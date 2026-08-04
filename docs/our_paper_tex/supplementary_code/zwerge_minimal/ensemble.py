"""
ZwerGe-UI — Deployed Decoder: Margin-Gated Fusion<->Native Ensemble
====================================================================
This file is the paper's HEADLINE deployment method: `run_p2p_ensemble()`,
described in the "Training-Free Gated Decoding" section:

    In one prefill we read both the fused point f, produced by a
    training-free posterior-to-point decoder [...], and the backbone's
    native autoregressive coordinate n, and route between them with a
    GT-free gate on the posterior sharpness: emit f when the margin
    m = p_final^(1) - p_final^(2), the difference between the top-two
    single-patch probabilities, exceeds tau, otherwise emit n.
    The default is tau=0.25, while the main results use 0.20.

i.e. exactly:

    margin = top1_patch_prob - top2_patch_prob
    use_fusion = margin > ensemble_margin_thr        # default 0.20 (main results)
    emit f (fusion point)  if use_fusion else n (native point)

This module also contains `decode_p2p`, the training-free
posterior-to-point decoder referenced above ("produced by a training-free
posterior-to-point decoder detailed in the supplementary material"),
composed of three composable, parameter-free levels:

    Level 1  region re-ranking        mass / mean / mass-over-sqrt(area)
    Level 2  cross-layer consensus    omega-weighted geometric mean of
                                       per-layer region mass
    Level 3  local Gaussian inversion weighted quadratic fit on
                                       log-posterior for sub-patch precision

None of these three levels have any learnable parameters — they only
consume the (already-trained) per-layer probe posteriors and the learned
fusion weights omega, both already computed by a single
`predict_layerwise()` prefill.

IMPORTANT — do not silently change the algorithm: the margin formula
above must remain `top1 - top2` of the FUSION posterior `p_final`
(`ensemble_gate="margin"`, the paper's default and the "robust winner"
mentioned in the source comments); the alternative `ensemble_gate="mass"`
(top-1 region posterior mass threshold) is kept only as an ablation.
"""

import math
from typing import List, Optional, Tuple

import torch

from inference import (
    RetrofitInference,
    get_zoom_crop_box,
    scores_to_point_and_topk,
)


# =============================================================================
# Level 1 — region extraction + region scoring
# =============================================================================

def _extract_candidate_regions(
    p_1d: torch.Tensor,
    n_width: int,
    n_height: int,
    activation_threshold: float = 0.3,
) -> List[List[Tuple[int, int, int, float]]]:
    """
    Threshold (relative to max) + 4-connectivity BFS -> connected regions.
    Each region is a list of (y, x, flat_idx, p) tuples.
    """
    scores = p_1d.float().cpu()
    if scores.dim() == 2:
        scores = scores.squeeze(0)
    max_score = scores.max().item()
    if max_score <= 0:
        return []
    threshold = max_score * activation_threshold
    valid_indices = (scores > threshold).nonzero(as_tuple=False).squeeze(-1)
    if valid_indices.numel() == 0:
        return []

    topk_values = scores[valid_indices]
    topk_coords = []
    for i, idx in enumerate(valid_indices.tolist()):
        y, x = idx // n_width, idx % n_width
        topk_coords.append((y, x, idx, topk_values[i].item()))

    regions: List[List[Tuple[int, int, int, float]]] = []
    visited: set = set()
    for y, x, idx, val in topk_coords:
        if idx in visited:
            continue
        region = [(y, x, idx, val)]
        visited.add(idx)
        queue = [(y, x, idx, val)]
        while queue:
            cy, cx, c_idx, c_val = queue.pop(0)
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = cy + dy, cx + dx
                if ny < 0 or ny >= n_height or nx < 0 or nx >= n_width:
                    continue
                for j, (ty, tx, t_idx, t_val) in enumerate(topk_coords):
                    if ty == ny and tx == nx and t_idx not in visited:
                        visited.add(t_idx)
                        region.append((ny, nx, t_idx, t_val))
                        queue.append((ny, nx, t_idx, t_val))
        regions.append(region)
    return regions


def _region_indices_dilated(
    region: List[Tuple[int, int, int, float]],
    n_width: int, n_height: int, dilate: int = 1,
) -> List[int]:
    """Flat patch indices in `region`, expanded by `dilate` cells (Chebyshev)."""
    cells = {(y, x) for y, x, _, _ in region}
    if dilate > 0:
        extra = set()
        for y, x in cells:
            for dy in range(-dilate, dilate + 1):
                for dx in range(-dilate, dilate + 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < n_height and 0 <= nx < n_width:
                        extra.add((ny, nx))
        cells = cells | extra
    return [y * n_width + x for y, x in cells]


def region_score(
    region: List[Tuple[int, int, int, float]],
    kind: str = "mass_sqrt_area",
    alpha: float = 0.5,
) -> float:
    """
    Level 1 — score a candidate region by its posterior mass.

      max            : peak patch probability (baseline behaviour)
      mass           : sum p_i over the region                (favours large regions)
      mean           : (1/|C|) sum p_i                        (favours sharp peaks)
      mass_sqrt_area : sum p_i / sqrt(|C|)                     (robust compromise — default)
      balanced       : sum p_i / |C|^alpha
    """
    if not region:
        return -1e18
    area = len(region)
    total = sum(item[3] for item in region)
    if kind == "max":
        return max(item[3] for item in region)
    if kind == "mass":
        return total
    if kind == "mean":
        return total / area
    if kind == "balanced":
        return total / (area ** alpha)
    return total / (math.sqrt(area) + 1e-12)


# =============================================================================
# Level 2 — cross-layer consensus
# =============================================================================

def consensus_score(
    region: List[Tuple[int, int, int, float]],
    per_layer_probs_active: List[torch.Tensor],
    omega: torch.Tensor,
    n_width: int, n_height: int,
    dilate: int = 1, eps: float = 1e-9,
) -> float:
    """
    Level 2 — cross-layer posterior consensus for a candidate region:

        S_l = sum_{i in dilated(region)} p_l(i)      (per-layer region mass)
        S   = sum_l omega_l * log(S_l + eps)          (omega-weighted geometric mean)

    Penalises single-layer spurious peaks: a candidate supported by only
    one active layer scores low under the geometric mean even if its
    FUSED mass is high.
    """
    if not per_layer_probs_active:
        return 0.0
    idxs = _region_indices_dilated(region, n_width, n_height, dilate)
    s = 0.0
    for j, p_l in enumerate(per_layer_probs_active):
        m = float(p_l.float().cpu().flatten()[idxs].sum())
        w = float(omega[j]) if j < omega.numel() else 0.0
        s += w * math.log(m + eps)
    return s


# =============================================================================
# Level 3 — local Gaussian inversion (sub-patch mode recovery)
# =============================================================================

def refine_local_mode(
    p: torch.Tensor, peak_x: int, peak_y: int,
    n_width: int, n_height: int,
    radius: int = 2, max_offset: float = 0.75,
) -> Optional[Tuple[float, float]]:
    """
    Level 3 — analytic sub-patch mode recovery.

    Fit a weighted 2-D quadratic to log p_final over a (2*radius+1)^2
    neighbourhood of the UNTHRESHOLDED posterior peak, then solve for the
    continuous mode via delta* = -H^-1 g. Returns (dx, dy) in patch-cell
    units, or None if the fit degenerates (non-concave Hessian, too few
    points, ill-conditioned) — callers must fall back to the region
    centroid in that case.
    """
    grid = p.float().cpu().reshape(n_height, n_width)
    xs, ys, values = [], [], []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            x, y = peak_x + dx, peak_y + dy
            if 0 <= x < n_width and 0 <= y < n_height:
                xs.append(dx); ys.append(dy)
                values.append(math.log(max(float(grid[y, x]), 1e-9)))

    if len(values) < 6:
        return None

    X = torch.tensor([[x * x, x * y, y * y, x, y, 1.0] for x, y in zip(xs, ys)], dtype=torch.float64)
    target = torch.tensor(values, dtype=torch.float64)

    dist2 = torch.tensor([x * x + y * y for x, y in zip(xs, ys)], dtype=torch.float64)
    weights = torch.exp(-0.5 * dist2)
    W = torch.diag(weights)

    try:
        beta = torch.linalg.solve(
            X.T @ W @ X + 1e-6 * torch.eye(6, dtype=torch.float64),
            X.T @ W @ target,
        )
    except RuntimeError:
        return None

    a, b, c, d, e, _ = beta.tolist()
    H = torch.tensor([[2 * a, b], [b, 2 * c]], dtype=torch.float64)
    g = torch.tensor([d, e], dtype=torch.float64)

    eigvals = torch.linalg.eigvalsh(H)
    if not torch.all(eigvals < -1e-6):   # must be a genuine concave peak
        return None

    try:
        offset = -torch.linalg.solve(H, g)
    except RuntimeError:
        return None
    if not torch.isfinite(offset).all():
        return None

    offset = offset.clamp(-max_offset, max_offset)
    return float(offset[0]), float(offset[1])


def _region_peak(region: List[Tuple[int, int, int, float]]) -> Tuple[int, int, int, float]:
    return max(region, key=lambda item: item[3])


def _region_centroid(
    region: List[Tuple[int, int, int, float]],
    n_width: int, n_height: int,
    decode_strategy: str = "centroid",
    peak_shift_alpha: float = 0.5, temperature: float = 0.5,
) -> Tuple[float, float]:
    """Continuous (x_norm, y_norm) for a region via the legacy decode rules (fallback for Level 3)."""
    norm_centers, weights = [], []
    for y, x, _, score in region:
        norm_centers.append(((x + 0.5) / n_width, (y + 0.5) / n_height))
        weights.append(score)
    max_idx = int(max(range(len(weights)), key=lambda i: weights[i]))
    argmax_center = norm_centers[max_idx]
    total_w = sum(weights)
    centroid = (
        sum(nc[0] * w for nc, w in zip(norm_centers, weights)) / total_w,
        sum(nc[1] * w for nc, w in zip(norm_centers, weights)) / total_w,
    )
    if decode_strategy == "argmax":
        return argmax_center
    if decode_strategy == "peak_shift":
        a = peak_shift_alpha
        return (a * argmax_center[0] + (1.0 - a) * centroid[0],
                a * argmax_center[1] + (1.0 - a) * centroid[1])
    if decode_strategy == "temperature":
        T = max(temperature, 1e-6)
        scaled_w = [w ** (1.0 / T) for w in weights]
        total_sw = sum(scaled_w) + 1e-12
        return (sum(nc[0] * sw for nc, sw in zip(norm_centers, scaled_w)) / total_sw,
                sum(nc[1] * sw for nc, sw in zip(norm_centers, scaled_w)) / total_sw)
    return centroid


def decode_p2p(
    p_final: torch.Tensor,
    n_width: int,
    n_height: int,
    activation_threshold: float = 0.3,
    topk: int = 3,
    # Level 1
    region_scorer: str = "mass_sqrt_area",
    # Level 2
    per_layer_probs: Optional[List[torch.Tensor]] = None,
    active_probe_indices: Optional[List[int]] = None,
    omega: Optional[torch.Tensor] = None,
    use_consensus: bool = True,
    consensus_dilate: int = 1,
    consensus_weight: float = 1.0,
    fusion_mass_weight: float = 0.5,
    spatial_disagree_weight: float = 0.25,
    # Level 3
    use_local_mode: bool = True,
    local_radius: int = 2,
    local_max_offset: float = 0.75,
    local_max_area: int = 0,
    # fallback decode
    fallback_decode: str = "centroid",
    peak_shift_alpha: float = 0.5,
    temperature: float = 0.5,
) -> Tuple[Tuple[float, float], List[Tuple[float, float]], List[float], dict]:
    """
    ZwerGe-P2P single-pass training-free decoder ("the fused point f,
    produced by a training-free posterior-to-point decoder", paper's
    "Training-Free Gated Decoding" section).

    Pipeline: extract candidate regions (BFS) -> Level-1 score -> optional
    Level-2 cross-layer-consensus re-rank -> sort -> Level-3 local-mode
    refinement of the winning region -> fallback to the region centroid if
    the local fit degenerates.

    If `per_layer_probs`/`omega` are None OR `use_consensus` is False, the
    score reduces to the pure Level-1 `region_scorer` (a Level-1-only
    ablation against the `max`-peak baseline).

    Returns (best_point, topk_points, region_scores, meta) where `meta`
    carries per-region diagnostics (peak idx, local-mode offset, patch
    indices for the top region — consumed by run_p2p_ensemble's mass
    computation).
    """
    p = p_final.float().cpu()
    if p.dim() == 2:
        p = p.squeeze(0)

    regions = _extract_candidate_regions(p, n_width, n_height, activation_threshold)

    if not regions:
        best_idx = int(p.argmax().item())
        y, x = best_idx // n_width, best_idx % n_width
        pt = ((x + 0.5) / n_width, (y + 0.5) / n_height)
        return pt, [pt], [float(p[best_idx].item())], {
            "n_regions": 0, "winner": None, "local_mode_applied": False,
            "fallback": "global_argmax",
        }

    plp_active: List[torch.Tensor] = []
    if use_consensus and per_layer_probs is not None and omega is not None:
        if active_probe_indices is None:
            active_probe_indices = list(range(len(per_layer_probs)))
        plp_active = [per_layer_probs[i] for i in active_probe_indices if i < len(per_layer_probs)]

    scored = []
    for region in regions:
        s1 = region_score(region, kind=region_scorer)
        meta_r: dict = {"region_score_l1": s1}
        if plp_active:
            s_cons = consensus_score(region, plp_active, omega, n_width, n_height, dilate=consensus_dilate)
            s_fuse = math.log(region_score(region, kind="mass") + 1e-9)
            # Cross-layer spatial-disagreement penalty: variance of
            # per-layer local centroids restricted to this region's patches.
            mus = []
            for p_l in plp_active:
                ws = p_l.float().cpu().flatten()[[i[2] for i in region]]
                wsum = ws.sum()
                if wsum > 0:
                    xs = torch.tensor([i[1] for i in region], dtype=torch.float32)
                    ys = torch.tensor([i[0] for i in region], dtype=torch.float32)
                    mus.append((float((ws * xs).sum() / wsum), float((ws * ys).sum() / wsum)))
            var_term = 0.0
            if len(mus) >= 2:
                cxs = torch.tensor([m[0] for m in mus])
                cys = torch.tensor([m[1] for m in mus])
                var_term = float(cxs.var() + cys.var())
            combined = (consensus_weight * s_cons + fusion_mass_weight * s_fuse
                        - spatial_disagree_weight * var_term)
            meta_r.update({"consensus": s_cons, "fusion_mass_log": s_fuse,
                           "spatial_var": var_term, "combined": combined})
            s_final = combined
        else:
            s_final = s1
        scored.append((region, s_final, meta_r))

    scored.sort(key=lambda t: t[1], reverse=True)
    top = scored[:max(topk, 1)]

    centers, scores_out, meta_regions = [], [], []
    winner_meta: Optional[dict] = None
    for rank, (region, s, meta_r) in enumerate(top):
        py, px, pidx, pval = _region_peak(region)
        offset, local_applied = None, False
        if use_local_mode and (local_max_area <= 0 or len(region) <= local_max_area):
            offset = refine_local_mode(p, px, py, n_width, n_height,
                                        radius=local_radius, max_offset=local_max_offset)
            local_applied = offset is not None
        if offset is not None:
            dx, dy = offset
            cx, cy = (px + 0.5 + dx) / n_width, (py + 0.5 + dy) / n_height
        else:
            cx, cy = _region_centroid(
                region, n_width, n_height, decode_strategy=fallback_decode,
                peak_shift_alpha=peak_shift_alpha, temperature=temperature,
            )
        center = (float(cx), float(cy))
        centers.append(center)
        scores_out.append(float(s))
        rm = {
            "rank": rank, "peak_idx": pidx, "peak_y": py, "peak_x": px,
            "peak_p": pval, "area": len(region),
            "patch_idxs": [item[2] for item in region],
            "local_mode_applied": local_applied,
            "local_offset": list(offset) if offset is not None else None,
            **meta_r,
        }
        meta_regions.append(rm)
        if rank == 0:
            winner_meta = rm

    best_point = centers[0]
    return best_point, centers[:topk], scores_out[:topk], {
        "n_regions": len(regions), "winner": winner_meta, "regions": meta_regions,
    }


def run_p2p_from_pred(pred: dict, head, p2p_cfg: dict):
    """`decode_p2p` wrapper that pulls active layer indices off the grounding head."""
    best, centers, scores, meta = decode_p2p(
        p_final=pred["p_final"], n_width=pred["n_width"], n_height=pred["n_height"],
        activation_threshold=p2p_cfg["activation_threshold"], topk=p2p_cfg["topk"],
        region_scorer=p2p_cfg["region_scorer"], per_layer_probs=pred["per_layer_probs"],
        active_probe_indices=list(head.active_probe_indices), omega=pred["omega"],
        use_consensus=p2p_cfg["use_consensus"], consensus_dilate=p2p_cfg["consensus_dilate"],
        consensus_weight=p2p_cfg["consensus_weight"], fusion_mass_weight=p2p_cfg["fusion_mass_weight"],
        spatial_disagree_weight=p2p_cfg["spatial_disagree_weight"],
        use_local_mode=p2p_cfg["use_local_mode"], local_radius=p2p_cfg["local_radius"],
        local_max_offset=p2p_cfg["local_max_offset"], fallback_decode=p2p_cfg["fallback_decode"],
        peak_shift_alpha=p2p_cfg["peak_shift_alpha"], temperature=p2p_cfg["temperature"],
    )
    return best, centers, meta


# =============================================================================
# THE DEPLOYED DECODER — margin-gated fusion<->native ensemble
# =============================================================================

def default_p2p_cfg(
    activation_threshold: float = 0.3,
    topk: int = 3,
    region_scorer: str = "mass_sqrt_area",
    use_consensus: bool = True,
    consensus_dilate: int = 1,
    consensus_weight: float = 1.0,
    fusion_mass_weight: float = 0.5,
    spatial_disagree_weight: float = 0.25,
    use_local_mode: bool = True,
    local_radius: int = 2,
    local_max_offset: float = 0.75,
    fallback_decode: str = "centroid",
    peak_shift_alpha: float = 0.5,
    temperature: float = 0.5,
    gate_dilate: int = 1,
    zoom_upscale_target: int = 0,
    ensemble_mass_thr: float = 0.4,
    ensemble_gate: str = "margin",
    ensemble_margin_thr: float = 0.20,
) -> dict:
    """
    Build the P2P/ensemble config dict.

    Defaults reproduce the paper's main-table configuration:
      ensemble_gate="margin", ensemble_margin_thr=0.20 (Table 2's tau=0.20;
      the paper also mentions a "default" tau=0.25 for the isolated
      description of the gate — 0.20 is what the MAIN RESULTS use).
    """
    return {
        "activation_threshold": activation_threshold, "topk": topk,
        "region_scorer": region_scorer, "use_consensus": use_consensus,
        "consensus_dilate": consensus_dilate, "consensus_weight": consensus_weight,
        "fusion_mass_weight": fusion_mass_weight, "spatial_disagree_weight": spatial_disagree_weight,
        "use_local_mode": use_local_mode, "local_radius": local_radius,
        "local_max_offset": local_max_offset, "fallback_decode": fallback_decode,
        "peak_shift_alpha": peak_shift_alpha, "temperature": temperature,
        "gate_dilate": gate_dilate, "zoom_upscale_target": zoom_upscale_target,
        "ensemble_mass_thr": ensemble_mass_thr, "ensemble_gate": ensemble_gate,
        "ensemble_margin_thr": ensemble_margin_thr,
    }


def run_p2p_ensemble(
    grounder: RetrofitInference,
    image,
    instruction: str,
    device,
    p2p_cfg: dict,
    zoom_max_new_tokens: int,
    activation_threshold: float,
    topk: int,
):
    """
    Confidence-gated fusion+native ensemble — THE DEPLOYED DECODER of the
    paper (§"Training-Free Gated Decoding", Table 2's "+ZwerGe" column).

    Computes BOTH:
      f (fusion point)  — via ONE ZwerGe prefill (predict_layerwise) +
                            decode_p2p (training-free posterior-to-point).
      n (native point)  — the backbone's own full-image autoregressive
                            coordinate (predict_zoom_backbone(full_image=True)),
                            i.e. the "Native" baseline column of Table 2.

    Gate (must stay EXACTLY this formula — see module docstring):
        margin = top1_patch_prob(p_final) - top2_patch_prob(p_final)
        use_fusion = margin > ensemble_margin_thr      (default 0.20)
        emit f if use_fusion else n

    A sharp fusion posterior (high margin) means the probe is confident in
    one location -> trust the fused point; a diffuse posterior (low margin)
    means the intermediate-layer signal is ambiguous -> defer to the
    backbone's own serializer.

    Cost: the native generate is shared with the baseline; the only added
    cost is one cheap fusion prefill (no extra generate) — hence "at the
    cost of a single forward pass" in the abstract.

    Returns:
      out          — (x, y) chosen point (fusion or native)
      f_centers    — list of candidate points for overlap@k bookkeeping
      pred         — the predict_layerwise() output dict (for logging/caching)
      meta         — dict with ensemble_used_fusion / ensemble_mass /
                     ensemble_margin / gate config (GT-free, useful for the
                     %Fus. column of Table 2)
      native_point — the raw native coordinate (x, y) or None
    """
    # ── Fusion point + posterior, from a single ZwerGe prefill. ────────────
    pred = grounder.predict_layerwise(
        image=image, instruction=instruction, device=device,
        activation_threshold=activation_threshold, topk=topk,
        decode_strategy="centroid",
        peak_shift_alpha=p2p_cfg["peak_shift_alpha"], temperature=p2p_cfg["temperature"],
    )
    f_best, f_centers, f_meta = run_p2p_from_pred(pred, grounder.model.layerwise_grounding_head, p2p_cfg)

    # ── Native full-image coordinate (the baseline; shared compute). ───────
    pred_nat = grounder.predict_zoom_backbone(
        image=image, instruction=instruction, device=device,
        activation_threshold=activation_threshold, topk=topk,
        max_new_tokens=zoom_max_new_tokens, full_image=True,
    )
    native_point = pred_nat.get("zoom_point")

    p_flat = pred["p_final"].float().cpu().flatten()
    mass = 0.0
    if f_meta.get("regions") and f_meta["regions"][0].get("patch_idxs"):
        idxs = torch.tensor(f_meta["regions"][0]["patch_idxs"])
        mass = float(p_flat[idxs].sum())

    # ── THE GATE: margin = top1 patch prob - top2 patch prob. ───────────────
    # This is a sharpness measure of the FUSED posterior p_final, NOT of any
    # single region. A sharp single mode -> high margin -> trust fusion.
    top2 = torch.topk(p_flat, 2).values
    margin = float(top2[0] - top2[1])

    gate = p2p_cfg.get("ensemble_gate", "margin")
    if gate == "mass":
        use_fusion = mass > p2p_cfg.get("ensemble_mass_thr", 0.4)
    else:  # "margin" — the paper's deployed, robust-winner gate
        use_fusion = margin > p2p_cfg.get("ensemble_margin_thr", 0.20)

    if use_fusion or native_point is None:
        out = (float(f_best[0]), float(f_best[1]))
        f_centers = list(f_centers)
    else:
        out = (float(native_point[0]), float(native_point[1]))
        f_centers = [out] + list(f_centers)

    meta = {
        "ensemble_used_fusion": bool(use_fusion),
        "ensemble_mass": float(mass),
        "ensemble_margin": float(margin),
        "ensemble_gate": gate,
        "ensemble_margin_thr": p2p_cfg.get("ensemble_margin_thr", 0.20),
        "ensemble_mass_thr": p2p_cfg.get("ensemble_mass_thr", 0.4),
    }
    return out, f_centers, pred, meta, native_point


# =============================================================================
# Posterior-Constrained Native Decoding (an alternative gate, kept for
# completeness / ablation — NOT the main-table deployed decoder above)
# =============================================================================

def _dilate_idxset(idxs: set, n_width: int, n_height: int, dilate: int) -> set:
    """Chebyshev dilation of a set of flat patch indices."""
    if dilate <= 0:
        return set(idxs)
    out = set()
    for i in idxs:
        y, x = i // n_width, i % n_width
        for dy in range(-dilate, dilate + 1):
            for dx in range(-dilate, dilate + 1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < n_height and 0 <= nx < n_width:
                    out.add(ny * n_width + nx)
    return out


def run_p2p_native_gate(pred: dict, head, native_point, p2p_cfg: dict):
    """
    Posterior-Constrained Native Decoding (ablation): map the backbone's
    native continuous coordinate onto the patch grid; if it lands inside
    any of the ZwerGe top-k candidate regions (optionally dilated), emit
    the native point, otherwise fall back to the ZwerGe-P2P point.

    This is a REGION-MEMBERSHIP gate (does the native point agree with
    where ZwerGe thinks the target is?), distinct from the margin gate of
    `run_p2p_ensemble` (is the fused posterior itself sharp?). Kept here
    for completeness; NOT the decoder used to produce Table 2.
    """
    best, centers, meta = run_p2p_from_pred(pred, head, p2p_cfg)
    n_w, n_h = pred["n_width"], pred["n_height"]
    px = int(min(n_w - 1, max(0, native_point[0] * n_w)))
    py = int(min(n_h - 1, max(0, native_point[1] * n_h)))
    patch_idx = py * n_w + px
    dilate = p2p_cfg["gate_dilate"]
    supported = False
    for rm in meta["regions"][:p2p_cfg["topk"]]:
        idxs = _dilate_idxset(set(rm["patch_idxs"]), n_w, n_h, dilate)
        if patch_idx in idxs:
            supported = True
            break
    meta["native_used"] = bool(supported)
    if supported:
        out_point = (float(native_point[0]), float(native_point[1]))
        f_centers = [out_point] + list(centers)
    else:
        out_point = best
        f_centers = list(centers)
    return out_point, f_centers, meta


def get_zoom_crop_box_p2p(
    p_final: torch.Tensor,
    per_layer_probs: List[torch.Tensor],
    active_probe_indices: List[int],
    omega: torch.Tensor,
    n_width: int, n_height: int,
    image_w: int, image_h: int,
    token_cell_px: int,
    activation_threshold: float = 0.3,
    padding_cells: int = 3,
    region_scorer: str = "mass_sqrt_area",
    use_consensus: bool = True,
    min_crop_frac: float = 0.15,
) -> Tuple[int, int, int, int]:
    """
    P2P-enhanced zoom crop box: like `get_zoom_crop_box` (inference.py) but
    the winning region is chosen by cross-layer posterior consensus
    (decode_p2p Level 1+2) instead of the single max-score patch, so the
    backbone's close-up is centred on the region that the fused,
    cross-layer-verified posterior endorses. Falls back to
    `get_zoom_crop_box` when P2P finds no region.
    """
    _, _, _, meta = decode_p2p(
        p_final=p_final, n_width=n_width, n_height=n_height,
        activation_threshold=activation_threshold, topk=1, region_scorer=region_scorer,
        per_layer_probs=per_layer_probs, active_probe_indices=active_probe_indices,
        omega=omega, use_consensus=use_consensus, use_local_mode=False,
    )
    regions = meta.get("regions") or []
    if not regions:
        return get_zoom_crop_box(
            p_final, n_width, n_height, image_w, image_h, token_cell_px,
            activation_threshold=activation_threshold, padding_cells=padding_cells,
        )
    idxs = regions[0]["patch_idxs"]
    rows = [i // n_width for i in idxs]
    cols = [i % n_width for i in idxs]
    min_col, max_col = min(cols), max(cols)
    min_row, max_row = min(rows), max(rows)
    x_min = max(0,        (min_col - padding_cells) * token_cell_px)
    y_min = max(0,        (min_row - padding_cells) * token_cell_px)
    x_max = min(image_w,  (max_col + 1 + padding_cells) * token_cell_px)
    y_max = min(image_h,  (max_row + 1 + padding_cells) * token_cell_px)
    min_w = max(1, int(image_w * min_crop_frac))
    min_h = max(1, int(image_h * min_crop_frac))
    if (x_max - x_min) < min_w:
        cx = (x_min + x_max) // 2
        x_min = max(0, cx - min_w // 2); x_max = min(image_w, x_min + min_w)
    if (y_max - y_min) < min_h:
        cy = (y_min + y_max) // 2
        y_min = max(0, cy - min_h // 2); y_max = min(image_h, y_min + min_h)
    return x_min, y_min, x_max, y_max
