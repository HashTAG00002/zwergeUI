#!/usr/bin/env python3
"""
p2p_sweep.py — Offline ZWERGE-P2P decode-config sweep + error decomposition
============================================================================

Loads per-sample posteriors cached by `eval_retrofit.py --cache_posteriors`
and sweeps dozens of training-free decode configurations in seconds — no 8B
model reload — using the *identical* `decode_p2p` code path as the live eval,
so sweep numbers match a real eval run.

For every config it reports strict top-1 metrics (hit@1 / overlap@1) plus the
top-k proposal-recall ceiling (hit@k / overlap@k), the four-quadrant error
decomposition (A/B/C/D, oracle §六), recovery vs. damage vs. the baseline
(oracle §十 第三步), and selection-recovery (oracle §十 第四步).

Usage:
  python p2p_sweep.py --cache_dir <posteriors_dir> [--out summary.json]
  python p2p_sweep.py --cache_dir <...> --baseline_only            # A/B/C/D only
  python p2p_sweep.py --cache_dir <...> --configs baseline,mass,bal,cons,cons_local
  python p2p_sweep.py --cache_dir <...> --native_gate              # also eval p2p_native gate

The baseline config (region_scorer=max, no consensus, no local-mode,
centroid fallback) reproduces the legacy `scores_to_point_and_topk` decode,
so its metrics should match the cached `fusion_overlap1/hit1` in results.json.
"""

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from inference_base import decode_p2p, point_in_bbox, do_boxes_overlap   # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Config presets (oracle §十 第二步 minimal matrix)
# ─────────────────────────────────────────────────────────────────────────────

CONFIGS: Dict[str, dict] = {
    "baseline":     dict(region_scorer="max",            use_consensus=False, use_local_mode=False),
    "mass":         dict(region_scorer="mass",           use_consensus=False, use_local_mode=False),
    "mean":         dict(region_scorer="mean",           use_consensus=False, use_local_mode=False),
    "bal":          dict(region_scorer="balanced",       use_consensus=False, use_local_mode=False),
    "msqrt":        dict(region_scorer="mass_sqrt_area", use_consensus=False, use_local_mode=False),
    "cons":         dict(region_scorer="mass_sqrt_area", use_consensus=True,  use_local_mode=False),
    # Level 3 with progressively tighter offset clamps — ±0.75 crosses into
    # neighbouring patches and damages already-correct hits; smaller clamps
    # keep the refinement sub-patch (the regime where Gaussian inversion is sound).
    "local":        dict(region_scorer="mass_sqrt_area", use_consensus=False, use_local_mode=True,  local_max_offset=0.75),
    "local_o45":    dict(region_scorer="mass_sqrt_area", use_consensus=False, use_local_mode=True,  local_max_offset=0.45),
    "local_o35":    dict(region_scorer="mass_sqrt_area", use_consensus=False, use_local_mode=True,  local_max_offset=0.35),
    "local_o25":    dict(region_scorer="mass_sqrt_area", use_consensus=False, use_local_mode=True,  local_max_offset=0.25),
    # area-gated Level 3: only refine small/peaky regions (≤2 patches) — the
    # sub-patch-quantization (C-quadrant) regime — sparing broad confident regions.
    "local_o25_a2": dict(region_scorer="mass_sqrt_area", use_consensus=False, use_local_mode=True,  local_max_offset=0.25, local_max_area=2),
    "local_o35_a2": dict(region_scorer="mass_sqrt_area", use_consensus=False, use_local_mode=True,  local_max_offset=0.35, local_max_area=2),
    "cons_local":   dict(region_scorer="mass_sqrt_area", use_consensus=True,  use_local_mode=True,  local_max_offset=0.75),
    "cons_local_o35":dict(region_scorer="mass_sqrt_area",use_consensus=True,  use_local_mode=True,  local_max_offset=0.35),
    "max_cons_loc": dict(region_scorer="max",            use_consensus=True,  use_local_mode=True,  local_max_offset=0.75),
}


def _patch_box(c, phx, phy):
    return (c[0] - phx, c[1] - phy, c[0] + phx, c[1] + phy)


def _metrics_for(best, centers, gt_bbox_norm, n_w, n_h, topk):
    """Strict top-1 + top-k proposal-recall metrics for one decoded sample."""
    phx = 0.5 / n_w
    phy = 0.5 / n_h
    hit1 = int(point_in_bbox(best[0], best[1], gt_bbox_norm))
    ov1 = int(do_boxes_overlap(_patch_box(best, phx, phy), gt_bbox_norm))
    cands = centers[:topk]
    hitk = int(any(point_in_bbox(c[0], c[1], gt_bbox_norm) for c in cands))
    ovk = int(any(do_boxes_overlap(_patch_box(c, phx, phy), gt_bbox_norm) for c in cands))
    return dict(hit1=hit1, ov1=ov1, hitk=hitk, ovk=ovk)


def _dilate_idxset(idxs, n_width, n_height, dilate):
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


def _native_gate(sample, p2p_cfg, topk, gate_dilate):
    """
    Posterior-constrained native decoding gate evaluated offline from the
    cached `native_point` (only present when the eval ran decode_strategy=
    p2p_native). Returns (out_point, centers, use_native); when no native point
    is cached the gate degenerates to the plain P2P fallback (use_native=False).
    """
    native = sample.get("native_point")
    best, centers, _, meta = decode_p2p(
        p_final=sample["p_final"],
        n_width=sample["n_width"], n_height=sample["n_height"],
        activation_threshold=p2p_cfg["activation_threshold"], topk=topk,
        region_scorer=p2p_cfg["region_scorer"],
        per_layer_probs=[sample["per_layer_probs"][i] for i in sample["active_probe_indices"].tolist()],
        active_probe_indices=list(range(int(sample["active_probe_indices"].numel()))),
        omega=sample["omega"],
        use_consensus=p2p_cfg["use_consensus"], use_local_mode=p2p_cfg["use_local_mode"],
    )
    n_w, n_h = sample["n_width"], sample["n_height"]
    use_native = False
    if native is not None:
        px = int(min(n_w - 1, max(0, native[0] * n_w)))
        py = int(min(n_h - 1, max(0, native[1] * n_h)))
        patch_idx = py * n_w + px
        for rm in meta["regions"][:topk]:
            if patch_idx in _dilate_idxset(set(rm["patch_idxs"]), n_w, n_h, gate_dilate):
                use_native = True
                break
    out = (float(native[0]), float(native[1])) if use_native else best
    return out, centers, use_native


def _decode_sample(sample, cfg, topk, activation_threshold, gate_dilate=1):
    """Run one config on one cached sample → (best, centers, metrics)."""
    n_w, n_h = sample["n_width"], sample["n_height"]
    if cfg == "__native_gate__":
        out, centers, _ = _native_gate(sample, dict(
            activation_threshold=activation_threshold, region_scorer="mass_sqrt_area",
            use_consensus=True, use_local_mode=True), topk, gate_dilate)
        return out, centers, _metrics_for(out, centers, sample["gt_bbox_norm"].tolist(), n_w, n_h, topk)
    best, centers, _, _ = decode_p2p(
        p_final=sample["p_final"],
        n_width=n_w, n_height=n_h,
        activation_threshold=activation_threshold, topk=topk,
        region_scorer=cfg["region_scorer"],
        per_layer_probs=sample["per_layer_probs"],
        active_probe_indices=sample["active_probe_indices"].tolist(),
        omega=sample["omega"],
        use_consensus=cfg["use_consensus"], consensus_dilate=1,
        use_local_mode=cfg["use_local_mode"], local_radius=2,
        local_max_offset=cfg.get("local_max_offset", 0.75),
        local_max_area=cfg.get("local_max_area", 0),
        fallback_decode="centroid",
    )
    return best, centers, _metrics_for(best, centers, sample["gt_bbox_norm"].tolist(), n_w, n_h, topk)


def load_samples(cache_dir: str) -> List[dict]:
    files = sorted(glob.glob(os.path.join(cache_dir, "idx*.pt")))
    samples = []
    for fp in files:
        samples.append(torch.load(fp, map_location="cpu"))
    return samples


def aggregate(samples, per_sample_metrics) -> dict:
    n = len(per_sample_metrics)
    s = dict(hit1=0, ov1=0, hitk=0, ovk=0)
    for m in per_sample_metrics:
        for k in s:
            s[k] += m[k]
    return {**{k: round(v / n * 100, 4) if n else 0.0 for k, v in s.items()}, "n": n}


def quadrant_of(m) -> str:
    """A/B/C/D error quadrant from BASELINE metrics (oracle §六)."""
    if m["hit1"]:
        return "A"
    if m["hitk"]:
        return "B"
    if m["ovk"]:
        return "C"
    return "D"


def run_sweep(cache_dir, config_names, topk, activation_threshold, gate_dilate, out_path):
    samples = load_samples(cache_dir)
    if not samples:
        raise SystemExit(f"No cached posteriors (*.pt) in {cache_dir}")
    print(f"[sweep] {len(samples)} samples from {cache_dir}")

    # Baseline metrics (computed from posteriors via the legacy max+centroid path).
    base_decoded = [_decode_sample(s, CONFIGS["baseline"], topk, activation_threshold) for s in samples]
    base_metrics = [b[2] for b in base_decoded]
    quadrants = [quadrant_of(m) for m in base_metrics]
    qcounts = {q: quadrants.count(q) for q in "ABCD"}
    print(f"\n[sweep] BASELINE (max + centroid, = legacy production decode):")
    print(f"        {aggregate(samples, base_metrics)}")
    qpct = {q: round(qcounts[q] / len(samples) * 100, 2) for q in "ABCD"}
    print(f"        A/B/C/D quadrants: "
          f"A(h1)={qcounts['A']} ({qpct['A']}%)  "
          f"B(h3\\h1)={qcounts['B']} ({qpct['B']}%)  "
          f"C(ov3\\h3)={qcounts['C']} ({qpct['C']}%)  "
          f"D(¬ov3)={qcounts['D']} ({qpct['D']}%)")

    results = {
        "cache_dir": cache_dir, "n_samples": len(samples), "topk": topk,
        "baseline": {"metrics": aggregate(samples, base_metrics), "quadrants": qpct},
        "configs": {},
    }

    # Pool of "overlap-recall but not hit" samples (B+C), the recovery target.
    bc_idx = [i for i, m in enumerate(base_metrics) if not m["hit1"] and m["ovk"]]
    n_old_hit1 = sum(m["hit1"] for m in base_metrics)
    n_old_ovk = sum(m["ovk"] for m in base_metrics)
    denom_sel = max(1, n_old_ovk - n_old_hit1)   # top-3 headroom over top-1

    cfg_names = list(config_names)
    if "__native_gate__" in [c for c in cfg_names]:
        # only sweep native gate if cached native_point present
        if not any(s.get("native_point") is not None for s in samples):
            print("[sweep] no native_point cached — skipping native-gate config")
            cfg_names = [c for c in cfg_names if c != "__native_gate__"]

    print(f"\n{'config':<16} {'hit@1':>7} {'ov@1':>7} {'hit@k':>7} {'ov@k':>7} "
          f"{'recov%':>7} {'dmg%':>6} {'selrec%':>8}")
    print("-" * 72)
    for name in cfg_names:
        cfg = "__native_gate__" if name == "__native_gate__" else CONFIGS[name]
        decoded = [_decode_sample(s, cfg, topk, activation_threshold, gate_dilate) for s in samples]
        ms = [d[2] for d in decoded]
        agg = aggregate(samples, ms)

        # recovery: among baseline (overlap-recall, not hit), fraction now hit
        recovered = sum(1 for i in bc_idx if ms[i]["hit1"])
        recovery = round(recovered / max(1, len(bc_idx)) * 100, 2)
        # damage: among baseline hits, fraction now broken
        damaged = sum(1 for i in range(len(samples)) if base_metrics[i]["hit1"] and not ms[i]["hit1"])
        damage = round(damaged / max(1, n_old_hit1) * 100, 2)
        # selection recovery (oracle §十 第四步)
        n_new_hit1 = sum(m["hit1"] for m in ms)
        selrec = round((n_new_hit1 - n_old_hit1) / denom_sel * 100, 2)

        # per-quadrant conversion: how many B→hit, C→hit
        b2h = sum(1 for i in range(len(samples)) if quadrants[i] == "B" and ms[i]["hit1"])
        c2h = sum(1 for i in range(len(samples)) if quadrants[i] == "C" and ms[i]["hit1"])

        print(f"{name:<16} {agg['hit1']:>7.2f} {agg['ov1']:>7.2f} {agg['hitk']:>7.2f} "
              f"{agg['ovk']:>7.2f} {recovery:>7.2f} {damage:>6.2f} {selrec:>8.2f}")
        results["configs"][name] = {
            "metrics": agg, "recovery_pct": recovery, "damage_pct": damage,
            "selection_recovery_pct": selrec, "B_to_hit": b2h, "C_to_hit": c2h,
        }

    if out_path:
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n[sweep] wrote {out_path}")
    return results



def main():
    p = argparse.ArgumentParser(description="Offline ZWERGE-P2P decode-config sweep")
    p.add_argument("--cache_dir", required=True, help="posteriors/ dir from eval_retrofit --cache_posteriors")
    p.add_argument("--out", default=None, help="output JSON path")
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--activation_threshold", type=float, default=0.3)
    p.add_argument("--gate_dilate", type=int, default=1)
    p.add_argument("--configs", default="baseline,mass,mean,bal,msqrt,cons,local,cons_local",
                   help="comma-separated config names (or 'all'); add 'native_gate' if cached")
    p.add_argument("--baseline_only", action="store_true")
    args = p.parse_args()

    if args.baseline_only:
        names = ["baseline"]
    else:
        names = args.configs.split(",")
        if "all" in names:
            names = list(CONFIGS.keys())
        if "native_gate" in names:
            names = [n for n in names if n != "native_gate"] + ["__native_gate__"]

    run_sweep(args.cache_dir, names, args.topk, args.activation_threshold,
              args.gate_dilate, args.out)


if __name__ == "__main__":
    main()
