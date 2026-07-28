"""
analysis_norm_census.py — P2-1: anchor / visual-patch hidden-state norm census.

Reviewer concern addressed:
  4sok-Q4 "pre-scale may discard activation-magnitude ~ token-confidence /
  attention-sink information" — the most direct link to Vision Transformers
  Need Registers (Darcet et al. 2023): high-norm "artifact/register" tokens
  absorb global information and serve as attention sinks.

We measure, PRE-scale (before the RMSNorm pre-scale that ZwerGe applies inside
the probe head), the per-layer ℓ2-norm distribution of the visual patch hidden
states. Three questions (Registers-paper analogs):
  (1) Is the norm distribution bimodal / heavy-tailed? (Registers Fig 2)
  (2) Does the high-norm outlier fraction grow with depth? (Registers Fig 3a,
      and our §3.1 claim "norms grow substantially with layer depth".)
  (3) Where do high-norm patches sit spatially — GUI chrome (status bars /
      toolbars, near-constant across screenshots) or uniform background?

Standalone: replicates predict_layerwise's forward (build_zwerge_inputs +
_forward_hidden_states_for_grounding) but captures ‖h_v‖₂ pre-scale. Does NOT
modify inference_base.py / modeling_base.py (parallel-agent boundary). 1 forward
per sample. ~100 samples × 2 models suffices for the census.
"""
import argparse
import gc
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

_PROBE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_PROBE_DIR, "../.."))
for _d in [_PROBE_DIR, os.path.join(_REPO_ROOT, "zwerge", "eval"),
           os.path.join(_REPO_ROOT, "zwerge", "src")]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from probe_utils import get_inference_class, load_eval_records, sample_records

OUT_DIR = Path(_PROBE_DIR) / "outputs"
FIG_DIR = OUT_DIR / "figures"
TBL_DIR = OUT_DIR / "tables"
for d in (FIG_DIR, TBL_DIR): d.mkdir(parents=True, exist_ok=True)

DATA_DIR = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/datasets/evaluation"
CKPT_BASE = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/.hdd/ckpt/zwerge"
# (model_type, display, A8 ckpt dir, conda env) — run one model type at a time.
MODEL_CFG = {
    "guiowl7b": ("GUI-Owl-7B (Qwen2.5-VL)", "guiowl7b_A8_cosmeta_ctx_exp001", "qwen25"),
    "guiowl":   ("GUI-Owl-1.5-8B (Qwen3-VL)", "guiowl_A8_cosmeta_ctx_exp001", "qwen3"),
    "uitars":   ("UI-TARS-1.5-7B (Qwen2.5-VL)", "uitars_A8_cosmeta_ctx_exp001", "qwen25"),
    "uivenus":  ("UI-Venus-1.5-8B (Qwen3-VL)", "uivenus_A8_cosmeta_ctx_exp003", "qwen3"),
}


def _get_image_path(rec, image_root):
    for k in ("image_path", "img_path", "image_filename", "image"):
        v = rec.get(k)
        if v:
            p = os.path.join(image_root, v)
            if os.path.exists(p): return p
    return None


def capture_norms(grounder, image, instruction, device, probe_layers, max_pixels):
    """One forward. Returns (norms_per_layer: dict layer->[N_vis] np.float32,
    n_width, n_height)."""
    from inference_base import build_zwerge_inputs, grid_thw_to_nwh
    from zwerge_retrofit.constants import GROUNDING_SYSTEM_MESSAGE, GROUND_RESPONSE_CLICK

    sys_msg = grounder.system_message if grounder.system_message is not None else GROUNDING_SYSTEM_MESSAGE
    grd_resp = grounder.ground_response if grounder.ground_response is not None else GROUND_RESPONSE_CLICK
    inputs = build_zwerge_inputs(
        image=image, instruction=instruction, processor=grounder.processor,
        system_message=sys_msg, ground_response=grd_resp,
        user_prompt_template=grounder.user_prompt_template,
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None: attention_mask = attention_mask.to(device)
    pixel_values = inputs.get("pixel_values")
    if pixel_values is not None: pixel_values = pixel_values.to(device, dtype=grounder.model.dtype)
    image_grid_thw = inputs.get("image_grid_thw")
    if image_grid_thw is not None: image_grid_thw = image_grid_thw.to(device)
    mm_token_type_ids = inputs.get("mm_token_type_ids")
    if mm_token_type_ids is not None: mm_token_type_ids = mm_token_type_ids.to(device)

    if image_grid_thw is not None:
        n_width, n_height = grid_thw_to_nwh(image_grid_thw, merge_size=grounder.merge_size)
    else:
        w, h = image.size
        patch_size = getattr(getattr(grounder.processor, "image_processor", grounder.processor),
                             "patch_size", grounder.patch_size)
        cell = patch_size * grounder.merge_size
        n_width, n_height = max(1, w // cell), max(1, h // cell)

    token_ids_1d = input_ids[0]
    with torch.no_grad():
        all_hs = grounder.model._forward_hidden_states_for_grounding(
            input_ids=input_ids, attention_mask=attention_mask,
            pixel_values=pixel_values, image_grid_thw=image_grid_thw,
            device=device, mm_token_type_ids=mm_token_type_ids,
        )
    visual_indices = grounder.model._get_visual_indices(token_ids_1d)
    norms = {}
    if visual_indices.numel() > 0:
        for ℓ in probe_layers:
            raw = all_hs[ℓ + 1]
            if raw is None: continue
            hs = raw[0] if raw.dim() == 3 else raw  # [seq, d]
            h_v = hs[visual_indices].float()        # [N_vis, d]  PRE-SCALE
            norms[ℓ] = h_v.norm(dim=-1).cpu().numpy().astype(np.float32)
    del all_hs, inputs
    gc.collect(); torch.cuda.empty_cache()
    return norms, n_width, n_height


def run(model_type, n_samples=100, max_pixels=6_400_000, device_str="cuda:0", seed=42):
    disp, mdir, _env = MODEL_CFG[model_type]
    ckpt = os.path.join(CKPT_BASE, mdir, "checkpoint-2800")
    eval_json = os.path.join(DATA_DIR, "ScreenSpot-Pro", "eval.json")
    image_root = os.path.join(DATA_DIR, "ScreenSpot-Pro")
    device = torch.device(device_str)

    print(f"[P2-1] Loading {disp} from {ckpt}")
    InfClass = get_inference_class(model_type)
    grounder = InfClass.from_checkpoint(ckpt, device=device_str, max_pixels=max_pixels)
    grounder.model.eval(); grounder.model.to(device)
    probe_layers = list(grounder.model.layerwise_grounding_head.probe_layers)
    print(f"[P2-1] probe_layers={probe_layers}")

    records = load_eval_records(eval_json)
    records = sample_records(records, n_samples, seed=seed)
    print(f"[P2-1] {len(records)} samples")

    # Per-layer accumulators
    per_layer_norms = {ℓ: [] for ℓ in probe_layers}     # pooled norms (subsampled)
    per_layer_stats = {ℓ: {"mean": [], "std": [], "max": [], "q90": [],
                           "outlier_frac": [], "n_vis": []} for ℓ in probe_layers}
    # Spatial accumulation of top-K high-norm patches (relative coords in [0,1]^2)
    spatial_hist = {ℓ: np.zeros((40, 40), dtype=np.float64) for ℓ in probe_layers}
    n_ok = 0
    for rec in tqdm(records, desc=f"P2-1/{model_type}"):
        img_path = _get_image_path(rec, image_root)
        if img_path is None: continue
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            warnings.warn(f"img open fail {img_path}: {e}"); continue
        instruction = rec.get("instruction") or rec.get("query") or ""
        try:
            norms, n_w, n_h = capture_norms(grounder, img, instruction, device,
                                            probe_layers, max_pixels)
        except Exception as e:
            warnings.warn(f"forward fail: {e}"); torch.cuda.empty_cache(); continue
        if not norms: continue
        n_ok += 1
        for ℓ, nrm in norms.items():
            # pooled (cap per-sample to 200 patches for memory)
            per_layer_norms[ℓ].append(nrm[:200])
            mu, sd = float(nrm.mean()), float(nrm.std())
            thr = mu + 3 * sd
            per_layer_stats[ℓ]["mean"].append(mu)
            per_layer_stats[ℓ]["std"].append(sd)
            per_layer_stats[ℓ]["max"].append(float(nrm.max()))
            per_layer_stats[ℓ]["q90"].append(float(np.percentile(nrm, 90)))
            per_layer_stats[ℓ]["outlier_frac"].append(float((nrm > thr).mean()))
            per_layer_stats[ℓ]["n_vis"].append(int(len(nrm)))
            # spatial: top-10 high-norm patches -> relative grid coords
            k = min(10, len(nrm))
            top_idx = np.argpartition(nrm, -k)[-k:]
            for ti in top_idx:
                x = (ti % n_w) / max(n_w, 1)
                y = (ti // n_w) / max(n_h, 1)
                hx = min(39, int(x * 40)); hy = min(39, int(y * 40))
                spatial_hist[ℓ][hy, hx] += 1
    print(f"[P2-1] ok={n_ok}/{len(records)}")

    # ---- Aggregate ----
    summary = {}
    for ℓ in probe_layers:
        s = per_layer_stats[ℓ]
        if not s["mean"]: continue
        pooled = np.concatenate(per_layer_norms[ℓ]) if per_layer_norms[ℓ] else np.array([])
        summary[ℓ] = {
            "mean_norm": float(np.mean(s["mean"])),
            "std_norm": float(np.mean(s["std"])),
            "max_norm": float(np.mean(s["max"])),
            "q90_norm": float(np.mean(s["q90"])),
            "outlier_frac": float(np.mean(s["outlier_frac"])),
            "n_vis": float(np.mean(s["n_vis"])),
            "pooled_mean": float(pooled.mean()) if pooled.size else 0.0,
            "pooled_std": float(pooled.std()) if pooled.size else 0.0,
        }
    out = {"model_type": model_type, "disp": disp, "probe_layers": probe_layers,
           "per_layer": {str(k): v for k, v in summary.items()},
           "n_samples": n_ok, "spatial_hist_path": None}
    # save pooled norms for a few representative layers (for histogram plot)
    rep_layers = [probe_layers[len(probe_layers)//4], probe_layers[len(probe_layers)//2],
                  probe_layers[-1]]
    rep = {str(ℓ): np.concatenate(per_layer_norms[ℓ]).tolist()
           for ℓ in rep_layers if per_layer_norms[ℓ]}
    out["rep_layer_norms"] = rep
    # spatial hist saved separately (numpy)
    sp_path = OUT_DIR / f"norm_spatial_{model_type}.npz"
    np.savez(sp_path, **{str(ℓ): spatial_hist[ℓ] for ℓ in probe_layers})
    out["spatial_hist_path"] = str(sp_path)
    jout = OUT_DIR / f"norm_census_{model_type}.json"
    json.dump(out, jout.open("w"), indent=2)
    print(f"[P2-1] saved {jout}")
    _plot(model_type, disp, probe_layers, summary, rep, spatial_hist)
    return out


def _plot(model_type, disp, probe_layers, summary, rep, spatial_hist):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot] skip: {e}"); return
    plt.rcParams.update({"font.size": 9})
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    # (a) outlier fraction vs depth
    ax = axes[0]
    Ls = sorted(summary.keys(), key=int)
    ax.plot([int(l) for l in Ls], [summary[l]["outlier_frac"] for l in Ls],
            "o-", color="#d6604d", lw=1.8, ms=5)
    ax.set_xlabel("Probe layer"); ax.set_ylabel("Outlier frac (norm $>\\mu+3\\sigma$)")
    ax.set_title("(a) High-norm sink fraction vs depth")
    # (b) mean norm vs depth
    ax = axes[1]
    ax.plot([int(l) for l in Ls], [summary[l]["mean_norm"] for l in Ls],
            "o-", color="#2166ac", lw=1.8, ms=5)
    ax.set_xlabel("Probe layer"); ax.set_ylabel("Mean $\\|h_v\\|_2$ (pre-scale)")
    ax.set_title("(b) Norm growth with depth")
    # (c) norm histogram for representative layers
    ax = axes[2]
    colors = ["#2166ac", "#4dac26", "#d6604d"]
    for c, (l, vals) in zip(colors, rep.items()):
        if vals:
            ax.hist(vals, bins=50, density=True, histtype="step", lw=1.8,
                    color=c, label=f"L{l}")
    ax.set_xlabel("$\\|h_v\\|_2$ (pre-scale)"); ax.set_ylabel("density")
    ax.set_title("(c) Norm distribution (early/mid/late)")
    ax.legend(fontsize=7)
    fig.suptitle(f"{disp}: visual-patch norm census (Registers analog)", fontsize=10)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig_norm_census_{model_type}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[plot] saved fig_norm_census_{model_type}.{{pdf,png}}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_type", required=True, choices=list(MODEL_CFG))
    p.add_argument("--n_samples", type=int, default=100)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max_pixels", type=int, default=6_400_000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    run(args.model_type, args.n_samples, args.max_pixels, args.device, args.seed)


if __name__ == "__main__":
    main()
