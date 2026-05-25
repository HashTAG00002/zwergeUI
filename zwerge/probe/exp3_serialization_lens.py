"""
exp3_serialization_lens.py — Experiment 3: Spatial Lens vs. Serialization Lens

Two-pass analysis per sample:
  Pass 1 (Spatial)   : predict_layerwise() → per-layer hit@1 and target_mass
  Pass 2 (Logit Lens): native-format GT prefill + forward hooks → per-layer coord NLL

Output JSONL per sample with per-layer spatial and serialization metrics.
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

# ── sys.path: probe_utils shares the same setup ────────────────────────────────
_PROBE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.abspath(os.path.join(_PROBE_DIR, "../../.."))
_EVAL_DIR   = os.path.join(_REPO_ROOT, "zwerge", "eval")
_SRC_DIR    = os.path.join(_REPO_ROOT, "zwerge", "src")
for _d in [_PROBE_DIR, _EVAL_DIR, _SRC_DIR]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from probe_utils import (
    build_native_gt_response,
    compute_spatial_metrics_from_pred,
    find_coord_token_positions,
    get_done_ids,
    get_inference_class,
    load_eval_records,
    logit_lens_nll_hooks,
    append_jsonl,
    sample_records,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_native_inputs(grounder, image: Image.Image, instruction: str,
                          native_response: str, max_pixels: int, device):
    """Build model inputs for the native-format logit-lens pass."""
    from inference_base import build_zwerge_inputs, _ZOOM_NOT_SET

    sys_msg = grounder._zoom_native_system_message
    if sys_msg is _ZOOM_NOT_SET:
        sys_msg = grounder.system_message

    user_tmpl = grounder._zoom_native_user_template
    if user_tmpl is _ZOOM_NOT_SET:
        user_tmpl = grounder.user_prompt_template

    return build_zwerge_inputs(
        image=image,
        instruction=instruction,
        processor=grounder.processor,
        system_message=sys_msg,
        ground_response=native_response,
        max_pixels=max_pixels,
        user_prompt_template=user_tmpl,
    )


def _parse_gt_bbox_norm(rec: dict):
    """Return (x1,y1,x2,y2) normalized to [0,1], or None on failure."""
    v = rec.get("gt_bbox_norm")
    if v and len(v) == 4:
        vals = [float(x) for x in v]
        if max(vals) > 1.5:
            vals = [x / 1000.0 for x in vals]
        return tuple(vals)
    # Fall back: compute from gt_bbox + image_size
    bbox = rec.get("gt_bbox") or rec.get("bbox")
    sz   = rec.get("image_size")
    if bbox and sz and len(bbox) == 4 and len(sz) == 2:
        W, H = float(sz[0]), float(sz[1])
        return (bbox[0]/W, bbox[1]/H, bbox[2]/W, bbox[3]/H)
    return None


def _get_image_path(rec: dict, image_root: str) -> str:
    for key in ("image_path", "img_path", "image_filename", "image"):
        v = rec.get(key)
        if v:
            p = os.path.join(image_root, v)
            if os.path.exists(p):
                return p
    return None


# ── Main experiment loop ──────────────────────────────────────────────────────

def run_exp3(
    ckpt_path: str,
    model_type: str,
    eval_json: str,
    image_root: str,
    output_path: str,
    n_samples: int = 200,
    max_pixels: int = 6_400_000,
    device_str: str = "cuda:0",
    seed: int = 42,
):
    device = torch.device(device_str)

    # ── 1. Load model ──────────────────────────────────────────────────────────
    print(f"[exp3] Loading {model_type} from {ckpt_path} …")
    InfClass = get_inference_class(model_type)
    grounder = InfClass.from_checkpoint(
        ckpt_path, device=device_str, max_pixels=max_pixels,
    )
    grounder.model.eval()
    grounder.model.to(device)

    probe_layers = list(grounder.model.layerwise_grounding_head.probe_layers)
    print(f"[exp3] Probe layers: {probe_layers}")

    # ── 2. Load and sample records ─────────────────────────────────────────────
    records = load_eval_records(eval_json)
    records = sample_records(records, n_samples, seed=seed)
    print(f"[exp3] Sampled {len(records)} records.")

    # Resume support
    done_ids = get_done_ids(output_path, id_key="sample_id")
    print(f"[exp3] Already done: {len(done_ids)} samples.")

    n_ok = 0
    n_skip_img = 0
    n_skip_coord = 0
    n_err = 0

    for i, rec in enumerate(tqdm(records, desc=f"exp3/{model_type}")):
        sample_id = rec.get("id") or rec.get("sample_id") or str(i)
        if sample_id in done_ids:
            continue

        # ── Image ──────────────────────────────────────────────────────────────
        img_path = _get_image_path(rec, image_root)
        if img_path is None:
            n_skip_img += 1
            continue
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            warnings.warn(f"[exp3] Cannot open {img_path}: {e}")
            n_skip_img += 1
            continue

        instruction = rec.get("instruction") or rec.get("query") or ""
        gt_bbox_norm = _parse_gt_bbox_norm(rec)
        if gt_bbox_norm is None:
            n_skip_coord += 1
            continue

        # ── Pass 1: Spatial probe ──────────────────────────────────────────────
        try:
            with torch.no_grad():
                pred = grounder.predict_layerwise(img, instruction, device=device)
        except Exception as e:
            warnings.warn(f"[exp3] predict_layerwise failed for {sample_id}: {e}")
            n_err += 1
            continue

        spatial_metrics = compute_spatial_metrics_from_pred(pred, gt_bbox_norm)

        # ── Pass 2: Logit-lens (native-format forward) ─────────────────────────
        gt_bbox  = rec.get("gt_bbox") or rec.get("bbox")
        img_size = rec.get("image_size")

        if gt_bbox is None or img_size is None:
            n_skip_coord += 1
            continue

        try:
            native_resp = build_native_gt_response(gt_bbox, img_size, model_type)
        except ValueError as e:
            warnings.warn(f"[exp3] build_native_gt_response failed: {e}")
            n_skip_coord += 1
            continue

        try:
            native_inputs = _build_native_inputs(
                grounder, img, instruction, native_resp, max_pixels, device
            )
        except Exception as e:
            warnings.warn(f"[exp3] build_zwerge_inputs failed for {sample_id}: {e}")
            n_err += 1
            continue

        input_ids_1d = native_inputs["input_ids"][0]
        coord_pos, coord_ids, proto_pos, proto_ids = find_coord_token_positions(
            input_ids_1d, grounder.processor.tokenizer, model_type
        )

        if len(coord_ids) == 0:
            n_skip_coord += 1
            continue

        try:
            coord_nlls = logit_lens_nll_hooks(
                grounder.model, native_inputs, coord_pos, coord_ids,
                probe_layers, device
            )
        except Exception as e:
            warnings.warn(f"[exp3] logit_lens_nll_hooks (coord) failed for {sample_id}: {e}")
            n_err += 1
            continue

        # Protocol tokens (optional — skip gracefully if empty)
        protocol_nlls = {}
        if proto_ids:
            try:
                protocol_nlls = logit_lens_nll_hooks(
                    grounder.model, native_inputs, proto_pos, proto_ids,
                    probe_layers, device
                )
            except Exception:
                pass

        # ── Save result ────────────────────────────────────────────────────────
        result = {
            "sample_id":      sample_id,
            "model_type":     model_type,
            "instruction":    instruction,
            "gt_bbox":        gt_bbox,
            "gt_bbox_norm":   list(gt_bbox_norm),
            "image_size":     img_size,
            "probe_layers":   probe_layers,
            # Spatial pass
            "spatial_hit1":   spatial_metrics["hit1_per_layer"],
            "spatial_mass":   spatial_metrics["mass_per_layer"],
            # Logit-lens pass
            "coord_nll":      [coord_nlls.get(l) for l in probe_layers],
            "protocol_nll":   [protocol_nlls.get(l) for l in probe_layers],
            # Metadata for plotting
            "n_coord_tokens":    len(coord_ids),
            "n_protocol_tokens": len(proto_ids),
        }
        append_jsonl(output_path, result)
        done_ids.add(sample_id)
        n_ok += 1

    print(
        f"[exp3] Done. ok={n_ok}, skip_img={n_skip_img}, "
        f"skip_coord={n_skip_coord}, err={n_err}"
    )
    print(f"[exp3] Output: {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Exp 3: Spatial Lens vs. Serialization Lens"
    )
    p.add_argument("--ckpt",        required=True, help="Path to ZwerGe A7 checkpoint")
    p.add_argument("--model_type",  required=True,
                   choices=["uitars", "guiowl7b", "guiowl", "uivenus", "uitars1"],
                   help="Model type")
    p.add_argument("--eval_json",   required=True, help="Path to eval.json")
    p.add_argument("--image_root",  required=True, help="Root directory for images")
    p.add_argument("--output",      required=True, help="Output .jsonl path")
    p.add_argument("--n_samples",   type=int, default=200)
    p.add_argument("--max_pixels",  type=int, default=6_400_000)
    p.add_argument("--device",      default="cuda:0")
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_exp3(
        ckpt_path=args.ckpt,
        model_type=args.model_type,
        eval_json=args.eval_json,
        image_root=args.image_root,
        output_path=args.output,
        n_samples=args.n_samples,
        max_pixels=args.max_pixels,
        device_str=args.device,
        seed=args.seed,
    )
