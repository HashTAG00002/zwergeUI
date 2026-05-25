"""
exp4_counterfactual.py — Experiment 4: Same-App Counterfactual Target Switch

For each (img_A, instruction_A) / (img_B, instruction_B) pair from the same
application (exact image_size match, non-overlapping GT bboxes):

  Run img_A with instruction_A  →  mass_A_given_IA  (control)
  Run img_A with instruction_B  →  mass_A_given_IB, mass_B_given_IB  (counterfactual)

Key metrics per probe layer:
  switch_score             = mass_B_given_IB - mass_A_given_IB
  old_target_suppression   = mass_A_given_IA - mass_A_given_IB

A positive switch_score means the posterior shifted toward instruction_B's target.
A positive old_target_suppression means instruction_B suppressed instruction_A's region.
Both peaking at mid-layers confirms instruction-conditioned (not generic saliency) grounding.
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

# ── sys.path setup ─────────────────────────────────────────────────────────────
_PROBE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.abspath(os.path.join(_PROBE_DIR, "../../.."))
_EVAL_DIR   = os.path.join(_REPO_ROOT, "zwerge", "eval")
_SRC_DIR    = os.path.join(_REPO_ROOT, "zwerge", "src")
for _d in [_PROBE_DIR, _EVAL_DIR, _SRC_DIR]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from probe_utils import (
    append_jsonl,
    compute_target_mass,
    get_done_ids,
    get_inference_class,
    load_eval_records,
    select_counterfactual_pairs,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_gt_bbox_norm(rec: dict):
    v = rec.get("gt_bbox_norm")
    if v and len(v) == 4:
        vals = [float(x) for x in v]
        if max(vals) > 1.5:
            vals = [x / 1000.0 for x in vals]
        return tuple(vals)
    bbox = rec.get("gt_bbox") or rec.get("bbox")
    sz   = rec.get("image_size")
    if bbox and sz and len(bbox) == 4 and len(sz) == 2:
        W, H = float(sz[0]), float(sz[1])
        return (bbox[0]/W, bbox[1]/H, bbox[2]/W, bbox[3]/H)
    return None


def _get_image_path(rec: dict, image_root: str):
    for key in ("image_path", "img_path", "image_filename", "image"):
        v = rec.get(key)
        if v:
            p = os.path.join(image_root, v)
            if os.path.exists(p):
                return p
    return None


def _pair_id(idx_A: int, idx_B: int) -> str:
    return f"{idx_A}_{idx_B}"


# ── Main experiment loop ──────────────────────────────────────────────────────

def run_exp4(
    ckpt_path: str,
    model_type: str,
    eval_json: str,
    image_root: str,
    output_path: str,
    max_pairs: int = 150,
    device_str: str = "cuda:0",
    seed: int = 42,
):
    device = torch.device(device_str)

    # ── 1. Load model ──────────────────────────────────────────────────────────
    print(f"[exp4] Loading {model_type} from {ckpt_path} …")
    InfClass = get_inference_class(model_type)
    grounder = InfClass.from_checkpoint(ckpt_path, device=device_str)
    grounder.model.eval()
    grounder.model.to(device)

    probe_layers = list(grounder.model.layerwise_grounding_head.probe_layers)
    print(f"[exp4] Probe layers: {probe_layers}")

    # ── 2. Load records and build pairs ───────────────────────────────────────
    records = load_eval_records(eval_json)
    print(f"[exp4] Loaded {len(records)} records.")

    pairs = select_counterfactual_pairs(records, max_pairs=max_pairs, seed=seed)
    print(f"[exp4] Selected {len(pairs)} counterfactual pairs.")

    # Resume support
    done_ids = get_done_ids(output_path, id_key="pair_id")
    print(f"[exp4] Already done: {len(done_ids)} pairs.")

    n_ok = 0
    n_skip = 0
    n_err  = 0

    for idx_A, idx_B in tqdm(pairs, desc=f"exp4/{model_type}"):
        pair_id = _pair_id(idx_A, idx_B)
        if pair_id in done_ids:
            continue

        rec_A = records[idx_A]
        rec_B = records[idx_B]

        img_path_A = _get_image_path(rec_A, image_root)
        if img_path_A is None:
            n_skip += 1
            continue

        try:
            img_A = Image.open(img_path_A).convert("RGB")
        except Exception as e:
            warnings.warn(f"[exp4] Cannot open {img_path_A}: {e}")
            n_skip += 1
            continue

        instruction_A = rec_A.get("instruction") or rec_A.get("query") or ""
        instruction_B = rec_B.get("instruction") or rec_B.get("query") or ""

        bbox_A_norm = _parse_gt_bbox_norm(rec_A)
        bbox_B_norm = _parse_gt_bbox_norm(rec_B)

        if bbox_A_norm is None or bbox_B_norm is None:
            n_skip += 1
            continue

        # ── Pass 1: img_A + instruction_A (control) ────────────────────────────
        try:
            with torch.no_grad():
                pred_AA = grounder.predict_layerwise(img_A, instruction_A, device=device)
        except Exception as e:
            warnings.warn(f"[exp4] predict_layerwise AA failed for pair {pair_id}: {e}")
            n_err += 1
            continue

        # ── Pass 2: img_A + instruction_B (counterfactual) ────────────────────
        try:
            with torch.no_grad():
                pred_AB = grounder.predict_layerwise(img_A, instruction_B, device=device)
        except Exception as e:
            warnings.warn(f"[exp4] predict_layerwise AB failed for pair {pair_id}: {e}")
            n_err += 1
            continue

        n_w = pred_AA["n_width"]
        n_h = pred_AA["n_height"]

        # Safety: verify n_width/n_height are consistent between passes
        if pred_AB["n_width"] != n_w or pred_AB["n_height"] != n_h:
            warnings.warn(f"[exp4] Grid size mismatch for pair {pair_id}, skipping.")
            n_skip += 1
            continue

        # ── Compute per-layer masses ───────────────────────────────────────────
        mass_A_given_IA = []
        mass_A_given_IB = []
        mass_B_given_IB = []
        switch_score    = []
        old_target_supp = []

        probs_AA = pred_AA["per_layer_probs"]
        probs_AB = pred_AB["per_layer_probs"]

        for li in range(len(probe_layers)):
            p_AA = probs_AA[li]
            p_AB = probs_AB[li]

            mAA = compute_target_mass(p_AA, bbox_A_norm, n_w, n_h)
            mAB = compute_target_mass(p_AB, bbox_A_norm, n_w, n_h)
            mBB = compute_target_mass(p_AB, bbox_B_norm, n_w, n_h)

            mass_A_given_IA.append(mAA)
            mass_A_given_IB.append(mAB)
            mass_B_given_IB.append(mBB)
            switch_score.append(mBB - mAB)       # positive = posterior moved toward B
            old_target_supp.append(mAA - mAB)    # positive = instruction B suppressed A's region

        result = {
            "pair_id":          pair_id,
            "idx_A":            idx_A,
            "idx_B":            idx_B,
            "model_type":       model_type,
            "app":              rec_A.get("application", rec_A.get("app", "")),
            "image_size":       rec_A.get("image_size"),
            "instruction_A":    instruction_A,
            "instruction_B":    instruction_B,
            "bbox_A_norm":      list(bbox_A_norm),
            "bbox_B_norm":      list(bbox_B_norm),
            "probe_layers":     probe_layers,
            "mass_A_given_IA":  mass_A_given_IA,
            "mass_A_given_IB":  mass_A_given_IB,
            "mass_B_given_IB":  mass_B_given_IB,
            "switch_score":     switch_score,
            "old_target_suppression": old_target_supp,
        }
        append_jsonl(output_path, result)
        done_ids.add(pair_id)
        n_ok += 1

    print(f"[exp4] Done. ok={n_ok}, skip={n_skip}, err={n_err}")
    print(f"[exp4] Output: {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Exp 4: Same-App Counterfactual Target Switch"
    )
    p.add_argument("--ckpt",       required=True, help="Path to ZwerGe A7 checkpoint")
    p.add_argument("--model_type", required=True,
                   choices=["uitars", "guiowl7b", "guiowl", "uivenus", "uitars1"],
                   help="Model type")
    p.add_argument("--eval_json",  required=True, help="Path to eval.json")
    p.add_argument("--image_root", required=True, help="Root directory for images")
    p.add_argument("--output",     required=True, help="Output .jsonl path")
    p.add_argument("--max_pairs",  type=int, default=150)
    p.add_argument("--device",     default="cuda:0")
    p.add_argument("--seed",       type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_exp4(
        ckpt_path=args.ckpt,
        model_type=args.model_type,
        eval_json=args.eval_json,
        image_root=args.image_root,
        output_path=args.output,
        max_pairs=args.max_pairs,
        device_str=args.device,
        seed=args.seed,
    )
