#!/usr/bin/env bash
# ==============================================================================
# ZwerGe-UI — example end-to-end reproduction of ONE Table-2 cell
#   (UI-TARS-1.5-7B backbone @ ScreenSpot-Pro, margin-gated ensemble, tau=0.20)
# ==============================================================================
#
# This script is a WORKED SKETCH, not a plug-and-run harness: this minimal
# supplementary package deliberately omits the full checkpoint/benchmark I/O
# used by the paper's internal eval CLI (see README.md, "What is
# intentionally NOT included"). It shows exactly which Python calls, in
# which order, reproduce a single strict-success-rate number from Table 2,
# so a reader can wire it into their own checkpoint-loading /
# dataset-loading code.
#
# Prereqs:
#   1. pip install -r requirements.txt
#   2. Download the UI-TARS-1.5-7B backbone (Qwen2.5-VL) separately, e.g.
#        huggingface-cli download ByteDance-Seed/UI-TARS-1.5-7B --local-dir ./ui-tars-1.5-7b
#      (see the backbone's own model card for its license.)
#   3. Obtain (train, or otherwise acquire) the ZwerGe retrofit checkpoint:
#      the probe bank (~42M params) + fusion head (~200K params) that sit
#      on top of the frozen backbone (see modeling.py's
#      LayerWiseGroundingHead). This minimal package does not include
#      training code or trained checkpoints.
#   4. Obtain the ScreenSpot-Pro benchmark (Li et al. 2025) images + eval.json
#      (gt_bbox, instruction, image_path, image_size per sample).
# ==============================================================================

set -euo pipefail

BACKBONE_PATH="${BACKBONE_PATH:-./ui-tars-1.5-7b}"          # frozen Qwen2.5-VL backbone
RETROFIT_CKPT="${RETROFIT_CKPT:-./ckpt/uitars_zwerge}"      # probe bank + fusion head weights
EVAL_JSON="${EVAL_JSON:-./ScreenSpot-Pro/eval.json}"        # benchmark annotations
IMAGE_ROOT="${IMAGE_ROOT:-./ScreenSpot-Pro}"                # benchmark images
DEVICE="${DEVICE:-cuda:0}"

python - "$BACKBONE_PATH" "$RETROFIT_CKPT" "$EVAL_JSON" "$IMAGE_ROOT" "$DEVICE" <<'PYEOF'
import json
import sys

import torch
from PIL import Image

# The deployed decoder (margin-gated fusion<->native ensemble) and its
# default config, matching the paper's main-table setting:
#   --p2p_ensemble_gate margin --p2p_ensemble_margin_thr 0.20
from ensemble import run_p2p_ensemble, default_p2p_cfg
from inference import point_in_bbox, do_boxes_overlap

backbone_path, retrofit_ckpt, eval_json, image_root, device_str = sys.argv[1:6]
device = torch.device(device_str)

# ── 1. Load the frozen backbone + grounding head + processor. ───────────────
# This step is intentionally left to the user's own checkpoint-loading code
# (see README.md's "What is intentionally NOT included"): it must produce
# an object exposing exactly the RetrofitInference interface used below:
#   .model                    a RetrofitModelMixin-based nn.Module (frozen
#                              backbone + trained layerwise_grounding_head)
#   .processor                the backbone's AutoProcessor
#   .system_message / .ground_response / .user_prompt_template
#   .predict_layerwise(...) / .predict_zoom_backbone(...)
#   .parse_backbone_coordinate(...)   (backbone-specific native-output parser)
#
# A concrete example wiring (illustrative — fill in your own prompt table):
#
#   from modeling import get_retrofit_model_class
#   from inference import RetrofitInference
#   from transformers import AutoProcessor, AutoConfig
#
#   class UITarsInference(RetrofitInference):
#       model_type, merge_size, patch_size = "uitars", 2, 14
#       def parse_backbone_coordinate(self, raw_text, crop_w_resized=None, crop_h_resized=None):
#           import re
#           m = re.search(r"<\|box_start\|>\((\d+),\s*(\d+)\)<\|box_end\|>", raw_text)
#           if not m:
#               return None
#           x, y = int(m.group(1)), int(m.group(2))
#           return x / crop_w_resized, y / crop_h_resized
#
#   ModelClass = get_retrofit_model_class("qwen2_5_vl")
#   config = AutoConfig.from_pretrained(retrofit_ckpt)
#   model  = ModelClass.from_pretrained(retrofit_ckpt, config=config,
#                                        attn_implementation="flash_attention_2",
#                                        torch_dtype=torch.bfloat16).to(device).eval()
#   processor = AutoProcessor.from_pretrained(backbone_path)
#   grounder = UITarsInference(
#       model=model, processor=processor,
#       system_message=..., ground_response=...,   # from your prompt table
#   )
raise SystemExit(
    "Fill in checkpoint loading per the comment above, then remove this "
    "line. This script only demonstrates the DECODE-TIME call sequence "
    "(run_p2p_ensemble) that reproduces Table 2's +ZwerGe column."
)

# ── 2. p2p / ensemble config matching the paper's main-table setting. ───────
p2p_cfg = default_p2p_cfg(
    ensemble_gate="margin",        # --p2p_ensemble_gate margin
    ensemble_margin_thr=0.20,      # --p2p_ensemble_margin_thr 0.20 (main results)
)

# ── 3. Evaluate over ScreenSpot-Pro under the strict single-point protocol. ──
with open(eval_json) as f:
    records = json.load(f)

n_correct, n_total = 0, 0
for rec in records:
    img = Image.open(f"{image_root}/{rec['image_path']}").convert("RGB")
    W, H = float(rec["image_size"][0]), float(rec["image_size"][1])
    x1, y1, x2, y2 = rec["gt_bbox"]
    gt_bbox_norm = (x1 / W, y1 / H, x2 / W, y2 / H)

    out_point, _, _, meta, native_point = run_p2p_ensemble(
        grounder=grounder, image=img, instruction=rec["instruction"], device=device,
        p2p_cfg=p2p_cfg, zoom_max_new_tokens=256,
        activation_threshold=0.3, topk=3,
    )

    px, py = out_point
    hit = point_in_bbox(px, py, gt_bbox_norm)   # strict single-point success (overlap@1)
    n_correct += int(hit)
    n_total += 1

print(f"Strict success rate (overlap@1): {100.0 * n_correct / max(n_total, 1):.2f}%  "
      f"(n={n_total})")
PYEOF
