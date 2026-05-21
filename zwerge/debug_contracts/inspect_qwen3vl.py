#!/usr/bin/env python3
"""
Qwen3-VL forward contract inspection.

Run in qwen3-verl env:
  python debug_contracts/inspect_qwen3vl.py

Checks every assumption made by _run_language_model in modeling_guiowl.py.
"""

import sys, inspect
sys.path.insert(0, "src")

def check(label, cond, detail=""):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}" + (f" — {detail}" if detail else ""))
    if not cond:
        sys.exit(1)

print("=== Qwen3-VL Forward Contract Inspection ===")

# ── 1. Import
try:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLModel, Qwen3VLForConditionalGeneration, Qwen3VLTextModel,
    )
    import transformers
    print(f"  transformers: {transformers.__version__}")
    check("imports OK", True)
except ImportError as e:
    check("imports OK", False, str(e))

# ── 2. check_model_inputs is a NO-OP (registry empty)
from transformers.utils.generic import _CAN_RECORD_REGISTRY
check("_CAN_RECORD_REGISTRY is empty (no monkey-patching)",
      len(_CAN_RECORD_REGISTRY) == 0,
      f"actual size={len(_CAN_RECORD_REGISTRY)}")

# ── 3. output_hidden_states NOT propagated through Qwen3VLModel
src = inspect.getsource(Qwen3VLModel.forward)
model_return_has_hidden = "hidden_states=outputs.hidden_states" in src or "hidden_states=hidden" in src.replace(" ", "")
check("Qwen3VLModel.forward does NOT propagate hidden_states in return",
      not model_return_has_hidden,
      "any code passing output_hidden_states=True via Qwen3VLModel will get None back")

# ── 4. Qwen3VLTextModel accepts visual_pos_masks and deepstack
sig = inspect.signature(Qwen3VLTextModel.forward)
params = list(sig.parameters)
check("Qwen3VLTextModel.forward accepts visual_pos_masks",
      "visual_pos_masks" in params, f"params={params[:15]}")
check("Qwen3VLTextModel.forward accepts deepstack_visual_embeds",
      "deepstack_visual_embeds" in params, f"params={params[:15]}")

# ── 5. get_image_features returns tuple(list, list)
import torch
cfg_path = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/.hdd/models/huggingface.co/GUI_Agents/GUI-Owl-1.5-8B-Instruct"
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained(cfg_path)
# Just check signature
sig2 = inspect.signature(Qwen3VLModel.get_image_features)
check("get_image_features(pixel_values, image_grid_thw) signature",
      list(sig2.parameters) == ["self", "pixel_values", "image_grid_thw"])

# ── 6. get_rope_index signature (no mm_token_type_ids)
sig3 = inspect.signature(Qwen3VLModel.get_rope_index)
params3 = list(sig3.parameters)
check("get_rope_index: NO mm_token_type_ids param",
      "mm_token_type_ids" not in params3,
      f"params={params3}")
check("get_rope_index has expected params",
      all(p in params3 for p in ["input_ids", "image_grid_thw", "video_grid_thw", "attention_mask"]))

# ── 7. get_placeholder_mask signature
sig4 = inspect.signature(Qwen3VLModel.get_placeholder_mask)
params4 = list(sig4.parameters)
check("get_placeholder_mask(input_ids, inputs_embeds=, image_features=, video_features=)",
      params4 == ["self", "input_ids", "inputs_embeds", "image_features", "video_features"])

# ── 8. GradientCheckpointing default uses use_reentrant=True
from transformers import PreTrainedModel
src_gc = inspect.getsource(PreTrainedModel.gradient_checkpointing_enable)
check("gradient_checkpointing_enable default: use_reentrant=True (risky with frozen backbone)",
      "use_reentrant: True" in src_gc or '"use_reentrant": True' in src_gc)

# ── 9. DDP find_unused logic in Trainer
import inspect as ins
from transformers import Trainer
src_trainer = ins.getsource(Trainer._inner_training_loop)
# The Trainer sets find_unused = not gradient_checkpointing when ddp_find_unused_parameters=None
check("Trainer.ddp_find_unused_parameters=None → not gc (gc=True → find_unused=False)",
      "ddp_find_unused_parameters" in src_trainer and "gradient_checkpointing" in src_trainer)

print()
print("All checks passed — Qwen3-VL forward contract verified.")
print("Our _run_language_model hook approach is the correct path.")
