"""
对比 uitars 和 guiowl 在相同样本上的推理分布

重点验证：
1. p_final 的 entropy / max 值
2. 哪一层信号最强
3. training system_message vs eval system_message 的影响
"""
import os, sys, json, warnings
import torch
from PIL import Image

sys.path.insert(0, "eval")
sys.path.insert(0, "src")

GUIOWL_CKPT = (
    "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03"
    "/.hdd/ckpt/zwerge/guiowl_grounding50k_A3-gaussian_cos_meta_20260521_210923"
    "/checkpoint-800"
)
UITARS_CKPT = (
    "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03"
    "/.hdd/ckpt/zwerge/uitars7b_grounding50k_A3-gaussian_cos_meta_L18-27_20260520_095304"
    "/checkpoint-2193"
)
SS_PRO_EVAL_JSON = (
    "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03"
    "/datasets/evaluation/ScreenSpot-Pro/eval.json"
)

# ── Training-time system message for guiowl (from args.json) ──────────────────
# NOTE: This is what the model was TRAINED with — different from GUI_OWL_SYSTEM_PROMPT
# used in constants.py (simplified)!
GUIOWL_TRAIN_SYSTEM_PROMPT = '''# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "computer_use", "description": "Use a mouse to interact with a computer.\n* The screen\'s resolution is 1000x1000.\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don\'t click boxes on their edges unless asked.\n* don\'t use any other computer use tool like type, key, scroll, left_click_drag and so on.\n* you can only use the left_click and mouse_move action to interact with the computer. if you can\'t find the element, you should terminate the task and report the failure.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:\n* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.\n* `left_click`: Click the left mouse button with coordinate (x, y) pixel coordinate on the screen.", "enum": ["mouse_move", "left_click"], "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=mouse_move` and `action=left_click`.", "type": "array"}}, "required": ["action"], "type": "object"}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

Additionally, if you think the task is infeasible (e.g., the task is not related to the image), return <tool_call>
{"name": "computer_use", "arguments": {"action": "terminate", "status": "failure"}}
</tool_call>'''


def load_sample(n=0):
    with open(SS_PRO_EVAL_JSON) as f:
        data = json.load(f)
    s = data[n]
    base_dir = os.path.dirname(SS_PRO_EVAL_JSON)
    img_path = os.path.join(base_dir, s["image_path"])
    return Image.open(img_path).convert("RGB"), s["instruction"], s.get("gt_bbox_norm"), s


def run_predict(model, processor, sys_msg, grd_resp, image, instruction, device, model_label):
    from inference_base import grid_thw_to_nwh, build_zwerge_inputs, scores_to_point_and_topk

    inputs = build_zwerge_inputs(
        image=image, instruction=instruction, processor=processor,
        system_message=sys_msg, ground_response=grd_resp,
    )
    input_ids      = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None: attention_mask = attention_mask.to(device)
    pixel_values   = inputs.get("pixel_values")
    if pixel_values is not None: pixel_values = pixel_values.to(device, dtype=model.dtype)
    image_grid_thw = inputs.get("image_grid_thw")
    if image_grid_thw is not None: image_grid_thw = image_grid_thw.to(device)

    if image_grid_thw is not None:
        merge_size = getattr(processor.image_processor, "merge_size", 2)
        n_width, n_height = grid_thw_to_nwh(image_grid_thw, merge_size=merge_size)
    else:
        n_width, n_height = 100, 50

    token_ids_1d = input_ids[0]

    with torch.no_grad():
        all_hs = model._forward_hidden_states_for_grounding(
            input_ids=input_ids, attention_mask=attention_mask,
            pixel_values=pixel_values, image_grid_thw=image_grid_thw, device=device,
        )

    anchor_idx, anchor_strategy = model._find_ground_anchor(token_ids=token_ids_1d, verbose=False)
    visual_indices = model._get_visual_indices(token_ids_1d)

    sample_hs = tuple(h[0] if h is not None else None for h in all_hs)
    with torch.no_grad():
        head_out = model.layerwise_grounding_head(
            all_hidden_states=sample_hs,
            ground_token_idx=anchor_idx,
            visual_indices=visual_indices,
            labels=None,
        )

    p_final = head_out["p_final"]
    omega   = head_out["omega"]
    per_layer_probs = head_out["per_layer_probs"]

    entropy = -(p_final * torch.log(p_final.clamp(min=1e-8))).sum().item()
    max_ent = torch.log(torch.tensor(float(p_final.shape[0]))).item()

    print(f"\n  [{model_label}] seq_len={input_ids.shape[1]}, n_vis={visual_indices.numel()}, "
          f"n_width={n_width}, n_height={n_height}")
    print(f"    anchor: idx={anchor_idx}, strategy={anchor_strategy.value}")
    print(f"    p_final: max={p_final.max().item():.5f}, entropy={entropy:.3f} / {max_ent:.3f}")
    print(f"    omega: {[f'{o:.3f}' for o in omega.cpu().tolist()]}")
    best, topk = scores_to_point_and_topk(p_final, n_width, n_height, 0.3, 3)
    print(f"    pred: {best}")

    # Per-layer best
    layer_stats = []
    for i, (li, p_l) in enumerate(zip(model.layerwise_grounding_head.probe_layers, per_layer_probs)):
        ent_l = -(p_l * torch.log(p_l.clamp(min=1e-8))).sum().item()
        max_l = p_l.max().item()
        layer_stats.append((li, max_l, ent_l, omega[i].item()))
    # Sort by entropy (lower = better focused)
    layer_stats.sort(key=lambda x: x[1], reverse=True)  # sort by max descending
    print(f"    Top-3 layers (by max):")
    for li, max_l, ent_l, om in layer_stats[:3]:
        print(f"      L{li}: max={max_l:.5f}, entropy={ent_l:.3f}, omega={om:.3f}")

    return best, p_final


def load_model(ckpt_path, model_type, device="cuda:0"):
    from zwerge_retrofit import get_model_class
    from transformers import AutoProcessor, AutoConfig

    ModelClass = get_model_class(model_type)
    config = AutoConfig.from_pretrained(ckpt_path)
    model = ModelClass.from_pretrained(
        ckpt_path, config=config,
        attn_implementation="sdpa", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )
    model.setup_special_token_ids(
        ground_token_id=config.ground_token_id,
        pointer_start_token_id=config.pointer_start_token_id,
        vision_end_token_id=config.vision_end_token_id,
        reinit_grounding_head=False,
    )
    model.config.use_cache = False
    model = model.to(device=device)
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)

    processor = AutoProcessor.from_pretrained(ckpt_path)
    return model, processor


if __name__ == "__main__":
    device = "cuda:0"

    print("\n" + "="*72)
    print("  Loading models...")
    print("="*72)

    # Load guiowl
    print("[GUIOwl] Loading...")
    model_owl, proc_owl = load_model(GUIOWL_CKPT, "guiowl", device)

    # Load uitars
    print("[UITARs] Loading...")
    model_tars, proc_tars = load_model(UITARS_CKPT, "uitars", device)

    from zwerge_retrofit.constants import (
        MODEL_TYPE_CONSTANTS, GUI_OWL_GROUND_RESPONSE, GUI_OWL_SYSTEM_PROMPT,
        GROUNDING_SYSTEM_MESSAGE, GROUND_RESPONSE_CLICK,
    )

    for n in range(3):
        print(f"\n{'='*72}")
        print(f"  SAMPLE #{n}")
        print(f"{'='*72}")
        image, instruction, gt_bbox_norm, sample = load_sample(n)
        print(f"  instruction: {instruction[:80]}")
        if gt_bbox_norm:
            b = gt_bbox_norm
            if max(b) > 1: b = [v/1000 for v in b]
            print(f"  gt_bbox_norm (0-1): {[f'{v:.4f}' for v in b]}")

        print("\n--- GUIOwl: EVAL system_message (constants.GUI_OWL_SYSTEM_PROMPT) ---")
        run_predict(model_owl, proc_owl,
                    GUI_OWL_SYSTEM_PROMPT, GUI_OWL_GROUND_RESPONSE,
                    image, instruction, torch.device(device), "guiowl/eval-sys")

        print("\n--- GUIOwl: TRAIN system_message (from args.json) ---")
        run_predict(model_owl, proc_owl,
                    GUIOWL_TRAIN_SYSTEM_PROMPT, GUI_OWL_GROUND_RESPONSE,
                    image, instruction, torch.device(device), "guiowl/train-sys")
