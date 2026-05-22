"""
debug_guiowl_inference.py
=========================
深度 debug GUI-Owl 推理链，逐步验证所有环节：
  1. token 序列结构（anchor 位置、visual token 数量）
  2. image_grid_thw → n_width/n_height
  3. hidden state 的 shape 和数值范围
  4. p_final 分布（entropy, max, argmax 位置）
  5. 与 uitars 相同样本的对比

用法：
  conda activate qwen3-verl
  cd /mnt/.../zwerge/code/zwerge
  python debug_guiowl_inference.py
"""

import os
import sys
import warnings
import json

import torch
from PIL import Image

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR   = os.path.join(SCRIPT_DIR, "eval")
SRC_DIR    = os.path.join(SCRIPT_DIR, "src")
sys.path.insert(0, EVAL_DIR)
sys.path.insert(0, SRC_DIR)

# ── Checkpoints ───────────────────────────────────────────────────────────────
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

# ── Test image ─────────────────────────────────────────────────────────────────
SS_PRO_EVAL_JSON = (
    "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03"
    "/datasets/evaluation/ScreenSpot-Pro/eval.json"
)

# ── Main ──────────────────────────────────────────────────────────────────────

def load_guiowl(device="cuda:0"):
    """Load guiowl retrofit model."""
    from inference_base import RetrofitInference, grid_thw_to_nwh
    from zwerge_retrofit import get_model_class
    from zwerge_retrofit.constants import MODEL_TYPE_CONSTANTS
    from transformers import AutoProcessor, AutoConfig

    cls = type("GUIOwlRetrofitInference", (RetrofitInference,), {
        "model_type": "guiowl", "merge_size": 2, "patch_size": 16
    })

    ModelClass = get_model_class("guiowl")
    config = AutoConfig.from_pretrained(GUIOWL_CKPT)

    print(f"[GUIOwl] probe_layers: {config.probe_layers}")
    print(f"[GUIOwl] grounding_proj_dim: {config.grounding_proj_dim}")
    print(f"[GUIOwl] ground_token_id: {config.ground_token_id}")
    print(f"[GUIOwl] pointer_start_token_id: {config.pointer_start_token_id}")
    print(f"[GUIOwl] vision_end_token_id: {config.vision_end_token_id}")
    print(f"[GUIOwl] image_token_id: {config.image_token_id}")
    text_cfg = getattr(config, "text_config", config)
    print(f"[GUIOwl] n_layers: {getattr(text_cfg, 'num_hidden_layers', 'N/A')}")
    print(f"[GUIOwl] hidden_size: {getattr(text_cfg, 'hidden_size', 'N/A')}")

    model = ModelClass.from_pretrained(
        GUIOWL_CKPT, config=config,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
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
    for p in model.parameters():
        p.requires_grad_(False)

    processor = AutoProcessor.from_pretrained(GUIOWL_CKPT)
    constants = MODEL_TYPE_CONSTANTS["guiowl"]

    return model, processor, constants, cls


def load_sample(n=0):
    """Load n-th sample from SS-Pro eval.json."""
    with open(SS_PRO_EVAL_JSON) as f:
        data = json.load(f)
    sample = data[n]
    base_dir = os.path.dirname(SS_PRO_EVAL_JSON)
    img_path = os.path.join(base_dir, sample["image_path"])
    image = Image.open(img_path).convert("RGB")
    instruction = sample["instruction"]
    gt_bbox_norm = sample.get("gt_bbox_norm", None)
    return image, instruction, gt_bbox_norm, sample


def run_guiowl_debug(n_samples=3, device="cuda:0"):
    print("=" * 72)
    print("  GUI-Owl DEBUG")
    print("=" * 72)

    model, processor, constants, InfCls = load_guiowl(device=device)
    sys_msg   = constants["system_message"]
    grd_resp  = constants["ground_response"]
    merge_sz  = constants["merge_size"]
    dev       = torch.device(device)

    print(f"\n[Config] ground_response repr:\n  {repr(grd_resp[:100])}")
    print(f"[Config] system_message (first 100 chars):\n  {repr(sys_msg[:100])}")

    from inference_base import grid_thw_to_nwh, build_zwerge_inputs, scores_to_point_and_topk
    from inference_zwerge import get_prediction_region_point

    for n in range(n_samples):
        print(f"\n{'─'*60}")
        print(f"  Sample #{n}")
        print(f"{'─'*60}")

        image, instruction, gt_bbox_norm, sample = load_sample(n)
        print(f"  instruction: {instruction[:80]}")
        print(f"  image_size: {image.size}")
        if gt_bbox_norm:
            # gt_bbox_norm might be in [0,1000] or [0,1]
            b = gt_bbox_norm
            print(f"  gt_bbox_norm: {b}")

        # ── Step 1: Build inputs ────────────────────────────────────────────
        inputs = build_zwerge_inputs(
            image=image, instruction=instruction, processor=processor,
            system_message=sys_msg, ground_response=grd_resp,
        )
        input_ids      = inputs["input_ids"].to(dev)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(dev)
        pixel_values   = inputs.get("pixel_values")
        if pixel_values is not None:
            pixel_values = pixel_values.to(dev, dtype=model.dtype)
        image_grid_thw = inputs.get("image_grid_thw")
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.to(dev)

        print(f"\n  [Tokens]")
        print(f"    seq_len        = {input_ids.shape[1]}")
        if image_grid_thw is not None:
            print(f"    image_grid_thw = {image_grid_thw}")
            n_width, n_height = grid_thw_to_nwh(image_grid_thw, merge_size=merge_sz)
            print(f"    n_width={n_width}, n_height={n_height}")
            print(f"    total vis tokens (grid) = {n_width * n_height}")
        if pixel_values is not None:
            print(f"    pixel_values.shape = {pixel_values.shape}")

        # Count tokens in sequence
        tok1d = input_ids[0]
        img_tok_id = model.config.image_token_id
        vis_count = (tok1d == img_tok_id).sum().item()
        print(f"    <|image_pad|> in seq = {vis_count}")

        # Check vision_end token
        ve_id = model._vision_end_token_id
        if ve_id is not None:
            ve_pos = (tok1d == ve_id).nonzero(as_tuple=False)
            print(f"    <|vision_end|> positions: {ve_pos.flatten().tolist()}")

        # Check ground token
        gt_id = model._ground_token_id
        if gt_id is not None:
            gt_pos = (tok1d == gt_id).nonzero(as_tuple=False)
            print(f"    <|ground|> positions:     {gt_pos.flatten().tolist()}")
            if gt_pos.numel() > 0:
                for gp in gt_pos.flatten().tolist():
                    # show surrounding tokens
                    ctx = tok1d[max(0, gp-3):gp+4].tolist()
                    print(f"      ctx around pos {gp}: {ctx}")

        # Check pointer_start token
        ps_id = model._pointer_start_token_id
        if ps_id is not None:
            ps_pos = (tok1d == ps_id).nonzero(as_tuple=False)
            print(f"    <|pointer_start|> positions: {ps_pos.flatten().tolist()}")

        # ── Step 2: Anchor strategy ──────────────────────────────────────────
        anchor_idx, anchor_strategy = model._find_ground_anchor(
            token_ids=tok1d, external_hint=None, verbose=True,
        )
        print(f"\n  [Anchor]")
        print(f"    anchor_idx = {anchor_idx}, strategy = {anchor_strategy.value}")
        ctx = tok1d[max(0, anchor_idx-5):anchor_idx+5].tolist()
        print(f"    anchor context tokens: {ctx}")

        visual_indices = model._get_visual_indices(tok1d)
        print(f"    visual_indices count = {visual_indices.numel()}")
        if visual_indices.numel() > 0:
            print(f"    visual_indices range: [{visual_indices[0].item()}, {visual_indices[-1].item()}]")

        # ── Step 3: Forward pass (hook-based for guiowl) ────────────────────
        print(f"\n  [Forward]")
        with torch.no_grad():
            all_hs = model._forward_hidden_states_for_grounding(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                device=dev,
            )

        non_none_layers = [i for i, h in enumerate(all_hs) if h is not None]
        print(f"    Total hidden state slots: {len(all_hs)}")
        print(f"    Non-None positions: {non_none_layers}")
        for li in non_none_layers:
            h = all_hs[li]
            if h is not None:
                print(f"    Layer slot {li}: shape={h.shape}, norm_mean={h.norm(dim=-1).mean().item():.2f}")

        # ── Step 4: Grounding head ───────────────────────────────────────────
        print(f"\n  [Grounding Head]")
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

        print(f"    p_final: shape={p_final.shape}, sum={p_final.sum().item():.4f}")
        print(f"    p_final: max={p_final.max().item():.4f}, argmax={p_final.argmax().item()}")
        entropy = -(p_final * torch.log(p_final.clamp(min=1e-8))).sum().item()
        print(f"    p_final: entropy={entropy:.4f}")
        print(f"    omega: {omega.cpu().tolist()}")

        # Per-layer probs
        for i, (li, p_l) in enumerate(zip(model.layerwise_grounding_head.probe_layers, per_layer_probs)):
            ent_l = -(p_l * torch.log(p_l.clamp(min=1e-8))).sum().item()
            max_l = p_l.max().item()
            print(f"    Layer {li}: max={max_l:.4f}, entropy={ent_l:.4f}, argmax={p_l.argmax().item()}")

        # ── Step 5: Coordinate decoding ─────────────────────────────────────
        print(f"\n  [Coordinate]")
        best, topk_pts = scores_to_point_and_topk(
            p=p_final, n_width=n_width, n_height=n_height,
            activation_threshold=0.3, topk=3,
        )
        print(f"    pred_point (norm): {best}")
        print(f"    topk_points: {topk_pts}")

        if gt_bbox_norm:
            # Determine if gt_bbox_norm is [0,1000] or [0,1]
            b = gt_bbox_norm
            if max(b) > 1:  # 0-1000 range
                b_norm = [v / 1000 for v in b]
            else:
                b_norm = b
            px, py = best
            hit = b_norm[0] <= px <= b_norm[2] and b_norm[1] <= py <= b_norm[3]
            print(f"    gt_bbox_norm (0-1): {[f'{v:.4f}' for v in b_norm]}")
            print(f"    hit={hit}")

        # ── Step 6: Check the system prompt → ground_token in system ────────
        print(f"\n  [Token ID check]")
        print(f"    ground_token_id in config = {model.config.ground_token_id}")
        print(f"    model._ground_token_id    = {model._ground_token_id}")
        print(f"    vision_end_token_id       = {model._vision_end_token_id}")
        print(f"    image_token_id (config)   = {model.config.image_token_id}")

        # Decode anchor token to see what it is
        try:
            anchor_tok_text = processor.tokenizer.decode([tok1d[anchor_idx].item()])
            print(f"    anchor token text: {repr(anchor_tok_text)}")
        except Exception as e:
            print(f"    (decode failed: {e})")

    print("\n" + "=" * 72)
    print("  DEBUG COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    run_guiowl_debug(n_samples=3, device="cuda:0")
