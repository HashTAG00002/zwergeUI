"""
ZwerGe-UI Inference Utilities
==============================
完全 prefill-only 推理：不生成任何 token，直接从中间层 hidden states 读取 grounding signal。

推理流程：
  1. 构造 conversation（system + user(image + instruction)），注入 <|ground|> 作为响应前缀
  2. Processor 处理得到 input_ids / pixel_values / image_grid_thw
  3. 单次 forward（output_hidden_states=True, output_attentions=False）
  4. LayerWiseGroundingHead 在 all_hidden_states 上做 layer-wise probe
  5. p_final [N_vis] → get_prediction_region_point → 归一化 (px, py) ∈ [0,1]
  6. topk 中心点列表用于 hit 判定

FA2 兼容：整个推理过程不需要 output_attentions，完全走 hidden-state 路径。

anchor token 选取（_find_ground_anchor 优先级）：
  P1: 序列中最后一个 <|ground|>（主方案，prefill 时模型已看完 image+instruction+click(，
      但没有见到任何坐标数字 → 无 label leakage）
  P2: <|pointer_start|> 之前的 token（pre-coordinate action-prefix token）
  P3-P5: fallback（见 modeling_uitars._find_ground_anchor）

image_grid_thw → n_width / n_height：
  n_h = H // merge_size,  n_w = W // merge_size
  对 Qwen2.5-VL 默认 merge_size=2 → 每个 visual token 对应 patch_size*merge_size=14*2=28 pixel
"""

import math
import warnings
from typing import List, Optional, Tuple

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info


# ─────────────────────────────────────────────────────────────────────────────
# patch posterior → click point  (mirrors GUI-AIMA get_prediction_region_point)
# ─────────────────────────────────────────────────────────────────────────────

def get_prediction_region_point(
    attn_scores: torch.Tensor,   # [1, N_vis]  or  [N_vis]
    n_width: int,
    n_height: int,
    activation_threshold: float = 0.3,
    return_all_regions: bool = True,
    decode_strategy: str = "centroid",
    peak_shift_alpha: float = 0.5,
    temperature: float = 0.5,
):
    """
    从 patch 后验概率中提取点坐标（与 GUI-AIMA inference.py 完全对齐，并扩展多种策略）。

    步骤（公共部分，所有策略共享）：
      1. 阈值化：选 p > max*threshold 的 patch
      2. BFS 4-连通分割成若干区域
      3. 各区域取 max(score) 作为 region score，选 region score 最高的区域

    策略（决定最佳区域内如何选点）：
      "centroid"    (默认/原始): score 加权质心。与 GUI-AIMA 完全对齐。
                                  缺点：质心可能漂移到 gt_bbox 边界外（hit < overlap 的根因）。
      "argmax"      :            直接取区域内最高分 patch 的中心点。
                                  最保守，完全不依赖邻域信息，避免质心漂移。
      "peak_shift"  :            argmax 中心 与 score 加权质心的线性插值。
                                  alpha * argmax_center + (1-alpha) * centroid。
                                  alpha=peak_shift_alpha（默认 0.5）。
                                  平衡两者：alpha 越大越像 argmax，越小越像 centroid。
      "temperature" :            对区域内 scores 做温度缩放（scores^(1/T)）后再做加权质心。
                                  T=temperature（默认 0.5，< 1 使分布更峰值化）。
                                  让高分 patch 权重更大，降低边缘低分 patch 对质心的拉偏。

    区域排序和 topk 中心点在所有策略下逻辑一致（仅 best_point 不同）。

    Returns (当 return_all_regions=True):
      best_point:     (px, py) 归一化坐标
      sorted_centers: 所有区域中心（按 region score 降序，使用所选策略）
      sorted_scores:  所有区域 score
      sorted_points:  所有区域的各 patch 归一化中心点列表
    """
    if attn_scores.dim() == 1:
        attn_scores = attn_scores.unsqueeze(0)   # → [1, N_vis]

    scores_1d = attn_scores[0]  # [N_vis]

    max_score = scores_1d.max().item()
    if max_score <= 0:
        # Degenerate case: return center
        if return_all_regions:
            return (0.5, 0.5), [(0.5, 0.5)], [0.0], [[(0.5, 0.5)]]
        return (0.5, 0.5)

    threshold = max_score * activation_threshold
    mask = scores_1d > threshold
    valid_indices = mask.nonzero(as_tuple=False).squeeze(-1)
    topk_values = scores_1d[valid_indices]

    if valid_indices.numel() == 0:
        # Fallback: pick argmax patch
        best_idx = int(scores_1d.argmax().item())
        y = best_idx // n_width
        x = best_idx % n_width
        pt = ((x + 0.5) / n_width, (y + 0.5) / n_height)
        if return_all_regions:
            return pt, [pt], [max_score], [[pt]]
        return pt

    # Convert indices to (row, col)
    topk_coords = []
    for i, idx in enumerate(valid_indices.tolist()):
        y = idx // n_width
        x = idx % n_width
        topk_coords.append((y, x, idx))

    # BFS connected-component clustering (4-connectivity)
    regions = []
    visited = set()
    for i, (y, x, idx) in enumerate(topk_coords):
        if idx in visited:
            continue
        region = [(y, x, idx, topk_values[i].item())]
        visited.add(idx)
        queue = [(y, x, idx, topk_values[i].item())]
        while queue:
            cy, cx, c_idx, c_val = queue.pop(0)
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = cy + dy, cx + dx
                n_idx = ny * n_width + nx
                for j, (ty, tx, t_idx) in enumerate(topk_coords):
                    if ty == ny and tx == nx and t_idx not in visited:
                        visited.add(t_idx)
                        region.append((ny, nx, t_idx, topk_values[j].item()))
                        queue.append((ny, nx, t_idx, topk_values[j].item()))
        regions.append(region)

    # ── 计算每个区域的 center（依策略）和 score ──────────────────────────────
    region_scores = []
    region_centers = []
    region_points_list = []

    for region in regions:
        reg_score = max(item[3] for item in region)   # max score (matches GUI-AIMA)
        region_scores.append(reg_score)

        norm_centers = []
        weights = []
        for y, x, _, score in region:
            cx_norm = (x + 0.5) / n_width
            cy_norm = (y + 0.5) / n_height
            norm_centers.append((cx_norm, cy_norm))
            weights.append(score)
        region_points_list.append(norm_centers)

        # argmax center（区域内最高分 patch 中心）
        max_idx_in_region = int(max(range(len(weights)), key=lambda i: weights[i]))
        argmax_center = norm_centers[max_idx_in_region]

        # score 加权质心（centroid 策略原始实现）
        total_w = sum(weights)
        wt_x = sum(nc[0] * w for nc, w in zip(norm_centers, weights)) / total_w
        wt_y = sum(nc[1] * w for nc, w in zip(norm_centers, weights)) / total_w
        centroid = (wt_x, wt_y)

        # 根据策略选择 center
        if decode_strategy == "argmax":
            center = argmax_center
        elif decode_strategy == "peak_shift":
            alpha = peak_shift_alpha
            center = (
                alpha * argmax_center[0] + (1.0 - alpha) * centroid[0],
                alpha * argmax_center[1] + (1.0 - alpha) * centroid[1],
            )
        elif decode_strategy == "temperature":
            T = max(temperature, 1e-6)
            # 温度缩放：分数升幂（T<1 使分布更集中于高分 patch）
            scaled_w = [w ** (1.0 / T) for w in weights]
            total_sw = sum(scaled_w) + 1e-12
            tw_x = sum(nc[0] * sw for nc, sw in zip(norm_centers, scaled_w)) / total_sw
            tw_y = sum(nc[1] * sw for nc, sw in zip(norm_centers, scaled_w)) / total_sw
            center = (tw_x, tw_y)
        else:
            # "centroid"（默认，与 GUI-AIMA 完全对齐）
            center = centroid

        region_centers.append(center)

    # Sort by region score descending
    sorted_idx = sorted(range(len(region_scores)), key=lambda i: region_scores[i], reverse=True)
    sorted_centers = [region_centers[i] for i in sorted_idx]
    sorted_scores = [region_scores[i] for i in sorted_idx]
    sorted_points = [region_points_list[i] for i in sorted_idx]
    best_point = sorted_centers[0]

    if return_all_regions:
        return best_point, sorted_centers, sorted_scores, sorted_points
    return best_point


# ─────────────────────────────────────────────────────────────────────────────
# image_grid_thw → (n_width, n_height)
# ─────────────────────────────────────────────────────────────────────────────

def grid_thw_to_nwh(image_grid_thw: torch.Tensor, merge_size: int = 2) -> Tuple[int, int]:
    """
    image_grid_thw: [T, H, W]  patch-level grid (patch_size=14 per dim).
    visual token grid = (T * (H // merge_size) * (W // merge_size)) 个 token，
    排列为 (H // merge_size) rows × (W // merge_size) cols（per frame）。

    此函数返回单帧的 (n_width, n_height)，对多帧取第一帧。
    """
    if image_grid_thw.dim() == 2:
        # batch of [T,H,W] rows → take first row
        thw = image_grid_thw[0]
    else:
        thw = image_grid_thw.squeeze()
    T, H, W = int(thw[0].item()), int(thw[1].item()), int(thw[2].item())
    n_h = H // merge_size
    n_w = W // merge_size
    return n_w, n_h   # (width_dim, height_dim)


# ─────────────────────────────────────────────────────────────────────────────
# Build input for ZwerGe inference (single sample)
# ─────────────────────────────────────────────────────────────────────────────

def build_zwerge_inputs(
    image: Image.Image,
    instruction: str,
    processor,
    system_message: str,
    ground_response: str,
    max_pixels: int = 5_760_000,
) -> dict:
    """
    构造 ZwerGe 评测时的 model inputs（单样本，batch_size=1）。

    注意：
      - 我们注入完整的 ground_response（含 <|ground|> 及 pointer tokens）作为 assistant turn
        的起始内容，整个序列一次性 prefill 进去，然后读取 <|ground|> 处的 hidden state。
      - 不需要 generate()，只需要 forward() 一次。

    Returns: dict with keys: input_ids, attention_mask, pixel_values, image_grid_thw
    """
    from zwerge_retrofit.constants import GROUNDING_SYSTEM_MESSAGE, GROUND_RESPONSE_CLICK

    conversation = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_message}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": instruction},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": ground_response}],
        },
    ]

    text = processor.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=False,   # assistant turn already included
    )
    image_inputs, video_inputs = process_vision_info(conversation)
    inputs = processor(
        text=[text],
        images=image_inputs if image_inputs else None,
        videos=video_inputs if video_inputs else None,
        return_tensors="pt",
        padding=True,
    )
    return inputs


# ─────────────────────────────────────────────────────────────────────────────
# Main single-sample inference function
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def zwerge_predict(
    image: Image.Image,
    instruction: str,
    model,
    processor,
    device: torch.device,
    topk: int = 3,
    activation_threshold: float = 0.3,
    system_message: Optional[str] = None,
    ground_response: Optional[str] = None,
    merge_size: int = 2,
    decode_strategy: str = "centroid",
    peak_shift_alpha: float = 0.5,
    temperature: float = 0.5,
) -> dict:
    """
    ZwerGe-UI 单样本推理（prefill-only）。

    Args:
        image:               PIL.Image.Image（原图，不需要提前 resize）
        instruction:         自然语言指令
        model:               UITARSRetrofitModel，已加载到 device，eval() 模式
        processor:           AutoProcessor（含 tokenizer 和 image_processor）
        device:              torch.device
        topk:                返回前 topk 个区域中心点（用于 overlap/hit@k 计算）
        activation_threshold: BFS 阈值（0.3 = max * 0.3，与 GUI-AIMA 对齐）
        system_message:      系统提示词（None → 使用 GROUNDING_SYSTEM_MESSAGE）
        ground_response:     prefill 的 assistant 响应前缀（None → GROUND_RESPONSE_CLICK）
        merge_size:          Qwen2.5-VL visual token merge size（默认 2）
        decode_strategy:     坐标提取策略，选项：
                               "centroid"   — score 加权质心（默认，与 GUI-AIMA 对齐）
                               "argmax"     — 最高分 patch 中心（避免质心漂移）
                               "peak_shift" — argmax 与 centroid 线性插值（alpha 控制）
                               "temperature"— 温度缩放后的加权质心（T<1 使分布更集中）
        peak_shift_alpha:    peak_shift 策略的插值系数（默认 0.5，越大越像 argmax）
        temperature:         temperature 策略的温度（默认 0.5，越小越集中于最高分 patch）

    Returns dict:
        pred_point:     (px, py) 归一化 [0,1]，top-1 预测点
        topk_points:    list[(px, py)]，前 k 个区域中心点
        topk_scores:    list[float]，对应 region score
        p_final:        torch.Tensor [N_vis]，patch 后验（CPU）
        omega:          torch.Tensor [num_probes]，层融合权重（CPU）
        n_width:        int
        n_height:       int
        anchor_strategy: AnchorStrategy string
    """
    from zwerge_retrofit.constants import GROUNDING_SYSTEM_MESSAGE, GROUND_RESPONSE_CLICK

    if system_message is None:
        system_message = GROUNDING_SYSTEM_MESSAGE
    if ground_response is None:
        ground_response = GROUND_RESPONSE_CLICK

    # ── 构造 inputs ──────────────────────────────────────────────────────────
    inputs = build_zwerge_inputs(
        image=image,
        instruction=instruction,
        processor=processor,
        system_message=system_message,
        ground_response=ground_response,
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    pixel_values = inputs.get("pixel_values")
    if pixel_values is not None:
        pixel_values = pixel_values.to(device, dtype=model.dtype)
    image_grid_thw = inputs.get("image_grid_thw")
    if image_grid_thw is not None:
        image_grid_thw = image_grid_thw.to(device)

    # ── Grid size (n_width, n_height) ─────────────────────────────────────────
    if image_grid_thw is not None:
        n_width, n_height = grid_thw_to_nwh(image_grid_thw, merge_size=merge_size)
    else:
        # Fallback: compute from image size
        w, h = image.size
        token_cell = 14 * merge_size   # patch_size=14, merge_size=2
        n_width  = max(1, w // token_cell)
        n_height = max(1, h // token_cell)

    # ── Forward（prefill-only，无 generate）──────────────────────────────────
    # 直接调用 _run_grounding_head，内部完成 embed → transformer → grounding head
    # （不走 model.forward，避免重复计算）
    p_final, omega, anchor_strategy_str = _run_grounding_head(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        device=device,
    )

    # ── patch posterior → 点坐标 ──────────────────────────────────────────────
    best_point, sorted_centers, sorted_scores, _ = get_prediction_region_point(
        attn_scores=p_final.unsqueeze(0),
        n_width=n_width,
        n_height=n_height,
        activation_threshold=activation_threshold,
        return_all_regions=True,
        decode_strategy=decode_strategy,
        peak_shift_alpha=peak_shift_alpha,
        temperature=temperature,
    )

    topk_points = sorted_centers[:topk]
    topk_scores = sorted_scores[:topk]

    return {
        "pred_point":      best_point,
        "topk_points":     topk_points,
        "topk_scores":     topk_scores,
        "p_final":         p_final.cpu(),
        "omega":           omega.cpu() if omega is not None else None,
        "n_width":         n_width,
        "n_height":        n_height,
        "anchor_strategy": anchor_strategy_str,
    }


def _run_grounding_head(
    model,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    pixel_values: Optional[torch.Tensor],
    image_grid_thw: Optional[torch.Tensor],
    device: torch.device,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], str]:
    """
    内部函数：运行完整 forward 并手动调用 grounding head。

    由于 model.forward 在 inference 模式（multi_patch_labels=None）下
    不运行 grounding head，我们在这里：
      1. 做一次完整 forward 取 all_hidden_states
      2. 手动调用 model.layerwise_grounding_head
      3. 用 model._find_ground_anchor 定位 anchor

    Returns: (p_final [N_vis], omega [num_probes], anchor_strategy_str)
    """
    model.eval()

    # Step 1: Embed tokens + visual tokens
    token_ids_1d = input_ids[0]   # [seq_len]

    # Build inputs_embeds
    with torch.no_grad():
        inputs_embeds = model.model.embed_tokens(input_ids)
        if pixel_values is not None:
            pv = pixel_values.to(model.dtype)
            image_embeds = model.visual(pv, grid_thw=image_grid_thw)
            n_img_tokens = (input_ids == model.config.image_token_id).sum().item()
            n_img_feats  = image_embeds.shape[0]
            if n_img_tokens != n_img_feats:
                warnings.warn(
                    f"Image token count mismatch: seq has {n_img_tokens} "
                    f"but visual encoder produced {n_img_feats} features. "
                    f"Attempting to proceed anyway."
                )
            image_mask = (
                (input_ids == model.config.image_token_id)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    # Step 2: Get RoPE position ids
    position_ids, rope_deltas = model.get_rope_index(
        input_ids, image_grid_thw, None, attention_mask
    )

    # Step 3: Full transformer forward (output_hidden_states=True)
    with torch.no_grad():
        transformer_out = model.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=None,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
    all_hidden_states = transformer_out.hidden_states  # tuple (L+1) × [seq_len, d_model]

    # Step 4: Find anchor and visual indices
    anchor_idx, anchor_strategy = model._find_ground_anchor(
        token_ids=token_ids_1d,
        external_hint=None,
        verbose=False,
    )
    visual_indices = model._get_visual_indices(token_ids_1d)

    if visual_indices.numel() == 0:
        warnings.warn("No visual tokens found in sequence! Check image_token_id config.")
        N_vis_fallback = 1
        p_final = torch.ones(N_vis_fallback, device=device) / N_vis_fallback
        return p_final, None, anchor_strategy.value

    # Step 5: Per-sample hidden states slice (single sample → same as full)
    sample_hidden_states = tuple(hs[0] for hs in all_hidden_states)  # [seq_len, d_model]

    # Step 6: Run grounding head (no labels → no loss)
    with torch.no_grad():
        head_out = model.layerwise_grounding_head(
            all_hidden_states=sample_hidden_states,
            ground_token_idx=anchor_idx,
            visual_indices=visual_indices,
            labels=None,
        )

    p_final = head_out["p_final"]   # [N_vis]
    omega   = head_out["omega"]     # [num_probes]

    return p_final, omega, anchor_strategy.value


# ─────────────────────────────────────────────────────────────────────────────
# Hit / overlap judgment  (mirrors GUI-AIMA/eval/screenSpot_pro.py)
# ─────────────────────────────────────────────────────────────────────────────

def point_in_bbox(px: float, py: float, bbox_norm: Tuple[float, float, float, float]) -> bool:
    """
    判断归一化点 (px, py) 是否落在归一化 bbox (x1, y1, x2, y2) 内。
    坐标均为 [0,1]。
    """
    x1, y1, x2, y2 = bbox_norm
    return x1 <= px <= x2 and y1 <= py <= y2


def do_boxes_overlap(
    box1: Tuple[float, float, float, float],
    box2: Tuple[float, float, float, float],
) -> bool:
    """
    判断两个 bbox 是否有交叠（与 GUI-AIMA/gui_aima/utils.py 完全一致）。
    两个 bbox 均为 (x1, y1, x2, y2) 格式，坐标系一致（均为归一化或均为绝对像素皆可）。

    用途：计算 overlap_top1 / overlap_topk 指标。
    以预测点为中心、patch 大小（0.5/n_width × 0.5/n_height）为半径构造预测框，
    与 gt_bbox 有任何重叠即视为 overlap。
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    if x1_max < x2_min or x2_max < x1_min:
        return False
    if y1_max < y2_min or y2_max < y1_min:
        return False
    return True


def do_boxes_overlap_norm(
    pred_point_norm: Tuple[float, float],
    gt_bbox_norm: Tuple[float, float, float, float],
    point_as_bbox_frac: float = 0.0,
) -> bool:
    """简单的点包含判定（如需 overlap 判定可扩展）。"""
    return point_in_bbox(pred_point_norm[0], pred_point_norm[1], gt_bbox_norm)


def topk_hit(
    topk_points: List[Tuple[float, float]],
    gt_bbox_norm: Tuple[float, float, float, float],
) -> bool:
    """Top-k 中是否有任意点落在 gt_bbox 内。"""
    return any(point_in_bbox(px, py, gt_bbox_norm) for px, py in topk_points)


# ─────────────────────────────────────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────────────────────────────────────

def load_zwerge_model(
    ckpt_path: str,
    attn_implementation: str = "flash_attention_2",
    device: str = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
):
    """
    加载 UITARSRetrofitModel + processor。

    Args:
        ckpt_path:           checkpoint 目录（含 config.json + safetensors）
        attn_implementation: "flash_attention_2" | "sdpa" | "eager"
        device:              目标设备
        dtype:               torch dtype

    Returns: (model, processor)

    注意：
      - config.json 中已经保存了 probe_layers / grounding_proj_dim 等参数，
        从 from_pretrained 读取时会自动读取这些 custom config 字段。
      - ground_token_id / pointer_start_token_id / vision_end_token_id
        也保存在 config 中，setup_special_token_ids() 会自动从 config 读取。
      - tokenizer 中的 added_tokens 也已经保存在 checkpoint，
        from_pretrained 会自动恢复。
    """
    import sys, os
    # 确保 zwerge_retrofit 包可以 import
    _zwerge_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
    if _zwerge_src not in sys.path:
        sys.path.insert(0, _zwerge_src)

    from transformers import AutoProcessor, AutoConfig
    from zwerge_retrofit.modeling_uitars import UITARSRetrofitModel

    print(f"[ZwerGe] Loading model from {ckpt_path}")
    print(f"[ZwerGe] attn_implementation={attn_implementation}, dtype={dtype}, device={device}")

    config = AutoConfig.from_pretrained(ckpt_path)
    print(f"[ZwerGe] probe_layers={getattr(config, 'probe_layers', 'N/A')}")
    print(f"[ZwerGe] grounding_proj_dim={getattr(config, 'grounding_proj_dim', 'N/A')}")

    model = UITARSRetrofitModel.from_pretrained(
        ckpt_path,
        config=config,
        attn_implementation=attn_implementation,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )

    # Register special token IDs
    # These are saved in config by training script; setup_special_token_ids also re-inits head LN.
    ground_token_id         = getattr(config, "ground_token_id", None)
    pointer_start_token_id  = getattr(config, "pointer_start_token_id", None)
    vision_end_token_id     = getattr(config, "vision_end_token_id", None)

    if ground_token_id is None or pointer_start_token_id is None:
        warnings.warn(
            "ground_token_id or pointer_start_token_id not found in config! "
            "Falling back to tokenizer lookup."
        )
        processor_tmp = AutoProcessor.from_pretrained(ckpt_path)
        tok = processor_tmp.tokenizer
        ground_token_id        = tok.convert_tokens_to_ids("<|ground|>")
        pointer_start_token_id = tok.convert_tokens_to_ids("<|pointer_start|>")
        vision_end_token_id    = tok.convert_tokens_to_ids("<|vision_end|>")

    model.setup_special_token_ids(
        ground_token_id=ground_token_id,
        pointer_start_token_id=pointer_start_token_id,
        vision_end_token_id=vision_end_token_id,
    )
    print(f"[ZwerGe] ground_token_id={ground_token_id}, "
          f"pointer_start_token_id={pointer_start_token_id}, "
          f"vision_end_token_id={vision_end_token_id}")

    model.config.use_cache = False
    model = model.to(device=device)
    model.eval()

    # Freeze all params (inference only)
    for p in model.parameters():
        p.requires_grad_(False)

    # Load processor
    processor = AutoProcessor.from_pretrained(ckpt_path)
    print(f"[ZwerGe] Model loaded. Total params: "
          f"{sum(p.numel() for p in model.parameters()):,}")
    return model, processor
