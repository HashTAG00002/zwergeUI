"""
ZwerGe-UI Inference Base
========================
Abstract base class, shared utilities, and RetrofitInference for all 3 retrofit model types.

Mirrors src/zwerge_retrofit/modeling_base.py on the inference side.
"""

import math
import os
import sys
import warnings
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR    = os.path.join(_SCRIPT_DIR, "..", "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# Shared utility functions
# ─────────────────────────────────────────────────────────────────────────────

def get_prediction_region_point(
    attn_scores: torch.Tensor,
    n_width: int,
    n_height: int,
    activation_threshold: float = 0.3,
    return_all_regions: bool = True,
    decode_strategy: str = "centroid",
    peak_shift_alpha: float = 0.5,
    temperature: float = 0.5,
):
    """patch 后验 → 点坐标（centroid/argmax/peak_shift/temperature 策略）。"""
    if attn_scores.dim() == 1:
        attn_scores = attn_scores.unsqueeze(0)
    scores_1d = attn_scores[0]

    max_score = scores_1d.max().item()
    if max_score <= 0:
        if return_all_regions:
            return (0.5, 0.5), [(0.5, 0.5)], [0.0], [[(0.5, 0.5)]]
        return (0.5, 0.5)

    threshold = max_score * activation_threshold
    mask = scores_1d > threshold
    valid_indices = mask.nonzero(as_tuple=False).squeeze(-1)
    topk_values = scores_1d[valid_indices]

    if valid_indices.numel() == 0:
        best_idx = int(scores_1d.argmax().item())
        y = best_idx // n_width
        x = best_idx % n_width
        pt = ((x + 0.5) / n_width, (y + 0.5) / n_height)
        if return_all_regions:
            return pt, [pt], [max_score], [[pt]]
        return pt

    topk_coords = []
    for i, idx in enumerate(valid_indices.tolist()):
        y = idx // n_width
        x = idx % n_width
        topk_coords.append((y, x, idx))

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
                if ny < 0 or ny >= n_height or nx < 0 or nx >= n_width:
                    continue
                n_idx = ny * n_width + nx
                for j, (ty, tx, t_idx) in enumerate(topk_coords):
                    if ty == ny and tx == nx and t_idx not in visited:
                        visited.add(t_idx)
                        region.append((ny, nx, t_idx, topk_values[j].item()))
                        queue.append((ny, nx, t_idx, topk_values[j].item()))
        regions.append(region)

    region_scores = []
    region_centers = []
    region_points_list = []

    for region in regions:
        reg_score = max(item[3] for item in region)
        region_scores.append(reg_score)

        norm_centers = []
        weights = []
        for y, x, _, score in region:
            cx_norm = (x + 0.5) / n_width
            cy_norm = (y + 0.5) / n_height
            norm_centers.append((cx_norm, cy_norm))
            weights.append(score)
        region_points_list.append(norm_centers)

        max_idx_in_region = int(max(range(len(weights)), key=lambda i: weights[i]))
        argmax_center = norm_centers[max_idx_in_region]

        total_w = sum(weights)
        wt_x = sum(nc[0] * w for nc, w in zip(norm_centers, weights)) / total_w
        wt_y = sum(nc[1] * w for nc, w in zip(norm_centers, weights)) / total_w
        centroid = (wt_x, wt_y)

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
            scaled_w = [w ** (1.0 / T) for w in weights]
            total_sw = sum(scaled_w) + 1e-12
            tw_x = sum(nc[0] * sw for nc, sw in zip(norm_centers, scaled_w)) / total_sw
            tw_y = sum(nc[1] * sw for nc, sw in zip(norm_centers, scaled_w)) / total_sw
            center = (tw_x, tw_y)
        else:
            center = centroid

        region_centers.append(center)

    sorted_idx = sorted(range(len(region_scores)), key=lambda i: region_scores[i], reverse=True)
    sorted_centers = [region_centers[i] for i in sorted_idx]
    sorted_scores  = [region_scores[i]  for i in sorted_idx]
    sorted_points  = [region_points_list[i] for i in sorted_idx]
    best_point = sorted_centers[0]

    if return_all_regions:
        return best_point, sorted_centers, sorted_scores, sorted_points
    return best_point


def scores_to_point_and_topk(
    p: torch.Tensor,
    n_width: int,
    n_height: int,
    activation_threshold: float,
    topk: int,
    decode_strategy: str = "centroid",
    peak_shift_alpha: float = 0.5,
    temperature: float = 0.5,
) -> Tuple[Tuple[float, float], List[Tuple[float, float]]]:
    """patch 后验 [N_vis] → (top-1 点, topk 候选点列表)。"""
    result = get_prediction_region_point(
        attn_scores=p.unsqueeze(0),
        n_width=n_width, n_height=n_height,
        activation_threshold=activation_threshold,
        return_all_regions=True,
        decode_strategy=decode_strategy,
        peak_shift_alpha=peak_shift_alpha,
        temperature=temperature,
    )
    best: Tuple[float, float] = result[0]          # type: ignore[index]
    centers: List[Tuple[float, float]] = result[1]  # type: ignore[index]
    return best, centers[:topk]


def get_zoom_crop_box(
    p_final: torch.Tensor,
    n_width: int,
    n_height: int,
    image_w: int,
    image_h: int,
    token_cell_px: int,
    activation_threshold: float = 0.3,
    padding_cells: int = 3,
) -> Tuple[int, int, int, int]:
    """
    Compute pixel-space crop box for zoom-in from patch posteriors.

    Uses the same threshold+BFS logic as get_prediction_region_point to find
    the best region, then returns its pixel bounding box with padding.

    Returns (x_min, y_min, x_max, y_max) clipped to image boundaries.
    padding_cells: number of extra patch cells added on each side.
    """
    scores_1d = p_final.float().cpu()
    max_score  = scores_1d.max().item()

    if max_score <= 0:
        # Fallback: center quarter
        qw, qh = max(1, image_w // 4), max(1, image_h // 4)
        cx, cy = image_w // 2, image_h // 2
        return max(0, cx - qw), max(0, cy - qh), min(image_w, cx + qw), min(image_h, cy + qh)

    threshold   = max_score * activation_threshold
    valid_mask  = scores_1d > threshold
    valid_idxs  = valid_mask.nonzero(as_tuple=False).squeeze(-1).tolist()
    valid_scores = scores_1d[valid_mask].tolist()

    if not valid_idxs:
        # Argmax fallback
        best_idx = int(scores_1d.argmax().item())
        col, row = best_idx % n_width, best_idx // n_width
        return (
            max(0, (col - padding_cells) * token_cell_px),
            max(0, (row - padding_cells) * token_cell_px),
            min(image_w, (col + 1 + padding_cells) * token_cell_px),
            min(image_h, (row + 1 + padding_cells) * token_cell_px),
        )

    # BFS to collect connected regions (mirrors get_prediction_region_point)
    topk_coords = [(idx // n_width, idx % n_width, idx) for idx in valid_idxs]
    patch_score = {idx: s for idx, s in zip(valid_idxs, valid_scores)}

    regions: List[List[int]] = []   # each element = list of flat patch indices
    visited: set = set()
    for row, col, idx in topk_coords:
        if idx in visited:
            continue
        region = [idx]
        visited.add(idx)
        queue = [(row, col)]
        while queue:
            r, c = queue.pop(0)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if nr < 0 or nr >= n_height or nc < 0 or nc >= n_width:
                    continue
                nidx = nr * n_width + nc
                if nidx in patch_score and nidx not in visited:
                    visited.add(nidx)
                    region.append(nidx)
                    queue.append((nr, nc))
        regions.append(region)

    # Select best region by peak score
    best_region = max(regions, key=lambda r: max(patch_score[i] for i in r))

    rows = [i // n_width for i in best_region]
    cols = [i %  n_width for i in best_region]
    min_col, max_col = min(cols), max(cols)
    min_row, max_row = min(rows), max(rows)

    x_min = max(0,        (min_col - padding_cells) * token_cell_px)
    y_min = max(0,        (min_row - padding_cells) * token_cell_px)
    x_max = min(image_w,  (max_col + 1 + padding_cells) * token_cell_px)
    y_max = min(image_h,  (max_row + 1 + padding_cells) * token_cell_px)
    return x_min, y_min, x_max, y_max


def grid_thw_to_nwh(image_grid_thw: torch.Tensor, merge_size: int = 2) -> Tuple[int, int]:
    """image_grid_thw [T,H,W] → (n_width, n_height)。"""
    if image_grid_thw.dim() == 2:
        thw = image_grid_thw[0]
    else:
        thw = image_grid_thw.squeeze()
    T, H, W = int(thw[0].item()), int(thw[1].item()), int(thw[2].item())
    return W // merge_size, H // merge_size


def build_zwerge_inputs(
    image: Image.Image,
    instruction: str,
    processor,
    system_message: Optional[str],
    ground_response: str,
    max_pixels: Optional[int] = None,
    user_prompt_template: Optional[str] = None,
) -> dict:
    """
    构造 prefill-only 推理的 model inputs（batch_size=1）。

    max_pixels: if provided, temporarily overrides processor.image_processor.max_pixels
                for this call only.  If None, uses the processor's current global setting
                (set during from_checkpoint()).  Both Stage-1 (grounding prefill) and
                Stage-2 (zoom backbone generate) should use the same value so that
                image_grid_thw reflects the actual resized dimensions.
    """
    user_text = user_prompt_template.format(instruction) if user_prompt_template else instruction

    conversation = []
    if system_message:
        conversation.append({
            "role": "system",
            "content": [{"type": "text", "text": system_message}],
        })
    conversation.append({
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": user_text},
        ],
    })
    conversation.append({
        "role": "assistant",
        "content": [{"type": "text", "text": ground_response}],
    })

    text = processor.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=False,
    )
    image_inputs, video_inputs = process_vision_info(conversation)

    # Override max_pixels for this call if explicitly provided
    _img_proc = getattr(processor, "image_processor", None)
    _old_max  = getattr(_img_proc, "max_pixels", None) if _img_proc else None
    if max_pixels is not None and _img_proc is not None:
        _img_proc.max_pixels = max_pixels
    try:
        inputs = processor(
            text=[text],
            images=image_inputs if image_inputs else None,
            videos=video_inputs if video_inputs else None,
            return_tensors="pt",
            padding=True,
        )
    finally:
        # Always restore the original max_pixels to avoid side-effects
        if max_pixels is not None and _img_proc is not None and _old_max is not None:
            _img_proc.max_pixels = _old_max
    return inputs


def point_in_bbox(px: float, py: float, bbox_norm: Tuple[float, float, float, float]) -> bool:
    x1, y1, x2, y2 = bbox_norm
    return x1 <= px <= x2 and y1 <= py <= y2


def do_boxes_overlap(
    box1: Tuple[float, float, float, float],
    box2: Tuple[float, float, float, float],
) -> bool:
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    if x1_max < x2_min or x2_max < x1_min:
        return False
    if y1_max < y2_min or y2_max < y1_min:
        return False
    return True


def topk_hit(
    topk_points: List[Tuple[float, float]],
    gt_bbox_norm: Tuple[float, float, float, float],
) -> bool:
    return any(point_in_bbox(px, py, gt_bbox_norm) for px, py in topk_points)


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark configs
# ─────────────────────────────────────────────────────────────────────────────

BENCH_CONFIGS = {
    "ss_pro": {
        "name":        "ScreenSpot-Pro",
        "eval_dir":    "ScreenSpot-Pro",
        "eval_json":   "eval.json",
        "group_field": "ui_type",
    },
    "ss_v2": {
        "name":        "ScreenSpot-v2",
        "eval_dir":    "ScreenSpot-v2",
        "eval_json":   "eval.json",
        "group_field": "data_type",
    },
    "osworld_g": {
        "name":        "OSWorld-G (refined)",
        "eval_dir":    "OSWorld-G",
        "eval_json":   "eval.json",
        "group_field": "GUI_types",
    },
    "osworld_g_orig": {
        "name":        "OSWorld-G (original)",
        "eval_dir":    "OSWorld-G",
        "eval_json":   "eval_orig.json",
        "group_field": "GUI_types",
    },
    "mmbench": {
        "name":        "MMBench-GUI-L2",
        "eval_dir":    "MMBench-GUI",
        "eval_json":   "eval.json",
        "group_field": "grounding_type",
    },
    "ui_vision": {
        "name":        "UI-Vision",
        "eval_dir":    "UI-Vision",
        "eval_json":   "eval.json",
        "group_field": "task_type",
    },
}

MAIN_BENCH_KEYS = ["ss_pro", "ss_v2", "osworld_g", "osworld_g_orig", "mmbench", "ui_vision"]


def _get_group_key(example: dict, group_field: Optional[str]) -> str:
    if group_field is None:
        return "all"
    val = example.get(group_field, "unknown")
    if isinstance(val, list):
        return ",".join(str(v) for v in val)
    return str(val)


def _print_layerwise_summary(summary: dict, topk: int):
    bench = summary["bench"]
    print(f"\n{'='*72}")
    print(f"  {bench}  [Layer-wise Accuracy]")
    print(f"{'='*72}")
    print(f"  Valid / Total : {summary['valid']} / {summary['total']}  "
          f"(skipped: {summary['skipped']})")
    print()
    print(f"  {'rank':>4}  {'layer':>7}  {'hit@1':>8}  {'ov@1':>8}  "
          f"{'hit@k':>8}  {'ov@k':>8}")
    print(f"  {'-'*52}")
    for i, la in enumerate(summary["layer_accs_sorted"]):
        marker = "  ← best" if i == 0 else ""
        print(f"  {i+1:>4}  L{la['layer_idx']:>6}  {la['hit_top1']:>7.2f}%  "
              f"{la['overlap_top1']:>7.2f}%  {la['hit_topk']:>7.2f}%  "
              f"{la['overlap_topk']:>7.2f}%{marker}")
    fa = summary["fusion_acc"]
    print(f"  {'-'*52}")
    has_topk = "hit_topk" in fa
    if has_topk:
        print(f"  {'fusion':>11}  {fa['hit_top1']:>7.2f}%  {fa['overlap_top1']:>7.2f}%"
              f"  {fa['hit_topk']:>7.2f}%  {fa['overlap_topk']:>7.2f}%")
    else:
        print(f"  {'fusion':>11}  {fa['hit_top1']:>7.2f}%  {fa['overlap_top1']:>7.2f}%")
    fga = summary.get("fusion_group_accs", {})
    if fga:
        print(f"\n  Fusion per-category breakdown:")
        print(f"  {'group':30s}  hit_top1   overlap_top1   n")
        print(f"  {'-'*58}")
        for grp, st in fga.items():
            print(f"  {grp:30s}  {st['hit_top1']:6.2f}%    {st['overlap_top1']:6.2f}%    {st['total']}")
    print(f"{'='*72}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base inference class
# ─────────────────────────────────────────────────────────────────────────────

class BaseZwergeInference(ABC):
    """
    ZwerGe-UI inference abstraction — mirrors RetrofitModelMixin on the eval side.

    Subclasses:
      RetrofitInference          ← base for all 3 retrofit models
        UITARSRetrofitInference  (inference_uitars.py)
        GUIOwlRetrofitInference  (inference_guiowl.py)
        UIVenusRetrofitInference (inference_uivenus.py)
      GUIOwlNativeInference      ← original GUI-Owl-1.5 model (inference_guiowl.py)
      UIVenusNativeInference     ← original UI-Venus-1.5 model (inference_uivenus.py)
    """

    model_type: str = ""
    merge_size: int = 2
    patch_size: int = 14    # 14 for Qwen2.5-VL, 16 for Qwen3-VL

    def __init__(
        self,
        model,
        processor,
        system_message: Optional[str] = None,
        ground_response: Optional[str] = None,
        user_prompt_template: Optional[str] = None,
    ):
        self.model              = model
        self.processor          = processor
        self.system_message     = system_message
        self.ground_response    = ground_response
        self.user_prompt_template = user_prompt_template

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path: str,
        attn_impl: str = "flash_attention_2",
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        max_pixels: Optional[int] = None,
    ) -> "BaseZwergeInference":
        """
        Load model + processor from checkpoint, look up model-specific constants,
        return an initialized inference instance.

        max_pixels: if None, resolved per model_type:
          uitars/guiowl7b/uitars1 (Qwen2.x-VL, patch_size=14): 16384 × 14² × 4 = 12,845,056
          guiowl/uivenus/qwen35 (Qwen3.x-VL, patch_size=16): 16384 × 16² × 4 = 16,777,216
        """
        raise NotImplementedError

    @abstractmethod
    @torch.no_grad()
    def predict_layerwise(
        self,
        image: Image.Image,
        instruction: str,
        device: torch.device,
        activation_threshold: float = 0.3,
        topk: int = 3,
        decode_strategy: str = "centroid",
        peak_shift_alpha: float = 0.5,
        temperature: float = 0.5,
    ) -> dict:
        """
        Returns dict with keys:
          per_layer_probs, per_layer_points, per_layer_topk, layer_indices,
          p_final, omega, n_width, n_height, anchor_strategy
        """
        ...


# ─────────────────────────────────────────────────────────────────────────────
# Retrofit inference base (shared by uitars / guiowl / uivenus)
# ─────────────────────────────────────────────────────────────────────────────

_ZOOM_NOT_SET = object()   # sentinel: "use the retrofit training value"


class RetrofitInference(BaseZwergeInference):
    """
    Implements predict_layerwise() for all retrofit models.

    Subclasses override:
      model_type  — "uitars" / "guiowl" / "uivenus" / "guiowl7b" / "qwen35" / "uitars1"
      patch_size  — 14 for Qwen2.x-VL, 16 for Qwen3.x-VL

    Zoom-backbone system message overrides (subclass sets these):
      _zoom_native_system_message — system message for backbone generate (NATIVE coord format).
          Use _ZOOM_NOT_SET (default) to fall back to self.system_message (retrofit training msg).
          Set to None explicitly for "no system message" (e.g., UI-Venus).
      _zoom_native_user_template  — user prompt template for backbone generate.
          Use _ZOOM_NOT_SET (default) to fall back to self.user_prompt_template.
    """
    _zoom_native_system_message = _ZOOM_NOT_SET
    _zoom_native_user_template  = _ZOOM_NOT_SET

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path: str,
        attn_impl: str = "flash_attention_2",
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        max_pixels: Optional[int] = None,
    ) -> "RetrofitInference":
        # Resolve max_pixels per model_type if not explicitly provided.
        # uitars/guiowl7b/uitars1 (Qwen2.x-VL, patch_size=14): 16384 × 14² × 4 = 12,845,056
        # guiowl/uivenus (Qwen3-VL, patch_size=16):              16384 × 16² × 4 = 16,777,216
        # qwen35 (Qwen3.5, patch_size=16, 暂定):                 16384 × 16² × 4 = 16,777,216
        if max_pixels is None:
            if cls.model_type in ("guiowl", "uivenus", "qwen35"):
                max_pixels = 16_777_216
            else:
                # uitars / guiowl7b / uitars1 (Qwen2.x-VL, patch_size=14)
                max_pixels = 12_845_056
        from zwerge_retrofit import get_model_class
        from zwerge_retrofit.constants import MODEL_TYPE_CONSTANTS
        from transformers import AutoProcessor, AutoConfig

        ModelClass = get_model_class(cls.model_type)
        config = AutoConfig.from_pretrained(ckpt_path)
        model = ModelClass.from_pretrained(
            ckpt_path, config=config,
            attn_implementation=attn_impl, torch_dtype=dtype, low_cpu_mem_usage=True,
        )

        ground_token_id        = getattr(config, "ground_token_id", None)
        pointer_start_token_id = getattr(config, "pointer_start_token_id", None)
        vision_end_token_id    = getattr(config, "vision_end_token_id", None)
        if ground_token_id is None or pointer_start_token_id is None:
            processor_tmp = AutoProcessor.from_pretrained(ckpt_path)
            tok = processor_tmp.tokenizer
            ground_token_id        = tok.convert_tokens_to_ids("<|ground|>")
            pointer_start_token_id = tok.convert_tokens_to_ids("<|pointer_start|>")
            vision_end_token_id    = tok.convert_tokens_to_ids("<|vision_end|>")

        model.setup_special_token_ids(
            ground_token_id=ground_token_id,
            pointer_start_token_id=pointer_start_token_id,
            vision_end_token_id=vision_end_token_id,
            reinit_grounding_head=False,
        )
        model.config.use_cache = False
        model = model.to(device=device)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        processor = AutoProcessor.from_pretrained(ckpt_path)
        if hasattr(processor, "image_processor"):
            processor.image_processor.max_pixels = max_pixels

        # ── Resolve system_message / ground_response ──────────────────────────
        # Priority: (1) args.json saved at training time — guaranteed to match
        #               what the checkpoint was trained with (handles multiple
        #               guiowl system-prompt variants, e.g. native vs. grounding-only)
        #           (2) MODEL_TYPE_CONSTANTS fallback — used when checkpoint was
        #               not produced by our training pipeline (e.g., raw base model)
        constants = MODEL_TYPE_CONSTANTS[cls.model_type]
        system_message       = constants["system_message"]
        ground_response      = constants["ground_response"]
        user_prompt_template = constants.get("user_prompt_template")

        # Look for args.json one level up from ckpt_path (i.e., the output_dir)
        ckpt_parent = os.path.dirname(ckpt_path)
        args_json_path = os.path.join(ckpt_parent, "args.json")
        if os.path.isfile(args_json_path):
            try:
                import json as _json
                with open(args_json_path) as _f:
                    _saved_args = _json.load(_f)
                _data_args = _saved_args.get("data_args", {})
                _sys = _data_args.get("system_message")
                _grd = _data_args.get("ground_response")
                _upt = _data_args.get("user_prompt_template")
                if _sys is not None:
                    system_message = _sys
                if _grd is not None:
                    ground_response = _grd
                if _upt is not None:
                    user_prompt_template = _upt
                print(
                    f"[RetrofitInference] system_message / ground_response loaded from "
                    f"{args_json_path} (len={len(system_message or '')})"
                )
            except Exception as _e:
                print(
                    f"[RetrofitInference] WARNING: failed to load args.json from "
                    f"{args_json_path}: {_e}. Falling back to MODEL_TYPE_CONSTANTS."
                )
        else:
            print(
                f"[RetrofitInference] args.json not found at {args_json_path}. "
                f"Using MODEL_TYPE_CONSTANTS defaults."
            )

        return cls(
            model=model, processor=processor,
            system_message=system_message,
            ground_response=ground_response,
            user_prompt_template=user_prompt_template,
        )

    @torch.no_grad()
    def predict_layerwise(
        self,
        image: Image.Image,
        instruction: str,
        device: torch.device,
        activation_threshold: float = 0.3,
        topk: int = 3,
        decode_strategy: str = "centroid",
        peak_shift_alpha: float = 0.5,
        temperature: float = 0.5,
    ) -> dict:
        from zwerge_retrofit.constants import GROUNDING_SYSTEM_MESSAGE, GROUND_RESPONSE_CLICK

        sys_msg  = self.system_message  if self.system_message  is not None else GROUNDING_SYSTEM_MESSAGE
        grd_resp = self.ground_response if self.ground_response is not None else GROUND_RESPONSE_CLICK

        inputs = build_zwerge_inputs(
            image=image, instruction=instruction, processor=self.processor,
            system_message=sys_msg, ground_response=grd_resp,
            user_prompt_template=self.user_prompt_template,
        )

        input_ids      = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        pixel_values = inputs.get("pixel_values")
        if pixel_values is not None:
            pixel_values = pixel_values.to(device, dtype=self.model.dtype)
        image_grid_thw = inputs.get("image_grid_thw")
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.to(device)
        # mm_token_type_ids: Qwen3.5 M-RoPE 必需字段；其他模型为 None
        mm_token_type_ids = inputs.get("mm_token_type_ids")
        if mm_token_type_ids is not None:
            mm_token_type_ids = mm_token_type_ids.to(device)

        # Grid size — reads patch_size from processor to support both Qwen2.5 (14) and Qwen3 (16)
        if image_grid_thw is not None:
            n_width, n_height = grid_thw_to_nwh(image_grid_thw, merge_size=self.merge_size)
        else:
            w, h = image.size
            patch_size = getattr(
                getattr(self.processor, "image_processor", self.processor), "patch_size", self.patch_size
            )
            cell = patch_size * self.merge_size
            n_width  = max(1, w // cell)
            n_height = max(1, h // cell)

        token_ids_1d = input_ids[0]

        all_hidden_states = self.model._forward_hidden_states_for_grounding(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            device=device,
            mm_token_type_ids=mm_token_type_ids,
        )

        anchor_idx, anchor_strategy = self.model._find_ground_anchor(
            token_ids=token_ids_1d, external_hint=None, verbose=False,
        )
        visual_indices = self.model._get_visual_indices(token_ids_1d)

        if visual_indices.numel() == 0:
            warnings.warn("No visual tokens found in sequence!")
            dummy = torch.ones(1, device=device) / 1
            n_probes = len(self.model.layerwise_grounding_head.probe_layers)
            return {
                "per_layer_probs":  [dummy] * n_probes,
                "per_layer_points": [(0.5, 0.5)] * n_probes,
                "per_layer_topk":   [[(0.5, 0.5)]] * n_probes,
                "layer_indices":    self.model.layerwise_grounding_head.probe_layers,
                "p_final":          dummy,
                "omega":            torch.ones(n_probes) / n_probes,
                "n_width": n_width, "n_height": n_height,
                "anchor_strategy":  anchor_strategy.value,
            }

        # Guiowl/UIVenus return a SPARSE tuple (Nones at non-probe positions).
        # UITARs returns a DENSE tuple (all layers present).
        # Handle both: skip-index with None for non-probe positions.
        sample_hs = tuple(hs[0] if hs is not None else None for hs in all_hidden_states)

        head_out = self.model.layerwise_grounding_head(
            all_hidden_states=sample_hs,
            ground_token_idx=anchor_idx,
            visual_indices=visual_indices,
            labels=None,
        )

        per_layer_probs = head_out["per_layer_probs"]
        p_final         = head_out["p_final"]
        omega           = head_out["omega"]
        layer_indices   = self.model.layerwise_grounding_head.probe_layers

        per_layer_points = []
        per_layer_topk   = []
        for p_l in per_layer_probs:
            best, centers = scores_to_point_and_topk(
                p=p_l, n_width=n_width, n_height=n_height,
                activation_threshold=activation_threshold, topk=topk,
                decode_strategy=decode_strategy,
                peak_shift_alpha=peak_shift_alpha, temperature=temperature,
            )
            per_layer_points.append(best)
            per_layer_topk.append(centers)

        return {
            "per_layer_probs":  [p.cpu() for p in per_layer_probs],
            "per_layer_points": per_layer_points,
            "per_layer_topk":   per_layer_topk,
            "layer_indices":    layer_indices,
            "p_final":          p_final.cpu(),
            "omega":            omega.cpu(),
            "n_width":          n_width,
            "n_height":         n_height,
            "anchor_strategy":  anchor_strategy.value,
        }

    # ── Zoom-backbone decode strategy ────────────────────────────────────────

    def parse_backbone_coordinate(
        self,
        raw_text: str,
        crop_w_resized: Optional[int] = None,
        crop_h_resized: Optional[int] = None,
    ) -> Optional[Tuple[float, float]]:
        """
        Parse the backbone's generated text to extract a normalized (x,y) point.

        Returns (x_norm, y_norm) in [0,1] (fraction of crop), or None on failure.
        Subclasses MUST override for their model's output format.

        Args:
          raw_text:        backbone-generated text
          crop_w_resized:  actual pixel width of the crop after processor smart_resize.
                           Needed by Qwen2.5-VL (UI-TARS) which outputs absolute pixel coords.
                           Qwen3-VL models use [0,1000] format and ignore this.
          crop_h_resized:  actual pixel height after smart_resize (same note as above).

        Coordinate conventions by model:
          Qwen3-VL (GUI-Owl, UI-Venus): [0,1000] relative → divide by 1000 → [0,1]
          Qwen2.5-VL (UI-TARS):        absolute pixels in smart-resized space
                                         → divide by (crop_w_resized, crop_h_resized) → [0,1]
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement parse_backbone_coordinate()."
        )

    def _build_generation_inputs(
        self,
        image: Image.Image,
        instruction: str,
        max_pixels: Optional[int] = None,
    ) -> dict:
        """
        Build inputs for backbone generate (Stage 2 of zoom_backbone).

        Uses _zoom_native_system_message / _zoom_native_user_template (class attrs)
        so the model outputs NATIVE coordinates (not <|ground|> special tokens).
          - If _zoom_native_system_message is _ZOOM_NOT_SET → fall back to self.system_message
          - If _zoom_native_system_message is None          → no system turn (e.g. UI-Venus)

        max_pixels: if provided, temporarily overrides processor.image_processor.max_pixels
                    for this call.  The same value should be used in Stage 1 so that
                    image_grid_thw faithfully reflects the actual smart_resize output and
                    crop_w_resized = W_patches * patch_size is consistent with what the
                    backbone actually sees.  Defaults to None (use processor global setting).
        """
        from qwen_vl_utils import process_vision_info

        # Use native coordinate-format prompts, not the retrofit training prompts
        # (retrofit prompts contain <|ground|> tokens; backbone outputs special tokens)
        sys_msg = (
            self.system_message
            if self._zoom_native_system_message is _ZOOM_NOT_SET
            else self._zoom_native_system_message
        )
        usr_tmpl = (
            self.user_prompt_template
            if self._zoom_native_user_template is _ZOOM_NOT_SET
            else self._zoom_native_user_template
        )

        user_text = usr_tmpl.format(instruction) if usr_tmpl else instruction
        conversation = []
        if sys_msg:
            conversation.append({
                "role": "system",
                "content": [{"type": "text", "text": sys_msg}],
            })
        conversation.append({
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text":  user_text},
            ],
        })
        text = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(conversation)

        # Override max_pixels for this call if explicitly provided
        _img_proc = getattr(self.processor, "image_processor", None)
        _old_max  = getattr(_img_proc, "max_pixels", None) if _img_proc else None
        if max_pixels is not None and _img_proc is not None:
            _img_proc.max_pixels = max_pixels
        try:
            result = self.processor(
                text=[text],
                images=image_inputs if image_inputs else None,
                videos=video_inputs if video_inputs else None,
                return_tensors="pt",
                padding=True,
            )
        finally:
            if max_pixels is not None and _img_proc is not None and _old_max is not None:
                _img_proc.max_pixels = _old_max
        return result

    @torch.no_grad()
    def predict_zoom_backbone(
        self,
        image: Image.Image,
        instruction: str,
        device: torch.device,
        activation_threshold: float = 0.3,
        padding_cells: int = 3,
        max_new_tokens: int = 256,
        topk: int = 3,
        decode_strategy: str = "centroid",
        peak_shift_alpha: float = 0.5,
        temperature: float = 0.5,
        full_image: bool = False,
    ) -> dict:
        """
        Two-stage decode strategy:
          Stage 1 — ZwerGe prefill → patch posteriors → select best region
          Stage 2 — Crop around best region → backbone generate → parse coordinate
                    → remap to original image

        Returns the same schema as predict_layerwise(), plus:
          'zoom_point':    (x_norm, y_norm) refined by backbone (falls back to ZwerGe centroid)
          'zoom_crop_box': (x_min, y_min, x_max, y_max) in pixels
          'backbone_raw':  raw generated text from backbone (for debugging)

        The eval code should use pred['zoom_point'] as the final prediction instead
        of running scores_to_point_and_topk on pred['p_final'].
        """
        # ── Stage 1: ZwerGe ───────────────────────────────────────────────────
        pred = self.predict_layerwise(
            image=image, instruction=instruction, device=device,
            activation_threshold=activation_threshold, topk=topk,
            decode_strategy=decode_strategy,
            peak_shift_alpha=peak_shift_alpha, temperature=temperature,
        )
        n_w, n_h = pred["n_width"], pred["n_height"]
        W, H     = image.size
        token_cell_px = self.patch_size * self.merge_size   # 28 for uitars, 32 for guiowl

        # ── Compute crop box ─────────────────────────────────────────────────
        if full_image:
            # native_backbone mode: no crop, pass original image to backbone.
            # Coordinate mapping degenerates to identity: (0 + bx*W)/W = bx
            crop_box = (0, 0, W, H)
            crop_img = image
        else:
            crop_box = get_zoom_crop_box(
                p_final=pred["p_final"],
                n_width=n_w, n_height=n_h,
                image_w=W, image_h=H,
                token_cell_px=token_cell_px,
                activation_threshold=activation_threshold,
                padding_cells=padding_cells,
            )
            crop_img = image.crop(crop_box)
        x_min, y_min, x_max, y_max = crop_box
        crop_w = max(1, x_max - x_min)
        crop_h = max(1, y_max - y_min)

        # ── Stage 2: backbone generate ────────────────────────────────────────
        gen_inputs = self._build_generation_inputs(crop_img, instruction)
        gen_inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in gen_inputs.items()
        }
        # Some inputs may need dtype cast for pixel_values
        if "pixel_values" in gen_inputs and gen_inputs["pixel_values"] is not None:
            gen_inputs["pixel_values"] = gen_inputs["pixel_values"].to(
                dtype=self.model.dtype
            )

        generated_ids = self.model.generate(
            **gen_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
        prompt_len = gen_inputs["input_ids"].shape[1]
        trimmed    = generated_ids[:, prompt_len:]
        raw_text   = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # ── Get crop's smart-resized dimensions ──────────────────────────────
        # image_grid_thw = [T, H_raw_patches, W_raw_patches] (raw patch units)
        # smart_resize → W_resized = W_raw_patches * patch_size (e.g. 14 for uitars)
        # Needed by UI-TARS (absolute pixel coords); ignored by Qwen3-VL ([0,1000] format)
        crop_thw = gen_inputs.get("image_grid_thw")
        if crop_thw is not None:
            thw = crop_thw[0] if crop_thw.dim() == 2 else crop_thw.squeeze()
            _, H_ptch, W_ptch = int(thw[0].item()), int(thw[1].item()), int(thw[2].item())
            crop_w_resized: Optional[int] = W_ptch * self.patch_size
            crop_h_resized: Optional[int] = H_ptch * self.patch_size
        else:
            crop_w_resized = crop_w
            crop_h_resized = crop_h

        # ── Parse and remap ───────────────────────────────────────────────────
        backbone_coord = self.parse_backbone_coordinate(
            raw_text, crop_w_resized=crop_w_resized, crop_h_resized=crop_h_resized,
        )
        if backbone_coord is not None:
            bx_crop, by_crop = backbone_coord           # [0,1] in crop space
            # remap to original image [0,1]
            ox = max(0.0, min(1.0, (x_min + bx_crop * crop_w) / W))
            oy = max(0.0, min(1.0, (y_min + by_crop * crop_h) / H))
            zoom_point = (ox, oy)
        else:
            # Fallback: ZwerGe centroid from p_final
            fb, _ = scores_to_point_and_topk(
                p=pred["p_final"], n_width=n_w, n_height=n_h,
                activation_threshold=activation_threshold, topk=1,
                decode_strategy="centroid",
            )
            zoom_point = (float(fb[0]), float(fb[1]))

        return {
            **pred,
            "zoom_point":    zoom_point,
            "zoom_crop_box": crop_box,
            "backbone_raw":  raw_text,
        }
