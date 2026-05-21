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
    max_pixels: int = 5_760_000,
    user_prompt_template: Optional[str] = None,
) -> dict:
    """构造 prefill-only 推理的 model inputs（batch_size=1）。"""
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
    inputs = processor(
        text=[text],
        images=image_inputs if image_inputs else None,
        videos=video_inputs if video_inputs else None,
        return_tensors="pt",
        padding=True,
    )
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

MAIN_BENCH_KEYS = ["ss_pro", "ss_v2", "osworld_g", "mmbench", "ui_vision"]


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
        max_pixels: int = 12_845_056,
    ) -> "BaseZwergeInference":
        """
        Load model + processor from checkpoint, look up model-specific constants,
        return an initialized inference instance.
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

class RetrofitInference(BaseZwergeInference):
    """
    Implements predict_layerwise() for all retrofit models.

    Subclasses override:
      model_type  — "uitars" / "guiowl" / "uivenus"
      patch_size  — 14 for Qwen2.5-VL, 16 for Qwen3-VL
    """

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path: str,
        attn_impl: str = "flash_attention_2",
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        max_pixels: int = 12_845_056,
    ) -> "RetrofitInference":
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

        constants = MODEL_TYPE_CONSTANTS[cls.model_type]
        return cls(
            model=model, processor=processor,
            system_message=constants["system_message"],
            ground_response=constants["ground_response"],
            user_prompt_template=constants.get("user_prompt_template"),
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
