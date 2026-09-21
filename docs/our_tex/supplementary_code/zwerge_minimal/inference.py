"""
ZwerGe-UI — Inference (minimal, self-contained)
================================================
Implements `RetrofitInference.predict_layerwise()` and
`.predict_zoom_backbone()`, the two entry points used by the paper's
evaluation:

  * predict_layerwise():
      One prefill-forced forward pass (see modeling.py's
      `_forward_hidden_states_for_grounding` / `_find_ground_anchor`) that
      returns, for every probed layer, its patch posterior p_l and decoded
      point, PLUS the cross-layer fusion posterior p_final and its
      layer weights omega. This is what the "Per-Layer Spatial Probes"
      and "Cross-Layer Fusion" sections of the paper measure.

  * predict_zoom_backbone(full_image=True):
      Runs the backbone's OWN autoregressive `generate()` on the (optionally
      cropped) image and parses its native coordinate output. With
      `full_image=True` this reproduces the vanilla backbone's own
      accuracy — i.e. the "Native" column of Table 2 in the paper.
      `ensemble.py`'s `run_p2p_ensemble` calls this to obtain the native
      coordinate `n` that competes against the fused point `f` under the
      margin gate.

Both functions share the SAME single prefill (paper: "Both reuse the
single prefill of [Per-Layer Spatial Probes], so the retrofit adds
negligible compute over the native baseline"): predict_zoom_backbone calls
predict_layerwise internally for Stage 1, then (optionally) crops the image
and calls the backbone's generate() for Stage 2.

FlashAttention-2 is used only as `attn_implementation="flash_attention_2"`
when loading the HuggingFace model (see `from_checkpoint`); no custom
FlashAttention kernel code is needed here, so there is no separate stub to
import.
"""

import warnings
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image

try:
    from qwen_vl_utils import process_vision_info
except ImportError:  # pragma: no cover - optional dependency, see requirements.txt
    def process_vision_info(conversation):
        raise NotImplementedError(
            "qwen_vl_utils is required for image/video preprocessing. "
            "Install with `pip install qwen-vl-utils`."
        )

from modeling import RetrofitModelMixin  # noqa: F401  (re-exported for convenience)


# =============================================================================
# Posterior -> point decoding (shared by predict_layerwise and ensemble.py)
# =============================================================================

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
    """
    Patch posterior [N_vis] -> continuous point(s) in [0,1]^2.

    Pipeline: threshold at `activation_threshold` * max, 4-connectivity BFS
    into candidate regions, decode each region's representative point via
    `decode_strategy` in {centroid, argmax, peak_shift, temperature}, then
    rank regions by their peak posterior value.

    This is the LEGACY single-pass decoder (region-growing + weighted
    centroid) used for the per-layer diagnostic points reported in Figure 1
    of the paper. The deployed decoder (ensemble.py's `decode_p2p`) refines
    this with cross-layer consensus re-ranking and sub-patch local-mode
    recovery; both share this function's region-extraction logic.
    """
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

    if valid_indices.numel() == 0:
        best_idx = int(scores_1d.argmax().item())
        y = best_idx // n_width
        x = best_idx % n_width
        pt = ((x + 0.5) / n_width, (y + 0.5) / n_height)
        if return_all_regions:
            return pt, [pt], [max_score], [[pt]]
        return pt

    topk_values = scores_1d[valid_indices]
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
                for j, (ty, tx, t_idx) in enumerate(topk_coords):
                    if ty == ny and tx == nx and t_idx not in visited:
                        visited.add(t_idx)
                        region.append((ny, nx, t_idx, topk_values[j].item()))
                        queue.append((ny, nx, t_idx, topk_values[j].item()))
        regions.append(region)

    region_scores, region_centers, region_points_list = [], [], []
    for region in regions:
        reg_score = max(item[3] for item in region)
        region_scores.append(reg_score)

        norm_centers, weights = [], []
        for y, x, _, score in region:
            norm_centers.append(((x + 0.5) / n_width, (y + 0.5) / n_height))
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
            center = (
                sum(nc[0] * sw for nc, sw in zip(norm_centers, scaled_w)) / total_sw,
                sum(nc[1] * sw for nc, sw in zip(norm_centers, scaled_w)) / total_sw,
            )
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
    """Patch posterior [N_vis] -> (top-1 point, top-k candidate points)."""
    result = get_prediction_region_point(
        attn_scores=p.unsqueeze(0), n_width=n_width, n_height=n_height,
        activation_threshold=activation_threshold, return_all_regions=True,
        decode_strategy=decode_strategy, peak_shift_alpha=peak_shift_alpha,
        temperature=temperature,
    )
    best: Tuple[float, float] = result[0]
    centers: List[Tuple[float, float]] = result[1]
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
    Pixel-space crop box around the best posterior region, used by
    predict_zoom_backbone's Stage-2 close-up decode (region_selector='max').
    """
    scores_1d = p_final.float().cpu()
    max_score = scores_1d.max().item()

    if max_score <= 0:
        qw, qh = max(1, image_w // 4), max(1, image_h // 4)
        cx, cy = image_w // 2, image_h // 2
        return max(0, cx - qw), max(0, cy - qh), min(image_w, cx + qw), min(image_h, cy + qh)

    threshold = max_score * activation_threshold
    valid_mask = scores_1d > threshold
    valid_idxs = valid_mask.nonzero(as_tuple=False).squeeze(-1).tolist()
    valid_scores = scores_1d[valid_mask].tolist()

    if not valid_idxs:
        best_idx = int(scores_1d.argmax().item())
        col, row = best_idx % n_width, best_idx // n_width
        return (
            max(0, (col - padding_cells) * token_cell_px),
            max(0, (row - padding_cells) * token_cell_px),
            min(image_w, (col + 1 + padding_cells) * token_cell_px),
            min(image_h, (row + 1 + padding_cells) * token_cell_px),
        )

    topk_coords = [(idx // n_width, idx % n_width, idx) for idx in valid_idxs]
    patch_score = {idx: s for idx, s in zip(valid_idxs, valid_scores)}

    regions: List[List[int]] = []
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
    """image_grid_thw [T,H,W] -> (n_width, n_height) patch-grid dimensions."""
    thw = image_grid_thw[0] if image_grid_thw.dim() == 2 else image_grid_thw.squeeze()
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
    Construct prefill-only model inputs (batch_size=1): system + user
    (image + instruction) + assistant turn ending in the `<|ground|>`
    anchor token sequence (`ground_response`), WITHOUT `add_generation_prompt`
    — i.e. exactly the "prefill a fixed template ending in an anchor token"
    procedure described in the paper's "Prefill-forced inference" paragraph.
    """
    user_text = user_prompt_template.format(instruction) if user_prompt_template else instruction

    conversation = []
    if system_message:
        conversation.append({"role": "system", "content": [{"type": "text", "text": system_message}]})
    conversation.append({
        "role": "user",
        "content": [{"type": "image", "image": image}, {"type": "text", "text": user_text}],
    })
    conversation.append({"role": "assistant", "content": [{"type": "text", "text": ground_response}]})

    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
    image_inputs, video_inputs = process_vision_info(conversation)

    img_proc = getattr(processor, "image_processor", None)
    old_max  = getattr(img_proc, "max_pixels", None) if img_proc else None
    if max_pixels is not None and img_proc is not None:
        img_proc.max_pixels = max_pixels
    try:
        inputs = processor(
            text=[text], images=image_inputs if image_inputs else None,
            videos=video_inputs if video_inputs else None,
            return_tensors="pt", padding=True,
        )
    finally:
        if max_pixels is not None and img_proc is not None and old_max is not None:
            img_proc.max_pixels = old_max
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


# =============================================================================
# RetrofitInference
# =============================================================================

_ZOOM_NOT_SET = object()   # sentinel: "use the retrofit training value"


class BaseZwergeInference(ABC):
    model_type: str = ""
    merge_size: int = 2
    patch_size: int = 14    # 14 for Qwen2.5-VL, 16 for Qwen3-VL

    def __init__(
        self, model, processor,
        system_message: Optional[str] = None,
        ground_response: Optional[str] = None,
        user_prompt_template: Optional[str] = None,
    ):
        self.model = model
        self.processor = processor
        self.system_message = system_message
        self.ground_response = ground_response
        self.user_prompt_template = user_prompt_template

    @classmethod
    def from_checkpoint(cls, ckpt_path: str, attn_impl: str = "flash_attention_2",
                         device: str = "cuda:0", dtype: torch.dtype = torch.bfloat16,
                         max_pixels: Optional[int] = None) -> "BaseZwergeInference":
        raise NotImplementedError

    @abstractmethod
    @torch.no_grad()
    def predict_layerwise(self, image, instruction, device,
                           activation_threshold: float = 0.3, topk: int = 3,
                           decode_strategy: str = "centroid",
                           peak_shift_alpha: float = 0.5, temperature: float = 0.5) -> dict:
        ...


class RetrofitInference(BaseZwergeInference):
    """
    ZwerGe-UI inference wrapper.

    Subclasses must set:
      model_type            — a key into a MODEL_TYPE_CONSTANTS-style table
                               (system_message / ground_response / prompt
                               template for this backbone+prompt-family)
      patch_size             — 14 for Qwen2.x-VL, 16 for Qwen3.x-VL
      parse_backbone_coordinate() — parses this backbone's native generated
                               text into a normalized (x, y) point.

    Zoom-backbone system-message overrides (used only by
    predict_zoom_backbone's Stage-2 native-format generate; leave as
    `_ZOOM_NOT_SET` to fall back to the retrofit training prompt):
      _zoom_native_system_message
      _zoom_native_user_template
    """
    _zoom_native_system_message = _ZOOM_NOT_SET
    _zoom_native_user_template  = _ZOOM_NOT_SET

    @classmethod
    def from_checkpoint(cls, ckpt_path: str, attn_impl: str = "flash_attention_2",
                         device: str = "cuda:0", dtype: torch.dtype = torch.bfloat16,
                         max_pixels: Optional[int] = None) -> "RetrofitInference":
        """
        Load the frozen backbone + grounding head from a HuggingFace-style
        checkpoint directory, and resolve this backbone's system_message /
        ground_response / user_prompt_template.

        NOTE: this method references a `model_type_constants` table and a
        `get_model_class` factory that are NOT included in this minimal
        package (they simply select which of the illustrative subclasses in
        modeling.py to instantiate and load the corresponding prompt
        strings). Wire these up to your own checkpoint / prompt config, or
        see the full repository for the exact tables used to produce the
        paper's numbers.
        """
        raise NotImplementedError(
            "from_checkpoint() requires a model-type -> (ModelClass, prompt) "
            "registry that is intentionally out of scope for this minimal "
            "package (see README.md). Instantiate RetrofitInference "
            "directly with an already-loaded `model` and `processor` "
            "instead, e.g.:\n"
            "    grounder = MyRetrofitInference(model=model, processor=processor,\n"
            "                                    system_message=..., ground_response=...)"
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
        """
        ONE prefill-forced forward pass -> per-layer posteriors + fused
        posterior. This is the function that both the paper's Figure-1
        diagnostic (per-layer hit@1) and the deployed ensemble decoder
        (ensemble.py) call.

        Returns dict with keys:
          per_layer_probs, per_layer_points, per_layer_topk, layer_indices,
          active_probe_layers, p_final, omega, n_width, n_height,
          anchor_strategy
        """
        sys_msg  = self.system_message
        grd_resp = self.ground_response
        if sys_msg is None and grd_resp is None:
            raise ValueError(
                "system_message / ground_response are not set. Either pass "
                "them to __init__ or load them from your checkpoint's "
                "training config."
            )

        inputs = build_zwerge_inputs(
            image=image, instruction=instruction, processor=self.processor,
            system_message=sys_msg, ground_response=grd_resp,
            user_prompt_template=self.user_prompt_template,
        )

        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        pixel_values = inputs.get("pixel_values")
        if pixel_values is not None:
            pixel_values = pixel_values.to(device, dtype=self.model.dtype)
        image_grid_thw = inputs.get("image_grid_thw")
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.to(device)
        mm_token_type_ids = inputs.get("mm_token_type_ids")
        if mm_token_type_ids is not None:
            mm_token_type_ids = mm_token_type_ids.to(device)

        if image_grid_thw is not None:
            n_width, n_height = grid_thw_to_nwh(image_grid_thw, merge_size=self.merge_size)
        else:
            w, h = image.size
            patch_size = getattr(
                getattr(self.processor, "image_processor", self.processor), "patch_size", self.patch_size
            )
            cell = patch_size * self.merge_size
            n_width, n_height = max(1, w // cell), max(1, h // cell)

        token_ids_1d = input_ids[0]

        # ── Single prefill: every probed layer's hidden state comes from
        # THIS ONE forward pass (paper: "Both reuse the single prefill ...
        # so the retrofit adds negligible compute over the native baseline"). ──
        all_hidden_states = self.model._forward_hidden_states_for_grounding(
            input_ids=input_ids, attention_mask=attention_mask,
            pixel_values=pixel_values, image_grid_thw=image_grid_thw,
            device=device, mm_token_type_ids=mm_token_type_ids,
        )

        anchor_idx, anchor_strategy = self.model._find_ground_anchor(
            token_ids=token_ids_1d, external_hint=None, verbose=False,
        )
        visual_indices = self.model._get_visual_indices(token_ids_1d)

        if visual_indices.numel() == 0:
            warnings.warn("No visual tokens found in sequence!")
            dummy = torch.ones(1, device=device)
            head = self.model.layerwise_grounding_head
            n_all = len(head.probe_layers)
            n_active = head.num_active_probes
            return {
                "per_layer_probs": [dummy] * n_all,
                "per_layer_points": [(0.5, 0.5)] * n_all,
                "per_layer_topk": [[(0.5, 0.5)]] * n_all,
                "layer_indices": head.probe_layers,
                "active_probe_layers": head.active_probe_layers,
                "p_final": dummy,
                "omega": torch.ones(n_active) / n_active,
                "n_width": n_width, "n_height": n_height,
                "anchor_strategy": anchor_strategy.value,
            }

        # Some backbones (e.g. Qwen3-VL / DeepStack) return a sparse tuple
        # with None at non-hidden-states positions; others (Qwen2.5-VL)
        # return a dense tuple. Handle both uniformly.
        sample_hs = tuple(hs[0] if hs is not None else None for hs in all_hidden_states)

        head_out = self.model.layerwise_grounding_head(
            all_hidden_states=sample_hs, ground_token_idx=anchor_idx,
            visual_indices=visual_indices, labels=None,
        )

        per_layer_probs = head_out["per_layer_probs"]
        p_final = head_out["p_final"]
        omega   = head_out["omega"]
        head    = self.model.layerwise_grounding_head

        per_layer_points, per_layer_topk = [], []
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
            "per_layer_probs": [p.cpu() for p in per_layer_probs],
            "per_layer_points": per_layer_points,
            "per_layer_topk": per_layer_topk,
            "layer_indices": head.probe_layers,
            "active_probe_layers": head.active_probe_layers,
            "p_final": p_final.cpu(),
            "omega": omega.cpu(),
            "n_width": n_width, "n_height": n_height,
            "anchor_strategy": anchor_strategy.value,
        }

    # ── Zoom-backbone / native decode strategy ─────────────────────────────

    def parse_backbone_coordinate(
        self, raw_text: str,
        crop_w_resized: Optional[int] = None, crop_h_resized: Optional[int] = None,
    ) -> Optional[Tuple[float, float]]:
        """
        Parse the backbone's own generated text into a normalized (x, y)
        point in [0,1]. Subclasses MUST override for their output format
        (see README.md for the 4 concrete formats used in the paper:
        UI-TARS-1.5 absolute-pixel `<|box_start|>(x,y)<|box_end|>`,
        GUI-Owl JSON tool-call `[0,1000]`, UI-Venus `[x,y]` `[0,1000]`, etc).
        """
        raise NotImplementedError(f"{type(self).__name__} must implement parse_backbone_coordinate().")

    def _build_generation_inputs(self, image: Image.Image, instruction: str,
                                  max_pixels: Optional[int] = None) -> dict:
        """Build inputs for the backbone's OWN generate() (native coordinate format, not <|ground|>)."""
        sys_msg = (self.system_message if self._zoom_native_system_message is _ZOOM_NOT_SET
                   else self._zoom_native_system_message)
        usr_tmpl = (self.user_prompt_template if self._zoom_native_user_template is _ZOOM_NOT_SET
                    else self._zoom_native_user_template)

        user_text = usr_tmpl.format(instruction) if usr_tmpl else instruction
        conversation = []
        if sys_msg:
            conversation.append({"role": "system", "content": [{"type": "text", "text": sys_msg}]})
        conversation.append({
            "role": "user",
            "content": [{"type": "image", "image": image}, {"type": "text", "text": user_text}],
        })
        text = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(conversation)

        img_proc = getattr(self.processor, "image_processor", None)
        old_max  = getattr(img_proc, "max_pixels", None) if img_proc else None
        if max_pixels is not None and img_proc is not None:
            img_proc.max_pixels = max_pixels
        try:
            result = self.processor(
                text=[text], images=image_inputs if image_inputs else None,
                videos=video_inputs if video_inputs else None,
                return_tensors="pt", padding=True,
            )
        finally:
            if max_pixels is not None and img_proc is not None and old_max is not None:
                img_proc.max_pixels = old_max
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
        region_selector: str = "max",
        p2p_region_scorer: str = "mass_sqrt_area",
        p2p_use_consensus: bool = True,
        min_crop_frac: float = 0.15,
        zoom_upscale_target: int = 0,
    ) -> dict:
        """
        Two-stage decode:
          Stage 1 — ZwerGe prefill (predict_layerwise) -> posteriors -> ROI
          Stage 2 — crop around ROI (or use the full image if `full_image`)
                     -> backbone's own `generate()` -> parse coordinate ->
                     remap to the original image's coordinate frame.

        With `full_image=True` (no crop, ROI selection skipped): this
        reproduces the vanilla backbone's own native accuracy — the
        "Native" baseline column of Table 2, and the `n` (native point)
        input consumed by `ensemble.py`'s `run_p2p_ensemble`.

        `region_selector='p2p'` requires the cross-layer-consensus ROI
        selector `get_zoom_crop_box_p2p` from ensemble.py (not imported
        here to avoid a circular dependency); pass a `crop_box_fn` override
        if you want the P2P-enhanced zoom variant described in the paper's
        supplementary material.
        """
        pred = self.predict_layerwise(
            image=image, instruction=instruction, device=device,
            activation_threshold=activation_threshold, topk=topk,
            decode_strategy=decode_strategy,
            peak_shift_alpha=peak_shift_alpha, temperature=temperature,
        )
        n_w, n_h = pred["n_width"], pred["n_height"]
        W, H = image.size
        token_cell_px = self.patch_size * self.merge_size

        if full_image:
            crop_box = (0, 0, W, H)
            crop_img = image
        else:
            if region_selector == "p2p":
                raise NotImplementedError(
                    "region_selector='p2p' requires get_zoom_crop_box_p2p from "
                    "ensemble.py (cross-layer consensus ROI). Import it there "
                    "and pass the crop_box explicitly, or use region_selector='max'."
                )
            crop_box = get_zoom_crop_box(
                p_final=pred["p_final"], n_width=n_w, n_height=n_h,
                image_w=W, image_h=H, token_cell_px=token_cell_px,
                activation_threshold=activation_threshold, padding_cells=padding_cells,
            )
            crop_img = image.crop(crop_box)

        x_min, y_min, x_max, y_max = crop_box
        crop_w = max(1, x_max - x_min)
        crop_h = max(1, y_max - y_min)

        if zoom_upscale_target and not full_image:
            cur_px = crop_w * crop_h
            if 0 < cur_px < zoom_upscale_target:
                scale = (zoom_upscale_target / cur_px) ** 0.5
                new_w, new_h = max(1, int(crop_w * scale)), max(1, int(crop_h * scale))
                crop_img = crop_img.resize((new_w, new_h), Image.LANCZOS)

        gen_inputs = self._build_generation_inputs(crop_img, instruction)
        gen_inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in gen_inputs.items()}
        if gen_inputs.get("pixel_values") is not None:
            gen_inputs["pixel_values"] = gen_inputs["pixel_values"].to(dtype=self.model.dtype)

        generated_ids = self.model.generate(
            **gen_inputs, max_new_tokens=max_new_tokens, do_sample=False,
            temperature=None, top_p=None,
        )
        prompt_len = gen_inputs["input_ids"].shape[1]
        trimmed = generated_ids[:, prompt_len:]
        raw_text = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0]

        crop_thw = gen_inputs.get("image_grid_thw")
        if crop_thw is not None:
            thw = crop_thw[0] if crop_thw.dim() == 2 else crop_thw.squeeze()
            _, H_ptch, W_ptch = int(thw[0].item()), int(thw[1].item()), int(thw[2].item())
            crop_w_resized: Optional[int] = W_ptch * self.patch_size
            crop_h_resized: Optional[int] = H_ptch * self.patch_size
        else:
            crop_w_resized, crop_h_resized = crop_w, crop_h

        backbone_coord = self.parse_backbone_coordinate(
            raw_text, crop_w_resized=crop_w_resized, crop_h_resized=crop_h_resized,
        )
        if backbone_coord is not None:
            bx_crop, by_crop = backbone_coord
            ox = max(0.0, min(1.0, (x_min + bx_crop * crop_w) / W))
            oy = max(0.0, min(1.0, (y_min + by_crop * crop_h) / H))
            zoom_point = (ox, oy)
        else:
            fb, _ = scores_to_point_and_topk(
                p=pred["p_final"], n_width=n_w, n_height=n_h,
                activation_threshold=activation_threshold, topk=1, decode_strategy="centroid",
            )
            zoom_point = (float(fb[0]), float(fb[1]))

        return {**pred, "zoom_point": zoom_point, "zoom_crop_box": crop_box, "backbone_raw": raw_text}
