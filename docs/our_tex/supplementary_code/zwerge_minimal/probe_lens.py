"""
ZwerGe-UI — Analytical Lenses (spatial lens + serialization lens)
==================================================================
This file implements the two "diagnostic lenses" from the paper's
§"The Coordinate Serialization Bottleneck" that together produce Figure 1
(spatial hit@1 vs. coordinate log-likelihood across layers):

  * SPATIAL LENS  (`compute_spatial_metrics_from_pred`):
      Uses the already-trained, frozen-backbone per-layer grounding probes
      (Stage 1 of modeling.py's LayerWiseGroundingHead) to measure, for
      every probed layer l, whether the decoded point falls inside the
      ground-truth bbox (hit@1) and how much posterior mass lands inside
      the bbox (target_mass). This is the "lightweight per-layer grounding
      head, Stage 1 of [the retrofit], which is about 0.5% of backbone
      parameters and is trained with the backbone completely frozen, so it
      introduces no new spatial knowledge" mentioned in the paper's
      "Experimental setup" paragraph.

  * SERIALIZATION LENS (`logit_lens_nll_hooks` + `build_native_gt_response`
    + `find_coord_token_positions`):
      A PARAMETER-FREE logit-lens: teacher-forced GT coordinate tokens are
      read out through the backbone's OWN (frozen) final LayerNorm + LM
      head, applied to EVERY intermediate layer's hidden state, and the
      resulting negative log-likelihood is reported per layer. No new
      parameters are introduced — this measures how "ready" each layer's
      representation already is to be read by the model's native output
      head, i.e. "coordinate serialization quality [...] measured via the
      backbone's native language model head under teacher forcing, with no
      additional parameters."

The lag between the spatial-lens peak layer L* and the serialization-lens
plateau layer L+ (L* < L+ in all four backbones evaluated) is the paper's
central empirical finding: the "coordinate serialization bottleneck".

Both lenses reuse the SAME single prefill machinery as inference.py
(`RetrofitModelMixin._forward_hidden_states_for_grounding`), so this module
imports directly from `modeling.py` / `inference.py` rather than
re-implementing hidden-state extraction.
"""

import gc
import math
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


# =============================================================================
# Backbone-architecture helpers (Qwen2.5-VL vs Qwen3-VL module layout)
# =============================================================================

def _get_decoder_layers(model):
    """
    Return the list of transformer decoder layers, for either backbone
    family used in the paper.

      Qwen2.5-VL (UI-TARS-1.5-7B, GUI-Owl-7B):
          model -> Qwen2_5_VLForConditionalGeneration
          .model -> Qwen2_5_VLModel
          .model.layers  <- here

      Qwen3-VL (GUI-Owl-1.5-8B, UI-Venus-1.5-8B):
          model -> Qwen3VLForConditionalGeneration
          .model -> Qwen3VLModel
          .model.language_model -> Qwen3VLTextModel
          .model.language_model.layers  <- here (note extra nesting level)
    """
    if hasattr(model, "model"):
        mm = model.model
        if hasattr(mm, "layers"):
            return mm.layers
        if hasattr(mm, "language_model") and hasattr(mm.language_model, "layers"):
            return mm.language_model.layers
    raise AttributeError(f"Cannot find decoder layers on {type(model)}")


def _get_lm_norm(model):
    """Final LM norm (pre-lm_head normalization); same dual layout as above."""
    if hasattr(model, "model"):
        mm = model.model
        if hasattr(mm, "norm"):
            return mm.norm
        if hasattr(mm, "language_model") and hasattr(mm.language_model, "norm"):
            return mm.language_model.norm
    raise AttributeError(f"Cannot find LM norm on {type(model)}")


def _get_lm_head(model):
    return model.lm_head


# =============================================================================
# SPATIAL LENS
# =============================================================================

def compute_target_mass(
    p: torch.Tensor,
    gt_bbox_norm: Tuple[float, float, float, float],
    n_width: int,
    n_height: int,
) -> float:
    """Total posterior probability mass (of a single-layer probe's p_l) inside gt_bbox_norm."""
    x1, y1, x2, y2 = gt_bbox_norm
    rows = torch.arange(n_height, dtype=torch.float32)
    cols = torch.arange(n_width, dtype=torch.float32)
    px1, px2 = cols / n_width, (cols + 1) / n_width
    py1, py2 = rows / n_height, (rows + 1) / n_height
    ox = (px1 < x2) & (px2 > x1)
    oy = (py1 < y2) & (py2 > y1)
    mask = (oy.unsqueeze(1) & ox.unsqueeze(0)).reshape(-1)
    N = min(len(p), len(mask))
    return float(p[:N][mask[:N]].sum().item())


def point_in_bbox(point: Tuple[float, float], bbox_norm: Tuple[float, float, float, float]) -> bool:
    px, py = point
    x1, y1, x2, y2 = bbox_norm
    return x1 <= px <= x2 and y1 <= py <= y2


def compute_spatial_metrics_from_pred(pred: dict, gt_bbox_norm: Tuple[float, float, float, float]) -> dict:
    """
    SPATIAL LENS: compute hit@1 and target_mass per probe layer from a
    `RetrofitInference.predict_layerwise()` output (see inference.py).

    This is Pass 1 ("Spatial") of the paper's two-pass per-sample analysis
    and requires no forward pass beyond the single prefill already done by
    predict_layerwise().

    Returns:
        {
            "hit1_per_layer":  [bool, ...],
            "mass_per_layer":  [float, ...],
            "layer_indices":   [int, ...],
        }
    """
    n_w, n_h = pred["n_width"], pred["n_height"]
    layers = pred["layer_indices"]
    points = pred["per_layer_points"]
    probs  = pred["per_layer_probs"]

    hit1 = [point_in_bbox(pt, gt_bbox_norm) for pt in points]
    mass = [compute_target_mass(p, gt_bbox_norm, n_w, n_h) for p in probs]
    return {"hit1_per_layer": hit1, "mass_per_layer": mass, "layer_indices": list(layers)}


# =============================================================================
# SERIALIZATION LENS
# =============================================================================

def build_native_gt_response(gt_bbox: List[float], image_size: List[int], model_type: str) -> str:
    """
    Build a native-format assistant response string containing the GT
    coordinate, in each backbone's own output vocabulary/format. Used as
    the teacher-forcing target for the serialization lens.

      uitars / guiowl7b / uitars1 (Qwen2.5-VL): absolute-pixel format
          click(start_box='<|box_start|>(x,y)<|box_end|>')
      guiowl (Qwen3-VL):   JSON tool-call, [0,1000]-relative format
          {"name": "computer_use", "arguments": {"action": "left_click",
           "coordinate": [x1000, y1000]}}
      uivenus (Qwen3-VL):  simple [x,y] [0,1000]-relative format
    """
    W, H = float(image_size[0]), float(image_size[1])
    x1, y1, x2, y2 = gt_bbox
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

    if model_type in ("uitars", "guiowl7b", "uitars1"):
        px, py = int(round(cx)), int(round(cy))
        return f"click(start_box='<|box_start|>({px},{py})<|box_end|>')"
    elif model_type == "guiowl":
        x1k, y1k = int(round(cx / W * 1000)), int(round(cy / H * 1000))
        return (f'{{"name": "computer_use", "arguments": '
                f'{{"action": "left_click", "coordinate": [{x1k}, {y1k}]}}}}')
    elif model_type == "uivenus":
        x1k, y1k = int(round(cx / W * 1000)), int(round(cy / H * 1000))
        return f"[{x1k},{y1k}]"
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def _tok_id(tokenizer, token_str: str) -> Optional[int]:
    try:
        tid = tokenizer.convert_tokens_to_ids(token_str)
        if isinstance(tid, int) and tid != tokenizer.unk_token_id:
            return tid
    except Exception:
        pass
    return None


def _find_last_subseq(haystack: List[int], needle: List[int]) -> int:
    n, m = len(haystack), len(needle)
    if m == 0 or m > n:
        return -1
    for i in range(n - m, -1, -1):
        if haystack[i:i + m] == needle:
            return i
    return -1


def _find_coord_by_subsequence(
    ids: List[int], tokenizer, native_response: str,
) -> Optional[Tuple[List[int], List[int], List[int], List[int]]]:
    """
    Tokenize candidate coordinate substrings from `native_response` and
    find their LAST occurrence as a contiguous subsequence in `ids`. This
    is the primary (robust) matcher for Qwen3-VL backbones whose tokenizer
    may split "[123, 456]" in several different ways depending on context.
    """
    coord_match = re.search(r"\[(\d+),\s*(\d+)\]", native_response)
    if coord_match is None:
        return None
    x_str, y_str = coord_match.group(1), coord_match.group(2)

    coord_interior_candidates = [f"{x_str},{y_str}", f"{x_str}, {y_str}",
                                  f" {x_str},{y_str}", f" {x_str}, {y_str}"]
    coord_full_candidates = [f"[{x_str},{y_str}]", f"[{x_str}, {y_str}]", f"[{x_str},{y_str} ]"]

    def _tokenize(s: str) -> List[int]:
        return tokenizer.encode(s, add_special_tokens=False)

    coord_pos: List[int] = []
    coord_ids_found: List[int] = []
    for cand in coord_interior_candidates:
        needle = _tokenize(cand)
        if not needle:
            continue
        start = _find_last_subseq(ids, needle)
        if start >= 0:
            coord_pos = list(range(start, start + len(needle)))
            coord_ids_found = needle
            break
    if not coord_pos:
        return None

    proto_pos: List[int] = []
    proto_ids_found: List[int] = []
    for cand in coord_full_candidates:
        needle = _tokenize(cand)
        if not needle:
            continue
        start = _find_last_subseq(ids, needle)
        if start < 0:
            continue
        end = start + len(needle)
        pre  = [i for i in range(start, coord_pos[0]) if i < end]
        post = [i for i in range(coord_pos[-1] + 1, end)]
        if pre or post:
            proto_pos = pre + post
            proto_ids_found = [ids[i] for i in proto_pos]
            break
    if not proto_pos and coord_pos[0] > 0:
        proto_pos = [coord_pos[0] - 1]
        proto_ids_found = [ids[coord_pos[0] - 1]]

    return coord_pos, coord_ids_found, proto_pos, proto_ids_found


def find_coord_token_positions(
    input_ids_1d: torch.Tensor,
    tokenizer,
    model_type: str,
    native_response: Optional[str] = None,
) -> Tuple[List[int], List[int], List[int], List[int]]:
    """
    Locate the coordinate VALUE tokens and the surrounding PROTOCOL tokens
    (e.g. `<|box_start|>` / `<|box_end|>`, or the enclosing `[` `]`) inside
    a tokenized prompt. Used to build teacher-forcing targets for the
    serialization lens's coordinate-vs-protocol NLL control (paper's
    "Logit-lens controls" / "Non-coordinate control" paragraph, which shows
    that late layers serialize PROTOCOL tokens almost perfectly while a
    substantial residual NLL gap remains specifically for the numeric
    coordinate VALUE, ruling out a generic-alignment confound).

    Returns (coord_positions, coord_ids, protocol_positions, protocol_ids).
    """
    ids = input_ids_1d.tolist()

    if model_type in ("uitars", "guiowl7b", "uitars1"):
        bs_id, be_id = _tok_id(tokenizer, "<|box_start|>"), _tok_id(tokenizer, "<|box_end|>")
        if bs_id is None or be_id is None:
            return [], [], [], []
        try:
            bs = ids.index(bs_id)
            be = ids.index(be_id, bs)
            return list(range(bs + 1, be)), ids[bs + 1:be], [bs, be], [bs_id, be_id]
        except ValueError:
            return [], [], [], []

    elif model_type in ("guiowl", "uivenus"):
        if native_response is not None:
            result = _find_coord_by_subsequence(ids, tokenizer, native_response)
            if result is not None:
                return result
        # Fallback: bracket-scan heuristic.
        last_bracket = -1
        for i in range(len(ids) - 1, -1, -1):
            tok_str = tokenizer.convert_ids_to_tokens([ids[i]])
            if tok_str and ("[" in tok_str[0] or "coordinate" in tok_str[0].lower()):
                last_bracket = i
                break
        if last_bracket < 0:
            return [], [], [], []
        coord_pos, coord_ids_out = [], []
        for i in range(last_bracket + 1, len(ids)):
            tok_str = tokenizer.convert_ids_to_tokens([ids[i]])
            if tok_str and "]" in tok_str[0]:
                break
            coord_pos.append(i)
            coord_ids_out.append(ids[i])
        return coord_pos, coord_ids_out, [last_bracket], [ids[last_bracket]]

    return [], [], [], []


def make_next_token_targets(
    target_positions: List[int], target_ids: List[int],
) -> Tuple[List[int], List[int]]:
    """
    Teacher-forcing shift: convert (position p, token at p) pairs into
    (predictor position p-1, target token at p), since a causal LM's
    hidden state at position p-1 is what predicts the token AT position p.
    Positions with p == 0 are dropped (no prior context to condition on).
    """
    pred_pos, pred_ids = [], []
    for pos, tid in zip(target_positions, target_ids):
        if pos > 0:
            pred_pos.append(pos - 1)
            pred_ids.append(tid)
    return pred_pos, pred_ids


def logit_lens_nll_hooks(
    model,
    inputs: dict,
    target_positions: List[int],
    target_ids: List[int],
    probe_layers: List[int],
    device: torch.device,
) -> Dict[int, float]:
    """
    SERIALIZATION LENS core routine: parameter-free logit-lens NLL.

    For every layer in `probe_layers`, apply the backbone's OWN (frozen)
    final LayerNorm + LM head directly to that layer's intermediate hidden
    state (at the teacher-forcing-shifted predictor position), and compute
    the cross-entropy of the resulting distribution against the true next
    coordinate/protocol token. No probe parameters are introduced: this
    literally reuses the model's existing output head as a "lens" onto
    every intermediate layer (nostalgebraist 2020's logit lens), which is
    exactly the zero-new-parameters measurement the paper uses for
    "coordinate serialization quality".

    Implementation note: this deliberately calls the SAME
    `model._forward_hidden_states_for_grounding` used by
    `RetrofitInference.predict_layerwise` (see modeling.py / inference.py)
    rather than registering forward hooks + raising an early-stop
    exception. The hook+exception approach can orphan intermediate CUDA
    tensors when a Python exception unwinds through a C++ forward call,
    causing persistent memory leaks on some backbones; a clean single
    forward pass with `output_hidden_states=True` has none of that risk.

    Returns {layer_idx: mean_nll_float}.
    """
    if not target_positions or not target_ids:
        return {}

    pred_positions, shifted_target_ids = make_next_token_targets(target_positions, target_ids)
    if not pred_positions:
        return {}

    norm, lm_head = _get_lm_norm(model), _get_lm_head(model)
    param = next(norm.parameters(), None)
    dtype = param.dtype if param is not None else torch.bfloat16
    target_t = torch.tensor(shifted_target_ids, dtype=torch.long, device=device)
    results: Dict[int, float] = {}

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    pixel_values = inputs.get("pixel_values")
    if pixel_values is not None:
        pixel_values = pixel_values.to(device=device, dtype=model.dtype)
    image_grid_thw = inputs.get("image_grid_thw")
    if image_grid_thw is not None:
        image_grid_thw = image_grid_thw.to(device)
    mm_token_type_ids = inputs.get("mm_token_type_ids")
    if mm_token_type_ids is not None:
        mm_token_type_ids = mm_token_type_ids.to(device)

    try:
        with torch.no_grad():
            all_hidden_states = model._forward_hidden_states_for_grounding(
                input_ids=input_ids, attention_mask=attention_mask,
                pixel_values=pixel_values, image_grid_thw=image_grid_thw,
                device=device, mm_token_type_ids=mm_token_type_ids,
            )
    finally:
        del input_ids, attention_mask, pixel_values, image_grid_thw, mm_token_type_ids

    try:
        for li in probe_layers:
            raw_hs = all_hidden_states[li + 1]   # output of decoder layer li
            if raw_hs is None:
                continue
            hs = raw_hs[0] if raw_hs.dim() == 3 else raw_hs
            h_pos  = hs[pred_positions].detach().to(dtype=dtype)
            h_n    = norm(h_pos)
            logits = lm_head(h_n).float()
            nll = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_t[:h_pos.shape[0]].to(logits.device),
                reduction="mean",
            )
            results[li] = float(nll.item())
            del h_pos, h_n, logits, nll, hs, raw_hs
    finally:
        del all_hidden_states, target_t
        gc.collect()
        torch.cuda.empty_cache()

    return results


# =============================================================================
# Counterfactual pair selection (§"Mid-Layer Posterior Is Instruction-
# Conditioned, Not Visual Saliency", Finding 3 / Figure 2)
# =============================================================================

def _bboxes_overlap_norm(bbox_a, bbox_b) -> bool:
    x1a, y1a, x2a, y2a = bbox_a
    x1b, y1b, x2b, y2b = bbox_b
    return not (x2a <= x1b or x2b <= x1a or y2a <= y1b or y2b <= y1a)


def _parse_bbox_norm(rec: dict) -> Optional[Tuple[float, float, float, float]]:
    v = rec.get("gt_bbox_norm")
    if v is None:
        return None
    if isinstance(v, (list, tuple)) and len(v) == 4:
        vals = [float(x) for x in v]
        if max(vals) > 1.5:
            vals = [x / 1000.0 for x in vals]
        return tuple(vals)
    return None


def select_counterfactual_pairs(
    records: List[dict], max_pairs: int = 150, seed: int = 42,
) -> List[Tuple[int, int]]:
    """
    Select (idx_A, idx_B) sample-index pairs for the instruction-switch
    counterfactual (paper's Finding 3): pairs share the same (application,
    image_size) — i.e. plausibly the SAME screenshot — but their
    ground-truth bboxes must NOT overlap (hard-negative filter), so that
    swapping instruction A <-> B genuinely points at a DIFFERENT target on
    the same screen. Used to measure how much the mid-layer spatial
    posterior mass shifts away from the original target when only the
    instruction changes (rules out a passive visual-saliency explanation).
    """
    import random
    groups: Dict[tuple, List[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        key = (rec.get("application", ""), tuple(rec.get("image_size", [])))
        groups[key].append(i)

    rng = random.Random(seed)
    pairs: List[Tuple[int, int]] = []
    for key, idxs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        if len(idxs) < 2:
            continue
        shuffled = list(idxs)
        rng.shuffle(shuffled)
        for i in range(len(shuffled)):
            for j in range(i + 1, len(shuffled)):
                bn_a = _parse_bbox_norm(records[shuffled[i]])
                bn_b = _parse_bbox_norm(records[shuffled[j]])
                if bn_a is None or bn_b is None:
                    continue
                if not _bboxes_overlap_norm(bn_a, bn_b):
                    pairs.append((shuffled[i], shuffled[j]))
                    if len(pairs) >= max_pairs:
                        return pairs
    return pairs
