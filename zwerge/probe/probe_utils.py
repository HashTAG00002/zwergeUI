"""
probe_utils.py — Shared utilities for ZwerGe probe experiments.

Experiment 3: Spatial Lens vs. Serialization Lens
Experiment 4: Same-App Counterfactual Target Switch

All imports from existing eval/ and src/ code are read-only.
No modifications to upstream code.
"""

import json
import math
import os
import random
import sys
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# ── sys.path: add eval/ and src/ directories ─────────────────────────────────
_PROBE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.abspath(os.path.join(_PROBE_DIR, "../../.."))
_EVAL_DIR   = os.path.join(_REPO_ROOT, "zwerge", "eval")
_SRC_DIR    = os.path.join(_REPO_ROOT, "zwerge", "src")

for _d in [_EVAL_DIR, _SRC_DIR]:
    if _d not in sys.path:
        sys.path.insert(0, _d)


# ── Architecture helpers ──────────────────────────────────────────────────────

def _get_decoder_layers(model):
    """Return the list of transformer decoder layers, handling Qwen2.5-VL and Qwen3-VL."""
    # Qwen2.5-VL: model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    # Qwen3-VL: model.model.language_model.model.layers
    if (hasattr(model, "model") and hasattr(model.model, "language_model")
            and hasattr(model.model.language_model, "model")
            and hasattr(model.model.language_model.model, "layers")):
        return model.model.language_model.model.layers
    raise AttributeError(f"Cannot find decoder layers on {type(model)}")


def _get_decoder_layer(model, layer_idx: int):
    return _get_decoder_layers(model)[layer_idx]


def _get_lm_norm(model):
    """Return the final LM norm module (pre-lm_head normalization)."""
    # Qwen2.5-VL
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm
    # Qwen3-VL
    if (hasattr(model, "model") and hasattr(model.model, "language_model")
            and hasattr(model.model.language_model, "model")
            and hasattr(model.model.language_model.model, "norm")):
        return model.model.language_model.model.norm
    raise AttributeError(f"Cannot find LM norm on {type(model)}")


def _get_lm_head(model):
    """Return the lm_head (vocabulary projection) module."""
    return model.lm_head


# ── Coordinate token building ─────────────────────────────────────────────────

def build_native_gt_response(
    gt_bbox: List[float],
    image_size: List[int],
    model_type: str,
) -> str:
    """
    Build a native-format assistant response string with GT coordinates.
    Used for the logit-lens (serialization) forward pass.

    For uitars/guiowl7b (Qwen2.5-VL): absolute pixel format
    For guiowl (Qwen3-VL): JSON tool-call [0,1000] format
    For uivenus (Qwen3-VL): simple [x,y] [0,1000] format
    """
    W, H = float(image_size[0]), float(image_size[1])
    x1, y1, x2, y2 = gt_bbox
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

    if model_type in ("uitars", "guiowl7b", "uitars1"):
        px, py = int(round(cx)), int(round(cy))
        return f"click(start_box='<|box_start|>({px},{py})<|box_end|>')"
    elif model_type == "guiowl":
        x1k = int(round(cx / W * 1000))
        y1k = int(round(cy / H * 1000))
        return (f'{{"name": "computer_use", "arguments": '
                f'{{"action": "left_click", "coordinate": [{x1k}, {y1k}]}}}}')
    elif model_type == "uivenus":
        x1k = int(round(cx / W * 1000))
        y1k = int(round(cy / H * 1000))
        return f"[{x1k},{y1k}]"
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def find_coord_token_positions(
    input_ids_1d: torch.Tensor,
    tokenizer,
    model_type: str,
    native_response: Optional[str] = None,
) -> Tuple[List[int], List[int], List[int], List[int]]:
    """
    Find positions of coordinate value tokens and protocol tokens in input_ids.

    For Qwen3/uivenus: uses exact token subsequence search over candidate
    coordinate substrings derived from native_response (most robust).
    Falls back to bracket-scan heuristic if subsequence search fails.

    Returns:
        coord_positions : list of int (positions of digit tokens)
        coord_ids       : list of int (token ids of those positions)
        protocol_positions : list of int (positions of <|box_start|>/<|box_end|> etc.)
        protocol_ids    : list of int
    """
    ids = input_ids_1d.tolist()

    if model_type in ("uitars", "guiowl7b", "uitars1"):
        # Format: <|box_start|>(x,y)<|box_end|>
        bs_id = _tok_id(tokenizer, "<|box_start|>")
        be_id = _tok_id(tokenizer, "<|box_end|>")
        if bs_id is None or be_id is None:
            return [], [], [], []
        try:
            bs = ids.index(bs_id)
            be = ids.index(be_id, bs)
            coord_pos = list(range(bs + 1, be))
            coord_ids = ids[bs + 1:be]
            return coord_pos, coord_ids, [bs, be], [bs_id, be_id]
        except ValueError:
            return [], [], [], []

    elif model_type in ("guiowl", "uivenus"):
        # ── Primary: exact token subsequence search ───────────────────────────
        if native_response is not None:
            result = _find_coord_by_subsequence(ids, tokenizer, native_response)
            if result is not None:
                return result

        # ── Fallback: bracket-scan heuristic ─────────────────────────────────
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


def _find_coord_by_subsequence(
    ids: List[int],
    tokenizer,
    native_response: str,
) -> Optional[Tuple[List[int], List[int], List[int], List[int]]]:
    """
    Tokenize candidate coordinate substrings from native_response and find
    their last occurrence as a contiguous subsequence in ids.

    Handles the coordinate part (digits inside [...]) and the surrounding
    bracket/JSON protocol part separately.
    """
    import re

    # Extract the coordinate portion: last [...] in the response
    coord_match = re.search(r'\[(\d+),\s*(\d+)\]', native_response)
    if coord_match is None:
        return None

    x_str = coord_match.group(1)
    y_str = coord_match.group(2)
    coord_span = coord_match.group(0)  # "[x,y]" or "[x, y]"

    # Build candidate strings to search for the coordinate interior (no brackets)
    coord_interior_candidates = [
        f"{x_str},{y_str}",
        f"{x_str}, {y_str}",
        f" {x_str},{y_str}",
        f" {x_str}, {y_str}",
    ]

    # Build candidate strings for the full coordinate span (with brackets)
    coord_full_candidates = [
        f"[{x_str},{y_str}]",
        f"[{x_str}, {y_str}]",
        f"[{x_str},{y_str} ]",
    ]

    def _find_last_subseq(haystack: List[int], needle: List[int]) -> int:
        """Return start index of last occurrence of needle in haystack, or -1."""
        n, m = len(haystack), len(needle)
        if m == 0 or m > n:
            return -1
        for i in range(n - m, -1, -1):
            if haystack[i:i + m] == needle:
                return i
        return -1

    def _tokenize(s: str) -> List[int]:
        return tokenizer.encode(s, add_special_tokens=False)

    # Search for the coordinate interior tokens
    coord_pos: List[int] = []
    coord_ids_found: List[int] = []
    for cand in coord_interior_candidates:
        needle = _tokenize(cand)
        if not needle:
            continue
        start = _find_last_subseq(ids, needle)
        if start >= 0:
            coord_pos      = list(range(start, start + len(needle)))
            coord_ids_found = needle
            break

    if not coord_pos:
        return None

    # Search for the protocol bracket tokens (the enclosing "[" and "]")
    proto_pos: List[int] = []
    proto_ids_found: List[int] = []
    for cand in coord_full_candidates:
        needle = _tokenize(cand)
        if not needle:
            continue
        start = _find_last_subseq(ids, needle)
        if start < 0:
            continue
        # Protocol = tokens before coord_pos[0] and after coord_pos[-1] within needle span
        end = start + len(needle)
        pre  = [i for i in range(start, coord_pos[0]) if i < end]
        post = [i for i in range(coord_pos[-1] + 1, end)]
        if pre or post:
            proto_pos      = pre + post
            proto_ids_found = [ids[i] for i in proto_pos]
            break

    # Fallback: just use the token immediately before coord_pos[0] as protocol
    if not proto_pos and coord_pos[0] > 0:
        proto_pos      = [coord_pos[0] - 1]
        proto_ids_found = [ids[coord_pos[0] - 1]]

    return coord_pos, coord_ids_found, proto_pos, proto_ids_found


def _tok_id(tokenizer, token_str: str) -> Optional[int]:
    """Convert a token string to id, return None if not in vocab."""
    try:
        tid = tokenizer.convert_tokens_to_ids(token_str)
        if isinstance(tid, int) and tid != tokenizer.unk_token_id:
            return tid
    except Exception:
        pass
    return None


def _tok_id_str(tokenizer, s: str) -> Optional[int]:
    """Tokenize a single string and return the first token id."""
    try:
        ids = tokenizer.encode(s, add_special_tokens=False)
        return ids[0] if ids else None
    except Exception:
        return None


# ── Logit-lens helpers ────────────────────────────────────────────────────────

class _StopForward(Exception):
    """Raised inside a forward hook to abort the forward pass after the last probe layer."""


def make_next_token_targets(
    target_positions: List[int],
    target_ids: List[int],
) -> Tuple[List[int], List[int]]:
    """
    Convert (target_positions, target_ids) into the teacher-forcing-safe form:

      predictor_position p-1  →  predicts token at position p

    Positions with p == 0 are dropped (no prior context).
    Returns (predictor_positions, shifted_target_ids).
    """
    pred_pos: List[int] = []
    pred_ids: List[int] = []
    for pos, tid in zip(target_positions, target_ids):
        if pos > 0:
            pred_pos.append(pos - 1)
            pred_ids.append(tid)
    return pred_pos, pred_ids


# ── Logit-lens NLL via forward hooks ─────────────────────────────────────────

def logit_lens_nll_hooks(
    model,
    inputs: dict,
    target_positions: List[int],
    target_ids: List[int],
    probe_layers: List[int],
    device: torch.device,
) -> Dict[int, float]:
    """
    Memory-efficient logit lens using PyTorch forward hooks.

    Token shift (teacher-forcing fix): for each target token at position p,
    we use the hidden state at position p-1 to predict token p.  This avoids
    the situation where position p's hidden state has already "seen" token p
    via the embedding, which would artificially deflate NLL at all layers.

    Early-exit via _StopForward: after the last probe layer fires its hook the
    forward pass is aborted, avoiding the cost of remaining layers and the
    final lm_head projection over the full sequence.

    Args:
        model           : the ZwerGe retrofit model (UITARSRetrofitModel etc.)
        inputs          : dict from build_zwerge_inputs() (input_ids, pixel_values, ...)
        target_positions: sequence positions whose tokens we want to predict
        target_ids      : ground-truth token ids AT those positions (p, not p-1)
        probe_layers    : layer indices to probe (must be sorted ascending)
        device          : torch device

    Returns:
        {layer_idx: mean_nll_float}
    """
    if not target_positions or not target_ids:
        return {}

    # Apply token shift: use hidden state at p-1 to predict token at p
    pred_positions, shifted_target_ids = make_next_token_targets(target_positions, target_ids)
    if not pred_positions:
        return {}

    norm     = _get_lm_norm(model)
    lm_head  = _get_lm_head(model)
    target_t = torch.tensor(shifted_target_ids, dtype=torch.long, device=device)
    results: Dict[int, float] = {}

    max_probe_layer = max(probe_layers)

    def make_hook(li: int):
        def hook_fn(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out  # [bsz, seq_len, d]
            # Use predictor positions (p-1) to predict tokens at p
            h_pos  = h[0, pred_positions, :].detach().float()  # [n_target, d]
            h_n    = norm(h_pos)                                # [n_target, d]
            logits = lm_head(h_n)                               # [n_target, vocab]
            nll = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_t[:h_pos.shape[0]].to(logits.device),
                reduction="mean",
            )
            results[li] = float(nll.item())
            del h_pos, h_n, logits, nll
            # Abort forward pass after the last probe layer — saves compute
            if li == max_probe_layer:
                raise _StopForward
        return hook_fn

    hooks = []
    for li in probe_layers:
        layer = _get_decoder_layer(model, li)
        hooks.append(layer.register_forward_hook(make_hook(li)))

    try:
        dev_inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }
        if "pixel_values" in dev_inputs and dev_inputs["pixel_values"] is not None:
            dev_inputs["pixel_values"] = dev_inputs["pixel_values"].to(
                device=device, dtype=model.dtype
            )

        with torch.no_grad():
            try:
                model(
                    output_hidden_states=False,
                    output_attentions=False,
                    use_cache=False,
                    return_dict=True,
                    **dev_inputs,
                )
            except _StopForward:
                pass  # expected early exit — results are populated
    finally:
        for h in hooks:
            h.remove()

    return results


# ── Spatial metrics ───────────────────────────────────────────────────────────

def compute_target_mass(
    p: torch.Tensor,
    gt_bbox_norm: Tuple[float, float, float, float],
    n_width: int,
    n_height: int,
) -> float:
    """
    Compute the total posterior probability mass inside gt_bbox_norm.

    p            : [N_vis] posterior (softmax output from probe head)
    gt_bbox_norm : (x1, y1, x2, y2) normalized to [0, 1]
    """
    x1, y1, x2, y2 = gt_bbox_norm
    rows = torch.arange(n_height, dtype=torch.float32)
    cols = torch.arange(n_width,  dtype=torch.float32)
    px1 = cols / n_width
    px2 = (cols + 1) / n_width
    py1 = rows / n_height
    py2 = (rows + 1) / n_height
    ox   = (px1 < x2) & (px2 > x1)   # [n_width]
    oy   = (py1 < y2) & (py2 > y1)   # [n_height]
    mask = (oy.unsqueeze(1) & ox.unsqueeze(0)).reshape(-1)   # [N_vis]
    N    = min(len(p), len(mask))
    return float(p[:N][mask[:N]].sum().item())


def point_in_bbox(
    point: Tuple[float, float],
    bbox_norm: Tuple[float, float, float, float],
) -> bool:
    px, py = point
    x1, y1, x2, y2 = bbox_norm
    return x1 <= px <= x2 and y1 <= py <= y2


def compute_spatial_metrics_from_pred(
    pred: dict,
    gt_bbox_norm: Tuple[float, float, float, float],
) -> dict:
    """
    Compute hit@1 and target_mass per probe layer from predict_layerwise() output.

    Returns:
        {
            "hit1_per_layer":  [bool, ...],
            "mass_per_layer":  [float, ...],
            "layer_indices":   [int, ...],
        }
    """
    n_w  = pred["n_width"]
    n_h  = pred["n_height"]
    layers = pred["layer_indices"]
    points = pred["per_layer_points"]
    probs  = pred["per_layer_probs"]

    hit1  = [point_in_bbox(pt, gt_bbox_norm) for pt in points]
    mass  = [
        compute_target_mass(p, gt_bbox_norm, n_w, n_h)
        for p in probs
    ]
    return {"hit1_per_layer": hit1, "mass_per_layer": mass, "layer_indices": list(layers)}


# ── Counterfactual pair selection ─────────────────────────────────────────────

def _bboxes_overlap_norm(
    bbox_a: Tuple[float, float, float, float],
    bbox_b: Tuple[float, float, float, float],
) -> bool:
    """Return True if two normalized bboxes have any intersection."""
    x1a, y1a, x2a, y2a = bbox_a
    x1b, y1b, x2b, y2b = bbox_b
    return not (x2a <= x1b or x2b <= x1a or y2a <= y1b or y2b <= y1a)


def select_counterfactual_pairs(
    records: List[dict],
    max_pairs: int = 150,
    seed: int = 42,
) -> List[Tuple[int, int]]:
    """
    Select (idx_A, idx_B) pairs for counterfactual experiment.

    Grouping: same (application, image_size) — exact match to avoid
    coordinate normalization issues.
    Hard-negative filter: gt_bboxes must NOT overlap.

    Returns list of (idx_A, idx_B) index pairs.
    """
    groups: Dict[tuple, List[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        key = (rec.get("application", ""), tuple(rec.get("image_size", [])))
        groups[key].append(i)

    rng = random.Random(seed)
    pairs: List[Tuple[int, int]] = []

    # Sort by group size descending for diversity
    for key, idxs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        if len(idxs) < 2:
            continue
        shuffled = list(idxs)
        rng.shuffle(shuffled)
        for i in range(len(shuffled)):
            for j in range(i + 1, len(shuffled)):
                a = records[shuffled[i]]
                b = records[shuffled[j]]
                # Parse gt_bbox_norm
                bn_a = _parse_bbox_norm(a)
                bn_b = _parse_bbox_norm(b)
                if bn_a is None or bn_b is None:
                    continue
                if not _bboxes_overlap_norm(bn_a, bn_b):
                    pairs.append((shuffled[i], shuffled[j]))
                    if len(pairs) >= max_pairs:
                        return pairs
    return pairs


def _parse_bbox_norm(rec: dict) -> Optional[Tuple[float, float, float, float]]:
    """Parse gt_bbox_norm from a dataset record. Returns (x1,y1,x2,y2) in [0,1]."""
    v = rec.get("gt_bbox_norm")
    if v is None:
        return None
    if isinstance(v, (list, tuple)) and len(v) == 4:
        vals = [float(x) for x in v]
        # If values > 1, they're in [0,1000] scale → normalize
        if max(vals) > 1.5:
            vals = [x / 1000.0 for x in vals]
        return tuple(vals)
    return None


# ── Data loading ──────────────────────────────────────────────────────────────

def load_eval_records(eval_json: str) -> List[dict]:
    """Load evaluation records from eval.json."""
    with open(eval_json, "r") as f:
        records = json.load(f)
    return records


def sample_records(
    records: List[dict],
    n: int,
    seed: int = 42,
    stratify_key: Optional[str] = "ui_type",
) -> List[dict]:
    """
    Sample n records, optionally stratified by stratify_key.
    Returns shuffled sample.
    """
    if n >= len(records):
        return list(records)

    rng = random.Random(seed)

    if stratify_key and records[0].get(stratify_key):
        groups: Dict[str, List[dict]] = defaultdict(list)
        for r in records:
            groups[r.get(stratify_key, "unknown")].append(r)
        per_group = max(1, n // len(groups))
        sampled = []
        for g in groups.values():
            shuffled = list(g)
            rng.shuffle(shuffled)
            sampled.extend(shuffled[:per_group])
        # Top up if needed
        remaining = [r for r in records if r not in set(sampled)]
        rng.shuffle(remaining)
        sampled.extend(remaining[:max(0, n - len(sampled))])
        return sampled[:n]

    shuffled = list(records)
    rng.shuffle(shuffled)
    return shuffled[:n]


# ── JSONL I/O ─────────────────────────────────────────────────────────────────

def append_jsonl(path: str, obj: dict) -> None:
    """Append a JSON object as a line to a .jsonl file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> List[dict]:
    """Read all records from a .jsonl file."""
    results = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def get_done_ids(output_path: str, id_key: str = "sample_id") -> set:
    """Return set of already-processed IDs for resumable runs."""
    if not os.path.exists(output_path):
        return set()
    done = set()
    for rec in read_jsonl(output_path):
        v = rec.get(id_key)
        if v is not None:
            done.add(v)
    return done


# ── Inference class loader ────────────────────────────────────────────────────

def get_inference_class(model_type: str):
    """Import and return the appropriate RetrofitInference subclass."""
    from eval_retrofit import get_inference_class as _gic
    return _gic(model_type)
