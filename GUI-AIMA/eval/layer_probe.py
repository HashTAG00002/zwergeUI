"""
Layer-wise Grounding Quality Probe  (v2 — ScreenSpot-Pro support + layer mask)
================================================================================

两种运行模式:
  Mode 1 -- "probe" (--mode probe):
      对每一层单独统计 grounding ACC，输出 layer→ACC/DIST/ENTROPY 曲线数据。
      纯分析，不影响模型的最终输出。

  Mode 2 -- "eval" (--mode eval):
      复现 AIMA 的完整推断流程，但支持 --layer_mask 指定哪些层参与聚合。
      例如:
        --layer_mask last        只用最后一层
        --layer_mask first_half  只用前一半层
        --layer_mask 0,5,10,15   只用指定的几层
        --layer_mask all         全部层（与 AIMA 原版等价）
        --layer_mask not_last    去除最后一层
      输出: 整体 ACC + 每个 ui_type 的 ACC，和正式评测格式对齐。

数据集支持:
  - ScreenSpot-Pro 本地 JSON（--dataset_path 指定 json 文件路径）
  - ScreenSpot-v2  HuggingFace  (--dataset screenspot_v2)

推断引擎:
  HuggingFace Transformers + flash_attention_2（前向）
  QK 重算（O(T*S) per layer）用于提取每层 attention map，避免保存完整 S×S 矩阵。

运行示例:
  # Mode 1: probe
  python eval/layer_probe.py --mode probe \\
      --model_path /path/to/GUI-AIMA-3B \\
      --dataset_path /path/to/ScreenSpot-Pro/eval.json \\
      --num_samples 200 --output_path eval_results/probe.json

  # Mode 2: eval with only last layer
  python eval/layer_probe.py --mode eval \\
      --model_path /path/to/GUI-AIMA-3B \\
      --dataset_path /path/to/ScreenSpot-Pro/eval.json \\
      --layer_mask last --output_path eval_results/eval_last_layer.json

  # Mode 2: eval without last layer (all shallow layers)
  python eval/layer_probe.py --mode eval \\
      --model_path /path/to/GUI-AIMA-3B \\
      --dataset_path /path/to/ScreenSpot-Pro/eval.json \\
      --layer_mask not_last --output_path eval_results/eval_no_last.json
"""

import argparse
import json
import math
import os
from collections import defaultdict

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, LogitsProcessorList
from qwen_vl_utils import process_vision_info

from gui_aima.constants import (
    DEFAULT_POINTER_END_TOKEN,
    DEFAULT_POINTER_PAD_TOKEN,
    DEFAULT_POINTER_PAD_TOKEN_list,
    DEFAULT_POINTER_START_TOKEN,
    chat_template,
    grounding_system_message,
)
from gui_aima.inference import ForceFollowTokensLogitsProcessorSimple
from gui_aima.modeling_qwen25vl import Qwen2_5_VLForConditionalGenerationWithPointer
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    apply_multimodal_rotary_pos_emb,
    repeat_kv,
)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def is_point_in_bbox(px, py, bbox_norm):
    """bbox_norm: (x1, y1, x2, y2) all in [0,1]"""
    x1, y1, x2, y2 = bbox_norm
    return x1 <= px <= x2 and y1 <= py <= y2


def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def to_norm_bbox(bbox_pixel, img_w, img_h):
    """Convert pixel bbox to [0,1] normalized bbox."""
    x1, y1, x2, y2 = bbox_pixel
    return (x1 / img_w, y1 / img_h, x2 / img_w, y2 / img_h)


# ---------------------------------------------------------------------------
# Dataset loader: supports both SS-Pro local JSON and HF datasets
# ---------------------------------------------------------------------------

def load_dataset_records(args):
    """
    Returns list of dicts with unified keys:
        image      : PIL.Image
        instruction: str
        bbox_norm  : (x1,y1,x2,y2) in [0,1]
        ui_type    : str  (for SS-Pro breakdown)
        meta       : dict (original record, for saving results)
    """
    if args.dataset_path:
        # ── ScreenSpot-Pro local JSON ──
        with open(args.dataset_path) as f:
            raw = json.load(f)
        records = []
        for item in raw:
            img_w, img_h = item["img_size"]
            bx1, by1, bx2, by2 = item["bbox"]
            bbox_n = to_norm_bbox((bx1, by1, bx2, by2), img_w, img_h)
            img_path = item["images"][0]
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"[warn] cannot open {img_path}: {e}")
                continue
            records.append({
                "image": img,
                "instruction": item["instruction"],
                "bbox_norm": bbox_n,
                "ui_type": item.get("ui_type", "unknown"),
                "meta": item,
            })
        return records
    else:
        # ── HuggingFace ScreenSpot-v2 ──
        from datasets import load_dataset as hf_load
        if args.dataset == "screenspot_v2":
            ds = hf_load("HongxinLi/ScreenSpot_v2")["test"]
        else:
            ds = hf_load("rootsautomation/ScreenSpot")["test"]
        records = []
        for item in ds:
            img = item["image"]
            bbox = item["bbox"]
            # HF screenspot bbox is already [0,1] normalized
            if all(0.0 <= v <= 1.0 for v in bbox):
                bbox_n = tuple(float(v) for v in bbox)
            else:
                bbox_n = to_norm_bbox(bbox, img.width, img.height)
            records.append({
                "image": img,
                "instruction": item["instruction"],
                "bbox_norm": bbox_n,
                "ui_type": item.get("data_type", "unknown"),
                "meta": {},
            })
        return records


# ---------------------------------------------------------------------------
# Connected-region peak finder (same as GUI-AIMA / GUI-Actor)
# ---------------------------------------------------------------------------

def patch_scores_to_point(scores_1d, n_width, n_height, threshold_ratio=0.3):
    max_s = scores_1d.max().item()
    if max_s < 1e-9:
        return (0.5, 0.5)
    thr = max_s * threshold_ratio
    valid = torch.nonzero(scores_1d > thr).squeeze(-1)
    if valid.numel() == 0:
        return (0.5, 0.5)

    vm = {i.item(): scores_1d[i].item() for i in valid}
    coords = [(i.item() // n_width, i.item() % n_width, i.item()) for i in valid]

    visited, regions = set(), []
    for y, x, idx in coords:
        if idx in visited:
            continue
        reg = [(y, x, idx, vm[idx])]
        visited.add(idx)
        q = [(y, x, idx)]
        while q:
            cy, cx, ci = q.pop(0)
            for dy, dx in ((-1,0),(1,0),(0,-1),(0,1)):
                ny, nx = cy+dy, cx+dx
                ni = ny*n_width+nx
                if 0<=ny<n_height and 0<=nx<n_width and ni not in visited and ni in vm:
                    visited.add(ni)
                    reg.append((ny, nx, ni, vm[ni]))
                    q.append((ny, nx, ni))
        regions.append(reg)

    best = max(regions, key=lambda r: max(it[3] for it in r))
    w = sum(it[3] for it in best)
    if w < 1e-9:
        return (0.5, 0.5)
    cx = sum((it[1]+0.5)/n_width * it[3] for it in best) / w
    cy = sum((it[0]+0.5)/n_height * it[3] for it in best) / w
    return (cx, cy)


# ---------------------------------------------------------------------------
# Parse layer_mask argument → set of layer indices
# ---------------------------------------------------------------------------

def parse_layer_mask(mask_str, n_layers):
    """
    Returns a set of layer indices to INCLUDE (i.e. keep non-zero weight).
    Supported formats:
      "all"        → {0,1,...,L-1}
      "last"       → {L-1}
      "not_last"   → {0,...,L-2}
      "first_half" → {0,...,L//2-1}
      "last_half"  → {L//2,...,L-1}
      "0,5,10"     → {0,5,10}  (comma-separated layer indices)
      "0-9"        → {0,1,...,9}
    """
    s = mask_str.strip().lower()
    if s == "all":
        return set(range(n_layers))
    if s == "last":
        return {n_layers - 1}
    if s == "not_last":
        return set(range(n_layers - 1))
    if s == "first_half":
        return set(range(n_layers // 2))
    if s == "last_half":
        return set(range(n_layers // 2, n_layers))
    # range syntax "a-b"
    if "-" in s and "," not in s:
        a, b = s.split("-")
        return set(range(int(a), int(b) + 1))
    # comma-separated
    return {int(x.strip()) for x in s.split(",")}


# ---------------------------------------------------------------------------
# Core: single-sample forward + layer-wise signal extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_layerwise_signals(
    model,
    all_hidden_states,  # tuple (n_layers+1): [l] = input of layer l; [L] = final output
    position_ids,       # (3, 1, seq_len)
    attention_mask,     # (1, seq_len) 2D
    visual_indices,     # LongTensor (V,)
    target_indices,     # LongTensor (T,) — pointer_pad token positions (ANCHOR)
    query_indices,      # LongTensor (Q,) — text tokens between last vis-tok and pointer_start
    signals=("A", "B", "C"),
):
    """
    Returns dict:
      "A"        → list[Tensor(V,)]  per-layer: ANCHOR→visual attn, averaged over heads (uniform weight)
      "A_hw"     → list[Tensor(V,)]  per-layer: ANCHOR→visual attn, AIMA-style head-weighted
                    head weight = softmax( Σ_v  attn_head(query_token_topk1 → visual_v) )
                    等价于 AIMA 的 kl_query_weighting=False, query_topk=1 配置
      "hw"       → list[Tensor(L*H,)] per-layer head weights  (for diagnostics)
      "B"        → list[Tensor(V,)]  per-layer: cosine-sim(ANCHOR_hs, visual_hs)
      "C"        → list[float]       per-layer: visual-patch attention entropy (uniform avg over heads)
    """
    qd = model.model
    L = len(qd.layers)
    device = all_hidden_states[0].device
    bsz, seq_len, _ = all_hidden_states[0].shape

    # 4D causal mask (built once)
    orig = qd.config._attn_implementation
    qd.config._attn_implementation = "eager"
    cmask = qd._update_causal_mask(
        attention_mask, all_hidden_states[0],
        cache_position=torch.arange(seq_len, device=device),
        past_key_values=None, output_attentions=True,
    )
    qd.config._attn_implementation = orig

    cos, sin = qd.rotary_emb(all_hidden_states[0], position_ids)
    t_idx = target_indices.tolist()        # ANCHOR token positions
    q_idx = query_indices.tolist()         # query text token positions (for head weighting)
    # merged: query_indices ++ target_indices, same as AIMA forward()
    merged_idx = q_idx + t_idx
    out = {s: [] for s in signals}
    # always compute A_hw alongside A
    out["A_hw"] = []
    out["hw"] = []

    for l in range(L):
        layer = qd.layers[l]
        sa = layer.self_attn
        ln = layer.input_layernorm(all_hidden_states[l])   # (1, S, d)

        nh, nkv, hd, ng = sa.num_heads, sa.num_key_value_heads, sa.head_dim, sa.num_key_value_groups

        # ── Signal B (output hidden-state cosine sim) ──
        if "B" in signals:
            lo = all_hidden_states[l + 1]
            vn = F.normalize(lo[0, visual_indices, :].float(), dim=-1)
            tn = F.normalize(lo[0, target_indices, :].float(), dim=-1)
            sim = torch.matmul(tn, vn.T).mean(0)           # (V,)
            sim = sim - sim.min()
            sim = sim / sim.sum().clamp_min(1e-9)
            out["B"].append(sim.cpu())
            del lo, vn, tn, sim

        # ── Signal A + C + A_hw (QK recompute) ──
        # Compute K once for the full sequence.
        # Compute Q for merged_idx = query_tokens + ANCHOR (same as AIMA's calculate_attention_from_qk).
        k = sa.k_proj(ln).view(bsz, seq_len, nkv, hd).transpose(1, 2)
        k = repeat_kv(k, ng)                               # (1, nh, S, hd)
        q_merged = sa.q_proj(ln[:, merged_idx, :]).view(bsz, len(merged_idx), nh, hd).transpose(1, 2)
        # (1, nh, Q+T, hd)

        k, _ = apply_multimodal_rotary_pos_emb(k, k.clone(), cos, sin, sa.rope_scaling["mrope_section"])
        q_merged, _ = apply_multimodal_rotary_pos_emb(
            q_merged, q_merged.clone(),
            cos[:, :, merged_idx, :], sin[:, :, merged_idx, :],
            sa.rope_scaling["mrope_section"]
        )

        sc_merged = torch.matmul(q_merged, k.transpose(-2, -1)) / math.sqrt(hd)  # (1, nh, Q+T, S)
        if cmask is not None:
            sc_merged = sc_merged + cmask[:, :, merged_idx, :].to(sc_merged.dtype)
        aw_merged = F.softmax(sc_merged, dim=-1, dtype=torch.float32).to(q_merged.dtype)
        # (1, nh, Q+T, S)
        del sc_merged, k, q_merged

        n_q = len(q_idx)   # number of query tokens
        n_t = len(t_idx)   # number of ANCHOR tokens

        # ---- ANCHOR attention on visual patches: aw_merged[0, :, n_q:, visual_indices]
        anchor_vis = aw_merged[0, :, n_q:, :][:, :, visual_indices]   # (nh, T, V)
        # ---- query attention on visual patches (for head weighting)
        query_vis  = aw_merged[0, :, :n_q, :][:, :, visual_indices]   # (nh, Q, V)

        # ── Signal A: uniform head average ──
        if "A" in signals:
            va = anchor_vis.mean(0).mean(0)                # (V,)  avg over heads, avg over T
            va = va / va.sum().clamp_min(1e-9)
            out["A"].append(va.cpu())

        # ── Signal A_hw: AIMA head-weighted (query_topk=1) ──
        # head_weight_h = Σ_q Σ_v  query_vis[h, q, v]  → softmax over all heads
        head_weights = query_vis.sum(dim=(-1, -2))         # (nh,)  sum over Q and V
        head_weights = head_weights.softmax(dim=-1)        # (nh,)  normalize → probability over heads
        out["hw"].append(head_weights.cpu())
        # weighted sum over heads: (nh,1,1) * (nh, T, V) → mean over T
        va_hw = (head_weights[:, None, None] * anchor_vis).sum(0).mean(0)  # (V,)
        va_hw = va_hw / va_hw.sum().clamp_min(1e-9)
        out["A_hw"].append(va_hw.cpu())

        # ── Signal C: entropy of visual-patch attention (uniform avg) ──
        if "C" in signals:
            ve = anchor_vis.float()                        # (nh, T, V)
            ve = ve / ve.sum(-1, keepdim=True).clamp_min(1e-9)
            ent = -(ve * (ve + 1e-9).log()).sum(-1).mean().item()
            out["C"].append(ent)

        del anchor_vis, query_vis, head_weights, va_hw, aw_merged
        del ln
        torch.cuda.empty_cache()

    return out


# ---------------------------------------------------------------------------
# AIMA-style aggregation with optional layer mask
# ---------------------------------------------------------------------------

def aima_aggregate(
    layer_attn_maps,   # list[Tensor(V,)] length = n_layers  (Signal A or A_hw per layer)
    layer_mask_set,    # set of int: which layers to include
    n_layers,
):
    """
    Layer-masked aggregation:
      - Only layers in layer_mask_set contribute (other layers are zeroed out / excluded)
      - Equal weight across selected layers (arithmetic mean)
      - Returns final grounding map Tensor(V,)

    When called with Signal "A_hw" maps (AIMA head-weighted per layer), this gives
    the exact AIMA behaviour restricted to the selected layers.
    When called with Signal "A" maps (uniform head avg), this is the uniform baseline.
    """
    selected = [layer_attn_maps[l] for l in sorted(layer_mask_set) if l < len(layer_attn_maps)]
    if not selected:
        return None
    stacked = torch.stack(selected, dim=0)      # (n_sel, V)
    merged = stacked.mean(dim=0)                # (V,)
    merged = merged / merged.sum().clamp_min(1e-9)
    return merged


# ---------------------------------------------------------------------------
# Run one batch (probe mode)
# ---------------------------------------------------------------------------

def run_probe_mode(model, proc, tok, records, args):
    n_layers = len(model.model.layers)
    acc_stats = {"A": [[] for _ in range(n_layers)],
                 "B": [[] for _ in range(n_layers)]}
    entropy_stats = [[] for _ in range(n_layers)]

    npts, ptr_pad_toks, ast_start, ptr_start_tok = _setup_tokens(model, tok)
    lp_list = LogitsProcessorList(
        [ForceFollowTokensLogitsProcessorSimple(tokenizer=tok, number_of_points=npts)]
    )

    n_done = n_skip = 0
    for rec in tqdm(records[:args.num_samples] if args.num_samples > 0 else records,
                    desc="[probe]"):
        res = _forward_one(model, proc, tok, rec, lp_list, ast_start)
        if res is None:
            n_skip += 1
            continue

        sigs, vis_idx, tgt_idx, nw, nh, bbox_n = res
        gt = ((bbox_n[0]+bbox_n[2])/2, (bbox_n[1]+bbox_n[3])/2)

        for li in range(n_layers):
            if sigs.get("C") and li < len(sigs["C"]):
                entropy_stats[li].append(sigs["C"][li])
            for sig in ("A", "B"):
                sl = sigs.get(sig, [])
                if li >= len(sl) or sl[li] is None or sl[li].numel() == 0:
                    continue
                px, py = patch_scores_to_point(sl[li], nw, nh, args.activation_threshold)
                acc_stats[sig][li].append(
                    (is_point_in_bbox(px, py, bbox_n), euclidean_distance((px,py), gt))
                )
        n_done += 1

    print(f"\n[probe] done={n_done}, skip={n_skip}")
    results = {"meta": vars(args), "layer_results": {}}
    print(f"\n{'L':>5} | {'ACC_A':>8} | {'ACC_B':>8} | {'ENTROPY':>9}")
    print("-" * 42)
    for li in range(n_layers):
        ld = {"layer_idx": li}
        for sig in ("A","B"):
            sp = acc_stats[sig][li]
            ld[f"acc_{sig}"] = float(sum(h for h,_ in sp)/max(len(sp),1))
            ld[f"mean_dist_{sig}"] = float(sum(d for _,d in sp)/max(len(sp),1)) if sp else 1.0
        ev = entropy_stats[li]
        ld["entropy_mean"] = float(sum(ev)/max(len(ev),1))
        ld["entropy_std"] = float((sum((e-ld["entropy_mean"])**2 for e in ev)/max(len(ev)-1,1))**0.5)
        results["layer_results"][str(li)] = ld
        print(f"{li:>5} | {ld['acc_A']:>8.4f} | {ld['acc_B']:>8.4f} | {ld['entropy_mean']:>9.4f}")

    _save(results, args.output_path)
    return results


# ---------------------------------------------------------------------------
# Run one batch (eval mode with layer mask)
# ---------------------------------------------------------------------------

def run_eval_mode(model, proc, tok, records, args):
    n_layers = len(model.model.layers)
    layer_mask_set = parse_layer_mask(args.layer_mask, n_layers)
    print(f"[eval] Layer mask '{args.layer_mask}' → using layers: {sorted(layer_mask_set)}")

    npts, ptr_pad_toks, ast_start, ptr_start_tok = _setup_tokens(model, tok)
    lp_list = LogitsProcessorList(
        [ForceFollowTokensLogitsProcessorSimple(tokenizer=tok, number_of_points=npts)]
    )

    # Track two variants:
    #   "hw"  = AIMA-style head-weighted (A_hw), exact replication with layer mask applied
    #   "uni" = uniform head average (A), ablation baseline
    n_done = n_skip = 0
    n_hit = {"hw": 0, "uni": 0}
    hit_by_type = {"hw": defaultdict(lambda: [0, 0]),
                   "uni": defaultdict(lambda: [0, 0])}
    per_sample_results = []

    recs = records[:args.num_samples] if args.num_samples > 0 else records
    for rec in tqdm(recs, desc=f"[eval|mask={args.layer_mask}]"):
        res = _forward_one(model, proc, tok, rec, lp_list, ast_start)
        if res is None:
            n_skip += 1
            per_sample_results.append({"skipped": True})
            continue

        sigs, vis_idx, tgt_idx, nw, nh, bbox_n = res
        gt = ((bbox_n[0]+bbox_n[2])/2, (bbox_n[1]+bbox_n[3])/2)

        # Aggregate with layer mask — two variants
        map_hw  = aima_aggregate(sigs.get("A_hw", []), layer_mask_set, n_layers)
        map_uni = aima_aggregate(sigs.get("A",    []), layer_mask_set, n_layers)

        if map_hw is None and map_uni is None:
            n_skip += 1
            per_sample_results.append({"skipped": True})
            continue

        sample_res = {
            "gt": gt, "ui_type": rec["ui_type"],
            "instruction": rec["instruction"],
            "bbox_norm": bbox_n, "skipped": False,
        }

        for variant, gmap in (("hw", map_hw), ("uni", map_uni)):
            if gmap is None:
                sample_res[f"hit_{variant}"] = False
                continue
            px, py = patch_scores_to_point(gmap, nw, nh, args.activation_threshold)
            hit = is_point_in_bbox(px, py, bbox_n)
            dist = euclidean_distance((px, py), gt)
            n_hit[variant] += int(hit)
            hit_by_type[variant][rec["ui_type"]][0] += int(hit)
            hit_by_type[variant][rec["ui_type"]][1] += 1
            sample_res[f"hit_{variant}"] = hit
            sample_res[f"pred_{variant}"] = (px, py)
            sample_res[f"dist_{variant}"] = dist

        n_done += 1
        per_sample_results.append(sample_res)

    acc_hw  = n_hit["hw"]  / max(n_done, 1)
    acc_uni = n_hit["uni"] / max(n_done, 1)
    print(f"\n[eval] mask={args.layer_mask}  done={n_done}, skip={n_skip}")
    print(f"  ACC (AIMA head-weighted): {acc_hw:.4f}  ({n_hit['hw']}/{n_done})")
    print(f"  ACC (uniform heads):      {acc_uni:.4f}  ({n_hit['uni']}/{n_done})")

    print(f"\n{'ui_type':>20} | {'ACC_hw':>8} | {'ACC_uni':>8} | {'n':>6}")
    print("-" * 52)
    all_types = set(hit_by_type["hw"]) | set(hit_by_type["uni"])
    for ut in sorted(all_types):
        h_hw, t_hw   = hit_by_type["hw"][ut]
        h_uni, t_uni = hit_by_type["uni"][ut]
        print(f"{ut:>20} | {h_hw/max(t_hw,1):>8.4f} | {h_uni/max(t_uni,1):>8.4f} | {t_hw:>6}")

    results = {
        "meta": {**vars(args), "layer_mask_set": sorted(layer_mask_set)},
        "overall_acc_hw": acc_hw,
        "overall_acc_uniform": acc_uni,
        "n_hit_hw": n_hit["hw"],
        "n_hit_uniform": n_hit["uni"],
        "n_total": n_done,
        "n_skip": n_skip,
        "acc_by_type_hw": {k: {"acc": h/max(t,1), "n": t, "hit": h}
                           for k,(h,t) in hit_by_type["hw"].items()},
        "acc_by_type_uniform": {k: {"acc": h/max(t,1), "n": t, "hit": h}
                                for k,(h,t) in hit_by_type["uni"].items()},
        "per_sample": per_sample_results,
    }
    _save(results, args.output_path)
    return results


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _setup_tokens(model, tok):
    ptr_pad_id = model.config.pointer_pad_token_id
    npts = len(ptr_pad_id) if isinstance(ptr_pad_id, list) else 1
    ptr_start_tok = tok.encode(DEFAULT_POINTER_START_TOKEN)[0]
    if npts == 1:
        ptr_pad_toks = [tok.encode(DEFAULT_POINTER_PAD_TOKEN)[0]]
        ast_start = ("<|im_start|>assistant<|recipient|>os\n"
                     "pyautogui.click(<|pointer_start|><|pointer_pad|><|pointer_end|>)")
    else:
        ptr_pad_toks = [tok.encode(DEFAULT_POINTER_PAD_TOKEN_list[k])[0] for k in range(npts)]
        inner = "".join(
            f"{DEFAULT_POINTER_START_TOKEN}{DEFAULT_POINTER_PAD_TOKEN_list[k]}{DEFAULT_POINTER_END_TOKEN}"
            for k in range(npts)
        )
        ast_start = f"<|im_start|>assistant<|recipient|>os\npyautogui.click({inner})"
    return npts, ptr_pad_toks, ast_start, ptr_start_tok


def _forward_one(model, proc, tok, rec, lp_list, ast_start):
    """
    Run one sample through the model (placeholder mode, 1 token).
    Returns (layer_signals, vis_idx, tgt_idx, n_width, n_height, bbox_norm)
    or None on error.

    query_indices: text tokens between last visual token and <pointer_start>.
    These are used for AIMA-style head weighting: the head that most strongly
    attends to visual patches (via these query tokens) gets higher weight.
    """
    ptr_start_tok = tok.encode(DEFAULT_POINTER_START_TOKEN)[0]
    npts_cfg = model.config.pointer_pad_token_id
    npts = len(npts_cfg) if isinstance(npts_cfg, list) else 1
    if npts == 1:
        ptr_pad_toks = [tok.encode(DEFAULT_POINTER_PAD_TOKEN)[0]]
    else:
        ptr_pad_toks = [tok.encode(DEFAULT_POINTER_PAD_TOKEN_list[k])[0] for k in range(npts)]

    conv = [
        {"role": "system", "content": [{"type": "text", "text": grounding_system_message}]},
        {"role": "user", "content": [
            {"type": "image", "image": rec["image"]},
            {"type": "text", "text": rec["instruction"]},
        ]},
    ]
    text = proc.apply_chat_template(conv, tokenize=False,
                                    add_generation_prompt=False,
                                    chat_template=chat_template) + ast_start
    imgs_in, vids_in = process_vision_info(conv)
    try:
        inputs = proc(text=[text], images=imgs_in, videos=vids_in,
                      padding=True, return_tensors="pt").to(model.device)
    except Exception as e:
        print(f"[warn] preproc error: {e}")
        return None

    with torch.no_grad():
        try:
            pos_ids, _ = model.get_rope_index(
                input_ids=inputs["input_ids"],
                image_grid_thw=inputs["image_grid_thw"],
                video_grid_thw=None,
                attention_mask=inputs["attention_mask"],
            )
            gen_out = model.generate(
                **inputs,
                max_new_tokens=1,
                logits_processor=lp_list,
                return_dict_in_generate=True,
                output_hidden_states=True,
                output_attentions=False,
                do_sample=False,
            )
        except Exception as e:
            print(f"[warn] forward error: {e}")
            return None

    ids = inputs["input_ids"][0]
    dev = ids.device
    vis_idx = torch.nonzero(ids == model.config.image_token_id, as_tuple=False).squeeze(-1)
    tgt_idx = torch.nonzero(
        torch.isin(ids, torch.tensor(ptr_pad_toks, device=dev)), as_tuple=False
    ).squeeze(-1)
    ps_pos = torch.nonzero(ids == ptr_start_tok, as_tuple=False).squeeze(-1)

    if vis_idx.numel() == 0 or tgt_idx.numel() == 0 or ps_pos.numel() == 0:
        return None

    # query_indices: tokens between last visual token and <pointer_start>
    # mirrors AIMA's modeling_qwen25vl.py lines 331-339
    query_start = vis_idx[-1].item() + 1   # first token after last visual token
    query_end   = ps_pos[0].item()         # position of <pointer_start> (exclusive)
    if query_end > query_start:
        qry_idx = torch.arange(query_start, query_end, device=dev)
    else:
        # fallback: use the single token just before pointer_start
        qry_idx = torch.tensor([max(0, query_end - 1)], device=dev)

    _, n_patch_h, n_patch_w = (inputs["image_grid_thw"][0] // model.visual.spatial_merge_size).tolist()
    prefill_hs = gen_out.hidden_states[0]   # tuple (n_layers+1)

    sigs = compute_layerwise_signals(
        model=model,
        all_hidden_states=prefill_hs,
        position_ids=pos_ids,
        attention_mask=inputs["attention_mask"],
        visual_indices=vis_idx,
        target_indices=tgt_idx,
        query_indices=qry_idx,
        signals=("A", "B", "C"),
    )
    return sigs, vis_idx, tgt_idx, n_patch_w, n_patch_h, rec["bbox_norm"]


def _save(data, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"[saved] → {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Layer-wise Grounding Probe + Eval with Layer Mask")
    p.add_argument("--mode", choices=["probe", "eval"], default="probe",
                   help="probe: per-layer ACC analysis | eval: full grounding with layer mask")
    p.add_argument("--model_path", type=str,
                   default="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/.hdd/models/huggingface.co/GUI_Agents/GUI-AIMA-3B")
    # Dataset options
    p.add_argument("--dataset_path", type=str, default=None,
                   help="Path to ScreenSpot-Pro eval.json (local). If set, overrides --dataset.")
    p.add_argument("--dataset", type=str, default="screenspot_v2",
                   choices=["screenspot", "screenspot_v2"],
                   help="HuggingFace dataset (used only if --dataset_path is not set)")
    p.add_argument("--num_samples", type=int, default=-1,
                   help="Max samples to evaluate. -1 = all.")
    p.add_argument("--output_path", type=str, default="eval_results/layer_probe.json")
    p.add_argument("--max_pixels", type=int, default=5760000)
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--activation_threshold", type=float, default=0.3)
    # Layer mask (for eval mode)
    p.add_argument("--layer_mask", type=str, default="all",
                   help=("Layer selection for eval mode. Options: "
                         "all | last | not_last | first_half | last_half | 0,5,10 | 0-9"))
    return p.parse_args()


def main():
    args = parse_args()
    print(f"[probe] Loading model: {args.model_path}")
    proc = AutoProcessor.from_pretrained(args.model_path, max_pixels=args.max_pixels)
    tok = proc.tokenizer
    model = Qwen2_5_VLForConditionalGenerationWithPointer.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=f"cuda:{args.gpu_id}",
        attn_implementation="flash_attention_2",
    ).eval()
    print(f"[probe] Layers: {len(model.model.layers)}, Device: {model.device}")

    print(f"[probe] Loading dataset ...")
    records = load_dataset_records(args)
    print(f"[probe] Dataset size: {len(records)}")

    if args.mode == "probe":
        run_probe_mode(model, proc, tok, records, args)
    else:
        run_eval_mode(model, proc, tok, records, args)


if __name__ == "__main__":
    main()
