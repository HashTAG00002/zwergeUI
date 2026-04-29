"""
Layer-wise Grounding Quality Probe  (v3 — 多GPU并行 + 断点续跑 + 定期落盘)
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

多GPU并行 (进程级数据并行，每张卡各自推断自己的数据分片，rank-0 汇总结果):
  # 单卡
  python eval/layer_probe.py --mode probe --gpu_id 0 ...

  # 双卡并行（推荐，速度 ≈ 2x）
  torchrun --nproc_per_node=2 eval/layer_probe.py --mode probe ...
  # 或者手动指定 rank（不依赖 torchrun）：
  CUDA_VISIBLE_DEVICES=0 python eval/layer_probe.py --rank 0 --world_size 2 ... &
  CUDA_VISIBLE_DEVICES=1 python eval/layer_probe.py --rank 1 --world_size 2 ... &
  wait

断点续跑:
  每 --save_every N 条落一次盘，写到 <output_path>.rank<R>.ckpt.json
  重新启动时自动检测 checkpoint 并跳过已完成的样本。
  所有进程都完成后，rank-0 合并所有分片 checkpoint 到最终 output_path。

数据集支持:
  - ScreenSpot-Pro 本地 JSON（--dataset_path 指定 json 文件路径）
  - ScreenSpot-v2  HuggingFace  (--dataset screenspot_v2)

推断引擎:
  HuggingFace Transformers + flash_attention_2（前向）
  QK 重算（O(T*S) per layer）用于提取每层 attention map，避免保存完整 S×S 矩阵。
"""

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

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
# 多GPU 辅助：rank / world_size 自动感知
# ---------------------------------------------------------------------------

def get_dist_info():
    """
    支持两种启动方式:
      1. torchrun: 自动读取 LOCAL_RANK / WORLD_SIZE 环境变量
      2. 手动:     通过 --rank / --world_size 命令行参数（在 parse_args 中处理）
    返回 (rank, world_size, local_rank)
    """
    # torchrun 注入的环境变量
    if "LOCAL_RANK" in os.environ:
        local_rank  = int(os.environ["LOCAL_RANK"])
        rank        = int(os.environ.get("RANK", local_rank))
        world_size  = int(os.environ.get("WORLD_SIZE", 1))
        return rank, world_size, local_rank
    # 手动模式：由 parse_args 注入 sys._dist_rank / sys._dist_world_size
    rank       = getattr(sys, "_dist_rank", 0)
    world_size = getattr(sys, "_dist_world_size", 1)
    local_rank = rank  # 单机多卡场景 rank == local_rank
    return rank, world_size, local_rank


# ---------------------------------------------------------------------------
# Checkpoint 辅助
# ---------------------------------------------------------------------------

def ckpt_path_for_rank(output_path: str, rank: int) -> str:
    """e.g. results/foo.json → results/foo.json.rank0.ckpt.json"""
    return output_path + f".rank{rank}.ckpt.json"


def load_checkpoint(ckpt_path: str):
    """
    返回已完成的样本列表 per_sample_results（list），
    以及 n_done_global（已计入统计的样本数，用于恢复计数器）。
    若 checkpoint 不存在则返回 ([], 0)。
    """
    if not os.path.exists(ckpt_path):
        return [], 0
    try:
        with open(ckpt_path) as f:
            ckpt = json.load(f)
        results = ckpt.get("per_sample_results", [])
        n_done  = ckpt.get("n_done", len(results))
        print(f"[ckpt] Resumed {len(results)} samples from {ckpt_path}")
        return results, n_done
    except Exception as e:
        print(f"[warn] Failed to load checkpoint {ckpt_path}: {e}, starting fresh.")
        return [], 0


def save_checkpoint(ckpt_path: str, per_sample_results: list, n_done: int,
                    extra_meta: dict = None):
    """原子写入（先写tmp再rename，防止断电损坏）"""
    data = {
        "n_done": n_done,
        "per_sample_results": per_sample_results,
    }
    if extra_meta:
        data["meta"] = extra_meta
    tmp = ckpt_path + ".tmp"
    os.makedirs(os.path.dirname(os.path.abspath(ckpt_path)), exist_ok=True)
    with open(tmp, "w") as f:
        json.dump(data, f, default=str)
    os.replace(tmp, ckpt_path)


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
      "hw"       → list[Tensor(nh,)] per-layer head weights  (for diagnostics)
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


def _atomic_save(data, path):
    """先写 .tmp 再 rename，防止写到一半断电损坏文件"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp, path)
    print(f"[saved] → {path}")


# ---------------------------------------------------------------------------
# Run one batch (probe mode) — 支持多GPU分片 + 断点续跑 + 定期落盘
# ---------------------------------------------------------------------------

def run_probe_mode(model, proc, tok, records, args, rank, world_size):
    n_layers = len(model.model.layers)
    ckpt_path = ckpt_path_for_rank(args.output_path, rank)

    # ── 断点续跑：加载已有 checkpoint ──
    prev_results, n_done_prev = load_checkpoint(ckpt_path)
    # checkpoint 里存的是 per-sample 聚合结果（probe模式每条存各层hit/dist/entropy）
    # 恢复各层累计统计
    acc_stats = {"A": [[] for _ in range(n_layers)],
                 "B": [[] for _ in range(n_layers)]}
    entropy_stats = [[] for _ in range(n_layers)]
    for entry in prev_results:
        if entry.get("skipped"):
            continue
        for li in range(n_layers):
            lk = str(li)
            if lk not in entry.get("layer_data", {}):
                continue
            ld = entry["layer_data"][lk]
            for sig in ("A", "B"):
                if f"hit_{sig}" in ld:
                    acc_stats[sig][li].append((ld[f"hit_{sig}"], ld[f"dist_{sig}"]))
            if "entropy" in ld:
                entropy_stats[li].append(ld["entropy"])

    # ── 数据分片：按 rank 切割，跳过已完成 ──
    recs_all = records[:args.num_samples] if args.num_samples > 0 else records
    # 当前 rank 负责的样本（stride 切割，保证均匀）
    my_recs = recs_all[rank::world_size]
    # 跳过已经做完的
    n_skip_ckpt = n_done_prev  # checkpoint 里已完成条数（不含 skip）
    # 注意：checkpoint 里记录的是 n_done（成功处理的），不含 forward 失败的
    # 所以跳过前 n_done_prev 条即可（checkpoint 已按顺序追加）
    my_recs_todo = my_recs[len(prev_results):]   # 用列表长度跳过（包含 skipped entry）

    npts, ptr_pad_toks, ast_start, ptr_start_tok = _setup_tokens(model, tok)
    lp_list = LogitsProcessorList(
        [ForceFollowTokensLogitsProcessorSimple(tokenizer=tok, number_of_points=npts)]
    )

    per_sample_results = list(prev_results)   # 从 checkpoint 继续追加
    n_done = n_done_prev
    n_skip = sum(1 for e in prev_results if e.get("skipped"))
    last_save_idx = len(per_sample_results)

    pbar = tqdm(my_recs_todo,
                desc=f"[rank{rank}|probe]",
                initial=len(prev_results),
                total=len(my_recs))

    for rec in pbar:
        res = _forward_one(model, proc, tok, rec, lp_list, ast_start)
        if res is None:
            n_skip += 1
            per_sample_results.append({"skipped": True})
            # 失败条也算入落盘判断
        else:
            sigs, vis_idx, tgt_idx, nw, nh, bbox_n = res
            gt = ((bbox_n[0]+bbox_n[2])/2, (bbox_n[1]+bbox_n[3])/2)
            entry = {"skipped": False, "layer_data": {}, "bbox_norm": bbox_n,
                     "ui_type": rec["ui_type"], "instruction": rec["instruction"]}
            for li in range(n_layers):
                ld = {}
                if sigs.get("C") and li < len(sigs["C"]):
                    ld["entropy"] = sigs["C"][li]
                    entropy_stats[li].append(sigs["C"][li])
                for sig in ("A", "B"):
                    sl = sigs.get(sig, [])
                    if li >= len(sl) or sl[li] is None or sl[li].numel() == 0:
                        continue
                    px, py = patch_scores_to_point(sl[li], nw, nh, args.activation_threshold)
                    hit = is_point_in_bbox(px, py, bbox_n)
                    dist = euclidean_distance((px, py), gt)
                    ld[f"hit_{sig}"] = hit
                    ld[f"dist_{sig}"] = dist
                    acc_stats[sig][li].append((hit, dist))
                entry["layer_data"][str(li)] = ld
            per_sample_results.append(entry)
            n_done += 1

        # ── 定期落盘 ──
        if len(per_sample_results) - last_save_idx >= args.save_every:
            save_checkpoint(ckpt_path, per_sample_results, n_done,
                            extra_meta={"rank": rank, "world_size": world_size,
                                        "mode": "probe", "args": vars(args)})
            last_save_idx = len(per_sample_results)
            pbar.set_postfix({"saved": len(per_sample_results), "done": n_done})

    # 最后再落一次盘
    save_checkpoint(ckpt_path, per_sample_results, n_done,
                    extra_meta={"rank": rank, "world_size": world_size,
                                "mode": "probe", "args": vars(args)})
    print(f"[rank{rank}|probe] done={n_done}, skip={n_skip}, total_processed={len(per_sample_results)}")

    # ── rank-0 等所有进程完成后合并并生成最终结果 ──
    if world_size > 1:
        _barrier_wait_all_ranks(args.output_path, rank, world_size, timeout=3600)

    if rank == 0:
        _merge_probe_results(args, world_size, n_layers)


def _barrier_wait_all_ranks(output_path: str, rank: int, world_size: int, timeout: int = 3600):
    """
    轻量级文件锁 barrier：
    每个 rank 写一个 done 标记文件，rank-0 等待所有 rank 都写完。
    非 rank-0 的进程在写完标记后直接退出等待。
    """
    done_flag = output_path + f".rank{rank}.done"
    Path(done_flag).touch()
    print(f"[rank{rank}] Wrote done flag: {done_flag}")

    if rank != 0:
        return  # 非 rank-0 不需要等待

    # rank-0 等待所有其他 rank 的 done 标记
    print(f"[rank0] Waiting for all {world_size} ranks to finish...")
    start = time.time()
    while True:
        all_done = all(
            os.path.exists(output_path + f".rank{r}.done")
            for r in range(world_size)
        )
        if all_done:
            break
        if time.time() - start > timeout:
            print(f"[rank0] Timeout waiting for other ranks after {timeout}s, proceeding with available checkpoints.")
            break
        time.sleep(5)

    # 清理 done 标记
    for r in range(world_size):
        flag = output_path + f".rank{r}.done"
        if os.path.exists(flag):
            os.remove(flag)


def _merge_probe_results(args, world_size, n_layers):
    """合并所有 rank 的 checkpoint，生成最终 JSON 结果"""
    print(f"[rank0] Merging {world_size} rank checkpoints...")
    acc_stats = {"A": [[] for _ in range(n_layers)],
                 "B": [[] for _ in range(n_layers)]}
    entropy_stats = [[] for _ in range(n_layers)]
    all_per_sample = []

    for r in range(world_size):
        ckpt_path = ckpt_path_for_rank(args.output_path, r)
        if not os.path.exists(ckpt_path):
            print(f"[warn] rank{r} checkpoint not found: {ckpt_path}")
            continue
        with open(ckpt_path) as f:
            ckpt = json.load(f)
        for entry in ckpt.get("per_sample_results", []):
            all_per_sample.append(entry)
            if entry.get("skipped"):
                continue
            for li in range(n_layers):
                lk = str(li)
                if lk not in entry.get("layer_data", {}):
                    continue
                ld = entry["layer_data"][lk]
                for sig in ("A", "B"):
                    if f"hit_{sig}" in ld:
                        acc_stats[sig][li].append((ld[f"hit_{sig}"], ld[f"dist_{sig}"]))
                if "entropy" in ld:
                    entropy_stats[li].append(ld["entropy"])

    # 汇总统计
    results = {
        "meta": {**vars(args), "world_size": world_size},
        "layer_results": {},
        "n_total": sum(1 for e in all_per_sample if not e.get("skipped")),
        "n_skip": sum(1 for e in all_per_sample if e.get("skipped")),
    }
    print(f"\n{'L':>5} | {'ACC_A':>8} | {'ACC_B':>8} | {'ENTROPY':>9}")
    print("-" * 42)
    for li in range(n_layers):
        ld = {"layer_idx": li}
        for sig in ("A", "B"):
            sp = acc_stats[sig][li]
            ld[f"acc_{sig}"] = float(sum(h for h,_ in sp) / max(len(sp), 1))
            ld[f"mean_dist_{sig}"] = float(sum(d for _,d in sp) / max(len(sp), 1)) if sp else 1.0
            ld[f"n_{sig}"] = len(sp)
        ev = entropy_stats[li]
        ld["entropy_mean"] = float(sum(ev) / max(len(ev), 1))
        ld["entropy_std"]  = float((sum((e - ld["entropy_mean"])**2 for e in ev) / max(len(ev)-1, 1))**0.5)
        results["layer_results"][str(li)] = ld
        print(f"{li:>5} | {ld['acc_A']:>8.4f} | {ld['acc_B']:>8.4f} | {ld['entropy_mean']:>9.4f}")

    _atomic_save(results, args.output_path)

    # 清理各 rank 的 checkpoint 文件
    if not args.keep_ckpt:
        for r in range(world_size):
            ckpt_path = ckpt_path_for_rank(args.output_path, r)
            if os.path.exists(ckpt_path):
                os.remove(ckpt_path)
                print(f"[rank0] Removed checkpoint: {ckpt_path}")
    return results


# ---------------------------------------------------------------------------
# Run one batch (eval mode with layer mask) — 支持多GPU分片 + 断点续跑 + 定期落盘
# ---------------------------------------------------------------------------

def run_eval_mode(model, proc, tok, records, args, rank, world_size):
    n_layers = len(model.model.layers)
    layer_mask_set = parse_layer_mask(args.layer_mask, n_layers)
    if rank == 0:
        print(f"[eval] Layer mask '{args.layer_mask}' → using layers: {sorted(layer_mask_set)}")

    ckpt_path = ckpt_path_for_rank(args.output_path, rank)

    # ── 断点续跑：加载已有 checkpoint ──
    prev_results, n_done_prev = load_checkpoint(ckpt_path)
    n_hit_prev = {"hw": sum(1 for e in prev_results if e.get("hit_hw")),
                  "uni": sum(1 for e in prev_results if e.get("hit_uni"))}

    # ── 数据分片 ──
    recs_all = records[:args.num_samples] if args.num_samples > 0 else records
    my_recs = recs_all[rank::world_size]
    my_recs_todo = my_recs[len(prev_results):]

    npts, ptr_pad_toks, ast_start, ptr_start_tok = _setup_tokens(model, tok)
    lp_list = LogitsProcessorList(
        [ForceFollowTokensLogitsProcessorSimple(tokenizer=tok, number_of_points=npts)]
    )

    n_done = n_done_prev
    n_skip = sum(1 for e in prev_results if e.get("skipped"))
    n_hit  = dict(n_hit_prev)
    hit_by_type = {"hw": defaultdict(lambda: [0, 0]),
                   "uni": defaultdict(lambda: [0, 0])}
    # 从 checkpoint 恢复 hit_by_type
    for entry in prev_results:
        if entry.get("skipped"):
            continue
        ut = entry.get("ui_type", "unknown")
        for variant in ("hw", "uni"):
            if entry.get(f"hit_{variant}"):
                hit_by_type[variant][ut][0] += 1
            if f"hit_{variant}" in entry:
                hit_by_type[variant][ut][1] += 1

    per_sample_results = list(prev_results)
    last_save_idx = len(per_sample_results)

    pbar = tqdm(my_recs_todo,
                desc=f"[rank{rank}|eval|mask={args.layer_mask}]",
                initial=len(prev_results),
                total=len(my_recs))

    for rec in pbar:
        res = _forward_one(model, proc, tok, rec, lp_list, ast_start)
        if res is None:
            n_skip += 1
            per_sample_results.append({"skipped": True})
        else:
            sigs, vis_idx, tgt_idx, nw, nh, bbox_n = res
            gt = ((bbox_n[0]+bbox_n[2])/2, (bbox_n[1]+bbox_n[3])/2)

            map_hw  = aima_aggregate(sigs.get("A_hw", []), layer_mask_set, n_layers)
            map_uni = aima_aggregate(sigs.get("A",    []), layer_mask_set, n_layers)

            if map_hw is None and map_uni is None:
                n_skip += 1
                per_sample_results.append({"skipped": True})
            else:
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

        # ── 定期落盘 ──
        if len(per_sample_results) - last_save_idx >= args.save_every:
            save_checkpoint(ckpt_path, per_sample_results, n_done,
                            extra_meta={"rank": rank, "world_size": world_size,
                                        "mode": "eval", "layer_mask": args.layer_mask,
                                        "args": vars(args)})
            last_save_idx = len(per_sample_results)
            acc_hw_cur = n_hit["hw"] / max(n_done, 1)
            pbar.set_postfix({"saved": len(per_sample_results),
                              "acc_hw": f"{acc_hw_cur:.3f}"})

    # 最后落盘
    save_checkpoint(ckpt_path, per_sample_results, n_done,
                    extra_meta={"rank": rank, "world_size": world_size,
                                "mode": "eval", "layer_mask": args.layer_mask,
                                "args": vars(args)})
    print(f"[rank{rank}|eval] done={n_done}, skip={n_skip}, "
          f"acc_hw={n_hit['hw']/max(n_done,1):.4f}, acc_uni={n_hit['uni']/max(n_done,1):.4f}")

    # ── rank-0 等待并合并 ──
    if world_size > 1:
        _barrier_wait_all_ranks(args.output_path, rank, world_size, timeout=3600)

    if rank == 0:
        _merge_eval_results(args, world_size, layer_mask_set, n_layers)


def _merge_eval_results(args, world_size, layer_mask_set, n_layers):
    """合并所有 rank 的 eval checkpoint，生成最终 JSON 结果"""
    print(f"[rank0] Merging {world_size} eval checkpoints...")
    n_done = 0
    n_skip = 0
    n_hit = {"hw": 0, "uni": 0}
    hit_by_type = {"hw": defaultdict(lambda: [0, 0]),
                   "uni": defaultdict(lambda: [0, 0])}
    all_per_sample = []

    for r in range(world_size):
        ckpt_path = ckpt_path_for_rank(args.output_path, r)
        if not os.path.exists(ckpt_path):
            print(f"[warn] rank{r} checkpoint not found: {ckpt_path}")
            continue
        with open(ckpt_path) as f:
            ckpt = json.load(f)
        for entry in ckpt.get("per_sample_results", []):
            all_per_sample.append(entry)
            if entry.get("skipped"):
                n_skip += 1
                continue
            n_done += 1
            ut = entry.get("ui_type", "unknown")
            for variant in ("hw", "uni"):
                if f"hit_{variant}" in entry:
                    n_hit[variant] += int(entry[f"hit_{variant}"])
                    hit_by_type[variant][ut][0] += int(entry[f"hit_{variant}"])
                    hit_by_type[variant][ut][1] += 1

    acc_hw  = n_hit["hw"]  / max(n_done, 1)
    acc_uni = n_hit["uni"] / max(n_done, 1)
    print(f"\n[eval] mask={args.layer_mask}  total_done={n_done}, total_skip={n_skip}")
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
        "meta": {**vars(args), "layer_mask_set": sorted(layer_mask_set),
                 "world_size": world_size},
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
        "per_sample": all_per_sample,
    }
    _atomic_save(results, args.output_path)

    if not args.keep_ckpt:
        for r in range(world_size):
            ckpt_path = ckpt_path_for_rank(args.output_path, r)
            if os.path.exists(ckpt_path):
                os.remove(ckpt_path)
                print(f"[rank0] Removed checkpoint: {ckpt_path}")
    return results


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
    p.add_argument("--activation_threshold", type=float, default=0.3)
    # Layer mask (for eval mode)
    p.add_argument("--layer_mask", type=str, default="all",
                   help=("Layer selection for eval mode. Options: "
                         "all | last | not_last | first_half | last_half | 0,5,10 | 0-9"))
    # 多GPU: 手动模式（不用 torchrun 时使用）
    p.add_argument("--gpu_id", type=int, default=None,
                   help="(单卡模式) 指定使用哪张 GPU。多卡时请改用 torchrun 或 --rank/--world_size。")
    p.add_argument("--rank", type=int, default=None,
                   help="(手动多卡) 当前进程的 rank (0-based)")
    p.add_argument("--world_size", type=int, default=None,
                   help="(手动多卡) 总进程数（=GPU 数量）")
    # 断点续跑 / 定期落盘
    p.add_argument("--save_every", type=int, default=20,
                   help="每处理 N 条样本保存一次 checkpoint（默认 20）")
    p.add_argument("--keep_ckpt", action="store_true",
                   help="合并完成后保留各 rank 的 checkpoint 文件（默认删除）")
    return p.parse_args()


def main():
    args = parse_args()

    # ── 确定 rank / world_size / 本进程用哪张 GPU ──
    # 优先级: torchrun 环境变量 > 手动 --rank/--world_size > 单卡 --gpu_id
    env_rank, env_ws, env_local = get_dist_info()

    if args.rank is not None:
        # 手动多卡模式：用 --rank 和 --world_size
        rank       = args.rank
        world_size = args.world_size if args.world_size is not None else 1
        local_rank = rank   # 单机假设
    elif env_ws > 1:
        # torchrun 启动
        rank       = env_rank
        world_size = env_ws
        local_rank = env_local
    else:
        # 纯单卡模式
        rank       = 0
        world_size = 1
        local_rank = 0

    # 注入到 sys 供 get_dist_info 回调使用（仅手动模式需要）
    sys._dist_rank       = rank
    sys._dist_world_size = world_size

    # 确定 GPU device
    if args.gpu_id is not None:
        device = f"cuda:{args.gpu_id}"
    else:
        n_gpu = torch.cuda.device_count()
        if n_gpu == 0:
            device = "cpu"
        else:
            device = f"cuda:{local_rank % n_gpu}"

    if rank == 0:
        print(f"[main] mode={args.mode}, world_size={world_size}, "
              f"rank={rank}/{world_size}, device={device}")

    # ── 加载模型（每个进程独立加载到自己的 GPU）──
    if rank == 0:
        print(f"[main] Loading model: {args.model_path}")
    proc = AutoProcessor.from_pretrained(args.model_path, max_pixels=args.max_pixels)
    tok  = proc.tokenizer
    model = Qwen2_5_VLForConditionalGenerationWithPointer.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="flash_attention_2",
    ).eval()
    if rank == 0:
        print(f"[main] Layers: {len(model.model.layers)}, Device: {model.device}")

    # ── 加载数据集 ──
    if rank == 0:
        print(f"[main] Loading dataset ...")
    records = load_dataset_records(args)
    if rank == 0:
        recs_all = records[:args.num_samples] if args.num_samples > 0 else records
        print(f"[main] Dataset size: {len(records)} "
              f"(using {len(recs_all)}, this rank handles ~{len(recs_all)//world_size} samples)")

    # ── 分发到对应模式 ──
    if args.mode == "probe":
        run_probe_mode(model, proc, tok, records, args, rank, world_size)
    else:
        run_eval_mode(model, proc, tok, records, args, rank, world_size)


if __name__ == "__main__":
    main()
