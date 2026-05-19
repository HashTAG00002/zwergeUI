#!/usr/bin/env python3
"""
ZwerGe-UI Layer-wise Accuracy Profiling
========================================
评测各 probe layer 的独立准确率分布（不使用 fusion，直接用每层 p_l 独立预测）。

目标：
  - 了解哪些层 grounding 能力最强
  - 验证层融合是否真的有增益（与 fusion 结果对比）
  - 辅助调参 probe_layers 选取

Probe Layer 来源：
  自动从 ckpt 的 config.probe_layers 读取，有多少层就评多少层，无需手动指定。
  例：config 里 probe_layers=[10,13,16,19,22,25,27] → 表格自动出 7 行，多一行 fusion 对比。

输出：
  - {bench_key}_layerwise_summary.json      各层 hit_top1 / overlap_top1 汇总
  - {bench_key}_layerwise_results.json      逐样本各层预测（可选，--save_per_sample）
  - {bench_key}_layerwise_aggregated.json   多卡合并后的最终结果（多卡时）
  - layerwise_all_summary.json              所有 bench 的汇总（bench=all 时）

用法：
  python eval_layerwise.py --ckpt <ckpt_dir> --bench ss_pro
  python eval_layerwise.py --ckpt <ckpt_dir> --bench all
  # 多卡自动并行（spawn 子进程，每 GPU 处理一个分片，最终汇总打印一次）
  python eval_layerwise.py --ckpt <ckpt_dir> --bench all
"""

import argparse
import glob
import json
import os
import sys
import time
import warnings
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR     = os.path.join(_SCRIPT_DIR, "..", "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from inference_zwerge import (
    load_zwerge_model,
    build_zwerge_inputs,
    grid_thw_to_nwh,
    get_prediction_region_point,
    point_in_bbox,
    do_boxes_overlap,
)


# ─────────────────────────────────────────────────────────────────────────────
# 辅助：明确类型的 patch posterior → 点坐标（消除 linter 联合类型警告）
# ─────────────────────────────────────────────────────────────────────────────

def _scores_to_point_and_topk(
    p: torch.Tensor,
    n_width: int,
    n_height: int,
    activation_threshold: float,
    topk: int,
) -> Tuple[Tuple[float, float], List[Tuple[float, float]]]:
    """
    将 patch 后验 p [N_vis] 转换为 top-1 预测点和 topk 候选点列表。
    明确返回类型，供 linter 静态分析使用（等价于 get_prediction_region_point 的 4-tuple 路径）。
    """
    result = get_prediction_region_point(
        attn_scores=p.unsqueeze(0),
        n_width=n_width,
        n_height=n_height,
        activation_threshold=activation_threshold,
        return_all_regions=True,
    )
    # result = (best_point, sorted_centers, sorted_scores, sorted_points)
    best: Tuple[float, float] = result[0]   # type: ignore[index]
    centers: List[Tuple[float, float]] = result[1]  # type: ignore[index]
    return best, centers[:topk]


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark 配置（与 eval_zwerge.py 完全一致）
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
        "name":        "OSWorld-G (non-refusal)",
        "eval_dir":    "OSWorld-G",
        "eval_json":   "eval.json",
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


# ─────────────────────────────────────────────────────────────────────────────
# Layer-wise forward：返回每层独立的 p_l，以及 fusion 的 p_final（作为 baseline 对比）
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def zwerge_predict_layerwise(
    image: Image.Image,
    instruction: str,
    model,
    processor,
    device: torch.device,
    activation_threshold: float = 0.3,
    topk: int = 3,
    merge_size: int = 2,
) -> dict:
    """
    ZwerGe-UI 逐层推理（prefill-only）。

    与 zwerge_predict 相比，额外返回：
      per_layer_probs:  list[Tensor[N_vis]]  每层独立的 patch 后验
      layer_indices:    list[int]            对应的 transformer 层 index
      per_layer_points: list[(px,py)]        每层 top-1 预测点

    Returns dict:
        per_layer_probs:   list[Tensor]   各层 patch 后验（CPU）
        per_layer_points:  list[(px,py)]  各层 top-1 归一化预测点
        per_layer_topk:    list[list]     各层 topk 预测点列表
        layer_indices:     list[int]      各 probe layer 的 transformer 层 index
        p_final:           Tensor[N_vis]  fusion 后验（CPU，作为对比）
        omega:             Tensor         层融合权重（CPU）
        n_width, n_height: int
        anchor_strategy:   str
    """
    from zwerge_retrofit.constants import GROUNDING_SYSTEM_MESSAGE, GROUND_RESPONSE_CLICK

    inputs = build_zwerge_inputs(
        image=image,
        instruction=instruction,
        processor=processor,
        system_message=GROUNDING_SYSTEM_MESSAGE,
        ground_response=GROUND_RESPONSE_CLICK,
    )

    input_ids      = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    pixel_values   = inputs.get("pixel_values")
    if pixel_values is not None:
        pixel_values = pixel_values.to(device, dtype=model.dtype)
    image_grid_thw = inputs.get("image_grid_thw")
    if image_grid_thw is not None:
        image_grid_thw = image_grid_thw.to(device)

    # Grid size
    if image_grid_thw is not None:
        n_width, n_height = grid_thw_to_nwh(image_grid_thw, merge_size=merge_size)
    else:
        w, h = image.size
        cell = 14 * merge_size
        n_width  = max(1, w // cell)
        n_height = max(1, h // cell)

    token_ids_1d = input_ids[0]

    # ── 1. Embed ──────────────────────────────────────────────────────────────
    inputs_embeds = model.model.embed_tokens(input_ids)
    if pixel_values is not None:
        pv = pixel_values.to(model.dtype)
        image_embeds = model.visual(pv, grid_thw=image_grid_thw)
        n_img_tokens = (input_ids == model.config.image_token_id).sum().item()
        n_img_feats  = image_embeds.shape[0]
        if n_img_tokens != n_img_feats:
            warnings.warn(
                f"Image token mismatch: seq={n_img_tokens}, visual={n_img_feats}"
            )
        image_mask = (
            (input_ids == model.config.image_token_id)
            .unsqueeze(-1).expand_as(inputs_embeds)
            .to(inputs_embeds.device)
        )
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    # ── 2. RoPE ──────────────────────────────────────────────────────────────
    position_ids, _ = model.get_rope_index(
        input_ids, image_grid_thw, None, attention_mask
    )

    # ── 3. Transformer forward ────────────────────────────────────────────────
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
    all_hidden_states = transformer_out.hidden_states   # (L+1) × [seq_len, d_model]

    # ── 4. Anchor & visual indices ────────────────────────────────────────────
    anchor_idx, anchor_strategy = model._find_ground_anchor(
        token_ids=token_ids_1d,
        external_hint=None,
        verbose=False,
    )
    visual_indices = model._get_visual_indices(token_ids_1d)

    if visual_indices.numel() == 0:
        warnings.warn("No visual tokens found in sequence!")
        dummy = torch.ones(1, device=device) / 1
        n_probes = len(model.layerwise_grounding_head.probe_layers)
        return {
            "per_layer_probs":  [dummy] * n_probes,
            "per_layer_points": [(0.5, 0.5)] * n_probes,
            "per_layer_topk":   [[(0.5, 0.5)]] * n_probes,
            "layer_indices":    model.layerwise_grounding_head.probe_layers,
            "p_final":          dummy,
            "omega":            torch.ones(n_probes) / n_probes,
            "n_width":          n_width,
            "n_height":         n_height,
            "anchor_strategy":  anchor_strategy.value,
        }

    sample_hs = tuple(hs[0] for hs in all_hidden_states)   # [seq_len, d_model]

    # ── 5. Run grounding head（with labels=None，不算 loss）────────────────────
    head_out = model.layerwise_grounding_head(
        all_hidden_states=sample_hs,
        ground_token_idx=anchor_idx,
        visual_indices=visual_indices,
        labels=None,
    )

    per_layer_probs = head_out["per_layer_probs"]   # list[Tensor[N_vis]]
    p_final         = head_out["p_final"]           # Tensor[N_vis]
    omega           = head_out["omega"]             # Tensor[num_probes]
    layer_indices   = model.layerwise_grounding_head.probe_layers  # list[int]

    # ── 6. 每层 p_l → 点坐标 ─────────────────────────────────────────────────
    per_layer_points = []
    per_layer_topk   = []
    for p_l in per_layer_probs:
        best, centers = _scores_to_point_and_topk(
            p=p_l,
            n_width=n_width,
            n_height=n_height,
            activation_threshold=activation_threshold,
            topk=topk,
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


# ─────────────────────────────────────────────────────────────────────────────
# 核心评测循环
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_bench_layerwise(
    bench_key: str,
    eval_root: str,
    model,
    processor,
    device: torch.device,
    output_dir: str,
    topk: int = 3,
    activation_threshold: float = 0.3,
    start: int = 0,
    end: int = -1,
    save_per_sample: bool = False,
    verbose: bool = True,   # False → 不打印汇总表（多卡分片子进程用，避免重复输出）
) -> Dict:
    cfg          = BENCH_CONFIGS[bench_key]
    bench_name   = cfg["name"]
    eval_json_path = os.path.join(eval_root, cfg["eval_dir"], cfg["eval_json"])
    img_root     = os.path.join(eval_root, cfg["eval_dir"])

    assert os.path.exists(eval_json_path), f"eval.json not found: {eval_json_path}"
    with open(eval_json_path) as f:
        data = json.load(f)

    if end == -1:
        end = len(data)
    data  = data[start:end]
    total = len(data)
    print(f"\n[{bench_name}] {total} samples (slice {start}:{end})")

    # probe layer 信息（第一个样本推理后才能拿到，先用 config 里的值）
    probe_layers = list(model.layerwise_grounding_head.probe_layers)
    n_probes     = len(probe_layers)

    # 每层独立的计数器
    # layer_stats[i] → {"hit1": int, "hitk": int, "overlap1": int, "overlapk": int, "total": int}
    layer_stats = [{
        "hit1": 0, "hitk": 0, "overlap1": 0, "overlapk": 0, "total": 0
    } for _ in range(n_probes)]

    # fusion 的计数器（对比用）
    fusion_stats = {"hit1": 0, "hitk": 0, "overlap1": 0, "overlapk": 0, "total": 0}

    results      = []
    skip_total   = 0

    pbar = tqdm(enumerate(data), total=total, desc=f"{bench_name} [layerwise]",
                dynamic_ncols=True)
    for idx, example in pbar:
        global_idx = start + idx

        img_path = os.path.join(img_root, example["image_path"])
        if not os.path.exists(img_path):
            warnings.warn(f"Image not found: {img_path}, skipping #{global_idx}")
            skip_total += 1
            continue

        W, H = float(example["image_size"][0]), float(example["image_size"][1])
        x1, y1, x2, y2 = example["gt_bbox"]
        gt_bbox_norm = (x1 / W, y1 / H, x2 / W, y2 / H)

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            warnings.warn(f"Failed to open image {img_path}: {e}")
            skip_total += 1
            continue

        try:
            pred = zwerge_predict_layerwise(
                image=image,
                instruction=example["instruction"],
                model=model,
                processor=processor,
                device=device,
                activation_threshold=activation_threshold,
                topk=topk,
            )
        except Exception as e:
            warnings.warn(f"Inference failed for #{global_idx}: {e}")
            skip_total += 1
            continue

        n_w, n_h = pred["n_width"], pred["n_height"]
        phx = 0.5 / n_w
        phy = 0.5 / n_h

        # ── 逐层指标 ──────────────────────────────────────────────────────────
        layer_metrics = []
    for li in range(n_probes):
        lpt       = pred["per_layer_points"][li]   # (px, py)
        px        = float(lpt[0])
        py        = float(lpt[1])
        tk_pts    = pred["per_layer_topk"][li]
        hit1      = int(point_in_bbox(px, py, gt_bbox_norm))
        hitk      = int(any(point_in_bbox(float(p[0]), float(p[1]), gt_bbox_norm) for p in tk_pts))
        pred_box  = (px - phx, py - phy, px + phx, py + phy)
        overlap1  = int(do_boxes_overlap(pred_box, gt_bbox_norm))
        overlapk  = overlap1
        for pk_pt in tk_pts[1:]:
            pk_x = float(pk_pt[0])
            pk_y = float(pk_pt[1])
            pk_box = (pk_x - phx, pk_y - phy, pk_x + phx, pk_y + phy)
            if do_boxes_overlap(pk_box, gt_bbox_norm):
                overlapk = 1
                break

            layer_stats[li]["hit1"]     += hit1
            layer_stats[li]["hitk"]     += hitk
            layer_stats[li]["overlap1"] += overlap1
            layer_stats[li]["overlapk"] += overlapk
            layer_stats[li]["total"]    += 1
            layer_metrics.append({
                "hit_top1": hit1, "hit_topk": hitk,
                "overlap_top1": overlap1, "overlap_topk": overlapk,
                "pred_point": list(pred["per_layer_points"][li]),
            })

        # ── fusion 指标（对比） ────────────────────────────────────────────────
        f_best, _ = _scores_to_point_and_topk(
            p=pred["p_final"],
            n_width=n_w, n_height=n_h,
            activation_threshold=activation_threshold,
            topk=1,
        )
        fpx = float(f_best[0])
        fpy = float(f_best[1])
        fhit1 = int(point_in_bbox(fpx, fpy, gt_bbox_norm))
        fpred_box = (fpx - phx, fpy - phy, fpx + phx, fpy + phy)
        fov1  = int(do_boxes_overlap(fpred_box, gt_bbox_norm))
        fusion_stats["hit1"]     += fhit1
        fusion_stats["overlap1"] += fov1
        fusion_stats["total"]    += 1

        if save_per_sample:
            rec = {
                "idx":             global_idx,
                "image_path":      example["image_path"],
                "instruction":     example["instruction"],
                "gt_bbox_norm":    list(gt_bbox_norm),
                "anchor_strategy": pred["anchor_strategy"],
                "n_width":         n_w,
                "n_height":        n_h,
                "omega":           pred["omega"].tolist(),
                "probe_layers":    probe_layers,
                "layer_metrics":   layer_metrics,
                "fusion_hit1":     fhit1,
                "fusion_overlap1": fov1,
            }
            for extra in ["id", "ui_type", "group", "platform", "application",
                          "data_type", "split", "grounding_type", "task_type",
                          "GUI_types", "category", "element_type"]:
                if extra in example:
                    rec[extra] = example[extra]
            results.append(rec)

        # 实时进度：显示 best layer 当前准确率（hit1 最高的层）
        valid_so_far = idx + 1 - skip_total
        if valid_so_far > 0:
            best_li   = max(range(n_probes), key=lambda i: layer_stats[i]["hit1"])
            best_acc  = layer_stats[best_li]["hit1"] / valid_so_far * 100
            fuse_acc  = fusion_stats["hit1"]          / valid_so_far * 100
            best_lay  = probe_layers[best_li]
            pbar.set_postfix({
                f"L{best_lay}(best)": f"{best_acc:.1f}%",
                "fusion":             f"{fuse_acc:.1f}%",
                "skip":               skip_total,
            })

    # ─────────────────────────────────────────────────────────────────────────
    # 汇总
    # ─────────────────────────────────────────────────────────────────────────
    valid_total = total - skip_total
    nn = valid_total if valid_total > 0 else 1

    layer_accs = []
    for li, ls in enumerate(layer_stats):
        n = ls["total"] if ls["total"] > 0 else 1
        layer_accs.append({
            "layer_idx":     probe_layers[li],
            "probe_rank":    li,
            "hit_top1":      round(ls["hit1"]     / n * 100, 4),
            "overlap_top1":  round(ls["overlap1"] / n * 100, 4),
            "hit_topk":      round(ls["hitk"]     / n * 100, 4),
            "overlap_topk":  round(ls["overlapk"] / n * 100, 4),
            "n":             ls["total"],
        })

    # 排序（hit_top1 降序）
    layer_accs_sorted = sorted(layer_accs, key=lambda x: x["hit_top1"], reverse=True)

    fn = fusion_stats["total"] if fusion_stats["total"] > 0 else 1
    fusion_acc = {
        "hit_top1":    round(fusion_stats["hit1"]     / fn * 100, 4),
        "overlap_top1":round(fusion_stats["overlap1"] / fn * 100, 4),
    }

    summary = {
        "bench":         bench_name,
        "bench_key":     bench_key,
        "total":         total,
        "valid":         valid_total,
        "skipped":       skip_total,
        "topk":          topk,
        "slice":         [start, end],
        "probe_layers":  probe_layers,
        "layer_accs":    layer_accs,        # 按 probe 顺序
        "layer_accs_sorted": layer_accs_sorted,  # 按 hit_top1 降序
        "fusion_acc":    fusion_acc,
    }

    os.makedirs(output_dir, exist_ok=True)
    suffix = f"_{start}-{end}" if (start > 0 or end != total + start) else ""

    spath = os.path.join(output_dir, f"{bench_key}_layerwise_summary{suffix}.json")
    with open(spath, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if save_per_sample and results:
        rpath = os.path.join(output_dir, f"{bench_key}_layerwise_results{suffix}.json")
        with open(rpath, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"[{bench_name}] Per-sample results → {rpath}")

    if verbose:
        _print_layerwise_summary(summary, topk)
    return summary


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
    print(f"  {'fusion':>11}  {fa['hit_top1']:>7.2f}%  {fa['overlap_top1']:>7.2f}%")
    print(f"{'='*72}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 多卡并行（与 eval_zwerge.py 相同的 spawn 分片方案）
# ─────────────────────────────────────────────────────────────────────────────

def _worker(gpu_id: int, ckpt_path: str, bench_key: str, eval_root: str,
            output_dir: str, topk: int, activation_threshold: float,
            attn_impl: str, start: int, end: int,
            max_pixels: int, save_per_sample: bool):
    import torch
    device = torch.device(f"cuda:{gpu_id}")
    model, processor = load_zwerge_model(
        ckpt_path=ckpt_path,
        attn_implementation=attn_impl,
        device=str(device),
        dtype=torch.bfloat16,
    )
    processor.image_processor.max_pixels = max_pixels
    evaluate_bench_layerwise(
        bench_key=bench_key,
        eval_root=eval_root,
        model=model,
        processor=processor,
        device=device,
        output_dir=output_dir,
        topk=topk,
        activation_threshold=activation_threshold,
        start=start,
        end=end,
        save_per_sample=save_per_sample,
        verbose=False,   # 分片不打印，合并后由主进程统一打印
    )


def _aggregate_layerwise_shards(output_dir: str, bench_key: str) -> Optional[Dict]:
    """合并多个分片的 layerwise_summary，加权平均各层准确率。"""
    pattern = os.path.join(output_dir, f"{bench_key}_layerwise_summary_*.json")
    files   = sorted(glob.glob(pattern))
    # 也包含无分片后缀的（单卡结果）
    single  = os.path.join(output_dir, f"{bench_key}_layerwise_summary.json")
    if os.path.exists(single) and not files:
        return json.load(open(single))
    if not files:
        print(f"[aggregate] No layerwise shard files found for {bench_key}")
        return None

    summaries = [json.load(open(fp)) for fp in files]

    probe_layers = summaries[0]["probe_layers"]
    n_probes     = len(probe_layers)
    bench_name   = summaries[0]["bench"]

    # 加权合并（按每个 shard 的 n 样本数）
    merged_layer_stats = [{"hit1": 0, "hitk": 0, "overlap1": 0, "overlapk": 0, "total": 0}
                          for _ in range(n_probes)]
    merged_fusion = {"hit1": 0, "overlap1": 0, "total": 0}

    for sm in summaries:
        for li, la in enumerate(sm["layer_accs"]):
            n = la["n"]
            merged_layer_stats[li]["hit1"]     += round(la["hit_top1"]    / 100 * n)
            merged_layer_stats[li]["hitk"]     += round(la["hit_topk"]    / 100 * n)
            merged_layer_stats[li]["overlap1"] += round(la["overlap_top1"]/ 100 * n)
            merged_layer_stats[li]["overlapk"] += round(la["overlap_topk"]/ 100 * n)
            merged_layer_stats[li]["total"]    += n
        fn = sm["valid"]
        merged_fusion["hit1"]     += round(sm["fusion_acc"]["hit_top1"]    / 100 * fn)
        merged_fusion["overlap1"] += round(sm["fusion_acc"]["overlap_top1"]/ 100 * fn)
        merged_fusion["total"]    += fn

    total  = sum(sm["total"]  for sm in summaries)
    valid  = sum(sm["valid"]  for sm in summaries)
    skipped= sum(sm["skipped"]for sm in summaries)

    layer_accs = []
    for li, ls in enumerate(merged_layer_stats):
        n = ls["total"] if ls["total"] > 0 else 1
        layer_accs.append({
            "layer_idx":    probe_layers[li],
            "probe_rank":   li,
            "hit_top1":     round(ls["hit1"]     / n * 100, 4),
            "overlap_top1": round(ls["overlap1"] / n * 100, 4),
            "hit_topk":     round(ls["hitk"]     / n * 100, 4),
            "overlap_topk": round(ls["overlapk"] / n * 100, 4),
            "n":            ls["total"],
        })
    layer_accs_sorted = sorted(layer_accs, key=lambda x: x["hit_top1"], reverse=True)

    fn = merged_fusion["total"] if merged_fusion["total"] > 0 else 1
    fusion_acc = {
        "hit_top1":     round(merged_fusion["hit1"]     / fn * 100, 4),
        "overlap_top1": round(merged_fusion["overlap1"] / fn * 100, 4),
    }

    summary = {
        "bench":              bench_name,
        "bench_key":          bench_key,
        "total":              total,
        "valid":              valid,
        "skipped":            skipped,
        "probe_layers":       probe_layers,
        "layer_accs":         layer_accs,
        "layer_accs_sorted":  layer_accs_sorted,
        "fusion_acc":         fusion_acc,
    }

    agg_path = os.path.join(output_dir, f"{bench_key}_layerwise_aggregated.json")
    with open(agg_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    _print_layerwise_summary(summary, topk=3)
    return summary


def run_bench_layerwise_parallel(
    bench_key: str,
    ckpt_path: str,
    eval_root: str,
    output_dir: str,
    topk: int,
    activation_threshold: float,
    attn_impl: str,
    max_pixels: int,
    save_per_sample: bool,
) -> Optional[Dict]:
    import multiprocessing as mp
    import torch

    n_gpu = torch.cuda.device_count()
    if n_gpu == 0:
        raise RuntimeError("No CUDA GPU found.")

    cfg = BENCH_CONFIGS[bench_key]
    eval_json_path = os.path.join(eval_root, cfg["eval_dir"], cfg["eval_json"])
    with open(eval_json_path) as f:
        N = len(json.load(f))
    print(f"[ZwerGe layerwise] {bench_key}: {N} samples, {n_gpu} GPU(s)")

    if n_gpu == 1:
        device = torch.device("cuda:0")
        model, processor = load_zwerge_model(
            ckpt_path=ckpt_path,
            attn_implementation=attn_impl,
            device="cuda:0",
            dtype=torch.bfloat16,
        )
        processor.image_processor.max_pixels = max_pixels
        return evaluate_bench_layerwise(
            bench_key=bench_key,
            eval_root=eval_root,
            model=model,
            processor=processor,
            device=device,
            output_dir=output_dir,
            topk=topk,
            activation_threshold=activation_threshold,
            start=0,
            end=N,
            save_per_sample=save_per_sample,
        )

    chunk  = (N + n_gpu - 1) // n_gpu
    slices = []
    for i in range(n_gpu):
        s = i * chunk
        e = min((i + 1) * chunk, N)
        if s >= N:
            break
        slices.append((i, s, e))

    ctx   = mp.get_context("spawn")
    procs = []
    for gpu_id, s, e in slices:
        print(f"[ZwerGe layerwise]   GPU{gpu_id}: slice [{s}, {e})")
        p = ctx.Process(
            target=_worker,
            args=(gpu_id, ckpt_path, bench_key, eval_root, output_dir,
                  topk, activation_threshold, attn_impl, s, e,
                  max_pixels, save_per_sample),
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"Worker process exited with code {p.exitcode}")

    return _aggregate_layerwise_shards(output_dir=output_dir, bench_key=bench_key)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="ZwerGe-UI Layer-wise Accuracy Profiling")
    parser.add_argument("--ckpt",       required=True,  help="Checkpoint 目录")
    parser.add_argument("--bench",      default="ss_pro",
                        choices=list(BENCH_CONFIGS.keys()) + ["all"])
    parser.add_argument("--eval_dir",   default="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/datasets/evaluation")
    parser.add_argument("--output_dir", default="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/zwerge/data/results/zwerge_layerwise")
    parser.add_argument("--attn_impl",  default="flash_attention_2",
                        choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--max_pixels",           type=int,   default=12_845_056)
    parser.add_argument("--activation_threshold", type=float, default=0.3)
    parser.add_argument("--topk",                 type=int,   default=3)
    parser.add_argument("--save_per_sample",      action="store_true",
                        help="保存每个样本各层的预测结果（磁盘空间较大，默认关闭）")
    return parser.parse_args()


def main():
    args = parse_args()

    assert os.path.isdir(args.ckpt), f"Checkpoint not found: {args.ckpt}"
    output_dir = os.path.join(args.output_dir, os.path.basename(args.ckpt))
    os.makedirs(output_dir, exist_ok=True)
    print(f"[ZwerGe layerwise] Output dir: {output_dir}")

    bench_keys    = list(BENCH_CONFIGS.keys()) if args.bench == "all" else [args.bench]
    all_summaries = {}

    for bench_key in bench_keys:
        t0 = time.time()
        summary = run_bench_layerwise_parallel(
            bench_key=bench_key,
            ckpt_path=args.ckpt,
            eval_root=args.eval_dir,
            output_dir=output_dir,
            topk=args.topk,
            activation_threshold=args.activation_threshold,
            attn_impl=args.attn_impl,
            max_pixels=args.max_pixels,
            save_per_sample=args.save_per_sample,
        )
        elapsed = time.time() - t0
        if summary:
            summary["elapsed_s"] = round(elapsed, 1)
            all_summaries[bench_key] = summary
        print(f"[{bench_key}] Elapsed: {elapsed/60:.1f} min")

    # ── 跨 bench 汇总打印 ─────────────────────────────────────────────────────
    if len(all_summaries) > 1:
        # 收集所有出现过的 probe layer
        all_layers = []
        for sm in all_summaries.values():
            for la in sm.get("layer_accs", []):
                if la["layer_idx"] not in all_layers:
                    all_layers.append(la["layer_idx"])
        all_layers = sorted(all_layers)

        print("\n" + "=" * 90)
        print("  FINAL LAYER-WISE SUMMARY  (hit_top1 %)")
        print("=" * 90)
        header = f"  {'benchmark':30s}" + "".join(f"  L{l:>2}" for l in all_layers) + "  fusion"
        print(header)
        print(f"  {'-'*86}")

        for bk, sm in all_summaries.items():
            bench_name = sm.get("bench", bk)
            # build layer_idx → hit_top1 map
            l2acc = {la["layer_idx"]: la["hit_top1"] for la in sm.get("layer_accs", [])}
            f_acc = sm.get("fusion_acc", {}).get("hit_top1", 0.0)
            row = f"  {bench_name:30s}"
            for l in all_layers:
                acc = l2acc.get(l, float("nan"))
                row += f"  {acc:5.1f}"
            row += f"  {f_acc:5.1f}"
            print(row)
        print("=" * 90)

    with open(os.path.join(output_dir, "layerwise_all_summary.json"), "w") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)
    print(f"[ZwerGe layerwise] Done. Results → {output_dir}")


if __name__ == "__main__":
    main()
