#!/usr/bin/env python3
"""
ZwerGe-UI Evaluation Script
=============================
支持五个评测 Benchmark（统一读取 eval.json）：
  - ScreenSpot-Pro    (SS-Pro)    P0 主指标
  - ScreenSpot-v2     (SS-v2)     P0 主指标
  - OSWorld-G         (non-refusal only，box_type=='bbox')
  - MMBench-GUI L2    (MMBench)
  - UI-Vision                     P3

使用标准化的 eval.json 格式（见 build_eval_jsons.py）：
  image_path:    相对于 eval.json 所在目录的图片路径
  image_size:    [W, H]
  gt_bbox:       [x1, y1, x2, y2] 绝对像素坐标
  gt_bbox_norm:  [x1n, y1n, x2n, y2n] 归一化 0-1000 整数
  instruction:   原始指令

推理原理（ZwerGe-UI prefill-only）：
  1. 构造 conversation（system + user(image+instruction) + assistant(<|ground|>前缀)）
  2. 单次 model forward（output_hidden_states=True, no generate）
  3. LayerWiseGroundingHead 输出 p_final [N_vis]（patch 后验概率分布）
  4. get_prediction_region_point → 归一化 (px, py) ∈ [0,1]
  5. 判断 px×W, py×H 是否落在 gt_bbox 内 → hit@1 / hit@k / overlap@1

评测指标：
  hit@1:  top-1 预测点落在 gt_bbox 内
  hit@k:  前 k 个候选点中有任意一个落在 gt_bbox 内（k=3）

用法：
  python eval_zwerge.py \\
      --ckpt /path/to/checkpoint \\
      --bench ss_pro \\
      --eval_dir /path/to/evaluation/ \\
      --output_dir /path/to/results/ \\
      [--attn_impl flash_attention_2] \\
      [--topk 3] \\
      [--max_pixels 5760000] \\
      [--start 0 --end -1] \\
      [--bs 1]

  bench 可选：ss_pro | ss_v2 | osworld_g | mmbench | ui_vision | all
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from tqdm import tqdm

# ── 添加 zwerge src 到 path ───────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR     = os.path.join(_SCRIPT_DIR, "..", "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from inference_zwerge import (
    load_zwerge_model,
    zwerge_predict,
    point_in_bbox,
    topk_hit,
)


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark configurations
# ─────────────────────────────────────────────────────────────────────────────

BENCH_CONFIGS = {
    "ss_pro": {
        "name":     "ScreenSpot-Pro",
        "eval_dir": "ScreenSpot-Pro",    # relative to eval_root
        "eval_json": "eval.json",
        "group_field": "ui_type",        # for per-category breakdown
        "filter": None,
    },
    "ss_v2": {
        "name":     "ScreenSpot-v2",
        "eval_dir": "ScreenSpot-v2",
        "eval_json": "eval.json",
        "group_field": "data_type",
        "filter": None,
    },
    "osworld_g": {
        "name":     "OSWorld-G (non-refusal)",
        "eval_dir": "OSWorld-G",
        "eval_json": "eval.json",
        "group_field": "GUI_types",
        "filter": lambda x: True,        # refusal items already excluded in eval.json
    },
    "mmbench": {
        "name":     "MMBench-GUI-L2",
        "eval_dir": "MMBench-GUI",
        "eval_json": "eval.json",
        "group_field": "grounding_type",
        "filter": None,
    },
    "ui_vision": {
        "name":     "UI-Vision",
        "eval_dir": "UI-Vision",
        "eval_json": "eval.json",
        "group_field": "task_type",
        "filter": None,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_bench(
    bench_key: str,
    eval_root: str,
    model,
    processor,
    device: torch.device,
    output_dir: str,
    topk: int = 3,
    max_pixels: int = 5_760_000,
    activation_threshold: float = 0.3,
    start: int = 0,
    end: int = -1,
    save_per_sample: bool = True,
) -> Dict:
    """
    评测单个 benchmark。

    Args:
        bench_key:           "ss_pro" | "ss_v2" | "osworld_g" | "mmbench" | "ui_vision"
        eval_root:           eval_json 所在根目录（含各 bench 子目录）
        model:               UITARSRetrofitModel（eval模式）
        processor:           AutoProcessor
        device:              torch.device
        output_dir:          结果保存目录
        topk:                top-k 候选点数量
        max_pixels:          processor 的 max_pixels（控制图像分辨率上限）
        activation_threshold: patch BFS 阈值
        start, end:          评测数据切片（方便多卡并行分段，-1 表示到末尾）
        save_per_sample:     是否保存每条结果（用于 debug / 可视化）

    Returns: 汇总结果 dict
    """
    cfg = BENCH_CONFIGS[bench_key]
    bench_name = cfg["name"]
    eval_json_path = os.path.join(eval_root, cfg["eval_dir"], cfg["eval_json"])
    img_root       = os.path.join(eval_root, cfg["eval_dir"])

    assert os.path.exists(eval_json_path), f"eval.json not found: {eval_json_path}"

    # ── 加载数据 ──────────────────────────────────────────────────────────────
    with open(eval_json_path, "r") as f:
        data = json.load(f)

    # 切片
    if end == -1:
        end = len(data)
    data = data[start:end]
    total = len(data)
    print(f"\n[{bench_name}] {total} samples (slice {start}:{end})")

    # ── 结果容器 ──────────────────────────────────────────────────────────────
    results = []
    hit1_total  = 0
    hitk_total  = 0
    skip_total  = 0
    group_stats: Dict[str, Dict] = {}   # group → {hit1, hitk, total}

    # ── 推理循环 ──────────────────────────────────────────────────────────────
    pbar = tqdm(enumerate(data), total=total, desc=bench_name, dynamic_ncols=True)
    for idx, example in pbar:
        global_idx = start + idx

        # 图片路径
        img_path = os.path.join(img_root, example["image_path"])
        if not os.path.exists(img_path):
            warnings.warn(f"Image not found: {img_path}, skipping sample {global_idx}")
            skip_total += 1
            continue

        # GT bbox（归一化 [0,1]）
        img_size = example["image_size"]  # [W, H]
        W, H = float(img_size[0]), float(img_size[1])
        gt_bbox_abs = example["gt_bbox"]  # [x1, y1, x2, y2] 绝对像素
        gt_bbox_norm = (
            gt_bbox_abs[0] / W,
            gt_bbox_abs[1] / H,
            gt_bbox_abs[2] / W,
            gt_bbox_abs[3] / H,
        )

        instruction = example["instruction"]

        # 加载图片
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            warnings.warn(f"Failed to open image {img_path}: {e}, skipping.")
            skip_total += 1
            continue

        # 推理
        try:
            pred = zwerge_predict(
                image=image,
                instruction=instruction,
                model=model,
                processor=processor,
                device=device,
                topk=topk,
                activation_threshold=activation_threshold,
            )
        except Exception as e:
            warnings.warn(f"Inference failed for sample {global_idx}: {e}")
            skip_total += 1
            continue

        # 判断 hit
        px, py = pred["pred_point"]
        hit1 = int(point_in_bbox(px, py, gt_bbox_norm))
        hitk = int(topk_hit(pred["topk_points"], gt_bbox_norm))

        hit1_total += hit1
        hitk_total += hitk

        # Group 统计
        group_key = _get_group_key(example, cfg["group_field"])
        if group_key not in group_stats:
            group_stats[group_key] = {"hit1": 0, "hitk": 0, "total": 0}
        group_stats[group_key]["hit1"]  += hit1
        group_stats[group_key]["hitk"]  += hitk
        group_stats[group_key]["total"] += 1

        # 记录结果
        rec = {
            "idx":              global_idx,
            "image_path":       example["image_path"],
            "instruction":      instruction,
            "gt_bbox_norm":     list(gt_bbox_norm),
            "pred_point":       list(pred["pred_point"]),
            "topk_points":      [list(p) for p in pred["topk_points"]],
            "hit1":             hit1,
            "hitk":             hitk,
            "anchor_strategy":  pred["anchor_strategy"],
            "n_width":          pred["n_width"],
            "n_height":         pred["n_height"],
        }
        # 保留 bench-specific fields
        for extra_key in ["id", "ui_type", "group", "platform", "application",
                           "data_type", "split", "grounding_type", "task_type",
                           "GUI_types", "category", "element_type", "box_type"]:
            if extra_key in example:
                rec[extra_key] = example[extra_key]
        results.append(rec)

        # 进度条更新
        valid_count = idx + 1 - skip_total
        acc1 = hit1_total / valid_count * 100 if valid_count > 0 else 0.0
        accK = hitk_total / valid_count * 100 if valid_count > 0 else 0.0
        pbar.set_postfix({"hit@1": f"{acc1:.1f}%", f"hit@{topk}": f"{accK:.1f}%",
                           "skip": skip_total})

    # ── 汇总 ──────────────────────────────────────────────────────────────────
    valid_total = total - skip_total
    acc1_overall = hit1_total / valid_total * 100 if valid_total > 0 else 0.0
    accK_overall = hitk_total / valid_total * 100 if valid_total > 0 else 0.0

    # Per-group accuracy
    group_accs = {}
    for grp, st in sorted(group_stats.items()):
        n = st["total"]
        if n == 0:
            continue
        group_accs[grp] = {
            "hit@1":   round(st["hit1"] / n * 100, 2),
            f"hit@{topk}": round(st["hitk"] / n * 100, 2),
            "total":   n,
        }

    summary = {
        "bench":          bench_name,
        "bench_key":      bench_key,
        "total":          total,
        "valid":          valid_total,
        "skipped":        skip_total,
        "hit1":           hit1_total,
        f"hit{topk}":     hitk_total,
        "acc@1":          round(acc1_overall, 4),
        f"acc@{topk}":    round(accK_overall, 4),
        "group_accs":     group_accs,
        "topk":           topk,
        "slice":          [start, end],
    }

    # ── 保存结果 ──────────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    suffix = f"_{start}-{end}" if (start > 0 or end != total + start) else ""
    results_path = os.path.join(output_dir, f"{bench_key}_results{suffix}.json")
    summary_path = os.path.join(output_dir, f"{bench_key}_summary{suffix}.json")

    if save_per_sample:
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"[{bench_name}] Per-sample results saved to {results_path}")

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ── 打印汇总 ──────────────────────────────────────────────────────────────
    _print_summary(summary, topk)

    return summary


def _get_group_key(example: dict, group_field: Optional[str]) -> str:
    """从样本中提取 group key（用于分类别统计）。"""
    if group_field is None:
        return "all"
    val = example.get(group_field, "unknown")
    if isinstance(val, list):
        return ",".join(str(v) for v in val)
    return str(val)


def _print_summary(summary: dict, topk: int):
    """Pretty-print evaluation summary."""
    bench = summary["bench"]
    print(f"\n{'='*60}")
    print(f"  {bench}")
    print(f"{'='*60}")
    print(f"  Valid / Total:  {summary['valid']} / {summary['total']}  (skipped: {summary['skipped']})")
    print(f"  Acc@1:          {summary['acc@1']:.2f}%  ({summary['hit1']} / {summary['valid']})")
    print(f"  Acc@{topk}:          {summary[f'acc@{topk}']:.2f}%  ({summary[f'hit{topk}']} / {summary['valid']})")
    if summary.get("group_accs"):
        print(f"\n  Per-category breakdown:")
        for grp, st in summary["group_accs"].items():
            print(f"    {grp:30s}  hit@1={st['hit@1']:6.2f}%  n={st['total']}")
    print(f"{'='*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate results from multiple bench runs (e.g., sharded evaluation)
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_shards(output_dir: str, bench_key: str, topk: int = 3) -> Optional[Dict]:
    """
    合并同一 bench 的多个分片结果（当使用 --start / --end 分段评测时）。

    会自动扫描 output_dir 下所有 {bench_key}_results_*.json 文件并合并。
    """
    import glob
    patterns = [
        os.path.join(output_dir, f"{bench_key}_results_*.json"),
        os.path.join(output_dir, f"{bench_key}_results.json"),
    ]
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    if not files:
        print(f"[aggregate] No result files found for {bench_key}")
        return None

    all_results = []
    for fp in sorted(files):
        with open(fp) as f:
            all_results.extend(json.load(f))

    # De-duplicate by idx
    seen = set()
    dedup = []
    for r in all_results:
        if r["idx"] not in seen:
            seen.add(r["idx"])
            dedup.append(r)
    dedup.sort(key=lambda x: x["idx"])

    hit1 = sum(r["hit1"] for r in dedup)
    hitk = sum(r["hitk"] for r in dedup)
    n    = len(dedup)
    summary = {
        "bench_key": bench_key,
        "total":     n,
        "hit1":      hit1,
        f"hit{topk}": hitk,
        "acc@1":     round(hit1 / n * 100, 4) if n > 0 else 0.0,
        f"acc@{topk}": round(hitk / n * 100, 4) if n > 0 else 0.0,
    }
    agg_path = os.path.join(output_dir, f"{bench_key}_aggregated.json")
    with open(agg_path, "w") as f:
        json.dump({"summary": summary, "results": dedup}, f, indent=2, ensure_ascii=False)
    print(f"[aggregate] {bench_key}: {n} samples, acc@1={summary['acc@1']:.2f}%")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="ZwerGe-UI Evaluation (prefill-only, layer-wise grounding head)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ckpt", type=str, required=True,
        help="ZwerGe checkpoint dir (e.g. .hdd/ckpt/zwerge/uitars7b_grounding50k_20260514_183342)",
    )
    parser.add_argument(
        "--bench", type=str, default="ss_pro",
        choices=list(BENCH_CONFIGS.keys()) + ["all"],
        help="Which benchmark to evaluate. 'all' runs all 5 benches sequentially.",
    )
    parser.add_argument(
        "--eval_dir", type=str,
        default="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/datasets/evaluation",
        help="Root directory containing benchmark eval.json files.",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/zwerge/data/results/zwerge",
        help="Directory to save evaluation results.",
    )
    parser.add_argument(
        "--attn_impl", type=str, default="flash_attention_2",
        choices=["flash_attention_2", "sdpa", "eager"],
        help="Attention implementation. Use 'sdpa' if flash_attention_2 is unavailable.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="Device string (e.g. 'cuda:0', 'cuda:1').",
    )
    parser.add_argument(
        "--topk", type=int, default=3,
        help="Number of top-k region candidates for hit@k metric.",
    )
    parser.add_argument(
        "--activation_threshold", type=float, default=0.3,
        help="Threshold for patch BFS region selection (fraction of max score).",
    )
    parser.add_argument(
        "--max_pixels", type=int, default=5_760_000,
        help="Max pixels for image processor (controls resolution cap). "
             "Default 5760000 ≈ 1.5× the standard 3840000.",
    )
    parser.add_argument(
        "--start", type=int, default=0,
        help="Start index for data slice (inclusive). Useful for multi-GPU parallel eval.",
    )
    parser.add_argument(
        "--end", type=int, default=-1,
        help="End index for data slice (exclusive). -1 = end of dataset.",
    )
    parser.add_argument(
        "--no_save_per_sample", action="store_true",
        help="Skip saving per-sample results (saves disk space).",
    )
    parser.add_argument(
        "--aggregate", action="store_true",
        help="After evaluation, aggregate all shards in output_dir.",
    )
    parser.add_argument(
        "--ckpt_subfolder", type=str, default=None,
        help="If specified, load checkpoint from ckpt/subfolder instead of ckpt directly. "
             "Useful for evaluating specific checkpoints like 'checkpoint-1500'.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve checkpoint path
    ckpt_path = args.ckpt
    if args.ckpt_subfolder:
        ckpt_path = os.path.join(ckpt_path, args.ckpt_subfolder)
    assert os.path.isdir(ckpt_path), f"Checkpoint not found: {ckpt_path}"

    # Resolve output dir (include bench name and ckpt basename for traceability)
    ckpt_basename = os.path.basename(ckpt_path)
    output_dir = os.path.join(args.output_dir, ckpt_basename)
    os.makedirs(output_dir, exist_ok=True)
    print(f"[ZwerGe eval] Output dir: {output_dir}")

    # Load model
    device = torch.device(args.device)
    model, processor = load_zwerge_model(
        ckpt_path=ckpt_path,
        attn_implementation=args.attn_impl,
        device=str(device),
        dtype=torch.bfloat16,
    )

    # Override processor max_pixels
    if hasattr(processor, "image_processor"):
        processor.image_processor.max_pixels = args.max_pixels
        print(f"[ZwerGe eval] image_processor.max_pixels = {args.max_pixels}")

    # Select benchmarks
    bench_keys = list(BENCH_CONFIGS.keys()) if args.bench == "all" else [args.bench]

    all_summaries = {}
    for bench_key in bench_keys:
        t0 = time.time()
        summary = evaluate_bench(
            bench_key=bench_key,
            eval_root=args.eval_dir,
            model=model,
            processor=processor,
            device=device,
            output_dir=output_dir,
            topk=args.topk,
            max_pixels=args.max_pixels,
            activation_threshold=args.activation_threshold,
            start=args.start,
            end=args.end,
            save_per_sample=not args.no_save_per_sample,
        )
        elapsed = time.time() - t0
        summary["elapsed_s"] = round(elapsed, 1)
        all_summaries[bench_key] = summary
        print(f"[{bench_key}] Elapsed: {elapsed/60:.1f} min")

        if args.aggregate:
            aggregate_shards(output_dir=output_dir, bench_key=bench_key, topk=args.topk)

    # ── Final aggregate summary ──────────────────────────────────────────────
    if len(all_summaries) > 1:
        print("\n" + "="*60)
        print("  FINAL SUMMARY")
        print("="*60)
        for bk, sm in all_summaries.items():
            print(f"  {sm['bench']:30s}  acc@1={sm['acc@1']:6.2f}%  acc@{args.topk}={sm.get(f'acc@{args.topk}', 0.0):6.2f}%  n={sm['valid']}")
        print("="*60)

    # Save combined summary
    combined_path = os.path.join(output_dir, "summary_all.json")
    with open(combined_path, "w") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)
    print(f"[ZwerGe eval] Combined summary saved to {combined_path}")


if __name__ == "__main__":
    main()
