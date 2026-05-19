#!/usr/bin/env python3
"""
ZwerGe-UI Evaluation Script

支持五个 Benchmark（统一读取 eval.json）：
  ss_pro     ScreenSpot-Pro    P0 主指标
  ss_v2      ScreenSpot-v2     P0
  osworld_g  OSWorld-G         P1（已排除 refusal 样本）
  mmbench    MMBench-GUI L2    P2
  ui_vision  UI-Vision         P3

四个评测指标（与 GUI-AIMA / GUI-Actor 完全对齐）：
  hit_top1    : 预测点 (px,py) 落在 gt_bbox 内
  overlap_top1: 以 (px,py) 为中心、patch 大小为边长的预测框与 gt_bbox 有交叠
  hit_topk    : 前 k 个候选区域中有任意一个 hit_top1
  overlap_topk: 前 k 个候选区域中有任意一个 overlap_top1

用法：
  # 直接调用（通常由 run_eval.sh 转发）
  python eval_zwerge.py --ckpt <ckpt_dir> --bench ss_pro
  python eval_zwerge.py --ckpt <ckpt_dir> --bench all
  # 多卡分片后合并
  python eval_zwerge.py --ckpt <ckpt_dir> --bench ss_pro --aggregate
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
    zwerge_predict,
    point_in_bbox,
    topk_hit,
    do_boxes_overlap,
)


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark 配置
# ─────────────────────────────────────────────────────────────────────────────

BENCH_CONFIGS = {
    "ss_pro": {
        "name":        "ScreenSpot-Pro",
        "eval_dir":    "ScreenSpot-Pro",
        "eval_json":   "eval.json",
        "group_field": "ui_type",      # text / icon
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
# 核心评测循环
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_bench(
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
    save_per_sample: bool = True,
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

    # ── 计数器 ────────────────────────────────────────────────────────────────
    results        = []
    hit1_total     = 0
    hitk_total     = 0
    overlap1_total = 0
    overlapk_total = 0
    skip_total     = 0
    group_stats: Dict[str, Dict] = {}

    # ── 推理循环 ──────────────────────────────────────────────────────────────
    pbar = tqdm(enumerate(data), total=total, desc=bench_name, dynamic_ncols=True)
    for idx, example in pbar:
        global_idx = start + idx

        img_path = os.path.join(img_root, example["image_path"])
        if not os.path.exists(img_path):
            warnings.warn(f"Image not found: {img_path}, skipping #{global_idx}")
            skip_total += 1
            continue

        W, H = float(example["image_size"][0]), float(example["image_size"][1])
        x1, y1, x2, y2 = example["gt_bbox"]   # 绝对像素
        gt_bbox_norm = (x1 / W, y1 / H, x2 / W, y2 / H)

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            warnings.warn(f"Failed to open image {img_path}: {e}")
            skip_total += 1
            continue

        try:
            pred = zwerge_predict(
                image=image,
                instruction=example["instruction"],
                model=model,
                processor=processor,
                device=device,
                topk=topk,
                activation_threshold=activation_threshold,
            )
        except Exception as e:
            warnings.warn(f"Inference failed for #{global_idx}: {e}")
            skip_total += 1
            continue

        # ── 四指标计算（与 GUI-AIMA screenSpot_pro.py 第 135-155 行完全对齐）────
        px, py   = pred["pred_point"]
        n_w, n_h = pred["n_width"], pred["n_height"]
        # patch 半径：一个 visual token 覆盖 1/n_w × 1/n_h 的归一化面积，取其一半
        phx = 0.5 / n_w
        phy = 0.5 / n_h

        hit1     = int(point_in_bbox(px, py, gt_bbox_norm))
        hitk     = int(topk_hit(pred["topk_points"], gt_bbox_norm))
        pred_box = (px - phx, py - phy, px + phx, py + phy)
        overlap1 = int(do_boxes_overlap(pred_box, gt_bbox_norm))
        overlapk = overlap1
        for pk_x, pk_y in pred["topk_points"][1:]:
            pk_box = (pk_x - phx, pk_y - phy, pk_x + phx, pk_y + phy)
            if do_boxes_overlap(pk_box, gt_bbox_norm):
                overlapk = 1

        hit1_total     += hit1
        hitk_total     += hitk
        overlap1_total += overlap1
        overlapk_total += overlapk

        # Group 统计
        group_key = _get_group_key(example, cfg["group_field"])
        if group_key not in group_stats:
            group_stats[group_key] = {"hit1": 0, "hitk": 0, "overlap1": 0, "overlapk": 0, "total": 0}
        group_stats[group_key]["hit1"]     += hit1
        group_stats[group_key]["hitk"]     += hitk
        group_stats[group_key]["overlap1"] += overlap1
        group_stats[group_key]["overlapk"] += overlapk
        group_stats[group_key]["total"]    += 1

        rec = {
            "idx":           global_idx,
            "image_path":    example["image_path"],
            "instruction":   example["instruction"],
            "gt_bbox_norm":  list(gt_bbox_norm),
            "pred_point":    list(pred["pred_point"]),
            "topk_points":   [list(p) for p in pred["topk_points"]],
            "hit_top1":      hit1,
            "overlap_top1":  overlap1,
            "hit_topk":      hitk,
            "overlap_topk":  overlapk,
            "anchor_strategy": pred["anchor_strategy"],
            "n_width":       pred["n_width"],
            "n_height":      pred["n_height"],
        }
        for extra in ["id", "ui_type", "group", "platform", "application",
                      "data_type", "split", "grounding_type", "task_type",
                      "GUI_types", "category", "element_type", "box_type"]:
            if extra in example:
                rec[extra] = example[extra]
        results.append(rec)

        valid_count = idx + 1 - skip_total
        acc1 = hit1_total / valid_count * 100 if valid_count > 0 else 0.0
        ov1  = overlap1_total / valid_count * 100 if valid_count > 0 else 0.0
        pbar.set_postfix({"hit@1": f"{acc1:.1f}%", "ov@1": f"{ov1:.1f}%", "skip": skip_total})

    # ── 汇总 ──────────────────────────────────────────────────────────────────
    valid_total = total - skip_total
    nn = valid_total if valid_total > 0 else 1

    group_accs = {}
    for grp, st in sorted(group_stats.items()):
        gn = st["total"]
        if gn == 0:
            continue
        group_accs[grp] = {
            "hit_top1":     round(st["hit1"]     / gn * 100, 2),
            "overlap_top1": round(st["overlap1"] / gn * 100, 2),
            "hit_topk":     round(st["hitk"]     / gn * 100, 2),
            "overlap_topk": round(st["overlapk"] / gn * 100, 2),
            "total":        gn,
        }

    summary = {
        "bench":        bench_name,
        "bench_key":    bench_key,
        "total":        total,
        "valid":        valid_total,
        "skipped":      skip_total,
        "hit_top1":     round(hit1_total     / nn * 100, 4),
        "overlap_top1": round(overlap1_total / nn * 100, 4),
        "hit_topk":     round(hitk_total     / nn * 100, 4),
        "overlap_topk": round(overlapk_total / nn * 100, 4),
        "topk":         topk,
        "slice":        [start, end],
        "group_accs":   group_accs,
    }

    # ── 保存 ──────────────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    suffix = f"_{start}-{end}" if (start > 0 or end != total + start) else ""
    if save_per_sample:
        rpath = os.path.join(output_dir, f"{bench_key}_results{suffix}.json")
        with open(rpath, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"[{bench_name}] Per-sample results → {rpath}")

    spath = os.path.join(output_dir, f"{bench_key}_summary{suffix}.json")
    with open(spath, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if verbose:
        _print_summary(summary, topk)
    return summary


def _get_group_key(example: dict, group_field: Optional[str]) -> str:
    if group_field is None:
        return "all"
    val = example.get(group_field, "unknown")
    if isinstance(val, list):
        return ",".join(str(v) for v in val)
    return str(val)


def _print_summary(summary: dict, topk: int):
    bench = summary["bench"]
    print(f"\n{'='*60}")
    print(f"  {bench}")
    print(f"{'='*60}")
    print(f"  Valid / Total   : {summary['valid']} / {summary['total']}  (skipped: {summary['skipped']})")
    print(f"  hit_top1        : {summary['hit_top1']:.2f}%")
    print(f"  overlap_top1    : {summary['overlap_top1']:.2f}%")
    print(f"  hit_top{topk}        : {summary['hit_topk']:.2f}%")
    print(f"  overlap_top{topk}    : {summary['overlap_topk']:.2f}%")
    if summary.get("group_accs"):
        print(f"\n  Per-category breakdown:")
        print(f"  {'group':30s}  hit_top1   overlap_top1   n")
        print(f"  {'-'*60}")
        for grp, st in summary["group_accs"].items():
            print(f"  {grp:30s}  {st['hit_top1']:6.2f}%    {st['overlap_top1']:6.2f}%    {st['total']}")
    print(f"{'='*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 分片合并（多卡并行后使用）
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_shards(output_dir: str, bench_key: str, topk: int = 3) -> Optional[Dict]:
    """合并同一 bench 的多个分片结果文件，输出 {bench_key}_aggregated.json 并打印完整汇总表。"""
    # ── 1. 合并逐样本结果 ─────────────────────────────────────────────────────
    patterns = [
        os.path.join(output_dir, f"{bench_key}_results_*.json"),
        os.path.join(output_dir, f"{bench_key}_results.json"),
    ]
    result_files = []
    for pat in patterns:
        result_files.extend(glob.glob(pat))
    if not result_files:
        print(f"[aggregate] No result files found for {bench_key}")
        return None

    all_results = []
    for fp in sorted(result_files):
        with open(fp) as f:
            all_results.extend(json.load(f))

    # 去重并排序
    seen, dedup = set(), []
    for r in all_results:
        if r["idx"] not in seen:
            seen.add(r["idx"])
            dedup.append(r)
    dedup.sort(key=lambda x: x["idx"])

    n        = len(dedup)
    nn       = n if n > 0 else 1
    hit1     = sum(r["hit_top1"]     for r in dedup)
    hitk     = sum(r["hit_topk"]     for r in dedup)
    overlap1 = sum(r["overlap_top1"] for r in dedup)
    overlapk = sum(r["overlap_topk"] for r in dedup)

    # ── 2. 从分片 summary 文件重建 group_accs（加权合并） ────────────────────
    shard_summaries = []
    for fp in sorted(glob.glob(os.path.join(output_dir, f"{bench_key}_summary_*.json"))):
        with open(fp) as f:
            shard_summaries.append(json.load(f))

    bench_name = BENCH_CONFIGS[bench_key]["name"]
    group_accs: Dict[str, Dict] = {}
    if shard_summaries:
        bench_name = shard_summaries[0].get("bench", bench_name)
        merged_groups: Dict[str, Dict] = {}
        for sm in shard_summaries:
            for grp, st in sm.get("group_accs", {}).items():
                gn = st["total"]
                if grp not in merged_groups:
                    merged_groups[grp] = {"hit1": 0, "hitk": 0, "overlap1": 0, "overlapk": 0, "total": 0}
                merged_groups[grp]["hit1"]     += round(st["hit_top1"]     / 100 * gn)
                merged_groups[grp]["hitk"]     += round(st["hit_topk"]     / 100 * gn)
                merged_groups[grp]["overlap1"] += round(st["overlap_top1"] / 100 * gn)
                merged_groups[grp]["overlapk"] += round(st["overlap_topk"] / 100 * gn)
                merged_groups[grp]["total"]    += gn
        for grp, mg in sorted(merged_groups.items()):
            gn = mg["total"] if mg["total"] > 0 else 1
            group_accs[grp] = {
                "hit_top1":     round(mg["hit1"]     / gn * 100, 2),
                "overlap_top1": round(mg["overlap1"] / gn * 100, 2),
                "hit_topk":     round(mg["hitk"]     / gn * 100, 2),
                "overlap_topk": round(mg["overlapk"] / gn * 100, 2),
                "total":        mg["total"],
            }

    skipped = sum(sm.get("skipped", 0) for sm in shard_summaries)

    summary = {
        "bench":        bench_name,
        "bench_key":    bench_key,
        "total":        n + skipped,
        "valid":        n,
        "skipped":      skipped,
        "hit_top1":     round(hit1     / nn * 100, 4),
        "overlap_top1": round(overlap1 / nn * 100, 4),
        "hit_topk":     round(hitk     / nn * 100, 4),
        "overlap_topk": round(overlapk / nn * 100, 4),
        "topk":         topk,
        "group_accs":   group_accs,
    }

    agg_path = os.path.join(output_dir, f"{bench_key}_aggregated.json")
    with open(agg_path, "w") as f:
        json.dump({"summary": summary, "results": dedup}, f, indent=2, ensure_ascii=False)

    _print_summary(summary, topk)
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# 多卡并行：每张卡作为独立子进程，处理数据集的一个分片
# ─────────────────────────────────────────────────────────────────────────────

def _worker(gpu_id: int, ckpt_path: str, bench_key: str, eval_root: str,
            output_dir: str, topk: int, activation_threshold: float,
            attn_impl: str, start: int, end: int, max_pixels: int = 20_358_912):
    """单张 GPU 上跑一个数据分片，结果写到 output_dir/{bench_key}_results_{start}-{end}.json。"""
    import torch
    device = torch.device(f"cuda:{gpu_id}")
    model, processor = load_zwerge_model(
        ckpt_path=ckpt_path,
        attn_implementation=attn_impl,
        device=str(device),
        dtype=torch.bfloat16,
    )
    processor.image_processor.max_pixels = max_pixels
    evaluate_bench(
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
        save_per_sample=True,
        verbose=False,   # 分片不打印，合并后由主进程统一打印
    )


def run_bench_parallel(bench_key: str, ckpt_path: str, eval_root: str,
                       output_dir: str, topk: int, activation_threshold: float,
                       attn_impl: str, max_pixels: int = 20_358_912) -> Optional[Dict]:
    """
    自动检测 GPU 数量，多卡并行跑一个 bench，最后合并分片返回汇总结果。
    单卡时退化为普通串行。
    """
    import multiprocessing as mp
    import torch

    n_gpu = torch.cuda.device_count()
    if n_gpu == 0:
        raise RuntimeError("No CUDA GPU found.")

    # 读取数据集总条数
    cfg = BENCH_CONFIGS[bench_key]
    eval_json_path = os.path.join(eval_root, cfg["eval_dir"], cfg["eval_json"])
    with open(eval_json_path) as f:
        N = len(json.load(f))

    print(f"[ZwerGe eval] {bench_key}: {N} samples, {n_gpu} GPU(s)")

    if n_gpu == 1:
        # 单卡：直接在当前进程跑，不 fork（省去进程启动开销）
        device = torch.device("cuda:0")
        model, processor = load_zwerge_model(
            ckpt_path=ckpt_path,
            attn_implementation=attn_impl,
            device="cuda:0",
            dtype=torch.bfloat16,
        )
        processor.image_processor.max_pixels = max_pixels
        return evaluate_bench(
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
            save_per_sample=True,
        )

    # 多卡：把 N 条数据等分给每张卡
    chunk = (N + n_gpu - 1) // n_gpu
    slices = []
    for i in range(n_gpu):
        s = i * chunk
        e = min((i + 1) * chunk, N)
        if s >= N:
            break
        slices.append((i, s, e))

    # 每张卡 fork 一个子进程（spawn 模式避免 CUDA init 冲突）
    ctx = mp.get_context("spawn")
    procs = []
    for gpu_id, s, e in slices:
        print(f"[ZwerGe eval]   GPU{gpu_id}: slice [{s}, {e})")
        p = ctx.Process(
            target=_worker,
            args=(gpu_id, ckpt_path, bench_key, eval_root,
                  output_dir, topk, activation_threshold, attn_impl, s, e, max_pixels),
        )
        p.start()
        procs.append(p)

    # 等待所有子进程结束
    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"Worker process exited with code {p.exitcode}")

    # 合并各 GPU 的分片结果文件
    return aggregate_shards(output_dir=output_dir, bench_key=bench_key, topk=topk)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="ZwerGe-UI Evaluation")
    parser.add_argument("--ckpt",       required=True,  help="Checkpoint 目录")
    parser.add_argument("--bench",      default="ss_pro",
                        choices=list(BENCH_CONFIGS.keys()) + ["all"])
    parser.add_argument("--eval_dir",   default="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/datasets/evaluation")
    parser.add_argument("--output_dir", default="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/zwerge/data/results/zwerge")
    parser.add_argument("--attn_impl",  default="flash_attention_2",
                        choices=["flash_attention_2", "sdpa", "eager"])
    # 图像分辨率上限：覆盖 preprocessor_config.json 里的默认值（5720064）
    # 与训练脚本 MAX_PIXELS=12845056 保持一致（= 16384 × 784，可覆盖 6016×3384 不缩放）
    parser.add_argument("--max_pixels", type=int, default=12_845_056)
    # 算法超参，默认值与 GUI-AIMA/Actor 一致
    parser.add_argument("--activation_threshold", type=float, default=0.3)
    parser.add_argument("--topk",                 type=int,   default=3)
    return parser.parse_args()


def main():
    args = parse_args()

    assert os.path.isdir(args.ckpt), f"Checkpoint not found: {args.ckpt}"
    output_dir = os.path.join(args.output_dir, os.path.basename(args.ckpt))
    os.makedirs(output_dir, exist_ok=True)
    print(f"[ZwerGe eval] Output dir: {output_dir}")

    bench_keys = list(BENCH_CONFIGS.keys()) if args.bench == "all" else [args.bench]
    all_summaries = {}

    for bench_key in bench_keys:
        t0 = time.time()
        summary = run_bench_parallel(
            bench_key=bench_key,
            ckpt_path=args.ckpt,
            eval_root=args.eval_dir,
            output_dir=output_dir,
            topk=args.topk,
            activation_threshold=args.activation_threshold,
            attn_impl=args.attn_impl,
            max_pixels=args.max_pixels,
        )
        elapsed = time.time() - t0
        if summary:
            summary["elapsed_s"] = round(elapsed, 1)
            all_summaries[bench_key] = summary
        print(f"[{bench_key}] Elapsed: {elapsed/60:.1f} min")

    # 汇总打印（bench=all 时）
    if len(all_summaries) > 1:
        print("\n" + "="*75)
        print("  FINAL SUMMARY")
        print("="*75)
        print(f"  {'benchmark':30s}  hit_top1  overlap_top1  hit_top{args.topk}  overlap_top{args.topk}  n")
        print(f"  {'-'*72}")
        for bk, sm in all_summaries.items():
            bench_name = sm.get("bench", bk)
            n = sm.get("valid", sm.get("total", "?"))
            print(f"  {bench_name:30s}  {sm['hit_top1']:6.2f}%    {sm['overlap_top1']:6.2f}%"
                  f"      {sm['hit_topk']:6.2f}%    {sm['overlap_topk']:6.2f}%    {n}")
        print("="*75)

    with open(os.path.join(output_dir, "summary_all.json"), "w") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)
    print(f"[ZwerGe eval] Done. Results → {output_dir}")


if __name__ == "__main__":
    main()
