#!/usr/bin/env python3
"""
ZwerGe-UI Evaluation + Visualization (Unified Entry Point)
===========================================================
Mirrors train_retrofit.py on the eval side: one script, --model_type dispatch.

Usage:
  # Eval + visualization (default):
  python eval_retrofit.py --model_type guiowl --ckpt <ckpt> --bench ss_pro

  # Metrics only (skip PNG generation, equivalent to old eval_layerwise.py):
  python eval_retrofit.py --model_type guiowl --ckpt <ckpt> --bench ss_pro --skip_vis

  # Multi-GPU auto-detected from CUDA_VISIBLE_DEVICES.

Output directory structure:
  {output_dir}/{decode_strategy}/{train_run}/{ckpt_name}/
  ├── {bench}_layerwise_summary.json
  ├── layerwise_all_summary.json        (bench=all)
  └── details/{bench}/
      ├── success/*.png                 (skipped when --skip_vis)
      ├── failure/*.png                 (skipped when --skip_vis)
      └── results.json
"""

import argparse
import glob
import json
import os
import sys
import time
import warnings
from typing import Dict, List, Optional

import torch
from PIL import Image
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR    = os.path.join(_SCRIPT_DIR, "..", "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from inference_base import (
    BENCH_CONFIGS, MAIN_BENCH_KEYS,
    scores_to_point_and_topk, point_in_bbox, do_boxes_overlap,
    _get_group_key, _print_layerwise_summary,
)


# ─────────────────────────────────────────────────────────────────────────────
# Inference class factory
# ─────────────────────────────────────────────────────────────────────────────

def get_inference_class(model_type: str, inference_type: str = "retrofit"):
    """
    Returns the inference class for the given model_type + inference_type.
    inference_type: "retrofit" (default) or "native"
    """
    from inference_uitars   import UITARSRetrofitInference
    from inference_guiowl   import GUIOwlRetrofitInference
    from inference_uivenus  import UIVenusRetrofitInference
    from inference_guiowl7b import GUIOwl7BRetrofitInference
    from inference_qwen35   import Qwen35RetrofitInference
    from inference_uitars1  import UITARS1RetrofitInference

    CLASSES = {
        ("uitars",    "retrofit"): UITARSRetrofitInference,
        ("guiowl",    "retrofit"): GUIOwlRetrofitInference,
        ("uivenus",   "retrofit"): UIVenusRetrofitInference,
        ("guiowl7b",  "retrofit"): GUIOwl7BRetrofitInference,
        ("qwen35",    "retrofit"): Qwen35RetrofitInference,
        ("uitars1",   "retrofit"): UITARS1RetrofitInference,
    }
    key = (model_type, inference_type)
    if key not in CLASSES:
        raise ValueError(f"Unknown (model_type={model_type!r}, inference_type={inference_type!r}). "
                         f"Available: {list(CLASSES.keys())}")
    return CLASSES[key]


# ─────────────────────────────────────────────────────────────────────────────
# Multi-GPU worker
# ─────────────────────────────────────────────────────────────────────────────

def _worker(
    gpu_id: int,
    ckpt_path: str,
    bench_key: str,
    eval_root: str,
    output_dir: str,
    start: int,
    end: int,
    attn_impl: str,
    max_pixels: int,
    decode_strategy: str,
    peak_shift_alpha: float,
    temperature: float,
    activation_threshold: float,
    topk: int,
    skip_vis: bool,
    cell_w: int,
    cell_h: int,
    alpha: float,
    group_stats: bool,
    model_type: str = "uitars",
    zoom_padding_cells: int = 3,
    zoom_max_new_tokens: int = 256,
):
    import torch as _torch
    _device = _torch.device(f"cuda:{gpu_id}")

    InfClass = get_inference_class(model_type)
    grounder = InfClass.from_checkpoint(
        ckpt_path=ckpt_path, attn_impl=attn_impl,
        device=str(_device), dtype=_torch.bfloat16, max_pixels=max_pixels,
    )

    cfg        = BENCH_CONFIGS[bench_key]
    bench_name = cfg["name"]
    group_field = cfg.get("group_field") if group_stats else None
    eval_json_path = os.path.join(eval_root, cfg["eval_dir"], cfg["eval_json"])
    img_root   = os.path.join(eval_root, cfg["eval_dir"])

    with open(eval_json_path) as f:
        shard = json.load(f)[start:end]

    if not skip_vis:
        success_dir = os.path.join(output_dir, "details", bench_key, "success")
        failure_dir = os.path.join(output_dir, "details", bench_key, "failure")
        os.makedirs(success_dir, exist_ok=True)
        os.makedirs(failure_dir, exist_ok=True)
    else:
        success_dir = failure_dir = ""

    _head               = grounder.model.layerwise_grounding_head
    probe_layers        = list(_head.probe_layers)
    active_probe_layers = list(_head.active_probe_layers)
    n_probes     = len(probe_layers)
    layer_stats  = [{"hit1": 0, "hitk": 0, "overlap1": 0, "overlapk": 0, "total": 0}
                    for _ in range(n_probes)]
    fusion_stats = {"hit1": 0, "hitk": 0, "overlap1": 0, "overlapk": 0, "total": 0}
    fusion_group_stats: Dict[str, Dict] = {}
    results    = []
    skip_total = 0

    pbar = tqdm(enumerate(shard), total=len(shard),
                desc=f"GPU{gpu_id} {bench_name} [{start}:{end}]", dynamic_ncols=True)
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
            orig_img = Image.open(img_path).convert("RGB")
        except Exception as e:
            warnings.warn(f"Failed to open image {img_path}: {e}")
            skip_total += 1
            continue

        try:
            if decode_strategy == "zoom_backbone":
                pred = grounder.predict_zoom_backbone(
                    image=orig_img, instruction=example["instruction"],
                    device=_device,
                    activation_threshold=activation_threshold,
                    topk=topk,
                    padding_cells=zoom_padding_cells,
                    max_new_tokens=zoom_max_new_tokens,
                    peak_shift_alpha=peak_shift_alpha,
                    temperature=temperature,
                    full_image=False,
                )
            elif decode_strategy == "native_backbone":
                # Full image → backbone generate (no ZwerGe-guided crop).
                # Stage 1 ZwerGe metrics still computed; only fusion/final from backbone.
                pred = grounder.predict_zoom_backbone(
                    image=orig_img, instruction=example["instruction"],
                    device=_device,
                    activation_threshold=activation_threshold,
                    topk=topk,
                    max_new_tokens=zoom_max_new_tokens,
                    full_image=True,   # ← KEY: no crop, use original image
                )
            else:
                pred = grounder.predict_layerwise(
                    image=orig_img, instruction=example["instruction"],
                    device=_device,
                    activation_threshold=activation_threshold,
                    topk=topk,
                    decode_strategy=decode_strategy,
                    peak_shift_alpha=peak_shift_alpha,
                    temperature=temperature,
                )
        except Exception as e:
            import traceback
            warnings.warn(f"Inference failed for #{global_idx}: {e}")
            traceback.print_exc()   # full stack trace to stderr for diagnostics
            skip_total += 1
            continue

        n_w, n_h = pred["n_width"], pred["n_height"]
        phx = 0.5 / n_w
        phy = 0.5 / n_h

        # Per-layer metrics
        layer_metrics = []
        for li in range(n_probes):
            lpt    = pred["per_layer_points"][li]
            px, py = float(lpt[0]), float(lpt[1])
            tk_pts = pred["per_layer_topk"][li]
            hit1   = int(point_in_bbox(px, py, gt_bbox_norm))
            hitk   = int(any(point_in_bbox(float(p[0]), float(p[1]), gt_bbox_norm) for p in tk_pts))
            pred_box = (px - phx, py - phy, px + phx, py + phy)
            overlap1 = int(do_boxes_overlap(pred_box, gt_bbox_norm))
            overlapk = overlap1
            for pk_pt in tk_pts[1:]:
                pk_box = (float(pk_pt[0]) - phx, float(pk_pt[1]) - phy,
                          float(pk_pt[0]) + phx, float(pk_pt[1]) + phy)
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

        # Fusion / final metrics
        # zoom_backbone: use backbone-refined point directly (no scores_to_point_and_topk)
        # other strategies: derive final point from p_final distribution
        if decode_strategy in ("zoom_backbone", "native_backbone") and "zoom_point" in pred:
            fpx, fpy  = float(pred["zoom_point"][0]), float(pred["zoom_point"][1])
            f_centers = [(fpx, fpy)]   # single refined point
        else:
            f_best, f_centers = scores_to_point_and_topk(
                p=pred["p_final"], n_width=n_w, n_height=n_h,
                activation_threshold=activation_threshold, topk=topk,
                decode_strategy=decode_strategy,
                peak_shift_alpha=peak_shift_alpha, temperature=temperature,
            )
            fpx, fpy = float(f_best[0]), float(f_best[1])
        fhit1     = int(point_in_bbox(fpx, fpy, gt_bbox_norm))
        fpred_box = (fpx - phx, fpy - phy, fpx + phx, fpy + phy)
        fov1      = int(do_boxes_overlap(fpred_box, gt_bbox_norm))
        fhitk     = int(any(point_in_bbox(float(p[0]), float(p[1]), gt_bbox_norm) for p in f_centers))
        fovk      = fov1
        for fk_pt in f_centers[1:]:
            fk_box = (float(fk_pt[0]) - phx, float(fk_pt[1]) - phy,
                      float(fk_pt[0]) + phx, float(fk_pt[1]) + phy)
            if do_boxes_overlap(fk_box, gt_bbox_norm):
                fovk = 1
                break
        fusion_stats["hit1"]     += fhit1
        fusion_stats["overlap1"] += fov1
        fusion_stats["hitk"]     += fhitk
        fusion_stats["overlapk"] += fovk
        fusion_stats["total"]    += 1

        if group_field is not None:
            grp = _get_group_key(example, group_field)
            if grp not in fusion_group_stats:
                fusion_group_stats[grp] = {"hit1": 0, "hitk": 0, "overlap1": 0, "overlapk": 0, "total": 0}
            fusion_group_stats[grp]["hit1"]     += fhit1
            fusion_group_stats[grp]["overlap1"] += fov1
            fusion_group_stats[grp]["hitk"]     += fhitk
            fusion_group_stats[grp]["overlapk"] += fovk
            fusion_group_stats[grp]["total"]    += 1

        # Visualization (skipped when --skip_vis)
        if not skip_vis and success_dir:
            meta = {"bench": bench_name}
            for k in ["ui_type", "data_type", "GUI_types", "grounding_type", "task_type",
                      "platform", "application", "element_type"]:
                if k in example:
                    meta[k] = example[k]
            try:
                from vis_utils import visualize_sample
                vis_img = visualize_sample(
                    orig_img=orig_img, pred=pred, gt_bbox_norm=gt_bbox_norm,
                    instruction=example["instruction"], meta=meta,
                    activation_threshold=activation_threshold,
                    decode_strategy=decode_strategy,
                    peak_shift_alpha=peak_shift_alpha, temperature=temperature,
                    cell_w=cell_w, cell_h=cell_h, alpha=alpha,
                )
                save_dir = success_dir if fhit1 else failure_dir
                vis_img.save(os.path.join(save_dir, f"idx{global_idx:05d}.png"))
            except Exception as e:
                warnings.warn(f"Visualization failed for #{global_idx}: {e}")

        # Per-sample record
        rec = {
            "idx": global_idx, "image_path": example["image_path"],
            "instruction": example["instruction"],
            "gt_bbox_norm": list(gt_bbox_norm),
            "anchor_strategy": pred["anchor_strategy"],
            "n_width": n_w, "n_height": n_h,
            "omega": pred["omega"].tolist(),
            "probe_layers": probe_layers,
            "active_probe_layers": pred.get("active_probe_layers", probe_layers),
            "layer_metrics": layer_metrics,
            "fusion_hit1": fhit1, "fusion_overlap1": fov1,
            "fusion_hitk": fhitk, "fusion_overlapk": fovk,
        }
        for extra in ["id", "ui_type", "group", "platform", "application",
                      "data_type", "split", "grounding_type", "task_type",
                      "GUI_types", "category", "element_type"]:
            if extra in example:
                rec[extra] = example[extra]
        results.append(rec)

        valid_so_far = idx + 1 - skip_total
        if valid_so_far > 0:
            pbar.set_postfix({
                "fusion_hit": f"{fusion_stats['hit1'] / valid_so_far * 100:.1f}%",
                "skip": skip_total,
            })

    # Save shard files
    if not skip_vis:
        details_dir = os.path.join(output_dir, "details", bench_key)
        os.makedirs(details_dir, exist_ok=True)
        with open(os.path.join(details_dir, f"results_{start}-{end}.json"), "w") as f:
            json.dump(results, f, ensure_ascii=False)

    valid_total = len(shard) - skip_total
    layer_accs = []
    for li, ls in enumerate(layer_stats):
        n = ls["total"] if ls["total"] > 0 else 1
        layer_accs.append({
            "layer_idx": probe_layers[li], "probe_rank": li,
            "hit_top1":     round(ls["hit1"]     / n * 100, 4),
            "overlap_top1": round(ls["overlap1"] / n * 100, 4),
            "hit_topk":     round(ls["hitk"]     / n * 100, 4),
            "overlap_topk": round(ls["overlapk"] / n * 100, 4),
            "n": ls["total"],
        })
    fn = fusion_stats["total"] if fusion_stats["total"] > 0 else 1
    fusion_acc = {
        "hit_top1":     round(fusion_stats["hit1"]     / fn * 100, 4),
        "overlap_top1": round(fusion_stats["overlap1"] / fn * 100, 4),
        "hit_topk":     round(fusion_stats["hitk"]     / fn * 100, 4),
        "overlap_topk": round(fusion_stats["overlapk"] / fn * 100, 4),
        "topk":         topk,
    }
    fusion_group_accs: Dict[str, Dict] = {}
    if group_field is not None:
        for grp, st in sorted(fusion_group_stats.items()):
            gn = st["total"] if st["total"] > 0 else 1
            fusion_group_accs[grp] = {
                "hit_top1":     round(st["hit1"]     / gn * 100, 2),
                "overlap_top1": round(st["overlap1"] / gn * 100, 2),
                "hit_topk":     round(st["hitk"]     / gn * 100, 2),
                "overlap_topk": round(st["overlapk"] / gn * 100, 2),
                "total":        st["total"],
            }

    shard_summary = {
        "bench": bench_name, "bench_key": bench_key,
        "total": len(shard), "valid": valid_total, "skipped": skip_total,
        "slice": [start, end], "probe_layers": probe_layers,
        "active_probe_layers": active_probe_layers,
        "layer_accs": layer_accs, "fusion_acc": fusion_acc,
        "fusion_group_accs": fusion_group_accs,
    }
    with open(os.path.join(output_dir, f"{bench_key}_layerwise_summary_{start}-{end}.json"), "w") as f:
        json.dump(shard_summary, f, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
# Shard aggregation
# ─────────────────────────────────────────────────────────────────────────────

def _aggregate_shards(output_dir: str, bench_key: str, topk: int, skip_vis: bool) -> Dict:
    shard_files = sorted(glob.glob(
        os.path.join(output_dir, f"{bench_key}_layerwise_summary_*.json")
    ))
    if not shard_files:
        raise FileNotFoundError(f"No shard summary files for {bench_key} in {output_dir}")

    summaries           = [json.load(open(fp)) for fp in shard_files]
    probe_layers        = summaries[0]["probe_layers"]
    active_probe_layers = summaries[0].get("active_probe_layers", probe_layers)
    n_probes     = len(probe_layers)
    bench_name   = summaries[0]["bench"]

    merged_layer  = [{"hit1": 0, "hitk": 0, "overlap1": 0, "overlapk": 0, "total": 0}
                     for _ in range(n_probes)]
    merged_fusion = {"hit1": 0, "hitk": 0, "overlap1": 0, "overlapk": 0, "total": 0}
    merged_groups: Dict[str, Dict] = {}

    for sm in summaries:
        for li, la in enumerate(sm["layer_accs"]):
            n = la["n"]
            merged_layer[li]["hit1"]     += round(la["hit_top1"]     / 100 * n)
            merged_layer[li]["hitk"]     += round(la["hit_topk"]     / 100 * n)
            merged_layer[li]["overlap1"] += round(la["overlap_top1"] / 100 * n)
            merged_layer[li]["overlapk"] += round(la["overlap_topk"] / 100 * n)
            merged_layer[li]["total"]    += n
        fn = sm["valid"]
        fa = sm["fusion_acc"]
        merged_fusion["hit1"]     += round(fa["hit_top1"]     / 100 * fn)
        merged_fusion["overlap1"] += round(fa["overlap_top1"] / 100 * fn)
        merged_fusion["hitk"]     += round(fa["hit_topk"]     / 100 * fn)
        merged_fusion["overlapk"] += round(fa["overlap_topk"] / 100 * fn)
        merged_fusion["total"]    += fn
        for grp, gst in sm.get("fusion_group_accs", {}).items():
            gn = gst["total"]
            if grp not in merged_groups:
                merged_groups[grp] = {"hit1": 0, "hitk": 0, "overlap1": 0, "overlapk": 0, "total": 0}
            merged_groups[grp]["hit1"]     += round(gst["hit_top1"]     / 100 * gn)
            merged_groups[grp]["overlap1"] += round(gst["overlap_top1"] / 100 * gn)
            merged_groups[grp]["hitk"]     += round(gst["hit_topk"]     / 100 * gn)
            merged_groups[grp]["overlapk"] += round(gst["overlap_topk"] / 100 * gn)
            merged_groups[grp]["total"]    += gn

    total   = sum(sm["total"]   for sm in summaries)
    valid   = sum(sm["valid"]   for sm in summaries)
    skipped = sum(sm["skipped"] for sm in summaries)

    layer_accs = []
    for li, ls in enumerate(merged_layer):
        n = ls["total"] if ls["total"] > 0 else 1
        layer_accs.append({
            "layer_idx":    probe_layers[li], "probe_rank": li,
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
        "hit_topk":     round(merged_fusion["hitk"]     / fn * 100, 4),
        "overlap_topk": round(merged_fusion["overlapk"] / fn * 100, 4),
        "topk":         topk,
    }
    fusion_group_accs: Dict[str, Dict] = {}
    for grp, mg in sorted(merged_groups.items()):
        gn = mg["total"] if mg["total"] > 0 else 1
        fusion_group_accs[grp] = {
            "hit_top1":     round(mg["hit1"]     / gn * 100, 2),
            "overlap_top1": round(mg["overlap1"] / gn * 100, 2),
            "hit_topk":     round(mg["hitk"]     / gn * 100, 2),
            "overlap_topk": round(mg["overlapk"] / gn * 100, 2),
            "total":        mg["total"],
        }

    summary = {
        "bench": bench_name, "bench_key": bench_key,
        "total": total, "valid": valid, "skipped": skipped, "topk": topk,
        "probe_layers": probe_layers,
        "active_probe_layers": active_probe_layers,
        "layer_accs": layer_accs, "layer_accs_sorted": layer_accs_sorted,
        "fusion_acc": fusion_acc, "fusion_group_accs": fusion_group_accs,
    }
    with open(os.path.join(output_dir, f"{bench_key}_layerwise_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Merge per-sample results (only if vis mode — details/ exists)
    if not skip_vis:
        details_dir   = os.path.join(output_dir, "details", bench_key)
        result_shards = sorted(glob.glob(os.path.join(details_dir, "results_*.json")))
        if result_shards:
            all_results = []
            for fp in result_shards:
                with open(fp) as f:
                    all_results.extend(json.load(f))
            all_results.sort(key=lambda x: x["idx"])
            with open(os.path.join(details_dir, "results.json"), "w") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            for fp in result_shards:
                try:
                    os.remove(fp)
                except OSError:
                    pass

    for fp in shard_files:
        try:
            os.remove(fp)
        except OSError:
            pass

    _print_layerwise_summary(summary, topk)
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Multi-GPU scheduling
# ─────────────────────────────────────────────────────────────────────────────

def run_bench_parallel(
    bench_key: str,
    ckpt_path: str,
    eval_root: str,
    output_dir: str,
    attn_impl: str,
    max_pixels: int,
    decode_strategy: str,
    peak_shift_alpha: float,
    temperature: float,
    activation_threshold: float,
    topk: int,
    skip_vis: bool,
    cell_w: int,
    cell_h: int,
    alpha: float,
    group_stats: bool = True,
    model_type: str = "uitars",
    zoom_padding_cells: int = 3,
    zoom_max_new_tokens: int = 256,
) -> Dict:
    import multiprocessing as mp

    n_gpu = torch.cuda.device_count()
    if n_gpu == 0:
        raise RuntimeError("No CUDA GPU found.")

    cfg = BENCH_CONFIGS[bench_key]
    with open(os.path.join(eval_root, cfg["eval_dir"], cfg["eval_json"])) as f:
        N = len(json.load(f))
    print(f"[ZwerGe] {bench_key}: {N} samples, {n_gpu} GPU(s), model_type={model_type}, skip_vis={skip_vis}")

    worker_kwargs = dict(
        ckpt_path=ckpt_path, bench_key=bench_key, eval_root=eval_root,
        output_dir=output_dir, attn_impl=attn_impl, max_pixels=max_pixels,
        decode_strategy=decode_strategy, peak_shift_alpha=peak_shift_alpha,
        temperature=temperature, activation_threshold=activation_threshold,
        topk=topk, skip_vis=skip_vis, cell_w=cell_w, cell_h=cell_h,
        alpha=alpha, group_stats=group_stats, model_type=model_type,
        zoom_padding_cells=zoom_padding_cells,
        zoom_max_new_tokens=zoom_max_new_tokens,
    )

    chunk  = (N + n_gpu - 1) // n_gpu
    slices = [(i, i * chunk, min((i + 1) * chunk, N))
              for i in range(n_gpu) if i * chunk < N]

    if n_gpu == 1:
        _worker(gpu_id=0, start=0, end=N, **worker_kwargs)
    else:
        ctx   = mp.get_context("spawn")
        procs = []
        for gpu_id, s, e in slices:
            print(f"[ZwerGe]   GPU{gpu_id}: slice [{s}, {e})")
            p = ctx.Process(
                target=_worker,
                kwargs={"gpu_id": gpu_id, "start": s, "end": e, **worker_kwargs},
            )
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"Worker process exited with code {p.exitcode}")

    return _aggregate_shards(output_dir, bench_key, topk, skip_vis)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="ZwerGe-UI Evaluation + Visualization (unified entry, mirrors train_retrofit.py)"
    )
    parser.add_argument("--ckpt",        required=True, help="Checkpoint directory")
    parser.add_argument(
        "--model_type", default="uitars",
        choices=["uitars", "guiowl", "uivenus", "guiowl7b", "qwen35", "uitars1"],
        help="Model type — affects prompt format and model loading class. "
             "guiowl7b: Qwen2.5-VL backbone + GUI-Owl-1.5 prompt (control variable). "
             "qwen35: Qwen3.5-VL, XML-style tool-call, relative 1000. "
             "uitars1: Qwen2-VL (UI-TARS-7B-SFT), UI-TARS prompt, relative 1000.",
    )
    parser.add_argument(
        "--bench", default="ss_pro",
        choices=list(BENCH_CONFIGS.keys()) + ["all"],
    )
    parser.add_argument("--eval_dir",
        default="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/datasets/evaluation")
    parser.add_argument("--output_dir",
        default="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/zwerge/data/results/zwerge_layerwise")
    parser.add_argument(
        "--output_dir_final", default=None,
        help="If set, use this directory directly as the output root (skip auto {basedir}/{basename} suffix). "
             "Use when you want full control over the output path.",
    )
    parser.add_argument("--attn_impl", default="flash_attention_2",
                        choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument(
        "--max_pixels", type=int, default=None,
        help=(
            "Max image pixels for processor. Defaults to model-type-specific value if not set: "
            "uitars=12845056 (16384 tokens @ patch14×merge2), "
            "guiowl/uivenus=16777216 (16384 tokens @ patch16×merge2). "
            "Always pass the same value used at training time."
        ),
    )
    parser.add_argument("--activation_threshold", type=float, default=0.3)
    parser.add_argument("--topk",                 type=int,   default=3)
    parser.add_argument(
        "--decode_strategy", default="centroid",
        choices=["centroid", "argmax", "peak_shift", "temperature",
                 "zoom_backbone", "native_backbone"],
        help=(
            "centroid/argmax/peak_shift/temperature: extract coordinate from p_final distribution. "
            "zoom_backbone: Stage1=ZwerGe ROI selection, Stage2=backbone generate on zoomed crop. "
            "native_backbone: Stage1=ZwerGe (for per-layer metrics), Stage2=backbone on FULL image "
            "(reproduces vanilla model accuracy; use to verify eval code and establish baseline)."
        ),
    )
    parser.add_argument("--peak_shift_alpha", type=float, default=0.5)
    parser.add_argument("--temperature",      type=float, default=0.5)
    # zoom_backbone strategy options
    parser.add_argument("--zoom_padding_cells",   type=int, default=3,
                        help="Extra patch cells of context around the selected region (zoom_backbone only)")
    parser.add_argument("--zoom_max_new_tokens",  type=int, default=256,
                        help="Max tokens for backbone generate in zoom_backbone strategy")
    # Visualization options
    parser.add_argument(
        "--skip_vis", action="store_true",
        help="Skip PNG generation; output metrics only (equivalent to old eval_layerwise.py)",
    )
    parser.add_argument("--cell_w",  type=int,   default=300)
    parser.add_argument("--cell_h",  type=int,   default=220)
    parser.add_argument("--alpha",   type=float, default=0.55)
    parser.add_argument("--no_group_stats", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    assert os.path.isdir(args.ckpt), f"Checkpoint not found: {args.ckpt}"

    # ── Resolve max_pixels default based on model_type ────────────────────────
    # uitars  (Qwen2.5-VL, patch_size=14): 16384 × 14² × 4 = 12,845,056
    # guiowl/uivenus (Qwen3-VL, patch_size=16): 16384 × 16² × 4 = 16,777,216
    # Using uitars' max_pixels for Qwen3-VL would give only ~12544 tokens instead of 16384.
    if args.max_pixels is None:
        if args.model_type in ("guiowl", "uivenus", "qwen35"):
            # Qwen3-VL / Qwen3.5: patch_size=16, max 16384 tokens → 16384 × 16² × 4 = 16,777,216
            args.max_pixels = 16_777_216
        else:
            # Qwen2-VL (uitars1) / Qwen2.5-VL (uitars / guiowl7b): patch_size=14 → 12,845,056
            args.max_pixels = 12_845_056
    print(f"[ZwerGe] max_pixels={args.max_pixels} (model_type={args.model_type})")

    if args.output_dir_final:
        output_dir = args.output_dir_final
    else:
        basename   = os.path.basename(args.ckpt)
        basedir    = os.path.basename(os.path.dirname(args.ckpt))
        output_dir = os.path.join(args.output_dir, basedir, basename)
    os.makedirs(output_dir, exist_ok=True)
    print(f"[ZwerGe] Output dir: {output_dir}")
    print(f"[ZwerGe] model_type={args.model_type}, skip_vis={args.skip_vis}")

    bench_keys    = MAIN_BENCH_KEYS if args.bench == "all" else [args.bench]
    all_summaries = {}

    for bench_key in bench_keys:
        t0 = time.time()
        summary = run_bench_parallel(
            bench_key=bench_key, ckpt_path=args.ckpt,
            eval_root=args.eval_dir, output_dir=output_dir,
            attn_impl=args.attn_impl, max_pixels=args.max_pixels,
            decode_strategy=args.decode_strategy,
            peak_shift_alpha=args.peak_shift_alpha,
            temperature=args.temperature,
            activation_threshold=args.activation_threshold,
            topk=args.topk, skip_vis=args.skip_vis,
            cell_w=args.cell_w, cell_h=args.cell_h, alpha=args.alpha,
            group_stats=not args.no_group_stats, model_type=args.model_type,
            zoom_padding_cells=args.zoom_padding_cells,
            zoom_max_new_tokens=args.zoom_max_new_tokens,
        )
        elapsed = time.time() - t0
        summary["elapsed_s"] = round(elapsed, 1)
        all_summaries[bench_key] = summary
        print(f"[{bench_key}] Elapsed: {elapsed / 60:.1f} min")

    if len(all_summaries) > 1:
        all_layers = sorted({
            la["layer_idx"]
            for sm in all_summaries.values()
            for la in sm.get("layer_accs", [])
        })
        print("\n" + "=" * 90)
        print("  FINAL LAYER-WISE SUMMARY  (hit_top1 %)")
        print("=" * 90)
        header = f"  {'benchmark':30s}" + "".join(f"  L{l:>2}" for l in all_layers) + "  fusion"
        print(header)
        print(f"  {'-' * 86}")
        for bk, sm in all_summaries.items():
            l2acc = {la["layer_idx"]: la["hit_top1"] for la in sm.get("layer_accs", [])}
            f_acc = sm.get("fusion_acc", {}).get("hit_top1", 0.0)
            row = f"  {sm.get('bench', bk):30s}"
            for l in all_layers:
                row += f"  {l2acc.get(l, float('nan')):5.1f}"
            row += f"  {f_acc:5.1f}"
            print(row)
        print("=" * 90)

    with open(os.path.join(output_dir, "layerwise_all_summary.json"), "w") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)
    print(f"[ZwerGe] Done. Results → {output_dir}")


if __name__ == "__main__":
    main()
