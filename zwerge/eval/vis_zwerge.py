#!/usr/bin/env python3
"""
ZwerGe-UI Evaluation + Visualization (All-in-One, Multi-GPU)
=============================================================
在一次 job 中完成评测（逐层+fusion 指标）和全量可视化。
多卡并行，每张卡处理数据集的一个分片，推理完成后聚合结果。

输出目录结构（与 eval_layerwise.py 相同路径）：
  {output_dir}/{decode_strategy}/{train_run}/{ckpt_name}/
  ├── {bench}_layerwise_summary.json    ← 与 eval_layerwise 完全相同格式
  ├── layerwise_all_summary.json
  └── details/
      └── {bench}/
          ├── success/                  ← fusion_hit1=1 样本可视化
          │   └── idx00042.png
          ├── failure/                  ← fusion_hit1=0 样本可视化
          │   └── idx00003.png
          └── results.json             ← 逐样本评测记录（同 eval_layerwise --save_per_sample）

用法：
  bash run_vis.sh ss_pro        # 单 bench，多卡自动
  bash run_vis.sh all           # 全部 5 个 bench
"""

import argparse
import glob
import json
import os
import sys
import time
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR    = os.path.join(_SCRIPT_DIR, "..", "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from inference_zwerge import (
    load_zwerge_model,
    point_in_bbox,
    do_boxes_overlap,
)
from eval_layerwise import (
    BENCH_CONFIGS,
    zwerge_predict_layerwise,
    scores_to_point_and_topk,
    _get_group_key,
    _print_layerwise_summary,
)


# ─────────────────────────────────────────────────────────────────────────────
# Colormap（与 GUI-AIMA layer_probe.py 完全一致）
# ─────────────────────────────────────────────────────────────────────────────
_CMAP_STOPS = np.array([
    [0,   0,   0  ],   # 0.00  black
    [0,   0,   255],   # 0.25  blue
    [0,   255, 255],   # 0.50  cyan
    [0,   255, 0  ],   # 0.625 green
    [255, 255, 0  ],   # 0.75  yellow
    [255, 0,   0  ],   # 1.00  red
], dtype=np.float32)
_CMAP_POS = np.array([0.0, 0.25, 0.50, 0.625, 0.75, 1.0], dtype=np.float32)


def _scores_to_rgb(scores_1d: torch.Tensor, n_h: int, n_w: int) -> np.ndarray:
    """patch 后验 → (n_h, n_w, 3) uint8 RGB 热图（冷→热色标）"""
    s = scores_1d.float().cpu().numpy()
    s_min, s_max = s.min(), s.max()
    s_norm = np.zeros_like(s) if s_max - s_min < 1e-9 else (s - s_min) / (s_max - s_min)
    flat = s_norm.reshape(n_h, n_w).reshape(-1)
    rgb_flat = np.zeros((len(flat), 3), dtype=np.float32)
    for i in range(len(_CMAP_POS) - 1):
        lo, hi = _CMAP_POS[i], _CMAP_POS[i + 1]
        mask = (flat >= lo) & (flat <= hi)
        if not mask.any():
            continue
        t = (flat[mask] - lo) / (hi - lo + 1e-9)
        c0, c1 = _CMAP_STOPS[i], _CMAP_STOPS[i + 1]
        rgb_flat[mask] = c0[None] * (1 - t[:, None]) + c1[None] * t[:, None]
    return rgb_flat.reshape(n_h, n_w, 3).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# 单格渲染
# ─────────────────────────────────────────────────────────────────────────────

def _try_load_font(size: int):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def render_one_cell(
    orig_img: Image.Image,
    scores_1d: torch.Tensor,
    n_w: int,
    n_h: int,
    gt_bbox_norm: Tuple[float, float, float, float],
    pred_xy: Tuple[float, float],
    hit: bool,
    overlap: bool,
    label: str,
    alpha: float = 0.55,
    cell_w: int = 320,
    cell_h: int = 240,
) -> Image.Image:
    img_resized = orig_img.convert("RGB").resize(
        (cell_w, cell_h),
        Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS,
    )
    W, H = cell_w, cell_h

    patch_rgb = _scores_to_rgb(scores_1d, n_h, n_w)
    hm_pil = Image.fromarray(patch_rgb, "RGB").resize(
        (W, H),
        Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR,
    )
    blend_arr = (
        (1 - alpha) * np.array(img_resized, dtype=np.float32) +
        alpha       * np.array(hm_pil,      dtype=np.float32)
    ).clip(0, 255).astype(np.uint8)
    result = Image.fromarray(blend_arr, "RGB")
    draw = ImageDraw.Draw(result)

    # GT bbox（绿色虚线框）
    x1n, y1n, x2n, y2n = gt_bbox_norm
    bx1, by1 = int(x1n * W), int(y1n * H)
    bx2, by2 = int(x2n * W), int(y2n * H)
    dash = 5
    for (ex1, ey1, ex2, ey2), axis in [
        ((bx1, by1, bx2, by1), "x"), ((bx1, by2, bx2, by2), "x"),
        ((bx1, by1, bx1, by2), "y"), ((bx2, by1, bx2, by2), "y"),
    ]:
        if axis == "x":
            x = ex1
            while x < ex2:
                draw.line([(x, ey1), (min(x + dash, ex2), ey1)], fill=(0, 220, 0), width=2)
                x += dash * 2
        else:
            y = ey1
            while y < ey2:
                draw.line([(ex1, y), (ex1, min(y + dash, ey2))], fill=(0, 220, 0), width=2)
                y += dash * 2

    # 预测点（hit→绿, near→橙, far→红）
    px_px = int(pred_xy[0] * W)
    py_px = int(pred_xy[1] * H)
    dot_color = (0, 220, 0) if hit else ((255, 165, 0) if overlap else (220, 0, 0))
    r = max(4, min(W, H) // 60)
    draw.ellipse([px_px - r, py_px - r, px_px + r, py_px + r],
                 fill=dot_color, outline=(255, 255, 255), width=1)
    draw.line([(px_px - r * 2, py_px), (px_px + r * 2, py_px)], fill=dot_color, width=1)
    draw.line([(px_px, py_px - r * 2), (px_px, py_px + r * 2)], fill=dot_color, width=1)

    # 底部标注条
    strip_h = max(18, H // 20)
    strip = Image.new("RGB", (W, strip_h), color=(20, 20, 20))
    sdraw = ImageDraw.Draw(strip)
    text_color = (80, 220, 80) if hit else ((255, 200, 50) if overlap else (220, 80, 80))
    sdraw.text((3, 1), label, fill=text_color, font=_try_load_font(strip_h - 4))

    combined = Image.new("RGB", (W, H + strip_h))
    combined.paste(result, (0, 0))
    combined.paste(strip,  (0, H))
    return combined


def render_omega_bar(
    omega: torch.Tensor,
    layer_indices: List[int],
    total_width: int,
    bar_h: int = 60,
) -> Image.Image:
    n  = len(omega)
    om = omega.float().cpu().numpy()
    img  = Image.new("RGB", (total_width, bar_h), (15, 15, 15))
    draw = ImageDraw.Draw(img)
    bar_w_each = total_width // max(n, 1)
    font = _try_load_font(max(10, bar_h // 5))
    max_inner_h = bar_h - 20
    for i, (w, li) in enumerate(zip(om, layer_indices)):
        x0 = i * bar_w_each
        red   = int(min(255, w * 255 * n))
        blue  = int(max(0, 255 - w * 255 * n))
        color = (red, 80, blue)
        inner_h = max(2, min(max_inner_h, int(w * max_inner_h * n)))
        y_bot = bar_h - 14
        y_top = y_bot - inner_h
        draw.rectangle([x0 + 2, y_top, x0 + bar_w_each - 2, y_bot], fill=color)
        draw.text((x0 + 2, y_bot + 1),  f"L{li}",   fill=(180, 180, 180), font=font)
        draw.text((x0 + 2, y_top - 13), f"{w:.2f}", fill=(220, 220, 220), font=font)
    return img


def render_info_bar(
    instruction: str,
    meta: Dict,
    total_width: int,
    info_h: int = 50,
) -> Image.Image:
    img  = Image.new("RGB", (total_width, info_h), (10, 10, 30))
    draw = ImageDraw.Draw(img)
    font_sm = _try_load_font(12)

    meta_parts = [
        f"{k}={meta[k]}"
        for k in ["bench", "ui_type", "data_type", "GUI_types", "grounding_type", "task_type"]
        if k in meta and meta[k]
    ]
    draw.text((6, 4),  "  |  ".join(meta_parts), fill=(160, 200, 255), font=font_sm)

    instr = instruction if len(instruction) <= 120 else instruction[:117] + "..."
    draw.text((6, 22), f"► {instr}", fill=(220, 220, 180), font=font_sm)

    fhit = meta.get("fusion_hit1", -1)
    fov  = meta.get("fusion_overlap1", -1)
    if fhit == 1:
        status, sc = "FUSION: HIT ✓", (80, 220, 80)
    elif fhit == 0 and fov == 1:
        status, sc = "FUSION: NEAR MISS ⚠", (255, 200, 50)
    elif fhit == 0 and fov == 0:
        status, sc = "FUSION: FAR MISS ✗", (220, 80, 80)
    else:
        status, sc = "", (200, 200, 200)
    draw.text((total_width - 250, 4), status, fill=sc, font=_try_load_font(14))
    return img


def visualize_sample(
    orig_img: Image.Image,
    pred: Dict,
    gt_bbox_norm: Tuple[float, float, float, float],
    instruction: str,
    meta: Dict,
    activation_threshold: float = 0.3,
    decode_strategy: str = "centroid",
    peak_shift_alpha: float = 0.5,
    temperature: float = 0.5,
    cell_w: int = 300,
    cell_h: int = 220,
    alpha: float = 0.55,
) -> Image.Image:
    n_w           = pred["n_width"]
    n_h_patches   = pred["n_height"]
    layer_indices = pred["layer_indices"]
    per_layer_probs = pred["per_layer_probs"]
    p_final = pred["p_final"]
    omega   = pred["omega"]
    phx = 0.5 / n_w
    phy = 0.5 / n_h_patches

    def _judge(px, py):
        pred_box = (px - phx, py - phy, px + phx, py + phy)
        return bool(point_in_bbox(px, py, gt_bbox_norm)), bool(do_boxes_overlap(pred_box, gt_bbox_norm))

    cells = []
    for p_l, layer_idx in zip(per_layer_probs, layer_indices):
        best, _ = scores_to_point_and_topk(
            p=p_l, n_width=n_w, n_height=n_h_patches,
            activation_threshold=activation_threshold, topk=1,
            decode_strategy=decode_strategy,
            peak_shift_alpha=peak_shift_alpha, temperature=temperature,
        )
        px, py  = float(best[0]), float(best[1])
        hit, ov = _judge(px, py)
        suffix  = "HIT ✓" if hit else ("NEAR ⚠" if ov else "MISS ✗")
        cells.append(render_one_cell(
            orig_img=orig_img, scores_1d=p_l, n_w=n_w, n_h=n_h_patches,
            gt_bbox_norm=gt_bbox_norm, pred_xy=(px, py), hit=hit, overlap=ov,
            label=f"L{layer_idx:02d}  {suffix}", alpha=alpha,
            cell_w=cell_w, cell_h=cell_h,
        ))

    f_best, _ = scores_to_point_and_topk(
        p=p_final, n_width=n_w, n_height=n_h_patches,
        activation_threshold=activation_threshold, topk=1,
        decode_strategy=decode_strategy,
        peak_shift_alpha=peak_shift_alpha, temperature=temperature,
    )
    fpx, fpy  = float(f_best[0]), float(f_best[1])
    fhit, fov = _judge(fpx, fpy)
    fsuffix   = "HIT ✓" if fhit else ("NEAR ⚠" if fov else "MISS ✗")
    cells.append(render_one_cell(
        orig_img=orig_img, scores_1d=p_final, n_w=n_w, n_h=n_h_patches,
        gt_bbox_norm=gt_bbox_norm, pred_xy=(fpx, fpy), hit=fhit, overlap=fov,
        label=f"fusion  {fsuffix}", alpha=alpha, cell_w=cell_w, cell_h=cell_h,
    ))

    cell_img_h  = cells[0].height
    total_width = cell_w * len(cells)
    grid = Image.new("RGB", (total_width, cell_img_h), (5, 5, 5))
    for i, cell in enumerate(cells):
        grid.paste(cell, (i * cell_w, 0))

    omega_bar = render_omega_bar(omega=omega, layer_indices=layer_indices,
                                 total_width=total_width, bar_h=60)
    info_meta = {**meta, "fusion_hit1": int(fhit), "fusion_overlap1": int(fov)}
    info_bar  = render_info_bar(instruction=instruction, meta=info_meta,
                                total_width=total_width, info_h=50)

    total_h = cell_img_h + omega_bar.height + info_bar.height
    canvas = Image.new("RGB", (total_width, total_h), (5, 5, 5))
    canvas.paste(grid,      (0, 0))
    canvas.paste(omega_bar, (0, cell_img_h))
    canvas.paste(info_bar,  (0, cell_img_h + omega_bar.height))
    return canvas


# ─────────────────────────────────────────────────────────────────────────────
# 多卡 Worker
# ─────────────────────────────────────────────────────────────────────────────

def _vis_worker(
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
    cell_w: int,
    cell_h: int,
    alpha: float,
    group_stats: bool,
):
    import torch
    device = torch.device(f"cuda:{gpu_id}")
    model, processor = load_zwerge_model(
        ckpt_path=ckpt_path,
        attn_implementation=attn_impl,
        device=str(device),
        dtype=torch.bfloat16,
    )
    processor.image_processor.max_pixels = max_pixels

    cfg        = BENCH_CONFIGS[bench_key]
    bench_name = cfg["name"]
    group_field = cfg.get("group_field") if group_stats else None
    eval_json_path = os.path.join(eval_root, cfg["eval_dir"], cfg["eval_json"])
    img_root   = os.path.join(eval_root, cfg["eval_dir"])

    with open(eval_json_path) as f:
        shard = json.load(f)[start:end]

    success_dir = os.path.join(output_dir, "details", bench_key, "success")
    failure_dir = os.path.join(output_dir, "details", bench_key, "failure")
    os.makedirs(success_dir, exist_ok=True)
    os.makedirs(failure_dir, exist_ok=True)

    probe_layers = list(model.layerwise_grounding_head.probe_layers)
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
            pred = zwerge_predict_layerwise(
                image=orig_img,
                instruction=example["instruction"],
                model=model,
                processor=processor,
                device=device,
                activation_threshold=activation_threshold,
                topk=topk,
                decode_strategy=decode_strategy,
                peak_shift_alpha=peak_shift_alpha,
                temperature=temperature,
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
                "hit_top1":     hit1,
                "hit_topk":     hitk,
                "overlap_top1": overlap1,
                "overlap_topk": overlapk,
                "pred_point":   list(pred["per_layer_points"][li]),
            })

        # ── fusion 指标 ────────────────────────────────────────────────────────
        f_best, f_centers = scores_to_point_and_topk(
            p=pred["p_final"], n_width=n_w, n_height=n_h,
            activation_threshold=activation_threshold, topk=topk,
            decode_strategy=decode_strategy,
            peak_shift_alpha=peak_shift_alpha, temperature=temperature,
        )
        fpx, fpy  = float(f_best[0]), float(f_best[1])
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

        # ── 可视化 ─────────────────────────────────────────────────────────────
        meta = {"bench": bench_name}
        for k in ["ui_type", "data_type", "GUI_types", "grounding_type", "task_type",
                  "platform", "application", "element_type"]:
            if k in example:
                meta[k] = example[k]

        try:
            vis_img  = visualize_sample(
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

        # ── per-sample 记录 ─────────────────────────────────────────────────────
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
            "fusion_hitk":     fhitk,
            "fusion_overlapk": fovk,
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

    # ── 保存分片数据 ─────────────────────────────────────────────────────────
    details_dir = os.path.join(output_dir, "details", bench_key)
    with open(os.path.join(details_dir, f"results_{start}-{end}.json"), "w") as f:
        json.dump(results, f, ensure_ascii=False)

    valid_total = len(shard) - skip_total
    layer_accs = []
    for li, ls in enumerate(layer_stats):
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
        "bench":             bench_name,
        "bench_key":         bench_key,
        "total":             len(shard),
        "valid":             valid_total,
        "skipped":           skip_total,
        "slice":             [start, end],
        "probe_layers":      probe_layers,
        "layer_accs":        layer_accs,
        "fusion_acc":        fusion_acc,
        "fusion_group_accs": fusion_group_accs,
    }
    with open(os.path.join(output_dir, f"{bench_key}_layerwise_summary_{start}-{end}.json"), "w") as f:
        json.dump(shard_summary, f, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
# 聚合分片结果
# ─────────────────────────────────────────────────────────────────────────────

def _aggregate_vis_shards(output_dir: str, bench_key: str, topk: int) -> Dict:
    shard_files = sorted(glob.glob(
        os.path.join(output_dir, f"{bench_key}_layerwise_summary_*.json")
    ))
    if not shard_files:
        raise FileNotFoundError(f"No shard summary files for {bench_key} in {output_dir}")

    summaries    = [json.load(open(fp)) for fp in shard_files]
    probe_layers = summaries[0]["probe_layers"]
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
        "bench":             bench_name,
        "bench_key":         bench_key,
        "total":             total,
        "valid":             valid,
        "skipped":           skipped,
        "topk":              topk,
        "probe_layers":      probe_layers,
        "layer_accs":        layer_accs,
        "layer_accs_sorted": layer_accs_sorted,
        "fusion_acc":        fusion_acc,
        "fusion_group_accs": fusion_group_accs,
    }

    with open(os.path.join(output_dir, f"{bench_key}_layerwise_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 合并 per-sample results
    details_dir   = os.path.join(output_dir, "details", bench_key)
    result_shards = sorted(glob.glob(os.path.join(details_dir, "results_*.json")))
    all_results   = []
    for fp in result_shards:
        with open(fp) as f:
            all_results.extend(json.load(f))
    all_results.sort(key=lambda x: x["idx"])
    with open(os.path.join(details_dir, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    for fp in shard_files + result_shards:
        try:
            os.remove(fp)
        except OSError:
            pass

    _print_layerwise_summary(summary, topk)
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# 多卡调度
# ─────────────────────────────────────────────────────────────────────────────

def run_bench_vis_parallel(
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
    cell_w: int,
    cell_h: int,
    alpha: float,
    group_stats: bool = True,
) -> Dict:
    import multiprocessing as mp
    import torch

    n_gpu = torch.cuda.device_count()
    if n_gpu == 0:
        raise RuntimeError("No CUDA GPU found.")

    cfg = BENCH_CONFIGS[bench_key]
    with open(os.path.join(eval_root, cfg["eval_dir"], cfg["eval_json"])) as f:
        N = len(json.load(f))
    print(f"[ZwerGe vis] {bench_key}: {N} samples, {n_gpu} GPU(s)")

    worker_kwargs = dict(
        ckpt_path=ckpt_path,
        bench_key=bench_key,
        eval_root=eval_root,
        output_dir=output_dir,
        attn_impl=attn_impl,
        max_pixels=max_pixels,
        decode_strategy=decode_strategy,
        peak_shift_alpha=peak_shift_alpha,
        temperature=temperature,
        activation_threshold=activation_threshold,
        topk=topk,
        cell_w=cell_w,
        cell_h=cell_h,
        alpha=alpha,
        group_stats=group_stats,
    )

    chunk  = (N + n_gpu - 1) // n_gpu
    slices = [(i, i * chunk, min((i + 1) * chunk, N))
              for i in range(n_gpu) if i * chunk < N]

    if n_gpu == 1:
        _vis_worker(gpu_id=0, start=0, end=N, **worker_kwargs)
    else:
        ctx   = mp.get_context("spawn")
        procs = []
        for gpu_id, s, e in slices:
            print(f"[ZwerGe vis]   GPU{gpu_id}: slice [{s}, {e})")
            p = ctx.Process(
                target=_vis_worker,
                kwargs={"gpu_id": gpu_id, "start": s, "end": e, **worker_kwargs},
            )
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"Worker process exited with code {p.exitcode}")

    return _aggregate_vis_shards(output_dir, bench_key, topk)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="ZwerGe-UI Evaluation + Visualization (All-in-One, Multi-GPU)"
    )
    parser.add_argument("--ckpt",       required=True)
    parser.add_argument("--bench",      default="ss_pro",
                        choices=list(BENCH_CONFIGS.keys()) + ["all"])
    parser.add_argument("--eval_dir",
                        default="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/datasets/evaluation")
    parser.add_argument("--output_dir",
                        default="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/zwerge/data/results/zwerge_layerwise")
    parser.add_argument("--attn_impl",  default="flash_attention_2",
                        choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--max_pixels",           type=int,   default=12_845_056)
    parser.add_argument("--activation_threshold", type=float, default=0.3)
    parser.add_argument("--topk",                 type=int,   default=3)
    parser.add_argument("--decode_strategy", default="centroid",
                        choices=["centroid", "argmax", "peak_shift", "temperature"])
    parser.add_argument("--peak_shift_alpha", type=float, default=0.5)
    parser.add_argument("--temperature",      type=float, default=0.5)
    parser.add_argument("--cell_w",  type=int,   default=300)
    parser.add_argument("--cell_h",  type=int,   default=220)
    parser.add_argument("--alpha",   type=float, default=0.55)
    parser.add_argument("--no_group_stats", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    assert os.path.isdir(args.ckpt), f"Checkpoint not found: {args.ckpt}"

    basename   = os.path.basename(args.ckpt)
    basedir    = os.path.basename(os.path.dirname(args.ckpt))
    output_dir = os.path.join(args.output_dir, basedir, basename)
    os.makedirs(output_dir, exist_ok=True)
    print(f"[ZwerGe vis] Output dir: {output_dir}")

    bench_keys    = list(BENCH_CONFIGS.keys()) if args.bench == "all" else [args.bench]
    all_summaries = {}

    for bench_key in bench_keys:
        t0 = time.time()
        summary = run_bench_vis_parallel(
            bench_key=bench_key,
            ckpt_path=args.ckpt,
            eval_root=args.eval_dir,
            output_dir=output_dir,
            attn_impl=args.attn_impl,
            max_pixels=args.max_pixels,
            decode_strategy=args.decode_strategy,
            peak_shift_alpha=args.peak_shift_alpha,
            temperature=args.temperature,
            activation_threshold=args.activation_threshold,
            topk=args.topk,
            cell_w=args.cell_w,
            cell_h=args.cell_h,
            alpha=args.alpha,
            group_stats=not args.no_group_stats,
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
    print(f"[ZwerGe vis] Done. Results → {output_dir}")


if __name__ == "__main__":
    main()
