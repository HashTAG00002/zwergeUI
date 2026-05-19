#!/usr/bin/env python3
"""
ZwerGe-UI Failure Case Visualization
======================================
对评测数据集中的 failure case（或任意样本）进行多层 attention 热图可视化。

输出格式（每个样本一张 PNG）：
  ┌─────────────────────────────────────────────────────────┐
  │  [L18] [L19] [L20] ... [L27] [fusion]   ← 热图网格       │
  │   N+1 列 × 1 行，每格: 原图 + attention 热图叠加         │
  │   绿虚框 = GT bbox，绿/红圆点 = 预测点（hit/miss）        │
  │─────────────────────────────────────────────────────────│
  │  omega bar: ████░░░░░░  各层融合权重                      │
  │  bench | ui_type | instruction                           │
  └─────────────────────────────────────────────────────────┘

用法：
  # 从 bench 数据集中随机取 failure case（重推理）
  python vis_zwerge.py \\
      --ckpt  <ckpt_dir> \\
      --bench ss_pro \\
      --n_samples 20 \\
      --case_type near_miss \\
      --output_dir <vis_dir>

  # 用已有 per-sample JSON 定位 failure index，再重推理可视化
  python vis_zwerge.py \\
      --ckpt  <ckpt_dir> \\
      --bench ss_pro \\
      --results_json <path/to/ss_pro_layerwise_results.json> \\
      --n_samples 20 \\
      --case_type near_miss \\
      --output_dir <vis_dir>

  # 只看随机样本（不筛选 case type）
  python vis_zwerge.py \\
      --ckpt  <ckpt_dir> \\
      --bench ss_pro \\
      --n_samples 8 \\
      --case_type random \\
      --output_dir <vis_dir>

case_type 说明：
  near_miss  : fusion_hit1=0 且 fusion_overlap1=1（预测点在 GT 外面但"接近"GT，hit-overlap 差距分析）
  far_miss   : fusion_hit1=0 且 fusion_overlap1=0（预测点与 GT 完全没有 overlap）
  hit        : fusion_hit1=1（正确预测，对照组）
  all_miss   : fusion_hit1=0（所有 miss）
  random     : 随机取 N 个样本（不过滤）
"""

import argparse
import json
import math
import os
import random
import sys
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR    = os.path.join(_SCRIPT_DIR, "..", "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from inference_zwerge import (
    load_zwerge_model,
    get_prediction_region_point,
    point_in_bbox,
    do_boxes_overlap,
)
from eval_layerwise import (
    BENCH_CONFIGS,
    zwerge_predict_layerwise,
    _scores_to_point_and_topk,
)


# ─────────────────────────────────────────────────────────────────────────────
# Colormap（与 GUI-AIMA/eval/layer_probe.py 完全一致）
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
    """patch 后验概率 → (n_h, n_w, 3) uint8 RGB 热图（冷→热色标）"""
    s = scores_1d.float().cpu().numpy()
    s_min, s_max = s.min(), s.max()
    if s_max - s_min < 1e-9:
        s_norm = np.zeros_like(s)
    else:
        s_norm = (s - s_min) / (s_max - s_min)
    s_grid = s_norm.reshape(n_h, n_w)
    flat = s_grid.reshape(-1)
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
# 单格渲染：原图 + 热图叠加 + GT bbox + 预测点 + 标注条
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
    """
    渲染单个可视化格（resize 到 cell_w × cell_h）。

    label: 底部标注条文字，如 "L21  HIT ✓" 或 "fusion  NEAR ⚠"
    hit:   是否 hit（绿/红 决定预测点颜色）
    overlap: 仅用于 label strip 颜色区分（near miss = yellow）
    """
    # 1. resize 原图到 cell 尺寸
    img_resized = orig_img.convert("RGB").resize((cell_w, cell_h), Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)
    W, H = cell_w, cell_h

    # 2. 热图叠加
    patch_rgb = _scores_to_rgb(scores_1d, n_h, n_w)
    hm_pil = Image.fromarray(patch_rgb, "RGB").resize((W, H), Image.Resampling.BILINEAR if hasattr(Image, 'Resampling') else Image.BILINEAR)
    hm_arr  = np.array(hm_pil, dtype=np.float32)
    orig_arr = np.array(img_resized, dtype=np.float32)
    blend_arr = ((1 - alpha) * orig_arr + alpha * hm_arr).clip(0, 255).astype(np.uint8)
    result = Image.fromarray(blend_arr, "RGB")
    draw = ImageDraw.Draw(result)

    # 3. GT bbox（绿色虚线框）
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

    # 4. 预测点（hit→绿, near_miss→橙, far_miss→红）
    px_px = int(pred_xy[0] * W)
    py_px = int(pred_xy[1] * H)
    if hit:
        dot_color = (0, 220, 0)
    elif overlap:
        dot_color = (255, 165, 0)   # orange = near miss
    else:
        dot_color = (220, 0, 0)     # red = far miss
    r = max(4, min(W, H) // 60)
    draw.ellipse([px_px - r, py_px - r, px_px + r, py_px + r],
                 fill=dot_color, outline=(255, 255, 255), width=1)
    draw.line([(px_px - r*2, py_px), (px_px + r*2, py_px)], fill=dot_color, width=1)
    draw.line([(px_px, py_px - r*2), (px_px, py_px + r*2)], fill=dot_color, width=1)

    # 5. 底部标注条
    strip_h = max(18, H // 20)
    strip = Image.new("RGB", (W, strip_h), color=(20, 20, 20))
    sdraw = ImageDraw.Draw(strip)
    font = _try_load_font(strip_h - 4)
    if hit:
        text_color = (80, 220, 80)
    elif overlap:
        text_color = (255, 200, 50)   # orange
    else:
        text_color = (220, 80, 80)
    sdraw.text((3, 1), label, fill=text_color, font=font)

    combined = Image.new("RGB", (W, H + strip_h))
    combined.paste(result, (0, 0))
    combined.paste(strip, (0, H))
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# omega 条形图（单行图片）
# ─────────────────────────────────────────────────────────────────────────────

def render_omega_bar(
    omega: torch.Tensor,
    layer_indices: List[int],
    total_width: int,
    bar_h: int = 60,
) -> Image.Image:
    """渲染 omega 层权重条形图（横向，各层一个色块）。"""
    n = len(omega)
    om = omega.float().cpu().numpy()
    img = Image.new("RGB", (total_width, bar_h), (15, 15, 15))
    draw = ImageDraw.Draw(img)
    bar_w_each = total_width // max(n, 1)
    font = _try_load_font(max(10, bar_h // 5))
    max_bar_inner_h = bar_h - 20  # 顶部留文字，底部留层号

    for i, (w, li) in enumerate(zip(om, layer_indices)):
        x0 = i * bar_w_each
        # 颜色：按权重从蓝（低）→红（高）
        red   = int(min(255, w * 255 * n))
        blue  = int(max(0, 255 - w * 255 * n))
        color = (red, 80, blue)
        bar_inner_h = int(w * max_bar_inner_h * n)  # 归一化到最大高度
        bar_inner_h = max(2, min(max_bar_inner_h, bar_inner_h))
        y_bot = bar_h - 14
        y_top = y_bot - bar_inner_h
        draw.rectangle([x0 + 2, y_top, x0 + bar_w_each - 2, y_bot], fill=color)
        draw.text((x0 + 2, y_bot + 1), f"L{li}", fill=(180, 180, 180), font=font)
        draw.text((x0 + 2, y_top - 13), f"{w:.2f}", fill=(220, 220, 220), font=font)

    return img


# ─────────────────────────────────────────────────────────────────────────────
# 信息条（instruction + meta）
# ─────────────────────────────────────────────────────────────────────────────

def render_info_bar(
    instruction: str,
    meta: Dict,
    total_width: int,
    info_h: int = 50,
) -> Image.Image:
    """底部文字信息条。"""
    img = Image.new("RGB", (total_width, info_h), (10, 10, 30))
    draw = ImageDraw.Draw(img)
    font_sm = _try_load_font(12)

    # meta 信息
    meta_str_parts = []
    for k in ["bench", "ui_type", "data_type", "GUI_types", "grounding_type", "task_type"]:
        if k in meta and meta[k]:
            meta_str_parts.append(f"{k}={meta[k]}")
    meta_str = "  |  ".join(meta_str_parts)
    draw.text((6, 4), meta_str, fill=(160, 200, 255), font=font_sm)

    # instruction（截断）
    instr_display = instruction if len(instruction) <= 120 else instruction[:117] + "..."
    draw.text((6, 22), f"► {instr_display}", fill=(220, 220, 180), font=font_sm)

    # hit/overlap 状态
    fhit  = meta.get("fusion_hit1", -1)
    fov   = meta.get("fusion_overlap1", -1)
    if fhit == 1:
        status = "FUSION: HIT ✓"
        sc = (80, 220, 80)
    elif fhit == 0 and fov == 1:
        status = "FUSION: NEAR MISS ⚠"
        sc = (255, 200, 50)
    elif fhit == 0 and fov == 0:
        status = "FUSION: FAR MISS ✗"
        sc = (220, 80, 80)
    else:
        status = ""
        sc = (200, 200, 200)
    draw.text((total_width - 250, 4), status, fill=sc, font=_try_load_font(14))

    return img


# ─────────────────────────────────────────────────────────────────────────────
# 单样本完整可视化（N 层 + fusion，拼接成宽幅图）
# ─────────────────────────────────────────────────────────────────────────────

def visualize_sample(
    orig_img: Image.Image,
    pred: Dict,
    gt_bbox_norm: Tuple[float, float, float, float],
    instruction: str,
    meta: Dict,
    activation_threshold: float = 0.3,
    decode_strategy: str = "centroid",
    cell_w: int = 300,
    cell_h: int = 220,
    alpha: float = 0.55,
) -> Image.Image:
    """
    生成单个样本的完整可视化图。

    pred: 来自 zwerge_predict_layerwise 的输出字典
    meta: 额外元信息（bench, ui_type 等），用于底部标注

    返回：一张合并好的 PIL Image
    """
    n_w = pred["n_width"]
    n_h_patches = pred["n_height"]
    layer_indices = pred["layer_indices"]
    per_layer_probs = pred["per_layer_probs"]
    p_final = pred["p_final"]
    omega   = pred["omega"]
    n_probes = len(layer_indices)

    phx = 0.5 / n_w
    phy = 0.5 / n_h_patches

    def _judge(px, py):
        hit = int(point_in_bbox(px, py, gt_bbox_norm))
        pred_box = (px - phx, py - phy, px + phx, py + phy)
        ov  = int(do_boxes_overlap(pred_box, gt_bbox_norm))
        return bool(hit), bool(ov)

    # ── 各层单格 ──
    cells = []
    for li, (p_l, layer_idx) in enumerate(zip(per_layer_probs, layer_indices)):
        best, _ = _scores_to_point_and_topk(
            p=p_l,
            n_width=n_w,
            n_height=n_h_patches,
            activation_threshold=activation_threshold,
            topk=1,
            decode_strategy=decode_strategy,
        )
        px, py = float(best[0]), float(best[1])
        hit, ov = _judge(px, py)
        if hit:
            suffix = "HIT ✓"
        elif ov:
            suffix = "NEAR ⚠"
        else:
            suffix = "MISS ✗"
        label = f"L{layer_idx:02d}  {suffix}"
        cell = render_one_cell(
            orig_img=orig_img,
            scores_1d=p_l,
            n_w=n_w,
            n_h=n_h_patches,
            gt_bbox_norm=gt_bbox_norm,
            pred_xy=(px, py),
            hit=hit,
            overlap=ov,
            label=label,
            alpha=alpha,
            cell_w=cell_w,
            cell_h=cell_h,
        )
        cells.append(cell)

    # ── fusion 格（用稍微高亮的边框区分）──
    f_best, _ = _scores_to_point_and_topk(
        p=p_final,
        n_width=n_w,
        n_height=n_h_patches,
        activation_threshold=activation_threshold,
        topk=1,
        decode_strategy=decode_strategy,
    )
    fpx, fpy = float(f_best[0]), float(f_best[1])
    fhit, fov = _judge(fpx, fpy)
    if fhit:
        fsuffix = "HIT ✓"
    elif fov:
        fsuffix = "NEAR ⚠"
    else:
        fsuffix = "MISS ✗"
    fusion_cell = render_one_cell(
        orig_img=orig_img,
        scores_1d=p_final,
        n_w=n_w,
        n_h=n_h_patches,
        gt_bbox_norm=gt_bbox_norm,
        pred_xy=(fpx, fpy),
        hit=fhit,
        overlap=fov,
        label=f"fusion  {fsuffix}",
        alpha=alpha,
        cell_w=cell_w,
        cell_h=cell_h,
    )
    cells.append(fusion_cell)

    # ── 拼成一行（所有 probe layer + fusion）──
    cell_img_h = cells[0].height
    n_cells = len(cells)
    total_width = cell_w * n_cells
    grid = Image.new("RGB", (total_width, cell_img_h), (5, 5, 5))
    for i, cell in enumerate(cells):
        grid.paste(cell, (i * cell_w, 0))

    # ── omega 条形图 ──
    omega_bar = render_omega_bar(
        omega=omega,
        layer_indices=layer_indices,
        total_width=total_width,
        bar_h=60,
    )

    # ── 信息条 ──
    info_meta = dict(meta)
    info_meta["fusion_hit1"]     = int(fhit)
    info_meta["fusion_overlap1"] = int(fov)
    info_bar = render_info_bar(
        instruction=instruction,
        meta=info_meta,
        total_width=total_width,
        info_h=50,
    )

    # ── 合并 ──
    total_h = cell_img_h + omega_bar.height + info_bar.height
    canvas = Image.new("RGB", (total_width, total_h), (5, 5, 5))
    canvas.paste(grid,      (0, 0))
    canvas.paste(omega_bar, (0, cell_img_h))
    canvas.paste(info_bar,  (0, cell_img_h + omega_bar.height))
    return canvas


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def _filter_indices_by_case_type(
    data: List[Dict],
    results_json: Optional[str],
    case_type: str,
    n_samples: int,
    seed: int,
) -> List[int]:
    """
    根据 case_type 筛选样本 index。

    若提供 results_json（eval_layerwise 的 --save_per_sample 输出），直接从中读 fusion hit/overlap；
    否则对 case_type=random 直接随机取，其他 case type 需要运行后再筛选（返回全量 index，由调用方判断）。

    Returns: list of global data indices（在 data 列表里的下标）
    """
    rng = random.Random(seed)

    if results_json and os.path.exists(results_json):
        with open(results_json) as f:
            per_sample = json.load(f)
        # per_sample 里的 idx 是 global index（与 data 对应）
        idx_to_rec = {r["idx"]: r for r in per_sample}
        candidate_indices = []
        for i, example in enumerate(data):
            rec = idx_to_rec.get(i)
            if rec is None:
                continue
            fhit = rec.get("fusion_hit1", -1)
            fov  = rec.get("fusion_overlap1", -1)
            if case_type == "near_miss"  and fhit == 0 and fov == 1:
                candidate_indices.append(i)
            elif case_type == "far_miss" and fhit == 0 and fov == 0:
                candidate_indices.append(i)
            elif case_type == "hit"      and fhit == 1:
                candidate_indices.append(i)
            elif case_type == "all_miss" and fhit == 0:
                candidate_indices.append(i)
            elif case_type == "random":
                candidate_indices.append(i)
        rng.shuffle(candidate_indices)
        return candidate_indices[:n_samples]
    else:
        # 无 results_json → 随机取（其他 case type 需要重推理后判断）
        all_indices = list(range(len(data)))
        rng.shuffle(all_indices)
        if case_type == "random":
            return all_indices[:n_samples]
        else:
            # 取更多候选，推理后再筛选
            # 取 n_samples * 10 个候选，推理后过滤
            return all_indices[:n_samples * 10]


def main():
    parser = argparse.ArgumentParser(
        description="ZwerGe-UI Failure Case Visualization"
    )
    parser.add_argument("--ckpt",        required=True,  help="ZwerGe checkpoint 路径")
    parser.add_argument("--bench",       default="ss_pro",
                        choices=list(BENCH_CONFIGS.keys()), help="benchmark 名称")
    parser.add_argument("--eval_dir",
                        default="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/datasets/evaluation",
                        help="eval 数据集根目录")
    parser.add_argument("--output_dir",
                        default="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/zwerge/data/results/vis_failures",
                        help="可视化输出目录")
    parser.add_argument("--results_json",   default=None,
                        help="eval_layerwise --save_per_sample 产出的 JSON，用于快速定位 failure index")
    parser.add_argument("--n_samples",  type=int, default=20,
                        help="可视化样本数（默认 20）")
    parser.add_argument("--case_type",  default="near_miss",
                        choices=["near_miss", "far_miss", "all_miss", "hit", "random"],
                        help=(
                            "near_miss=hit=0&overlap=1（接近但未命中，是主要分析场景）; "
                            "far_miss=hit=0&overlap=0; hit=正确; all_miss=所有 miss; random=随机"
                        ))
    parser.add_argument("--max_pixels",  type=int, default=12845056)
    parser.add_argument("--cell_w",     type=int, default=300, help="每格宽度（像素）")
    parser.add_argument("--cell_h",     type=int, default=220, help="每格高度（像素）")
    parser.add_argument("--alpha",      type=float, default=0.55, help="热图叠加透明度")
    parser.add_argument("--decode_strategy", default="centroid",
                        choices=["centroid", "argmax", "peak_shift", "temperature"])
    parser.add_argument("--activation_threshold", type=float, default=0.3)
    parser.add_argument("--gpu_id",     type=int, default=0)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--attn_impl",  default="flash_attention_2",
                        choices=["flash_attention_2", "sdpa", "eager"])
    args = parser.parse_args()

    # ── 配置 ──
    cfg       = BENCH_CONFIGS[args.bench]
    bench_name = cfg["name"]
    eval_json_path = os.path.join(args.eval_dir, cfg["eval_dir"], cfg["eval_json"])
    img_root  = os.path.join(args.eval_dir, cfg["eval_dir"])
    group_field = cfg.get("group_field")

    assert os.path.exists(eval_json_path), f"eval.json not found: {eval_json_path}"
    with open(eval_json_path) as f:
        data = json.load(f)
    print(f"[vis] bench={bench_name}, total={len(data)} samples")

    # ── 筛选 index ──
    candidate_indices = _filter_indices_by_case_type(
        data=data,
        results_json=args.results_json,
        case_type=args.case_type,
        n_samples=args.n_samples,
        seed=args.seed,
    )
    print(f"[vis] candidate_indices: {len(candidate_indices)} → will infer up to {args.n_samples}")

    # ── 加载模型 ──
    device = torch.device(f"cuda:{args.gpu_id}")
    print(f"[vis] Loading model from {args.ckpt} ...")
    model, processor = load_zwerge_model(
        ckpt_path=args.ckpt,
        attn_implementation=args.attn_impl,
        device=str(device),
        dtype=torch.bfloat16,
    )
    processor.image_processor.max_pixels = args.max_pixels
    print("[vis] Model loaded.")

    # ── 输出目录 ──
    run_tag = f"{args.bench}_{args.case_type}"
    out_dir = os.path.join(args.output_dir, run_tag)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[vis] Output dir: {out_dir}")

    # ── 推理 + 可视化 ──
    saved = 0
    meta_records = []

    for cand_idx in candidate_indices:
        if saved >= args.n_samples:
            break

        example = data[cand_idx]
        img_path = os.path.join(img_root, example["image_path"])
        if not os.path.exists(img_path):
            warnings.warn(f"Image not found: {img_path}")
            continue

        W, H = float(example["image_size"][0]), float(example["image_size"][1])
        x1, y1, x2, y2 = example["gt_bbox"]
        gt_bbox_norm = (x1 / W, y1 / H, x2 / W, y2 / H)

        try:
            orig_img = Image.open(img_path).convert("RGB")
        except Exception as e:
            warnings.warn(f"Failed to open {img_path}: {e}")
            continue

        try:
            pred = zwerge_predict_layerwise(
                image=orig_img,
                instruction=example["instruction"],
                model=model,
                processor=processor,
                device=device,
                activation_threshold=args.activation_threshold,
                topk=1,
                decode_strategy=args.decode_strategy,
            )
        except Exception as e:
            warnings.warn(f"Inference failed for #{cand_idx}: {e}")
            continue

        # 判断 fusion 的 hit / overlap（用于 case_type 筛选）
        n_w = pred["n_width"]
        n_h_patches = pred["n_height"]
        phx = 0.5 / n_w
        phy = 0.5 / n_h_patches
        f_best, _ = _scores_to_point_and_topk(
            p=pred["p_final"],
            n_width=n_w,
            n_height=n_h_patches,
            activation_threshold=args.activation_threshold,
            topk=1,
            decode_strategy=args.decode_strategy,
        )
        fpx, fpy = float(f_best[0]), float(f_best[1])
        fhit = int(point_in_bbox(fpx, fpy, gt_bbox_norm))
        fpred_box = (fpx - phx, fpy - phy, fpx + phx, fpy + phy)
        fov  = int(do_boxes_overlap(fpred_box, gt_bbox_norm))

        # 如果没有 results_json，在这里根据实际结果过滤 case_type
        if args.results_json is None or not os.path.exists(args.results_json or ""):
            if args.case_type == "near_miss" and not (fhit == 0 and fov == 1):
                continue
            elif args.case_type == "far_miss" and not (fhit == 0 and fov == 0):
                continue
            elif args.case_type == "hit" and fhit != 1:
                continue
            elif args.case_type == "all_miss" and fhit != 0:
                continue
            # random → 不过滤

        # meta 信息
        meta = {
            "bench":           bench_name,
            "global_idx":      cand_idx,
            "image_path":      example.get("image_path", ""),
            "instruction":     example.get("instruction", ""),
            "fusion_hit1":     fhit,
            "fusion_overlap1": fov,
        }
        if group_field and group_field in example:
            meta[group_field] = example[group_field]
        for k in ["ui_type", "data_type", "GUI_types", "grounding_type", "task_type",
                  "platform", "application", "element_type"]:
            if k in example:
                meta[k] = example[k]

        # 可视化
        try:
            vis_img = visualize_sample(
                orig_img=orig_img,
                pred=pred,
                gt_bbox_norm=gt_bbox_norm,
                instruction=example["instruction"],
                meta=meta,
                activation_threshold=args.activation_threshold,
                decode_strategy=args.decode_strategy,
                cell_w=args.cell_w,
                cell_h=args.cell_h,
                alpha=args.alpha,
            )
        except Exception as e:
            warnings.warn(f"Visualization failed for #{cand_idx}: {e}")
            import traceback; traceback.print_exc()
            continue

        # 保存
        if fhit:
            prefix = "hit"
        elif fov:
            prefix = "near"
        else:
            prefix = "far"
        fname = f"{prefix}_{saved:04d}_idx{cand_idx:05d}.png"
        fpath = os.path.join(out_dir, fname)
        vis_img.save(fpath)
        meta_records.append({"file": fname, **meta})
        saved += 1
        print(f"  [{saved}/{args.n_samples}] saved → {fname}  "
              f"(fusion: {'HIT' if fhit else 'NEAR' if fov else 'MISS'}  "
              f"omega_max_layer=L{pred['layer_indices'][int(pred['omega'].argmax())]:02d})")

    # ── 保存 meta JSON ──
    meta_path = os.path.join(out_dir, "vis_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta_records, f, indent=2, ensure_ascii=False)

    print(f"\n[vis] Done. {saved} images saved to {out_dir}")
    print(f"[vis] Meta → {meta_path}")


if __name__ == "__main__":
    main()
