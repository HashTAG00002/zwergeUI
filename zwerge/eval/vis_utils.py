"""
ZwerGe-UI Visualization Utilities
===================================
Rendering helpers for layer-wise grounding visualization.
Previously embedded in vis_zwerge.py.
"""

from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from inference_base import scores_to_point_and_topk, point_in_bbox, do_boxes_overlap


# ─────────────────────────────────────────────────────────────────────────────
# Colormap (cold→hot, same as GUI-AIMA layer_probe.py)
# ─────────────────────────────────────────────────────────────────────────────

_CMAP_STOPS = np.array([
    [0,   0,   0  ],
    [0,   0,   255],
    [0,   255, 255],
    [0,   255, 0  ],
    [255, 255, 0  ],
    [255, 0,   0  ],
], dtype=np.float32)
_CMAP_POS = np.array([0.0, 0.25, 0.50, 0.625, 0.75, 1.0], dtype=np.float32)


def _scores_to_rgb(scores_1d: torch.Tensor, n_h: int, n_w: int) -> np.ndarray:
    """patch 后验 → (n_h, n_w, 3) uint8 RGB 热图。"""
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

    px_px = int(pred_xy[0] * W)
    py_px = int(pred_xy[1] * H)
    dot_color = (0, 220, 0) if hit else ((255, 165, 0) if overlap else (220, 0, 0))
    r = max(4, min(W, H) // 60)
    draw.ellipse([px_px - r, py_px - r, px_px + r, py_px + r],
                 fill=dot_color, outline=(255, 255, 255), width=1)
    draw.line([(px_px - r * 2, py_px), (px_px + r * 2, py_px)], fill=dot_color, width=1)
    draw.line([(px_px, py_px - r * 2), (px_px, py_px + r * 2)], fill=dot_color, width=1)

    strip_h = max(18, H // 20)
    strip   = Image.new("RGB", (W, strip_h), color=(20, 20, 20))
    sdraw   = ImageDraw.Draw(strip)
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
