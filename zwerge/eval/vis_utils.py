"""
ZwerGe-UI Visualization Utilities
===================================
Rendering helpers for layer-wise grounding visualization.

Layout:
  - Multi-row grid: cells arranged in rows of COLS_PER_ROW (default 4)
  - Each cell: heatmap + gt_bbox dashed border + prediction dot
  - Below each cell: a styled label strip with layer index and hit/miss status
  - Below the grid: omega bar chart (layer fusion weights)
  - Below omega bar: info panel (instruction + meta + hit status badge)
"""

import math
import textwrap
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from inference_base import scores_to_point_and_topk, point_in_bbox, do_boxes_overlap


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

COLS_PER_ROW  = 4       # cells per row
CELL_W        = 320     # default cell width (px)
CELL_H        = 240     # default cell content height (px)
LABEL_H       = 28      # label strip height below each cell
OMEGA_H       = 72      # omega bar chart height
INFO_H        = 80      # info panel height
BG_COLOR      = (18, 18, 28)   # dark navy background
GRID_PAD      = 6              # padding between cells (px)

# Colors
COL_HIT     = (60,  210, 100)   # green
COL_NEAR    = (255, 180,  50)   # amber
COL_MISS    = (220,  60,  60)   # red
COL_GT_BOX  = (60,  220,  60)   # dashed GT bbox
COL_DOT_OUT = (255, 255, 255)   # dot outline

# ─────────────────────────────────────────────────────────────────────────────
# Colormap: cold (black→blue) → warm (cyan→green→yellow→red)
# ─────────────────────────────────────────────────────────────────────────────

_CMAP_STOPS = np.array([
    [  0,   0,   0],
    [  0,   0, 255],
    [  0, 255, 255],
    [  0, 255,   0],
    [255, 255,   0],
    [255,   0,   0],
], dtype=np.float32)
_CMAP_POS = np.array([0.0, 0.20, 0.45, 0.60, 0.78, 1.0], dtype=np.float32)


def _scores_to_rgb(scores_1d: torch.Tensor, n_h: int, n_w: int) -> np.ndarray:
    """patch posterior → (n_h, n_w, 3) uint8 RGB heatmap."""
    s = scores_1d.float().cpu().numpy()
    s_min, s_max = s.min(), s.max()
    s_norm = np.zeros_like(s) if s_max - s_min < 1e-9 else (s - s_min) / (s_max - s_min)
    flat = s_norm.reshape(-1)
    rgb = np.zeros((len(flat), 3), dtype=np.float32)
    for i in range(len(_CMAP_POS) - 1):
        lo, hi = _CMAP_POS[i], _CMAP_POS[i + 1]
        mask = (flat >= lo) & (flat <= hi)
        if not mask.any():
            continue
        t = (flat[mask] - lo) / (hi - lo + 1e-9)
        c0, c1 = _CMAP_STOPS[i], _CMAP_STOPS[i + 1]
        rgb[mask] = c0[None] * (1 - t[:, None]) + c1[None] * t[:, None]
    return rgb.reshape(n_h, n_w, 3).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Font loader (falls back to PIL default)
# ─────────────────────────────────────────────────────────────────────────────

_FONT_CACHE: Dict[int, ImageFont.FreeTypeFont] = {}

def _font(size: int) -> ImageFont.FreeTypeFont:
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
    ]
    for path in candidates:
        try:
            f = ImageFont.truetype(path, size=size)
            _FONT_CACHE[size] = f
            return f
        except Exception:
            pass
    f = ImageFont.load_default()
    _FONT_CACHE[size] = f
    return f


# ─────────────────────────────────────────────────────────────────────────────
# Draw dashed rectangle
# ─────────────────────────────────────────────────────────────────────────────

def _draw_dashed_rect(
    draw: ImageDraw.ImageDraw,
    x1: int, y1: int, x2: int, y2: int,
    color: Tuple[int, int, int],
    dash: int = 6, width: int = 2,
) -> None:
    """Draw a dashed rectangle outline."""
    for x in range(x1, x2, dash * 2):
        draw.line([(x, y1), (min(x + dash, x2), y1)], fill=color, width=width)
        draw.line([(x, y2), (min(x + dash, x2), y2)], fill=color, width=width)
    for y in range(y1, y2, dash * 2):
        draw.line([(x1, y), (x1, min(y + dash, y2))], fill=color, width=width)
        draw.line([(x2, y), (x2, min(y + dash, y2))], fill=color, width=width)


# ─────────────────────────────────────────────────────────────────────────────
# Single cell renderer
# ─────────────────────────────────────────────────────────────────────────────

def render_one_cell(
    orig_img: Image.Image,
    scores_1d: torch.Tensor,
    n_w: int,
    n_h: int,
    gt_bbox_norm: Tuple[float, float, float, float],
    pred_xy: Tuple[float, float],
    hit: bool,
    overlap: bool,
    layer_label: str,         # e.g. "L18" or "Fusion"
    alpha: float = 0.55,
    cell_w: int = CELL_W,
    cell_h: int = CELL_H,
    label_h: int = LABEL_H,
) -> Image.Image:
    """Render one heatmap cell with GT bbox, prediction dot, and label strip."""
    W, H = cell_w, cell_h

    # Heatmap blend
    img_resized = orig_img.convert("RGB").resize(
        (W, H), Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
    )
    patch_rgb = _scores_to_rgb(scores_1d, n_h, n_w)
    hm_pil = Image.fromarray(patch_rgb, "RGB").resize(
        (W, H), Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
    )
    blend_arr = (
        (1 - alpha) * np.array(img_resized, dtype=np.float32) +
        alpha       * np.array(hm_pil,      dtype=np.float32)
    ).clip(0, 255).astype(np.uint8)
    result = Image.fromarray(blend_arr, "RGB")
    draw   = ImageDraw.Draw(result)

    # GT bbox dashed border
    x1n, y1n, x2n, y2n = gt_bbox_norm
    bx1, by1 = max(1, int(x1n * W)), max(1, int(y1n * H))
    bx2, by2 = min(W - 1, int(x2n * W)), min(H - 1, int(y2n * H))
    _draw_dashed_rect(draw, bx1, by1, bx2, by2, COL_GT_BOX, dash=5, width=2)

    # Prediction dot + crosshair
    px_px = int(pred_xy[0] * W)
    py_px = int(pred_xy[1] * H)
    dot_color = COL_HIT if hit else (COL_NEAR if overlap else COL_MISS)
    r = max(5, min(W, H) // 55)
    draw.ellipse([px_px - r, py_px - r, px_px + r, py_px + r],
                 fill=dot_color, outline=COL_DOT_OUT, width=1)
    draw.line([(px_px - r * 2, py_px), (px_px + r * 2, py_px)], fill=dot_color, width=1)
    draw.line([(px_px, py_px - r * 2), (px_px, py_px + r * 2)], fill=dot_color, width=1)

    # ── Label strip (below cell image) ──
    strip = Image.new("RGB", (W, label_h), (25, 25, 38))
    sdraw = ImageDraw.Draw(strip)

    status_txt = "✓ HIT" if hit else ("≈ NEAR" if overlap else "✗ MISS")
    txt_color  = dot_color
    font_lbl   = _font(max(11, label_h - 10))
    font_stat  = _font(max(11, label_h - 8))

    # Left: layer label  |  Right: status
    sdraw.text((6, (label_h - 14) // 2), layer_label, fill=(200, 210, 255), font=font_lbl)
    stat_w = sdraw.textlength(status_txt, font=font_stat) if hasattr(sdraw, "textlength") else 60
    sdraw.text((W - int(stat_w) - 6, (label_h - 14) // 2), status_txt,
               fill=txt_color, font=font_stat)

    # Thin colored bar along top edge of strip
    sdraw.line([(0, 0), (W - 1, 0)], fill=dot_color, width=2)

    combined = Image.new("RGB", (W, H + label_h), BG_COLOR)
    combined.paste(result, (0, 0))
    combined.paste(strip,  (0, H))
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# Omega bar chart
# ─────────────────────────────────────────────────────────────────────────────

def render_omega_bar(
    omega: torch.Tensor,
    layer_indices: List[int],
    total_width: int,
    bar_h: int = OMEGA_H,
) -> Image.Image:
    """Render layer fusion weight omega as a colored bar chart."""
    n  = len(omega)
    om = omega.float().cpu().numpy()
    img  = Image.new("RGB", (total_width, bar_h), (12, 12, 22))
    draw = ImageDraw.Draw(img)

    bar_w = max(1, (total_width - 2 * GRID_PAD) // max(n, 1))
    font_sm = _font(max(9, bar_h // 7))
    font_md = _font(max(11, bar_h // 5))
    max_inner_h = bar_h - 26

    for i, (w, li) in enumerate(zip(om, layer_indices)):
        x0 = GRID_PAD + i * bar_w
        # Blue→Red gradient based on weight magnitude (relative to uniform 1/n)
        rel = w * n   # rel=1 → uniform, rel>1 → above uniform
        red  = int(min(255, rel * 140))
        blue = int(max(30, 200 - rel * 100))
        bar_color = (red, 80, blue)

        inner_h = max(3, min(max_inner_h, int(w * max_inner_h * n)))
        y_bot = bar_h - 18
        y_top = y_bot - inner_h
        draw.rectangle([x0 + 1, y_top, x0 + bar_w - 2, y_bot], fill=bar_color)

        # Layer label at bottom
        draw.text((x0 + 2, y_bot + 2), f"L{li}", fill=(160, 170, 200), font=font_sm)
        # Weight value above bar
        if inner_h > 10:
            draw.text((x0 + 2, max(2, y_top - 13)), f"{w:.2f}", fill=(220, 225, 240), font=font_sm)

    # Title
    title = "Layer Fusion Weights (ω)"
    draw.text((total_width // 2 - 80, 2), title, fill=(140, 150, 190), font=font_md)
    return img


# ─────────────────────────────────────────────────────────────────────────────
# Info panel (instruction + metadata + result badge)
# ─────────────────────────────────────────────────────────────────────────────

def render_info_panel(
    instruction: str,
    meta: Dict,
    total_width: int,
    info_h: int = INFO_H,
) -> Image.Image:
    """Render bottom info panel with instruction text, metadata and hit badge."""
    img  = Image.new("RGB", (total_width, info_h), (12, 16, 36))
    draw = ImageDraw.Draw(img)

    fhit = meta.get("fusion_hit1", -1)
    fov  = meta.get("fusion_overlap1", -1)

    # Determine badge text + color
    if fhit == 1:
        badge, badge_col = "● FUSION  HIT ✓", COL_HIT
    elif fhit == 0 and fov == 1:
        badge, badge_col = "● FUSION  NEAR MISS ≈", COL_NEAR
    else:
        badge, badge_col = "● FUSION  FAR MISS ✗", COL_MISS

    # Draw left accent bar
    draw.rectangle([0, 0, 3, info_h - 1], fill=badge_col)

    font_meta  = _font(12)
    font_instr = _font(13)
    font_badge = _font(14)

    # Row 1: meta tags
    meta_order = ["bench", "group", "platform", "ui_type", "data_type",
                  "grounding_type", "task_type", "GUI_types"]
    meta_parts = []
    for k in meta_order:
        v = meta.get(k)
        if v:
            meta_parts.append(f"{k}={v}")
    meta_str = "  │  ".join(meta_parts)
    draw.text((10, 5), meta_str, fill=(120, 150, 210), font=font_meta)

    # Row 2: instruction (truncated, wrapped at ~100 chars per line, max 2 lines)
    instr_avail_w = total_width - 220
    max_chars = max(40, instr_avail_w // 7)
    lines = textwrap.wrap(instruction, width=max_chars)[:2]
    instr_display = "\n".join(lines)
    if len(textwrap.wrap(instruction, width=max_chars)) > 2:
        instr_display = instr_display.rstrip() + " …"
    draw.text((10, 22), instr_display, fill=(220, 215, 180), font=font_instr)

    # Badge at top-right
    badge_w = draw.textlength(badge, font=font_badge) if hasattr(draw, "textlength") else 180
    badge_x = total_width - int(badge_w) - 14
    draw.text((badge_x, 5), badge, fill=badge_col, font=font_badge)

    # Thin top border
    draw.line([(0, 0), (total_width - 1, 0)], fill=(40, 50, 80), width=1)
    return img


# ─────────────────────────────────────────────────────────────────────────────
# Main entry: visualize_sample
# ─────────────────────────────────────────────────────────────────────────────

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
    cell_w: int = CELL_W,
    cell_h: int = CELL_H,
    alpha: float = 0.55,
    cols_per_row: int = COLS_PER_ROW,
) -> Image.Image:
    """
    Produce a multi-row grid visualization for one sample.

    Layout:
      Row(s) of cells (each cell = heatmap + label strip)
        ↓
      Omega bar chart
        ↓
      Info panel (instruction + meta + fusion badge)
    """
    n_w            = pred["n_width"]
    n_h_patches    = pred["n_height"]
    layer_indices  = pred["layer_indices"]
    per_layer_probs = pred["per_layer_probs"]
    p_final = pred["p_final"]
    omega   = pred["omega"]
    phx = 0.5 / n_w
    phy = 0.5 / n_h_patches

    def _judge(px: float, py: float):
        pred_box = (px - phx, py - phy, px + phx, py + phy)
        return bool(point_in_bbox(px, py, gt_bbox_norm)), bool(do_boxes_overlap(pred_box, gt_bbox_norm))

    label_h = LABEL_H
    cell_total_h = cell_h + label_h

    # ── Collect all cells ──
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
        cells.append(render_one_cell(
            orig_img=orig_img, scores_1d=p_l, n_w=n_w, n_h=n_h_patches,
            gt_bbox_norm=gt_bbox_norm, pred_xy=(px, py), hit=hit, overlap=ov,
            layer_label=f"L{layer_idx:02d}", alpha=alpha,
            cell_w=cell_w, cell_h=cell_h, label_h=label_h,
        ))

    # Fusion cell (last)
    f_best, _ = scores_to_point_and_topk(
        p=p_final, n_width=n_w, n_height=n_h_patches,
        activation_threshold=activation_threshold, topk=1,
        decode_strategy=decode_strategy,
        peak_shift_alpha=peak_shift_alpha, temperature=temperature,
    )
    fpx, fpy  = float(f_best[0]), float(f_best[1])
    fhit, fov = _judge(fpx, fpy)
    cells.append(render_one_cell(
        orig_img=orig_img, scores_1d=p_final, n_w=n_w, n_h=n_h_patches,
        gt_bbox_norm=gt_bbox_norm, pred_xy=(fpx, fpy), hit=fhit, overlap=fov,
        layer_label="Fusion", alpha=alpha,
        cell_w=cell_w, cell_h=cell_h, label_h=label_h,
    ))

    n_cells = len(cells)
    n_cols  = min(cols_per_row, n_cells)
    n_rows  = math.ceil(n_cells / n_cols)

    # ── Assemble grid ──
    grid_w = n_cols * cell_w + (n_cols - 1) * GRID_PAD
    grid_h = n_rows * cell_total_h + (n_rows - 1) * GRID_PAD

    grid = Image.new("RGB", (grid_w, grid_h), BG_COLOR)
    for ci, cell in enumerate(cells):
        row = ci // n_cols
        col = ci % n_cols
        x   = col * (cell_w + GRID_PAD)
        y   = row * (cell_total_h + GRID_PAD)
        grid.paste(cell, (x, y))

    # ── Omega bar ──
    omega_bar = render_omega_bar(
        omega=omega, layer_indices=layer_indices,
        total_width=grid_w, bar_h=OMEGA_H,
    )

    # ── Info panel ──
    info_meta = {**meta, "fusion_hit1": int(fhit), "fusion_overlap1": int(fov)}
    info_panel = render_info_panel(
        instruction=instruction, meta=info_meta,
        total_width=grid_w, info_h=INFO_H,
    )

    # ── Compose canvas ──
    total_h = grid_h + GRID_PAD + OMEGA_H + 2 + INFO_H
    canvas  = Image.new("RGB", (grid_w, total_h), BG_COLOR)
    canvas.paste(grid,       (0, 0))
    canvas.paste(omega_bar,  (0, grid_h + GRID_PAD))
    canvas.paste(info_panel, (0, grid_h + GRID_PAD + OMEGA_H + 2))
    return canvas
