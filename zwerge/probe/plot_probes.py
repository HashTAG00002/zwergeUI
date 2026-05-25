"""
plot_probes.py — Unified figure generator for Exp 3 and Exp 4.

Figure 3 (Exp 3): Dual-axis crossing plot (4 panels, one per model)
  Left Y  : spatial hit@1 (%) — solid blue
  Right Y : mean logit-lens NLL of coord tokens (inverted: lower NLL = better) — dashed orange
  Shaded bands: mid-layer spatial zone (light blue) / late serialization zone (light red)

Figure 4 (Exp 4): Switch-score plot (2 panels per model pair + optional heatmap)
  Mean switch_score per probe layer
  Mean old_target_suppression per probe layer (secondary line)

Usage:
  python plot_probes.py \\
      --exp3_dir outputs/exp3 \\
      --exp4_dir outputs/exp4 \\
      --output_dir outputs/figures

Outputs (PDF + PNG at 300 dpi):
  figures/fig3_serialization_lens.pdf
  figures/fig4_switch_score.pdf
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Matplotlib: non-interactive backend for server environments
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# ── Style constants ────────────────────────────────────────────────────────────

_BLUE       = "#2166ac"
_ORANGE     = "#d6604d"
_BLUE_ALPHA = "#2166ac33"
_RED_ALPHA  = "#d6604d22"
_GREY       = "#666666"

_MODEL_LABELS = {
    "uitars":   "UI-TARS-7B",
    "guiowl7b": "GUI-Owl-7B",
    "guiowl":   "GUI-Owl-72B",
    "uivenus":  "UI-Venus-7B",
    "uitars1":  "UI-TARS-1.5-7B",
}

_FIG_PARAMS = {
    "font.family":   "DejaVu Sans",
    "font.size":     9,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.framealpha": 0.85,
    "figure.dpi": 150,
}


# ── JSONL reader ──────────────────────────────────────────────────────────────

def read_jsonl(path: str) -> List[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ── Exp 3 aggregation ─────────────────────────────────────────────────────────

def aggregate_exp3(records: List[dict]) -> dict:
    """Aggregate per-sample results into mean curves over probe layers."""
    if not records:
        return {}

    layers = records[0]["probe_layers"]
    n_layers = len(layers)

    hit1_mat  = []
    mass_mat  = []
    cnll_mat  = []

    for rec in records:
        if len(rec.get("spatial_hit1", [])) != n_layers:
            continue
        if len(rec.get("coord_nll", [])) != n_layers:
            continue
        # Skip samples where any coord_nll is None
        cnll = rec["coord_nll"]
        if any(v is None for v in cnll):
            continue

        hit1_mat.append([float(v) for v in rec["spatial_hit1"]])
        mass_mat.append([float(v) for v in rec["spatial_mass"]])
        cnll_mat.append([float(v) for v in cnll])

    if not hit1_mat:
        return {}

    hit1_arr = np.array(hit1_mat)   # [N, L]
    mass_arr = np.array(mass_mat)
    cnll_arr = np.array(cnll_mat)

    return {
        "layers":          layers,
        "mean_hit1":       hit1_arr.mean(axis=0) * 100.0,   # → percentage
        "sem_hit1":        hit1_arr.std(axis=0) / np.sqrt(len(hit1_arr)) * 100.0,
        "mean_mass":       mass_arr.mean(axis=0),
        "mean_coord_nll":  cnll_arr.mean(axis=0),
        "sem_coord_nll":   cnll_arr.std(axis=0) / np.sqrt(len(cnll_arr)),
        "n_samples":       len(hit1_mat),
    }


# ── Exp 4 aggregation ─────────────────────────────────────────────────────────

def aggregate_exp4(records: List[dict]) -> dict:
    if not records:
        return {}

    layers = records[0]["probe_layers"]
    n_layers = len(layers)

    sw_mat   = []
    supp_mat = []

    for rec in records:
        if len(rec.get("switch_score", [])) != n_layers:
            continue
        sw_mat.append([float(v) for v in rec["switch_score"]])
        supp_mat.append([float(v) for v in rec["old_target_suppression"]])

    if not sw_mat:
        return {}

    sw_arr   = np.array(sw_mat)
    supp_arr = np.array(supp_mat)

    return {
        "layers":           layers,
        "mean_switch":      sw_arr.mean(axis=0),
        "sem_switch":       sw_arr.std(axis=0) / np.sqrt(len(sw_arr)),
        "mean_suppression": supp_arr.mean(axis=0),
        "sem_suppression":  supp_arr.std(axis=0) / np.sqrt(len(supp_arr)),
        "n_pairs":          len(sw_mat),
    }


# ── Figure 3: Crossing plot ───────────────────────────────────────────────────

def _find_peak(arr: np.ndarray) -> int:
    return int(np.argmax(arr))


def _find_min(arr: np.ndarray) -> int:
    return int(np.argmin(arr))


def plot_fig3(
    exp3_data: Dict[str, dict],   # model_type → aggregated dict
    output_dir: str,
):
    models = [m for m in ["uitars", "guiowl7b", "guiowl", "uivenus"] if m in exp3_data]
    if not models:
        print("[plot] No exp3 data found, skipping Fig 3.")
        return

    plt.rcParams.update(_FIG_PARAMS)
    fig, axes = plt.subplots(1, len(models), figsize=(3.2 * len(models), 3.2))
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        d = exp3_data[model]
        layers = d["layers"]
        xs     = np.arange(len(layers))
        tick_labels = [str(l) for l in layers]

        hit1  = d["mean_hit1"]
        cnll  = d["mean_coord_nll"]
        n     = d["n_samples"]

        peak_spatial = _find_peak(hit1)
        peak_serial  = _find_min(cnll)   # lower NLL = better serialization

        # ── Shaded regions ────────────────────────────────────────────────────
        # Spatial zone: from first non-zero to peak
        first_nz = next((i for i, v in enumerate(hit1) if v > 1.0), 0)
        ax.axvspan(first_nz - 0.5, peak_spatial + 0.5,
                   color=_BLUE_ALPHA, zorder=0, label="_spatial_zone")
        # Serialization zone: from peak_serial to end
        ax.axvspan(peak_serial - 0.5, xs[-1] + 0.5,
                   color=_RED_ALPHA, zorder=0, label="_serial_zone")

        # ── Left axis: hit@1 ──────────────────────────────────────────────────
        ax.set_xlabel("Probe layer", fontsize=8)
        ax.set_ylabel("Spatial hit@1 (%)", color=_BLUE, fontsize=8)
        ax.tick_params(axis="y", labelcolor=_BLUE)

        line_hit, = ax.plot(xs, hit1, color=_BLUE, lw=1.8, label="Spatial hit@1")
        ax.fill_between(xs, hit1 - d["sem_hit1"], hit1 + d["sem_hit1"],
                        color=_BLUE, alpha=0.15)
        ax.plot(xs[peak_spatial], hit1[peak_spatial], "*", color=_BLUE,
                ms=10, zorder=5)

        # ── Right axis: coord NLL ─────────────────────────────────────────────
        ax2 = ax.twinx()
        ax2.set_ylabel("Coord token NLL ↓", color=_ORANGE, fontsize=8)
        ax2.tick_params(axis="y", labelcolor=_ORANGE)

        line_nll, = ax2.plot(xs, cnll, color=_ORANGE, lw=1.8,
                             linestyle="--", label="Coord NLL")
        ax2.fill_between(xs, cnll - d["sem_coord_nll"], cnll + d["sem_coord_nll"],
                         color=_ORANGE, alpha=0.15)
        ax2.plot(xs[peak_serial], cnll[peak_serial], "^", color=_ORANGE,
                 ms=8, zorder=5)

        # ── X ticks (sparse for readability) ──────────────────────────────────
        step = max(1, len(layers) // 8)
        ax.set_xticks(xs[::step])
        ax.set_xticklabels(tick_labels[::step], fontsize=7)

        ax.set_title(_MODEL_LABELS.get(model, model) + f"\n(n={n})", fontsize=9)

    # ── Shared legend ─────────────────────────────────────────────────────────
    legend_elements = [
        Line2D([0], [0], color=_BLUE,   lw=1.8,              label="Spatial hit@1"),
        Line2D([0], [0], color=_ORANGE, lw=1.8, ls="--",     label="Coord token NLL"),
        Line2D([0], [0], color=_BLUE,   marker="*", ms=8, lw=0, label="Spatial peak"),
        Line2D([0], [0], color=_ORANGE, marker="^", ms=7, lw=0, label="NLL plateau"),
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Spatial Grounding Peaks Before Coordinate Serialization",
                 fontsize=10, y=1.02)
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        out = os.path.join(output_dir, f"fig3_serialization_lens.{ext}")
        fig.savefig(out, bbox_inches="tight", dpi=300)
        print(f"[plot] Saved {out}")
    plt.close(fig)


# ── Figure 4: Switch-score plot ───────────────────────────────────────────────

def plot_fig4(
    exp4_data: Dict[str, dict],
    output_dir: str,
):
    models = [m for m in ["uitars", "guiowl7b", "guiowl", "uivenus"] if m in exp4_data]
    if not models:
        print("[plot] No exp4 data found, skipping Fig 4.")
        return

    plt.rcParams.update(_FIG_PARAMS)
    fig, axes = plt.subplots(1, len(models), figsize=(3.2 * len(models), 3.2))
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        d      = exp4_data[model]
        layers = d["layers"]
        xs     = np.arange(len(layers))
        tick_labels = [str(l) for l in layers]
        n      = d["n_pairs"]

        sw   = d["mean_switch"]
        supp = d["mean_suppression"]

        peak_sw = _find_peak(sw)

        # Zero reference
        ax.axhline(0, color=_GREY, lw=0.8, ls=":")

        # Highlight mid-layer peak region (±2 layers around peak)
        lo = max(0, peak_sw - 2)
        hi = min(len(xs) - 1, peak_sw + 2)
        ax.axvspan(lo - 0.5, hi + 0.5, color=_BLUE_ALPHA, zorder=0)

        line_sw, = ax.plot(xs, sw, color=_BLUE, lw=1.8,
                           label="Switch score (↑ target shift)")
        ax.fill_between(xs, sw - d["sem_switch"], sw + d["sem_switch"],
                        color=_BLUE, alpha=0.15)
        ax.plot(xs[peak_sw], sw[peak_sw], "*", color=_BLUE, ms=10, zorder=5)

        line_supp, = ax.plot(xs, supp, color=_ORANGE, lw=1.8, ls="--",
                             label="Old-target suppression")
        ax.fill_between(xs, supp - d["sem_suppression"], supp + d["sem_suppression"],
                        color=_ORANGE, alpha=0.15)

        step = max(1, len(layers) // 8)
        ax.set_xticks(xs[::step])
        ax.set_xticklabels(tick_labels[::step], fontsize=7)
        ax.set_xlabel("Probe layer", fontsize=8)
        ax.set_ylabel("Posterior mass delta", fontsize=8)
        ax.set_title(_MODEL_LABELS.get(model, model) + f"\n(n={n} pairs)", fontsize=9)

    legend_elements = [
        Line2D([0], [0], color=_BLUE,   lw=1.8,          label="Switch score"),
        Line2D([0], [0], color=_ORANGE, lw=1.8, ls="--", label="Old-target suppression"),
        Line2D([0], [0], color=_BLUE,   marker="*", ms=8, lw=0, label="Switch peak"),
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Mid-Layer Posterior Responds to Instruction Switch",
                 fontsize=10, y=1.02)
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        out = os.path.join(output_dir, f"fig4_switch_score.{ext}")
        fig.savefig(out, bbox_inches="tight", dpi=300)
        print(f"[plot] Saved {out}")
    plt.close(fig)


# ── Combined Fig 5: single-page summary (optional) ───────────────────────────

def plot_fig5_combined(
    exp3_data: Dict[str, dict],
    exp4_data: Dict[str, dict],
    output_dir: str,
):
    """2-row figure: top row = Exp3 crossing plots, bottom row = Exp4 switch scores."""
    models3 = [m for m in ["uitars", "guiowl7b", "guiowl", "uivenus"] if m in exp3_data]
    models4 = [m for m in ["uitars", "guiowl7b", "guiowl", "uivenus"] if m in exp4_data]
    n_cols = max(len(models3), len(models4), 1)

    plt.rcParams.update(_FIG_PARAMS)
    fig = plt.figure(figsize=(3.2 * n_cols, 6.8))
    gs  = gridspec.GridSpec(2, n_cols, hspace=0.55, wspace=0.45)

    # ── Row 0: Exp 3 ──────────────────────────────────────────────────────────
    for col, model in enumerate(models3):
        ax  = fig.add_subplot(gs[0, col])
        ax2 = ax.twinx()
        d   = exp3_data[model]
        xs  = np.arange(len(d["layers"]))
        step = max(1, len(d["layers"]) // 7)

        hit1 = d["mean_hit1"]
        cnll = d["mean_coord_nll"]
        pk_sp = _find_peak(hit1)
        pk_se = _find_min(cnll)

        first_nz = next((i for i, v in enumerate(hit1) if v > 1.0), 0)
        ax.axvspan(first_nz - 0.5, pk_sp + 0.5,  color=_BLUE_ALPHA, zorder=0)
        ax.axvspan(pk_se - 0.5,    xs[-1] + 0.5, color=_RED_ALPHA,  zorder=0)

        ax.plot(xs, hit1, color=_BLUE, lw=1.6)
        ax.plot(xs[pk_sp], hit1[pk_sp], "*", color=_BLUE, ms=9, zorder=5)
        ax.fill_between(xs, hit1 - d["sem_hit1"], hit1 + d["sem_hit1"],
                        color=_BLUE, alpha=0.15)

        ax2.plot(xs, cnll, color=_ORANGE, lw=1.6, ls="--")
        ax2.plot(xs[pk_se], cnll[pk_se], "^", color=_ORANGE, ms=7, zorder=5)
        ax2.fill_between(xs, cnll - d["sem_coord_nll"], cnll + d["sem_coord_nll"],
                         color=_ORANGE, alpha=0.15)

        ax.set_xticks(xs[::step])
        ax.set_xticklabels([str(d["layers"][i]) for i in range(0, len(d["layers"]), step)],
                           fontsize=6)
        ax.tick_params(axis="y", labelcolor=_BLUE, labelsize=7)
        ax2.tick_params(axis="y", labelcolor=_ORANGE, labelsize=7)
        ax.set_title(_MODEL_LABELS.get(model, model), fontsize=8, pad=4)
        if col == 0:
            ax.set_ylabel("Hit@1 (%)", color=_BLUE, fontsize=7)
        if col == len(models3) - 1:
            ax2.set_ylabel("Coord NLL ↓", color=_ORANGE, fontsize=7)

    # ── Row 1: Exp 4 ──────────────────────────────────────────────────────────
    for col, model in enumerate(models4):
        ax = fig.add_subplot(gs[1, col])
        d  = exp4_data[model]
        xs = np.arange(len(d["layers"]))
        step = max(1, len(d["layers"]) // 7)

        sw   = d["mean_switch"]
        supp = d["mean_suppression"]
        pk   = _find_peak(sw)

        ax.axhline(0, color=_GREY, lw=0.7, ls=":")
        lo, hi = max(0, pk-2), min(len(xs)-1, pk+2)
        ax.axvspan(lo - 0.5, hi + 0.5, color=_BLUE_ALPHA, zorder=0)

        ax.plot(xs, sw,   color=_BLUE,   lw=1.6)
        ax.plot(xs, supp, color=_ORANGE, lw=1.6, ls="--")
        ax.plot(xs[pk], sw[pk], "*", color=_BLUE, ms=9, zorder=5)
        ax.fill_between(xs, sw - d["sem_switch"], sw + d["sem_switch"],
                        color=_BLUE, alpha=0.15)

        ax.set_xticks(xs[::step])
        ax.set_xticklabels([str(d["layers"][i]) for i in range(0, len(d["layers"]), step)],
                           fontsize=6)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("Layer", fontsize=7)
        if col == 0:
            ax.set_ylabel("Mass delta", fontsize=7)

    # ── Row labels ────────────────────────────────────────────────────────────
    fig.text(0.01, 0.75, "(a) Exp 3: Serialization Lens", va="center",
             rotation="vertical", fontsize=8, fontweight="bold")
    fig.text(0.01, 0.27, "(b) Exp 4: Counterfactual Switch", va="center",
             rotation="vertical", fontsize=8, fontweight="bold")

    os.makedirs(output_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        out = os.path.join(output_dir, f"fig5_combined.{ext}")
        fig.savefig(out, bbox_inches="tight", dpi=300)
        print(f"[plot] Saved {out}")
    plt.close(fig)


# ── Data discovery ────────────────────────────────────────────────────────────

_EXP3_SUFFIX = "_lens.jsonl"
_EXP4_SUFFIX = "_cf.jsonl"


def discover_model_files(directory: str, suffix: str) -> Dict[str, str]:
    """Find model_type → file_path mapping from a directory."""
    result = {}
    if not os.path.isdir(directory):
        return result
    for fname in os.listdir(directory):
        if fname.endswith(suffix):
            model = fname[:-len(suffix)]
            result[model] = os.path.join(directory, fname)
    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Plot probe experiment figures")
    p.add_argument("--exp3_dir",    default="outputs/exp3",
                   help="Directory containing exp3 *_lens.jsonl files")
    p.add_argument("--exp4_dir",    default="outputs/exp4",
                   help="Directory containing exp4 *_cf.jsonl files")
    p.add_argument("--output_dir",  default="outputs/figures")
    p.add_argument("--combined",    action="store_true",
                   help="Also generate combined Fig 5")
    # Allow explicit file paths for single-model runs
    p.add_argument("--exp3_file",   default=None,
                   help="Single exp3 .jsonl file (overrides --exp3_dir discovery)")
    p.add_argument("--exp4_file",   default=None,
                   help="Single exp4 .jsonl file (overrides --exp4_dir discovery)")
    p.add_argument("--model_type",  default=None,
                   help="Model type label when using --exp3_file or --exp4_file")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Load exp3 ──────────────────────────────────────────────────────────────
    exp3_data: Dict[str, dict] = {}
    if args.exp3_file and args.model_type:
        records = read_jsonl(args.exp3_file)
        agg = aggregate_exp3(records)
        if agg:
            exp3_data[args.model_type] = agg
            print(f"[plot] exp3 {args.model_type}: {agg['n_samples']} samples")
    else:
        for model, fpath in discover_model_files(args.exp3_dir, _EXP3_SUFFIX).items():
            records = read_jsonl(fpath)
            agg = aggregate_exp3(records)
            if agg:
                exp3_data[model] = agg
                print(f"[plot] exp3 {model}: {agg['n_samples']} samples")

    # ── Load exp4 ──────────────────────────────────────────────────────────────
    exp4_data: Dict[str, dict] = {}
    if args.exp4_file and args.model_type:
        records = read_jsonl(args.exp4_file)
        agg = aggregate_exp4(records)
        if agg:
            exp4_data[args.model_type] = agg
            print(f"[plot] exp4 {args.model_type}: {agg['n_pairs']} pairs")
    else:
        for model, fpath in discover_model_files(args.exp4_dir, _EXP4_SUFFIX).items():
            records = read_jsonl(fpath)
            agg = aggregate_exp4(records)
            if agg:
                exp4_data[model] = agg
                print(f"[plot] exp4 {model}: {agg['n_pairs']} pairs")

    if not exp3_data and not exp4_data:
        print("[plot] No data found. Check --exp3_dir and --exp4_dir paths.")
        return

    # ── Generate figures ───────────────────────────────────────────────────────
    if exp3_data:
        plot_fig3(exp3_data, args.output_dir)
    if exp4_data:
        plot_fig4(exp4_data, args.output_dir)
    if args.combined and exp3_data and exp4_data:
        plot_fig5_combined(exp3_data, exp4_data, args.output_dir)


if __name__ == "__main__":
    main()
