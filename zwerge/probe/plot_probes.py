"""
plot_probes.py — Unified figure generator for Exp 3 and Exp 4.

Outputs
-------
figures/fig3_serialization_lens.{pdf,png}
    Dual-axis crossing plot (4 panels).  Left Y = spatial hit@1 (%),
    Right Y = coord-token NLL (↓ = better serialization).

figures/fig_exp3_coord_protocol_split.{pdf,png}
    Separate panels showing coord NLL vs. protocol-token NLL per layer.
    Supports "late layers are protocol/serialization layers" claim.

figures/fig4_instruction_suppression.{pdf,png}
    Primary Exp4 figure: old_target_suppression + posterior_js per layer.
    Switch-score only plotted when exact_image pairs are available.

figures/fig5_combined.{pdf,png}   (--combined flag)
    2-row summary: Exp3 crossing (top) + Exp4 suppression (bottom).

tables/tab_exp3_peak_lag.tex
    Per-model table: spatial-peak layer, NLL-plateau layer, lag (plateau − peak).

tables/tab_exp4_instruction_sensitivity.tex
    Per-model table: peak suppression, peak JS, peak layer.

Usage:
  python plot_probes.py \\
      --exp3_dir outputs/exp3 \\
      --exp4_dir outputs/exp4 \\
      --output_dir outputs/figures
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# ── Style ──────────────────────────────────────────────────────────────────────

_BLUE       = "#2166ac"
_ORANGE     = "#d6604d"
_GREEN      = "#4dac26"
_BLUE_ALPHA = "#2166ac33"
_RED_ALPHA  = "#d6604d22"
_GREEN_ALPHA = "#4dac2622"
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

_MODEL_ORDER = ["uitars", "guiowl7b", "guiowl", "uivenus"]


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
    if not records:
        return {}
    layers   = records[0]["probe_layers"]
    n_layers = len(layers)

    hit1_mat  = []
    mass_mat  = []
    cnll_mat  = []
    pnll_mat  = []   # protocol NLL (may be None)

    for rec in records:
        if len(rec.get("spatial_hit1", [])) != n_layers:
            continue
        cnll = rec.get("coord_nll", [])
        if len(cnll) != n_layers or any(v is None for v in cnll):
            continue
        hit1_mat.append([float(v) for v in rec["spatial_hit1"]])
        mass_mat.append([float(v) for v in rec["spatial_mass"]])
        cnll_mat.append([float(v) for v in cnll])
        pnll = rec.get("protocol_nll", [])
        if len(pnll) == n_layers and all(v is not None for v in pnll):
            pnll_mat.append([float(v) for v in pnll])

    if not hit1_mat:
        return {}

    hit1_arr = np.array(hit1_mat)
    mass_arr = np.array(mass_mat)
    cnll_arr = np.array(cnll_mat)

    out = {
        "layers":         layers,
        "mean_hit1":      hit1_arr.mean(axis=0) * 100.0,
        "sem_hit1":       hit1_arr.std(axis=0) / np.sqrt(len(hit1_arr)) * 100.0,
        "mean_mass":      mass_arr.mean(axis=0),
        "mean_coord_nll": cnll_arr.mean(axis=0),
        "sem_coord_nll":  cnll_arr.std(axis=0) / np.sqrt(len(cnll_arr)),
        "n_samples":      len(hit1_mat),
    }
    if pnll_mat:
        pnll_arr = np.array(pnll_mat)
        out["mean_protocol_nll"] = pnll_arr.mean(axis=0)
        out["sem_protocol_nll"]  = pnll_arr.std(axis=0) / np.sqrt(len(pnll_arr))
        out["n_protocol"]        = len(pnll_mat)
    return out


# ── Exp 4 aggregation ─────────────────────────────────────────────────────────

def aggregate_exp4(records: List[dict]) -> dict:
    if not records:
        return {}
    layers   = records[0]["probe_layers"]
    n_layers = len(layers)

    supp_mat = []
    js_mat   = []
    sw_mat   = []
    sw_exact = []   # switch_score only for exact_image pairs

    for rec in records:
        supp = rec.get("old_target_suppression", [])
        js   = rec.get("posterior_js", [])
        sw   = rec.get("switch_score", [])
        if len(supp) != n_layers:
            continue
        supp_mat.append([float(v) for v in supp])
        if len(js) == n_layers:
            js_mat.append([float(v) for v in js])
        if len(sw) == n_layers:
            sw_mat.append([float(v) for v in sw])
            if rec.get("pair_type") == "exact_image":
                sw_exact.append([float(v) for v in sw])

    if not supp_mat:
        return {}

    supp_arr = np.array(supp_mat)
    out = {
        "layers":           layers,
        "mean_suppression": supp_arr.mean(axis=0),
        "sem_suppression":  supp_arr.std(axis=0) / np.sqrt(len(supp_arr)),
        "n_pairs":          len(supp_mat),
    }
    if js_mat:
        js_arr = np.array(js_mat)
        out["mean_js"]  = js_arr.mean(axis=0)
        out["sem_js"]   = js_arr.std(axis=0) / np.sqrt(len(js_arr))
    if sw_mat:
        sw_arr = np.array(sw_mat)
        out["mean_switch"] = sw_arr.mean(axis=0)
        out["sem_switch"]  = sw_arr.std(axis=0) / np.sqrt(len(sw_arr))
    if sw_exact:
        se_arr = np.array(sw_exact)
        out["mean_switch_exact"] = se_arr.mean(axis=0)
        out["n_exact"] = len(sw_exact)
    return out


# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_peak(arr: np.ndarray) -> int:
    return int(np.argmax(arr))

def _find_min(arr: np.ndarray) -> int:
    return int(np.argmin(arr))

def _save_fig(fig, output_dir: str, stem: str):
    os.makedirs(output_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        out = os.path.join(output_dir, f"{stem}.{ext}")
        fig.savefig(out, bbox_inches="tight", dpi=300)
        print(f"[plot] Saved {out}")
    plt.close(fig)

def _xticks(ax, layers, xs, max_ticks=8):
    step = max(1, len(layers) // max_ticks)
    ax.set_xticks(xs[::step])
    ax.set_xticklabels([str(layers[i]) for i in range(0, len(layers), step)], fontsize=7)


# ── Figure 3: Dual-axis crossing plot ────────────────────────────────────────

def plot_fig3(exp3_data: Dict[str, dict], output_dir: str):
    models = [m for m in _MODEL_ORDER if m in exp3_data]
    if not models:
        print("[plot] No exp3 data, skipping Fig 3.")
        return

    plt.rcParams.update(_FIG_PARAMS)
    fig, axes = plt.subplots(1, len(models), figsize=(3.2 * len(models), 3.2))
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        d   = exp3_data[model]
        xs  = np.arange(len(d["layers"]))
        hit1 = d["mean_hit1"]
        cnll = d["mean_coord_nll"]

        pk_sp = _find_peak(hit1)
        pk_se = _find_min(cnll)
        first_nz = next((i for i, v in enumerate(hit1) if v > 1.0), 0)

        ax.axvspan(first_nz - 0.5, pk_sp + 0.5, color=_BLUE_ALPHA, zorder=0)
        ax.axvspan(pk_se - 0.5, xs[-1] + 0.5,   color=_RED_ALPHA,  zorder=0)

        ax.set_ylabel("Spatial hit@1 (%)", color=_BLUE, fontsize=8)
        ax.tick_params(axis="y", labelcolor=_BLUE)
        ax.plot(xs, hit1, color=_BLUE, lw=1.8, label="Spatial hit@1")
        ax.fill_between(xs, hit1 - d["sem_hit1"], hit1 + d["sem_hit1"],
                        color=_BLUE, alpha=0.15)
        ax.plot(xs[pk_sp], hit1[pk_sp], "*", color=_BLUE, ms=10, zorder=5)

        ax2 = ax.twinx()
        ax2.set_ylabel("Coord NLL ↓", color=_ORANGE, fontsize=8)
        ax2.tick_params(axis="y", labelcolor=_ORANGE)
        ax2.plot(xs, cnll, color=_ORANGE, lw=1.8, ls="--", label="Coord NLL")
        ax2.fill_between(xs, cnll - d["sem_coord_nll"], cnll + d["sem_coord_nll"],
                         color=_ORANGE, alpha=0.15)
        ax2.plot(xs[pk_se], cnll[pk_se], "^", color=_ORANGE, ms=8, zorder=5)

        _xticks(ax, d["layers"], xs)
        ax.set_xlabel("Probe layer", fontsize=8)
        ax.set_title(_MODEL_LABELS.get(model, model) + f"\n(n={d['n_samples']})", fontsize=9)

    legend_elems = [
        Line2D([0], [0], color=_BLUE,   lw=1.8,           label="Spatial hit@1"),
        Line2D([0], [0], color=_ORANGE, lw=1.8, ls="--",  label="Coord token NLL"),
        Line2D([0], [0], color=_BLUE,   marker="*", ms=8, lw=0, label="Spatial peak"),
        Line2D([0], [0], color=_ORANGE, marker="^", ms=7, lw=0, label="NLL plateau"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=4,
               fontsize=8, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Spatial Grounding Peaks Before Coordinate Serialization",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    _save_fig(fig, output_dir, "fig3_serialization_lens")


# ── Figure Exp3 coord-vs-protocol split ──────────────────────────────────────

def plot_exp3_coord_protocol_split(exp3_data: Dict[str, dict], output_dir: str):
    """
    Coord NLL vs. protocol-token NLL per layer for each model.
    If late layers are "serialization layers", both NLLs should be low late —
    but protocol NLL drops faster/earlier than coord NLL, showing the model
    first learns the output format then fills in the coordinate values.
    """
    # Only include models that have protocol_nll data
    models = [m for m in _MODEL_ORDER
              if m in exp3_data and "mean_protocol_nll" in exp3_data[m]]
    if not models:
        print("[plot] No protocol_nll data, skipping coord-protocol split figure.")
        return

    plt.rcParams.update(_FIG_PARAMS)
    fig, axes = plt.subplots(1, len(models), figsize=(3.2 * len(models), 3.0))
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        d  = exp3_data[model]
        xs = np.arange(len(d["layers"]))

        cnll = d["mean_coord_nll"]
        pnll = d["mean_protocol_nll"]

        ax.plot(xs, cnll, color=_ORANGE, lw=1.8, label="Coord tokens")
        ax.fill_between(xs, cnll - d["sem_coord_nll"], cnll + d["sem_coord_nll"],
                        color=_ORANGE, alpha=0.15)
        ax.plot(xs, pnll, color=_GREEN,  lw=1.8, ls="--", label="Protocol tokens")
        ax.fill_between(xs, pnll - d["sem_protocol_nll"], pnll + d["sem_protocol_nll"],
                        color=_GREEN, alpha=0.15)

        # Mark where each NLL reaches 110% of its minimum (plateau onset)
        pk_c = _find_min(cnll)
        pk_p = _find_min(pnll)
        ax.axvline(pk_c, color=_ORANGE, lw=0.8, ls=":", alpha=0.7)
        ax.axvline(pk_p, color=_GREEN,  lw=0.8, ls=":", alpha=0.7)

        _xticks(ax, d["layers"], xs)
        ax.set_xlabel("Probe layer", fontsize=8)
        ax.set_ylabel("NLL ↓", fontsize=8)
        ax.set_title(_MODEL_LABELS.get(model, model), fontsize=9)
        ax.legend(fontsize=7)

    fig.suptitle("Protocol Format vs. Coordinate Value Serialization",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    _save_fig(fig, output_dir, "fig_exp3_coord_protocol_split")


# ── Figure 4: Instruction suppression (primary Exp4 figure) ──────────────────

def plot_fig4_instruction_suppression(exp4_data: Dict[str, dict], output_dir: str):
    """
    Primary Exp4 figure: old_target_suppression + posterior_js per layer.
    Safe for both exact_image and same_app_hard_negative pairs.
    """
    models = [m for m in _MODEL_ORDER if m in exp4_data]
    if not models:
        print("[plot] No exp4 data, skipping Fig 4.")
        return

    plt.rcParams.update(_FIG_PARAMS)
    fig, axes = plt.subplots(1, len(models), figsize=(3.2 * len(models), 3.2))
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        d   = exp4_data[model]
        xs  = np.arange(len(d["layers"]))
        n   = d["n_pairs"]

        supp = d["mean_suppression"]
        pk   = _find_peak(supp)

        ax.axhline(0, color=_GREY, lw=0.8, ls=":")
        lo, hi = max(0, pk - 2), min(len(xs) - 1, pk + 2)
        ax.axvspan(lo - 0.5, hi + 0.5, color=_BLUE_ALPHA, zorder=0)

        ax.plot(xs, supp, color=_BLUE, lw=1.8, label="Old-target suppression")
        ax.fill_between(xs, supp - d["sem_suppression"], supp + d["sem_suppression"],
                        color=_BLUE, alpha=0.15)
        ax.plot(xs[pk], supp[pk], "*", color=_BLUE, ms=10, zorder=5)

        # Posterior JS on secondary axis (same direction: both peak at mid-layers)
        if "mean_js" in d:
            ax2 = ax.twinx()
            js  = d["mean_js"]
            ax2.plot(xs, js, color=_GREEN, lw=1.8, ls=":", label="Posterior JS")
            ax2.fill_between(xs, js - d["sem_js"], js + d["sem_js"],
                             color=_GREEN, alpha=0.12)
            ax2.set_ylabel("JS divergence", color=_GREEN, fontsize=7)
            ax2.tick_params(axis="y", labelcolor=_GREEN, labelsize=7)

        _xticks(ax, d["layers"], xs)
        ax.set_xlabel("Probe layer", fontsize=8)
        ax.set_ylabel("Posterior mass delta", fontsize=8)
        ax.set_title(_MODEL_LABELS.get(model, model) + f"\n(n={n} pairs)", fontsize=9)

    legend_elems = [
        Line2D([0], [0], color=_BLUE,  lw=1.8,          label="Old-target suppression"),
        Line2D([0], [0], color=_GREEN, lw=1.8, ls=":",   label="Posterior JS"),
        Line2D([0], [0], color=_BLUE,  marker="*", ms=8, lw=0, label="Peak layer"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=3,
               fontsize=8, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Instruction Switch Suppresses Old Target at Mid-Layers",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    _save_fig(fig, output_dir, "fig4_instruction_suppression")


# ── Figure 5: Combined summary ────────────────────────────────────────────────

def plot_fig5_combined(exp3_data: Dict[str, dict], exp4_data: Dict[str, dict],
                       output_dir: str):
    models3 = [m for m in _MODEL_ORDER if m in exp3_data]
    models4 = [m for m in _MODEL_ORDER if m in exp4_data]
    n_cols  = max(len(models3), len(models4), 1)

    plt.rcParams.update(_FIG_PARAMS)
    fig = plt.figure(figsize=(3.2 * n_cols, 6.8))
    gs  = gridspec.GridSpec(2, n_cols, hspace=0.55, wspace=0.45)

    for col, model in enumerate(models3):
        ax  = fig.add_subplot(gs[0, col])
        ax2 = ax.twinx()
        d   = exp3_data[model]
        xs  = np.arange(len(d["layers"]))
        hit1, cnll = d["mean_hit1"], d["mean_coord_nll"]
        pk_sp, pk_se = _find_peak(hit1), _find_min(cnll)
        first_nz = next((i for i, v in enumerate(hit1) if v > 1.0), 0)

        ax.axvspan(first_nz - 0.5, pk_sp + 0.5, color=_BLUE_ALPHA, zorder=0)
        ax.axvspan(pk_se - 0.5, xs[-1] + 0.5,   color=_RED_ALPHA,  zorder=0)
        ax.plot(xs, hit1, color=_BLUE, lw=1.6)
        ax.plot(xs[pk_sp], hit1[pk_sp], "*", color=_BLUE, ms=9, zorder=5)
        ax.fill_between(xs, hit1 - d["sem_hit1"], hit1 + d["sem_hit1"],
                        color=_BLUE, alpha=0.15)
        ax2.plot(xs, cnll, color=_ORANGE, lw=1.6, ls="--")
        ax2.plot(xs[pk_se], cnll[pk_se], "^", color=_ORANGE, ms=7, zorder=5)
        ax2.fill_between(xs, cnll - d["sem_coord_nll"], cnll + d["sem_coord_nll"],
                         color=_ORANGE, alpha=0.15)
        _xticks(ax, d["layers"], xs, max_ticks=7)
        ax.tick_params(axis="y", labelcolor=_BLUE, labelsize=7)
        ax2.tick_params(axis="y", labelcolor=_ORANGE, labelsize=7)
        ax.set_title(_MODEL_LABELS.get(model, model), fontsize=8, pad=4)
        if col == 0:
            ax.set_ylabel("Hit@1 (%)", color=_BLUE, fontsize=7)
        if col == len(models3) - 1:
            ax2.set_ylabel("Coord NLL ↓", color=_ORANGE, fontsize=7)

    for col, model in enumerate(models4):
        ax = fig.add_subplot(gs[1, col])
        d  = exp4_data[model]
        xs = np.arange(len(d["layers"]))
        supp = d["mean_suppression"]
        pk   = _find_peak(supp)
        ax.axhline(0, color=_GREY, lw=0.7, ls=":")
        lo, hi = max(0, pk - 2), min(len(xs) - 1, pk + 2)
        ax.axvspan(lo - 0.5, hi + 0.5, color=_BLUE_ALPHA, zorder=0)
        ax.plot(xs, supp, color=_BLUE, lw=1.6)
        ax.plot(xs[pk], supp[pk], "*", color=_BLUE, ms=9, zorder=5)
        ax.fill_between(xs, supp - d["sem_suppression"], supp + d["sem_suppression"],
                        color=_BLUE, alpha=0.15)
        if "mean_js" in d:
            ax2 = ax.twinx()
            ax2.plot(xs, d["mean_js"], color=_GREEN, lw=1.4, ls=":")
            ax2.tick_params(axis="y", labelcolor=_GREEN, labelsize=6)
        _xticks(ax, d["layers"], xs, max_ticks=7)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("Layer", fontsize=7)
        if col == 0:
            ax.set_ylabel("Mass delta", fontsize=7)

    fig.text(0.01, 0.75, "(a) Exp 3", va="center", rotation="vertical",
             fontsize=8, fontweight="bold")
    fig.text(0.01, 0.27, "(b) Exp 4", va="center", rotation="vertical",
             fontsize=8, fontweight="bold")

    _save_fig(fig, output_dir, "fig5_combined")


# ── LaTeX table: Exp3 peak-lag ────────────────────────────────────────────────

def write_tab_exp3_peak_lag(exp3_data: Dict[str, dict], output_dir: str):
    """
    Table: for each model, spatial peak layer, NLL plateau layer, lag.
    Lag = (NLL plateau − spatial peak): positive = serialization lags grounding.
    """
    rows = []
    for model in _MODEL_ORDER:
        if model not in exp3_data:
            continue
        d = exp3_data[model]
        layers = d["layers"]
        pk_sp  = _find_peak(d["mean_hit1"])
        pk_se  = _find_min(d["mean_coord_nll"])
        lag    = layers[pk_se] - layers[pk_sp]
        rows.append((
            _MODEL_LABELS.get(model, model),
            layers[pk_sp],
            layers[pk_se],
            f"{lag:+d}",
            d["n_samples"],
        ))

    table_dir = os.path.join(output_dir, "..", "tables")
    os.makedirs(table_dir, exist_ok=True)
    out_path = os.path.join(table_dir, "tab_exp3_peak_lag.tex")

    with open(out_path, "w") as f:
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\toprule\n")
        f.write("Model & Spatial peak $L^*$ & Serialization plateau $L^\\dagger$ "
                "& Lag & $n$ \\\\\n")
        f.write("\\midrule\n")
        for row in rows:
            f.write(" & ".join(str(x) for x in row) + " \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
    print(f"[plot] Saved {out_path}")


# ── LaTeX table: Exp4 instruction sensitivity ─────────────────────────────────

def write_tab_exp4_instruction_sensitivity(exp4_data: Dict[str, dict], output_dir: str):
    """
    Table: for each model, peak suppression value, peak JS value, peak layer, n_pairs.
    """
    rows = []
    for model in _MODEL_ORDER:
        if model not in exp4_data:
            continue
        d   = exp4_data[model]
        layers = d["layers"]
        pk_supp = _find_peak(d["mean_suppression"])
        peak_supp_val = d["mean_suppression"][pk_supp]
        peak_js_val   = d["mean_js"][pk_supp] if "mean_js" in d else float("nan")
        rows.append((
            _MODEL_LABELS.get(model, model),
            layers[pk_supp],
            f"{peak_supp_val:.3f}",
            f"{peak_js_val:.3f}" if not (peak_js_val != peak_js_val) else "--",
            d["n_pairs"],
        ))

    table_dir = os.path.join(output_dir, "..", "tables")
    os.makedirs(table_dir, exist_ok=True)
    out_path  = os.path.join(table_dir, "tab_exp4_instruction_sensitivity.tex")

    with open(out_path, "w") as f:
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\toprule\n")
        f.write("Model & Peak layer & Suppression$_{\\max}$ & "
                "JS$_{\\max}$ & $n_{\\text{pairs}}$ \\\\\n")
        f.write("\\midrule\n")
        for row in rows:
            f.write(" & ".join(str(x) for x in row) + " \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
    print(f"[plot] Saved {out_path}")


# ── Data discovery ────────────────────────────────────────────────────────────

def discover_model_files(directory: str, suffix: str) -> Dict[str, str]:
    result = {}
    if not os.path.isdir(directory):
        return result
    for fname in os.listdir(directory):
        if fname.endswith(suffix):
            model = fname[: -len(suffix)]
            result[model] = os.path.join(directory, fname)
    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Plot probe experiment figures")
    p.add_argument("--exp3_dir",   default="outputs/exp3")
    p.add_argument("--exp4_dir",   default="outputs/exp4")
    p.add_argument("--output_dir", default="outputs/figures")
    p.add_argument("--combined",   action="store_true",
                   help="Also generate combined Fig 5")
    p.add_argument("--exp3_file",  default=None,
                   help="Single exp3 .jsonl file (overrides --exp3_dir)")
    p.add_argument("--exp4_file",  default=None,
                   help="Single exp4 .jsonl file (overrides --exp4_dir)")
    p.add_argument("--model_type", default=None,
                   help="Model type label when using --exp3_file / --exp4_file")
    return p.parse_args()


def main():
    args = parse_args()

    exp3_data: Dict[str, dict] = {}
    if args.exp3_file and args.model_type:
        agg = aggregate_exp3(read_jsonl(args.exp3_file))
        if agg:
            exp3_data[args.model_type] = agg
            print(f"[plot] exp3 {args.model_type}: {agg['n_samples']} samples")
    else:
        for model, fpath in discover_model_files(args.exp3_dir, "_lens.jsonl").items():
            agg = aggregate_exp3(read_jsonl(fpath))
            if agg:
                exp3_data[model] = agg
                print(f"[plot] exp3 {model}: {agg['n_samples']} samples")

    exp4_data: Dict[str, dict] = {}
    if args.exp4_file and args.model_type:
        agg = aggregate_exp4(read_jsonl(args.exp4_file))
        if agg:
            exp4_data[args.model_type] = agg
            print(f"[plot] exp4 {args.model_type}: {agg['n_pairs']} pairs")
    else:
        for model, fpath in discover_model_files(args.exp4_dir, "_cf.jsonl").items():
            agg = aggregate_exp4(read_jsonl(fpath))
            if agg:
                exp4_data[model] = agg
                print(f"[plot] exp4 {model}: {agg['n_pairs']} pairs")

    if not exp3_data and not exp4_data:
        print("[plot] No data found. Check --exp3_dir and --exp4_dir.")
        return

    if exp3_data:
        plot_fig3(exp3_data, args.output_dir)
        plot_exp3_coord_protocol_split(exp3_data, args.output_dir)
        write_tab_exp3_peak_lag(exp3_data, args.output_dir)

    if exp4_data:
        plot_fig4_instruction_suppression(exp4_data, args.output_dir)
        write_tab_exp4_instruction_sensitivity(exp4_data, args.output_dir)

    if args.combined and exp3_data and exp4_data:
        plot_fig5_combined(exp3_data, exp4_data, args.output_dir)


if __name__ == "__main__":
    main()
