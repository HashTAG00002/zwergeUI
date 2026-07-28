#!/usr/bin/env python3
"""
quadrant_plot.py — Four-quadrant error decomposition (A/B/C/D) figure for §3.

Reads the consolidated sweep summary (/tmp/p2p_all_summary.json by default) and
plots a stacked horizontal bar per (backbone, benchmark): A=already hit, B=top-3
has GT but ranking wrong, C=patch overlaps GT but point misses (sub-patch
quantization), D=top-3 misses GT (localization failure). This attributes every
error to a mechanism the P2P decoder can or cannot fix.
"""
import argparse, json, os, sys
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL = True
except Exception:
    _MPL = False

_FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "docs", "our_paper_tex", "figs")
ORDER = ["guiowl7b_ss_pro","guiowl7b_ss_v2","uitars_ss_pro","uitars_ss_v2",
         "guiowl_ss_pro","guiowl_ss_v2","uivenus_ss_pro","uivenus_ss_v2"]
LABEL = {"guiowl7b":"GUI-Owl-7B","uitars":"UI-TARS-1.5-7B","guiowl":"GUI-Owl-1.5-8B","uivenus":"UI-Venus-1.5-8B"}
COLOR = {"A":"#4daf4a","B":"#ff7f00","C":"#377eb8","D":"#e41a1c"}


def _nice(name):
    bench = "SS-Pro" if name.endswith("ss_pro") else "SS-v2"
    for k in LABEL:
        if name.startswith(k):
            return f"{LABEL[k]}  ({bench})"
    return name


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary", default="/tmp/p2p_all_summary.json")
    p.add_argument("--out", default=os.path.join(_FIG, "error_quadrants.pdf"))
    args = p.parse_args()
    data = json.load(open(args.summary))
    names = [n for n in ORDER if n in data] + [n for n in data if n not in ORDER]
    if not _MPL:
        print("[quad] matplotlib unavailable; A/B/C/D per (model,bench):")
        for n in names:
            q = data[n]["quadrants"]
            print(f"  {n}: A={q['A']} B={q['B']} C={q['C']} D={q['D']}")
        return

    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ys = list(range(len(names)))[::-1]
    for yi, n in zip(ys, names):
        q = data[n]["quadrants"]
        left = 0
        for k in "ABCD":
            v = q[k]
            ax.barh(yi, v, left=left, color=COLOR[k], height=0.62, edgecolor="w", linewidth=0.4)
            if v > 4:
                ax.text(left + v / 2, yi, f"{v:.0f}", ha="center", va="center", fontsize=7, color="white", fontweight="bold")
            left += v
    ax.set_yticks(ys)
    ax.set_yticklabels([_nice(n) for n in names], fontsize=8)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of samples (%)")
    ax.set_title("Error decomposition (baseline fusion decode)")
    from matplotlib.patches import Patch
    _desc = {"A": "already hit", "B": "ranking wrong (top-3 has GT)",
             "C": "sub-patch quantization", "D": "localization failure"}
    ax.legend(handles=[Patch(color=COLOR[k], label=f"{k}: {_desc[k]}") for k in "ABCD"],
              loc="lower right", fontsize=7, framealpha=0.9)
    fig.tight_layout()
    os.makedirs(_FIG, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"[quad] wrote {args.out}")


if __name__ == "__main__":
    main()
