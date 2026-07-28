#!/usr/bin/env python3
"""
bottleneck_severity_plot.py — Reframe the Qwen3 negative optimization.

Scatter, per (backbone, benchmark):
  x = retrofit headroom = (best intermediate-layer probe overlap@1) − (native baseline overlap@1)
  y = ZwerGe-UI delta    = (ZwerGe overlap@1) − (native baseline overlap@1)   [fair, same protocol]
                            (+ a companion series for the current overlap@3 delta)

Hypothesis (oracle §五 / §三): when the native decoder already matches or
exceeds the best intermediate probe (headroom ≤ 0), the retrofit has no signal
to exploit, so its delta is small or negative — an EXPECTED null result, not a
method failure. Qwen3-VL backbones sit in the negative-headroom / negative-delta
quadrant; Qwen2.5-VL in the positive/positive quadrant.

Reads the canonical A8 checkpoint-2800 layerwise_all_summary.json for the 4
backbones; native-baseline overlap@1 (Overall) is transcribed from the main
tables (docs/our_paper_tex/secs/5_experiment.tex).

Usage:
  python bottleneck_severity_plot.py --out figs/bottleneck_severity.pdf
  python bottleneck_severity_plot.py --out figs/bottleneck_severity.pdf --p2p_json p2p_summary.json
        # if --p2p_json given, also plot the P2P overlap@1 delta (post-sweep)
"""
import argparse, json, os, sys
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

_HERE = os.path.dirname(os.path.abspath(__file__))
_CKPT_BASE = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/.hdd/ckpt/zwerge"
_FIG_ROOT = os.path.join(_HERE, "..", "..", "docs", "our_paper_tex", "figs")

RUNS = {
    "GUI-Owl-7B":     ("guiowl7b_A8_cosmeta_ctx_exp001", "Qwen2.5-VL"),
    "UI-TARS-1.5-7B": ("uitars_A8_cosmeta_ctx_exp001",   "Qwen2.5-VL"),
    "GUI-Owl-1.5-8B": ("guiowl_A8_cosmeta_ctx_exp001",   "Qwen3-VL"),
    "UI-Venus-1.5-8B":("uivenus_A8_cosmeta_ctx_exp003",  "Qwen3-VL"),
}
# native baseline overlap@1 (Overall) per (model, bench) — from 5_experiment.tex main tables.
BASE_OV1 = {
    "GUI-Owl-7B":     {"ss_pro":52.12,"ss_v2":89.69,"osworld_g":65.69,"mmbench":72.84,"ui_vision":27.94},
    "UI-TARS-1.5-7B": {"ss_pro":51.11,"ss_v2":90.72,"osworld_g":71.18,"mmbench":75.99,"ui_vision":24.01},
    "GUI-Owl-1.5-8B": {"ss_pro":74.83,"ss_v2":94.18,"osworld_g":80.98,"mmbench":83.50,"ui_vision":43.25},
    "UI-Venus-1.5-8B":{"ss_pro":70.46,"ss_v2":96.85,"osworld_g":85.69,"mmbench":88.98,"ui_vision":52.35},
}
BENCHES = ["ss_pro", "ss_v2", "osworld_g", "mmbench", "ui_vision"]
BENCH_LABEL = {"ss_pro":"SS-Pro","ss_v2":"SS-v2","osworld_g":"OSW-G",
               "mmbench":"MMBench","ui_vision":"UI-Vis"}
FAMILY_COLOR = {"Qwen2.5-VL": "#1f77b4", "Qwen3-VL": "#d62728"}


def load_layerwise(model, run):
    p = os.path.join(_CKPT_BASE, run, "checkpoint-2800", "results", "layerwise_all_summary.json")
    with open(p) as f:
        return json.load(f)


def collect(p2p_json=None):
    """Return list of dicts: {model, family, bench, headroom, delta_ov1, delta_ov3, delta_p2p_ov1}."""
    p2p = {}
    if p2p_json and os.path.exists(p2p_json):
        p2p = json.load(open(p2p_json))   # {(model,bench): {overlap1: ...}}
    rows = []
    for model, (run, family) in RUNS.items():
        d = load_layerwise(model, run)
        for bench in BENCHES:
            if bench not in d:
                continue
            fa = d[bench]["fusion_acc"]
            peaks = [la.get("overlap_top1", 0) for la in d[bench].get("layer_accs", [])]
            peak = max(peaks) if peaks else 0.0
            base = BASE_OV1[model][bench]
            fus_ov1 = fa.get("overlap_top1", 0.0)
            fus_ov3 = fa.get("overlap_topk", 0.0)
            p2p_ov1 = p2p.get((model, bench), {}).get("overlap1", fus_ov1)
            rows.append(dict(model=model, family=family, bench=bench,
                             base=base, peak=peak, fus_ov1=fus_ov1, fus_ov3=fus_ov3,
                             p2p_ov1=p2p_ov1,
                             headroom=peak - base,
                             delta_ov1=fus_ov1 - base,
                             delta_ov3=fus_ov3 - base,
                             delta_p2p_ov1=p2p_ov1 - base))
    return rows


def plot(rows, out_path):
    if not _HAS_MPL:
        print("[plot] matplotlib unavailable — skipping figure, data table above is the source of truth.")
        return
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    # zero lines
    ax.axhline(0, color="0.6", lw=0.8, zorder=1)
    ax.axvline(0, color="0.6", lw=0.8, zorder=1)
    seen = set()
    for r in rows:
        fam = r["family"]; c = FAMILY_COLOR[fam]
        m = {"Qwen2.5-VL": "o", "Qwen3-VL": "s"}[fam]
        ax.scatter(r["headroom"], r["delta_ov1"], c=c, marker=m, s=42,
                   edgecolors="k", linewidths=0.4, zorder=3,
                   label=fam if fam not in seen else "")
        seen.add(fam)
    # optional P2P arrows (headroom → p2p delta)
    p2p_rows = [r for r in rows if "delta_p2p_ov1" in r and r["p2p_ov1"] != r["fus_ov1"]]
    for r in p2p_rows:
        c = FAMILY_COLOR[r["family"]]
        ax.scatter(r["headroom"], r["delta_p2p_ov1"], c=c, marker="*", s=120,
                   edgecolors="k", linewidths=0.4, zorder=4)
        ax.annotate("", xy=(r["headroom"], r["delta_p2p_ov1"]),
                    xytext=(r["headroom"], r["delta_ov1"]),
                    arrowprops=dict(arrowstyle="->", color=c, lw=0.8, alpha=0.7))
    ax.set_xlabel("Retrofit headroom: peak-probe overlap@1 $-$ native baseline overlap@1 (pp)")
    ax.set_ylabel("ZwerGe-UI delta over native baseline (pp)")
    ax.set_title("Bottleneck severity predicts retrofit gain")
    ax.legend(title="Backbone family", loc="upper left", fontsize=8)
    # quadrant annotations
    ax.text(0.97, 0.97, "headroom > 0, gain > 0\n(expected: method helps)",
            transform=ax.transAxes, ha="right", va="top", fontsize=7, color="#1f77b4")
    ax.text(0.03, 0.03, "headroom < 0, gain < 0\n(expected null: native already at ceiling)",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=7, color="#d62728")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"[plot] wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join(_FIG_ROOT, "bottleneck_severity.pdf"))
    p.add_argument("--p2p_json", default=None,
                   help="optional JSON {model: {bench: {overlap1}}} from p2p_sweep to overlay P2P deltas")
    p.add_argument("--data_out", default=None, help="dump the per-row data as JSON here")
    args = p.parse_args()
    rows = collect(p2p_json=args.p2p_json)
    # print table
    print(f"{'model':<18}{'bench':<10}{'base':>7}{'peak':>7}{'fus_ov1':>8}{'fus_ov3':>8}"
          f"{'headrm':>8}{'d_ov1':>7}{'d_ov3':>7}")
    for r in rows:
        print(f"{r['model']:<18}{BENCH_LABEL[r['bench']]:<10}{r['base']:>7.2f}{r['peak']:>7.2f}"
              f"{r['fus_ov1']:>8.2f}{r['fus_ov3']:>8.2f}{r['headroom']:>+8.2f}"
              f"{r['delta_ov1']:>+7.2f}{r['delta_ov3']:>+7.2f}")
    plot(rows, args.out)
    if args.data_out:
        json.dump(rows, open(args.data_out, "w"), indent=2)
        print(f"[plot] data → {args.data_out}")


if __name__ == "__main__":
    main()
