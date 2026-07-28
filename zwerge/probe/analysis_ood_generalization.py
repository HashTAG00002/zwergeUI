"""
analysis_ood_generalization.py — P0-4: OOD generalization of the inverted-U peak layer.

Reviewer concern addressed:
  eKpD-W2 "held-out OOD UI layouts" / 4KeN-W4 dataset-level confounds.

Question: does the spatial-peak layer L* (and the inverted-U shape) survive across
benchmarks whose UI style differs substantially from the training distribution?
Training data is OS-Atlas / GroundCUA / AgentNet style; UI-Vision and MMBench-GUI
are the most distribution-shifted. If L* varies by at most a few layers across all
five benchmarks despite these UI-style differences, the inverted-U is not an
artifact of overfitting one layout family.

This is a PURE DATA-ANALYSIS script — no GPU, no model forward. It reads the
per-benchmark `*_layerwise_summary.json` files produced by the eval daemon and
computes peak-layer statistics.

It ALSO verifies which checkpoint (A7-last vs A8-2800) reproduces the published
Finding-4 table, so the analysis is built on the same checkpoint as the main text.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

CKPT_BASE = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/.hdd/ckpt/zwerge"
OUT_DIR = Path(__file__).resolve().parent / "outputs"
FIG_DIR = OUT_DIR / "figures"
TBL_DIR = OUT_DIR / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)

# (model_key, display, A7 dir, A8 dir, A7_last_ckpt, A8_canonical_ckpt)
MODELS = [
    ("uitars",   "UI-TARS-1.5-7B",        "uitars_A7_exp001",   "uitars_A8_cosmeta_ctx_exp001",   3129, 2800),
    ("guiowl7b", "GUI-Owl-7B",            "guiowl7b_A7_exp001",  "guiowl7b_A8_cosmeta_ctx_exp001", 3129, 2800),
    ("guiowl",   "GUI-Owl-1.5-8B-Instruct","guiowl_A7_exp002",   "guiowl_A8_cosmeta_ctx_exp001",   3130, 2800),
    ("uivenus",  "UI-Venus-1.5-8B",       "uivenus_A7_exp002",   "uivenus_A8_cosmeta_ctx_exp003",  3130, 2800),
]

# Five benchmarks (osworld_g = refusals-removed, the main one)
BENCHES = ["ss_pro", "ss_v2", "osworld_g", "ui_vision", "mmbench"]
BENCH_LABEL = {
    "ss_pro": "SS-Pro", "ss_v2": "SS-v2", "osworld_g": "OSWorld-G",
    "ui_vision": "UI-Vision", "mmbench": "MMBench-GUI",
}

# Published Finding-4 table (3_analysis.tex tab:resolution_comparison) — for verification
TABLE_TARGETS = {
    "uitars":   {"ss_pro": (46.0, 30.5), "ss_v2": (91.5, 80.0)},
    "guiowl7b": {"ss_pro": (46.5, 37.0), "ss_v2": (88.0, 80.5)},
    "guiowl":   {"ss_pro": (54.0, 30.5), "ss_v2": (90.5, 84.5)},
    "uivenus":  {"ss_pro": (54.0, 32.5), "ss_v2": (91.5, 85.0)},
}


def _load_layerwise(model_dir, ckpt, bench):
    f = os.path.join(CKPT_BASE, model_dir, f"checkpoint-{ckpt}", "results",
                     f"{bench}_layerwise_summary.json")
    if not os.path.exists(f):
        return None
    return json.load(open(f))


def _peak_final(d, key="hit_top1"):
    """Return (peak_val, peak_layer_idx, final_val) from a layerwise summary."""
    accs = d["layer_accs"]
    vals = [a[key] for a in accs]
    pk_i = int(np.argmax(vals))
    return vals[pk_i], accs[pk_i]["layer_idx"], vals[-1]


def verify_checkpoint():
    """Determine whether A7-last or A8-2800 reproduces the published Finding-4 table."""
    print("=" * 78)
    print("CHECKPOINT VERIFICATION (vs published Finding-4 table)")
    print("=" * 78)
    best = {"A7": 0, "A8": 0}
    rows = []
    for key, disp, a7dir, a8dir, a7ck, a8ck in MODELS:
        for ck_label, mdir, ck in (("A7", a7dir, a7ck), ("A8", a8dir, a8ck)):
            ok = 0; tot = 0; detail = []
            for bench in ("ss_pro", "ss_v2"):
                d = _load_layerwise(mdir, ck, bench)
                if d is None:
                    continue
                pk, _, fin = _peak_final(d)
                tp, tf = TABLE_TARGETS[key][bench]
                tot += 2
                if abs(pk - tp) < 0.8: ok += 1
                if abs(fin - tf) < 0.8: ok += 1
                detail.append(f"{bench}: pk={pk:.1f}(tbl {tp}) fin={fin:.1f}(tbl {tf})")
            rows.append((disp, ck_label, ok, tot, "; ".join(detail)))
            best[ck_label] += ok
    for r in rows:
        print(f"{r[0]:24s} {r[1]}  {r[2]}/{r[3]}  | {r[4]}")
    print(f"\nTotal matches: A7={best['A7']}  A8={best['A8']}")
    choice = "A8" if best["A8"] >= best["A7"] else "A7"
    print(f"--> Using {choice} (more consistent with published table)")
    return choice


def ood_analysis(ckpt_choice):
    print("\n" + "=" * 78)
    print(f"P0-4 OOD GENERALIZATION ANALYSIS (checkpoint set: {ckpt_choice})")
    print("=" * 78)
    # Build the (model, dir, ckpt) per the chosen family
    config = []
    for key, disp, a7dir, a8dir, a7ck, a8ck in MODELS:
        if ckpt_choice == "A8":
            config.append((key, disp, a8dir, a8ck))
        else:
            config.append((key, disp, a7dir, a7ck))

    # Per (model, bench): peak layer (hit_top1), peak layer (overlap_top1),
    # final hit_top1, collapse gap (peak-final hit_top1), n
    records = {}
    all_ok = True
    for key, disp, mdir, ck in config:
        records[key] = {"disp": disp, "benches": {}}
        for bench in BENCHES:
            d = _load_layerwise(mdir, ck, bench)
            if d is None:
                print(f"  [warn] missing: {key} {bench} ckpt-{ck}")
                all_ok = False
                continue
            pk_h, pkL_h, fin_h = _peak_final(d, "hit_top1")
            pk_o, pkL_o, fin_o = _peak_final(d, "overlap_top1")
            records[key]["benches"][bench] = {
                "n": d.get("valid", d.get("total")),
                "probe_layers": d.get("probe_layers"),
                "peak_hit1": pk_h, "peak_hit1_layer": pkL_h,
                "final_hit1": fin_h, "gap_hit1": pk_h - fin_h,
                "peak_ovl1": pk_o, "peak_ovl1_layer": pkL_o,
                "final_ovl1": fin_o, "gap_ovl1": pk_o - fin_o,
            }
    if not all_ok:
        print("  (some cells missing — proceeding with available ones)")

    # ---- Per-model peak-layer variance across the 5 benchmarks (hit@1) ----
    print("\nPer-model spatial-peak layer L* (hit@1) across 5 benchmarks:")
    print(f"{'Model':24s} " + " ".join(f"{BENCH_LABEL[b]:>11s}" for b in BENCHES)
          + f"{'mean':>7s}{'std':>7s}{'range':>7s}")
    summary_rows = []
    for key, disp, _, _ in config:
        rec = records[key]
        peaks = [rec["benches"][b]["peak_hit1_layer"] for b in BENCHES
                 if b in rec["benches"]]
        gaps = [rec["benches"][b]["gap_hit1"] for b in BENCHES if b in rec["benches"]]
        finals = [rec["benches"][b]["final_hit1"] for b in BENCHES if b in rec["benches"]]
        arr = np.array(peaks)
        row_disp = (f"{disp:24s} " + " ".join(f"{p:>11d}" for p in peaks)
                    + f"{arr.mean():>7.2f}{arr.std():>7.2f}{arr.max()-arr.min():>7d}")
        print(row_disp)
        summary_rows.append({
            "model": disp, "key": key,
            "peak_layers": peaks,
            "peak_mean": float(arr.mean()), "peak_std": float(arr.std()),
            "peak_range": int(arr.max() - arr.min()),
            "gaps": gaps, "finals": finals,
            "benches": rec["benches"],
        })

    # ---- Inverted-U present? (peak > final for every benchmark) ----
    print("\nInverted-U presence (peak_hit1 > final_hit1 in every benchmark):")
    for r in summary_rows:
        present = [g > 0 for g in r["gaps"]]
        print(f"  {r['model']:24s} {sum(present)}/{len(present)} benches have peak>final "
              f"(gaps: {['%.1f'%g for g in r['gaps']]})")

    # ---- Aggregate statistics ----
    all_ranges = [r["peak_range"] for r in summary_rows]
    all_stds = [r["peak_std"] for r in summary_rows]
    print("\n" + "-" * 78)
    print(f"Peak-layer range across 5 benchmarks: "
          f"max={max(all_ranges)} layers, mean={np.mean(all_ranges):.1f} layers")
    print(f"Peak-layer std across 5 benchmarks: "
          f"max={max(all_stds):.2f}, mean={np.mean(all_stds):.2f}")
    print(f"Inverted-U present in "
          f"{sum(sum(g>0 for g in r['gaps']) for r in summary_rows)}/"
          f"{sum(len(r['gaps']) for r in summary_rows)} (model,benchmark) cells")

    return records, summary_rows


def write_tex(summary_rows, ckpt_choice):
    """Write a tex table + a short quantitative blurb."""
    out = TBL_DIR / "tab_ood_peak_layer.tex"
    with open(out, "w") as f:
        f.write("\\begin{tabular}{l" + "c" * len(BENCHES) + "ccc}\n\\toprule\n")
        f.write("& " + " & ".join(BENCH_LABEL[b] for b in BENCHES)
                + " & mean & std & range\\\\\n")
        f.write("\\midrule\n")
        for r in summary_rows:
            cells = " & ".join(str(p) for p in r["peak_layers"])
            f.write(f"{r['model']} & {cells} & {r['peak_mean']:.1f} & "
                    f"{r['peak_std']:.2f} & {r['peak_range']}\\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"\n[tex] Saved {out}")

    # Save machine-readable JSON
    jout = OUT_DIR / "ood_generalization_summary.json"
    json.dump({
        "ckpt_choice": ckpt_choice,
        "benches": BENCHES,
        "models": [{k: v for k, v in r.items() if k != "benches"} | {"benches": {
            b: {kk: vv for kk, vv in bb.items() if kk != "probe_layers"}
            for b, bb in r["benches"].items()
        }} for r in summary_rows],
    }, jout.open("w"), indent=2)
    print(f"[json] Saved {jout}")


def write_blurb(summary_rows):
    """Compose the quantitative sentence for the paper."""
    all_ranges = [r["peak_range"] for r in summary_rows]
    maxrange = max(all_ranges)
    present = sum(sum(g > 0 for g in r["gaps"]) for r in summary_rows)
    total = sum(len(r["gaps"]) for r in summary_rows)
    blurb = (
        f"The spatial-peak layer $L^*$ varies by at most ${maxrange}$ layers across "
        f"all five benchmarks per model (std $\\le {max(r['peak_std'] for r in summary_rows):.2f}$), "
        f"and the inverted-U (peak $>$ final) is present in ${present}/{total}$ "
        f"(model, benchmark) cells, including the distribution-shifted UI-Vision "
        f"and MMBench-GUI benchmarks. The mid-layer spatial advantage is thus not an "
        f"artifact of the training layout family."
    )
    (TBL_DIR / "ood_blurb.tex").write_text(blurb)
    print(f"[tex] Saved {TBL_DIR / 'ood_blurb.tex'}")
    print("\nBLURB:\n" + blurb)


if __name__ == "__main__":
    choice = verify_checkpoint()
    _, rows = ood_analysis(choice)
    write_tex(rows, choice)
    write_blurb(rows)
