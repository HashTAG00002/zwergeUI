#!/usr/bin/env python3
"""
run_all_sweeps.py — Consolidated P2P sweep across all cached (model, bench) dirs.

For each p2p_cache/<model>_<bench> and p2p_p2p_native/<model>_<bench> dir, runs
the baseline + key P2P configs (+ the native-gate config for native dirs) and
collects: strict overlap@1/hit@1, proposal recall ov@3, A/B/C/D quadrant counts,
recovery/damage. Outputs a consolidated JSON + a LaTeX-ready summary table for
the paper's P2P rows and the §3 error-decomposition figure.
"""
import argparse, glob, json, os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import p2p_sweep as S   # noqa: E402

_CACHE = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/zwerge/data/results/p2p_cache"
_NATIVE = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/zwerge/data/results/p2p_p2p_native"

CONFIGS = ["baseline", "msqrt", "local_o25_a2", "cons"]


def _dirs(base):
    out = {}
    for d in sorted(glob.glob(os.path.join(base, "*"))):
        name = os.path.basename(d)   # <model>_<bench>
        pdir = os.path.join(d, "details")
        # find the bench subdir under details/
        subs = glob.glob(os.path.join(pdir, "*", "posteriors"))
        if not subs:
            continue
        posteriors = subs[0]
        bench = os.path.basename(os.path.dirname(posteriors))
        out[name] = (d, posteriors, bench)
    return out


def _run_one(label, posteriors, bench, is_native, topk, thr, gd):
    samples = S.load_samples(posteriors)
    if not samples:
        return None
    base_dec = [S._decode_sample(s, S.CONFIGS["baseline"], topk, thr) for s in samples]
    base_m = [b[2] for b in base_dec]
    quads = [S.quadrant_of(m) for m in base_m]
    qc = {q: quads.count(q) for q in "ABCD"}
    n = len(samples)
    res = {
        "label": label, "bench": bench, "n": n, "is_native": is_native,
        "quadrants": {q: round(qc[q] / n * 100, 2) for q in "ABCD"},
        "quadrants_n": qc,
        "configs": {},
    }
    cfgs = list(CONFIGS)
    if is_native:
        cfgs.append("__native_gate__")
    for c in cfgs:
        cfg = "__native_gate__" if c == "__native_gate__" else S.CONFIGS[c]
        dec = [S._decode_sample(s, cfg, topk, thr, gd) for s in samples]
        ms = [d[2] for d in dec]
        agg = S.aggregate(samples, ms)
        # recovery/damage vs baseline
        bc = [i for i, m in enumerate(base_m) if not m["hit1"] and m["ovk"]]
        rec = sum(1 for i in bc if ms[i]["hit1"])
        dmg = sum(1 for i in range(n) if base_m[i]["hit1"] and not ms[i]["hit1"])
        n_old_h1 = sum(m["hit1"] for m in base_m)
        res["configs"][c] = {
            "hit1": agg["hit1"], "ov1": agg["ov1"], "hitk": agg["hitk"], "ovk": agg["ovk"],
            "recovery_pct": round(rec / max(1, len(bc)) * 100, 2),
            "damage_pct": round(dmg / max(1, n_old_h1) * 100, 2),
        }
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--thr", type=float, default=0.3)
    p.add_argument("--gate_dilate", type=int, default=1)
    p.add_argument("--out", default="/tmp/p2p_all_summary.json")
    args = p.parse_args()

    results = {}
    for kind, base in [("cache", _CACHE), ("native", _NATIVE)]:
        for name, (d, pdir, bench) in _dirs(base).items():
            print(f"[..] {kind} {name} ({bench}) ...", flush=True)
            r = _run_one(f"{kind}:{name}", pdir, bench, kind == "native",
                         args.topk, args.thr, args.gate_dilate)
            if r:
                results[name] = r

    # print summary
    print("\n" + "=" * 110)
    print(f"{'model_bench':<22}{'cfg':<16}{'hit1':>7}{'ov1':>7}{'ov3':>7}{'rec%':>7}{'dmg%':>7} | "
          f"{'A%':>5}{'B%':>5}{'C%':>5}{'D%':>5}")
    print("-" * 110)
    for name, r in sorted(results.items()):
        for c in ["baseline", "msqrt", "local_o25_a2", "cons", "__native_gate__"]:
            if c not in r["configs"]:
                continue
            x = r["configs"][c]
            q = r["quadrants"]
            print(f"{name:<22}{c:<16}{x['hit1']:>7.2f}{x['ov1']:>7.2f}{x['ovk']:>7.2f}"
                  f"{x['recovery_pct']:>7}{x['damage_pct']:>7} | "
                  f"{q['A']:>5}{q['B']:>5}{q['C']:>5}{q['D']:>5}")
        print()
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
