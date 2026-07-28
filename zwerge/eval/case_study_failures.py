#!/usr/bin/env python3
"""
case_study_failures.py — visual case study of ZwerGe SS-Pro failures.

Loads the cached posteriors (p2p_cache/guiowl_ss_pro), classifies each sample
into the A/B/C/D error quadrant (baseline max+centroid decode), picks a balanced
set of failures, and renders PNGs (image + GT bbox + predicted point + top-3
region boxes + p_final heatmap) for visual diagnosis.
"""
import glob, json, os, sys, math
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from inference_base import decode_p2p, point_in_bbox, do_boxes_overlap  # noqa
try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    import torch
except Exception as e:
    raise SystemExit(f"missing dep: {e}")

CACHE = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/zwerge/data/results/p2p_cache/guiowl_ss_pro/details/ss_pro/posteriors"
EVAL_DIR = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/datasets/evaluation/ScreenSpot-Pro"
OUT = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/zwerge/data/results/case_study"
TOPK = 3


def metrics(best, centers, gt, n_w, n_h, topk):
    phx, phy = 0.5 / n_w, 0.5 / n_h
    def box(p): return (p[0]-phx, p[1]-phy, p[0]+phx, p[1]+phy)
    ov1 = int(do_boxes_overlap(box(best), gt))
    hit1 = int(point_in_bbox(best[0], best[1], gt))
    cs = centers[:topk]
    ov3 = int(any(do_boxes_overlap(box(c), gt) for c in cs))
    hit3 = int(any(point_in_bbox(c[0], c[1], gt) for c in cs))
    return ov1, hit1, ov3, hit3


def load_all():
    samples = []
    for fp in sorted(glob.glob(os.path.join(CACHE, "idx*.pt"))):
        d = torch.load(fp, map_location="cpu")
        samples.append(d)
    return samples


def classify(samples):
    rows = []
    for s in samples:
        n_w, n_h = s["n_width"], s["n_height"]
        gt = tuple(s["gt_bbox_norm"].tolist())
        best, centers, _, _ = decode_p2p(
            s["p_final"], n_w, n_h, 0.3, TOPK, "max",
            per_layer_probs=None, omega=None, use_consensus=False, use_local_mode=False)
        ov1, hit1, ov3, hit3 = metrics(best, centers, gt, n_w, n_h, TOPK)
        if hit1: q = "A"
        elif hit3: q = "B"
        elif ov3: q = "C"
        else: q = "D"
        rows.append(dict(s=s, q=q, best=best, centers=centers, ov1=ov1, ov3=ov3, hit1=hit1,
                         gt=gt, n_w=n_w, n_h=n_h))
    return rows


def render(row, out_png):
    s = row["s"]; n_w, n_h = row["n_w"], row["n_h"]
    img_path = os.path.join(EVAL_DIR, s["image_path"])
    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    gx1, gy1, gx2, gy2 = row["gt"]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(img)
    # heatmap
    grid = s["p_final"].float().reshape(n_h, n_w)
    extent = (0, W, H, 0)
    im = ax.imshow(grid.numpy(), extent=extent, cmap="jet", alpha=0.45,
                   aspect="auto", interpolation="bilinear")
    # GT bbox
    ax.add_patch(plt.Rectangle((gx1*W, gy1*H), (gx2-gx1)*W, (gy2-gy1)*H,
                 fill=False, edgecolor="lime", linewidth=2.5, linestyle="--"))
    # top-3 region centers
    for i, c in enumerate(row["centers"][:TOPK]):
        cx, cy = c[0]*W, c[1]*H
        col = "red" if i == 0 else "orange"
        ax.add_patch(plt.Rectangle((cx-0.5*W/n_w, cy-0.5*H/n_h), W/n_w, H/n_h,
                     fill=False, edgecolor=col, linewidth=1.5))
        ax.plot(cx, cy, marker="x", color=col, ms=12, mew=2)
    ax.set_title(f"idx={s['idx']} q={row['q']} ov1={row['ov1']} ov3={row['ov3']} | "
                 f"{s.get('ui_type','?')} / {s.get('group','?')}\n{s['instruction'][:60]}",
                 fontsize=9)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    print("loading posteriors...")
    samples = load_all()
    print(f"{len(samples)} samples")
    rows = classify(samples)
    from collections import Counter
    qc = Counter(r["q"] for r in rows)
    print(f"quadrants: {dict(qc)}  ({ {q: round(qc[q]/len(rows)*100,1) for q in 'ABCD'} })")
    # pick failures: 3 from B (ranking wrong) + 3 from D (localization fail) + 2 from C
    picks = []
    for q in ["B", "D", "C"]:
        sub = [r for r in rows if r["q"] == q]
        # spread across ui_type
        seen_type = set()
        for r in sub:
            t = r["s"].get("ui_type", "?")
            if t not in seen_type:
                picks.append(r); seen_type.add(t)
            if len([p for p in picks if p["q"] == q]) >= (4 if q == "D" else 3):
                break
    print(f"rendering {len(picks)} failure cases...")
    paths = []
    for r in picks:
        p = os.path.join(OUT, f"fail_{r['q']}_idx{r['s']['idx']:05d}.png")
        render(r, p)
        paths.append((r, p))
        print(f"  {os.path.basename(p)}: q={r['q']} {r['s'].get('ui_type','')} / {r['s'].get('group','')} | instr='{r['s']['instruction'][:50]}'")
    # dump a tiny json index
    json.dump([{"q": r["q"], "idx": r["s"]["idx"], "image_path": r["s"]["image_path"],
                "instruction": r["s"]["instruction"], "ui_type": r["s"].get("ui_type"),
                "group": r["s"].get("group"), "gt_bbox_norm": list(r["gt"]),
                "ov1": r["ov1"], "ov3": r["ov3"], "best": list(r["best"]),
                "centers": [list(c) for c in r["centers"][:TOPK]],
                "png": p} for r, p in paths],
              open(os.path.join(OUT, "cases.json"), "w"), indent=2)
    print(f"done → {OUT}")


if __name__ == "__main__":
    main()
