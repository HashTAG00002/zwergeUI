"""
plot_probe.py — 从 layer_probe checkpoint（单卡或多卡）快速统计各层指标并绘制折线图

用法:
  # 读单个 checkpoint
  python eval/plot_probe.py \
      --ckpt /path/to/sspro_probe.json.rank0.ckpt.json

  # 合并多卡 checkpoint（自动发现 rank0/rank1/...）
  python eval/plot_probe.py \
      --ckpt /path/to/sspro_probe.json.rank0.ckpt.json \
             /path/to/sspro_probe.json.rank1.ckpt.json

  # 自动扫描目录下所有 rank checkpoint
  python eval/plot_probe.py \
      --output_base /path/to/sspro_probe.json

  # 保存图片（不弹窗）
  python eval/plot_probe.py --output_base ... --save_fig results/probe_curves.png
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")   # 无显示器环境下也能保存图片；如需弹窗改为 TkAgg
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ---------------------------------------------------------------------------
# 加载并合并多个 rank checkpoint
# ---------------------------------------------------------------------------

def load_ckpts(ckpt_paths: list[str]):
    """
    合并多个 rank checkpoint，返回 per_sample_results 列表（只含有效条目）。
    每个有效 entry 格式：
      { "layer_data": { "0": {"hit_A":bool,"dist_A":float,"hit_B":bool,"dist_B":float,"entropy":float}, ... },
        "ui_type": str, "instruction": str, ... }
    """
    all_entries = []
    for path in ckpt_paths:
        if not os.path.exists(path):
            print(f"[warn] not found: {path}")
            continue
        with open(path) as f:
            ckpt = json.load(f)
        entries = ckpt.get("per_sample_results", [])
        valid = [e for e in entries if not e.get("skipped")]
        print(f"  {os.path.basename(path)}: total={len(entries)}, valid={len(valid)}")
        all_entries.extend(valid)
    return all_entries


def auto_find_ckpts(output_base: str) -> list[str]:
    """扫描 <output_base>.rank*.ckpt.json"""
    pattern = output_base + ".rank*.ckpt.json"
    found = sorted(glob.glob(pattern))
    return found


# ---------------------------------------------------------------------------
# 统计每层指标
# ---------------------------------------------------------------------------

def compute_layer_stats(entries: list[dict]):
    """
    返回 dict:
      layer_idx  → { acc_A, acc_B, mean_dist_A, mean_dist_B, entropy_mean, entropy_std, n }
    """
    # 先探测层数
    n_layers = max(int(k) for e in entries for k in e["layer_data"].keys()) + 1

    acc_A      = np.zeros(n_layers)
    acc_B      = np.zeros(n_layers)
    dist_A     = np.zeros(n_layers)
    dist_B     = np.zeros(n_layers)
    entropy    = np.zeros(n_layers)
    ent_sq     = np.zeros(n_layers)   # for std
    cnt_A      = np.zeros(n_layers, dtype=int)
    cnt_B      = np.zeros(n_layers, dtype=int)
    cnt_ent    = np.zeros(n_layers, dtype=int)

    for e in entries:
        for li_str, ld in e["layer_data"].items():
            li = int(li_str)
            if "hit_A" in ld:
                acc_A[li]  += int(ld["hit_A"])
                dist_A[li] += ld["dist_A"]
                cnt_A[li]  += 1
            if "hit_B" in ld:
                acc_B[li]  += int(ld["hit_B"])
                dist_B[li] += ld["dist_B"]
                cnt_B[li]  += 1
            if "entropy" in ld:
                entropy[li] += ld["entropy"]
                ent_sq[li]  += ld["entropy"] ** 2
                cnt_ent[li] += 1

    stats = {}
    for li in range(n_layers):
        nA = max(cnt_A[li], 1);  nB = max(cnt_B[li], 1);  nE = max(cnt_ent[li], 1)
        ent_mean = entropy[li] / nE
        ent_var  = max(ent_sq[li]/nE - ent_mean**2, 0.0)
        stats[li] = {
            "acc_A":       float(acc_A[li]  / nA),
            "acc_B":       float(acc_B[li]  / nB),
            "dist_A":      float(dist_A[li] / nA),
            "dist_B":      float(dist_B[li] / nB),
            "entropy_mean": float(ent_mean),
            "entropy_std":  float(ent_var ** 0.5),
            "n_A": int(cnt_A[li]),
            "n_B": int(cnt_B[li]),
        }
    return stats, n_layers


# ---------------------------------------------------------------------------
# 按 ui_type 分组统计 ACC_A
# ---------------------------------------------------------------------------

def compute_per_type_acc(entries, n_layers):
    """返回 {ui_type: np.array(n_layers) of ACC_A}"""
    hit_by_type  = defaultdict(lambda: np.zeros(n_layers))
    cnt_by_type  = defaultdict(lambda: np.zeros(n_layers, dtype=int))
    for e in entries:
        ut = e.get("ui_type", "unknown")
        for li_str, ld in e["layer_data"].items():
            li = int(li_str)
            if "hit_A" in ld:
                hit_by_type[ut][li] += int(ld["hit_A"])
                cnt_by_type[ut][li] += 1
    return {ut: hit_by_type[ut] / np.maximum(cnt_by_type[ut], 1)
            for ut in hit_by_type}


# ---------------------------------------------------------------------------
# 绘图
# ---------------------------------------------------------------------------

COLORS = [
    "#2196F3", "#E91E63", "#4CAF50", "#FF9800",
    "#9C27B0", "#00BCD4", "#F44336", "#8BC34A",
]


def plot_curves(stats, n_layers, per_type_acc, save_path=None, title_suffix=""):
    layers = np.arange(n_layers)

    acc_A  = np.array([stats[l]["acc_A"]        for l in layers])
    acc_B  = np.array([stats[l]["acc_B"]        for l in layers])
    dist_A = np.array([stats[l]["dist_A"]       for l in layers])
    dist_B = np.array([stats[l]["dist_B"]       for l in layers])
    ent_m  = np.array([stats[l]["entropy_mean"] for l in layers])
    ent_s  = np.array([stats[l]["entropy_std"]  for l in layers])

    n_total = stats[0]["n_A"]   # 各层样本数相同

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"Layer-wise Probe Curves{title_suffix}  (n={n_total})",
                 fontsize=13, fontweight="bold")

    # ── (0,0) ACC_A & ACC_B ──
    ax = axes[0, 0]
    ax.plot(layers, acc_A * 100, color=COLORS[0], lw=1.8, label="ACC_A (attn, uniform)")
    ax.plot(layers, acc_B * 100, color=COLORS[1], lw=1.8, linestyle="--", label="ACC_B (cosine-sim)")
    ax.axhline(acc_A.max() * 100, color=COLORS[0], lw=0.7, linestyle=":", alpha=0.6)
    best_A = int(acc_A.argmax());  best_B = int(acc_B.argmax())
    ax.axvline(best_A, color=COLORS[0], lw=0.7, linestyle=":", alpha=0.6)
    ax.axvline(best_B, color=COLORS[1], lw=0.7, linestyle=":", alpha=0.4)
    ax.set_title(f"Grounding ACC  (best A@L{best_A}={acc_A.max()*100:.1f}%,"
                 f"  best B@L{best_B}={acc_B.max()*100:.1f}%)")
    ax.set_xlabel("Layer"); ax.set_ylabel("ACC (%)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    # ── (0,1) Mean Distance ──
    ax = axes[0, 1]
    ax.plot(layers, dist_A, color=COLORS[0], lw=1.8, label="Dist_A")
    ax.plot(layers, dist_B, color=COLORS[1], lw=1.8, linestyle="--", label="Dist_B")
    ax.set_title("Mean Euclidean Distance to GT")
    ax.set_xlabel("Layer"); ax.set_ylabel("Distance (normalized)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ── (1,0) Attention Entropy ──
    ax = axes[1, 0]
    ax.plot(layers, ent_m, color=COLORS[2], lw=1.8, label="Entropy (mean)")
    ax.fill_between(layers, ent_m - ent_s, ent_m + ent_s,
                    color=COLORS[2], alpha=0.15, label="±1 std")
    ax.set_title("Visual-patch Attention Entropy  (lower = more concentrated)")
    ax.set_xlabel("Layer"); ax.set_ylabel("Entropy (nats)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ── (1,1) ACC_A by ui_type ──
    ax = axes[1, 1]
    for i, (ut, curve) in enumerate(sorted(per_type_acc.items())):
        c = COLORS[i % len(COLORS)]
        ax.plot(layers, curve * 100, color=c, lw=1.4, label=ut)
    ax.set_title("ACC_A by UI Type")
    ax.set_xlabel("Layer"); ax.set_ylabel("ACC (%)")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[saved] figure → {save_path}")
    else:
        plt.savefig("/tmp/probe_curves.png", dpi=150, bbox_inches="tight")
        print("[saved] figure → /tmp/probe_curves.png  (use --save_fig to specify path)")

    plt.close(fig)


# ---------------------------------------------------------------------------
# 打印汇总表
# ---------------------------------------------------------------------------

def print_summary(stats, n_layers):
    print(f"\n{'L':>4} | {'ACC_A':>7} | {'ACC_B':>7} | {'Dist_A':>7} | {'Dist_B':>7} | {'Entropy':>8} | {'n':>5}")
    print("-" * 60)
    for li in range(n_layers):
        s = stats[li]
        print(f"{li:>4} | {s['acc_A']*100:>6.2f}% | {s['acc_B']*100:>6.2f}% | "
              f"{s['dist_A']:>7.4f} | {s['dist_B']:>7.4f} | "
              f"{s['entropy_mean']:>8.4f} | {s['n_A']:>5}")

    best_A = max(range(n_layers), key=lambda l: stats[l]["acc_A"])
    best_B = max(range(n_layers), key=lambda l: stats[l]["acc_B"])
    low_e  = min(range(n_layers), key=lambda l: stats[l]["entropy_mean"])
    print(f"\n★ Best ACC_A: layer {best_A}  ({stats[best_A]['acc_A']*100:.2f}%)")
    print(f"★ Best ACC_B: layer {best_B}  ({stats[best_B]['acc_B']*100:.2f}%)")
    print(f"★ Min Entropy: layer {low_e}  ({stats[low_e]['entropy_mean']:.4f})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Plot layer-wise probe curves from checkpoint(s)")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--ckpt", nargs="+",
                     help="一个或多个 rank checkpoint 文件路径（直接指定）")
    grp.add_argument("--output_base",
                     help="自动扫描 <output_base>.rank*.ckpt.json，e.g. results/sspro_probe.json")
    p.add_argument("--save_fig", default=None,
                   help="图片保存路径（不填则存到 /tmp/probe_curves.png）")
    p.add_argument("--no_table", action="store_true",
                   help="不打印逐层数字表")
    return p.parse_args()


def main():
    args = parse_args()

    if args.output_base:
        ckpt_paths = auto_find_ckpts(args.output_base)
        if not ckpt_paths:
            print(f"[error] no checkpoints found matching {args.output_base}.rank*.ckpt.json")
            return
    else:
        ckpt_paths = args.ckpt

    print(f"Loading {len(ckpt_paths)} checkpoint(s):")
    entries = load_ckpts(ckpt_paths)
    if not entries:
        print("[error] no valid entries found in checkpoints")
        return
    print(f"Total valid samples: {len(entries)}")

    stats, n_layers = compute_layer_stats(entries)
    per_type_acc    = compute_per_type_acc(entries, n_layers)

    if not args.no_table:
        print_summary(stats, n_layers)

    suffix = f"  [{len(ckpt_paths)} rank(s)]"
    plot_curves(stats, n_layers, per_type_acc,
                save_path=args.save_fig, title_suffix=suffix)


if __name__ == "__main__":
    main()
