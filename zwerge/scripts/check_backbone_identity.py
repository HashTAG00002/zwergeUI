#!/usr/bin/env python3
"""
check_backbone_identity.py
==========================
验证 ZwerGe retrofit checkpoint 的 backbone 权重
是否与原始预训练模型完全一致（bit-level，bf16 精度下）。

比对策略：
  ckpt[key].to(bf16)  ==  orig[key].to(bf16)
  期望：对所有 backbone key 完全一致（backbone 被完全冻结）。

排除的 key（不属于"纯backbone"）：
  layerwise_grounding_head.*  — ZwerGe grounding head（新增，有训练）
  model.embed_tokens.weight   — 新 token embedding（新增 token 行有训练）

性能优化：
  先检查各 safetensors 分片是否有 backbone key，如果整个分片都是 head key 则跳过。
  每个 tensor 用 torch.equal 做精确比对（无需 sha256，加载即比较，避免重复遍历）。

用法：
  conda run -n gui_actor python3 scripts/check_backbone_identity.py 2>&1 | tee /tmp/backbone_check.log
  # 后台：
  conda run -n gui_actor python3 scripts/check_backbone_identity.py > /tmp/backbone_check.log 2>&1 &
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from safetensors import safe_open

# ─────────────────────────────────────────────────────────────────
# 配置：6 个 ckpt（uitars × 2, guiowl × 2, uivenus × 2）
# ─────────────────────────────────────────────────────────────────
CKPT_ROOT = Path("/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/.hdd/ckpt/zwerge")
ORIG_ROOT = Path("/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/.hdd/models/huggingface.co/GUI_Agents")

CHECKPOINTS: List[Dict] = [
    # ── UI-TARS (Qwen2.5-VL backbone, frozen) ────────────────────
    {
        "label":    "uitars / A3-L18-27 / ckpt-1200",
        "ckpt_dir": CKPT_ROOT / "uitars7b_grounding50k_A3-gaussian_cos_meta_L18-27_20260520_095304" / "checkpoint-1200",
        "orig_dir": ORIG_ROOT / "UI-TARS-1.5-7B",
    },
    {
        "label":    "uitars / A3-L18-27 / ckpt-2193",
        "ckpt_dir": CKPT_ROOT / "uitars7b_grounding50k_A3-gaussian_cos_meta_L18-27_20260520_104330" / "checkpoint-2193",
        "orig_dir": ORIG_ROOT / "UI-TARS-1.5-7B",
    },
    # ── GUI-Owl-1.5 (Qwen3-VL backbone, frozen) ──────────────────
    {
        "label":    "guiowl / A3 / ckpt-400 (run1)",
        "ckpt_dir": CKPT_ROOT / "guiowl_grounding50k_A3-gaussian_cos_meta_20260522_034634" / "checkpoint-400",
        "orig_dir": ORIG_ROOT / "GUI-Owl-1.5-8B-Instruct",
    },
    {
        "label":    "guiowl / A3 / ckpt-400 (run2)",
        "ckpt_dir": CKPT_ROOT / "guiowl_grounding50k_A3-gaussian_cos_meta_20260521_204025" / "checkpoint-400",
        "orig_dir": ORIG_ROOT / "GUI-Owl-1.5-8B-Instruct",
    },
    # ── UI-Venus-1.5 (Qwen3-VL backbone, frozen) ─────────────────
    {
        "label":    "uivenus / A3 / ckpt-400 (run1)",
        "ckpt_dir": CKPT_ROOT / "uivenus_grounding50k_A3-gaussian_cos_meta_20260522_123032" / "checkpoint-400",
        "orig_dir": ORIG_ROOT / "UI-Venus-1.5-8B",
    },
    {
        "label":    "uivenus / A3 / ckpt-2193 (run2)",
        "ckpt_dir": CKPT_ROOT / "uivenus_grounding50k_A3-gaussian_cos_meta_20260522_031059" / "checkpoint-2193",
        "orig_dir": ORIG_ROOT / "UI-Venus-1.5-8B",
    },
]


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def load_index(model_dir: Path) -> Dict[str, str]:
    """返回 {param_key: safetensors_filename} 映射。"""
    idx_path = model_dir / "model.safetensors.index.json"
    if idx_path.exists():
        return json.loads(idx_path.read_text())["weight_map"]
    single = model_dir / "model.safetensors"
    if single.exists():
        with safe_open(str(single), framework="pt") as f:
            return {k: "model.safetensors" for k in f.keys()}
    raise FileNotFoundError(f"No safetensors index or single file found in {model_dir}")


def is_backbone_key(key: str) -> bool:
    """
    backbone key = 全部 key 排除以下（vocab size 因新增 token 不同，或属于 head）：
      layerwise_grounding_head.*              新增 grounding head，含训练权重
      model.embed_tokens.weight               Qwen2.5-VL embed（新 token 行被训练）
      model.language_model.embed_tokens.weight Qwen3-VL embed（同上）
      lm_head.weight                          vocab 输出层（行数随 vocab size 变化）
    """
    if key.startswith("layerwise_grounding_head."):
        return False
    # embed_tokens: Qwen2.5-VL path vs Qwen3-VL path
    if key in ("model.embed_tokens.weight",
               "model.language_model.embed_tokens.weight"):
        return False
    # lm_head: vocab size changed (new special tokens added)
    if key == "lm_head.weight":
        return False
    return True


def compare_ckpt_to_orig(label: str, ckpt_dir: Path, orig_dir: Path) -> bool:
    print(f"\n{'='*72}")
    print(f"  {label}")
    print(f"  ckpt : {ckpt_dir}")
    print(f"  orig : {orig_dir}")
    print(f"{'='*72}")
    t0 = time.time()

    if not ckpt_dir.exists():
        print(f"  [SKIP] ckpt_dir not found")
        return False
    if not orig_dir.exists():
        print(f"  [SKIP] orig_dir not found")
        return False

    ckpt_idx = load_index(ckpt_dir)
    orig_idx  = load_index(orig_dir)

    # Backbone keys = ckpt keys minus head & embed_tokens
    ckpt_bb = {k for k in ckpt_idx if is_backbone_key(k)}
    orig_bb  = {k for k in orig_idx  if k != "model.embed_tokens.weight"}

    only_ckpt = ckpt_bb - orig_bb
    only_orig  = orig_bb - ckpt_bb
    common     = ckpt_bb & orig_bb

    print(f"  Keys — ckpt_backbone: {len(ckpt_bb)}, orig_backbone: {len(orig_bb)}, common: {len(common)}")
    if only_ckpt:
        print(f"  [WARN] Only in ckpt ({len(only_ckpt)}): {sorted(only_ckpt)[:3]}")
    if only_orig:
        print(f"  [WARN] Only in orig ({len(only_orig)}): {sorted(only_orig)[:3]}")
    if not common:
        print("  [ERROR] No common keys to compare!")
        return False

    # Group common keys by (ckpt_file, orig_file) to open each file pair at most once
    # key -> (ckpt_file, orig_file)
    file_pairs: Dict[Tuple[str, str], List[str]] = {}
    for key in sorted(common):
        pair = (ckpt_idx[key], orig_idx[key])
        file_pairs.setdefault(pair, []).append(key)

    n_ok       = 0
    n_mismatch = 0
    mismatch_info: List[Tuple[str, float, float, str, str]] = []

    for (cf, of), keys in file_pairs.items():
        ckpt_path = str(ckpt_dir / cf)
        orig_path = str(orig_dir  / of)
        with safe_open(ckpt_path, framework="pt") as fc, \
             safe_open(orig_path,  framework="pt") as fo:
            for key in keys:
                ct = fc.get_tensor(key).to(torch.bfloat16)
                ot = fo.get_tensor(key).to(torch.bfloat16)

                if ct.shape != ot.shape:
                    mismatch_info.append((key, -1.0, -1.0, f"shape {ct.shape} vs {ot.shape}", ""))
                    n_mismatch += 1
                    continue

                if torch.equal(ct, ot):
                    n_ok += 1
                else:
                    diff = (ct.float() - ot.float()).abs()
                    mx   = diff.max().item()
                    mn   = diff.mean().item()
                    n_mismatch += 1
                    mismatch_info.append((key, mx, mn, "", f"{cf}"))

    elapsed = time.time() - t0

    if n_mismatch == 0:
        print(f"  [PASS ✅]  All {n_ok} backbone tensors are bit-level identical (bf16). ({elapsed:.1f}s)")
        return True
    else:
        print(f"  [FAIL ❌]  {n_mismatch}/{n_ok+n_mismatch} tensors differ! ({elapsed:.1f}s)")
        mismatch_info.sort(key=lambda x: x[1], reverse=True)
        print(f"  Top mismatches:")
        for key, mx, mn, note, fname in mismatch_info[:8]:
            if mx < 0:
                print(f"    {key}  → {note}")
            else:
                print(f"    {key}  max={mx:.4e}  mean={mn:.4e}  [{fname}]")
        return False


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = []
    for cfg in CHECKPOINTS:
        ok = compare_ckpt_to_orig(
            label    = cfg["label"],
            ckpt_dir = cfg["ckpt_dir"],
            orig_dir = cfg["orig_dir"],
        )
        results.append((cfg["label"], ok))

    print(f"\n{'='*72}")
    print("  SUMMARY")
    print(f"{'='*72}")
    all_pass = True
    for label, ok in results:
        status = "PASS ✅" if ok else "FAIL ❌"
        print(f"  [{status}]  {label}")
        all_pass = all_pass and ok
    print(f"{'='*72}")
    print(f"  Overall: {'ALL PASS ✅' if all_pass else 'SOME FAILED ❌'}")
    print(f"{'='*72}\n")
    sys.exit(0 if all_pass else 1)
