#!/usr/bin/env python3
"""
verify_qwen3_env_fast.py
=========================
快速版本验证脚本（只测 N=2040，避免在 qwen3-verl 下 Conv3d 极慢导致超时）。

使用场景：qwen3-verl 环境下运行（因为 Conv3d 极慢，需要等待）
与 verify_qwen3_env.py 相同逻辑，仅减少 N_patches 测试规模。
"""

import sys
import time
import types
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 70)
print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
try:
    import transformers
    print(f"Transformers: {transformers.__version__}")
except ImportError:
    print("Transformers: NOT INSTALLED")
print("=" * 70)

MODEL_PATH = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/.hdd/models/huggingface.co/Qwen/Qwen3-VL-2B-Instruct"

# ─────────────────────────────────────────────────────────────────────────────
# 问题 1: output_hidden_states=True 是否真的返回 non-None hidden_states
# ─────────────────────────────────────────────────────────────────────────────
print("\n【问题 1】output_hidden_states=True 是否返回 non-None hidden_states?")
print("-" * 70)

try:
    from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer
    print("[OK] Qwen3VLForConditionalGeneration 导入成功")
except ImportError as e:
    print(f"[FAIL] Qwen3VL 导入失败: {e}")
    sys.exit(1)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"加载模型到 {device}...")

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map=device,
)
model.eval()
print(f"[OK] 模型加载完成 (hidden={model.config.text_config.hidden_size}, layers={model.config.text_config.num_hidden_layers})")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
inputs = tokenizer("Hello", return_tensors="pt").to(device)

# 测试1a：顶层
with torch.no_grad():
    out = model(**inputs, output_hidden_states=True, return_dict=True)
hs = out.hidden_states
if hs is None:
    print("❌ 顶层 hidden_states = None (BUG)")
else:
    print(f"✅ 顶层 hidden_states: {type(hs).__name__}, len={len(hs)}, shape[0]={hs[0].shape}")

# 测试1b：Qwen3VLModel 内层
with torch.no_grad():
    base_out = model.model(**inputs, output_hidden_states=True, return_dict=True)
base_hs = getattr(base_out, 'hidden_states', None)
if base_hs is None:
    print("❌ Qwen3VLModel.hidden_states = None (inner BUG - Qwen3VLModel drops it)")
else:
    print(f"✅ Qwen3VLModel.hidden_states: len={len(base_hs)}")

# ─────────────────────────────────────────────────────────────────────────────
# 问题 2: Conv3d 性能（只测 N=2040 快速验证）
# ─────────────────────────────────────────────────────────────────────────────
print("\n【问题 2】PatchEmbed Conv3d vs F.linear 性能对比（N=2040）")
print("-" * 70)

patch_embed = model.model.visual.patch_embed
proj = patch_embed.proj
print(f"patch_embed.proj 类型: {type(proj).__name__}")

if isinstance(proj, nn.Conv3d):
    C = proj.in_channels
    T, H, W = proj.kernel_size
    flat_dim = C * T * H * W
    out_ch = proj.out_channels
    print(f"Conv3d: in={C}, out={out_ch}, kernel=({T},{H},{W}), flat_dim={flat_dim}")
    
    if torch.cuda.is_available():
        proj_cuda = proj.to(device)
        
        # N=2040（1 张 1920×1080 图等效）
        N = 2040
        dummy_flat = torch.randn(N, flat_dim, dtype=torch.bfloat16, device=device)
        dummy_3d = dummy_flat.view(N, C, T, H, W)
        w_flat = proj_cuda.weight.view(out_ch, -1)
        b = proj_cuda.bias
        
        # 预热
        print("预热中...")
        for _ in range(2):
            with torch.no_grad():
                _ = proj_cuda(dummy_3d)
        torch.cuda.synchronize()
        print("预热完成，开始计时...")
        
        # Conv3d 计时（只跑 1 次，因为 verl 可能很慢）
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out_c = proj_cuda(dummy_3d).view(N, -1)
        torch.cuda.synchronize()
        t_conv3d = (time.perf_counter() - t0) * 1000
        
        # F.linear 计时
        t0 = time.perf_counter()
        with torch.no_grad():
            out_l = F.linear(dummy_flat, w_flat, b)
        torch.cuda.synchronize()
        t_linear = (time.perf_counter() - t0) * 1000
        
        max_diff = (out_c - out_l).abs().max().item()
        speedup = t_conv3d / max(t_linear, 1e-6)
        
        print(f"\n  N_patches = {N}")
        print(f"  Conv3d time:   {t_conv3d:.3f} ms")
        print(f"  F.linear time: {t_linear:.3f} ms")
        print(f"  Speedup:       {speedup:.1f}x")
        print(f"  Max diff:      {max_diff:.3e}")
        
        if t_conv3d < 10:
            print(f"  ✅ Conv3d 速度正常 (<10ms) — qwen3 环境已修复")
        elif t_conv3d < 100:
            print(f"  ⚠️ Conv3d 偏慢 (10~100ms)")
        elif t_conv3d < 1000:
            print(f"  ⚠️ Conv3d 很慢 (100ms~1s)")
        else:
            print(f"  ❌ Conv3d 极慢 (>1s) — 原始 qwen3-verl bug 仍存在")
            
        if speedup > 1000:
            print(f"  ❌ speedup={speedup:.0f}x — 原始 bug（cuDNN退化）仍存在")
        elif speedup > 100:
            print(f"  ⚠️ speedup={speedup:.0f}x — 有改善但仍需 monkey-patch")
        elif speedup > 10:
            print(f"  ⚠️ speedup={speedup:.0f}x — 有差距，建议保留 monkey-patch")
        else:
            print(f"  ✅ speedup={speedup:.1f}x — Conv3d 已接近 F.linear，patch 不是必须的")
    else:
        print("[SKIP] 无 GPU")
else:
    print(f"patch_embed.proj = {type(proj).__name__} (非 Conv3d)")
    print("→ Conv3d 性能问题已从 transformers 层面修复！不再需要 monkey-patch。")
    # 检查是什么类型
    if hasattr(proj, 'weight'):
        print(f"  weight.shape: {proj.weight.shape}")

print("\n" + "=" * 70)
print("【完成】")
print("=" * 70)
