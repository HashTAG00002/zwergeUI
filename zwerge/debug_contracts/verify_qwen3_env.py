#!/usr/bin/env python3
"""
verify_qwen3_env.py
=====================
验证两个问题（在 qwen3 / qwen3-verl 两个 conda 环境中运行）：

问题 1: output_hidden_states=True 是否真的返回 non-None hidden_states？
  - qwen3-verl (torch 2.9.1): Qwen3VLModel.forward 会 silently drop hidden_states
    导致 Qwen3VLForConditionalGeneration.forward(output_hidden_states=True).hidden_states = None
  - qwen3 (torch 2.8.0): 同一代码是否已修复？

问题 2: PatchEmbed Conv3d 性能问题
  - qwen3-verl 下 Conv3d 处理 N=large_N patches 的 batch 极慢（~88937ms），
    因为 N 个独立 1×1×1 spatial size 使得 cuDNN GEMM 退化
  - qwen3 (torch 2.8.0) 是否已修复 Conv3d 在这种退化场景下的速度？
  - 对比：原始 Conv3d 前向 vs F.linear 等价操作，与 qwen3-verl 的结果对比

使用 Qwen3-VL-2B-Instruct（最小模型，快速验证），不加载完整模型权重，
只测 vision encoder 部分以加快速度。

运行方式：
  # 在 qwen3 环境下：
  conda run -n qwen3 python verify_qwen3_env.py
  # 或直接激活环境后运行：
  python verify_qwen3_env.py
"""

import sys
import time
import types
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 70)
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
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
print("\n" + "=" * 70)
print("【问题 1】output_hidden_states=True 是否返回 non-None hidden_states?")
print("=" * 70)

try:
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, AutoTokenizer
    print("[OK] Qwen3VLForConditionalGeneration 导入成功")
except ImportError as e:
    print(f"[FAIL] Qwen3VL 导入失败: {e}")
    sys.exit(1)

print(f"\n加载模型: {MODEL_PATH}")
print("(加载到 CPU，仅用于功能验证，不需要 GPU)")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map=device,
)
model.eval()
print(f"[OK] 模型加载完成")
print(f"     hidden_size: {model.config.text_config.hidden_size}")
print(f"     num_hidden_layers: {model.config.text_config.num_hidden_layers}")

# 构造一个最小的 dummy 输入（不含图片，只有文字，以快速测试 hidden_states 是否返回）
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
dummy_text = "Hello, what is in this image?"
inputs = tokenizer(dummy_text, return_tensors="pt").to(device)

print(f"\n[测试 1a] 传入 output_hidden_states=True，检查返回值...")
with torch.no_grad():
    outputs = model(
        **inputs,
        output_hidden_states=True,
        return_dict=True,
    )

hidden_states = outputs.hidden_states
if hidden_states is None:
    print("[RESULT] ❌ outputs.hidden_states = None  （bug 仍然存在！）")
    q1_result = "BUG: hidden_states=None even with output_hidden_states=True"
elif isinstance(hidden_states, (tuple, list)):
    n_layers = len(hidden_states)
    first_shape = hidden_states[0].shape if hidden_states[0] is not None else None
    last_shape = hidden_states[-1].shape if hidden_states[-1] is not None else None
    print(f"[RESULT] ✅ outputs.hidden_states 非 None!")
    print(f"          type: {type(hidden_states).__name__}")
    print(f"          len (num_layers+1): {n_layers}")
    print(f"          hidden_states[0].shape: {first_shape}")
    print(f"          hidden_states[-1].shape: {last_shape}")
    # 检查是否全是非None
    non_none = sum(1 for hs in hidden_states if hs is not None)
    print(f"          非None的层数: {non_none}/{n_layers}")
    q1_result = f"FIXED: hidden_states is tuple of {n_layers} tensors, all non-None: {non_none == n_layers}"
else:
    print(f"[RESULT] ⚠️  outputs.hidden_states 类型意外: {type(hidden_states)}")
    q1_result = f"UNEXPECTED: type={type(hidden_states)}"

print(f"\n[测试 1b] 通过 model.model (Qwen3VLModel) 直接测试...")
# 测试 Qwen3VLModel 内层是否 drop hidden_states
with torch.no_grad():
    # Qwen3VLModel.forward() 直接调用
    base_outputs = model.model(
        **inputs,
        output_hidden_states=True,
        return_dict=True,
    )
base_hs = getattr(base_outputs, 'hidden_states', 'attribute_not_found')
if base_hs == 'attribute_not_found':
    print("[RESULT] ❌ Qwen3VLModel 返回值没有 hidden_states 属性")
elif base_hs is None:
    print("[RESULT] ❌ Qwen3VLModel.hidden_states = None  （Qwen3VLModel 内部丢弃了 hidden_states）")
    q1_inner_result = "Qwen3VLModel.forward() SILENTLY DROPS hidden_states"
elif isinstance(base_hs, (tuple, list)):
    print(f"[RESULT] ✅ Qwen3VLModel.hidden_states 非 None! len={len(base_hs)}")
    q1_inner_result = f"Qwen3VLModel.forward() PASSES THROUGH hidden_states (len={len(base_hs)})"
else:
    q1_inner_result = f"Unexpected: {type(base_hs)}"
    print(f"[RESULT] ⚠️ {q1_inner_result}")

# ─────────────────────────────────────────────────────────────────────────────
# 问题 2: PatchEmbed Conv3d 性能问题
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("【问题 2】PatchEmbed Conv3d 性能验证")
print("  （测试 N_patches 大 batch 下 Conv3d vs F.linear 速度差异）")
print("=" * 70)

# 获取 vision encoder 的 patch_embed
try:
    patch_embed = model.model.visual.patch_embed
    proj = patch_embed.proj
    print(f"[OK] 找到 patch_embed.proj: {type(proj).__name__}")
    if isinstance(proj, nn.Conv3d):
        print(f"     Conv3d: in_channels={proj.in_channels}, out_channels={proj.out_channels}")
        print(f"     kernel_size={proj.kernel_size}, stride={proj.stride}")
        # 计算 flat_dim
        flat_dim = proj.in_channels
        for k in proj.kernel_size:
            flat_dim *= k
        print(f"     flat_dim (C*T*H*W) = {flat_dim}")
        print(f"     out_channels = {proj.out_channels}")
        is_conv3d = True
    else:
        print(f"[INFO] patch_embed.proj 不是 Conv3d，而是 {type(proj).__name__}")
        print("       可能已经是 Linear 实现，不需要 patch（新版本可能已经修复）")
        is_conv3d = False
except AttributeError as e:
    print(f"[FAIL] 无法访问 patch_embed: {e}")
    is_conv3d = False

if is_conv3d:
    # 构建符合 Qwen3-VL PatchEmbed 输入格式的 dummy 数据
    # 真实场景：1920×1080 图片 → N_patches ≈ 1920/2/16 × 1080/2/16 = 60 × 34 ≈ 2040 patches
    # N = 2040（模拟真实大图，但不加载图片）
    # 但 qwen3-verl 测的是 N=11360（2张图片），我们也用类似大小
    N_patches_list = [2040, 5680, 11360]  # 1张/5张/10张图片等效
    
    C = proj.in_channels
    T, H, W = proj.kernel_size  # kernel_size == stride for PatchEmbed
    flat_dim = C * T * H * W
    out_ch = proj.out_channels
    
    print(f"\n参数：C={C}, T={T}, H={H}, W={W}, flat_dim={flat_dim}, out_ch={out_ch}")
    
    if torch.cuda.is_available():
        proj_cuda = proj.to(device)
        
        for N in N_patches_list:
            print(f"\n--- N_patches = {N} ---")
            
            # dummy 输入：[N, flat_dim]（已 flat，符合 Qwen3VL processor 输出格式）
            dummy_input_flat = torch.randn(N, flat_dim, dtype=torch.bfloat16, device=device)
            # reshape 为 Conv3d 需要的 [N, C, T, H, W]
            dummy_input_3d = dummy_input_flat.view(N, C, T, H, W)
            
            # ── 方法 A: 原始 Conv3d ──────────────────────────────────────────
            # 预热
            for _ in range(3):
                with torch.no_grad():
                    _ = proj_cuda(dummy_input_3d).view(N, -1)
            torch.cuda.synchronize()
            
            # 计时（N_runs 次，取平均）
            N_RUNS = 5
            t0 = time.perf_counter()
            for _ in range(N_RUNS):
                with torch.no_grad():
                    out_conv3d = proj_cuda(dummy_input_3d).view(N, -1)
            torch.cuda.synchronize()
            t_conv3d_ms = (time.perf_counter() - t0) * 1000 / N_RUNS
            
            # ── 方法 B: F.linear 等价实现 ────────────────────────────────────
            w_flat = proj_cuda.weight.view(out_ch, -1)  # [out_ch, flat_dim]
            b = proj_cuda.bias
            
            # 预热
            for _ in range(3):
                with torch.no_grad():
                    _ = F.linear(dummy_input_flat, w_flat, b)
            torch.cuda.synchronize()
            
            t0 = time.perf_counter()
            for _ in range(N_RUNS):
                with torch.no_grad():
                    out_linear = F.linear(dummy_input_flat, w_flat, b)
            torch.cuda.synchronize()
            t_linear_ms = (time.perf_counter() - t0) * 1000 / N_RUNS
            
            # ── 验证数学等价性 ─────────────────────────────────────────────
            max_diff = (out_conv3d - out_linear).abs().max().item()
            
            speedup = t_conv3d_ms / t_linear_ms if t_linear_ms > 0 else float('inf')
            
            print(f"  Conv3d time:   {t_conv3d_ms:.3f} ms")
            print(f"  F.linear time: {t_linear_ms:.3f} ms")
            print(f"  Max diff (Conv3d vs Linear): {max_diff:.6e}")
            print(f"  Speedup (Conv3d→Linear): {speedup:.1f}x")
            
            if t_conv3d_ms > 1000:
                status = "❌ STILL SLOW (>1000ms)"
            elif t_conv3d_ms > 100:
                status = "⚠️ SLOW (100-1000ms)"
            elif t_conv3d_ms > 10:
                status = "⚠️ MODERATE (10-100ms)"
            else:
                status = "✅ FAST (<10ms)"
            print(f"  Conv3d 速度状态: {status}")
            
            if speedup < 5:
                fix_status = "✅ FIXED: Conv3d已经不慢了（差距 <5x，原问题已解决）"
            elif speedup < 100:
                fix_status = "⚠️ IMPROVED but still slower than Linear"
            else:
                fix_status = f"❌ BUG仍存在: Conv3d 比 F.linear 慢 {speedup:.0f}x"
            print(f"  结论: {fix_status}")
    else:
        print("[SKIP] 无 GPU，跳过 Conv3d 性能测试")
        
    # ── 额外测试：验证 types.MethodType monkey-patch 是否正常工作 ────────────────
    print(f"\n--- Monkey-patch 验证（复用 modeling_guiowl.py 相同代码）---")
    
    # 保存原始 forward
    original_forward = patch_embed.forward
    original_forward_qualname = getattr(original_forward, '__qualname__', str(original_forward))
    print(f"  原始 forward: {original_forward_qualname}")
    
    # 应用 monkey-patch（与 modeling_guiowl.py 完全相同的代码）
    def _fast_patch_embed_forward(self_pe, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self_pe.proj.weight.dtype
        x = hidden_states.to(dtype=target_dtype)
        w = self_pe.proj.weight.view(self_pe.proj.weight.shape[0], -1)
        b = self_pe.proj.bias
        return F.linear(x, w, b)
    
    patch_embed.forward = types.MethodType(_fast_patch_embed_forward, patch_embed)
    patched_qualname = patch_embed.forward.__func__.__qualname__
    print(f"  Patch 后 forward.__func__.__qualname__: {patched_qualname}")
    patch_ok = '_fast_patch_embed_forward' in patched_qualname
    print(f"  Patch 应用: {'✅ 成功' if patch_ok else '❌ 失败'}")

else:
    print("\n[INFO] patch_embed.proj 不是 Conv3d，跳过 Conv3d 性能测试")
    print("       这可能意味着：")
    print("       (a) 当前 transformers 版本已经将 PatchEmbed 改为 Linear 实现")
    print("       (b) 或者模型结构有变化")
    # 检查具体是什么
    if hasattr(model.model, 'visual') and hasattr(model.model.visual, 'patch_embed'):
        pe = model.model.visual.patch_embed
        print(f"\n  patch_embed 的类: {type(pe).__name__}")
        print(f"  patch_embed 的所有属性:")
        for attr in dir(pe):
            if not attr.startswith('_'):
                val = getattr(pe, attr, None)
                if isinstance(val, nn.Module):
                    print(f"    {attr}: {type(val).__name__}")

# ─────────────────────────────────────────────────────────────────────────────
# 最终汇总报告
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("【最终汇总报告】")
print("=" * 70)
print(f"环境: PyTorch {torch.__version__}, Transformers {transformers.__version__}")
print(f"设备: {device}")
print()
print(f"问题 1 (output_hidden_states):")
print(f"  顶层 (Qwen3VLForConditionalGeneration): {q1_result}")
try:
    print(f"  内层 (Qwen3VLModel):                  {q1_inner_result}")
except NameError:
    pass
print()
print(f"问题 2 (Conv3d 性能):")
if is_conv3d:
    print(f"  patch_embed.proj 仍然是 nn.Conv3d")
    if torch.cuda.is_available():
        print(f"  → 请查看上方每个 N_patches 规模的计时结果")
    else:
        print(f"  → 无 GPU，跳过性能测试")
else:
    print(f"  patch_embed.proj 不是 nn.Conv3d（可能已改为 Linear 实现）")
    print(f"  → Conv3d 性能问题可能已从 transformers 层面修复")
print("=" * 70)
