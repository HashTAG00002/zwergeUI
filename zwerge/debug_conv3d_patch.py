"""
深度诊断 Conv3d Monkey-Patch 正确性
=====================================
关键问题：
  Qwen3VLVisionPatchEmbed.forward() 接收的 hidden_states 到底是什么形状？

根据 transformers/models/qwen3_vl/modeling_qwen3_vl.py 源码：
  Qwen3VLVisionPatchEmbed.forward(hidden_states):
    hidden_states = hidden_states.view(-1, self.in_channels, self.temporal_patch_size,
                                        self.patch_size, self.patch_size)
    hidden_states = self.proj(hidden_states.to(dtype)).view(-1, self.embed_dim)

即：传入的 hidden_states 是 [N, in_channels * temporal_patch_size * patch_size * patch_size]
   forward 内部先 reshape 成 [N, C, T, H, W] 再送给 Conv3d

我们的 _fast_patch_embed_forward 接收的 hidden_states 同样是 [N, flat]（还没被reshape）
但我们的代码直接做 F.linear(x, w, b)，其中 w = conv.weight.view(out_ch, -1)

问题：conv.weight 的形状是 [out_ch, C, T, H, W]，展平后是 [out_ch, C*T*H*W]
     输入 x = hidden_states [N, C*T*H*W]（flat）
     F.linear(x, w, b) = x @ w.T + b → [N, out_ch]  ✅

但等等：Conv3d 的输入是 [N, C, T, H, W]（不是 [N, C*T*H*W]！）
     Conv3d stride=kernel_size 时：output = sum over C*T*H*W position of weight * input
     这等价于 F.linear(input.view(N, C*T*H*W), weight.view(out_ch, C*T*H*W)) 
     ——前提是 spatial dimension ordering 要对。

关键疑问：当 Conv3d kernel=(T,H,W) stride=(T,H,W) 时，weight 的内存顺序是 [out_ch, C, T, H, W]
对应展平是 C×T×H×W 的 contiguous layout。
但原始 forward 先把 [N, C*T*H*W] reshape 成 [N, C, T, H, W] 再做 Conv3d。
我们直接用 [N, C*T*H*W] flat 做 F.linear，这是否等价？

答案：等价！
  - Conv3d([N, C, T, H, W]) 等价于对每个 spatial position 做 linear：
    out[n] = W.view(out_ch, C*T*H*W) @ in[n].view(C*T*H*W) + b
  - F.linear([N, C*T*H*W], W.view(out_ch, C*T*H*W)) 结果完全相同
  - 因为 in[n].view(C*T*H*W) 就是 in[n, :, :, :].reshape(-1)，contiguous，ordering 相同

但还有一个问题：_fast_patch_embed_forward 收到的 hidden_states 是 [N, flat] 还是已经 [N, C, T, H, W]？
这是真正要检查的！
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

print("=" * 70)
print("Part 1: 验证 Conv3d → F.linear 数学等价性")
print("=" * 70)

C, T, H, W = 3, 2, 16, 16
out_ch = 1152
flat = C * T * H * W  # 1536
N = 128

conv = nn.Conv3d(C, out_ch, kernel_size=(T, H, W), stride=(T, H, W), bias=True)
conv.eval()

# Case A: 输入是 [N, flat]（processor 实际输出形式）
x_flat = torch.randn(N, flat)

# 原始路径：先 reshape 再 Conv3d
x_3d = x_flat.view(N, C, T, H, W)
with torch.no_grad():
    out_conv = conv(x_3d).view(N, out_ch)

# 我们的 _fast_patch_embed_forward
with torch.no_grad():
    x_in = x_flat.to(dtype=conv.weight.dtype)
    w = conv.weight.view(conv.weight.shape[0], -1)
    out_linear = F.linear(x_in, w, conv.bias)

print(f"Input to patch_embed: [N={N}, flat={flat}] (processor output)")
print(f"Conv3d output shape:  {out_conv.shape}")
print(f"F.linear output shape: {out_linear.shape}")
print(f"Max abs diff: {(out_conv - out_linear).abs().max().item():.2e}")
print(f"All close (atol=1e-5): {torch.allclose(out_conv, out_linear, atol=1e-5)}")
print()

print("=" * 70)
print("Part 2: 检查 Qwen3VL PatchEmbed 的实际输入 shape")
print("=" * 70)

try:
    from transformers import Qwen3VLForConditionalGeneration
    import inspect
    try:
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionPatchEmbed
        src = inspect.getsource(Qwen3VLVisionPatchEmbed.forward)
        print("Qwen3VLVisionPatchEmbed.forward source:")
        print(src[:2000])
    except Exception as e:
        print(f"Cannot get source: {e}")

    # 加载真实模型做一次 forward，观察实际 patch_embed.forward 的输入
    model_path = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/.hdd/models/huggingface.co/GUI_Agents/GUI-Owl-1.5-8B-Instruct"
    print(f"\n尝试加载 {model_path} ...")
    # 只加载 config，不加载权重（快速）
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_path)
    print(f"Config loaded. model_type: {config.model_type}")
    vis_cfg = getattr(config, 'vision_config', None)
    if vis_cfg:
        print(f"  patch_size: {getattr(vis_cfg, 'patch_size', '?')}")
        print(f"  temporal_patch_size: {getattr(vis_cfg, 'temporal_patch_size', '?')}")
        print(f"  in_channels: {getattr(vis_cfg, 'in_channels', '?')}")
        print(f"  hidden_size: {getattr(vis_cfg, 'hidden_size', '?')}")

except ImportError as e:
    print(f"Qwen3VL not available in this env: {e}")
    sys.exit(0)

print()
print("=" * 70)
print("Part 3: Monkey-patch 之后是否改变了 conv.weight 的值")
print("=" * 70)

import types

class FakePatchEmbed(nn.Module):
    def __init__(self, C, out_ch, T, H, W):
        super().__init__()
        self.proj = nn.Conv3d(C, out_ch, kernel_size=(T,H,W), stride=(T,H,W), bias=True)
        self.embed_dim = out_ch
        self.in_channels = C
        self.temporal_patch_size = T
        self.patch_size = H
        
    def forward(self, hidden_states):
        # 原始 Qwen3VL 实现
        hidden_states = hidden_states.view(-1, self.in_channels,
                                           self.temporal_patch_size,
                                           self.patch_size, self.patch_size)
        hidden_states = self.proj(hidden_states.to(dtype=self.proj.weight.dtype))
        hidden_states = hidden_states.view(-1, self.embed_dim)
        return hidden_states

pe = FakePatchEmbed(C, out_ch, T, H, W)
pe.eval()

# Record original weight checksum
w_before = pe.proj.weight.clone()

# Apply monkey-patch
def _fast_patch_embed_forward(self_pe, hidden_states):
    target_dtype = self_pe.proj.weight.dtype
    x = hidden_states.to(dtype=target_dtype)
    w = self_pe.proj.weight.view(self_pe.proj.weight.shape[0], -1)
    b = self_pe.proj.bias
    return F.linear(x, w, b)

pe.forward = types.MethodType(_fast_patch_embed_forward, pe)

w_after = pe.proj.weight.clone()
print(f"Weight unchanged after patch: {torch.equal(w_before, w_after)}")

# Test with original-style input [N, C*T*H*W]
x_test = torch.randn(N, flat)

# Original (before patch) — recreate without patch
pe_orig = FakePatchEmbed(C, out_ch, T, H, W)
pe_orig.proj.weight.data = w_before.clone()
pe_orig.proj.bias.data = pe.proj.bias.clone()
pe_orig.eval()

with torch.no_grad():
    out_orig = pe_orig(x_test)
    out_patched = pe(x_test)

print(f"Original forward input shape: {x_test.shape}")
print(f"Patched forward output shape: {out_patched.shape}")
print(f"Max abs diff orig vs patched: {(out_orig - out_patched).abs().max().item():.2e}")
print(f"Numerically equivalent: {torch.allclose(out_orig, out_patched, atol=1e-5)}")

print()
print("=" * 70)
print("Part 4: 如果 Qwen3VL 传入的是 [N, C, T, H, W] 而不是 [N, flat]，会出什么问题？")
print("=" * 70)

# 如果有人传入了 [N, C, T, H, W]，而我们用 F.linear，会怎样？
x_3d_wrong = torch.randn(N, C, T, H, W)
try:
    out_wrong = F.linear(x_3d_wrong.view(x_3d_wrong.shape[0], -1),
                         w_before.view(out_ch, -1), pe.proj.bias)
    print("If input is [N,C,T,H,W]: F.linear after reshape gives same result as Conv3d")
    # Note: view(N, -1) on [N, C, T, H, W] gives [N, C*T*H*W] = same as [N, flat]
    print(f"  x_3d_wrong.view(N,-1) shape: {x_3d_wrong.view(N, -1).shape}")
    print(f"  This IS correct because view(N,-1) flattens the same way as C*T*H*W order")
except Exception as e:
    print(f"Error: {e}")

print()
print("=" * 70)
print("CONCLUSION")
print("=" * 70)
print("""
Conv3d monkey-patch 数学等价性：✅ 完全正确

真正可能导致 Qwen3-VL 性能差的原因（与 Conv3d patch 无关）：

1. deepstack hidden state 注入顺序问题？
   - Qwen3-VL 在 L8/L16/L24 重注入视觉 embedding
   - 如果 hook 在这些层也触发了，hidden state 包含了重注入的视觉信息 → 正确
   - 但如果 deepstack_visual_embeds 传递有问题 → hidden state 不含 deepstack → 性能差

2. _run_language_model 里的 deepstack_visual_embeds 是否正确传递？
   - 检查 self.model.language_model() 的 visual_pos_masks / deepstack_visual_embeds 参数

3. gradient checkpointing 设置问题？
   - Qwen3-VL 强制 gradient_checkpointing=False
   - 而 uitars 可以开 GC → 训练步数可能不同？

请检查 _run_language_model 是否正确传了 deepstack_visual_embeds
""")
