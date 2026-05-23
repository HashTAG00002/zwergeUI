"""
深度诊断 Qwen3-VL deepstack 传递正确性
=========================================
关键假设：modeling_guiowl.py 的 _run_language_model 里通过
  self.model.language_model(..., deepstack_visual_embeds=...)
传递 deepstack。

问题：Qwen3VLTextModel.forward 的实际签名里有 deepstack_visual_embeds 吗？
      如果没有，参数会被 **extra_kwargs 静默忽略，deepstack 注入根本没有发生！
"""

import sys, inspect

try:
    from transformers import Qwen3VLForConditionalGeneration
    from transformers.models.qwen3_vl import modeling_qwen3_vl as mod
    print("=== Qwen3VL module loaded ===")
    print()

    # 1. 检查 Qwen3VLTextModel.forward 签名
    text_model_cls = getattr(mod, 'Qwen3VLTextModel', None)
    if text_model_cls:
        sig = inspect.signature(text_model_cls.forward)
        params = list(sig.parameters.keys())
        print("Qwen3VLTextModel.forward params:")
        for p in params:
            print(f"  {p}")
        print()
        has_deepstack = 'deepstack_visual_embeds' in params
        has_visual_pos = 'visual_pos_masks' in params
        print(f"Has deepstack_visual_embeds param: {has_deepstack}")
        print(f"Has visual_pos_masks param: {has_visual_pos}")
        if not has_deepstack:
            print("!!! WARNING: deepstack_visual_embeds NOT in Qwen3VLTextModel.forward !!!")
            print("    Our _run_language_model call passes it via **kwargs but it's SILENTLY IGNORED")
    else:
        print("Qwen3VLTextModel not found in module")

    print()

    # 2. 检查 language_model 是什么类型
    # In Qwen3VL, model.language_model could be Qwen3VLTextModel or Qwen3TextModel
    # Let's check Qwen3VLModel.forward for deepstack handling
    qwen3vl_model_cls = getattr(mod, 'Qwen3VLModel', None)
    if qwen3vl_model_cls:
        print("Qwen3VLModel.forward params:")
        sig2 = inspect.signature(qwen3vl_model_cls.forward)
        for p in list(sig2.parameters.keys()):
            print(f"  {p}")
    print()

    # 3. 检查 Qwen3VLForConditionalGeneration 中 deepstack 处理
    print("=== Scanning for deepstack references in Qwen3VL source ===")
    src = inspect.getsource(mod)
    # Find all occurrences of deepstack
    lines = src.split('\n')
    deepstack_lines = [(i+1, l.strip()) for i, l in enumerate(lines) if 'deepstack' in l.lower()]
    print(f"Total deepstack references: {len(deepstack_lines)}")
    for lineno, line in deepstack_lines[:40]:
        print(f"  L{lineno}: {line}")

except ImportError as e:
    print(f"Cannot import Qwen3VL: {e}")
    sys.exit(1)

print()
print("=== Checking actual language_model attribute ===")
try:
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(
        "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/.hdd/models/huggingface.co/GUI_Agents/GUI-Owl-1.5-8B-Instruct"
    )
    model = Qwen3VLForConditionalGeneration(cfg)
    lm = model.model.language_model
    print(f"model.model.language_model type: {type(lm).__name__}")
    sig3 = inspect.signature(lm.forward)
    params3 = list(sig3.parameters.keys())
    print(f"language_model.forward params count: {len(params3)}")
    has_deepstack_lm = 'deepstack_visual_embeds' in params3
    has_visual_lm = 'visual_pos_masks' in params3
    print(f"  deepstack_visual_embeds: {has_deepstack_lm}")
    print(f"  visual_pos_masks: {has_visual_lm}")
    if not has_deepstack_lm:
        print()
        print("!!! CRITICAL BUG FOUND !!!")
        print("model.model.language_model.forward does NOT accept deepstack_visual_embeds")
        print("Our _run_language_model passes it, but it's silently ignored via **kwargs")
        print("=> Qwen3-VL deepstack re-injection NEVER happens during our training/inference!")
        print("=> This is equivalent to running without deepstack, degrading hidden states at L8/L16/L24")
    else:
        print()
        print("Deepstack params found correctly in language_model.forward")
        print("Conv3d patch is NOT the cause of Qwen3-VL performance issues")
        print("Need to investigate other causes...")
    del model
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
