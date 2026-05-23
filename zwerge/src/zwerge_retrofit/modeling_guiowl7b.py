"""
ZwerGe-UI Retrofit: GUIOwl7BRetrofitModel
==========================================
GUI-Owl-7B（Qwen2.5-VL 系列）retrofit 版本。

架构信息（GUI-Owl-7B/config.json 实测）：
  - architectures: Qwen2_5_VLForConditionalGeneration（与 UI-TARS-1.5-7B 完全相同）
  - num_hidden_layers: 28
  - hidden_size: 3584
  - patch_size: 14, spatial_merge_size: 2
  - 无 deepstack（与 UI-TARS 相同，与 GUI-Owl-1.5 不同）
  - conda 环境：gui_actor（transformers 4.51.3）

与 UITARSRetrofitModel 的唯一区别：
  - MODEL_TYPE = "guiowl7b"
  - prompt 使用 GUI-Owl-1.5 的 tool_call 格式（GUI_OWL_GROUND_RESPONSE / GUI_OWL_SYSTEM_PROMPT）
  - 坐标系：Qwen2.5-VL 绝对像素坐标（与 uitars 相同）

实验目的（控制变量）：
  - guiowl7b(Qwen2.5-VL) vs guiowl(Qwen3-VL)
    → 两者 prompt 相同，架构不同 → 分离 Qwen2.5 vs Qwen3 能力差距
  - guiowl7b(GUI-Owl prompt) vs uitars(TARS prompt)
    → 两者架构相同，prompt 不同 → 分离 prompt 格式的影响
  - 若 guiowl7b ≈ uitars：说明 GUI-Owl-1.5 的 prompt 格式无额外加分，
    且 Retrofit 框架在 Qwen2.5-VL 上运行正常
  - 若 guiowl7b >> guiowl(Qwen3-VL)：说明问题在 Qwen3-VL 架构本身

Action format (GUI-Owl-7B retrofit prefill)：与 GUI-Owl-1.5 相同，
  <tool_call>
  {"name": "computer_use", "arguments": {"action": "left_click", "coordinate": [
  <|ground|><|pointer_start|><|pointer_pad|><|pointer_end|>]}}
  </tool_call>
"""

from .modeling_uitars import UITARSRetrofitModel


class GUIOwl7BRetrofitModel(UITARSRetrofitModel):
    """
    GUI-Owl-7B（Qwen2.5-VL, 28层）retrofitted with layer-wise grounding head.

    继承 UITARSRetrofitModel（完全相同的 forward() 逻辑，Qwen2.5-VL backbone），
    唯一区别是 MODEL_TYPE 标识（用于加载 constants.py 中对应的 prompt 配置）。

    所有训练/推断逻辑完全复用 UITARSRetrofitModel：
      - output_hidden_states=True（Qwen2.5-VL 支持，无 deepstack 问题）
      - 无 Conv3d patch（Qwen2.5-VL 用的是 Conv2d/Linear，不是 Conv3d）
      - 无 forward hook（直接用 outputs.hidden_states，与 uitars 完全相同）
    """
    # MODEL_TYPE 仅用于日志/checkpoint 标识，不影响任何运行时逻辑
    # 真正的 prompt 配置由 train_retrofit.py 根据 --model_type guiowl7b 加载
    pass
