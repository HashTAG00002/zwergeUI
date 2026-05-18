"""
ZwerGe-UI Retrofit Constants
针对 UI-TARS-1.5-7B (Qwen2.5-VL 架构) 的 coordinate-free grounding retrofit。

核心设计（来自 chatgpt-export.txt 第 8888-8944 行）：
  - 冻结 backbone，只训练轻量级 layer-wise grounding head
  - Query anchor token 的主方案：pre-coordinate action-prefix token
      即 "pyautogui.click(" 之后、坐标生成之前的最后一个 token
      此时模型已经看过 image + instruction + "click("，但还没有看到任何坐标
      → 无 label leakage，天然与 action interface 对齐
  - 实现方式：引入 <GROUND> token，放置于 action prefix "click(" 之后：
        pyautogui.click(<|ground|><|pointer_start|><|pointer_pad|><|pointer_end|>)
      <|ground|> 的 hidden state 即为 pre-coordinate action-prefix token
  - 对 GUI-Actor/GUI-AIMA 格式（已有 pointer tokens）：
        <|pointer_start|><|ground|><|pointer_pad|><|pointer_end|>
      注入在 pointer_start 之后
  - 支持多层 hidden-state probe（probe_layers 配置见下）
"""

import json

# ─────────────────────────────────────────────
# 基础常量
# ─────────────────────────────────────────────
IGNORE_INDEX = -100
DEFAULT_IMAGE_TOKEN = "<image>"

# ─────────────────────────────────────────────
# Grounding 特殊 token
#
# 主方案（chatgpt-export.txt §4.2）：
#   <|ground|> 作为 pre-coordinate action-prefix token，
#   放置于 "pyautogui.click(" 之后（action type 已知，坐标尚未生成）。
#   这与 GUI-Actor <ACTOR> token 类比，但更贴近坐标生成模型的 interface：
#     pyautogui.click(<|ground|><|pointer_start|><|pointer_pad|><|pointer_end|>)
#
# 备选方案（ablation 用）：
#   - 直接用 pointer_start 之前的最后一个 token（需在 forward 中动态查找）
#   - instruction pooling（不需要特殊 token）
# ─────────────────────────────────────────────
DEFAULT_GROUND_TOKEN = "<|ground|>"          # query anchor token（pre-coordinate action-prefix）
DEFAULT_POINTER_START_TOKEN = "<|pointer_start|>"
DEFAULT_POINTER_END_TOKEN = "<|pointer_end|>"
DEFAULT_POINTER_PAD_TOKEN = "<|pointer_pad|>"

# ─────────────────────────────────────────────
# tokenizer 需要额外添加的 special tokens
# ─────────────────────────────────────────────
ADDITIONAL_SPECIAL_TOKENS = [
    DEFAULT_GROUND_TOKEN,
    DEFAULT_POINTER_START_TOKEN,
    DEFAULT_POINTER_END_TOKEN,
    DEFAULT_POINTER_PAD_TOKEN,
]

# ─────────────────────────────────────────────
# Response 模板（各种 action 类型的 grounding response）
#
# 标准格式：action_prefix + <|ground|> + pointer tokens
# 这样 <|ground|> 的 hidden state = pre-coordinate action-prefix token
# 已经看过 image + instruction + action_prefix，但未看到坐标
# ─────────────────────────────────────────────
# click/double_click/right_click/hover 共用格式
GROUND_RESPONSE_CLICK = (
    f"pyautogui.click("
    f"{DEFAULT_GROUND_TOKEN}"
    f"{DEFAULT_POINTER_START_TOKEN}"
    f"{DEFAULT_POINTER_PAD_TOKEN}"
    f"{DEFAULT_POINTER_END_TOKEN})"
)

# GUI-Actor/GUI-AIMA 格式注入（pointer_start 已存在时，在其后插入 <|ground|>）
GROUND_INJECTION_AFTER_POINTER_START = (
    f"{DEFAULT_POINTER_START_TOKEN}{DEFAULT_GROUND_TOKEN}"
)

# ─────────────────────────────────────────────
# UI-TARS 原生坐标格式正则（用于检测并替换为 retrofit 格式）
# UI-TARS 的 click 格式：click([x, y]) 或 pyautogui.click(x, y)
# ─────────────────────────────────────────────
UITARS_CLICK_PATTERNS = [
    # pyautogui.click(x, y) → 替换整个 "pyautogui.click(...)"
    r"pyautogui\.click\([^)]*\)",
    # click([x, y]) → 替换整个 "click([...])"
    r"click\(\[[^\]]*\]\)",
    # action_type=click coordinates=[x, y] → 替换 coordinates=[...]
    r"coordinates=\[[^\]]*\]",
]

# ─────────────────────────────────────────────
# UI-TARS grounding system message
# 与 GUI-Actor 对齐，但强调从 hidden-state 读取空间证据
# ─────────────────────────────────────────────
GROUNDING_SYSTEM_MESSAGE = (
    "You are a GUI agent. Given a screenshot of the current GUI and a human instruction, "
    "your task is to locate the screen element that corresponds to the instruction. "
    "You should output a PyAutoGUI action that performs a click on the correct position. "
    "To indicate the click location, we will use some special tokens, which is used to refer to a visual patch later. "
    f"For example, you can output: {GROUND_RESPONSE_CLICK}."
)

# ─────────────────────────────────────────────
# Chat template（与 GUI-Actor/GUI-AIMA 一致，适配 Qwen2.5-VL）
# ─────────────────────────────────────────────
CHAT_TEMPLATE = (
    "{% set image_count = namespace(value=0) %}"
    "{% set video_count = namespace(value=0) %}"
    "{% for message in messages %}"
    "<|im_start|>{{ message['role'] }}\n"
    "{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n"
    "{% else %}"
    "{% for content in message['content'] %}"
    "{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}"
    "{% set image_count.value = image_count.value + 1 %}"
    "{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}"
    "<|vision_start|><|image_pad|><|vision_end|>"
    "{% elif content['type'] == 'video' or 'video' in content %}"
    "{% set video_count.value = video_count.value + 1 %}"
    "{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}"
    "<|vision_start|><|video_pad|><|vision_end|>"
    "{% elif 'text' in content %}{{ content['text'] }}"
    "{% endif %}"
    "{% endfor %}<|im_end|>\n"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)

# ─────────────────────────────────────────────
# Layer 配置
# UI-TARS-1.5-7B 共 28 层，选取用于 probe 的层
# 参考 chatgpt-export.txt 中讨论：建议先用 L/2, 2L/3, 3L/4 等关键层
# ─────────────────────────────────────────────
UI_TARS_7B_NUM_LAYERS = 28
UI_TARS_7B_HIDDEN_SIZE = 3584

# 默认 probe 层（可通过 --probe_layers 覆盖）
DEFAULT_PROBE_LAYERS = [14, 18, 21, 24, 26, 27]  # ~L/2, 2L/3, 3L/4, last few

# ─────────────────────────────────────────────
# Action pattern（与 GUI-Actor 一致，用于解析坐标并替换为 pointer tokens）
# ─────────────────────────────────────────────
ACTION_PATTERNS_XY = [
    r"x=([0-9.]+),\s*y=([0-9.]+)",
    r"from_coord=\[([0-9.]+),\s*([0-9.]+)\],\s*to_coord=\[([0-9.]+),\s*([0-9.]+)\]",
    r"pyautogui\.click\(([0-9.]+),\s*([0-9.]+)\)",
    # UI-TARS 原生坐标格式
    r"\[([0-9.]+),\s*([0-9.]+)\]",
    r"coordinate\s*=\s*\(([0-9.]+),\s*([0-9.]+)\)",
]
