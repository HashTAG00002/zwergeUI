"""
ZwerGe-UI Retrofit Constants
针对 UI-TARS-1.5-7B (Qwen2.5-VL 架构) 的 coordinate-free grounding retrofit。

核心设计（来自 chatgpt-export.txt 第 8888-8944 行）：
  - 冻结 backbone，只训练轻量级 layer-wise grounding head
  - Query anchor token 的主方案：pre-coordinate action-prefix token
      即 "pyautogui.click(" 之后、坐标生成之前的最后一个 token
      此时模型已经看过 image + instruction + "click("，但还没有看到任何坐标
      → 无 label leakage，天然与 action interface 对齐
  - 实现方式：引入 <GROUND> token，放置于 action prefix "click(start_box='" 之后：
        click(start_box='<|ground|><|pointer_start|><|pointer_pad|><|pointer_end|>')
      对应 UI-TARS-1.5 真实固化输出格式（issue #183/#138 确认），坐标部分替换为 pointer tokens
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
# ⚠️  必须与 UI-TARS-1.5 真实固化输出格式一致（issue #183/#138 确认）：
#   click(start_box='<|box_start|>(x,y)<|box_end|>')
# 我们用 <|ground|> 替代坐标部分，放在 start_box=' 之后：
#   click(start_box='<|ground|><|pointer_start|><|pointer_pad|><|pointer_end|>')
GROUND_RESPONSE_CLICK = (
    f"click(start_box='"
    f"{DEFAULT_GROUND_TOKEN}"
    f"{DEFAULT_POINTER_START_TOKEN}"
    f"{DEFAULT_POINTER_PAD_TOKEN}"
    f"{DEFAULT_POINTER_END_TOKEN}')"
)

# GUI-Actor/GUI-AIMA 格式注入（pointer_start 已存在时，在其后插入 <|ground|>）
GROUND_INJECTION_AFTER_POINTER_START = (
    f"{DEFAULT_POINTER_START_TOKEN}{DEFAULT_GROUND_TOKEN}"
)

# ─────────────────────────────────────────────
# UI-TARS 原生坐标格式正则（用于检测并替换为 retrofit 格式）
#
# UI-TARS-1.5 真实固化输出格式（来源：issue #183/#138 官方 collaborator 确认）：
#   click(start_box='<|box_start|>(x,y)<|box_end|>')
# prompt.py 里的 <point> 格式是过时/错误的，模型实际不输出那个格式。
#
# 训练数据来源可能混有多种格式，全部需要覆盖：
#   1. click(start_box='<|box_start|>(x,y)<|box_end|>')  ← UI-TARS-1.5 真实格式（最重要）
#   2. click(start_box='(x,y)')                          ← 无 box token 的变体
#   3. click(point='(x,y)') 或 click(point='<point>x y</point>') ← 旧版/prompt.py 格式
#   4. click([x, y])                                     ← 早期 UI-TARS 格式
#   5. pyautogui.click(x, y)                             ← GUI-Actor/OS-Atlas 格式
# ─────────────────────────────────────────────
UITARS_CLICK_PATTERNS = [
    # UI-TARS-1.5 真实固化格式：click(start_box='<|box_start|>(...)<|box_end|>')
    r"click\(start_box='[^']*'\)",
    # <point> 格式（prompt.py 旧版，部分数据可能含有）
    r"click\(point='[^']*'\)",
    # pyautogui.click(x, y) → GUI-Actor/OS-Atlas 格式
    r"pyautogui\.click\([^)]*\)",
    # click([x, y]) → 早期 UI-TARS 格式
    r"click\(\[[^\]]*\]\)",
    # coordinates=[x, y] → 其他格式
    r"coordinates=\[[^\]]*\]",
]

# ─────────────────────────────────────────────
# UI-TARS grounding system message
#
# 尽量与 UI-TARS-1.5 官方 GROUNDING_PROMPT 保持一致：
#   - 保留 "## Output Format / ## Action Space / ## User Instruction" 结构
#   - Action Space 中的坐标部分替换为我们的 pointer tokens（<|ground|> 等）
#   - User turn 只放 instruction 本身，system message 不含 {instruction} 占位符
#     （因为我们采用 system + user 分离的 chat 格式）
# ─────────────────────────────────────────────
GROUNDING_SYSTEM_MESSAGE = (
    "You are a GUI agent. You are given a task and a screenshot. "
    "You need to perform the next action to complete the task.\n\n\n\n"
    "## Output Format\n\n"
    "Action: ...\n\n\n\n"
    "## Action Space\n\n"
    # ⚠️  必须与 GROUND_RESPONSE_CLICK 以及 UI-TARS-1.5 真实固化格式完全一致：
    #   click(start_box='<|ground|><|pointer_start|><|pointer_pad|><|pointer_end|>')
    # 来源：issue #138 官方 collaborator JjjFangg 提供的 grounding-only prompt，
    #       其中坐标 '<|box_start|>(x1,y1)<|box_end|>' 替换为我们的 pointer tokens
    f"click(start_box='{DEFAULT_GROUND_TOKEN}"
    f"{DEFAULT_POINTER_START_TOKEN}"
    f"{DEFAULT_POINTER_PAD_TOKEN}"
    f"{DEFAULT_POINTER_END_TOKEN}')\n"
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
