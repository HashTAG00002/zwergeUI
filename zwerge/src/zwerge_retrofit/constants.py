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


# =============================================================================
# GUI-Owl-1.5 constants (Qwen3-VL based)
# =============================================================================
# ─────────────────────────────────────────────────────────────────────────────
# Ground response（assistant prefill）— 定义在 system prompt 之前，以便 system prompt 引用
#
# 格式：tool_call JSON，坐标部分 [x, y] → [<|ground|><|pointer_start|><|pointer_pad|><|pointer_end|>]
# <|ground|> 是坐标前锚点（pre-coordinate action-prefix token，无 label leakage）。
#
# _find_ground_anchor() P1：candidates = positions[positions > vision_cut]，取最后一个
#   → system prompt 里的 <|ground|> 位于 <|vision_end|> 之前，被 vision_cut 过滤
#   → P1 始终命中 assistant prefill 里的 <|ground|>，与 UI-TARS 完全等价
# ─────────────────────────────────────────────────────────────────────────────
GUI_OWL_GROUND_RESPONSE = (
    "<tool_call>\n"
    '{"name": "computer_use", "arguments": {"action": "left_click", "coordinate": ['
    f"{DEFAULT_GROUND_TOKEN}"
    f"{DEFAULT_POINTER_START_TOKEN}"
    f"{DEFAULT_POINTER_PAD_TOKEN}"
    f"{DEFAULT_POINTER_END_TOKEN}"
    "]}}\n"
    "</tool_call>"
)

# ─────────────────────────────────────────────────────────────────────────────
# System prompt（grounding-only 改造版，基于 MobileAgent-v3.5 / GUI-Owl-1.5 官方格式）
#
# 对原始 system prompt 的两处修改：
#   1. 动作空间仅保留 left_click（移除 mouse_move；移除 infeasible/terminate 指令）
#   2. "For each function call" 示例中，坐标改为 pointer tokens（与 assistant prefill 完全一致）
#      模型在 context 中见到该格式，有助于对齐坐标输出方式
# ─────────────────────────────────────────────────────────────────────────────
_GUI_OWL_SYSTEM_PREFIX = '''# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {
"name": "computer_use", 
"description": "Use a mouse and keyboard to interact with a computer, and take screenshots. This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications. * The screen\'s resolution is 1000x1000.\\n* Make sure to click buttons, links, icons, etc with the cursor tip in the center of the element. Don\'t click boxes on their edges unless asked.", 
"parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:\\n* `left_click`: Click the left mouse button at coordinate (x, y) pixel coordinate on the screen.", "enum": ["left_click"], "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to click. Required only by `action=left_click`.", "type": "array"}}, "required": ["action", "coordinate"], "type": "object"}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
'''
GUI_OWL_SYSTEM_PROMPT = _GUI_OWL_SYSTEM_PREFIX + GUI_OWL_GROUND_RESPONSE

# ─────────────────────────────────────────────────────────────────────────────
# GUI-Owl-1.5 Layer 配置（实测自 GUI-Owl-1.5-8B-Instruct/config.json）
#   text_config.num_hidden_layers = 36    <- LLM decoder 层数
#   text_config.hidden_size       = 4096
#   vision_config.depth           = 27    <- ViT 层数
#   vision_config.deepstack_visual_indexes = [8, 16, 24]
#       ⚠️  8/16/24 是 ViT 层号，不是 LLM 层号！
#       ViT block 8/16/24 的中间输出分别注入到 LLM decoder 第 0/1/2 层之后
#   vision_config.patch_size      = 16  (vs Qwen2.5-VL 的 14)
#   vision_config.spatial_merge_size = 2
# ─────────────────────────────────────────────────────────────────────────────
GUI_OWL_15_NUM_LAYERS  = 36
GUI_OWL_15_HIDDEN_SIZE = 4096
GUI_OWL_15_DEFAULT_PROBE_LAYERS = [26, 27, 28, 29, 30, 31, 32, 33, 34, 35]  # last 10 of 36


# =============================================================================
# UI-Venus-1.5 constants (Qwen3-VL based)
# =============================================================================
# ─────────────────────────────────────────────────────────────────────────────
# UI-Venus-1.5 无 system message，直接使用 user turn 中的指令模板
#
# 原始 native 推理模板保留（backward compat，供 UIVenusNativeInference 使用）：
# ─────────────────────────────────────────────────────────────────────────────
UI_VENUS_USER_PROMPT_TEMPLATE_WITH_REFUSAL = (
    "Output the center point of the position corresponding to the following instruction: \n{}. "
    "\n\nThe output should just be the coordinates of a point, in the format [x,y]. "
    "Additionally, if the task is infeasible (e.g., the task is not related to the image), "
    "the output should be [-1,-1]."
)

UI_VENUS_USER_PROMPT_TEMPLATE_NO_REFUSAL = (
    "Output the center point of the position corresponding to the following instruction: \n{}. "
    "\n\nThe output should just be the coordinates of a point, in the format [x,y]."
)

# ─────────────────────────────────────────────────────────────────────────────
# Ground response（assistant prefill）— 坐标 [x, y] → [<|ground|>...]
#
# _find_ground_anchor() P1：
#   user prompt 里的 <|ground|> 在 <|vision_end|> 之后 → 候选
#   assistant prefill 里的 <|ground|> 更靠后 → P1 取最后一个 = assistant prefill ✅
# ─────────────────────────────────────────────────────────────────────────────
UI_VENUS_GROUND_RESPONSE = (
    f"[{DEFAULT_GROUND_TOKEN}"
    f"{DEFAULT_POINTER_START_TOKEN}"
    f"{DEFAULT_POINTER_PAD_TOKEN}"
    f"{DEFAULT_POINTER_END_TOKEN}]"
)

# ─────────────────────────────────────────────────────────────────────────────
# Retrofit grounding 专用 user prompt 模板（grounding-only 改造版）
#
# 对原始 native 模板的两处修改：
#   1. 移除 infeasible / [-1,-1] 选项（grounding 训练不需要拒绝分支）
#   2. 输出格式示例由 [x,y] 改为 pointer tokens（与 assistant prefill 完全一致）
#      → 模型在 user turn 中见到 pointer tokens 格式，有助于对齐输出
#      → user turn 里的 <|ground|> 在 <|vision_end|> 之后，但 assistant prefill 更靠后，
#         P1 仍取 assistant prefill（见注释 above）
# ─────────────────────────────────────────────────────────────────────────────
UI_VENUS_USER_PROMPT_TEMPLATE = (
    f"Output the center point of the position corresponding to the following instruction: \n{{}}. "
    f"\n\nThe output should just be the coordinates of a point, in the format "
    f"[{DEFAULT_GROUND_TOKEN}{DEFAULT_POINTER_START_TOKEN}"
    f"{DEFAULT_POINTER_PAD_TOKEN}{DEFAULT_POINTER_END_TOKEN}]."
)

# ─────────────────────────────────────────────────────────────────────────────
# UI-Venus-1.5 Layer 配置（实测自 UI-Venus-1.5-8B/config.json，架构与 GUI-Owl 完全相同）
#   text_config.num_hidden_layers = 36    <- LLM decoder 层数
#   text_config.hidden_size       = 4096
#   vision_config.depth           = 27    <- ViT 层数
#   vision_config.deepstack_visual_indexes = [8, 16, 24]
#       ⚠️  8/16/24 是 ViT 层号，不是 LLM 层号！
#       ViT block 8/16/24 的中间输出分别注入到 LLM decoder 第 0/1/2 层之后
# ─────────────────────────────────────────────────────────────────────────────
UI_VENUS_15_NUM_LAYERS  = 36
UI_VENUS_15_HIDDEN_SIZE = 4096
UI_VENUS_15_DEFAULT_PROBE_LAYERS = [26, 27, 28, 29, 30, 31, 32, 33, 34, 35]  # last 10 of 36


# =============================================================================
# GUI-Owl-7B constants (Qwen2.5-VL based — 控制变量)
# =============================================================================
# GUI-Owl-7B 是 GUI-Owl 系列的 Qwen2.5-VL 版本：
#   - architecture: Qwen2_5_VLForConditionalGeneration（与 UI-TARS-1.5-7B 完全相同）
#   - num_hidden_layers: 28，hidden_size: 3584，patch_size: 14（与 UI-TARS 相同）
#   - 无 deepstack（与 UI-TARS 相同）
#   - conda 环境：gui_actor（transformers 4.51.3）
#
# 与 uitars 的唯一区别：
#   - prompt 格式使用 GUI-Owl-1.5 的 tool_call 格式（GUI_OWL_GROUND_RESPONSE）
#   - 坐标系：Qwen2.5-VL 绝对像素坐标（与 uitars 相同，与 guiowl1.5 的 [0,1000] 不同）
#
# 实验目的：通过与 guiowl(Qwen3-VL) 对比，分离"模型本身能力"与"Retrofit框架正确性"的影响
# ─────────────────────────────────────────────────────────────────────────────
GUI_OWL_7B_NUM_LAYERS  = 28
GUI_OWL_7B_HIDDEN_SIZE = 3584
GUI_OWL_7B_DEFAULT_PROBE_LAYERS = [18, 19, 20, 21, 22, 23, 24, 25, 26, 27]  # last 10 of 28


# =============================================================================
# Model type → constants 映射（供 train/eval 脚本使用）
# =============================================================================
MODEL_TYPE_CONSTANTS = {
    "uitars": {
        "system_message": GROUNDING_SYSTEM_MESSAGE,
        "ground_response": GROUND_RESPONSE_CLICK,
        "default_probe_layers": DEFAULT_PROBE_LAYERS,
        "merge_size": 2,
        # None → user turn = image + instruction (原始格式，无模板包装)
        "user_prompt_template": None,
    },
    "guiowl": {
        "system_message": GUI_OWL_SYSTEM_PROMPT,
        "ground_response": GUI_OWL_GROUND_RESPONSE,
        "default_probe_layers": GUI_OWL_15_DEFAULT_PROBE_LAYERS,
        "merge_size": 2,
        "user_prompt_template": None,  # user turn = image + instruction
    },
    "uivenus": {
        "system_message": None,   # UI-Venus 无 system message（传 None 跳过 system turn）
        "ground_response": UI_VENUS_GROUND_RESPONSE,
        "default_probe_layers": UI_VENUS_15_DEFAULT_PROBE_LAYERS,
        "merge_size": 2,
        # user turn = image + UI_VENUS_USER_PROMPT_TEMPLATE.format(instruction)
        # grounding-only: no refusal branch, format example uses pointer tokens
        "user_prompt_template": UI_VENUS_USER_PROMPT_TEMPLATE,
    },
    # ── GUI-Owl-7B: Qwen2.5-VL + GUI-Owl prompt（控制变量）────────────────────
    "guiowl7b": {
        "system_message": GUI_OWL_SYSTEM_PROMPT,    # 与 guiowl1.5 相同的 prompt 格式
        "ground_response": GUI_OWL_GROUND_RESPONSE, # 与 guiowl1.5 相同的 response 格式
        "default_probe_layers": GUI_OWL_7B_DEFAULT_PROBE_LAYERS,  # last 10 of 28
        "merge_size": 2,
        "user_prompt_template": None,               # user turn = image + instruction
    },
}
