"""
GUI-Owl-7B Retrofit Inference
==============================
Mirrors src/zwerge_retrofit/modeling_guiowl7b.py on the inference side.

GUI-Owl-7B 是控制变量：
  - 架构: Qwen2.5-VL（与 UI-TARS-1.5-7B 完全相同, 28层, patch_size=14）
  - prompt: GUI-Owl-1.5 的 tool_call 格式（与 guiowl1.5 相同）
  - 坐标系: Qwen2.5-VL 绝对像素坐标（与 uitars 相同, 需除以 crop_w_resized）

实验目的：
  - guiowl7b vs uitars:   prompt 格式的影响（架构相同，prompt 不同）
  - guiowl7b vs guiowl:   Qwen2.5 vs Qwen3 的性能差距（prompt 相同，架构不同）
"""

from inference_uitars import UITARSRetrofitInference


# Native GUI-Owl-7B grounding system message for zoom_backbone strategy.
# GUI-Owl-7B is Qwen2.5-VL, outputs absolute pixel coordinates like UI-TARS.
# Format: {"name": "computer_use", "arguments": {"action": "left_click", "coordinate": [x, y]}}
# where x, y are absolute pixel coords in smart_resize'd space.
_GUIOWL7B_NATIVE_GROUNDING_SYSTEM_MESSAGE = (
    "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n"
    "<tools>\n"
    '{"type": "function", "function": {'
    '"name": "computer_use", '
    '"description": "Use a mouse and keyboard to interact with a computer, and take screenshots.", '
    '"parameters": {"properties": {"action": {"description": "The action to perform. '
    'The available actions are:\\n* `left_click`: Click the left mouse button at coordinate (x, y).", '
    '"enum": ["left_click"], "type": "string"}, '
    '"coordinate": {"description": "(x, y): The x and y pixel coordinates.", "type": "array"}}, '
    '"required": ["action", "coordinate"], "type": "object"}}}\n'
    "</tools>\n\n"
    "For each function call, return a json object with function name and arguments within "
    "<tool_call></tool_call> XML tags:\n"
    "<tool_call>\n"
    '{"name": <function-name>, "arguments": <args-json-object>}\n'
    "</tool_call>\n"
)


class GUIOwl7BRetrofitInference(UITARSRetrofitInference):
    """
    Retrofit inference for GUI-Owl-7B (Qwen2.5-VL, 28 layers).

    继承 UITARSRetrofitInference：
      - 完全相同的模型加载逻辑（Qwen2.5-VL backbone）
      - 完全相同的 forward/grounding head 逻辑
      - 完全相同的坐标解析（Qwen2.5-VL 绝对像素坐标 → / crop_w_resized）

    唯一区别：
      - model_type = "guiowl7b"（用于加载 constants.py 中对应的 prompt 配置）
      - _zoom_native_system_message 使用 GUI-Owl 风格的原生系统消息
    """
    model_type = "guiowl7b"
    merge_size = 2
    patch_size = 14   # Qwen2.5-VL（与 uitars 完全相同）

    # zoom_backbone 策略时使用的原生系统消息（无 <|ground|> tokens）
    # GUI-Owl-7B 是 Qwen2.5-VL，输出绝对像素坐标（与 uitars 相同）
    _zoom_native_system_message = _GUIOWL7B_NATIVE_GROUNDING_SYSTEM_MESSAGE

    # parse_backbone_coordinate 继承自 UITARSRetrofitInference，
    # 解析 click(start_box='<|box_start|>(x,y)<|box_end|>') 格式
    # 或尝试从 {"coordinate": [x, y]} JSON 格式解析
    def parse_backbone_coordinate(
        self,
        raw_text: str,
        crop_w_resized: int = None,
        crop_h_resized: int = None,
    ):
        """
        Parse GUI-Owl-7B native output coordinate.

        GUI-Owl-7B generates Qwen2.5-VL absolute pixel coordinates.
        Try GUI-Owl tool_call format first, then fall back to UI-TARS format.

        Both formats output absolute pixel coordinates → normalize by crop_w_resized.
        """
        import re
        # Try GUI-Owl JSON format: {"coordinate": [x, y]}
        m = re.search(r'"coordinate"\s*:\s*\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]', raw_text)
        if m:
            x_abs = float(m.group(1))
            y_abs = float(m.group(2))
            if crop_w_resized and crop_h_resized:
                return x_abs / crop_w_resized, y_abs / crop_h_resized
            else:
                import warnings
                warnings.warn("[zoom] crop_w/h_resized not provided for GUI-Owl-7B — fallback to /1000")
                return x_abs / 1000.0, y_abs / 1000.0

        # Fall back to UI-TARS format: click(start_box='<|box_start|>(x,y)<|box_end|>')
        return super().parse_backbone_coordinate(raw_text, crop_w_resized, crop_h_resized)
