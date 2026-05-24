"""
UI-TARS-7B-SFT Retrofit Inference
===================================
Mirrors src/zwerge_retrofit/modeling_uitars1.py on the inference side.

Two classes:
  UITARS1RetrofitInference  — retrofit head on top of UI-TARS-7B-SFT (Qwen2-VL architecture)

Architecture: Qwen2VLForConditionalGeneration, 28 layers, patch_size=14.
Prompt format: UI-TARS-1.5 format (identical to uitars/guiowl7b).
Coordinate system: relative 1000 (same as guiowl/uivenus/qwen35, NOT absolute pixels like uitars).

与 UITARSRetrofitInference 的关键区别：
  - model_type = "uitars1"（加载 Qwen2VL 而非 Qwen2.5-VL）
  - parse_backbone_coordinate：相对 1000 坐标（÷1000），非绝对像素坐标
  - _zoom_native_system_message：UI-TARS-1.5 原生格式（与 uitars 相同）
"""

from __future__ import annotations

import re
from typing import Optional

from inference_base import RetrofitInference

from zwerge_retrofit.constants import (
    GROUNDING_SYSTEM_MESSAGE,
)

# Native UI-TARS-1.5 grounding system message — shows actual coordinate format.
# This replaces the retrofit training message (which contains <|ground|> tokens)
# so the backbone generates real coordinates: click(start_box='<|box_start|>(x,y)<|box_end|>')
# Note: UI-TARS-7B-SFT is Qwen2-VL and outputs absolute pixel coordinates natively,
# but we parse them as relative 1000 because that is how the model was trained (retrofit).
_UITARS1_NATIVE_GROUNDING_SYSTEM_MESSAGE = (
    "You are a GUI agent. You are given a task and a screenshot. "
    "You need to perform the next action to complete the task.\n\n\n\n"
    "## Output Format\n\n"
    "Action: ...\n\n\n\n"
    "## Action Space\n\n"
    "click(start_box='<|box_start|>(x,y)<|box_end|>')\n"
)


class UITARS1RetrofitInference(RetrofitInference):
    """
    Retrofit inference for UI-TARS-7B-SFT (Qwen2-VL, 28 layers).

    patch_size=14: each visual token = 14*2=28 px (Qwen2-VL default).
    Coordinate system: relative 1000 (x/1000, y/1000 → [0,1]).

    与 uitars 的唯一区别：
      - model_type = "uitars1" → 加载 Qwen2VLForConditionalGeneration
      - parse_backbone_coordinate 将坐标除以 1000（而非 crop_w_resized）
    """
    model_type = "uitars1"
    merge_size = 2
    patch_size = 14   # Qwen2-VL（与 uitars 完全相同）

    # zoom_backbone 第二阶段使用 native prompt（不含 pointer tokens）
    _zoom_native_system_message = _UITARS1_NATIVE_GROUNDING_SYSTEM_MESSAGE
    _zoom_native_user_template  = None  # user turn = image + instruction

    def parse_backbone_coordinate(
        self,
        raw_text: str,
        crop_w_resized: Optional[int] = None,   # ignored: uitars1 uses [0,1000] format
        crop_h_resized: Optional[int] = None,
    ):
        """
        Parse UI-TARS-7B-SFT native output: click(start_box='<|box_start|>(x,y)<|box_end|>')

        UI-TARS-7B-SFT native coordinates are in [0,1000] scale (relative to input image),
        same as Qwen3-VL family. Divide by 1000 → [0,1].
        crop_w/h_resized are ignored.
        """
        # Primary: <|box_start|>(x,y)<|box_end|>
        m = re.search(r"<\|box_start\|>\((\d+),\s*(\d+)\)<\|box_end\|>", raw_text)
        if m:
            return float(m.group(1)) / 1000.0, float(m.group(2)) / 1000.0
        # Fallback: click(start_box='...(x,y)...')
        m = re.search(r"click\s*\([^)]*'[^']*\((\d+),\s*(\d+)\)[^']*'\)", raw_text)
        if m:
            return float(m.group(1)) / 1000.0, float(m.group(2)) / 1000.0
        # Last resort: bare [x, y]
        m = re.search(r"\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\]", raw_text)
        if m:
            return float(m.group(1)) / 1000.0, float(m.group(2)) / 1000.0
        return None
