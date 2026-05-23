"""
UI-TARS Retrofit Inference
==========================
Mirrors src/zwerge_retrofit/modeling_uitars.py on the inference side.
"""

from inference_base import RetrofitInference, _ZOOM_NOT_SET

# Native UI-TARS-1.5 grounding system message — shows actual coordinate format.
# This replaces the retrofit training message (which contains <|ground|> tokens)
# so the backbone generates real pixel coordinates: click(start_box='<|box_start|>(x,y)<|box_end|>')
_UITARS_NATIVE_GROUNDING_SYSTEM_MESSAGE = (
    "You are a GUI agent. You are given a task and a screenshot. "
    "You need to perform the next action to complete the task.\n\n\n\n"
    "## Output Format\n\n"
    "Action: ...\n\n\n\n"
    "## Action Space\n\n"
    "click(start_box='<|box_start|>(x,y)<|box_end|>')\n"
)


class UITARSRetrofitInference(RetrofitInference):
    """
    Retrofit inference for UI-TARS-1.5-7B (Qwen2.5-VL).

    patch_size=14: each visual token = 14*2=28 px (Qwen2.5-VL default).
    """
    model_type = "uitars"
    merge_size = 2
    patch_size = 14   # Qwen2.5-VL

    # Use native coordinate format for backbone generate (actual pixel coords, not <|ground|> tokens)
    _zoom_native_system_message = _UITARS_NATIVE_GROUNDING_SYSTEM_MESSAGE

    def parse_backbone_coordinate(
        self,
        raw_text: str,
        crop_w_resized: int = None,
        crop_h_resized: int = None,
    ):
        """
        Parse UI-TARS native output: click(start_box='<|box_start|>(x,y)<|box_end|>')

        UI-TARS outputs ABSOLUTE PIXEL coordinates in the smart_resize'd image space.
        Must normalize by the actual resized crop dimensions, NOT by 1000.

        crop_w_resized / crop_h_resized come from image_grid_thw[W/H] * patch_size (=14).
        """
        import re
        # UI-TARS-1.5 real format:  <|box_start|>(x,y)<|box_end|>
        m = re.search(r"<\|box_start\|>\((\d+),\s*(\d+)\)<\|box_end\|>", raw_text)
        if not m:
            m = re.search(r"click\s*\([^)]*'[^']*\((\d+),\s*(\d+)\)[^']*'\)", raw_text)
        if not m:
            m = re.search(r"\[(\d+),\s*(\d+)\]", raw_text)
        if m:
            x_abs = int(m.group(1))
            y_abs = int(m.group(2))
            if crop_w_resized and crop_h_resized:
                # Correct: normalize by smart-resized crop dimensions
                return float(x_abs) / crop_w_resized, float(y_abs) / crop_h_resized
            else:
                # Fallback when image_grid_thw not available (should not happen normally)
                import warnings
                warnings.warn("[zoom] crop_w/h_resized not provided for UI-TARS — fallback to /1000")
                return float(x_abs) / 1000.0, float(y_abs) / 1000.0
        return None
