"""
GUI-Owl Inference
=================
Mirrors src/zwerge_retrofit/modeling_guiowl.py on the inference side.

Two classes:
  GUIOwlRetrofitInference  — retrofit head on top of GUI-Owl-1.5 (Qwen3-VL)
  GUIOwlNativeInference    — original GUI-Owl-1.5 model (generate + parse coordinate)
                             Absorbs the former gui_owl15_grounding_infer.py.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from PIL import Image
from tqdm import tqdm
from qwen_vl_utils import process_vision_info

from inference_base import RetrofitInference, _ZOOM_NOT_SET


# ─────────────────────────────────────────────────────────────────────────────
# Retrofit model
# ─────────────────────────────────────────────────────────────────────────────

class GUIOwlRetrofitInference(RetrofitInference):
    """
    Retrofit inference for GUI-Owl-1.5-8B-Instruct (Qwen3-VL).

    patch_size=16: each visual token = 16*2=32 px (Qwen3-VL default).
    """
    model_type = "guiowl"
    merge_size = 2
    patch_size = 16   # Qwen3-VL

    # _zoom_native_system_message is set below (after ONLY_TWO_ACTION_SYSTEM_PROMPT is defined)

    def parse_backbone_coordinate(
        self,
        raw_text: str,
        crop_w_resized: int = None,   # ignored: GUI-Owl uses [0,1000] format
        crop_h_resized: int = None,
    ):
        """
        Parse GUI-Owl native output: <tool_call>{"coordinate": [x, y]}</tool_call>

        GUI-Owl outputs coordinates in [0,1000] scale (relative to input image),
        regardless of the actual pixel dimensions. Divide by 1000 → [0,1].
        crop_w/h_resized are ignored.
        """
        import re
        # Primary: {"coordinate": [x, y]} (JSON tool_call format)
        m = re.search(r'"coordinate"\s*:\s*\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]', raw_text)
        if m:
            return float(m.group(1)) / 1000.0, float(m.group(2)) / 1000.0
        # Fallback: any [x, y]
        m = re.search(r'\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\]', raw_text)
        if m:
            return float(m.group(1)) / 1000.0, float(m.group(2)) / 1000.0
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Native GUI-Owl-1.5 inference (original model — no retrofit head)
# Originally in gui_owl15_grounding_infer.py
# ─────────────────────────────────────────────────────────────────────────────

MIN_PIXELS = 196 * 32 * 32
MAX_PIXELS = 9800 * 32 * 32

# Official prompt from MobileAgent-v3.5
ONLY_TWO_ACTION_SYSTEM_PROMPT = '''# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "computer_use", "description": "Use a mouse to interact with a computer.\n* The screen\'s resolution is 1000x1000.\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don\'t click boxes on their edges unless asked.\n* don\'t use any other computer use tool like type, key, scroll, left_click_drag and so on.\n* you can only use the left_click and mouse_move action to interact with the computer. if you can\'t find the element, you should terminate the task and report the failure.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:\n* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.\n* `left_click`: Click the left mouse button with coordinate (x, y) pixel coordinate on the screen.", "enum": ["mouse_move", "left_click"], "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=mouse_move` and `action=left_click`.", "type": "array"}}, "required": ["action"], "type": "object"}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
'''

INFESIBLE_PREFIX = '''Additionally, if you think the task is infeasible (e.g., the task is not related to the image), return <tool_call>
{"name": "computer_use", "arguments": {"action": "terminate", "status": "failure"}}
</tool_call>'''

# Set zoom system message AFTER ONLY_TWO_ACTION_SYSTEM_PROMPT is defined.
# (Cannot set as class body attr because constant is defined after the class.)
# GUI-Owl backbone outputs [0,1000] coords with this prompt — no <|ground|> tokens.
GUIOwlRetrofitInference._zoom_native_system_message = ONLY_TWO_ACTION_SYSTEM_PROMPT


@dataclass
class NativePrediction:
    id: Optional[str]
    image: str
    instruction: str
    raw_response: str
    point_1000: Optional[List[int]]
    point_norm: Optional[List[float]]
    point_pixel: Optional[List[float]]
    status: str


def _floor_by_factor(number: float, factor: int) -> int:
    return math.floor(number / factor) * factor


def _ceil_by_factor(number: float, factor: int) -> int:
    return math.ceil(number / factor) * factor


def _round_by_factor(number: float, factor: int) -> int:
    return round(number / factor) * factor


def _smart_resize(
    height: int, width: int, factor: int = 32,
    min_pixels: int = 56 * 56, max_pixels: int = 32 * 32 * 9800,
    max_long_side: int = 8192,
) -> Tuple[int, int]:
    if height < 2 or width < 2:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"absolute aspect ratio must be smaller than 200")
    if max(height, width) > max_long_side:
        beta = max(height, width) / max_long_side
        height, width = int(height / beta), int(width / beta)
    h_bar = _round_by_factor(height, factor)
    w_bar = _round_by_factor(width,  factor)
    if h_bar * w_bar > max_pixels:
        beta  = math.sqrt((height * width) / max_pixels)
        h_bar = _floor_by_factor(height / beta, factor)
        w_bar = _floor_by_factor(width  / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta  = math.sqrt(min_pixels / (height * width))
        h_bar = _ceil_by_factor(height * beta, factor)
        w_bar = _ceil_by_factor(width  * beta, factor)
    return h_bar, w_bar


def _update_image_size_(image_ele: Dict[str, Any], min_tokens: int = 1,
                         max_tokens: int = 9800, merge_base: int = 2,
                         patch_size: int = 16) -> Dict[str, Any]:
    height, width = image_ele["height"], image_ele["width"]
    pixels_per_token = patch_size * patch_size * merge_base * merge_base
    rh, rw = _smart_resize(
        height, width, factor=merge_base * patch_size,
        min_pixels=pixels_per_token * min_tokens,
        max_pixels=pixels_per_token * max_tokens,
        max_long_side=50000,
    )
    image_ele.update({"resized_height": rh, "resized_width": rw,
                       "seq_len": rh * rw // pixels_per_token + 2})
    return image_ele


def _make_qwen_image_item(img_path: str, max_tokens: int = 9800,
                           patch_size: int = 16) -> Dict[str, Any]:
    path = str(Path(img_path).expanduser())
    if not (path.startswith("http://") or path.startswith("https://") or path.startswith("oss://")):
        path = str(Path(path).absolute())
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with Image.open(path) as img:
            width, height = img.size
    else:
        width, height = 1000, 1000
    image_ele = {"image": path, "height": height, "width": width, "type": "image"}
    return _update_image_size_(image_ele, max_tokens=max_tokens, patch_size=patch_size)


def _build_native_messages(image_path: str, instruction: str,
                            add_infesible_prefix: bool = True) -> List[Dict[str, Any]]:
    image_ele  = _make_qwen_image_item(image_path, max_tokens=9800, patch_size=16)
    system_text = ONLY_TWO_ACTION_SYSTEM_PROMPT
    if add_infesible_prefix:
        system_text = system_text + "\n" + INFESIBLE_PREFIX
    return [
        {"role": "system", "content": [{"text": system_text, "type": "text"}]},
        {"role": "user", "content": [image_ele, {"type": "text", "text": instruction}]},
    ]


def parse_gui_owl_point(raw_response: str) -> Tuple[Optional[List[int]], str]:
    """Parse (x,y) or [x,y] coordinate from raw generation output."""
    if not raw_response:
        return None, "empty"
    lower = raw_response.lower()
    if "terminate" in lower or '"status": "failure"' in lower or '"status":"failure"' in lower:
        return None, "infeasible"
    matches = re.findall(r"\((\d+),\s*(\d+)\)", raw_response)
    if not matches:
        matches = re.findall(r"\[(\d+),\s*(\d+)\]", raw_response)
    if not matches:
        return None, "no_coordinate"
    x, y = matches[-1]
    return [int(x), int(y)], "positive"


def _point_1000_to_norm_pixel(point_1000: List[int],
                                image_path: str) -> Tuple[List[float], List[float]]:
    x, y = point_1000
    norm = [float(x) / 1000.0, float(y) / 1000.0]
    with Image.open(image_path) as img:
        width, height = img.size
    return norm, [norm[0] * width, norm[1] * height]


def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _normalize_row(row: Dict[str, Any], images_dir: Optional[str]) -> Tuple[Optional[str], str, str]:
    image = (row.get("image") or row.get("image_path") or row.get("img")
             or row.get("img_path") or row.get("img_filename") or row.get("file_name"))
    instruction = (row.get("instruction") or row.get("query") or row.get("text")
                   or row.get("prompt") or row.get("task"))
    sample_id = row.get("id") or row.get("sample_id") or row.get("uid")
    if image is None or instruction is None:
        raise ValueError(f"Cannot find image/instruction fields in row: {row}")
    image_path = Path(str(image)).expanduser()
    if not image_path.is_absolute() and images_dir:
        image_path = Path(images_dir).expanduser() / image_path
    return None if sample_id is None else str(sample_id), str(image_path), str(instruction)


class GUIOwlNativeInference:
    """
    Original GUI-Owl-1.5 inference (Qwen3VLForConditionalGeneration, no retrofit head).
    CLI: python eval/inference_guiowl.py --mode native --model_path ... --image ... --instruction ...
    """

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        use_flash_attention: bool = True,
    ):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        kwargs: Dict[str, Any] = {
            "torch_dtype": torch.bfloat16 if self.device.type == "cuda" else torch.float32,
        }
        if self.device.type == "cuda" and use_flash_attention:
            kwargs["attn_implementation"] = "flash_attention_2"
        self.model = (Qwen3VLForConditionalGeneration
                      .from_pretrained(model_path, **kwargs)
                      .to(self.device).eval())
        self.processor = AutoProcessor.from_pretrained(
            model_path, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS
        )

    @torch.inference_mode()
    def infer(
        self,
        image_path: str,
        instruction: str,
        sample_id: Optional[str] = None,
        add_infesible_prefix: bool = True,
    ) -> NativePrediction:
        messages = _build_native_messages(image_path, instruction, add_infesible_prefix)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(self.model.device)
        gen_config = {"top_p": 0.01, "top_k": 1, "temperature": 0.01, "repetition_penalty": 1.0}
        generated_ids = self.model.generate(**inputs, **gen_config, max_new_tokens=2048)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        raw_response = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        point_1000, status = parse_gui_owl_point(raw_response)
        point_norm = point_pixel = None
        if point_1000 is not None:
            point_norm, point_pixel = _point_1000_to_norm_pixel(point_1000, image_path)
        return NativePrediction(
            id=sample_id, image=image_path, instruction=instruction,
            raw_response=raw_response, point_1000=point_1000,
            point_norm=point_norm, point_pixel=point_pixel, status=status,
        )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _main_native():
    import argparse
    parser = argparse.ArgumentParser(description="GUI-Owl-1.5 native grounding inference")
    parser.add_argument("--model_path",           required=True)
    parser.add_argument("--device",               default=None)
    parser.add_argument("--no_flash_attention",   action="store_true")
    parser.add_argument("--no_infesible_prefix",  action="store_true")
    parser.add_argument("--image")
    parser.add_argument("--instruction")
    parser.add_argument("--jsonl")
    parser.add_argument("--images_dir")
    parser.add_argument("--output",               default="gui_owl_native_predictions.jsonl")
    args = parser.parse_args()

    grounder = GUIOwlNativeInference(
        model_path=args.model_path, device=args.device,
        use_flash_attention=not args.no_flash_attention,
    )
    add_prefix = not args.no_infesible_prefix

    if args.image and args.instruction:
        pred = grounder.infer(args.image, args.instruction, add_infesible_prefix=add_prefix)
        print(json.dumps(asdict(pred), ensure_ascii=False, indent=2))
        return

    if not args.jsonl:
        raise SystemExit("Provide either --image + --instruction or --jsonl")

    with open(args.output, "w", encoding="utf-8") as wf:
        for row in tqdm(_iter_jsonl(args.jsonl), desc="GUI-Owl-1.5 native"):
            sid, image, instruction = _normalize_row(row, args.images_dir)
            pred = grounder.infer(image, instruction, sample_id=sid, add_infesible_prefix=add_prefix)
            out = asdict(pred)
            out["source"] = row
            wf.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    _main_native()
