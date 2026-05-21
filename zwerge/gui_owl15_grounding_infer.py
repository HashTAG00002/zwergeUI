#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GUI-Owl-1.5 grounding inference script, adapted from the official
MobileAgent-v3.5/grounding_and_kb/eval_grounding_benchmarks.py.

Key official behaviours preserved:
  1) Uses Qwen3VLForConditionalGeneration + AutoProcessor.
  2) Uses the official two-action computer_use system prompt.
  3) For OSWorld-G-style grounding, adds the infeasible/terminate instruction.
  4) Builds qwen image items with resized_height/resized_width/seq_len.
  5) Uses apply_chat_template + qwen_vl_utils.process_vision_info.
  6) Parses the last coordinate from model output, matching official eval logic.

Input JSONL format for batch mode:
  {"id": "optional", "image": "path/to.png", "instruction": "..."}
Aliases accepted for image: image_path, img, img_path, img_filename, file_name
Aliases accepted for instruction: query, text, prompt, task

Example single image:
  python gui_owl15_grounding_infer.py \
    --model_path /path/to/GUI-Owl-1.5-8B-Instruct \
    --image ./screen.png \
    --instruction "Click the search box"

Example batch:
  python gui_owl15_grounding_infer.py \
    --model_path /path/to/GUI-Owl-1.5-8B-Instruct \
    --jsonl ./samples.jsonl \
    --output ./pred_gui_owl15.jsonl
"""

from __future__ import annotations

import argparse
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
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


MIN_PIXELS = 196 * 32 * 32
MAX_PIXELS = 9800 * 32 * 32

# Official prompt from MobileAgent-v3.5/grounding_and_kb/eval_grounding_benchmarks.py
ONLY_TWO_ACTION_SYSTEM_PROMPT = '''# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "computer_use", "description": "Use a mouse to interact with a computer.\n* The screen's resolution is 1000x1000.\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.\n* don't use any other computer use tool like type, key, scroll, left_click_drag and so on.\n* you can only use the left_click and mouse_move action to interact with the computer. if you can't find the element, you should terminate the task and report the failure.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:\n* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.\n* `left_click`: Click the left mouse button with coordinate (x, y) pixel coordinate on the screen.", "enum": ["mouse_move", "left_click"], "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=mouse_move` and `action=left_click`.", "type": "array"}}, "required": ["action"], "type": "object"}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
'''

# Official spelling preserved: infesible_prefix.
INFESIBLE_PREFIX = '''Additionally, if you think the task is infeasible (e.g., the task is not related to the image), return <tool_call>
{"name": "computer_use", "arguments": {"action": "terminate", "status": "failure"}}
</tool_call>'''


@dataclass
class Prediction:
    id: Optional[str]
    image: str
    instruction: str
    raw_response: str
    point_1000: Optional[List[int]]
    point_norm: Optional[List[float]]
    point_pixel: Optional[List[float]]
    status: str


def floor_by_factor(number: float, factor: int) -> int:
    return math.floor(number / factor) * factor


def ceil_by_factor(number: float, factor: int) -> int:
    return math.ceil(number / factor) * factor


def round_by_factor(number: float, factor: int) -> int:
    return round(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = 32,
    min_pixels: int = 56 * 56,
    max_pixels: int = 32 * 32 * 9800,
    max_long_side: int = 8192,
) -> Tuple[int, int]:
    """Same resize helper shape as the official GUI-Owl grounding script."""
    if height < 2 or width < 2:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"absolute aspect ratio must be smaller than 200, got {height} / {width}")

    if max(height, width) > max_long_side:
        beta = max(height, width) / max_long_side
        height, width = int(height / beta), int(width / beta)

    h_bar = round_by_factor(height, factor)
    w_bar = round_by_factor(width, factor)

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)

    return h_bar, w_bar


def update_image_size_(image_ele: Dict[str, Any], min_tokens: int = 1, max_tokens: int = 9800, merge_base: int = 2, patch_size: int = 16) -> Dict[str, Any]:
    height, width = image_ele["height"], image_ele["width"]
    pixels_per_token = patch_size * patch_size * merge_base * merge_base
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=merge_base * patch_size,
        min_pixels=pixels_per_token * min_tokens,
        max_pixels=pixels_per_token * max_tokens,
        max_long_side=50000,
    )
    image_ele.update(
        {
            "resized_height": resized_height,
            "resized_width": resized_width,
            "seq_len": resized_height * resized_width // pixels_per_token + 2,
        }
    )
    return image_ele


def make_qwen_image_item(img_path: str, max_tokens: int = 9800, patch_size: int = 16) -> Dict[str, Any]:
    path = str(Path(img_path).expanduser())
    if not (path.startswith("http://") or path.startswith("https://") or path.startswith("oss://")):
        path = str(Path(path).absolute())
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with Image.open(path) as img:
            width, height = img.size
    else:
        # For URLs, process_vision_info can fetch the image, but width/height are unknown here.
        # Prefer local files for exact pixel back-conversion. Use 1000 as a safe placeholder.
        width, height = 1000, 1000

    image_ele = {"image": path, "height": height, "width": width, "type": "image"}
    return update_image_size_(image_ele, max_tokens=max_tokens, patch_size=patch_size)


def build_messages(image_path: str, instruction: str, add_infesible_prefix: bool = True) -> List[Dict[str, Any]]:
    """Official message shape: system prompt + user image item + raw instruction."""
    image_ele = make_qwen_image_item(image_path, max_tokens=9800, patch_size=16)
    system_text = ONLY_TWO_ACTION_SYSTEM_PROMPT
    if add_infesible_prefix:
        system_text = system_text + "\n" + INFESIBLE_PREFIX

    return [
        {"role": "system", "content": [{"text": system_text, "type": "text"}]},
        {
            "role": "user",
            "content": [
                image_ele,
                {"type": "text", "text": instruction},
            ],
        },
    ]


def parse_gui_owl_point(raw_response: str) -> Tuple[Optional[List[int]], str]:
    """
    Match official eval logic closely:
      - first tries '(x, y)'
      - if none, tries '[x, y]' with a space after comma
      - uses the last parsed coordinate
      - no coordinate -> [0,0] in official eval; here status marks it as no_coordinate.
    This parser also accepts '[x,y]' to make inference output easier to consume.
    """
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


def point_to_norm_and_pixel(point_1000: List[int], image_path: str) -> Tuple[List[float], List[float]]:
    x, y = point_1000
    norm = [float(x) / 1000.0, float(y) / 1000.0]
    with Image.open(image_path) as img:
        width, height = img.size
    pixel = [norm[0] * width, norm[1] * height]
    return norm, pixel


class GUIOwl15Grounder:
    def __init__(self, model_path: str, device: Optional[str] = None, use_flash_attention: bool = True):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        kwargs = {
            "torch_dtype": torch.bfloat16 if self.device.type == "cuda" else torch.float32,
        }
        if self.device.type == "cuda" and use_flash_attention:
            kwargs["attn_implementation"] = "flash_attention_2"

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, **kwargs).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_path, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)

    @torch.inference_mode()
    def infer(self, image_path: str, instruction: str, sample_id: Optional[str] = None, add_infesible_prefix: bool = True) -> Prediction:
        messages = build_messages(image_path, instruction, add_infesible_prefix=add_infesible_prefix)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        # Official eval updates the generation kwargs this way.
        gen_config = {
            "top_p": 0.01,
            "top_k": 1,
            "temperature": 0.01,
            "repetition_penalty": 1.0,
        }
        generated_ids = self.model.generate(**inputs, **gen_config, max_new_tokens=2048)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        raw_response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        point_1000, status = parse_gui_owl_point(raw_response)
        point_norm = None
        point_pixel = None
        if point_1000 is not None:
            point_norm, point_pixel = point_to_norm_and_pixel(point_1000, image_path)

        return Prediction(
            id=sample_id,
            image=image_path,
            instruction=instruction,
            raw_response=raw_response,
            point_1000=point_1000,
            point_norm=point_norm,
            point_pixel=point_pixel,
            status=status,
        )


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def normalize_row(row: Dict[str, Any], images_dir: Optional[str]) -> Tuple[Optional[str], str, str]:
    image = row.get("image") or row.get("image_path") or row.get("img") or row.get("img_path") or row.get("img_filename") or row.get("file_name")
    instruction = row.get("instruction") or row.get("query") or row.get("text") or row.get("prompt") or row.get("task")
    sample_id = row.get("id") or row.get("sample_id") or row.get("uid")
    if image is None or instruction is None:
        raise ValueError(f"Cannot find image/instruction fields in row: {row}")
    image_path = Path(str(image)).expanduser()
    if not image_path.is_absolute() and images_dir:
        image_path = Path(images_dir).expanduser() / image_path
    return None if sample_id is None else str(sample_id), str(image_path), str(instruction)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no_flash_attention", action="store_true")
    parser.add_argument("--no_infesible_prefix", action="store_true", help="Disable official OSWorld-G infeasible terminate prefix.")

    parser.add_argument("--image")
    parser.add_argument("--instruction")
    parser.add_argument("--jsonl")
    parser.add_argument("--images_dir")
    parser.add_argument("--output", default="gui_owl15_grounding_predictions.jsonl")
    args = parser.parse_args()

    grounder = GUIOwl15Grounder(
        model_path=args.model_path,
        device=args.device,
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
        for row in tqdm(iter_jsonl(args.jsonl), desc="GUI-Owl-1.5 grounding"):
            sample_id, image, instruction = normalize_row(row, args.images_dir)
            pred = grounder.infer(image, instruction, sample_id=sample_id, add_infesible_prefix=add_prefix)
            out = asdict(pred)
            out["source"] = row
            wf.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
