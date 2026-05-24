"""
Qwen3.5-VL Inference
=====================
Mirrors src/zwerge_retrofit/modeling_qwen35.py on the inference side.

Two classes:
  Qwen35RetrofitInference  — retrofit head on top of Qwen3.5-VL (Qwen3-VL architecture)
  Qwen35NativeInference    — original Qwen3.5-VL model (generate + parse coordinate)

Coordinate system: relative 1000 (same as guiowl/uivenus, NOT absolute pixels).

Native output format (XML-style tool-call):
  <tool_call>
  <computer_use>
  <action>left_click</action>
  <coordinate>[x, y]</coordinate>
  </computer_use>
  </tool_call>
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from tqdm import tqdm
from qwen_vl_utils import process_vision_info

from inference_base import RetrofitInference

import sys
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_SCRIPT_DIR, "..", "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from zwerge_retrofit.constants import (
    QWEN35_NATIVE_SYSTEM_PROMPT,
    QWEN35_NATIVE_USER_PROMPT_TEMPLATE,
)


# ─────────────────────────────────────────────────────────────────────────────
# Retrofit model
# ─────────────────────────────────────────────────────────────────────────────

class Qwen35RetrofitInference(RetrofitInference):
    """
    Retrofit inference for Qwen3.5-9B.

    patch_size=16: each visual token = 16*2=32 px (暂定，与 Qwen3-VL 相同).
    Coordinate system: relative 1000 (x/1000, y/1000 → [0,1]).
    """
    model_type = "qwen35"
    merge_size = 2
    patch_size = 16   # 暂定，待 Qwen3.5 config 确认

    # zoom-backbone 第二阶段使用 native prompt（不含 pointer tokens）
    _zoom_native_system_message = QWEN35_NATIVE_SYSTEM_PROMPT
    _zoom_native_user_template  = QWEN35_NATIVE_USER_PROMPT_TEMPLATE

    def parse_backbone_coordinate(
        self,
        raw_text: str,
        crop_w_resized: Optional[int] = None,   # ignored: Qwen3.5 uses [0,1000] format
        crop_h_resized: Optional[int] = None,
    ):
        """
        Parse Qwen3.5-VL native output: <coordinate>[x, y]</coordinate>

        Qwen3.5-VL outputs coordinates in [0,1000] scale (relative to input image),
        same as Qwen3-VL family. Divide by 1000 → [0,1].
        crop_w/h_resized are ignored.
        """
        # Primary: <coordinate>[x, y]</coordinate>  (XML-style tool-call)
        m = re.search(
            r'<coordinate>\s*\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]\s*</coordinate>',
            raw_text,
        )
        if m:
            return float(m.group(1)) / 1000.0, float(m.group(2)) / 1000.0
        # Fallback: bare [x, y]
        m = re.search(r'\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\]', raw_text)
        if m:
            return float(m.group(1)) / 1000.0, float(m.group(2)) / 1000.0
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Native Qwen3.5-VL inference (original model — no retrofit head)
# ─────────────────────────────────────────────────────────────────────────────

def parse_qwen35_coordinate(text: str) -> Optional[Tuple[float, float]]:
    """
    Parse Qwen3.5-VL native output coordinate.
    Returns (x_norm, y_norm) in [0,1], or None on failure.

    Primary: <coordinate>[x, y]</coordinate>  (XML-style, [0,1000])
    Fallback: bare [x, y] anywhere in the text.
    """
    m = re.search(
        r'<coordinate>\s*\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]\s*</coordinate>',
        text,
    )
    if m:
        return float(m.group(1)) / 1000.0, float(m.group(2)) / 1000.0
    m = re.search(r'\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\]', text)
    if m:
        return float(m.group(1)) / 1000.0, float(m.group(2)) / 1000.0
    return None


@dataclass
class Qwen35NativePrediction:
    id: Optional[str]
    image: str
    instruction: str
    full_prompt: str
    raw_response: str
    result: str
    point_norm: Optional[Tuple[float, float]]
    point_pixel: Optional[Tuple[float, float]]


class Qwen35NativeInference:
    """
    Original Qwen3.5-VL inference (AutoModelForImageTextToText, no retrofit head).
    Uses XML-style tool-call format, relative 1000 coordinate system.
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: Optional[str] = None,
        use_flash_attention: bool = True,
    ):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        kwargs: Dict[str, Any] = {
            "torch_dtype": torch.bfloat16 if self.device.type == "cuda" else torch.float32,
            "low_cpu_mem_usage": True,
        }
        if self.device.type == "cuda" and use_flash_attention:
            kwargs["attn_implementation"] = "flash_attention_2"

        self.model = (
            AutoModelForImageTextToText
            .from_pretrained(model_name_or_path, **kwargs)
            .to(self.device)
            .eval()
        )
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)

    @staticmethod
    def build_messages(image_path: str, instruction: str) -> List[Dict]:
        prompt = QWEN35_NATIVE_USER_PROMPT_TEMPLATE.format(instruction)
        return [
            {
                "role": "system",
                "content": QWEN35_NATIVE_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text",  "text":  prompt},
                ],
            },
        ]

    @torch.inference_mode()
    def infer(
        self,
        instruction: str,
        image_path: str,
        sample_id: Optional[str] = None,
        max_new_tokens: int = 256,
    ) -> Qwen35NativePrediction:
        assert os.path.exists(image_path), f"Invalid image path: {image_path}"

        messages = self.build_messages(image_path, instruction)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        raw_response = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0]

        coord = parse_qwen35_coordinate(raw_response)
        if coord is None:
            result = "wrong_format"
            point_norm = None
            point_pixel = None
        else:
            result = "positive"
            point_norm = coord
            with Image.open(image_path) as img:
                w, h = img.size
            point_pixel = (coord[0] * w, coord[1] * h)

        full_prompt = QWEN35_NATIVE_USER_PROMPT_TEMPLATE.format(instruction)
        return Qwen35NativePrediction(
            id=sample_id,
            image=image_path,
            instruction=instruction,
            full_prompt=full_prompt,
            raw_response=raw_response,
            result=result,
            point_norm=point_norm,
            point_pixel=point_pixel,
        )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _main_native():
    import argparse
    parser = argparse.ArgumentParser(description="Qwen3.5-VL native grounding inference")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--device",             default=None)
    parser.add_argument("--no_flash_attention", action="store_true")
    parser.add_argument("--image")
    parser.add_argument("--instruction")
    parser.add_argument("--jsonl")
    parser.add_argument("--images_dir")
    parser.add_argument("--output", default="qwen35_native_predictions.jsonl")
    args = parser.parse_args()

    grounder = Qwen35NativeInference(
        model_name_or_path=args.model_name_or_path,
        device=args.device,
        use_flash_attention=not args.no_flash_attention,
    )

    if args.image and args.instruction:
        pred = grounder.infer(instruction=args.instruction, image_path=args.image)
        print(json.dumps(asdict(pred), ensure_ascii=False, indent=2))
        return

    if not args.jsonl:
        raise SystemExit("Provide either --image + --instruction or --jsonl")

    def _iter_jsonl(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    with open(args.output, "w", encoding="utf-8") as wf:
        for row in tqdm(_iter_jsonl(args.jsonl), desc="Qwen3.5-VL native"):
            image = row.get("image") or row.get("image_path") or row.get("img")
            instruction = row.get("instruction") or row.get("query") or row.get("text")
            sample_id = str(row.get("id") or row.get("sample_id") or "")
            if not image or not instruction:
                continue
            image_path = str(image)
            if not os.path.isabs(image_path) and args.images_dir:
                image_path = os.path.join(args.images_dir, image_path)
            pred = grounder.infer(instruction=instruction, image_path=image_path, sample_id=sample_id)
            out = asdict(pred)
            out["source"] = row
            wf.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    _main_native()
