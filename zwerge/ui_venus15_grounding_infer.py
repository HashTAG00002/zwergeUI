#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UI-Venus-1.5 grounding inference script, adapted from the official
UI-Venus-1.5/models/grounding/ui_venus1_5_gd.py.

Key official behaviours preserved:
  1) Uses AutoModelForImageTextToText + AutoProcessor + AutoTokenizer.
  2) Uses the exact official grounding prompt template.
  3) Removes a trailing period from instruction before prompt formatting.
  4) Uses a single user message containing image then text.
  5) Uses deterministic generation config and max_new_tokens=128 in inference.
  6) Parses [x,y], [x1,y1,x2,y2], or [x1,y1],[x2,y2], then normalizes by /1000.

Input JSONL format for batch mode:
  {"id": "optional", "image": "path/to.png", "instruction": "..."}
Aliases accepted for image: image_path, img, img_path, img_filename, file_name
Aliases accepted for instruction: query, text, prompt, task

Example single image:
  python ui_venus15_grounding_infer.py \
    --model_name_or_path /path/to/UI-Venus-1.5-Pro \
    --image ./screen.png \
    --instruction "search box"

Example batch:
  python ui_venus15_grounding_infer.py \
    --model_name_or_path /path/to/UI-Venus-1.5-Pro \
    --jsonl ./samples.jsonl \
    --output ./pred_ui_venus15.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from PIL import Image
from tqdm import tqdm
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
from transformers.generation import GenerationConfig


PROMPT_WITH_REFUSAL = (
    "Output the center point of the position corresponding to the following instruction: \n{}. "
    "\n\nThe output should just be the coordinates of a point, in the format [x,y]. "
    "Additionally, if the task is infeasible (e.g., the task is not related to the image), "
    "the output should be [-1,-1]."
)

PROMPT_NO_REFUSAL = (
    "Output the center point of the position corresponding to the following instruction: \n{}. "
    "\n\nThe output should just be the coordinates of a point, in the format [x,y]."
)


@dataclass
class Prediction:
    id: Optional[str]
    image: str
    instruction: str
    full_prompt: str
    raw_response: str
    result: str
    point_norm: Optional[List[float]]
    point_1000: Optional[List[float]]
    point_pixel: Optional[List[float]]


class UIVenus15Grounder:
    def __init__(
        self,
        model_name_or_path: str = "UI-Venus/UI-Venus-1.5-Pro",
        device: Optional[str] = None,
        use_flash_attention: bool = True,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        kwargs = {
            "torch_dtype": torch.bfloat16 if self.device.type == "cuda" else torch.float32,
            "low_cpu_mem_usage": True,
        }
        if self.device.type == "cuda" and use_flash_attention:
            kwargs["attn_implementation"] = "flash_attention_2"

        self.model = AutoModelForImageTextToText.from_pretrained(model_name_or_path, **kwargs).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)

        # Official default generation config, then overridden for deterministic eval.
        self.generation_config = GenerationConfig.from_pretrained(model_name_or_path, trust_remote_code=True).to_dict()
        self.set_generation_config(max_new_tokens=256, do_sample=False, temperature=0.0)

    def set_generation_config(self, **kwargs: Any) -> None:
        self.generation_config.update(kwargs)
        self.model.generation_config = GenerationConfig(**self.generation_config)

    @staticmethod
    def build_prompt(instruction: str, do_not_use_refusal: bool = False) -> str:
        # Official: strip only a final English period.
        if instruction.endswith("."):
            instruction = instruction[:-1]
        template = PROMPT_NO_REFUSAL if do_not_use_refusal else PROMPT_WITH_REFUSAL
        return template.format(instruction)

    @staticmethod
    def parse_point(text: str) -> Optional[List[float]]:
        """
        Official parse behaviour, but using ast/json-safe logic instead of eval:
          - [x,y] -> normalized [x/1000, y/1000]
          - [x1,y1,x2,y2] -> center then normalized
          - [x1,y1], [x2,y2] -> center then normalized
          - [-1,-1] -> [-1,-1]
          - parse error -> None
        """
        text = text.strip()
        pattern_bbox = r"\[\s*-?\d+\s*,\s*-?\d+\s*,\s*-?\d+\s*,\s*-?\d+\s*\]"
        pattern_point = r"\[\s*-?\d+\s*,\s*-?\d+\s*\]"
        pattern_two_points = r"\[\s*-?\d+\s*,\s*-?\d+\s*\],\s*\[\s*-?\d+\s*,\s*-?\d+\s*\]"

        try:
            if re.fullmatch(pattern_bbox, text, re.DOTALL):
                nums = [int(x) for x in re.findall(r"-?\d+", text)]
                point = [(nums[0] + nums[2]) / 2, (nums[1] + nums[3]) / 2]
            elif re.fullmatch(pattern_point, text, re.DOTALL):
                point = [int(x) for x in re.findall(r"-?\d+", text)]
            elif re.fullmatch(pattern_two_points, text.replace(" ", ""), re.DOTALL):
                nums = [int(x) for x in re.findall(r"-?\d+", text)]
                point = [(nums[0] + nums[2]) / 2, (nums[1] + nums[3]) / 2]
            else:
                # Official fallback: extract numbers before the first closing bracket.
                head = text.split("]")[0].split("[")[1]
                point = list(map(int, head.split(",")))

            if point == [-1, -1]:
                return [-1, -1]
            return [float(point[0]) / 1000.0, float(point[1]) / 1000.0]
        except Exception:
            return None

    @staticmethod
    def norm_to_1000_and_pixel(point_norm: Optional[List[float]], image_path: str) -> Tuple[Optional[List[float]], Optional[List[float]]]:
        if point_norm is None or point_norm == [-1, -1]:
            return point_norm, None
        point_1000 = [point_norm[0] * 1000.0, point_norm[1] * 1000.0]
        with Image.open(image_path) as img:
            width, height = img.size
        point_pixel = [point_norm[0] * width, point_norm[1] * height]
        return point_1000, point_pixel

    @torch.inference_mode()
    def infer(
        self,
        instruction: str,
        image_path: str,
        sample_id: Optional[str] = None,
        do_not_use_refusal: bool = False,
    ) -> Prediction:
        if isinstance(image_path, str):
            assert os.path.exists(image_path) and os.path.isfile(image_path), f"Invalid image path: {image_path}"
        else:
            raise ValueError("image must be a file path (str)")

        full_prompt = self.build_prompt(instruction, do_not_use_refusal=do_not_use_refusal)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": full_prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        raw_response = output_text[0]

        point_norm = self.parse_point(raw_response)
        if point_norm is None:
            result = "wrong_format"
        elif point_norm == [-1, -1]:
            result = "infeasible"
        else:
            result = "positive"

        point_1000, point_pixel = self.norm_to_1000_and_pixel(point_norm, image_path)

        return Prediction(
            id=sample_id,
            image=image_path,
            instruction=instruction,
            full_prompt=full_prompt,
            raw_response=raw_response,
            result=result,
            point_norm=point_norm,
            point_1000=point_1000,
            point_pixel=point_pixel,
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
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no_flash_attention", action="store_true")
    parser.add_argument("--do_not_use_refusal", action="store_true", help="Use official no-refusal prompt variant.")

    parser.add_argument("--image")
    parser.add_argument("--instruction")
    parser.add_argument("--jsonl")
    parser.add_argument("--images_dir")
    parser.add_argument("--output", default="ui_venus15_grounding_predictions.jsonl")
    args = parser.parse_args()

    grounder = UIVenus15Grounder(
        model_name_or_path=args.model_name_or_path,
        device=args.device,
        use_flash_attention=not args.no_flash_attention,
    )

    if args.image and args.instruction:
        pred = grounder.infer(
            instruction=args.instruction,
            image_path=args.image,
            do_not_use_refusal=args.do_not_use_refusal,
        )
        print(json.dumps(asdict(pred), ensure_ascii=False, indent=2))
        return

    if not args.jsonl:
        raise SystemExit("Provide either --image + --instruction or --jsonl")

    with open(args.output, "w", encoding="utf-8") as wf:
        for row in tqdm(iter_jsonl(args.jsonl), desc="UI-Venus-1.5 grounding"):
            sample_id, image, instruction = normalize_row(row, args.images_dir)
            pred = grounder.infer(
                instruction=instruction,
                image_path=image,
                sample_id=sample_id,
                do_not_use_refusal=args.do_not_use_refusal,
            )
            out = asdict(pred)
            out["source"] = row
            wf.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
