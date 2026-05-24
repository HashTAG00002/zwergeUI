"""
UI-Venus Inference
==================
Mirrors src/zwerge_retrofit/modeling_uivenus.py on the inference side.

Two classes:
  UIVenusRetrofitInference  — retrofit head on top of UI-Venus-1.5 (Qwen3-VL)
  UIVenusNativeInference    — original UI-Venus-1.5 model (generate + parse coordinate)
                              Absorbs the former ui_venus15_grounding_infer.py.
"""

from __future__ import annotations

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

from inference_base import RetrofitInference, _ZOOM_NOT_SET


# ─────────────────────────────────────────────────────────────────────────────
# Retrofit model
# ─────────────────────────────────────────────────────────────────────────────

class UIVenusRetrofitInference(RetrofitInference):
    """
    Retrofit inference for UI-Venus-1.5-8B (Qwen3-VL).

    patch_size=16: each visual token = 16*2=32 px (Qwen3-VL default).
    """
    model_type = "uivenus"
    merge_size = 2
    patch_size = 16   # Qwen3-VL

    # _zoom_native_system_message and _zoom_native_user_template are set after
    # UI_VENUS_USER_PROMPT_TEMPLATE_NO_REFUSAL is defined (later in this file)
    _zoom_native_system_message = None   # explicit None: no system turn for UI-Venus

    def parse_backbone_coordinate(
        self,
        raw_text: str,
        crop_w_resized: int = None,   # ignored: UI-Venus uses [0,1000] format
        crop_h_resized: int = None,
    ):
        """
        Parse UI-Venus native output: [x, y] (or [-1,-1] for infeasible)

        UI-Venus outputs [0,1000] relative coordinates.
        parse_venus_point() already divides by 1000 → returns [0,1].
        crop_w/h_resized are ignored.
        """
        coord = parse_venus_point(raw_text)
        if coord is None or coord == [-1, -1]:
            return None
        return float(coord[0]), float(coord[1])   # already [0,1]


# ─────────────────────────────────────────────────────────────────────────────
# Native UI-Venus-1.5 inference (original model — no retrofit head)
# Originally in ui_venus15_grounding_infer.py
# ─────────────────────────────────────────────────────────────────────────────

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

# Set zoom user template after PROMPT_NO_REFUSAL is defined
UIVenusRetrofitInference._zoom_native_user_template = PROMPT_NO_REFUSAL


@dataclass
class NativePrediction:
    id: Optional[str]
    image: str
    instruction: str
    full_prompt: str
    raw_response: str
    result: str
    point_norm: Optional[List[float]]
    point_1000: Optional[List[float]]
    point_pixel: Optional[List[float]]


def parse_venus_point(text: str) -> Optional[List[float]]:
    """Official UI-Venus coordinate parsing logic."""
    text = text.strip()
    pattern_bbox       = r"\[\s*-?\d+\s*,\s*-?\d+\s*,\s*-?\d+\s*,\s*-?\d+\s*\]"
    pattern_point      = r"\[\s*-?\d+\s*,\s*-?\d+\s*\]"
    pattern_two_points = r"\[\s*-?\d+\s*,\s*-?\d+\s*\],\s*\[\s*-?\d+\s*,\s*-?\d+\s*\]"
    try:
        if re.fullmatch(pattern_bbox, text, re.DOTALL):
            nums  = [int(x) for x in re.findall(r"-?\d+", text)]
            point = [(nums[0] + nums[2]) / 2, (nums[1] + nums[3]) / 2]
        elif re.fullmatch(pattern_point, text, re.DOTALL):
            point = [int(x) for x in re.findall(r"-?\d+", text)]
        elif re.fullmatch(pattern_two_points, text.replace(" ", ""), re.DOTALL):
            nums  = [int(x) for x in re.findall(r"-?\d+", text)]
            point = [(nums[0] + nums[2]) / 2, (nums[1] + nums[3]) / 2]
        else:
            head  = text.split("]")[0].split("[")[1]
            point = list(map(int, head.split(",")))
        if point == [-1, -1]:
            return [-1, -1]
        return [float(point[0]) / 1000.0, float(point[1]) / 1000.0]
    except Exception:
        return None


def _norm_to_1000_pixel(point_norm: Optional[List[float]],
                         image_path: str) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    if point_norm is None or point_norm == [-1, -1]:
        return point_norm, None
    point_1000 = [point_norm[0] * 1000.0, point_norm[1] * 1000.0]
    with Image.open(image_path) as img:
        width, height = img.size
    return point_1000, [point_norm[0] * width, point_norm[1] * height]


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


class UIVenusNativeInference:
    """
    Original UI-Venus-1.5 inference (AutoModelForImageTextToText, no retrofit head).
    CLI: python eval/inference_uivenus.py --mode native --model_name_or_path ... --image ... --instruction ...
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: Optional[str] = None,
        use_flash_attention: bool = True,
    ):
        from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
        from transformers.generation import GenerationConfig

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        kwargs: Dict[str, Any] = {
            "torch_dtype": torch.bfloat16 if self.device.type == "cuda" else torch.float32,
            "low_cpu_mem_usage": True,
        }
        if self.device.type == "cuda" and use_flash_attention:
            kwargs["attn_implementation"] = "flash_attention_2"

        self.model     = (AutoModelForImageTextToText
                          .from_pretrained(model_name_or_path, **kwargs)
                          .to(self.device).eval())
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)

        gen_cfg = GenerationConfig.from_pretrained(
            model_name_or_path, trust_remote_code=True
        ).to_dict()
        gen_cfg.update({"max_new_tokens": 256, "do_sample": False, "temperature": 0.0})
        self.model.generation_config = GenerationConfig(**gen_cfg)

    @staticmethod
    def build_prompt(instruction: str, do_not_use_refusal: bool = False) -> str:
        if instruction.endswith("."):
            instruction = instruction[:-1]
        template = PROMPT_NO_REFUSAL if do_not_use_refusal else PROMPT_WITH_REFUSAL
        return template.format(instruction)

    @torch.inference_mode()
    def infer(
        self,
        instruction: str,
        image_path: str,
        sample_id: Optional[str] = None,
        do_not_use_refusal: bool = False,
    ) -> NativePrediction:
        assert os.path.exists(image_path) and os.path.isfile(image_path), \
            f"Invalid image path: {image_path}"

        full_prompt = self.build_prompt(instruction, do_not_use_refusal=do_not_use_refusal)
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image_path},
                {"type": "text",  "text":  full_prompt},
            ]}
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(self.model.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        raw_response = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0]

        point_norm = parse_venus_point(raw_response)
        if point_norm is None:
            result = "wrong_format"
        elif point_norm == [-1, -1]:
            result = "infeasible"
        else:
            result = "positive"

        point_1000, point_pixel = _norm_to_1000_pixel(point_norm, image_path)
        return NativePrediction(
            id=sample_id, image=image_path, instruction=instruction,
            full_prompt=full_prompt, raw_response=raw_response, result=result,
            point_norm=point_norm, point_1000=point_1000, point_pixel=point_pixel,
        )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _main_native():
    import argparse
    parser = argparse.ArgumentParser(description="UI-Venus-1.5 native grounding inference")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--device",              default=None)
    parser.add_argument("--no_flash_attention",  action="store_true")
    parser.add_argument("--do_not_use_refusal",  action="store_true")
    parser.add_argument("--image")
    parser.add_argument("--instruction")
    parser.add_argument("--jsonl")
    parser.add_argument("--images_dir")
    parser.add_argument("--output",              default="ui_venus_native_predictions.jsonl")
    args = parser.parse_args()

    grounder = UIVenusNativeInference(
        model_name_or_path=args.model_name_or_path, device=args.device,
        use_flash_attention=not args.no_flash_attention,
    )

    if args.image and args.instruction:
        pred = grounder.infer(
            instruction=args.instruction, image_path=args.image,
            do_not_use_refusal=args.do_not_use_refusal,
        )
        print(json.dumps(asdict(pred), ensure_ascii=False, indent=2))
        return

    if not args.jsonl:
        raise SystemExit("Provide either --image + --instruction or --jsonl")

    with open(args.output, "w", encoding="utf-8") as wf:
        for row in tqdm(_iter_jsonl(args.jsonl), desc="UI-Venus-1.5 native"):
            sid, image, instruction = _normalize_row(row, args.images_dir)
            pred = grounder.infer(
                instruction=instruction, image_path=image, sample_id=sid,
                do_not_use_refusal=args.do_not_use_refusal,
            )
            out = asdict(pred)
            out["source"] = row
            wf.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    _main_native()
