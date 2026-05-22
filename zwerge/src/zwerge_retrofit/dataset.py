"""
ZwerGe-UI Retrofit Dataset

数据格式：
  OS-Atlas 原始格式（每个 JSON 文件是一个 list）：
    [
      {
        "img_filename": "screenshot.png",
        "elements": [
          {
            "instruction": "Click the OK button",
            "bbox": [x_min, y_min, x_max, y_max],  # [0,1] normalized
            "data_type": "push-button"
          },
          ...
        ]
      },
      ...
    ]

  也支持 GUI-Actor/GUI-AIMA 格式（conversations）：
    [
      {
        "id": "xxx",
        "image": "relative/path/to/image.png",
        "conversations": [
          {"from": "human", "value": "<image>\nInstruction text"},
          {"from": "gpt", "value": "click(start_box='<|pointer_start|><|pointer_pad|><|pointer_end|>')"}
        ],
        "bbox": [x_min, y_min, x_max, y_max]
      },
      ...
    ]

  通用 ms-swift 格式（扁平化单轮）：
    [
      {
        "image": "path/to/image.png",
        "query": "Click the OK button",
        "response": "click(start_box='<|box_start|>(x,y)<|box_end|>')",
        "bbox": [x_min, y_min, x_max, y_max]
      },
      ...
    ]

训练时生成：
  - input_ids / labels (LM supervision)
  - pixel_values + image_grid_thw
  - ground_token_indices: <GROUND> token 位置（作为 query）
  - multi_patch_labels: patch-level soft label [N_vis]
"""

import copy
import json
import math
import os
import random
import re
from typing import Dict, List, Optional, Tuple

import torch
import transformers
import yaml
from PIL import Image
from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset

from .constants import (
    IGNORE_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_GROUND_TOKEN,
    DEFAULT_POINTER_START_TOKEN,
    DEFAULT_POINTER_END_TOKEN,
    DEFAULT_POINTER_PAD_TOKEN,
    ADDITIONAL_SPECIAL_TOKENS,
    GROUNDING_SYSTEM_MESSAGE,
    CHAT_TEMPLATE,
    GROUND_RESPONSE_CLICK,
    GROUND_INJECTION_AFTER_POINTER_START,
    UITARS_CLICK_PATTERNS,
)
from .trainer import rank0_print


# ─────────────────────────────────────────────────────────────────────────────
# Patch label utilities
# ─────────────────────────────────────────────────────────────────────────────

def get_patch_soft_label_from_point(
    image_processor,
    image: Image.Image,
    point_x: float,
    point_y: float,
    sigma_ratio: float = 0.05,
) -> torch.Tensor:
    """
    Convert GT click point (normalized [0,1]) to Gaussian soft label over patches.

    l_i = exp(-||patch_center_i - gt_point||^2 / (2σ^2))
    l = l / l.sum()

    sigma = sigma_ratio * min(image_width, image_height) in pixel space

    NOTE on patch layout (applies to both Qwen2.5-VL and Qwen3-VL):
      image_grid_thw [T, H, W] uses H, W = patch-level grid.
        Qwen2.5-VL (uitars):    patch_size=14, merge_size=2 → token_cell=28px
        Qwen3-VL (guiowl/uivenus): patch_size=16, merge_size=2 → token_cell=32px
      Each visual token in input_ids corresponds to merge_size² patches merged.
      So the actual token grid is (H // merge_size) × (W // merge_size).
      This function computes label at the MERGED token level, i.e.
        grid_h = image_height // (patch_size * merge_size)
        grid_w = image_width  // (patch_size * merge_size)
      which equals T * (H // merge_size) * (W // merge_size).
      token_cell_size is read dynamically from image_processor.patch_size × merge_size.
    """
    w, h = image.size
    # Each visual token spans patch_size * merge_size pixels.
    # patch_size is read dynamically: 14 for Qwen2.5-VL, 16 for Qwen3-VL.
    token_cell_size = image_processor.patch_size * image_processor.merge_size
    grid_w = max(1, w // token_cell_size)
    grid_h = max(1, h // token_cell_size)

    gt_px = point_x * w
    gt_py = point_y * h
    sigma = sigma_ratio * min(w, h)

    label = torch.zeros(grid_h * grid_w)
    for y_idx in range(grid_h):
        for x_idx in range(grid_w):
            cx = (x_idx + 0.5) * token_cell_size
            cy = (y_idx + 0.5) * token_cell_size
            dist2 = (cx - gt_px) ** 2 + (cy - gt_py) ** 2
            label[y_idx * grid_w + x_idx] = math.exp(-dist2 / (2 * sigma ** 2))

    label_sum = label.sum()
    if label_sum > 0:
        label = label / label_sum
    else:
        # Fallback: uniform
        label = torch.ones(grid_h * grid_w) / (grid_h * grid_w)
    return label


def get_patch_binary_label_from_bbox(
    image_processor,
    image: Image.Image,
    bbox: List[float],
) -> torch.Tensor:
    """
    Convert GT bbox (normalized [0,1]) to binary patch label over patches.
    Patches with any overlap with bbox are positive.
    Normalizes to sum=1.

    NOTE on patch layout (Qwen2.5-VL and Qwen3-VL):
      Each visual token corresponds to a token_cell_size × token_cell_size pixel region,
      where token_cell_size = patch_size * merge_size (read dynamically from image_processor).
        uitars (Qwen2.5-VL): patch_size=14, merge_size=2 → token_cell=28px
        guiowl/uivenus (Qwen3-VL): patch_size=16, merge_size=2 → token_cell=32px
      Grid size matches n_image_tokens in input_ids (= H * W / merge_size²).
    """
    w, h = image.size
    # Token-level cell size: each visual token covers this many pixels.
    # patch_size read dynamically: 14 for Qwen2.5-VL, 16 for Qwen3-VL.
    token_cell_size = image_processor.patch_size * image_processor.merge_size

    # Handle potentially non-divisible sizes
    grid_w = max(1, w // token_cell_size)
    grid_h = max(1, h // token_cell_size)

    x_min, y_min, x_max, y_max = bbox
    x_min_px = max(0.0, x_min * w)
    y_min_px = max(0.0, y_min * h)
    x_max_px = min(float(w), x_max * w)
    y_max_px = min(float(h), y_max * h)

    label = torch.zeros(grid_h * grid_w)
    for y_idx in range(grid_h):
        for x_idx in range(grid_w):
            patch_x_min = x_idx * token_cell_size
            patch_y_min = y_idx * token_cell_size
            patch_x_max = patch_x_min + token_cell_size
            patch_y_max = patch_y_min + token_cell_size

            # Overlap check
            if not (patch_x_max <= x_min_px or patch_x_min >= x_max_px or
                    patch_y_max <= y_min_px or patch_y_min >= y_max_px):
                label[y_idx * grid_w + x_idx] = 1.0

    label_sum = label.sum()
    if label_sum > 0:
        label = label / label_sum
    else:
        # Fallback: point label at bbox center
        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0
        label = get_patch_soft_label_from_point(
            image_processor, image, cx, cy
        )
    return label


def get_patch_gaussian_label_from_bbox(
    image_processor,
    image: Image.Image,
    bbox: List[float],
    sigma_factor: float = 0.5,
) -> torch.Tensor:
    """
    各向异性 Gaussian patch label：以 GT bbox 中心为均值，
    σ_x = bbox_width_px * sigma_factor，σ_y = bbox_height_px * sigma_factor。

    宽 bbox → σ_x 大（x 方向分布更均匀）；高 bbox → σ_y 大（y 方向分布更均匀）。
    归一化到 sum=1。

    sigma_factor 推荐值 0.5：σ 等于 bbox 半宽/半高，bbox 内覆盖约 68% 的概率质量。
    """
    w, h = image.size
    token_cell_size = image_processor.patch_size * image_processor.merge_size
    grid_w = max(1, w // token_cell_size)
    grid_h = max(1, h // token_cell_size)

    x_min, y_min, x_max, y_max = bbox
    cx_px = ((x_min + x_max) / 2.0) * w
    cy_px = ((y_min + y_max) / 2.0) * h
    bbox_w_px = (x_max - x_min) * w
    bbox_h_px = (y_max - y_min) * h

    sigma_x = max(bbox_w_px * sigma_factor, 1e-3)
    sigma_y = max(bbox_h_px * sigma_factor, 1e-3)

    # Patch center coordinates in pixels: vectorized
    x_idx = torch.arange(grid_w, dtype=torch.float32)
    y_idx = torch.arange(grid_h, dtype=torch.float32)
    cx_patches = (x_idx + 0.5) * token_cell_size   # [grid_w]
    cy_patches = (y_idx + 0.5) * token_cell_size   # [grid_h]

    dx2 = (cx_patches - cx_px) ** 2                         # [grid_w]
    dy2 = (cy_patches - cy_px) ** 2                         # [grid_h]
    label_2d = torch.exp(
        -(dx2.unsqueeze(0) / (2 * sigma_x ** 2) +
          dy2.unsqueeze(1) / (2 * sigma_y ** 2))
    )                                                        # [grid_h, grid_w]

    label = label_2d.reshape(-1)
    label_sum = label.sum()
    if label_sum > 1e-9:
        return label / label_sum
    # 极端情况（sigma 过小使所有 patch 权重趋零）退回 binary
    return get_patch_binary_label_from_bbox(image_processor, image, bbox)


def get_ground_token_idx_in_sequence(
    input_ids: torch.Tensor,
    ground_token_id: int,
    pointer_start_token_id: Optional[int] = None,
) -> Optional[int]:
    """
    Return the best anchor token index for grounding query.

    Priority (mirrors modeling_uitars._find_ground_anchor):
      P1. First <|ground|> token (explicitly injected, no leakage)
      P2. Token immediately before <|pointer_start|>
          (at this position: has seen action prefix, not yet coordinates)
      P3. None — let modeling_uitars handle fallback dynamically

    NOTE: We intentionally do NOT fall back to "last non-pad token" here,
    because in UI-TARS that token often follows coordinate values → label leakage.
    The caller (modeling_uitars._find_ground_anchor) has better context to
    handle P3/P4/P5 cases with appropriate warnings.
    """
    # P1: explicit <|ground|> token
    # NOTE: Use the LAST occurrence, not the first, because the system message
    # Action Space example contains a spurious <|ground|> at an early position:
    #   "click(start_box='<|ground|><|pointer_start|><|pointer_pad|><|pointer_end|>')"
    # The true anchor is in the assistant response (later in the sequence).
    positions = (input_ids == ground_token_id).nonzero(as_tuple=False)
    if positions.numel() > 0:
        return positions[-1].item()

    # P2: token immediately before <|pointer_start|>
    # NOTE: P2 only triggers when P1 failed (no <|ground|> in sequence at all).
    # Since GROUNDING_SYSTEM_MESSAGE always contains <|ground|>, P1 will always
    # fire first whenever the system message is present. P2 is a safety fallback
    # for data that has <|pointer_start|> but NO <|ground|> token at all
    # (e.g. raw GUI-Actor format without <|ground|> injection).
    # In that edge case there is only ONE <|pointer_start|> (in assistant response),
    # so first vs last does not matter. Using first is fine.
    if pointer_start_token_id is not None:
        ptr_positions = (input_ids == pointer_start_token_id).nonzero(as_tuple=False)
        if ptr_positions.numel() > 0:
            ptr_pos = ptr_positions[0].item()   # first occurrence (see note above)
            if ptr_pos > 0:
                return ptr_pos - 1   # action-prefix token just before pointer_start

    # P3: let modeling_uitars handle it dynamically
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Query anchor injection utilities
#
# 核心设计：
#   主方案：pre-coordinate action-prefix token
#     格式：click(start_box='<|ground|><|pointer_start|><|pointer_pad|><|pointer_end|>')
#     <|ground|> 在 start_box=' 之后，hidden state 已看到 image+instruction+action_type，
#     但还没看到坐标，无 label leakage
#     对应 UI-TARS-1.5 真实固化输出格式（issue #183/#138）：
#       click(start_box='<|box_start|>(x,y)<|box_end|>')
#
#   对已有 pointer token 的格式（GUI-Actor/GUI-AIMA）：
#     注入：<|pointer_start|><|ground|><|pointer_pad|><|pointer_end|>
#
#   对各种原始坐标格式（UITARS_CLICK_PATTERNS 全部覆盖）：
#     click(start_box='<|box_start|>(x,y)<|box_end|>') → GROUND_RESPONSE_CLICK
#     click(point='(x,y)')                             → GROUND_RESPONSE_CLICK
#     pyautogui.click(x, y)                           → GROUND_RESPONSE_CLICK
#     click([x, y])                                    → GROUND_RESPONSE_CLICK
# ─────────────────────────────────────────────────────────────────────────────

def _try_parse_point_from_text(text: str) -> Optional[List[float]]:
    """Try to extract (x, y) from various coordinate formats. (module-level helper)

    All formats return coordinates in the SAME space as stored in the original response:
      - Normalized [0,1]: pyautogui.click / click([]) formats
      - Absolute pixels: UI-TARS-1.5 click(start_box='<|box_start|>(x,y)<|box_end|>')
        → returned as-is; caller is responsible for normalization if needed.

    ⚠️ WARNING: UI-TARS-1.5 real format (issue #183/#138 confirmed) uses ABSOLUTE pixel
    coords (e.g. 1327, 864). When gt_point comes from this path AND bbox is None,
    get_patch_soft_label_from_point (which expects normalized [0,1]) will receive
    out-of-range coordinates and produce a completely wrong Gaussian label.
    Mitigation: _get_item always tries bbox-derived label FIRST; only falls back to
    gt_point if bbox is None. Ensure training data always has bbox when possible.
    """
    # UI-TARS-1.5 真实固化格式：click(start_box='<|box_start|>(x,y)<|box_end|>')
    # 坐标是绝对像素值，如 (1327,864)
    m = re.search(r"click\(start_box='[^']*\(([0-9]+),\s*([0-9]+)\)[^']*'\)", text)
    if m:
        # 绝对像素坐标，返回时注明（调用方需自行归一化）
        return [float(m.group(1)), float(m.group(2))]
    # click(point='(x,y)') or click(point='x y') — 旧版/prompt.py 格式（绝对像素）
    m = re.search(r"click\(point='[^']*?([0-9]+)[,\s]+([0-9]+)[^']*'\)", text)
    if m:
        return [float(m.group(1)), float(m.group(2))]
    # GUI-Actor format: x=0.5, y=0.3（归一化 [0,1]）
    m = re.search(r"x=([0-9.]+),\s*y=([0-9.]+)", text)
    if m:
        return [float(m.group(1)), float(m.group(2))]
    # pyautogui.click(x, y)（归一化 [0,1]）
    m = re.search(r"pyautogui\.click\(([0-9.]+),\s*([0-9.]+)\)", text)
    if m:
        return [float(m.group(1)), float(m.group(2))]
    # click([x, y])（归一化 [0,1]）
    m = re.search(r"click\(\[([0-9.]+),\s*([0-9.]+)\]\)", text)
    if m:
        return [float(m.group(1)), float(m.group(2))]
    # [x, y]（归一化 [0,1]，最宽泛的 fallback）
    m = re.search(r"\[([0-9.]+),\s*([0-9.]+)\]", text)
    if m:
        return [float(m.group(1)), float(m.group(2))]
    return None


def inject_ground_token_into_response(response: str) -> str:
    """
    Inject <|ground|> token into a response string to create the
    pre-coordinate action-prefix anchor.

    Strategy (priority order):
      1. Match any pattern in UITARS_CLICK_PATTERNS → replace the entire matched
         call with GROUND_RESPONSE_CLICK (first match wins, count=1).
         This covers all known click coordinate formats:
           - click(start_box='<|box_start|>(x,y)<|box_end|>')  ← UI-TARS-1.5 real format
           - click(start_box='(x,y)')
           - click(point='...')
           - pyautogui.click(x, y)
           - click([x, y])
      2. If pointer_start already present but no <|ground|> → inject after pointer_start
         e.g. <|pointer_start|><|pointer_pad|>... → <|pointer_start|><|ground|><|pointer_pad|>...
      3. If response already has <|ground|> → return as-is
      4. Fallback: wrap the whole response in GROUND_RESPONSE_CLICK format
    """
    # Case 0 (FIRST): if <|ground|> already present, return immediately.
    # MUST be checked before Case 1, because GROUND_RESPONSE_CLICK itself contains
    # "click(start_box='...'" which matches UITARS_CLICK_PATTERNS[0]. Without this
    # guard, a response that was already injected would be replaced again.
    if DEFAULT_GROUND_TOKEN in response:
        return response

    # Case 1: replace any known native coordinate call with our retrofit format
    for pattern in UITARS_CLICK_PATTERNS:
        new_response = re.sub(pattern, GROUND_RESPONSE_CLICK, response, count=1)
        if new_response != response:
            return new_response

    # Case 2: pointer_start exists but no ground token
    if DEFAULT_POINTER_START_TOKEN in response and DEFAULT_GROUND_TOKEN not in response:
        return response.replace(
            DEFAULT_POINTER_START_TOKEN,
            GROUND_INJECTION_AFTER_POINTER_START,
            1,
        )

    # Case 3: fallback - wrap response with our standard format
    # (e.g. response is just plain text with no action)
    return GROUND_RESPONSE_CLICK


# ─────────────────────────────────────────────────────────────────────────────
# Dataset class
# ─────────────────────────────────────────────────────────────────────────────

class RetrofitDataset(Dataset):
    """
    Lazy-loading dataset for UI-TARS Retrofit training.

    Supports three input formats:
      1. OS-Atlas raw format (img_filename + elements list)
      2. GUI-Actor/GUI-AIMA conversations format
      3. ms-swift flat format (query/response)

    All formats produce:
      - input_ids, labels (for optional LM loss)
      - pixel_values, image_grid_thw
      - ground_token_indices (position of <GROUND> token in sequence)
      - multi_patch_labels (soft patch label for grounding loss)
    """

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        processor: transformers.ProcessorMixin,
        data_path: str,
        data_args,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.processor = processor
        self.data_args = data_args

        self.ground_token_id        = tokenizer.convert_tokens_to_ids(DEFAULT_GROUND_TOKEN)
        self.pointer_pad_token_id   = tokenizer.convert_tokens_to_ids(DEFAULT_POINTER_PAD_TOKEN)
        self.pointer_start_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_POINTER_START_TOKEN)
        self.pointer_end_token_id   = tokenizer.convert_tokens_to_ids(DEFAULT_POINTER_END_TOKEN)

        # Model-specific prompt constants (injected from data_args in train_retrofit.py)
        # ground_response: assistant prefill template (e.g. GUI-Owl uses tool_call format)
        # user_prompt_template: None or a .format(instruction) template (e.g. UI-Venus)
        # system_message: None → skip system turn (UI-Venus); non-None → include system turn
        self._ground_response      = getattr(data_args, "ground_response",      GROUND_RESPONSE_CLICK)
        self._user_prompt_template = getattr(data_args, "user_prompt_template", None)
        self._system_message       = getattr(data_args, "system_message",       GROUNDING_SYSTEM_MESSAGE)

        self.gt_label_type = getattr(data_args, "gt_label_type", "binary")
        self.gaussian_sigma_factor = getattr(data_args, "gaussian_sigma_factor", 0.5)
        rank0_print(f"[RetrofitDataset] gt_label_type={self.gt_label_type}"
                    + (f"  sigma_factor={self.gaussian_sigma_factor}"
                       if self.gt_label_type == "gaussian" else ""))

        # Load all data into flat list: each item has unified format
        self.samples = []  # list of dicts: {image_path, instruction, bbox, conversations}

        self._load_data(data_path)
        random.shuffle(self.samples)
        rank0_print(f"[RetrofitDataset] Total samples loaded: {len(self.samples)} (shuffled)")

    def _load_data(self, data_path: str):
        """Load data from single JSON, comma/newline-separated paths, brace-pattern, or YAML."""
        default_images_folder = getattr(self.data_args, "image_folder", "") or ""
        if data_path.endswith(".yaml"):
            self._load_yaml(data_path)
        elif "{" in data_path and "}" in data_path:
            # Brace-expansion pattern: /path/to/{file1,file2,file3}.json
            base, pattern = re.match(r"^(.*)\{(.*)\}\.json$", data_path).groups()
            for fn in pattern.split(","):
                self._load_json(f"{base}{fn.strip()}.json", default_images_folder)
        elif "," in data_path or "\n" in data_path:
            # Comma- or newline-separated list of full paths
            for path in re.split(r"[,\n]+", data_path):
                path = path.strip()
                if not path:
                    continue
                if path.endswith(".yaml"):
                    self._load_yaml(path)
                else:
                    self._load_json(path, default_images_folder)
        else:
            self._load_json(data_path, default_images_folder)

    def _load_yaml(self, yaml_path: str):
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)
        for ds in cfg.get("datasets", []):
            json_path = ds["json_path"]
            images_folder = ds.get("images_folder", "")
            sampling = ds.get("sampling_strategy", "all")
            items = self._read_json_file(json_path)
            items = self._apply_sampling(items, sampling)
            self._parse_and_extend(items, images_folder)
            rank0_print(f"  Loaded {len(items)} samples from {json_path}")

    def _load_json(self, json_path: str, images_folder: str = ""):
        items = self._read_json_file(json_path)
        self._parse_and_extend(items, images_folder)
        rank0_print(f"  Loaded {len(items)} samples from {json_path}")

    @staticmethod
    def _read_json_file(path: str) -> List[Dict]:
        with open(path) as f:
            if path.endswith(".jsonl"):
                return [json.loads(line) for line in f]
            return json.load(f)

    @staticmethod
    def _apply_sampling(items: List, strategy: str) -> List:
        if ":" not in strategy:
            return items
        mode, num_str = strategy.split(":", 1)
        if "%" in num_str:
            n = math.ceil(int(num_str.replace("%", "")) * len(items) / 100)
        else:
            n = int(num_str)
        if mode == "first":
            return items[:n]
        elif mode == "end":
            return items[-n:]
        elif mode == "random":
            random.shuffle(items)
            return items[:n]
        return items

    def _parse_and_extend(self, items: List[Dict], images_folder: str):
        """Parse items in any supported format to unified internal format."""
        for item in items:
            parsed = self._parse_item(item, images_folder)
            if parsed is not None:
                if isinstance(parsed, list):
                    self.samples.extend(parsed)
                else:
                    self.samples.append(parsed)

    def _parse_item(self, item: Dict, images_folder: str) -> Optional[Dict]:
        """
        Detect item format and parse to unified dict:
        {
          "image_path": str,
          "conversations": [{"from": "human"|"gpt", "value": str}, ...],
          "bbox": [x_min, y_min, x_max, y_max],   # normalized [0,1]
          "gt_point": [x, y] or None,
        }
        Returns list if one item expands to multiple samples (OS-Atlas format with elements).
        """
        # ── Format 1: OS-Atlas raw (img_filename + elements) ──────────────
        # OS-Atlas 没有坐标格式，直接构建标准 retrofit 格式
        if "img_filename" in item and "elements" in item:
            results = []
            img_path = os.path.join(images_folder, item["img_filename"])
            for elem in item["elements"]:
                inst = elem.get("instruction", "")
                bbox = elem.get("bbox", None)
                if not inst or bbox is None:
                    continue
                # Compute GT point as bbox center
                cx = (bbox[0] + bbox[2]) / 2.0
                cy = (bbox[1] + bbox[3]) / 2.0
                human_text = self._user_prompt_template.format(inst) if self._user_prompt_template else inst
                convs = [
                    {
                        "from": "human",
                        "value": f"<image>\n{human_text}",
                    },
                    {
                        "from": "gpt",
                        "value": self._ground_response,
                    },
                ]
                results.append({
                    "image_path": img_path,
                    "conversations": convs,
                    "bbox": bbox,
                    "gt_point": [cx, cy],
                })
            return results if results else None

        # ── Format 2: GUI-Actor/GUI-AIMA conversations ────────────────────
        # 可能包含：
        #   - 已有 pointer tokens 的格式（GUI-Actor）
        #   - 原始坐标格式（需替换）
        if "conversations" in item:
            image_file = item.get("image", "")
            img_path = os.path.join(images_folder, image_file) if image_file else ""
            bbox = item.get("bbox", None)
            gt_point = item.get("gt_point", None)

            convs = copy.deepcopy(item["conversations"])
            for conv in convs:
                if conv["from"] == "gpt":
                    original_val = conv["value"]
                    # 先尝试从原始 response 解析 GT point（替换前）
                    if bbox is None and gt_point is None:
                        gt_point = _try_parse_point_from_text(original_val)
                    # 替换 gpt response 为 model-specific ground_response 格式
                    # 注意：不直接使用 inject_ground_token_into_response()，因为它硬编码
                    # 替换为 UITARS 的 GROUND_RESPONSE_CLICK 格式，对 guiowl/uivenus 不适用。
                    # 正确做法：始终用 self._ground_response（由 MODEL_TYPE_CONSTANTS 决定）。
                    conv["value"] = self._ground_response

            return {
                "image_path": img_path,
                "conversations": convs,
                "bbox": bbox,
                "gt_point": gt_point,
            }

        # ── Format 3: ms-swift flat format (query/response) ──────────────
        # 可能包含 UI-TARS 格式：pyautogui.click(x, y) 或 click([x, y])
        # 也支持 grounding_50k 格式：{image, instruction, gt_bbox_norm, gt_bbox_abs, img_width, img_height}
        if "query" in item or "instruction" in item:
            image_file = item.get("image", item.get("image_path", ""))
            img_path = os.path.join(images_folder, image_file) if image_file else ""
            query = item.get("query", item.get("instruction", ""))
            response = item.get("response", item.get("answer", ""))
            # bbox 优先级：bbox（已归一化）> gt_bbox_norm（已归一化）> gt_bbox_abs（像素绝对值，需归一化）
            bbox = item.get("bbox", None)
            if bbox is None:
                bbox = item.get("gt_bbox_norm", None)
            if bbox is None and "gt_bbox_abs" in item:
                w = item.get("img_width", 1)
                h = item.get("img_height", 1)
                ab = item["gt_bbox_abs"]
                if len(ab) >= 4 and w > 0 and h > 0:
                    bbox = [ab[0] / w, ab[1] / h, ab[2] / w, ab[3] / h]
            gt_point = item.get("gt_point", item.get("point", None))

            # 先从原始 response 解析 GT point（在坐标被替换之前）
            if bbox is None and gt_point is None:
                gt_point = _try_parse_point_from_text(response)

            # 始终用 model-specific ground_response 格式替换 gpt 部分
            # 不使用 inject_ground_token_into_response()，因为它硬编码替换为 UITARS 的
            # GROUND_RESPONSE_CLICK 格式（click(start_box='...')），对 guiowl/uivenus 不适用。
            # 正确做法：始终用 self._ground_response（由 MODEL_TYPE_CONSTANTS 决定），
            # 这样 uitars → GROUND_RESPONSE_CLICK, guiowl → GUI_OWL_GROUND_RESPONSE,
            #      uivenus → UI_VENUS_GROUND_RESPONSE，每个模型训练时格式完全一致。
            response = self._ground_response

            human_text = self._user_prompt_template.format(query) if self._user_prompt_template else query
            convs = [
                {"from": "human", "value": f"<image>\n{human_text}"},
                {"from": "gpt", "value": response},
            ]

            return {
                "image_path": img_path,
                "conversations": convs,
                "bbox": bbox,
                "gt_point": gt_point,
            }

        return None

    @staticmethod
    def _try_parse_point(text: str) -> Optional[List[float]]:
        """Try to extract (x, y) from various coordinate formats."""
        return _try_parse_point_from_text(text)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i) -> Optional[Dict[str, torch.Tensor]]:
        max_retries = min(10, len(self.samples))
        for attempt in range(max_retries):
            try:
                idx = i if attempt == 0 else random.randint(0, len(self.samples) - 1)
                sample = self._get_item(idx)
                if sample is not None:
                    return sample
            except Exception as e:
                rank0_print(f"[RetrofitDataset] Failed to load sample {idx} (attempt {attempt+1}): {e}")
        # If all retries fail, return None (collator will filter it out)
        rank0_print(f"[RetrofitDataset] All {max_retries} retries failed for index {i}, returning None")
        return None

    def _get_item(self, i: int) -> Optional[Dict[str, torch.Tensor]]:
        sample = self.samples[i]
        image_path = sample["image_path"]
        conversations = sample["conversations"]
        bbox = sample.get("bbox", None)
        gt_point = sample.get("gt_point", None)

        # ── Load image ─────────────────────────────────────────────────────
        if not image_path or not os.path.exists(image_path):
            return None
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            rank0_print(f"  Failed to open image {image_path}: {e}")
            return None

        # ── Tokenize conversation ──────────────────────────────────────────
        data_dict = self._preprocess(
            conversations=conversations,
            image=image,
            image_path=image_path,
        )
        if data_dict is None:
            return None

        input_ids = data_dict["input_ids"]

        # ── Compute patch label ─────────────────────────────────────────────
        image_processor = self.processor.image_processor
        # Resize image to match what processor would use
        # We use the *processed* image size for patch label computation
        processed_w = data_dict.get("image_width", image.width)
        processed_h = data_dict.get("image_height", image.height)
        resized_image = image.resize((processed_w, processed_h), Image.LANCZOS)

        patch_label = None
        if bbox is not None:
            try:
                if self.gt_label_type == "gaussian":
                    patch_label = get_patch_gaussian_label_from_bbox(
                        image_processor, resized_image, bbox,
                        sigma_factor=self.gaussian_sigma_factor,
                    )
                else:
                    patch_label = get_patch_binary_label_from_bbox(
                        image_processor, resized_image, bbox
                    )
            except Exception as e:
                patch_label = None
        if patch_label is None and gt_point is not None:
            try:
                # ⚠️ gt_point 坐标空间关键检查：
                # - OS-Atlas / GUI-Actor 格式：归一化 [0,1]  → 直接使用
                # - UI-TARS-1.5 click(start_box='<|box_start|>(x,y)<|box_end|>') 格式：
                #   绝对像素坐标，相对于模型实际接收的 smart_resize 后图像尺寸
                #   （来源：action_parser.py parse_action_to_structure_output，
                #    qwen25vl 分支用 smart_resize_width/height 归一化，不是原图尺寸）
                #   判断方法：如果 max(x,y) > 1，认为是绝对像素
                gx, gy = float(gt_point[0]), float(gt_point[1])
                if max(gx, gy) > 1.0:
                    # 绝对像素坐标 → 除以 smart_resize 后的图像尺寸（即 processed_w/h）进行归一化
                    # 注意：必须用 processed_w/processed_h，不能用原图 image.width/height
                    if processed_w > 0 and processed_h > 0:
                        gx = gx / processed_w
                        gy = gy / processed_h
                patch_label = get_patch_soft_label_from_point(
                    image_processor, resized_image, gx, gy
                )
            except Exception as e:
                patch_label = None

        if patch_label is None:
            # No grounding supervision for this sample; skip grounding loss
            patch_label = torch.zeros(1)   # placeholder

        # ── Find <|ground|> anchor token index ────────────────────────────
        # Priority: explicit <|ground|> token > before <|pointer_start|> > None
        # NOTE: We do NOT fall back to "last non-pad" here — that would risk label leakage
        # for UI-TARS (last token may be after coordinates). If this returns None,
        # modeling_uitars._find_ground_anchor() will handle further fallback dynamically
        # with appropriate warnings (P3: after vision_end, P5: last non-pad with warning).
        ground_token_idx = get_ground_token_idx_in_sequence(
            input_ids, self.ground_token_id, self.pointer_start_token_id
        )
        # ground_token_idx=None is acceptable; forward() handles it via _find_ground_anchor

        # ── Check sequence length ──────────────────────────────────────────
        if len(input_ids) > self.tokenizer.model_max_length:
            return None

        return {
            "input_ids": data_dict["input_ids"],
            "labels": data_dict["labels"],
            "pixel_values": data_dict["pixel_values"],
            "image_grid_thw": data_dict["image_grid_thw"],
            "ground_token_indices": ground_token_idx,
            "multi_patch_labels": patch_label,
        }

    def _preprocess(
        self,
        conversations: List[Dict],
        image: Image.Image,
        image_path: str,
    ) -> Optional[Dict]:
        """
        Tokenize conversations with image using the model's processor
        (Qwen2.5-VL for uitars, Qwen3-VL for guiowl/uivenus).
        Builds input_ids and labels (mask human turns with IGNORE_INDEX).

        Returns dict with: input_ids, labels, pixel_values, image_grid_thw,
                            image_width, image_height
        """
        tokenizer = self.tokenizer
        processor = self.processor

        # Build messages list for chat template
        messages = []

        # System message (None or empty → skip; UI-Venus has no system message)
        if self._system_message:
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": self._system_message}],
            })

        # User/assistant turns
        for conv in conversations:
            role = "user" if conv["from"] == "human" else "assistant"
            value = conv["value"]

            if "<image>" in value and role == "user":
                # Replace <image> with actual image content for processor
                content = [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": value.replace("<image>\n", "").replace("<image>", "").strip()},
                ]
                messages.append({"role": role, "content": content})
            else:
                messages.append({
                    "role": role,
                    "content": [{"type": "text", "text": value}],
                })

        # Apply chat template to get text.
        # We use processor.tokenizer.chat_template (set to CHAT_TEMPLATE from constants)
        # rather than falling back to the base model's template, to ensure our
        # custom special tokens (<|ground|> etc.) are handled correctly.
        try:
            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            # Use process_vision_info to get image input
            image_inputs, _ = process_vision_info(messages)

            # Processor call
            model_inputs = processor(
                text=[text],
                images=image_inputs if image_inputs else None,
                return_tensors="pt",
            )
        except Exception as e:
            rank0_print(f"  Processor error: {e}")
            return None

        input_ids = model_inputs["input_ids"][0]   # [seq_len]

        # Build labels: mask everything except assistant turns
        labels = torch.full_like(input_ids, IGNORE_INDEX)

        # Find assistant turn boundaries by re-tokenizing
        # Approach: tokenize system + user portion, then label assistant portion
        try:
            # Build prefix (system + user only)
            prefix_messages = [m for m in messages if m["role"] != "assistant"]
            prefix_text = processor.apply_chat_template(
                prefix_messages,
                tokenize=False,
                add_generation_prompt=True,  # adds "<|im_start|>assistant\n"
            )
            prefix_image_inputs, _ = process_vision_info(prefix_messages)
            prefix_inputs = processor(
                text=[prefix_text],
                images=prefix_image_inputs if prefix_image_inputs else None,
                return_tensors="pt",
            )
            prefix_len = prefix_inputs["input_ids"].shape[1]
        except Exception as e:
            # Prefix tokenization failed: we cannot reliably determine where the
            # assistant response starts, so we cannot build correct labels.
            # len(input_ids)//2 would wrongly include image tokens as labels → skip sample.
            rank0_print(f"  Prefix tokenization error (skipping sample): {e}")
            return None

        # Set labels for assistant tokens
        labels[prefix_len:] = input_ids[prefix_len:]

        # Don't supervise special tokens in assistant response that are just formatting
        # (Keep POINTER tokens as supervision since we want to generate them)

        pixel_values = model_inputs.get("pixel_values", None)
        image_grid_thw = model_inputs.get("image_grid_thw", None)

        if pixel_values is not None:
            pixel_values = pixel_values  # [n_patches, 3*merge^2*patch^2]
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw  # [1, 3] = [T, H, W]

        # Compute processed image dimensions for patch label.
        #
        # image_grid_thw [T, H, W] (same for Qwen2.5-VL uitars and Qwen3-VL guiowl/uivenus):
        #   H, W = raw patch count (each patch is patch_size × patch_size pixels).
        #   patch_size: 14 for Qwen2.5-VL, 16 for Qwen3-VL (read dynamically).
        #   Actual pixel size = H * patch_size × W * patch_size.
        #   Visual tokens = T * (H / merge_size) * (W / merge_size).
        #
        # When we call get_patch_*_label(image), the function computes:
        #   token_cell_size = patch_size * merge_size
        #   grid_h = img_h // token_cell_size = (H * patch_size) // (patch_size * merge_size)
        #          = H // merge_size
        # This gives label size = H//merge_size * W//merge_size = visual token count. ✓
        processed_w, processed_h = image.width, image.height
        if image_grid_thw is not None:
            _, H, W = image_grid_thw[0].tolist()
            patch_size = processor.image_processor.patch_size  # 14 (Qwen2.5-VL) or 16 (Qwen3-VL)
            # Image pixel size = raw patch grid × patch size (NOT × merge_size)
            processed_w = int(W * patch_size)
            processed_h = int(H * patch_size)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "image_width": processed_w,
            "image_height": processed_h,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Data collator
# ─────────────────────────────────────────────────────────────────────────────

class RetrofitDataCollator:
    """
    Collate function for RetrofitDataset.
    Pads input_ids and labels; stacks pixel_values; collects grounding supervision.
    """

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances: List[Dict]) -> Dict[str, object]:
        # Filter out None samples
        instances = [inst for inst in instances if inst is not None]
        if not instances:
            raise ValueError("All samples in batch are None!")

        input_ids = [inst["input_ids"] for inst in instances]
        labels = [inst["labels"] for inst in instances]
        ground_indices_raw = [inst.get("ground_token_indices") for inst in instances]

        # Truncate to model_max_length
        max_len = self.tokenizer.model_max_length
        orig_lens = [len(ids) for ids in input_ids]
        input_ids = [ids[:max_len] for ids in input_ids]
        labels = [lbl[:max_len] for lbl in labels]

        # If truncation occurred AND the pre-computed ground_token_idx falls in the
        # truncated region, invalidate it (set None) so _find_ground_anchor will
        # re-scan the truncated sequence and find the correct position.
        ground_indices_raw = [
            idx if (idx is None or idx < len(ids))
            else None
            for idx, ids in zip(ground_indices_raw, input_ids)
        ]

        # Pad
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        max_seq_len = max(len(ids) for ids in input_ids)
        padded_input_ids = torch.zeros(len(input_ids), max_seq_len, dtype=torch.long).fill_(pad_id)
        padded_labels = torch.zeros(len(labels), max_seq_len, dtype=torch.long).fill_(IGNORE_INDEX)

        for j, (ids, lbl) in enumerate(zip(input_ids, labels)):
            padded_input_ids[j, :len(ids)] = ids
            padded_labels[j, :len(lbl)] = lbl

        batch = {
            "input_ids": padded_input_ids,
            "labels": padded_labels,
            "attention_mask": padded_input_ids.ne(pad_id),
        }

        # Pixel values (all images from the batch concatenated)
        if instances[0].get("pixel_values") is not None:
            batch["pixel_values"] = torch.cat(
                [inst["pixel_values"] for inst in instances if inst.get("pixel_values") is not None],
                dim=0,
            )
            batch["image_grid_thw"] = torch.cat(
                [inst["image_grid_thw"] for inst in instances if inst.get("image_grid_thw") is not None],
                dim=0,
            )

        # Grounding supervision (use ground_indices_raw which was already validated
        # against post-truncation sequence lengths above)
        batch["ground_token_indices"] = ground_indices_raw
        batch["multi_patch_labels"] = [
            inst["multi_patch_labels"] for inst in instances
        ]

        return batch
