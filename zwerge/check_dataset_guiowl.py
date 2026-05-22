"""
检查 guiowl 训练数据的 patch label 对齐问题
关键问题：
  1. processor.image_processor.patch_size 是否等于 16？
  2. dataset 计算的 processed_w/h 是否正确？
  3. patch_label.shape[0] 是否 == n_vis?
  4. image_grid_thw 的值是否正确？
"""
import os
import sys
import json

sys.path.insert(0, "src")

from transformers import AutoProcessor

GUIOWL_CKPT = (
    "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03"
    "/.hdd/ckpt/zwerge/guiowl_grounding50k_A3-gaussian_cos_meta_20260521_210923"
    "/checkpoint-800"
)

UITARS_CKPT = (
    "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03"
    "/.hdd/ckpt/zwerge/uitars7b_grounding50k_A3-gaussian_cos_meta_L18-27_20260520_095304"
    "/checkpoint-2193"
)

SS_PRO_EVAL_JSON = (
    "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03"
    "/datasets/evaluation/ScreenSpot-Pro/eval.json"
)

from PIL import Image

def load_sample(n=0):
    with open(SS_PRO_EVAL_JSON) as f:
        data = json.load(f)
    sample = data[n]
    base_dir = os.path.dirname(SS_PRO_EVAL_JSON)
    img_path = os.path.join(base_dir, sample["image_path"])
    image = Image.open(img_path).convert("RGB")
    instruction = sample["instruction"]
    gt_bbox_norm = sample.get("gt_bbox_norm", None)
    return image, instruction, gt_bbox_norm, sample, img_path


def check_processor(ckpt_path, label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    processor = AutoProcessor.from_pretrained(ckpt_path)
    ip = processor.image_processor

    print(f"  processor type: {type(processor).__name__}")
    print(f"  image_processor type: {type(ip).__name__}")
    print(f"  patch_size: {getattr(ip, 'patch_size', 'N/A')}")
    print(f"  merge_size: {getattr(ip, 'merge_size', 'N/A')}")
    print(f"  max_pixels: {getattr(ip, 'max_pixels', 'N/A')}")
    print(f"  min_pixels: {getattr(ip, 'min_pixels', 'N/A')}")
    print(f"  temporal_patch_size: {getattr(ip, 'temporal_patch_size', 'N/A')}")

    # Test with an actual image
    image, instruction, gt_bbox_norm, sample, img_path = load_sample(0)
    print(f"\n  Test image: {image.size} ({img_path[-50:]})")

    from qwen_vl_utils import process_vision_info
    from zwerge_retrofit.constants import MODEL_TYPE_CONSTANTS

    # Determine model type from ckpt path
    if "guiowl" in ckpt_path:
        model_type = "guiowl"
    else:
        model_type = "uitars"

    constants = MODEL_TYPE_CONSTANTS[model_type]
    sys_msg = constants["system_message"]
    grd_resp = constants["ground_response"]

    messages = []
    if sys_msg:
        messages.append({"role": "system", "content": [{"type": "text", "text": sys_msg}]})
    messages.append({
        "role": "user",
        "content": [
            {"type": "image", "image": img_path},
            {"type": "text", "text": instruction},
        ],
    })
    messages.append({
        "role": "assistant",
        "content": [{"type": "text", "text": grd_resp}],
    })

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    image_inputs, _ = process_vision_info(messages)
    model_inputs = processor(
        text=[text],
        images=image_inputs if image_inputs else None,
        return_tensors="pt",
    )

    input_ids = model_inputs["input_ids"]
    pixel_values = model_inputs.get("pixel_values")
    image_grid_thw = model_inputs.get("image_grid_thw")

    print(f"\n  input_ids.shape: {input_ids.shape}")
    print(f"  pixel_values.shape: {pixel_values.shape if pixel_values is not None else None}")
    print(f"  image_grid_thw: {image_grid_thw}")

    if image_grid_thw is not None:
        T, H, W = image_grid_thw[0].tolist()
        patch_size = getattr(ip, "patch_size", 14)
        merge_size = getattr(ip, "merge_size", 2)
        print(f"\n  T={T}, H={H}, W={W}")
        print(f"  patch_size={patch_size}, merge_size={merge_size}")
        n_width = int(W) // merge_size
        n_height = int(H) // merge_size
        n_vis = n_width * n_height
        print(f"  n_width={n_width}, n_height={n_height}, n_vis={n_vis}")

        # Check actual visual tokens in sequence
        image_token_id = 151655  # <|image_pad|>
        vis_count = (input_ids[0] == image_token_id).sum().item()
        print(f"  <|image_pad|> count in seq: {vis_count}")
        print(f"  match (n_vis == vis_count): {n_vis == vis_count}")

        # What dataset._preprocess would compute
        processed_w_dataset = int(W) * patch_size
        processed_h_dataset = int(H) * patch_size
        print(f"\n  dataset._preprocess computed processed_w={processed_w_dataset}, processed_h={processed_h_dataset}")

        # Now compute patch label using get_patch_binary_label_from_bbox
        import importlib.util, sys
        from zwerge_retrofit.dataset import get_patch_binary_label_from_bbox, get_patch_gaussian_label_from_bbox

        resized_image = image.resize((processed_w_dataset, processed_h_dataset), Image.LANCZOS)
        print(f"  resized_image.size: {resized_image.size}")

        if gt_bbox_norm:
            b = gt_bbox_norm
            if max(b) > 1:
                b_norm = [v / 1000 for v in b]
            else:
                b_norm = b
            print(f"\n  gt_bbox_norm (0-1): {b_norm}")

            # Binary label
            binary_label = get_patch_binary_label_from_bbox(ip, resized_image, b_norm)
            print(f"  binary_label.shape: {binary_label.shape}")
            print(f"  binary_label.sum(): {binary_label.sum().item()}")
            print(f"  binary_label nonzero: {(binary_label > 0).nonzero().squeeze().tolist()}")
            print(f"  match (label_len == n_vis): {len(binary_label) == n_vis}")

            # Gaussian label
            try:
                gaussian_label = get_patch_gaussian_label_from_bbox(ip, resized_image, b_norm)
                print(f"\n  gaussian_label.shape: {gaussian_label.shape}")
                print(f"  gaussian_label.sum(): {gaussian_label.sum().item():.4f}")
                print(f"  gaussian_label max: {gaussian_label.max().item():.4f}")
                print(f"  gaussian_label argmax: {gaussian_label.argmax().item()}")
                print(f"  match (label_len == n_vis): {len(gaussian_label) == n_vis}")
            except Exception as e:
                print(f"  gaussian_label error: {e}")

    return processor


if __name__ == "__main__":
    print("Checking GUIOwl processor...")
    proc_owl = check_processor(GUIOWL_CKPT, "GUI-Owl-1.5 (Qwen3-VL)")

    print("\n\nChecking UI-TARS processor...")
    proc_tars = check_processor(UITARS_CKPT, "UI-TARS-1.5 (Qwen2.5-VL)")
