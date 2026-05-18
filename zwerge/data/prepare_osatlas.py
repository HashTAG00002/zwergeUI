"""
OS-Atlas Data Preparation Utility

Scans the OS-Atlas dataset directory and generates a YAML config
pointing to all available JSON files. Also verifies that the image
files referenced by the JSONs actually exist.

Usage:
    python data/prepare_osatlas.py \
        --osatlas_root /path/to/OS-Atlas \
        --output_yaml data/train_data.yaml \
        --verify_images   # optional: check image existence

Output YAML format (compatible with RetrofitDataset):
    datasets:
      - json_path: /path/to/data.json
        images_folder: /path/to/screenshots
        sampling_strategy: all
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import yaml


# ─────────────────────────────────────────────────────────────────────────────
# Known OS-Atlas dataset structure
# ─────────────────────────────────────────────────────────────────────────────
KNOWN_SPLITS = {
    "desktop/linux": {
        "json": "linux_splited.json",
        "images": "screenshots",
    },
    # Add more as you discover them:
    # "desktop/windows": {"json": "...", "images": "screenshots"},
    # "desktop/macos": {"json": "...", "images": "screenshots"},
    # "web/fineweb": {"json": "...", "images": "screenshots"},
    # "web/seeclick": {"json": "...", "images": "screenshots"},
    # "mobile/rico": {"json": "...", "images": "screenshots"},
    # "mobile/android_world": {"json": "...", "images": "screenshots"},
}


def find_json_files(root: str):
    """Recursively find all .json files under root (up to 2 levels deep)."""
    results = []
    root_path = Path(root)
    for subdir in root_path.iterdir():
        if not subdir.is_dir():
            continue
        for subsubdir in [subdir] + list(subdir.iterdir()):
            if not subsubdir.is_dir():
                continue
            for f in subsubdir.iterdir():
                if f.suffix == ".json" and f.is_file():
                    # Try to find screenshots directory
                    screenshots = subsubdir / "screenshots"
                    if not screenshots.exists():
                        screenshots = subsubdir
                    results.append({
                        "json_path": str(f),
                        "images_folder": str(screenshots),
                        "num_samples": None,
                    })
    return results


def count_samples(json_path: str) -> int:
    try:
        with open(json_path) as f:
            data = json.load(f)
        if isinstance(data, list):
            # OS-Atlas raw format: count elements
            total = sum(len(item.get("elements", [1])) for item in data)
            return total
        return len(data)
    except Exception as e:
        print(f"  Warning: could not read {json_path}: {e}")
        return 0


def verify_sample_images(json_path: str, images_folder: str, n_check: int = 20) -> dict:
    """Spot-check that image files exist."""
    try:
        with open(json_path) as f:
            data = json.load(f)
    except Exception:
        return {"ok": 0, "missing": 0}

    if not isinstance(data, list):
        return {"ok": 0, "missing": 0}

    samples = random.sample(data, min(n_check, len(data)))
    ok, missing = 0, 0
    for item in samples:
        img = item.get("img_filename", item.get("image", ""))
        if img:
            full_path = os.path.join(images_folder, img)
            if os.path.exists(full_path):
                ok += 1
            else:
                missing += 1
    return {"ok": ok, "missing": missing}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--osatlas_root", default="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/datasets/OS-Atlas")
    parser.add_argument("--output_yaml", default="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/zwerge/code/zwerge/data/train_data.yaml")
    parser.add_argument("--verify_images", action="store_true")
    parser.add_argument("--max_per_dataset", type=int, default=None,
                        help="Cap each dataset at this many samples (None = all)")
    args = parser.parse_args()

    print(f"Scanning {args.osatlas_root}...")
    found = find_json_files(args.osatlas_root)

    datasets_cfg = []
    total_samples = 0

    for entry in found:
        json_path = entry["json_path"]
        images_folder = entry["images_folder"]

        n = count_samples(json_path)
        print(f"  {json_path}: ~{n} samples")

        if args.verify_images:
            check = verify_sample_images(json_path, images_folder)
            print(f"    Image check: {check['ok']} ok, {check['missing']} missing (out of 20)")

        if n == 0:
            continue

        if args.max_per_dataset and n > args.max_per_dataset:
            sampling = f"random:{args.max_per_dataset}"
            effective_n = args.max_per_dataset
        else:
            sampling = "all"
            effective_n = n

        total_samples += effective_n
        datasets_cfg.append({
            "json_path": json_path,
            "images_folder": images_folder,
            "sampling_strategy": sampling,
        })

    # Write YAML
    cfg = {"datasets": datasets_cfg}
    os.makedirs(os.path.dirname(args.output_yaml), exist_ok=True)
    with open(args.output_yaml, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

    print(f"\nGenerated {args.output_yaml}")
    print(f"Total datasets: {len(datasets_cfg)}")
    print(f"Total samples (after sampling): ~{total_samples:,}")


if __name__ == "__main__":
    main()
