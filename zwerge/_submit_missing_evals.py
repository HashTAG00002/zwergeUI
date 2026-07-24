#!/usr/bin/env python3
"""
手动提交所有无 results 的 checkpoint 评估任务。

覆盖 4 个实验下共 15 个 ckpt：
  guiowl7b_A7_exp001 : checkpoint-400,800,1200,1600,2000,2400,2800,3100,3129 (Qwen2.5-VL, eval_zwerge.hope, eval_zwerge_guiowl7b.sh)
  uitars_A7_exp001   : checkpoint-3100, 3129         (Qwen2.5-VL, eval_zwerge.hope, eval_zoom_backbone.sh)
  guiowl_A7_exp002   : checkpoint-3100, 3130         (Qwen3-VL,   eval_zwerge_qwen3.hope, eval_zoom_backbone.sh)
  uivenus_A7_exp002  : checkpoint-3100, 3130         (Qwen3-VL,   eval_zwerge_qwen3.hope, eval_zoom_backbone.sh)

用法：
  python _submit_missing_evals.py           # 实际提交
  python _submit_missing_evals.py --dry_run # 仅打印 hope 文件，不提交
"""

import argparse
import datetime
import json
import logging
import os
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from eval_daemon import generate_hope_file, hope_run, _tmp_dir, load_submitted, save_submitted

ZWERGE_ROOT = pathlib.Path(__file__).parent.resolve()
BASE_CKPT_DIR = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/.hdd/ckpt/zwerge"

# ─────────────────────────────────────────────────────────────────────────────
# 任务清单
# 字段说明：
#   exp_name        实验名（用于 output_dir 和 .eval_submitted.json 定位）
#   ckpt_steps      需要补评的 step 列表
#   hope_template   相对 ZWERGE_ROOT 的 hope 模板路径
#   bash_script     相对 ZWERGE_ROOT 的 bash 脚本路径（覆盖 hope 模板中的 worker.script）
#   env             注入 worker.script 前缀的环境变量
#   bench           评估 bench（all/ss_pro/...）
#   queue           覆盖 hope 模板中的 queue（None=保留模板默认值）
# ─────────────────────────────────────────────────────────────────────────────

EVAL_TASKS = [
    # ── guiowl7b_A7_exp001 (Qwen2.5-VL) ────────────────────────────────────
    {
        "exp_name":      "guiowl7b_A7_exp001",
        "ckpt_steps":    [400, 800, 1200, 1600, 2000, 2400, 2800, 3100, 3129],
        "hope_template": "scripts/eval/eval_zwerge.hope",
        "bash_script":   "scripts/eval/eval_zwerge_guiowl7b.sh",
        "env": {
            "MODEL_TYPE":       "guiowl7b",
            "DECODE_STRATEGY":  "centroid",
            "ZOOM_PADDING_CELLS": "3",
        },
        "bench": "all",
        "queue": "root.zw05_training_cluster.hadoop-aipnlp.elastic",
    },
    # ── uitars_A7_exp001 (Qwen2.5-VL) ──────────────────────────────────────
    {
        "exp_name":      "uitars_A7_exp001",
        "ckpt_steps":    [3100, 3129],
        "hope_template": "scripts/eval/eval_zwerge.hope",
        "bash_script":   "scripts/eval/eval_zoom_backbone.sh",
        "env": {
            "MODEL_TYPE":       "uitars",
            "DECODE_STRATEGY":  "centroid",
            "ZOOM_PADDING_CELLS": "3",
        },
        "bench": "all",
        "queue": "root.zw05_training_cluster.hadoop-aipnlp.elastic",
    },
    # ── guiowl_A7_exp002 (Qwen3-VL) ────────────────────────────────────────
    {
        "exp_name":      "guiowl_A7_exp002",
        "ckpt_steps":    [3100, 3130],
        "hope_template": "scripts/eval/eval_zwerge_qwen3.hope",
        "bash_script":   "scripts/eval/eval_zoom_backbone.sh",
        "env": {
            "MODEL_TYPE":       "guiowl",
            "DECODE_STRATEGY":  "centroid",
            "ZOOM_PADDING_CELLS": "3",
        },
        "bench": "all",
        "queue": "root.zw05_training_cluster.hadoop-vision.elastic_job",
    },
    # ── uivenus_A7_exp002 (Qwen3-VL) ───────────────────────────────────────
    {
        "exp_name":      "uivenus_A7_exp002",
        "ckpt_steps":    [3100, 3130],
        "hope_template": "scripts/eval/eval_zwerge_qwen3.hope",
        "bash_script":   "scripts/eval/eval_zoom_backbone.sh",
        "env": {
            "MODEL_TYPE":       "uivenus",
            "DECODE_STRATEGY":  "centroid",
            "ZOOM_PADDING_CELLS": "3",
        },
        "bench": "all",
        "queue": "root.zw05_training_cluster.hadoop-vision.elastic_job",
    },
]


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("submit_missing")
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(ch)
    return logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    logger = setup_logger()
    dry_run = args.dry_run
    if dry_run:
        logger.info("=== DRY RUN MODE — hope files will be printed but NOT submitted ===")

    total_submitted = 0
    total_skipped = 0

    for task in EVAL_TASKS:
        exp_name      = task["exp_name"]
        output_dir    = pathlib.Path(BASE_CKPT_DIR) / exp_name
        template_path = str(ZWERGE_ROOT / task["hope_template"])
        bash_script   = str(ZWERGE_ROOT / task["bash_script"])
        bench         = task["bench"]
        queue         = task.get("queue")

        submitted = load_submitted(output_dir)

        for step in task["ckpt_steps"]:
            ckpt_name    = f"checkpoint-{step}"
            ckpt_path    = output_dir / ckpt_name
            results_dir  = ckpt_path / "results"

            # 跳过已有 results 的 ckpt（幂等保护）
            if results_dir.exists():
                logger.info(f"[SKIP] {exp_name}/{ckpt_name}: results already exist")
                total_skipped += 1
                continue

            if not ckpt_path.exists():
                logger.warning(f"[WARN] {exp_name}/{ckpt_name}: ckpt dir not found, skipping")
                total_skipped += 1
                continue

            output_dir_final = results_dir

            env_vars = {
                "CKPT": str(ckpt_path),
                "OUTPUT_DIR_FINAL": str(output_dir_final),
            }
            env_vars.update(task["env"])

            # 临时 hope 文件路径
            tmp_dir = output_dir / ".eval_daemon_tmp"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            out_hope = tmp_dir / f"eval_{exp_name}_ckpt{step}_manual.hope"

            worker_script = generate_hope_file(
                template_path,
                str(out_hope),
                env_vars,
                positional_args=bench,
                bash_script=bash_script,
                queue=queue,
            )

            logger.info(f"\n{'='*60}")
            logger.info(f"[SUBMIT] {exp_name}/{ckpt_name}")
            logger.info(f"  hope       = {out_hope}")
            logger.info(f"  bash       = {bash_script}")
            logger.info(f"  queue      = {queue}")
            logger.info(f"  worker.script = {worker_script}")
            logger.info(f"--- hope file contents ---")
            logger.info(out_hope.read_text())
            logger.info(f"--- end hope file ---")

            ok = hope_run(out_hope, logger, dry_run=dry_run)

            if ok and not dry_run:
                submitted[ckpt_name] = {
                    "submitted_at": datetime.datetime.now().isoformat(),
                    "status": "submitted",
                    "ckpt_path": str(ckpt_path),
                    "wandb_logged": False,
                }
                save_submitted(output_dir, submitted)
                logger.info(f"[OK] Recorded in .eval_submitted.json: {ckpt_name}")
                total_submitted += 1
            elif ok and dry_run:
                total_submitted += 1
            else:
                logger.error(f"[FAIL] Submission failed for {exp_name}/{ckpt_name}")

    logger.info(f"\n{'='*60}")
    logger.info(f"Done. submitted={total_submitted}, skipped={total_skipped}")


if __name__ == "__main__":
    main()
