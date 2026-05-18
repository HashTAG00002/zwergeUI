#!/bin/bash
# =========================================================
# ZwerGe-UI Evaluation Runner
# =========================================================
#
# Python 环境：使用 gui_aima conda 环境（含 torch 2.6+, transformers 4.51+）
# 如需切换，修改 PYTHON 变量。
#
# 用法示例：
#
#   # 评测最新 checkpoint 在 ScreenSpot-Pro 上
#   bash run_eval.sh ss_pro
#
#   # 评测所有 benchmark
#   bash run_eval.sh all
#
#   # 指定 checkpoint（不用最新的）
#   CKPT=.../uitars7b_grounding50k_20260509_032324 bash run_eval.sh ss_pro
#
#   # 评测特定 checkpoint 子目录（如 checkpoint-1500）
#   CKPT=... CKPT_SUBFOLDER=checkpoint-1500 bash run_eval.sh ss_pro
#
#   # 多卡并行（以 SS-Pro 1581条为例，4卡）：
#   CUDA_VISIBLE_DEVICES=0 bash run_eval.sh ss_pro 0    396  &
#   CUDA_VISIBLE_DEVICES=1 bash run_eval.sh ss_pro 396  792  &
#   CUDA_VISIBLE_DEVICES=2 bash run_eval.sh ss_pro 792  1188 &
#   CUDA_VISIBLE_DEVICES=3 bash run_eval.sh ss_pro 1188 -1   &
#   wait
#   # 合并分片（运行一次）：
#   python eval_zwerge.py --ckpt "$CKPT" --bench ss_pro --aggregate --no_save_per_sample
#
# bench 参数：ss_pro | ss_v2 | osworld_g | mmbench | ui_vision | all
# =========================================================

set -euo pipefail

# ── 路径配置 ─────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZWERGE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Python 环境（含 torch / transformers / qwen_vl_utils 等）
# qwen25 环境：torch 2.7, transformers 4.51，无 triton 编译问题
PYTHON="${PYTHON:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/conda/envs/qwen25/bin/python3}"

# 命令行参数
BENCH="${1:-ss_pro}"    # bench key
START="${2:-0}"          # start index (inclusive)
END="${3:--1}"           # end index (exclusive), -1 = all

# ── 数据/结果路径 ─────────────────────────────────────────────────────────────
EVAL_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/datasets/evaluation"
RESULT_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/zwerge/data/results/zwerge"
CKPT_ROOT="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/.hdd/ckpt/zwerge"

# ── Checkpoint（自动选取最新）────────────────────────────────────────────────
if [[ -z "${CKPT:-}" ]]; then
    CKPT=$(ls -d "${CKPT_ROOT}"/uitars7b_grounding50k_* 2>/dev/null | sort | tail -1)
    if [[ -z "$CKPT" ]]; then
        echo "[ERROR] No checkpoint found under ${CKPT_ROOT}"
        echo "  Set CKPT env variable manually."
        exit 1
    fi
fi
echo "[run_eval] Checkpoint: ${CKPT}"

# 可选：加载某个 checkpoint-XXXX 子目录
CKPT_SUBFOLDER="${CKPT_SUBFOLDER:-}"

# ── GPU & Attention ───────────────────────────────────────────────────────────
DEVICE="${DEVICE:-cuda:0}"
ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"

# ── 推理超参 ─────────────────────────────────────────────────────────────────
TOPK="${TOPK:-3}"
MAX_PIXELS="${MAX_PIXELS:-5760000}"          # ~2400×2400，覆盖高分辨率截图
ACTIVATION_THRESHOLD="${ACTIVATION_THRESHOLD:-0.3}"

# ── 构造命令 ─────────────────────────────────────────────────────────────────
CMD="${PYTHON} ${SCRIPT_DIR}/eval_zwerge.py"
CMD="${CMD} --ckpt ${CKPT}"
CMD="${CMD} --bench ${BENCH}"
CMD="${CMD} --eval_dir ${EVAL_DIR}"
CMD="${CMD} --output_dir ${RESULT_DIR}"
CMD="${CMD} --attn_impl ${ATTN_IMPL}"
CMD="${CMD} --device ${DEVICE}"
CMD="${CMD} --topk ${TOPK}"
CMD="${CMD} --max_pixels ${MAX_PIXELS}"
CMD="${CMD} --activation_threshold ${ACTIVATION_THRESHOLD}"
CMD="${CMD} --start ${START}"
CMD="${CMD} --end ${END}"

if [[ -n "${CKPT_SUBFOLDER}" ]]; then
    CMD="${CMD} --ckpt_subfolder ${CKPT_SUBFOLDER}"
fi

echo "[run_eval] Bench: ${BENCH},  Slice: [${START}, ${END}]"
echo "[run_eval] PYTHONPATH: ${ZWERGE_ROOT}/src"
echo ""

# ── 执行 ─────────────────────────────────────────────────────────────────────
cd "${SCRIPT_DIR}"
export PYTHONPATH="${ZWERGE_ROOT}/src:${PYTHONPATH:-}"

echo "[run_eval] Running: ${CMD}"
eval "${CMD}"

echo ""
echo "[run_eval] Done. Results in: ${RESULT_DIR}/$(basename ${CKPT})"
