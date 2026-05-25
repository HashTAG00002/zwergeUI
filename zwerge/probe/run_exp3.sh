#!/bin/bash
# run_exp3.sh — Experiment 3: Spatial Lens vs. Serialization Lens
# Runtime estimate: ~45 min on 1 GPU (4 models × 200 samples × 2 passes)
#
# Qwen2.5-VL models (uitars, guiowl7b): qwen25 conda env
# Qwen3-VL models  (guiowl, uivenus):   qwen3-verl conda env
#
# Usage:
#   bash run_exp3.sh [CUDA_ID]
#
# To run a single model:
#   MODEL=uitars bash run_exp3.sh 0

set -euo pipefail

CUDA_ID="${1:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_ID}"

CKPT_BASE="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/.hdd/ckpt/zwerge"
DATA_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/datasets/evaluation"
PROBE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${PROBE_DIR}/outputs/exp3"
EVAL_JSON="${DATA_DIR}/ScreenSpot-Pro/eval.json"
IMAGE_ROOT="${DATA_DIR}/ScreenSpot-Pro"

mkdir -p "${OUT_DIR}"

N_SAMPLES=200
MAX_PIXELS=6400000
SEED=42
DEVICE="cuda:0"

run_model() {
    local MODEL_TYPE="$1"
    local CKPT="$2"
    local CONDA_ENV="$3"
    local OUTPUT="${OUT_DIR}/${MODEL_TYPE}_lens.jsonl"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[exp3] Model: ${MODEL_TYPE}  |  ckpt: ${CKPT}  |  conda: ${CONDA_ENV}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    conda run -n "${CONDA_ENV}" --no-capture-output \
        python "${PROBE_DIR}/exp3_serialization_lens.py" \
            --ckpt        "${CKPT}" \
            --model_type  "${MODEL_TYPE}" \
            --eval_json   "${EVAL_JSON}" \
            --image_root  "${IMAGE_ROOT}" \
            --output      "${OUTPUT}" \
            --n_samples   "${N_SAMPLES}" \
            --max_pixels  "${MAX_PIXELS}" \
            --device      "${DEVICE}" \
            --seed        "${SEED}"
}

MODEL="${MODEL:-all}"

if [[ "${MODEL}" == "all" || "${MODEL}" == "uitars" ]]; then
    run_model uitars   "${CKPT_BASE}/uitars_A7_exp001/checkpoint-2800"   qwen25
fi

if [[ "${MODEL}" == "all" || "${MODEL}" == "guiowl7b" ]]; then
    run_model guiowl7b "${CKPT_BASE}/guiowl7b_A7_exp001/checkpoint-2800" qwen25
fi

if [[ "${MODEL}" == "all" || "${MODEL}" == "guiowl" ]]; then
    run_model guiowl   "${CKPT_BASE}/guiowl_A7_exp002/checkpoint-2800"   qwen3-verl
fi

if [[ "${MODEL}" == "all" || "${MODEL}" == "uivenus" ]]; then
    run_model uivenus  "${CKPT_BASE}/uivenus_A7_exp002/checkpoint-2800"  qwen3-verl
fi

echo ""
echo "[exp3] All done. Generating figures …"
conda run -n qwen25 --no-capture-output \
    python "${PROBE_DIR}/plot_probes.py" \
        --exp3_dir   "${OUT_DIR}" \
        --output_dir "${PROBE_DIR}/outputs/figures"

echo "[exp3] Figures saved to ${PROBE_DIR}/outputs/figures/"
