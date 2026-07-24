#!/bin/bash
# run_exp3.sh — Experiment 3: Spatial Lens vs. Serialization Lens
# Runtime estimate: ~45 min on 1 GPU per dataset (4 models × 200 samples × 2 passes)
#
# Qwen2.5-VL models (uitars, guiowl7b): qwen25 conda env
# Qwen3-VL models  (guiowl, uivenus):   qwen3 conda env
#
# Usage:
#   bash run_exp3.sh [CUDA_ID]              # SS-Pro only (default)
#   DATASET=v2 bash run_exp3.sh [CUDA_ID]  # SS-v2 only
#   DATASET=both bash run_exp3.sh [CUDA_ID] # both datasets (generates resolution comparison)
#
# To run a single model:
#   MODEL=guiowl bash run_exp3.sh 0

set -euo pipefail

CUDA_ID="${1:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_ID}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

CKPT_BASE="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/.hdd/ckpt/zwerge"
DATA_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/datasets/evaluation"
PROBE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

N_SAMPLES=200
MAX_PIXELS=6400000
SEED=42
DEVICE="cuda:0"
MODEL="${MODEL:-all}"
DATASET="${DATASET:-pro}"   # pro | v2 | both

run_model() {
    local MODEL_TYPE="$1"
    local CKPT="$2"
    local CONDA_ENV="$3"
    local EVAL_JSON="$4"
    local IMAGE_ROOT="$5"
    local OUTPUT="$6"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[exp3] Model: ${MODEL_TYPE}  |  ckpt: ${CKPT}  |  conda: ${CONDA_ENV}"
    echo "[exp3] eval: ${EVAL_JSON}"
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

run_dataset() {
    local DS="$1"       # "pro" or "v2"
    local OUT_DIR

    if [[ "${DS}" == "pro" ]]; then
        OUT_DIR="${PROBE_DIR}/outputs/exp3"
        EVAL_JSON="${DATA_DIR}/ScreenSpot-Pro/eval.json"
        IMAGE_ROOT="${DATA_DIR}/ScreenSpot-Pro"
    else
        OUT_DIR="${PROBE_DIR}/outputs/exp3_v2"
        EVAL_JSON="${DATA_DIR}/ScreenSpot-v2/eval.json"
        IMAGE_ROOT="${DATA_DIR}/ScreenSpot-v2"
    fi
    mkdir -p "${OUT_DIR}"

    if [[ "${MODEL}" == "all" || "${MODEL}" == "uitars" ]]; then
        run_model uitars   "${CKPT_BASE}/uitars_A7_exp001/checkpoint-3129" \
                  qwen25 "${EVAL_JSON}" "${IMAGE_ROOT}" "${OUT_DIR}/uitars_lens.jsonl"
    fi
    if [[ "${MODEL}" == "all" || "${MODEL}" == "guiowl7b" ]]; then
        run_model guiowl7b "${CKPT_BASE}/guiowl7b_A7_exp001/checkpoint-3129" \
                  qwen25 "${EVAL_JSON}" "${IMAGE_ROOT}" "${OUT_DIR}/guiowl7b_lens.jsonl"
    fi
    if [[ "${MODEL}" == "all" || "${MODEL}" == "guiowl" ]]; then
        run_model guiowl   "${CKPT_BASE}/guiowl_A7_exp002/checkpoint-3130" \
                  qwen3 "${EVAL_JSON}" "${IMAGE_ROOT}" "${OUT_DIR}/guiowl_lens.jsonl"
    fi
    if [[ "${MODEL}" == "all" || "${MODEL}" == "uivenus" ]]; then
        run_model uivenus  "${CKPT_BASE}/uivenus_A7_exp002/checkpoint-3130" \
                  qwen3 "${EVAL_JSON}" "${IMAGE_ROOT}" "${OUT_DIR}/uivenus_lens.jsonl"
    fi
}

# ── Run selected dataset(s) ───────────────────────────────────────────────────

if [[ "${DATASET}" == "pro" || "${DATASET}" == "both" ]]; then
    echo "[exp3] Running on SS-Pro …"
    run_dataset pro
fi

if [[ "${DATASET}" == "v2" || "${DATASET}" == "both" ]]; then
    echo "[exp3] Running on SS-v2 …"
    run_dataset v2
fi

# ── Generate figures ──────────────────────────────────────────────────────────

echo ""
echo "[exp3] All runs done. Generating figures …"

PLOT_ARGS=(
    --exp3_dir   "${PROBE_DIR}/outputs/exp3"
    --output_dir "${PROBE_DIR}/outputs/figures"
)
if [[ "${DATASET}" == "both" || "${DATASET}" == "v2" ]]; then
    PLOT_ARGS+=(--exp3_dir_v2 "${PROBE_DIR}/outputs/exp3_v2")
fi

conda run -n qwen25 --no-capture-output \
    python "${PROBE_DIR}/plot_probes.py" "${PLOT_ARGS[@]}"

echo "[exp3] Figures saved to ${PROBE_DIR}/outputs/figures/"
