#!/bin/bash
# ============================================================
# ZwerGe-UI 评测 — GUI-Owl-7B (Qwen2.5-VL, model_type=guiowl7b)
# ============================================================
# eval_zoom_backbone.sh 里没有 guiowl7b 分支，本脚本专门处理该模型。
# Qwen2.5-VL 架构，与 uitars 相同 backbone → CONDA_ENV=qwen25, MAX_PIXELS=12845056
#
# 用法（eval_daemon.py 自动调用，也可手动）：
#   bash scripts/eval/eval_zwerge_guiowl7b.sh all
#   bash scripts/eval/eval_zwerge_guiowl7b.sh ss_pro
#
# 可选环境变量（eval_daemon.py 注入）：
#   CKPT                checkpoint 路径
#   OUTPUT_DIR_FINAL    输出目录（优先，eval_daemon.py 用此注入）
#   DECODE_STRATEGY     centroid/argmax/peak_shift/temperature（默认 centroid）
#   ZOOM_PADDING_CELLS  ROI 外扩 patch 数（默认 3）
#   ZOOM_MAX_NEW_TOKENS backbone generate 最大 token 数（默认 256）
#   SKIP_VIS            设 1 则只输出指标 JSON（默认 1）
# ============================================================

set -euo pipefail
unset http_proxy https_proxy

if [[ -z "${AFO_ENV_CLUSTER_SPEC:-}" ]]; then
    echo "[guiowl7b_eval] AFO_ENV_CLUSTER_SPEC not set — debug/local mode"
else
    nvidia-smi
    conda config --add envs_dirs /mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/conda/envs
    conda env list
    export NODE_RANK="$(jq -r '.index | tonumber' <<<"$AFO_ENV_CLUSTER_SPEC")"
    export NNODES="$(jq -r '.worker | length' <<<"$AFO_ENV_CLUSTER_SPEC")"
    master=$(jq -r '.worker[0]' <<<"$AFO_ENV_CLUSTER_SPEC")
    export MASTER_ADDR="${master%%:*}"
    export MASTER_PORT="${master##*:}"
    echo "NODE_RANK=$NODE_RANK  NNODES=$NNODES  MASTER_ADDR=$MASTER_ADDR:$MASTER_PORT"
fi

export NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)
echo "[guiowl7b_eval] NPROC_PER_NODE=$NPROC_PER_NODE"
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NPROC_PER_NODE - 1)))
echo "[guiowl7b_eval] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZWERGE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
EVAL_DIR="${ZWERGE_ROOT}/eval"
cd "${EVAL_DIR}"
export PYTHONPATH="${ZWERGE_ROOT}/src:${PYTHONPATH:-}"

BENCH="${1:-all}"

# ── 模型参数（guiowl7b: Qwen2.5-VL, 与 uitars 相同架构）──────────────
MODEL_TYPE="guiowl7b"
CKPT="${CKPT:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/.hdd/ckpt/zwerge/guiowl7b_A7_exp001/checkpoint-2800}"
CONDA_ENV="qwen25"
MAX_PIXELS=12845056

# ── 解码策略 ──────────────────────────────────────────────────────────
DECODE_STRATEGY="${DECODE_STRATEGY:-centroid}"

# ── zoom_backbone 专用参数 ────────────────────────────────────────────
ZOOM_PADDING_CELLS="${ZOOM_PADDING_CELLS:-3}"
ZOOM_MAX_NEW_TOKENS="${ZOOM_MAX_NEW_TOKENS:-256}"

# ── 输出目录 ──────────────────────────────────────────────────────────
_BASE_OUTPUT="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/zwerge/data/results/zwerge_layerwise/${DECODE_STRATEGY}"
if [[ -n "${OUTPUT_DIR_FINAL:-}" ]]; then
    _OUTPUT_DIR_KEY="--output_dir_final"
    _OUTPUT_DIR_VAL="${OUTPUT_DIR_FINAL}"
else
    _OUTPUT_DIR_KEY="--output_dir"
    _OUTPUT_DIR_VAL="${_BASE_OUTPUT}"
fi

# ── 可视化控制 ────────────────────────────────────────────────────────
SKIP_VIS="${SKIP_VIS:-1}"
EXTRA_FLAGS=""
[[ "${SKIP_VIS}" == "1" ]] && EXTRA_FLAGS="${EXTRA_FLAGS} --skip_vis"

echo "[guiowl7b_eval] MODEL_TYPE          = ${MODEL_TYPE}"
echo "[guiowl7b_eval] CKPT                = ${CKPT}"
echo "[guiowl7b_eval] BENCH               = ${BENCH}"
echo "[guiowl7b_eval] DECODE_STRATEGY     = ${DECODE_STRATEGY}"
echo "[guiowl7b_eval] ZOOM_PADDING_CELLS  = ${ZOOM_PADDING_CELLS}"
echo "[guiowl7b_eval] ZOOM_MAX_NEW_TOKENS = ${ZOOM_MAX_NEW_TOKENS}"
echo "[guiowl7b_eval] SKIP_VIS            = ${SKIP_VIS}"
echo "[guiowl7b_eval] CONDA_ENV           = ${CONDA_ENV}"

conda run --no-capture-output -n "${CONDA_ENV}" \
python eval_retrofit.py \
    --model_type          "${MODEL_TYPE}" \
    --ckpt                "${CKPT}" \
    --bench               "${BENCH}" \
    --eval_dir            "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/datasets/evaluation" \
    "${_OUTPUT_DIR_KEY}"  "${_OUTPUT_DIR_VAL}" \
    --max_pixels          "${MAX_PIXELS}" \
    --decode_strategy     "${DECODE_STRATEGY}" \
    --zoom_padding_cells  "${ZOOM_PADDING_CELLS}" \
    --zoom_max_new_tokens "${ZOOM_MAX_NEW_TOKENS}" \
    ${EXTRA_FLAGS}
