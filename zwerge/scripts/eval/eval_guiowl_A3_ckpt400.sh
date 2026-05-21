#!/bin/bash
# GUI-Owl-1.5 A3 (Gaussian + cos_meta) checkpoint-800 全量评测 + 可视化
#
# 用法：
#   bash scripts/eval/eval_guiowl_A3_ckpt800.sh        # 全5个bench，含可视化
#   SKIP_VIS=1 bash scripts/eval/eval_guiowl_A3_ckpt800.sh  # 仅指标，不生成PNG
#   bash scripts/eval/eval_guiowl_A3_ckpt800.sh ss_pro  # 单bench快速验证

set -euo pipefail
unset http_proxy https_proxy

if [[ -z "${AFO_ENV_CLUSTER_SPEC:-}" ]]; then
    echo "[eval] AFO_ENV_CLUSTER_SPEC not set — debug/local mode"
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
echo "[eval] NPROC_PER_NODE=$NPROC_PER_NODE"
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NPROC_PER_NODE - 1)))
echo "[eval] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZWERGE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
EVAL_DIR="${ZWERGE_ROOT}/eval"
cd "${EVAL_DIR}"
export PYTHONPATH="${ZWERGE_ROOT}/src:${PYTHONPATH:-}"

# ── 固定参数 ──────────────────────────────────────────────────────────────────
CKPT="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/.hdd/ckpt/zwerge/uitars7b_grounding50k_A4-gaussian_cos_meta_L18-25_20260520_042031/checkpoint-2193"
MODEL_TYPE="uitars"
BENCH="${1:-all}"

OUTPUT_DIR_FINAL="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/zwerge/data/results/zwerge-uitars/retry"

DECODE_STRATEGY="centroid"
PEAK_SHIFT_ALPHA="0.5"

# ── 可视化控制 ────────────────────────────────────────────────────────────────
SKIP_VIS="${SKIP_VIS:-0}"
EXTRA_FLAGS=""
[[ "${SKIP_VIS}" == "1" ]] && EXTRA_FLAGS="${EXTRA_FLAGS} --skip_vis"

echo "[eval] ckpt        = ${CKPT}"
echo "[eval] bench       = ${BENCH}"
echo "[eval] output_dir  = ${OUTPUT_DIR_FINAL}"
echo "[eval] skip_vis    = ${SKIP_VIS}"

conda run --no-capture-output -n gui_actor \
python eval_retrofit.py \
    --model_type       "${MODEL_TYPE}" \
    --ckpt             "${CKPT}" \
    --bench            "${BENCH}" \
    --output_dir_final "${OUTPUT_DIR_FINAL}" \
    --eval_dir         "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/datasets/evaluation" \
    --max_pixels       12845056 \
    --decode_strategy  "${DECODE_STRATEGY}" \
    --peak_shift_alpha "${PEAK_SHIFT_ALPHA}" \
    --attn_impl        sdpa \
    ${EXTRA_FLAGS}
