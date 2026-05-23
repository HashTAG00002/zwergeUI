#!/bin/bash
# ============================================================
# ZwerGe-UI native_backbone 策略评测（原始模型指标复现）
# ============================================================
# 用途：通过与 zoom_backbone 相同的代码路径，把完整原图传给 backbone，
#       复现 GUI-Owl / UI-Venus 原始（不含 ZwerGe head）的 grounding 指标。
#
# 验证两件事：
#   1. 我们的 eval 代码（conv3d patch、坐标解析、系统消息）是否正确
#   2. 原始模型本地能跑出多少分（对标 leaderboard 数字）
#
# 指标解读：
#   - layer_accs per-layer/fusion  → ZwerGe head 指标（顺带验证 ZwerGe 是否工作）
#   - fusion_hit1 / fusion_overlap1 → backbone 原始指标（这才是主要关注点）
#
# 期望结果（如果 eval 代码正确）：
#   GUI-Owl SS-Pro  ≈ 73.2%（KV-Ground-8B 无ZoomIn）
#   UI-Venus SS-v2  ≈ 94.1%
#   如果相差 > 5pp → eval 代码可能有 bug
#
# 用法：
#   bash scripts/eval/eval_native_backbone.sh             # guiowl，全5个bench
#   bash scripts/eval/eval_native_backbone.sh ss_pro      # 单bench快速验证
#   MODEL_TYPE=uivenus bash scripts/eval/eval_native_backbone.sh ss_v2
#   MODEL_TYPE=uitars  bash scripts/eval/eval_native_backbone.sh ss_pro  # uitars baseline
#
# 可选环境变量：
#   MODEL_TYPE          uitars/guiowl/uivenus（默认 guiowl）
#   CKPT                checkpoint 路径（有默认值）
#   ZOOM_MAX_NEW_TOKENS backbone generate 最大 token 数（默认 256）
#   SKIP_VIS            设 1 则只输出指标 JSON（默认 1）
# ============================================================

set -euo pipefail
unset http_proxy https_proxy

if [[ -z "${AFO_ENV_CLUSTER_SPEC:-}" ]]; then
    echo "[native_eval] AFO_ENV_CLUSTER_SPEC not set — debug/local mode"
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
echo "[native_eval] NPROC_PER_NODE=$NPROC_PER_NODE"
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NPROC_PER_NODE - 1)))
echo "[native_eval] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZWERGE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
EVAL_DIR="${ZWERGE_ROOT}/eval"
cd "${EVAL_DIR}"
export PYTHONPATH="${ZWERGE_ROOT}/src:${PYTHONPATH:-}"

BENCH="${1:-all}"

# ── 模型类型（默认 guiowl，主要用来复现 Qwen3-VL 模型指标）──
MODEL_TYPE="${MODEL_TYPE:-guiowl}"

if [[ "${MODEL_TYPE}" == "guiowl" ]]; then
    CKPT="${CKPT:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/.hdd/ckpt/zwerge/guiowl_grounding50k_A3-gaussian_cos_meta_20260522_034634/checkpoint-1600}"
    CONDA_ENV="qwen3-verl"
    MAX_PIXELS=16777216
elif [[ "${MODEL_TYPE}" == "uivenus" ]]; then
    CKPT="${CKPT:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/.hdd/ckpt/zwerge/uivenus_grounding50k_A3-gaussian_cos_meta_20260522_031059/checkpoint-1600}"
    CONDA_ENV="qwen3-verl"
    MAX_PIXELS=16777216
else
    MODEL_TYPE="uitars"
    CKPT="${CKPT:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/.hdd/ckpt/zwerge/uitars7b_grounding50k_20260519_015331/checkpoint-2193}"
    CONDA_ENV="qwen25"
    MAX_PIXELS=12845056
fi

ZOOM_MAX_NEW_TOKENS="${ZOOM_MAX_NEW_TOKENS:-256}"
SKIP_VIS="${SKIP_VIS:-1}"

_BASE_OUTPUT="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/zwerge/data/results/zwerge_layerwise/native_backbone"
if [[ -n "${OUTPUT_DIR_FINAL:-}" ]]; then
    _OUTPUT_DIR_KEY="--output_dir_final"
    _OUTPUT_DIR_VAL="${OUTPUT_DIR_FINAL}"
else
    _OUTPUT_DIR_KEY="--output_dir"
    _OUTPUT_DIR_VAL="${_BASE_OUTPUT}"
fi

EXTRA_FLAGS=""
[[ "${SKIP_VIS}" == "1" ]] && EXTRA_FLAGS="${EXTRA_FLAGS} --skip_vis"

echo "[native_eval] MODEL_TYPE          = ${MODEL_TYPE}"
echo "[native_eval] CKPT                = ${CKPT}"
echo "[native_eval] BENCH               = ${BENCH}"
echo "[native_eval] ZOOM_MAX_NEW_TOKENS = ${ZOOM_MAX_NEW_TOKENS}"
echo "[native_eval] CONDA_ENV           = ${CONDA_ENV}"
echo "[native_eval] Note: fusion_hit1 = backbone native accuracy (compare to leaderboard)"

conda run --no-capture-output -n "${CONDA_ENV}" \
python eval_retrofit.py \
    --model_type          "${MODEL_TYPE}" \
    --ckpt                "${CKPT}" \
    --bench               "${BENCH}" \
    --eval_dir            "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/datasets/evaluation" \
    "${_OUTPUT_DIR_KEY}"  "${_OUTPUT_DIR_VAL}" \
    --max_pixels          "${MAX_PIXELS}" \
    --decode_strategy     "native_backbone" \
    --zoom_max_new_tokens "${ZOOM_MAX_NEW_TOKENS}" \
    ${EXTRA_FLAGS}
