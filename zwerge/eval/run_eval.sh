#!/bin/bash
# ZwerGe-UI 评测入口（统一走 eval_retrofit.py，支持所有模型类型）
#
# 用法：
#   bash run_eval.sh ss_pro               # 评测 SS-Pro（指标 + 可视化）
#   bash run_eval.sh all                  # 全部 5 个 benchmark
#   SKIP_VIS=1 bash run_eval.sh ss_pro   # 指标 only（不生成 PNG，快速评测）
#   MODEL_TYPE=guiowl bash run_eval.sh ss_pro
#   MODEL_TYPE=uivenus bash run_eval.sh all
#
# 可选环境变量：
#   MODEL_TYPE        模型类型（默认 uitars，选项：uitars/guiowl/uivenus）
#   CKPT              checkpoint 路径（有默认值，见下方）
#   DECODE_STRATEGY   坐标提取策略（默认 peak_shift）
#   PEAK_SHIFT_ALPHA  peak_shift 插值系数（默认 0.5）
#   TEMPERATURE       temperature 策略温度（默认 0.5）
#   SKIP_VIS          设为 1 则只计算指标，不生成 PNG（默认 0）
#   NO_GROUP_STATS    设为 1 则关闭 group 细分域统计

set -euo pipefail
unset http_proxy https_proxy
if [[ -z "${AFO_ENV_CLUSTER_SPEC:-}" ]]; then
    echo "AFO_ENV_CLUSTER_SPEC not set yet, debug mode"
else
    nvidia-smi
    conda config --add envs_dirs /mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/conda/envs
    conda env list
    export NODE_RANK="$(jq -r '.index | tonumber' <<<"$AFO_ENV_CLUSTER_SPEC")"
    export NNODES="$(jq -r '.worker | length' <<<"$AFO_ENV_CLUSTER_SPEC")"
    master=$(jq -r '.worker[0]' <<<"$AFO_ENV_CLUSTER_SPEC")
    export MASTER_ADDR="${master%%:*}"
    export MASTER_PORT="${master##*:}"

    echo "AFO_ENV_CLUSTER_SPEC = $AFO_ENV_CLUSTER_SPEC"
    echo "NODE_RANK            = $NODE_RANK"
    echo "NNODES               = $NNODES"
    echo "MASTER_ADDR          = $MASTER_ADDR"
    echo "MASTER_PORT          = $MASTER_PORT"
fi

export NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)
echo "NPROC_PER_NODE       = $NPROC_PER_NODE"
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NPROC_PER_NODE - 1)))
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZWERGE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${SCRIPT_DIR}"
export PYTHONPATH="${ZWERGE_ROOT}/src:${PYTHONPATH:-}"

BENCH="${1:-all}"

# ── 模型类型 ──────────────────────────────────────────────────────────────────
MODEL_TYPE="${MODEL_TYPE:-uitars}"

# Conda environment: Qwen3-VL (guiowl/uivenus) requires qwen3-verl env
if [[ "${MODEL_TYPE}" == "guiowl" || "${MODEL_TYPE}" == "uivenus" ]]; then
    CONDA_ENV="${CONDA_ENV:-qwen3-verl}"
else
    CONDA_ENV="${CONDA_ENV:-qwen25}"
fi
echo "MODEL_TYPE           = $MODEL_TYPE"
echo "CONDA_ENV            = $CONDA_ENV"

# ── Checkpoint ────────────────────────────────────────────────────────────────
if [[ "${MODEL_TYPE}" == "guiowl" ]]; then
    CKPT="${CKPT:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/.hdd/ckpt/zwerge/guiowl_grounding/checkpoint-latest}"
elif [[ "${MODEL_TYPE}" == "uivenus" ]]; then
    CKPT="${CKPT:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/.hdd/ckpt/zwerge/uivenus_grounding/checkpoint-latest}"
else
    CKPT="${CKPT:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/.hdd/ckpt/zwerge/uitars7b_grounding50k_20260519_015331/checkpoint-2193}"
fi

# ── 坐标提取策略 ───────────────────────────────────────────────────────────────
DECODE_STRATEGY="${DECODE_STRATEGY:-peak_shift}"
PEAK_SHIFT_ALPHA="${PEAK_SHIFT_ALPHA:-0.5}"
TEMPERATURE="${TEMPERATURE:-0.5}"

# ── 可视化控制 ────────────────────────────────────────────────────────────────
# SKIP_VIS=1 → 只输出指标 JSON，不生成 PNG（快，等价于旧 eval_layerwise.py）
# SKIP_VIS=0 → 生成指标 + 全量可视化 PNG（较慢）
SKIP_VIS="${SKIP_VIS:-0}"

EXTRA_FLAGS=""
[[ "${SKIP_VIS}"       == "1" ]] && EXTRA_FLAGS="${EXTRA_FLAGS} --skip_vis"
[[ "${NO_GROUP_STATS:-0}" == "1" ]] && EXTRA_FLAGS="${EXTRA_FLAGS} --no_group_stats"

conda run --no-capture-output -n "${CONDA_ENV}" \
python eval_retrofit.py \
    --model_type "${MODEL_TYPE}" \
    --ckpt       "${CKPT}" \
    --bench      "${BENCH}" \
    --eval_dir   "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/datasets/evaluation" \
    --output_dir "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/zwerge/data/results/zwerge_layerwise/${DECODE_STRATEGY}" \
    --max_pixels 12845056 \
    --decode_strategy  "${DECODE_STRATEGY}" \
    --peak_shift_alpha "${PEAK_SHIFT_ALPHA}" \
    --temperature      "${TEMPERATURE}" \
    ${EXTRA_FLAGS}
