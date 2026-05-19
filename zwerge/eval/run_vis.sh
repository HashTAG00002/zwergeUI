#!/bin/bash
# ZwerGe-UI Evaluation + Visualization (All-in-One)
#
# 在单次 AFO job 中完成全量评测和可视化，多卡自动并行。
# 输出目录结构与 run_eval.sh 相同，额外生成 details/{bench}/success/ 和 failure/。
#
# 用法：
#   bash run_vis.sh ss_pro    # 单 bench
#   bash run_vis.sh all       # 全部 5 个 benchmark
#
# 可选环境变量：
#   CKPT            checkpoint 路径
#   DECODE_STRATEGY centroid（默认）| argmax | peak_shift | temperature
#   PEAK_SHIFT_ALPHA peak_shift 插值系数（默认 0.5）
#   TEMPERATURE     temperature 策略温度（默认 0.5）
#   CELL_W          每格热图宽度像素（默认 300）
#   CELL_H          每格热图高度像素（默认 220）
#   ALPHA           热图叠加透明度（默认 0.55）
#   MAX_PIXELS      图像最大像素（默认 12845056）

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
CKPT="${CKPT:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/.hdd/ckpt/zwerge/uitars7b_grounding50k_20260519_015331/checkpoint-2193}"

DECODE_STRATEGY="${DECODE_STRATEGY:-centroid}"
PEAK_SHIFT_ALPHA="${PEAK_SHIFT_ALPHA:-0.5}"
TEMPERATURE="${TEMPERATURE:-0.5}"
CELL_W="${CELL_W:-300}"
CELL_H="${CELL_H:-220}"
ALPHA="${ALPHA:-0.55}"
MAX_PIXELS="${MAX_PIXELS:-12845056}"

echo "============================================================"
echo "  ZwerGe-UI Evaluation + Visualization"
echo "============================================================"
echo "  bench           = ${BENCH}"
echo "  ckpt            = ${CKPT}"
echo "  decode_strategy = ${DECODE_STRATEGY}"
echo "  cell_w x cell_h = ${CELL_W} x ${CELL_H}"
echo "  alpha           = ${ALPHA}"
echo "  max_pixels      = ${MAX_PIXELS}"
echo "============================================================"

conda run --no-capture-output -n qwen25 \
python vis_zwerge.py \
    --ckpt              "${CKPT}" \
    --bench             "${BENCH}" \
    --eval_dir          "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/datasets/evaluation" \
    --output_dir        "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/zwerge/data/results/zwerge_layerwise/${DECODE_STRATEGY}" \
    --max_pixels        "${MAX_PIXELS}" \
    --decode_strategy   "${DECODE_STRATEGY}" \
    --peak_shift_alpha  "${PEAK_SHIFT_ALPHA}" \
    --temperature       "${TEMPERATURE}" \
    --cell_w            "${CELL_W}" \
    --cell_h            "${CELL_H}" \
    --alpha             "${ALPHA}"
