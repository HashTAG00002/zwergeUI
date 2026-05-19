#!/bin/bash
# ZwerGe-UI 评测入口（统一走 eval_layerwise.py，支持逐层分析 + fusion 完整指标 + group 细分域）
#
# 用法：
#   bash run_eval.sh ss_pro   # 评测 SS-Pro（自动检测 GPU 数量并行）
#   bash run_eval.sh all      # 全部 5 个 benchmark
#
#   CKPT=<路径> bash run_eval.sh ss_pro            # 指定 checkpoint
#
# bench 参数：ss_pro | ss_v2 | osworld_g | mmbench | ui_vision | all
#
# 可选环境变量：
#   DECODE_STRATEGY   坐标提取策略（默认 centroid，见下方说明）
#   PEAK_SHIFT_ALPHA  peak_shift 插值系数（默认 0.5）
#   TEMPERATURE       temperature 策略温度（默认 0.5）
#   NO_GROUP_STATS    设为 1 则关闭 group 细分域统计
#   NO_FUSION_TOPK    设为 1 则关闭 fusion topk 指标

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
CKPT="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/.hdd/ckpt/zwerge/uitars7b_grounding50k_20260519_015331/checkpoint-2193"

# ── 坐标提取策略 ───────────────────────────────────────────────────────────────
# DECODE_STRATEGY: centroid（默认，与 GUI-AIMA 对齐）| argmax | peak_shift | temperature
#   centroid    : score 加权质心，与 GUI-AIMA 基线完全一致
#   argmax      : 最高分 patch 中心，避免质心漂移 → 预期 hit ≈ overlap
#   peak_shift  : argmax 与 centroid 线性插值（PEAK_SHIFT_ALPHA 控制，0=centroid,1=argmax）
#   temperature : 温度缩放加权质心（TEMPERATURE<1 使分布更集中，减少低分 patch 影响）
DECODE_STRATEGY="${DECODE_STRATEGY:-peak_shift}"
PEAK_SHIFT_ALPHA="${PEAK_SHIFT_ALPHA:-0.5}"
TEMPERATURE="${TEMPERATURE:-0.5}"
# 示例：
#   DECODE_STRATEGY=argmax bash run_eval.sh ss_pro
#   DECODE_STRATEGY=peak_shift PEAK_SHIFT_ALPHA=0.7 bash run_eval.sh ss_pro
#   DECODE_STRATEGY=temperature TEMPERATURE=0.3 bash run_eval.sh ss_pro

# ── 可选开关 ──────────────────────────────────────────────────────────────────
# NO_GROUP_STATS=1 关闭 group 细分域统计（默认开启）
# NO_FUSION_TOPK=1 关闭 fusion topk 指标（默认开启）
EXTRA_FLAGS=""
[[ "${NO_GROUP_STATS:-0}" == "1" ]] && EXTRA_FLAGS="${EXTRA_FLAGS} --no_group_stats"
[[ "${NO_FUSION_TOPK:-0}" == "1" ]] && EXTRA_FLAGS="${EXTRA_FLAGS} --no_fusion_full_topk"

conda run --no-capture-output -n qwen25 \
python eval_layerwise.py \
    --ckpt    "${CKPT}" \
    --bench   "${BENCH}" \
    --eval_dir   "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/datasets/evaluation" \
    --output_dir "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/zwerge/data/results/zwerge_layerwise/${DECODE_STRATEGY}" \
    --max_pixels 12845056 \
    --decode_strategy  "${DECODE_STRATEGY}" \
    --peak_shift_alpha "${PEAK_SHIFT_ALPHA}" \
    --temperature      "${TEMPERATURE}" \
    --save_per_sample \
    ${EXTRA_FLAGS}
    # 加 --save_per_sample 可保存每个样本各层的预测结果，供 vis_zwerge.py 快速定位 failure index
    # 加 --no_group_stats  可关闭 group 细分域统计（加速约 5%）
    # 加 --no_fusion_full_topk 可关闭 fusion topk 计算

# ── Failure Case 可视化（vis 模式，单卡运行） ─────────────────────────────────
# 用法：MODE=vis bash run_eval.sh ss_pro
#
# 可选环境变量：
#   CASE_TYPE    near_miss（默认）| far_miss | hit | all_miss | random
#   N_VIS        可视化样本数（默认 20）
#   RESULTS_JSON 指定 --save_per_sample 产出的 JSON，加速 failure index 定位
#   VIS_DIR      输出目录（默认 .../results/vis_failures）
#   GPU_ID       使用的 GPU（默认 0，vis 模式单卡即可）

if [[ "${MODE:-eval}" == "vis" ]]; then
    CASE_TYPE="${CASE_TYPE:-near_miss}"
    N_VIS="${N_VIS:-20}"
    RESULTS_JSON="${RESULTS_JSON:-}"
    VIS_DIR="${VIS_DIR:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/zwerge/data/results/vis_failures}"
    GPU_ID="${GPU_ID:-0}"

    VIS_FLAGS="--case_type ${CASE_TYPE} --n_samples ${N_VIS} --gpu_id ${GPU_ID}"
    [[ -n "${RESULTS_JSON}" ]] && VIS_FLAGS="${VIS_FLAGS} --results_json ${RESULTS_JSON}"

    conda run --no-capture-output -n qwen25 \
    python vis_zwerge.py \
        --ckpt      "${CKPT}" \
        --bench     "${BENCH}" \
        --eval_dir  "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/datasets/evaluation" \
        --output_dir "${VIS_DIR}" \
        --max_pixels "${MAX_PIXELS:-12845056}" \
        --decode_strategy "${DECODE_STRATEGY}" \
        ${VIS_FLAGS}
fi
