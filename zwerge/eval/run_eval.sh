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
#   MODEL_TYPE        模型类型（默认 uitars，选项：uitars/guiowl/uivenus/guiowl7b/qwen35/uitars1）
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

# Conda environment:
#   Qwen3-VL (guiowl/uivenus):      qwen3  (transformers>=4.57.1)
#   Qwen3.5  (qwen35):              qwen35 (独立 conda env)
#   Qwen2-VL (uitars1):             qwen2  (独立 conda env)
#   Qwen2.5-VL (uitars/guiowl7b):  gui_actor / qwen25 (transformers>=4.51.3)
if [[ "${MODEL_TYPE}" == "guiowl" || "${MODEL_TYPE}" == "uivenus" ]]; then
    CONDA_ENV="${CONDA_ENV:-qwen3}"
elif [[ "${MODEL_TYPE}" == "qwen35" ]]; then
    CONDA_ENV="${CONDA_ENV:-qwen35}"
elif [[ "${MODEL_TYPE}" == "uitars1" ]]; then
    CONDA_ENV="${CONDA_ENV:-qwen2}"
else
    CONDA_ENV="${CONDA_ENV:-qwen25}"
fi
echo "MODEL_TYPE           = $MODEL_TYPE"
echo "CONDA_ENV            = $CONDA_ENV"

# ── Checkpoint ────────────────────────────────────────────────────────────────
if [[ "${MODEL_TYPE}" == "guiowl" ]]; then
    # 最新 guiowl checkpoint（grounding-only system prompt, 1255 chars）
    # system_message 从 checkpoint 目录的 args.json 自动读取（inference_base.py 已支持）
    CKPT="${CKPT:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/.hdd/ckpt/zwerge/guiowl_grounding50k_A3-gaussian_cos_meta_20260522_034634/checkpoint-1600}"
elif [[ "${MODEL_TYPE}" == "uivenus" ]]; then
    # 最新 uivenus checkpoint
    CKPT="${CKPT:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/.hdd/ckpt/zwerge/uivenus_grounding50k_A3-gaussian_cos_meta_20260522_031059/checkpoint-1600}"
elif [[ "${MODEL_TYPE}" == "guiowl7b" ]]; then
    # GUI-Owl-7B 控制变量 checkpoint（Qwen2.5-VL + GUI-Owl-1.5 prompt）
    # 首次训练完成后更新此路径
    CKPT="${CKPT:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/.hdd/ckpt/zwerge/guiowl7b_grounding50k_A3-gaussian_cos_meta}"
elif [[ "${MODEL_TYPE}" == "qwen35" ]]; then
    # Qwen3.5-VL checkpoint（训练完成后更新此路径）
    CKPT="${CKPT:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/.hdd/ckpt/zwerge/qwen35_grounding}"
elif [[ "${MODEL_TYPE}" == "uitars1" ]]; then
    # UI-TARS-7B-SFT (Qwen2-VL) checkpoint（训练完成后更新此路径）
    CKPT="${CKPT:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/.hdd/ckpt/zwerge/uitars1_grounding}"
else
    CKPT="${CKPT:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/.hdd/ckpt/zwerge/uitars7b_grounding50k_20260519_015331/checkpoint-2193}"
fi

# ── 坐标提取策略 ───────────────────────────────────────────────────────────────
DECODE_STRATEGY="${DECODE_STRATEGY:-peak_shift}"
PEAK_SHIFT_ALPHA="${PEAK_SHIFT_ALPHA:-0.5}"
TEMPERATURE="${TEMPERATURE:-0.5}"

# zoom_backbone 专属参数（非 zoom 策略时忽略）
ZOOM_PADDING_CELLS="${ZOOM_PADDING_CELLS:-3}"    # 感兴趣区域外扩 patch 数
ZOOM_MAX_NEW_TOKENS="${ZOOM_MAX_NEW_TOKENS:-256}" # backbone generate 最大 token 数

# ── MAX_PIXELS（与训练时一致）─────────────────────────────────────────────────
# uitars/guiowl7b/uitars1 (Qwen2.x-VL, patch_size=14): 16384 × 14² × 4 = 12,845,056
# guiowl/uivenus (Qwen3-VL, patch_size=16):             16384 × 16² × 4 = 16,777,216
# qwen35 (Qwen3.5, patch_size=16):                      16384 × 16² × 4 = 16,777,216  (暂定，待验证)
# 用 uitars 的值跑 Qwen3-VL 会导致输入图片被压缩过度（仅约 12544 tokens），
# 造成坐标精度下降。必须与训练脚本中的 MAX_PIXELS 保持一致。
if [[ -z "${MAX_PIXELS:-}" ]]; then
    if [[ "${MODEL_TYPE}" == "guiowl" || "${MODEL_TYPE}" == "uivenus" || "${MODEL_TYPE}" == "qwen35" ]]; then
        # Qwen3.x-VL family: patch_size=16 → 16384 × 16² × 4 = 16,777,216
        MAX_PIXELS=16777216
    else
        # Qwen2.x-VL (uitars / guiowl7b / uitars1): patch_size=14 → 16384 × 14² × 4 = 12,845,056
        MAX_PIXELS=12845056
    fi
fi
echo "MAX_PIXELS           = $MAX_PIXELS"

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
    --max_pixels "${MAX_PIXELS}" \
    --decode_strategy     "${DECODE_STRATEGY}" \
    --peak_shift_alpha    "${PEAK_SHIFT_ALPHA}" \
    --temperature         "${TEMPERATURE}" \
    --zoom_padding_cells  "${ZOOM_PADDING_CELLS}" \
    --zoom_max_new_tokens "${ZOOM_MAX_NEW_TOKENS}" \
    ${EXTRA_FLAGS}
