#!/bin/bash
# ============================================================
# ZwerGe-UI zoom_backbone 策略评测
# ============================================================
# 两阶段推断：
#   Stage 1: ZwerGe prefill → patch posteriors → BFS 选最佳区域 → crop box
#   Stage 2: backbone.generate(crop) → 解析坐标 → 映射回原图
#
# 适用场景：验证 ZwerGe 的 ROI 定位 + backbone 精化是否能弥补粗粒度网格误差
#
# 用法：
#   bash scripts/eval/eval_zoom_backbone.sh             # uitars，全5个bench
#   bash scripts/eval/eval_zoom_backbone.sh ss_pro      # 单bench快速验证
#   MODEL_TYPE=guiowl bash scripts/eval/eval_zoom_backbone.sh
#   MODEL_TYPE=uivenus bash scripts/eval/eval_zoom_backbone.sh ss_pro
#
# 可选环境变量：
#   MODEL_TYPE             uitars/guiowl/uivenus（默认 uitars）
#   CKPT                   checkpoint 路径（有默认值）
#   ZOOM_PADDING_CELLS     感兴趣区域外扩 patch 数（默认 3）
#   ZOOM_MAX_NEW_TOKENS    backbone generate 最大 token 数（默认 256）
#   SKIP_VIS               设 1 则只输出指标 JSON（默认 1，zoom 较慢建议先跑指标）
#   OUTPUT_DIR_FINAL       自定义输出目录（可选，否则自动推断）
#
# 注意：zoom_backbone 每个样本比 centroid 多一次 backbone generate，
#       建议先 SKIP_VIS=1 快速验证准确率，有提升再开可视化。
# ============================================================

set -euo pipefail
unset http_proxy https_proxy

if [[ -z "${AFO_ENV_CLUSTER_SPEC:-}" ]]; then
    echo "[zoom_eval] AFO_ENV_CLUSTER_SPEC not set — debug/local mode"
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
echo "[zoom_eval] NPROC_PER_NODE=$NPROC_PER_NODE"
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NPROC_PER_NODE - 1)))
echo "[zoom_eval] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZWERGE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
EVAL_DIR="${ZWERGE_ROOT}/eval"
cd "${EVAL_DIR}"
export PYTHONPATH="${ZWERGE_ROOT}/src:${PYTHONPATH:-}"

BENCH="${1:-all}"

# ── 模型类型 ─────────────────────────────────────────────────
MODEL_TYPE="${MODEL_TYPE:-uitars}"

if [[ "${MODEL_TYPE}" == "guiowl" ]]; then
    CKPT="${CKPT:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/.hdd/ckpt/zwerge/guiowl_grounding50k_A3-gaussian_cos_meta_20260522_034634/checkpoint-1600}"
    CONDA_ENV="qwen3"
    MAX_PIXELS=16777216
elif [[ "${MODEL_TYPE}" == "uivenus" ]]; then
    CKPT="${CKPT:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/.hdd/ckpt/zwerge/uivenus_grounding50k_A3-gaussian_cos_meta_20260522_031059/checkpoint-2193}"
    CONDA_ENV="qwen3"
    MAX_PIXELS=16777216
else
    MODEL_TYPE="uitars"
    CKPT="${CKPT:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/.hdd/ckpt/zwerge/uitars7b_grounding50k_A4-gaussian_cos_meta_L18-25_20260520_042031/checkpoint-2193}"
    CONDA_ENV="qwen25"
    MAX_PIXELS=12845056
fi

# ── 解码策略（可通过 DECODE_STRATEGY 环境变量覆盖）─────────────
# centroid（默认）：patch posterior 加权质心，无额外 backbone 调用，快
# zoom_backbone：先用 ZwerGe 定位 ROI，再裁图喂给 backbone 精化，慢但精度高
DECODE_STRATEGY="${DECODE_STRATEGY:-centroid}"

# ── zoom_backbone 专用参数（DECODE_STRATEGY=centroid 时忽略）────
ZOOM_PADDING_CELLS="${ZOOM_PADDING_CELLS:-3}"
ZOOM_MAX_NEW_TOKENS="${ZOOM_MAX_NEW_TOKENS:-256}"

# ── 输出目录 ──────────────────────────────────────────────────
# 默认：.../zwerge_layerwise/zoom_backbone/{run}/{ckpt}/
# 也可通过 OUTPUT_DIR_FINAL 直接指定完整路径（eval_daemon.py 用此注入每个 ckpt 的输出路径）
_BASE_OUTPUT="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/zwerge/data/results/zwerge_layerwise/zoom_backbone"
if [[ -n "${OUTPUT_DIR_FINAL:-}" ]]; then
    _OUTPUT_DIR_KEY="--output_dir_final"
    _OUTPUT_DIR_VAL="${OUTPUT_DIR_FINAL}"
else
    _OUTPUT_DIR_KEY="--output_dir"
    _OUTPUT_DIR_VAL="${_BASE_OUTPUT}"
fi

# ── 可视化控制（zoom 较慢，默认关闭 vis）─────────────────────
SKIP_VIS="${SKIP_VIS:-1}"
EXTRA_FLAGS=""
[[ "${SKIP_VIS}" == "1" ]] && EXTRA_FLAGS="${EXTRA_FLAGS} --skip_vis"

echo "[zoom_eval] MODEL_TYPE          = ${MODEL_TYPE}"
echo "[zoom_eval] CKPT                = ${CKPT}"
echo "[zoom_eval] BENCH               = ${BENCH}"
echo "[zoom_eval] ZOOM_PADDING_CELLS  = ${ZOOM_PADDING_CELLS}"
echo "[zoom_eval] ZOOM_MAX_NEW_TOKENS = ${ZOOM_MAX_NEW_TOKENS}"
echo "[zoom_eval] SKIP_VIS            = ${SKIP_VIS}"
echo "[zoom_eval] CONDA_ENV           = ${CONDA_ENV}"

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
