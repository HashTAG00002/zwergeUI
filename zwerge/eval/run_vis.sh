#!/bin/bash
# ZwerGe-UI Failure Case 可视化脚本
#
# 用法：
#   bash run_vis.sh ss_pro              # SS-Pro near_miss（默认）
#   bash run_vis.sh all                 # 全部 5 个 benchmark
#
#   CASE_TYPE=far_miss  bash run_vis.sh ss_pro    # 只看 far miss
#   CASE_TYPE=hit       bash run_vis.sh ss_pro    # 正确样本对照组
#   CASE_TYPE=all_miss  bash run_vis.sh ss_pro    # 所有 miss
#   CASE_TYPE=random    bash run_vis.sh ss_pro    # 随机样本
#   N_VIS=50            bash run_vis.sh ss_pro    # 更多样本
#
# bench 参数：ss_pro | ss_v2 | osworld_g | mmbench | ui_vision | all
#
# 可选环境变量：
#   CKPT           checkpoint 路径（覆盖默认值）
#   CASE_TYPE      near_miss（默认）| far_miss | hit | all_miss | random
#   N_VIS          可视化样本数（默认 20）
#   RESULTS_JSON   eval_layerwise --save_per_sample 产出的 JSON 路径
#                  指定后可跳过全量重推理，直接定位 failure index（强烈推荐）
#   DECODE_STRATEGY centroid（默认）| argmax | peak_shift | temperature
#   CELL_W         每格宽度像素（默认 300）
#   CELL_H         每格高度像素（默认 220）
#   ALPHA          热图叠加透明度，0~1（默认 0.55）
#   GPU_ID         使用哪块 GPU（默认 0）
#   MAX_PIXELS     最大图像像素数（默认 12845056）
#   VIS_DIR        输出根目录（默认 .../results/vis_failures）
#   SEED           随机种子（默认 42）
#
# 典型工作流：
#   Step 1 — 先跑评测并保存逐样本结果（只需跑一次）：
#     SAVE_PER_SAMPLE=1 bash run_eval.sh ss_pro
#     → 产出 .../zwerge_layerwise/.../ss_pro_layerwise_results.json
#
#   Step 2 — 用结果 JSON 定位 failure，避免全量重推理：
#     RESULTS_JSON=".../ss_pro_layerwise_results.json" CASE_TYPE=near_miss bash run_vis.sh ss_pro
#
#   Step 3 — 查看输出目录下的 PNG 图（一张图 = 一个样本的所有层热图）：
#     ls .../vis_failures/ss_pro_near_miss/
#
# 注意：
#   - vis 脚本单卡运行（无需多卡并行），每个样本做 1 次 prefill forward
#   - 默认跑 GPU 0，若要换卡：GPU_ID=2 bash run_vis.sh ss_pro
#   - 没有 RESULTS_JSON 时，脚本会随机取 N_VIS×10 个候选样本挨个推理来凑够 N_VIS 个 case

set -euo pipefail
unset http_proxy https_proxy

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZWERGE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${SCRIPT_DIR}"
export PYTHONPATH="${ZWERGE_ROOT}/src:${PYTHONPATH:-}"

# ── 参数 ──────────────────────────────────────────────────────────────────────
BENCH="${1:-ss_pro}"

CKPT="${CKPT:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/.hdd/ckpt/zwerge/uitars7b_grounding50k_20260519_015331/checkpoint-2193}"

# case type：near_miss | far_miss | hit | all_miss | random
CASE_TYPE="${CASE_TYPE:-near_miss}"

# 可视化样本数
N_VIS="${N_VIS:-20}"

# 解码策略（与评测时保持一致才有可比性）
DECODE_STRATEGY="${DECODE_STRATEGY:-centroid}"

# 每个格子的尺寸
CELL_W="${CELL_W:-300}"
CELL_H="${CELL_H:-220}"

# 热图叠加透明度
ALPHA="${ALPHA:-0.55}"

# GPU
GPU_ID="${GPU_ID:-0}"

# 最大像素数
MAX_PIXELS="${MAX_PIXELS:-12845056}"

# 随机种子
SEED="${SEED:-42}"

# 输出根目录
VIS_DIR="${VIS_DIR:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/zwerge/data/results/vis_failures}"

# per-sample JSON（可选，用于快速定位 failure index，避免全量重推理）
RESULTS_JSON="${RESULTS_JSON:-}"

# ── 打印配置 ──────────────────────────────────────────────────────────────────
echo "============================================================"
echo "  ZwerGe-UI Failure Case Visualization"
echo "============================================================"
echo "  bench         = ${BENCH}"
echo "  ckpt          = ${CKPT}"
echo "  case_type     = ${CASE_TYPE}"
echo "  n_vis         = ${N_VIS}"
echo "  decode_strat  = ${DECODE_STRATEGY}"
echo "  cell_w×h      = ${CELL_W}×${CELL_H}"
echo "  alpha         = ${ALPHA}"
echo "  gpu_id        = ${GPU_ID}"
echo "  max_pixels    = ${MAX_PIXELS}"
echo "  vis_dir       = ${VIS_DIR}"
if [[ -n "${RESULTS_JSON}" ]]; then
echo "  results_json  = ${RESULTS_JSON}"
else
echo "  results_json  = (不指定，随机候选推理)"
fi
echo "============================================================"

# ── 拼 flags ──────────────────────────────────────────────────────────────────
EXTRA_FLAGS=""
[[ -n "${RESULTS_JSON}" ]] && EXTRA_FLAGS="${EXTRA_FLAGS} --results_json ${RESULTS_JSON}"

# ── 单 bench 可视化函数 ────────────────────────────────────────────────────────
run_vis_bench() {
    local bench="$1"
    echo ""
    echo "[vis] bench=${bench}  case_type=${CASE_TYPE}  n_vis=${N_VIS}"

    conda run --no-capture-output -n qwen25 \
    python vis_zwerge.py \
        --ckpt           "${CKPT}" \
        --bench          "${bench}" \
        --eval_dir       "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/datasets/evaluation" \
        --output_dir     "${VIS_DIR}" \
        --max_pixels     "${MAX_PIXELS}" \
        --decode_strategy "${DECODE_STRATEGY}" \
        --case_type      "${CASE_TYPE}" \
        --n_samples      "${N_VIS}" \
        --cell_w         "${CELL_W}" \
        --cell_h         "${CELL_H}" \
        --alpha          "${ALPHA}" \
        --gpu_id         "${GPU_ID}" \
        --seed           "${SEED}" \
        ${EXTRA_FLAGS}
}

# ── 分发 bench ────────────────────────────────────────────────────────────────
if [[ "${BENCH}" == "all" ]]; then
    for b in ss_pro ss_v2 osworld_g mmbench ui_vision; do
        run_vis_bench "${b}"
    done
else
    run_vis_bench "${BENCH}"
fi

echo ""
echo "[vis] Done. Images saved to: ${VIS_DIR}"
echo "[vis] 每个子目录对应一个 bench+case_type 组合，PNG 文件名："
echo "        near_XXXX_idxYYYYY.png  = near_miss 样本"
echo "        far_XXXX_idxYYYYY.png   = far_miss 样本"
echo "        hit_XXXX_idxYYYYY.png   = 正确样本"
