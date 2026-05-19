#!/bin/bash
# ZwerGe-UI 评测入口
#
# 用法：
#   bash run_eval.sh ss_pro   # 评测 SS-Pro（自动检测 GPU 数量并行）
#   bash run_eval.sh all      # 全部 5 个 benchmark
#
#   CKPT=<路径> bash run_eval.sh ss_pro            # 指定 checkpoint
#   CKPT=<路径> CKPT_SUBFOLDER=checkpoint-2000 bash run_eval.sh ss_pro
#
# bench 参数：ss_pro | ss_v2 | osworld_g | mmbench | ui_vision | all

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
# MODE: fusion（默认，eval_zwerge.py）| layerwise（eval_layerwise.py，不做 fusion，逐层准确率分布）
MODE="${2:-fusion}"
CKPT="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/.hdd/ckpt/zwerge/uitars7b_grounding50k_20260519_015331/checkpoint-2193"

if [[ "${MODE}" == "layerwise" ]]; then
    conda run --no-capture-output -n qwen25 \
    python eval_layerwise.py \
        --ckpt    "${CKPT}" \
        --bench   "${BENCH}" \
        --eval_dir   "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/datasets/evaluation" \
        --output_dir "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/zwerge/data/results/zwerge_layerwise" \
        --max_pixels 12845056
        # 加 --save_per_sample 可保存每个样本各层的预测结果
else
    conda run --no-capture-output -n qwen25 \
    python eval_zwerge.py \
        --ckpt    "${CKPT}" \
        --bench   "${BENCH}" \
        --eval_dir   "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/datasets/evaluation" \
        --output_dir "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/zwerge/data/results/zwerge_sparse_high_res" \
        --max_pixels 12845056
        #--max_pixels 5720064
fi
