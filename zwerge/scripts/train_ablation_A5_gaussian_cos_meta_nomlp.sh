#!/bin/bash
# ============================================================
# Ablation A5: Gaussian GT + cos_meta + 纯 LoRA（去掉共享 MLP）
# ============================================================
# 变量点：
#   GT_LABEL_TYPE         = gaussian   ← 各向异性高斯 GT（同 A3）
#   FUSION_TYPE           = cos_meta   ← alpha_l + cos(q_meta, q_l)（同 A3）
#   USE_SHARED_MLP        = false      ← 核心：去掉共享 MLP，纯 LoRA 模式
#   PROBE_LAYERS          = 18-27      ← 不变（同 A3）
#
# 对照 A3 的唯一差异：是否使用共享 MLP 投影。
# 目的：验证 MLP 是否真的有帮助，还是 LoRA adapter 足够提取 grounding 信号。
#
# 参数量对比：
#   A3（有 MLP, d_proj=1024）: ~12M（其中 MLP 占 9.4M）
#   A5（无 MLP）             : ~2.6M（只有 LoRA adapter，q_meta 扩为 d_model=3584）
#
# 预期：
#   若 A5 ≈ A3：MLP 是冗余的，纯 LoRA 已足够
#   若 A5 < A3：共享 MLP 提供了有效的跨层 grounding 空间映射
#   若 A5 > A3：MLP 反而是噪声，固定非线性结构损害了层特异性
# ============================================================

unset http_proxy https_proxy

if [[ -z "${AFO_ENV_CLUSTER_SPEC:-}" ]]; then
    echo "===== [A5-gaussian+cos_meta+nomlp] DEBUG MODE ====="
    export NPROC_PER_NODE=2
    export NODE_RANK=0
    export NNODES=1
    export MASTER_ADDR="127.0.0.1"
    export MASTER_PORT=29500
    PER_DEVICE_BATCH_SIZE=1
    GRADIENT_ACCUMULATION_STEPS=2
    NUM_EPOCHS=1
    MAX_STEPS=30
    SAVE_STEPS=30
    MAX_PIXELS=2494464
    FLASH_ATTN=False
    conda config --add envs_dirs /mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/conda/envs 2>/dev/null || true
else
    echo "===== [A5-gaussian+cos_meta+nomlp] JOB MODE ====="
    nvidia-smi
    conda config --add envs_dirs /mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/conda/envs
    conda env list
    export NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)
    export NODE_RANK="$(jq -r '.index | tonumber' <<<"$AFO_ENV_CLUSTER_SPEC")"
    export NNODES="$(jq -r '.worker | length' <<<"$AFO_ENV_CLUSTER_SPEC")"
    _REAL_IP=$(ip route get 1.0.0.0 2>/dev/null \
        | awk '/src/{for(i=1;i<=NF;i++) if($i=="src") print $(i+1)}' | head -1)
    [[ -z "$_REAL_IP" ]] && _REAL_IP=$(hostname -I 2>/dev/null | tr ' ' '\n' | grep -E '^10\.' | head -1)
    [[ -z "$_REAL_IP" ]] && _REAL_IP=$(hostname -I 2>/dev/null | tr ' ' '\n' | grep -vE '^127\.' | head -1)
    _WORKER0_RAW="$(jq -r '.worker[0]' <<<"$AFO_ENV_CLUSTER_SPEC")"
    if [[ "$NODE_RANK" == "0" ]] || [[ -z "${NODE_RANK}" ]]; then
        export MASTER_ADDR="${_REAL_IP:-127.0.0.1}"
    else
        _W0_IP="${_WORKER0_RAW%%:*}"
        if [[ "$_W0_IP" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
            export MASTER_ADDR="${_W0_IP}"
        else
            export MASTER_ADDR="${_REAL_IP:-127.0.0.1}"
        fi
    fi
    _W0_PORT="${_WORKER0_RAW##*:}"
    if [[ "$_W0_PORT" =~ ^[0-9]+$ ]] && [[ "$_W0_PORT" -ge 1024 ]] && [[ "$_W0_PORT" -le 65535 ]]; then
        export MASTER_PORT="${_W0_PORT}"
    else
        export MASTER_PORT=29500
    fi
    PER_DEVICE_BATCH_SIZE=2
    GRADIENT_ACCUMULATION_STEPS=4
    NUM_EPOCHS=3
    MAX_STEPS=-1
    SAVE_STEPS=400
    MAX_PIXELS=12845056
    FLASH_ATTN=True
fi

export http_proxy=http://10.70.11.143:8412
export https_proxy=http://10.70.11.143:8412
export WANDB_API_KEY=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
export WANDB_PROJECT=zwerge
export WANDB_RUN_NAME="zwerge-A5-gaussian_cos_meta_nomlp-$(date +%Y%m%d-%H%M%S)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_PATH="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/.hdd/models/huggingface.co/GUI_Agents/UI-TARS-1.5-7B"
DATA_PATH="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/datasets/grounding_50k.json"

RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/.hdd/ckpt/zwerge/uitars7b_grounding50k_A5-gaussian_cos_meta_nomlp_L18-27_${RUN_TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"

echo "WANDB_RUN_NAME = ${WANDB_RUN_NAME}"
echo "OUTPUT_DIR     = ${OUTPUT_DIR}"

# ── 5. 消融参数 ─────────────────────────────────────────────────
PROBE_LAYERS="18,19,20,21,22,23,24,25,26,27"
GT_LABEL_TYPE="gaussian"          # ← 同 A3
GAUSSIAN_SIGMA_FACTOR="0.5"
FUSION_TYPE="cos_meta"            # ← 同 A3
USE_SHARED_MLP="false"            # ← A5 核心：纯 LoRA，无共享 MLP

# GROUNDING_PROJ_DIM 在 use_shared_mlp=false 时不影响 dot product，
# 仅在读取 config 时被存储，实际 d_eff = d_model=3584
GROUNDING_PROJ_DIM=1024
GROUNDING_ADAPTER_RANK=16
GROUNDING_LAMBDA_LAYER=0.5
GROUNDING_LOSS_WEIGHT=1.0
LM_LOSS_WEIGHT=0.0
LEARNING_RATE=2e-4
LEARNING_RATE_NEW_TOKENS=2e-4
MIN_PIXELS=3136
MODEL_MAX_LENGTH=18432

VAL_STEPS=400
VAL_BENCH="all"
VAL_N_SAMPLES=-1
VAL_DECODE_STRATEGY="centroid"
VAL_OUTPUT_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/zwerge/data/results/zwerge_layerwise"
VAL_CELL_W=300
VAL_CELL_H=220
VAL_ALPHA=0.55

CONDA_TORCHRUN="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/conda/envs/gui_actor/bin/torchrun"
TORCHRUN="$([ -f "${CONDA_TORCHRUN}" ] && echo "${CONDA_TORCHRUN}" || which torchrun 2>/dev/null || echo torchrun)"

cd "${PROJECT_ROOT}"

${TORCHRUN} \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    train_retrofit.py \
    \
    --model_name_or_path "${MODEL_PATH}" \
    --flash_attn_2_enabled ${FLASH_ATTN} \
    \
    --probe_layers "${PROBE_LAYERS}" \
    --grounding_proj_dim ${GROUNDING_PROJ_DIM} \
    --grounding_adapter_rank ${GROUNDING_ADAPTER_RANK} \
    --grounding_lambda_layer ${GROUNDING_LAMBDA_LAYER} \
    --grounding_use_shared_mlp ${USE_SHARED_MLP} \
    \
    --data_path "${DATA_PATH}" \
    --image_folder "" \
    --min_pixels ${MIN_PIXELS} \
    --max_pixels ${MAX_PIXELS} \
    --max_conv_turns 10 \
    --gt_label_type ${GT_LABEL_TYPE} \
    --gaussian_sigma_factor ${GAUSSIAN_SIGMA_FACTOR} \
    --grounding_fusion_type ${FUSION_TYPE} \
    \
    --val_steps           ${VAL_STEPS} \
    --val_bench           ${VAL_BENCH} \
    --val_n_samples       ${VAL_N_SAMPLES} \
    --val_decode_strategy ${VAL_DECODE_STRATEGY} \
    --val_eval_dir        "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/datasets/evaluation" \
    --val_output_dir      "${VAL_OUTPUT_DIR}" \
    --val_max_pixels      "${MAX_PIXELS}" \
    --val_cell_w          "${VAL_CELL_W}" \
    --val_cell_h          "${VAL_CELL_H}" \
    --val_alpha           "${VAL_ALPHA}" \
    \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs ${NUM_EPOCHS} \
    --max_steps ${MAX_STEPS} \
    --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --learning_rate ${LEARNING_RATE} \
    --learning_rate_new_tokens ${LEARNING_RATE_NEW_TOKENS} \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    \
    --model_max_length ${MODEL_MAX_LENGTH} \
    --bf16 True \
    --fp16 False \
    --gradient_checkpointing True \
    \
    --grounding_loss_weight ${GROUNDING_LOSS_WEIGHT} \
    --lm_loss_weight ${LM_LOSS_WEIGHT} \
    \
    --unfreeze_all_parameters False \
    --unfreeze_grounding_head True \
    --unfreeze_new_tokens True \
    --unfreeze_lm_head False \
    --unfreeze_last_n_layers -1 \
    --unfreeze_visual_encoder False \
    \
    --empty_cache_every_n_steps 20 \
    \
    --save_strategy "steps" \
    --save_steps ${SAVE_STEPS} \
    --logging_steps 10 \
    --dataloader_num_workers 4 \
    \
    --report_to wandb \
    --run_name "${WANDB_RUN_NAME}" \
    \
    --verbose_logging False \
    2>&1 | tee "${OUTPUT_DIR}/train.log"

echo "===== [A5-gaussian+cos_meta+nomlp] Training complete. Output: ${OUTPUT_DIR} ====="
