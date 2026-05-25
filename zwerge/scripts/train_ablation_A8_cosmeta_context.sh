#!/bin/bash
# ============================================================
# Ablation A8 Stage-2: Context-Aware Cos-Meta Fusion
# ============================================================
# 第二阶段训练：在 A7 已训练好的 CrossAttn probes 基础上，
# 引入约 200K 参数的 ContextLoRACosMetaFusion head。
#
# 核心设计：
#   - 从 A7 checkpoint 加载 14 层 CrossAttn probe 权重
#   - 选取其中 10 层（active_probe_layers）参与 fusion 训练
#   - 剩余 4 层 probe 保留在模型中（用于 checkpoint 兼容），但冻结
#   - backbone / visual encoder / lm_head 继续冻结
#   - 完全移除 readiness fusion，改用 cos-meta context fusion
#
# 参数量对比：
#   A7  10-layer CrossAttn probes: ~36M（继续训练）
#   A8  ContextLoRACosMetaFusion:  ~200K（新增）
#   A8  new token embeddings:      ~14K（继续训练）
#   Total trainable A8:            ~37.06M
#
# Fusion 参数（d_attn=512, lora_rank=128, context_rank=64, M=10）：
#   A_f/B_f: 131,072 + A_c/B_c: 65,536 + 3 LNs: 3,072
#   q_meta: 512 + alpha: 10 + rho: 1  →  Total: 200,203
#
# 用法：
#   # uitars (默认)
#   bash scripts/train_ablation_A8_cosmeta_context.sh
#
#   # guiowl7b
#   MODEL_TYPE=guiowl7b bash scripts/train_ablation_A8_cosmeta_context.sh
#
#   # 自定义 active 层（默认 16-25）：
#   ACTIVE_PROBE_LAYERS="18,19,20,21,22,23,24,25,26,27" \
#     bash scripts/train_ablation_A8_cosmeta_context.sh
#
# 注意：MODEL_NAME_OR_PATH 必须指向 A7 最后 checkpoint，通过 yaml 环境变量传入：
#   ZWERGE_JOB_NAME=uitars_A8_cosmeta_ctx_exp001
#   MODEL_NAME_OR_PATH=/mnt/.../uitars_A7_exp001/checkpoint-3129
# ============================================================

unset http_proxy https_proxy

if [[ -z "${AFO_ENV_CLUSTER_SPEC:-}" ]]; then
    echo "===== [A8-cosmeta_context] DEBUG MODE ====="
    export NPROC_PER_NODE=2
    export NODE_RANK=0
    export NNODES=1
    export MASTER_ADDR="127.0.0.1"
    export MASTER_PORT=29500
    PER_DEVICE_BATCH_SIZE=1
    GRADIENT_ACCUMULATION_STEPS=2
    NUM_EPOCHS=1
    MAX_STEPS=50
    SAVE_STEPS="${SAVE_STEPS:-50}"
    SAVE_STEPS_ONLY_FOR_RESUME="${SAVE_STEPS_ONLY_FOR_RESUME:--1}"
    FLASH_ATTN=False
    conda config --add envs_dirs /mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/conda/envs 2>/dev/null || true
else
    echo "===== [A8-cosmeta_context] JOB MODE ====="
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
    NUM_EPOCHS=1
    MAX_STEPS="${MAX_STEPS:--1}"
    SAVE_STEPS="${SAVE_STEPS:-400}"
    SAVE_STEPS_ONLY_FOR_RESUME="${SAVE_STEPS_ONLY_FOR_RESUME:-100}"
    FLASH_ATTN=True
fi

export http_proxy=http://10.70.11.143:8412
export https_proxy=http://10.70.11.143:8412
export WANDB_API_KEY=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
export WANDB_PROJECT=zwerge
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── 模型类型 ─────────────────────────────────────────────────
MODEL_TYPE="${MODEL_TYPE:-uitars}"

# MODEL_NAME_OR_PATH 必须通过 yaml env 或环境变量传入（指向 A7 checkpoint）
if [[ -z "${MODEL_NAME_OR_PATH:-}" ]]; then
    echo "[ERROR] MODEL_NAME_OR_PATH must be set to A7 checkpoint path"
    exit 1
fi

if [[ "${MODEL_TYPE}" == "guiowl7b" ]]; then
    CONDA_ENV="gui_actor"
    MAX_PIXELS="${MAX_PIXELS:-12845056}"
else
    # uitars (默认)
    MODEL_TYPE="uitars"
    CONDA_ENV="gui_actor"
    MAX_PIXELS="${MAX_PIXELS:-12845056}"
fi

# A7 probe_layers (全部 14 层，必须与 A7 checkpoint config 对齐)
PROBE_LAYERS="${PROBE_LAYERS:-14,15,16,17,18,19,20,21,22,23,24,25,26,27}"

# A8 active_probe_layers (参与 fusion 训练的 10 层子集)
ACTIVE_PROBE_LAYERS="${ACTIVE_PROBE_LAYERS:-16,17,18,19,20,21,22,23,24,25}"

DATA_PATH="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/datasets/grounding_200k.jsonl"

BASE_CKPT_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/.hdd/ckpt/zwerge"
if [ -n "${ZWERGE_JOB_NAME}" ]; then
    OUTPUT_DIR="${BASE_CKPT_DIR}/${ZWERGE_JOB_NAME}"
else
    RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_DIR="${BASE_CKPT_DIR}/${MODEL_TYPE}_grounding200k_A8-cosmeta_context_${RUN_TIMESTAMP}"
fi
mkdir -p "${OUTPUT_DIR}"
VAL_OUTPUT_DIR="${OUTPUT_DIR}"
export WANDB_RUN_NAME="$(basename "${OUTPUT_DIR}")"

echo "MODEL_TYPE          = ${MODEL_TYPE}"
echo "MODEL_NAME_OR_PATH  = ${MODEL_NAME_OR_PATH}"
echo "PROBE_LAYERS        = ${PROBE_LAYERS}"
echo "ACTIVE_PROBE_LAYERS = ${ACTIVE_PROBE_LAYERS}"
echo "WANDB_RUN_NAME      = ${WANDB_RUN_NAME}"
echo "OUTPUT_DIR          = ${OUTPUT_DIR}"

# ── A8 核心参数 ─────────────────────────────────────────────────
GT_LABEL_TYPE="gaussian"
GAUSSIAN_SIGMA_FACTOR="${GAUSSIAN_SIGMA_FACTOR:-0.35}"

# probe 架构：与 A7 保持一致
ADAPTER_TYPE="attn"
ATTN_HEADS="${ATTN_HEADS:-8}"
ATTN_HEAD_DIM="${ATTN_HEAD_DIM:-64}"

# fusion 参数
INDEPENDENT_LAYERS=false
FUSION_TYPE="cos_meta_context_lora"
FUSION_LORA_RANK="${FUSION_LORA_RANK:-128}"
FUSION_CONTEXT_RANK="${FUSION_CONTEXT_RANK:-64}"
FUSION_DETACH_QUERIES="${FUSION_DETACH_QUERIES:-true}"
FUSION_LEARN_TEMPERATURE="${FUSION_LEARN_TEMPERATURE:-true}"

GROUNDING_PROJ_DIM=1024    # 不影响 attn 模式，保留以防 compat
GROUNDING_ADAPTER_RANK=16  # 不影响 attn 模式
GROUNDING_LAMBDA_LAYER="${GROUNDING_LAMBDA_LAYER:-0.2}"
GROUNDING_LOSS_WEIGHT=1.0
LM_LOSS_WEIGHT=0.0

# 阶段二：不重置 A7 已训练好的 probe 权重
REINIT_GROUNDING_HEAD="false"
STAGE2_FROM_CKPT="true"

# 学习率
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
LEARNING_RATE_NEW_TOKENS="${LEARNING_RATE_NEW_TOKENS:-5e-5}"
MIN_PIXELS=3136
MODEL_MAX_LENGTH=18432

# ── 同步/异步评估互斥开关 ─────────────────────────────────────
EVAL_MODE="${EVAL_MODE:-async}"
VAL_FREQ="${VAL_FREQ:-400}"
if [[ "${EVAL_MODE}" == "async" ]]; then
    VAL_STEPS=-1
else
    VAL_STEPS="${VAL_FREQ}"
fi
VAL_BENCH="all"
VAL_N_SAMPLES=-1
VAL_DECODE_STRATEGY="centroid"
VAL_CELL_W=300
VAL_CELL_H=220
VAL_ALPHA=0.55

echo "ADAPTER_TYPE        = ${ADAPTER_TYPE}"
echo "ATTN_HEADS          = ${ATTN_HEADS}  d_attn=$((ATTN_HEADS * ATTN_HEAD_DIM))"
echo "FUSION_TYPE         = ${FUSION_TYPE}"
echo "GROUNDING_LAMBDA_LAYER = ${GROUNDING_LAMBDA_LAYER}"

CONDA_BASE="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/conda/envs"
CONDA_TORCHRUN="${CONDA_BASE}/${CONDA_ENV}/bin/torchrun"
TORCHRUN="$([ -f "${CONDA_TORCHRUN}" ] && echo "${CONDA_TORCHRUN}" || which torchrun 2>/dev/null || echo torchrun)"
echo "CONDA_ENV           = ${CONDA_ENV}"
echo "TORCHRUN            = ${TORCHRUN}"

cd "${PROJECT_ROOT}"

${TORCHRUN} \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    train_retrofit.py \
    \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --model_type "${MODEL_TYPE}" \
    --flash_attn_2_enabled ${FLASH_ATTN} \
    \
    --probe_layers "${PROBE_LAYERS}" \
    --grounding_proj_dim ${GROUNDING_PROJ_DIM} \
    --grounding_adapter_rank ${GROUNDING_ADAPTER_RANK} \
    --grounding_lambda_layer ${GROUNDING_LAMBDA_LAYER} \
    --grounding_independent_layers ${INDEPENDENT_LAYERS} \
    --grounding_adapter_type ${ADAPTER_TYPE} \
    --grounding_attn_heads ${ATTN_HEADS} \
    --grounding_attn_head_dim ${ATTN_HEAD_DIM} \
    --grounding_fusion_type ${FUSION_TYPE} \
    --grounding_active_probe_layers "${ACTIVE_PROBE_LAYERS}" \
    --grounding_fusion_lora_rank ${FUSION_LORA_RANK} \
    --grounding_fusion_context_rank ${FUSION_CONTEXT_RANK} \
    --grounding_fusion_detach_queries ${FUSION_DETACH_QUERIES} \
    --grounding_fusion_learn_temperature ${FUSION_LEARN_TEMPERATURE} \
    --reinit_grounding_head ${REINIT_GROUNDING_HEAD} \
    --stage2_from_retrofit_checkpoint ${STAGE2_FROM_CKPT} \
    \
    --data_path "${DATA_PATH}" \
    --image_folder "" \
    --min_pixels ${MIN_PIXELS} \
    --max_pixels ${MAX_PIXELS} \
    --max_conv_turns 10 \
    --gt_label_type ${GT_LABEL_TYPE} \
    --gaussian_sigma_factor ${GAUSSIAN_SIGMA_FACTOR} \
    \
    --grounding_loss_weight ${GROUNDING_LOSS_WEIGHT} \
    --lm_loss_weight ${LM_LOSS_WEIGHT} \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs ${NUM_EPOCHS} \
    --max_steps ${MAX_STEPS} \
    --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --learning_rate ${LEARNING_RATE} \
    --learning_rate_new_tokens ${LEARNING_RATE_NEW_TOKENS} \
    --weight_decay 0.0 \
    --warmup_ratio 0.02 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --bf16 True \
    --fp16 False \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --logging_steps 20 \
    --save_steps ${SAVE_STEPS} \
    --save_steps_only_for_resume ${SAVE_STEPS_ONLY_FOR_RESUME} \
    --model_max_length ${MODEL_MAX_LENGTH} \
    --report_to wandb \
    --run_name "${WANDB_RUN_NAME}" \
    \
    --val_steps ${VAL_STEPS} \
    --val_bench "${VAL_BENCH}" \
    --val_n_samples ${VAL_N_SAMPLES} \
    --val_eval_dir "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/datasets/evaluation" \
    --val_output_dir "${VAL_OUTPUT_DIR}" \
    --val_decode_strategy "${VAL_DECODE_STRATEGY}" \
    --val_max_pixels ${MAX_PIXELS} \
    --val_cell_w ${VAL_CELL_W} \
    --val_cell_h ${VAL_CELL_H} \
    --val_alpha ${VAL_ALPHA} \
    2>&1 | tee "${OUTPUT_DIR}/train.log"
