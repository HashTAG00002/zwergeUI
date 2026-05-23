#!/bin/bash
# ============================================================
# Ablation A3: Gaussian GT + cos_meta fusion
# ============================================================
# 变量点：
#   GT_LABEL_TYPE = gaussian   ← 各向异性高斯 GT
#   FUSION_TYPE   = cos_meta   ← alpha_l + cos(q_meta, q_l)
#   PROBE_LAYERS  = 18-27      ← 不变
# A1 + A2 同时启用，验证两项改进是否有协同效果。
# ============================================================

unset http_proxy https_proxy

if [[ -z "${AFO_ENV_CLUSTER_SPEC:-}" ]]; then
    echo "===== [A3-gaussian+cos_meta] DEBUG MODE ====="
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
    SAVE_STEPS_ONLY_FOR_RESUME=-1
    # MAX_PIXELS 将在 MODEL_TYPE 分支中设置（与 JOB MODE 保持一致）
    FLASH_ATTN=False
    conda config --add envs_dirs /mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/conda/envs 2>/dev/null || true
else
    echo "===== [A3-gaussian+cos_meta] JOB MODE ====="
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
    SAVE_STEPS_ONLY_FOR_RESUME=100
    # MAX_PIXELS 将在 MODEL_TYPE 分支中设置（与 JOB MODE 保持一致）
    FLASH_ATTN=True
fi

export http_proxy=http://10.70.11.143:8412
export https_proxy=http://10.70.11.143:8412
export WANDB_API_KEY=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
export WANDB_PROJECT=zwerge

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── 模型类型（决定使用哪个 backbone 和 prompt 格式）──────────────
# 可选值：uitars | guiowl7b | guiowl | uivenus
# 用法：MODEL_TYPE=guiowl7b bash scripts/train_ablation_A3_gaussian_cos_meta.sh
MODEL_TYPE="${MODEL_TYPE:-uitars}"

# ── 根据 MODEL_TYPE 设置 backbone 路径、输出目录、probe 层 ────────
if [[ "${MODEL_TYPE}" == "guiowl7b" ]]; then
    # GUI-Owl-7B: Qwen2.5-VL, 28层, hidden=3584（与 uitars 完全相同架构）
    # prompt 格式使用 GUI-Owl-1.5 的 tool_call 格式（控制变量）
    # patch_size=14, merge_size=2 → token_cell=28px
    # MAX_PIXELS: 16384 tokens × 14² × 2² = 12,845,056（与 uitars 完全相同）
    MODEL_PATH="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/.hdd/models/huggingface.co/GUI_Agents/GUI-Owl-7B"
    PROBE_LAYERS="18,19,20,21,22,23,24,25,26,27"   # last 10 of 28（与 uitars 相同）
    VAL_OUTPUT_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/zwerge/data/results/zwerge-guiowl7b"
    # Qwen2.5-VL 使用 gui_actor 环境（transformers 4.51.3）
    CONDA_ENV="gui_actor"
    MAX_PIXELS="${MAX_PIXELS:-12845056}"   # 16384 × 14² × 4 = 12,845,056
elif [[ "${MODEL_TYPE}" == "guiowl" ]]; then
    # GUI-Owl-1.5-8B: Qwen3-VL, 36层, hidden=4096, deepstack=[8,16,24]
    # patch_size=16, merge_size=2 → token_cell=32px
    # MAX_PIXELS: 16384 tokens × 16² × 2² = 16384 × 1024 = 16,777,216
    # (保持与 uitars 相同的 16384 token 预算; 使用 uitars 的 12845056 则仅约 12544 tokens)
    MODEL_PATH="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/models/huggingface.co/GUI_Agents/GUI-Owl-1.5-8B-Instruct"
    PROBE_LAYERS="21,22,23,24,25,26,27,28,29,30"   # last 10 of 36
    VAL_OUTPUT_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/zwerge/data/results/zwerge-guiowl"
    # Qwen3-VL 需要 transformers>=4.57.1，使用 qwen3 环境
    CONDA_ENV="qwen3-verl"
    MAX_PIXELS="${MAX_PIXELS:-16777216}"   # 16384 × 16² × 4 = 16,777,216
elif [[ "${MODEL_TYPE}" == "uivenus" ]]; then
    # UI-Venus-1.5-8B: Qwen3-VL, 36层, hidden=4096, deepstack=[8,16,24]
    # patch_size=16, merge_size=2 → token_cell=32px
    # MAX_PIXELS: 16384 tokens × 16² × 2² = 16,777,216
    MODEL_PATH="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/models/huggingface.co/GUI_Agents/UI-Venus-1.5-8B"
    PROBE_LAYERS="21,22,23,24,25,26,27,28,29,30"   # last 10 of 36
    VAL_OUTPUT_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/zwerge/data/results/zwerge-uivenus"
    # Qwen3-VL 需要 transformers>=4.57.1，使用 qwen3 环境
    CONDA_ENV="qwen3-verl"
    MAX_PIXELS="${MAX_PIXELS:-16777216}"   # 16384 × 16² × 4 = 16,777,216
else
    # uitars（默认）: Qwen2.5-VL-7B, 28层, hidden=3584
    # patch_size=14, merge_size=2 → token_cell=28px
    # MAX_PIXELS: 16384 tokens × 14² × 2² = 16384 × 784 = 12,845,056
    MODEL_TYPE="uitars"
    MODEL_PATH="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/.hdd/models/huggingface.co/GUI_Agents/UI-TARS-1.5-7B"
    PROBE_LAYERS="18,19,20,21,22,23,24,25,26,27"   # last 10 of 28
    VAL_OUTPUT_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/zwerge/data/results/zwerge-uitars"
    # Qwen2.5-VL 使用 gui_actor 环境（transformers 4.51.3）
    CONDA_ENV="gui_actor"
    MAX_PIXELS="${MAX_PIXELS:-12845056}"   # 16384 × 14² × 4 = 12,845,056
fi

DATA_PATH="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/datasets/grounding_50k.json"

BASE_CKPT_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/.hdd/ckpt/zwerge"
if [ -n "${ZWERGE_JOB_NAME}" ]; then
    OUTPUT_DIR="${BASE_CKPT_DIR}/${ZWERGE_JOB_NAME}"
else
    RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_DIR="${BASE_CKPT_DIR}/${MODEL_TYPE}_grounding50k_A3-gaussian_cos_meta_${RUN_TIMESTAMP}"
fi
mkdir -p "${OUTPUT_DIR}"
export WANDB_RUN_NAME="$(basename "${OUTPUT_DIR}")"

echo "MODEL_TYPE     = ${MODEL_TYPE}"
echo "MODEL_PATH     = ${MODEL_PATH}"
echo "WANDB_RUN_NAME = ${WANDB_RUN_NAME}"
echo "OUTPUT_DIR     = ${OUTPUT_DIR}"
echo "VAL_OUTPUT_DIR = ${VAL_OUTPUT_DIR}"

# ── 5. 消融参数 ─────────────────────────────────────────────────
# PROBE_LAYERS 已在上方 MODEL_TYPE 分支中设置
GT_LABEL_TYPE="gaussian"          # ← A3 核心：Gaussian GT
GAUSSIAN_SIGMA_FACTOR="0.5"
FUSION_TYPE="cos_meta"            # ← A3 核心：cos_meta fusion

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
# VAL_OUTPUT_DIR is set above based on MODEL_TYPE
VAL_CELL_W=300
VAL_CELL_H=220
VAL_ALPHA=0.55

# torchrun 路径：根据 MODEL_TYPE 选择对应 conda 环境
# uitars → gui_actor (transformers 4.51.3, Qwen2.5-VL)
# guiowl / uivenus → qwen3 (transformers 4.57.1, Qwen3-VL)
CONDA_BASE="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/conda/envs"
CONDA_TORCHRUN="${CONDA_BASE}/${CONDA_ENV}/bin/torchrun"
TORCHRUN="$([ -f "${CONDA_TORCHRUN}" ] && echo "${CONDA_TORCHRUN}" || which torchrun 2>/dev/null || echo torchrun)"
echo "CONDA_ENV      = ${CONDA_ENV}"
echo "TORCHRUN       = ${TORCHRUN}"

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
    --model_type "${MODEL_TYPE}" \
    --flash_attn_2_enabled ${FLASH_ATTN} \
    \
    --probe_layers "${PROBE_LAYERS}" \
    --grounding_proj_dim ${GROUNDING_PROJ_DIM} \
    --grounding_adapter_rank ${GROUNDING_ADAPTER_RANK} \
    --grounding_lambda_layer ${GROUNDING_LAMBDA_LAYER} \
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
    --save_steps_only_for_resume ${SAVE_STEPS_ONLY_FOR_RESUME} \
    --logging_steps 10 \
    --dataloader_num_workers 4 \
    \
    --report_to wandb \
    --run_name "${WANDB_RUN_NAME}" \
    \
    --verbose_logging False \
    2>&1 | tee "${OUTPUT_DIR}/train.log"

echo "===== [A3-gaussian+cos_meta] MODEL_TYPE=${MODEL_TYPE} Training complete. Output: ${OUTPUT_DIR} ====="
