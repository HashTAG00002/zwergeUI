#!/bin/bash
# ============================================================
# Ablation A6: 逐层独立监督（无融合，无共享 MLP）
# ============================================================
# 变量点：
#   INDEPENDENT_LAYERS = true   ← 核心：每层完全独立监督
#   GT_LABEL_TYPE      = gaussian
#   PROBE_LAYERS       = 与 A3 相同（同 MODEL_TYPE 自动选取）
#
# 与 A3 的差异（三处同时关闭）：
#   1. 无 LayerFusionScorer（不存在 omega 参数）
#   2. 无共享 MLP q_proj/k_proj（forced use_shared_mlp=false）
#   3. Loss = mean_l KL(y || p_l)，无 L_fuse 项
#
# 参数量对比（uitars, 10 probe layers）：
#   A3  (MLP + cos_meta fusion): ~12M
#   A5  (无 MLP, 有 fusion)    : ~2.6M
#   A6  (无 MLP, 无 fusion)    : ~1.3M  ← 最轻量，纯 LoRA adapter 监督
#
# 实验目标：
#   验证 grounding signal 是否在没有任何跨层协调的情况下依然能被每层独立学到。
#   若 A6 ≈ A3/A5：fusion 是冗余的，每层 adapter 已有足够判别力。
#   若 A6 < A5 << A3：fusion 带来的跨层权重分配是关键贡献。
#
# 用法：
#   bash scripts/train_ablation_A6_independent_layers.sh         # uitars (默认)
#   MODEL_TYPE=guiowl  bash scripts/train_ablation_A6_independent_layers.sh
#   MODEL_TYPE=uivenus bash scripts/train_ablation_A6_independent_layers.sh
# ============================================================

unset http_proxy https_proxy

if [[ -z "${AFO_ENV_CLUSTER_SPEC:-}" ]]; then
    echo "===== [A6-independent_layers] DEBUG MODE ====="
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
    FLASH_ATTN=False
    conda config --add envs_dirs /mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/conda/envs 2>/dev/null || true
else
    echo "===== [A6-independent_layers] JOB MODE ====="
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
    MAX_STEPS=-1
    SAVE_STEPS=400
    SAVE_STEPS_ONLY_FOR_RESUME=100
    FLASH_ATTN=True
fi

export http_proxy=http://10.70.11.143:8412
export https_proxy=http://10.70.11.143:8412
export WANDB_API_KEY="${WANDB_API_KEY:-your_wandb_api_key_here}"
export WANDB_PROJECT=zwerge

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── 模型类型 ─────────────────────────────────────────────────
MODEL_TYPE="${MODEL_TYPE:-uitars}"

if [[ "${MODEL_TYPE}" == "guiowl" ]]; then
    MODEL_PATH="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/models/huggingface.co/GUI_Agents/GUI-Owl-1.5-8B-Instruct"
    PROBE_LAYERS="21,22,23,24,25,26,27,28,29,30"
    VAL_OUTPUT_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/zwerge/data/results/zwerge-guiowl"
    CONDA_ENV="qwen3"
    MAX_PIXELS="${MAX_PIXELS:-16777216}"
elif [[ "${MODEL_TYPE}" == "uivenus" ]]; then
    MODEL_PATH="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/models/huggingface.co/GUI_Agents/UI-Venus-1.5-8B"
    PROBE_LAYERS="21,22,23,24,25,26,27,28,29,30"
    VAL_OUTPUT_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/zwerge/data/results/zwerge-uivenus"
    CONDA_ENV="qwen3"
    MAX_PIXELS="${MAX_PIXELS:-16777216}"
else
    MODEL_TYPE="uitars"
    MODEL_PATH="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/.hdd/models/huggingface.co/GUI_Agents/UI-TARS-1.5-7B"
    PROBE_LAYERS="18,19,20,21,22,23,24,25,26,27"
    VAL_OUTPUT_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/zwerge/data/results/zwerge-uitars"
    CONDA_ENV="gui_actor"
    MAX_PIXELS="${MAX_PIXELS:-12845056}"
fi

# 训练集：多个文件换行分隔，dataset.py 合并后随机打乱
_DS=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/datasets
DATA_PATH="${_DS}/grounding_50k.json
${_DS}/grounding_jedi_4k.json"

BASE_CKPT_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/.hdd/ckpt/zwerge"
if [ -n "${ZWERGE_JOB_NAME}" ]; then
    OUTPUT_DIR="${BASE_CKPT_DIR}/${ZWERGE_JOB_NAME}"
else
    RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_DIR="${BASE_CKPT_DIR}/${MODEL_TYPE}_grounding50k_A6-independent_layers_${RUN_TIMESTAMP}"
fi
mkdir -p "${OUTPUT_DIR}"
export WANDB_RUN_NAME="$(basename "${OUTPUT_DIR}")"

echo "MODEL_TYPE     = ${MODEL_TYPE}"
echo "MODEL_PATH     = ${MODEL_PATH}"
echo "WANDB_RUN_NAME = ${WANDB_RUN_NAME}"
echo "OUTPUT_DIR     = ${OUTPUT_DIR}"

# ── A6 核心参数 ─────────────────────────────────────────────────
# INDEPENDENT_LAYERS=true  → 自动禁用 fusion 和共享 MLP
GT_LABEL_TYPE="gaussian"
GAUSSIAN_SIGMA_FACTOR="0.5"
INDEPENDENT_LAYERS=true
# FUSION_TYPE 不影响结果（independent_layers=true 时 fusion 不被创建）
# USE_SHARED_MLP 不影响结果（independent_layers=true 时强制关闭）

GROUNDING_PROJ_DIM=1024    # 有 MLP 时用，independent 模式下 d_proj 不生效但配置保留
GROUNDING_ADAPTER_RANK=16
GROUNDING_LAMBDA_LAYER=0.5  # 不影响 independent 模式（只有 loss_layer）
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
VAL_CELL_W=300
VAL_CELL_H=220
VAL_ALPHA=0.55

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
    --grounding_independent_layers ${INDEPENDENT_LAYERS} \
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
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --bf16 True \
    --tf32 True \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --save_steps ${SAVE_STEPS} \
    --save_steps_only_for_resume ${SAVE_STEPS_ONLY_FOR_RESUME} \
    --save_total_limit 3 \
    --model_max_length ${MODEL_MAX_LENGTH} \
    --report_to wandb \
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
