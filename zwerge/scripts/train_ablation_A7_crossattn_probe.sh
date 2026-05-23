#!/bin/bash
# ============================================================
# Ablation A7: Cross-Attention Probe（多头全秩注意力 adapter）
# ============================================================
# 在 A6 基础上替换 adapter 架构：
#   A6: LoRA rank=16（每方向 16 自由度，表达力受限）
#   A7: CrossAttnProbe（全秩 W_q/W_k + 多头注意力打分）
#
# 核心参数：
#   ADAPTER_TYPE   = attn            ← 替换 LoRA
#   ATTN_HEADS     = 8               ← n_heads
#   ATTN_HEAD_DIM  = 64              ← d_head，d_attn = 8×64 = 512
#   INDEPENDENT_LAYERS = true        ← 继承自 A6（每层独立监督，无 fusion）
#
# 参数量对比（10 probe layers）：
#   A6  LoRA r=16:          ~2.5M / ~2.8M
#   A7  CrossAttn n=8,d=64: ~36M  / ~42M  (≈0.52% of 7B/8B)  ← 在 0.5-0.75% 目标区间内
#
# 实验目标：
#   A6 中 LoRA rank=16 限制了 guiowl/uivenus 在 4096-d 空间的表达能力
#   CrossAttn 通过全秩投影 + 多头注意力提升每层的定位判别力
#
# 用法：
#   bash scripts/train_ablation_A7_crossattn_probe.sh           # uitars 默认
#   MODEL_TYPE=guiowl7b bash scripts/train_ablation_A7_crossattn_probe.sh
#   MODEL_TYPE=guiowl   bash scripts/train_ablation_A7_crossattn_probe.sh
#   MODEL_TYPE=uivenus  bash scripts/train_ablation_A7_crossattn_probe.sh
#
#   # 自定义 head 数（减小参数量）：
#   ATTN_HEADS=4 ATTN_HEAD_DIM=64 bash scripts/train_ablation_A7_crossattn_probe.sh
# ============================================================

unset http_proxy https_proxy

if [[ -z "${AFO_ENV_CLUSTER_SPEC:-}" ]]; then
    echo "===== [A7-crossattn_probe] DEBUG MODE ====="
    export NPROC_PER_NODE=2
    export NODE_RANK=0
    export NNODES=1
    export MASTER_ADDR="127.0.0.1"
    export MASTER_PORT=29500
    PER_DEVICE_BATCH_SIZE=1
    GRADIENT_ACCUMULATION_STEPS=2
    NUM_EPOCHS=1
    MAX_STEPS=30
    SAVE_STEPS="${SAVE_STEPS:-30}"
    SAVE_STEPS_ONLY_FOR_RESUME="${SAVE_STEPS_ONLY_FOR_RESUME:--1}"
    FLASH_ATTN=False
    conda config --add envs_dirs /mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/conda/envs 2>/dev/null || true
else
    echo "===== [A7-crossattn_probe] JOB MODE ====="
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── 模型类型 ─────────────────────────────────────────────────
MODEL_TYPE="${MODEL_TYPE:-uitars}"

if [[ "${MODEL_TYPE}" == "guiowl7b" ]]; then
    # GUI-Owl-7B: Qwen2.5-VL, 28层, hidden=3584
    MODEL_PATH="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/.hdd/models/huggingface.co/GUI_Agents/GUI-Owl-7B"
    PROBE_LAYERS="14,15,16,17,18,19,20,21,22,23,24,25,26,27"   # last 14 of 28 (last 1/2)
    CONDA_ENV="gui_actor"
    MAX_PIXELS="${MAX_PIXELS:-12845056}"
elif [[ "${MODEL_TYPE}" == "guiowl" ]]; then
    # GUI-Owl-1.5-8B: Qwen3-VL, 36层
    MODEL_PATH="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/models/huggingface.co/GUI_Agents/GUI-Owl-1.5-8B-Instruct"
    PROBE_LAYERS="18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35"   # last 18 of 36 (last 1/2)
    CONDA_ENV="qwen3-verl"
    MAX_PIXELS="${MAX_PIXELS:-16777216}"
elif [[ "${MODEL_TYPE}" == "uivenus" ]]; then
    # UI-Venus-1.5-8B: Qwen3-VL, 36层
    MODEL_PATH="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/models/huggingface.co/GUI_Agents/UI-Venus-1.5-8B"
    PROBE_LAYERS="18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35"   # last 18 of 36 (last 1/2)
    CONDA_ENV="qwen3-verl"
    MAX_PIXELS="${MAX_PIXELS:-16777216}"
else
    # UI-TARS-1.5-7B: Qwen2.5-VL, 28层（默认）
    MODEL_TYPE="uitars"
    MODEL_PATH="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/.hdd/models/huggingface.co/GUI_Agents/UI-TARS-1.5-7B"
    PROBE_LAYERS="14,15,16,17,18,19,20,21,22,23,24,25,26,27"   # last 14 of 28 (last 1/2)
    CONDA_ENV="gui_actor"
    MAX_PIXELS="${MAX_PIXELS:-12845056}"
fi

DATA_PATH="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/datasets/grounding_200k.jsonl"

BASE_CKPT_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/.hdd/ckpt/zwerge"
if [ -n "${ZWERGE_JOB_NAME}" ]; then
    OUTPUT_DIR="${BASE_CKPT_DIR}/${ZWERGE_JOB_NAME}"
else
    RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_DIR="${BASE_CKPT_DIR}/${MODEL_TYPE}_grounding200k_A7-crossattn_probe_${RUN_TIMESTAMP}"
fi
mkdir -p "${OUTPUT_DIR}"
# 评估结果写到对应 checkpoint 目录下的 results/ 子目录（与 eval_daemon 异步评估路径一致）
VAL_OUTPUT_DIR="${OUTPUT_DIR}"
export WANDB_RUN_NAME="$(basename "${OUTPUT_DIR}")"

echo "MODEL_TYPE     = ${MODEL_TYPE}"
echo "MODEL_PATH     = ${MODEL_PATH}"
echo "WANDB_RUN_NAME = ${WANDB_RUN_NAME}"
echo "OUTPUT_DIR     = ${OUTPUT_DIR}"

# ── A7 核心参数 ─────────────────────────────────────────────────
# CrossAttn probe: 全秩 W_q/W_k + 多头注意力，替换 LoRA adapter
GT_LABEL_TYPE="gaussian"
GAUSSIAN_SIGMA_FACTOR="${GAUSSIAN_SIGMA_FACTOR:-0.35}"
INDEPENDENT_LAYERS=true       # 每层独立监督（继承自 A6）
ADAPTER_TYPE="attn"           # ← A7 核心：CrossAttnGroundingProbe
ATTN_HEADS="${ATTN_HEADS:-8}"          # n_heads（默认 8）
ATTN_HEAD_DIM="${ATTN_HEAD_DIM:-64}"   # d_head（默认 64，d_attn=512）

GROUNDING_PROJ_DIM=1024    # 不影响 attn 模式（probe 有自己的 W_q/W_k），保留以防 compat
GROUNDING_ADAPTER_RANK=16  # 不影响 attn 模式
GROUNDING_LAMBDA_LAYER=0.5 # 不影响 independent 模式（只有 loss_layer）
GROUNDING_LOSS_WEIGHT=1.0
LM_LOSS_WEIGHT=0.0
LEARNING_RATE=2e-4
LEARNING_RATE_NEW_TOKENS=2e-4
MIN_PIXELS=3136
MODEL_MAX_LENGTH=18432

# ── 同步/异步评估互斥开关 ─────────────────────────────────────
# EVAL_MODE=async（默认）：禁用同步评估，依靠 eval_daemon 异步提交 hope job 评估
# EVAL_MODE=sync：启用同步评估，训练中每 VAL_FREQ 步评测一次（调试用，生产慎用）
EVAL_MODE="${EVAL_MODE:-async}"
VAL_FREQ="${VAL_FREQ:-400}"     # 仅 sync 模式生效
if [[ "${EVAL_MODE}" == "async" ]]; then
    VAL_STEPS=-1                # 完全关闭，不占训练时间
else
    VAL_STEPS="${VAL_FREQ}"
fi
VAL_BENCH="all"
VAL_N_SAMPLES=-1
VAL_DECODE_STRATEGY="centroid"
VAL_CELL_W=300
VAL_CELL_H=220
VAL_ALPHA=0.55

echo "ADAPTER_TYPE   = ${ADAPTER_TYPE}"
echo "ATTN_HEADS     = ${ATTN_HEADS}"
echo "ATTN_HEAD_DIM  = ${ATTN_HEAD_DIM}  (d_attn = $((ATTN_HEADS * ATTN_HEAD_DIM)))"

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
    --grounding_adapter_type ${ADAPTER_TYPE} \
    --grounding_attn_heads ${ATTN_HEADS} \
    --grounding_attn_head_dim ${ATTN_HEAD_DIM} \
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
