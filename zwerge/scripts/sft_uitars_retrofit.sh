#!/bin/bash
# ============================================================
# ZwerGe-UI Retrofit Training Script
# UI-TARS-1.5-7B → Coordinate-Free Grounding Retrofit
#
# 训练理念（来自 chatgpt-export.txt Phase 4）：
#   - 冻结 backbone（UI-TARS-1.5-7B，28层，Qwen2.5-VL架构）
#   - 只训练 LayerWiseGroundingHead + 新 token embeddings
#   - Loss: KL divergence 在 patch posterior 上
#
# 用法：
#   # 单机多卡（8 GPU）
#   bash scripts/sft_uitars_retrofit.sh
#
#   # 多机（需设置 NNODES, NODE_RANK, MASTER_ADDR）
#   NNODES=2 NODE_RANK=0 MASTER_ADDR=192.168.1.1 bash scripts/sft_uitars_retrofit.sh
# ============================================================

set -e

# ── WandB 配置 ──────────────────────────────────────────────
export WANDB_API_KEY=05140d124018012288eaf1d7166bef50eb16eb3b
export WANDB_PROJECT=Look-Ahead-Agent
export WANDB_RUN_NAME="uitars-retrofit-$(date +%Y%m%d-%H%M%S)"

# ── 路径配置 ────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_PATH="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/.hdd/models/huggingface.co/GUI_Agents/UI-TARS-1.5-7B"
DATA_ROOT="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/datasets/OS-Atlas"
OUTPUT_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/zwerge/data/checkpoints/uitars-retrofit"
RESULTS_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/zwerge/data/results"

mkdir -p "${OUTPUT_DIR}" "${RESULTS_DIR}"

# ── 分布式配置 ──────────────────────────────────────────────
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-29500}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}   # 修改为你实际的 GPU 数量

# ── 训练超参 ────────────────────────────────────────────────
# Layer 配置：UI-TARS-1.5-7B 共 28 层（0-indexed），
# 参考 chatgpt-export.txt 中的分析，选取中后层作为 probe 目标
# Layer 14 ≈ L/2, 18 ≈ 2L/3, 21 ≈ 3L/4, 24-27 为最后几层前的关键层
PROBE_LAYERS="14,18,21,24,26,27"

# 投影维度（比 hidden_size=3584 小，节省显存）
GROUNDING_PROJ_DIM=512

# LoRA adapter rank（per-layer 自适应）
GROUNDING_ADAPTER_RANK=16

# Per-layer loss 权重（loss = loss_fuse + lambda * loss_per_layer）
GROUNDING_LAMBDA_LAYER=0.5

# 主 loss 权重
GROUNDING_LOSS_WEIGHT=1.0

# LM loss 权重（设 0 = 完全冻结，不计算 LM loss，节省计算）
LM_LOSS_WEIGHT=0.0

# 批次设置（根据 GPU 显存调整）
PER_DEVICE_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=4
# 等效全局 batch size = NPROC_PER_NODE * PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS

# 训练轮次
NUM_EPOCHS=3

# 学习率（grounding head 使用较高 LR，因为是随机初始化）
LEARNING_RATE=2e-4
LEARNING_RATE_NEW_TOKENS=2e-4

# 分辨率设置（与 GUI-Actor/GUI-AIMA 一致）
MIN_PIXELS=3136       # 56*56
MAX_PIXELS=5720064    # ~3192*1792

# 序列长度
MODEL_MAX_LENGTH=8192

# ── 数据路径（YAML 配置，支持多数据集混合）──────────────────
# 使用 YAML 配置文件指定多个数据集
DATA_CONFIG="${PROJECT_ROOT}/data/train_data.yaml"

# 如果 YAML 不存在，自动生成一个默认配置
if [ ! -f "${DATA_CONFIG}" ]; then
    mkdir -p "${PROJECT_ROOT}/data"
    cat > "${DATA_CONFIG}" << YAML_EOF
# ZwerGe Retrofit Training Data Configuration
# 修改 sampling_strategy 来控制每个数据集的采样量
datasets:
  # Desktop: Linux
  - json_path: ${DATA_ROOT}/desktop/linux/linux_splited.json
    images_folder: ${DATA_ROOT}/desktop/linux/screenshots
    sampling_strategy: all
  # Web: SeeClick
  - json_path: ${DATA_ROOT}/web/seeclick
    images_folder: ${DATA_ROOT}/web/seeclick
    sampling_strategy: random:5000
YAML_EOF
    echo "Created default data config at ${DATA_CONFIG}"
    echo "Please review and modify ${DATA_CONFIG} before training."
fi

# ── 启动训练 ────────────────────────────────────────────────
cd "${PROJECT_ROOT}"

torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    train_retrofit.py \
    \
    --model_name_or_path "${MODEL_PATH}" \
    --flash_attn_2_enabled True \
    \
    --probe_layers "${PROBE_LAYERS}" \
    --grounding_proj_dim ${GROUNDING_PROJ_DIM} \
    --grounding_adapter_rank ${GROUNDING_ADAPTER_RANK} \
    --grounding_lambda_layer ${GROUNDING_LAMBDA_LAYER} \
    \
    --data_path "${DATA_CONFIG}" \
    --image_folder "" \
    --min_pixels ${MIN_PIXELS} \
    --max_pixels ${MAX_PIXELS} \
    --max_conv_turns 10 \
    \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs ${NUM_EPOCHS} \
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
    --save_steps 500 \
    --save_total_limit 3 \
    --logging_steps 10 \
    --dataloader_num_workers 4 \
    \
    --report_to wandb \
    --run_name "${WANDB_RUN_NAME}" \
    \
    --verbose_logging False \
    2>&1 | tee "${OUTPUT_DIR}/train_$(date +%Y%m%d_%H%M%S).log"

echo "Training complete. Output at ${OUTPUT_DIR}"
