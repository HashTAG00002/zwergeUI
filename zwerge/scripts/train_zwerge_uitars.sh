#!/bin/bash
# ============================================================
# ZwerGe-UI  Retrofit Training: UI-TARS-1.5-7B
# Coordinate-Free Grounding Head on frozen backbone
#
# 使用方法：
#   本地 codelab debug：  bash scripts/train_zwerge_uitars.sh
#   AFO job 队列：        由 train_zwerge_uitars.hope 自动调用
# ============================================================

# ── 1. 环境检测：本地 codelab vs AFO job ─────────────────────
unset http_proxy https_proxy

if [[ -z "${AFO_ENV_CLUSTER_SPEC:-}" ]]; then
    # ── 本地 codelab：2 张 A100-40G ──
    echo "===== [ZwerGe] DEBUG MODE (codelab) ====="
    export NPROC_PER_NODE=2
    export NODE_RANK=0
    export NNODES=1
    export MASTER_ADDR="127.0.0.1"
    export MASTER_PORT=29500

    # DEBUG 超参：小 batch，只跑 30 steps 验证流程
    PER_DEVICE_BATCH_SIZE=1
    GRADIENT_ACCUMULATION_STEPS=2
    NUM_EPOCHS=1
    MAX_STEPS=30
    SAVE_STEPS=30
    MAX_PIXELS=2494464      # A100-40G 显存较小，限制分辨率

    # codelab 的 gcc 版本太老（缺 stdatomic.h），triton kernel 无法编译
    # 改用 sdpa（PyTorch 原生实现，无需 triton），debug 阶段够用
    FLASH_ATTN=False

    conda config --add envs_dirs /mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/conda/envs 2>/dev/null || true

else
    # ── AFO job 队列：8 张 A100-80G ──
    echo "===== [ZwerGe] JOB MODE (AFO cluster) ====="
    nvidia-smi
    conda config --add envs_dirs /mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/conda/envs
    conda env list

    export NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)
    export NODE_RANK="$(jq -r '.index | tonumber' <<<"$AFO_ENV_CLUSTER_SPEC")"
    export NNODES="$(jq -r '.worker | length' <<<"$AFO_ENV_CLUSTER_SPEC")"
    _REAL_IP=""
    # 策略1：出口路由对应的源 IP（最可靠）
    _REAL_IP=$(ip route get 1.0.0.0 2>/dev/null \
        | awk '/src/{for(i=1;i<=NF;i++) if($i=="src") print $(i+1)}' \
        | head -1)
    # 策略2：从 hostname -I 取第一个 10.x.x.x 地址
    if [[ -z "$_REAL_IP" ]]; then
        _REAL_IP=$(hostname -I 2>/dev/null | tr ' ' '\n' | grep -E '^10\.' | head -1)
    fi
    # 策略3：取所有非 127 的第一个地址兜底
    if [[ -z "$_REAL_IP" ]]; then
        _REAL_IP=$(hostname -I 2>/dev/null | tr ' ' '\n' | grep -vE '^127\.' | head -1)
    fi

    # rank-0 用真实 IP 作为 MASTER_ADDR，其他 rank 也使用同一地址
    # 多节点时 MASTER_ADDR 应是 rank-0 的 IP：
    #   若 NODE_RANK==0 → 用本机探测的真实 IP
    #   若 NODE_RANK!=0 → 仍用 AFO worker[0] 的地址（但做端口解析保护）
    _WORKER0_RAW="$(jq -r '.worker[0]' <<<"$AFO_ENV_CLUSTER_SPEC")"
    if [[ "$NODE_RANK" == "0" ]] || [[ -z "${NODE_RANK}" ]]; then
        # rank-0：用本机探测的真实内网 IP
        export MASTER_ADDR="${_REAL_IP:-127.0.0.1}"
    else
        # 非 rank-0：尝试从 worker[0] 解析 IP，格式 "ip:port" 或纯域名
        _W0_IP="${_WORKER0_RAW%%:*}"
        # 若解析到的是数字 IP（10.x / 172.x / 192.x），直接用；否则用原始值
        if [[ "$_W0_IP" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
            export MASTER_ADDR="${_W0_IP}"
        else
            # K8s svc 域名形式，无法直接 resolve；退回探测的真实 IP
            export MASTER_ADDR="${_REAL_IP:-127.0.0.1}"
        fi
    fi

    # MASTER_PORT：从 worker[0] 尝试解析，取不到则用 29500
    _W0_PORT="${_WORKER0_RAW##*:}"
    if [[ "$_W0_PORT" =~ ^[0-9]+$ ]] && [[ "$_W0_PORT" -ge 1024 ]] && [[ "$_W0_PORT" -le 65535 ]]; then
        export MASTER_PORT="${_W0_PORT}"
    else
        export MASTER_PORT=29500
    fi

    echo "AFO worker[0] raw   = ${_WORKER0_RAW}"
    echo "Real IP (detected)  = ${_REAL_IP}"
    echo "MASTER_ADDR (final) = ${MASTER_ADDR}"
    echo "MASTER_PORT (final) = ${MASTER_PORT}"

    # job 正式超参
    PER_DEVICE_BATCH_SIZE=2
    GRADIENT_ACCUMULATION_STEPS=4
    NUM_EPOCHS=3
    MAX_STEPS=-1            # -1 = 跑完所有 epoch
    SAVE_STEPS=400
    MAX_PIXELS=12845056    # A100-80G 全分辨率

    # job 环境 gcc 新，可用 flash_attention_2
    FLASH_ATTN=True
fi

# ── 2. WandB & 代理（job 队列也需要通过代理连外网）──────────────
export http_proxy=http://10.70.11.143:8412
export https_proxy=http://10.70.11.143:8412
export WANDB_API_KEY=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
export WANDB_PROJECT=zwerge
export WANDB_RUN_NAME="zwerge-uitars7b-grounding50k-$(date +%Y%m%d-%H%M%S)"

# ── 3. 打印环境信息 ────────────────────────────────────────────
echo "NPROC_PER_NODE = ${NPROC_PER_NODE}"
echo "NODE_RANK      = ${NODE_RANK}"
echo "NNODES         = ${NNODES}"
echo "MASTER_ADDR    = ${MASTER_ADDR}"
echo "MASTER_PORT    = ${MASTER_PORT}"
echo "FLASH_ATTN     = ${FLASH_ATTN}"
echo "WANDB_RUN_NAME = ${WANDB_RUN_NAME}"

# ── 4. 路径配置 ────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"    # .../zwerge/code/zwerge/

MODEL_PATH="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/.hdd/models/huggingface.co/GUI_Agents/UI-TARS-1.5-7B"
DATA_PATH="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/datasets/grounding_50k.json"

RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/.hdd/ckpt/zwerge/uitars7b_grounding50k_${RUN_TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"

echo "PROJECT_ROOT = ${PROJECT_ROOT}"
echo "MODEL_PATH   = ${MODEL_PATH}"
echo "DATA_PATH    = ${DATA_PATH}"
echo "OUTPUT_DIR   = ${OUTPUT_DIR}"

# ── 5. Head 超参 ───────────────────────────────────────────────
# UI-TARS-1.5-7B 共 28 层（0-indexed），选取中后层 probe
#PROBE_LAYERS="10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27"
PROBE_LAYERS="18,19,20,21,22,23,24,25,26,27"
GROUNDING_PROJ_DIM=1024
GROUNDING_ADAPTER_RANK=16
GROUNDING_LAMBDA_LAYER=0.5
GROUNDING_LOSS_WEIGHT=1.0
LM_LOSS_WEIGHT=0.0          # backbone 完全冻结，不计算 LM loss

LEARNING_RATE=2e-4
LEARNING_RATE_NEW_TOKENS=2e-4
MIN_PIXELS=3136             # 56×56
# MODEL_MAX_LENGTH 计算：
#   MAX_PIXELS=12845056，Qwen2.5-VL patch=14, merge=2，每 token 覆盖 28×28 像素
#   最大视觉 token 数 = 12845056 / (28×28) = 16384
#   + system message (~150) + instruction (~200) + response (~30) = ~16800
#   取 16384 + 2048 = 18432 留足余量，确保高分辨率样本不被过滤掉
MODEL_MAX_LENGTH=18432

# ── 6. 启动 torchrun ───────────────────────────────────────────
# 优先使用 conda 环境里的 torchrun（避免系统 PATH 找不到）
CONDA_TORCHRUN="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/conda/envs/gui_actor/bin/torchrun"
if [[ -f "${CONDA_TORCHRUN}" ]]; then
    TORCHRUN="${CONDA_TORCHRUN}"
else
    TORCHRUN="$(which torchrun 2>/dev/null || echo torchrun)"
fi
echo "TORCHRUN = ${TORCHRUN}"

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
    \
    --data_path "${DATA_PATH}" \
    --image_folder "" \
    --min_pixels ${MIN_PIXELS} \
    --max_pixels ${MAX_PIXELS} \
    --max_conv_turns 10 \
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

echo "===== [ZwerGe] Training complete. Output: ${OUTPUT_DIR} ====="
