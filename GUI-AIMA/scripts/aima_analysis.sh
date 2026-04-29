MODEL_PATH="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/.hdd/models/huggingface.co/GUI_Agents/GUI-AIMA-3B"
DATASET_PATH="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/zwerge/data/eval/ScreenSpot-Pro/eval.json"
RESULT_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/zwerge/data/results"
CODE_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/zwerge/code/GUI-AIMA"

cd "$CODE_DIR"
mkdir -p "$RESULT_DIR"

torchrun --nproc_per_node=2 eval/layer_probe.py \
    --mode probe \
    --model_path "$MODEL_PATH" \
    --dataset_path "$DATASET_PATH" \
    --num_samples -1 \
    --output_path "$RESULT_DIR/sspro_probe.json" \
    --save_every 20


# ============================================================
# eval 模式：复现 AIMA baseline（所有层）
# ============================================================
# python eval/layer_probe.py \
#     --mode eval \
#     --model_path "$MODEL_PATH" \
#     --dataset_path "$DATASET_PATH" \
#     --num_samples -1 \
#     --layer_mask all \
#     --output_path "$RESULT_DIR/sspro_eval_all.json" \
#     --save_every 20 \
#     --gpu_id 0

# ============================================================
# eval 模式：只用最后一层 / 去掉最后一层 / 前一半层
# ============================================================
# for MASK in last not_last first_half last_half; do
#     python eval/layer_probe.py \
#         --mode eval \
#         --model_path "$MODEL_PATH" \
#         --dataset_path "$DATASET_PATH" \
#         --num_samples -1 \
#         --layer_mask "$MASK" \
#         --output_path "$RESULT_DIR/sspro_eval_${MASK}.json" \
#         --save_every 20 \
#         --gpu_id 0
# done

echo "Done. Results in $RESULT_DIR"
