MODEL_PATH="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/.hdd/models/huggingface.co/GUI_Agents/GUI-AIMA-3B"
DATASET_PATH="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/zwerge/data/eval/ScreenSpot-Pro/eval.json"
CODE_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/zwerge/code/GUI-AIMA"

cd "$CODE_DIR"

# ============================================================
# Exp 1: probe 模式 — 记录每一层的 ACC_A / ACC_B / Entropy
#         用于绘制 Figure: layer-wise grounding quality curve
# ============================================================
python eval/layer_probe.py \
    --mode probe \
    --model_path "$MODEL_PATH" \
    --dataset_path "$DATASET_PATH" \
    --num_samples -1 \
    --output_path eval_results/sspro_probe_all.json \
    --gpu_id 0
# # ============================================================
# # Exp 2: eval 模式 — 使用全部层（复现 AIMA baseline）
# # ============================================================
# python eval/layer_probe.py \
#     --mode eval \
#     --model_path "$MODEL_PATH" \
#     --dataset_path "$DATASET_PATH" \
#     --num_samples -1 \
#     --layer_mask all \
#     --output_path eval_results/sspro_eval_all_layers.json \
#     --gpu_id 0

# # ============================================================
# # Exp 3: eval 模式 — 只用最后一层（测试单层能力上限）
# # ============================================================
# python eval/layer_probe.py \
#     --mode eval \
#     --model_path "$MODEL_PATH" \
#     --dataset_path "$DATASET_PATH" \
#     --num_samples -1 \
#     --layer_mask last \
#     --output_path eval_results/sspro_eval_last_layer.json \
#     --gpu_id 0

# # ============================================================
# # Exp 4: eval 模式 — 去除最后一层（浅层的信息论上限）
# # ============================================================
# python eval/layer_probe.py \
#     --mode eval \
#     --model_path "$MODEL_PATH" \
#     --dataset_path "$DATASET_PATH" \
#     --num_samples -1 \
#     --layer_mask not_last \
#     --output_path eval_results/sspro_eval_no_last.json \
#     --gpu_id 0

# # ============================================================
# # Exp 5: eval 模式 — 只用前一半层（最浅层信息论能力）
# # ============================================================
# python eval/layer_probe.py \
#     --mode eval \
#     --model_path "$MODEL_PATH" \
#     --dataset_path "$DATASET_PATH" \
#     --num_samples -1 \
#     --layer_mask first_half \
#     --output_path eval_results/sspro_eval_first_half.json \
#     --gpu_id 0

echo "All experiments done. Results in eval_results/"
