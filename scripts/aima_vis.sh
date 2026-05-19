MODEL_PATH="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/.hdd/models/huggingface.co/GUI_Agents/GUI-AIMA-3B"
DATASET_PATH="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/zwerge/data/eval/ScreenSpot-Pro/eval.json"
RESULT_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/zwerge/data/results/gui_aima"
CODE_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/zwerge/code/GUI-AIMA"

cd "$CODE_DIR"
mkdir -p "$RESULT_DIR"


# ============================================================
# vis 模式：逐层注意力热图可视化（Signal A / B，层 18-36）
# 输出：$RESULT_DIR/gui_aima/sample_XXXX/{A_layerXX.png, grid_A.png, meta.json}
# ============================================================
python eval/layer_probe.py \
    --mode vis \
    --model_path "$MODEL_PATH" \
    --dataset_path "$DATASET_PATH" \
    --vis_samples 100 \
    --vis_layers "18-36" \
    --vis_signal "A,B" \
    --vis_output_dir "$RESULT_DIR/vis" \
    --vis_alpha 0.55 \
    --gpu_id 0

echo "Done. Results in $RESULT_DIR"
