#!/bin/bash
# run_all_probes.sh — 重跑 Exp3 + Exp4，覆盖 SS-Pro 和 SS-v2 两个数据集
#
# 运行完成后自动生成全部图表（含分辨率对比图）。
#
# 用法：
#   bash run_all_probes.sh [CUDA_ID]
#
# 估算时间（单卡）：
#   Exp3 SS-Pro  ~45 min  (4 模型 × 200 样本 × 2 pass)
#   Exp3 SS-v2   ~45 min
#   Exp4 SS-Pro  ~35 min  (4 模型 × 150 对)
#   Exp4 SS-v2   ~35 min
#   合计         ~160 min
#
# 如果只想跑单模型：
#   MODEL=guiowl bash run_all_probes.sh 0

set -euo pipefail

CUDA_ID="${1:-0}"
PROBE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================================"
echo "[all] Starting full probe rerun on GPU ${CUDA_ID}"
echo "[all] PROBE_DIR = ${PROBE_DIR}"
echo "============================================================"

# ── 清空旧输出（避免 resume 跳过） ──────────────────────────────────────────
echo ""
echo "[all] Clearing old outputs …"
rm -f "${PROBE_DIR}/outputs/exp3"/*.jsonl 2>/dev/null || true
rm -f "${PROBE_DIR}/outputs/exp3_v2"/*.jsonl 2>/dev/null || true
rm -f "${PROBE_DIR}/outputs/exp4"/*.jsonl 2>/dev/null || true
rm -f "${PROBE_DIR}/outputs/exp4_v2"/*.jsonl 2>/dev/null || true

# ── Exp 3: Spatial Lens vs. Serialization Lens ──────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[all] EXP 3  (SS-Pro + SS-v2)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
DATASET=both bash "${PROBE_DIR}/run_exp3.sh" "${CUDA_ID}"

# ── Exp 4: Counterfactual Instruction Switch ─────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[all] EXP 4  (SS-Pro + SS-v2)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
DATASET=both bash "${PROBE_DIR}/run_exp4.sh" "${CUDA_ID}"

# ── 最终合并绘图（exp3 + exp4 combined，含分辨率对比） ─────────────────────
echo ""
echo "[all] Generating final combined figures …"
conda run -n qwen25 --no-capture-output \
    python "${PROBE_DIR}/plot_probes.py" \
        --exp3_dir    "${PROBE_DIR}/outputs/exp3" \
        --exp4_dir    "${PROBE_DIR}/outputs/exp4" \
        --exp3_dir_v2 "${PROBE_DIR}/outputs/exp3_v2" \
        --exp4_dir_v2 "${PROBE_DIR}/outputs/exp4_v2" \
        --output_dir  "${PROBE_DIR}/outputs/figures" \
        --combined

echo ""
echo "============================================================"
echo "[all] All done."
echo "[all] Figures  → ${PROBE_DIR}/outputs/figures/"
echo "[all] Tables   → ${PROBE_DIR}/outputs/tables/"
echo "============================================================"
