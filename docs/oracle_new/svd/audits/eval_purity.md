# Gate-C.4 Eval 纯净性审计（Stage-1 SVD 实验用）

审计日期 2026-09-22，审计对象 = 当前 HEAD（== baseline 9329b27）的评测代码。
结论先行：**`decode_strategy="centroid"` 时，逐层 `hit_top1`/`overlap_top1` 不经过任何
gate / fallback / ensemble / threshold-tuning；independent(A7) 模式下 `fusion_acc` 只是均匀均值（eval 展示用）。**

## 1. 逐层统计来源（无任何 trick 渗入）

- `zwerge/eval/eval_retrofit.py:543-567`：逐层指标循环只读
  `pred["per_layer_points"][li]` 与 `pred["per_layer_topk"][li]`，直接累积进
  `layer_stats[li]["hit1"/"hitk"/"overlap1"/"overlapk"]`。
- `eval_retrofit.py:520-529`（else 分支）：`decode_strategy="centroid"` 时走
  `grounder.predict_layerwise(..., decode_strategy="centroid", ...)`。
  p2p / p2p_zoom_gated / p2p_ensemble / p2p_native / zoom_backbone 各分支（L462-519）
  只影响**最终融合点** `fpx,fpy`（L573-595），不改写 `per_layer_points`；
  它们污染的是 `fusion_stats`，不是 `layer_stats`。
- `zwerge/eval/inference_base.py:1343-1353`（`predict_layerwise`）：每层后验 `p_l`
  独立调用同一个 `scores_to_point_and_topk(..., decode_strategy=decode_strategy)`，
  层与层之间无交叉项、无跨层共识、无重排。

## 2. centroid 解码器本身

- `inference_base.py:150-172` `scores_to_point_and_topk` → `get_prediction_region_point`。
- `inference_base.py:46-143`：threshold = max × activation_threshold（L52）→
  4-连通 BFS 区域生长（L72-92）→ region score = 区域内 max（L99）→
  centroid = 概率加权重心（L114-117）→ **`else: center = centroid`（L134-135，centroid 即默认分支）**
  → top-1 = 分数最高区域的 centroid（L139-143）。
- `peak_shift` / `temperature` 只在各自分支生效（L121-133），centroid 路径不经过。
- baseline 口径：A7 评测 `DECODE_STRATEGY=centroid`、`VAL_ALPHA=0.55`（即 activation_threshold=0.55，
  `eval_retrofit.py:1153 activation_threshold=args.activation_threshold`），cell 由
  `eval_retrofit.py:537-539` 的 `phx=0.5/n_w, phy=0.5/n_h` 给出（overlap 判定框，L549-556）。
  注意：论文口径的 "cell 300×220" 是训练脚本 `--val_cell_w/h` 传入的另一套 grid 评估参数，
  与 layerwise summary 的 `phx/phy` 半格框不是同一对象；本轮统一沿用 layerwise summary 口径。

## 3. independent(A7) 模式的 fusion_acc

- `zwerge/src/zwerge_retrofit/modeling_base.py:497-503`：`independent_layers=True` 时
  `p_final = sum(active_p)/num_active_probes`、`omega = uniform` —— 纯均匀均值，无可学参数，
  仅 eval 展示，不进 loss（L523-533 每层独立 KL）。

## 4. 本轮使用约束

- 只允许 `decode_strategy="centroid"` 的 else 分支（`eval_retrofit.py:520-529`）；
  禁止把 p2p/ensemble/gated 分支的 `fusion_stats` 当作本轮 Stage-1 指标引用。
- 与旧 Xavier 基线对比时，直接取各 ckpt `results/layerwise_all_summary.json` 的
  `layer_accs[*].hit_top1/overlap_top1`，与本轮新 run 同管线产出，口径天然一致。
