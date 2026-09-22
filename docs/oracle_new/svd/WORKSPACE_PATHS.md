# SVD Stage-1 实验：Workspace 路径与基线速查

> 审计 agent 实地核实（2026-09-22），所有路径均已 `ls` 验证存在。
> Baseline commit：`9329b271709aabceb54ca16224f75733116a9ee0`（当前 HEAD 与此一致，tracked 代码零差异，sha256 与 `source_manifest.json` 全对得上）。

## 1. 是什么 → 在哪

| 是什么 | 在哪 |
|---|---|
| 200K 训练数据 | `/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/datasets/grounding_200k.jsonl`（85MB，AgentNet42k+OS-Atlas35k+GroundCUA123k） |
| 评测数据 | `/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/datasets/evaluation/{ScreenSpot-Pro,ScreenSpot-v2,OSWorld-G,MMBench-GUI,UI-Vision}` + `eval_all.json` |
| A7 训练脚本（Stage-1 唯一入口） | `zwerge/scripts/train_ablation_A7_crossattn_probe.sh`（`MODEL_TYPE` 切 backbone） |
| A8 训练脚本（本轮禁用，仅参考） | `zwerge/scripts/train_ablation_A8_cosmeta_context.sh` |
| 训练主入口 | `zwerge/train_retrofit.py`；SVD 插入点 = L700-705 `setup_special_token_ids` 之后、L733 `setup_trainable_params` 之前 |
| Probe 实现 | `zwerge/src/zwerge_retrofit/modeling_base.py`：`CrossAttnGroundingProbe` L199-275；读 `hidden_states[layer_idx+1]` L482；`reinit_grounding_head()`（Xavier 覆盖点）L639-654 |
| SVD 实现包（GPT 交付，已过本地审计） | `docs/oracle_new/svd/{paired_qk_svd.py,test_paired_qk_svd.py,test_results.txt,README.md,source_manifest.json}` |
| Xavier baseline checkpoint（论文主结果） | `/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/.hdd/ckpt/zwerge/{uitars_A7_exp001,guiowl7b_A7_exp001,guiowl_A7_exp002,uivenus_A7_exp002}/checkpoint-2800`（终态 3129） |
| 无妥协 Stage-1 逐层结果（hit+overlap） | 上述 ckpt 下 `results/layerwise_all_summary.json`（字段 `hit_top1`/`overlap_top1`/`layer_accs`，centroid 解码，无 gate/ensemble） |
| 训练日志 / loss 曲线 | 各 exp 目录 `train.log` + `checkpoint-*/trainer_state.json` + WandB project `zwerge` |
| hope 训练作业模板 | `zwerge/scripts/train_zwerge_uitars.hope`（单 worker 8×80G，queue `root.zw05_training_cluster.hadoop-vision.elastic_job`） |
| hope 评测模板 | `zwerge/scripts/eval/eval_zwerge.hope`（+`_qwen3` 版） |
| eval_daemon 实验配置 | `zwerge/experiments/A7_*.yaml` |
| 统一评测入口 | `zwerge/eval/eval_retrofit.py`（layerwise summary 产出者，`decode_strategy=centroid`） |

## 2. 各 backbone 关键参数

| MODEL_TYPE | 架构 | 层数 | hidden | Q/KV heads | head_dim | probe_layers | 训练 env | 评测 env |
|---|---|---|---|---|---|---|---|---|
| `uitars` (UI-TARS-1.5-7B) | Qwen2.5-VL | 28 | 3584 | 28/4 | 128 | 14-27 | `gui_actor` (tf 4.51.3, torch 2.5.1) | `qwen25` |
| `guiowl7b` (GUI-Owl-7B) | Qwen2.5-VL | 28 | 3584 | 28/4 | 128 | 14-27 | `gui_actor` | `qwen25` |
| `guiowl` (GUI-Owl-1.5-8B) | Qwen3-VL | 36 | 4096 | 32/8 | 128 | 18-35 | `qwen3` (tf 4.57.1, torch 2.8.0) | `qwen3` |
| `uivenus` (UI-Venus-1.5-8B) | Qwen3-VL | 36 | 4096 | 32/8 | 128 | 18-35 | `qwen3` | `qwen3` |

- 权重 key：Qwen2.5-VL = `model.layers.N.self_attn.q_proj.weight`（monolithic，`Qwen2_5_VLModel.layers`，无 `language_model` 中间层）；Qwen3-VL = `model.language_model.layers.N...` 且带 `q_norm/k_norm`。`paired_qk_svd.decoder_layers()` 三条 fallback 路径均已覆盖。
- SVD 截断：Qwen2.5-VL 核宽度 hkv·d0=512=probe 宽 → **零截断**（energy=1.0，实测重建误差 1.6e-6）；Qwen3-VL 核宽 1024→512，L19 实测保留 **95.6%** 能量（tail 21% rel-Frobenius），逐层须记 report。

## 3. 旧训练配置（复现基线必须逐字对齐）

`lr=2e-4`（cosine, warmup 3%）、`adamw_torch`、bf16、1 epoch=3129 steps（job 模式 per_device 2 × accum 4 × 8 GPU = global 64）、`gt_label_type=gaussian`、`gaussian_sigma_factor=0.35`、`ATTN_HEADS=8 × ATTN_HEAD_DIM=64`（投影宽 512）、`unfreeze_new_tokens` 默认 True、`unfreeze_grounding_head=True`、`lm_loss_weight=0`、`MAX_PIXELS`：Qwen2.5=12845056 / Qwen3=16777216、eval `decode=centroid`、`cell 300×220`、`alpha 0.55`。旧 run 尾部：loss 5.9→~1.65，终态 LR≈5e-9（cosine 已退火到 0），`p_final_entropy` 3.66 远未饱和——"是否欠训"无法从旧曲线判定，正是本轮要回答的问题。

## 4. Xavier baseline（无妥协 Stage-1，ckpt-2800，逐层独立 probe，centroid）

| backbone | bench | best overlap@1 (layer) | best hit@1 (layer) | first layer ovl/hit | last layer ovl/hit |
|---|---|---|---|---|---|
| uitars | SS-Pro | **52.44** (L20) | **41.62** (L20) | 34.16/23.78 (L14) | 42.19/30.93 (L27) |
| uitars | SS-v2 | 91.27 (L21) | 90.01 (L21) | 71.83/69.87 | 82.06/80.02 |
| uitars | OSWorld-G | 75.69 (L22) | 61.18 (L22) | 57.65/45.69 | 64.71/50.20 |
| uitars | MMBench | 78.35 (L22) | 74.60 (L19) | 56.76/52.39 | 68.67/63.27 |
| uitars | UI-Vision | 37.34 (L24) | 28.26 (L24) | 29.05/20.64 | 33.49/24.63 |
| guiowl7b | SS-Pro | **58.95** (L21) | **47.44** (L21) | 40.73/29.41 (L14) | 48.20/37.19 (L27) |
| guiowl7b | SS-v2 | 91.11 (L21) | 89.93 (L21) | 67.43/65.46 | 84.58/82.77 |
| guiowl7b | OSWorld-G | 77.45 (L20) | 63.14 (L20) | 58.24/47.06 | 68.04/53.33 |
| guiowl7b | MMBench | 81.52 (L21) | 77.38 (L21) | 58.71/53.84 | 73.68/68.14 |
| guiowl7b | UI-Vision | 52.49 (L21) | 39.45 (L20) | 35.89/25.15 | 48.74/35.35 |
| guiowl1.5 | SS-Pro | **68.94** (L24) | **51.55** (L24) | 51.11/36.56 (L18) | 50.60/33.08 (L35) |
| guiowl1.5 | SS-v2 | 92.60 (L23) | 91.11 (L23) | 83.40/81.51 | 89.22/85.84 |
| guiowl1.5 | OSWorld-G | 81.57 (L22) | 61.76 (L26) | 69.61/50.20 | 72.35/49.22 |
| guiowl1.5 | MMBench | 83.39 (L23) | 78.35 (L24) | 70.81/65.00 | 76.57/68.25 |
| guiowl1.5 | UI-Vision | 47.37 (L22) | 31.69 (L23) | 36.56/23.25 | 40.17/24.73 |
| uivenus | SS-Pro | **66.54** (L23) | **51.23** (L23) | 50.28/34.54 (L18) | 48.39/30.17 (L35) |
| uivenus | SS-v2 | 94.57 (L26) | 92.60 (L26) | 85.44/83.48 | 87.10/83.95 |
| uivenus | OSWorld-G | 81.18 (L23) | 60.59 (L23) | 63.14/47.06 | 64.51/42.75 |
| uivenus | MMBench | 86.48 (L24) | 80.63 (L24) | 73.04/66.58 | 75.54/66.17 |
| uivenus | UI-Vision | 50.31 (L23) | 34.08 (L23) | 38.01/23.80 | 40.19/24.32 |

指标定义（`zwerge/eval/eval_retrofit.py`）：**hit@1** = top-1 预测点落入 GT bbox；**overlap@1** = 以预测点为中心按 cell 300×220（alpha 0.55 膨胀）构造的预测框与 GT 有交集（strict top-1，非 @k）。中间层峰、末层回落在所有 backbone×bench 上一致（如 guiowl7b SS-Pro：L21 58.95 → L27 48.20，-10.7pp）。

## 5. 审计已确认的坑（子 agent 必读）

1. SVD 初始化必须发生在 `setup_special_token_ids(reinit=True)` **之后**（否则被 Xavier 覆盖）；且仅在 fresh run（`_is_resuming=False`）、`independent_layers=True`、`adapter_type="attn"` 时允许，否则 hard error。
2. **两架构末层约定相反（2026-09-22 实测定论）**：Qwen2.5-VL(4.51.3) 最后一个 hidden state **已过 final norm**（probe L27 端点离群，max_abs=0 实证，`audits/hidden_contract_guiowl7b.json`）；Qwen3-VL(4.57.1) 最后一个 hidden state 是 **RAW block-35 输出，未过 final norm**（output_35 误差=0，vs final_norm rel_l2=30.6，`audits/hidden_contract_guiowl15.json`），即 Qwen3 各层约定一致、无端点离群，但末层 raw 表示带 norm 爆炸（probe 的 RMS pre-scale 在吃它）。跨层/跨架构比较时 Qwen2.5 的 L27 单独标注；`probe_utils.py:420-429` 的旧 lens 在 Qwen2.5 端点构成二次 norm、在 Qwen3 端点是单次 norm，别拿它当 ground truth。
3. `next` donor 对 probe L27/L35 不存在 block 28/36 → 显式 fallback 到 same 并写 report，禁止静默。
4. Qwen3-VL 原生 attention 有 per-head QK RMSNorm，weight-only 核是近似；report 已含 `source_has_qk_norm` 标记，解释 Qwen3 结果时必须带上这个 caveat。
5. 旧 recipe 训练 new-token embeddings（`unfreeze_new_tokens=True`）。主实验组保持旧 recipe 不动（保证与 §4 基线可比）；如加"固定表示"诊断组，所有 arm 统一 `--unfreeze_new_tokens false`。
6. 本轮禁用：Stage-2 fusion / margin gate / native fallback / P2P / zoom / ensemble / 任何 threshold tuning。评测只出逐层 `hit_top1`/`overlap_top1` + uniform-mean 参考行。
