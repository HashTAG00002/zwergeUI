# ZwerGe：W&B 最小训练动力学记录改造说明

**交付对象：workspace coding agent。目标：在不改变训练算法的前提下，让一次 loss 脉冲能够被定位到数据、probe、梯度更新或 A8 融合环节。**

审计基线：`9329b271709aabceb54ca16224f75733116a9ee0`；本地源码来自用户上传的 `ZwerGeUI_main_9329b27.zip`。审计日期：2026-09-22。在线固定提交的核心源码也已核对。

本次没有独立的 W&B run 导出、训练 checkpoint 或 GPU 运行日志，不能宣称已经找到了过去某次训练崩溃的根因。下文严格区分源码事实、建议的记录方案、需要真实环境确认的实现细节。

这是一份**改造规格，不是已经接入仓库并验证过的补丁**。附带的 9 项 CPU 测试复现了部分原代码行为并验证统计口径；没有执行真实模型训练、在线 W&B、GPU 或多进程 DDP 测试。

---

## 0. 执行范围：先修记录，再增加必要的观测

首轮工作分成三个小提交：

1. **P0：修复记录口径和写入路径。**修复 loss 分母、跨 rank 聚合、同 run 多 writer；把已经计算的逐层损失接到日志。
2. **P1：补充 probe 动力学与事件记录。**常规指标仅记录本文指定的少量标量；异常时输出有样本身份的 JSONL 记录。
3. **P2：小型固定诊断集。**采用独立开发集，定期执行无 AR 解码、无 PNG 渲染的直接读出诊断。

本次禁止顺手改变 loss 实现、无效样本处理、梯度累积、采样器、学习率、初始化、冻结策略或 gate。发现这些环节有问题，另建修复提交和对照实验。尤其不要为了让曲线好看而自动跳过 batch、`nan_to_num`、重置参数或调整学习率。

不新增常驻监控服务，不默认开启 `wandb.watch(model, log="all")`，不保存全量激活或参数快照，不逐 token 上传，不定期做参数 SVD，不常开 `autograd.detect_anomaly()`。

---

## 1. 源码已经支持的事实

以下行号均为上传快照中的源文件行号；浏览器渲染行号可能不同。

### 1.1 当前并非只有一条 loss，但缺少重要的层级信息

`zwerge/src/zwerge_retrofit/trainer.py:199–244, 859–962` 已记录：

- grounding loss、可选 LM loss、总 loss、全局 grad norm、一个 learning rate；
- A8 的平均 layer weights 和 omega entropy；
- 最终空间分布的 entropy 与最大概率。

但是：

- `modeling_base.py:77` 虽声明 `per_layer_losses`，当前 wrapper 没有将逐层 KL 填入该字段；
- `LayerWiseGroundingHead.forward():518–546` 已逐层计算 KL，只累加成均值；
- `CrossAttnGroundingProbe.forward():263–275` 已计算 Q、K、逐 head scores、最终 logits；
- 上层调用 `modeling_base.py:486` 用 `_` 丢掉了 logits；
- `_compute_grounding_loss():1028–1033` 仅向外传总 grounding loss、最终空间概率和 layer weights。

因此，补逐层 loss 和 logits 尺度不需要额外跑 backbone。

### 1.2 自定义 component loss 的日志分母错误

`RetrofitTrainer.compute_loss():874–883` 每个 microbatch 把 scalar loss 累加到 `_custom_metrics`。

`RetrofitTrainer.log():944–955` 只除以 `gradient_accumulation_steps`，没有除实际累计的 microbatch 数。

若梯度累积为 A，每 K 个 optimizer update 打印一次，窗口完整、每个 microbatch 的 grounding loss 恒为 c，则旧日志为：

    K × A × c / A = K × c

A7/A8 脚本设置 `logging_steps=20`。在上述条件下，自定义 grounding_loss 是真实 microbatch 均值的 20 倍。窗口因训练结束、额外 log 或重启变短时，这个放大因子可能变化。

这是**日志数值错误**，不是这里的 backward loss 被放大 20 倍。不能据此断定历史 HF `train/loss` 的每次脉冲都属于显示错误。

附带测试从原文件 AST 抽出 `compute_loss` / `log` 执行：恒定 loss=2，A=4，K=20，旧自定义日志得到 40；K=5 时得到 10。

### 1.3 自定义指标目前未显式做 DDP 全局 sum/count 聚合

`compute_loss()` 的 `_custom_metrics` 是各 rank 的本地状态。不能假设 HF 对原生 loss 的处理，会自动替自定义字段聚合。

不论最终谁写 W&B，这些字段在当前代码中都没有明确的“所有 rank、所有有效样本”统计语义。

### 1.4 无效 grounding 样本以零损失进入 batch mean

`modeling_base.py:962–1015` 的下列分支加入 `_zero_grounding_loss()`：

- 没有视觉 token；
- 没有标签；
- `[0]` 占位标签；
- 标签长度与视觉 token 数差异大于 10。

`1032` 对整个列表取 mean。标签长度差异不超过 10 的分支则插值修复后继续使用。

因此，有效样本损失完全不变时，仅有效比例改变，返回的 grounding loss 就可能改变。附带测试复现：“一个有效样本 loss=10，另一个占位无效样本”为返回 loss=5。

**本提交只记录此事实，不修改分母，也不把 graph-connected zero 删除。**

### 1.5 数据可能被替换，且身份未传到 Trainer

`dataset.py:670–682` 在失败后随机选择其他样本重试，最多 `min(10,len(dataset))` 次。

`dataset.py:770–781` 可能丢弃超长样本；返回数据不含稳定样本 ID / bbox / 原始记录定位。

`RetrofitDataCollator:946–969` 过滤 None，并可能截断序列、清除越界 anchor hint。

`get_patch_gaussian_label_from_bbox():244–249` 在权重总和过小时退回 binary label。

所以只保存 sampler 请求的 index，不足以知道实际训练了哪条数据。

### 1.6 当前训练概率有 floor，应该记录是否触及目标区域

`modeling_base.py:519–546` 使用：

    F.kl_div(torch.log(p.clamp(min=1e-8)), label_dist, reduction="sum")

这不是数值上与 `log_softmax(logits)` 在所有区间等价的实现。极小概率触及 clamp 时，对那些 probability 的直接导数会被截断。

一个单目标 toy example 中，logits=[0,-30]、目标为第二项，旧表达式得到有限 loss 但梯度为零；log_softmax 版本梯度非零。真实 Gaussian 标签还有其他项，因此不能推广成“只要有任何 floor，所有参数梯度都为零”。

只加监控；loss 数值实现是否替换另开实验。

### 1.7 记录路径存在重复 writer / 时间轴风险

- `report_to=wandb` 已启用 HF 自带 WandbCallback；
- 另外注册的 `WandbRetrofitCallback` 再次 `wandb.log(..., step=state.global_step)`；
- 同步验证 callback 也直接调用 `wandb.log`；
- `eval_daemon.py:270–305` 以训练 run ID `resume="must"`，用 checkpoint step 记录异步结果，再 `finish()`。

W&B 内部 history step 与 optimizer global_step 不是同一对象。训练已推进到 1000 时才得到 checkpoint-400 的评估，不应向 history step=400 回填。官方共享模式需要显式配置，不能把普通 resume 当成多进程写入协议。

`eval_daemon.py:299–301` 的“按名称恢复”注释也不可靠：name 是显示名，不能代替稳定 run ID。

---

## 2. 最小指标集合：五类逐层标量，少量全局标量，细节放事件记录

### 2.1 命名约定

本文表格中的 `diag/...`、`probe/...`、`data/...`、`optim/...`、`fusion/...` 是**传给 Trainer.log 的键**。

沿用已核对的 HF 4.57.1 默认集成时，W&B 会添加 `train/` 前缀，例如：

    Trainer.log:  probe/L18/kl_mean
    W&B:          train/probe/L18/kl_mean

不要再手工给这些键添加 `train/`，避免出现 `train/train/...`。

固定诊断集采用 `eval_diag/...` 作为 Trainer 输入键，按已核对的 `eval_` 重写规则映射成 W&B 的 `eval/diag/...`。必须在目标版本的离线测试中确认实际映射；不能硬编码假定所有版本一致。

### 2.2 核心全局量

| Trainer 输入键 | 定义 / 统计方式 | 用途 |
|---|---|---|
| `diag/loss_model_mean` | 每次原始 `outputs.loss` 的 SUM / 实际 forward 次数，先统计再跨 rank 合并；尚未做 GA 缩放 | 可对照旧 HF loss，但不要假定非均匀 batch 下与优化权重完全一致 |
| `diag/loss_micro_max` | 窗口内所有 rank、所有 microbatch 的原始 model loss 最大值 | 不让短脉冲被均值抹掉 |
| `diag/grounding_valid_mean` | 逐有效样本 grounding loss 的 SUM / 有效样本数 | 排除 zero-skip 对均值的稀释 |
| `diag/nonfinite_forward_count` | loss/logits 等指定边界出现非有限值的样本-层记录次数 | 不依赖 HF 对 NaN/Inf 的显示过滤 |
| `data/grounding_valid_frac` | 有效 grounding 样本数 / 实际送入 model 的样本数 | 解释有效监督量变化 |
| `data/repair_or_fallback_frac` | 发生标签插值、标签类型回退、数据重试替换或非预期 anchor 的样本比例，按样本去重 | 发现数据路径变化；具体原因放事件记录 |
| `data/uniform_ref_kl_mean` | 有效样本上的 `log(N_vis) - H(label_dist)` 的均值 | 区分输出空间/标签形状变化与模型退化 |
| `optim/grad_norm_preclip_max` | 本窗口 optimizer update 中，裁剪函数实际返回的全局 preclip norm 最大值 | 发现短时梯度脉冲 |
| `optim/clip_frac` | 触发全局 norm 裁剪的 update 数 / 实际有梯度的 update 数 | 判断是否长期被裁剪 |
| `optim/lr_head`, `optim/lr_new_tokens` | 实际 optimizer group 当前 LR，不读静态 config 替代；不存在的组不报 0 | 检查 schedule 与 resume |
| `optim/new_tokens_grad_rms_postclip` | 新 token 小参数的梯度 RMS（有这个 trainable group 时） | 检查影响所有 probe 的共享输入是否在异常更新 |

保留 HF 原有 loss / runtime / system metrics。修复旧 `grounding_loss` / `lm_loss` 的窗口均值口径后，尽量停止重复 alias；不要为了同一数值额外维护三条曲线。

`diag/loss_model_mean` 是 microbatch 平均的平均；`diag/grounding_valid_mean` 是有效样本平均，两者故意采用不同分母。精确分母写入本地窗口记录，让 agent 能重算。

当观测到非有限值时，不得把其当 0 加入平均。该窗口相应均值标记 unavailable/nonfinite（本地 JSON 使用 null 并给原因），同时记录 count；只有明确命名为 finite-only 的辅助量才可以忽略非有限项。不能无声地产生好看的有限均值。

### 2.3 每个 active probe 只保留五类常规量

以实际 backbone 层号 L18 为例，不使用 `probes` 的数组序号替代层号。

| Trainer 输入键 | 定义 | 解释 |
|---|---|---|
| `probe/L18/kl_mean` | 有效样本上的当前逐层 KL 均值 | 哪一层先偏离 |
| `probe/L18/logit_rms_centered` | 每个样本先算 `sqrt(mean((z-mean(z))^2))`，再按有效样本平均 | softmax 输入是否突然放大；不受整体平移影响 |
| `probe/L18/entropy_norm` | 每个样本 `H(p)/log(N_vis)`，再平均 | 接近 1 更分散，接近 0 更尖锐；不把尖锐等同正确 |
| `probe/L18/gt_floor_mass` | `sum_j label_dist[j] * 1[p[j] < 1e-8]`，再平均 | 有多少目标监督质量落入概率 floor 区域 |
| `probe/L18/grad_rms_postclip_max` | 每个 update 将 Wq/Wk 的梯度平方和除以它们参数元素数并开根号；窗口取最大值 | 哪个读出器的梯度异常，避免被最后一次采样漏掉 |

补充约定：

- `N_vis=1` 时 normalized entropy 定义为 0 并保留 N_vis；`N_vis=0` 不进入有效样本统计。
- `grad is None` 与数值为零的 gradient 区别保留在事件 / coverage 统计，不能默认为相同。
- 这里的梯度组只包含 Wq/Wk。发生异常时，在事件记录中补出 Wq、Wk、LayerNorm、head_gate 各自的状态；无需常规创建四套额外曲线。
- 对 A8 非 active/frozen probes，默认不记录训练梯度，也不把“冻结无梯度”当异常。固定诊断可仍检查全部保留层。
- KL、logits RMS、entropy 的窗口峰值和峰值所属样本保留在本地 ring/event 数据；W&B 无需为所有字段再复制 mean/max/min 三套。
- 五类指标乘以 14–18 层是合理的定位分辨率；不扩展到整个 backbone 每个 Linear/LayerNorm。

### 2.4 A8 才启用的最少附加量

| Trainer 输入键 | 定义 |
|---|---|
| `fusion/kl_mean` | 未乘任何全局权重的 `loss_fuse`，有效样本平均 |
| `fusion/aux_kl_mean` | 未乘 lambda 的 `loss_layer`，有效样本平均 |
| `fusion/omega/L18` 等 | 当前已有 layer weights，改成有效样本 sum/count |
| `fusion/omega_entropy_norm` | 每个样本先算 `H(omega)/log(M)` 再平均，不算均值 omega 的熵 |
| `fusion/tau` | 实际 `softplus(rho)+1e-4` |
| `fusion/grad_rms_postclip_max` | fusion 全部 trainable 参数的梯度 RMS，窗口最大 |

在代码中 tau 是 **乘在 cosine score 上的系数**，越大一般越强化分数差异，不要误读成“除以 tau”的温度。

A7 `independent_layers=True` 时 omega 固定均匀，p_final 只是多个 probe 的均值：不要用 A7 的 omega entropy 判断“融合塌缩”；默认不绘制融合页。

---

## 3. 为什么选这些量，而不是继续增加几十种曲线

### 3.1 输入与输出不同，不必把所有激活都画出来

正常 W&B 图用 logits RMS + entropy 足以观察输出尺度。为定位“输入就异常，还是 Wq/Wk 放大”，probe 在每次 forward 额外生成以下**分离计算图的边界摘要**，只放本地滚动缓冲：

    layer_idx
    h_query_raw_rms
    h_vis_raw_token_rms_mean / max
    Q_rms
    K_rms
    logits_centered_rms
    first_nonfinite_boundary
    KL, entropy_norm, gt_floor_mass

只检查实际 probed query / visual tokens；不检查整个 backbone 所有 token 和 module。优先复用现有归一化统计，不为保存摘要复制整张特征图。涉及 norm/square 的诊断归约用 FP32，不能先用低精度平方溢出再转 FP32。

若完整视觉归约确实形成瓶颈，可以固定选择有限视觉行作为**sampled** 激活摘要，名称和记录中必须注明 sampled；此时“未检测到异常”不能解释为全部 token 无异常。KL、最终 logits、最终概率和梯度的核心记录不能随意抽掉。

不默认对参数求谱、不监控 Adam 全部 moment、不计算 Hessian、不保存 attention 图。

### 3.2 logits 比 max probability 更适合观察 SVD 起点

Wq/Wk 初始化改变时，logit RMS 和初始 entropy 能显示是否一开始就过尖。用 step-0 的固定诊断记录初始化状态；正式训练前执行、关闭 dropout。

同时只记录一次 initializer、donor layer、rank、gain、seed 和初始 head 配置到 manifest。不要在每次 log 重新做 SVD。

### 3.3 目标分布的统计不可省

均匀预测 u_j=1/N 时：

    KL(y || u) = log(N) - H(y)

因此，loss 的变化可能来自 N 或 y 的形状变化。Gaussian label 也不等于 binary bbox mask，不能拿“y>0 区域”直接当点击成功区域。

事件记录同时保留 N_vis、原始/修复后 label_len、label_sum、target_entropy、bbox 和处理后尺寸。标签质量最终仍需要检查原图/标注；任何标量组合都不能自动证明标签正确或错误。

---

## 4. 采集频率与事件记录

### 4.1 默认频率

- 每个 training microbatch：采集标量、sum/count/max 和样本身份；不直接调用 W&B。
- 每个 optimizer update：采集全局裁剪结果和 trainable 小模块 gradient RMS；只对实际参与更新的模块做。
- 每 20 个 optimizer update：统一 all-reduce 窗口统计并由 global rank 0 写一次常规日志；沿用现有 logging cadence。
- step 0、每 100 个 optimizer update：32 条固定 dev 样本的轻量诊断；计算开销较高时改 200，记录实际配置。
- 异常发生时：立刻把当前 rank 的事件与已有上下文写本地 JSONL；无需等待下一次 W&B 上传。
- 普通 W&B 上传同时上报本窗口 event count。不要为每个样本或每层调用 `wandb.log()`。

**采集频率与上传频率是两件事。**上传仍可稀疏，但窗口内部峰值和事件不能丢。

### 4.2 小型事件记录，不是全量 dump

每个 rank 维护最近 4 个 optimizer-update 窗口的 scalar/metadata ring。异常触发时保存此前上下文、当前帧，并继续收集后 2 个 update。所有对象 detached，不保留计算图、hidden-state 张量、截图或 gradient 全量数组。

建议文件：

    output_dir/diagnostics/manifest.json
    output_dir/diagnostics/windows.segment-<id>.jsonl
    output_dir/diagnostics/events.segment-<id>.rank-<global_rank>.jsonl

只有 windows 由全局主进程写。events 按 rank 分文件，避免多进程争写同一文件。DataLoader worker 的加载错误保留带 worker ID 的结构化错误记录，或通过返回 metadata 传回；不要让 worker 登录 W&B。

训练机已有原图时只保存稳定定位；默认不上传截图或完整任务文本。排查时 agent 按 ID 从原数据读取。

### 4.3 触发器保持简单

只启用：

1. loss、probe 指定边界或 gradient 出现非有限值；
2. grounding 样本被 skip、label repair、Gaussian fallback、anchor fallback 或数据替换；
3. 单 microbatch model loss 相对最近正常历史显著升高。

第 3 项可先用一个可配置起点：累计至少 100 个本地 finite microbatch 后，当前 loss 大于 `max(3 * 历史中位数, 历史中位数 + 1.0)`。阈值不是理论常数，只是决定保存事件，不用于改变训练。

重复数据问题可以按原因计数并每个上传窗口保留前 3 个代表事件；非有限值保存首次完整上下文和后续计数，避免日志风暴。不要因为限流而丢失计数。

某个 rank 触发事件，只进行本地操作；不在条件分支里插入 collective，避免 DDP 死锁。全局合并仅在固定的所有 rank 都进入的 update/log 边界进行。

### 4.4 事件 schema

下例仅示意字段，不是本次真实运行记录：

```json
{
  "schema_version": 1,
  "segment_id": "<restart segment>",
  "phase": "train",
  "optimizer_step_before_update": 417,
  "micro_index_in_update": 2,
  "global_rank": 3,
  "trigger": "loss_spike",
  "requested_sample_id": "<source record originally requested>",
  "actual_sample_id": "<record actually consumed after retry>",
  "source_record": {"dataset_id": "<manifest hash>", "row": 123, "child": 0},
  "sample": {
    "image_ref": "<relative path or stable identifier>",
    "bbox_norm": [0.1, 0.2, 0.3, 0.4],
    "processed_width": 1280,
    "processed_height": 768,
    "seq_len_before_truncation": 1200,
    "seq_len_after_truncation": 1200,
    "n_vis": 960,
    "label_len_before_repair": 960,
    "label_len_after_repair": 960,
    "target_entropy": 1.7,
    "anchor_strategy": "P1:explicit_ground_token",
    "anchor_index": 1195,
    "anchor_after_visual_end": true,
    "retry_count": 0,
    "label_fallback": null
  },
  "suspect_layer": 24,
  "layer_summaries": "<small scalar map; no tensors>",
  "optimizer": "<actual LR and pre/postclip gradient summaries>"
}
```

不要只保存 shuffle 后的整数 index。应在载入/展开源数据时保留源数据版本、原始行号和元素 child index；或使用规范化字段的稳定 SHA-256。不要使用会跨进程/会话变化的 Python 内置 `hash()`。

`_monitor` metadata 在 collator 中保留，在 `RetrofitTrainer.compute_loss` 进入 `model(**inputs)` 之前 pop。不得传入模型 forward kwargs，也不能让 remove_unused_columns 提前删掉。明确测试该数据路径。

---

## 5. 正确聚合：不能平均 rank 的平均值，也不能在 on_log 才观察梯度

### 5.1 forward 统计

用 detached device scalar / 小向量积累：

    loss_sum, forward_count
    valid_loss_sum, valid_sample_count
    model_sample_count
    layer_kl_sum[layer], layer_valid_count[layer]
    micro_loss_max
    nonfinite_count
    reason_counts

同一个指标必须有明确分母。跨 rank 用 SUM 聚合分子/计数、用 MAX 聚合峰值，再计算最终比值。

不要每层 `.item()` / `.cpu()` 一次。批量合并后再搬运。不要把整个 batch 已经做过平均的 entropy 再当成等样本权重的平均（None 过滤可能使 batch 大小不同）。

不需要更改当前 backward 的模型损失分母；监控可以同时报告“模型使用的 loss”和“有效样本诊断 loss”。

测试例：rank 0 有 2 条样本均值 10，rank 1 有 8 条均值 1。有效样本全局均值为 2.8，不是 5.5。

### 5.2 梯度的观察时机

优先对当前脚本的 torchrun/DDP + bf16 路径实现和验证；脚本未传 DeepSpeed 配置，不需要为了本次首版强做全后端支持。但必须在 manifest 确认实际 backend。

需要区分：

    backward 完成
    → DDP 梯度归约完成
    → 如有 AMP scaler，unscale
    → [global preclip norm]
    → gradient clipping
    → [per-probe grad_rms_postclip]
    → optimizer.step
    → zero_grad
    → on_log

已核对的 HF callback 文档中，`on_pre_optimizer_step` 位于**裁剪之后**；不能在这里得到 gradient 后把它命名为 preclip。

推荐保留现有 Trainer clipping 行为，用极小的观测适配器取得实际 clipping 函数返回的 global preclip norm；随后从同一次已裁剪、已归约的 `.grad` 读取分组 postclip RMS。不得重复调用 clipping，不得重复 unscale，不得修改 optimizer.step。

HF `on_log` 的 `grad_norm` 常常只对应某个 update，不能代表整个 logging window 的最大值。缓存每次实际返回值，窗口 max 才能留住脉冲。

DDP 归约后的梯度在各 rank 复制存在；不要再把这些 norm 平方跨 rank SUM，导致多乘 world_size。forward 数据统计与 DDP 已同步梯度统计不是同一种聚合。

如果未来使用 ZeRO/FSDP，普通 param.grad 可能不是完整梯度。没有实现 backend-aware observer 时，将该指标标为 unavailable 并说明 backend，不得输出伪 0。FP16 scaler、clip disabled 路径同样需要单独确认 unscale 时机。不要为兼容它们重写整段 Trainer 训练循环。

### 5.3 基准测试覆盖“观察不改变训练”

在相同种子、相同 batch 顺序下比较 monitor on/off 的前向、梯度和参数更新。额外 fixed-eval 也应保存并恢复模型模式、processor 设置与 RNG 状态，不消耗训练 DataLoader，不改变其采样顺序。

固定输入可以缓存 tokenization / 图像预处理；**不要跨训练阶段缓存 backbone hidden states**来当诊断结果，因为当前新 token embeddings 可能仍然可训练。

---

## 6. W&B writer 统一与断点恢复

### 6.1 训练只有一个 writer

保留 HF 默认 WandbCallback。将 `WandbRetrofitCallback` 的功能移入 Trainer.log 的字段整理，停止它独立 `wandb.log`。

在调用 `super().log(...)` 前：

1. 合并所有 rank 的窗口统计；
2. 展开 layer weights 为 scalar keys；
3. 写本地窗口 JSONL；
4. 交给默认 HF integration 进行一次 W&B 写入。

global rank 0 负责远端写入，不使用 `local_rank == 0` 代替多机全局主进程。all-reduce 仍然必须所有 rank 参加。验证多机时每台的 local_rank=0 不会重复创建 writer。

### 6.2 异步评估不要 resume 活跃训练 run

默认采用简单方案：

- 训练 run：`job_type=train`；
- 独立评估 run：`job_type=eval`，以训练 run ID 作为 group 或显式 config 关联；
- evaluator 使用自己的稳定 run ID，不继承训练 `WANDB_RUN_ID`；
- 用 `eval/checkpoint_step` 作评估横轴，不把 checkpoint step 传入 W&B 内部 `step=...`；
- `finish()` 只结束 evaluator 自己的 run。

评估结果先到 800 再到 400也允许 append；自定义横轴携带 800/400，内部 history step 始终前进。agent 读取时按 checkpoint step 整理，不凭写入顺序推断训练先后。

这不新增服务：复用已有 eval_daemon 和已落盘的 summary JSON。若以后强要求训练评估同 run，改为同一个 writer 读取已完成结果并 append，不恢复当前多个普通 writer 的方案。

### 6.3 每次恢复写一个 segment 标记

manifest / segment event 至少包含：

    schema_version
    repo commit + dirty status / modified file hashes
    torch / transformers / accelerate / wandb versions
    backbone checkpoint identity
    probe_layers / active layers / independent_layers
    init method / SVD donor map / rank / gain / seed
    gradient accumulation / per-device batch / world_size
    dtype / actual backend / clip threshold
    head and new-token LR group identity
    dataset/preprocessor identity
    resume checkpoint / restored optimizer step
    segment_id / prior run ID

W&B 继续同 run 时，global_step 可能回滚到最新保存的 checkpoint。内部 history 仍 append，用自定义 optimizer step 和 segment_id 表示轨迹；不能“伪造补齐丢掉的训练步”。修复统计口径的第一次运行要使用新 schema/run 或显式新 segment，不能把修正前后的 loss 数值当成连续同口径曲线。

---

## 7. 固定小诊断集：区分“这批数据更难”与“同一模型真的变差”

选 32 条**独立开发数据**，固定 sample IDs 和预处理。应覆盖若干大小目标、分辨率和数据来源，但它不是新 benchmark，也不报告正式测试集结论。

step 0 和每 100 update：

- 一次直接读出，不生成 native AR 坐标，不走 gate，不渲染 PNG；
- 计算各层 KL，A8 则另算 fusion KL；
- 使用实际部署的 posterior-to-point decoder 与 bbox，报告 strict point-in-box；
- 明确 direct branch 指标，不用原生 fallback 掩盖 head 变化；
- 每条数据的处理策略与 decoder 配置写入 manifest；
- fixed-eval 的 forward 不得混入训练 accumulator，不得清空训练窗口。

W&B 最少量：

    eval/diag/grounding_valid_mean
    eval/diag/point_in_bbox
    eval/diag/probe/Lxx/kl_mean
    eval/diag/valid_count

32 条样本的命中率有统计波动，不能用单次涨跌判定泛化，也不能证明全部故障来源。但它为“同一批输入上的模型行为变化”提供必要参照。

已有同步 ValEvalCallback 会做完整 benchmark / PNG，不直接把它改成每 100 步调用。新增轻量路径或复用其中的纯 decoder/metric 函数即可。避免改动训练 processor.max_pixels 后未恢复。

---

## 8. dashboard 与 agent 的阅读顺序

只安排六个面板组：

1. **总体：**HF loss、校正的 valid loss、micro max；显示 unsmoothed 原始 max。
2. **层：**逐层 KL，同一图比较或 layer×step 热力图。
3. **分布：**逐层 logits RMS、normalized entropy、GT floor mass。
4. **更新：**global preclip max、clip fraction、逐层 postclip gradient、新 token gradient 与两组 LR。
5. **数据：**有效比例、repair/fallback 比例、uniform reference KL。
6. **A8 / fixed dev：**A8 fusion 特有量和固定诊断集；A7 不展示虚假的 fusion 动态。

agent 初步诊断规则（都是待证实的解释，不是自动定罪）：

| 观察组合 | 优先检查 |
|---|---|
| loss_micro_max 单点升高，固定 dev 稳定，uniform_ref_kl 或 repair 比例同时变 | batch 组成、分辨率、标签、重试替换 |
| model loss 下降，但 valid loss 不变、valid fraction 下降 | zero-skip 稀释，不是模型变好 |
| 某层 KL 与 logits RMS 突升、entropy 下降、GT floor mass 增加 | 该层读出过尖/目标低概率被 floor；检查前一更新和 Q/K 摘要 |
| 多层同时变化，新 token grad 异常，当前输入 h_query 摘要也改变 | 共享输入、token embeddings、数据与预处理；不能直接怪所有独立 head |
| probe KL 都稳定，仅 fusion KL 变坏，omega entropy 很低且 tau 变大 | A8 权重集中、fusion 参数更新；集中也可能是正确选择，需看 fixed dev |
| global grad norm 很大，clip fraction 长期很高 | 梯度尺度、样本与 LR；裁剪常触发本身不证明发散 |
| 训练指标改善，但固定 dev 持续恶化 | 泛化/过拟合或诊断分布问题；先核实 dev 独立性 |
| 曲线在 checkpoint/restart 附近变形，数据/层级量无对应事件 | 日志窗口、writer、时间轴、resume 和状态恢复 |

以 update t 计：当前 forward 的异常会产生当前 gradient；其参数更新主要影响之后的 forward。不要把同 step 梯度峰值直接当作该步 loss 峰值的先行原因。事件中显式标明 optimizer_step_before_update。

允许最终结论为“记录足以定位到模块，但还不能证明数据错误/具体根因”；再重放少量已定位样本，而不是重训整个实验。

---

## 9. 源码改动清单

| 文件 / 位置 | 限定改动 |
|---|---|
| `modeling_base.py:199–275` | probe boundary scalar summaries；保留 detached logits 统计；不改 Q/K、softmax 与 loss |
| `modeling_base.py:464–548` | 暴露已经计算的逐层 KL；传出 fuse/aux 原始分量；区分 active/frozen；不增加一遍 head forward |
| `modeling_base.py:927–1033` | 统计 valid / skip / repair 原因、sample loss 与身份；保留原模型 loss 行为 |
| `BaseRetrofitOutput` + 各 model wrapper | 传递一个有文档的轻量 diagnostics payload；不要让既有 4-tuple 返回值调用方被悄悄破坏 |
| `dataset.py:202–249, 670–785` | 传播 stable ID、实际重试样本、label fallback、bbox 与实际处理尺寸 |
| `dataset.py:946–1023` | 保留 `_monitor` metadata；统计 None、truncation；不送 metadata 给 backbone |
| `trainer.py:859–962` | sum/count/max accumulator；排除 eval；pop metadata；DDP 归并；统一 writer |
| `trainer.py:199–244` | 取消第二次独立 W&B 写入；scalar flatten 放到 super.log 之前 |
| optimizer boundary adapter | 观测实际 preclip 返回值和 postclip grads，不再在 on_log 查 `.grad` |
| `train_retrofit.py:838–878` | 接入 observer / 轻量 dev callback；记录 manifest；避免 callback 重复 |
| `eval_daemon.py:270–305` | 独立 evaluator run，checkpoint 自定义轴，自己持久化 run ID |
| A7/A8 shell | 可配置监控频率；去掉源码内的 API key，改环境注入 |

实现时统一一个 `TrainingDiagnostics` 数据契约，不要在五个 backbone wrapper 里各自发明一套命名。没有 supervision 的 inference 路径允许 diagnostics=None。

---

## 10. workspace agent 必须完成的验收

首轮验收不需要完整 200K 训练。

1. **loss 窗口：**constant loss 在 logging_steps=1/5/20、GA=1/4 及末尾短窗口下均值相同；旧错误有回归测试。
2. **变长 / 无效样本：**model loss 维持原行为，valid-only loss 和各类 count 可重算。
3. **DDP：**至少 2 个真实进程，非主 rank 插入异常/无效样本，global max/count 正确，只有一个 W&B writer，无条件 collective 死锁。
4. **梯度时序：**构造 preclip norm=10、阈值1 的例子，正确得到 preclip=10、postclip≈1，zero_grad 后不误报0；未训练的参数不当作缺梯度错误。
5. **layer mapping：**probe slot 0 对应 L14/L18 时命名正确；A8 inactive 层不误判无梯度。
6. **非有限值 / floor：**注入 NaN 或过尖 logits，有正确边界事件；不静默替换0，不依赖 `train/loss` 的过滤。
7. **无侵入：**monitor on/off 同种子同 batch 的 loss、grad、update 相同或在已说明的后端确定性容差内；不能只是“程序没报错”。
8. **GPU 内存：**运行几十步后无持续增长；diagnostic buffers 不持有计算图。
9. **W&B 离线：**每 logging window 一个 training log event；键无重复前缀；内部 history step 与 optimizer step 分开。
10. **乱序评估与恢复：**checkpoint-800 后到 checkpoint-400 都可读；独立 evaluator finish 不影响 training；恢复写 segment marker。
11. **样本身份：**强制重试替换后记录 actual 而非仅 requested ID；None / 超长分支有结构化原因。
12. **开销：**同一短运行比较 median step time 和峰值显存，记录实测开销；建议目标≤3%，不是既定事实或硬保证。超预算先减少非必要激活归约/固定 dev 频率，不删除 loss max / valid count / nonfinite 和样本身份。

提交给用户的总结应包括：改了哪些文件；每项指标单位/分母/窗口/同步语义；CPU、DDP、GPU、W&B 各测试是否实际执行；一例合成事件；开销。未测项要明确列出。

---

## 11. 单独处理的一项凭证问题

快照中的 A7/A8 shell 与 `eval_daemon.py` 包含硬编码 W&B API credential。本说明与附件不复制凭证，也没有用其访问任何账户。

应在服务端撤销该凭证，再用环境变量/密钥管理注入新凭证。仅从当前源码删掉字符串不能撤销已经暴露在历史中的值。修复时不要把新凭证、完整环境变量或包含密钥的 diff 上传到 W&B/code artifacts。

---

## 12. 本次实际验证与资料

附带 `audit_repro_tests.py`，9 项 CPU 测试通过。包含从原源码抽出的 component-loss log 与 zero-skip 行为，以及 uniform KL、entropy normalization、probability floor、无侵入 detached probe 统计等小型测试。

运行方式：

    python audit_repro_tests.py --repo-root /path/to/zwergeUI

它不是生产监控实现，也不能替代上一节的实际 DDP/GPU/W&B 验收。

本地源文件 SHA-256 和审计版本在 `audit_manifest.json` 中。用户提供的是 Git commit ID；manifest 中的 SHA-256 是为审计文件内容另算的哈希，两者不要混淆。

已核对的一手参考（API 接入以用户实际安装版本为准）：

```text
Pinned repository:
https://github.com/HashTAG00002/zwergeUI/tree/9329b271709aabceb54ca16224f75733116a9ee0

HF 4.57.1 Trainer:
https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/trainer

HF 4.57.1 W&B integration implementation:
https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/src/transformers/integrations/integration_utils.py

HF callback event order:
https://huggingface.co/docs/transformers/main/en/trainer_callbacks

W&B logging:
https://docs.wandb.ai/models/track/log

W&B custom x-axis:
https://docs.wandb.ai/models/track/log/customize-logging-axes

W&B distributed/shared logging:
https://docs.wandb.ai/models/track/log/distributed-training
```
