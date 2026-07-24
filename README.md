# ZwerGe-UI 项目记忆存档（Revision 阶段核心 README）

> 本文件是全项目的唯一入口记忆。写作目的：ACL ARR 被拒后（原从 EMNLP 转投，AAAI2027 提交号 Submission2403），
> 基于三位审稿人意见做大规模补充实验与论文重写。以后新开对话，**先看这一份**，能想起所有关键决策、
> 待办实验、代码风险点。详细工程细节仍在 `.mrules`，本文件只做"接口层"索引 + 决策摘要，不重复贴长代码。

---

## 0. 一句话现状

论文核心 claim（**Coordinate Serialization Bottleneck**：GUI grounding 在中间层已形成，最后几层更偏向把
空间意图"序列化"成坐标 token）已经被论文正文采纳并写成了 4 个 Finding + 完整方法+实验章节（`docs/our_paper_tex/`）。
三位审稿人评分为 **2 / 3 / 3.5**（10 分制大致对应"拒稿重投 / Findings 弱接收 / 边缘会议"），
核心问题不是"idea 不行"，而是 **证据链不够硬、部分数字与文字 claim 不一致、部分技术描述有逻辑漏洞**。
本轮修订目标：**不换核心叙事，把审稿人指出的每一个"证据不够"补成"证据够硬"**。

---

## 1. 论文当前结构（已实际写好，采纳了 oracle 写作建议）

路径：`docs/our_paper_tex/secs/`

```
1_intro.tex       — Coordinate Serialization Bottleneck 提出，4条贡献
2_related.tex     — GUI Agent Grounding / Visual Transformers Grounding 两小节（偏短）
3_analysis.tex    — 核心分析章节，4个 Finding（框在 tcolorbox 里）：
                    §3.1 Finding1: 中间层 > 最后层（spatial lens）
                    §3.2 Finding2: spatial peak 早于 coordinate serialization plateau（双 lens 错位）
                    §3.3 Finding3: instruction-switch 反事实实验，排除 saliency 解释
                    §3.4 Finding4: collapse gap 在高分辨率小目标（SS-Pro）上比 SS-v2 更大
4_method.tex      — ZwerGe-UI 方法：Stage1 独立 CrossAttn probe，Stage2 Cross-Layer Fusion
5_experiment.tex  — 4个backbone × 5个benchmark 主表 + Ablation(A0-A5)
6_conclusion.tex  — 结论
a_appendix.tex    — Limitations + 逐层曲线（A7 CrossAttn probe 全曲线）+ prompt 附录 + 可视化 case
```

评测的 4 个模型：GUI-Owl-7B / UI-TARS-1.5-7B（Qwen2.5-VL, 28层）、GUI-Owl-1.5-8B / UI-Venus-1.5-8B（Qwen3-VL, 36层）。
5 个 benchmark：ScreenSpot-Pro（主战场，高分辨率小目标）、ScreenSpot-v2（已饱和）、OSWorld-G、MMBench-GUI、UI-Vision。
指标统一用 **overlap@1**（patch 级别，对 baseline 和 ZwerGe 都公平）。

**当前主结果的真实模式（写论文时必须诚实面对，不能回避）**：
- Qwen2.5-VL 系（GUI-Owl-7B, UI-TARS-1.5-7B）：普遍涨点，GUI-Owl-7B 在 SS-Pro +7.97pp，UI-Vision Spatial +25.95pp。
- Qwen3-VL 系（GUI-Owl-1.5-8B, UI-Venus-1.5-8B）：**SS-Pro 上明显掉点**（GUI-Owl-1.5-8B: 74.83→68.69，−6.14pp），
  SS-v2/MMBench 部分掉点，只有 OSWorld-G / UI-Vision 仍然涨。
- 这个"强弱backbone分化"模式本身就是论文的一个论点（"backbone 自身序列化能力越强，retrofit 收益越小"），
  但目前只是"事后解释"，**没有独立证据支撑**，这正是 4KeN 的核心批评之一。

---

## 2. 三位审稿人核心意见（决定这一轮要做什么）

原文见 `docs/reviews/Reviewer-{eKpD,4KeN,4sok}`。整理成可执行任务，按优先级：

### P0（不解决论文过不了）

1. **【eKpD 致命项】Stage 2 loss 存在逻辑漏洞**
   论文 `4_method.tex` 第129行写"auxiliary per-layer KL term prevents fusion from collapsing by keeping
   probe posteriors individually calibrated"。但 Stage 2 backbone 和 probe 都冻结，如果 per-layer posterior
   `p_l` 不是 fusion 参数的函数，这一项对 fusion 参数就是常数，起不到防 collapse 的作用。
   **必须做**：明确回答"Stage 2 梯度到底流到哪里"，即回顾 `.mrules` [2026-05-25] A8 章节的梯度路径分析——
   结论应该是：`loss_layer` 的梯度流向 **active probe 的 W_q/W_k**（不是 frozen 的，A8 里 10/12 层是继续训练的，
   不是完全冻结！），`loss_fuse` 也通过 `p_l`（非 detach 的）反传到 probe。**当前论文写"probe frozen"是不准确的**，
   需要在方法section明确写清楚 Stage2 到底哪些层参数继续训练、哪些冻结、auxiliary loss 具体作用于谁。
   如果实际实现里 active probe 确实继续训练，这个批评可以通过"更正描述"解决，不需要重新做实验。

2. **【eKpD+4KeN 共同致命项】"bottleneck"只是 probe-capacity artifact 的怀疑**
   审稿人怀疑：42M 参数的 full-rank cross-attn probe 在 200k 样本上训练，本身就可能学出"哪层都行"的强 readout，
   中间层更好可能只是这个 probe 架构在中间层学得更好，不代表 backbone 天生如此。
   **必须补的 controls**（这是本轮最大的工作量）：
   - 低容量 linear probe（Level-1 Rank-1 probe，`docs/oracle/report` §6.4 已有设计：2×d×n_layers ≈ 43K 参数）
   - 跨层共享 probe（同一个 probe 权重跑所有层，只有输入不同）
   - random-label / shuffled-label control（probe 在乱标签上应该学不出同样的中间层优势）
   - held-out OOD UI layout（不同分布的界面上验证中间层优势是否保持）
   - probe-capacity sweep（Level 0 无参数 cosine → Level 1 rank-1 → Level 2 当前 CrossAttn，做完整对比表）
   - 与 attention-only / saliency baseline 直接比较

3. **【eKpD】Serialization lens 需要更多 control（logit-lens 混淆问题）**
   用最终 LM head 解码中间层 hidden state 天然会让"越深越像输出空间"，这本身可能和 coordinate serialization
   无关，只是 logit lens 的通病（表征需要"对齐"到输出 embedding 空间才能被最终层解码，这是已知现象，非本文独有）。
   **必须补**：non-coordinate action token 对照、random coordinate 对照、text-only token 对照、
   **tuned lens**（每层单独训一个仿射变换再解码，而不是复用最终 LM head，参考 arXiv:2303.08112 Tuned Lens）、
   或者做因果干预实验（把中间层 spatial peak 路由到 decoder 是否真的提升坐标预测）。
   代码位置：`docs/our_paper_tex` 引用的 serialization lens 实现应在 `zwerge/probe/` 下（`exp3_serialization_lens.py`
   看名字很可能就是这个，需要检查其中 LM head 用法是否有上述 confound，并补 tuned lens 变体）。

4. **【eKpD】训练数据 200k 样本来源、去重、benchmark 泄漏完全未描述**
   论文只写"200k grounding samples for 1 epoch"，没有列数据集名称、许可、如何去重、是否与测试集视觉相似。
   **必须补**：完整数据来源表（GroundCUA 110k / OS-Atlas 48k / AgentNet 42k，来自 `.mrules` A7 训练记录），
   写明是否检查过与 5 个 benchmark 的图像重复（可用 perceptual hash / CLIP embedding 相似度扫一遍）。

5. **【eKpD】评测指标可能对 patch-level 方法有利，需要补标准 point-in-box 精度**
   当前 `overlap@1` 对 baseline 和 ZwerGe 都用 patch 中心/patch 相交判定，论文自己也承认这对 baseline 更宽松
   （`5_experiment.tex` 第126行"more forgiving for baselines"）。但审稿人要的是**标准点击成功率**（原始像素级
   hit@1，不做 patch 宽容化）。**必须补**：标准 hit@1（pixel-level，非 patch 宽容）作为并列指标，
   这在 `.mrules` 里已经有过很多讨论（gap = overlap@1 - hit@1，机制是 near_miss patch 量化误差），
   只是论文正文目前只报了 overlap@1，需要把 hit@1 也放进主表或至少 appendix。

6. **【eKpD 数值错误，最容易修】Appendix fusion claim 与图不符**
   `a_appendix.tex` 第39-41行明确写"Fusion consistently outperforms the best single layer... dotted line
   lies above the peak... in nearly all panels"，但 `5_experiment.tex` 主表 Table ablation 里 A4 fusion
   结果 48.7 是否真的高于所有单层峰值需要重新核对（`.mrules` [2026-05-20] 记录显示：A4@ckpt1600 SS-Pro 上
   fusion 39.97% 只比最优单层 L20/L21 (39.9%/40.16%) 高 0.06-0.81pp，且部分 bench 如 OSWorld-G fusion
   反而比最优单层差 1.4pp）。**这是一个可以立刻验证的检查项**：拉出 A7 appendix 逐层曲线图数据，
   逐个 panel 核实 fusion dotted line 是否真的在所有 peak 之上，不是则改文字为"fusion matches or
   moderately exceeds the best single layer in most but not all settings"，并补一张 Table（best single
   layer / avg active layers / Stage1-only / Stage2 fusion 四列对比，这是 eKpD 明确要的表）。

### P1（显著提升接收概率）

7. **【eKpD】样本量与不确定性报告缺失**
   Figure 1 用 n=200，instruction-switch 用 n=150 pairs，报告的 gap 常常只有 6-11pp，量级上可能落在
   噪声范围。**必须补**：bootstrap CI（对 hit@1、coordinate log-likelihood、suppression、JS divergence、
   collapse gap 都要），以及至少 3 个随机种子训练 probe/fusion 看方差。

8. **【eKpD+4KeN】需要直接对比其他"加同等容量监督参数"的基线**
   不能只跟 base model 比，要跟同样数据、同样参数量训练的：final-layer probe、简单 MLP/linear probe、
   LoRA adapter、supervised coordinate head 比。这直接对应 P0 第2条的 probe-capacity sweep，可以合并做。

9. **【eKpD】Related Work 中 GUI-Actor / Re-Prefill 的实验对比不够**
   目前 `2_related.tex` 只有两个小节且偏短。GUI-Actor 是最直接的 coordinate-free 前驱（也是 frozen backbone
   + attention-based head），Re-Prefill 是最像的并发工作。**必须补**：至少一张定量对比表（同 backbone 下
   GUI-Actor vs ZwerGe-UI 的 SS-Pro 数字），Re-Prefill 由于方法不同（training-free, attention-based,
   FA2不兼容）不好直接跑数字对比，但要在文字上把区别讲透（见下面 §4 Re-Prefill 专节）。

10. **【4sok】CosMeta fusion 的"异构语义空间"疑问**
    审稿人指出 §3.1 承认不同层语义空间异构（heterogeneous），但又用单一全局 `q_meta` 去跟所有层的 `q_l`
    算 cosine，逻辑上有张力。回答方向：`q_l` 不是原始 hidden state，而是经过 per-layer 独立 `W_q^{(l)}`
    投影后的**统一维度**表征（`d=512` for CrossAttn probe），这个投影本身就是在学习"把异构层空间映射到
    一个可比较的子空间"，`q_meta` 是在这个共享子空间里的原型，不是在原始 hidden state 空间。需要在论文里
    补一句解释这个设计动机，必要时加一个小实验：可视化不同层的 `q_l` 在这个共享空间里的分布，看是否真的
    存在可比较的几何结构。

11. **【4sok】小目标下 Gaussian label 方差趋零、退化成 hard one-hot 的问题**
    `σ = η·w_b`（η=0.35），当 bbox 很小时 σ 也很小，接近退化成 one-hot，soft label 的意义消失。
    **需要**：报告 σ 的实际分布（尤其 SS-Pro 上小图标的 σ 统计），并讨论是否需要设置 `σ_min` 下限。
    这个问题在 `.mrules` 里目前没有被讨论过，是一个新发现的点，值得做一个快速 ablation：固定最小 σ
    （比如至少覆盖 1 个 patch）vs 当前的纯比例方案。

12. **【4sok】Prefill-forced inference 的 OOD 问题**
    backbone 原生训练是在 CoT/reasoning token 之后才生成 `<|ground|><|pointer_start|>`，而 ZwerGe 直接
    prefill 这个模板、跳过 CoT，这构成一种分布外输入。`.mrules` [2026-05-18] 已经讨论过这个问题并给出了
    "方案A: Rule-based CoT + 强制 prefill grounding token" 的集成设计，但**从未做过消融实验**验证"有无CoT
    prefill 对 grounding 质量的影响"。这正好和 oracle 报告里"实验4: pre-decoding vs post-decoding anchor
    probe"的方案C（no-think/direct-action）与方案B（post-think pre-action）重合，是本轮该补的实验。

13. **【4sok】训练用连续 KL 散度目标，推理却用离散 BFS centroid 解码，两者是否对齐**
    这个问题目前没有被正面回答过。方向：可以证明 centroid 解码在小 σ 高斯分布下渐近等价于 argmax（连续），
    或者做实验直接对比"训练 loss 下降"和"BFS 解码后 hit@1 上升"两条曲线是否单调一致。

### P2（值得做但优先级低于以上）

- 4KeN 提到的 patch size / box-to-patch mapping 需要在论文里明确写清楚（当前只在 `.mrules` 里有：
  uitars/guiowl7b patch≈28px，guiowl/uivenus patch≈32px）。
- naming 统一："ZWERGE-UI" / "ZwerGe" / "ZwerGe-UI" 三种写法，论文里必须统一成 `\textsc{ZwerGe-UI}`（LaTeX
  宏已经在用，检查全文有无手打的不一致写法）。
- Equation 5（fusion score）需要显式定义符号，不能只在文字里描述。
- 需要补 compute/inference overhead 报告：hooked layer 数量、显存开销、相对自回归解码的延迟。

---

## 3. 与 Re-Prefill（arXiv:2605.12549）的区分——审稿人反复提到，必须讲透

Re-Prefill 论文标题是 *"What Happens Before Decoding? Prefill Determines GUI Grounding in VLMs"*，
是目前**时间上最近、主张上最接近**的工作。已读取全文（`docs/references/2605.12549/`）。核心区分点：

| 维度 | Re-Prefill | ZwerGe-UI |
|---|---|---|
| 训练方式 | Training-free（推理时启发式） | 有监督训练 probe + fusion head |
| 信号来源 | 最后 token 跨层 **attention**，需要 `output_attentions=True` | hidden-state probe，**FA2 兼容** |
| 核心机制 | 取高 attention token → append 回输入 → 二次 prefill | 单次 prefill，probe 直接读中间层 hidden state |
| 层级建模 | 无——把"跨层高attention token"当工程手段，不研究哪层最优 | **把 layer-wise posterior 演化本身作为研究对象**（定量分析哪层最先成熟） |
| 对论文的意义 | 可作为"prefill 决定 grounding"这一大类 claim 的旁证 | 本文的问题不是"prefill vs decode"，而是"**prefill 内部哪一层**" |

**论文写作策略（来自 oracle writing_suggestion，已被论文正文部分采纳）**：
不要把 ZwerGe-UI 包装成"another prefill paper"，标题和摘要不要出现"before decoding"这种会被联想到
Re-Prefill 的表达。目前论文标题/abstract 已经用了 "Coordinate Serialization Bottleneck" 这个更精确的
框架，这是对的方向，**继续保持**，Related Work 里对 Re-Prefill 只写一段做边界声明即可，不需要展开攻击。

---

## 4. 竞品定位三维坐标系（写 Related Work 时的检查清单）

来自 `.mrules` §六 长期积累 + oracle report §9.3，三个维度上 ZwerGe-UI 的位置：

1. **Uncertainty/信号来源**：解码后多次采样（UI-Zoomer, AutoFocus, MVP）vs. **单次前向内部状态（ZwerGe ✅）**
2. **Layer-wise 建模**：无层概念/最后层（GUI-Actor）、全层 attention 需 `output_attentions`（GUI-AIMA, Re-Prefill）
   vs. **多层 hidden-state probe，FA2 兼容（ZwerGe ✅）**
3. **Zoom/决策触发**：启发式规则（ZoomClick）、外部 RL（SE-GUI）vs. **learned controller from internal
   posterior（ZwerGe 的设计方向，当前论文版本未包含 zoom controller，只到 probe+fusion）**

**重要提醒**：当前投稿版本的论文（`docs/our_paper_tex`）**没有包含 zoom-in / uncertainty-gated controller**，
只到 Stage1 probe + Stage2 fusion。`.mrules` 里大量关于 zoom_backbone 策略、GRPO controller 的记录是
**代码库已实现但论文未使用**的部分，如果这轮修订要加 zoom 实验来回应"patch granularity ceiling"（Limitations
里提到的问题），`zoom_backbone` decode strategy（`.mrules` [2026-05-22]）是现成可用的实现，直接可以拿来跑。

---

## 5. 代码历史 Bug 复发风险清单（防止重复踩坑）

以下按"复现概率从高到低"排列，来自 `docs/record/debug`（原始调试日志）+ `.mrules` 沉淀总结。

### 高风险（几乎一定会在扩展新模型/新实验时复现）

1. **reinit_grounding_head 逻辑错误**：训练脚本默认按"output_dir 是否有 checkpoint"判断要不要重新初始化
   grounding head。**任何 Stage2/继续训练场景**（从别的目录加载 checkpoint 到一个新 output_dir）都会被
   误判为"fresh run"从而**擦除已训练权重**。已在 A8 引入 `--reinit_grounding_head` 显式开关修复，
   但如果做新的 stage/新模型迁移时忘记显式传这个参数，会静默复发且没有报错，训练 loss 看起来正常但
   全部白训。**每次做新的"接着训"实验前，必须显式确认这个参数、并检查日志里是否打印了 reinit=False。**

2. **bfloat16 下新增参数（LayerNorm 等）未初始化为 NaN**：`from_pretrained(torch_dtype=bfloat16)` 对
   checkpoint 里没有的新增模块，可能分配未初始化的 bfloat16 内存，NaN 概率极高。**任何新增模型分支
   （目前已有 uitars/uitars1/guiowl/guiowl7b/uivenus/qwen35）第一次跑通时都要做一次 `reinit_grounding_head()`
   校验**（强制 LN weight=1/bias=0, LoRA B=0），已封装为方法但要记得在新分支的 `setup_special_token_ids()`
   末尾调用。

3. **深层 hidden state L2 norm 爆炸导致 MLP/LayerNorm 后数值溢出**：本质原因是不同层激活空间的几何未对齐，
   越深层残差累积 norm 越大（7B模型 L26 norm 可达 14000+）。**任何新增 probe 架构（LoRA adapter、CrossAttn
   probe）如果不在 MLP/LN 之前做 RMS pre-scale（`h / (‖h‖/√d)`），几乎必然在深层 NaN**。当前 CrossAttn
   probe 已经内置了这个 pre-scale（对应论文 `4_method.tex` "RMSNorm pre-scale" 段落），但如果之后写新的
   probe 变体（比如 P2 待做的 linear probe / rank-1 probe ablation），**必须重新加上这一步**，否则大概率
   复现同样的 NaN。

4. **Qwen3-VL (Qwen3VLVisionPatchEmbed) 的 Conv3d 性能退化 —— 已被证明是"伪修复"，不要再引入**：
   历史上因为 N=11360 个 patch 走 Conv3d 在特定 PyTorch/cuDNN 版本下慢 3-4 万倍（88秒 vs 0.28ms），
   一度做了 Conv3d→Linear 的 monkey-patch。但后续 [2026-05-24] 严格数值验证发现：**这个 patch 在
   bfloat16 下会引入系统性数值漂移**（视觉编码器逐层放大，最终 LM 层 hidden state 偏差可达 1e2~1e3 量级，
   足以显著影响 grounding head 的判断）。**该 patch 已被彻底移除**（`modeling_guiowl.py` 现已改回官方
   `super().forward()` 路径）。根本解决是 PyTorch 2.8.0 本身已修复该 Conv3d 退化问题（PyTorch 2.9.1 有
   retrogression）。**教训**：任何 monkey-patch 类"性能优化"，必须做逐样本、逐层的数值一致性验证
   （patch前后 max diff），不能只测速度。**如果之后又遇到 Qwen3-VL 训练异常慢，第一反应应该是检查
   PyTorch 版本，而不是重新引入 Conv3d monkey-patch。**

### 中风险（架构/环境迁移时容易碰到）

5. **DDP + gradient_checkpointing 组合报错**：`find_unused_parameters` 与 `gradient_checkpointing` 的
   Trainer 隐式推导逻辑在 guiowl/uivenus（backbone 冻结 + deepstack 结构）下会冲突。正确配置是**显式**
   设置 `gradient_checkpointing_kwargs={"use_reentrant": False}` + `ddp_find_unused_parameters=True`
   （不要依赖 Trainer 自动推导）。`use_reentrant=True` 会导致 GC recompute 时重新产生全层 hidden_states
   （如果开了 `output_hidden_states=True`），带来数百秒/step 的开销。

6. **新模型接入时的类继承层级错误**：transformers 库里同一系列模型（如 Qwen3-VL vs Qwen3.5、Qwen2-VL vs
   Qwen2.5-VL）在新版本 transformers 下往往是**完全独立、无继承关系**的类。历史上 `modeling_qwen35.py`
   最初错误地继承了 `GUIOwlRetrofitModel`（内部指向 `Qwen3VLForConditionalGeneration`），导致
   `from_pretrained` 类型不匹配。**任何接入新模型家族之前，先确认其 `config.json` 的 `architectures`
   字段对应的 transformers 类，不要假设"看起来像的模型"共享基类**。

7. **Anchor token 查找误选 system prompt 里的示例 token**：`<|ground|>` 如果在 system prompt 里出现过
   示例，查找逻辑必须取**最后一个**出现位置（assistant 回复里的），而不是第一个。当前实现（`_find_ground_anchor`
   P1-P5 优先级）已经处理了这个问题，但如果之后改 prompt 模板、加新的 few-shot 示例，要重新检查这个逻辑
   是否仍然成立。

8. **Skip 样本的 zero loss 断开计算图**：batch 内某些样本因为找不到 anchor token 或 label 全零需要跳过时，
   如果用裸 `torch.zeros(1)` 作为占位 loss，DDP 场景下会导致该 rank 的计算图与其他 rank 不一致而卡死/报错。
   正确做法是用 `sum(p * 0 for p in trainable_params)` 这种"连接所有可训练参数但值为0"的写法
   （`_zero_grounding_loss` 方法），保证 DDP 图结构一致。

### Qwen3 系列数值问题——用户明确指出"没有完全解决"，需要重点复查

这是用户在本次任务里**明确点名**的关切点，梳理 `.mrules` 中所有相关记录的时间线和当前状态：

| 时间 | 问题 | 状态 |
|---|---|---|
| 2026-05-08 | float32 下深层 MLP projector 输入 norm 爆炸 → NaN | ✅ 已修复（RMS pre-scale before MLP） |
| 2026-05-08 | bfloat16 下 `from_pretrained` 新增 LayerNorm 权重是 NaN（未初始化内存） | ✅ 已修复（`reinit_grounding_head`） |
| 2026-05-08 | bfloat16 精度比 float32 更容易在 LoRA adapter 内部 LayerNorm 处溢出 | ✅ 已修复（adapter 输入前二次 RMS scale） |
| 2026-05-21 | GUI-Owl (Qwen3-VL) Conv3d 训练极慢（600s/step） | ⚠️ 用 Conv3d→Linear monkey-patch"修复"，**但这是错误方向** |
| 2026-05-24 | 严格数值验证：Conv3d→Linear patch 在 bf16 下产生系统性 drift（视觉编码器逐层放大，LM 层 HS 偏差达 1e2~1e3） | ❌ 发现 05-21 的"修复"实际引入了新的数值不一致问题 |
| 2026-05-24 | 验证 PyTorch 2.8.0（qwen3 conda env）本身已解决 Conv3d 退化，2.9.1（qwen3-verl）仍有 bug | ✅ 确认根因是 PyTorch 版本，不是模型代码问题 |
| 2026-05-24 | 彻底移除 Conv3d monkey-patch + 移除历史遗留的 `_run_language_model` hook 方案，改回官方 `output_hidden_states=True` 路径，统一新旧模型 forward 对称性 | ✅ 代码层面已清理干净 |

**当前结论（回应用户"我感觉这个问题没有完全解决"的担忧）**：
- 数值 NaN 问题（前3条）：**已经过 debug_nan1~11 完整迭代验证，backward pass 成功、训练 loss finite，
  可以认为已解决**，这部分证据链是扎实的。
- Conv3d 性能问题：**绕了一圈弯路**（先用有 bug 的 monkey-patch"修复"，后来才发现这个修复本身有数值问题，
  最后靠升级 PyTorch 版本真正解决）。当前代码状态是干净的（已移除 patch），但**这暴露了一个更深的隐患**：
  guiowl/uivenus 的所有历史训练结果（包括论文里报告的 GUI-Owl-1.5-8B / UI-Venus-1.5-8B 数字）**如果是在
  引入 Conv3d monkey-patch 期间训练的，其 backbone 前向路径与"干净路径"存在系统性数值差异**，需要确认：
  1. 论文里报告的 GUI-Owl-1.5-8B / UI-Venus-1.5-8B checkpoint 具体是哪个训练时间点产出的
  2. 该训练时间点代码里是否包含 Conv3d monkey-patch
  3. 如果包含，训练和推理是否**同时**用了这个 patch（`.mrules` 里说"训练推理一致性有保证"，因为两边都走
     同一个 `_GUIOwlImpl.__init__()`），如果训练推理两边一致，那么这只是一个"内部自洽但与官方实现不同"的
     系统，数字本身仍然有效，但**如果要与其他论文/其他人复现的 GUI-Owl-1.5 结果比较，可能存在系统性偏差**。
  **建议行动**：在这轮补实验之前，先确认当前所有要复用的 GUI-Owl-1.5/UI-Venus-1.5 checkpoint 是否是用
  已清理的干净代码（无 Conv3d patch）重新训练的。如果不是，**建议至少对主结果涉及的两个 Qwen3-VL 模型
  重新跑一遍训练**，确保论文数字来自干净代码路径，这样审稿人如果深挖复现细节不会有隐患。
- **另一个尚未被充分验证的点**：`output_hidden_states=True` 目前用于 guiowl/uivenus 的 forward
  （代替旧的 hook 方案），这比 hook 方案多占约 2.2GB 显存（37层 vs 15层），**如果后续要在同一批 GPU
  上同时增加新的 ablation（比如本轮要补的 linear probe / rank-1 probe 等 P0 实验），需要重新做一次
  显存预算检查**，避免 OOM。

---

## 6. 论文叙事写作原则（避免再被审稿人抓到同类问题）

综合三份审稿意见 + oracle writing_suggestion，写作时的红线：

1. **不要用绝对化表达**："improves across five benchmarks"这类表达必须改成精确表述（哪些模型涨、哪些掉，
   给出 wins/losses 计数和平均 delta，Qwen2.5 vs Qwen3 分开报）。
2. **任何 appendix 里的定性描述（"consistently outperforms"之类）必须先跟数据表逐项核对**，不能凭图感觉写。
3. **凡是用 KL/loss 之类连续目标训练、却用离散策略（BFS centroid）解码的地方**，都要么给出理论对齐说明，
   要么做实验验证两者行为一致。
4. **提到"计算量小/参数少"时要给出具体数字和对比**（inference overhead 表，hooked layer 数，显存开销，
   相对自回归 decode 的延迟）。
5. **凡是提到"backbone 天生具备 XX 能力"，必须用 low-capacity / shuffled-label / OOD 等 control 排除
   "是 probe 学出来的"这个 alternative explanation**，这是本轮最核心的方法论补强方向。
6. **不要主打"before decoding"这个表达**（容易被联想到 Re-Prefill），继续用"coordinate serialization
   bottleneck"这个更精确的框架。
7. **naming 统一**：全文用 `\textsc{ZwerGe-UI}`，禁止手打 "ZWERGE-UI" 等不一致大小写。

---

## 7. 本轮补实验的推荐执行顺序

按"性价比"（对应审稿人诉求的覆盖面 / 实现难度）排序，不是按上面列出的编号顺序：

1. **先做免费的**（不需要新训练，纯分析/复核，1天内可完成）：
   - 核对 appendix fusion claim 与实际数字是否一致（P0-6），错了就改文字，不需要新实验。
   - 明确写清楚 Stage2 到底哪些参数训练/冻结、梯度路径（P0-1），这是纯写作修复。
   - 给现有指标补 bootstrap CI（P1-7），用已有 eval 结果重新算，不需要重训。
   - naming 统一、Equation 5 符号定义、compute overhead 报告（P2），纯写作。

2. **中等成本**（复用现有 probe 训练框架，改一下架构/label，1-3天）：
   - Level-0/1/2 probe capacity sweep：无参数 cosine → rank-1 linear probe → 当前 CrossAttn（P0-2 的核心部分）。
   - Random-label / shuffled-label control（P0-2）。
   - 标准 pixel-level hit@1 补充报告，不只报 overlap@1（P0-5，这个其实评测脚本里数据已经都有，
     只是没有把 hit@1 单独摘出来放主表，最低成本）。
   - Gaussian label σ_min 下限 ablation（P1-11）。
   - No-CoT vs post-think anchor 消融（P1-12，对应 oracle 实验4的方案B/C）。

3. **较高成本**（需要新的分析脚本/新一轮训练评测，3-5天）：
   - Tuned lens 替代 logit lens 做 serialization lens（P0-3）。
   - 200k 训练数据的完整来源表 + benchmark 重复检测（P0-4）。
   - 跟 GUI-Actor 同 backbone 下的定量对比表（P1-9）。
   - held-out OOD layout 上验证中间层优势（P0-2 的一部分）。

4. **视时间决定是否做**（P2 + 锦上添花）：
   - CosMeta 共享子空间的可视化验证（P1-10）。
   - 训练/解码目标对齐的理论或实验说明（P1-13）。

---

## 8. 关键文件/路径速查

```
论文正文：       docs/our_paper_tex/secs/*.tex（1_intro ~ a_appendix）
审稿意见：       docs/reviews/Reviewer-{eKpD,4KeN,4sok}
写作策略调研：   docs/oracle/writing_suggestion（论文结构/图表/标题建议，已部分采纳）
方法论调研：     docs/oracle/report（6方向文献调研：attention sink/layer probing/token pruning/
                 uncertainty/sparse QK/frozen probe 先例，含大量可引用 arXiv ID）
实验设计调研：   docs/oracle/chatgpt-export_坐标瓶颈假说实验设计（G_l/C_l/LLI/CBI/DBR 等指标形式化，
                 8个具体可执行 probe 实验设计，含数学定义/怎么跑/预期图/降级方案）
最早期方法讨论： docs/oracle/chatgpt-export（历史背景，端到端问题讨论，核心结论已沉淀进 .mrules）
Re-Prefill 原文：docs/references/2605.12549/neurips_2026.tex
调试历史全record：docs/record/debug（原始日志，6247行，核心结论已提炼进 .mrules，一般不需要重读）
Stage2/A8设计：  docs/record/fusion（A8 ContextLoRACosMetaFusion 完整数学定义+代码改造点，已实现）
核心工程细节：   .mrules（模型路径、训练脚本、evaluate流程、所有历史bug修复记录，本README的"底层数据库"）
```

**代码结构提醒**（详见 `.mrules` 一/二节）：主代码在 `zwerge/` 子模块，`src/zwerge_retrofit/` 是核心建模代码
（`modeling_base.py` 公共组件 + `modeling_{uitars,uitars1,guiowl,uivenus,qwen35}.py` 各模型分支），
`eval/` 镜像同样结构，`experiments/*.yaml` + `eval_daemon.py` 是训练/评测异步流水线入口。

**`zwerge/probe/` 目录（论文 §3 分析实验的实际代码，本轮补实验应优先复用/扩展这里，而不是从零写）**：
```
probe_utils.py             — 公共工具（24个函数/类），posterior 收集、hit/overlap/KL 等指标计算的底层实现
exp3_serialization_lens.py — 对应论文 Finding 2（spatial lens vs serialization lens 双曲线错位）
exp4_counterfactual.py     — 对应论文 Finding 3（instruction-switch 反事实实验）
plot_probes.py             — 画图脚本（很可能也覆盖了 Finding 1 的逐层曲线，即 Experiment 1/2 未独立成文件，
                              被合并进这里或直接用 A7 eval 的逐层结果画的）
run_exp3.sh / run_exp4.sh / run_all_probes.sh — 对应的启动脚本
summary                    — **重要**：另一份 oracle 实验设计文档（904行），是 exp3/exp4 代码的直接设计蓝图，
                              还包含了 Experiment 1（Layerwise Fingerprint）、Experiment 2（Layer Aggregation
                              vs Layer-Specific）、Experiment 5（Small Target Stratification）的完整设计
                              （数学定义+代码方案+CLI），但这两个实验最终**没有生成独立脚本**，本轮如果要做
                              P0-2 的 probe-capacity sweep / baseline 对比，可以参考这份文档里 `probe_patch/`
                              方案设计（最小侵入式、只读原代码、新建 collectors.py/metrics.py/token_lens.py
                              等），虽然目录本身未被采用，但设计思路仍然适用。
```
该目录明确降级/不用的实验（避免重复踩坑）：Prefill-only vs Generate（容易被误认为抄 Re-Prefill）、
occlusion causal experiment（因果性论证薄弱，易被质疑"遮的本来就是GT附近"）——这两个见 `summary` 第15-31行。

---

## 9. 变更日志（本文件自身的维护记录）

- [2026-07-22] 首次创建。基于用户要求，系统阅读 `docs/reviews`、`docs/oracle`、`docs/our_paper_tex`、
  `docs/references/2605.12549`、`docs/record` 全部内容 + 重新梳理 `.mrules` 全文后撰写。
  核心目的：为基于审稿人意见的补充实验阶段提供唯一记忆入口。
