# 交接文档：ZwerGe-UI 分析性 Probe 实验加固（与刷点 agent 并行）

## 0. 你的定位与边界

你和另一个 agent **并行工作**，那个 agent 正在做 `docs/oracle/chatgpt-export_Qwen3负优化与OPD.txt` 里定义的 ZWERGE-P2P 训练-free 刷点任务（改 `zwerge/eval/inference_base.py`、`eval_retrofit.py`、跑 hope job、改 `4_method.tex`/`5_experiment.tex`）。

**你的任务完全不同、互不冲突**：不碰方法刷点、不碰 P2P 代码、不碰主表数字。你只负责把 `docs/our_paper_tex/secs/3_analysis.tex`（The Coordinate Serialization Bottleneck，四个 Finding）以及相关 appendix 变成一篇**即使方法部分完全不刷点，也能独立站得住的分析性论文**。

**核心心态**：这篇论文现在的定位是 **analysis-primary, method-secondary**（已在 `.mrules` 2456 行确认）。三位审稿人对分析部分提出的质疑比对方法刷点的质疑更致命、更本质——如果分析部分的因果链不够硬，方法刷不刷点都救不了这篇论文；反过来，如果分析部分做得像 *Vision Transformers Need Registers*（`docs/references/2309.16588/`）那样严谨完备，即使方法部分毫无起色，这也会是一篇很强的 Findings/分析论文。

**唯一的产出要求**：新增/加固分析实验的代码 + 结果 + 图表 + 对应的论文文字修改（`3_analysis.tex`、可能新增 appendix 小节）。**不要创建新的 README/说明文档**；只在 `.mrules` 变更日志区块追加你做了什么（这是仓库既有习惯，见 `.mrules` 第 248 行的 memory）。

---

## 1. 论文当前状态（你要加固的对象）

- 论文标题：*Middle Layers Know Where to Click: Coordinate Serialization Bottlenecks in GUI Agents*，投 AAAI 2027（`docs/our_paper_tex/AnonymousSubmission2027.tex`）。
- 核心假说："coordinate serialization bottleneck"——GUI agent 把"定位目标"和"把坐标序列化成 token"这两个计算糅合在一起，二者在 transformer 深度上的最优层不同：**中间层更适合空间定位，末层更适合坐标序列化**。
- 支撑这个假说的分析章节是 `docs/our_paper_tex/secs/3_analysis.tex`，四个 Finding：
  1. **Finding 1**（`sec:analysis:layerwise`）：中间层比末层更善于空间解码（用 Stage-1 CrossAttn probe 测的 spatial hit@1，见 Fig. `fig3_serialization_lens.pdf` 蓝线）。
  2. **Finding 2**（`sec:analysis:serialization`）：serialization lens——用 backbone 原生 LM head 在 teacher-forcing 下测坐标 token 的 log-likelihood，逐层，发现空间峰值层 $L^*$ 系统性早于坐标似然峰值层 $L^\dagger$（Fig. 同上，橙线）。
  3. **Finding 3**（`sec:analysis:counterfactual`）：instruction-switch 反事实实验——同一张图换指令，测中间层 posterior 是否显著偏移（排除"纯视觉显著性"的替代解释），见 Fig. `fig4_instruction_suppression.pdf`。
  4. **Finding 4**（`sec:analysis:resolution`）：collapse gap（peak-final 层 hit@1 差）在高分辨率小目标 benchmark（SS-Pro）上系统性大于中等分辨率（SS-v2），见 `tab:resolution_comparison`。
- 支撑这四个 Finding 的代码已经存在，你不用从零写：
  - `zwerge/probe/exp3_serialization_lens.py` + `zwerge/probe/probe_utils.py`：产出 Finding 1+2 的数据（spatial hit@1 逐层 + coord NLL 逐层，两路 forward）。
  - `zwerge/probe/exp4_counterfactual.py`：产出 Finding 3 的数据（instruction-switch pair 选取 + JS divergence + suppression）。
  - `zwerge/probe/plot_probes.py`：画图脚本。
  - `GUI-AIMA/eval/layer_probe.py`：更早期的、**无参数** attention-based probe 工具（Signal A/B/C，见下文第 4 节），是一个重要的"对照工具"，但目前论文正文没有直接引用它的产出。
  - Stage-1/Stage-2 CrossAttn probe 训练/推理代码：`zwerge/src/zwerge_retrofit/modeling_base.py`（`CrossAttnGroundingProbe`、`LayerWiseGroundingHead`）+ `zwerge/eval/inference_base.py`（`predict_layerwise`）。

---

## 2. 三位审稿人对"分析部分"的完整质疑清单（逐条，全部要处理或至少书面回应）

请先自己完整读一遍原文：`docs/reviews/Reviewer-4KeN`、`docs/reviews/Reviewer-eKpD`、`docs/reviews/Reviewer-4sok`。以下是我提炼的、**专门针对分析/probe 部分**的质疑（方法刷点相关的质疑已经在另一个 agent 那边处理，这里不重复）：

### 4KeN（Findings 分数 3，主要担心分析的因果性不够）
- **[4KeN-W1]** "The main evidence comes from large supervised cross-attention probes, not directly from the original GUI agents. This shows that target locations are decodable from intermediate layers, but it does not fully prove that the model naturally knows where to click." → **需要一个不训练/低容量的对照，证明结论不是探针学出来的伪象**。
- **[4KeN-W2]** "The probes are expressive, so the middle-layer advantage may partly reflect where this probe design learns best. The claim that late layers are 'coordinate serializers, not spatial refiners' is too strong; late layers may still encode spatial/action information in a different form." → **需要 probe-capacity sweep**（不同参数量的 probe 是否都得到同样的 inverted-U，还是容量越大中间层优势越明显——这直接决定 Finding 1/2 的说服力）。
- **[4KeN-W3]** "Figure 2 supports instruction sensitivity, but for some models the strongest effects appear in later layers, not clearly middle layers." → 需要**逐模型**报告 instruction-switch 峰值层与 spatial-peak 层是否对齐，如果不对齐要如实报告并讨论，而不是笼统地画一条平均曲线。
- **[4KeN-W4]** "The high-resolution/small-target claim is also based on dataset-level comparisons, without controlling for target size, resolution, UI density, or error type." → Finding 4 目前是 benchmark 级别的粗对比（SS-Pro vs SS-v2 两个数据集），**需要在同一个 benchmark 内部按 target size/resolution 分桶**做受控回归，而不是换数据集当作换变量。
- **[4KeN-W5]** "Since the method predicts patches and uses overlap@1, the paper should clearly report patch size, box-to-patch mapping, and target-size effects. Metrics beyond top-1 overlap, such as target-region probability mass or top-k coverage, would also help." → 需要补充 patch size/box-to-patch 映射的明确说明，以及 probability-mass 与 top-k coverage 指标（`probe_utils.py` 里已有 `compute_target_mass`，可以直接用）。

### eKpD（分数最低，Soundness=2，最系统的质疑清单，重点看 W2/W3/W8）
- **[eKpD-W2]**（最核心）："a full-rank cross-attention probe can learn substantial task-specific readout geometry and dataset priors... More controls are needed: **lower-capacity linear probes, shared probes across layers, random-label controls, held-out OOD UI layouts, probe-capacity sweeps, and direct comparison with attention-only or saliency baselines**." → 这是审稿人明确列出的**六项具体控制实验**，几乎是一份现成的 TODO 清单，见下文第 3 节逐条展开。
- **[eKpD-W3]**（serialization lens 的方法学漏洞，非常致命）："applying a final LM head to intermediate states is a **logit-lens-style analysis that can be heavily confounded by layerwise alignment to the output embedding space**. It is expected that later layers will become more compatible with the final LM head, even for reasons unrelated to coordinate serialization... controls such as **non-coordinate action tokens, random coordinates, text-only tokens, calibrated tuned lenses, or causal interventions**." → logit lens 是个已知有 confound 的技术（层与最终输出空间的对齐程度本身随深度增加，不代表"内容"变化）。这是 Finding 2 的根基，必须加控制，否则整个"serialization lens"论证都立不住。
- **[eKpD-W6]**（诊断样本量/置信区间）："The key layerwise analysis in Figure 1 uses (n=200), and the instruction-switch experiment uses (n=150) pairs... The paper should report confidence intervals or bootstrap intervals... and should report whether the probe training is stable across random seeds." → 需要 bootstrap CI + 扩大样本量 + 多 seed 稳定性。
- **[eKpD-W7]**（appendix fusion claim 与图不符，已被上一个 agent 部分修复，见 `.mrules` 2558 行；但你要独立核实数字是否真的站得住，如果发现新的不一致要继续纠正）。
- **[eKpD-W8]**（消融不直接验证最终方法）："The ablation section uses a 'fast LoRA-probe variant' separate from the final CrossAttn probes... The final CrossAttn architecture should be ablated directly, including **probe capacity, number of active layers, final-layer-only probe, intermediate-only probe, static averaging, learned fusion without CosMeta, and patch decoding strategy**." → 这也是分析性质的消融，不是刷点，属于你的范围。
- **[eKpD-Comments]** 补充的具体控制要求（原文摘录）："random instruction, random target, shuffled labels, image-only, text-only, saliency-only, final-layer tuned probe with higher capacity, and lower-capacity linear probes."

### 4sok（分数最高 3.5，问的是架构细节，但每条都可以变成一个漂亮的诊断实验）
- **[4sok-Q1]** CosMeta fusion 的全局 meta-query 如何在"异构语义空间"（不同层的表示空间不对齐）里算余弦相似度还有意义？→ 可以做一个诊断：**逐层 anchor query 之间的表示相似度/CKA，量化"异构性"到底有多大**，用数据回答这个质疑而不是空口辩解。
- **[4sok-Q2]** 小目标下 Gaussian label 的方差趋近于零，KL loss 是否退化成 one-hot？→ 可以做**小目标 vs 大目标分层的定量分析**（label 的有效熵 vs bbox 大小的关系），直接是 Finding 4 的一个自然延伸。
- **[4sok-Q3]** prefill-forced 推理是否有 OOD 偏移（backbone 通常在 CoT/推理 token 之后才生成坐标）？→ 需要**长度/位置扰动实验**：在 anchor token 前人为插入不同长度的填充 token，看 probe 读出的 posterior 是否稳定。
- **[4sok-Q4]** pre-scale 是否丢失了"activation magnitude ~ 注意力汇聚/置信度"的信息（对应 Registers 论文里 high-norm token 的现象）？→ **这是与 Registers 论文最直接的联系点**，见第 4 节。
- **[4sok-Q5]** 连续 KL 训练目标与离散 BFS-centroid 解码策略是否一致？→ 可以做训练目标（Gaussian KL）与解码几何（centroid vs mode）之间的失配定量分析。

---

## 3. 具体新增分析实验清单（按优先级，每条标注对应审稿人问题 + 建议实现方式 + 预期产出）

> 原则：**每个实验都要能在现有代码基础上，用几百行新代码 + 复用 `probe_utils.py`/`inference_base.py` 的接口，在 1-2 天内跑完**。不要设计需要重新训练大模型的实验（那是刷点 agent 的事，你没有训练资源冲突的余地）。所有实验都应该是"训练一个小 probe/线性层"或"跑现有 checkpoint 做统计"级别的工作量。

### P0（必须做，直接回应审稿人最集中的质疑）

**P0-1. Probe-capacity sweep（回应 4KeN-W2, eKpD-W2, eKpD-W8）**
- 目的：证明"中间层空间优势"不是因为大容量 CrossAttn probe（~4.2M/层）学到的伪影，而是不同容量的 probe 都收敛到同一个 inverted-U。
- 实现：在现有 `CrossAttnGroundingProbe`（`zwerge/src/zwerge_retrofit/modeling_base.py`）旁边新增至少 2 个更低容量的 probe 变体：
  - **线性 probe**：`p_l = softmax(W_k h_v · W_q h_q / sqrt(d))`，`W_q`/`W_k` 都是单层线性投影（无多头、无 gate），参数量降到 ~0.1M/层量级。
  - **共享 probe（shared across layers）**：所有 probe layer 共用同一组 `W_q`/`W_k` 权重（只有 anchor/visual 的 hidden state 输入不同），验证"是探针的可学习几何在起作用还是层的表示本身在起作用"。
  - 只需要在少量层（比如 probe_layers 里挑 6-8 层）上训练这些低容量变体，数据量可以比 A7 的 200k 小（5-10 万条即可，能跑出稳定 inverted-U 就行），不需要追求刷点意义上的高精度。
- 产出：一张图——x 轴层数，y 轴 spatial hit@1（或 overlap@1），叠加"CrossAttn probe（当前方法）/ 线性 probe / 共享 probe"三条曲线，证明 inverted-U 形状在所有容量下都存在（哪怕绝对数值不同）。
- 论文位置：`3_analysis.tex` Finding 1 段落后新增一个 "Robustness to probe capacity" 小段 + 新图/新 appendix 图。

**P0-2. Serialization lens 的 logit-lens 控制实验（回应 eKpD-W3，最高优先级，因为这是 Finding 2 的根基）**
- 目的：排除"后期层只是碰巧和输出 embedding 空间更对齐"这个混淆解释。
- 实现（`zwerge/probe/exp3_serialization_lens.py` 里 `logit_lens_nll_hooks` 已经是核心工具，扩展调用方式即可，不需要重写）：
  1. **非坐标 action token 对照**：除了坐标 token 的 NLL，额外测量同一序列中**其他非坐标、非特殊 token**（比如 `pyautogui.click(` 里的普通文本 token）的逐层 NLL。若这些 token 的逐层 NLL 曲线形状与坐标 token几乎一致（都是"越深越准"），说明这只是通用的 logit-lens 效应，不特定于坐标序列化；若坐标 token 曲线明显比一般 token 更陡/更晚才追平，才能说明有坐标特异的东西在发生。
  2. **随机坐标对照**：把 GT 坐标替换成同分布的随机坐标（或均匀采样的坐标），重新算 NLL 曲线。这条曲线不应该表现出"逐层单调上升"的规律（因为不存在真实待预测的信息，模型只会给出先验概率），如果它也单调上升说明测的是与内容无关的量。
  3. **Tuned lens（简单版本，用 eKpD 原文措辞 "calibrated tuned lenses"）**：为每个 probe layer 学一个轻量的仿射变换 $\hat h_\ell = A_\ell h_\ell + b_\ell$，把中间层 hidden state 校准到输出 embedding 空间（这是 nostalgebraist 2020 "logit lens" 之后 Belrose et al. 2023 "tuned lens" 的标准做法，可以用几千条数据快速拟合一个线性/仿射层），再测 tuned NLL。如果 raw logit lens 和 tuned logit lens 的 $L^\dagger$（serialization 峰值层）位置一致，说明 raw lens 的结论是稳健的，不是对齐 confound 导致的。
- 产出：`fig3_serialization_lens.pdf` 旁新增一张"控制对照图"：坐标 token vs 非坐标 token vs 随机坐标 vs tuned-lens 校准后的坐标 token，四条曲线叠加，证明 $L^\dagger$ 的位置在控制了对齐 confound 之后依然存在/依然稳定。
- 论文位置：`3_analysis.tex` Finding 2 段落大幅扩写 + 附一个 appendix 小节详细说明四种控制的定义与结果。

**P0-3. 随机标签/random control（回应 eKpD-W2, eKpD-Comments）**
- 目的：验证 probe 的高准确率不是因为它记住了训练分布的位置先验（比如"图标总是在屏幕中间/左上角"这种数据集偏置），而是真的在读取图像+指令条件下的空间信息。
- 实现：复用现有 Stage-1 训练管线（`zwerge/train_retrofit.py`），在一个小规模子集上训练两个对照 probe：
  - **shuffled-label**：训练时把 GT bbox 标签在 batch 内随机打乱（图不变，label 换成另一个样本的），若模型仍能训出不错的 loss 下降，说明存在某种可以被数据集统计规律利用的捷径；预期这个对照的 test 准确率应该显著低（接近随机/先验分布水平）。
  - **random-instruction**：instruction 替换成与图像无关的随机字符串/无意义占位符，其余不变。这条对照直接呼应 Finding 3 的 instruction-switch 逻辑，但反过来做：完全去掉真实指令语义，测 probe 输出是否退化为纯粹的位置先验分布（比如始终指向图像中心或统计上最常见的图标位置）。
- 产出：一张小表，列出 {正常训练, shuffled-label, random-instruction} 三种设置下的 test hit@1，预期看到显著的性能坍缩，作为"probe 学到的是真实定位能力而非数据集偏置"的直接证据。
- 论文位置：新增到 `3_analysis.tex` Finding 1 附近，或单独放入 appendix 作为 "Sanity Controls" 小节。

**P0-4. OOD held-out UI layout 泛化测试（回应 eKpD-W2 "held-out OOD UI layouts"）**
- 目的：证明 inverted-U 层级模式不是对训练分布（GroundCUA/OS-Atlas/AgentNet 200k 混合数据）过拟合的产物。
- 实现：直接用**已有的 5 个评测 benchmark 之间的交叉验证**即可，不需要新数据——训练数据主要来自 OS-Atlas/GroundCUA/AgentNet 风格，UI-Vision 和 MMBench-GUI 的 UI 布局风格与之差异较大（可以在 `docs/oracle` 或 `zwerge/eval/inference_base.py` 里的 `BENCH_CONFIGS` 查这几个 benchmark 的来源说明）。已经有的 Appendix Figure（`a7_layerwise_combined.pdf`，逐 benchmark 逐层曲线）本身就是很好的材料——只是目前没有专门从"OOD 泛化"这个角度去写一段论述。
- 你要做的是**分析现有数据**而非重新训练：从已经跑出来的 per-benchmark 逐层 overlap@1 数据中，量化"inverted-U 峰值层位置"在 5 个 benchmark 之间的方差/一致性（比如算一个"peak layer 跨 benchmark 标准差"的小统计量），如果峰值层在分布差异很大的 benchmark 之间依然稳定，就是很强的 OOD 泛化证据。
- 产出：一个小表或一句定量论述："peak layer varies by at most ±k layers across all 5 benchmarks despite substantial UI-style differences between (train-adjacent) OS-Atlas-style benchmarks and (OOD) UI-Vision/MMBench-GUI"。
- 论文位置：`3_analysis.tex` Finding 1 或 appendix，工作量很小但论证力度很强，优先做。

### P1（重要，能显著加固但工作量适中）

**P1-1. Attention-only / saliency-only baseline（回应 eKpD-W2, eKpD-Comments）**
- 目的：`GUI-AIMA/eval/layer_probe.py` 里已经实现了**无参数**的注意力探针（Signal A：ANCHOR→visual 的原生 attention，无需训练任何东西）。用这个作为"零参数下界"和当前 42M 参数的 CrossAttn probe 对比，如果无参数的原生 attention 也能大致复现同样的 inverted-U（哪怕绝对数值低很多），这是**最强的证据**，直接正面回应 4KeN-W1（"这展示的是探针的可解码性,不是模型本身天然知道"）——因为原生 attention 完全不训练，是模型的真实内部信号。
- 实现：直接复用 `GUI-AIMA/eval/layer_probe.py` 的 `compute_layerwise_signals`（Signal A，第 312-430 行左右）和 `patch_scores_to_point`，在与 Stage-1 CrossAttn probe **相同的评测样本**上跑一遍，逐层输出 hit@1/overlap@1。
- 此外可以加一个纯视觉 saliency baseline（不看指令，只用图像本身的某种显著性图，比如简单的边缘/对比度显著性，或者干脆用"图像自身与自身的 patch-wise 余弦相似度找孤立/独特 patch"这种无监督方法），进一步压低下界。
- 产出：在 P0-1 的图上再叠加一条"zero-parameter native attention"曲线，与"random/saliency-only"曲线一起，构成从 0 参数到 42M 参数的完整能力谱系。
- 论文位置：`3_analysis.tex` Finding 1，或者单独起一个 "From Attention to Trained Probes: A Capability Spectrum" 小节。

**P1-2. Bootstrap 置信区间 + 多 seed 稳定性（回应 eKpD-W6）**
- 目的：当前 n=200（layerwise 分析）、n=150（instruction-switch）的样本量太小，报告的 6-11pp 差距可能落在噪声范围内。
- 实现：
  1. 直接扩大样本量：`exp3_serialization_lens.py`/`exp4_counterfactual.py` 的 `--n_samples` 参数直接调大（比如从 200/150 提到 1000+，反正 SS-Pro 有 1581 条、SS-v2 有 1271 条，够采）。这是最低成本的改进。
  2. 对关键统计量（spatial hit@1 逐层曲线、$L^*$/$L^\dagger$ 的层位置、instruction-switch suppression 幅度）做 bootstrap resampling（对样本做有放回重采样 1000 次），报告 95% CI，画成 shaded band 或误差棒。
  3. 如果时间允许，用不同随机种子重新训练一次 Stage-1 probe（哪怕只训一个小规模子集），验证 inverted-U 的峰值层不因训练随机性而大幅漂移。
- 产出：Fig 3/4 的曲线加上阴影置信区间；正文加一句"$L^* $ is stable within $\pm$X layers across bootstrap resamples / random seeds"。
- 论文位置：`3_analysis.tex` 全篇加 CI，appendix 补充多 seed 稳定性小表。

**P1-3. Finding 4 的受控回归（回应 4KeN-W4）**
- 目的：Finding 4 目前是"SS-Pro vs SS-v2 两个数据集"的粗对比，审稿人指出这混淆了 target size/resolution/UI density 等多个变量。
- 实现：**不需要跨数据集**，改成在单一较大的数据集内部（推荐用 ScreenSpot-Pro，本身跨越了从小图标到大按钮/多种分辨率），把每个样本的 target bbox 面积（相对图像面积的比例）、图像分辨率记录下来，按这两个维度分桶（比如 bbox 面积四分位数 × 分辨率三分位数），在每个桶内单独计算 collapse gap（peak-final hit@1 差）。
- 产出：一个 collapse gap 对 (bbox_size, resolution) 的热力图或分桶表，证明"gap 随 bbox 变小/分辨率变高而单调增大"这个关系在**受控**（同一数据集内部、真正只改变一个变量时）下依然成立，而不是数据集选择效应。
- 论文位置：替换或补充 `3_analysis.tex` 的 `tab:resolution_comparison`。

**P1-4. Coordinate-free 消融的直接化（回应 eKpD-W8）**
- 目的：现有 ablation（`docs/reviews` 提到的 "fast LoRA-probe variant"，只训 50k、eval 在 checkpoint 800）不是最终 CrossAttn 架构的直接消融。
- 实现：如果时间允许，在**现有的 A7 CrossAttn checkpoint 基础上**（不重新训练全部，而是做 inference-time 的功能性消融）：
  - **final-layer-only probe**：只用最后一层 probe 的输出当最终预测（zeroing out 其它层的贡献，或者单独评测最后一层 probe），对比 fusion 结果。
  - **intermediate-only probe**：只用中间层（比如 Finding 1 里峰值附近 2-3 层）当最终预测。
  - **static uniform averaging**：所有 active probe 等权平均（不用 learned fusion scorer），对比 learned CosMeta fusion。
  - 这些都可以在**已有 checkpoint 上直接跑推理时改一下聚合方式**，不需要重新训练，工作量很小（`zwerge/eval/inference_base.py` 的 `decode_p2p`/fusion 相关代码已经暴露了这些中间量，比如 `per_layer_probs`、`omega`，直接手写聚合逻辑做后处理评测即可）。
- 产出：一个表，行是 {final-layer-only, intermediate-only (peak band), static-average, learned CosMeta fusion (当前方法)}，列是各 benchmark 的 overlap@1，直接量化"为什么要用中间层 + learned fusion，而不是更简单的方案"。
- 论文位置：`a_appendix.tex` 新增消融表，与现有 A7 层曲线图相邻。

### P2（锦上添花，若时间充裕再做，直接呼应 Registers 论文和 4sok 的架构问题）

**P2-1. Anchor 表示的"异常范数" / attention-sink 现象普查（直接对应 4sok-Q4，并与 Registers 论文强关联）**
- 这是**最值得做的一条**，因为它能把整篇论文的分析部分和用户指定要参考的 *Vision Transformers Need Registers* 直接对上，形成一个新的、原创的 Finding，而不只是回应审稿人。详见第 4 节的详细设计。

**P2-2. 逐层表示相似度 / CKA（回应 4sok-Q1）**
- 目的：定量刻画"不同层的语义空间有多异构"，直接回答 CosMeta fusion 用单一全局 meta-query 是否合理。
- 实现：对 probe layers 之间两两计算 anchor query 表示的 **CKA (Centered Kernel Alignment)** 或简单的表示子空间夹角，画一个层×层的相似度矩阵热力图。
- 产出：如果相邻层高度相似、远层低相似度，说明"异构性"确实存在但是平滑变化的（而非突变），可以用来论证 CosMeta 的全局 meta-query 之所以能工作，是因为它其实是在一个连续变化的流形上找相似度，而不是在完全不相关的空间里瞎比。

**P2-3. 训练目标与解码几何的失配分析（回应 4sok-Q5）**
- 目的：定量刻画"训练时用 Gaussian KL 目标，推理时用 BFS-centroid/argmax 离散解码"之间的 gap 有多大。
- 实现：对一批样本，比较"如果直接对训练时的 Gaussian 目标做解析最优解码（连续 mode）"与"当前的 BFS-centroid 离散解码"两者的预测点误差分布，这其实和刷点 agent 正在做的 P2P Level-3（`refine_local_mode`，见 `zwerge/eval/inference_base.py` 323-394 行）高度相关——**你可以直接复用这个函数做分析，而不用重新实现**，只是这次不是为了刷点，而是为了在论文里定量描述"训练-推理目标失配"这个现象本身。
- 论文位置：可以直接放在 P2P 方法小节旁边补一句分析性的话，或者单独放入 appendix。

---

## 4. 参考模板：*Vision Transformers Need Registers*（`docs/references/2309.16588/`）给你的具体设计启发

这篇论文（Darcet et al. 2023, Meta FAIR）是**纯分析驱动方法**的典范，请通读 `1_intro.tex`、`3_problem_formulation.tex`、`4_experiments.tex`（已经在这次对话里读过，以下是提炼出的可复用范式，直接映射到 ZwerGe 该怎么做）：

### 4.1 它的分析范式（五步闭环）

1. **定性发现异常**（Fig 1: 多个模型的 attention map 都有离群的高分 patch）→ ZwerGe 已有对应（inverted-U 曲线）。
2. **定量刻画异常的统计特征**（`3_problem_formulation.tex` §3.1）：
   - 用一个简单可复现的判据定义"异常"（norm > 150，双峰分布里的阈值），**不是拍脑袋，而是从直方图的双峰性里读出来的**。
   - 系统性地扫描"异常在什么条件下出现"：**层数**（哪一层开始分化，Fig 3a）、**训练迭代数**（训到多久才出现，Fig 3b）、**模型规模**（多大的模型才出现，Fig 3c）。这是一个三因素的受控扫描，而不是单一维度的观察。
   - 这一步直接映射：ZwerGe 目前只有"层数"这一个维度的分析（Finding 1），**没有做"训练量/模型规模"维度**。如果时间允许，P2 可以补一个"probe 训练数据量 vs inverted-U 清晰度"的扫描（用不同大小的训练子集训 probe，看多少数据量开始出现清晰的中间层峰值），这是 Registers 论文 Fig 3b 的直接类比。
3. **机制探测：这个异常携带什么信息**（`3_problem_formulation.tex` "High-norm tokens hold little local information" + "Artifacts hold global information"）：
   - 用两个**互补的线性探针**（position prediction + pixel reconstruction）证明异常 token 丢失了局部信息；
   - 用一个**下游任务**（ImageNet 线性分类）证明异常 token 反而携带更多全局信息。
   - 这一步的核心方法论是：**用最简单的线性模型做探针**（而不是用复杂的、可能自带偏置的探针），这样"探针学到了什么"这个混淆因素被最小化——**这正是 eKpD-W2 对 ZwerGe 的批评**：ZwerGe 现在用的是 42M 参数的全秩 CrossAttn probe，Registers 论文全程只用线性探针。**这是你在 P0-1 里必须补的最重要的对照**。
4. **提出可证伪的假设**（"模型学会识别冗余 patch 并用它们存储全局信息"）→ ZwerGe 对应的是"coordinate serialization bottleneck"假设，已经提出，但证据链的严谨度需要按上面的模式补强。
5. **设计干预并验证假设**（加 register token，训练前后对比，异常消失）→ **这一步 ZwerGe 目前没有对应物**！Registers 论文最有说服力的地方在于，它不仅是相关性分析，还做了一个**因果干预**：加了 register token 之后异常真的消失了，而且下游任务性能不降反升。ZwerGe 目前的"方法"（ZwerGe-UI retrofit）某种意义上是这个"干预"的对应物，但目前写法上是"方法"章节，跟"分析"章节是分开的。**建议**：在 `3_analysis.tex` 末尾或作为衔接段，明确写一句话把方法框成"assumption 的因果验证"：如果把中间层的 posterior 单独读出来生成最终点击，效果确实比末层自回归坐标好，这本身就是对"末层不是最优空间解码位置"这个假设的因果验证，而不仅是工程性能提升。这是一个纯粹的**叙事重构**，不需要写新代码，只需要在 `3_analysis.tex` 和 `4_method.tex` 之间加一两句衔接。
6. **消融控制变量**（Fig `scores_n_reg`: register 数量从 0 到 16 的扫描，测下游性能和视觉伪影消失情况）→ ZwerGe 对应 P1-4（probe/fusion 架构消融），已经在清单里。
7. **额外的定性分析**（Fig `slot_attn`：不同 register token 学到了不同的注意力模式，纯观察性、诚实地说"我们不确定为什么"）→ 你在 P2-1（异常 token/attention sink 普查）里如果做出定性可视化，也可以用这种"诚实报告、不过度诠释"的风格写。

### 4.2 最值得直接借鉴的一条：high-norm token / attention sink 分析（对应上面的 P2-1，强烈建议做）

这是这篇论文给你最直接的可复用灵感，而且和 4sok-Q4（"pre-scale 是否丢弃了 activation magnitude ~ token confidence/attention sink 的信息"）直接相关：

- **背景**：ZwerGe 的 RMSNorm pre-scale（`4_method.tex` 第 43-45 行，"Hidden-state $\ell_2$ norms grow substantially with layer depth"）是为了数值稳定性引入的，做法是把每层输入的 norm 强制缩放到 $\sqrt d$。但 Registers 论文和后续大量 LLM 可解释性工作（attention sink，Xiao et al. 2023 StreamingLLM；massive activations，Sun et al. 2024）都发现：**某些 token（往往是极少数几个，比如序列首 token、或者语义上"无意义"的 token）会积累异常大的 hidden-state norm，这些高 norm token 在 attention 里被大量 head 用作"信息汇聚点"（sink），丢弃它们的局部语义、专注做全局聚合**。
- **可做的具体分析**（复用你已有的评测 pipeline，不需要新训练）：
  1. 对一批评测样本，在 ZwerGe 的 forward 里，**在 pre-scale 之前**记录每个 visual patch token 在各层的 hidden-state $\ell_2$ norm（`zwerge/src/zwerge_retrofit/modeling_base.py` 里 `LayerGroundingProbe`/`CrossAttnGroundingProbe` 的 forward 已经能拿到 `h_v`，加几行记录 norm 分布即可，完全无需改动训练逻辑，可以用 hook 或者直接改一份分析专用的 inference 脚本）。
  2. 画出 norm 分布的直方图（是否双峰？双峰的话，异常 token 占比是多少？）随层数的变化（类比 Registers Fig 3a）。
  3. 如果发现 GUI 截图里也存在类似的"高 norm patch"，**检查它们的空间位置**：是否集中在图像的均匀背景区域（类似 Registers 发现的"信息冗余区域"）？还是集中在 GUI 特有的位置（比如状态栏、固定的工具栏图标——这些在几乎每张截图里都高度相似，天然冗余，是 GUI 场景下 Registers 现象最可能出现的位置）？
  4. **关键的因果检验**：这些高 norm token 是否正是 probe 在做 BFS 阈值化（`activation_threshold=0.3`）时被系统性排除或系统性误选的 patch？如果高 norm token 恰好是低分空间信号但因为某种数值原因被 threshold 误纳入候选区域，这可能是 hit@1 < overlap@1 gap 的**另一个此前未被识别的根因**（目前 `.mrules` 只归因于"patch 量化误差"，这会是一个新发现）。
  5. 如果观测到这个现象，你甚至可以提出一个简单的**训练-free 干预**（呼应 Registers 论文"干预验证假设"的做法）：在 P2P 的候选区域筛选阶段，加一条"排除 norm 异常大的 patch"规则，看是否能略微提升 hit@1（这是一个正当的、纯分析驱动的、几行代码的干预，不是刷点 agent 在做的那种工程优化）。
- **产出**：一张类似 Registers Fig 2/3 的图（GUI 截图 + 高 norm patch 叠加可视化 + norm 分布直方图 + 逐层演化），可以成为论文里一个全新的、原创性很强的 Finding（"Finding 5" 或者作为 Finding 1 的补充证据），并且直接回答 4sok-Q4。
- **这条分析不需要与刷点 agent 协调**：纯只读分析，不改训练代码、不改推理主路径、不影响任何评测数字。

---

## 5. 你可以直接复用的代码资产（不要重新发明轮子）

| 需求 | 现成代码 |
|---|---|
| 加载任意 checkpoint 做推理 | `zwerge/probe/probe_utils.py::get_inference_class(model_type)` + `InfClass.from_checkpoint(...)` |
| 逐层 spatial hit@1/mass | `zwerge/probe/probe_utils.py::compute_spatial_metrics_from_pred`, `compute_target_mass` |
| Logit-lens 逐层 NLL（hook 实现） | `zwerge/probe/probe_utils.py::logit_lens_nll_hooks`（P0-2 直接扩展这个函数的调用方式） |
| 坐标 token 定位 | `zwerge/probe/probe_utils.py::find_coord_token_positions`, `_find_coord_by_subsequence` |
| Instruction-switch 反事实配对 | `zwerge/probe/probe_utils.py::select_counterfactual_pairs` + `zwerge/probe/exp4_counterfactual.py::_js_divergence` |
| Patch posterior → 点坐标（含多种策略） | `zwerge/eval/inference_base.py::get_prediction_region_point`, `decode_p2p`, `refine_local_mode`（P2-3 直接复用） |
| 无参数原生 attention probe（P1-1 用） | `GUI-AIMA/eval/layer_probe.py::compute_layerwise_signals`（Signal A）, `patch_scores_to_point` |
| Stage-1 probe 架构（做低容量变体的起点） | `zwerge/src/zwerge_retrofit/modeling_base.py::CrossAttnGroundingProbe`, `LayerGroundingProbe`（已有 LoRA 版本，容量介于线性和全秩 CrossAttn 之间，可以直接作为"中等容量"对照，不用再新写一个） |
| 训练脚本（做 P0-1/P0-3 的小规模训练） | `zwerge/train_retrofit.py` + `zwerge/scripts/train_ablation_A7_crossattn_probe.sh`（改 `--grounding_adapter_type lora` 就是较低容量版本，已经是现成的开关，见 `.mrules` 1522-1525 行） |
| 评测数据集统一格式 | `/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/datasets/evaluation/{bench}/eval.json`（见 `.mrules` 第 [2026-05-18] 节，含 5 个 benchmark 的统一 schema） |
| 模型权重路径 | `.hdd/models/huggingface.co/GUI_Agents/`、`.hdd/models/huggingface.co/Qwen/`（先 ls 确认目录名，不要凭记忆猜） |
| 已训练的 A7/A8 checkpoint | 通过 `zwerge/experiments/*.yaml` 里的 `experiment_name` 找到 `.hdd/ckpt/zwerge/{experiment_name}/checkpoint-*`；已知 A8 主 checkpoint 是 `checkpoint-2800`（4 个模型：uitars/guiowl7b/guiowl/uivenus 各一份，见 `.mrules` P2P 章节） |
| Hope job 提交（如果分析实验需要 GPU 批量跑） | `zwerge/eval_daemon.py`（异步评估架构，见 `.mrules` [2026-05-26] 节）；但更推荐直接用 `zwerge/probe/exp3_serialization_lens.py` / `exp4_counterfactual.py` 的 CLI 方式在单卡上跑分析脚本，不需要走完整的训练-评估 daemon 流程，因为分析实验通常样本量不大（几百到几千条），单卡几十分钟到几小时能跑完 |

---

## 6. 与刷点 agent 的协作边界（避免冲突）

- **不要修改**：`zwerge/eval/inference_base.py` 里 P2P 相关函数（`decode_p2p`/`region_score`/`consensus_score`/`refine_local_mode`/`run_p2p_*`）、`zwerge/eval/eval_retrofit.py` 的 `--decode_strategy` 相关逻辑、`4_method.tex` 的 `sec:method:p2p` 小节、`5_experiment.tex` 的主表/`tab:proposal_recall`/`tab:p2p_results`。这些都是刷点 agent 的地盘。
- **可以只读引用**：P2-3 直接调用 `refine_local_mode` 做分析（只读，不改），完全安全。
- **共享但要小心**：`.mrules` 变更日志——两个 agent 都要写，按时间顺序追加即可，不要互相覆盖对方的记录。
- **共享文件（需要留意冲突）**：`docs/our_paper_tex/secs/3_analysis.tex`（你主要负责）、`a_appendix.tex`（你和刷点 agent 都可能会碰——刷点 agent 之前动过 fusion claim 那部分，见 `.mrules` 2558 行；你如果要新增消融表/probe-capacity 图，加在其他小节，不要动 fusion claim 那几行，除非你重新核实后发现还有问题）。
- 如果你发现自己的分析实验结果与刷点 agent 报告的某个数字（比如 headroom、Qwen3 负优化）有冲突或者能提供互补解释，**在 `.mrules` 里明确记录，但不要擅自改动刷点 agent 负责的表格数字**，先记录发现，供后续统一决策。

---

## 7. 执行建议（给你自己做任务分解用，非强制顺序）

1. 先做 **P0-4**（OOD 泛化，纯数据分析，零训练成本，半小时到一小时就能出结果，立刻能用）。
2. 再做 **P0-1**（probe-capacity sweep，线性/共享 probe），因为这是回应最多条审稿意见的单一实验（4KeN-W2 + eKpD-W2 + eKpD-W8 都能部分被它回应）。
3. 并行做 **P0-2**（serialization lens 控制实验），这是 eKpD 打分最低的直接原因，逻辑上独立于 P0-1，可以另开一个进程/GPU 跑。
4. 做 **P0-3**（random control），复用 P0-1 的训练管线，增量成本很低。
5. 时间富余的话做 **P1** 系列（尤其 P1-1 无参数 attention baseline，性价比最高）。
6. 如果还有余裕，做 **P2-1**（attention sink/register 现象普查）——这是最有可能产出一个"意外惊喜"式新发现的实验，直接呼应用户想要的"审稿人看完分析部分后觉得已经很完备，然后发现还有方法涨点"的叙事效果，但反过来：这里是"分析部分已经很完备了，居然还挖出一个新机制"的效果。

完成每一项后，请在 `.mrules` 追加变更日志（复用现有格式：`- [日期] 【实验名】...`），并在论文对应 `.tex` 文件里落笔（哪怕先用简短占位段落 + TODO 注释，也要让 diff 可追踪）。

祝顺利。
