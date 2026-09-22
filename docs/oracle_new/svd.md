I


**我建议做 SVD 初始化，但看过代码后，我不建议第一版直接拿 FFN 的某个矩阵照搬 TMEM。更适合你当前架构的是：从预训练 attention 的 Q/K 匹配关系出发，做成对 SVD，初始化你现有的 `W_q`、`W_k`，其余架构先不动。**

你提出的“读的是本层末尾，因此应该借下一层的先验”**有明确的代码依据**。但这只能说明下一层是更自然的候选来源，不能提前保证训练效果更好；而“最后一层改用词表矩阵”会引入不同类型的读出先验，我不建议这样混用。

我已经看到了指定提交 `9329b271709aabceb54ca16224f75733116a9ee0` 及固定在该提交上的核心源码。容器无法通过 Git 拉取仓库，因此本地检查使用了你上传的 ZIP，并与在线核心源码交叉核对；不是拿旧版 README 推测架构。([GitHub][1])

下面先把架构、层索引和矩阵对应关系讲清楚，再给具体公式与修改位置。**初始化实现、运行时检查工具和已通过的 12 项 CPU 测试已经打包：**

[下载 SVD 初始化实现与代码审计包](sandbox:/mnt/data/ZwerGe_SVD_9329b27_review.zip)

这里验证的是矩阵关系、代码接入和小张量梯度，**还没有用你的预训练 checkpoint 跑 grounding 训练，不能把它当成效果已经得到验证的方法。**

---

## 一、你现在实际训练的架构是什么？

### 1. 当前 A7 不是“LoRA＋共享 MLP”，而是每层独立的 Q/K 读出器

代码里确实还保留着旧的：

* `LayerLoRAAdapter`
* `LayerGroundingProbe`
* `MLP2`

但你当前 A7 脚本选择的是 `adapter_type="attn"`，实际使用 **`CrossAttnGroundingProbe`**。旧版的低秩 adapter 和共享 MLP 不是这条训练路径的核心。([GitHub][2])

当前每个被探测层做的是：

```text
该层的完整 hidden states
          │
          ├── 取 <|ground|> 位置：h_query [d]
          │
          └── 取视觉 token：H_vis [N_vis, d]
                         │
               各自进行归一化
                         │
          ┌──────────────┴──────────────┐
          │                             │
      W_q: d → 512                  W_k: d → 512
          │                             │
      Q [8, 64]                  K [N_vis, 8, 64]
          └──────────────┬──────────────┘
                         │
                每个 head 做点积
                         │
               head_gate 加权 logits
                         │
                对视觉位置做 softmax
                         │
                   p_l [N_vis]
```

注意：**这里没有 V 投影、没有 O 投影，也没有把 attention 的输出加回 backbone。**它是一个针对 query–visual token 的匹配读出器，而不是另加一个完整 Transformer block。代码位置是：

`zwerge/src/zwerge_retrofit/modeling_base.py:199–275`。([GitHub][2])

这直接决定了你的初始化问题：

> **不是“给一个任意的新 MLP 找预训练权重”，而是“给两个新的匹配投影找已有的 query–key 几何关系”。**

### 2. 真正生效的维度是 512，不是脚本里的 1024 或 rank 16

当前默认：

$$
H_p=8,\qquad d_p=64,\qquad r=H_p d_p=512.
$$

因此，按 PyTorch 的 `[out_features, in_features]` 约定：

$$
W_q,W_k\in\mathbb R^{512\times d}.
$$

A7 脚本里的 `GROUNDING_PROJ_DIM=1024` 和 `GROUNDING_ADAPTER_RANK=16`，**在当前 attn 分支都不控制这两个矩阵的大小**；脚本本身也注明了这一点。不要误把 SVD rank 设成那个遗留的 16。([GitHub][3])

第一轮实验我建议保持：

$$
\boxed{r=512,\quad W_q,W_k\text{ 的形状和训练自由度完全不变。}}
$$

这样是在比较初始化，而不是同时比较不同模型容量。

### 3. 当前归一化和初始化也必须纳入考虑

你的输入不是直接乘 `W_q/W_k`，而是经过：

$$
h
\rightarrow \text{RMS 安全缩放}
\rightarrow \text{LayerNorm}
\rightarrow \text{再次 RMS 缩放}
\rightarrow W.
$$

两个投影目前独立采用 `Xavier(gain=0.02)`，`head_gate` 初始化为零，因此初始时 8 个 head 等权。**先加权各 head 的 logits，然后才做一次空间 softmax。**不是分别 softmax 后再平均概率。([GitHub][2])

这两个细节很重要：

**第一，预训练矩阵和新 probe 接收的归一化输入并不完全一样。**因此，“从下一层复制参数”不等于“完整复现下一层的 attention”。

**第二，随机初始化与预训练初始化的 logit 尺度可能差很多。**实验必须排除“只是 softmax 温度、梯度尺度不同”的解释，后面会给具体控制方法。

### 4. A8 的 LoRA 不是你这次应该初始化的对象

A8 的 `ContextLoRACosMetaFusion` 接收的是每个 probe 已经投影出来的：

$$
q_l\in\mathbb R^{512},
$$

不是原始的 \(d\) 维 backbone hidden state。它包含 query LoRA、跨层均值 context、context LoRA，最后用 cosine-meta scorer 给不同层分配权重。([GitHub][2])

所以不要看到 A8 里有 LoRA，就把 backbone FFN 的 SVD 因子往这里塞。**两者所在的特征空间不一致。**

这次应当只改：

$$
\boxed{\text{A7 各层 probe 的 }W_q,W_k\text{ 初始化。}}
$$

A7 训练后，A8 正常加载新 checkpoint。融合结构和融合初始化先保持原样。

---

## 二、你读的是本层开头还是末尾？应该用本层还是下一层？

### 1. 代码明确读取 `hidden_states[layer_idx + 1]`

当前读取逻辑的核心是：

```python
hs = all_hidden_states[layer_idx + 1]
```

随后从 `hs` 提取 anchor 和视觉 token。对应本地源码：

`modeling_base.py:481–489`。([GitHub][2])

对标准 Qwen2.5-VL 实现，除最后一个特殊端点外，这意味着：

$$
\text{你探测的 block }l\text{ 的输出}
=
\text{block }l+1\text{ 的输入}.
$$

例如，使用代码中的 **0-based 索引**：

| probe 配置              | 实际读取                            | 更自然的下一层 attention 来源 |
| --------------------- | ------------------------------- | -------------------- |
| `layer_idx=14`        | `hidden_states[15]`，block 14 之后 | block 15 的 Q/K       |
| `layer_idx=20`        | `hidden_states[21]`，block 20 之后 | block 21 的 Q/K       |
| `layer_idx=26`        | `hidden_states[27]`，block 26 之后 | block 27 的 Q/K       |
| `layer_idx=27`，共 28 层 | 最后一个 hidden-state 返回值           | 没有 block 28          |

官方实现是在进入各 decoder block 之前保存当前 hidden state，最后另行保存经过 final norm 的输出。([GitHub][4])

**所以，你关于“下一层读入先验”的直觉是成立的。**

但还要精确一点：

```text
block l 输出
    ↓
block l+1 的 input_layernorm
    ↓
block l+1 的 attention Q/K/V
    ↓
attention residual
    ↓
post_attention_layernorm
    ↓
block l+1 的 FFN
```

因此，**紧接着读这份表示的是下一层的 attention，不是下一层的 FFN。**这也是我优先选 Q/K 而不是 FFN 的第二个理由。

### 2. 本层初始化并不等于“重复执行一遍本层”

这里可以消除你的一个顾虑。

把 block \(l\) 的参数拿来初始化一个新 probe，并不是重新跑 block \(l\)。它只是借用一套已有的匹配方向；整个 block 的 attention、MLP、residual 都没有重做。

因此：

* **下一层 Q/K**：输入位置的对应关系更自然。
* **本层 Q/K**：也完全可以作为初始化；可能保留了有利于 grounding 的匹配结构。
* **哪个训练得更好**：需要比较，不能由层索引单独推出。

我的建议是：**实现同时支持 `same` 和 `next`，优先检验 `next`，但把 `same` 作为必做对照。**

### 3. 最后一层不能自动换成 `lm_head`

你的思路在“谁接着消费最后的表示”这个意义上是对的：最后确实接 final norm 和词表预测。

但你的新 probe 要做的是：

$$
\text{query token 与 visual token 的匹配},
$$

而 `lm_head` 做的是：

$$
\text{hidden state 与 vocabulary 的匹配}.
$$

它们不是同一类算子。

如果前面所有层都用 attention prior，唯独最后一层用 vocabulary prior，那么最后一层结果差时，很难区分：

> 是该层表示不好，还是你给它换了一种更偏语言输出的初始化？

这反而会重新引入你想排除的读出器偏差。

**第一版更稳妥的处理是：非末层用下一层 Q/K；末层明确回退到本层 Q/K，并把这个端点单独报告。**在论证“next 比 same 更合理”时，只比较两者都存在的共同层区间，不把末层当作完全等价的证据。

### 4. 这里还有一个应该先检查的 final norm 问题

这件事与层比较直接相关，不是旁枝。

标准 Qwen2.5-VL 返回的最后一个 `hidden_states`，**已经经过 final norm**。而你的 serialization lens 在 `probe_utils.py:420–429` 对所有选定层统一再调用一次 `norm(h_pos)`。两段代码组合起来，会让这个端点出现二次 final norm。([GitHub][4])

这不能当作必然无害：带学习权重的 RMSNorm 一般不是幂等运算。

我建议先用真实运行环境核查：

$$
\texttt{hidden\_states[l+1]}
\quad\text{分别等于}\quad
\begin{cases}
\text{block }l\text{ 的原始输出？}\\
\text{block }l+1\text{ 的输入？}\\
\text{final norm 的输出？}
\end{cases}
$$

Qwen3-VL 的 hidden-state 捕获路径与版本有关，不能机械套用 Qwen2.5 的结论。审计包提供了 hook 检查函数，专门比较这些张量。

**这不说明你的深度趋势一定有问题；它说明在用 SVD 增强论证之前，应该先统一端点的表示与归一化约定。**也不要静默改变旧 checkpoint 的输入约定后，直接与旧结果比较。

---

## 三、TMEM 的 SVD 到底做了什么？为什么不能原样搬过来？

TMEM 对选定的预训练 FFN 权重：

$$
W\in\mathbb R^{d_{\rm out}\times d_{\rm in}}
$$

做：

$$
W=U\Sigma V^\top,
$$

再初始化 LoRA：

$$
A_0=\Sigma_rV_r^\top,\qquad B_0=0.
$$

它保留原始 \(W\)，新增的是：

$$
W_{\rm effective}=W+B A.
$$

训练时固定 \(A\)、更新 \(B\)，因此初始增量为零，但更新所用的输入子空间来自预训练权重，而不是随机方向。([arXiv][5])

**你当前的新 probe 没有那个已经预训练好的底座投影 \(W\)。**

你的 `W_q/W_k` 本身就是新建的全部读出路径。因此不能：

> 把某个因子设零，然后认为整个读出器已经继承了非零的预训练行为。

特别是，**不要把新 Q 和 K 都初始化成零**；点积结构下，两侧都为零会使对应权重梯度也为零。

你应该借鉴的是：

> **用已有权重定义初始读取方向。**

不必继承“固定 \(A\)、只训 \(B\)”的参数化方式。第一轮保持现有两个 dense projection 全部可训练，比较才干净。

### 不同矩阵各自提供什么先验？

设 residual hidden 维度为 \(d\)，FFN 中间维度为 \(d_{\rm ff}\)。

| 来源矩阵                        |        PyTorch 权重形状 | 能提供什么                  | 对你的建议              |
| --------------------------- | ------------------: | ---------------------- | ------------------ |
| Attention `q_proj`、`k_proj` | `[Q维度,d]`、`[K维度,d]` | token 间 query–key 匹配关系 | **主方案，成对处理**       |
| FFN `gate_proj`、`up_proj`   |          `[d_ff,d]` | residual 输入空间中的特征方向    | 可做 TMEM 风格对照       |
| FFN `down_proj`             |          `[d,d_ff]` | FFN 特征到 residual 的输出方向 | 不能直接拿右奇异向量塞进 probe |
| `lm_head`                   |    `[vocab_size,d]` | 词表预测的输入方向              | 可研究，但别只给最后一层特殊使用   |

其中 `down_proj` 很容易用错：

$$
V_r^\top\in\mathbb R^{r\times d_{\rm ff}},
$$

而你的 probe 需要：

$$
W_q,W_k\in\mathbb R^{512\times d}.
$$

维度就不对应。`down_proj` 的左奇异向量 \(U_r\) 才位于 \(d\) 维 residual 输出空间，但那已经是另一种先验。

**FFN 并不是不能用，只是不能因为 TMEM 用 FFN，就跳过“你的新模块究竟在做什么”这个对应关系。**

---

## 四、我建议的具体方法：对 Q/K 的匹配算子做成对 SVD

下面是基于你当前代码推导的候选初始化，**不是 TMEM 原文的方法，也不是已经验证有精度提升的结果。**

### 1. 为什么不能随便对 Q、K 分别 SVD？

假设原生投影分别为：

$$
Q=U_Q\Sigma_QV_Q^\top,\qquad
K=U_K\Sigma_KV_K^\top.
$$

它们的点积关系包含：

$$
Q^\top K
=
V_Q\Sigma_Q
\underbrace{U_Q^\top U_K}_{\text{两侧输出空间的配对关系}}
\Sigma_KV_K^\top.
$$

如果分别保留两套右奇异向量，再直接把新输出相乘，中间这项通常就丢了。

也就是说：

> **Q 和 K 各自保留较大的奇异方向，不代表保留了 Q–K 原本如何匹配。**

你真正应该保留的是两者共同定义的关系。

### 2. 从选定的原生 attention 构造一个匹配矩阵

设来源 attention 有 \(H\) 个 query heads，原生 head 维度为 \(d_0\)。考虑 GQA，query head \(h\) 使用的 key head 记为 \(g(h)\)。

定义：

$$
\boxed{
M
=
\frac{1}{H\sqrt{d_0}}
\sum_{h=1}^{H}
\left(W^{\rm native}_{Q,h}\right)^\top
W^{\rm native}_{K,g(h)}
}
$$

于是：

$$
M\in\mathbb R^{d\times d}.
$$

它描述一个明确、但有所简化的对象：

> **只考虑权重、忽略 RoPE 等操作时，原生各 head 平均的内容匹配 logits。**

这里取的是完整预训练 checkpoint 中的 `q_proj.weight` 与 `k_proj.weight`，不是 LoRA 增量，也不是你随机初始化的新 head。

GQA 必须正确处理，不能直接对整个 `q_proj.weight.T @ k_proj.weight` 做乘法：两者的 head 数量及投影行数可能不同。

### 3. 对这个矩阵做 rank-512 SVD

$$
M\approx M_r=U_r\Sigma_rV_r^\top,\qquad r=512.
$$

接下来把这个匹配关系写进现有 probe。

你当前的 `head_gate` 初始均匀，因此其初始 logits 可以写成：

$$
s(x,y)
=
\frac{1}{H_p\sqrt{d_p}}
x^\top W_q^\top W_k y,
$$

这里 \(x,y\) 是**经过当前 probe 归一化后的** query 和 visual feature。

令：

$$
c=H_p\sqrt{d_p}=8\sqrt{64}=64,
$$

设置：

$$
\boxed{
W_{q,0}
=
\sqrt c\,\Sigma_r^{1/2}U_r^\top
}
$$

$$
\boxed{
W_{k,0}
=
\sqrt c\,\Sigma_r^{1/2}V_r^\top
}
$$

两个矩阵的形状都正好是：

$$
[512,d].
$$

这样就有：

$$
\frac{1}{c}W_{q,0}^\top W_{k,0}
=
U_r\Sigma_rV_r^\top=M_r.
$$

**这就是完整的“从哪个矩阵做 SVD，再把结果放到哪里”。**

```text
选定来源 block 的原生 q_proj、k_proj
                 ↓
      按原生 head / GQA 对齐
                 ↓
   构造 query–key 内容匹配算子 M
                 ↓
       rank-512 的成对分解
                 ↓
        W_q ← sqrt(cΣ) Uᵀ
        W_k ← sqrt(cΣ) Vᵀ
                 ↓
       现有 A7 训练，两个 W 都更新
```

**不改 backbone，不新增一层 MLP，不新增低秩分支，不改变 probe 参数量。**

### 4. 不必真的构造巨大方阵再做完整 SVD

公式里 \(M\) 是 \(d\times d\)，但实现可以利用：

$$
M=\alpha A^\top B.
$$

对 \(A^\top,B^\top\) 分别做 reduced QR，再对较小的核心矩阵做 SVD，最后还原左右奇异向量。

提供的代码就是这样实现的，并利用 GQA 分组避免重复展开所有 key heads。

另外，`full_matrices=False` 指的是使用经济型 SVD，**不意味着只分解了部分预训练权重**。不要为了“全量矩阵”四个字去申请没有必要的巨大 \(U,V\)。

### 5. 这个方案保留什么，又没有保留什么？

保留的是：

$$
\text{一个预训练的、低秩近似后的 query–key 内容匹配关系。}
$$

**没有完整保留原生 attention。**第一版实现没有纳入原生输入 RMSNorm、Q/K bias、Q/K normalization、RoPE 和各 head 的独立 softmax。Qwen3-VL 原生 attention 包含 Q/K normalization，因此它的实际打分函数不能仅由一个固定的 \(Q^\top K\) 矩阵精确表示。([GitHub][6])

另外，平均 logits 也不等于平均 attention probabilities；平均不同 head 可能抵消一些有用的专门化信号。

所以我建议把它称为：

> **预训练 QK 匹配算子的谱初始化。**

不要声称“完整无损迁移了这一层 attention 知识”。

它的数学保证只是：对于**我们定义的这个 \(M\)**，截断 SVD 提供最优的固定秩 Frobenius 近似。这个保证不是 grounding 精度保证，更不是泛化保证。

---

## 五、真的要用 FFN 做一个 TMEM 式版本，该怎么做？

这个可以保留为一个很清楚的对照，不必先否定。

例如取来源层：

```python
layers[j].mlp.gate_proj.weight
```

其形状：

$$
W_{\rm gate}\in\mathbb R^{d_{\rm ff}\times d}.
$$

做 SVD 后取：

$$
P=\Sigma_rV_r^\top\in\mathbb R^{512\times d}.
$$

一个形状正确、关系明确的初始化是：

$$
W_{q,0}=P,\qquad W_{k,0}=P,
$$

再控制尺度，之后允许两者独立训练。

它初始定义的相似度是：

$$
x^\top P^\top Py
=
x^\top V_r\Sigma_r^2V_r^\top y.
$$

这表示：

> **在该 FFN 最敏感的一组输入方向上，比较 query 与 visual feature。**

它是一个对称、半正定的相似度，不是从原生 attention 继承的不对称 query–key 匹配关系。

因此两个方案可以这样区分：

| 方案         | 初始化时继承的东西                        |
| ---------- | -------------------------------- |
| FFN 右奇异子空间 | “这组 residual 特征方向在预训练 FFN 中比较重要” |
| 成对 QK-SVD  | “原生 query 与 key 通过这些方向相互匹配”      |

**针对你现在的模块功能，我优先测试后者；前者作为 TMEM 风格对照很合适。**没有训练结果之前，我不会断言后者最终精度一定更高。

---

## 六、具体改代码的位置：有一个覆盖初始化的坑必须避开

### 1. SVD 必须放在 `setup_special_token_ids` 之后

你的 `reinit_grounding_head()` 会再次把 `W_q/W_k` 重置成 Xavier；而 `setup_special_token_ids(..., reinit_grounding_head=True)` 会触发这个过程。

实际调用在：

`zwerge/train_retrofit.py:700–705`。([GitHub][2])

因此，下面这种顺序是错的：

```text
from_pretrained
    ↓
SVD 初始化
    ↓
setup_special_token_ids(reinit=True)
    ↓
SVD 结果被覆盖
```

正确顺序应该是：

```text
加载预训练模型
    ↓
扩展 tokenizer / embedding
    ↓
setup_special_token_ids，完成默认重置
    ↓
【这里做 SVD 初始化】
    ↓
设置可训练参数
    ↓
创建 optimizer
    ↓
开始 A7
```

### 2. 提供的初始化函数可以这样调用

把包里的 `paired_qk_svd.py` 放到：

`zwerge/src/zwerge_retrofit/`

然后在**新增的 fresh A7-SVD 初始化分支**中，默认重置完成后调用：

```python
from zwerge_retrofit.paired_qk_svd import initialize_a7_probes

svd_report = initialize_a7_probes(
    model,
    donor_mode="next",       # 对照实验改成 "same"
    terminal_policy="same",  # 末层回退会明确写入 report
    device="cpu",           # 也可在可用 GPU 上一次性预计算
    logit_gain=1.0,
)
```

这是新增接入示例，**不是仓库已经存在的 CLI 选项**。

外层必须检查：不是 resume、不是 A8、确实是 fresh head。不要替换已经训练过的 probe 后继续沿用旧 optimizer momentum。

来源索引也不能直接用 `ModuleList` 下标。你的 `probes[0]` 可能对应 backbone 的 block 14，不是 block 0。函数依据实际 `head.probe_layers` 做映射。

### 3. 第一版不要冻结 SVD 因子

建议初始化后仍然训练现有的：

* `W_q/W_k`
* probe 的归一化参数
* `head_gate`

不要同时改成“只训练奇异值”或者“冻结左右子空间”。那会把初始化实验变成新的低秩约束实验。

### 4. A8 保持现有 checkpoint 继承逻辑

A8 原脚本已经设置不重置 A7 probe，并从 A7 checkpoint 加载。你只需把来源换成新的 A7-SVD checkpoint；这轮不要再给 A8 fusion 做 SVD。([GitHub][7])

分布式训练时，建议**单进程预计算一次，保存初始化后的 head 权重及来源记录，各 rank 加载同一份**。没必要每个 rank 重复分解，也不需要复制整个 backbone checkpoint。

---

## 七、SVD 能不能增强你的“中间层更适合 grounding”论证？

**能补强，但不是因为“初始化来自本层，所以本层读出一定充分”。**

你现在的动机应该更准确地写成：

> 随机读出器需要通过有限 grounding 监督重新学习读取几何；预训练谱初始化可能减少这一负担，使层间比较对优化起点更稳健。

这比“随机头没有任何先验、SVD 后就有全部先验”更准确。随机头接收的 backbone 特征本来就有预训练知识；欠缺的是**模型特定的读出先验**。

而且，预训练权重中最大的奇异方向未必恰好是 grounding 最需要的方向。TMEM 自己关于近似优势的理论，也依赖目标更新与预训练子空间的对齐条件，不是无条件保证。([arXiv][5])

### 1. 非各向同性没有问题，但它引入了新的层特定偏置

你的担心是对的：SVD 初始化不是各向同性初始化。

这不是缺点本身——**你正是为了引入有方向性的先验**。问题在于，不能用它替代所有公平性控制。

各层来源权重的奇异值谱、匹配结构不同，会影响初始 logits、梯度和优化速度。因此，真正有说服力的不是只画一条 SVD 后的层曲线，而是：

$$
\boxed{
\text{中间层峰值是否在多种初始化与充分优化条件下保持？}
}
$$

### 2. 我建议先做四组，而不是上来全模型全面铺开

| 组别 | 初始化               | 回答的问题             |
| -- | ----------------- | ----------------- |
| A  | 当前 Xavier         | 原始结果基线            |
| B  | 与预训练谱匹配，但随机旋转左右方向 | 收益来自方向先验，还是谱与条件数？ |
| C  | 本层 paired QK-SVD  | 本层参数作为读出先验是否有效？   |
| D  | 下一层 paired QK-SVD | 输入位置对齐是否更有帮助？     |

B、C、D 尽量再匹配初始 logit 尺度。需要时给 A 增加同尺度版本。

**初始尺度用训练集的一小批无标签 hidden states 来测，不用测试集。**记录空间 logits 的去均值 RMS 和分布熵；不要只匹配权重 Frobenius norm，因为不同层特征分布也不同。

包里的 `random_orientation_seed` 和 `match_probe_logit_rms` 就分别支持这两个控制。

### 3. 还有一个“冻结 backbone 不等于固定表示”的细节

你的训练入口允许新加入的 special-token embeddings 训练，并有专门的新 token 参数与替换路径。因此，不能仅凭 backbone 大部分权重冻结，就把所有 probe 输入视为始终固定。([GitHub][8])

用于**层间机制诊断**的第一轮，我建议冻结这些新 token embedding，或让各初始化组使用同一份固定 embedding checkpoint。最终追求精度的训练可以继续联合优化，但不要把两种实验解释混在一起。

### 4. 看学习曲线，而不是只看一个 epoch 的终点

先挑同一 backbone 的中间层、较深层、倒数第二层，末层单独处理。记录初始性能、早期学习速度、验证集曲线和较长预算后的结果。

结果可能有不同解释：

**SVD 只让早期收敛更快，最终随机初始化追上。**
说明它主要提高训练效率，不能解释成扩大了可达到的能力边界。

**SVD 在未见数据上持续更好，随机方向谱匹配组不能复现。**
这才更支持预训练方向本身提供了有用先验。

**深层在 SVD 或更长训练后追上中间层。**
那就需要相应收紧原先“中间层本征更优”的解释。这也是有价值的研究结果，不应为了保住原叙事回避。

**多种初始化、预算下都保持中间层峰值。**
这时你可以更有底气地说：深度差异不是单一随机初始化或有限训练预算的偶然产物。

---

## 八、我建议你现在就采用的版本

把这轮工作严格限定为：

$$
\boxed{
\begin{gathered}
\text{核实 hidden-state 与 final norm 约定}\\
\downarrow\\
\text{从原生 Q/K 匹配算子做 paired SVD}\\
\downarrow\\
\text{初始化现有 }512\times d\text{ 的 }W_q,W_k\\
\downarrow\\
\text{保留原参数量、原监督目标、原训练自由度}\\
\downarrow\\
\text{比较 same / next / random，再进入 A8}
\end{gathered}
}
$$

**暂时不改 FFN、不改 fusion、不把末层换成词表读出、不顺手加新的 loss。**FFN-SVD 留作一个明确的来源对照即可。

实现包中的 12 项测试已通过，包括 GQA 对齐、成对重建、截断近似、你仓库原始 probe 类的小张量前向与梯度、尺度匹配、层映射、末层处理和防止覆盖 A8。运行时 hidden-state 审计目前只用测试模型验证了工具逻辑，真实 Qwen checkpoint 仍需在你的环境执行。

[初始化与审计完整包](sandbox:/mnt/data/ZwerGe_SVD_9329b27_review.zip) · [单独查看初始化代码](sandbox:/mnt/data/zwerge_svd_review/deliverable/paired_qk_svd.py)

**最终我的判断是：你的 SVD 动机值得做；当前最需要继承的不是泛泛的“这一层有知识”，而是“已有模型如何把一个 query 与视觉 token 匹配起来”。在你的实际架构里，Q/K 的成对关系比任意一个 FFN 矩阵更直接；下一层是合理的首选假设，但必须让本层对照和学习曲线来决定它是否真的更好。**

[1]: https://github.com/HashTAG00002/zwergeUI/commit/9329b271709aabceb54ca16224f75733116a9ee0 "https://github.com/HashTAG00002/zwergeUI/commit/9329b271709aabceb54ca16224f75733116a9ee0"
[2]: https://raw.githubusercontent.com/HashTAG00002/zwergeUI/9329b271709aabceb54ca16224f75733116a9ee0/zwerge/src/zwerge_retrofit/modeling_base.py "https://raw.githubusercontent.com/HashTAG00002/zwergeUI/9329b271709aabceb54ca16224f75733116a9ee0/zwerge/src/zwerge_retrofit/modeling_base.py"
[3]: https://raw.githubusercontent.com/HashTAG00002/zwergeUI/9329b271709aabceb54ca16224f75733116a9ee0/zwerge/scripts/train_ablation_A7_crossattn_probe.sh "https://raw.githubusercontent.com/HashTAG00002/zwergeUI/9329b271709aabceb54ca16224f75733116a9ee0/zwerge/scripts/train_ablation_A7_crossattn_probe.sh"
[4]: https://raw.githubusercontent.com/huggingface/transformers/v4.51.3/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py "https://raw.githubusercontent.com/huggingface/transformers/v4.51.3/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py"
[5]: https://arxiv.org/html/2606.04536v1 "https://arxiv.org/html/2606.04536v1"
[6]: https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py "https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py"
[7]: https://raw.githubusercontent.com/HashTAG00002/zwergeUI/9329b271709aabceb54ca16224f75733116a9ee0/zwerge/scripts/train_ablation_A8_cosmeta_context.sh "https://raw.githubusercontent.com/HashTAG00002/zwergeUI/9329b271709aabceb54ca16224f75733116a9ee0/zwerge/scripts/train_ablation_A8_cosmeta_context.sh"
[8]: https://raw.githubusercontent.com/HashTAG00002/zwergeUI/9329b271709aabceb54ca16224f75733116a9ee0/zwerge/train_retrofit.py "https://raw.githubusercontent.com/HashTAG00002/zwergeUI/9329b271709aabceb54ca16224f75733116a9ee0/zwerge/train_retrofit.py"
