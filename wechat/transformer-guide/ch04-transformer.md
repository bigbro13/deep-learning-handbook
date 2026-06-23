# Transformer 架构完全拆解：从 RNN 到注意力机制的演进

> 本文是「Transformer 完全拆解」系列之一。从 RNN 的硬伤讲起，逐步拆解 Self-Attention、多头注意力、位置编码、Encoder/Decoder Block、训练与推理的不对称，最后给出 PyTorch 极简实现，帮你把 Transformer 的每一块骨头都啃干净。
>
> 下一篇：[ViT 与 Swin Transformer]()

---

# Ch4 — Transformer 架构

## 4.1 为什么需要 Transformer

### 4.1.1 从 RNN 谈起

在 Transformer 出现之前,处理序列数据(机器翻译、语音识别、文本摘要)的主力是循环神经网络(RNN)及其变体 LSTM、GRU。RNN 的核心思想很朴素:**把序列一个时间步一个时间步地喂给网络,网络维护一个不断更新的隐藏状态 $h_t$,把过去的信息"记忆"在里面**。

![image-20260622150315079](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch04/RNN.png)

如图所示，以处理自然语言句子“我 爱 人工 智能”为例，输入序列被转化为向量 $x_1, x_2, x_3, x_4$。RNN 的核心思想是引入了**循环结构**，使得信息能够在时间步之间传递。

**计算机制**：在每一个时间步 $t$，RNN 单元（RNN Cell）接收当前时刻的输入 $x_t$ 以及来自前一时刻的隐状态（Hidden State）$h_{t-1}$。从**图2**的局部放大细节可以看出，模型将当前输入与前一状态进行融合（图中简化为相加），并通过一个非线性激活函数（通常为 $\tanh$）进行映射，从而计算出当前时刻的隐状态 $h_t$。这个隐状态 $h_t$ 不仅作为当前步的输出，还会被传递至下一个时间步，成为计算 $h_{t+1}$ 的基础。

数学上 RNN 的更新可以写成:

$$h_t = \tanh(W_h h_{t-1} + W_x x_t + b)$$

这种"信息流像水流一样从左向右递推"的设计,在 NLP 任务上跑了将近 30 年。但它有三个绕不开的硬伤,这些硬伤直接催生了 Transformer。

**硬伤一:无法并行**

RNN 计算 $h_t$ 必须先有 $h_{t-1}$,而 $h_{t-1}$ 又依赖 $h_{t-2}$……这种串行依赖让 GPU 这种擅长大规模并行的硬件几乎无用武之地。一个长度 100 的句子,你必须老老实实跑 100 步,前一步没算完后一步动不了。这在小模型时代还能忍,但当我们想训练亿级、千亿级参数的模型时,串行计算成了致命瓶颈。

**硬伤二:长程依赖难以捕捉**

理论上,RNN 的隐藏状态 $h_t$ 编码了从 $x_1$ 到 $x_t$ 的所有历史信息;但实际上,信息每经过一次 $W_h$ 矩阵相乘就被"挤压、扭曲"一次。当序列长度达到几十、上百个 token 时,早期信息已经被反复挤压得几乎无法分辨。LSTM、GRU 通过门控机制缓解了这一问题(下一节我们会展开讲),但本质上没有解决——只是把"指数级遗忘"变得"温柔一点"。

**硬伤三:每个 token 只能看到上文**

标准 RNN 在 $t$ 时刻只能看到 $x_1, \ldots, x_t$,看不到 $x_{t+1}, \ldots, x_T$。双向 RNN(BiRNN)通过反向再跑一遍来弥补,但本质上还是两个单向流的拼接,不是真正意义上的"全局信息融合"。

> RNN 的核心矛盾:**串行计算让它跑不快,信息挤压让它记不住,单向递推让它看不全**。

### 4.1.2 LSTM:用门控缓解长程依赖

为了缓解标准 RNN 无法保留长距离信息的缺陷，Hochreiter 和 Schmidhuber 于 1997 年提出了长短期记忆网络（Long Short-Term Memory, LSTM），这也是循环神经网络发展史上的一座重要里程碑。![lstm](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch04/lstm.png)

如图所示，在处理相同的文本序列时，LSTM 在结构上进行了重大改进。最显著的变化在于，它在隐状态 $h_t$（可视为短期记忆）的基础上，引入了一个贯穿整个时间序列的**细胞状态（Cell State, $c_t$）**，充当长期记忆的载体。

详细展示了 LSTM Cell 内部精细的**门控机制（Gated Mechanism）**。LSTM 主要通过以下三个门来保护和控制细胞状态：![lstm-cell](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch04/lstm-cell.png)

1.  **遗忘门（Forget Gate, $f_t$）**：如**图1**中最左侧的 $\sigma$（Sigmoid）层所示。它接收前一时刻的隐状态 $h_{t-1}$ 和当前输入 $x_t$，输出一个介于 $0$ 到 $1$ 之间的数值，用以决定前一时刻的细胞状态 $c_{t-1}$ 中有多少信息需要被保留或遗忘。
2.  **输入门（Input Gate, $i_t$）与候选细胞状态**：中间的 $\sigma$ 层为输入门，决定哪些新信息将被选择性更新；旁边的 $\tanh$ 层则用于生成新的候选细胞状态。两者经过逐元素相乘（$\otimes$）后，与遗忘门处理后的旧细胞状态相加，从而更新得到当前时刻的细胞状态 $c_t$（即**图1**中最上方的水平主线通道）。
3.  **输出门（Output Gate, $o_t$）**：最右侧的 $\sigma$ 层决定细胞状态的哪些部分可以作为当前的隐状态 $h_t$ 输出。更新后的细胞状态 $c_t$ 经过 $\tanh$ 函数激活后，与输出门 $o_t$ 的结果相乘（$\otimes$），最终产生当前的隐状态 $h_t$。

通过这种设计，细胞状态 $c_t$ 类似于一条传送带，信息可以相对无损地在时间通道中流动，从而显著缓解了长序列训练中的梯度消失问题。

**但 LSTM 没有彻底解决问题,只是"把指数级遗忘变得温柔一点":**

- **串行计算的瓶颈依旧**。LSTM 仍然是 RNN 结构,第 $t$ 步必须等第 $t-1$ 步算完,GPU 并行优势用不上。这成了大模型时代的硬伤。
- **长程依赖只是缓解,不是根治**。当序列长到几百、上千,即使有细胞状态这条高速公路,遗忘门也会在反复更新中逐渐"磨损"早期信息,长程依赖的精度仍然下降。
- **参数量是同维度 RNN 的 4 倍左右**。每个时间步都要走一遍完整的门控计算(四个线性变换 + 三个激活),训练和推理都更慢。

GRU(Cho 等 2014)是 LSTM 的简化版——把遗忘门和输入门合成一个"更新门",并合并细胞状态和隐藏状态,参数更少、效果接近。但本质上仍是门控 RNN,串行这一根本问题没动。

> LSTM 用门控和加性更新让 RNN "记得更久、训得更深",但**串行计算的根本瓶颈没动**——这正是 Transformer 用注意力替代循环要解决的核心问题。

### 4.1.3 CNN 也想替代 RNN,但只走了半步

在 Transformer 之前,业界还有另一条尝试路线——用 CNN 处理序列。代表作是 Facebook 2017 年的 ConvS2S。CNN 的优势很明显:卷积天然并行,GPU 利用率拉满。

但 CNN 也有自己的问题:**单层卷积的感受野(Receptive Field)受限于 kernel size**。要捕捉跨度 100 个 token 的依赖关系,要么用极大的 kernel(参数爆炸),要么堆很多层(梯度衰减再次出现),要么用空洞卷积(信息有缝隙)。CNN 在序列上表现尚可,但始终没能彻底击败 RNN。

### 4.1.4 Transformer 的设计哲学

2017 年 6 月,Google 的 Vaswani 团队发表了那篇划时代的论文——《Attention Is All You Need》。论文提出的 Transformer 架构做了一件极其大胆的事:**完全抛弃循环和卷积,只用注意力机制**。

这个设计有三个核心 idea:

第一,**用注意力机制实现"任意两个位置直接交互"**。不论两个 token 距离多远,它们之间都只有一次矩阵运算的距离,长程依赖一步到位。

第二,**所有位置的计算可以并行**。第 5 个 token 算注意力时,不需要等第 4 个 token 算完——所有 token 同时参与一次大矩阵乘法即可。

第三,**用位置编码补偿"丢失的顺序信息"**。注意力机制本身是位置无关的(打乱输入顺序结果一样),所以需要显式注入位置信息。

> Transformer 的核心创新可以用一句话概括:**任意两个 token 之间的"距离"都是 1,且所有交互可并行计算**。

接下来我们就一步步拆解,看 Transformer 是怎么用注意力机制做到这一点的。

## 4.2 注意力机制的本质

### 4.2.1 一个生活化的类比

理解注意力机制,最直观的类比是**查字典**。

假设你有一本字典(数据库),里面存了一堆"键值对"——每个键(Key)是一个词条索引,每个值(Value)是该词条对应的解释。现在你拿着一个查询(Query),想从字典里找到相关的内容。

最简单的查法是精确匹配:Query 必须和某个 Key 完全相同,才返回对应的 Value。但现实世界很少这么干净——你想查"机器学习"的解释,字典里只有"AI"、"深度学习"、"统计学"、"算法"几个 Key。这时候你会怎么办?

**你会综合所有 Key 的相关程度,加权融合所有 Value**:

- "AI" 和"机器学习"高度相关,权重 0.4
- "深度学习" 和"机器学习"高度相关,权重 0.4
- "统计学" 和"机器学习"中等相关,权重 0.15
- "算法" 和"机器学习"较弱相关,权重 0.05

最终返回的"答案" = 0.4 × Value(AI) + 0.4 × Value(深度学习) + 0.15 × Value(统计学) + 0.05 × Value(算法)。

这就是注意力机制的全部直觉:

$$\text{Attention}(Q, K, V) = \sum_i \alpha_i \cdot V_i, \quad \alpha_i = \text{normalize}(\text{sim}(Q, K_i))$$

> 注意力 = **基于相似度的加权求和**。Query 决定"要找什么",Key 决定"我是谁",Value 决定"我能贡献什么"。

### 4.2.2 从加性注意力到点积注意力

衡量 Query 和 Key 相似度的方法有很多,Transformer 之前最常用的是 Bahdanau 提出的**加性注意力**(Additive Attention):

$$\text{score}(Q, K) = v^T \tanh(W_q Q + W_k K)$$

它用一个小 MLP 来打分,表达力强但计算慢——每对 (Q, K) 都要走一遍 MLP。

Transformer 选择了更简洁的**点积注意力**(Dot-Product Attention):

$$\text{score}(Q, K) = Q \cdot K^T$$

直接做向量点积。两个向量越"同向",点积越大,相似度越高。点积注意力的最大优势是**可以批量化为矩阵乘法**,这正是 GPU 的强项。

所以 Transformer 选择了"用点积换速度"——但点积有个副作用:数值不稳定。下一节我们会看到 Transformer 怎么解决这个问题。

### 4.2.3 Self-Attention vs Cross-Attention

注意力机制有两种工作模式,后面会反复用到,这里先建立概念。

**Self-Attention(自注意力)**:Q、K、V 都来自**同一个序列**。比如在编码 "The cat sat on the mat" 时,每个词都要去"看看"句子里其他词,理解自己在上下文中的角色。这是 Transformer 编码器和解码器内部都用到的核心机制。

**Cross-Attention(交叉注意力)**:Q 来自一个序列,K 和 V 来自**另一个序列**。比如机器翻译时,解码器生成中文时要去"看"编码器输出的英文表示,这就是 Cross-Attention 的用武之地。

![self-vs-cross-attention](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch04/self-vs-cross-attention.png)

> Self-Attention 让序列"自己看自己",理解内部依赖;Cross-Attention 让一个序列"看另一个序列",建立两个序列间的映射。

## 4.3 Self-Attention 拆解

这一节我们把 Self-Attention 的计算过程一步步拆开。建议读这一节时手边备张纸,跟着推一遍——这是 Transformer 最核心的部分,值得花时间。



### 4.3.1 输入设定

假设我们要处理的句子是 "Thinking Machines",经过分词和 Embedding 后变成两个向量:

- $x_1$ = Embedding("Thinking"), 维度 $d_{model} = 512$
- $x_2$ = Embedding("Machines"), 维度 $d_{model} = 512$

我们的目标是:为每个 token 输出一个新的、融合了上下文信息的向量 $z_1, z_2$。

### 4.3.2 第一步:从输入向量生成 Q、K、V

Self-Attention 的第一步,是给每个输入向量算出三个"角色":

- **Query 向量 $q$**:代表"当前这个 token 要去问什么"
- **Key 向量 $k$**:代表"当前这个 token 用来被别人查询的'索引'"
- **Value 向量 $v$**:代表"当前这个 token 实际能贡献的内容"

这三个向量都通过对原始输入做线性变换得到——三个不同的权重矩阵 $W^Q, W^K, W^V$ 是模型可学习的参数:

$$q_i = x_i W^Q, \quad k_i = x_i W^K, \quad v_i = x_i W^V$$

注意一个细节:在原始论文中,$x_i$ 的维度是 $d_{model} = 512$,但 $q, k, v$ 的维度是 $d_k = 64$。这是因为后面会用 8 个头(head)的多头注意力,每个头独立做一份 64 维的注意力计算,$8 \times 64 = 512$ 维度刚好对得上。

![qkv-projection](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch04/qkv-projection.png)

> $W^Q, W^K, W^V$ 是 Self-Attention 中**唯一需要学习**的核心参数。模型在训练中学到的就是"如何把同一个输入投影成三种不同的角色"。

### 4.3.3 第二步:计算注意力分数

要为 token $i$ 计算它对其他 token 的关注程度,就用 $q_i$ 去和**所有** token 的 $k_j$ 做点积:

$$\text{score}(i, j) = q_i \cdot k_j$$

以"Thinking Machines"为例,我们要算 $q_1$(代表 Thinking)对每个 token 的关注度:

- $\text{score}(1, 1) = q_1 \cdot k_1$ (Thinking 对自己的关注)
- $\text{score}(1, 2) = q_1 \cdot k_2$ (Thinking 对 Machines 的关注)

直觉上,如果 $q_1$ 和 $k_2$ 在向量空间中"指向相似的方向",点积就大,说明 Thinking 想从 Machines 那里获取信息;反之点积小,说明这两个 token 关系不大。

### 4.3.4 第三步:除以 $\sqrt{d_k}$ 进行缩放

直接拿原始点积去做 softmax,会遇到一个隐蔽但严重的问题——**点积的方差会随着维度 $d_k$ 增长而变大**。我们来推一下。

假设 $q$ 和 $k$ 的每个分量都独立同分布,均值为 0、方差为 1。那么点积:

$$q \cdot k = \sum_{i=1}^{d_k} q_i k_i$$

的方差是:

$$\text{Var}(q \cdot k) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) = d_k$$

(用了 $\text{Var}(XY) = \text{Var}(X)\text{Var}(Y) = 1$,以及独立项方差可加。)

也就是说,$d_k$ 越大,点积的数值波动越剧烈。当 $d_k = 64$ 时,点积的标准差约为 $\sqrt{64} = 8$,数值动辄到几十。把这种数值丢给 softmax 会怎样?

softmax 是 $e^x / \sum e^x$。当某个 $x$ 比其他大很多时,它对应的概率会**急剧逼近 1**,其他位置的概率几乎为 0。这种极端的概率分布会让梯度变得非常小(因为 softmax 的梯度在饱和区接近 0),训练直接卡住。

解决方法很简单:**给点积除以 $\sqrt{d_k}$**,把方差拉回到 1 附近:

$$\text{Var}\left(\frac{q \cdot k}{\sqrt{d_k}}\right) = \frac{d_k}{d_k} = 1$$

> $\sqrt{d_k}$ 缩放的本质是**温度调节**——避免 softmax 输入过大导致梯度消失。这是 Transformer 的一个看似不起眼但极其关键的设计。

### 4.3.5 第四步:Softmax 归一化

把缩放后的分数过一遍 softmax,得到每个位置的注意力权重:

$$\alpha_{ij} = \text{softmax}_j\left(\frac{q_i \cdot k_j}{\sqrt{d_k}}\right) = \frac{\exp(q_i \cdot k_j / \sqrt{d_k})}{\sum_m \exp(q_i \cdot k_m / \sqrt{d_k})}$$

这一步保证所有 $\alpha_{ij}$(对固定的 $i$,所有 $j$)加起来等于 1,可以解读为概率分布。

### 4.3.6 第五步:用注意力权重加权 Value

最后一步,用刚算出的权重对所有 Value 加权求和,得到 token $i$ 的输出表示:

$$z_i = \sum_j \alpha_{ij} v_j$$

到这里,Self-Attention 一次完整的计算就结束了。让我们把六步串起来再看一遍流程图:

![image-20260623145545258](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch04/attention.png)

### 4.3.7 矩阵化:把所有 token 一次算完

逐 token 计算太慢了。实际实现时,我们把所有 token 拼成矩阵,一次性算出所有 token 的输出。

设输入序列有 $n$ 个 token,每个 token 是 $d_{model}$ 维向量,堆成矩阵 $X \in \mathbb{R}^{n \times d_{model}}$​。一步矩阵乘法就能拿到所有 token 的 Q、K、V:

![image-20260623145948593](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch04/QKV.png)

$$Q = X W^Q, \quad K = X W^K, \quad V = X W^V$$

其中 $W^Q, W^K \in \mathbb{R}^{d_{model} \times d_k}$,$W^V \in \mathbb{R}^{d_{model} \times d_v}$。结果矩阵 $Q, K \in \mathbb{R}^{n \times d_k}$,$V \in \mathbb{R}^{n \times d_v}$。

接着一次性算出所有的注意力分数矩阵和输出:

![image-20260623150020087](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch04/attentionQkV.png)

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

这个公式是 Transformer 最核心的公式,务必背下来。我们逐项解读它的形状:

- $Q K^T$ 的形状是 $n \times n$,第 $(i, j)$ 元就是 $q_i \cdot k_j$
- 除以 $\sqrt{d_k}$ 后形状不变
- softmax 沿**最后一维**(列方向)归一化,得到 $n \times n$ 的注意力矩阵 $A$, 第 $i$ 行就是 token $i$ 对所有 token 的注意力分布
- $A V$ 形状是 $n \times d_v$,每一行就是对应 token 的输出表示



### 4.3.8 复杂度对比:为什么 Transformer 在长序列上慢?

很多人有个误解,觉得 Transformer 一定比 RNN 快。其实这取决于序列长度 $n$ 和隐藏维度 $d$ 的相对大小:

- **RNN**:每层每步是一个 $d \times d$ 的矩阵乘法,共 $n$ 步。总复杂度 $O(n \cdot d^2)$,但**串行**,不能并行。
- **Self-Attention**:核心是 $Q K^T$($n \times n$ 矩阵)乘 $V$。总复杂度 $O(n^2 \cdot d)$,但**完全并行**,所有位置可以同时算。

什么时候 Transformer 更快?当 $n < d$ 时,$n^2 \cdot d < n \cdot d^2$,Transformer 计算量更小,加上并行优势,速度优势巨大。这正是大多数 NLP 任务的常态——句子长度几百,隐藏维度上千。

但当 $n$ 远大于 $d$ 时(比如长文档建模、基因组序列),$O(n^2)$ 的内存和计算开销就成了痛点。这也是后来 Longformer、Linformer、FlashAttention 等改进工作的出发点,我们后面会简单提到。

> **Transformer 的注意力是 $O(n^2)$ 的,这是它最大的瓶颈,也是后续无数研究试图突破的方向**。

## 4.4 多头注意力:让模型从多个视角理解关系

### 4.4.1 为什么单头不够

到目前为止我们讨论的都是单头注意力——一组 $W^Q, W^K, W^V$。但这有个问题:**单头注意力只能学到一种"关系模式"**。

举个例子,在句子 "The animal didn't cross the street because it was too tired" 中,"it" 指代什么?这里至少有两种关系需要建模:

- **指代关系**:it 指代 animal
- **修饰关系**:tired 修饰 it

如果只有一组 $W^Q, W^K, W^V$,模型很难同时把这两种关系都学到——它只能学一个"折中"的注意力模式。

**多头注意力(Multi-Head Attention)** 的解决方案非常自然:**搞 $h$ 组独立的 $W^Q, W^K, W^V$,让每组关注不同的子空间**。然后把所有头的输出拼起来,再过一个线性层做最终融合。

### 4.4.2 多头注意力的数学表达

假设有 $h$ 个头(原论文中 $h = 8$)。每个头有自己的投影矩阵:

$$\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$$

其中 $W_i^Q \in \mathbb{R}^{d_{model} \times d_k}$,$W_i^K \in \mathbb{R}^{d_{model} \times d_k}$,$W_i^V \in \mathbb{R}^{d_{model} \times d_v}$。

然后把所有头的输出拼接,过一个最终线性变换 $W^O \in \mathbb{R}^{h d_v \times d_{model}}$:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$

**一个常见的工程选择是设置 $d_k = d_v = d_{model} / h$**。这样总参数量和单头注意力(用满 $d_{model}$)是一样的——多头不是"用更多参数",而是"把同样的参数切成多份,各管一摊"。

![image-20260623150127889](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch04/multi-head.png)

### 4.4.3 多头注意力学到了什么

通过可视化训练好的 Transformer,研究者发现不同的头确实在学不同的模式:

- 有的头专注于**句法关系**——主语找谓语,宾语找动词
- 有的头专注于**指代关系**——代词找它指代的名词
- 有的头专注于**位置关系**——每个 token 关注它的左/右邻居
- 还有的头看起来在做"全局总结"——所有 token 都关注同一个特殊位置(如 [CLS])

> 多头注意力相当于"集成学习"——多个简单的注意力头协作,共同完成对复杂关系的建模。

### 4.4.4 高效实现:把多头折叠进矩阵运算

实际工程实现里,我们不会真的开 $h$ 个独立的 for 循环。一个标准技巧是:

1. 用一个大的 $W^Q \in \mathbb{R}^{d_{model} \times d_{model}}$ 一次性把 $X$ 投影成 $h \cdot d_k = d_{model}$ 维
2. **Reshape** 成 $(n, h, d_k)$,再 transpose 成 $(h, n, d_k)$
3. 在 batch 维上做并行的注意力计算
4. 最后 reshape 回 $(n, h \cdot d_v)$,过 $W^O$

这种"先合后切"的做法让多头注意力和单头注意力的代码结构几乎一样,只多了两个 reshape 操作。我们在 4.9 节给出完整代码实现。

## 4.5 输入层:Embedding 与位置编码

注意力机制本身有一个被忽略的特性——**它对输入顺序无感**。把 "I love you" 和 "You love I" 输入 Self-Attention,虽然每个词的表示会不同,但**输出向量的集合是完全一样的**(只是顺序换了),模型分不出"主谓宾"和"宾谓主"的区别。

这显然不行。NLP 中顺序至关重要——"狗咬人"和"人咬狗"的语义天差地别。Transformer 的解决方案是:**用位置编码(Positional Encoding)显式注入位置信息**。

### 4.5.1 Token Embedding

在讨论位置编码之前,先回顾一下 token embedding。这部分对 NLP 有基础的同学很熟悉,这里只快速过一下。

文本经过分词器(Tokenizer)拆成 token id 序列后,通过一个**可学习的嵌入矩阵** $E \in \mathbb{R}^{|V| \times d_{model}}$ 查表得到每个 token 的稠密向量表示:

$$x_i = E[\text{token}_i]$$

其中 $|V|$ 是词表大小,$d_{model}$ 是嵌入维度。这一步把离散的 token id 变成连续的向量,使得后续的矩阵运算成为可能。

值得一提的细节:**原论文在嵌入后会乘以 $\sqrt{d_{model}}$**:

$$x_i = E[\text{token}_i] \times \sqrt{d_{model}}$$

这是为了让 Embedding 的数值范围与位置编码的数值范围(后面会看到,sin/cos 取值在 $[-1, 1]$)在同一量级,避免位置编码相加后被淹没。

### 4.5.2 位置编码的设计目标

位置编码要满足几个要求:

1. **每个位置有唯一的编码**——不能两个位置撞车
2. **编码值有界**——不要随序列长度无限增长
3. **能编码相对位置**——网络应该容易学到"相隔 5 个位置"这种关系
4. **能外推到训练时没见过的长度**——比如训练时序列长 512,推理时希望能处理 1024

最朴素的想法是用整数编码——位置 0、1、2、3……但这样编码值会随序列变长无限增长,显然不行。归一化到 $[0, 1]$ 又有另一个问题——同样是位置 5,在长度 10 的序列里是中间,在长度 100 的序列里是开头,语义不一致。

### 4.5.3 Sinusoidal 位置编码

Vaswani 给出的方案是用一组**不同频率的正弦/余弦函数**:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

公式里的符号:

- $pos$ 是 token 在序列中的位置(0, 1, 2, ...)
- $i$ 是维度索引,$0 \le i < d_{model}/2$
- 偶数维度($2i$)用 sin,奇数维度($2i+1$)用 cos

每个 token 拿到一个 $d_{model}$ 维的向量,前后相邻的两个维度共享同一个频率,但一个用 sin、一个用 cos。频率从 $1$(高频)指数衰减到 $1/10000$(低频),覆盖了从极短到极长各种尺度的周期信息。

最终的输入是 Embedding + Position Encoding:

$$\text{input}_i = x_i + PE_i$$

![embedding-plus-pos-encoding](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch04/embedding-plus-pos-encoding.png)

### 4.5.4 为什么 sin/cos 能编码相对位置?

这是这个设计最巧妙的地方。我们来推一下。

固定一个频率 $\omega = 1/10000^{2i/d_{model}}$。对于位置 $pos$,有:

$$PE_{pos} = (\sin(\omega \cdot pos), \cos(\omega \cdot pos))$$

(这里只看一对维度。)

那么**对于偏移 $k$,$PE_{pos+k}$ 可以表示为 $PE_{pos}$ 的一个线性变换**:

$$\begin{pmatrix} \sin(\omega(pos+k)) \\ \cos(\omega(pos+k)) \end{pmatrix} = \begin{pmatrix} \cos(\omega k) & \sin(\omega k) \\ -\sin(\omega k) & \cos(\omega k) \end{pmatrix} \begin{pmatrix} \sin(\omega \cdot pos) \\ \cos(\omega \cdot pos) \end{pmatrix}$$

这个 $2 \times 2$ 矩阵就是熟悉的**旋转矩阵**!

意思是说:从位置 $pos$ 到位置 $pos+k$,在每对 (sin, cos) 维度上,都是一个固定的旋转——**旋转角度只依赖于 $k$,不依赖于 $pos$**。

这意味着:**模型只要学到一个固定的旋转(不管原始位置是什么),就掌握了"距离 $k$"这个概念**。神经网络很擅长学线性变换,所以这种设计让相对位置极易学习。

> Sinusoidal 位置编码的精妙之处:**绝对位置用 sin/cos 编码,相对位置自然由旋转关系隐含表示**。这是数学美感和工程实用性的完美结合。

### 4.5.5 可学习位置编码 vs 固定位置编码

后来 BERT 和大多数实用模型选择了**可学习位置编码**——直接用一个 $\mathbb{R}^{n_{max} \times d_{model}}$ 的随机初始化矩阵当位置编码,让模型自己学。这种做法在已知最大长度的场景下表现略好,但**不能外推**——训练时见过的最大位置是 512,推理时遇到第 513 个位置就没有编码可用。

| 方式 | 优势 | 劣势 |
|:---|:---|:---|
| Sinusoidal(固定) | 可外推到任意长度;无需训练参数 | 表达力略弱于可学习版本 |
| Learnable(可学习) | 任务特定的最优编码;BERT 等模型实测略好 | 不能外推;额外参数 |

更近代的位置编码方案如 **RoPE(旋转位置编码)** 和 **ALiBi**,我们会在 Part 3 LLM 章节详细讨论。

## 4.6 Encoder Block 的内部结构

到这里,我们已经能把一段输入文本变成一组带位置信息的向量了。接下来这组向量要进入 Encoder——这是 Transformer 真正发挥作用的地方。

在开始之前，我们先看下transformer的整体架构，其中左侧为encoder块，右侧为decoder块，接下来我们将一一拆解
![image-20260623150625525](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch04/transfomer.png)

原论文的 Transformer 由 6 个相同结构的 Encoder Block 堆叠而成，请看图中的左侧部分。每个 Block 内部包含两个子层:**多头自注意力**和**前馈神经网络(FFN)**。每个子层外面包了一层"残差 + LayerNorm"的壳，拆解如下图。

![encoder-block-structure](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch04/encoder-block-structure.png)

我们逐个来看每个组件。多头注意力前面已经讲过了,这一节重点讲**残差**、**LayerNorm**、**FFN**。

### 4.6.1 残差连接:让深层网络训得动

Transformer 的两个子层都用了残差连接(Residual Connection),也叫 skip connection:

$$\text{output} = \text{LayerNorm}(x + \text{Sublayer}(x))$$

残差连接最早由 ResNet 提出,用来解决**深度神经网络的退化问题**——堆得越深,性能反而越差。退化的根源不是过拟合(训练 loss 也变差),而是**梯度难以有效传播**到浅层。

残差连接给反向传播开了一条"高速公路":

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial \text{output}} \cdot \left(1 + \frac{\partial \text{Sublayer}}{\partial x}\right)$$

注意那个**加号 1**——即使 Sublayer 的梯度变得很小,整体梯度也至少有个 1 兜底,不会消失。

> 残差连接的本质是给梯度提供"短路通道",使得深度网络的训练不再被"梯度消失"卡住。

对于 Transformer 这种动辄几十上百层的架构,残差连接是必需品,没有它根本训不动。

### 4.6.2 LayerNorm vs BatchNorm:为什么 NLP 选了 LayerNorm?

归一化(Normalization)的目的是让网络中间层的激活值分布稳定。CV 任务中我们最熟悉的是 BatchNorm,但 Transformer 用的是 LayerNorm。这俩有什么区别?

**BatchNorm(批归一化)**:对**同一个特征通道**,在**整个 batch** 中所有样本的所有空间位置上计算均值和方差,然后归一化。形状 `[B, C, H, W]` 的张量,对每个通道 $c$,在 $B$ 个样本和 $H \times W$ 个空间位置上算均值方差,用这一对统计量归一化该通道下所有 $B \times H \times W$ 个值。

**LayerNorm(层归一化)**:对**同一个样本**,在**该层所有特征维度**上计算均值和方差,然后归一化。形状 `[B, C, H, W]` 的张量,对每个样本 $b$,在它自己的 $C \times H \times W$ 个特征值上算均值方差,用这对统计量归一化这个样本的所有特征。

两者的核心差异如下图:

![layernorm-vs-batchnorm](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch04/layernorm-vs-batchnorm.png)

**为什么 NLP 选 LayerNorm 而不是 BatchNorm?** 三个原因:

第一,**变长序列的统计不稳定**。NLP 任务中不同样本的序列长度差异很大——同一个 batch 里有 10 个 token 的句子,也有 200 个 token 的句子。padding 后再算 BN 统计量会被填充位置严重污染。

第二,**小 batch size 不友好**。Transformer 模型大,显存吃紧,batch size 经常只能开到 8、16,BN 的统计量在小 batch 下波动剧烈,效果反而变差。LayerNorm 完全在样本内部算,不受 batch 大小影响。

第三,**推理时无需维护 running mean/var**。BN 在训练时算每个 batch 的统计量,推理时用整体的滑动平均——这套机制在序列模型上经常出 bug(比如 dropout 行为差异)。LayerNorm 训练和推理逻辑完全一样,工程上简单。

> **CV 用 BN,NLP 用 LN**——本质是因为 CV 任务数据形状规整、batch 内统计稳定,NLP 任务序列变长、batch 小,LN 更鲁棒。

### 4.6.3 Pre-LN vs Post-LN

原论文的设计是 **Post-LN**:子层输出加残差后再过 LN,即 $\text{LN}(x + \text{Sublayer}(x))$。但训练大模型时人们发现 Post-LN 训练不稳定,需要精心设计的 warmup 学习率调度才能收敛。

后续研究改为 **Pre-LN**:在子层之前先做 LN,即 $x + \text{Sublayer}(\text{LN}(x))$。Pre-LN 的好处是梯度直接从输出残差传到输入,无需经过 LN 的非线性变换,**训练显著更稳定**。GPT、LLaMA 等现代大模型几乎都改用了 Pre-LN。

![pre-ln-vs-post-ln](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch04/pre-ln-vs-post-ln.png)

> Post-LN 训练敏感、效果好;Pre-LN 训练稳定、易扩展。**现代大模型几乎全部改用 Pre-LN**。

### 4.6.4 Feed Forward Network(FFN)

每个 Encoder Block 的第二个子层是一个看起来很普通的两层 MLP:

$$\text{FFN}(x) = \max(0, x W_1 + b_1) W_2 + b_2$$

或者更现代的写法用 GELU 替代 ReLU:

$$\text{FFN}(x) = \text{GELU}(x W_1 + b_1) W_2 + b_2$$

关键的设计是**中间层维度远大于输入输出**。在原论文里,$d_{model} = 512$,但 FFN 中间层 $d_{ff} = 2048$,先升维 4 倍再降回去。这个 4 倍是个工程经验值,现代大模型(LLaMA 等)有时用 2.7 倍或别的比例。

**FFN 的作用是什么?** 这是面试常考问题,答案有几个层次:

1. **引入非线性**。注意力本身是加权求和(线性操作),FFN 提供了网络真正"思考"的非线性能力。
2. **逐位置(position-wise)的特征加工**。FFN 对每个 token 独立作用,与序列长度无关。可以理解为"注意力负责跨位置融合,FFN 负责单位置加工"。
3. **承载模型的"知识"**。最近的研究表明,FFN 层可以被解读为一个 key-value 存储——大模型里的事实知识主要存在 FFN 权重里。

> **注意力跨位置交流,FFN 单位置加工**——这是 Transformer Block 的两个互补支柱。

把 4.6 节所有部件组装起来,一个完整的 Encoder Block 长这样(Pre-LN 版伪代码):

```python
def encoder_block(x):
    # 子层 1: Multi-Head Self-Attention
    h = LayerNorm(x)
    h = MultiHeadAttention(h, h, h)  # Q=K=V=h
    x = x + Dropout(h)

    # 子层 2: Feed Forward
    h = LayerNorm(x)
    h = FFN(h)
    x = x + Dropout(h)

    return x
```

注意每个子层的 dropout——这是原论文里不显眼但重要的正则化技巧。

## 4.7 Decoder Block:三个子层的协作

Decoder 比 Encoder 多了一个子层。Encoder 有 2 个子层(MHA + FFN),Decoder 有 3 个子层:

1. **Masked Multi-Head Self-Attention**(带因果掩码的自注意力)
2. **Multi-Head Cross-Attention**(对 Encoder 输出的交叉注意力)
3. **Feed Forward**

![decoder-block-structure](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch04/decoder-block-structure.png)

### 4.7.1 Masked Self-Attention:防止"看见未来"

Decoder 是用来生成的——一个 token 一个 token 地输出。在训练阶段,我们用 Teacher Forcing:把整个目标序列一次性喂给 Decoder,并行计算所有位置的损失。但这样有个问题:**第 3 个位置不应该能看到第 4、5 个位置**(否则就是作弊,推理时根本看不到未来)。

解决方法是**因果掩码(Causal Mask)**——一个上三角全是 $-\infty$ 的矩阵。把这个矩阵加到注意力分数上:

$$\text{Score}_{i,j} = \begin{cases} q_i \cdot k_j / \sqrt{d_k} & j \le i \\ -\infty & j > i \end{cases}$$

经过 softmax 后,$-\infty$ 位置的注意力权重变成 0——就像那些位置的 token 根本不存在。

![causal-mask](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch04/causal-mask.png)

举个例子,假设序列长度是 4,Mask 矩阵长这样(用 $0$ 表示保留,$-\infty$ 表示屏蔽):

```
位置  1  2  3  4
1   [ 0  -∞  -∞  -∞ ]
2   [ 0   0  -∞  -∞ ]
3   [ 0   0   0  -∞ ]
4   [ 0   0   0   0 ]
```

第 1 行只能看自己;第 2 行能看前两个;以此类推。这就是"自回归"的本质。

> 因果掩码是 Decoder 的"道德底线"——保证训练阶段并行计算和推理阶段串行生成的行为一致。

### 4.7.2 Cross-Attention:连接 Encoder 和 Decoder

Cross-Attention 是 Decoder 的第二个子层,也是 Encoder-Decoder 架构(如机器翻译)中两个序列的"桥梁"。

它和普通 Self-Attention 的唯一区别是 Q、K、V 的来源不同:

- **Q 来自 Decoder 当前层**(我想问什么?)
- **K、V 都来自 Encoder 的最终输出**(源序列的表示)

直觉上,Decoder 在生成每个 token 时都会去"问"Encoder——"源序列里哪些位置和我现在想生成的目标 token 最相关?"对于翻译"I love you"→"我爱你"时,生成"爱"的时候 Q 会和 Encoder 输出做交叉注意力,理想情况下注意力主要聚焦在"love"上。

数学形式上和普通注意力没有任何差异:

$$\text{CrossAttention}(Q_{dec}, K_{enc}, V_{enc}) = \text{softmax}\left(\frac{Q_{dec} K_{enc}^T}{\sqrt{d_k}}\right) V_{enc}$$

注意 Cross-Attention **没有因果掩码**——Decoder 看 Encoder 的整个输出是合理的,因为源序列在解码前就完整已知了。但通常会有 **Padding Mask**——防止 Decoder 把注意力放在源序列的填充位置上。

### 4.7.3 两种 Mask 的区别

Transformer 里其实有两种掩码,容易混淆,这里特别澄清下。

**Padding Mask(填充掩码)**

输入序列长度不一(一个 batch 里有的句子 10 个 token,有的 50 个),我们会把短的 pad 到统一长度。Padding Mask 的作用是告诉注意力机制:**这些填充位置是"假"的,不要分配注意力**。

实现上同样是把对应位置的注意力分数设为 $-\infty$。Padding Mask 在 Encoder 和 Decoder 中都用,在 Self-Attention 和 Cross-Attention 中都用。

**Sequence Mask / Causal Mask(因果掩码)**

只在 Decoder 的 Self-Attention 中用,作用前面讲过——防止看见未来。

| 掩码类型 | 何处使用 | 目的 |
|:---|:---|:---|
| Padding Mask | Encoder/Decoder, 所有注意力 | 屏蔽填充位置 |
| Causal Mask | Decoder Self-Attention | 屏蔽未来位置 |

实际工程中这两种 Mask 通常是合并的——一个叫 `attn_mask` 的 0/1 矩阵,把不能看的位置全标 0,能看的标 1。

![mask-combine](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch04/mask-combine.png)

## 4.8 输出层与训练/推理的不对称

走过 Encoder、走过 Decoder,最后一步是把 Decoder 输出的向量变成预测的 token。

### 4.8.1 输出层:Linear + Softmax

Decoder 输出的是一组 $d_{model}$ 维的向量。要变成对词表的概率分布,需要两步:

$$\text{logits} = \text{DecoderOutput} \cdot W_{vocab}$$

$$P(\text{token}) = \text{softmax}(\text{logits})$$

其中 $W_{vocab} \in \mathbb{R}^{d_{model} \times |V|}$。一个常见的优化是**权重共享**(weight tying)——让输出层的 $W_{vocab}$ 与输入 Embedding 矩阵 $E$ 共享(转置关系):$W_{vocab} = E^T$。这样可以省下 $|V| \times d_{model}$ 的参数,在大词表场景下省的不少。

最终选择概率最高的 token 作为预测输出(贪心解码),或者用 Beam Search、Top-k Sampling、Top-p Sampling 等策略采样。

### 4.8.2 Teacher Forcing:训练时的并行技巧

训练 Transformer 时,我们用一种叫 **Teacher Forcing** 的技巧:**不论模型上一步预测对了没,下一步都把"正确答案"喂给它**。

比如训练翻译"I love you"→"我爱你":

- 输入 Decoder 的是 `<bos> 我 爱 你`(标准答案,带起始符)
- 期望 Decoder 输出的是 `我 爱 你 <eos>`
- 损失就是这两组的交叉熵

注意一个关键点:**Decoder 一次性接收整个目标序列,所有位置的损失同时计算**。这能成立的原因正是因果掩码——第 $i$ 个位置的输出只依赖前 $i$ 个输入,即使一次性把整个序列都喂进去,因果性也不会被破坏。

![teacher-forcing-vs-autoregressive](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch04/teacher-forcing-vs-autoregressive.png)

### 4.8.3 推理时的"串行困境"

推理阶段就没那么舒服了——我们不知道目标序列是什么,得**一个 token 一个 token 地生成**:

1. 先输入 `<bos>`,Decoder 输出概率分布,选最高概率的 token,假设是"我"
2. 输入 `<bos> 我`,Decoder 输出第二个位置的分布,选出"爱"
3. 输入 `<bos> 我 爱`,选出"你"
4. 输入 `<bos> 我 爱 你`,选出 `<eos>`,生成结束

这是不可避免的串行——因为第 $t$ 步的输入依赖第 $t-1$ 步的输出。**Transformer 的训练和推理在这里出现了根本不对称**:训练 $O(1)$ 步搞定,推理 $O(n)$ 步。这也是为什么大模型推理慢——千亿参数的模型,每个 token 都得跑一遍完整的网络。

为了缓解这个问题,实践中有 **KV Cache** 技术:在推理时把每一步的 K、V 缓存下来,下一步只需要为新 token 算新的 Q,而不必重算所有历史 token 的 K、V。这能把推理复杂度从 $O(n^3)$ 降到 $O(n^2)$,是大模型推理优化的基础。具体细节我们留到 Part 3 LLM 章节。

> 训练时 Teacher Forcing 让 Transformer 享受完全并行的红利,但推理时不可避免地退化为串行——这是自回归生成模型的"原罪"。

### 4.8.4 Label Smoothing:小技巧大收益

原论文里还有一个常被忽略的训练技巧叫 **Label Smoothing**(标签平滑)。

标准的交叉熵损失,目标是让正确 token 的概率尽量逼近 1,其他 token 概率逼近 0。但这种"非黑即白"的目标会让模型变得过度自信(overconfident),概率分布尖锐到接近 one-hot,泛化能力反而变差。

Label Smoothing 把硬标签(0 或 1)平滑成软标签:正确类别赋值 $1 - \epsilon$,其他类别均分 $\epsilon$:

$$y_{smooth} = (1 - \epsilon) \cdot y_{onehot} + \frac{\epsilon}{|V|}$$

原论文用 $\epsilon = 0.1$,让模型学会"我有把握,但不必绝对"。这个改动在大词表的翻译任务上能带来 0.5~1 BLEU 的提升,是个性价比极高的 trick。

## 4.9 PyTorch 极简实现

讲了这么多理论,最后我们手写一个完整的 Transformer Block,把所有概念落到代码上。这一节我们实现的是 Encoder Block(Pre-LN 版),Decoder Block 留给读者作为练习——结构上只是多了一个 Cross-Attention 子层,逻辑上没有新东西。

### 4.9.1 Scaled Dot-Product Attention

最底层的注意力函数:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: (batch, num_heads, seq_q, d_k)
    K: (batch, num_heads, seq_k, d_k)
    V: (batch, num_heads, seq_k, d_v)
    mask: (batch, 1, seq_q, seq_k) 或 None,1 表示保留,0 表示屏蔽
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    attn = F.softmax(scores, dim=-1)
    out = torch.matmul(attn, V)
    return out, attn
```

注意 `masked_fill` 把 mask=0 的位置填成 $-\infty$,softmax 后这些位置的权重就是 0。

### 4.9.2 Multi-Head Attention

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        # 投影并切成多头: (B, seq, d_model) -> (B, num_heads, seq, d_k)
        Q = self.W_Q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        out, attn = scaled_dot_product_attention(Q, K, V, mask)

        # 合并多头: (B, num_heads, seq, d_k) -> (B, seq, d_model)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        out = self.W_O(out)
        return self.dropout(out)
```

注意几个工程细节:

- `W_Q/W_K/W_V` 输入输出维度都是 `d_model`,等价于 $h$ 个独立的 $d_{model} \to d_k$ 投影矩阵拼起来,只是用一个大矩阵加 reshape 实现
- `transpose(1, 2)` 把头维度移到 batch 之后,方便后续在 (head, seq, d_k) 上做并行注意力
- `contiguous()` 必须加,否则后面的 `view` 会因为内存非连续而报错

### 4.9.3 Position-wise Feed Forward

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))
```

简洁明了——升维、激活、dropout、降维。

### 4.9.4 Positional Encoding

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (batch, seq, d_model)
        return x + self.pe[:, :x.size(1)]
```

`register_buffer` 让 `pe` 成为模块的一部分(会被 `.to(device)` 一起移动)但不是可训练参数。

### 4.9.5 Encoder Block(Pre-LN 版)

把上面所有部件组装起来:

```python
class EncoderBlock(nn.Module):
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 子层 1: Self-Attention (Pre-LN)
        h = self.norm1(x)
        x = x + self.dropout(self.attn(h, h, h, mask))

        # 子层 2: FFN (Pre-LN)
        h = self.norm2(x)
        x = x + self.dropout(self.ffn(h))

        return x
```

### 4.9.6 完整的 Encoder

把 $N$ 个 EncoderBlock 堆叠起来,加上 Embedding 和位置编码:

```python
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8, d_ff=2048,
                 num_layers=6, dropout=0.1, max_len=5000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

        self.layers = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # x: (batch, seq) token id
        x = self.embedding(x) * math.sqrt(self.d_model)  # 别忘了 sqrt(d_model) 缩放
        x = self.pos_encoding(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)
```

来跑一下:

```python
# 简单测试
encoder = TransformerEncoder(vocab_size=10000, num_layers=6)
x = torch.randint(0, 10000, (2, 20))  # batch_size=2, seq_len=20
out = encoder(x)
print(out.shape)  # torch.Size([2, 20, 512])
```

到这里,你已经从零搭出了一个能跑的 Transformer Encoder 了。Decoder Block 你应该能照葫芦画瓢实现——多一个 Cross-Attention 子层,Self-Attention 处加因果掩码即可。

## 4.10 Transformer 家族速览

原始的 Transformer 是 Encoder-Decoder 架构(用于机器翻译)。但人们很快发现,Encoder 和 Decoder 各自就足以承担很多任务,于是衍生出了三大流派:

![transformer-family](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch04/transformer-family.png)

**Encoder-only(以 BERT 为代表)**:只用编码器,所有位置可以双向看上下文(没有因果掩码)。适合"理解类"任务——文本分类、命名实体识别、问答抽取。BERT 用 Masked Language Model(完形填空)预训练。

**Decoder-only(以 GPT 系列为代表)**:只用解码器,带因果掩码,所有位置只能看前面的内容。适合"生成类"任务——文本生成、对话、创作。GPT 用自回归语言建模(下一个 token 预测)预训练。这是当前大模型(LLaMA、Claude、GPT-4)的主流架构。

**Encoder-Decoder(以 T5、BART 为代表)**:保留原始的 Encoder-Decoder 结构,适合"输入到输出"的任务——机器翻译、摘要、改写。T5 把所有 NLP 任务都统一成"text-to-text"格式来做。

为什么 Decoder-only 在大模型时代成了赢家?核心原因是**自回归语言建模这一种简单目标可以解决一切**——只要你模型够大、数据够多。Encoder-Decoder 的"输入-输出"结构在小模型时代是优势,但在大模型时代,统一的生成目标让 scaling 更顺滑。

> **Encoder 擅长理解,Decoder 擅长生成,Encoder-Decoder 擅长翻译**——这是粗略但实用的分类记忆法。

至于现代大模型的更多创新——RoPE 位置编码替代 Sinusoidal、SwiGLU 激活替代 ReLU/GELU、RMSNorm 替代 LayerNorm、Grouped Query Attention 替代标准多头、FlashAttention 实现 IO 高效计算——这些我们留到 Part 3 LLM 章节再展开。

## 本章小结

这一章我们完整拆解了 Transformer 架构。核心要点回顾如下。

**第一,Transformer 的设计动机是"全局并行的注意力"**。RNN 的串行计算和长程依赖问题,CNN 的局部感受野问题,促成了"用注意力机制实现任意两个位置直接交互"这一根本创新。任意两 token 之间的"距离"被压缩到 1 步矩阵运算,且所有交互完全可并行。

**第二,Self-Attention 的核心公式 $\text{Attention}(Q, K, V) = \text{softmax}(Q K^T / \sqrt{d_k}) V$ 必须烂熟于心**。其中 $\sqrt{d_k}$ 缩放来自点积方差控制(避免 softmax 饱和),Q/K/V 来自同一输入的三种线性投影。计算复杂度为 $O(n^2 \cdot d)$,并行性强但内存吃序列长度的平方。

**第三,多头注意力是"集成学习思路"在注意力上的应用**。多个头各管一摊不同的关系模式,最后拼接 + 线性融合。工程上通过 reshape 让多头与单头共享同一份矩阵乘法骨架,几乎不增加额外开销。

**第四,位置编码用 sin/cos 解决"注意力位置无关"的副作用**。Sinusoidal 编码的精妙在于:相对位置可以通过固定的旋转矩阵线性表示,使模型易于学到"距离 $k$"这种相对关系。可学习位置编码效果略好但不能外推。

**第五,Encoder Block 的两个子层互补,各司其职**。多头注意力跨位置融合信息,FFN 单位置非线性加工。两个子层都包了"Add + LayerNorm"。LayerNorm 之所以替代 BatchNorm,核心原因是 NLP 序列变长 + 小 batch 不友好。Pre-LN 替代 Post-LN 是大模型时代的事实标准。

**第六,Decoder 比 Encoder 多一个 Cross-Attention 子层,且 Self-Attention 带因果掩码**。因果掩码保证训练并行和推理串行的行为一致——这是自回归生成的"道德底线"。Padding Mask 处处用,Causal Mask 仅在 Decoder Self-Attention 用。

**第七,Teacher Forcing 让训练享受并行红利,但推理只能串行**——这是自回归模型不可避免的代价。KV Cache 是推理优化的关键技术,Label Smoothing 是训练时性价比极高的小技巧。

**第八,Transformer 家族分三大流派**。Encoder-only(BERT)做理解,Decoder-only(GPT)做生成,Encoder-Decoder(T5)做映射。Decoder-only 在大模型时代胜出,核心是统一的自回归目标与 scaling 兼容性。

掌握了 Transformer,下一章我们会进入 Part 2 计算机视觉的世界,看看注意力机制如何征服图像领域(ViT、Swin Transformer)。再之后的 Part 3 我们会深入到 LLM 的现代版本——位置编码新方案、激活函数新选择、推理优化新技术,本章是这一切的基础。
