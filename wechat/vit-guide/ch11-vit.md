# ViT 完全拆解：把图像切成「视觉单词」塞进 Transformer

> 本文承接「Transformer 完全拆解」系列。上一篇我们啃完了 NLP 里的 Transformer 架构，结尾留下一个悬念：注意力机制如何征服图像领域？本篇就回答这个问题--看 Google Brain 如何用一次大胆的「减法」，把图像切成 16×16 个「视觉单词」，喂进和 NLP 完全相同的 Encoder，在大数据预训练下超越统治 CV 二十五年的 CNN。
>
> 上一篇：[Transformer 架构完全拆解：从 RNN 到注意力机制的演进](../transformer-guide/ch04-transformer.md)
> 下一篇：[DETR：用 Transformer 做目标检测]()

---

## 一、为什么视觉领域需要 Transformer

### CNN 统治视觉的二十五年

从 1998 年 LeNet 诞生算起，卷积神经网络（CNN）在计算机视觉领域统治了将近 25 年。2012 年 AlexNet 在 ImageNet 上一战成名之后，VGG、GoogLeNet、ResNet、DenseNet、EfficientNet 一代接一代，把图像分类精度推向新高。

CNN 的成功，离不开两个写进网络结构里的「经验法则」--也就是所谓的**归纳偏置（inductive bias）**。它们让模型在少量数据下也能学到合理的视觉特征。

**第一个是局部性（locality）。** 卷积核每次只看特征图上的一个小窗口（典型 $3\times3$），这等于在假设：图像上相邻的像素关系最紧密，距离越远关系越弱。这个假设对自然图像几乎总是成立--一只猫的眼睛和它旁边的眼眶、毛发、鼻子强相关，而和图像另一角的背景树没什么关系。局部性让卷积只需极少参数就能扫过整张图。

**第二个是平移等变性（translation equivariance）。** 卷积在图像上滑动时，输出特征图会随输入的平移而平移--你把猫的图片向右移 10 像素，卷积输出的「猫耳朵」特征也向右移 10 像素。同一个特征检测器可以在整张图上复用，不必为每个位置单独学习。

> CNN 的归纳偏置就像给它戴上了一副「局部视野的眼镜」--看什么都先看局部，再层层堆叠拼出全局。这在数据稀缺的时代是巨大优势，但在大数据时代却可能变成束缚。

### 归纳偏置：是先验，也是天花板

随着数据集规模从 ImageNet 的 130 万张一路涨到 JFT-300M 的 3 亿张、Instagram 的 35 亿张，CNN 的归纳偏置开始显得「过度保守」。

原因很直接：**当数据足够多时，模型完全可以从数据里自己学到「局部性」和「平移等变性」，根本不需要人为注入**；而人为注入的偏置反而限制了表达能力--CNN 无论如何堆叠，每一层都还是只能在小窗口内做局部运算，想让图像左上角和右下角的特征直接交互，只能靠极深的网络层层传递。

2020 年 10 月，Google Brain 团队的论文《An Image is Worth 16x16 Words》提出了 **Vision Transformer（ViT）**，做了一件令当时的 CV 圈颇感意外的事：**把图像当成一串「视觉单词」，直接喂进 NLP 里那个原封不动的 Transformer Encoder**。

![vit-overview-flow](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch11/vit-overview-flow.png)

整个流程看上去非常「暴力」：不引入任何卷积，不设计任何针对图像的特殊结构，只把 NLP 那套 Transformer 几乎原封不动搬过来。但实验证明：在足够大的数据集上预训练后，ViT 不仅追平了 ResNet，还超越了当时最强的 CNN 架构。

> ViT 的核心赌注是：**当数据量足够大时，Transformer 强大的全局建模能力会压过 CNN 的归纳偏置，获得更高的性能上限**。

### 与 NLP Transformer 输入的差异

要理解 ViT 的设计，先回答一个最基本的问题：NLP 里的 Transformer 接收的是一串一维的 token 序列（每个 token 是一个词向量），而图像是三维张量 $H\times W\times C$。两者维度不统一，没法直接复用 NLP 的 Transformer 结构。

![nlp-vs-cv-input](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch11/nlp-vs-cv-input.png)

解决这个维度不匹配的关键步骤，就是 **Patch Embedding**：把三维图像切成一个个小块，每个小块展平成一个向量，这样就得到了和 NLP 中完全相同的二维 token 序列形式。

这个看似简单的转换，是 ViT 把 Transformer 从 NLP 搬到 CV 的第一块、也是最关键的奠基石。接下来我们就一步步拆开 ViT 的每个组件，看它具体如何把「图像」翻译成「Transformer 能理解的语言」。

---

## 二、ViT 整体架构概览

深入细节之前，先从宏观上把 ViT 的整体框架看清楚。整个前向过程可以拆成五个阶段：**分块嵌入**、**拼接 Class Token**、**加位置编码**、**过 Transformer Encoder**、**分类头输出**。

![vit-architecture](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch11/vit-architecture.png)

整个流程的核心思想浓缩成一句话：**把图像切成 patch 序列，当成「视觉句子」喂给 Transformer，最后用一个特殊的 Class Token 汇聚全局信息做分类**。

这个流程里有几个关键设计选择，先在脑子里记一下，后面逐一展开：

- **Patch 是基本单元**：图像被切成 $16\times16$ 的小块，每块对应 NLP 中的一个「词」。
- **Class Token 是汇聚点**：一个可学习的特殊 token，放在序列最前面，在 Encoder 里通过自注意力「收集」全局信息。
- **位置编码是可学习的**：不像原始 Transformer 用正弦/余弦固定编码，ViT 用可学习向量注入位置信息。
- **Encoder 与 NLP 完全一致**：不做任何针对图像的修改，只复用 NLP 的标准结构。

> 把图像切成 patch 当成 token 序列，是 ViT 最关键也最优雅的设计--它让 NLP 里那套成熟的 Transformer 几乎可以零成本迁移到 CV。

---

## 三、Patch Embedding：把图像变成 token 序列

### 分块的核心思想

Patch Embedding 要解决的问题是：把三维图像 $H\times W\times C$ 转换成一串二维的 token 序列 $N\times D$，其中 $N$ 是序列长度（类似句子中词的数量），$D$ 是每个 token 的嵌入维度（类似词向量维度）。

最直观的做法：把图像切成若干个不重叠的小块（patch），每个 patch 内部所有像素展平成一个长向量，这个长向量就是最初的 token。

![patch-split](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch11/patch-split.png)

假设原图尺寸为 $H\times W\times C$，每个 patch 大小为 $m\times n\times C$，那么会得到 $\frac{H}{m}\times\frac{W}{n}$ 个 patch，每个 patch 展平后是 $m\cdot n\cdot C$ 维向量。最终完成了从 $H\times W\times C$ 到 $\left(\frac{H}{m}\cdot\frac{W}{n}\right)\times(m\cdot n\cdot C)$ 的维度变换--三维图像变成了二维 token 序列。

### 用卷积优雅地实现分块

理论上，分块 + 展平两步就够了。但在工程实现上，ViT 用了一个更优雅的等价做法：**一次卷积**。

具体地，用一个卷积核大小为 patch 大小、步长等于 patch 大小、卷积核个数等于嵌入维度的卷积，就能一次性完成「分块 + 展平 + 线性投影」三件事。

以 ViT-Base/16 为例：输入图像 $224\times224\times3$，patch 大小 $16\times16$，嵌入维度 $D=768$。卷积参数为 `kernel_size=16, stride=16, in_channels=3, out_channels=768`。卷积后得到的特征图尺寸为：

$$\frac{224}{16}\times\frac{224}{16}\times 768 = 14\times14\times768$$

把空间维度 $14\times14$ 展平，就得到了 $196\times768$ 的二维张量--正好是 token 序列的形式。

![patch-embedding-conv](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch11/patch-embedding-conv.png)

这个卷积实现有一个微妙但很重要的细节：卷积核是**可学习的**。也就是说，虽然它形式上在做「分块」，但每个卷积核实际上在学习如何把 $16\times16\times3=768$ 个像素线性投影成 1 维嵌入。这和 NLP 里的词嵌入矩阵本质同构--词嵌入矩阵把 one-hot 词向量投影成稠密词向量，这里的卷积把 patch 像素投影成稠密 patch 嵌入。

> Patch Embedding 的本质是一个**可学习的线性投影**：它不仅在做「切块」，还在同时学习如何把每个 patch 的像素压缩成有意义的特征向量。

### 为什么是 patch，而不是像素

一个自然的疑问：为什么不直接把每个像素当成一个 token？要回答这个问题，得先理解自注意力的**计算复杂度**。

自注意力要让序列里**任意两个 token** 都互相算一次相似度，再根据相似度加权融合。如果序列长度是 $N$，「两两配对」的组合数就是 $N\times N=N^2$--这就是为什么自注意力的复杂度是 $O(N^2)$。序列稍微变长一点，计算量就会平方级爆炸。

如果把每个像素都当成一个 token，$224\times224$ 的图像序列长度 $N=50176$，那么 $N^2=2.5\times10^9$--光是注意力矩阵就有 25 亿个元素，显存和算力都吃不消，根本训不动。

patch 化把序列长度从 50176 降到 196，$N^2$ 从 25 亿降到约 3.8 万，自注意力的计算量降低了 6 万多倍--这是 ViT 能用得起 Transformer 的关键妥协。

但妥协也付出了代价：patch 内部的像素被强行「压平」成一个向量，**子 patch 尺度的细节信息会被线性投影抹掉**。这也是 ViT 在小数据集上效果差的原因之一--CNN 通过多层卷积可以保留细粒度的局部特征，而 ViT 一上来就把局部信息压缩到了 patch 级别。

| 设计选择 | 序列长度 $N$ | 注意力复杂度 $O(N^2)$ | 局部细节 |
| --- | --- | --- | --- |
| 像素级 token | 50176 | $2.5\times10^9$ | 完整保留 |
| Patch（16×16） | 196 | 38416 | 被压缩 |
| Patch（32×32） | 49 | 2401 | 严重丢失 |

可以看到，patch 大小是一个**计算成本 vs 细节保留**的权衡。ViT 默认选 $16\times16$ 是经验上的甜点--既能让序列长度足够短以高效训练，又能让每个 patch 包含足够语义信息。

---

## 四、Class Token 与位置编码

### Class Token：用一个特殊 token 汇聚全局信息

经过 Patch Embedding，我们得到了 $196\times768$ 的 token 序列。如果直接把它送进 Transformer Encoder，输出仍然是 $196\times768$--196 个 token 各自携带了不同 patch 的信息，但要做分类的话，**到底该拿哪个 token 来代表整张图？**

这是个棘手的问题。如果取所有 token 的平均池化（mean pooling），会损失掉自注意力学到的位置差异；如果取中心 patch 或某个固定 patch，又缺乏理论依据。ViT 借鉴了 NLP 中 BERT 的 [CLS] token 设计，引入了一个**可学习的 Class Token**。

![class-token-concat](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch11/class-token-concat.png)

具体做法是：在 Patch Embedding 之后，定义一个可学习的 `cls_token`，形状为 $1\times1\times768$，初始化为全 0。在前向时把它**拼接到 patch 序列的最前面**，序列从 $196\times768$ 变成 $197\times768$。

经过 Encoder 之后，这个 Class Token 通过自注意力和所有 patch token 进行了信息交互--它可以「看到」所有 patch，所有 patch 也可以「看到」它。在分类阶段，只需取序列的第 0 个位置（即 Class Token 的最终状态）送入分类头即可。

> Class Token 就像一个「专门负责总结的学员」--在 Encoder 的每一层里，它都通过自注意力去「询问」所有 patch，最后变成了整张图的全局表征。

### 为什么需要位置编码

Transformer 的自注意力机制有一个反直觉的特性：**它是位置无关的（permutation-invariant）**。如果你把输入序列打乱顺序，自注意力的输出也只是相应地打乱顺序，内容本身不会改变。

这对 NLP 来说是问题（否则「狗咬人」和「人咬狗」就成了一回事），对 CV 更是问题--否则 ViT 就分不清「猫在桌子上」和「桌子在猫上」。

为了给 Transformer 注入位置信息，ViT 沿用了原始 Transformer 的位置编码思路，但做了一个改变：**用可学习的位置编码，而不是固定的正弦/余弦编码**。

![positional-encoding](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch11/positional-encoding.png)

`pos_embed` 也是一个 $197\times768$ 的可学习参数，初始化为全 0，和拼接 cls_token 后的序列**逐元素相加**。这样每个位置（包括 Class Token）都会被注入它所在位置的信息。

值得注意的是，这个位置编码是 1D 的（沿序列维度），而不是 2D 的（沿图像的 H、W 维度）。这看起来有点反直觉--既然 patch 来自 2D 图像，为什么不直接用 2D 位置编码？实验表明，模型完全可以从 1D 位置编码中学到 2D 邻接关系，而且 1D 编码参数更少、更简单。这也间接说明：**位置编码只是给模型一个「位置标记」，具体如何利用由模型自己学**。

> 位置编码的本质是给每个 token 贴上一个「位置标签」，让自注意力能区分同一内容出现在不同位置的情况。ViT 用可学习的 1D 编码，既简单又足够有效。

---

## 五、Transformer Encoder

把前面几步合起来，我们已经得到了带位置信息的 $197\times768$ token 序列。接下来就是 ViT 的「心脏」--Transformer Encoder。这一部分和上一篇拆解的 NLP Transformer Encoder 几乎完全一致，这里只做简要回顾，重点关注 ViT 中的实现细节。

> **前置知识小抄**：理解下面四个概念就够跟上本节了。
>
> - **自注意力（Self-Attention）**：让序列里每个 token 都去「看」其他所有 token，根据相似度加权融合信息。核心公式 $\text{Attention}(Q,K,V)=\text{softmax}(QK^\top/\sqrt{d})V$，其中 $Q,K,V$ 是输入 $x$ 经过三个线性投影得到的「查询/键/值」。
> - **多头注意力（Multi-Head Attention）**：把 $Q,K,V$ 沿通道维度拆成 $h$ 份，每份独立做一次自注意力，最后拼回来。这让模型能同时关注不同子空间的不同关系。
> - **残差连接（Residual Connection）**：子层输出加上子层输入，即 $x_{\text{out}}=x+\text{SubLayer}(x)$。让梯度能直通，深层网络才训得动。
> - **LayerNorm**：对单个样本的特征维度做归一化（减均值、除标准差），稳定训练。

![encoder-block](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch11/encoder-block.png)

每个 Encoder Block 包含两个子层：**多头自注意力（MSA）** 和 **多层感知机（MLP）**，每个子层前都做了 LayerNorm，每个子层后都做了残差连接。这和原始 Transformer 的 Post-LN 结构不同，ViT 采用的是 **Pre-LN** 结构--LayerNorm 放在子层前面，这让深层 Transformer 的训练更稳定。

自注意力部分的核心计算是：

$$\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

其中 $Q,K,V$ 由输入 $x$ 经过三个独立线性投影得到。ViT-Base 用 12 个注意力头，每个头处理 64 维（$768/12$），最后把 12 个头的输出拼接送回 768 维。

MLP Block 是一个「先扩维再缩回」的两层全连接网络：第一层把 768 维扩到 3072 维（扩大 4 倍），经过 GELU 激活和 Dropout，再缩回 768 维。这个「先扩后缩」的设计让模型能在更高维空间里做非线性变换，学到更丰富的特征。

ViT 还用 **DropPath**（Stochastic Depth）替代了部分 Dropout--以一定概率丢弃整个残差分支，而不是单个神经元。这种正则化方式在深层网络中更有效，能避免训练时梯度退化。

> Encoder Block 的输入输出维度始终是 $197\times768$--维度不变是 Transformer 的关键特性，这让任意深的堆叠都成为可能。ViT-Base 堆叠 12 层，ViT-Large 堆叠 24 层，ViT-Huge 堆叠 32 层。

---

## 六、MLP Head 与分类输出

经过 $L$ 层 Encoder 之后，序列仍然是 $197\times768$，但每个 token 都已经经过多层自注意力的反复融合，携带了丰富的全局信息。最终分类只需要取 Class Token（第 0 个位置）对应的向量。

![mlp-head](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch11/mlp-head.png)

分类头的设计有一个有趣的细节：**Pre-Logits 层是否启用，取决于预训练数据集的规模**。

当预训练数据集是 ImageNet-21k（约 1400 万张图、21843 类）时，会启用 Pre-Logits 层--一个 Linear+Tanh 的组合，把 Class Token 先映射到 `representation_size`（通常是 768）再送入最终的分类 Linear。这个额外的非线性变换让模型在迁移到下游任务时更有表达力。

当预训练数据集是普通 ImageNet（约 130 万张图、1000 类）时，Pre-Logits 层被设为 Identity（什么都不做），Class Token 直接送入最终的分类 Linear。

最终的分类头就是一个简单的 `nn.Linear(hidden_size, num_classes)`，输出每个类别的 logits，送入交叉熵损失即可。

> Pre-Logits 这个看似不起眼的设计选择，反映了 ViT 对「大数据集需要更大模型容量」的回应--数据越多，模型可以承担更复杂的迁移映射，所以多加一层非线性变换。

---

## 七、ViT 的三种模型规格

论文给出了 ViT 的三种规格，它们的差异主要在**层数、隐藏维度、注意力头数**上，patch 大小则在前两种中保持 $16\times16$，Huge 模型用 $14\times14$。

| Model | Patch size | Layers | Hidden Size | MLP size | Heads | Params |
| --- | --- | --- | --- | --- | --- | --- |
| ViT-Base | 16×16 | 12 | 768 | 3072 | 12 | 86M |
| ViT-Large | 16×16 | 24 | 1024 | 4096 | 16 | 307M |
| ViT-Huge | 14×14 | 32 | 1280 | 5120 | 16 | 632M |

几个关键参数的含义回顾：

- **Patch size**：每个 patch 的边长（像素）。Patch 越小，序列越长，模型能看到的细节越多，但计算成本越高。
- **Layers**：Encoder Block 重复的次数，直接决定模型深度。
- **Hidden Size**：每个 token 的嵌入维度 $D$，也是 Encoder 内部所有计算的维度。
- **MLP size**：MLP Block 第一个全连接层的输出维度，通常是 Hidden Size 的 4 倍。
- **Heads**：多头注意力的头数，每个头维度为 Hidden Size / Heads。

可以看到，从 Base 到 Huge，模型参数量从 86M 涨到 632M--但和 GPT-3 的 175B 相比，ViT 即便最大的版本也只能算「中等规模」。这也从侧面印证了 ViT 论文的核心论点：**视觉 Transformer 想超越 CNN，靠的不是模型规模，而是预训练数据规模**。

---

## 八、训练特性：数据规模决定上限

ViT 论文最关键的实验结论，是关于「数据规模」和「性能」的关系。这个结论直接决定了我们今天如何使用 ViT。

### 三档数据集上的对比

论文在三个规模递增的数据集上做了预训练实验，然后迁移到 ImageNet 上做分类评估：

![dataset-scale-comparison](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch11/dataset-scale-comparison.png)

实验结论可以总结为三句话：

- **在小数据集（ImageNet）上预训练**：ViT 的效果普遍**低于** BiT（基于 ResNet 的强 CNN 基线）。CNN 的归纳偏置在数据稀缺时发挥了关键作用--它内置的局部性和平移等变性让它「自带先验」，即便数据不多也能学到合理特征。而 ViT 缺乏这些先验，需要更多数据才能学到等价能力。
- **在中等数据集（ImageNet-21k）上预训练**：ViT 和 BiT 表现**接近**，ViT 在某些配置下略好，某些配置下略差。这时 CNN 的归纳偏置优势开始被数据量稀释。
- **在大数据集（JFT-300M）上预训练**：ViT 的最佳效果**显著超越** BiT。当数据量足够大时，Transformer 强大的全局建模能力得到释放，归纳偏置的缺失不再成为短板，反而因为没有「局部先验的束缚」而获得了更高的性能上限。

### 这给我们的实践启示

这个结论对实际使用 ViT 有几个非常重要的指导意义：

**第一，直接从头训练 ViT 几乎一定不是好选择。** 除非你有 JFT-300M 量级的数据，否则 ViT 的表现很可能不如同等规模的 ResNet。这也是为什么开源社区发布的 ViT 权重几乎全部是基于大数据集预训练的--没有预训练的 ViT 在小数据集上效果很差。

**第二，预训练数据集越大，迁移效果越好。** 在 ImageNet-21k 上预训练的 ViT 通常优于在 ImageNet 上预训练的 ViT，在 JFT-300M 上预训练的 ViT 又优于 ImageNet-21k。在做下游任务时，尽量选择大数据集预训练的权重作为初始化。

**第三，CNN 的归纳偏置在「数据稀缺 + 中小模型」场景仍然有价值。** 这也是为什么后续工作 DeiT 通过更强的数据增强和蒸馏，让 ViT 在 ImageNet 上从头训练也能接近 ResNet 的效果--它本质上是用「数据增强」和「教师网络」作为外部先验，弥补了 ViT 内部归纳偏置的缺失。

> ViT 的核心实验结论可以浓缩为一句话：**CNN 靠归纳偏置在小数据上赢，ViT 靠全局建模在大数据上赢**。这决定了 ViT 的使用范式--预训练 + 微调，而不是从头训练。

---

## 九、ViT 代码实战

讲完原理，我们用 PyTorch 实现一个完整的 ViT-Base/16。代码会严格对应前面讲过的每个模块，方便对照学习。

### Patch Embedding

```python
class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    用一个卷积 + 展平，把 224x224x3 的图像变成 196x768 的 token 序列
    """
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]  # 196
        # 一个卷积同时完成分块 + 展平 + 线性投影
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # [B, 3, 224, 224] -> [B, 768, 14, 14]
        x = self.proj(x).flatten(2).transpose(1, 2)  # [B, 196, 768]
        x = self.norm(x)
        return x
```

这段代码的关键是 `self.proj(x).flatten(2).transpose(1, 2)` 这一行--一次卷积得到 $[B,768,14,14]$，然后 `flatten(2)` 把空间维度 $14\times14$ 展平成 $196$，得到 $[B,768,196]$，最后 `transpose(1, 2)` 调换成 $[B,196,768]$，正好对应「196 个 token，每个 768 维」。

### Class Token 拼接与位置编码

```python
# 定义可学习的 Class Token，形状 [1, 1, 768]，初始化为全 0
self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
# 定义可学习的位置编码，形状 [1, 197, 768]，初始化为全 0
self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
self.pos_drop = nn.Dropout(p=drop_ratio)

def forward_features(self, x):
    # x 初始形状 [B, 3, 224, 224]
    x = self.patch_embed(x)  # [B, 196, 768]

    # 把 cls_token 扩展到当前 batch_size，拼接到 patch 序列最前面
    cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # [B, 1, 768]
    x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]

    # 加位置编码
    x = x + self.pos_embed  # [B, 197, 768]
    x = self.pos_drop(x)
    return x
```

注意 `cls_token` 和 `pos_embed` 都用 `nn.Parameter` 包装，这意味着它们是可学习参数，会在训练中通过反向传播自动更新。初始化为全 0 是 ViT 论文的做法--让模型从「无信息」状态出发，自己学到合适的位置标记。

### 多头自注意力

这段代码的难点不在原理，而在张量维度的反复变换。先看一张「形状流转图」，把每一步的张量形状在脑子里画清楚，代码就好读了：

![attention-shape-flow](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch11/attention-shape-flow.png)

理解这张图的关键是抓住两次「形状重组」：

- **第一次重组**（reshape + permute）：把 $[B,197,768]$ 投影成 $[B,197,2304]$，再拆成 $[B,197,3,12,64]$--这里的 3 对应 $Q/K/V$，12 对应 12 个头，64 是每个头的维度。`permute(2,0,3,1,4)` 把「3」这个维度提到最前面，是为了能直接用 `qkv[0]/qkv[1]/qkv[2]` 切出 $Q,K,V$。
- **第二次重组**（transpose + reshape）：注意力算完后形状是 $[B,12,197,64]$，要把 12 个头拼回 768 维，所以先 `transpose(1,2)` 调成 $[B,197,12,64]$，再 `reshape` 成 $[B,197,768]$。

```python
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop_ratio=0., proj_drop_ratio=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        # 一次线性投影同时得到 Q、K、V，提升效率
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # x: [B, 197, 768]
        B, N, C = x.shape
        # qkv: [B, 197, 2304] -> reshape -> [B, 197, 3, 12, 64] -> permute -> [3, B, 12, 197, 64]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 各 [B, 12, 197, 64]

        # attn: [B, 12, 197, 197]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # [B, 12, 197, 64] -> [B, 197, 12, 64] -> [B, 197, 768]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
```

对着形状流转图读代码，`permute(2,0,3,1,4)` 这种看上去很劝退的四维以上变换，本质只是在为「切片取 QKV」和「批量矩阵乘法」做准备--它不改变数据内容，只调整维度的顺序。

这段代码还有两个工程上的优化值得注意。第一，`self.qkv` 用一个 `Linear(dim, dim*3)` 一次投影出 $Q,K,V$，而不是分三个 `Linear`--这样只需要一次矩阵乘法，效率更高。第二，`scale = head_dim ** -0.5` 是为了缓解点积过大导致 softmax 进入梯度饱和区，这是原始 Transformer 就有的设计。

### MLP Block

```python
class Mlp(nn.Module):
    """
    Vision Transformer 中的 MLP Block：先扩维 4 倍，经 GELU，再缩回原维度
    """
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)  # 768 -> 3072
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)  # 3072 -> 768
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
```

`hidden_features` 对应模型参数表里的 MLP size（3072），是 `in_features`（768）的 4 倍--这个比例来自原始 Transformer，实践中被证明是个甜点。

### 完整的 Encoder Block

```python
class Block(nn.Module):
    """ViT 的 Encoder Block，采用 Pre-LN 结构"""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_ratio=0., attn_drop_ratio=0., drop_path_ratio=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              qk_scale=qk_scale, attn_drop_ratio=attn_drop_ratio,
                              proj_drop_ratio=drop_ratio)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        # Pre-LN：LayerNorm 放在子层前面，残差连接跨过整个子层
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
```

注意两行 `x = x + self.drop_path(...)`：这是 Pre-LN 结构的标志性写法--LayerNorm 在子层内部，残差连接从子层输入直接跨到子层输出。这种结构让梯度在深层网络中流动更顺畅，是 ViT 训练稳定的关键。

### 完整的 ViT 模型

把以上模块组装起来，就得到了完整的 ViT：

```python
class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0,
                 qkv_bias=True, representation_size=None,
                 drop_ratio=0., attn_drop_ratio=0., drop_path_ratio=0.,
                 norm_layer=None, act_layer=nn.GELU):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        # 1. Patch Embedding
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                      in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # 2. Class Token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # 3. 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        # 4. Encoder Block 堆叠 12 次
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, drop_ratio=drop_ratio,
                  attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # 5. Pre-Logits（可选）+ 分类头
        if representation_size:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)                    # [B, 196, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)       # [B, 197, 768]
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x[:, 0]  # 取 Class Token 对应的输出

    def forward(self, x):
        x = self.forward_features(x)
        x = self.pre_logits(x)
        x = self.head(x)
        return x
```

`forward` 方法清晰地对应了 ViT 的五个阶段：`patch_embed` → 拼接 `cls_token` → 加 `pos_embed` → `blocks`（Encoder 堆叠）→ 取 `cls_token` 经 `pre_logits` 和 `head` 输出。

> 这里我去掉了原论文实现里的 `distilled` 分支--那是 DeiT 为知识蒸馏预留的双 token 设计，标准 ViT 不需要。理解标准 ViT，上面的代码已经完整。

### 实验验证：为什么 ViT 必须用预训练

实际训练时，ViT 对预训练的依赖非常明显。以花的五分类任务为例：

| 训练方式 | ResNet 准确率 | ViT 准确率 |
| --- | --- | --- |
| 从头训 50 epoch | ~0.79 | ~0.56 |
| ImageNet 预训练 + 微调 10 epoch | ~0.915 | ~0.971 |

不用预训练时，ViT 的效果显著弱于 ResNet--这正是 CNN 归纳偏置在小数据上的优势。而一旦使用预训练，ViT 反超 ResNet 接近 6 个百分点，印证了「大数据预训练 + 小数据微调」是 ViT 的正确使用范式。

> 实践中，几乎不会有人从头训练 ViT。开源社区（如 timm 库）提供了大量在大数据集上预训练好的 ViT 权重，直接加载并微调才是标准做法。

---

## 十、ViT 的影响与演进

ViT 的意义远不止「把 Transformer 搬到 CV」--它打开了一扇门，让 CV 进入了「统一架构」时代。ViT 之后涌现了大量后续工作，这里简要提三个最有影响力的方向。

**DeiT：让 ViT 在小数据上也能从头训。** Facebook AI 在 2021 年发表的 DeiT（Data-efficient Image Transformers）通过更强的数据增强（RandAugment、Mixup、CutMix）、知识蒸馏（用 ResNet 作为教师网络）和更精细的训练超参，让 ViT 在 ImageNet 上从头训练也能达到和 ResNet 相当的效果。DeiT 的核心思想是**用「数据增强」和「教师网络」作为外部先验，弥补 ViT 内部归纳偏置的缺失**。

**Swin Transformer：引入层级结构和局部注意力。** 微软亚洲研究院的 Swin 把 ViT 的全局注意力改成了**窗口局部注意力**（在局部窗口内做自注意力，再通过窗口移位让相邻窗口交互），并引入了类似 CNN 的金字塔结构（逐层降采样、特征图尺寸递减）。这种设计既保留了 Transformer 的全局建模能力，又获得了 CNN 的多尺度特征和计算效率，成为了 ViT 在密集预测任务（检测、分割）上的强力替代者。

**MAE：自监督预训练让 ViT 摆脱标注数据依赖。** 何恺明团队的 MAE（Masked Autoencoders）把 NLP 里的「掩码语言建模」思路搬到 CV：随机遮住图像 75% 的 patch，让 ViT Encoder 只处理可见的 25%，再用一个轻量 Decoder 重建被遮住的部分。这种自监督预训练让 ViT 不再依赖海量标注数据，在 ImageNet 上达到了当时的最优性能，也成了后来视觉大模型预训练的事实标准之一。

![vit-evolution](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch11/vit-evolution.png)

> ViT 之后，CV 不再是 CNN 一家独大--Transformer 成了视觉任务的通用架构，甚至让 CV 和 NLP 第一次共享同一套骨干。这种「统一架构」的趋势，直接为后来的多模态大模型（CLIP、Flamingo、GPT-4V）铺平了道路。

---

## 写在最后

ViT 的核心贡献不是发明了什么新结构，而是用一次大胆的「减法」证明了：Transformer 不需要任何针对图像的特殊设计，只要把图像切成 patch 序列，就能在大数据预训练下超越 CNN。这篇文章我们拆解了 ViT 的五个关键组件--Patch Embedding 把图像变成 token 序列，Class Token 作为全局信息汇聚点，位置编码注入位置先验，Transformer Encoder 做特征融合，MLP Head 输出分类结果。

ViT 留给我们两条最重要的经验：第一，**归纳偏置是先验也是天花板**--CNN 的局部性和平移等变性在数据稀缺时是优势，在大数据时代却成了束缚；第二，**预训练决定上限**--ViT 几乎必须依赖大数据集预训练，这也是为什么后续 DeiT、MAE 等工作都在围绕「如何让 ViT 摆脱大数据依赖」做文章。

理解了 ViT，你就掌握了视觉 Transformer 的基本语言--后面的 Swin、MAE、CLIP，乃至多模态大模型，都是在这套语言上做的扩展和改造。下一篇，我们会看 Transformer 如何进一步征服目标检测任务（DETR）。
