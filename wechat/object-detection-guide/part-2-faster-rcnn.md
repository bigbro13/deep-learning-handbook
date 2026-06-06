# 目标检测完全指南（中）：Faster R-CNN 原理详解

> 本文是「目标检测完全指南」系列的中篇，聚焦两阶段检测器的集大成者——Faster R-CNN。我们将从 R-CNN 的前传故事讲起，逐层拆解 RPN、Anchor、ROI Pooling 和端到端训练的设计精髓。
>
> 上篇：[原理、指标与核心组件]()
> 下篇：[YOLO 系列全解析]()

---

## 6.5 Two-Stage 检测器：Faster R-CNN 原理详解

Faster R-CNN 是两阶段检测器的集大成者，2015 年发布至今仍被广泛使用和改进。理解它不仅是为了使用它，更是因为**它定义了目标检测的许多基础概念**——anchor、RPN、ROI Pooling、端到端训练——这些概念深刻影响了后续几乎所有检测器的设计。

### 6.5.1 前传：为什么需要两阶段？

在 Faster R-CNN 出现之前，目标检测的流程是这样的：

**R-CNN（2014）** 是第一个把 CNN 引入目标检测的工作。思路很直接：

1. 用 Selective Search 算法在原图上生成大概 2000 个候选框
2. 把每个候选框从原图上裁下来，缩放到统一尺寸（227×227）
3. 把 2000 个裁剪图逐个送进 CNN（比如 AlexNet）提取特征
4. 用 SVM 分类器对每个特征向量做分类
5. 用线性回归器对框的位置做微调

每一步思路都很清晰，但实战中慢得让人绝望——**2000 个候选框意味着 2000 次 CNN 前向传播**。而且训练是分步走的（先训 CNN、再训 SVM、再训回归器），整个流程不优雅。

**Fast R-CNN（2015）** 的关键改进一句话就说明白了：**整张图只过一次 CNN，所有候选框共享这张特征图**。这就像你用 Google Maps 提前加载了整个区域的卫星图，每个具体地点只需要在上面标位置就行，不用每次都重新加载。Fast R-CNN 还引入了 ROI Pooling（后面会细讲），把不同大小的候选框统一映射为固定尺寸的特征向量，方便后面接全连接层。

![selective-search](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch06/selective-search.png)

但 Fast R-CNN 还留了一个尾巴——**候选框仍然是用 Selective Search 这种外部算法生成的**，不是神经网络自己产生的。

### 6.5.2 Faster R-CNN 的核心思路



Faster R-CNN 的突破就是把这个"尾巴"也砍掉了——**用 RPN（Region Proposal Network，区域提议网络）替代 Selective Search，让生成候选框的过程也由神经网络完成**。至此，整个目标检测流程首次实现了完全的端到端。

![image-20260606152317595](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch06/faster-r-cnn.png)

Faster R-CNN 的完整流程如下：

![faster-rcnn-pipeline](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch06/faster-rcnn-pipeline.png)

可以看到，Faster R-CNN 实际上就是 **RPN（候选框生成）+ Fast R-CNN（候选框分类+回归）**，两者共享同一个 Backbone 的特征图。

### 6.5.3 Backbone —— 特征提取基底

Backbone 是整条流水线的基础，负责把原始图像转化成多层次的语义特征。Faster R-CNN 论文使用的是 **VGG16**——一个在当时效果不错、结构又比较规整的网络。

![image-20260606152443408](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch06/vgg-16.png)

以 VGG16 为例来分析 Backbone 做了什么：
- 13 个卷积层，卷积核全是 3×3，padding = 1（卷积后特征图大小不变）
- 4 个池化层（Max Pooling 2×2, stride = 2），每次池化特征图尺寸减半
- VGG16 原始有 5 个池化层，但 Faster R-CNN **丢弃了最后一个**——否则特征图会被下采样 32 倍，过于粗糙

所以一张 $M \times N$ 的图片经过 Backbone 后，特征图的大小变为 $\frac{M}{16} \times \frac{N}{16}$。比如输入 800×600，特征图就是 50×38。

需要强调的是——Backbone 是**可替换**的。后来大家普遍改用 ResNet（残差网络，训练更稳定）、FPN（特征金字塔，多尺度更好）等作为 Backbone。RPN + Detection Head 的设计与具体 Backbone 解耦，这也是 Faster R-CNN 框架灵活性的体现。

### 6.5.4 RPN —— 网络自己学会"看哪里"

RPN 是 Faster R-CNN 的灵魂。它的工作原理可以这样理解：

在共享特征图上，用一个 3×3 的窗口滑过每个位置。对于每个位置，RPN 要做两件事——**判断这里是不是"可能有东西"（分类），以及可能是什么样的框（回归）**。

![rpn-workflow](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch06/rpn-workflow.png)

两个分支的作用很明确：

![image-20260606152737829](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch06/RPN.png)

- **分类分支**：对 9 个 anchor 分别打分，判断它是"前景"（包含真实目标）还是"背景"（啥也不是）。为什么是 18 个通道？因为 9 个 anchor × 2 个类别（前景和背景）= 18
- **回归分支**：对 9 个 anchor 分别输出 4 个微调参数 ($t_x, t_y, t_w, t_h$)，告诉每个 anchor 应该如何调整自己才能更贴近真实目标。为什么是 36 个通道？因为 9 个 anchor × 4 个偏移量 = 36

**特征图上的一个点对应原图的哪个位置？** 这是一个关键问题。因为 Backbone 做了 4 次 2× 池化，总下采样倍数是 16。所以特征图上坐标 $(x_{feat}, y_{feat})$ 对应原图的中心点为 $(16 \times x_{feat}, 16 \times y_{feat})$。在这个原图位置上，画出 9 个 anchor——它们就是 RPN 后续处理的基本单元。

**什么算正样本？什么算负样本？**

训练 RPN 时，并不是所有 anchor 都参与训练：
- **正样本**：和某个真实框 IoU 最大的 anchor，或者和任何一个真实框 IoU > 0.7 的 anchor（绝大多数情况满足后者）
- **负样本**：和所有真实框的 IoU 都 < 0.3 的 anchor
- IoU 在 0.3~0.7 之间的 anchor **直接忽略**，不参与计算损失

> RPN 的设计哲学是"分而治之"——分类分支专做前景/背景判断，回归分支专做位置修正，最后在 Proposal 层汇合，组合出高质量的候选框。

### 6.5.5 RPN 的损失函数

RPN 的损失也由两部分组成，和它的两个输出分支对应：

$$L(\{p_i\}, \{t_i\}) = \frac{1}{N_{cls}} \sum_i L_{cls}(p_i, p_i^*) + \lambda \frac{1}{N_{reg}} \sum_i p_i^* \cdot L_{reg}(t_i, t_i^*)$$

拆开来看：
- $L_{cls}$：二分类交叉熵损失，判断每个 anchor 是前景还是背景
- $L_{reg}$：Smooth L1 损失，**只对正样本计算**（因为 $p_i^* = 1$ 时回归才有意义——你让模型对一个全是背景的 anchor 回归位置就离谱了）
- $p_i^*$：真实标签（1 表示该 anchor 是正样本，0 表示负样本）
- $\lambda$：平衡系数，让两类损失的数值量级相仿

**怎么把 anchor 变成最终的预测框？** RPN 输出的是偏移量 $(t_x, t_y, t_w, t_h)$，需要解码到绝对坐标。编码方式（以 anchor $(x_a, y_a, w_a, h_a)$ 为基准）：

$$t_x = \frac{x - x_a}{w_a}, \quad t_y = \frac{y - y_a}{h_a}, \quad t_w = \log\frac{w}{w_a}, \quad t_h = \log\frac{h}{h_a}$$

为什么要用 $\log$？因为框的宽高可能差几个数量级，取对数后数值范围被压缩，网络更容易学习。推理时反过来解码就行。

### 6.5.6 ROI Pooling —— 把不同大小的框变成统一尺寸

RPN 输出了约 300 个高质量的候选框（经过 NMS 筛选后）。但这些候选框大小不一——有的是 20×30 的小目标，有的是 200×150 的大目标。而后面接的全连接层只能处理固定维度的输入。

![image-20260606153913378](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch06/roi_pooling.png)

**ROI Pooling 的作用就是把一个任意大小的候选框映射为一个固定大小的特征向量（比如 7×7）**。

操作步骤：
1. 候选框坐标除以 stride（16），缩放到特征图尺度
2. 把候选区域划分为 $k \times k$（比如 7×7 = 49）个大小相等的子区域
3. 每个子区域内做 Max Pooling，输出一个值
4. 最终得到一个 $k^2 \times C$（49 × 通道数）的固定长度特征向量

![roi-pooling-flow](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch06/roi-pooling-flow.png)

**ROI Pooling 存在一个问题——量化误差**

划分过程中有两次"取整"操作：
1. 将原图坐标缩放到特征图时，坐标是浮点数但像素位置是整数——取整
2. 分配子区域时，区域大小可能不是整数——取整

这两次取整在检测任务中问题还不算太大（差一两个像素一般也不影响判断），但在**实例分割**这种需要像素级精度的任务里就很致命了。

### 6.5.7 ROI Align —— 消除量化误差

ROI Align（出自 Mask R-CNN）的改进思路很自然：**既然取整会造成误差，那就不取整，用插值**。

ROI Align 的具体做法：
1. 保留浮点坐标，不取整
2. 划分子区域后，在每个子区域内**均匀采样 4 个点**
3. 对每个采样点用**双线性插值**（利用周围 4 个真实像素计算插值结果）获得精确的像素值
4. 取 4 个采样点的 Max 或 Average

全程没有量化取整，每个值都是"算出来的精确值"。ROI Align 现在是目标检测和实例分割任务的标准配置。

### 6.5.8 最终的分类与精修

ROI Pooling/Align 之后，流程就进入了收尾阶段：

- 特征向量送入全连接层
- 分类分支：Softmax 输出该候选框属于每个类别的概率（包括一个"背景"类）
- 回归分支：对每个类别输出一组精修偏移量，再次微调边界框

这里的损失函数和 RPN 类似，也是分类损失 + 回归损失的多任务形式。最后再过一次 NMS，去除同一类别中高度重叠的框，就得到了最终的检测结果。

### 6.5.9 小结

回顾 Faster R-CNN 的完整流程：

1. 整张图送入 Backbone → 共享特征图
2. RPN 在特征图上滑动，为每个位置生成 9 个 anchor，输出前景/背景分数和位置修正量 → 约 300 个高质量候选框
3. 候选框映射回特征图 → ROI Pooling/Align 统一尺寸 → 全连接层
4. 最终分类 + 回归精修 → NMS 去重 → 输出

Faster R-CNN 的核心贡献是什么？**用 RPN 替代外部算法，使整个系统"学"出了候选框生成的能力，实现了真正的端到端训练**。这个架构范式影响深远——今天很多检测器虽然外观不同，但底层大多能找到 RPN + ROI 的影子。

---
