# 目标检测完全指南（下）：YOLO 系列全解析

> 本文是「目标检测完全指南」系列的下篇，完整梳理 YOLO 从 v1 到 v26 的技术演进——从 7×7 网格的朴素想法，到 Anchor-Free 架构革命，再到多任务统一和极致边缘部署。还会对比 YOLOv5 / v8 / v11 / v26 四代版本的核心差异。
>
> 上篇：[原理、指标与核心组件]()
> 中篇：[Faster R-CNN 原理详解]()

---

## 6.6 One-Stage 检测器：YOLO 系列

如果说 Faster R-CNN 代表了两阶段范式的最高水准，那 YOLO 系列就是单阶段范式的旗帜。YOLO 的名字本身就是宣言——**You Only Look Once（你只需要看一次）**——强调的就是直接把检测问题一步到位地解决。

### 6.6.1 YOLOv1 —— 把检测变成一个回归问题

YOLOv1（2015）提出的思想在当时是颠覆性的。之前的主流做法都是"先找候选区域，再逐个分类"，而 YOLOv1 说：**不用那么麻烦，把整张图喂给一个神经网络，直接输出所有检测结果**。

![yolov1-pipeline](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch06/yolov1-pipeline.png)

核心机制是**网格划分**。让我一步一步拆解：

**第一步**：把输入图像平均分成 $S \times S$ 个格子（$S=7$）。就像在图像上画了一个 7×7 的棋盘。

**第二步**：如果某个目标的**中心点**落在了某个格子里，那么这个格子就"承包"了这个目标的检测任务。每个格子最多预测 $B$ 个目标（$B=2$）。

**第三步**：每个格子要输出三类信息：

- **$B$ 个边界框**，每个框 5 个值：$(x, y, w, h, confidence)$
  - $(x, y)$：框的中心坐标，**相对于当前格子的偏移**（不是整张图），范围 0~1
  - $(w, h)$：框的宽高，**相对于整张图的比例**（不是像素值），范围 0~1
  - $confidence$：置信度 = $\text{Pr(Object)} \times \text{IoU}_{\text{pred}}^{\text{truth}}$（即该框包含目标的概率乘上预测框与真实框的 IoU）
- **$C$ 个类别概率**：条件概率 $\text{Pr(class} \mid \text{object)}$（在该格子已包含目标的前提下，属于各个类别的概率），VOC 数据集 20 类

**第四步**：把所有信息拼起来。输出张量的维度是 $S \times S \times (5B + C)$ = $7 \times 7 \times (10 + 20)$ = $7 \times 7 \times 30$。

![yolov1-grid](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch06/yolov1-grid.png)

YOLOv1 的设计非常优雅——检测被完全简化为一个回归问题，没有复杂的中间步骤。但代价也很明显：7×7 的粗糙网格意味着**小目标和密集目标效果很差**——一个格子里挤了多个小目标，YOLOv1 也只能检测出 2 个。而且直接用全连接层预测，空间信息丢失较多。

> **理解 YOLOv1 的关键**：它把检测问题重新定义为"空间划分 + 每个格子独立预测"。虽然精度不如两阶段方法，但它证明了单阶段检测器的可行性，为后续所有 YOLO 版本奠定了基础。

### 6.6.2 奠基阶段（YOLOv2 ~ YOLOv3）

**YOLOv2（2016）** 是多点开花的改进版本：

最关键的改动是**引入 Anchor 机制**。YOLOv1 让网络直接预测绝对坐标，YOLOv2 改成了预测"相对于 anchor 的偏移"。怎么确定 anchor 的尺寸？用 **K-means 在训练集标注框上聚类**，自动找出最好的一组 anchor。这不仅更贴合数据，还大幅提升了定位精度。

其他重要改进：
- 每层都加 **Batch Normalization**，训练更稳定，mAP 涨了 2%
- 先在 448×448 的高分辨率上微调分类网络，再切到检测——让网络适应高分辨率输入
- **多尺度训练**：每 10 个 batch 随机换一个输入分辨率，网络对不同大小的图都能处理

**YOLOv3（2018）** 最重要的改进是**多尺度预测**。它不再只在最后一层特征图上做检测，而是在 3 个不同分辨率的特征图上各做一次——大分辨率图检测小目标，小分辨率图检测大目标。这和 FPN 的思路一致，大幅提升了小目标的检测效果。

另外，YOLOv3 的 Backbone 换成了 **Darknet-53**（53 个卷积层 + 残差连接），ImageNet 分类精度匹敌 ResNet-152，但速度快了 2 倍。

### 6.6.3 工程化精进阶段（YOLOv4 ~ YOLOv7）

这一阶段的共同特点不是单一突破，而是**系统性地整合各种技巧，在工程上做到极致**。

**YOLOv4（2020）** 最经典——它把当时所有有效的技巧系统地组织成两类：

- **Bag of Freebies（白送的技巧）**：训练时用、推理时不增加计算量——Mosaic 数据增强、CIoU 损失、标签平滑、余弦退火学习率……全加上
- **Bag of Specials（花点小钱的技巧）**：推理时略微增加计算但大幅提升精度——CSPDarknet53 骨干、SPP 模块、PANet 特征融合、Mish 激活函数

YOLOv4 的整体架构确立了后续 YOLO 版本的基础框架——Backbone + Neck + Head 三段式结构：

![yolo-evolution-engineering](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch06/yolo-evolution-engineering.png)

#### 6.6.3.1 YOLOv5 —— 工程化标杆

YOLOv5（2020）虽然没发论文，但在工业界的影响力远超大多数有论文的模型。它的核心贡献不是某个算法上的突破，而是**把目标检测的工程化做到了极致**。

**整体架构**

YOLOv5 继承了 YOLOv4 的 Backbone-Neck-Head 三段式设计，但在每个阶段都做了精心的工程优化：

![image-20260606155640508](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch06/YOLOv5.png)

**C3 模块 —— 骨干的基本单元**

C3（CSP Bottleneck with 3 convolutions）是 YOLOv5 骨干网络的核心构建块，它借鉴了 CSPNet（跨阶段部分网络）的思想：

![yolov5-architecture](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch06/yolov5-architecture.png)

C3 的设计哲学是"分而治之"——将输入通道一分为二，一半经过 Bottleneck 做深度特征提取，另一半走捷径直接透传，最后再拼回来。这样做的效果是：**梯度可以通过 shortcut 路径更直接地回传到浅层**（缓解深层网络的梯度消失），同时只对一半通道做卷积也降低了计算量。

**模型族 —— 一套设计，五种规格**

YOLOv5 最大的工程创新之一就是通过两个简单的参数——深度乘数（depth_multiple）和宽度乘数（width_multiple）——控制模型大小，生成了 5 种规格：

![yolov5-output](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch06/yolov5-output.png)

- **深度乘数**：控制每个 C3 模块中 Bottleneck 的重复次数——乘数越大，网络越深，特征提取能力越强
- **宽度乘数**：控制每层卷积的输出通道数——乘数越大，每层能学到的特征越多

> 从 YOLOv5n（1.9M 参数，适合树莓派）到 YOLOv5x（86.7M 参数，适合云端 GPU），用的都是**同一套架构代码**，只是改了这两个数字。这种设计让开发者可以根据硬件条件灵活选用，而不需要重新设计网络。

**YOLOv5输出结构解析**

![image-20260606162103299](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch06/YOLOv5_output.png)

维度含义

![image-20260606162158073](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch06/YOLOv5_out1.png)

结果解析
![image-20260606162246387](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch06/YOLOv5_out2.png)

**为什么 YOLOv5 如此流行？**

YOLOv5 的火爆不是偶然。它同时拥有三样东西：
- **PyTorch 原生实现**：代码简洁易读，社区贡献门槛低
- **完善的部署工具链**：一键导出到 TensorRT、ONNX、CoreML、TFLite，覆盖云-边-端全场景
- **开箱即用的训练流程**：自动数据增强、自动锚框聚类、自动学习率调度，降低了上手成本

这套"论文可以不发，代码一定要好用"的理念，让 YOLOv5 成了目标检测领域事实上的工程标准。

**YOLOv6（2022）** 美团出品，主打**结构重参数化**——训练时用多分支结构追求更好性能，推理前把多分支"融合"成单路来提速。RepVGG 风格的 EfficientRep 骨干就是这个思想的体现。

**YOLOv7（2022）** 提出了 **E-ELAN（扩展高效层聚合网络）**，在保持梯度完整性的同时提升网络学习能力，并大规模应用了模型重参数化技术。

### 6.6.4 现代化转型阶段（YOLOv8 ~ YOLOv13）

从 YOLOv8 开始，YOLO 系列进入了全新的阶段——抛掉历史包袱，全面拥抱现代化设计。

![image-20260606155914187](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch06/YOLOv8.png)

**YOLOv8（2023）是最重要的转折点**，它做了三个根本性的架构改变：

**改变一：C2f 替换 C3 模块**

![image-20260606160110100](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch06/c3-c2f.png)

C2f 借鉴了 YOLOv7 的 ELAN 设计——更多的跳层连接（skip connection）+ Split 操作。这带来了**更丰富的梯度回传路径**（训练更容易），同时因为砍掉了分支中的部分卷积，计算量反而更低。

**改变二：解耦头（Decoupled Head）替代耦合头**

![image-20260606160214900](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch06/head.png)

在 YOLOv5 中，分类和定位是用同一层卷积完成的（耦合设计）。YOLOv8 把它们拆成了两条独立的分支——分类分支专注语义特征，回归分支专注空间特征。道理很简单：**判断"这是什么"和"它在哪"是两个不同性质的任务，不应该混在一起**。

**改变三：彻底抛弃 Anchor，转向 Anchor-Free**

这可能是最激进的一步。YOLOv8 不再预设任何 anchor，而是直接让网络预测目标的中心点和宽高。配合 DFL（Distribution Focal Loss，把边界框回归建模为位置概率分布的预测）+ CIoU 损失，定位比有 anchor 时更灵活。

YOLOv8 还用了一个叫 **TaskAlignedAssigner** 的正样本匹配策略：

$$\text{align\_score} = \text{class\_score}^\alpha \times \text{IoU}^\beta$$

不是简单地看 IoU 高低，而是把分类质量和定位质量捆在一起评估——"你的分类有把握 AND 你的定位精准"才算好样本。这个设计让置信度分数更有意义：它直接衡量了预测的**整体质量**。

**DFL 机制详解**

YOLOv8 中另一个关键创新是 DFL（Distribution Focal Loss，分布焦点损失）。传统边界框回归直接预测一个坐标值（比如宽度 = 203 像素），但这对模型来说要求很高——网络需要非常精确地输出一个数字。DFL 换了一个思路：**不直接预测一个值，而是预测一个位置的概率分布**。

![yolov8-architecture](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch06/yolov8-architecture.png)

具体来说，DFL 把边界框的每条边（左、上、右、下）的坐标范围划分为 $n$ 个离散位置，网络为每个位置输出一个概率值。最终的坐标值通过对这 $n$ 个位置做**概率加权求和**得到。这样做有两个好处：

- **定位更精确**：概率分布可以"捕捉"到亚像素级别的偏移——即使离散位置的步长是 1 像素，通过概率加权也可以输出 201.8 这样的浮点值
- **对小偏移敏感**：网络不仅要让"正确位置"附近的概率高，还要让离正确位置越远的概率越低（用 cross-entropy 约束整个分布的形状），这比直接回归 L1/L2 损失提供了更丰富的梯度信号

> DFL 可以理解为：把"定位"这个回归问题重新表述为一个"在有限个候选位置上做概率分类"的问题。回归 → 分类，网络学起来更稳定。

**YOLOv9（2024）** 聚焦在深层网络的信息瓶颈问题上，提出 PGI（可编程梯度信息），让深层也能接收到完整的输入信息。

**YOLOv10（2024）** 实现了 YOLO 系列多年来的终极目标——**真正的端到端、无需 NMS**：

![c3-c2f-comparison](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch06/c3-c2f-comparison.png)

核心思想是**让模型在训练阶段就学会"做选择"**——通过一对一标签分配强迫模型为每个目标只保留一个最优的预测框。既然模型训练时已经学会了"一个目标只出一个框"，推理时自然就不再需要 NMS 去冗余了。

**YOLOv11（2024）—— 多任务统一**

如果说 YOLOv8 是"内部架构重构"，那 YOLOv11 就是"能力横向扩展"。它是 Ultralytics 首个官方多任务统一框架——一个模型同时支持检测、实例分割、旋转框检测（OBB）和人体姿态估计。

![image-20260606160731858](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch06/YOLOv11.jpg)

**C3k2 模块 —— 更高效的骨干单元**

YOLOv11 把骨干瓶颈块从 C2f 替换为 C3k2。C3k2 的设计目标是——用更少的计算成本实现同等或更好的特征提取效率：

![decoupled-head](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch06/decoupled-head.png)

C3k2 的核心思路是在更低的通道维度上做计算——通过额外的 Split 操作将特征拆得更细，只在必要的地方保留 Bottleneck 的深度处理能力。结果就是：计算量比 C2f 更小，但对关键特征的提取能力不打折扣。

**多任务 Head 架构**

YOLOv11 的真正亮点在于 Head 的多任务扩展。不同于训练 4 个独立模型，YOLOv11 共享 Backbone 和 Neck，只在 Head 层分支处理不同任务：

![image-20260606161506939](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch06/YOLOv11_head.png)

四个 Head 各有特点：

- **检测 Head**：标准配置，分类分支 + 回归分支，输出 4 个坐标值定义边界框
- **分割 Head**：分两步——先生成一组共享的**掩膜原型**（prototype masks，覆盖全图的基础掩膜模板），再为每个检测到的实例预测一组**掩膜系数**（mask coefficients，在当前原型上的线性组合权重）。原型 × 系数的组合 = 每个实例独有的二值掩膜。这种"原型 + 系数"的设计非常高效——原型只需算一次，每个实例只需多输出几个系数
- **姿态估计 Head**：在检测头的基础上增加关键点回归分支，直接预测人体 17 个关键点的坐标和每个点的可见性（可见/遮挡/不在图中）。关键是：关键点坐标是基于检测框中心做归一化偏移，不依赖全图坐标，这让预测更稳定
- **OBB Head（旋转框检测）**：传统的水平框只能输出 $(cx, cy, w, h)$，旋转框多了一个角度维度 $(cx, cy, w, h, angle)$，能更紧密地包裹倾斜的物体（比如航拍图中的船只）。损失函数也做了相应调整以适应角度的周期性

> YOLOv11 的核心意义：**一个模型 = 四个任务的解决方案**。以前需要部署 3-4 个模型才能覆盖检测、分割、姿态估计的需求，现在一个模型全包了，不仅节省了部署维护成本，共享的 Backbone+Neck 也让多任务之间可以相互增强。

**YOLOv12/YOLOv13（2025）** 分别引入了区域注意力模块和超图关联增强模块，继续在精度和效率上做精细打磨。

### 6.6.5 YOLOv26 —— 极简主义的边缘革命

到了 YOLOv26，迭代方向发生了根本性转变——**不是再加功能，而是做减法**。一切都是为了一个目标：让模型在 CPU 和边缘设备上以代际飞跃的速度运行。

YOLOv26 把四个"不必要的"东西砍掉了：

**1. 砍掉 DFL（Distribution Focal Loss）**

DFL 的机制前面介绍过——把边界框回归建模为位置概率分布，用加权求和得到最终坐标。这确实提升了定位精度，但它有两个实际代价：

- **推理开销**：DFL 需要维护 $n$ 个离散位置的概率输出（通常 $n=16$），然后做 softmax + 加权求和。在 GPU 上这点开销不算什么，但在 CPU/边缘 NPU 上，每一步额外的数学运算都在挤压本就不富裕的算力预算
- **部署复杂度**：DFL 的输出结构不是标准的回归头，有些推理框架（尤其是边缘端的轻量框架）需要特殊适配

YOLOv26 直接砍掉 DFL，回到更直接的回归方式——用实践证明在大量边缘场景下，DFL 带来的那点精度提升不值得那个计算成本。

**2. 砍掉 NMS —— 原生端到端**

YOLOv10 已经探索了"训练时端到端、推理时无 NMS"的路子，YOLOv26 做得更彻底。关键不在于"不用 NMS"这个结果，而在于**训练阶段的设计让模型自然而然地就不出重复框**：

- **一对一标签分配**：每个真实目标在训练时只匹配一个最优预测框——模型从根源上学到了"一个目标只出一个框"
- **内部竞争机制**：相邻位置的预测之间形成隐式竞争，迫使模型自己决定由哪个位置"负责"某个目标

推理时直接输出，没有 NMS 后处理延迟——这在对延迟敏感的实时系统中意义重大。

![yolov26-philosophy](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch06/yolov26-philosophy.png)

**3. ProgLoss + STAL —— 让训练更稳、小目标更好**

边缘部署场景的痛点是什么？图像质量差、小目标多、光照不足。传统的标签分配和损失加权策略对这些情况不够敏感。YOLOv26 引入了两个针对性策略：

- **ProgLoss（Progressive Loss Balancing，渐进式损失平衡）**：任务A（分类）、任务B（定位）、任务C（DFL 已被移除但可能还有置信度回归）的损失不是简单相加，而是让模型在训练早期优先学好"容易的"任务，随着训练推进逐步增加"难的"任务的权重。这就像学开车——先练直线，再练弯道，最后上高速
- **STAL（Small Target Aware Label assignment，小目标感知标签分配）**：针对小目标（面积很小，特征弱）修改了正负样本匹配逻辑——给靠近小目标中心点的位置更高的匹配优先级，让网络更不容易漏掉这些"小东西"

两项结合，模型在边缘场景中的可靠性明显提升——小目标召回率提升而不牺牲大目标精度。

**4. MuSGD 优化器 —— 从大模型训练中"偷师"**

MuSGD 的灵感来自大语言模型训练——Moonshot AI 的 Kimi K2 训练经验被融合到了 SGD 框架中。SGD 的传统优势是**泛化能力强**（因为每次更新的噪声天然起到了正则化作用），但**收敛慢**（因为步长和方向都不够"聪明"）。MuSGD 保留了 SGD 的泛化优势，同时通过动量和自适应调节大幅提升了收敛速度和训练稳定性。

> YOLOv26 的哲学是"能不要的都不要"。这不是为了炫技，而是为了回答一个根本问题：**一个检测器最少需要什么才能有效工作？** 答案是——不需要 DFL、不需要 NMS、不需要复杂的优化器。极简不代表简陋，而是用更聪明的训练策略弥补架构的"朴素"。

### 6.6.6 YOLO 系列演进总结

![yolo-evolution-timeline](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch06/yolo-evolution-timeline.png)

演进主线非常清晰：**精度 → 速度与精度的平衡 → 多任务统一 → 极致边缘部署**。

---

## 6.7 YOLO 版本对比

下面选取三代关键版本（YOLOv5 → YOLOv8 → YOLOv11 → YOLOv26），对比它们在架构设计上的核心变迁。理解这种变迁，比记住每个版本的细节更重要。

### 6.7.1 YOLOv5 → YOLOv8：从 Anchor-Based 到 Anchor-Free 的跨越

这是 YOLO 历史上最彻底的一次架构重构，几乎每个模块都变了。

**Backbone 变了什么？**

最简单的变化是模块名从 C3 换成了 C2f，但背后的设计哲学完全不同。C3 是一个比较简单直接的 CSP 模块——三条分支汇合，计算沿着固定路径走。C2f 借鉴了 YOLOv7 ELAN 的设计，在 C3 基础上加了更多的跳层连接和 Split 操作，取消了分支中的部分卷积。

理解 C2f 优势的关键是**梯度流**——训练时反向传播的梯度通过更多路径回流到浅层，深层网络收到更丰富的梯度信息。同时因为砍掉了一些卷积，计算量还少了。这就像修路——C3 是三条主干道，C2f 是主干道 + 一堆小支路，交通（梯度）更通畅，还更省地（计算量）。

**Neck 变了什么？**

两者都使用 PAN-FPN 结构（一种在 FPN 自顶向下基础上又加了自底向上路径的特征融合方式）。YOLOv8 在 Neck 中也用 C2f 替换了 C3，并去掉了上采样阶段的部分卷积。

**Head —— 变化最大的部分**



三个核心差异：

| 维度 | YOLOv5 | YOLOv8 | 为什么更好 |
|---|---|---|---|
| **Head 结构** | 耦合头：一层卷积分时做分类+定位 | 解耦头：两条独立分支 | 分类关注"是什么"（语义），定位关注"在哪"（空间），分开后互不干扰 |
| **Anchor 方式** | Anchor-Based：预设锚框，预测相对偏移 | Anchor-Free：直接预测中心+宽高 | 去掉锚框超参数，流程更简洁，小目标更友好 |
| **标签分配** | 基于 IoU 的简单匹配 | TaskAlignedAssigner | 同时评估分类质量和定位质量，匹配更合理 |
| **置信度意义** | 存在性分数 × 类别分数 | 对齐分数（分类和定位的联合质量） | 置信度高 = 分类有把握 AND 定位精准，更可信 |

从 YOLOv8 往后，YOLO 系列全面转向 Anchor-Free。这不是偶然——去掉锚框意味着更少的超参数、更简单的正负样本定义、更高效的解码。实践证明 Anchor-Free 不仅没有损失精度，反而因为更灵活带来了提升。

### 6.7.2 YOLOv8 → YOLOv11：从单一检测到多任务统一

如果说 v5→v8 是"内部重构"，那 v8→v11 就是"能力扩展"。

![yolo-multi-task](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch06/yolo-multi-task.png)

架构层面，YOLOv11 把骨干瓶颈块从 C2f 换成了 **C3k2**——更小的计算成本，更高效的特征提取。其余基础配置（Anchor-Free、解耦头、TaskAlignedAssigner）基本保持不变。

真正的亮点在于 **Head 的多任务扩展**：

- **检测**：标准分类分支 + 回归分支，输出边界框和置信度
- **分割**：在检测头基础上，增加掩膜原型分支（生成共享的原型掩膜）和掩膜系数分支（为每个实例选择原型掩膜的权重），最终每个实例得到独立的二值掩膜
- **姿态估计**：在检测头基础上，增加关键点回归分支，直接预测人体关键点的坐标和可见性
- **旋转框（OBB）**：标准回归输出 4 个参数，旋转框输出 5 个——多了个角度 $(cx, cy, w, h, angle)$，并调整损失函数以适应角度的周期性



YOLOv11 的核心意义在于：**一个模型 = 多个任务的解决方案**。以前需要部署 3-4 个模型才能覆盖检测、分割、姿态估计的需求，现在一个模型全包了。

| **任务类型**         | **核心输出/目标**                                 | **检测头关键设计**                                           | **主要损失函数**                                             |
| -------------------- | ------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **检测 (Detect)**    | 类别、矩形框(`x,y,w,h`)、置信度                   | 分类与回归解耦的双分支头，无锚框设计。                       | – 分类: `BCE Loss`– 回归: `DFL Loss` + `CIoU Loss`           |
| **实例分割 (Seg)**   | **每个实例**的二值掩膜                            | 在检测头基础上，**增加一个掩膜原型分支**和**掩膜系数分支**。 | – 检测部分: 同上– 分割: `BCE Loss` (掩膜与GT掩膜的损失)      |
| **姿态估计 (Pose)**  | 每个人体实例的**关键点坐标** (`x,y`) 和**可见性** | 在检测头基础上，**增加一个关键点回归分支**，直接预测每个关键点的相对坐标。 | – 检测部分: 同上– 关键点: `BCE Loss` (可见性) + `MSE Loss` (坐标) |
| **旋转框检测 (OBB)** | 带角度的矩形框 (`cx,cy,w,h,angle`)                | 在回归分支中将4个参数扩展为**5个参数** (`cx,cy,w,h,angle`)，并调整损失函数以适应角度周期性。 | – 分类: `BCE Loss`– 回归: `DFL Loss` + `Rotated CIoU Loss`   |

### 6.7.3 YOLOv11 → YOLOv26：为边缘而生的极简主义

如果说 v8→v11 是"做加法"（加任务、加能力），那 v11→v26 就是**做减法**——把一切对边缘部署不必要的东西都砍掉。

![yolov26-edge](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch06/yolov26-edge.png)

逐条拆解 YOLOv26 为什么这么做：

**1. 去掉 DFL**

DFL 的机制是把边界框回归建模成一个概率分布预测——比如宽度不是直接预测"203 像素"，而是预测"可能是 200、201、202……"等一系列离散值上的概率，然后加权平均。这样做的好处是定位更精确（尤其对小偏移敏感），但代价是增加了推理时的计算和部署复杂度。

在算力充裕的 GPU 上这不叫事，但在 CPU/边缘设备上，每多一步计算就是一个瓶颈。YOLOv26 直接砍掉 DFL，回到更简单的回归方式——用实践证明在很多场景下，DFL 带来的那点精度提升不值得那个计算成本。

**2. 原生端到端，彻底不用 NMS**

YOLOv10 已经探索了端到端的路子，YOLOv26 做得更彻底。关键不在于"不用 NMS"这个结果，而在于**训练阶段的设计让模型自然而然地就不出重复框**——一对一标签分配 + 内部竞争机制确保了每个目标只有一个最优预测。推理时直接输出，没有后处理延迟。

**3. ProgLoss + STAL**

边缘场景的痛点是什么？小目标多、低光照、可见度低。传统的标签分配和损失加权策略对这些情况不够敏感。ProgLoss（渐进式损失平衡）让模型在训练过程中逐步调整对不同难度样本的注意力，STAL（小目标感知标签分配）则专门为小目标优化了正负样本的匹配逻辑。两项结合，模型在边缘场景中的可靠性明显提升。

**4. MuSGD 优化器**

这个优化器的灵感来自大语言模型训练——Moonshot AI 的 Kimi K2 训练经验被融合到了 SGD 框架中。SGD 的传统优势是泛化能力强（因为噪声大），但收敛慢；MuSGD 保留了 SGD 的泛化优势，同时大幅提升了收敛速度和训练稳定性，在大型复杂数据集上表现突出。

### 6.7.4 三代演进大图景

把三代版本的关键变化放在一起，演进图景非常清晰：

| 对比维度 | YOLOv5 | YOLOv8 | YOLOv11 | YOLOv26 |
|---|---|---|---|---|
| **Anchor** | Anchor-Based | Anchor-Free | Anchor-Free | Anchor-Free |
| **Head** | 耦合头 | 解耦头 | 解耦头 | 解耦头 + 简化 |
| **骨干模块** | C3 | C2f | C3k2 | 极致轻量 |
| **DFL** | 无 | 有 | 有 | **移除** |
| **NMS** | 需要 | 需要 | 需要 | **不需要** |
| **多任务** | 仅检测 | 检测 + 分割 | + OBB + 姿态 | + 优化版全任务 |
| **优化器** | SGD | SGD/Adam | Adam | **MuSGD** |
| **设计哲学** | 工程化落地 | 架构现代化 | 能力横向扩展 | 边缘纵向深挖 |

概括三代演进的主题：**v5→v8 是架构革命（去掉历史包袱），v8→v11 是能力扩展（一个模型干更多事），v11→v26 是极致回归（砍掉一切非必要的，让模型在最小设备上跑起来）**。

---

## 本章小结

这一章我们从零开始，把目标检测的核心知识体系走了一遍。来回顾一下学到的东西：

- **原理层面**：目标检测 = 分类 + 定位，Two-Stage（先搜后看）和 One-Stage（一眼报答案）各有取舍。核心挑战是尺度变化、遮挡、小目标和速度精度的平衡
- **评价层面**：从 TP/FP/FN 的基础概念出发，通过班级分类例子和街景检测例子建立直觉。Precision（不乱报）和 Recall（不漏检）互为制约，mAP 以 IoU 为基础，通过 5 步 P-R 曲线积分同时衡量误检、漏检和定位精度。COCO 的 AP@[.5:.95] 是目前最通用的标准
- **组件层面**：Anchor 提供了一个"位置先验"（但正在被 Anchor-Free 取代），NMS 用来消除重复检测（正在被端到端机制替代），数据增强通过随机变换扩充训练集，Label Smoothing 防止模型对标签过度自信
- **损失函数层面**：从 Smooth L1 到 CIoU 的五个版本，每一代都在损失函数中编码了更多的几何信息——重叠面积 → 最小闭包 → 中心距离 → 宽高比。CIoU 是目前的标配，DFL 通过把定位重构为概率分布问题，为 Anchor-Free 时代提供了关键补充
- **两阶段方法**：Faster R-CNN 的 RPN 是核心创新——用网络学会"看哪里"，替代了外部算法。ROI Align 用双线性插值消除了 ROI Pooling 的量化误差
- **单阶段方法（YOLO 系列）**：从 v1 的 7×7 网格检测到 v26 的极简边缘部署，YOLO 的演进是一部活生生的技术迭代史。关键里程碑：v5 定义了工程化标准（模块化 + 模型族 + 工具链），v8 完成了架构革命（C2f + 解耦头 + Anchor-Free），v11 实现了多任务统一（一个模型覆盖检测/分割/姿态/OBB），v26 回归极简（砍掉 DFL/NMS，为边缘设备做减法）
- **版本对比主线**：v5→v8 架构革命，v8→v11 能力扩展，v11→v26 极简回归。三次迭代恰好代表了三种不同的工程追求
