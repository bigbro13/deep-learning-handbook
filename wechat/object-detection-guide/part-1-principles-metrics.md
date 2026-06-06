# 目标检测完全指南（上）：原理、指标与核心组件

> 本文是「目标检测完全指南」系列的上篇，涵盖目标检测的基本原理、评价指标（IoU / mAP）、核心组件（NMS / 数据增强 / Label Smoothing）以及 Bounding Box 回归损失函数的完整演进。
>
> 中篇：[Faster R-CNN 原理详解]()
> 下篇：[YOLO 系列全解析]()

---

# Ch6 — 目标检测

## 6.1 目标检测原理概述

### 6.1.1 目标检测要解决什么问题？

在一张日常生活中的图片——比如街景照片。你一眼扫过去，能看到车、行人、交通灯、路牌……你的大脑不仅识别出了这些物体是什么，还精确地知道它们在画面中的哪个位置。

让计算机做到同样的事，就是目标检测要做的事情。用更正式的语言来说：**目标检测（Object Detection）同时解决两个问题——分类（这是什么？）和定位（它在哪里？）**。

具体来说，把一张图像 $I$ 输入模型，模型需要输出一组检测结果，每组结果包含三条信息：

- **类别标签 $c_i$**：这个目标是什么（车？人？狗？）
- **边界框 $b_i$**：目标在图像中的精确位置，通常用 $(x, y, w, h)$ 表示中心坐标和宽高，或者用 $(x_1, y_1, x_2, y_2)$ 表示左上角和右下角坐标
- **置信度 $s_i$**：模型对这个检测结果有多"确信"，一个 0 到 1 之间的分数

### 6.1.2 目标检测和图像分类有什么不同？

很多同学可能先接触过图像分类。图像分类只需要回答"这张图里是什么"这一个问题，输出的结果是整张图的一个类别标签。比如输入一张猫的照片，输出"猫"。

目标检测要难得多——一张图里可能有多个目标，你需要：

1. **找出所有目标的位置**（不能漏掉任何一个）
2. **对每个目标正确分类**（不能把狗认成猫）
3. **同一个目标不能重复检测**（每个目标只应该有一个框）

拿图像分类和目标检测做个对比，差距一目了然：

| | 图像分类 | 目标检测 |
|---|---|---|
| 输入 | 一张图像 | 一张图像 |
| 输出 | 一个类别标签 | 多个 (类别 + 位置 + 置信度) |
| 一张图检测几个目标 | 1 个 | 不限 |
| 知道目标在哪吗？ | 不知道 | 知道（精确到像素的边界框） |
| 典型场景 | 猫狗识别 | 自动驾驶、安防监控 |

### 6.1.3 两种主流检测范式

既然目标检测要做这么多事，那从架构上怎么设计呢？深度学习时代发展出了两大流派。

**Two-Stage（两阶段检测）**

两阶段方法的思想很直观——与其在一整张图上大海捞针，不如先找出"可能有东西"的区域，再对这些区域仔细判断。

![two-stage-pipeline](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch06/two-stage-pipeline.png)

- **第一阶段**：生成大约几百到几千个"候选框"，把图像中最有可能包含目标的区域框出来。这就像你扫一眼照片，先注意到"那边有个东西"
- **第二阶段**：对每个候选框，仔细做两件事——判断它属于哪个类别（分类），以及修正它的边界让它更精确地包围目标（回归）

代表方法：Faster R-CNN 系列。它的优势是精度高——因为第二阶段筛掉了大量不是目标的背景，正负样本比例可控。代价是速度慢一些，毕竟分两步走。

**One-Stage（单阶段检测）**

单阶段方法更"直接"——不先生成候选框，而是一步到位，从图像像素直接回归出所有目标的类别和位置。

![one-stage-pipeline](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch06/one-stage-pipeline.png)

- 没有中间的候选区域生成步骤
- 输入图像 → 卷积网络 → 直接输出 N 个检测框

代表方法：YOLO 系列。优势是速度快（适合实时场景），但早期版本在精度上略逊于两阶段方法。不过后面的发展让两者的边界越来越模糊——最新的单阶段方法在精度上完全不输两阶段。



> **核心理解**：Two-Stage 是"先扫一眼找到可能位置，再仔细看"；One-Stage 是"一眼看完直接报答案"。选择哪个取决于你的场景——需要极致精度选 Two-Stage，需要实时速度选 One-Stage。但近年来这俩的界限越来越模糊了。

### 6.1.4 这个领域到底难在哪里？

目标检测发展到今天，看似模型已经很强了，但这些核心挑战仍然存在：

| 核心挑战         | 具体表现/难点描述                                            |
| :--------------- | :----------------------------------------------------------- |
| **尺度变化**     | • 同一类目标大小差异巨大 <br />• 小目标可能仅占几个像素 <br />• 大目标可能占满整个画面 |
| **遮挡问题**     | • 目标之间互相遮挡 <br />• 部分关键特征不可见 <br />• 模型需要“脑补”缺失信息以完成识别 |
| **小目标检测**   | • 有效信息量太少<br /> • 特征难以有效提取<br />• 被认为是目前最困难的方向之一 |
| **密集场景**     | • 目标扎堆出现 <br />• 边界框（Bounding Box）高度重叠<br /> • 容易导致漏检或重复检测 |
| **速度精度平衡** | • 实时应用如自动驾驶要求 30+ FPS <br />• 高精度模型通常计算量大、速度慢 <br />• 需要在速度与精度之间寻找最优解 |
| **背景干扰**     | • 背景包含复杂纹理和颜色<br /> • 相似物体容易产生误检<br /> • 需要模型具备更强的语义理解能力 |

这些挑战也正好解释了为什么目标检测的方法一直在迭代——**每一个新的网络结构、每一个新的损失函数，几乎都是为了解决上面某一个或某几个具体问题而诞生的**。理解了挑战，才能理解后面的各种设计选择。

---

## 6.2 核心评价指标

在深入学习具体的检测方法之前，得先搞清楚"怎么评价一个模型做得好不好"。目标检测的评价比分类复杂得多——你不仅要看分类对不对，还要看定位准不准。

### 6.2.1 IoU —— 模型和正确答案有多"重叠"？

IoU（Intersection over Union，交并比）是目标检测中最基础、最核心的度量。它衡量的是**两个框的重叠程度**——预测框和真实框"撞"了多少。

$$\text{IoU} = \frac{|A \cap B|}{|A \cup B|}$$

公式含义：IoU = 两个框的交集面积 ÷ 两个框的并集面积。用图来理解：

![iou](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch06/iou.png)

几种典型情况：

- IoU = 1：两个框完全重叠，预测完美无瑕
- IoU ≈ 0.7~0.9：预测很不错，框的位置比较精准
- IoU ≈ 0.5：勉强可以接受的检测，但位置偏差明显
- IoU = 0：预测框和真实框完全不沾边

在实际使用中，我们需要设定一个 IoU 阈值（通常取 0.5）来判断一个预测算不算"正确"。如果预测框与某个真实框的 IoU ≥ 0.5 **且**类别预测正确，那这个预测就算命中目标。

> IoU 看起来简单，但它不仅用来评价结果好坏，还被直接用作损失函数的设计基础（后面 6.4 节会详细讲）。**IoU 唯一的"盲区"是它只看重叠面积，不关心框的中心在哪、形状匹不匹配**。这直接催生了后来的 GIoU、DIoU、CIoU 等改进版本。

### 6.2.2  Precision 与 Recall—— 你的模型有多少"对的"检测？

光说"这个框对不对"还不够，我们还需要从全局角度评价模型的检测质量。这里引入四个基本概念：

**TP、FP、FN、TN 到底是什么？**

先来拆解这四个字母的含义：

- **T / F**（True / False）：判断对了还是判断错了
- **P / N**（Positive / Negative）：模型给出的预测是"有目标"还是"没目标"

两两组合就得到了四个概念：

| 概念 | 全称 | 通俗理解 |
|---|---|---|
| TP (True Positive) | 真阳性 | 判断对了，模型说有目标，实际确实有目标——**检测对了** |
| FP (False Positive) | 假阳性 | 判断错了，模型说有目标，但实际没有——**误检（虚警）** |
| FN (False Negative) | 假阴性 | 判断错了，模型说没目标，但实际有——**漏检** |
| TN (True Negative) | 真阴性 | 判断对了，模型说没目标，实际也确实没有——对了但不太重要 |

这四个概念初看有点绕，我们用两个例子来彻底搞懂——先从一个简单的分类场景开始，再回到目标检测。

**例子一：班级里的"找女生"（分类场景）**

这个例子帮你建立最直观的理解。已知：班级共 100 人，其中男生 80 人，女生 20 人。目标是把所有女生找出来（女生 = 正类）。

现在某个"模型"选出了 50 个人，声称这些都是女生。逐一核实后发现：

- 20 个真正的女生被成功找了出来 → **TP = 20**
- 30 个男生被错误地当成了女生 → **FP = 30**
- 0 个女生被漏掉（所有女生都在选出的 50 人中）→ **FN = 0**
- 50 个男生没有被选出来，他们也确实是男生 → **TN = 50**

> 注意：TN 在这个例子里是 50，但在公式中一般不出现。因为"正确的拒绝"太多了，统计它没有意义。

有了 TP、FP、FN，就可以计算两个核心指标了：

$$\text{Precision} = \frac{TP}{TP + FP} = \frac{20}{20 + 30} = 0.4$$

Precision 回答的问题是：**模型说"是女生"的人里，真正是女生的比例有多大？** 它衡量的是模型会不会"乱报"。

$$\text{Recall} = \frac{TP}{TP + FN} = \frac{20}{20 + 0} = 1.0$$

Recall 回答的问题是：**所有真正是女生的人里，模型成功找出了多少？** 它衡量的是模型会不会"漏掉"。

在这个例子里，Recall 完美（一个没漏），但 Precision 只有 0.4（选出来的 50 人里混了 30 个男生）。模型太"激进"了——宁可错杀也不放过。

**例子二：街景中的"找人"（检测场景）**

现在回到目标检测的语境。一张图中实际有 10 个人。模型检测出了 8 个框。逐一核对：

- 6 个框正确框住了人（TP = 6）
- 2 个框框在了错误的物体上，比如把垃圾桶认成了人（FP = 2）
- 剩下的 4 个人模型根本没检测到（FN = 4，因为 10 - 6 = 4）

套用同样的公式：

$$\text{Precision} = \frac{6}{6 + 2} = 0.75,\quad \text{Recall} = \frac{6}{6 + 4} = 0.6$$

> **从分类到检测，TP/FP/FN 的含义没变，但场景变了**。分类场景中每个样本是一个独立个体（学生），检测场景中每个样本是一个预测框。核心逻辑完全一样——"判断对了没有" × "模型说了什么"。



> **Precision 和 Recall 是天生的冤家**——你把检测门槛设高（只出置信度最高的框），Precision 很高但 Recall 很低（漏了一堆目标）；你把门槛设低（什么框都输出），Recall 很高但 Precision 很差（误报满天飞）。一个好的检测器就是要在两者之间找到最优平衡。

### 6.2.3 从 Precision 和 Recall 到 AP

但光说 Precision 高或者 Recall 高还不够——你调整一下置信度阈值，Precision 和 Recall 都会跟着变。我们想要一个**综合了所有可能阈值情况**的单一指标。这就是 AP（Average Precision）的来源。

AP 衡量的是某一个类别在**所有可能的 Recall 水平下**的整体 Precision 表现。下面用五个步骤把计算过程彻底拆解清楚。

**步骤 1：收集某一类别的所有预测框**

对某个类别（比如"人"），把模型输出的所有预测框收集起来。每个框带两个信息：它的置信度（模型觉得它有多靠谱），以及它和哪个真实框匹配上了（IoU 是否超过阈值、类别是否一致——是不是 TP）。

**步骤 2：按置信度降序排序**

把所有预测框按置信度从高到低排好。置信度最高的排第一个，最低的排最后。为什么从高往低排？因为实际使用时，我们会按照置信度来决定"展示哪些结果给用户"——展示越多的框（把阈值调低），Recall 就越高，但 Precision 会下降。

**步骤 3：逐个累加，计算 Precision 和 Recall**

从置信度最高的框开始，一个一个往结果集中加。每加一个框，更新当前的累计 TP 和 FP，算出当前的 Precision 和 Recall。这样得到一系列 $(Recall, Precision)$ 点——每个点对应"把置信度阈值设到这里为止"时的表现。

![pr-curve-computation](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch06/pr-curve-computation.png)

**步骤 4：构建 P-R 曲线并插值平滑**

![P-R](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch06/P-R.png)

将上面得到的点绘制成 P-R 曲线。随着 Recall 增加，Precision 总体呈下降趋势——因为越来越多的低置信度框被加入，其中错误的可能性也越来越大。曲线可能不是单调下降的（会有波动），所以通常会对 Precision 做**插值平滑**——在每个 Recall 水平上，取它"右边所有 Precision 的最大值"作为平滑后的值：

$$p_{\text{interp}}(r) = \max_{\tilde{r} \geq r} p(\tilde{r})$$

**步骤 5：计算 AP**

AP 就是平滑后的 P-R 曲线下方的面积。关于怎么算这个"面积"，有两种主流方法：

- **VOC 2007 方式**：在 Recall 0 到 1 之间均匀取 11 个点（0, 0.1, 0.2, ..., 1.0），在每个点上取插值后的 Precision 值，11 个值取平均：

  $$\text{AP} = \frac{1}{11} \sum_{r \in \{0, 0.1, ..., 1.0\}} p_{\text{interp}}(r)$$

- **VOC 2010 及 COCO 方式**：对整个 Recall 范围上的 P-R 曲线做积分（即曲线下面积），比 11 点法更精准：

  $$\text{AP} = \int_0^1 p_{\text{interp}}(r) \, dr$$

> AP 的强大之处在于：**它用一个数字同时衡量了漏检（低 Recall 时 Precision 也低）和误检（高 Recall 时 Precision 掉得多）**，而且不受置信度阈值选择的干扰。

### 6.2.4 mAP —— 多类别下的终极指标

实际任务中通常不只检测一个类别（比如 COCO 数据集有 80 个类）。**mAP（mean Average Precision）就是把所有类的 AP 取个平均**：

$$\text{mAP} = \frac{1}{C} \sum_{c=1}^{C} \text{AP}_c$$

它是目标检测领域最核心、最通用的评价指标。

**COCO 评价体系**是目前业界使用最广的标准，它考核得很细致：

| 指标名 | 含义 |
|---|---|
| AP@[0.5:0.95] | IoU 阈值从 0.5 到 0.95，步长 0.05，共 10 个阈值下的 mAP 取平均。这是**最核心**的指标 |
| AP@0.50 | 只取 IoU ≥ 0.5 就算对（VOC 传统标准，相对宽松） |
| AP@0.75 | IoU ≥ 0.75 才算对（对定位精度要求更高） |

> **为什么 mAP 能成为行业标准？** 因为它一个指标同时衡量了**分类对不对、定位准不准、有没有漏检、会不会误检、在不同类别上表现是否均衡**。五个维度，一个数字，简洁有力。

### 6.2.5 别忘了速度

在实际落地中，光精度高是不够的。自动驾驶需要实时处理视频流（≥ 30 FPS），移动端应用需要模型小到能跑在手机上。所以通常还会关注：

- **FPS（Frames Per Second）**：每秒能处理多少张图，衡量推理速度
- **FLOPs（浮点运算次数）**：模型跑一次要多少计算量，和硬件要求直接相关
- **参数量（Params）**：模型有多少参数，影响内存占用和部署可行性

---

## 6.3 核心组件与技术

了解了"怎么评价"之后，我们来看看目标检测系统内部的几个关键组件。这些组件就像乐高积木，不同模型用不同的方式组合它们。理解了这些零件，再看各种模型的设计就会觉得顺理成章。

### 6.3.1 Anchor —— 给模型一个"先验"

想象一下，你要教一个初学者怎么画边界框。你不会让他从一张白纸开始随便画，而是会给他一些参考——"大概在这个位置，大概是这个比例"。

**Anchor（锚框）就是给网络提供的这种"参考"**。它在特征图的每个位置上预先放置一组固定大小和比例的框，网络不需要从零开始预测坐标，只需要在这些锚框的基础上做"微调"。

**为什么需要锚框？** 因为直接预测绝对坐标（比如"这个框的中心在 (342, 567)，宽 203，高 411"）是非常困难的——数值范围太大，网络不好学习。但如果有一个参考锚框，网络只需要学习"在锚框基础上向左偏 10%，高度增加 20%"，这种相对偏移就稳定多了。

以 Faster R-CNN 为例，它在每个特征图位置上生成 9 个锚框：3 种尺度（128²、256²、512²）× 3 种宽高比（1:1、1:2、2:1）。

YOLOv2 更进一步——它觉得人工设计的锚框尺寸未必最适合当前数据集，于是用 **K-means 聚类**在训练集的所有标注框上自动找出最典型的几种尺寸。聚类的距离度量也很聪明，用的是 $d = 1 - \text{IoU}$——两个框越不重叠，距离越远。语义上就是：我们关心的是"框长得像不像"，而不是"框的坐标差多少"。

**那锚框有什么缺点？**

说实话锚框挺"麻烦"的：
- 你需要精心设计锚框的尺寸和比例（超参数调优地狱）
- 每个位置生成一堆锚框，大部分都是背景（正负样本极度不平衡）
- 对于特别大或特别小的目标，预设的锚框可能覆盖不到

这些缺点正好推动了 **Anchor-Free 方法**的兴起（我们会在 6.6 和 6.7 节详细讨论）。核心思想就是——扔掉锚框，直接让网络自己去想怎么定位。

### 6.3.2 NMS —— 消除"一物多框"

检测器有个毛病——对同一个目标，它经常输出好几个高度重叠的预测框（每个都认为自己是最准的）。这其实不难理解：特征图上相邻的几个位置可能都"看到"了同一个物体。

**NMS（Non-Maximum Suppression，非极大值抑制）的任务就是：从这些重复的框中，只保留最好的那个。**

算法步骤其实很简单：

1. 把当前所有预测框按置信度从高到低排好序
2. 取出最高置信度的框（一定保留它，因为它最有把握），把它放进结果列表
3. 计算剩下所有框与这个"冠军框"的 IoU。如果 IoU > 阈值（常见 0.5），说明它们检测到的是同一个目标——删掉
4. 对剩余的框，重复步骤 2-3，直到没有框剩下
5. 结果列表里就是最终保留的所有框

用代码实现起来非常直观：

```python
import numpy as np


def nms(boxes, scores, iou_threshold=0.5):
    """
    NMS（非极大值抑制）

    参数:
        boxes:   ndarray, shape (N, 4), 预测框坐标 (x1, y1, x2, y2)
        scores:  ndarray, shape (N,),   每个框的置信度
        iou_threshold: float, IoU 阈值, 超过此值视为"同一目标"

    返回:
        keep: list[int], 保留下来的框的索引
    """
    # 1. 计算每个框的面积（用于 IoU 计算）
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    # 2. 按置信度从高到低排序
    order = np.argsort(scores)[::-1]

    keep = []
    while order.size > 0:
        # 3. 取出当前最高置信度的框，直接保留
        best_idx = order[0]
        keep.append(best_idx)

        if order.size == 1:
            break

        # 4. 计算"冠军框"与其余框的 IoU
        xx1 = np.maximum(x1[best_idx], x1[order[1:]])
        yy1 = np.maximum(y1[best_idx], y1[order[1:]])
        xx2 = np.minimum(x2[best_idx], x2[order[1:]])
        yy2 = np.minimum(y2[best_idx], y2[order[1:]])

        # 交集宽高（不相交时为负，clamp 到 0）
        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        inter = inter_w * inter_h

        # IoU = 交集 / 并集
        union = areas[best_idx] + areas[order[1:]] - inter
        iou = inter / union

        # 5. IoU > 阈值的框视为检测到同一目标，删掉
        remain = np.where(iou <= iou_threshold)[0]
        order = order[remain + 1]  # +1 是因为 order[1:] 的索引偏移

    return keep
```

关键细节解释：

- `np.argsort(scores)[::-1]` 按置信度**降序**排列，保证每次先处理最有把握的框
- `order[remain + 1]` 里的 `+1` 是因为 `order[1:]` 比原 `order` 少了第一个元素（冠军框已取出），索引需要补偿
- IoU 计算中的 `np.maximum(0.0, ...)` 确保不相交的框不会出现负面积

**Soft-NMS：当两个目标真的离得很近时**

标准 NMS 有一个致命问题：如果两个同类目标挨得很近（比如马路上一排紧挨着的车），其中一个的预测框可能和另一个目标的框 IoU 太高，被误删了。

Soft-NMS 的解决方案很优雅——不直接删，而是**降低这些框的置信度**：

$$s_i = s_i \cdot e^{-\frac{\text{IoU}(M, b_i)^2}{\sigma}}$$

如果一个框和冠军框 IoU 越高，它的置信度就被降得越厉害。但不会直接归零，所以如果它其实是附近另一个真正的目标，它仍然有机会在后面的迭代中胜出。

### 6.3.3 数据增强 —— 用"变体"扩充训练集

一张图，稍微改一改，对人类来说还是同一张图，但对模型来说就是一个全新的训练样本。**数据增强就是利用这一点，通过对训练图像做各种随机变换，凭空"造"出更多训练数据。**

为什么要这么做？因为真实场景中光照、角度、尺度千变万化，而标注数据永远不够多。数据增强让模型见过更多"变体"，泛化能力自然更好。

**常用的数据增强方法**：

![data-augmentation](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch06/data-augmentation.png)

特别要提一下 **Mosaic 增强**（YOLOv4 引入）——它把 4 张不同的训练图片随机缩放后拼成一张大图。这样做有几个好处：拼出来的图片中包含更多的小目标（因为被缩小了），上下文也更丰富（一张图里有 4 种场景）。这是 YOLO 系列最经典的数据增强手段之一。

以下参数在yolo模型训练中可以调整这些设置以满足数据集和当前任务的具体要求。尝试不同的值有助于找到能带来最佳模型性能的最优增强策略。

| 参数                                                         | 描述                                                         | 图例                                                         |
| :----------------------------------------------------------- | :----------------------------------------------------------- | ------------------------------------------------------------ |
| [`hsv_h`](https://docs.ultralytics.com/zh/guides/yolo-data-augmentation#hue-adjustment-hsv_h) | 按色相环的一小部分调整图像的色相，引入颜色变异性。有助于模型在不同的光照条件下进行泛化。 | ![hsv_h](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch06/hsv_h.png)                 |
| [`hsv_s`](https://docs.ultralytics.com/zh/guides/yolo-data-augmentation#saturation-adjustment-hsv_s) | 按一小部分改变图像的饱和度，从而影响颜色的强度。有助于模拟不同的环境条件。 | ![image-20260605174158963](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch06/hsv_s.png) |
| [`hsv_v`](https://docs.ultralytics.com/zh/guides/yolo-data-augmentation#brightness-adjustment-hsv_v) | 按一小部分修改图像的值（亮度），帮助模型在各种光照条件下良好运行。 | ![image-20260605174220113](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch06/hsv_v.png) |
| [`degrees`](https://docs.ultralytics.com/zh/guides/yolo-data-augmentation#rotation-degrees) | 在指定的角度范围内随机旋转图像，提高模型识别不同方向物体的能力。 | ![image-20260605174259202](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch06/degrees.png) |
| [`translate`](https://docs.ultralytics.com/zh/guides/yolo-data-augmentation#translation-translate) | 按图像大小的一小部分水平和垂直平移图像，有助于学习检测部分可见的物体。 | ![image-20260605174312685](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch06/translate.png) |
| [`scale`](https://docs.ultralytics.com/zh/guides/yolo-data-augmentation#scale-scale) | 通过增益因子缩放图像，模拟相机视角中距离不同的物体。         | ![image-20260605174327841](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch06/scale.png) |
| [`perspective`](https://docs.ultralytics.com/zh/guides/yolo-data-augmentation#perspective-perspective) | 对图像应用随机透视变换，增强模型理解 3D 空间中物体的能力。   | ![image-20260605174343746](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch06/perspective.png) |
| [`flipud`](https://docs.ultralytics.com/zh/guides/yolo-data-augmentation#flip-up-down-flipud) | 以指定的概率将图像上下翻转，在不影响物体特征的情况下增加数据变异性。 | ![image-20260606150217525](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch06/flipud.png) |
| [`fliplr`](https://docs.ultralytics.com/zh/guides/yolo-data-augmentation#flip-left-right-fliplr) | 以指定的概率将图像左右翻转，这对于学习对称物体和增加数据集多样性非常有用。 | ![image-20260606150245630](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch06/fliplr.png) |
| [`bgr`](https://docs.ultralytics.com/zh/guides/yolo-data-augmentation#bgr-channel-swap-bgr) | 以指定的概率将图像通道从 RGB 翻转为 BGR，这对于提高对通道顺序错误的稳健性非常有用。 | ![image-20260606150300278](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch06/bgr.png) |
| [`mosaic`](https://docs.ultralytics.com/zh/guides/yolo-data-augmentation#mosaic-mosaic) | 将四张训练图像合并为一张，模拟不同的场景构图和目标交互。对于复杂场景理解非常有效。 | ![image-20260606150316836](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch06/mosaic.png) |
| [`mixup`](https://docs.ultralytics.com/zh/guides/yolo-data-augmentation#mixup-mixup) | 混合两张图像及其标签，创建一个合成图像。通过引入标签噪声和视觉多变性，增强模型的泛化能力。 | ![image-20260606150328281](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch06/mixup.png) |
| [`cutmix`](https://docs.ultralytics.com/zh/guides/yolo-data-augmentation#cutmix-cutmix) | 结合两张图像的部分内容，在保持不同区域的同时创建部分混合。通过创建遮挡场景增强模型的鲁棒性。 | ![image-20260606150342977](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch06/cutmix.png) |

### 6.3.4 Label Smoothing —— 别让模型太"自信"

分类模型通常用 One-Hot 标签训练：正确的类别标为 1，其他全标为 0。比如"这是一只猫" → [0, 1, 0, 0, 0]。

这种硬标签有个潜台词：**模型被要求对正确答案百分百确信**。但现实中，有些图片本身就是模糊的、模棱两可的。强制模型输出极端概率会导致两个问题：一是过拟合（模型认为自己的判断永远是"绝无错误"的），二是泛化差（遇到没见过的东西时过于武断）。

**Label Smoothing 的做法是"软化"标签**：

$$\tilde{y}_i = (1 - \epsilon) \cdot y_i + \frac{\epsilon}{K}$$

其中 $\epsilon$ 是平滑因子（常见 0.1），$K$ 是类别数。比如原来的 One-Hot 标签 [0, 1, 0, 0, 0]（共 5 类，$\epsilon=0.1$）变成：[0.02, 0.92, 0.02, 0.02, 0.02]——正确类别只给 0.92 的确信度，其余每个类给 0.02 的"小概率"。

这样做让模型学到的不是一个"绝对的是非判断"，而是一个**更平滑、更宽容的概率分布**。结果就是模型对训练集不那么"偏执"，在测试集上表现更稳健。

不过 Label Smoothing 也有局限：它本质上是在给标签加均匀分布的随机噪声，并不能反映类别之间真实的关系（猫和狗确实比猫和汽车更相似，但 Label Smoothing 对所有非正确类一视同仁）。在知识蒸馏场景中，使用 Label Smoothing 训练的教师网络质量反而更差。

---

## 6.4 Bounding Box 回归损失函数

你可能已经注意到了——目标检测的很多设计难题都围绕一个核心问题：**怎么让预测框更精准地靠近真实框？** 这个"靠近"的度量方式，就体现在回归损失函数的设计上。

边界框回归损失函数是目标检测中演进脉络最清晰的部分，每一代的改进都直指前代的痛点。发展路径如下：

$$\text{Smooth L1} \;\rightarrow\; \text{IoU Loss} \;\rightarrow\; \text{GIoU} \;\rightarrow\; \text{DIoU} \;\rightarrow\; \text{CIoU}$$

下面我们一个一个来看，重点是理解**每个版本解决了什么问题，又留下了什么问题**。

### 6.4.1 Smooth L1 Loss —— 第一代回归损失

Faster R-CNN 用的是 Smooth L1 Loss，它在 L1 和 L2 之间取了个折中：

$$L_{\text{smoothL1}}(x) = \begin{cases} 0.5x^2 & \text{if } |x| < 1 \\ |x| - 0.5 & \text{otherwise} \end{cases}$$

解释一下这个分段设计：当预测值和真实值差距不大时（$|x| < 1$），用 L2（平方）损失，梯度随误差减小而减小，收敛精细又稳定；当差距很大时，切换到 L1（绝对值）损失，梯度恒定，不会被大误差带偏。这个设计在工程实践上很聪明。

**但它有一个根本性的问题**：Smooth L1 把边界框的四个坐标（$x, y, w, h$）当成四个独立的变量，分别算损失再加起来。可实际上这四个坐标不是独立的——$x$ 的误差对 IoU 的影响和 $y$ 的误差完全不同。结果是：两个预测框可能有同样的 Smooth L1 损失值，但实际的 IoU 天差地别。**损失函数的优化目标和最终的评价指标（IoU）脱节了**。

### 6.4.2 IoU Loss —— 直接优化"正主"

![image-20260605172631739](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch06/iou_loss.png)

那为什么不用 IoU 本身作为损失函数呢？IoU Loss 就是这么想的：

$$L_{\text{IoU}} = 1 - \text{IoU}(B_p, B_g) = 1 - \frac{|B_p \cap B_g|}{|B_p \cup B_g|}$$

优点明显：**优化什么就评价什么**，损失值直接反映了定位质量。

但 IoU Loss 有两个硬伤，一个比一个严重：

**硬伤一：预测框和真实框完全不相交时，怎么办？**

![image-20260605172656764](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch06/iou_loss_1.png)

此时 IoU = 0，$L_{\text{IoU}} = 1$，是一个常数。常数意味着梯度为零——模型不知道应该让预测框往哪个方向移动才能靠近真实框。在训练初期，预测框很可能和真实框完全不沾边，优化直接卡死。

**硬伤二：相同 IoU 值的预测框，位置可能完全不同。**

比如下面三种情况，每一对的 IoU 都一样，但你肯定能看出它们的对齐质量完全不同：

- 情况 A：预测框偏左，有些区域重叠
- 情况 B：预测框偏右，相同程度重叠
- 情况 C：预测框在目标框正中间偏上

IoU 值相同，IoU Loss 完全无法区分这三种情况。这对精确定位来说是一个很大的盲区。

### 6.4.3 GIoU Loss —— 用"最小闭包"解决不相交问题

GIoU（Generalized IoU）通过引入**最小闭包矩形**来弥补 IoU 的缺陷。什么是最小闭包矩形？就是同时包含预测框和真实框的最小矩形。

![image-20260605172930790](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch06/giou_loss.png)

$$L_{\text{GIoU}} = 1 - \text{IoU} + \frac{|C - (B_p \cup B_g)|}{|C|}$$

其中 $C$ 是那个最小闭包矩形。公式的后半部分可以理解为：**最小闭包中不属于两个框的部分越大，惩罚就越大**。

你现在看出 GIoU 的聪明之处了吗？当预测框和真实框不相交时，虽然 IoU = 0，但两个框离得越远，最小闭包 $C$ 就越大，惩罚项就越大——**不相交情况下也有了可优化的梯度**。模型知道应该让框往哪个方向靠拢了。

但 GIoU 仍然有一个问题没有解决：**当预测框完全被真实框包含时**（或者反过来），此时 $C = B_p \cup B_g = B_g$​，惩罚项退化为 0，GIoU 直接退化成 IoU，无法区分"框在中心"和"框在角落"。

![image-20260605173032011](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch06/giou_loss_1.png)

### 6.4.4 DIoU Loss —— 让中心点直接靠近

![image-20260605173057842](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch06/diou_loss.png)

DIoU（Distance IoU）的改进直击要害——**既然中心点距离很重要，那就直接在损失里加上它**：

$$L_{\text{DIoU}} = 1 - \text{IoU} + \frac{\rho^2(b_p, b_g)}{c^2}$$

其中 $\rho(b_p, b_g)$ 是两框中心点的欧氏距离，$c$ 是最小闭包矩形的对角线长度。

这个设计带来了三个好处：
- **收敛更快**：中心距离直接拉到损失里，优化路径更直接

- **完全包含也不怕**：即使预测框在真实框内部，中心点距离仍然能衡量"偏离了多少"，GIoU 退化的问题不复存在

- **优化行为更合理**：训练时预测框会自然地朝真实框中心靠拢

  

- ![image-20260605173331100](https://raw.githubusercontent.com/bigbro13/deep-learning-handbook/master/assets/images/ch06/diou_loss_1.png)

### 6.4.5 CIoU Loss —— 把宽高比也考虑进来

DIoU 还差最后一块拼图——**两个框可能中心重叠，但形状不同**（一个正方形、一个长条形）。它们应该有不同分数，但 DIoU 无法区分。

CIoU（Complete IoU）把宽高比也加进了损失函数，做到了当时的最完整建模：

$$L_{\text{CIoU}} = 1 - \text{IoU} + \frac{\rho^2(b_p, b_g)}{c^2} + \alpha v$$

其中宽高比惩罚项 $v$ 和平衡权重 $\alpha$ 为：

$$v = \frac{4}{\pi^2}\left(\arctan\frac{w_g}{h_g} - \arctan\frac{w_p}{h_p}\right)^2, \quad \alpha = \frac{v}{(1 - \text{IoU}) + v}$$

$v$ 用 $\arctan$ 来比较宽高比（因为 $\arctan$ 在宽高差异很大时也不会爆炸），$\alpha$ 是一个自适应权重——当 IoU 较小时，让宽高比的影响弱一些，优先把框的位置和大小对齐。

### 6.4.6 综合对比

把五个损失函数放在一起看，演进逻辑一目了然：

每个版本增加的维度：

| 损失函数 | 重叠面积 | 中心距离 | 宽高比 | 不相交可优化 |
|---|---|---|---|---|
| Smooth L1 | ✗ | ✗ | ✗ | ✓ |
| IoU Loss | ✓ | ✗ | ✗ | ✗ |
| GIoU Loss | ✓ | ✗ (间接) | ✗ | ✓ |
| DIoU Loss | ✓ | ✓ | ✗ | ✓ |
| CIoU Loss | ✓ | ✓ | ✓ | ✓ |

> **演进规律**：每一步都在损失中编码一个额外的几何维度——重叠面积 → 最小闭包 → 中心距离 → 宽高比。最终 CIoU 在这四个维度上做到了全覆盖，也成了今天几乎每款 YOLO 版本的标准配置。

---
