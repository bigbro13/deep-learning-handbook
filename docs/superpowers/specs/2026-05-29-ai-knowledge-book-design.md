# AI 知识库书籍 — 设计文档

## 概述

一部覆盖全 AI 领域的教科书式知识库，以 Markdown 源文件为核心，面向初学者和有经验的开发者。

## 核心决策

| 维度 | 选择 |
|---|---|
| 目标读者 | AI/ML 初学者 + 有经验的开发者 |
| 内容范围 | 全 AI 领域（ML/DL/CV/NLP/LLM/多模态/RL/工程/安全） |
| 交付形式 | Markdown 源文件，格式无关，后期可渲染为任意格式 |
| 书籍风格 | 教科书风格，严谨章节结构，概念→原理→公式 |
| 数学深度 | 公式 + 直觉解释，读者理解原理但不需要手推 |
| 语言 | 纯中文，术语保留英文原词 |
| 组织结构 | 混合结构：基础篇 → 各领域独立展开，均衡分层 |
| 架构模式 | 均衡分层：基础篇覆盖所有共用知识（5 章），各领域篇聚焦独有内容 |

## 全书结构（5 部分 · 24 章）

### Part 1 — 基础篇（工具 & 架构驱动）

| 章 | 主题 |
|---|---|
| Ch1 | NumPy 科学计算 |
| Ch2 | PyTorch 深度学习框架 |
| Ch3 | CNN 卷积神经网络 |
| Ch4 | Transformer 架构 |

### Part 2 — 图像处理（任务驱动）

| 章 | 主题 |
|---|---|
| Ch5 | OpenCV 基础与图像处理 |
| Ch6 | 目标检测 |
| Ch7 | 图像分割 |
| Ch8 | 姿态估计 |
| Ch9 | 目标追踪 |
| Ch10 | OCR 光学字符识别 |
| Ch11 | ViT 与视觉 Transformer 应用 |

### Part 3 — 大模型 & NLP（任务 & 工具驱动）

| 章 | 主题 |
|---|---|
| Ch12 | NLP 基础与经典任务 |
| Ch13 | HuggingFace 生态 |
| Ch14 | 预训练语言模型 |
| Ch15 | 大语言模型原理 |
| Ch16 | Prompt Engineering |
| Ch17 | RAG 检索增强生成 |
| Ch18 | Agent 智能体 |
| Ch19 | 微调与对齐 |

### Part 4 — 多模态

| 章 | 主题 |
|---|---|
| Ch20 | 多模态理解（CLIP、BLIP、LLaVA、VQA） |
| Ch21 | Qwen3-VL 系列解读 |
| Ch22 | 多模态生成（Stable Diffusion、DALL-E、Sora） |
| Ch23 | 跨模态对齐与融合（ImageBind、统一表示空间） |

### Part 5 — 实战与面试

| 章 | 主题 |
|---|---|
| Ch24 | AI 面试题精解 |

> 章节可根据后续需要增删，此结构为初始基线。

## 文件目录结构

```
ai-book/
├── README.md
├── book.toml                     # 元数据
├── parts/
│   ├── part-1-basics/
│   ├── part-2-cv/
│   ├── part-3-nlp-llm/
│   ├── part-4-multimodal/
│   └── part-5-interview/
├── assets/
│   └── images/
└── code/
    ├── part-1/
    ├── part-2/
    ├── part-3/
    ├── part-4/
    └── part-5/
```

- 每章一个 `.md` 文件，按 Part 分目录
- 图片统一放 `assets/images/`
- 配套可运行代码放 `code/`，独立于正文

## 写作规范

### 文件格式

- 编码：UTF-8，LF 换行
- 每章文件以 YAML frontmatter 开头（title, part, chapter, status, updated）
- 标题层级：`#` 章 → `##` 节 → `###` 小节，最多 4 级

### 内容约定

- **公式**：LaTeX，行内 `$...$`，独立行 `$$...$$`，复杂推导编号
- **代码块**：标注语言，用 ` ```python ` 等
- **插图**：`![描述](path)` + 斜体图注
- **关键直觉**：重要概念用 `>` 块引用给出直觉解释
- **参考文献**：`[标题](url)`，每章末尾有 `## 参考`
- **每章末尾**：`## 本章小结`

### 单章模板

```markdown
---
title: "章节标题"
part: 1
chapter: 1
status: draft
updated: 2026-05-29
---

# Ch1 — 章节标题

## 1.1 第一小节

正文...

> **关键直觉**：一句话解释核心思想。

​```python
# 代码示例
​```

![插图描述](./assets/images/xxx.png)
*图 1.1：说明文字*

## 参考

- [论文/文章标题](url)

## 本章小结

简要回顾本章要点。
```

## Notion → Markdown 工作流

1. 用户在 Notion 撰写内容
2. 提供 Notion 页面 URL
3. 通过 Notion MCP server 读取页面 block 内容
4. 自动映射为 Markdown 格式（标题、代码块、表格、图片等）
5. 写入对应 `parts/` 章节文件

需配置 Notion API token 和 MCP server（如 `notion-mcp-server`）。

## 状态说明

- `draft`：初稿写作中
- `review`：技术审校中
- `done`：定稿

## 后续扩展

- 章节可根据需要增删
- 后期可接入渲染工具（mdBook、GitBook 等）生成网站或 PDF
- 代码示例需确保可运行并通过 CI 验证
