# deep-learning-handbook

AI 全领域教科书式知识库，Markdown 源文件为核心，mdBook 渲染。目标读者：AI/ML 初学者 + 有经验的开发者。

## 项目结构

```
parts/
├── part-1-basics/        # NumPy, PyTorch, CNN, Transformer
├── part-2-cv/            # OpenCV, 目标检测, 分割, 姿态, 追踪, OCR, ViT
├── part-3-nlp-llm/       # NLP基础, HuggingFace, 预训练, LLM, Prompt, RAG, Agent, 微调
├── part-4-multimodal/    # 多模态理解, Qwen3-VL, 生成, 跨模态对齐
└── part-5-interview/     # AI 面试题
code/                     # 配套可运行代码
assets/images/            # 图片资源
```

## 写作规范

- 语言：纯中文，术语保留英文原词（如 Bounding Box、Anchor）
- 风格：**图文并茂、理论讲解、内容充分展开后总结**
  - 使用 Mermaid 图表穿插在文字之间（架构图、流程图、对比图）
  - 像老师讲课一样一步步拆解理论，确保零基础能读懂
  - 每个概念讲透，关键架构逐层拆解，避免一句话总结
  - 避免纯表格堆砌、避免 bullet list 罗列，用段落式讲解 + 图表辅助
- 公式：LaTeX，行内 `$...$`，独立行 `$$...$$`，**公式内不得出现中文字符**（Typora 兼容性）
- 代码块：标注语言 ` ```python ` 等
- 关键直觉用 `>` 块引用给出
- 每章 frontmatter：title, part, chapter, status, updated
- 每章末尾 `## 本章小结`，不要 `## 参考` 节

## Notion 工作流

1. 用户在 Notion 撰写初稿 → 提供页面 URL
2. 通过 Notion MCP server 读取页面内容
3. 基于自身知识扩充改写，不可直接搬运原文
4. 写入对应 `parts/` 章节文件

## 命令

- `mdbook serve` — 本地预览
