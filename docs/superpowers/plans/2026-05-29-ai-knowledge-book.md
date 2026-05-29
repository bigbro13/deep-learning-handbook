# AI 知识库书籍 — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 搭建 AI 知识库书籍的项目骨架，包含目录结构、元数据、24 章模板和 Notion MCP 集成。

**Architecture:** 纯 Markdown 源文件项目，`parts/` 下按 5 个 Part 分目录管理 24 个章节文件，`assets/` 统一管理图片，`code/` 存放配套代码。通过 Notion MCP server 实现 Notion → Markdown 的内容迁移。

**Tech Stack:** Markdown (源文件), YAML frontmatter (元数据), Notion MCP (内容获取), 可选 mdBook 渲染

---

### Task 1: 创建项目骨架

**Files:**
- Create: `README.md`
- Create: `book.toml`
- Create: `parts/part-1-basics/`, `parts/part-2-cv/`, `parts/part-3-nlp-llm/`, `parts/part-4-multimodal/`, `parts/part-5-interview/`
- Create: `assets/images/`
- Create: `code/part-1/`, `code/part-2/`, `code/part-3/`, `code/part-4/`, `code/part-5/`

- [ ] **Step 1: 创建目录结构**

```bash
mkdir -p parts/part-1-basics \
         parts/part-2-cv \
         parts/part-3-nlp-llm \
         parts/part-4-multimodal \
         parts/part-5-interview \
         assets/images \
         code/part-1 code/part-2 code/part-3 code/part-4 code/part-5
```

- [ ] **Step 2: 创建 book.toml**

```toml
[book]
title = "AI 知识全书"
authors = ["Your Name"]
version = "0.1.0"
language = "zh-CN"

[structure]
parts = ["basics", "cv", "nlp-llm", "multimodal", "interview"]

[rendering]
math = true
highlight = true

[status]
draft = "part-1-basics, part-2-cv, part-3-nlp-llm, part-4-multimodal, part-5-interview"
review = ""
done = ""
```

- [ ] **Step 3: 创建 README.md**

```markdown
# AI 知识全书

覆盖全 AI 领域的教科书式知识库，面向初学者和有经验的开发者。

## 结构

- **Part 1 — 基础篇**：NumPy、PyTorch、CNN、Transformer
- **Part 2 — 图像处理**：OpenCV、目标检测、图像分割、姿态估计、目标追踪、OCR、ViT
- **Part 3 — 大模型 & NLP**：NLP 基础、HuggingFace、预训练模型、LLM、Prompt、RAG、Agent、微调
- **Part 4 — 多模态**：多模态理解、Qwen3-VL、多模态生成、跨模态对齐
- **Part 5 — 实战与面试**：AI 面试题精解

## 构建

```bash
# 安装 mdBook（可选，用于渲染为 HTML/PDF）
cargo install mdbook

# 渲染为 HTML
mdbook build
```

## 许可

CC BY-NC-SA 4.0
```

- [ ] **Step 4: 创建 .gitignore**

```gitignore
.superpowers/
*.log
```

- [ ] **Step 5: 验证目录结构**

```bash
find . -type d | sort
```

Expected: 15+ 目录包括所有 part 目录、assets、code 子目录。

- [ ] **Step 6: Commit**

```bash
git add README.md book.toml .gitignore parts/ assets/ code/
git commit -m "feat: initialize AI book project skeleton"
```

---

### Task 2: 创建 24 章模板文件

**Files:**
- Create: `parts/part-1-basics/ch01-numpy.md`
- Create: `parts/part-1-basics/ch02-pytorch.md`
- Create: `parts/part-1-basics/ch03-cnn.md`
- Create: `parts/part-1-basics/ch04-transformer.md`
- Create: `parts/part-2-cv/ch05-opencv.md`
- Create: `parts/part-2-cv/ch06-object-detection.md`
- Create: `parts/part-2-cv/ch07-segmentation.md`
- Create: `parts/part-2-cv/ch08-pose-estimation.md`
- Create: `parts/part-2-cv/ch09-tracking.md`
- Create: `parts/part-2-cv/ch10-ocr.md`
- Create: `parts/part-2-cv/ch11-vit-vision.md`
- Create: `parts/part-3-nlp-llm/ch12-nlp-basics.md`
- Create: `parts/part-3-nlp-llm/ch13-huggingface.md`
- Create: `parts/part-3-nlp-llm/ch14-pretrained-lm.md`
- Create: `parts/part-3-nlp-llm/ch15-llm-principles.md`
- Create: `parts/part-3-nlp-llm/ch16-prompt-engineering.md`
- Create: `parts/part-3-nlp-llm/ch17-rag.md`
- Create: `parts/part-3-nlp-llm/ch18-agent.md`
- Create: `parts/part-3-nlp-llm/ch19-finetuning.md`
- Create: `parts/part-4-multimodal/ch20-multimodal-understanding.md`
- Create: `parts/part-4-multimodal/ch21-qwen3-vl.md`
- Create: `parts/part-4-multimodal/ch22-multimodal-generation.md`
- Create: `parts/part-4-multimodal/ch23-crossmodal-alignment.md`
- Create: `parts/part-5-interview/ch24-ai-interview.md`

- [ ] **Step 1: 创建章节模板 script 并批量生成所有章节文件**

```bash
#!/usr/bin/env bash
# 批量生成 24 个章节 stub 文件

declare -A CHAPTERS
CHAPTERS=(
  ["part-1-basics/ch01-numpy"]="NumPy 科学计算"
  ["part-1-basics/ch02-pytorch"]="PyTorch 深度学习框架"
  ["part-1-basics/ch03-cnn"]="CNN 卷积神经网络"
  ["part-1-basics/ch04-transformer"]="Transformer 架构"
  ["part-2-cv/ch05-opencv"]="OpenCV 基础与图像处理"
  ["part-2-cv/ch06-object-detection"]="目标检测"
  ["part-2-cv/ch07-segmentation"]="图像分割"
  ["part-2-cv/ch08-pose-estimation"]="姿态估计"
  ["part-2-cv/ch09-tracking"]="目标追踪"
  ["part-2-cv/ch10-ocr"]="OCR 光学字符识别"
  ["part-2-cv/ch11-vit-vision"]="ViT 与视觉 Transformer 应用"
  ["part-3-nlp-llm/ch12-nlp-basics"]="NLP 基础与经典任务"
  ["part-3-nlp-llm/ch13-huggingface"]="HuggingFace 生态"
  ["part-3-nlp-llm/ch14-pretrained-lm"]="预训练语言模型"
  ["part-3-nlp-llm/ch15-llm-principles"]="大语言模型原理"
  ["part-3-nlp-llm/ch16-prompt-engineering"]="Prompt Engineering"
  ["part-3-nlp-llm/ch17-rag"]="RAG 检索增强生成"
  ["part-3-nlp-llm/ch18-agent"]="Agent 智能体"
  ["part-3-nlp-llm/ch19-finetuning"]="微调与对齐"
  ["part-4-multimodal/ch20-multimodal-understanding"]="多模态理解"
  ["part-4-multimodal/ch21-qwen3-vl"]="Qwen3-VL 系列解读"
  ["part-4-multimodal/ch22-multimodal-generation"]="多模态生成"
  ["part-4-multimodal/ch23-crossmodal-alignment"]="跨模态对齐与融合"
  ["part-5-interview/ch24-ai-interview"]="AI 面试题精解"
)

for key in "${!CHAPTERS[@]}"; do
  file="parts/${key}.md"
  title="${CHAPTERS[$key]}"
  part_num=$(echo "$key" | sed 's/part-\([0-9]\)-.*/\1/')
  ch_num=$(echo "$key" | sed 's/.*ch0\?\([0-9]*\)-.*/\1/')
  ch_num=$((10#$ch_num))

  cat > "$file" <<EOF
---
title: "${title}"
part: ${part_num}
chapter: ${ch_num}
status: draft
updated: $(date +%Y-%m-%d)
---

# Ch${ch_num} — ${title}

[待撰写]

## 参考

## 本章小结
EOF

  echo "Created: $file"
done
```

- [ ] **Step 2: 运行脚本生成所有章节文件**

```bash
cd /home/wusong/project/knowledge_bash && bash -c '
declare -A CHAPTERS
CHAPTERS=(
  ["part-1-basics/ch01-numpy"]="NumPy 科学计算"
  ["part-1-basics/ch02-pytorch"]="PyTorch 深度学习框架"
  ["part-1-basics/ch03-cnn"]="CNN 卷积神经网络"
  ["part-1-basics/ch04-transformer"]="Transformer 架构"
  ["part-2-cv/ch05-opencv"]="OpenCV 基础与图像处理"
  ["part-2-cv/ch06-object-detection"]="目标检测"
  ["part-2-cv/ch07-segmentation"]="图像分割"
  ["part-2-cv/ch08-pose-estimation"]="姿态估计"
  ["part-2-cv/ch09-tracking"]="目标追踪"
  ["part-2-cv/ch10-ocr"]="OCR 光学字符识别"
  ["part-2-cv/ch11-vit-vision"]="ViT 与视觉 Transformer 应用"
  ["part-3-nlp-llm/ch12-nlp-basics"]="NLP 基础与经典任务"
  ["part-3-nlp-llm/ch13-huggingface"]="HuggingFace 生态"
  ["part-3-nlp-llm/ch14-pretrained-lm"]="预训练语言模型"
  ["part-3-nlp-llm/ch15-llm-principles"]="大语言模型原理"
  ["part-3-nlp-llm/ch16-prompt-engineering"]="Prompt Engineering"
  ["part-3-nlp-llm/ch17-rag"]="RAG 检索增强生成"
  ["part-3-nlp-llm/ch18-agent"]="Agent 智能体"
  ["part-3-nlp-llm/ch19-finetuning"]="微调与对齐"
  ["part-4-multimodal/ch20-multimodal-understanding"]="多模态理解"
  ["part-4-multimodal/ch21-qwen3-vl"]="Qwen3-VL 系列解读"
  ["part-4-multimodal/ch22-multimodal-generation"]="多模态生成"
  ["part-4-multimodal/ch23-crossmodal-alignment"]="跨模态对齐与融合"
  ["part-5-interview/ch24-ai-interview"]="AI 面试题精解"
)
for key in "${!CHAPTERS[@]}"; do
  file="parts/${key}.md"
  title="${CHAPTERS[$key]}"
  part_num=$(echo "$key" | sed "s/part-\([0-9]\)-.*/\1/")
  ch_num=$(echo "$key" | sed "s/.*ch0\?\([0-9]*\)-.*/\1/")
  ch_num=$((10#$ch_num))
  cat > "$file" <<INNEREOF
---
title: "${title}"
part: ${part_num}
chapter: ${ch_num}
status: draft
updated: $(date +%Y-%m-%d)
---

# Ch${ch_num} — ${title}

[待撰写]

## 参考

## 本章小结
INNEREOF
  echo "Created: parts/${key}.md"
done
'
```

Expected: 24 files created.

- [ ] **Step 3: 验证章节文件数量和 frontmatter**

```bash
find parts -name "ch*.md" | wc -l       # Expected: 24
head -6 parts/part-1-basics/ch01-numpy.md   # Check frontmatter
head -6 parts/part-2-cv/ch11-vit-vision.md # Check frontmatter
head -6 parts/part-3-nlp-llm/ch18-agent.md # Check frontmatter
```

- [ ] **Step 4: Commit**

```bash
git add parts/
git commit -m "feat: add 24 chapter stub files with frontmatter"
```

---

### Task 3: 设置 Notion MCP 集成

**Files:**
- Create: `.claude/settings.local.json` (或更新已有)

- [ ] **Step 1: 确认 Notion API Token 已就绪**

用户需要在 [Notion Integrations](https://www.notion.so/my-integrations) 创建集成并获取 token，同时将需要访问的 Notion 页面 share 给该集成。

提示用户：
```
请在 https://www.notion.so/my-integrations 创建一个集成，获取 Internal Integration Token。
然后将你的 Notion 知识库页面 share 给该集成（页面右上角 ... → Connections → 选择你的集成）。
完成后把 token 发给我。
```

- [ ] **Step 2: 安装 notion-mcp-server 并配置环境变量**

```bash
# 检查是否已有 notion MCP server
# 推荐: https://github.com/suekou/mcp-notion-server
# 或使用 npm 安装
```

给出配置指引（待用户提供 token 后完成）：

```json
{
  "mcpServers": {
    "notion": {
      "command": "npx",
      "args": ["-y", "@notionhq/notion-mcp-server"],
      "env": {
        "NOTION_API_TOKEN": "<user-provided-token>"
      }
    }
  }
}
```

> **注意**: 具体的 MCP server 包名和配置取决于选定哪个 Notion MCP 实现。执行前需确认可用的 MCP server。

- [ ] **Step 3: 测试 Notion 读取流程**

用户提供 Notion 页面 URL 后，通过 MCP 工具读取内容，验证能正确获取 block 结构（标题、文本、代码块、图片、表格）。

- [ ] **Step 4: 编写 Notion → Markdown 转换脚本（可选）**

```bash
# 位置: scripts/notion-to-md.sh
# 功能: 读取 Notion 页面内容，映射为对应章节的 Markdown 格式
# 此脚本在 MCP 就绪后按需实现
```

- [ ] **Step 5: Commit**

```bash
git add scripts/ .claude/
git commit -m "feat: add Notion MCP integration config"
```

---

### Task 4: 添加渲染脚本（可选 — 后期执行）

**Files:**
- Create: `scripts/build-html.sh`
- Create: `scripts/build-pdf.sh`

> 此任务为后期扩展，当前不执行。在至少一个 Part 内容完成后才启用。

- [ ] **Step 1: 创建 mdBook 渲染脚本**

```bash
#!/usr/bin/env bash
# scripts/build-html.sh
set -e
cd "$(dirname "$0")/.."

# 生成 mdBook 所需的 SUMMARY.md
echo "# Summary" > SUMMARY.md
echo "" >> SUMMARY.md

for part_dir in parts/part-*/; do
  part_name=$(basename "$part_dir")
  echo "## ${part_name}" >> SUMMARY.md
  for ch in "$part_dir"*.md; do
    title=$(head -3 "$ch" | grep 'title:' | sed 's/title: "\(.*\)"/\1/')
    echo "  - [${title}](${ch})" >> SUMMARY.md
  done
  echo "" >> SUMMARY.md
done

mdbook build
echo "Built to book/html/"
```

### 执行顺序

Task 1 → Task 2 → Task 3（需用户提供 Notion token）→ Task 4（后期可选）

Task 1 和 Task 2 无依赖，可并行执行。
