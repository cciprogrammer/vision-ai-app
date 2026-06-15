# Vision AI - 多模态图片理解应用

基于 **Qwen2-VL** 开源多模态大模型构建的智能图片理解与分析平台。

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Huggingface](https://img.shields.io/badge/🤗-Huggingface-orange)

## ✨ 功能特性

- 📷 **图片问答**：对任意图片提问，获得精准回答
- 🔍 **智能分析**：一键分析图片内容、场景、风格
- 📝 **描述生成**：自动生成图片简短描述
- 💬 **多轮对话**：支持上下文连续对话
- 🖥️ **Web界面**：简洁美观的Gradio界面

## 🛠️ 技术栈

- **模型**: [Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)
- **框架**: Gradio
- **后端**: PyTorch + Transformers
- **部署**: Hugging Face Spaces (免费)

## 🚀 快速开始

### 本地运行

```bash
# 1. 克隆项目
git clone https://github.com/YOUR_USERNAME/vision-ai-app.git
cd vision-ai-app

# 2. 安装依赖
pip install -r requirements.txt

# 3. 运行应用
python app.py
```

### Hugging Face Spaces 部署

1. Fork 本项目到你的 GitHub
2. 访问 [Hugging Face Spaces](https://huggingface.co/new-space)
3. 选择 GitHub，创建新 Space
4. 选择 Gradio SDK，自动部署！

## 📖 使用方法

1. **上传图片**：点击上传按钮或拖拽图片
2. **选择功能**：
   - 💬 问答模式：输入问题，获得回答
   - 🔍 分析图片：获取详细分析
   - 📝 图片描述：获取简短描述

## 📁 项目结构

```
vision-ai-app/
├── app.py              # Gradio 主应用
├── requirements.txt    # Python 依赖
├── README.md           # 项目说明
└── examples/           # 示例图片
```

## 💻 开发说明

### 模型要求

- 默认使用 `Qwen/Qwen2-VL-2B-Instruct` 模型
- 显存需求：约 4GB（2B模型）或 14GB（7B模型）

### 自定义模型

修改 `app.py` 中的 `MODEL_NAME`:

```python
MODEL_NAME = "你的模型名称"
```

## 📝 简历写法

### 项目描述

> 基于 Qwen2-VL 多模态大模型构建的智能图片理解与分析平台，实现了图片内容问答、智能分析、描述生成等功能。

### 技术栈

- 使用 Python + Gradio 构建 Web 应用
- 集成 Qwen2-VL 开源多模态模型
- 实现图片处理与 LLM 推理流程

### 个人贡献

- 设计并实现多模态图片理解完整流程
- 优化 Prompt 提升理解准确率
- 完成 Gradio Web 界面开发

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 License

MIT License

---

⭐ 如果对你有帮助，请给个项目一个 Star！
