# Diffusion Prompt Embedder

[![PyPI version](https://img.shields.io/pypi/v/diffusion-prompt-embedder.svg)](https://pypi.org/project/diffusion-prompt-embedder/)
[![Python Version](https://img.shields.io/pypi/pyversions/diffusion-prompt-embedder.svg)](https://pypi.org/project/diffusion-prompt-embedder/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen.svg)](https://github.com/jannchie/diffusion-prompt-embedder)

一个专门用于解析和处理带有权重的提示文本的 Python 库，支持嵌入生成和标记化，为 Stable Diffusion 等 AI 模型提供增强的文本处理能力。它兼容 SD Web UI 的权重提示，但不包括调度部分。

## 特性

- 💬 **提示解析**: 解析带有权重标记的文本提示（例如 `a (cat:1.5) in the garden`）
- 🔢 **权重管理**: 支持正向权重 `(text)` 和负向权重 `[text]` 语法
- 📚 **CLIP 集成**: 无缝集成 CLIP 文本模型进行嵌入生成
- 🔄 **批处理支持**: 高效处理多个提示的批处理
- 🪄 **长文本处理**: 支持超出 CLIP 标准上下文长度的长提示

## 安装

使用 pip 安装基础库:

```bash
pip install diffusion-prompt-embedder
```

## 使用示例

### 解析带有权重的提示

```python
from diffusion_prompt_embedder import parse_prompt_attention

# 基本解析
result = parse_prompt_attention("a (cat:1.5) in the garden")
print(result)  # [['a ', 1.0], ['cat', 1.5], [' in the garden', 1.0]]

# 使用方括号降低权重
result = parse_prompt_attention("a [cat] in the garden")
print(result)  # [['a ', 1.0], ['cat', 0.9090909090909091], [' in the garden', 1.0]]

# 复杂提示示例
result = parse_prompt_attention("a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).")
print(result)
```

### 生成 CLIP 嵌入

```python
import torch
from transformers import CLIPTokenizer, CLIPTextModel
from prompt_parser import get_embeddings_sd15

# 初始化 CLIP 模型
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained(
    "openai/clip-vit-large-patch14",
    torch_dtype=torch.float16
).to("cuda")

# 生成嵌入
prompt_embeds, neg_prompt_embeds = get_embeddings_sd15(
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    prompt="a (white:1.2) cat",
    neg_prompt="blur, bad quality",
    clip_skip=1  # 可选：跳过 CLIP 模型中的层
)

# 批处理多个提示
from prompt_parser import get_embeddings_sd_15_batch

batch_embeds = get_embeddings_sd_15_batch(
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    prompts=["a (white:1.2) cat", "a (blue:1.4) dog", "a red bird"]
)
```

## 提示语法

### 基本权重语法

- `(text)` - 将提示的权重提升 1.1 倍
- `(text:1.5)` - 将提示的权重设置为 1.5
- `[text]` - 将提示的权重降低为原来的 1/1.1
- `\( \[ \) \]` - 使用反斜杠转义括号字符

### BREAK 语法

使用 `BREAK` 关键字在提示中创建断点：

```python
result = parse_prompt_attention("text1 BREAK text2")
# 结果: [["text1", 1.0], ["BREAK", -1], ["text2", 1.0]]
```

## 开发

克隆仓库并安装开发依赖：

```bash
git clone https://github.com/jannchie/diffusion-prompt-parser.git
cd diffusion-prompt-parser
pip install -e ".[dev]"
```

运行测试：

```bash
pytest
```

## 许可证

[MIT](https://opensource.org/licenses/MIT)

## 作者

- Jianqi Pan ([@jannchie](https://github.com/jannchie))
