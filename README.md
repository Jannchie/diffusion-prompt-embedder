# Diffusion Prompt Embedder

[![PyPI version](https://img.shields.io/pypi/v/diffusion-prompt-embedder.svg)](https://pypi.org/project/diffusion-prompt-embedder/)
[![Python Version](https://img.shields.io/pypi/pyversions/diffusion-prompt-embedder.svg)](https://pypi.org/project/diffusion-prompt-embedder/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen.svg)](https://github.com/jannchie/diffusion-prompt-embedder)

A Python library specialized for parsing and processing weighted prompt text, supporting embedding generation and tokenization to enhance text processing for AI models like Stable Diffusion. It's compatible with SD Web UI's weighted prompts but doesn't include scheduling.

## Features

- 💬 **Prompt Parsing**: Parse text prompts with weight markers (e.g., `a (cat:1.5) in the garden`)
- 🔢 **Weight Management**: Support for positive weight `(text)` and negative weight `[text]` syntax
- 📚 **CLIP Integration**: Seamless integration with CLIP text models for embedding generation
- 🔄 **Batch Processing**: Efficiently process batches of multiple prompts
- 🪄 **Long Text Support**: Handle prompts that exceed standard CLIP context length

## Installation

Install the base library using pip:

```bash
pip install diffusion-prompt-embedder
```

## Usage Examples

### Parse Weighted Prompts

```python
from diffusion_prompt_embedder import parse_prompt_attention

# Basic parsing
result = parse_prompt_attention("a (cat:1.5) in the garden")
print(result)  # [['a ', 1.0], ['cat', 1.5], [' in the garden', 1.0]]

# Using brackets to lower weight
result = parse_prompt_attention("a [cat] in the garden")
print(result)  # [['a ', 1.0], ['cat', 0.9090909090909091], [' in the garden', 1.0]]

# Complex prompt example
result = parse_prompt_attention("a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).")
print(result)
```

### Generate CLIP Embeddings

```python
import torch
from transformers import CLIPTokenizer, CLIPTextModel
from prompt_parser import get_embeddings_sd15

# Initialize CLIP model
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained(
    "openai/clip-vit-large-patch14",
    torch_dtype=torch.float16
).to("cuda")

# Generate embeddings
prompt_embeds, neg_prompt_embeds = get_embeddings_sd15(
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    prompt="a (white:1.2) cat",
    neg_prompt="blur, bad quality",
    clip_skip=1  # Optional: skip layers in CLIP model
)

# Batch processing multiple prompts
from prompt_parser import get_embeddings_sd_15_batch

batch_embeds = get_embeddings_sd_15_batch(
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    prompts=["a (white:1.2) cat", "a (blue:1.4) dog", "a red bird"]
)
```

## Prompt Syntax

### Basic Weight Syntax

- `(text)` - Increases the prompt weight by 1.1x
- `(text:1.5)` - Sets the prompt weight to 1.5
- `[text]` - Decreases the prompt weight to 1/1.1 of original
- `\( \[ \) \]` - Use backslash to escape bracket characters

### BREAK Syntax

Use the `BREAK` keyword to create breakpoints in prompts:

```python
result = parse_prompt_attention("text1 BREAK text2")
# Result: [["text1", 1.0], ["BREAK", -1], ["text2", 1.0]]
```

## Development

Clone the repository and install development dependencies:

```bash
git clone https://github.com/jannchie/diffusion-prompt-parser.git
cd diffusion-prompt-parser
pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```

## License

[MIT](https://opensource.org/licenses/MIT)

## Author

- Jianqi Pan ([@jannchie](https://github.com/jannchie))
