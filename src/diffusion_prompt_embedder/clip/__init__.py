"""
CLIP model functionality for embedding generation.
"""

from diffusion_prompt_embedder.clip.tokenization import (
    get_prompts_tokens_with_weights,
    group_tokens_and_weights,
)

__all__ = [
    "get_prompts_tokens_with_weights",
    "group_tokens_and_weights",
]
