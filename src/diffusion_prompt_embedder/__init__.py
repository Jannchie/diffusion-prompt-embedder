"""
prompt_parser: A library for parsing and processing text prompts with attention weights.

This package provides tools for parsing text prompts with attention weights syntax,
tokenizing prompts, and generating embeddings for use with Stable Diffusion models.
"""

from __future__ import annotations

from diffusion_prompt_embedder.core.parser import parse_prompt_attention
from diffusion_prompt_embedder.core.sd15 import get_embeddings_sd15, get_embeddings_sd15_batch

__all__ = [
    "get_embeddings_sd15",
    "get_embeddings_sd15_batch",
    "parse_prompt_attention",
]
