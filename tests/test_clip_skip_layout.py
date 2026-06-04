"""
Lightweight tests for clip_skip handling across transformers 4.x and 5.x layouts.

transformers 4.x wraps the transformer in ``text_encoder.text_model`` while
transformers 5.x flattens ``CLIPTextModel`` so ``encoder`` lives on the model
itself. These tests use minimal fakes (no model download) to verify that
clip_skip truncates and restores ``encoder.layers`` on both layouts.
"""

from __future__ import annotations

import torch

from diffusion_prompt_embedder.core.embedding import clip_inner_model, setup_clip_for_embedding


class _FakeEncoder:
    def __init__(self, num_layers: int) -> None:
        # A plain list stands in for nn.ModuleList; slicing semantics match.
        self.layers = list(range(num_layers))


class _FlatTextEncoder:
    """transformers 5.x style: encoder lives directly on the model."""

    def __init__(self, num_layers: int = 12) -> None:
        self.encoder = _FakeEncoder(num_layers)
        self.device = torch.device("cpu")
        self.dtype = torch.float32


class _NestedTextEncoder:
    """transformers 4.x style: encoder lives under .text_model."""

    def __init__(self, num_layers: int = 12) -> None:
        self.text_model = _FlatTextEncoder(num_layers)
        self.device = torch.device("cpu")
        self.dtype = torch.float32


class TestClipInnerModel:
    def test_flat_layout_returns_self(self) -> None:
        encoder = _FlatTextEncoder()
        assert clip_inner_model(encoder) is encoder

    def test_nested_layout_returns_text_model(self) -> None:
        encoder = _NestedTextEncoder()
        assert clip_inner_model(encoder) is encoder.text_model


class TestSetupClipForEmbedding:
    def test_flat_layout_truncates_and_restores(self) -> None:
        """transformers 5.x: clip_skip must take effect (regression guard)."""
        encoder = _FlatTextEncoder(num_layers=12)
        original = encoder.encoder.layers

        _, _, original_clip_layers, _ = setup_clip_for_embedding(encoder, clip_skip=2)

        # Layers were truncated by clip_skip, not silently skipped.
        assert original_clip_layers is original
        assert len(encoder.encoder.layers) == 10

        # Restore path used by the embedding functions.
        clip_inner_model(encoder).encoder.layers = original_clip_layers
        assert encoder.encoder.layers is original
        assert len(encoder.encoder.layers) == 12

    def test_nested_layout_truncates_and_restores(self) -> None:
        """transformers 4.x: nested .text_model path still works."""
        encoder = _NestedTextEncoder(num_layers=12)
        original = encoder.text_model.encoder.layers

        _, _, original_clip_layers, _ = setup_clip_for_embedding(encoder, clip_skip=2)

        assert original_clip_layers is original
        assert len(encoder.text_model.encoder.layers) == 10

        clip_inner_model(encoder).encoder.layers = original_clip_layers
        assert encoder.text_model.encoder.layers is original
        assert len(encoder.text_model.encoder.layers) == 12

    def test_clip_skip_zero_leaves_layers_untouched(self) -> None:
        encoder = _FlatTextEncoder(num_layers=12)
        original = encoder.encoder.layers

        _, _, original_clip_layers, applied = setup_clip_for_embedding(encoder, clip_skip=0)

        assert original_clip_layers is None
        assert applied == 0
        assert encoder.encoder.layers is original
        assert len(encoder.encoder.layers) == 12
