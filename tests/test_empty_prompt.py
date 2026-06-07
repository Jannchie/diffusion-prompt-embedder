"""
Regression tests for empty-prompt encoding.

Versions <= 0.4.1 substituted the literal word "empty" for empty/None prompts,
so the "unconditional" embedding was actually the conditional embedding of the
token "empty". The ecosystem convention (diffusers / WebUI / sd-scripts) for an
empty prompt is the fixed sequence [BOS, EOS*76]. This matters for training:
condition dropout feeds "" to learn the CFG unconditional branch, and inference
CFG uses the [BOS, EOS*76] anchor - they must be the same distribution.

These tests pin the contract: an empty prompt yields zero content tokens, the
chunker emits the standard unconditional block, and the resulting embedding is
bit-identical to forwarding [BOS, EOS*76] directly (all weights are 1.0, and
multiplying by 1.0 is exact in IEEE arithmetic).

Everything runs on a tiny randomly initialized CLIPTextModel with a fake
deterministic tokenizer - no network, no model downloads.
"""

from __future__ import annotations

import zlib
from types import SimpleNamespace

import pytest
import torch
from transformers import CLIPTextConfig, CLIPTextModel

from diffusion_prompt_embedder import get_embeddings_sd15, get_embeddings_sd15_batch
from diffusion_prompt_embedder.clip.tokenization import get_prompts_tokens_with_weights, group_tokens_and_weights

BOS, EOS = 49406, 49407
UNCOND_BLOCK = [BOS, *([EOS] * 76)]


class _FakeTokenizer:
    """Deterministic stand-in for CLIPTokenizer (word -> 1..3 stable token ids)."""

    bos_token_id = BOS
    eos_token_id = EOS

    def __call__(self, text: str, truncation: bool = False) -> SimpleNamespace:  # noqa: ARG002, FBT001, FBT002
        ids: list[int] = []
        for word in text.split():
            num_tokens = (len(word) % 3) + 1
            base = (zlib.crc32(word.encode()) % 40000) + 1000
            ids.extend((base + offset) % 49000 + 100 for offset in range(num_tokens))
        return SimpleNamespace(input_ids=[self.bos_token_id, *ids, self.eos_token_id])


@pytest.fixture(scope="module")
def tokenizer() -> _FakeTokenizer:
    return _FakeTokenizer()


@pytest.fixture(scope="module")
def text_encoder() -> CLIPTextModel:
    config = CLIPTextConfig(
        vocab_size=49408,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        max_position_embeddings=77,
    )
    torch.manual_seed(42)
    return CLIPTextModel(config).eval()


class TestEmptyPromptTokens:
    def test_empty_string_yields_no_tokens(self, tokenizer: _FakeTokenizer) -> None:
        tokens, weights = get_prompts_tokens_with_weights(tokenizer, "")  # type: ignore[arg-type]
        assert tokens == []
        assert weights == []

    def test_none_yields_no_tokens(self, tokenizer: _FakeTokenizer) -> None:
        tokens, weights = get_prompts_tokens_with_weights(tokenizer, None)  # type: ignore[arg-type]
        assert tokens == []
        assert weights == []


class TestEmptyChunking:
    def test_zero_tokens_emit_standard_uncond_block(self) -> None:
        token_groups, weight_groups = group_tokens_and_weights([], [], pad_last_block=True)
        assert token_groups == [UNCOND_BLOCK]
        assert weight_groups == [[1.0] * 77]

    def test_zero_tokens_unpadded(self) -> None:
        token_groups, weight_groups = group_tokens_and_weights([], [], pad_last_block=False)
        assert token_groups == [[BOS, EOS]]
        assert weight_groups == [[1.0, 1.0]]

    def test_nonempty_chunking_unchanged(self) -> None:
        token_groups, _ = group_tokens_and_weights([100, 200, 300], [1.0, 1.0, 1.0], pad_last_block=True)
        assert token_groups == [[BOS, 100, 200, 300, *([EOS] * 72), EOS]]


class TestEmptyPromptEmbeddings:
    def test_matches_native_uncond_bitwise(self, tokenizer: _FakeTokenizer, text_encoder: CLIPTextModel) -> None:
        """"" must encode exactly as a direct forward of [BOS, EOS*76]."""
        ids = torch.tensor([UNCOND_BLOCK], dtype=torch.long)
        with torch.no_grad():
            expected = text_encoder(ids)[0]
            actual = get_embeddings_sd15_batch(tokenizer, text_encoder, prompts=[""])  # type: ignore[arg-type]
        assert torch.equal(actual, expected)

    def test_not_the_word_empty(self, tokenizer: _FakeTokenizer, text_encoder: CLIPTextModel) -> None:
        """Regression: "" must NOT encode as the literal word "empty" (<= 0.4.1 bug)."""
        with torch.no_grad():
            uncond = get_embeddings_sd15_batch(tokenizer, text_encoder, prompts=[""])  # type: ignore[arg-type]
            word_empty = get_embeddings_sd15_batch(tokenizer, text_encoder, prompts=["empty"])  # type: ignore[arg-type]
        assert not torch.equal(uncond, word_empty)

    def test_mixed_batch(self, tokenizer: _FakeTokenizer, text_encoder: CLIPTextModel) -> None:
        """Condition-dropout shape: empty rows ride along with captioned rows."""
        with torch.no_grad():
            batch = get_embeddings_sd15_batch(tokenizer, text_encoder, prompts=["a white cat", ""])  # type: ignore[arg-type]
            uncond = get_embeddings_sd15_batch(tokenizer, text_encoder, prompts=[""])  # type: ignore[arg-type]
        assert batch.shape[0] == 2
        torch.testing.assert_close(batch[1:2], uncond)

    def test_single_prompt_pair_both_empty(self, tokenizer: _FakeTokenizer, text_encoder: CLIPTextModel) -> None:
        with torch.no_grad():
            pos, neg = get_embeddings_sd15(tokenizer, text_encoder, prompt="", neg_prompt="", pad_last_block=True)  # type: ignore[arg-type]
        assert pos.shape == neg.shape == (1, 77, 64)
        assert torch.equal(pos, neg)
