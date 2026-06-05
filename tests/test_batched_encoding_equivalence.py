"""
Equivalence tests for the batched encoding rewrite.

The 0.3.x implementation encoded one batch-1 chunk per text-encoder forward and
applied attention weights with a per-token Python loop; at training batch sizes
this meant batch*chunks forwards per step and starved the GPU. The rewrite
encodes each chunk position for the whole batch in a single forward and applies
weights with one vectorized multiply.

These tests pin the rewrite to a reference implementation of the old algorithm
(kept here, built from the legacy helpers) so outputs stay numerically
identical, and guard the forward count so the perf fix cannot silently regress.

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
from diffusion_prompt_embedder.core.embedding import clip_inner_model, encode_tokens_with_weights, setup_clip_for_embedding


class _FakeTokenizer:
    """Deterministic stand-in for CLIPTokenizer (word -> 1..3 stable token ids)."""

    bos_token_id = 49406
    eos_token_id = 49407

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


def _reference_batch(
    tokenizer: _FakeTokenizer,
    text_encoder: CLIPTextModel,
    prompts: list[str],
    *,
    pad_last_block: bool = True,
    clip_skip: int = 0,
) -> torch.Tensor:
    """The 0.3.x algorithm: per-prompt loop, one batch-1 forward per chunk."""
    device, dtype, original_clip_layers, _ = setup_clip_for_embedding(text_encoder, clip_skip)
    eos = tokenizer.eos_token_id

    all_tokens: list[list[int]] = []
    all_weights: list[list[float]] = []
    max_token_len = 0
    for prompt in prompts:
        tokens, weights = get_prompts_tokens_with_weights(tokenizer, prompt)  # type: ignore[arg-type]
        all_tokens.append(tokens)
        all_weights.append(weights)
        max_token_len = max(max_token_len, len(tokens))

    outputs: list[torch.Tensor] = []
    for tokens, weights in zip(all_tokens, all_weights, strict=True):
        padding_len = max_token_len - len(tokens)
        token_groups, weight_groups = group_tokens_and_weights(
            [*tokens, *([eos] * padding_len)],
            [*weights, *([1.0] * padding_len)],
            pad_last_block=pad_last_block,
        )
        embeds = encode_tokens_with_weights(text_encoder, token_groups, weight_groups, device, dtype)
        outputs.append(torch.cat(embeds, dim=1))

    result = torch.cat(outputs, dim=0)
    if clip_skip > 1 and original_clip_layers is not None:
        clip_inner_model(text_encoder).encoder.layers = original_clip_layers  # type: ignore[attr-defined]
    return result


SHORT_PROMPTS = [
    "a (white:1.2) cat sitting on a mat",
    "a (blue:1.4) dog, (best quality:1.1)",
    "red bird",
]

# 30 words x exactly 3 tokens (len 8 -> 8 % 3 + 1 = 3) = 90 tokens,
# reliably spilling past the 75-token first chunk
LONG_PROMPT = " ".join(f"tag{index:03d}ab" for index in range(30))


class TestBatchMatchesReference:
    def test_short_prompts(self, tokenizer: _FakeTokenizer, text_encoder: CLIPTextModel) -> None:
        with torch.no_grad():
            expected = _reference_batch(tokenizer, text_encoder, SHORT_PROMPTS)
            actual = get_embeddings_sd15_batch(tokenizer, text_encoder, prompts=SHORT_PROMPTS)  # type: ignore[arg-type]
        torch.testing.assert_close(actual, expected)

    def test_multi_chunk_with_clip_skip(self, tokenizer: _FakeTokenizer, text_encoder: CLIPTextModel) -> None:
        prompts = [LONG_PROMPT, "short (prompt:1.3)"]
        with torch.no_grad():
            expected = _reference_batch(tokenizer, text_encoder, prompts, clip_skip=2)
            actual = get_embeddings_sd15_batch(tokenizer, text_encoder, prompts=prompts, clip_skip=2)  # type: ignore[arg-type]
        assert actual.shape[1] > 77  # really exercised the multi-chunk path
        torch.testing.assert_close(actual, expected)

    def test_unpadded_last_block(self, tokenizer: _FakeTokenizer, text_encoder: CLIPTextModel) -> None:
        prompts = [LONG_PROMPT, LONG_PROMPT + " extra tail"]
        with torch.no_grad():
            expected = _reference_batch(tokenizer, text_encoder, prompts, pad_last_block=False)
            actual = get_embeddings_sd15_batch(tokenizer, text_encoder, prompts=prompts, pad_last_block=False)  # type: ignore[arg-type]
        torch.testing.assert_close(actual, expected)

    def test_clip_skip_layers_left_untouched(self, tokenizer: _FakeTokenizer, text_encoder: CLIPTextModel) -> None:
        layers_before = clip_inner_model(text_encoder).encoder.layers  # type: ignore[attr-defined]
        with torch.no_grad():
            get_embeddings_sd15_batch(tokenizer, text_encoder, prompts=SHORT_PROMPTS, clip_skip=2)  # type: ignore[arg-type]
        layers_after = clip_inner_model(text_encoder).encoder.layers  # type: ignore[attr-defined]
        assert layers_after is layers_before
        assert len(layers_after) == 4


class TestSinglePromptPair:
    def test_pos_neg_bitwise_match_reference(self, tokenizer: _FakeTokenizer, text_encoder: CLIPTextModel) -> None:
        """Preview path contract: pos/neg stay in separate batch-1 forwards.

        Inference stacks (diffusers/WebUI) encode each prompt individually, so
        the single-prompt API must be BIT-identical to the per-prompt reference -
        batching pos+neg into one forward changes fp16 matmul reduction order.
        torch.equal (not assert_close) pins that contract.
        """
        prompt, neg_prompt = "a (white:1.2) cat", "blur, bad quality, (worst:1.4)"
        with torch.no_grad():
            expected = _reference_batch(tokenizer, text_encoder, [prompt, neg_prompt], pad_last_block=True)
            actual_pos, actual_neg = get_embeddings_sd15(
                tokenizer,  # type: ignore[arg-type]
                text_encoder,
                prompt=prompt,
                neg_prompt=neg_prompt,
                pad_last_block=True,
            )
        assert torch.equal(actual_pos, expected[0:1])
        assert torch.equal(actual_neg, expected[1:2])

    def test_long_pos_short_neg(self, tokenizer: _FakeTokenizer, text_encoder: CLIPTextModel) -> None:
        with torch.no_grad():
            pos, neg = get_embeddings_sd15(
                tokenizer,  # type: ignore[arg-type]
                text_encoder,
                prompt=LONG_PROMPT,
                neg_prompt="blur",
                pad_last_block=True,
            )
        # Both sides must share the (multi-chunk) sequence length for CFG concat
        assert pos.shape == neg.shape
        assert pos.shape[1] > 77


class TestForwardCount:
    def test_one_forward_per_chunk_position(self, tokenizer: _FakeTokenizer, text_encoder: CLIPTextModel) -> None:
        """4 prompts x 2 chunks must cost 2 forwards, not 8 (the old behavior)."""
        prompts = [LONG_PROMPT, "a cat", "a (dog:1.2)", LONG_PROMPT + " tail"]
        calls: list[int] = []
        handle = text_encoder.register_forward_pre_hook(lambda *_: calls.append(1))
        try:
            with torch.no_grad():
                embeds = get_embeddings_sd15_batch(tokenizer, text_encoder, prompts=prompts)  # type: ignore[arg-type]
        finally:
            handle.remove()
        num_chunks = embeds.shape[1] // 77
        assert num_chunks >= 2
        assert len(calls) == num_chunks
