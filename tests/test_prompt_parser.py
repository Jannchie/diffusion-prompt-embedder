import pytest
import torch
from transformers import CLIPTextModel, CLIPTokenizer

from diffusion_prompt_embedder import (
    parse_prompt_attention,
)
from diffusion_prompt_embedder.core.sd15.get_embeddings_sd15 import get_embeddings_sd15
from diffusion_prompt_embedder.core.sd15.get_embeddings_sd15_batch import get_embeddings_sd15_batch


@pytest.fixture(scope="module")
def text_encoder() -> CLIPTextModel:
    """Initialize the CLIP text encoder once for all tests in this module."""
    return CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")


@pytest.fixture(scope="module")
def tokenizer() -> CLIPTokenizer:
    """Initialize the CLIP tokenizer once for all tests in this module."""
    return CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")


class TestParsePromptAttention:
    def test_normal_text(self) -> None:
        result = parse_prompt_attention("normal text")
        assert result == [["normal text", 1.0]]

    def test_important_word(self) -> None:
        result = parse_prompt_attention("an (important) word")
        assert result == [["an ", 1.0], ["important", 1.1], [" word", 1.0]]

    def test_unbalanced_bracket(self) -> None:
        result = parse_prompt_attention("(unbalanced")
        assert result == [["unbalanced", 1.1]]

    def test_escaped_literals(self) -> None:
        result = parse_prompt_attention(r"\(literal\]")
        assert result == [["(literal]", 1.0]]

    def test_unnecessary_parens(self) -> None:
        result = parse_prompt_attention("(unnecessary)(parens)")
        assert result == [["unnecessaryparens", 1.1]]

    def test_complex_attention(self) -> None:
        result = parse_prompt_attention("a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).")
        assert result == [
            ["a ", 1.0],
            ["house", 1.5730000000000004],
            [" ", 1.1],
            ["on", 1.0],
            [" a ", 1.1],
            ["hill", 0.55],
            [", sun, ", 1.1],
            ["sky", 1.4641000000000006],
            [".", 1.1],
        ]

    def test_specified_weight(self) -> None:
        result = parse_prompt_attention("a (word:2.5) test")
        assert result == [["a ", 1.0], ["word", 2.5], [" test", 1.0]]

    def test_negative_weight(self) -> None:
        result = parse_prompt_attention("a (bad:-0.5) test")
        assert result == [["a ", 1.0], ["bad", -0.5], [" test", 1.0]]

    def test_square_brackets(self) -> None:
        result = parse_prompt_attention("a [less] important [thing]")
        expected_weight: float = 1 / 1.1
        assert result == [["a ", 1.0], ["less", expected_weight], [" important ", 1.0], ["thing", expected_weight]]

    def test_nested_brackets_mixed(self) -> None:
        result = parse_prompt_attention("a ([mixed]:1.5) test")
        assert result == [["a ", 1.0], ["mixed", 1.5 / 1.1], [" test", 1.0]]

    def test_uncluse_square_brackets(self) -> None:
        result = parse_prompt_attention("a [unclosed")
        assert result == [["a ", 1.0], ["unclosed", 1 / 1.1]]

    def test_empty_string(self) -> None:
        result = parse_prompt_attention("")
        assert result == [["", 1.0]]

    def test_with_break(self) -> None:
        result = parse_prompt_attention("text1 BREAK text2")
        assert result == [["text1", 1.0], ["BREAK", -1], ["text2", 1.0]]


class TestGetEmbeddingsSd15:
    def test_embedding_generation(self, text_encoder: CLIPTextModel, tokenizer: CLIPTokenizer) -> None:
        # Test prompt with weights
        prompt = "cat"

        # Generate embeddings
        prompt_embeds, negative_prompt_embeds = get_embeddings_sd15(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            prompt=prompt,
            pad_last_block=True,
        )

        # Verify the output shapes and types
        assert isinstance(prompt_embeds, torch.Tensor)
        assert isinstance(negative_prompt_embeds, torch.Tensor)
        assert prompt_embeds.shape == (1, 77, 768)
        assert negative_prompt_embeds.shape == (1, 77, 768)

    def test_long_embedding(self, text_encoder: CLIPTextModel, tokenizer: CLIPTokenizer) -> None:
        # Test prompt with weights that exceeds the maximum length
        tags = ["cat"] * 80
        prompt = " ".join(tags)

        # Generate embeddings
        prompt_embeds, negative_prompt_embeds = get_embeddings_sd15(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            prompt=prompt,
            pad_last_block=True,
        )

        # Verify the output shapes and types
        assert isinstance(prompt_embeds, torch.Tensor)
        assert isinstance(negative_prompt_embeds, torch.Tensor)
        assert prompt_embeds.shape == (1, 77 * 2, 768)
        assert negative_prompt_embeds.shape == (1, 77 * 2, 768)

    def test_clip_skip(self, text_encoder: CLIPTextModel, tokenizer: CLIPTokenizer) -> None:
        # Test prompt with weights that exceeds the maximum length
        prompt = "cat"

        # Generate embeddings
        prompt_embeds, negative_prompt_embeds = get_embeddings_sd15(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            prompt=prompt,
            pad_last_block=True,
            clip_skip=2,
        )

        assert isinstance(prompt_embeds, torch.Tensor)
        # Verify the output shapes and types
        assert isinstance(negative_prompt_embeds, torch.Tensor)
        assert prompt_embeds.shape == (1, 77, 768)
        assert negative_prompt_embeds.shape == (1, 77, 768)


class TestGetEmbeddingsSd15Batch:
    def test_batch_embedding_generation(self, text_encoder: CLIPTextModel, tokenizer: CLIPTokenizer) -> None:
        # Test prompts with different lengths
        prompts = ["cat", "a (dog:1.2)", "beautiful landscape with mountains"]

        # Generate embeddings
        batch_embeds = get_embeddings_sd15_batch(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            prompts=prompts,
            pad_last_block=True,
        )

        # Verify the output shapes and types
        assert isinstance(batch_embeds, torch.Tensor)
        assert batch_embeds.shape == (3, 77, 768)  # Batch size 3, 77 tokens, 768 dimensions

    def test_long_prompts_batch(self, text_encoder: CLIPTextModel, tokenizer: CLIPTokenizer) -> None:
        # Create one short prompt and one long prompt
        short_prompt = "cat"
        tags = ["dog"] * 80
        long_prompt = " ".join(tags)
        prompts = [short_prompt, long_prompt]

        # Generate embeddings
        batch_embeds = get_embeddings_sd15_batch(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            prompts=prompts,
        )

        # Verify the output shapes and types
        assert isinstance(batch_embeds, torch.Tensor)
        assert batch_embeds.shape[0] == 2  # Batch size 2
        assert batch_embeds.shape[2] == 768  # 768 dimensions
        # The sequence length should accommodate the longest prompt
        # Long prompt should need 2 blocks of 77 tokens
        assert batch_embeds.shape[1] == 77 * 2

    def test_with_weighted_prompts(self, text_encoder: CLIPTextModel, tokenizer: CLIPTokenizer) -> None:
        # Test prompts with various weights
        prompts = [
            "a (cat:1.5) in the garden",
            "a [dog:-0.5] with (flowers:1.2)",
            "a normal prompt without weights",
        ]

        # Generate embeddings
        batch_embeds = get_embeddings_sd15_batch(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            prompts=prompts,
        )

        # Verify the output shapes and types
        assert isinstance(batch_embeds, torch.Tensor)
        assert batch_embeds.shape == (3, 77, 768)  # Batch size 3, 77 tokens, 768 dimensions

    def test_clip_skip(self, text_encoder: CLIPTextModel, tokenizer: CLIPTokenizer) -> None:
        # Test prompts
        prompts = ["cat", "dog"]

        # Generate embeddings with clip_skip
        batch_embeds = get_embeddings_sd15_batch(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            prompts=prompts,
            clip_skip=2,
        )

        # Verify the output shapes and types
        assert isinstance(batch_embeds, torch.Tensor)
        assert batch_embeds.shape == (2, 77, 768)  # Batch size 2, 77 tokens, 768 dimensions

    def test_one_long_one_short(self, text_encoder: CLIPTextModel, tokenizer: CLIPTokenizer) -> None:
        # Create one short prompt and one long prompt
        short_prompt = "cat"
        tags = ["dog"] * 80
        long_prompt = " ".join(tags)
        prompts = [short_prompt, long_prompt]

        # Generate embeddings with clip_skip
        batch_embeds = get_embeddings_sd15_batch(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            prompts=prompts,
        )

        # Verify the output shapes and types
        assert isinstance(batch_embeds, torch.Tensor)
