import torch
from transformers import CLIPTextModel, CLIPTokenizer

from diffusion_prompt_embedder.clip.tokenization import get_prompts_tokens_with_weights, group_tokens_and_weights
from diffusion_prompt_embedder.core.embedding import encode_prompt_chunks_batched


def get_embeddings_sd15_batch(
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    *,
    prompts: list[str],
    pad_last_block: bool = True,
    clip_skip: int = 0,
) -> torch.Tensor:
    """
    Generate weighted text embeddings for multiple prompts in a batch.

    This function processes a list of prompts with weights and generates CLIP text
    embeddings for use in batch inference. It handles arbitrarily long prompts
    by processing them in chunks, pads all prompts to the same length, and supports
    clip-skip for style control.

    All prompts are padded with EOS to the longest prompt's token length, so every
    prompt yields the same chunk layout; each chunk position is then encoded with a
    single batched text-encoder forward (training hot path: a handful of forwards
    per step instead of one batch-1 forward per chunk per prompt).

    Args:
        tokenizer (CLIPTokenizer): The CLIP tokenizer instance
        text_encoder (CLIPTextModel): The CLIP text encoder model
        prompts (list[str]): List of prompts, each with optional weights in parentheses
        pad_last_block (bool): Whether to pad the last token block to full length
        clip_skip (int): CLIP skip, A1111/WebUI semantics (1 = last layer, 2 = penultimate layer)

    Returns:
        torch.Tensor: Tensor of embeddings for all prompts, shape [batch_size, seq_len, hidden_size]

    Example:
        from transformers import CLIPTokenizer, CLIPTextModel

        tokenizer = CLIPTokenizer.from_pretrained(
            "openai/clip-vit-large-patch14",
        )
        text_encoder = CLIPTextModel.from_pretrained(
            "openai/clip-vit-large-patch14",
            torch_dtype=torch.float16
        ).to("cuda")

        prompt_embeds = get_embeddings_sd15_batch(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            prompts=["a (white:1.2) cat", "a (blue:1.4) dog", "a red bird"],
        )
    """
    device = text_encoder.device
    dtype = text_encoder.dtype

    # Get the eos token id from tokenizer
    eos = tokenizer.eos_token_id

    # Tokenize all prompts with weights
    all_prompt_tokens: list[list[int]] = []
    all_prompt_weights: list[list[float]] = []
    max_token_len: int = 0

    for prompt in prompts:
        prompt_tokens, prompt_weights = get_prompts_tokens_with_weights(
            tokenizer,
            prompt,
        )
        all_prompt_tokens.append(prompt_tokens)
        all_prompt_weights.append(prompt_weights)
        max_token_len = max(max_token_len, len(prompt_tokens))

    # Pad all prompts to the same length and group into 77-token chunks
    token_groups: list[list[list[int]]] = []
    weight_groups: list[list[list[float]]] = []
    for prompt_tokens, prompt_weights in zip(all_prompt_tokens, all_prompt_weights, strict=True):
        padding_len = max_token_len - len(prompt_tokens)
        prompt_token_groups, prompt_weight_groups = group_tokens_and_weights(
            [*prompt_tokens, *([eos] * padding_len)],
            [*prompt_weights, *([1.0] * padding_len)],
            pad_last_block=pad_last_block,
        )
        token_groups.append(prompt_token_groups)
        weight_groups.append(prompt_weight_groups)

    # One batched text-encoder forward per chunk position
    return encode_prompt_chunks_batched(
        text_encoder,
        token_groups,
        weight_groups,
        device,
        dtype,
        clip_skip,
    )
