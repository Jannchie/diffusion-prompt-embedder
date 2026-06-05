import torch
from transformers import CLIPTextModel, CLIPTokenizer

from diffusion_prompt_embedder.clip.tokenization import get_prompts_tokens_with_weights, group_tokens_and_weights
from diffusion_prompt_embedder.core.embedding import encode_prompt_chunks_batched


def get_embeddings_sd15(  # noqa: PLR0913
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    *,
    prompt: str = "",
    neg_prompt: str = "",
    pad_last_block: bool = False,
    clip_skip: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate weighted text embeddings for Stable Diffusion 1.5 models.

    This function processes both positive and negative prompts with weights and
    generates CLIP text embeddings for use in Stable Diffusion inference. It can
    handle arbitrarily long prompts by processing them in chunks and supports
    clip-skip for style control.

    The positive and negative prompts are padded to the same token length and
    encoded together as a 2-row batch (one text-encoder forward per chunk
    position) instead of one forward per chunk per prompt.

    Args:
        tokenizer (CLIPTokenizer): The CLIP tokenizer instance
        text_encoder (CLIPTextModel): The CLIP text encoder model
        prompt (str): The positive prompt with optional weights in parentheses
        neg_prompt (str): The negative prompt with optional weights in parentheses
        pad_last_block (bool): Whether to pad the last token block to full length
        clip_skip (int): CLIP skip, A1111/WebUI semantics (1 = last layer, 2 = penultimate layer)

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - prompt_embeds: Tensor of positive prompt embeddings
            - neg_prompt_embeds: Tensor of negative prompt embeddings

    Example:
        from transformers import CLIPTokenizer, CLIPTextModel

        tokenizer = CLIPTokenizer.from_pretrained(
            "openai/clip-vit-large-patch14",
        )
        text_encoder = CLIPTextModel.from_pretrained(
            "openai/clip-vit-large-patch14",
            torch_dtype=torch.float16
        ).to("cuda")

        prompt_embeds, neg_prompt_embeds = get_embeddings_sd15(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            prompt="a (white:1.2) cat",
            neg_prompt="blur, bad quality",
        )
    """
    device = text_encoder.device
    dtype = text_encoder.dtype

    # Get the eos token id from tokenizer
    eos = tokenizer.eos_token_id

    # Tokenize prompts with weights
    prompt_tokens, prompt_weights = get_prompts_tokens_with_weights(
        tokenizer,
        prompt,
    )
    neg_prompt_tokens, neg_prompt_weights = get_prompts_tokens_with_weights(
        tokenizer,
        neg_prompt,
    )

    # Pad the shorter prompt to match the longer one for consistent batch processing
    max_token_len = max(len(prompt_tokens), len(neg_prompt_tokens))
    token_groups: list[list[list[int]]] = []
    weight_groups: list[list[list[float]]] = []
    for tokens, weights in ((prompt_tokens, prompt_weights), (neg_prompt_tokens, neg_prompt_weights)):
        padding_len = max_token_len - len(tokens)
        groups, group_weights = group_tokens_and_weights(
            [*tokens, *([eos] * padding_len)],
            [*weights, *([1.0] * padding_len)],
            pad_last_block=pad_last_block,
        )
        token_groups.append(groups)
        weight_groups.append(group_weights)

    # Encode positive and negative prompts together, one forward per chunk position
    embeds = encode_prompt_chunks_batched(
        text_encoder,
        token_groups,
        weight_groups,
        device,
        dtype,
        clip_skip,
    )
    return embeds[0:1], embeds[1:2]
