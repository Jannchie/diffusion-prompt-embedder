import torch
from transformers import CLIPTextModel, CLIPTokenizer

from diffusion_prompt_embedder.clip.tokenization import get_prompts_tokens_with_weights, group_tokens_and_weights
from diffusion_prompt_embedder.core.embedding import encode_tokens_with_weights, setup_clip_for_embedding


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

    Args:
        tokenizer (CLIPTokenizer): The CLIP tokenizer instance
        text_encoder (CLIPTextModel): The CLIP text encoder model
        prompt (str): The positive prompt with optional weights in parentheses
        neg_prompt (str): The negative prompt with optional weights in parentheses
        pad_last_block (bool): Whether to pad the last token block to full length
        clip_skip (int): Number of layers to skip in CLIP model for style control

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

        prompt_embeds, neg_prompt_embeds = get_weighted_text_embeddings_sd15(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            prompt="a (white:1.2) cat",
            neg_prompt="blur, bad quality",
        )
    """
    # Setup CLIP model and get common parameters
    device, dtype, original_clip_layers, _ = setup_clip_for_embedding(
        text_encoder,
        clip_skip,
    )

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
    prompt_token_len = len(prompt_tokens)
    neg_prompt_token_len = len(neg_prompt_tokens)
    if prompt_token_len > neg_prompt_token_len:
        # Pad negative prompt with EOS tokens to match positive prompt length
        neg_prompt_tokens = neg_prompt_tokens + [eos] * abs(prompt_token_len - neg_prompt_token_len)
        neg_prompt_weights = neg_prompt_weights + [1.0] * abs(prompt_token_len - neg_prompt_token_len)
    else:
        # Pad positive prompt with EOS tokens to match negative prompt length
        prompt_tokens = prompt_tokens + [eos] * abs(prompt_token_len - neg_prompt_token_len)
        prompt_weights = prompt_weights + [1.0] * abs(prompt_token_len - neg_prompt_token_len)

    # Group tokens for processing in CLIP-compatible chunks (77 tokens per chunk)
    prompt_token_groups, prompt_weight_groups = group_tokens_and_weights(
        prompt_tokens.copy(),
        prompt_weights.copy(),
        pad_last_block=pad_last_block,
    )
    neg_prompt_token_groups, neg_prompt_weight_groups = group_tokens_and_weights(
        neg_prompt_tokens.copy(),
        neg_prompt_weights.copy(),
        pad_last_block=pad_last_block,
    )

    # Process token groups through the shared encoder function
    embeds = encode_tokens_with_weights(
        text_encoder,
        prompt_token_groups,
        prompt_weight_groups,
        device,
        dtype,
    )

    neg_embeds = encode_tokens_with_weights(
        text_encoder,
        neg_prompt_token_groups,
        neg_prompt_weight_groups,
        device,
        dtype,
    )

    # Concatenate all token group embeddings
    prompt_embeds = torch.cat(embeds, dim=1)
    neg_prompt_embeds = torch.cat(neg_embeds, dim=1)

    # Restore original CLIP layers if clip_skip was used
    if clip_skip > 0 and original_clip_layers is not None:
        text_encoder.text_model.encoder.layers = original_clip_layers

    return prompt_embeds, neg_prompt_embeds
