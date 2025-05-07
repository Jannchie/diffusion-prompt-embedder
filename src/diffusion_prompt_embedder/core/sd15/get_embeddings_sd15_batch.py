import torch
from transformers import CLIPTextModel, CLIPTokenizer

from diffusion_prompt_embedder.clip.tokenization import get_prompts_tokens_with_weights, group_tokens_and_weights
from diffusion_prompt_embedder.core.embedding import encode_tokens_with_weights, setup_clip_for_embedding


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

    Args:
        tokenizer (CLIPTokenizer): The CLIP tokenizer instance
        text_encoder (CLIPTextModel): The CLIP text encoder model
        prompts (list[str]): List of prompts, each with optional weights in parentheses
        pad_last_block (bool): Whether to pad the last token block to full length
        clip_skip (int): Number of layers to skip in CLIP model for style control

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

        prompt_embeds = get_weighted_text_embeddings_batch(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            prompts=["a (white:1.2) cat", "a (blue:1.4) dog", "a red bird"],
        )
    """
    # Setup CLIP model and get common parameters
    device, dtype, original_clip_layers, _ = setup_clip_for_embedding(
        text_encoder,
        clip_skip,
    )

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

    # Pad all prompts to the same length
    for i in range(len(all_prompt_tokens)):
        token_len = len(all_prompt_tokens[i])
        if token_len < max_token_len:
            padding_len = max_token_len - token_len
            all_prompt_tokens[i] = all_prompt_tokens[i] + [eos] * padding_len
            all_prompt_weights[i] = all_prompt_weights[i] + [1.0] * padding_len

    # Initialize list to hold embeddings for each prompt
    all_embeds = []

    # Process each prompt separately
    for prompt_idx in range(len(prompts)):
        # Group tokens for processing in CLIP-compatible chunks (77 tokens per chunk)
        prompt_token_groups, prompt_weight_groups = group_tokens_and_weights(
            all_prompt_tokens[prompt_idx].copy(),
            all_prompt_weights[prompt_idx].copy(),
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

        # Concatenate all token group embeddings for this prompt
        prompt_embeds = torch.cat(embeds, dim=1)
        all_embeds.append(prompt_embeds)

    # Stack all prompt embeddings into a batch
    batched_embeds = torch.cat(all_embeds, dim=0)

    # Restore original CLIP layers if clip_skip was used
    if clip_skip > 0 and original_clip_layers is not None:
        text_encoder.text_model.encoder.layers = original_clip_layers

    return batched_embeds
