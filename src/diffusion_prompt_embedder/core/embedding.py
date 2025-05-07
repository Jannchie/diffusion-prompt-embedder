import torch
from transformers import CLIPTextModel, CLIPTokenizer

from diffusion_prompt_embedder.clip.tokenization import get_prompts_tokens_with_weights, group_tokens_and_weights


def _encode_tokens_with_weights(
    text_encoder: CLIPTextModel,
    token_groups: list[list[int]],
    weight_groups: list[list[float]],
    device: torch.device,
    dtype: torch.dtype,
) -> list[torch.Tensor]:
    """
    Internal helper function to encode token groups and apply weights.

    Args:
        text_encoder: The CLIP text encoder model
        token_groups: Grouped token IDs, each group has 77 tokens
        weight_groups: Grouped weights matching the token IDs
        device: Device to run encoding on
        dtype: Data type for tensors

    Returns:
        list[torch.Tensor]: List of encoded embeddings for each token group
    """
    embeds = []

    # Process each token group through the text encoder
    for i in range(len(token_groups)):
        # Process tokens
        token_tensor = torch.tensor(
            [token_groups[i]],
            dtype=torch.long,
            device=device,
        )
        weight_tensor = torch.tensor(
            weight_groups[i],
            dtype=dtype,
            device=device,
        )

        # Get embeddings from text encoder
        token_embedding = text_encoder(token_tensor)[0].squeeze(0)

        # Apply attention weights to token embeddings
        for j in range(len(weight_tensor)):
            token_embedding[j] = token_embedding[j] * weight_tensor[j]

        # Add batch dimension back and append to results
        token_embedding = token_embedding.unsqueeze(0)
        embeds.append(token_embedding)

    return embeds


def _setup_clip_for_embedding(
    text_encoder: CLIPTextModel,
    clip_skip: int = 0,
) -> tuple[torch.device, torch.dtype, object | None, int]:
    """
    Setup CLIP model for embedding generation and return common parameters.

    Args:
        text_encoder: The CLIP text encoder model
        clip_skip: Number of layers to skip in CLIP model

    Returns:
        tuple: (device, dtype, original_clip_layers, clip_skip_applied)
    """
    # Get the device and dtype from the text encoder
    device = text_encoder.device
    dtype = text_encoder.dtype

    # Store original layers for clip skip feature
    original_clip_layers = None
    if clip_skip > 0 and hasattr(text_encoder, "text_model"):
        original_clip_layers = text_encoder.text_model.encoder.layers
        text_encoder.text_model.encoder.layers = original_clip_layers[:-clip_skip]

    return device, dtype, original_clip_layers, clip_skip


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
    device, dtype, original_clip_layers, _ = _setup_clip_for_embedding(
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
    embeds = _encode_tokens_with_weights(
        text_encoder,
        prompt_token_groups,
        prompt_weight_groups,
        device,
        dtype,
    )

    neg_embeds = _encode_tokens_with_weights(
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


def get_embeddings_sd_15_batch(
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
    device, dtype, original_clip_layers, _ = _setup_clip_for_embedding(
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
        embeds = _encode_tokens_with_weights(
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
