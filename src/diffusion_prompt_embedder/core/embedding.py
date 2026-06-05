import torch
from transformers import CLIPTextModel


def clip_inner_model(text_encoder: CLIPTextModel) -> object:
    """
    Resolve the inner CLIP module that owns ``encoder.layers``.

    transformers 4.x wraps the transformer in ``text_encoder.text_model``,
    while transformers 5.x flattens ``CLIPTextModel`` so ``embeddings`` /
    ``encoder`` / ``final_layer_norm`` live directly on the model. The same
    holds for ``CLIPTextModelWithProjection``. Returning ``text_model`` when
    present and falling back to the encoder itself keeps both layouts working.

    Args:
        text_encoder: The CLIP text encoder model

    Returns:
        object: The module exposing ``encoder.layers``
    """
    return getattr(text_encoder, "text_model", text_encoder)


def encode_chunks_with_clip_skip(
    text_encoder: CLIPTextModel,
    input_ids: torch.Tensor,
    clip_skip: int = 0,
) -> torch.Tensor:
    """
    Run one text-encoder forward over a batch of token chunks with clip_skip.

    clip_skip uses A1111/WebUI semantics (1 = last layer, 2 = penultimate layer).
    Instead of truncating ``encoder.layers`` in place (which mutates the model and
    leaves it truncated if an exception fires before restore), this selects the
    matching hidden state and applies ``final_layer_norm`` - mathematically
    identical to running a truncated encoder, since CLIP applies the final norm
    after the last encoder layer.

    Args:
        text_encoder: The CLIP text encoder model
        input_ids: Token IDs of shape [batch, seq_len]
        clip_skip: CLIP skip in A1111/WebUI semantics; values <= 1 use the last layer

    Returns:
        torch.Tensor: Hidden states of shape [batch, seq_len, hidden_size]
    """
    if clip_skip > 1:
        output = text_encoder(input_ids, output_hidden_states=True)
        hidden_states = output.hidden_states
        # hidden_states[0] is the embedding output; clamp to the deepest real layer.
        index = max(-clip_skip, -(len(hidden_states) - 1))
        return clip_inner_model(text_encoder).final_layer_norm(hidden_states[index])
    return text_encoder(input_ids)[0]


def encode_prompt_chunks_batched(  # noqa: PLR0913
    text_encoder: CLIPTextModel,
    token_groups: list[list[list[int]]],
    weight_groups: list[list[list[float]]],
    device: torch.device,
    dtype: torch.dtype,
    clip_skip: int = 0,
) -> torch.Tensor:
    """
    Encode chunked prompts for a whole batch with one forward per chunk position.

    Every prompt must contribute the same number of chunks (callers guarantee this
    by padding all prompts to a common token length before grouping). Chunk
    position ``c`` across all prompts is encoded as a single ``[batch, chunk_len]``
    forward, and attention weights are applied with one vectorized multiply -
    instead of one batch-1 forward per chunk per prompt plus a per-token Python
    loop, which starves the GPU at training batch sizes.

    Args:
        text_encoder: The CLIP text encoder model
        token_groups: Per prompt, the list of 77-token chunks
        weight_groups: Per prompt, the weight chunks matching token_groups
        device: Device to run encoding on
        dtype: Data type for the weight tensors
        clip_skip: CLIP skip in A1111/WebUI semantics (1 = last layer)

    Returns:
        torch.Tensor: Embeddings of shape [batch, total_seq_len, hidden_size]
    """
    num_chunks = len(token_groups[0])
    chunk_embeds: list[torch.Tensor] = []
    for chunk_index in range(num_chunks):
        ids = torch.tensor(
            [groups[chunk_index] for groups in token_groups],
            dtype=torch.long,
            device=device,
        )
        weights = torch.tensor(
            [groups[chunk_index] for groups in weight_groups],
            dtype=dtype,
            device=device,
        )
        embeds = encode_chunks_with_clip_skip(text_encoder, ids, clip_skip)
        chunk_embeds.append(embeds * weights.unsqueeze(-1))
    return torch.cat(chunk_embeds, dim=1)


def encode_tokens_with_weights(
    text_encoder: CLIPTextModel,
    token_groups: list[list[int]],
    weight_groups: list[list[float]],
    device: torch.device,
    dtype: torch.dtype,
) -> list[torch.Tensor]:
    """
    Legacy helper to encode token groups of a single prompt and apply weights.

    Kept for backward compatibility; clip_skip is expected to be applied by the
    caller via ``setup_clip_for_embedding``. New code should prefer
    ``encode_prompt_chunks_batched``, which batches across prompts and handles
    clip_skip without mutating the model.

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
    for tokens, weights in zip(token_groups, weight_groups, strict=True):
        token_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
        weight_tensor = torch.tensor(weights, dtype=dtype, device=device)
        token_embedding = text_encoder(token_tensor)[0]
        # One vectorized multiply instead of one GPU op per token
        embeds.append(token_embedding * weight_tensor.view(1, -1, 1))
    return embeds


def setup_clip_for_embedding(
    text_encoder: CLIPTextModel,
    clip_skip: int = 0,
) -> tuple[torch.device, torch.dtype, object | None, int]:
    """
    Setup CLIP model for embedding generation and return common parameters.

    Args:
        text_encoder: The CLIP text encoder model
        clip_skip: CLIP skip in A1111/WebUI semantics (1 = last layer,
            2 = penultimate layer, ...); values <= 1 leave the model untouched

    Returns:
        tuple: (device, dtype, original_clip_layers, clip_skip_applied)
    """
    # Get the device and dtype from the text encoder
    device = text_encoder.device
    dtype = text_encoder.dtype

    # Store original layers for clip skip feature.
    # A1111/WebUI semantics: clip_skip=1 uses the last layer (no skip),
    # clip_skip=2 the penultimate layer (NAI convention) - k skips k-1 layers.
    original_clip_layers = None
    if clip_skip > 1:
        inner = clip_inner_model(text_encoder)
        original_clip_layers = inner.encoder.layers
        inner.encoder.layers = original_clip_layers[: -(clip_skip - 1)]

    return device, dtype, original_clip_layers, clip_skip
