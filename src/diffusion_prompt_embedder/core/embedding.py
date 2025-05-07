import torch
from transformers import CLIPTextModel


def encode_tokens_with_weights(
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


def setup_clip_for_embedding(
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
