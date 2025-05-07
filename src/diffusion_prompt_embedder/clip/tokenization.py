from transformers import CLIPTokenizer

from diffusion_prompt_embedder.core.parser import parse_prompt_attention


def group_tokens_and_weights(
    token_ids: list[int],
    weights: list[float],
    *,
    pad_last_block: bool = True,
) -> tuple[list[list[int]], list[list[float]]]:
    """
    Group tokenized IDs and weights into CLIP-compatible chunks of 77 tokens.

    This function takes tokenized IDs and their corresponding weights, then groups them
    into chunks of 77 tokens (75 content tokens + BOS and EOS tokens). The last block
    can be padded with EOS tokens based on the pad_last_block parameter.

    Args:
        token_ids (list): Token IDs generated from the CLIP tokenizer
        weights (list): Corresponding weights for each token
        pad_last_block (bool): Whether to pad the last block to 75 tokens with EOS tokens

    Returns:
        tuple: A tuple containing:
            - list[list[int]]: Grouped token IDs with each sublist containing 77 tokens
            - list[list[float]]: Grouped weights matching the token IDs structure

    Example:
        token_groups, weight_groups = group_tokens_and_weights(
            token_ids=token_id_list,
            weights=token_weight_list
        )
    """
    # Define beginning-of-sequence and end-of-sequence token IDs
    bos, eos = 49406, 49407

    # Initialize empty lists for storing grouped tokens and weights
    new_token_ids = []
    new_weights = []

    # Process complete blocks of 75 tokens
    while len(token_ids) >= 75:
        # Extract the first 75 tokens and their weights
        head_75_tokens = [token_ids.pop(0) for _ in range(75)]
        head_75_weights = [weights.pop(0) for _ in range(75)]

        # Create a complete block with BOS and EOS tokens
        temp_77_token_ids = [bos, *head_75_tokens, eos]
        temp_77_weights = [1.0, *head_75_weights, 1.0]

        # Add the completed block to our result lists
        new_token_ids.append(temp_77_token_ids)
        new_weights.append(temp_77_weights)

    # Process remaining tokens if any exist
    if len(token_ids) > 0:
        # Calculate padding length if pad_last_block is True
        padding_len = 75 - len(token_ids) if pad_last_block else 0

        # Create the final block with appropriate padding
        temp_77_token_ids = [bos] + token_ids + [eos] * padding_len + [eos]
        new_token_ids.append(temp_77_token_ids)

        temp_77_weights = [1.0] + weights + [1.0] * padding_len + [1.0]
        new_weights.append(temp_77_weights)

    return new_token_ids, new_weights


def get_prompts_tokens_with_weights(
    clip_tokenizer: CLIPTokenizer,
    prompt: str | None,
) -> tuple[list[int], list[float]]:
    """
    Tokenize a prompt with attention weights into token IDs and their corresponding weights.

    This function processes prompts with weighted terms (like "a (cat:1.2) in the garden")
    and returns both the token IDs and their respective weights. Works for both positive
    and negative prompts in Stable Diffusion.

    Args:
        clip_tokenizer (CLIPTokenizer): The CLIP tokenizer instance
        prompt (str | None): A prompt string with optional weights in parentheses
                            If None or empty, defaults to "empty"

    Returns:
        tuple: A tuple containing:
            - list[int]: List of token IDs
            - list[float]: List of weights corresponding to each token

    Example:
        token_id_list, token_weight_list = get_prompts_tokens_with_weights(
            clip_tokenizer=clip_tokenizer,
            prompt="a (red:1.5) cat"
        )
    """
    # Use "empty" as default if prompt is None or empty
    if (prompt is None) or (len(prompt) < 1):
        prompt = "empty"

    # Parse the prompt to get text chunks and their weights
    texts_and_weights = parse_prompt_attention(prompt)
    text_tokens: list[int] = []
    text_weights: list[float] = []

    for word, weight in texts_and_weights:
        # Tokenize the text chunk, removing BOS/EOS tokens (positions 0 and -1)
        token = clip_tokenizer(
            word,
            truncation=False,  # Allow processing prompts of any length
        ).input_ids[1:-1]

        # Append new tokens to the full token list
        text_tokens = [*text_tokens, *token]

        # Apply the same weight to all tokens in this text chunk
        chunk_weights = [weight] * len(token)

        # Append weights to the full weights list
        text_weights = [*text_weights, *chunk_weights]

    return text_tokens, text_weights
