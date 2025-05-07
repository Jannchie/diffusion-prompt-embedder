import re

# Regular expressions for prompt processing
# Matches the "AND" keyword (used to split prompts)
re_and = re.compile(r"\bAND\b")
# Matches weight format: "text:1.5", captures text and weight value
re_weight = re.compile(r"^((?:\s|.)*?)(?:\s*:\s*([-+]?(?:\d+\.?|\d*\.\d+)))?\s*$")
# Matches the "BREAK" keyword (used to insert separators in prompts)
re_break = re.compile(r"\s*\bBREAK\b\s*", re.DOTALL)

# Complex regular expression for parsing attention markers
# This regex identifies various brackets and weight markers used to enhance or reduce specific parts of prompts
re_attention = re.compile(
    r"""
    \\\(|      # Escaped left parenthesis \(
    \\\)|      # Escaped right parenthesis \)
    \\\[|      # Escaped left bracket \[
    \\]|       # Escaped right bracket \]
    \\\\|      # Escaped backslash \\
    \\|        # Single backslash (escape character)
    \(|        # Left parenthesis - starts an enhanced attention area
    \[|        # Left bracket - starts a reduced attention area
    :\s*([+-]?[.\d]+)\s*\)|  # Colon followed by number and right parenthesis - custom weight value
    \)|        # Right parenthesis - ends enhanced attention area
    ]|         # Right bracket - ends reduced attention area
    [^\\()\[\]:]+|  # Regular text (any text not containing special characters)
    :          # Single colon
    """,
    re.VERBOSE,  # Enables verbose mode, allowing comments and whitespace in regex
)


def apply_multiplier_to_range(
    tokens: list[list[str | float]],
    start_position: int,
    multiplier: float,
) -> None:
    """
    Applies a weight multiplier to a range of tokens starting from a specified position.

    This function is used to process weight adjustments for text within brackets,
    such as weight changes in (text) or [text].

    Args:
        tokens: List of [text, weight] pairs to modify
        start_position: Position to start applying the multiplier
        multiplier: Weight multiplier to apply
    """
    for p in range(start_position, len(tokens)):
        tokens[p][1] *= multiplier


def process_text_token(text: str) -> list[list[str | float]]:
    """
    Processes text tokens, specifically handling BREAK markers in the text.

    BREAK markers are used to insert special separators in prompts,
    typically used to divide different concepts or regions.

    Args:
        text: Text to process

    Returns:
        List of [text, weight] pairs
    """
    result = []
    # Split text by BREAK keyword
    parts = re.split(re_break, text)
    for i, part in enumerate(parts):
        if i > 0:
            # Add a special marker after each BREAK with weight -1
            result.append(["BREAK", -1])
        # Add regular text with default weight 1.0
        result.append([part, 1.0])
    return result


def merge_identical_weights(tokens: list[list[str | float]]) -> list[list[str | float]]:
    """
    Merges consecutive tokens with identical weights.

    When multiple consecutive text fragments have the same weight, this function
    combines them into one to simplify output and improve efficiency.

    Args:
        tokens: List of [text, weight] pairs

    Returns:
        List of merged tokens
    """
    if not tokens:
        return [["", 1.0]]  # Return a default value if list is empty

    i = 0
    while i + 1 < len(tokens):
        if tokens[i][1] == tokens[i + 1][1]:
            # When two consecutive tokens have the same weight, merge their text
            tokens[i][0] += tokens[i + 1][0]
            tokens.pop(i + 1)  # Remove the merged token
        else:
            i += 1

    return tokens


def parse_prompt_attention(text: str) -> list[list[str | float]]:
    """
    Parses a string with attention markers and returns a list of text and associated weight pairs.

    This function is the core of prompt parsing, handling various attention control symbols
    like parentheses and brackets used to adjust focus on different parts of the prompt during generation.

    Supported markers:
      (abc) - increases attention to abc by a multiplier of 1.1
      (abc:3.12) - increases attention to abc by a multiplier of 3.12
      [abc] - decreases attention to abc by a multiplier of 1.1
      \\( - literal character '('
      \\[ - literal character '['
      \\) - literal character ')'
      \\] - literal character ']'
      \\ - literal character '\'
      anything else - just text

    Args:
        text: Prompt text to parse

    Returns:
        List of [text, weight] pairs representing the parsed prompt parts and their weights
    """
    res: list[list[str | float]] = []  # Result list storing [text, weight] pairs
    round_brackets: list[int] = []  # Stack for parentheses, stores opening position
    square_brackets: list[int] = []  # Stack for brackets, stores opening position

    # Define weight multiplier constants
    round_bracket_multiplier = 1.1  # Default enhancement factor for parentheses
    square_bracket_multiplier = 1 / 1.1  # Default reduction factor for brackets (reciprocal)

    # Parse each token in the text using regex
    for m in re_attention.finditer(text):
        token_text = m.group(0)  # Current matched text
        weight = m.group(1)  # Possible weight value (if any)

        if token_text.startswith("\\"):
            # Handle escape characters - remove backslash, preserve original character
            res.append([token_text[1:], 1.0])
        elif token_text == "(":
            # Left parenthesis - push current position to stack, mark start of enhancement area
            round_brackets.append(len(res))
        elif token_text == "[":
            # Left bracket - push current position to stack, mark start of reduction area
            square_brackets.append(len(res))
        elif weight is not None and round_brackets:
            # Right parenthesis with custom weight - adjust area with specified weight
            apply_multiplier_to_range(res, round_brackets.pop(), float(weight))
        elif token_text == ")" and round_brackets:
            # Regular right parenthesis - enhance area with default multiplier
            apply_multiplier_to_range(res, round_brackets.pop(), round_bracket_multiplier)
        elif token_text == "]" and square_brackets:
            # Right bracket - reduce area with default multiplier
            apply_multiplier_to_range(res, square_brackets.pop(), square_bracket_multiplier)
        else:
            # Process regular text or unmatched brackets
            res.extend(process_text_token(token_text))

    # Handle unclosed brackets (ensure all opening brackets have corresponding closing brackets)
    for pos in round_brackets:
        # Apply default enhancement for unclosed parentheses
        apply_multiplier_to_range(res, pos, round_bracket_multiplier)

    for pos in square_brackets:
        # Apply default reduction for unclosed brackets
        apply_multiplier_to_range(res, pos, square_bracket_multiplier)

    # Merge consecutive tokens with identical weights
    res = merge_identical_weights(res)

    # Ensure all elements in the returned list have the correct types
    return [[str(text), float(weight)] for text, weight in res]
