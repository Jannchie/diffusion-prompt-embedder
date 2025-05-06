from prompt_parser import (
    get_learned_conditioning_prompt_schedules,
    parse_prompt_attention,
)


class TestGetLearnedConditioningPromptSchedules:
    def test_basic_prompt(self) -> None:
        result = get_learned_conditioning_prompt_schedules(["test"], 10)[0]
        assert result == [[10, "test"]]

    def test_simple_scheduled_prompt(self) -> None:
        result = get_learned_conditioning_prompt_schedules(["a [b:3]"], 10)[0]
        assert result == [[3, "a "], [10, "a b"]]

    def test_scheduled_prompt_with_space(self) -> None:
        result = get_learned_conditioning_prompt_schedules(["a [b: 3]"], 10)[0]
        assert result == [[3, "a "], [10, "a b"]]

    def test_nested_brackets(self) -> None:
        result = get_learned_conditioning_prompt_schedules(["a [[[b]]:2]"], 10)[0]
        assert result == [[2, "a "], [10, "a [[b]]"]]

    def test_nested_parentheses(self) -> None:
        result = get_learned_conditioning_prompt_schedules(["[(a:2):3]"], 10)[0]
        assert result == [[3, ""], [10, "(a:2)"]]

    def test_complex_prompt_with_spaces(self) -> None:
        result = get_learned_conditioning_prompt_schedules(["a [b : c : 1] d"], 10)[0]
        assert result == [[1, "a b  d"], [10, "a  c  d"]]

    def test_complex_nested_scheduling(self) -> None:
        result = get_learned_conditioning_prompt_schedules(["a[b:[c:d:2]:1]e"], 10)[0]
        assert result == [[1, "abe"], [2, "ace"], [10, "ade"]]

    def test_unbalanced_bracket(self) -> None:
        result = get_learned_conditioning_prompt_schedules(["a [unbalanced"], 10)[0]
        assert result == [[10, "a [unbalanced"]]

    def test_decimal_step(self) -> None:
        result = get_learned_conditioning_prompt_schedules(["a [b:.5] c"], 10)[0]
        assert result == [[5, "a  c"], [10, "a b c"]]

    def test_unbalanced_mixed_brackets(self) -> None:
        result = get_learned_conditioning_prompt_schedules(["((a][:b:c [d:3]"], 10)[0]
        assert result == [[3, "((a][:b:c "], [10, "((a][:b:c d"]]

    def test_alternating_prompts(self) -> None:
        result = get_learned_conditioning_prompt_schedules(["[a|(b:1.1)]"], 10)[0]
        assert result == [
            [1, "a"],
            [2, "(b:1.1)"],
            [3, "a"],
            [4, "(b:1.1)"],
            [5, "a"],
            [6, "(b:1.1)"],
            [7, "a"],
            [8, "(b:1.1)"],
            [9, "a"],
            [10, "(b:1.1)"],
        ]

    def test_alternating_with_empty(self) -> None:
        result = get_learned_conditioning_prompt_schedules(["[fe|]male"], 10)[0]
        assert result == [
            [1, "female"],
            [2, "male"],
            [3, "female"],
            [4, "male"],
            [5, "female"],
            [6, "male"],
            [7, "female"],
            [8, "male"],
            [9, "female"],
            [10, "male"],
        ]

    def test_multiple_alternates(self) -> None:
        result = get_learned_conditioning_prompt_schedules(["[fe|||]male"], 10)[0]
        assert result == [
            [1, "female"],
            [2, "male"],
            [3, "male"],
            [4, "male"],
            [5, "female"],
            [6, "male"],
            [7, "male"],
            [8, "male"],
            [9, "female"],
            [10, "male"],
        ]

    def test_hires_steps(self) -> None:
        result = get_learned_conditioning_prompt_schedules(["a [b:.5] c"], 10, 10)[0]
        assert result == [[10, "a b c"]]

    def test_hires_steps_with_larger_value(self) -> None:
        result = get_learned_conditioning_prompt_schedules(["a [b:1.5] c"], 10, 10)[0]
        assert result == [[5, "a  c"], [10, "a b c"]]


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
