from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple, TypeVar

import lark
import torch

if TYPE_CHECKING:
    from collections.abc import Generator


# a prompt like this: "fantasy landscape with a [mountain:lake:0.25] and [an oak:a christmas tree:0.75][ in foreground::0.6][: in background:0.25] [shoddy:masterful:0.5]"
# will be represented with prompt_schedule like this (assuming steps=100):
# [25, 'fantasy landscape with a mountain and an oak in foreground shoddy']
# [50, 'fantasy landscape with a lake and an oak in foreground in background shoddy']
# [60, 'fantasy landscape with a lake and an oak in foreground in background masterful']
# [75, 'fantasy landscape with a lake and an oak in background masterful']
# [100, 'fantasy landscape with a lake and a christmas tree in background masterful']

schedule_parser = lark.Lark(
    r"""
!start: (prompt | /[][():]/+)*
prompt: (emphasized | scheduled | alternate | plain | WHITESPACE)*
!emphasized: "(" prompt ")"
        | "(" prompt ":" prompt ")"
        | "[" prompt "]"
scheduled: "[" [prompt ":"] prompt ":" [WHITESPACE] NUMBER [WHITESPACE] "]"
alternate: "[" prompt ("|" [prompt])+ "]"
WHITESPACE: /\s+/
plain: /([^\\\[\]():|]|\\.)+/
%import common.SIGNED_NUMBER -> NUMBER
""",
)

T = TypeVar("T")


def get_learned_conditioning_prompt_schedules(
    prompts: list[str],
    base_steps: int,
    hires_steps: int | None = None,
    *,
    use_old_scheduling: bool = False,
) -> list[list[tuple[int, str]]]:
    """
    Parses a list of prompts into a list of prompt schedules.
    Each schedule is a list of tuples (step, prompt) where step is the sampling step
    at which the prompt should be replaced with the next one.
    """

    if hires_steps is None or use_old_scheduling:
        int_offset = 0
        flt_offset = 0
        steps = base_steps
    else:
        int_offset = base_steps
        flt_offset = 1.0
        steps = hires_steps

    def collect_steps(steps: int, tree: lark.Tree) -> list[int]:
        res = [steps]

        class CollectSteps(lark.Visitor):
            def scheduled(self, tree: lark.Tree) -> None:
                s = tree.children[-2]
                v = float(s)
                v = (v * steps if v < 1 else v) if use_old_scheduling else ((v - flt_offset) * steps if "." in s else v - int_offset)
                tree.children[-2] = min(steps, int(v))
                if tree.children[-2] >= 1:
                    res.append(tree.children[-2])

            def alternate(self, _: lark.Tree) -> None:
                res.extend(range(1, steps + 1))

        CollectSteps().visit(tree)
        return sorted(set(res))

    def at_step(step: int, tree: lark.Tree) -> str:
        class AtStep(lark.Transformer):
            def scheduled(self, args: list) -> Generator[str, None, None]:
                before, after, _, when, _ = args
                yield before or () if step <= when else after

            def alternate(self, args: list) -> Generator[str, None, None]:
                args = [arg or "" for arg in args]
                yield args[(step - 1) % len(args)]

            def start(self, args: list) -> str:
                def flatten(x: Any) -> Generator[str, None, None]:
                    if isinstance(x, str):
                        yield x
                    else:
                        for gen in x:
                            yield from flatten(gen)

                return "".join(flatten(args))

            def plain(self, args: list) -> Generator[str, None, None]:
                yield args[0].value

            def __default__(self, data: Any, children: list, meta: Any) -> Generator[str, None, None]:
                yield from children

        return AtStep().transform(tree)

    def get_schedule(prompt: str) -> list[tuple[int, str]]:
        try:
            tree = schedule_parser.parse(prompt)
        except lark.exceptions.LarkError:
            return [[steps, prompt]]
        return [[t, at_step(t, tree)] for t in collect_steps(steps, tree)]

    promptdict = {prompt: get_schedule(prompt) for prompt in set(prompts)}
    return [promptdict[prompt] for prompt in prompts]


class ScheduledPromptConditioning(NamedTuple):
    """A scheduled prompt conditioning with the step it ends at and the condition"""

    end_at_step: int
    cond: Any


class SdConditioning(list):
    """
    A list with prompts for stable diffusion's conditioner model.
    Can also specify width and height of created image - SDXL needs it.
    """

    def __init__(
        self,
        prompts: list[str],
        width: int | None = None,
        height: int | None = None,
        copy_from: SdConditioning | list[str] | None = None,
        *,
        is_negative_prompt: bool = False,
    ) -> None:
        super().__init__()
        self.extend(prompts)

        if copy_from is None:
            copy_from = prompts

        self.is_negative_prompt = is_negative_prompt or getattr(copy_from, "is_negative_prompt", False)
        self.width = width or getattr(copy_from, "width", None)
        self.height = height or getattr(copy_from, "height", None)


def get_learned_conditioning(
    model: Any,
    prompts: SdConditioning | list[str],
    steps: int,
    hires_steps: int | None = None,
    *,
    use_old_scheduling: bool = False,
) -> list[list[ScheduledPromptConditioning]]:
    """converts a list of prompts into a list of prompt schedules - each schedule is a list of ScheduledPromptConditioning, specifying the comdition (cond),
    and the sampling step at which this condition is to be replaced by the next one.

    Input:
    (model, ['a red crown', 'a [blue:green:5] jeweled crown'], 20)

    Output:
    [
        [
            ScheduledPromptConditioning(end_at_step=20, cond=tensor([[-0.3886,  0.0229,  ..., -0.4901, -0.3066,  0.0674], ..., [ 0.3317, -0.5102, -0.4066,  ...,  0.4119, -0.7647, -1.0160]], device='cuda:0'))
        ],
        [
            ScheduledPromptConditioning(end_at_step=5, cond=tensor([[-0.3886,  0.0229, ..., -0.4901, -0.3067,  0.0673], ..., [-0.0192,  0.3867, -0.4644,  ...,  0.1135, -0.3696, -0.4625]], device='cuda:0')),
            ScheduledPromptConditioning(end_at_step=20, cond=tensor([[-0.3886,  0.0229, -0.0522,  ..., -0.4901, -0.3067,  0.0673], ..., [-0.7352, -0.4356, -0.7888,  ...,  0.6994, -0.4312, -1.2593]], device='cuda:0'))
        ]
    ]
    """
    res = []

    prompt_schedules = get_learned_conditioning_prompt_schedules(prompts, steps, hires_steps, use_old_scheduling=use_old_scheduling)
    cache = {}

    for prompt, prompt_schedule in zip(prompts, prompt_schedules, strict=False):
        cached = cache.get(prompt)
        if cached is not None:
            res.append(cached)
            continue

        texts = SdConditioning([x[1] for x in prompt_schedule], copy_from=prompts)
        conds = model.get_learned_conditioning(texts)

        cond_schedule = []
        for i, (end_at_step, _) in enumerate(prompt_schedule):
            cond = {k: v[i] for k, v in conds.items()} if isinstance(conds, dict) else conds[i]

            cond_schedule.append(ScheduledPromptConditioning(end_at_step, cond))

        cache[prompt] = cond_schedule
        res.append(cond_schedule)

    return res


re_and = re.compile(r"\bAND\b")
re_weight = re.compile(r"^((?:\s|.)*?)(?:\s*:\s*([-+]?(?:\d+\.?|\d*\.\d+)))?\s*$")


def get_multicond_prompt_list(
    prompts: SdConditioning | list[str],
) -> tuple[list[list[tuple[int, float]]], SdConditioning, dict[str, int]]:
    res_indexes = []

    prompt_indexes = {}
    prompt_flat_list = SdConditioning(prompts)
    prompt_flat_list.clear()

    for prompt in prompts:
        subprompts = re_and.split(prompt)

        indexes = []
        for subprompt in subprompts:
            match = re_weight.search(subprompt)

            text, weight = match.groups() if match is not None else (subprompt, 1.0)

            weight = float(weight) if weight is not None else 1.0

            index = prompt_indexes.get(text)
            if index is None:
                index = len(prompt_flat_list)
                prompt_flat_list.append(text)
                prompt_indexes[text] = index

            indexes.append((index, weight))

        res_indexes.append(indexes)

    return res_indexes, prompt_flat_list, prompt_indexes


@dataclass
class ComposableScheduledPromptConditioning:
    schedules: list[ScheduledPromptConditioning]
    weight: float = 1.0


@dataclass
class MulticondLearnedConditioning:
    shape: tuple
    batch: list[list[ComposableScheduledPromptConditioning]]


def get_multicond_learned_conditioning(
    model: Any,
    prompts: SdConditioning | list[str],
    steps: int,
    hires_steps: int | None = None,
    *,
    use_old_scheduling: bool = False,
) -> MulticondLearnedConditioning:
    """same as get_learned_conditioning, but returns a list of ScheduledPromptConditioning along with the weight objects for each prompt.
    For each prompt, the list is obtained by splitting the prompt using the AND separator.

    https://energy-based-model.github.io/Compositional-Visual-Generation-with-Composable-Diffusion-Models/
    """

    res_indexes, prompt_flat_list, prompt_indexes = get_multicond_prompt_list(prompts)

    learned_conditioning = get_learned_conditioning(model, prompt_flat_list, steps, hires_steps, use_old_scheduling)

    res = [[ComposableScheduledPromptConditioning(learned_conditioning[i], weight) for i, weight in indexes] for indexes in res_indexes]
    return MulticondLearnedConditioning(shape=(len(prompts),), batch=res)


class DictWithShape(dict):
    def __init__(self, x: dict[str, Any]) -> None:
        super().__init__()
        self.update(x)

    @property
    def shape(self) -> tuple:
        return self["crossattn"].shape

    def to(self, *args: Any, **kwargs: Any) -> DictWithShape:
        for k in self.keys():
            if isinstance(self[k], torch.Tensor):
                self[k] = self[k].to(*args, **kwargs)
        return self

    def advanced_indexing(self, item: Any) -> DictWithShape:
        result = {k: self[k][item] for k in self.keys() if isinstance(self[k], torch.Tensor)}
        return DictWithShape(result)


def reconstruct_cond_batch(c: list[list[ScheduledPromptConditioning]], current_step: int) -> torch.Tensor | DictWithShape:
    param = c[0][0].cond
    is_dict = isinstance(param, dict)

    if is_dict:
        dict_cond = param
        res = {k: torch.zeros((len(c), *param.shape), device=param.device, dtype=param.dtype) for k, param in dict_cond.items()}
        res = DictWithShape(res)
    else:
        res = torch.zeros((len(c), *param.shape), device=param.device, dtype=param.dtype)

    for i, cond_schedule in enumerate(c):
        target_index = next(
            (current for current, entry in enumerate(cond_schedule) if current_step <= entry.end_at_step),
            0,
        )
        if is_dict:
            for k, param in cond_schedule[target_index].cond.items():
                res[k][i] = param
        else:
            res[i] = cond_schedule[target_index].cond

    return res


def stack_conds(tensors: list[torch.Tensor]) -> torch.Tensor:
    # if prompts have wildly different lengths above the limit we'll get tensors of different shapes
    # and won't be able to torch.stack them. So this fixes that.
    token_count = max(x.shape[0] for x in tensors)
    for i in range(len(tensors)):
        if tensors[i].shape[0] != token_count:
            last_vector = tensors[i][-1:]
            last_vector_repeated = last_vector.repeat([token_count - tensors[i].shape[0], 1])
            tensors[i] = torch.vstack([tensors[i], last_vector_repeated])

    return torch.stack(tensors)


def reconstruct_multicond_batch(c: MulticondLearnedConditioning, current_step: int) -> tuple[list[list[tuple[int, float]]], torch.Tensor | DictWithShape]:
    param = c.batch[0][0].schedules[0].cond

    tensors = []
    conds_list = []

    for composable_prompts in c.batch:
        conds_for_batch = []

        for composable_prompt in composable_prompts:
            target_index = next(
                (current for current, entry in enumerate(composable_prompt.schedules) if current_step <= entry.end_at_step),
                0,
            )
            conds_for_batch.append((len(tensors), composable_prompt.weight))
            tensors.append(composable_prompt.schedules[target_index].cond)

        conds_list.append(conds_for_batch)

    if isinstance(tensors[0], dict):
        keys = list(tensors[0].keys())
        stacked = {k: stack_conds([x[k] for x in tensors]) for k in keys}
        stacked = DictWithShape(stacked)
    else:
        stacked = stack_conds(tensors).to(device=param.device, dtype=param.dtype)

    return conds_list, stacked


re_attention = re.compile(
    r"""
\\\(|
\\\)|
\\\[|
\\]|
\\\\|
\\|
\(|
\[|
:\s*([+-]?[.\d]+)\s*\)|
\)|
]|
[^\\()\[\]:]+|
:
""",
    re.VERBOSE,
)

re_break = re.compile(r"\s*\bBREAK\b\s*", re.DOTALL)


def parse_prompt_attention(text: str) -> list[list[tuple[str, float]]]:
    """
    Parses a string with attention tokens and returns a list of pairs: text and its associated weight.
    Accepted tokens are:
      (abc) - increases attention to abc by a multiplier of 1.1
      (abc:3.12) - increases attention to abc by a multiplier of 3.12
      [abc] - decreases attention to abc by a multiplier of 1.1
      \\( - literal character '('
      \\[ - literal character '['
      \\) - literal character ')'
      \\] - literal character ']'
      \\ - literal character '\'
      anything else - just text
    """

    res = []
    round_brackets = []
    square_brackets = []

    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position: int, multiplier: float) -> None:
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    for m in re_attention.finditer(text):
        text = m.group(0)
        weight = m.group(1)

        if text.startswith("\\"):
            res.append([text[1:], 1.0])
        elif text == "(":
            round_brackets.append(len(res))
        elif text == "[":
            square_brackets.append(len(res))
        elif weight is not None and round_brackets:
            multiply_range(round_brackets.pop(), float(weight))
        elif text == ")" and round_brackets:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif text == "]" and square_brackets:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            parts = re.split(re_break, text)
            for i, part in enumerate(parts):
                if i > 0:
                    res.append(["BREAK", -1])
                res.append([part, 1.0])

    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)

    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    if not res:
        res = [["", 1.0]]

    # merge runs of identical weights
    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1

    return res
