"""What a benchmark has to provide for the survey and for later evaluation steps.

A benchmark loads its items from the downloaded files, gives each item the one
prompt that every model state will see, and grades a reply with a program. Adding a
job means writing one subclass and registering it in ``benchmarks/__init__.py``.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..store import stable_hash


@dataclass
class Item:
    id: str
    messages: list[dict]  # the prompt, identical for every model state
    answer: Any = None  # what the grader compares against
    group: str = ""  # sub-category, for breakdowns
    cluster: str = ""  # items that must be resampled together (the two orders of a pair)
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.cluster = self.cluster or self.id

    def fingerprint(self) -> str:
        return stable_hash(self.messages if self.messages else self.extra)[:12]


class Benchmark:
    name = ""
    title = ""
    job = ""  # the job from the study design this benchmark stands for
    role = "candidate"  # "target", "candidate" or "watch"
    graded_by = ""
    source = ""  # group name in benchmarks/sources.py
    choices: tuple[str, ...] = ()  # one-token answers whose probabilities are worth keeping
    chance: float | None = None
    macro = False  # score is the mean over groups, not over items
    multi_step = False  # an item is a conversation with tool results, not one reply
    parallel_grading = False  # the grader may run on several replies at once
    reference_size: int | None = None  # size of the effort_0 subsample, if not the survey's default
    reads_tool_calls = False  # calls made in the model's own tool-call format count as the reply
    max_tokens_fast = 256
    max_tokens_slow = 8192

    def load(self, root: Path, seed: int) -> list[Item]:
        raise NotImplementedError

    def grade(self, item: Item, reply: dict) -> dict:
        """Grade one stored reply. Must return at least ``{"score": 0.0 or 1.0}``."""
        raise NotImplementedError

    def extra_metrics(self, items: list[Item], grades: list[dict]) -> dict:
        """Benchmark-specific numbers beyond the score (error types, consistency)."""
        return {}

    def prepare(self, root: Path) -> None:
        """One-time setup before grading (imports of downloaded checkers and so on)."""


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_parquet(path: Path) -> list[dict]:
    import pyarrow.parquet as parquet

    return parquet.read_table(path).to_pylist()


def fixed_sample(items: list, n: int, seed: int) -> list:
    """The same ``n`` items for a given seed, in their original order.

    The items are ranked once by a seeded shuffle and the first ``n`` are taken, so
    a larger sample with the same seed contains every smaller one."""
    if len(items) <= n:
        return list(items)
    order = list(range(len(items)))
    random.Random(seed).shuffle(order)
    chosen = set(order[:n])
    return [item for index, item in enumerate(items) if index in chosen]


def user(text: str) -> list[dict]:
    return [{"role": "user", "content": text}]


_WRAPPING = "*`'\"()[]{}.,:; \n\t"


def parse_choice(text: str, choices: tuple[str, ...]) -> tuple[str | None, bool]:
    """Read a one-word verdict from a reply. Returns (choice, strict).

    ``strict`` is true when the reply is the bare choice and nothing else. Otherwise
    the choice is looked for in the usual phrasings, and failing that at the very
    start of the reply. Matching ignores case and returns the choice as listed.
    """
    lookup = {choice.lower(): choice for choice in choices}
    cleaned = text.strip().strip(_WRAPPING).lower()
    if cleaned in lookup:
        return lookup[cleaned], True
    alternatives = "|".join(re.escape(c) for c in sorted(lookup, key=len, reverse=True))
    # With the answer opening forced, the model sometimes writes its choice twice ("AA").
    repeats = re.findall(alternatives, cleaned)
    if repeats and len(set(repeats)) == 1 and "".join(repeats) == re.sub(r"[\s,.]", "", cleaned):
        return lookup[repeats[0]], False
    phrased = re.findall(
        rf"(?:answer|response|output|option|choice|verdict)\s*(?:is|:)?\s*[*(\[\"']*({alternatives})\b", text, re.IGNORECASE
    )
    if phrased:
        return lookup[phrased[-1].lower()], False
    opening = re.match(rf"\W*({alternatives})\b", text, re.IGNORECASE)
    if opening:
        return lookup[opening.group(1).lower()], False
    return None, False


def strip_code_fence(text: str, language: str = "") -> str:
    """The contents of the first fenced block, or the text itself if there is none."""
    match = re.search(rf"```(?:{language}|\w*)[^\n]*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    opened = re.search(rf"```(?:{language}|\w*)[^\n]*\n(.*)", text, re.DOTALL)  # fence never closed
    return (opened.group(1) if opened else text).strip()
