"""Prompt rendering and response parsing, independent of any SDK.

The model's chat format frames every message with special tokens. A reply looks like

    <message_model> <content_thinking> ...reasoning... <end_message>
    <message_model> <content_text> ...answer... <end_message> <end_sampling>

where the thinking message is optional. ``segment_response`` splits a sampled token
sequence along that framing, which is how thinking tokens are counted separately from
answer tokens and how the position of the first answer token is found.

``Renderer`` is the small interface the rest of the package uses. The real one wraps
the Tinker cookbook renderer (see ``backends/tinker_backend.py``); tests use a fake.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Protocol, Sequence

DEFAULT_EFFORT = 0.9  # the renderer's default thinking effort ("high")
FAST_EFFORT = 0.0  # conditions the model toward no reasoning

Message = dict  # {"role": "system" | "user" | "assistant", "content": str}


@dataclass(frozen=True)
class SpecialTokens:
    """Token IDs of the framing tokens in a model reply."""

    message_model: int
    content_thinking: int
    content_text: int
    end_message: int
    end_sampling: int


@dataclass
class ParsedResponse:
    """A sampled reply split into thinking and answer parts."""

    text: str = ""
    thinking: str = ""
    n_tokens: int = 0
    n_thinking_tokens: int = 0  # content tokens inside thinking messages
    n_text_tokens: int = 0  # content tokens inside text messages
    n_framing_tokens: int = 0  # special tokens around the content
    first_kind: str | None = None  # "thinking" or "text": what the reply opened with
    first_text_index: int | None = None  # index of the first answer token in the reply
    ended: bool = False  # the model closed its turn (it was not cut off by max_tokens)
    well_formed: bool = True  # the framing was as expected
    text_token_ids: list[int] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict:
        data = asdict(self)
        data.pop("text_token_ids")
        return data


class Renderer(Protocol):
    """What the client layer needs from a chat template."""

    special: SpecialTokens
    stop_tokens: list[int]

    def build_prompt(self, messages: Sequence[Message], effort: float) -> list[int]:
        """Tokens of the conversation, ready for the model to start its reply."""
        ...

    def build_example(self, messages: Sequence[Message], effort: float) -> tuple[list[int], list[float]]:
        """Tokens of a full conversation ending in an assistant reply, with per-token
        loss weights (1.0 on the reply, 0.0 on everything before it)."""
        ...

    def encode(self, text: str) -> list[int]: ...

    def decode(self, tokens: Sequence[int]) -> str: ...


def segment_response(tokens: Sequence[int], special: SpecialTokens, decode) -> ParsedResponse:
    """Split sampled tokens into thinking and answer parts by their framing tokens."""
    parsed = ParsedResponse(n_tokens=len(tokens))
    thinking_ids: list[int] = []
    kind: str | None = None  # the kind of message currently open
    expect_kind = False  # the previous token opened a model message
    for index, token in enumerate(tokens):
        if token == special.end_sampling:
            parsed.ended = True
            parsed.n_framing_tokens += 1
            if kind is not None:
                parsed.well_formed = False
            kind = None
            continue
        if token == special.message_model:
            parsed.n_framing_tokens += 1
            if kind is not None:
                parsed.well_formed = False
            kind = None
            expect_kind = True
            continue
        if expect_kind:
            expect_kind = False
            if token == special.content_thinking:
                kind = "thinking"
            elif token == special.content_text:
                kind = "text"
            else:
                # An unknown content type (a tool call, for example). Count it as framing.
                kind = "other"
                parsed.well_formed = False
            parsed.n_framing_tokens += 1
            if parsed.first_kind is None:
                parsed.first_kind = kind
            continue
        if token == special.end_message:
            parsed.n_framing_tokens += 1
            kind = None
            continue
        if kind == "thinking":
            thinking_ids.append(token)
        elif kind == "text":
            if parsed.first_text_index is None:
                parsed.first_text_index = index
            parsed.text_token_ids.append(token)
        else:
            # Content outside any message: the framing is not what we expect.
            parsed.well_formed = False
            parsed.n_framing_tokens += 1
    parsed.n_thinking_tokens = len(thinking_ids)
    parsed.n_text_tokens = len(parsed.text_token_ids)
    parsed.thinking = decode(thinking_ids) if thinking_ids else ""
    parsed.text = decode(parsed.text_token_ids) if parsed.text_token_ids else ""
    return parsed


def parse_reply(
    tokens: Sequence[int], special: SpecialTokens, decode, forced_opening: Sequence[int] = ()
) -> ParsedResponse:
    """Parse sampled tokens. If the prompt ended with a forced opening of the reply
    (see ``ModelClient.sample``), the opening is put back for parsing and then left
    out of the counts, which describe only what the model generated."""
    parsed = segment_response([*forced_opening, *tokens], special, decode)
    skip = len(forced_opening)
    if skip:
        parsed.n_tokens -= skip
        parsed.n_framing_tokens -= skip
        if parsed.first_text_index is not None:
            parsed.first_text_index -= skip
    return parsed


def answer_position(example_tokens: Sequence[int], special: SpecialTokens) -> int:
    """Index of the first answer token in a rendered conversation that ends with an
    assistant text reply. Used to score candidate answer tokens at that position."""
    for index in range(len(example_tokens) - 2, 0, -1):
        if (
            example_tokens[index] == special.content_text
            and example_tokens[index - 1] == special.message_model
        ):
            return index + 1
    raise ValueError("The rendered conversation has no assistant text reply")
