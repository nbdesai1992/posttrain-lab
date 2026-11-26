"""The interface between the client layer and a model service."""

from __future__ import annotations

import contextvars
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

# The call record of the request being made, so a backend can attach the service
# events it observes (throttling, queue pauses) to the call that caused them.
CURRENT_CALL: contextvars.ContextVar[Any] = contextvars.ContextVar("posttrain_lab_call", default=None)

Example = tuple[list[int], list[float]]  # (tokens, per-token loss weights)


class MissingKeyError(RuntimeError):
    """The API key is not in the environment."""


@dataclass
class RawSample:
    """One sampled sequence as the service returned it."""

    tokens: list[int]
    logprobs: list[float] | None = None
    # For each sampled position, the most likely tokens as (token_id, logprob) pairs.
    top_logprobs: list[list[tuple[int, float]]] | None = None
    stop_reason: str | None = None
    # For each requested prompt position, the logprobs of the requested token IDs.
    target_logprobs: dict[int, list[float]] | None = None
    prompt_cache_hit_tokens: int = 0


@dataclass
class ErrorInfo:
    """How the client should react to an exception from the backend.

    ``kind`` is one of "throttle" (rate limited, retry after a pause), "transient"
    (retry with backoff), "account" (a billing or permission problem, stop everything)
    or "fatal" (the request itself is wrong, do not retry).
    """

    kind: str
    status: int | None = None
    retry_after_s: float | None = None


@dataclass
class ServiceEvent:
    """Something the service did that is not an ordinary response."""

    ts: float
    type: str  # "http_error", "queue_state", "sdk_log"
    detail: dict = field(default_factory=dict)
    call_id: int | None = None

    def to_dict(self) -> dict:
        return {"ts": self.ts, "type": self.type, "call_id": self.call_id, **self.detail}


class Backend(Protocol):
    events: list[ServiceEvent]

    async def sample(
        self,
        model_id: str,
        model_path: str | None,
        prompt_tokens: Sequence[int],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: int | None,
        stop: Sequence[int],
        top_logprobs: int,
        targets: dict[int, list[int]] | None,
    ) -> RawSample: ...

    async def sampler_base_model(self, model_id: str, model_path: str | None) -> str:
        """Create (or reuse) the sampler and return the base model the service reports."""
        ...

    async def create_training_run(self, model_id: str, *, rank: int, seed: int | None) -> Any: ...

    async def forward(self, run: Any, examples: Sequence[Example], *, backward: bool) -> dict: ...

    async def optim_step(self, run: Any, learning_rate: float) -> dict: ...

    async def train_step(self, run: Any, examples: Sequence[Example], learning_rate: float) -> dict: ...

    async def train_pipelined(
        self, run: Any, batches: Sequence[Sequence[Example]], learning_rate: float
    ) -> list[dict]:
        """Several updates, each submitted before the previous one is awaited."""
        ...

    async def save_sampler_weights(self, run: Any, name: str, ttl_seconds: int | None) -> str: ...

    async def checkpoint_info(self, tinker_path: str) -> dict:
        """Base model, size and expiry of a saved checkpoint."""
        ...

    async def client_config(self) -> dict:
        """Limits the service hands to its SDK at session start."""
        ...

    async def account_snapshot(self) -> dict:
        """Checkpoint storage in use and recent metered usage, with no identifiers."""
        ...

    def classify_error(self, exc: BaseException) -> ErrorInfo: ...

    def fatal_condition(self) -> str | None:
        """A reason to stop all work now (for example a billing pause), or None."""
        ...

    async def close(self) -> None: ...
