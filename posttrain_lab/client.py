"""The client layer: the one place that makes model calls.

Every sampling and training call in the package goes through ``ModelClient``. It

* refuses model IDs that are not on the free tier (``guard.check_model_id``),
* bounds how many calls are in flight,
* retries throttled and transient failures with backoff, and stops everything on an
  account-level problem,
* writes one record per call (tokens in and out, thinking tokens, latency, effort,
  errors) to an append-only log, and
* caches sampling responses on disk, keyed by a hash of the model, the rendered
  prompt tokens, the sampling parameters, the effort and the seed, so a repeated
  request is answered from disk.
"""

from __future__ import annotations

import asyncio
import math
import random
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Sequence

from .backends.base import CURRENT_CALL, Backend, Example, RawSample
from .guard import FREE_MODEL_ID, PaidModelError, check_model_id
from .render import (
    DEFAULT_EFFORT,
    FAST_EFFORT,
    Message,
    ParsedResponse,
    Renderer,
    answer_position,
    parse_reply,
)
from .store import ResponseCache, append_jsonl, stable_hash

CACHE_VERSION = 1


class AccountBlockedError(RuntimeError):
    """The service reported a billing, balance or permission problem. Stop and report."""


class CallTimeout(TimeoutError):
    pass


@dataclass
class RetryPolicy:
    max_attempts: int = 5
    base_delay_s: float = 2.0
    max_delay_s: float = 60.0
    # A last resort against a call that never returns, not a latency bound. The SDK
    # queues and retries on its own, and a cancelled request goes to the back of the
    # service's queue when it is sent again, so keep this generous.
    call_timeout_s: float = 1800.0

    def delay(self, attempt: int, retry_after_s: float | None = None) -> float:
        """Seconds to wait before attempt number ``attempt + 1``."""
        backoff = min(self.base_delay_s * (2 ** (attempt - 1)), self.max_delay_s)
        backoff *= 0.75 + 0.5 * random.random()
        if retry_after_s is not None:
            backoff = max(backoff, min(retry_after_s, self.max_delay_s))
        return backoff


@dataclass
class CallRecord:
    """What was sent and what it cost, for one call."""

    call_id: int
    ts: str
    step: str
    tag: str
    kind: str  # sample, score, forward, forward_backward, optim_step, train_step, ...
    model_id: str
    model_path: str | None = None
    effort: float | None = None
    key: str | None = None
    cache_hit: bool = False
    prompt_tokens: int = 0
    cached_prompt_tokens: int = 0  # prompt tokens the service served from its prefix cache
    completion_tokens: int = 0
    thinking_tokens: int = 0
    answer_tokens: int = 0
    max_tokens: int | None = None
    temperature: float | None = None
    seed: int | None = None
    stop_reason: str | None = None
    examples: int | None = None
    started_at: float = 0.0  # epoch seconds, when the call got a concurrency slot
    ended_at: float = 0.0
    queued_s: float = 0.0  # time spent waiting for a slot on this side
    latency_s: float = 0.0  # time from getting a slot to having the result
    attempts: int = 0
    throttle_events: int = 0
    retried_errors: list[str] = field(default_factory=list)
    error_class: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SampleResult:
    tokens: list[int]
    parsed: ParsedResponse
    record: CallRecord
    logprobs: list[float] | None = None
    top_logprobs: list[list[tuple[int, float]]] | None = None
    target_logprobs: dict[int, list[float]] | None = None
    stop_reason: str | None = None

    @property
    def text(self) -> str:
        return self.parsed.text


@dataclass
class ChoiceScores:
    """Probabilities of candidate first answer tokens, read at a forced position."""

    logprobs: dict[str, float]  # candidate -> log probability of its token
    probs: dict[str, float]  # the same, normalized over the candidates
    mass: float  # total probability the candidates hold before normalizing
    p_text_first: float  # probability the reply opens with an answer
    p_thinking_first: float  # probability the reply opens with reasoning
    record: CallRecord


class ModelClient:
    def __init__(
        self,
        backend: Backend,
        renderer: Renderer,
        *,
        model_id: str = FREE_MODEL_ID,
        allow_paid: bool = False,
        cache_dir: Path | str,
        call_log: Path | str | None = None,
        max_concurrency: int = 8,
        retry: RetryPolicy | None = None,
        step: str = "",
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        poll_s: float = 1.0,
    ):
        self.model_id = check_model_id(model_id, allow_paid=allow_paid)
        self.allow_paid = allow_paid
        self.backend = backend
        self.renderer = renderer
        self.cache = ResponseCache(cache_dir)
        self.call_log = Path(call_log) if call_log else None
        self.retry = retry or RetryPolicy()
        self.step = step
        self.totals = {
            "calls": 0,
            "api_calls": 0,
            "cache_hits": 0,
            "errors": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "thinking_tokens": 0,
            "training_tokens": 0,
        }
        self._sleep = sleep
        self._poll_s = poll_s
        self._next_id = 0
        self._checked_paths: set[str] = set()
        self.set_concurrency(max_concurrency)

    # ------------------------------------------------------------ concurrency

    def set_concurrency(self, limit: int) -> None:
        """Change how many calls may be in flight. Call it between batches of work."""
        if limit < 1:
            raise ValueError("Concurrency must be at least 1")
        self.max_concurrency = limit
        self._semaphore = asyncio.Semaphore(limit)

    async def warm_up(self) -> str:
        """Create the base-model sampler before any timed work, and check that the
        service reports a free base model for it. Returns the reported base model."""
        base_model = await self.backend.sampler_base_model(self.model_id, None)
        check_model_id(base_model, allow_paid=self.allow_paid)
        return base_model

    # ---------------------------------------------------------------- records

    def _new_record(self, kind: str, tag: str, **fields: Any) -> CallRecord:
        self._next_id += 1
        return CallRecord(
            call_id=self._next_id,
            ts=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            step=self.step,
            tag=tag,
            kind=kind,
            model_id=self.model_id,
            **fields,
        )

    def _finish(self, record: CallRecord) -> None:
        self.totals["calls"] += 1
        if record.cache_hit:
            self.totals["cache_hits"] += 1
        elif record.error_class:
            self.totals["errors"] += 1
        else:
            self.totals["api_calls"] += 1
            if record.kind in ("sample", "score"):
                self.totals["prompt_tokens"] += record.prompt_tokens
                self.totals["completion_tokens"] += record.completion_tokens
                self.totals["thinking_tokens"] += record.thinking_tokens
            elif record.kind in ("forward", "forward_backward", "train_step", "train_pipelined"):
                self.totals["training_tokens"] += record.prompt_tokens
        if self.call_log is not None:
            append_jsonl(self.call_log, record.to_dict())

    # ------------------------------------------------------------- call logic

    async def _attempt(self, make_call: Callable[[], Awaitable[Any]]) -> Any:
        """Run one attempt, watching for a stop condition and a time limit."""
        task = asyncio.ensure_future(make_call())
        deadline = time.monotonic() + self.retry.call_timeout_s
        try:
            while True:
                done, _ = await asyncio.wait({task}, timeout=self._poll_s)
                if done:
                    return task.result()
                reason = self.backend.fatal_condition()
                if reason:
                    raise AccountBlockedError(reason)
                if time.monotonic() > deadline:
                    raise CallTimeout(f"No result after {self.retry.call_timeout_s:.0f} s")
        finally:
            if not task.done():
                task.cancel()

    async def _run(self, record: CallRecord, make_call: Callable[[], Awaitable[Any]]) -> Any:
        """Make one logical call: wait for a slot, try, back off and retry if warranted.

        A call that fails for good is logged here. A call that succeeds is logged by
        the caller, once it has filled in what came back.
        """
        try:
            return await self._run_with_retries(record, make_call)
        except BaseException as exc:
            record.error_class = type(exc).__name__
            record.error_message = str(exc)[:500]
            self._finish(record)
            raise

    async def _run_with_retries(self, record: CallRecord, make_call: Callable[[], Awaitable[Any]]) -> Any:
        queued = time.monotonic()
        async with self._semaphore:
            started = time.monotonic()
            record.queued_s = started - queued
            record.started_at = time.time()
            token = CURRENT_CALL.set(record)
            try:
                while True:
                    record.attempts += 1
                    try:
                        return await self._attempt(make_call)
                    except AccountBlockedError:
                        raise
                    except Exception as exc:
                        if isinstance(exc, CallTimeout):
                            kind, retry_after = "transient", None
                        else:
                            info = self.backend.classify_error(exc)
                            kind, retry_after = info.kind, info.retry_after_s
                        if kind == "account":
                            raise AccountBlockedError(f"{type(exc).__name__}: {exc}") from exc
                        if kind == "fatal" or record.attempts >= self.retry.max_attempts:
                            raise
                        if kind == "throttle":
                            record.throttle_events += 1
                        record.retried_errors.append(type(exc).__name__)
                        await self._sleep(self.retry.delay(record.attempts, retry_after))
            finally:
                CURRENT_CALL.reset(token)
                record.latency_s = time.monotonic() - started
                record.ended_at = time.time()

    # --------------------------------------------------------------- sampling

    def cache_key(
        self,
        prompt_tokens: Sequence[int],
        *,
        effort: float | None,
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: int | None,
        top_logprobs: int,
        targets: dict[int, list[int]] | None,
        model_path: str | None,
        salt: str | None,
    ) -> str:
        return stable_hash(
            {
                "v": CACHE_VERSION,
                "model_id": self.model_id,
                "model_path": model_path,
                "prompt": list(prompt_tokens),
                "effort": effort,
                "params": {
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "seed": seed,
                    "stop": list(self.renderer.stop_tokens),
                    "top_logprobs": top_logprobs,
                },
                "targets": sorted((int(p), list(ids)) for p, ids in (targets or {}).items()),
                "salt": salt,
            }
        )

    async def sample(
        self,
        messages: Sequence[Message],
        *,
        effort: float = DEFAULT_EFFORT,
        max_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int | None = None,
        top_logprobs: int = 0,
        force_answer: bool = False,
        model_path: str | None = None,
        salt: str | None = None,
        tag: str = "",
    ) -> SampleResult:
        """Sample one reply to a conversation.

        ``force_answer`` opens the reply as a text answer on the model's behalf, so it
        cannot reason first. Effort 0.0 alone does not stop the base model reasoning.

        ``salt`` is an extra cache-key ingredient. Timing probes pass a fresh one so
        that they always reach the service; everything else leaves it unset.
        """
        prompt = self.renderer.build_prompt(messages, effort)
        opening: list[int] = []
        if force_answer:
            special = self.renderer.special
            opening = [special.message_model, special.content_text]
        return await self.sample_tokens(
            [*prompt, *opening],
            forced_opening=opening,
            effort=effort,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            top_logprobs=top_logprobs,
            model_path=model_path,
            salt=salt,
            tag=tag,
        )

    async def sample_tokens(
        self,
        prompt_tokens: Sequence[int],
        *,
        effort: float | None,
        max_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int | None = None,
        top_logprobs: int = 0,
        targets: dict[int, list[int]] | None = None,
        forced_opening: Sequence[int] = (),
        model_path: str | None = None,
        salt: str | None = None,
        tag: str = "",
        kind: str = "sample",
    ) -> SampleResult:
        """Sample from already rendered prompt tokens. ``sample`` is the usual entry.

        ``forced_opening`` names the reply tokens the prompt already ends with, so the
        reply can be parsed as a whole."""
        if model_path is not None:
            await self._check_adapter(model_path)
        key = self.cache_key(
            prompt_tokens,
            effort=effort,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            top_logprobs=top_logprobs,
            targets=targets,
            model_path=model_path,
            salt=salt,
        )
        record = self._new_record(
            kind,
            tag,
            model_path=model_path,
            effort=effort,
            key=key,
            prompt_tokens=len(prompt_tokens),
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
        )
        cached = self.cache.get(key)
        if cached is not None:
            record.cache_hit = True
            raw = RawSample(
                tokens=cached["tokens"],
                logprobs=cached.get("logprobs"),
                top_logprobs=[[tuple(pair) for pair in row] for row in cached["top_logprobs"]]
                if cached.get("top_logprobs") is not None
                else None,
                stop_reason=cached.get("stop_reason"),
                target_logprobs={int(p): v for p, v in cached["target_logprobs"].items()}
                if cached.get("target_logprobs") is not None
                else None,
                prompt_cache_hit_tokens=cached.get("prompt_cache_hit_tokens", 0),
            )
        else:
            raw = await self._run(
                record,
                lambda: self.backend.sample(
                    self.model_id,
                    model_path,
                    prompt_tokens,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    seed=seed,
                    stop=self.renderer.stop_tokens,
                    top_logprobs=top_logprobs,
                    targets=targets,
                ),
            )
            self.cache.put(
                key,
                {
                    "ts": record.ts,
                    "kind": kind,
                    "model_id": self.model_id,
                    "model_path": model_path,
                    "effort": effort,
                    "n_prompt_tokens": len(prompt_tokens),
                    "tokens": raw.tokens,
                    "logprobs": raw.logprobs,
                    "top_logprobs": raw.top_logprobs,
                    "stop_reason": raw.stop_reason,
                    "target_logprobs": {str(p): v for p, v in raw.target_logprobs.items()}
                    if raw.target_logprobs is not None
                    else None,
                    "prompt_cache_hit_tokens": raw.prompt_cache_hit_tokens,
                },
            )
        parsed = parse_reply(raw.tokens, self.renderer.special, self.renderer.decode, forced_opening)
        record.completion_tokens = len(raw.tokens)
        record.cached_prompt_tokens = raw.prompt_cache_hit_tokens
        record.thinking_tokens = parsed.n_thinking_tokens
        record.answer_tokens = parsed.n_text_tokens
        record.stop_reason = raw.stop_reason
        self._finish(record)
        return SampleResult(
            tokens=raw.tokens,
            parsed=parsed,
            record=record,
            logprobs=raw.logprobs,
            top_logprobs=raw.top_logprobs,
            target_logprobs=raw.target_logprobs,
            stop_reason=raw.stop_reason,
        )

    async def score_choices(
        self,
        messages: Sequence[Message],
        choices: Sequence[str],
        *,
        effort: float = FAST_EFFORT,
        model_path: str | None = None,
        salt: str | None = None,
        tag: str = "",
    ) -> ChoiceScores:
        """Read the probability of each candidate answer at the first answer position.

        The conversation is rendered with the reply already opened as a text answer,
        so the model's choice to reason first cannot get in the way. The same request
        also reads how likely the reply was to open with reasoning instead. Each
        candidate must be a single token. One request, one generated token.
        """
        special = self.renderer.special
        choice_ids: dict[str, int] = {}
        for choice in choices:
            ids = self.renderer.encode(choice)
            if len(ids) != 1:
                raise ValueError(f"Candidate {choice!r} is {len(ids)} tokens; it must be exactly one")
            choice_ids[choice] = ids[0]
        example, _ = self.renderer.build_example(
            [*messages, {"role": "assistant", "content": choices[0]}], effort
        )
        position = answer_position(example, special)
        prompt = example[: position + 1]
        targets = {
            position: [choice_ids[c] for c in choices],
            position - 1: [special.content_text, special.content_thinking],
        }
        result = await self.sample_tokens(
            prompt,
            effort=effort,
            max_tokens=1,
            targets=targets,
            model_path=model_path,
            salt=salt,
            tag=tag,
            kind="score",
        )
        assert result.target_logprobs is not None
        logprobs = dict(zip(choices, result.target_logprobs[position]))
        mass = sum(math.exp(lp) for lp in logprobs.values())
        text_lp, thinking_lp = result.target_logprobs[position - 1]
        return ChoiceScores(
            logprobs=logprobs,
            probs={c: (math.exp(lp) / mass if mass > 0 else 0.0) for c, lp in logprobs.items()},
            mass=mass,
            p_text_first=math.exp(text_lp),
            p_thinking_first=math.exp(thinking_lp),
            record=result.record,
        )

    async def _check_adapter(self, model_path: str) -> None:
        """An adapter may be sampled only if the service says its base is a free ID."""
        if model_path in self._checked_paths:
            return
        base_model = await self.backend.sampler_base_model(self.model_id, model_path)
        try:
            check_model_id(base_model, allow_paid=self.allow_paid)
        except PaidModelError as exc:
            raise PaidModelError(f"Adapter {model_path} is built on {base_model!r}. {exc}") from None
        self._checked_paths.add(model_path)

    async def load_adapter(self, model_path: str, tag: str = "") -> CallRecord:
        """Create the sampler for a saved adapter and check its base model. Timed."""
        record = self._new_record("create_sampling_client", tag, model_path=model_path)
        await self._run(record, lambda: self._check_adapter(model_path))
        self._finish(record)
        return record

    # --------------------------------------------------------------- training

    async def create_training_run(self, *, rank: int = 16, seed: int | None = None, tag: str = "") -> "TrainingRun":
        record = self._new_record("create_training_client", tag, seed=seed)
        handle = await self._run(
            record, lambda: self.backend.create_training_run(self.model_id, rank=rank, seed=seed)
        )
        self._finish(record)
        return TrainingRun(self, handle, rank, record)

    async def _training_call(
        self, kind: str, tag: str, examples: Sequence[Example] | None, make_call: Callable[[], Awaitable[Any]]
    ) -> tuple[Any, CallRecord]:
        record = self._new_record(kind, tag)
        if examples is not None:
            record.examples = len(examples)
            record.prompt_tokens = sum(len(tokens) for tokens, _ in examples)
        result = await self._run(record, make_call)
        self._finish(record)
        return result, record


class TrainingRun:
    """A LoRA training run on the free model. Calls are logged but never cached."""

    def __init__(self, client: ModelClient, handle: Any, rank: int, created: CallRecord):
        self._client = client
        self._handle = handle
        self.rank = rank
        self.created = created

    async def forward(self, examples: Sequence[Example], tag: str = "") -> tuple[dict, CallRecord]:
        """Loss on a batch without touching the weights."""
        backend = self._client.backend
        return await self._client._training_call(
            "forward", tag, examples, lambda: backend.forward(self._handle, examples, backward=False)
        )

    async def forward_backward(self, examples: Sequence[Example], tag: str = "") -> tuple[dict, CallRecord]:
        """Loss and gradients for a batch. Follow with ``optim_step``."""
        backend = self._client.backend
        return await self._client._training_call(
            "forward_backward", tag, examples, lambda: backend.forward(self._handle, examples, backward=True)
        )

    async def optim_step(self, learning_rate: float, tag: str = "") -> tuple[dict, CallRecord]:
        backend = self._client.backend
        return await self._client._training_call(
            "optim_step", tag, None, lambda: backend.optim_step(self._handle, learning_rate)
        )

    async def train_step(
        self, examples: Sequence[Example], learning_rate: float, tag: str = ""
    ) -> tuple[dict, CallRecord]:
        """One update with the gradient and optimizer requests sent back to back."""
        backend = self._client.backend
        return await self._client._training_call(
            "train_step", tag, examples, lambda: backend.train_step(self._handle, examples, learning_rate)
        )

    async def train_pipelined(
        self, batches: Sequence[Sequence[Example]], learning_rate: float | Sequence[float], tag: str = ""
    ) -> tuple[list[dict], CallRecord]:
        """Several updates in a row, each submitted before the previous one is awaited.
        ``learning_rate`` is one value for all of them or one value per batch.

        This is the fastest way to train: the service works in fixed cycles, and a
        loop that waits for each result before sending the next batch misses cycles.
        Logged as one call covering all the batches.
        """
        backend = self._client.backend
        flat = [example for examples in batches for example in examples]
        return await self._client._training_call(
            "train_pipelined", tag, flat, lambda: backend.train_pipelined(self._handle, batches, learning_rate)
        )

    async def save_sampler_weights(
        self, name: str, ttl_seconds: int | None, tag: str = ""
    ) -> tuple[str, CallRecord]:
        """Save the adapter for sampling. Returns its ``tinker://`` path."""
        backend = self._client.backend
        return await self._client._training_call(
            "save_sampler_weights", tag, None, lambda: backend.save_sampler_weights(self._handle, name, ttl_seconds)
        )
