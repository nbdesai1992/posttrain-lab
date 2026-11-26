"""A fake renderer and backend, so everything above the SDK can run offline.

The tests use these, and they are handy when developing a new step: the whole
step runs in milliseconds and no request leaves the machine.
"""

from __future__ import annotations

import asyncio
import math
from typing import Any, Callable, Sequence

from .backends.base import ErrorInfo, Example, RawSample, ServiceEvent
from .render import Message, SpecialTokens

MESSAGE_MODEL, CONTENT_THINKING, CONTENT_TEXT, END_MESSAGE, END_SAMPLING = 1, 2, 3, 4, 5
MESSAGE_SYSTEM, MESSAGE_USER = 6, 7
FIRST_WORD_ID = 100


class FakeThrottle(Exception):
    pass


class FakeTransient(Exception):
    pass


class FakeAccountProblem(Exception):
    pass


class FakeRenderer:
    """Whitespace tokenizer with the same message framing as the real chat template."""

    special = SpecialTokens(
        message_model=MESSAGE_MODEL,
        content_thinking=CONTENT_THINKING,
        content_text=CONTENT_TEXT,
        end_message=END_MESSAGE,
        end_sampling=END_SAMPLING,
    )
    stop_tokens = [END_SAMPLING]

    def __init__(self) -> None:
        self._ids: dict[str, int] = {}
        self._words: dict[int, str] = {}

    def encode(self, text: str) -> list[int]:
        ids = []
        for word in text.split():
            if word not in self._ids:
                token = FIRST_WORD_ID + len(self._ids)
                self._ids[word] = token
                self._words[token] = word
            ids.append(self._ids[word])
        return ids

    def decode(self, tokens: Sequence[int]) -> str:
        return " ".join(self._words.get(t, f"<{t}>") for t in tokens)

    def build_prompt(self, messages: Sequence[Message], effort: float) -> list[int]:
        tokens: list[int] = []
        system = [m for m in messages if m["role"] == "system"]
        for message in system:
            tokens += [MESSAGE_SYSTEM, CONTENT_TEXT, *self.encode(message["content"]), END_MESSAGE]
        tokens += [MESSAGE_SYSTEM, CONTENT_TEXT, *self.encode(f"effort {effort}"), END_MESSAGE]
        for message in messages:
            if message["role"] == "user":
                tokens += [MESSAGE_USER, CONTENT_TEXT, *self.encode(message["content"]), END_MESSAGE]
        return tokens

    def build_example(self, messages: Sequence[Message], effort: float) -> tuple[list[int], list[float]]:
        prompt = self.build_prompt(messages[:-1], effort)
        reply = [MESSAGE_MODEL, CONTENT_TEXT, *self.encode(messages[-1]["content"]), END_MESSAGE, END_SAMPLING]
        return prompt + reply, [0.0] * len(prompt) + [1.0] * len(reply)

    def effort_of(self, prompt_tokens: Sequence[int]) -> float:
        marker = self._ids.get("effort")
        for index, token in enumerate(prompt_tokens):
            if token == marker and index + 1 < len(prompt_tokens):
                try:
                    return float(self._words[prompt_tokens[index + 1]])
                except (KeyError, ValueError):
                    pass
        return 0.0


class FakeBackend:
    """Answers every prompt locally. Thinks first whenever the effort is above zero."""

    def __init__(
        self,
        renderer: FakeRenderer,
        *,
        answer: Callable[[str], str] | str = "yes",
        forced_answer: Callable[[str], str] | str | None = None,
        thinking_words: int = 6,
        latency_s: float = 0.0,
        failures: Sequence[BaseException | None] = (),
    ):
        self.renderer = renderer
        self.answer = answer
        self.forced_answer = forced_answer  # what to say when the answer opening is forced, if different
        self.thinking_words = thinking_words
        self.latency_s = latency_s
        self.failures = list(failures)  # raised one per call, in order; None means succeed
        self.events: list[ServiceEvent] = []
        self.sample_calls: list[dict] = []
        self.training_calls: list[str] = []
        self.in_flight = 0
        self.max_in_flight = 0
        self.fatal: str | None = None
        self.adapter_base_model: str | None = None

    async def _enter(self) -> None:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(self.latency_s)
        finally:
            self.in_flight -= 1

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
    ) -> RawSample:
        self.sample_calls.append({"model_id": model_id, "model_path": model_path, "prompt": list(prompt_tokens)})
        await self._enter()
        if self.failures:
            failure = self.failures.pop(0)
            if failure is not None:
                raise failure
        prompt_text = self.renderer.decode([t for t in prompt_tokens if t >= FIRST_WORD_ID])
        answer_forced = list(prompt_tokens[-2:]) == [MESSAGE_MODEL, CONTENT_TEXT]
        source = self.forced_answer if (answer_forced and self.forced_answer is not None) else self.answer
        answer = source(prompt_text) if callable(source) else source
        tokens: list[int] = []
        if answer_forced:
            tokens += [*self.renderer.encode(answer), END_MESSAGE, END_SAMPLING]
        else:
            if self.renderer.effort_of(prompt_tokens) > 0 and not targets:
                thought = self.renderer.encode(" ".join(["hmm"] * self.thinking_words))
                tokens += [MESSAGE_MODEL, CONTENT_THINKING, *thought, END_MESSAGE]
            tokens += [MESSAGE_MODEL, CONTENT_TEXT, *self.renderer.encode(answer), END_MESSAGE, END_SAMPLING]
        stop_reason = "stop"
        if len(tokens) > max_tokens:
            tokens, stop_reason = tokens[:max_tokens], "length"
        top = None
        if top_logprobs:
            top = [[(t, math.log(0.9)), (t + 1, math.log(0.1))][:top_logprobs] for t in tokens]
        target_logprobs = None
        if targets:
            # The first requested token gets 0.7 and the rest share 0.3.
            target_logprobs = {
                position: [math.log(0.7 if i == 0 else 0.3 / max(len(ids) - 1, 1)) for i in range(len(ids))]
                for position, ids in targets.items()
            }
        return RawSample(
            tokens=tokens,
            logprobs=[math.log(0.9)] * len(tokens),
            top_logprobs=top,
            stop_reason=stop_reason,
            target_logprobs=target_logprobs,
        )

    async def sampler_base_model(self, model_id: str, model_path: str | None) -> str:
        return self.adapter_base_model or model_id

    async def create_training_run(self, model_id: str, *, rank: int, seed: int | None) -> Any:
        self.training_calls.append("create")
        return {"model_id": model_id, "rank": rank, "steps": 0}

    async def forward(self, run: Any, examples: Sequence[Example], *, backward: bool) -> dict:
        self.training_calls.append("forward_backward" if backward else "forward")
        await self._enter()
        weight = sum(sum(weights) for _, weights in examples)
        return {"metrics": {"loss:sum": 2.0 * weight / (1 + run["steps"])}}

    async def optim_step(self, run: Any, learning_rate: float) -> dict:
        self.training_calls.append("optim_step")
        run["steps"] += 1
        return {"metrics": {}}

    async def train_step(self, run: Any, examples: Sequence[Example], learning_rate: float) -> dict:
        result = await self.forward(run, examples, backward=True)
        await self.optim_step(run, learning_rate)
        return {**result, "optim_metrics": {}, "forward_backward_s": 0.0, "total_s": 0.0}

    async def train_pipelined(self, run: Any, batches: Sequence[Sequence[Example]], learning_rate: float) -> list[dict]:
        results = []
        for index, examples in enumerate(batches):
            step = await self.train_step(run, examples, learning_rate)
            results.append({"metrics": step["metrics"], "seconds": 0.5, "elapsed_s": 0.5 * (index + 1)})
        return results

    async def save_sampler_weights(self, run: Any, name: str, ttl_seconds: int | None) -> str:
        self.training_calls.append("save")
        return f"tinker://fake-run/sampler_weights/{name}"

    async def checkpoint_info(self, tinker_path: str) -> dict:
        return {"base_model": self.adapter_base_model, "size_bytes": 1024, "expires_at": None}

    async def client_config(self) -> dict:
        return {}

    async def account_snapshot(self) -> dict:
        return {
            "storage_rate_usd_per_gb_month": 0.0,
            "checkpoint_count": 0,
            "checkpoint_bytes": 0,
            "checkpoints_listed": 0,
            "checkpoints_listed_bytes": 0,
            "usage_estimated_cost_usd": 0.0,
            "usage_by_model_and_type": {},
        }

    def classify_error(self, exc: BaseException) -> ErrorInfo:
        if isinstance(exc, FakeThrottle):
            return ErrorInfo("throttle", 429, 0.0)
        if isinstance(exc, FakeTransient):
            return ErrorInfo("transient", 503)
        if isinstance(exc, FakeAccountProblem):
            return ErrorInfo("account", 402)
        return ErrorInfo("fatal")

    def fatal_condition(self) -> str | None:
        return self.fatal

    async def close(self) -> None:
        pass
