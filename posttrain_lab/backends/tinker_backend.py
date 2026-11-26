"""Tinker backend. This is the only module that imports the Tinker SDK.

Two things here are worth knowing before changing it.

The SDK hides throttling. A rate-limited submit (HTTP 429) is swallowed and resent
after a second, a billing problem (HTTP 402) is retried silently for up to an hour,
and a queued request reports its state only to an observer object. So this backend
installs three small probes: one around the submit call, one on the queue-state
observers, and one on the SDK's logger. What they see lands in ``self.events``.

The key is read from the environment by the SDK. This module checks that it is set,
never reads it from a file, and strips it from any text it records.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import re
import time
import warnings
from datetime import datetime, timedelta, timezone
from typing import Any, Sequence

from ..guard import check_model_id
from ..render import Message, SpecialTokens
from .base import CURRENT_CALL, ErrorInfo, Example, MissingKeyError, RawSample, ServiceEvent

KEY_ENV = "TINKER_API_KEY"
MAX_EVENTS = 20_000
RATE_HEADER_PREFIXES = ("retry-after", "x-ratelimit", "ratelimit", "x-rate-limit")
BILLING_PAUSE_MARKER = "paused due to billing status"
_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE)


def require_key() -> None:
    if not os.environ.get(KEY_ENV):
        raise MissingKeyError(
            f"{KEY_ENV} is not set. Export a Tinker API key in your shell first, for example "
            f"`export {KEY_ENV}=\"$(cat ~/.config/tinker/api_key)\"`."
        )


async def _resolve(value: Any) -> Any:
    """SDK calls return objects that are awaitable, future-like, or plain values."""
    if inspect.isawaitable(value):
        return await value
    if hasattr(value, "result"):
        return await asyncio.to_thread(value.result)
    return value


def _quiet_imports() -> None:
    # The cookbook and its dependencies warn at import time about optional extras.
    warnings.filterwarnings("ignore", module=r"(tinker_cookbook|transformers|torch|pydantic)(\..*)?")


class InklingRenderer:
    """The cookbook's renderer for Inkling models behind the package's small interface."""

    def __init__(self, model_id: str):
        _quiet_imports()
        from tinker_cookbook import model_info
        from tinker_cookbook.renderers import get_renderer
        from tinker_cookbook.tokenizer_utils import get_tokenizer

        self.model_id = model_id
        self._tokenizer = get_tokenizer(model_id)
        self._renderer = get_renderer(model_info.get_recommended_renderer_name(model_id), self._tokenizer)
        self.stop_tokens: list[int] = [int(t) for t in self._renderer.get_stop_sequences()]
        self.special = self._find_special_tokens()

    def build_prompt(self, messages: Sequence[Message], effort: float) -> list[int]:
        return self._renderer.build_generation_prompt(list(messages), effort=effort).to_ints()

    def build_example(self, messages: Sequence[Message], effort: float) -> tuple[list[int], list[float]]:
        model_input, weights = self._renderer.build_supervised_example(list(messages), effort=effort)
        return model_input.to_ints(), [float(w) for w in weights]

    def encode(self, text: str) -> list[int]:
        return list(self._tokenizer.encode(text, add_special_tokens=False))

    def decode(self, tokens: Sequence[int]) -> str:
        decoded = self._tokenizer.decode(list(tokens))
        return decoded if isinstance(decoded, str) else "".join(decoded)

    def _find_token(self, name: str) -> int:
        """The ID of a framing token by its name, or -1 if this template has none."""
        for token in range(199990, 200600):
            try:
                if self.decode([token]) == name:
                    return token
            except Exception:
                continue
        return -1

    def _find_special_tokens(self) -> SpecialTokens:
        reference = [
            {"role": "user", "content": "x"},
            {
                "role": "assistant",
                "content": [{"type": "thinking", "thinking": "y"}, {"type": "text", "text": "z"}],
            },
        ]
        tokens, _ = self.build_example(reference, 0.9)
        names = {self.decode([token]): token for token in set(tokens)}
        try:
            return SpecialTokens(
                message_model=names["<|message_model|>"],
                content_thinking=names["<|content_thinking|>"],
                content_text=names["<|content_text|>"],
                end_message=names["<|end_message|>"],
                end_sampling=names["<|content_model_end_sampling|>"],
                content_tool_call=self._find_token("<|content_invoke_tool_json|>"),
            )
        except KeyError as missing:
            raise RuntimeError(
                f"The chat template for {self.model_id} has no {missing} token; "
                "the response parser needs updating for this renderer version."
            ) from None


class _SdkLogCapture(logging.Handler):
    def __init__(self, backend: "TinkerBackend"):
        super().__init__(level=logging.WARNING)
        self._backend = backend

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
        except Exception:
            return
        self._backend._on_sdk_log(record.name, record.levelname, message)


_PROBE_SINKS: list["TinkerBackend"] = []
_PROBE_INSTALLED = False


def _install_submit_probe() -> bool:
    """Wrap the SDK's submit call so HTTP errors it would swallow are still recorded."""
    global _PROBE_INSTALLED
    if _PROBE_INSTALLED:
        return True
    try:
        import tinker
        from tinker.resources.sampling import AsyncSamplingResource

        original = AsyncSamplingResource.asample

        async def asample(resource, **kwargs):
            try:
                return await original(resource, **kwargs)
            except tinker.APIStatusError as exc:
                for sink in list(_PROBE_SINKS):
                    sink._on_http_error("sample_submit", exc)
                raise

        AsyncSamplingResource.asample = asample
        _PROBE_INSTALLED = True
    except Exception:
        _PROBE_INSTALLED = False
    return _PROBE_INSTALLED


class TinkerBackend:
    def __init__(self, *, allow_paid: bool = False):
        require_key()
        _quiet_imports()
        self.allow_paid = allow_paid
        self.events: list[ServiceEvent] = []
        self.dropped_events = 0
        self._secret = os.environ[KEY_ENV]
        self._fatal: str | None = None
        self._service: Any = None
        self._rest: Any = None
        self._admin: Any = None
        self._samplers: dict[tuple[str, str | None], Any] = {}
        self._lock = asyncio.Lock()
        self._log_capture = _SdkLogCapture(self)
        logging.getLogger("tinker").addHandler(self._log_capture)
        _PROBE_SINKS.append(self)
        self.submit_probe_installed = _install_submit_probe()

    # ------------------------------------------------------------------ probes

    def _redact(self, text: Any) -> str:
        """Strip the key, and session or account IDs, from text that will be stored."""
        text = str(text)
        if self._secret:
            text = text.replace(self._secret, "[redacted]")
        return _UUID.sub("<id>", text)

    def _record(self, kind: str, detail: dict, *, throttle: bool = False) -> None:
        record = CURRENT_CALL.get()
        if throttle and record is not None:
            record.throttle_events += 1
        if len(self.events) >= MAX_EVENTS:
            self.dropped_events += 1
            return
        call_id = getattr(record, "call_id", None)
        self.events.append(ServiceEvent(ts=time.time(), type=kind, detail=detail, call_id=call_id))

    def _on_sdk_log(self, logger_name: str, level: str, message: str) -> None:
        message = self._redact(message)[:500]
        if BILLING_PAUSE_MARKER in message:
            self._fatal = f"The SDK reports a billing pause: {message}"
        self._record("sdk_log", {"logger": logger_name, "level": level, "message": message})

    def _on_http_error(self, where: str, exc: Any) -> None:
        status = getattr(exc, "status_code", None)
        headers = {}
        header_names: list[str] = []
        response = getattr(exc, "response", None)
        if response is not None:
            header_names = sorted({name.lower() for name in response.headers.keys()})
            headers = {
                name.lower(): self._redact(value)[:200]
                for name, value in response.headers.items()
                if name.lower().startswith(RATE_HEADER_PREFIXES)
            }
        detail = {
            "where": where,
            "status": status,
            "exception": type(exc).__name__,
            "message": self._redact(getattr(exc, "message", exc))[:500],
            "rate_headers": headers,
            "response_header_names": header_names,
        }
        if status in (401, 402, 403):
            self._fatal = f"HTTP {status} from the service ({type(exc).__name__}): {detail['message']}"
        self._record("http_error", detail, throttle=(status == 429))

    def _watch_queue(self, observer: Any, where: str) -> None:
        original = observer.on_queue_state_change

        def on_queue_state_change(queue_state, queue_state_reason):
            state = getattr(queue_state, "value", str(queue_state))
            if state != "active":
                self._record(
                    "queue_state",
                    {"where": where, "state": state, "reason": queue_state_reason},
                    throttle=True,
                )
            return original(queue_state, queue_state_reason)

        try:
            observer.on_queue_state_change = on_queue_state_change
        except Exception:
            pass

    def fatal_condition(self) -> str | None:
        return self._fatal

    def classify_error(self, exc: BaseException) -> ErrorInfo:
        import tinker

        status_error = exc if isinstance(exc, tinker.APIStatusError) else None
        if status_error is None and isinstance(exc.__cause__, tinker.APIStatusError):
            status_error = exc.__cause__
        if status_error is not None:
            status = status_error.status_code
            retry_after = None
            try:
                retry_after = float(status_error.response.headers.get("retry-after"))
            except (TypeError, ValueError, AttributeError):
                pass
            if status in (401, 402, 403):
                return ErrorInfo("account", status)
            if status == 429:
                return ErrorInfo("throttle", status, retry_after)
            if status in (408, 409, 410) or status >= 500:
                return ErrorInfo("transient", status, retry_after)
            return ErrorInfo("fatal", status)
        if isinstance(exc, (tinker.APIConnectionError, asyncio.TimeoutError, TimeoutError)):
            return ErrorInfo("transient")
        if isinstance(exc, tinker.RequestFailedError):
            category = str(getattr(exc, "category", "")).lower()
            return ErrorInfo("transient" if category == "server" else "fatal")
        if type(exc).__name__ == "RetryableException":
            return ErrorInfo("transient")
        return ErrorInfo("fatal")

    # ----------------------------------------------------------------- clients

    async def _service_client(self) -> Any:
        async with self._lock:
            if self._service is None:
                import tinker

                self._service = await asyncio.to_thread(tinker.ServiceClient)
            return self._service

    async def _rest_client(self) -> Any:
        if self._rest is None:
            import tinker

            self._rest = await asyncio.to_thread(tinker.RestClient)
        return self._rest

    async def _admin_client(self) -> Any:
        if self._admin is None:
            import tinker

            self._admin = await asyncio.to_thread(tinker.AdminClient)
        return self._admin

    async def _sampler(self, model_id: str, model_path: str | None) -> Any:
        check_model_id(model_id, allow_paid=self.allow_paid)
        key = (model_id, model_path)
        if key in self._samplers:
            return self._samplers[key]
        service = await self._service_client()
        async with self._lock:
            if key not in self._samplers:
                if model_path is None:
                    sampler = await _resolve(service.create_sampling_client(base_model=model_id))
                else:
                    sampler = await _resolve(service.create_sampling_client(model_path=model_path))
                self._watch_queue(sampler, "sampling")
                self._samplers[key] = sampler
        return self._samplers[key]

    async def sampler_base_model(self, model_id: str, model_path: str | None) -> str:
        sampler = await self._sampler(model_id, model_path)
        return str(sampler.get_base_model())

    async def client_config(self) -> dict:
        """The limits the service hands to the SDK at session start."""
        service = await self._service_client()
        config = service.holder.get_client_config()
        return {k: v for k, v in config.model_dump().items() if isinstance(v, (int, float, bool, str))}

    # ---------------------------------------------------------------- sampling

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
        import tinker

        sampler = await self._sampler(model_id, model_path)
        extra: dict[str, Any] = {}
        if top_logprobs:
            extra["topk_sample_logprobs"] = int(top_logprobs)
        positions: list[int] = []
        if targets:
            import torch

            positions = sorted(targets)
            width = max(len(targets[p]) for p in positions)
            # Row i of the tensor names the token IDs to score at prompt position i + 1.
            ids = torch.full((len(prompt_tokens) - 1, width), -1, dtype=torch.int64)
            for position in positions:
                if not 1 <= position <= len(prompt_tokens) - 1:
                    raise ValueError(f"Cannot score position {position} of a {len(prompt_tokens)}-token prompt")
                wanted = targets[position]
                ids[position - 1, : len(wanted)] = torch.tensor(wanted, dtype=torch.int64)
            extra["target_prompt_logprobs"] = tinker.TensorData.from_torch_sparse(ids, pad_value=-1)

        response = await sampler.sample(
            prompt=tinker.ModelInput.from_ints([int(t) for t in prompt_tokens]),
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                max_tokens=int(max_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
                seed=seed,
                stop=[int(t) for t in stop],
            ),
            **extra,
        )
        sequence = response.sequences[0]
        target_logprobs = None
        if targets:
            if response.target_prompt_logprobs is None:
                raise RuntimeError("The service returned no target_prompt_logprobs")
            table = response.target_prompt_logprobs.to_torch()
            target_logprobs = {
                p: [float(x) for x in table[p - 1, : len(targets[p])]] for p in positions
            }
        top = None
        if top_logprobs and sequence.topk_logprobs is not None:
            top = [[(int(t), float(lp)) for t, lp in (row or [])] for row in sequence.topk_logprobs]
        logprobs = sequence.logprobs
        return RawSample(
            tokens=[int(t) for t in sequence.tokens],
            logprobs=[float(x) for x in logprobs] if logprobs is not None else None,
            top_logprobs=top,
            stop_reason=str(sequence.stop_reason) if sequence.stop_reason is not None else None,
            target_logprobs=target_logprobs,
            prompt_cache_hit_tokens=int(getattr(response, "prompt_cache_hit_tokens", 0) or 0),
        )

    # ---------------------------------------------------------------- training

    async def create_training_run(self, model_id: str, *, rank: int, seed: int | None) -> Any:
        check_model_id(model_id, allow_paid=self.allow_paid)
        service = await self._service_client()
        run = await _resolve(service.create_lora_training_client(base_model=model_id, rank=rank, seed=seed))
        logger = getattr(run, "_queue_state_logger", None)
        if logger is not None:
            self._watch_queue(logger, "training")
        return run

    @staticmethod
    def _to_data(examples: Sequence[Example]) -> list[Any]:
        import tinker
        import torch
        from tinker_cookbook.supervised.common import datum_from_model_input_weights

        return [
            datum_from_model_input_weights(
                tinker.ModelInput.from_ints([int(t) for t in tokens]),
                torch.tensor(weights, dtype=torch.float32),
            )
            for tokens, weights in examples
        ]

    @staticmethod
    def _metrics(output: Any) -> dict:
        metrics = getattr(output, "metrics", None) or {}
        return {str(k): float(v) for k, v in metrics.items() if isinstance(v, (int, float))}

    async def forward(self, run: Any, examples: Sequence[Example], *, backward: bool) -> dict:
        data = self._to_data(examples)
        call = run.forward_backward if backward else run.forward
        output = await _resolve(call(data, "cross_entropy"))
        return {"metrics": self._metrics(output)}

    async def optim_step(self, run: Any, learning_rate: float) -> dict:
        import tinker

        output = await _resolve(run.optim_step(tinker.AdamParams(learning_rate=learning_rate)))
        return {"metrics": self._metrics(output)}

    async def train_step(self, run: Any, examples: Sequence[Example], learning_rate: float) -> dict:
        """One update, submitting the gradient and the optimizer step back to back so
        the service can run them without a gap (the pattern the Tinker docs recommend)."""
        import tinker

        data = self._to_data(examples)
        started = time.monotonic()
        gradient = run.forward_backward(data, "cross_entropy")
        update = run.optim_step(tinker.AdamParams(learning_rate=learning_rate))
        output = await _resolve(gradient)
        gradient_s = time.monotonic() - started
        update_output = await _resolve(update)
        return {
            "metrics": self._metrics(output),
            "optim_metrics": self._metrics(update_output),
            "forward_backward_s": gradient_s,
            "total_s": time.monotonic() - started,
        }

    async def train_pipelined(
        self, run: Any, batches: Sequence[Sequence[Example]], learning_rate: float
    ) -> list[dict]:
        """Several updates with the next batch submitted before the current one is
        awaited, so a request is always queued when the service starts a cycle.
        Returns one entry per update with the time since the previous one finished."""
        import tinker

        results: list[dict] = []
        started = last = time.monotonic()

        async def collect(pending: tuple[Any, Any]) -> None:
            nonlocal last
            output = await _resolve(pending[0])
            await _resolve(pending[1])
            now = time.monotonic()
            results.append({"metrics": self._metrics(output), "seconds": now - last, "elapsed_s": now - started})
            last = now

        pending = None
        for index, examples in enumerate(batches):
            data = self._to_data(examples)
            rate = learning_rate if isinstance(learning_rate, (int, float)) else learning_rate[index]
            submitted = (
                run.forward_backward(data, "cross_entropy"),
                run.optim_step(tinker.AdamParams(learning_rate=rate)),
            )
            if pending is not None:
                await collect(pending)
            pending = submitted
        if pending is not None:
            await collect(pending)
        return results

    async def save_sampler_weights(self, run: Any, name: str, ttl_seconds: int | None) -> str:
        response = await _resolve(run.save_weights_for_sampler(name=name, ttl_seconds=ttl_seconds))
        return str(response.path)

    # ----------------------------------------------------------------- account

    async def checkpoint_info(self, tinker_path: str) -> dict:
        """Base model, size and expiry of a saved checkpoint."""
        rest = await self._rest_client()
        run = await _resolve(rest.get_training_run_by_tinker_path(tinker_path))
        info: dict[str, Any] = {"base_model": run.base_model, "lora_rank": run.lora_rank}
        listing = await _resolve(rest.list_checkpoints(run.training_run_id))
        for checkpoint in listing.checkpoints:
            if checkpoint.tinker_path == tinker_path:
                info["size_bytes"] = checkpoint.size_bytes
                info["expires_at"] = checkpoint.expires_at.isoformat() if checkpoint.expires_at else None
                info["checkpoint_type"] = checkpoint.checkpoint_type
        return info

    async def account_snapshot(self) -> dict:
        """Checkpoint storage in use and today's metered usage, with no identifiers."""
        admin = await self._admin_client()
        snapshot: dict[str, Any] = {}
        storage = await _resolve(admin.get_current_checkpoint_storage_usage())
        snapshot["storage_rate_usd_per_gb_month"] = storage.effective_rate_usd_per_gigabyte_month
        snapshot["checkpoint_count"] = sum(item.checkpoint_count for item in storage.data)
        snapshot["checkpoint_bytes"] = sum(item.size_bytes for item in storage.data)
        costs = [item.estimated_monthly_cost_usd for item in storage.data]
        snapshot["storage_estimated_monthly_cost_usd"] = sum(c for c in costs if c is not None)
        # The storage totals above are refreshed with a delay. The listing is current.
        rest = await self._rest_client()
        listing = await _resolve(rest.list_user_checkpoints(limit=100))
        snapshot["checkpoints_listed"] = len(listing.checkpoints)
        snapshot["checkpoints_listed_bytes"] = sum(c.size_bytes or 0 for c in listing.checkpoints)

        # The usage endpoint wants both ends on a UTC hour boundary.
        end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        start = end - timedelta(hours=24)
        usage = await _resolve(admin.get_billing_usage(start, end))
        totals: dict[str, dict[str, float]] = {}
        for event in usage.data:
            kind = getattr(event.event_info, "type", "unrecognized")
            model = event.base_model or "(none)"
            entry = totals.setdefault(f"{model} | {kind}", {"tokens": 0, "estimated_cost_usd": 0.0, "events": 0})
            entry["tokens"] += int(getattr(event.event_info, "token_count", 0) or 0)
            entry["estimated_cost_usd"] += float(event.estimated_cost_usd or 0.0)
            entry["events"] += 1
        snapshot["usage_window_start"] = start.isoformat()
        snapshot["usage_cost_data_through"] = (
            usage.cost_data_through.isoformat() if usage.cost_data_through else None
        )
        snapshot["usage_by_model_and_type"] = totals
        snapshot["usage_estimated_cost_usd"] = sum(v["estimated_cost_usd"] for v in totals.values())
        return snapshot

    async def close(self) -> None:
        logging.getLogger("tinker").removeHandler(self._log_capture)
        if self in _PROBE_SINKS:
            _PROBE_SINKS.remove(self)
