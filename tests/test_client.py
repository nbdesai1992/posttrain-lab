"""The client layer: caching, retries, concurrency, call records."""

from __future__ import annotations

import asyncio
import json
import math

import pytest

from posttrain_lab.client import AccountBlockedError
from posttrain_lab.render import SpecialTokens, answer_position, parse_reply, segment_response
from posttrain_lab.testing import (
    FakeAccountProblem,
    FakeBackend,
    FakeThrottle,
    FakeTransient,
)

from conftest import run

QUESTION = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "is two even"}]
OTHER = [{"role": "user", "content": "is three even"}]


def read_log(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


# ------------------------------------------------------------------- cache


def test_second_identical_call_is_served_from_the_cache(make_client):
    client, backend = make_client()
    first = run(client.sample(QUESTION, effort=0.0, max_tokens=32))
    second = run(client.sample(QUESTION, effort=0.0, max_tokens=32))
    assert len(backend.sample_calls) == 1
    assert not first.record.cache_hit and second.record.cache_hit
    assert second.tokens == first.tokens and second.text == first.text
    assert client.totals["api_calls"] == 1 and client.totals["cache_hits"] == 1


def test_cache_persists_across_clients(make_client, renderer):
    client, backend = make_client()
    run(client.sample(QUESTION, effort=0.0, max_tokens=32, top_logprobs=2))
    later, later_backend = make_client(FakeBackend(renderer))
    result = run(later.sample(QUESTION, effort=0.0, max_tokens=32, top_logprobs=2))
    assert result.record.cache_hit
    assert later_backend.sample_calls == []
    assert result.top_logprobs is not None and isinstance(result.top_logprobs[0][0], tuple)


@pytest.mark.parametrize(
    "change",
    [
        {"effort": 0.9},
        {"max_tokens": 33},
        {"temperature": 0.7},
        {"seed": 1},
        {"top_logprobs": 3},
        {"salt": "again"},
        {"model_path": "tinker://run/sampler_weights/x"},
        {"force_answer": True},
    ],
)
def test_any_change_to_the_request_is_a_cache_miss(make_client, change):
    client, backend = make_client()
    base = {"effort": 0.0, "max_tokens": 32}
    run(client.sample(QUESTION, **base))
    run(client.sample(QUESTION, **{**base, **change}))
    assert len(backend.sample_calls) == 2


def test_a_different_prompt_is_a_cache_miss(make_client):
    client, backend = make_client()
    run(client.sample(QUESTION, effort=0.0, max_tokens=32))
    run(client.sample(OTHER, effort=0.0, max_tokens=32))
    assert len(backend.sample_calls) == 2


def test_a_different_model_id_is_a_cache_miss(make_client, renderer):
    client, backend = make_client()
    run(client.sample(QUESTION, effort=0.0, max_tokens=32))
    other, other_backend = make_client(FakeBackend(renderer), model_id="some/other-model:free")
    result = run(other.sample(QUESTION, effort=0.0, max_tokens=32))
    assert not result.record.cache_hit and len(other_backend.sample_calls) == 1


def test_failed_calls_are_not_cached(make_client, renderer):
    backend = FakeBackend(renderer, failures=[ValueError("bad request")])
    client, _ = make_client(backend)
    with pytest.raises(ValueError):
        run(client.sample(QUESTION, effort=0.0, max_tokens=32))
    result = run(client.sample(QUESTION, effort=0.0, max_tokens=32))
    assert not result.record.cache_hit and len(backend.sample_calls) == 2


# ----------------------------------------------------------------- retries


def test_throttled_call_is_retried_with_backoff(make_client, renderer):
    backend = FakeBackend(renderer, failures=[FakeThrottle(), FakeTransient(), None])
    slept: list[float] = []

    async def sleep(seconds):
        slept.append(seconds)

    client, _ = make_client(backend, sleep=sleep)
    result = run(client.sample(QUESTION, effort=0.0, max_tokens=32))
    assert result.text == "yes"
    assert result.record.attempts == 3
    assert result.record.throttle_events == 1
    assert result.record.retried_errors == ["FakeThrottle", "FakeTransient"]
    assert len(slept) == 2 and slept[1] > slept[0] * 0.9


def test_retries_stop_at_the_attempt_limit(make_client, renderer, tmp_path):
    backend = FakeBackend(renderer, failures=[FakeTransient()] * 10)
    client, _ = make_client(backend)
    with pytest.raises(FakeTransient):
        run(client.sample(QUESTION, effort=0.0, max_tokens=32))
    assert len(backend.sample_calls) == 4
    record = read_log(tmp_path / "calls.jsonl")[-1]
    assert record["error_class"] == "FakeTransient" and record["attempts"] == 4


def test_request_errors_are_not_retried(make_client, renderer):
    backend = FakeBackend(renderer, failures=[ValueError("bad request"), None])
    client, _ = make_client(backend)
    with pytest.raises(ValueError):
        run(client.sample(QUESTION, effort=0.0, max_tokens=32))
    assert len(backend.sample_calls) == 1


def test_account_problem_stops_without_retrying(make_client, renderer):
    backend = FakeBackend(renderer, failures=[FakeAccountProblem("balance"), None])
    client, _ = make_client(backend)
    with pytest.raises(AccountBlockedError):
        run(client.sample(QUESTION, effort=0.0, max_tokens=32))
    assert len(backend.sample_calls) == 1


def test_a_stop_condition_from_the_backend_aborts_a_waiting_call(make_client, renderer):
    backend = FakeBackend(renderer, latency_s=5.0)
    backend.fatal = "billing pause"
    client, _ = make_client(backend)
    with pytest.raises(AccountBlockedError, match="billing pause"):
        run(client.sample(QUESTION, effort=0.0, max_tokens=32))


# ------------------------------------------------------------- concurrency


def test_calls_in_flight_never_exceed_the_limit(make_client, renderer):
    backend = FakeBackend(renderer, latency_s=0.01)
    client, _ = make_client(backend, max_concurrency=3)

    async def burst():
        prompts = [[{"role": "user", "content": f"question {i}"}] for i in range(20)]
        return await asyncio.gather(*(client.sample(p, effort=0.0, max_tokens=32) for p in prompts))

    results = run(burst())
    assert len(results) == 20
    assert backend.max_in_flight == 3


def test_concurrency_can_be_changed_between_batches(make_client, renderer):
    backend = FakeBackend(renderer, latency_s=0.01)
    client, _ = make_client(backend, max_concurrency=1)

    async def burst(tag):
        prompts = [[{"role": "user", "content": f"{tag} {i}"}] for i in range(8)]
        await asyncio.gather(*(client.sample(p, effort=0.0, max_tokens=32) for p in prompts))

    run(burst("a"))
    assert backend.max_in_flight == 1
    client.set_concurrency(4)
    run(burst("b"))
    assert backend.max_in_flight == 4


# ----------------------------------------------------------------- records


def test_every_call_is_recorded_with_tokens_and_effort(make_client, renderer, tmp_path):
    client, backend = make_client(step="unit-test")
    result = run(client.sample(QUESTION, effort=0.9, max_tokens=64, seed=7, tag="t1"))
    run(client.sample(QUESTION, effort=0.9, max_tokens=64, seed=7, tag="t2"))
    records = read_log(tmp_path / "calls.jsonl")
    assert len(records) == 2
    first, second = records
    assert first["step"] == "unit-test" and first["tag"] == "t1" and first["kind"] == "sample"
    assert first["effort"] == 0.9 and first["seed"] == 7 and first["max_tokens"] == 64
    assert first["prompt_tokens"] == len(renderer.build_prompt(QUESTION, 0.9))
    assert first["completion_tokens"] == len(result.tokens)
    assert first["thinking_tokens"] == backend.thinking_words
    assert first["answer_tokens"] == 1
    assert first["latency_s"] >= 0 and first["attempts"] == 1 and first["error_class"] is None
    assert first["cache_hit"] is False and second["cache_hit"] is True
    assert first["key"] == second["key"]


def test_thinking_tokens_are_counted_apart_from_the_answer(make_client):
    client, backend = make_client()
    slow = run(client.sample(QUESTION, effort=0.9, max_tokens=64))
    fast = run(client.sample(QUESTION, effort=0.0, max_tokens=64))
    assert slow.parsed.first_kind == "thinking" and slow.record.thinking_tokens == backend.thinking_words
    assert fast.parsed.first_kind == "text" and fast.record.thinking_tokens == 0
    assert slow.text == fast.text == "yes"
    assert slow.parsed.ended and fast.parsed.ended


def test_truncated_reply_is_marked_as_not_ended(make_client):
    client, _ = make_client()
    result = run(client.sample(QUESTION, effort=0.9, max_tokens=4))
    assert result.stop_reason == "length"
    assert not result.parsed.ended and result.text == ""


def test_forced_answer_skips_thinking(make_client, renderer):
    client, backend = make_client()
    result = run(client.sample(QUESTION, effort=0.9, max_tokens=64, force_answer=True))
    special = renderer.special
    assert backend.sample_calls[0]["prompt"][-2:] == [special.message_model, special.content_text]
    assert result.text == "yes" and result.record.thinking_tokens == 0
    assert result.parsed.first_text_index == 0
    assert result.parsed.n_tokens == len(result.tokens) == 3


def test_score_choices_reads_the_answer_position(make_client, renderer):
    client, backend = make_client()
    scores = run(client.score_choices(QUESTION, ["yes", "no"], effort=0.0))
    prompt = backend.sample_calls[0]["prompt"]
    special = renderer.special
    # The prompt is cut right after the first answer token of a text reply.
    assert prompt[-3:-1] == [special.message_model, special.content_text]
    assert math.isclose(scores.probs["yes"], 0.7) and math.isclose(scores.probs["no"], 0.3)
    assert math.isclose(scores.mass, 1.0)
    assert math.isclose(scores.p_text_first, 0.7) and math.isclose(scores.p_thinking_first, 0.3)
    assert scores.record.kind == "score" and scores.record.max_tokens == 1
    run(client.score_choices(QUESTION, ["yes", "no"], effort=0.0))
    assert len(backend.sample_calls) == 1  # the repeat came from the cache


def test_score_choices_needs_single_token_candidates(make_client):
    client, _ = make_client()
    with pytest.raises(ValueError, match="exactly one"):
        run(client.score_choices(QUESTION, ["yes", "not sure"]))


def test_training_calls_are_recorded_and_never_cached(make_client, renderer, tmp_path):
    client, backend = make_client()
    example = renderer.build_example([*QUESTION, {"role": "assistant", "content": "yes"}], 0.0)

    async def train():
        training = await client.create_training_run(rank=16, seed=1)
        await training.forward([example, example])
        await training.forward_backward([example])
        await training.optim_step(1e-4)
        await training.train_step([example], 1e-4)
        await training.train_step([example], 1e-4)
        path, _ = await training.save_sampler_weights("smoke", ttl_seconds=60)
        await client.load_adapter(path)
        return path

    path = run(train())
    assert path.startswith("tinker://")
    kinds = [r["kind"] for r in read_log(tmp_path / "calls.jsonl")]
    assert kinds == [
        "create_training_client",
        "forward",
        "forward_backward",
        "optim_step",
        "train_step",
        "train_step",
        "save_sampler_weights",
        "create_sampling_client",
    ]
    forward = read_log(tmp_path / "calls.jsonl")[1]
    assert forward["examples"] == 2 and forward["prompt_tokens"] == 2 * len(example[0])
    assert client.totals["training_tokens"] == 5 * len(example[0])
    assert backend.training_calls.count("forward_backward") == 3


# ----------------------------------------------------------------- parsing

SPECIAL = SpecialTokens(message_model=1, content_thinking=2, content_text=3, end_message=4, end_sampling=5)


def decode(tokens):
    return " ".join(str(t) for t in tokens)


def test_segmenting_a_reply_with_thinking():
    parsed = segment_response([1, 2, 10, 11, 12, 4, 1, 3, 20, 21, 4, 5], SPECIAL, decode)
    assert parsed.first_kind == "thinking"
    assert parsed.n_thinking_tokens == 3 and parsed.n_text_tokens == 2 and parsed.n_framing_tokens == 7
    assert parsed.thinking == "10 11 12" and parsed.text == "20 21"
    assert parsed.first_text_index == 8
    assert parsed.ended and parsed.well_formed


def test_segmenting_a_reply_with_native_tool_calls():
    special = SpecialTokens(message_model=1, content_thinking=2, content_text=3, end_message=4, end_sampling=5, content_tool_call=6)
    words = {50: '{"name": "pwd", "args": {}}', 51: '{"name": "cd", "args": {"folder": "docs"}}', 52: "not json"}
    tokens = [1, 2, 10, 4, 1, 70, 6, 50, 4, 1, 71, 72, 6, 51, 4, 5]
    parsed = segment_response(tokens, special, lambda ids: "".join(words.get(t, str(t)) for t in ids))
    assert parsed.tool_calls == [{"name": "pwd", "args": {}}, {"name": "cd", "args": {"folder": "docs"}}]
    assert parsed.text == "" and parsed.n_thinking_tokens == 1 and parsed.ended and parsed.well_formed
    assert parsed.first_kind == "thinking"
    broken = segment_response([1, 70, 6, 52, 4, 5], special, lambda ids: "".join(words.get(t, str(t)) for t in ids))
    assert broken.tool_calls == [] and not broken.well_formed and broken.first_kind == "tool_call"


def test_segmenting_a_reply_cut_off_mid_thought():
    parsed = segment_response([1, 2, 10, 11], SPECIAL, decode)
    assert not parsed.ended and parsed.text == "" and parsed.first_text_index is None
    assert parsed.n_thinking_tokens == 2


def test_parsing_a_reply_after_a_forced_opening():
    parsed = parse_reply([20, 4, 5], SPECIAL, decode, forced_opening=[1, 3])
    assert parsed.text == "20" and parsed.first_text_index == 0
    assert parsed.n_tokens == 3 and parsed.n_framing_tokens == 2 and parsed.ended


def test_answer_position_finds_the_first_answer_token():
    example = [9, 9, 9, 1, 3, 20, 4, 5]
    assert answer_position(example, SPECIAL) == 5
    with pytest.raises(ValueError):
        answer_position([9, 9, 9], SPECIAL)
