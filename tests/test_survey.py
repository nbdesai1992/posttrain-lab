"""The survey step end to end, on made-up benchmarks and the fake backend."""

from __future__ import annotations

import json

from posttrain_lab.benchmarks.base import Benchmark, Item, user
from posttrain_lab.benchmarks.simple import pairwise_metrics
from posttrain_lab.steps.base import StepContext, run_step
from posttrain_lab.steps.survey import Survey, soak_timeline
from posttrain_lab.store import JsonlStore
from posttrain_lab.testing import FakeBackend

from conftest import run


class Parity(Benchmark):
    """Is a number even? Twenty items, one of them far too long for the context."""

    name, title, job, role, source = "parity", "Parity", "Judging answers", "target", "parity"
    choices = ("yes", "no")
    max_tokens_fast, max_tokens_slow = 8, 64

    def load(self, root, seed):
        items = [Item(id=f"n{n}", messages=user(f"is {n} even"), answer="yes" if n % 2 == 0 else "no", group="small" if n < 10 else "large") for n in range(20)]
        items.append(Item(id="huge", messages=user("word " * 70000), answer="yes", group="large"))
        return items

    def grade(self, item, reply):
        return {"score": float(reply["text"].strip() == item.answer), "choice": reply["text"].strip() or None, "strict": True}


class Pairs(Benchmark):
    name, title, job, role, source = "pairs", "Pairs", "Judging answers", "watch", "pairs"
    choices = ("A", "B")

    def load(self, root, seed):
        return [Item(id=f"p{p}:{o}", messages=user(f"pair {p} order {o}"), answer="A" if o == "ab" else "B", cluster=f"p{p}") for p in range(15) for o in ("ab", "ba")]

    def grade(self, item, reply):
        return {"score": float(reply["text"].strip() == item.answer), "choice": reply["text"].strip(), "strict": True}

    def extra_metrics(self, items, grades):
        return pairwise_metrics(items, grades, first="A")


def survey(tmp_path, renderer, make_client, answer, **config):
    client, backend = make_client(FakeBackend(renderer, answer=answer), max_concurrency=4, call_log=tmp_path / "runs" / "survey" / "calls.jsonl")
    lines: list[str] = []
    ctx = StepContext(
        step="survey",
        config={"seed": 3, "reference_sample": 6, "_registry": {"parity": Parity(), "pairs": Pairs()}, "_skip_download": True,
                "benchmarks": ["parity", "pairs"], "client": {"max_concurrency": 4}, **config},
        renderer=renderer,
        runs_dir=tmp_path / "runs" / "survey",
        results_dir=tmp_path / "results" / "survey",
        store=JsonlStore(tmp_path / "runs" / "survey" / "results.jsonl"),
        client=client,
        out=lines.append,
    )
    return ctx, backend, lines


def even_judge(prompt: str) -> str:
    """Right about small numbers, says yes to every large one, and always picks A."""
    if "pair" in prompt:
        return "A"
    number = int(prompt.split()[-2])
    return "yes" if number % 2 == 0 or number >= 10 else "no"


def test_survey_dry_run_counts_units_per_setting(tmp_path, renderer, make_client):
    ctx, backend, lines = survey(tmp_path, renderer, make_client, even_judge)
    assert run(run_step(Survey(), ctx, dry_run=True)) == 0
    assert backend.sample_calls == []
    counts = {line.split()[0]: int(line.split()[1]) for line in lines if line.split()[0] in ("parity", "pairs", "total")}
    assert counts == {"parity": 21 + 21 + 6, "pairs": 30 + 30 + 6, "total": 114}


def test_survey_scores_each_setting_and_applies_the_target_rule(tmp_path, renderer, make_client):
    ctx, backend, lines = survey(tmp_path, renderer, make_client, even_judge)
    assert run(run_step(Survey(), ctx)) == 0
    summary = json.loads((ctx.results_dir / "summary.json").read_text())
    parity = summary["benchmarks"]["parity"]
    fast, slow = parity["settings"]["fast"], parity["settings"]["slow"]
    # 10 small right, 5 of 10 large right, and the oversized item counted as a failure.
    assert fast["n"] == 21 and fast["score"]["estimate"] == round(15 / 21, 4)
    assert fast["context_overflow"] == 1 and slow["context_overflow"] == 1
    assert fast["score"]["low"] <= fast["score"]["estimate"] <= fast["score"]["high"]
    assert fast["by_group"] == {"large": {"n": 11, "score": round(5 / 11, 4)}, "small": {"n": 10, "score": 1.0}}
    # The fake model thinks when it is not forced to answer, and its answers do not change.
    assert fast["tokens_per_answer"]["thinking"]["max"] == 0 and slow["tokens_per_answer"]["thinking"]["mean"] == 6
    assert parity["settings"]["effort_0"]["n"] == 6 and parity["settings"]["effort_0"]["tokens_per_answer"]["thinking"]["max"] == 0
    assert parity["slow_minus_fast"]["estimate"] == 0.0 and parity["gap_points"] == 0.0
    assert parity["classification"] == "usable as is"
    assert parity["effort_0_subsample"]["items"] == 6 and parity["effort_0_subsample"]["fast_on_subsample"]["n"] == 6
    pairs = summary["benchmarks"]["pairs"]
    assert pairs["classification"] == "watch only (usable as is)"
    assert pairs["settings"]["fast"]["swap_consistency"] == 0.0 and pairs["settings"]["fast"]["picked_first_position"] == 1.0
    assert set(summary["not_run"]) == {"llm_aggrefact", "wildguardtest"}
    too_long = sum(s["context_overflow"] for s in parity["settings"].values())
    assert too_long in (2, 3)  # the oversized prompt, in fast and slow and perhaps in the subsample
    assert summary["usage"]["api_calls"] == len(backend.sample_calls) == 114 - too_long  # it was never sent
    # First-token probabilities are kept for one-token answers in the fast setting.
    rows = [json.loads(line) for line in (ctx.runs_dir / "results.jsonl").read_text().splitlines()]
    fast_row = next(r for r in rows if r["benchmark"] == "parity" and r["setting"] == "fast" and r["item"] == "n2")
    assert fast_row["first_token"][0][0] == "yes" and fast_row["thinking_tokens"] == 0 and fast_row["text"] == "yes"

    figure = json.loads((ctx.results_dir / "figure-data.json").read_text())
    assert [row["benchmark"] for row in figure["capability_map"]] == ["parity", "pairs"]
    assert figure["capability_map"][0]["fast"]["score"] == fast["score"]["estimate"]
    assert len(figure["tokens_per_answer_against_score"]) == 6 and len(figure["not_run"]) == 2
    report = (ctx.results_dir / "REPORT.md").read_text()
    for heading in ("## Capability map", "## Tokens and time per answer", "## Judging in detail", "## Throughput over the run"):
        assert heading in report
    assert "not run, needs a token" in report
    ids = json.loads((ctx.results_dir / "item-ids.json").read_text())
    assert len(ids["parity"]["items"]) == 21 and len(ids["pairs"]["effort_0_subsample"]) == 6


def test_survey_resumes_and_regrades_without_calling_the_model(tmp_path, renderer, make_client):
    ctx, backend, _ = survey(tmp_path, renderer, make_client, even_judge)
    assert run(run_step(Survey(), ctx)) == 0
    first = (ctx.results_dir / "summary.json").read_text()
    ctx, backend, lines = survey(tmp_path, renderer, make_client, even_judge)
    assert run(run_step(Survey(), ctx)) == 0
    assert backend.sample_calls == []
    assert json.loads(first)["benchmarks"] == json.loads((ctx.results_dir / "summary.json").read_text())["benchmarks"]


def test_a_gap_of_ten_points_makes_a_training_target(tmp_path, renderer, make_client):
    def thinks_better(prompt: str) -> str:
        return "A" if "pair" in prompt else ("yes" if int(prompt.split()[-2]) % 2 == 0 else "no")

    class Lazy(FakeBackend):
        """Right when it reasons first, yes to everything when the answer is forced."""

        async def sample(self, model_id, model_path, prompt_tokens, **kwargs):
            self.answer = (lambda prompt: "yes") if list(prompt_tokens[-2:]) == [1, 3] else thinks_better
            return await super().sample(model_id, model_path, prompt_tokens, **kwargs)

    ctx, _, _ = survey(tmp_path, renderer, make_client, thinks_better, settings=["fast", "slow"], benchmarks=["parity"])
    ctx.client.backend = Lazy(renderer)
    assert run(run_step(Survey(), ctx)) == 0
    parity = json.loads((ctx.results_dir / "summary.json").read_text())["benchmarks"]["parity"]
    assert parity["settings"]["fast"]["score"]["estimate"] == round(10 / 21, 4)
    assert parity["settings"]["slow"]["score"]["estimate"] == round(20 / 21, 4)
    assert parity["gap_points"] == round(10 / 21 * 100, 1) and parity["classification"] == "training target"
    assert parity["slow_minus_fast"]["gained"] == 10 and parity["slow_minus_fast"]["lost"] == 0 and parity["slow_minus_fast"]["mcnemar_p"] < 0.01


def test_soak_timeline_buckets_calls_by_time():
    calls = [
        {"kind": "sample", "cache_hit": False, "error_class": None, "started_at": 1000.0 + i, "ended_at": 1003.0 + i, "latency_s": 3.0,
         "prompt_tokens": 100, "completion_tokens": 50, "attempts": 1, "throttle_events": int(i == 5)}
        for i in range(60)
    ] + [{"kind": "sample", "cache_hit": True, "ended_at": 1010.0, "started_at": 1010.0}]
    events = [{"ts": 1010.0, "type": "queue_state", "state": "paused_rate_limit"}]
    (window,) = soak_timeline(calls, events)
    assert window["requests"] == 60 and window["throttle_events"] == 1 and window["latency_p50_s"] == 3.0
    assert window["service_events"] == {"paused_rate_limit": 1}
    assert 55 <= window["requests_per_min"] <= 62 and window["completion_tokens_per_min"] > 2800
    assert soak_timeline([], []) == []
