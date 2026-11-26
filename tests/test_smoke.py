"""The smoke step end to end on the fake backend, and its prompt checks."""

from __future__ import annotations

import json
import random

import pytest

from posttrain_lab import cli
from posttrain_lab.steps import smoke_prompts as prompts
from posttrain_lab.testing import FakeBackend

CONFIG = """
model_id = "thinkingmachines/Inkling-2-Mini:peft:65536:free"
seed = 7

[client]
max_concurrency = 4
call_timeout_s = 10

[estimates]
completion_tokens_effort_0 = 10
completion_tokens_effort_default = 20
completion_tokens_forced = 3

[throughput]
efforts = [0.0, 0.9]
concurrency = [1, 2]
requests = [3, 4]
max_tokens = 32
max_seconds = 30
throttled_fraction_stop = 0.5

[prefill]
concurrency = 2
requests = 3
source_sentences = 5

[behaviour]
per_kind = 2
max_tokens = 32
concurrency = 4

[verdict]
items = 2
max_tokens = 32
concurrency = 2

[training]
enabled = true
rank = 16
learning_rate = 1e-4
short_examples = 4
long_examples = 2
long_sentences = 3
small_batch = 2
forward_calls = 1
separate_steps = 1
adapter_samples = 1
checkpoint_ttl_seconds = 60

[[training.phases]]
name = "2 short examples"
pool = "short"
batch = 2
steps = 2

[[training.phases]]
name = "2 long examples"
pool = "long"
batch = 2
steps = 1

[context]
enabled = true
limit_tokens = 600
over_limit_tokens = 700
margin_tokens = 20
near_limit = true
sum_over_limit = true
sum_over_max_tokens = 40
"""

# throughput 2 * (3 + 4), prefill 3, behaviour 3 modes * 6 prompts, verdict 4 * 2,
# training (create, save, load, 1 forward, 2 separate, 1 sample, 3 updates), context 3
EXPECTED_CALLS = 14 + 3 + 18 + 8 + 10 + 3
EXPECTED_SAMPLING_CALLS = 14 + 3 + 18 + 8 + 1 + 3


@pytest.fixture
def smoke(tmp_path, renderer):
    config = tmp_path / "smoke.toml"
    config.write_text(CONFIG)
    backends: list[FakeBackend] = []

    def make_backend(allow_paid):
        backends.append(FakeBackend(renderer))
        return backends[-1]

    def invoke(*extra):
        lines: list[str] = []
        code = cli.main(
            ["smoke", "--config", str(config), "--runs-dir", str(tmp_path / "runs"), "--results-dir", str(tmp_path / "results"), *extra],
            make_renderer=lambda model_id: renderer,
            make_backend=make_backend,
            out=lines.append,
        )
        return code, lines

    invoke.backends = backends
    invoke.root = tmp_path
    return invoke


def test_smoke_dry_run_counts_calls_without_a_backend(smoke):
    code, lines = smoke("--dry-run")
    assert code == 0
    assert smoke.backends == []  # no backend was even constructed
    total = next(line for line in lines if line.startswith("total")).split()
    assert total[1] == "13"  # units: 2 account, 4 throughput, prefill, 3 behaviour, verdict, training, context
    assert int(total[3]) == EXPECTED_CALLS
    assert int(total[4].replace(",", "")) > 0  # prompt tokens were counted with the renderer
    assert not (smoke.root / "runs").exists()


def test_smoke_runs_end_to_end_and_writes_the_report(smoke):
    code, lines = smoke()
    assert code == 0, "\n".join(lines)
    backend = smoke.backends[0]
    assert len(backend.sample_calls) == EXPECTED_SAMPLING_CALLS

    summary = json.loads((smoke.root / "results" / "smoke" / "summary.json").read_text())
    assert [row["concurrency"] for row in summary["throughput"]] == [1, 2, 1, 2]
    assert all(row["requests_completed"] == row["requests_planned"] for row in summary["throughput"])
    assert set(summary["behaviour"]) == {"effort_0.0", "effort_0.9", "effort_0.0_forced_answer"}
    assert summary["behaviour"]["effort_0.9"]["overall"]["share_with_thinking"] == 1.0
    assert summary["behaviour"]["effort_0.0_forced_answer"]["overall"]["share_with_thinking"] == 0.0
    assert summary["verdict"][0]["summary"]["n"] == 2
    training = summary["training"][0]
    assert training["tinker_path"].startswith("tinker://")
    assert list(training["phases"]) == ["forward only", "separate calls", "2 short examples", "2 long examples"]
    assert len(summary["context"][0]["probes"]) == 3
    assert summary["context"][0]["probes"][0]["prompt_tokens"] > 600
    assert summary["context"][0]["probes"][1]["prompt_tokens"] <= 580

    report = (smoke.root / "results" / "smoke" / "REPORT.md").read_text()
    for heading in ("## Sampling throughput", "## Throttling", "## What effort 0.0 does", "## Training throughput", "## Context limit", "## Spend"):
        assert heading in report

    rows = (smoke.root / "runs" / "smoke" / "results.jsonl").read_text().splitlines()
    assert len(rows) == 13
    calls = (smoke.root / "runs" / "smoke" / "calls.jsonl").read_text().splitlines()
    assert len(calls) == EXPECTED_CALLS


def test_smoke_rerun_makes_no_calls(smoke):
    assert smoke()[0] == 0
    code, lines = smoke()
    assert code == 0
    assert smoke.backends[1].sample_calls == [] and smoke.backends[1].training_calls == []
    assert next(line for line in lines if line.startswith("total")).split()[2] == "0"


def test_throughput_stops_climbing_when_throttling_dominates(smoke, renderer):
    from posttrain_lab.backends.base import CURRENT_CALL

    class Throttling(FakeBackend):
        async def sample(self, *args, **kwargs):
            record = CURRENT_CALL.get()
            if record is not None and record.tag.startswith("throughput"):
                record.throttle_events += 1
            return await super().sample(*args, **kwargs)

    smoke.backends.append(Throttling(renderer))
    lines: list[str] = []
    config = smoke.root / "smoke.toml"
    code = cli.main(
        ["smoke", "--config", str(config), "--runs-dir", str(smoke.root / "runs"), "--results-dir", str(smoke.root / "results")],
        make_renderer=lambda model_id: renderer,
        make_backend=lambda allow_paid: smoke.backends[0],
        out=lines.append,
    )
    assert code == 0
    summary = json.loads((smoke.root / "results" / "smoke" / "summary.json").read_text())
    levels = summary["throughput"]
    assert levels[0]["throttle_dominated"] and levels[0]["throttled_requests"] == 3
    assert levels[1] == {**levels[1], "skipped": True}
    assert "throttling dominated at concurrency 1" in levels[1]["reason"]


# ---------------------------------------------------------------- prompts


def test_prompt_sets_are_reproducible_and_mixed():
    first = prompts.behaviour_items(3, 4)
    again = prompts.behaviour_items(3, 4)
    assert first == again
    assert [i["kind"] for i in first[:3]] == ["arithmetic", "judgment", "extraction"]
    assert {i["answer"] for i in first if i["kind"] == "judgment"} == {"yes", "no"}
    assert len({i["id"] for i in first}) == 12


def test_judgment_labels_match_the_arithmetic():
    rng = random.Random(1)
    for _ in range(50):
        item = prompts.judgment_item(rng)
        question, answered = item["messages"][0]["content"].split("\n")[:2]
        a, op, b = question.removeprefix("A student was asked: What is ").rstrip("?").split()
        truth = {"+": int(a) + int(b), "-": int(a) - int(b), "*": int(a) * int(b)}[op]
        given = int(answered.removeprefix("The student answered: "))
        assert (given == truth) == (item["answer"] == "yes")


def test_long_claims_are_true_or_false_by_construction():
    rng = random.Random(2)
    for _ in range(20):
        item = prompts.long_judgment_item(rng, 8)
        text = item["messages"][0]["content"]
        claim = text.split("Claim: ")[1].split("\n")[0]
        number = claim.split(" says ")[0]
        stated = claim.split(" ordered ")[1].split(" units")[0]
        source_line = next(line for line in text.split("\n") if line.startswith(number + ":"))
        assert (f"ordered {stated} units" in source_line) == (item["answer"] == "yes")


@pytest.mark.parametrize(
    "kind, expected, text, parseable, correct",
    [
        ("arithmetic", "861", "861", True, True),
        ("arithmetic", "861", " 861.\n", True, True),
        ("arithmetic", "861", "1,861", True, False),
        ("arithmetic", "861", "The answer is 861.", False, True),
        ("arithmetic", "-12", "-12", True, True),
        ("arithmetic", "861", "", False, False),
        ("judgment", "yes", "Yes.", True, True),
        ("judgment", "no", "yes", True, False),
        ("judgment", "no", "No, the answer is wrong.", False, True),
        ("extraction", "INV-48213", "INV-48213", True, True),
        ("extraction", "INV-48213", "The invoice number is INV-48213, as stated in the text.", False, True),
        ("extraction", "Lena Ortiz", "Lena Ortiz", True, True),
        ("extraction", "Lisbon", "Tucson", True, False),
    ],
)
def test_check_answer(kind, expected, text, parseable, correct):
    assert prompts.check_answer({"kind": kind, "answer": expected}, text) == {"parseable": parseable, "correct": correct}
