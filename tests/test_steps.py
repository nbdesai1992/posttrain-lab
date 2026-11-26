"""The step framework: planning, dry runs, resuming, and the command line."""

from __future__ import annotations

import pytest

from posttrain_lab import cli
from posttrain_lab.steps import STEPS
from posttrain_lab.steps.base import Step, StepContext, Unit, run_step
from posttrain_lab.store import JsonlStore
from posttrain_lab.testing import FakeAccountProblem, FakeBackend

from conftest import run


class Counting(Step):
    """Five units, one sampling call each. Can be told to fail at one unit."""

    name = "counting"
    summary = "test step"

    def __init__(self, fail_at: int | None = None, parallel: bool = False):
        self.fail_at = fail_at
        self.parallel_units = parallel
        self.ran: list[int] = []
        self.finished_with: list[dict] | None = None

    def plan(self, ctx):
        return [
            Unit("ask", {"n": n}, calls=1, prompt_tokens=10, completion_tokens=3, max_completion_tokens=16)
            for n in range(5)
        ]

    async def run_unit(self, unit, ctx):
        n = unit.params["n"]
        if n == self.fail_at:
            raise RuntimeError(f"unit {n} failed")
        result = await ctx.client.sample([{"role": "user", "content": f"question {n}"}], effort=0.0, max_tokens=16)
        self.ran.append(n)
        return {"text": result.text}

    async def finish(self, ctx, rows):
        self.finished_with = rows


def context(tmp_path, renderer, client=None, lines=None):
    return StepContext(
        step="counting",
        config={},
        renderer=renderer,
        runs_dir=tmp_path / "runs" / "counting",
        results_dir=tmp_path / "results" / "counting",
        store=JsonlStore(tmp_path / "runs" / "counting" / "results.jsonl"),
        client=client,
        out=(lines.append if lines is not None else lambda line: None),
    )


def test_dry_run_prints_counts_and_calls_nothing(tmp_path, renderer, make_client):
    client, backend = make_client()
    lines: list[str] = []
    step = Counting()
    code = run(run_step(step, context(tmp_path, renderer, client, lines), dry_run=True))
    assert code == 0
    assert backend.sample_calls == [] and step.ran == []
    total = next(line for line in lines if line.startswith("total"))
    assert total.split() == ["total", "5", "5", "5", "50", "15", "80", "0"]
    assert any("no API calls" in line for line in lines)


def test_dry_run_needs_no_client(tmp_path, renderer):
    assert run(run_step(Counting(), context(tmp_path, renderer), dry_run=True)) == 0


def test_run_writes_one_row_per_unit_and_calls_finish(tmp_path, renderer, make_client):
    client, backend = make_client()
    step = Counting()
    ctx = context(tmp_path, renderer, client)
    assert run(run_step(step, ctx)) == 0
    assert step.ran == [0, 1, 2, 3, 4]
    assert len(JsonlStore(ctx.store.path)) == 5
    assert [row["params"]["n"] for row in step.finished_with] == [0, 1, 2, 3, 4]
    assert all(row["text"] == "yes" and row["kind"] == "ask" for row in step.finished_with)


def test_rerun_resumes_after_a_failure(tmp_path, renderer, make_client):
    client, backend = make_client()
    failing = Counting(fail_at=2)
    with pytest.raises(RuntimeError):
        run(run_step(failing, context(tmp_path, renderer, client)))
    assert failing.ran == [0, 1] and failing.finished_with is None
    assert len(backend.sample_calls) == 2

    client, backend = make_client()
    resumed = Counting()
    lines: list[str] = []
    assert run(run_step(resumed, context(tmp_path, renderer, client, lines))) == 0
    assert resumed.ran == [2, 3, 4]  # units 0 and 1 were not run again
    assert len(backend.sample_calls) == 3
    assert len(resumed.finished_with) == 5
    assert next(line for line in lines if line.startswith("total")).split()[:4] == ["total", "5", "3", "3"]


def test_a_finished_step_reruns_nothing(tmp_path, renderer, make_client):
    client, _ = make_client()
    run(run_step(Counting(), context(tmp_path, renderer, client)))
    client, backend = make_client()
    again = Counting()
    assert run(run_step(again, context(tmp_path, renderer, client))) == 0
    assert again.ran == [] and backend.sample_calls == []
    assert len(again.finished_with) == 5


def test_fresh_reruns_units_but_reuses_cached_responses(tmp_path, renderer, make_client):
    client, _ = make_client()
    run(run_step(Counting(), context(tmp_path, renderer, client)))
    client, backend = make_client()
    again = Counting()
    ctx = context(tmp_path, renderer, client)
    assert run(run_step(again, ctx, fresh=True)) == 0
    assert again.ran == [0, 1, 2, 3, 4]
    assert backend.sample_calls == []  # the model responses came from the cache
    assert len(ctx.store.path.read_text().splitlines()) == 10  # appended, not rewritten


def test_only_reruns_units_of_one_kind(tmp_path, renderer, make_client):
    class TwoKinds(Counting):
        def plan(self, ctx):
            return [Unit("ask" if n < 3 else "tell", {"n": n}, calls=1) for n in range(5)]

    client, _ = make_client()
    run(run_step(TwoKinds(), context(tmp_path, renderer, client)))
    client, _ = make_client()
    again = TwoKinds()
    assert run(run_step(again, context(tmp_path, renderer, client), fresh=True, only="tell")) == 0
    assert again.ran == [3, 4]
    assert len(again.finished_with) == 5  # the summary still sees every unit
    with pytest.raises(ValueError, match="no unit of kind"):
        run(run_step(TwoKinds(), context(tmp_path, renderer, client), only="nothing"))


def test_parallel_units_all_run(tmp_path, renderer, make_client):
    client, backend = make_client(FakeBackend(renderer, latency_s=0.01), max_concurrency=3)
    step = Counting(parallel=True)
    assert run(run_step(step, context(tmp_path, renderer, client))) == 0
    assert sorted(step.ran) == [0, 1, 2, 3, 4]
    assert backend.max_in_flight == 3


def test_account_problem_stops_the_step_and_keeps_finished_units(tmp_path, renderer, make_client):
    backend = FakeBackend(renderer, failures=[None, None, FakeAccountProblem("no balance")])
    client, _ = make_client(backend)
    lines: list[str] = []
    step = Counting()
    ctx = context(tmp_path, renderer, client, lines)
    assert run(run_step(step, ctx)) == 3
    assert step.ran == [0, 1] and step.finished_with is None
    assert len(JsonlStore(ctx.store.path)) == 2
    assert any("account problem" in line for line in lines)


def test_unit_keys_depend_on_step_kind_and_params():
    assert Unit("a", {"n": 1}).key("s") == Unit("a", {"n": 1}, calls=9).key("s")
    assert Unit("a", {"n": 1}).key("s") != Unit("a", {"n": 2}).key("s")
    assert Unit("a", {"n": 1}).key("s") != Unit("b", {"n": 1}).key("s")
    assert Unit("a", {"n": 1}).key("s") != Unit("a", {"n": 1}).key("t")


# ------------------------------------------------------------ command line


def test_help_lists_every_step(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--help"])
    assert exit_info.value.code == 0
    text = capsys.readouterr().out
    for name in ("smoke", "survey", "build-data", "filter", "train", "eval", "report"):
        assert name in text
    assert list(STEPS) == ["smoke", "survey", "build-data", "filter", "train", "eval", "report"]


@pytest.mark.parametrize("name", ["report"])
def test_later_steps_say_not_implemented(name, tmp_path, renderer):
    lines: list[str] = []
    made: list[str] = []
    code = cli.main(
        [name, "--runs-dir", str(tmp_path / "runs")],
        make_renderer=lambda model_id: renderer,
        make_backend=lambda allow_paid: made.append("backend"),
        out=lines.append,
    )
    assert code == 2
    assert lines == [f"{name}: not implemented yet"]
    assert made == []


def test_every_step_accepts_dry_run(capsys):
    for name in STEPS:
        with pytest.raises(SystemExit):
            cli.main([name, "--help"])
        assert "--dry-run" in capsys.readouterr().out
