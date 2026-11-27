"""The train and eval steps on the fake backend."""

from __future__ import annotations

import gzip
import json
import random

import pytest

from posttrain_lab.data.sets import build_sets
from posttrain_lab.steps.base import StepContext, run_step
from posttrain_lab.steps.evaluate import Eval, decide, share_of_gap
from posttrain_lab.steps.train import Train, fast_example, schedule, set_items, updates_per_epoch
from posttrain_lab.client import RetryPolicy
from posttrain_lab.store import JsonlStore
from posttrain_lab.testing import FakeBackend, FakeTransient

from conftest import run
from test_survey import Pairs, Parity


@pytest.fixture
def data_dir(tmp_path, renderer):
    rng = random.Random(3)
    items = []
    for k in range(400):
        prompt, answer = f"what is {k} plus {k}", str(2 * k)
        items.append({"id": f"i{k:03d}", "cell": ["middle", "middle", "easy", "hard"][k % 4], "family": f"f{k % 9}", "prompt": prompt, "answer": answer,
                      "answer_type": "number", "train_tokens": len(renderer.build_example([{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}], 0.0)[0])})
    built = build_sets(items, seed=1, dev_per_cell=9, pilot_items=6, sizes=(10, 20), main_size=10)
    out = tmp_path / "data"
    out.mkdir()
    with gzip.open(out / "items.jsonl.gz", "wt") as handle:
        handle.write(json.dumps({"canary": "x"}) + "\n" + "".join(json.dumps(i) + "\n" for i in items))
    (out / "sets.json").write_text(json.dumps(built))
    return out


def context(tmp_path, renderer, step, client, **config):
    lines: list[str] = []
    return StepContext(step=step, config=config, renderer=renderer, runs_dir=tmp_path / "runs" / step, results_dir=tmp_path / "results" / step,
                       store=JsonlStore(tmp_path / "runs" / step / "results.jsonl"), client=client, out=lines.append), lines


RUNS = [{"name": "middle-s1", "set": "middle-10", "seed": 1}, {"name": "wrong-labels", "set": "shuffled-labels-10", "seed": 1}, {"name": "pilot-r8", "set": "pilot-train", "rank": 8, "learning_rate": 1e-3}]


def test_schedule_covers_every_item_once_per_pass():
    batches = schedule(10, 4, 3, seed=5)
    assert len(batches) == 9 and [len(b) for b in batches[:3]] == [4, 4, 2]
    assert sorted(i for b in batches[:3] for i in b) == list(range(10)) and batches[:3] != batches[3:6]
    assert batches == schedule(10, 4, 3, seed=5)


def test_updates_sized_in_tokens_give_token_matched_sets_the_same_number_of_updates():
    assert updates_per_epoch(221_061, 3500) == updates_per_epoch(221_081, 3500) == 63 and updates_per_epoch(900, 3500) == 1
    assert updates_per_epoch(221_061, None) is None
    short, long = schedule(2000, 32, 4, seed=1, per_epoch=63), schedule(2865, 32, 4, seed=1, per_epoch=63)
    assert len(short) == len(long) == 252
    assert {len(b) for b in short} <= {31, 32} and {len(b) for b in long} <= {45, 46}
    for batches, n in ((short, 2000), (long, 2865)):
        assert all(sorted(i for b in batches[e * 63 : (e + 1) * 63] for i in b) == list(range(n)) for e in range(4))
    assert len(schedule(5, 32, 2, seed=1, per_epoch=9)) == 10  # never more batches than items


def test_train_records_each_run_and_decays_the_learning_rate(tmp_path, renderer, make_client, data_dir):
    def double(text):  # the fake adapter has learned the task
        return str(2 * int(text.split()[-3]))

    backend = FakeBackend(renderer, answer="0", adapter_answer=double)
    client, _ = make_client(backend)
    config = {"data_dir": str(data_dir), "runs": RUNS, "batch_size": 4, "epochs": 2, "learning_rate": 4e-4, "updates_per_call": 3, "model_id": "m:free"}
    ctx, lines = context(tmp_path, renderer, "train", client, **config)
    assert run(run_step(Train(), ctx, dry_run=True)) == 0 and backend.training_calls == []
    assert run(run_step(Train(), ctx)) == 0
    summary = json.loads((ctx.results_dir / "summary.json").read_text())
    runs = {r["name"]: r for r in summary["runs"]}
    main = runs["middle-s1"]
    assert summary["complete"] and summary["planned_runs"] == [r["name"] for r in RUNS] and summary["totals"]["updates"] == 6 + 6 + 4
    assert main["stalls"] == {"updates": 0, "seconds": 0.0, "longest_s": 0.0} and main["median_seconds_per_update"] == 0.5 and main["started_at"] <= main["finished_at"]
    assert "losses" not in main and len(main["group_losses"]) == 2
    progress = (ctx.results_dir / "PROGRESS.md").read_text()
    assert "3 of 3 runs are done" in progress and "| middle-s1 | 6 |" in progress and "Still to run: nothing." in progress
    report = (ctx.results_dir / "REPORT.md").read_text()
    assert "| wrong-labels | shuffled-labels-10 |" in report and "tinker://" in report
    figure = json.loads((ctx.results_dir / "figure-data.json").read_text())
    assert [r["run"] for r in figure["runs"]] == [r["name"] for r in RUNS] and len(figure["runs"][0]["loss_by_group_of_updates"]) == 2
    rows = [json.loads(line) for line in (ctx.runs_dir / "progress.jsonl").read_text().splitlines()]
    assert rows[0]["update_seconds"] == [0.5, 0.5, 0.5] and rows[0]["eval_active"] is False and summary["overlap_with_evaluation"]["median_ratio"] is None
    assert (main["items"], main["updates"], main["rank"], main["learning_rate"], main["seed"]) == (10, 6, 16, 4e-4, 1)
    assert main["train_tokens"] == 2 * main["set_train_tokens"] and main["tinker_path"].startswith("tinker://")
    assert main["loss_last"] < main["loss_first"] and main["dev_items"] == 27 and main["dev_accuracy"] == {"easy": 1.0, "hard": 1.0, "middle": 1.0}
    assert runs["pilot-r8"]["rank"] == 8 and runs["pilot-r8"]["learning_rate"] == 1e-3 and runs["pilot-r8"]["items"] == 6
    assert runs["pilot-r8"]["dev_items"] == 27 and main["dev_unfinished_replies"] == 0
    assert set(main["dev_loss"]) == {"easy", "middle", "hard"} and main["dev_loss"]["middle"] < main["dev_loss_before"]["middle"]
    # The prompt and the forced opening carry no loss: the answer and the two closing tokens do.
    tokens, weights = fast_example(renderer, "what is 3 plus 3", "6")
    forced = [*renderer.build_prompt([{"role": "user", "content": "what is 3 plus 3"}], 0.0), renderer.special.message_model, renderer.special.content_text]
    assert tokens[: len(forced)] == forced and weights == [0.0] * len(forced) + [1.0] * (len(tokens) - len(forced)) and len(tokens) - len(forced) == 3
    assert main["loss_weighted_tokens"] == 30
    assert runs["wrong-labels"]["set_fingerprint"] != main["set_fingerprint"]
    assert backend.training_calls.count("create") == 3 and backend.training_calls.count("save") == 3
    assert backend.training_calls.count("new_session") == 3 and backend.training_calls[-1] == "new_session"  # each run ends its session
    rates = backend.learning_rates[:6]
    assert rates[0] == 4e-4 and rates == sorted(rates, reverse=True) and 0 < rates[-1] < 1e-4
    # Examples are the prompt and the answer alone, and the shuffled set trains on moved answers.
    items, sets = Train()._data(ctx)
    assert [r["answer"] for r in set_items("shuffled-labels-10", items, sets)] != [r["answer"] for r in set_items("middle-10", items, sets)]
    assert sorted(r["id"] for r in set_items("shuffled-labels-10", items, sets)) == sorted(r["id"] for r in set_items("middle-10", items, sets))
    # A finished run is not trained again.
    client, again = make_client(FakeBackend(renderer, answer="0", adapter_answer=double))
    ctx, _ = context(tmp_path, renderer, "train", client, **config)
    assert run(run_step(Train(), ctx)) == 0 and again.training_calls == []


def test_train_stops_at_the_checkpoint_limit_and_refuses_a_paid_base(tmp_path, renderer, make_client, data_dir):
    backend = FakeBackend(renderer)
    backend.stored_checkpoints = 95
    client, _ = make_client(backend)
    ctx, lines = context(tmp_path, renderer, "train", client, data_dir=str(data_dir), runs=RUNS[:1], batch_size=4, epochs=1)
    assert run(run_step(Train(), ctx)) == 3 and backend.training_calls == []
    assert any("checkpoints are stored" in line for line in lines)
    backend = FakeBackend(renderer)
    backend.adapter_base_model = "thinkingmachines/Inkling-2-Mini"
    client, _ = make_client(backend)
    ctx, _ = context(tmp_path / "paid", renderer, "train", client, data_dir=str(data_dir), runs=RUNS[:1], batch_size=4, epochs=1)
    assert run(run_step(Train(), ctx)) == 1  # the adapter is never sampled, and the run stays pending
    summary = json.loads((ctx.results_dir / "summary.json").read_text())
    assert backend.sample_calls == [] and summary["runs"] == [] and not summary["complete"]
    assert summary["failures"][0]["run"] == "middle-s1" and "PaidModelError" in summary["failures"][0]["error"]
    assert "middle-s1 failed" in (ctx.results_dir / "PROGRESS.md").read_text()


def test_a_run_that_fails_leaves_the_others_to_finish(tmp_path, renderer, make_client, data_dir):
    class Flaky(FakeBackend):
        async def train_pipelined(self, run_handle, batches, learning_rate):
            if self.training_calls.count("create") == 2 and not getattr(self, "failed", False):
                self.failed = True
                raise ConnectionError("the service went away")
            return await super().train_pipelined(run_handle, batches, learning_rate)

    backend = Flaky(renderer)
    client, _ = make_client(backend, retry=RetryPolicy(max_attempts=1, call_timeout_s=5))
    config = {"data_dir": str(data_dir), "runs": RUNS, "batch_size": 4, "epochs": 1, "updates_per_call": 8}
    ctx, lines = context(tmp_path, renderer, "train", client, **config)
    (tmp_path / "runs" / "eval").mkdir(parents=True)
    (tmp_path / "runs" / "eval" / "ACTIVE").write_text("")  # another process is evaluating adapters
    assert run(run_step(Train(), ctx)) == 1
    summary = json.loads((ctx.results_dir / "summary.json").read_text())
    assert [r["name"] for r in summary["runs"]] == ["middle-s1", "pilot-r8"] and not summary["complete"]
    assert summary["failures"][0]["run"] == "wrong-labels" and backend.training_calls.count("save") == 2
    assert "Still to run: wrong-labels." in (ctx.results_dir / "PROGRESS.md").read_text()
    assert summary["overlap_with_evaluation"]["while_evaluating"]["updates"] == 3 + 2 and summary["overlap_with_evaluation"]["alone"]["updates"] == 0
    # The next invocation trains only the run that failed.
    client, _ = make_client(backend, retry=RetryPolicy(max_attempts=1, call_timeout_s=5))
    ctx, _ = context(tmp_path, renderer, "train", client, **config)
    assert run(run_step(Train(), ctx)) == 0 and backend.training_calls.count("create") == 4
    summary = json.loads((ctx.results_dir / "summary.json").read_text())
    assert summary["complete"] and [r["name"] for r in summary["runs"]] == [r["name"] for r in RUNS]


def test_a_process_can_be_limited_to_one_run(tmp_path, renderer, make_client, data_dir):
    config = {"data_dir": str(data_dir), "runs": RUNS, "batch_size": 4, "epochs": 1, "runs_per_process": 1}
    for done in (1, 2, 3):
        backend = FakeBackend(renderer)
        client, _ = make_client(backend)
        ctx, lines = context(tmp_path, renderer, "train", client, **config)
        assert run(run_step(Train(), ctx)) == (0 if done == 3 else 1) and backend.training_calls.count("create") == 1
        summary = json.loads((ctx.results_dir / "summary.json").read_text())
        assert [r["name"] for r in summary["runs"]] == [r["name"] for r in RUNS[:done]] and summary["failures"] == []


def test_a_group_of_updates_sent_twice_or_a_rising_loss_ends_the_run(tmp_path, renderer, make_client, data_dir):
    class Hiccup(FakeBackend):
        async def train_pipelined(self, run_handle, batches, learning_rate):
            if not getattr(self, "failed", False):
                self.failed = True
                raise FakeTransient("no answer")
            return await super().train_pipelined(run_handle, batches, learning_rate)

    backend = Hiccup(renderer)
    client, _ = make_client(backend, retry=RetryPolicy(max_attempts=3, call_timeout_s=5))
    ctx, lines = context(tmp_path, renderer, "train", client, data_dir=str(data_dir), runs=RUNS[:1], batch_size=4, epochs=1)
    run(run_step(Train(), ctx))
    summary = json.loads((ctx.results_dir / "summary.json").read_text())
    assert summary["runs"] == [] and "sent more than once" in summary["failures"][0]["error"] and "save" not in backend.training_calls

    class Rising(FakeBackend):
        async def forward(self, run_handle, examples, *, backward):
            out = await super().forward(run_handle, examples, backward=backward)
            return {"metrics": {"loss:sum": out["metrics"]["loss:sum"] * (1 + run_handle["steps"]) ** 2}}

    backend = Rising(renderer)
    client, _ = make_client(backend)
    ctx, lines = context(tmp_path / "rising", renderer, "train", client, data_dir=str(data_dir), runs=RUNS[:1], batch_size=2, epochs=2, updates_per_call=2)
    run(run_step(Train(), ctx))
    summary = json.loads((ctx.results_dir / "summary.json").read_text())
    assert summary["runs"] == [] and "diverged" in summary["failures"][0]["error"] and "save" not in backend.training_calls


def test_eval_compares_each_state_with_the_base_on_the_same_items(tmp_path, renderer, make_client, data_dir):
    def base(text):  # yes to everything
        return "yes"

    def adapter(text):  # right about every number
        return "yes" if int(text.split()[-2]) % 2 == 0 else "no"

    def with_examples(text):  # the worked examples are in the prompt, and help a little
        number = int(text.split()[-2])
        return adapter(text) if ("plus" in text and number < 10) else "yes"

    backend = FakeBackend(renderer, answer=with_examples, adapter_answer=adapter)
    client, _ = make_client(backend, max_concurrency=4)
    runs_file = tmp_path / "runs.json"
    runs_file.write_text(json.dumps({"runs": [{"name": "middle-s1", "tinker_path": "tinker://run/sampler_weights/a", "set": "middle-10"}]}))
    config = {"seed": 2, "data_dir": str(data_dir), "runs_file": str(runs_file), "_registry": {"parity": Parity()}, "_skip_download": True,
              "benchmarks": ["parity"], "items": {"default": 12}, "target_benchmarks": ["parity"], "states": [{"name": "prompt-only", "few_shot": True}]}
    ctx, lines = context(tmp_path, renderer, "eval", client, **config)
    assert run(run_step(Eval(), ctx, dry_run=True)) == 0 and backend.sample_calls == []
    assert run(run_step(Eval(), ctx)) == 0
    summary = json.loads((ctx.results_dir / "summary.json").read_text())
    assert list(summary["states"]) == ["base", "middle-s1", "prompt-only"]
    base_cell, adapter_cell = summary["states"]["base"]["benchmarks"]["parity"], summary["states"]["middle-s1"]["benchmarks"]["parity"]
    assert base_cell["n"] == adapter_cell["n"] == 12 and "minus_base" not in base_cell
    too_long = adapter_cell["context_overflow"]  # the benchmark's one oversized item may be in the sample
    assert adapter_cell["score"]["estimate"] == round((12 - too_long) / 12, 4) and adapter_cell["role"] == "target"
    gained = adapter_cell["score"]["estimate"] - base_cell["score"]["estimate"]
    assert abs(adapter_cell["minus_base"]["estimate"] - gained) < 1e-3 and adapter_cell["minus_base"]["lost"] == 0 and adapter_cell["regression"] is False
    assert summary["states"]["middle-s1"]["model_path"] == "tinker://run/sampler_weights/a"
    # The adapter was sampled through its path, and the prompt-only state saw the eight examples.
    assert sum(1 for call in backend.sample_calls if call["model_path"]) == 12 - too_long
    few_shot = summary["states"]["prompt-only"]["benchmarks"]["parity"]
    assert few_shot["tokens_per_answer"]["prompt"] > 3 * base_cell["tokens_per_answer"]["prompt"]
    assert base_cell["score"]["estimate"] <= few_shot["score"]["estimate"] <= 1.0
    assert "| middle-s1 |" in (ctx.results_dir / "REPORT.md").read_text()


def test_eval_marks_a_watch_cell_that_falls(tmp_path, renderer, make_client, data_dir):
    backend = FakeBackend(renderer, answer=lambda text: "yes" if int(text.split()[-2]) % 2 == 0 else "no", adapter_answer="no")
    client, _ = make_client(backend, max_concurrency=4)
    runs_file = tmp_path / "runs.json"
    runs_file.write_text(json.dumps({"runs": [{"name": "broken", "tinker_path": "tinker://run/sampler_weights/b", "set": "middle-10"}]}))
    ctx, _ = context(tmp_path, renderer, "eval", client, seed=2, data_dir=str(data_dir), runs_file=str(runs_file), _registry={"parity": Parity()}, _skip_download=True,
                     benchmarks=["parity"], items={"default": 0}, target_benchmarks=["gsm8k"])
    assert run(run_step(Eval(), ctx)) == 0
    cell = json.loads((ctx.results_dir / "summary.json").read_text())["states"]["broken"]["benchmarks"]["parity"]
    assert cell["role"] == "watch" and cell["regression"] is True and cell["minus_base"]["estimate"] < -0.4 and cell["n"] == 21
    assert "regression" in (ctx.results_dir / "REPORT.md").read_text()


# ---------------------------------------------------------------- the arms and the rules


def right(text):
    if "even" in text:
        return "yes" if int(text.split()[-2]) % 2 == 0 else "no"
    return "A" if text.split()[-1] == "ab" else "B"


def lazy(text):  # what the fast base model says: the same answer every time
    return "yes" if "even" in text else "A"


def half(text):  # right about the small numbers only
    return right(text) if "even" in text and int(text.split()[-2]) < 10 else lazy(text)


def never(text):
    return "no" if "even" in text else "B"


ARMS = {
    "middle-2000-s1": right, "random-2000-s1": half, "middle-2000-s2": right, "random-2000-s2": half, "middle-2000-s3": right, "random-2000-s3": lazy,
    "shuffled-labels-2000": right, "easy-2000": lazy, "hard-2000": never, "length-matched-random-2000": half,
    "middle-250": lazy, "random-250": lazy, "middle-4000": right, "random-4000": half,
}
EXTRA_STATES = [
    {"name": "prompt-only", "few_shot": True},
    {"name": "base-reasoning", "setting": "reasoning"},
    {"name": "middle-2000-s1-reasoning", "setting": "reasoning", "adapter": "middle-2000-s1"},
]


def arms_setup(tmp_path, renderer, make_client, data_dir, arms=ARMS, **more):
    backend = FakeBackend(renderer, answer=right, forced_answer=lazy, adapter_answer={f"tinker://{name}": answer for name, answer in arms.items()})
    client, _ = make_client(backend, max_concurrency=8)
    runs_file = tmp_path / "summary.json"
    runs_file.write_text(json.dumps({"runs": [{"name": name, "tinker_path": f"tinker://{name}", "set": name, "seed": int(name[-1]) if name[-2] == "s" else 1} for name in arms]}))
    config = {"seed": 2, "data_dir": str(data_dir), "runs_file": str(runs_file), "_registry": {"parity": Parity(), "pairs": Pairs()}, "_skip_download": True,
              "benchmarks": ["parity", "pairs"], "items": {"default": 0}, "target_benchmarks": ["parity"], "states": EXTRA_STATES, **more}
    ctx, lines = context(tmp_path, renderer, "eval", client, **config)
    return ctx, lines, backend


def test_eval_compares_the_arms_and_applies_the_rules(tmp_path, renderer, make_client, data_dir):
    ctx, lines, backend = arms_setup(tmp_path, renderer, make_client, data_dir, expected_states=[*ARMS, "prompt-only", "random-1000"])
    assert run(run_step(Eval(), ctx)) == 0
    summary = json.loads((ctx.results_dir / "summary.json").read_text())
    assert list(summary["states"])[:2] == ["base", "middle-2000-s1"] and list(summary["states"])[-3:] == [s["name"] for s in EXTRA_STATES]
    assert summary["missing"] == ["random-1000: not evaluated"] and not (ctx.runs_dir / "ACTIVE").exists()

    # The base model's gap between the two settings, and how much of it a state closes.
    gap = summary["base_gap"]["parity"]
    assert (gap["base_fast"], gap["base_reasoning"]) == (round(10 / 21, 4), round(20 / 21, 4))
    main = summary["states"]["middle-2000-s1"]["benchmarks"]["parity"]
    assert main["gap_closed"]["estimate"] == 1.0 and summary["states"]["random-2000-s1"]["benchmarks"]["parity"]["gap_closed"]["estimate"] == 0.5
    assert main["minus_base"]["gained"] == 10 and main["minus_base"]["lost"] == 0 and main["minus_base"]["mcnemar_p"] < 0.05
    assert "gap_closed" not in summary["states"]["middle-2000-s1"]["benchmarks"]["pairs"] and "minus_base" not in summary["states"]["base-reasoning"]["benchmarks"]["parity"]

    # Middle band against random, pooled over three seeds, with the seed-to-seed range.
    comparison = summary["main_comparison"]["parity"]
    pooled = comparison["middle_minus_random"]
    assert comparison["seeds"] == [1, 2, 3] and pooled["n"] == 21 and abs(pooled["estimate"] - (20 - (15 + 15 + 10) / 3) / 21) < 1e-3 and pooled["low"] > 0
    assert comparison["seed_range"] == {"low": round(5 / 21, 4), "high": round(10 / 21, 4)}
    assert comparison["middle_scores"] == [round(20 / 21, 4)] * 3 and comparison["random_scores"] == [round(15 / 21, 4), round(15 / 21, 4), round(10 / 21, 4)]
    controls = comparison["controls"]
    assert controls["shuffled-labels-2000"]["middle_minus_control"]["estimate"] == 0.0 and controls["easy-2000"]["middle_minus_control"]["gained"] == 10
    assert controls["hard-2000"]["label"] == "hard only" and controls["length-matched-random-2000"]["middle_minus_control"]["estimate"] == round(5 / 21, 4)

    # The size sweep, the prompt-only state, the watch cells and the reasoning setting.
    sweep = summary["size_sweep"]["parity"]
    assert [point["size"] for point in sweep["middle"]] == [250, 2000, 4000] and [point["state"] for point in sweep["random"]] == ["random-250", "random-2000-s1", "random-4000"]
    assert [point["score"]["estimate"] for point in sweep["middle"]] == [round(10 / 21, 4), round(20 / 21, 4), round(20 / 21, 4)]
    prompt_only = summary["prompt_only"]["parity"]
    assert prompt_only["minus_base"]["estimate"] == 0.0 and prompt_only["main_minus_prompt_only"]["gained"] == 10
    assert [cell["benchmark"] for cell in summary["watch_cells"]] == ["pairs"] and summary["watch_cells"][0]["regression"] is False and summary["watch_cells"][0]["difference"]["estimate"] == 0.5
    slow = summary["reasoning_pass"]
    assert (slow["base_state"], slow["adapter_state"]) == ("base-reasoning", "middle-2000-s1-reasoning")
    assert slow["benchmarks"]["parity"]["difference"]["estimate"] == 0.0 and slow["benchmarks"]["pairs"]["adapter"]["estimate"] == 1.0

    # The rules, applied as written.
    outcomes = {(d["rule"], d["benchmark"]): d["outcome"] for d in summary["decisions"]}
    assert outcomes == {
        ("P2", "parity"): "holds", ("P2, against prompt-only", "parity"): "holds", ("P3", "parity"): "holds",
        ("P3, against easy only", "parity"): "supports", ("P3, against hard only", "parity"): "supports", ("P4", "parity"): "the two match", ("P5", None): "holds",
    }
    report = (ctx.results_dir / "REPORT.md").read_text()
    assert all(d["line"] in report for d in summary["decisions"]) and "the gain is the model getting used to the format" in report
    figure = json.loads((ctx.results_dir / "figure-data.json").read_text())
    assert figure["size_sweep"]["parity"]["middle"][0] == {"size": 250, "score": round(10 / 21, 4), "low": figure["size_sweep"]["parity"]["middle"][0]["low"], "high": figure["size_sweep"]["parity"]["middle"][0]["high"]}
    assert len(figure["targets_by_state"]) == 1 + len(ARMS) + 1 and figure["watch_cells"][0]["title"] == "Pairs"
    # The base model in the reasoning setting and every adapter were sampled as asked.
    assert sum(1 for call in backend.sample_calls if call["model_path"] == "tinker://middle-2000-s1") == 2 * (20 + 30)


def test_eval_can_run_ahead_of_training_and_leave_the_report(tmp_path, renderer, make_client, data_dir):
    first = {name: ARMS[name] for name in ("random-2000-s1",)}  # the main adapter is not trained yet
    ctx, lines, backend = arms_setup(tmp_path, renderer, make_client, data_dir, arms=first, no_report=True)
    assert run(run_step(Eval(), ctx)) == 0
    status = json.loads((ctx.runs_dir / "status.json").read_text())
    assert list(status["states"]) == ["base", "random-2000-s1", "prompt-only", "base-reasoning"]  # no state for an adapter that does not exist
    assert all(state["done"] == state["of"] == 51 for state in status["states"].values())
    assert not (ctx.results_dir / "summary.json").exists() and not (ctx.runs_dir / "ACTIVE").exists()
    assert any("No report was written" in line for line in lines)
    # Once the main adapter exists, only its replies are missing, and the report can be written.
    made = len(backend.sample_calls)
    ctx, lines, backend = arms_setup(tmp_path, renderer, make_client, data_dir, arms={"random-2000-s1": half, "middle-2000-s1": right})
    assert run(run_step(Eval(), ctx)) == 0
    assert len(backend.sample_calls) == 2 * 50  # the main adapter in the fast setting and in the reasoning setting
    summary = json.loads((ctx.results_dir / "summary.json").read_text())
    assert summary["main_comparison"]["parity"]["seeds"] == [1] and summary["main_comparison"]["parity"]["middle_minus_random"]["estimate"] == round(5 / 21, 4)
    assert {d["outcome"] for d in summary["decisions"] if d["rule"] in ("P4", "P3, against easy only")} == {"not decided"}


def test_the_rules_say_wrong_when_the_numbers_say_so():
    def difference(estimate, low, high, p=0.5):
        return {"estimate": estimate, "low": low, "high": high, "mcnemar_p": p, "gained": 1, "lost": 1}

    summary = {
        "main_adapter": "m", "targets": ["t"], "titles": {"t": "Target", "w1": "Watch one", "w2": "Watch two"},
        "states": {"m": {"benchmarks": {"t": {"minus_base": difference(0.03, -0.01, 0.07, 0.2)}}}},
        "prompt_only": {"t": {"main_minus_prompt_only": difference(-0.02, -0.05, 0.01, 0.3)}},
        "main_comparison": {"t": {"seeds": [1, 2, 3], "middle_minus_random": {"estimate": 0.01, "low": -0.02, "high": 0.04}, "seed_range": {"low": -0.01, "high": 0.03},
                                  "controls": {"easy-2000": {"label": "easy only", "middle_minus_control": difference(0.02, -0.01, 0.05)},
                                               "hard-2000": {"label": "hard only", "middle_minus_control": difference(-0.06, -0.09, -0.03)},
                                               "shuffled-labels-2000": {"label": "shuffled labels", "middle_minus_control": difference(0.2, 0.15, 0.25)}}}},
        "watch_cells": [{"benchmark": "w1", "regression": True, "difference": difference(-0.031, -0.06, 0.0)}, {"benchmark": "w2", "regression": False, "difference": difference(0.01, -0.02, 0.04)}],
    }
    decisions = {(d["rule"], d["benchmark"]): d for d in decide(summary)}
    assert decisions[("P2", "t")]["outcome"] == "wrong" and "McNemar's test does not separate" in decisions[("P2", "t")]["line"]
    assert decisions[("P2, against prompt-only", "t")]["outcome"] == "wrong"
    assert decisions[("P3", "t")]["outcome"] == "wrong" and "the interval includes zero" in decisions[("P3", "t")]["line"] and "seed by seed from -1.0 to +3.0" in decisions[("P3", "t")]["line"]
    assert decisions[("P3, against easy only", "t")]["outcome"] == "counts against" and decisions[("P3, against hard only", "t")]["outcome"] == "counts against"
    assert "hard only is ahead" in decisions[("P3, against hard only", "t")]["line"]
    assert decisions[("P4", "t")]["outcome"] == "the two differ" and "more than the format" in decisions[("P4", "t")]["line"]
    assert decisions[("P5", None)]["outcome"] == "wrong" and "1 of 2 watch cells" in decisions[("P5", None)]["line"] and "Watch one -3.1 points" in decisions[("P5", None)]["line"]
    # A score that falls is wrong whatever the test says, and a pooled difference below zero is wrong for the middle band.
    summary["states"]["m"]["benchmarks"]["t"]["minus_base"] = difference(-0.1, -0.15, -0.05, 0.001)
    summary["main_comparison"]["t"]["middle_minus_random"] = {"estimate": -0.05, "low": -0.08, "high": -0.02}
    decisions = {(d["rule"], d["benchmark"]): d for d in decide(summary)}
    assert decisions[("P2", "t")]["outcome"] == "wrong" and "the score does not rise" in decisions[("P2", "t")]["line"]
    assert decisions[("P3", "t")]["outcome"] == "wrong" and "random is ahead" in decisions[("P3", "t")]["line"]


def test_share_of_gap():
    fast, reference = [0, 0, 0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1]
    result = share_of_gap(fast, reference, [1, 1, 0, 0, 1, 1, 1, 1], seed=1)
    assert result["estimate"] == 0.5 and 0 <= result["low"] <= 0.5 <= result["high"] <= 1
    assert share_of_gap(fast, fast, fast, seed=1) is None and share_of_gap([], [], [], seed=1) is None


def test_a_config_can_keep_its_runs_in_their_own_folder(tmp_path, renderer, data_dir):
    from posttrain_lab import cli

    config = tmp_path / "pilot.toml"
    config.write_text(
        f'model_id = "thinkingmachines/Inkling-2-Mini:peft:65536:free"\nfolder = "train-pilot"\ndata_dir = "{data_dir}"\nbatch_size = 4\nepochs = 1\n'
        '[[runs]]\nname = "pilot-a"\nset = "pilot-train"\nrank = 8\nlearning_rate = 1e-3\n'
    )
    backend = FakeBackend(renderer)
    code = cli.main(
        ["train", "--config", str(config), "--runs-dir", str(tmp_path / "runs"), "--results-dir", str(tmp_path / "results")],
        make_renderer=lambda model_id: renderer, make_backend=lambda allow_paid: backend, out=lambda line: None,
    )
    assert code == 0
    assert (tmp_path / "runs" / "train-pilot" / "results.jsonl").exists() and not (tmp_path / "runs" / "train").exists()
    assert json.loads((tmp_path / "results" / "train-pilot" / "summary.json").read_text())["runs"][0]["name"] == "pilot-a"
