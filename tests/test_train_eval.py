"""The train and eval steps on the fake backend."""

from __future__ import annotations

import gzip
import json
import random

import pytest

from posttrain_lab.data.sets import build_sets
from posttrain_lab.steps.base import StepContext, run_step
from posttrain_lab.steps.evaluate import Eval
from posttrain_lab.steps.train import Train, fast_example, schedule, set_items, updates_per_epoch
from posttrain_lab.client import RetryPolicy
from posttrain_lab.store import JsonlStore
from posttrain_lab.testing import FakeBackend, FakeTransient

from conftest import run
from test_survey import Parity


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
    assert list(summary["states"]) == ["base", "prompt-only", "middle-s1"]
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
