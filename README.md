# posttrain-lab

This repository holds a post-training study of Inkling 2 Mini, a small open-weight model that Tinker lets anyone train and sample at no charge. The study asks which jobs the model already does well enough on public benchmarks, and whether cheap synthetic data with program-made labels can close the gaps that remain. Everything runs on Tinker's free model ID, so a stranger with one free key can rerun it and extend it. Results are scored only on public benchmarks, before and after training, including jobs that were never trained.

## Status

The smoke test and the survey have run, both on 2026-11-26. The later steps (`build-data`, `filter`, `train`, `eval`, `report`) are registered as subcommands and print "not implemented yet".

The numbers are in [results/smoke/REPORT.md](results/smoke/REPORT.md) and [results/survey/REPORT.md](results/survey/REPORT.md), with the full data in the `summary.json` next to each.

## What the survey found

The survey scores the base model on public benchmarks in three settings. `fast` is a forced direct answer at effort 0.0, with no reasoning block. `slow` is effort 0.9, free-running. `effort_0` is free-running effort 0.0, the model's shipped no-effort setting, which still reasons, and it ran on a fixed subsample of each benchmark. Scores are percentages, and the report has the intervals.

| benchmark | job | fast | slow | slow minus fast | effort_0 (subsample) | by the target rule |
|---|---|---|---|---|---|---|
| JudgeBench, pairwise | Judging answers | 69.0 | 64.3 | -4.6 | 72.5 | usable as is |
| JudgeBench, one response at a time | Judging answers | 61.3 | 61.1 | -0.2 | 77.5 | usable as is |
| RewardBench 2, best of four | Judging answers | 70.9 | 70.8 | -0.1 | 82.2 | usable as is |
| LLMBar | Judging answers | 74.6 | 86.2 | +11.6 | 84.5 | training target |
| BFCL V4, single turn | Calling tools | 74.1 | 78.3 | +4.2 | 80.3 | in between |
| JSONSchemaBench (sample of 1,000) | Filling a schema | 95.6 | 97.9 | +2.3 | 98.5 | usable as is |
| GSM8K | Reasoning without the wait | 37.5 | 95.8 | +58.3 | 96.5 | training target |
| MBPP (sanitized test split) | Reasoning without the wait | 82.1 | 92.2 | +10.1 | 93.5 | training target |
| IFEval | Instructions and knowledge | 68.4 | 75.8 | +7.4 | 87.5 | watch only (in between) |
| MMLU-Pro (sample of 500) | Instructions and knowledge | 56.4 | 73.0 | +16.6 | 80.5 | watch only (training target) |
| BFCL V4, multi turn (sample of 200) | Calling tools | 28.5 | 40.5 | +12.0 | 33.2 | training target |

LLM-AggreFact and WildGuardTest were not run. Both are gated on Hugging Face and need a token.

The rule was fixed before the run. A job is a training target when slow beats fast by 10 points or more, and usable as is when the gap is 3 points or less. Three things in the table need care. At effort 0.9 between a fifth and a third of the replies on the judging and knowledge benchmarks ran past the 8,192-token limit without answering, which pulls the slow score down. Free-running effort 0.0 reasons for about half as long, is cut off far less often, and on every single-turn benchmark in the subsample it was either ahead of effort 0.9 or within a point of it. And published leaderboard scores were left out, so the half of the rule that compares against the frontier was not applied.

## What the smoke test found

The smoke test came first and shaped how the survey was run.

At effort 0.0 the base model still reasons before it answers. A reply with no reasoning needs the answer opening to be forced. `ModelClient.sample(..., force_answer=True)` does that by ending the prompt with the two tokens that open a text answer.

The probability of a one-token verdict is read with `ModelClient.score_choices`, which scores the candidate tokens at the answer position in one request. Reading it from a sampled reply is of no use, because once the model has reasoned its way to a verdict the verdict token has a probability of 1.0.

Sampling scaled close to linearly up to 64 requests in flight with no throttling, and the survey then held 32 in flight for two and a half hours without a throttling event. Training runs in cycles of five to ten seconds at any batch size tested, so large batches and back-to-back submission (`TrainingRun.train_pipelined`) are what make it fast.

## Quickstart

You need a Tinker API key and a Python 3.12 that `tml-renderers` has a wheel for. Tinker's pricing page says a prepaid account must hold a positive balance to use the free model ID, though calls to that ID are not charged. Version 0.5.0 publishes wheels for Apple-silicon macOS and for Linux, and none for Intel macOS, so on a Mac the interpreter must be an arm64 build. The commands below use [uv](https://docs.astral.sh/uv/) to fetch one.

```bash
uv venv --python-preference only-managed --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.lock

export TINKER_API_KEY="..."            # the key is read from the environment and nowhere else

.venv/bin/python -m pytest             # offline tests, no key needed
.venv/bin/python -m posttrain_lab --help
.venv/bin/python -m posttrain_lab smoke --dry-run   # planned calls and a token estimate, no API call
.venv/bin/python -m posttrain_lab smoke             # about twenty minutes, free

docker pull python:3.11-slim                        # once; MBPP answers run in this image with no network
.venv/bin/python -m posttrain_lab survey --dry-run
caffeinate -i .venv/bin/python -m posttrain_lab survey   # about three hours, free; caffeinate keeps a Mac awake
```

The survey downloads its benchmark files itself, without a login, into `runs/survey/sources/`. Each file is pinned to a revision and checked against a SHA-256 value in `posttrain_lab/benchmarks/pins.json`. A survey that stops for any reason picks up where it left off when the same command is run again, and `--only gsm8k` limits a run to one benchmark.

`requirements.txt` pins the three direct dependencies and `requirements.lock` pins everything that was installed alongside them.

If the checkout lives inside iCloud Drive, keep the heavy directories out of sync by giving them names that end in `.nosync` and linking the usual names to them, for example `.venv.nosync` with a `.venv` symlink and `runs.nosync` with a `runs` symlink. The `.gitignore` already covers both spellings.

## Hard constraints

These are fixed for the whole study.

1. Nothing may cost money. The only model ID that is called is `thinkingmachines/Inkling-2-Mini:peft:65536:free`. The client refuses any model ID that does not end in `:free` unless `--allow-paid` is passed, and it refuses to sample an adapter whose base model is not a free ID. No other provider's SDK is imported, and a test enforces that.
2. Evaluation uses public benchmarks that can be graded without a paid model, by a program, a validator, unit tests or the labels that ship with the benchmark.
3. Training data is synthetic. No benchmark item, and nothing close to one, may enter training.
4. Every adapter is checked for forgetting by rerunning the benchmarks it was not trained for.
5. Benchmark items stay under `runs/survey/sources/`. No step that builds training data may read them, other than to check for leaks.
6. Model-written code runs only inside a Docker container with no network.
7. Keys come from the environment. They are never written to disk or to a log.
8. Tokens in and out are recorded for every call, so cost per answer is a measured number.

## How the code is laid out

`posttrain_lab/client.py` is the one place that makes model calls. It applies the free-ID guard, bounds how many calls are in flight, retries throttled and transient failures with backoff, stops on an account problem, logs one record per call, and caches every sampling response on disk. The cache key is a hash of the model ID, the rendered prompt tokens, the sampling parameters, the effort and the seed, so an identical request is answered from disk.

`posttrain_lab/backends/tinker_backend.py` is the only module that imports the Tinker SDK. `posttrain_lab/testing.py` has a fake renderer and backend, which is what the tests run on.

`posttrain_lab/benchmarks/` holds the public benchmarks. Each one is a small class that loads its items, gives every item one prompt, and grades a reply with a program. `sources.py` lists where every file comes from. BFCL and IFEval are graded by their own public checkers, imported from the pinned copies, and BFCL replies are decoded and executed without `eval`.

Each step is a subcommand that follows one convention, defined in `posttrain_lab/steps/base.py`. A step reads a TOML config from `configs/`, plans its work as a list of units, and appends one row per finished unit to `runs/<step>/results.jsonl`, keyed by a hash of the unit's inputs. A rerun skips the units that are already there. Every step accepts `--dry-run`, which prints the planned call counts and a token estimate and calls nothing. `--fresh` runs units again even if their rows are stored, and `--only KIND` limits a run to one kind of unit.

## Where results live

`runs/` holds everything large or raw, and is not committed. That covers the response cache in `runs/cache/`, and for each step its result rows, its call log (`calls.jsonl`) and the service events it saw.

`results/` holds the small summaries that are committed, one folder per step.
