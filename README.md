# posttrain-lab

This repository holds a post-training study of Inkling 2 Mini, a small open-weight model that Tinker lets anyone train and sample at no charge. The study asks which jobs the model already does well enough on public benchmarks, and whether cheap synthetic data with program-made labels can close the gaps that remain. Everything runs on Tinker's free model ID, so a stranger with one free key can rerun it and extend it. Results are scored only on public benchmarks, before and after training, including jobs that were never trained.

## Status

Only the smoke test exists so far. It measures what the free tier can do, which is what the rest of the plan will be sized on. The later steps (`survey`, `build-data`, `filter`, `train`, `eval`, `report`) are registered as subcommands and print "not implemented yet".

The smoke test's numbers are in [results/smoke/REPORT.md](results/smoke/REPORT.md), with the full data in `results/smoke/summary.json`.

## Quickstart

You need a Tinker API key and a Python 3.12 that `tml-renderers` has a wheel for. Version 0.5.0 publishes wheels for Apple-silicon macOS and for Linux, and none for Intel macOS, so on a Mac the interpreter must be an arm64 build. The commands below use [uv](https://docs.astral.sh/uv/) to fetch one.

```bash
uv venv --python-preference only-managed --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.lock

export TINKER_API_KEY="..."            # the key is read from the environment and nowhere else

.venv/bin/python -m pytest             # offline tests, no key needed
.venv/bin/python -m posttrain_lab --help
.venv/bin/python -m posttrain_lab smoke --dry-run   # planned calls and a token estimate, no API call
.venv/bin/python -m posttrain_lab smoke             # about fifteen minutes, free
```

`requirements.txt` pins the three direct dependencies and `requirements.lock` pins everything that was installed alongside them.

If the checkout lives inside iCloud Drive, keep the heavy directories out of sync by giving them names that end in `.nosync` and linking the usual names to them, for example `.venv.nosync` with a `.venv` symlink and `runs.nosync` with a `runs` symlink. The `.gitignore` already covers both spellings.

## Hard constraints

These are fixed for the whole study.

1. Nothing may cost money. The only model ID that is called is `thinkingmachines/Inkling-2-Mini:peft:65536:free`. The client refuses any model ID that does not end in `:free` unless `--allow-paid` is passed, and it refuses to sample an adapter whose base model is not a free ID. No other provider's SDK is imported, and a test enforces that.
2. Evaluation uses public benchmarks that can be graded without a paid model, by a program, a validator, unit tests or the labels that ship with the benchmark.
3. Training data is synthetic. No benchmark item, and nothing close to one, may enter training.
4. Every adapter is checked for forgetting by rerunning the benchmarks it was not trained for.
5. Keys come from the environment. They are never written to disk or to a log.
6. Tokens in and out are recorded for every call, so cost per answer is a measured number.

## How the code is laid out

`posttrain_lab/client.py` is the one place that makes model calls. It applies the free-ID guard, bounds how many calls are in flight, retries throttled and transient failures with backoff, stops on an account problem, logs one record per call, and caches every sampling response on disk. The cache key is a hash of the model ID, the rendered prompt tokens, the sampling parameters, the effort and the seed, so an identical request is answered from disk.

`posttrain_lab/backends/tinker_backend.py` is the only module that imports the Tinker SDK. `posttrain_lab/testing.py` has a fake renderer and backend, which is what the tests run on.

Each step is a subcommand that follows one convention, defined in `posttrain_lab/steps/base.py`. A step reads a TOML config from `configs/`, plans its work as a list of units, and appends one row per finished unit to `runs/<step>/results.jsonl`, keyed by a hash of the unit's inputs. A rerun skips the units that are already there. Every step accepts `--dry-run`, which prints the planned call counts and a token estimate and calls nothing.

## Where results live

`runs/` holds everything large or raw, and is not committed. That covers the response cache in `runs/cache/`, and for each step its result rows, its call log (`calls.jsonl`) and the service events it saw.

`results/` holds the small summaries that are committed, one folder per step.
