# posttrain-lab

This repository holds a post-training study of Inkling 2 Mini, a small open-weight model that Tinker lets anyone train and sample at no charge. The study asks which jobs the model already does well enough on public benchmarks, and whether cheap synthetic data with program-made labels can close the gaps that remain. Everything runs on Tinker's free model ID, so a stranger with one free key can rerun it and extend it. Results are scored only on public benchmarks, before and after training, including jobs that were never trained.

## Status

All planned runs are done. The smoke test and the survey ran on 2026-11-26, and the survey includes a full pass in the reasoning setting, which is the study's reference. The training data for the first target, reasoning without the wait, was generated, checked for leaks and labeled on 2026-11-27, a pilot fixed the LoRA rank at 16 and the learning rate at 1.8e-4, and the 18 training runs in `configs/train.toml` and the evaluation of all 20 model states finished the same day. Every call used the free model ID, and the service's usage report prices all of it at $0.

The filter did not pick better training data than chance. On GSM8K in the fast setting, three adapters trained on 2,000 middle-band problems scored 42.1%, 43.0% and 44.8%, and three trained on token-matched random problems scored 42.8%, 44.5% and 44.6%, against 37.5% untrained. Pooled over seeds, middle band minus random is -0.7 points (-1.8 to +0.5) on GSM8K and +0.9 (-0.8 to +2.6) on MBPP. Easy-only and length-matched random sets gained as much, a hard-only set gained nothing, and shuffled answers sent nearly every score to zero. Left free to reason, the main adapter often ends its turn without an answer (165 of 1,319 GSM8K items, 263 of 500 LLMBar items), which costs it 12.4 and 44.2 points there. The decision rules are applied in [results/eval/REPORT.md](results/eval/REPORT.md): P2 holds on both targets, P3 is wrong on both, P4 holds, and P5 is wrong on one watch cell.

The numbers are in [results/smoke/REPORT.md](results/smoke/REPORT.md), [results/survey/REPORT.md](results/survey/REPORT.md), [results/data/REPORT.md](results/data/REPORT.md), [results/train-pilot/REPORT.md](results/train-pilot/REPORT.md), [results/train/REPORT.md](results/train/REPORT.md) and [results/eval/REPORT.md](results/eval/REPORT.md), with the full data in the JSON file next to each. [docs/training-recipe.md](docs/training-recipe.md) states the recipe.

## What the survey found

The survey scores the base model on public benchmarks in three settings. `fast` is a forced direct answer at effort 0.0, with no reasoning block. `effort 0.9` is free-running at the model's high effort setting. `reasoning` is free-running effort 0.0, the model's shipped no-effort setting, which still reasons before it answers. Both free-running settings have an 8,192-token limit. Scores are percentages, and the report has the intervals.

| benchmark | job | items | fast | effort 0.9 | reasoning | reasoning minus fast |
|---|---|---|---|---|---|---|
| JudgeBench, pairwise | Judging answers | 1,240 | 69.0 | 64.3 | 74.7 | +5.7 |
| JudgeBench, one response at a time | Judging answers | 1,240 | 61.3 | 61.1 | 73.3 | +12.0 |
| RewardBench 2, best of four | Judging answers | 1,763 | 70.9 | 70.8 | 78.5 | +7.6 |
| LLMBar | Judging answers | 838 | 74.6 | 86.2 | 85.9 | +11.3 |
| BFCL V4, single turn | Calling tools | 3,491 | 74.1 | 78.3 | 78.3 | +4.2 |
| BFCL V4, multi turn (sample of 200) | Calling tools | 200 | 28.5 | 40.5 | 37.0 | +8.5 |
| JSONSchemaBench (sample of 1,000) | Filling a schema | 1,000 | 95.6 | 97.9 | 97.9 | +2.3 |
| GSM8K | Reasoning without the wait | 1,319 | 37.5 | 95.8 | 96.7 | +59.2 |
| MBPP (sanitized test split) | Reasoning without the wait | 257 | 82.1 | 92.2 | 94.5 | +12.4 |
| IFEval | Instructions and knowledge | 541 | 68.4 | 75.8 | 87.4 | +19.0 |
| MMLU-Pro (sample of 500) | Instructions and knowledge | 500 | 56.4 | 73.0 | 80.2 | +23.8 |

LLM-AggreFact and WildGuardTest were not run. Both are gated on Hugging Face and need a token.

The reference for the rest of the study is the reasoning setting. Effort 0.9 was the first choice and was measured in full. On the judging and knowledge benchmarks between a fifth and a third of its replies reached the 8,192-token limit without an answer, which pulls its score down. Free-running effort 0.0 reasons for about half as long and is cut off less often. Paired on the same items, it is ahead of effort 0.9 on six of the eleven benchmarks with an interval that excludes zero and behind on none. Its largest shortfall is 3.5 points on the multi-turn BFCL sample, with an interval that includes zero. Effort 0.9 stays in the tables as measured.

The target rule is applied to jobs. A job's gap is the unweighted mean, over the benchmarks listed for it beforehand, of reference minus fast in points. A gap of 10 or more makes the job a training target, and a gap of 3 or less means the fast setting is usable as it is.

| job | benchmarks | gap against effort 0.9 | gap against the reasoning setting | class |
|---|---|---|---|---|
| Reasoning without the wait | GSM8K, MBPP | +34.2 | +35.8 | training target |
| Judging answers | JudgeBench pairwise, RewardBench 2, LLMBar | +2.3 | +8.2 | in between (usable as is against effort 0.9) |
| Calling tools | BFCL single turn, BFCL multi turn | +8.1 | +6.4 | in between |
| Filling a schema | JSONSchemaBench | +2.3 | +2.3 | usable as is |

Reasoning without the wait is the one training target. IFEval and MMLU-Pro are watched for forgetting and never trained. The pointwise form of JudgeBench is outside the list that was fixed beforehand, so anything built on it would be exploratory.

## The training data

The training data for the reasoning target is made of generated problems whose answers are fixed by the program that writes them. There are eleven families (arithmetic, word problems about stock and rates, sequences, number theory, tables, code tracing, clock and calendar arithmetic, counting, equations and ordering) at up to seven levels of difficulty. No benchmark item is used to make them. Each candidate is compared with every survey benchmark item, and 72 of 11,942 were dropped for resembling one.

Each of the 11,870 that remained was run in the fast setting and in the reasoning setting, and graded by its program. That sorts the problems into four cells.

| | reasoning right | reasoning wrong |
|---|---|---|
| fast right | easy, 3,162 | rare, 2 |
| fast wrong | middle band, 6,447 | hard, 2,259 |

The middle band is where reasoning helps, and it is the data the main training arm uses. Nearly all of the hard cell (2,139 of 2,259) is there because the reasoning reply reached its 8,192-token limit before giving an answer, so that cell mostly holds problems that run past the reasoning setting's token limit. Every training target is the answer the program computed; the reasoning setting only decides which cell a problem lands in.

The training sets are drawn with fixed seeds and saved as lists of item ids in `results/data/sets.json`. Sets that are compared with each other are matched on training tokens. There are middle-band sets of 250, 500, 1,000, 2,000 and 4,000 items, each inside the next, and random draws from all cells with the same token counts. At the size of the 2,000-item middle-band set there are also an easy-only set, a hard-only set, a copy of the middle-band set with its answers moved between items, and a random set matched item by item on length. Held out of all of them are 150 dev items from each cell, 500 middle-band items for the pilot that fixes rank and learning rate, and eight worked examples for a prompt-only arm. A training example is the prompt followed by the answer alone, rendered for the fast setting.

[`docs/training-recipe.md`](docs/training-recipe.md) states the full training recipe in one place: what is trained on and what is only scored, an example as the model sees it, every set, the LoRA and optimizer settings, and why a run takes about 40 minutes.

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
caffeinate -i .venv/bin/python -m posttrain_lab survey   # about four hours, free; caffeinate keeps a Mac awake

.venv/bin/python -m posttrain_lab build-data             # a minute, no API call: generates problems and runs the leak check
caffeinate -i .venv/bin/python -m posttrain_lab filter   # runs every problem in both settings and builds the training sets
.venv/bin/python -m posttrain_lab train --config configs/train-pilot.toml   # six short runs that fix rank and learning rate
caffeinate -i scripts/train_all.sh       # the 18 planned runs, one adapter each; a run that fails is started again
caffeinate -i scripts/eval_follow.sh     # in a second shell: evaluates each adapter as it finishes, then writes the report
```

The two scripts wrap `python -m posttrain_lab train` and `python -m posttrain_lab eval`, which can also be run on their own. While the runs train, `results/train/PROGRESS.md` says how far they are.

`build-data` reads the benchmark files the survey downloaded, so the survey's sources must be in place first. `train` reads the committed sets in `results/data/`, so the filter does not have to be rerun to train on the same items.

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

`posttrain_lab/data/` builds the training data. `generators.py` writes problems in eleven families at up to seven levels of difficulty, each with an answer computed by the program that wrote it. `leak.py` compares every candidate with every survey benchmark item. `sets.py` sorts labeled items into the four cells and draws the training sets with fixed seeds.

The steps run in this order. `survey` scores the base model. `build-data` generates candidates and drops any that resemble a benchmark item. `filter` runs each candidate in the fast and reasoning settings, labels it, and writes the training sets. `train` makes one LoRA adapter per run from a saved set, with examples rendered for the fast setting and the answer alone as the target. `eval` reruns the benchmarks in the fast setting for the base model, for every adapter and for a prompt-only state, and compares each with the base model on the same items.

Each step is a subcommand that follows one convention, defined in `posttrain_lab/steps/base.py`. A step reads a TOML config from `configs/`, plans its work as a list of units, and appends one row per finished unit to `runs/<step>/results.jsonl`, keyed by a hash of the unit's inputs. A rerun skips the units that are already there. Every step accepts `--dry-run`, which prints the planned call counts and a token estimate and calls nothing. `--fresh` runs units again even if their rows are stored, and `--only KIND` limits a run to one kind of unit.

## Where results live

`runs/` holds everything large or raw, and is not committed. That covers the response cache in `runs/cache/`, and for each step its result rows, its call log (`calls.jsonl`) and the service events it saw.

`results/` holds the small summaries that are committed, one folder per step. `results/data/` also holds the labeled items and the training sets themselves, with a manifest of SHA-256 values, because they are small and every later step reads them.

## License

The code and the generated data are under the MIT license in [`LICENSE`](LICENSE). The public benchmarks keep their own licenses and are downloaded at run time, not stored here.
