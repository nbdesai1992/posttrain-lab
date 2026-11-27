# Training recipe

This page states what is trained, on what, and with which settings, for the study's first training target, reasoning without the wait. Every run uses the same recipe. Only the training set and the seed change between runs. The settings live in `configs/train.toml`, and the code is `posttrain_lab/steps/train.py`.

## What the model is trained on, and what it is scored on

These are kept apart on purpose.

- **Training data is generated, not taken from benchmarks.** Programs write reasoning problems in 11 families (arithmetic expressions, date arithmetic, tracing a short Python program, counting, linear systems, number theory, ordering puzzles, rates and distances, sequences, stock word problems, tables) and compute each answer before any model sees the problem. `posttrain_lab/data/generators.py` holds the generators, and `configs/build-data.toml` the counts and the seed (20261126).
- **Benchmarks are only for scoring.** No benchmark item, and nothing derived from one, is in any training set. A leak check compared every generated problem with the question text of 9,679 survey benchmark items and dropped the 72 near copies (character 5-gram Jaccard of 0.5 or more with digits masked; no exact or normalized matches).
- **The two target benchmarks** are GSM8K (1,319 grade-school math word problems) and MBPP (257 short Python programming problems, sanitized test split). They measure the trained job. No training item asks for code, so any MBPP change is transfer.
- **Nine watch scores** check for damage elsewhere: JudgeBench (pairwise and one response at a time), RewardBench 2, LLMBar, BFCL V4 single and multi turn, JSONSchemaBench, IFEval and MMLU-Pro. A fall of more than 2 points under the main adapter is reported as a regression.

## What one training example looks like

A user message with the problem and a one-line answer instruction, then the assistant turn in the fast setting: reasoning effort 0.0 with the two tokens that open an answer, so there is no reasoning block, then the answer and the end of the turn.

```
user:      On a 24-hour clock, a session starts at 16:40 and runs for 141 minutes with no
           breaks. Counting from the midnight before it started, how many minutes after
           midnight does it end?

           Solve the problem. Give only the final answer as a number, with no units and no
           explanation.
assistant: [answer opens at once] 1141 [end of turn]
```

Only the answer and the two tokens that close the turn carry loss. The prompt and the forced answer opening carry none, because they are given at sampling time. A middle-band item averages about 110 training tokens. The answer instruction is the same one-line instruction the GSM8K survey prompt uses.

## Which problems go into which set

Every generated problem was run once in the fast setting (a forced direct answer, 256-token limit) and once in the slow setting (effort 0.0 left free to reason, 8,192-token limit), graded by program, and put in one of four cells.

| cell | fast setting | slow setting | problems |
|---|---|---|---:|
| easy | right | right | 3,162 |
| middle band | wrong | right | 6,447 |
| hard | wrong | wrong | 2,259 |
| rare | right | wrong | 2 |

2,139 of the 2,259 hard problems are there because the slow reply reached its token limit before giving an answer. 958 problems are held out of every set: 450 as a check set (150 per cell), 500 middle-band problems for the pilot below, and 8 worked examples for the prompt-only state.

The sets each run trains on are in `results/data/sets.json`. Sets that are compared are matched on training tokens, not item counts.

| run | set | items | training tokens per pass |
|---|---|---:|---:|
| middle-2000, seeds 1 to 3 | middle band, three independent draws | 2,000 each | about 221,000 |
| random-2000, seeds 1 to 3 | all cells, three draws, each matched to its middle-band draw | 2,143 to 2,211 | about 221,000 |
| easy-2000 | easy cell only | 2,865 | about 221,000 |
| hard-2000 | hard cell only | 1,949 | about 221,000 |
| shuffled-labels-2000 | the seed-1 middle-band items with their answers permuted among them | 2,000 | 221,061 |
| length-matched-random-2000 | all cells, matched to the middle band's length distribution | 2,000 | 221,061 |
| middle-250 to middle-4000 | nested middle-band sets | 250, 500, 1,000, 4,000 | 29,248 to 435,137 |
| random-250 to random-4000 | nested random sets, token-matched to the middle-band set of the same size | 274, 537, 1,096, 4,301 | 29,214 to 435,090 |

Draws 2 and 3 of the 2,000-item sets share 676 to 694 middle-band items and 418 to 457 random items with each other draw.

## Settings

| setting | value |
|---|---|
| base model | `thinkingmachines/Inkling-2-Mini:peft:65536:free` on Tinker 0.40.1 |
| method | LoRA through Tinker's training API, on the attention, MLP and unembedding layers (the service's default) |
| rank | 16 |
| optimizer | Adam, beta1 0.9, beta2 0.95, epsilon 1e-12, no weight decay, no gradient clipping (the service's defaults) |
| learning rate | 1.8e-4, falling linearly to zero over the run, no warmup |
| update size | about 3,500 training tokens, so about 32 middle-band items; each pass over a set is cut into the number of updates its tokens call for |
| passes | 4, reshuffled each pass with the run's seed |
| updates per run | 252 for every 2,000-item set and its token-matched controls |
| training tokens per run | about 884,000 at 2,000 items (221,000 per pass times 4) |
| seed | changes the adapter's starting weights, the batch order and, for the main comparison, the draw of the data |
| checkpoint | one sampler checkpoint at the end of each run, kept 30 days |
| sampling for scoring | temperature 0, the same prompts and items as the baseline survey |

Rank and learning rate were fixed once by a pilot (`configs/train-pilot.toml`, `results/train-pilot/REPORT.md`): six short runs on the 500 held-out middle-band problems at rank 8, 16 and 32 and learning rates from 6e-5 to 1.6e-3, scored only on the held-out generated check set, never on a benchmark. Rank 16 at 1.8e-4 answered 12.7% of 150 held-out middle-band problems directly, up from none, and kept 89.3% of 150 easy ones. The top three settings were within noise of each other at that sample size.

## The runs

18 runs, in the order they train: the main comparison (middle band and random at 2,000 items, three seeds each), the controls (shuffled labels, easy only, hard only, length-matched random), then the size sweep (middle band and random at 250, 500, 1,000 and 4,000 items; the 2,000-item points are the seed-1 main runs). Each run is scored on 5,776 benchmark items in the fast setting: GSM8K and MBPP in full, 500 from each of eight other benchmarks, and the 200 multi-turn tool items. The base model, the base model with the eight worked examples in its prompt, and the main adapter left free to reason are scored on the same items.

## Why a run takes about 40 minutes

The work per update is small, about 3,500 tokens, and the time goes to the service. On the free ID an update takes a median of about 6 seconds as a round trip, a run makes 252 of them, and some wait 30 seconds or more in the service's queue. A run also waits about two minutes for a training client before its first update. Runs go one at a time, because a second training client from the same account sat in the queue for minutes with rate-limit pauses when it was tried. Scoring shares the same free quota, so updates slow down while adapters are being scored.
