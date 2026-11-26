# Survey report

The base model `thinkingmachines/Inkling-2-Mini:peft:65536:free` on public benchmarks, generated 2026-11-26T23:53:47+00:00 with tinker 0.40.1, tml-renderers 0.5.0 and tinker-cookbook 0.7.1. Every score comes from this repository's harness at temperature 0 with one prompt per benchmark, and every reply was graded by a program. Scores are percentages with 95% bootstrap intervals. The full data is in `summary.json`, and the raw replies are in `runs/survey/`.

The settings:

- `fast` is a forced direct answer at effort 0.0 (no reasoning block).
- `reasoning` is effort 0.0, free-running, with an 8,192-token limit: the reference and teacher from here on.
- `effort 0.9` is effort 0.9, free-running, with an 8,192-token limit: measured once and not used as the reference.

## The reference from here on

The study's reference line, and the teacher for its training data, is the reasoning setting: free-running effort 0.0 with the 8,192-token limit. Effort 0.9 was the first choice and was measured in full with the same limit. It is set aside because on the judging and knowledge benchmarks a fifth to a third of its replies ran to the limit without giving an answer, while effort 0.0 reasons for about half as long, is cut off less often and scores as well or better. Effort 0.9 stays in every table as measured and is not run again.

## Capability map

| benchmark | job | items | fast | effort 0.9 | reasoning | reasoning minus fast (points) | reasoning replies at the limit | reasoning tokens per answer |
|---|---|---|---|---|---|---|---|---|
| JudgeBench, pairwise | Judging answers | 1,240 | 69.0 (65.9 to 71.9) | 64.3 (60.9 to 67.7) | 74.7 (71.5 to 77.7) | +5.7 (+1.9 to +9.4) | 15.2% | 2,651 |
| JudgeBench, one response at a time | Judging answers | 1,240 | 61.3 (59.2 to 63.5) | 61.1 (58.0 to 64.1) | 73.3 (70.5 to 76.0) | +12.0 (+9.1 to +14.9) | 9.9% | 2,289 |
| RewardBench 2, best of four | Judging answers | 1,763 | 70.9 (68.6 to 73.3) | 70.8 (68.5 to 73.1) | 78.5 (76.4 to 80.7) | +7.6 (+4.9 to +10.2) | 6.1% | 1,109 |
| LLMBar | Judging answers | 838 | 74.6 (71.0 to 78.2) | 86.2 (83.2 to 89.0) | 85.9 (82.8 to 88.7) | +11.3 (+8.3 to +14.4) | 0.0% | 270 |
| BFCL V4, single turn | Calling tools | 3,491 | 74.1 (70.6 to 77.4) | 78.3 (75.0 to 81.7) | 78.3 (74.8 to 81.6) | +4.2 (+1.6 to +6.7) | 0.1% | 155 |
| JSONSchemaBench (sample of 1,000) | Filling a schema | 1,000 | 95.6 (94.3 to 96.9) | 97.9 (97.0 to 98.7) | 97.9 (97.0 to 98.8) | +2.3 (+1.2 to +3.6) | 0.1% | 301 |
| GSM8K | Reasoning without the wait | 1,319 | 37.5 (34.8 to 40.1) | 95.8 (94.6 to 96.8) | 96.7 (95.8 to 97.7) | +59.2 (+56.5 to +61.9) | 0.1% | 226 |
| MBPP (sanitized test split) | Reasoning without the wait | 257 | 82.1 (77.4 to 86.8) | 92.2 (88.7 to 95.3) | 94.5 (91.4 to 97.3) | +12.4 (+8.2 to +17.1) | 4.7% | 831 |
| IFEval | Instructions and knowledge | 541 | 68.4 (64.3 to 72.5) | 75.8 (72.3 to 79.3) | 87.4 (84.8 to 90.2) | +19.0 (+14.8 to +23.1) | 7.0% | 1,062 |
| MMLU-Pro (sample of 500) | Instructions and knowledge | 500 | 56.4 (52.0 to 61.0) | 73.0 (69.0 to 76.6) | 80.2 (76.6 to 83.4) | +23.8 (+19.4 to +28.2) | 6.4% | 1,154 |
| BFCL V4, multi turn (sample of 200) | Calling tools | 200 | 28.5 (22.0 to 35.0) | 40.5 (34.0 to 47.0) | 37.0 (30.5 to 43.5) | +8.5 (+1.0 to +16.0) | 0.5% | 1,728 |
| LLM-AggreFact | Checking claims against a source | not run, needs a token: the dataset is gated on Hugging Face (licence CC-BY-ND-4.0) | | | | | | |
| WildGuardTest | Screening for harm | not run, needs a token: the dataset is gated on Hugging Face (licence ODC-BY) | | | | | | |

Differences are paired on the same items. A score that is a mean over groups (RewardBench 2 subsets, BFCL categories) weights every group equally. Published leaderboard numbers are left out of this report.

## The target rule, by job

The rule: a job's gap is the unweighted mean, over its benchmarks, of reference minus fast in points; it is a training target at 10 or more and usable as is at 3 or less, with at most 3 targets. The benchmarks that stand for each job were fixed before the reference pass.

| job | benchmarks | gap against effort 0.9 | class | gap against the reasoning setting | class | class changed |
|---|---|---|---|---|---|---|
| Judging answers | JudgeBench, pairwise; RewardBench 2, best of four; LLMBar | +2.3 (+0.4 to +4.2) | usable as is | +8.2 (+6.4 to +10.0) | in between | yes |
| Calling tools | BFCL V4, single turn; BFCL V4, multi turn (sample of 200) | +8.1 (+4.1 to +12.3) | in between | +6.4 (+2.2 to +10.4) | in between | no |
| Filling a schema | JSONSchemaBench (sample of 1,000) | +2.3 (+1.1 to +3.5) | usable as is | +2.3 (+1.2 to +3.6) | usable as is | no |
| Reasoning without the wait | GSM8K; MBPP (sanitized test split) | +34.2 (+31.7 to +36.8) | training target | +35.8 (+33.4 to +38.5) | training target | no |

Benchmarks outside the rule, with their own gaps for the record:

| benchmark | why it is outside | effort 0.9 minus fast | reasoning minus fast |
|---|---|---|---|
| JudgeBench, one response at a time | not in the list fixed beforehand, so anything built on it is exploratory | -0.2 (-3.4 to +2.9) | +12.0 (+9.1 to +14.9) |
| IFEval | watch only, never trained | +7.4 (+3.0 to +12.2) | +19.0 (+14.8 to +23.1) |
| MMLU-Pro (sample of 500) | watch only, never trained | +16.6 (+11.4 to +21.4) | +23.8 (+19.4 to +28.2) |

Checking claims against a source has no measurement: LLM-AggreFact was not run, needs a token: the dataset is gated on Hugging Face (licence CC-BY-ND-4.0).

Screening for harm has no measurement: WildGuardTest was not run, needs a token: the dataset is gated on Hugging Face (licence ODC-BY).

## Replies that ran out of budget

A reply is cut off when it reaches the 8,192-token limit before the model ends its turn. A cut-off reply has no answer and scores zero. The table separates that from a wrong answer.

| benchmark | setting | replies | cut off | share cut off | score | score among replies that finished |
|---|---|---|---|---|---|---|
| JudgeBench, pairwise | reasoning | 1,240 | 188 | 15% | 74.7 | 88.0 |
| JudgeBench, pairwise | effort 0.9 | 1,240 | 370 | 30% | 64.3 | 91.7 |
| JudgeBench, one response at a time | reasoning | 1,240 | 123 | 10% | 73.3 | 81.4 |
| JudgeBench, one response at a time | effort 0.9 | 1,240 | 349 | 28% | 61.1 | 85.1 |
| RewardBench 2, best of four | reasoning | 1,763 | 108 | 6% | 78.5 | 85.2 |
| RewardBench 2, best of four | effort 0.9 | 1,763 | 354 | 20% | 70.8 | 90.0 |
| LLMBar | reasoning | 838 | 0 | 0% | 85.9 | 85.9 |
| LLMBar | effort 0.9 | 838 | 9 | 1% | 86.2 | 87.1 |
| BFCL V4, single turn | reasoning | 3,491 | 3 | 0% | 78.3 | 78.3 |
| BFCL V4, single turn | effort 0.9 | 3,491 | 16 | 0% | 78.3 | 78.3 |
| JSONSchemaBench (sample of 1,000) | reasoning | 1,000 | 1 | 0% | 97.9 | 98.4 |
| JSONSchemaBench (sample of 1,000) | effort 0.9 | 1,000 | 10 | 1% | 97.9 | 99.3 |
| GSM8K | reasoning | 1,319 | 1 | 0% | 96.7 | 96.7 |
| GSM8K | effort 0.9 | 1,319 | 13 | 1% | 95.8 | 96.7 |
| MBPP (sanitized test split) | reasoning | 257 | 12 | 5% | 94.5 | 99.2 |
| MBPP (sanitized test split) | effort 0.9 | 257 | 16 | 6% | 92.2 | 98.3 |
| IFEval | reasoning | 541 | 38 | 7% | 87.4 | 93.6 |
| IFEval | effort 0.9 | 541 | 104 | 19% | 75.8 | 93.6 |
| MMLU-Pro (sample of 500) | reasoning | 500 | 32 | 6% | 80.2 | 85.7 |
| MMLU-Pro (sample of 500) | effort 0.9 | 500 | 92 | 18% | 73.0 | 89.5 |
| BFCL V4, multi turn (sample of 200) | reasoning | 200 | 1 | 0% | 37.0 | 37.2 |
| BFCL V4, multi turn (sample of 200) | effort 0.9 | 200 | 10 | 5% | 40.5 | 42.5 |

## Tokens and time per answer

| benchmark | setting | prompt tokens | completion tokens mean | p50 | p90 | max | thinking tokens mean | p90 | latency p50 (s) | p95 (s) | cut off | too long for the context |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| JudgeBench, pairwise | fast | 1,360 | 4 | 3 | 7 | 16 | 0 | 0 | 2.35 | 3.84 | 48 | 0 |
| JudgeBench, pairwise | reasoning | 1,358 | 2,658 | 1,436 | 8,192 | 8,192 | 2,651 | 8,190 | 18.96 | 87.66 | 188 | 0 |
| JudgeBench, pairwise | effort 0.9 | 1,360 | 4,270 | 3,310 | 8,192 | 8,192 | 4,264 | 8,190 | 34.16 | 86.01 | 370 | 0 |
| JudgeBench, one response at a time | fast | 840 | 4 | 3 | 3 | 16 | 1 | 0 | 1.91 | 3.74 | 63 | 0 |
| JudgeBench, one response at a time | reasoning | 838 | 2,297 | 1,384 | 7,976 | 8,192 | 2,289 | 7,968 | 16.07 | 72.75 | 123 | 0 |
| JudgeBench, one response at a time | effort 0.9 | 840 | 4,160 | 3,304 | 8,192 | 8,192 | 4,154 | 8,190 | 38.76 | 121.72 | 349 | 0 |
| RewardBench 2, best of four | fast | 1,858 | 3 | 3 | 3 | 16 | 0 | 0 | 12.52 | 13.09 | 7 | 0 |
| RewardBench 2, best of four | reasoning | 1,856 | 1,117 | 381 | 2,508 | 8,192 | 1,109 | 2,500 | 6.74 | 47.19 | 108 | 0 |
| RewardBench 2, best of four | effort 0.9 | 1,858 | 2,535 | 826 | 8,192 | 8,192 | 2,529 | 8,190 | 34.11 | 111.07 | 354 | 0 |
| LLMBar | fast | 332 | 3 | 3 | 3 | 16 | 0 | 0 | 1.83 | 2.94 | 5 | 0 |
| LLMBar | reasoning | 330 | 278 | 188 | 517 | 4,595 | 270 | 509 | 5.69 | 20.68 | 0 | 0 |
| LLMBar | effort 0.9 | 332 | 608 | 282 | 1,173 | 8,192 | 600 | 1,165 | 4.41 | 21.31 | 9 | 0 |
| BFCL V4, single turn | fast | 1,284 | 58 | 37 | 111 | 512 | 4 | 0 |  |  | 23 | 0 |
| BFCL V4, single turn | reasoning | 1,282 | 215 | 148 | 400 | 8,192 | 155 | 314 | 3.52 | 10.2 | 3 | 0 |
| BFCL V4, single turn | effort 0.9 | 1,284 | 394 | 218 | 764 | 8,192 | 322 | 649 |  |  | 16 | 0 |
| JSONSchemaBench (sample of 1,000) | fast | 2,335 | 136 | 68 | 332 | 2,048 | 0 | 0 | 2.53 | 5.49 | 3 | 4 |
| JSONSchemaBench (sample of 1,000) | reasoning | 2,333 | 423 | 267 | 876 | 8,192 | 301 | 594 | 3.73 | 10.84 | 1 | 4 |
| JSONSchemaBench (sample of 1,000) | effort 0.9 | 2,335 | 657 | 346 | 1,265 | 8,192 | 511 | 922 | 4.29 | 13.6 | 10 | 4 |
| GSM8K | fast | 99 | 18 | 4 | 39 | 256 | 5 | 0 | 1.9 | 3.34 | 25 | 0 |
| GSM8K | reasoning | 97 | 235 | 125 | 424 | 8,192 | 226 | 416 | 2.99 | 9.9 | 1 | 0 |
| GSM8K | effort 0.9 | 99 | 428 | 165 | 719 | 8,192 | 419 | 710 | 3.18 | 13.7 | 13 | 0 |
| MBPP (sanitized test split) | fast | 160 | 73 | 45 | 131 | 1,024 | 0 | 0 | 2.27 | 3.92 | 1 | 0 |
| MBPP (sanitized test split) | reasoning | 158 | 891 | 312 | 1,719 | 8,192 | 831 | 1,554 | 5.29 | 26.53 | 12 | 0 |
| MBPP (sanitized test split) | effort 0.9 | 160 | 1,163 | 406 | 2,207 | 8,192 | 1,104 | 2,105 | 4.64 | 36.05 | 16 | 0 |
| IFEval | fast | 64 | 295 | 186 | 557 | 2,048 | 2 | 0 | 4.59 | 10.88 | 19 | 0 |
| IFEval | reasoning | 62 | 1,354 | 703 | 2,240 | 8,192 | 1,062 | 1,618 | 10.38 | 43.31 | 38 | 0 |
| IFEval | effort 0.9 | 64 | 2,529 | 1,149 | 8,192 | 8,192 | 2,287 | 8,190 | 12.6 | 46.55 | 104 | 0 |
| MMLU-Pro (sample of 500) | fast | 224 | 4 | 3 | 7 | 16 | 0 | 0 | 1.57 | 2.63 | 14 | 0 |
| MMLU-Pro (sample of 500) | reasoning | 222 | 1,162 | 330 | 2,889 | 8,192 | 1,154 | 2,881 | 4.79 | 44.2 | 32 | 0 |
| MMLU-Pro (sample of 500) | effort 0.9 | 224 | 2,310 | 662 | 8,192 | 8,192 | 2,303 | 8,190 | 6.87 | 55.55 | 92 | 0 |
| BFCL V4, multi turn (sample of 200) | fast | 182,346 | 608 | 509 | 1,122 | 1,971 | 45 | 144 | 41.97 | 63.48 | 9 | 1 |
| BFCL V4, multi turn (sample of 200) | reasoning | 165,403 | 2,507 | 2,114 | 4,191 | 21,683 | 1,728 | 3,030 | 38.19 | 84.54 | 1 | 0 |
| BFCL V4, multi turn (sample of 200) | effort 0.9 | 175,362 | 4,469 | 3,116 | 9,571 | 33,461 | 3,460 | 7,391 | 44.43 | 117.67 | 10 | 2 |

For the multi-turn benchmark the figures are totals over all the model calls an item took. Latency leaves out replies that were read from the cache.

## Judging in detail

| benchmark | setting | score | false accepts | false rejects | same verdict after swapping | right in both orders | picked the first position | unparsed |
|---|---|---|---|---|---|---|---|---|
| JudgeBench, pairwise | fast | 69.0 |  |  | 72.3 | 55.2 | 46.2 | 1 |
| JudgeBench, pairwise | reasoning | 74.7 |  |  | 76.1 | 68.2 | 49.1 | 188 |
| JudgeBench, pairwise | effort 0.9 | 64.3 |  |  | 58.7 | 55.0 | 49.8 | 370 |
| JudgeBench, one response at a time | fast | 61.3 | 41.9 | 34.8 |  |  |  | 4 |
| JudgeBench, one response at a time | reasoning | 73.3 | 16.4 | 17.1 |  |  |  | 123 |
| JudgeBench, one response at a time | effort 0.9 | 61.1 | 10.7 | 10.8 |  |  |  | 349 |
| RewardBench 2, best of four | fast | 70.9 |  |  |  |  |  | 0 |
| RewardBench 2, best of four | reasoning | 78.5 |  |  |  |  |  | 108 |
| RewardBench 2, best of four | effort 0.9 | 70.8 |  |  |  |  |  | 354 |
| LLMBar | fast | 74.6 |  |  | 84.5 | 66.8 | 48.9 | 0 |
| LLMBar | reasoning | 85.9 |  |  | 92.4 | 82.1 | 52.4 | 0 |
| LLMBar | effort 0.9 | 86.2 |  |  | 91.6 | 82.1 | 52.2 | 9 |

False accepts and false rejects need a yes or no verdict on one response, so they come from the pointwise form of JudgeBench: a false accept is a yes to an incorrect response and a false reject is a no to a correct one. The pairwise benchmarks show each pair in both orders, and their intervals resample whole pairs.

## Scores by group

JudgeBench, pairwise:

| group | items | fast | effort 0.9 | reasoning |
|---|---|---|---|---|
| livebench-math | 180 | 78.9 | 81.7 | 87.2 |
| livebench-reasoning | 298 | 65.8 | 74.8 | 85.2 |
| livecodebench | 146 | 71.2 | 55.5 | 69.2 |
| mmlu-pro | 616 | 67.0 | 56.3 | 67.2 |

JudgeBench, one response at a time:

| group | items | fast | effort 0.9 | reasoning |
|---|---|---|---|---|
| livebench-math | 180 | 69.4 | 73.3 | 80.0 |
| livebench-reasoning | 298 | 54.4 | 64.8 | 77.5 |
| livecodebench | 146 | 61.6 | 62.3 | 67.8 |
| mmlu-pro | 616 | 62.2 | 55.5 | 70.6 |

RewardBench 2, best of four:

| group | items | fast | effort 0.9 | reasoning |
|---|---|---|---|---|
| Factuality | 475 | 73.7 | 53.3 | 72.6 |
| Focus | 495 | 89.3 | 81.6 | 85.7 |
| Math | 183 | 66.1 | 88.0 | 89.1 |
| Precise IF | 160 | 39.4 | 51.2 | 59.4 |
| Safety | 450 | 86.2 | 80.0 | 86.0 |

LLMBar:

| group | items | fast | effort 0.9 | reasoning |
|---|---|---|---|---|
| Adversarial_GPTInst | 184 | 77.7 | 89.1 | 90.2 |
| Adversarial_GPTOut | 94 | 74.5 | 77.7 | 77.7 |
| Adversarial_Manual | 92 | 70.7 | 78.3 | 80.4 |
| Adversarial_Neighbor | 268 | 62.7 | 84.0 | 82.1 |
| Natural | 200 | 89.5 | 94.0 | 93.5 |

BFCL V4, single turn:

| group | items | fast | effort 0.9 | reasoning |
|---|---|---|---|---|
| irrelevance | 240 | 48.3 | 84.6 | 86.7 |
| live_irrelevance | 884 | 57.9 | 80.8 | 79.5 |
| live_multiple | 1053 | 66.6 | 65.8 | 68.2 |
| live_parallel | 16 | 81.2 | 81.2 | 81.2 |
| live_parallel_multiple | 24 | 66.7 | 62.5 | 62.5 |
| live_relevance | 16 | 68.8 | 68.8 | 68.8 |
| live_simple | 258 | 83.3 | 83.7 | 84.5 |
| multiple | 200 | 90.5 | 88.0 | 84.5 |
| parallel | 200 | 87.0 | 82.5 | 83.5 |
| parallel_multiple | 200 | 75.5 | 79.0 | 76.0 |
| simple_python | 400 | 89.2 | 84.8 | 86.0 |

MMLU-Pro (sample of 500):

| group | items | fast | effort 0.9 | reasoning |
|---|---|---|---|---|
| biology | 33 | 87.9 | 84.9 | 87.9 |
| business | 39 | 59.0 | 76.9 | 89.7 |
| chemistry | 53 | 26.4 | 83.0 | 86.8 |
| computer science | 19 | 68.4 | 79.0 | 84.2 |
| economics | 31 | 74.2 | 77.4 | 77.4 |
| engineering | 39 | 51.3 | 56.4 | 74.4 |
| health | 43 | 74.4 | 81.4 | 81.4 |
| history | 16 | 75.0 | 68.8 | 81.2 |
| law | 42 | 50.0 | 35.7 | 52.4 |
| math | 50 | 38.0 | 90.0 | 96.0 |
| other | 36 | 47.2 | 66.7 | 72.2 |
| philosophy | 16 | 37.5 | 56.2 | 62.5 |
| physics | 48 | 47.9 | 75.0 | 79.2 |
| psychology | 35 | 85.7 | 77.1 | 85.7 |

BFCL V4, multi turn (sample of 200):

| group | items | fast | effort 0.9 | reasoning |
|---|---|---|---|---|
| multi_turn_base | 50 | 34.0 | 40.0 | 42.0 |
| multi_turn_long_context | 50 | 32.0 | 32.0 | 32.0 |
| multi_turn_miss_func | 50 | 26.0 | 42.0 | 38.0 |
| multi_turn_miss_param | 50 | 22.0 | 48.0 | 36.0 |

## Throughput over the run

The run kept up to 32 requests in flight. It made 45,544 sampling calls with 151,549,967 prompt tokens and 34,703,163 completion tokens, of which 32,496,600 were thinking tokens. 0 calls failed, 0 were retried and 0 saw a throttling event.

| window start (UTC) | requests | req/min | prompt tok/min | completion tok/min | latency p50 (s) | p95 (s) | max (s) | errors | throttle events |
|---|---|---|---|---|---|---|---|---|---|
| 11-26 19:30 | 373 | 156.5 | 1,004,714 | 64,763 | 3.32 | 19.74 | 95.4 | 0 | 0 |
| 11-26 19:40 | 1,421 | 139.3 | 247,378 | 59,346 | 2.43 | 31.48 | 138.5 | 0 | 0 |
| 11-26 19:50 | 510 | 44.6 | 78,326 | 179,107 | 30.1 | 86.68 | 119.3 | 0 | 0 |
| 11-26 20:00 | 495 | 44.4 | 42,998 | 189,972 | 35.51 | 83.55 | 146.8 | 0 | 0 |
| 11-26 20:10 | 1,748 | 154.0 | 142,060 | 172,614 | 2.29 | 55.55 | 136.5 | 0 | 0 |
| 11-26 20:20 | 379 | 33.8 | 36,357 | 141,853 | 44.07 | 124.71 | 173.3 | 0 | 0 |
| 11-26 20:30 | 331 | 27.4 | 18,055 | 115,477 | 47.56 | 146.79 | 223.5 | 0 | 0 |
| 11-26 20:40 | 2,242 | 190.7 | 307,656 | 147,556 | 2.46 | 51.08 | 136.5 | 0 | 0 |
| 11-26 20:50 | 961 | 92.3 | 151,192 | 284,152 | 11.23 | 56.91 | 88.0 | 0 | 0 |
| 11-26 21:00 | 2,773 | 260.2 | 234,883 | 192,176 | 3.58 | 30.49 | 98.2 | 0 | 0 |
| 11-26 21:10 | 6,052 | 600.2 | 763,538 | 96,112 | 2.5 | 6.95 | 45.2 | 0 | 0 |
| 11-26 21:20 | 4,462 | 441.1 | 644,925 | 152,029 | 3.1 | 10.04 | 78.3 | 0 | 0 |
| 11-26 21:30 | 3,087 | 307.4 | 31,853 | 178,231 | 3.61 | 20.47 | 120.1 | 0 | 0 |
| 11-26 21:40 | 1,903 | 171.2 | 238,970 | 220,179 | 4.34 | 43.31 | 92.7 | 0 | 0 |
| 11-26 21:50 | 3,016 | 278.0 | 3,249,598 | 33,907 | 4.86 | 15.03 | 86.0 | 0 | 0 |
| 11-26 22:00 | 1,630 | 265.4 | 3,450,877 | 94,742 | 4.01 | 13.18 | 63.0 | 0 | 0 |
| 11-26 22:10 | 2,272 | 298.1 | 3,853,860 | 80,883 | 3.01 | 10.0 | 83.8 | 0 | 0 |
| 11-26 22:20 | 236 | 24.9 | 55,455 | 37,155 | 9.13 | 62.92 | 129.9 | 0 | 0 |
| 11-26 22:30 | 2 | 0.7 | 1,689 | 4,740 | 161.04 | 169.26 | 170.2 | 0 | 0 |
| 11-26 22:40 | 641 | 63.8 | 100,532 | 157,596 | 17.06 | 84.2 | 210.6 | 0 | 0 |
| 11-26 22:50 | 644 | 48.7 | 48,103 | 118,872 | 17.82 | 85.43 | 229.8 | 0 | 0 |
| 11-26 23:00 | 785 | 67.8 | 55,126 | 163,734 | 16.2 | 71.86 | 162.3 | 0 | 0 |
| 11-26 23:10 | 1,621 | 139.9 | 230,534 | 151,240 | 6.46 | 48.0 | 178.9 | 0 | 0 |
| 11-26 23:20 | 3,858 | 336.3 | 397,672 | 78,391 | 3.63 | 12.12 | 140.5 | 0 | 0 |
| 11-26 23:30 | 2,976 | 295.2 | 683,813 | 145,159 | 3.64 | 18.97 | 194.2 | 0 | 0 |
| 11-26 23:40 | 1,126 | 226.2 | 3,085,304 | 50,031 | 3.07 | 7.32 | 98.0 | 0 | 0 |

The service raised no throttling event, no paused queue state and no billing pause during the run.

Estimated cost of all metered usage in the 24 hours up to the end of the run, as reported by the service: $0.0. Its usage records lag behind the calls.

## What was run, and under which licence

| benchmark | items | graded by | licence |
|---|---|---|---|
| JudgeBench, pairwise | 1,240 | labels that ship with the benchmark | MIT |
| JudgeBench, one response at a time | 1,240 | labels that ship with the benchmark | MIT |
| RewardBench 2, best of four | 1,763 | labels that ship with the benchmark | ODC-BY |
| LLMBar | 838 | labels that ship with the benchmark | MIT |
| BFCL V4, single turn | 3,491 | BFCL's AST checker | Apache-2.0 |
| JSONSchemaBench (sample of 1,000) | 1,000 | a JSON Schema validator | MIT |
| GSM8K | 1,319 | answer key | MIT |
| MBPP (sanitized test split) | 257 | the benchmark's unit tests, run in a container with no network | CC-BY-4.0 |
| IFEval | 541 | the benchmark's public checker | Apache-2.0 |
| MMLU-Pro (sample of 500) | 500 | answer key | MIT |
| BFCL V4, multi turn (sample of 200) | 200 | BFCL's simulated back ends, with its state and response checks | Apache-2.0 |
| LLM-AggreFact | | | not run, needs a token: the dataset is gated on Hugging Face (licence CC-BY-ND-4.0) |
| WildGuardTest | | | not run, needs a token: the dataset is gated on Hugging Face (licence ODC-BY) |

Every file is downloaded without a login, pinned to a revision and checked against a SHA-256 value (75 files). BFCL is revision `6ea57973c7a6` of the gorilla repository and the IFEval checker is revision `e6890f85757d` of google-research. From BFCL V4 the survey runs the Python single-turn categories and the four multi-turn categories. It leaves out `simple_java` and `simple_javascript`, whose decoders need tree-sitter, and the agentic `web_search` and `memory` categories, which need an outside search service or an embedding model. BFCL's prompt text, AST checker, simulated back ends and state checks are used unchanged, while the reply decoding and the call execution are this repository's own, because BFCL's run model output through `eval`. When the model made a call in its own tool-call format instead of the text format the prompt asks for, the call was read as the reply. The multi-turn benchmark is a fixed sample of 50 items from each of its four categories. From RewardBench 2 the Ties subset is left out: it has several right answers per prompt and its own rating metric, which a pick-one task cannot score. JSONSchemaBench items whose schema does not fit the 64K context are counted as failures.
