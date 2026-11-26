# Survey report

The base model `thinkingmachines/Inkling-2-Mini:peft:65536:free` on public benchmarks, generated 2026-11-26T22:22:54+00:00 with tinker 0.40.1, tml-renderers 0.5.0 and tinker-cookbook 0.7.1. Every score comes from this repository's harness at temperature 0 with one prompt per benchmark, and every reply was graded by a program. Scores are percentages with 95% bootstrap intervals. The full data is in `summary.json`, and the raw replies are in `runs/survey/`.

The settings:

- `fast` is forced direct answer at effort 0.0 (no reasoning block).
- `slow` is effort 0.9, free-running.
- `effort_0` is effort 0.0, free-running (the shipped no-effort setting), on a fixed subsample.

## Capability map

| benchmark | job | items | fast | slow | slow minus fast (points) | by the target rule |
|---|---|---|---|---|---|---|
| JudgeBench, pairwise | Judging answers | 1,240 | 69.0 (65.9 to 71.9) | 64.3 (60.9 to 67.7) | -4.6 (-8.6 to -0.6) | usable as is |
| JudgeBench, one response at a time | Judging answers | 1,240 | 61.3 (59.2 to 63.5) | 61.1 (58.0 to 64.1) | -0.2 (-3.4 to +2.9) | usable as is |
| RewardBench 2, best of four | Judging answers | 1,723 | 71.2 (68.7 to 73.5) | 71.2 (68.8 to 73.3) | -0.1 (-2.7 to +2.6) | usable as is |
| LLMBar | Judging answers | 838 | 74.6 (71.0 to 78.2) | 86.2 (83.2 to 89.0) | +11.6 (+8.5 to +14.8) | training target |
| BFCL V4, single turn | Calling tools | 3,491 | 74.1 (70.6 to 77.4) | 78.3 (75.0 to 81.7) | +4.2 (+1.5 to +7.1) | in between |
| JSONSchemaBench (sample of 1,000) | Filling a schema | 1,000 | 95.6 (94.3 to 96.9) | 97.9 (97.0 to 98.7) | +2.3 (+1.1 to +3.5) | usable as is |
| GSM8K | Reasoning without the wait | 1,319 | 37.5 (34.8 to 40.1) | 95.8 (94.6 to 96.8) | +58.3 (+55.4 to +61.1) | training target |
| MBPP (sanitized test split) | Reasoning without the wait | 257 | 82.1 (77.4 to 86.8) | 92.2 (88.7 to 95.3) | +10.1 (+6.2 to +14.4) | training target |
| IFEval | Instructions and knowledge | 541 | 68.4 (64.3 to 72.5) | 75.8 (72.3 to 79.3) | +7.4 (+3.0 to +12.2) | watch only (in between) |
| MMLU-Pro (sample of 500) | Instructions and knowledge | 500 | 56.4 (52.0 to 61.0) | 73.0 (69.0 to 76.6) | +16.6 (+11.4 to +21.4) | watch only (training target) |
| BFCL V4, multi turn (sample of 200) | Calling tools | 200 | 28.5 (22.0 to 35.0) | 40.5 (34.0 to 47.0) | +12.0 (+4.0 to +20.0) | training target |
| LLM-AggreFact | Checking claims against a source | | | | | not run, needs a token: the dataset is gated on Hugging Face (licence CC-BY-ND-4.0) |
| WildGuardTest | Screening for harm | | | | | not run, needs a token: the dataset is gated on Hugging Face (licence ODC-BY) |

The rule, fixed before the run: training target if slow minus fast is at least 10 points, usable as is if it is at most 3. Jobs marked watch only are never trained. A score that is a mean over groups (RewardBench 2 subsets, BFCL categories) weights every group equally. Published leaderboard numbers are left out of this report, so the rule is applied to the gap between the two settings only.

## The shipped no-effort setting, on a subsample

Free-running effort 0.0 still reasons before it answers. It ran on a fixed subsample of each benchmark, shown next to the other two settings on the same items. The last column applies the same rule with effort_0 in place of slow.

| benchmark | items | fast | effort_0 | slow | effort_0 minus fast | slow minus effort_0 | thinking tokens at effort_0 | rule against effort_0 |
|---|---|---|---|---|---|---|---|---|
| JudgeBench, pairwise | 200 | 66.0 (58.5 to 73.0) | 72.5 (64.5 to 80.0) | 67.5 (59.0 to 76.0) | +6.5 (-2.5 to +16.0) | -5.0 (-10.5 to +0.5) | 2,617 | in between |
| JudgeBench, one response at a time | 200 | 62.5 (58.0 to 67.5) | 77.5 (71.5 to 83.5) | 69.5 (62.5 to 76.5) | +15.0 (+8.0 to +22.0) | -8.0 (-13.5 to -2.5) | 2,159 | training target |
| RewardBench 2, best of four | 195 | 79.2 (72.9 to 85.2) | 82.6 (76.6 to 88.0) | 73.4 (67.0 to 79.5) | +3.4 (-4.3 to +10.9) | -9.2 (-13.9 to -4.7) | 1,130 | in between |
| LLMBar | 200 | 77.5 (70.0 to 84.5) | 84.5 (78.0 to 91.0) | 85.0 (78.5 to 90.5) | +7.0 (+2.0 to +12.0) | +0.5 (-3.0 to +4.0) | 263 | in between |
| BFCL V4, single turn | 200 | 80.7 (74.0 to 87.2) | 80.3 (73.4 to 87.0) | 81.0 (74.4 to 87.8) | -0.4 (-4.5 to +3.4) | +0.7 (-3.4 to +4.6) | 148 | usable as is |
| JSONSchemaBench (sample of 1,000) | 200 | 96.5 (93.5 to 99.0) | 98.5 (96.5 to 100.0) | 99.0 (97.5 to 100.0) | +2.0 (+0.0 to +4.5) | +0.5 (+0.0 to +1.5) | 299 | usable as is |
| GSM8K | 200 | 39.0 (32.0 to 45.5) | 96.5 (94.0 to 98.5) | 95.5 (92.5 to 98.0) | +57.5 (+50.5 to +64.5) | -1.0 (-3.0 to +1.0) | 271 | training target |
| MBPP (sanitized test split) | 200 | 84.0 (78.5 to 89.0) | 93.5 (90.0 to 97.0) | 92.0 (88.0 to 95.5) | +9.5 (+5.0 to +14.0) | -1.5 (-4.0 to +1.0) | 863 | in between |
| IFEval | 200 | 65.0 (58.5 to 71.5) | 87.5 (83.0 to 92.0) | 74.5 (68.0 to 80.5) | +22.5 (+15.5 to +29.5) | -13.0 (-19.0 to -7.5) | 1,064 | watch only (training target) |
| MMLU-Pro (sample of 500) | 200 | 50.5 (43.5 to 57.0) | 80.5 (74.5 to 86.0) | 73.5 (67.0 to 79.5) | +30.0 (+22.5 to +37.5) | -7.0 (-11.0 to -3.5) | 1,143 | watch only (training target) |
| BFCL V4, multi turn (sample of 200) | 60 | 39.0 (27.1 to 51.4) | 33.2 (21.8 to 44.9) | 41.1 (29.0 to 53.1) | -5.8 (-20.0 to +8.3) | +7.8 (-6.4 to +22.3) | 1,781 | usable as is |

## Replies that ran out of budget

A reply is cut off when it reaches the token limit (8,192 tokens for slow and effort_0) before the model ends its turn. A cut-off reply has no answer and scores zero. The table separates that from a wrong answer.

| benchmark | setting | replies | cut off | share cut off | score | score among replies that finished |
|---|---|---|---|---|---|---|
| JudgeBench, pairwise | slow | 1,240 | 370 | 30% | 64.3 | 91.7 |
| JudgeBench, pairwise | effort_0 | 200 | 33 | 16% | 72.5 | 86.8 |
| JudgeBench, one response at a time | slow | 1,240 | 349 | 28% | 61.1 | 85.1 |
| JudgeBench, one response at a time | effort_0 | 200 | 13 | 6% | 77.5 | 82.9 |
| RewardBench 2, best of four | slow | 1,723 | 335 | 19% | 71.2 | 90.3 |
| RewardBench 2, best of four | effort_0 | 195 | 13 | 7% | 82.6 | 90.4 |
| LLMBar | slow | 838 | 9 | 1% | 86.2 | 87.1 |
| LLMBar | effort_0 | 200 | 0 | 0% | 84.5 | 84.5 |
| BFCL V4, single turn | slow | 3,491 | 16 | 0% | 78.3 | 78.3 |
| BFCL V4, single turn | effort_0 | 200 | 0 | 0% | 80.3 | 80.3 |
| JSONSchemaBench (sample of 1,000) | slow | 1,000 | 10 | 1% | 97.9 | 99.3 |
| JSONSchemaBench (sample of 1,000) | effort_0 | 200 | 0 | 0% | 98.5 | 98.5 |
| GSM8K | slow | 1,319 | 13 | 1% | 95.8 | 96.7 |
| GSM8K | effort_0 | 200 | 1 | 0% | 96.5 | 97.0 |
| MBPP (sanitized test split) | slow | 257 | 16 | 6% | 92.2 | 98.3 |
| MBPP (sanitized test split) | effort_0 | 200 | 11 | 6% | 93.5 | 98.9 |
| IFEval | slow | 541 | 104 | 19% | 75.8 | 93.6 |
| IFEval | effort_0 | 200 | 13 | 6% | 87.5 | 93.0 |
| MMLU-Pro (sample of 500) | slow | 500 | 92 | 18% | 73.0 | 89.5 |
| MMLU-Pro (sample of 500) | effort_0 | 200 | 12 | 6% | 80.5 | 85.6 |
| BFCL V4, multi turn (sample of 200) | slow | 200 | 10 | 5% | 40.5 | 42.5 |
| BFCL V4, multi turn (sample of 200) | effort_0 | 60 | 0 | 0% | 33.2 | 33.2 |

## Tokens and time per answer

| benchmark | setting | prompt tokens | completion tokens mean | p50 | p90 | max | thinking tokens mean | p90 | latency p50 (s) | p95 (s) | cut off | too long for the context |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| JudgeBench, pairwise | fast | 1,360 | 4 | 3 | 7 | 16 | 0 | 0 | 2.35 | 3.84 | 48 | 0 |
| JudgeBench, pairwise | slow | 1,360 | 4,270 | 3,310 | 8,192 | 8,192 | 4,264 | 8,190 | 34.16 | 86.01 | 370 | 0 |
| JudgeBench, pairwise | effort_0 | 1,351 | 2,624 | 1,361 | 8,192 | 8,192 | 2,617 | 8,190 | 13.88 | 69.06 | 33 | 0 |
| JudgeBench, one response at a time | fast | 840 | 4 | 3 | 3 | 16 | 1 | 0 | 1.91 | 3.74 | 63 | 0 |
| JudgeBench, one response at a time | slow | 840 | 4,160 | 3,304 | 8,192 | 8,192 | 4,154 | 8,190 | 38.76 | 121.72 | 349 | 0 |
| JudgeBench, one response at a time | effort_0 | 828 | 2,167 | 1,456 | 4,899 | 8,192 | 2,159 | 4,891 | 13.75 | 44.77 | 13 | 0 |
| RewardBench 2, best of four | fast | 1,851 | 3 | 3 | 3 | 16 | 0 | 0 | 2.2 | 3.97 | 7 | 0 |
| RewardBench 2, best of four | slow | 1,851 | 2,479 | 785 | 8,192 | 8,192 | 2,472 | 8,190 | 8.53 | 52.3 | 335 | 0 |
| RewardBench 2, best of four | effort_0 | 1,797 | 1,137 | 342 | 2,412 | 8,192 | 1,130 | 2,404 | 5.45 | 39.04 | 13 | 0 |
| LLMBar | fast | 332 | 3 | 3 | 3 | 16 | 0 | 0 | 1.83 | 2.94 | 5 | 0 |
| LLMBar | slow | 332 | 608 | 282 | 1,173 | 8,192 | 600 | 1,165 | 4.41 | 21.31 | 9 | 0 |
| LLMBar | effort_0 | 341 | 271 | 186 | 514 | 2,793 | 263 | 506 | 3.08 | 7.61 | 0 | 0 |
| BFCL V4, single turn | fast | 1,284 | 58 | 37 | 111 | 512 | 4 | 0 | 0.0 | 0.0 | 23 | 0 |
| BFCL V4, single turn | slow | 1,284 | 394 | 218 | 764 | 8,192 | 322 | 649 | 0.0 | 0.0 | 16 | 0 |
| BFCL V4, single turn | effort_0 | 1,290 | 210 | 150 | 407 | 1,366 | 148 | 313 | 0.0 | 0.0 | 0 | 0 |
| JSONSchemaBench (sample of 1,000) | fast | 2,335 | 136 | 68 | 332 | 2,048 | 0 | 0 | 2.52 | 5.46 | 3 | 4 |
| JSONSchemaBench (sample of 1,000) | slow | 2,335 | 657 | 346 | 1,265 | 8,192 | 511 | 922 | 4.28 | 13.59 | 10 | 4 |
| JSONSchemaBench (sample of 1,000) | effort_0 | 1,927 | 422 | 257 | 873 | 3,413 | 299 | 608 | 3.96 | 10.54 | 0 | 0 |
| GSM8K | fast | 99 | 18 | 4 | 39 | 256 | 5 | 0 | 1.9 | 3.34 | 25 | 0 |
| GSM8K | slow | 99 | 428 | 165 | 719 | 8,192 | 419 | 710 | 3.18 | 13.7 | 13 | 0 |
| GSM8K | effort_0 | 97 | 279 | 132 | 486 | 8,192 | 271 | 478 | 2.95 | 7.23 | 1 | 0 |
| MBPP (sanitized test split) | fast | 160 | 73 | 45 | 131 | 1,024 | 0 | 0 | 2.27 | 3.92 | 1 | 0 |
| MBPP (sanitized test split) | slow | 160 | 1,163 | 406 | 2,207 | 8,192 | 1,104 | 2,105 | 4.64 | 36.05 | 16 | 0 |
| MBPP (sanitized test split) | effort_0 | 156 | 922 | 300 | 1,790 | 8,192 | 863 | 1,609 | 4.53 | 29.59 | 11 | 0 |
| IFEval | fast | 64 | 295 | 186 | 557 | 2,048 | 2 | 0 | 4.59 | 10.88 | 19 | 0 |
| IFEval | slow | 64 | 2,529 | 1,149 | 8,192 | 8,192 | 2,287 | 8,190 | 12.6 | 46.55 | 104 | 0 |
| IFEval | effort_0 | 64 | 1,351 | 733 | 2,192 | 8,192 | 1,064 | 1,510 | 9.01 | 37.08 | 13 | 0 |
| MMLU-Pro (sample of 500) | fast | 224 | 4 | 3 | 7 | 16 | 0 | 0 | 1.57 | 2.63 | 14 | 0 |
| MMLU-Pro (sample of 500) | slow | 224 | 2,310 | 662 | 8,192 | 8,192 | 2,303 | 8,190 | 6.87 | 55.55 | 92 | 0 |
| MMLU-Pro (sample of 500) | effort_0 | 220 | 1,151 | 314 | 3,093 | 8,192 | 1,143 | 3,085 | 4.45 | 37.34 | 12 | 0 |
| BFCL V4, multi turn (sample of 200) | fast | 182,346 | 608 | 509 | 1,122 | 1,971 | 45 | 144 | 0.0 | 43.81 | 9 | 1 |
| BFCL V4, multi turn (sample of 200) | slow | 175,362 | 4,469 | 3,116 | 9,571 | 33,461 | 3,460 | 7,391 | 31.2 | 106.97 | 10 | 2 |
| BFCL V4, multi turn (sample of 200) | effort_0 | 157,982 | 2,461 | 2,136 | 4,604 | 7,326 | 1,781 | 3,739 | 2.7 | 65.07 | 0 | 0 |

For the multi-turn benchmark the figures are totals over all the model calls an item took.

## Judging in detail

| benchmark | setting | score | false accepts | false rejects | same verdict after swapping | right in both orders | picked the first position | unparsed |
|---|---|---|---|---|---|---|---|---|
| JudgeBench, pairwise | fast | 69.0 |  |  | 72.3 | 55.2 | 46.2 | 1 |
| JudgeBench, pairwise | slow | 64.3 |  |  | 58.7 | 55.0 | 49.8 | 370 |
| JudgeBench, pairwise | effort_0 | 72.5 |  |  | 73.0 | 65.0 | 49.7 | 33 |
| JudgeBench, one response at a time | fast | 61.3 | 41.9 | 34.8 |  |  |  | 4 |
| JudgeBench, one response at a time | slow | 61.1 | 10.7 | 10.8 |  |  |  | 349 |
| JudgeBench, one response at a time | effort_0 | 77.5 | 14.0 | 18.0 |  |  |  | 13 |
| RewardBench 2, best of four | fast | 71.2 |  |  |  |  |  | 0 |
| RewardBench 2, best of four | slow | 71.2 |  |  |  |  |  | 335 |
| RewardBench 2, best of four | effort_0 | 82.6 |  |  |  |  |  | 13 |
| LLMBar | fast | 74.6 |  |  | 84.5 | 66.8 | 48.9 | 0 |
| LLMBar | slow | 86.2 |  |  | 91.6 | 82.1 | 52.2 | 9 |
| LLMBar | effort_0 | 84.5 |  |  | 91.0 | 80.0 | 53.5 | 0 |

False accepts and false rejects need a yes or no verdict on one response, so they come from the pointwise form of JudgeBench: a false accept is a yes to an incorrect response and a false reject is a no to a correct one. The pairwise benchmarks show each pair in both orders, and their intervals resample whole pairs.

## Scores by group

JudgeBench, pairwise:

| group | items | fast | slow |
|---|---|---|---|
| livebench-math | 180 | 78.9 | 81.7 |
| livebench-reasoning | 298 | 65.8 | 74.8 |
| livecodebench | 146 | 71.2 | 55.5 |
| mmlu-pro | 616 | 67.0 | 56.3 |

JudgeBench, one response at a time:

| group | items | fast | slow |
|---|---|---|---|
| livebench-math | 180 | 69.4 | 73.3 |
| livebench-reasoning | 298 | 54.4 | 64.8 |
| livecodebench | 146 | 61.6 | 62.3 |
| mmlu-pro | 616 | 62.2 | 55.5 |

RewardBench 2, best of four:

| group | items | fast | slow |
|---|---|---|---|
| Factuality | 435 | 75.2 | 54.9 |
| Focus | 495 | 89.3 | 81.6 |
| Math | 183 | 66.1 | 88.0 |
| Precise IF | 160 | 39.4 | 51.2 |
| Safety | 450 | 86.2 | 80.0 |

LLMBar:

| group | items | fast | slow |
|---|---|---|---|
| Adversarial_GPTInst | 184 | 77.7 | 89.1 |
| Adversarial_GPTOut | 94 | 74.5 | 77.7 |
| Adversarial_Manual | 92 | 70.7 | 78.3 |
| Adversarial_Neighbor | 268 | 62.7 | 84.0 |
| Natural | 200 | 89.5 | 94.0 |

BFCL V4, single turn:

| group | items | fast | slow |
|---|---|---|---|
| irrelevance | 240 | 48.3 | 84.6 |
| live_irrelevance | 884 | 57.9 | 80.8 |
| live_multiple | 1053 | 66.6 | 65.8 |
| live_parallel | 16 | 81.2 | 81.2 |
| live_parallel_multiple | 24 | 66.7 | 62.5 |
| live_relevance | 16 | 68.8 | 68.8 |
| live_simple | 258 | 83.3 | 83.7 |
| multiple | 200 | 90.5 | 88.0 |
| parallel | 200 | 87.0 | 82.5 |
| parallel_multiple | 200 | 75.5 | 79.0 |
| simple_python | 400 | 89.2 | 84.8 |

MMLU-Pro (sample of 500):

| group | items | fast | slow |
|---|---|---|---|
| biology | 33 | 87.9 | 84.9 |
| business | 39 | 59.0 | 76.9 |
| chemistry | 53 | 26.4 | 83.0 |
| computer science | 19 | 68.4 | 79.0 |
| economics | 31 | 74.2 | 77.4 |
| engineering | 39 | 51.3 | 56.4 |
| health | 43 | 74.4 | 81.4 |
| history | 16 | 75.0 | 68.8 |
| law | 42 | 50.0 | 35.7 |
| math | 50 | 38.0 | 90.0 |
| other | 36 | 47.2 | 66.7 |
| philosophy | 16 | 37.5 | 56.2 |
| physics | 48 | 47.9 | 75.0 |
| psychology | 35 | 85.7 | 77.1 |

BFCL V4, multi turn (sample of 200):

| group | items | fast | slow |
|---|---|---|---|
| multi_turn_base | 50 | 34.0 | 40.0 |
| multi_turn_long_context | 50 | 32.0 | 32.0 |
| multi_turn_miss_func | 50 | 26.0 | 42.0 |
| multi_turn_miss_param | 50 | 22.0 | 48.0 |

## Throughput over the run

The run kept up to 32 requests in flight. It made 33,659 sampling calls with 119,311,070 prompt tokens and 24,930,315 completion tokens, of which 23,266,765 were thinking tokens. 0 calls failed, 0 were retried and 0 saw a throttling event.

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
| 11-26 22:20 | 4 | 6.4 | 103,810 | 16,247 | 8.83 | 29.09 | 31.9 | 0 | 0 |

The service raised no throttling event, no paused queue state and no billing pause during the run.

Estimated cost of all metered usage in the 24 hours up to the end of the run, as reported by the service: $0.0. Its usage records lag behind the calls.

## What was run, and under which licence

| benchmark | items | graded by | licence |
|---|---|---|---|
| JudgeBench, pairwise | 1,240 | labels that ship with the benchmark | MIT |
| JudgeBench, one response at a time | 1,240 | labels that ship with the benchmark | MIT |
| RewardBench 2, best of four | 1,723 | labels that ship with the benchmark | ODC-BY |
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
