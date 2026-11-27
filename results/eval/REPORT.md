# Evaluation of the adapters

Every model state answered the same benchmark items with the same prompt, and a program graded every reply. The fast setting is a forced direct answer at effort 0.0. Scores are percentages with 95% bootstrap intervals over items. A difference is paired on the same items, in points, with its 95% bootstrap interval, and the p-value is from an exact McNemar test on the items where the two states disagree. The main adapter is `middle-2000-s1`. Everything in this report is measured unless it says otherwise. Generated 2026-11-27T15:37:47+00:00.

Items per benchmark: GSM8K 1,319, MBPP (sanitized test split) 257, JudgeBench, pairwise 500, JudgeBench, one response at a time 500, RewardBench 2, best of four 500, LLMBar 500, BFCL V4, single turn 500, JSONSchemaBench (sample of 1,000) 500, IFEval 500, MMLU-Pro (sample of 500) 500, BFCL V4, multi turn (sample of 200) 200.

## The decision rules

The rules were fixed before any adapter was trained. They are applied as written, on GSM8K and on MBPP separately.

- P2 on GSM8K: holds. The main adapter changes the fast score by +4.6 points (+2.3 to +7.0), McNemar p = 0.000253; the score rises and McNemar's test separates before from after.
- P2 against the prompt-only state on GSM8K: holds. The main adapter minus eight examples in the prompt is +6.1 points (+3.7 to +8.4), McNemar p = 1e-06.
- P3 on GSM8K: wrong. Middle band minus random over 3 seeds is -0.7 points (-1.8 to +0.5), seed by seed from -1.5 to +0.2; the interval includes zero.
- P3 against easy only on GSM8K: counts against. Middle band minus easy only is +0.0 points (-2.0 to +2.1); the interval includes zero.
- P3 against hard only on GSM8K: supports. Middle band minus hard only is +4.9 points (+2.6 to +7.2); the middle band is ahead.
- P4 on GSM8K: real labels minus shuffled labels is +42.1 points (+39.3 to +44.7). The interval excludes zero: real labels are ahead, so the gain is more than the format.
- P2 on MBPP (sanitized test split): holds. The main adapter changes the fast score by +3.5 points (+0.8 to +6.2), McNemar p = 0.02246; the score rises and McNemar's test separates before from after.
- P2 against the prompt-only state on MBPP (sanitized test split): wrong. The main adapter minus eight examples in the prompt is +1.2 points (-1.2 to +3.9), McNemar p = 0.5488.
- P3 on MBPP (sanitized test split): wrong. Middle band minus random over 3 seeds is +0.9 points (-0.8 to +2.6), seed by seed from -1.2 to +1.9; the interval includes zero.
- P3 against easy only on MBPP (sanitized test split): counts against. Middle band minus easy only is +1.6 points (-1.2 to +4.3); the interval includes zero.
- P3 against hard only on MBPP (sanitized test split): counts against. Middle band minus hard only is +1.6 points (-1.2 to +4.3); the interval includes zero.
- P4 on MBPP (sanitized test split): real labels minus shuffled labels is +85.6 points (+81.3 to +89.9). The interval excludes zero: real labels are ahead, so the gain is more than the format.
- P5: wrong. 1 of 9 watch cells falls by more than 2 points under the main adapter: BFCL V4, multi turn (sample of 200) -3.0 points (-9.0 to +3.0).

## The target benchmarks

On GSM8K the base model scores 37.5 in the fast setting and 96.7 in the reasoning setting on the same items, a gap of +59.2 (+56.5 to +61.9) points.
On MBPP (sanitized test split) the base model scores 82.1 in the fast setting and 94.5 in the reasoning setting on the same items, a gap of +12.4 (+8.2 to +17.1) points.
The share of the gap closed is a state's gain over the base model's fast score, divided by that gap.

GSM8K:

| state | score | minus base | McNemar p | share of the gap closed | completion tokens per answer |
|---|---|---|---|---|---|
| base | 37.5 (34.8 to 40.1) |  |  |  | 18 |
| middle-2000-s1 | 42.1 (39.3 to 44.7) | +4.6 (+2.3 to +7.0) | <0.001 | 8% (4% to 12%) | 3 |
| random-2000-s1 | 42.8 (40.1 to 45.3) | +5.3 (+2.9 to +7.7) | <0.001 | 9% (5% to 13%) | 3 |
| middle-2000-s2 | 43.0 (40.3 to 45.6) | +5.5 (+3.3 to +7.7) | <0.001 | 9% (6% to 13%) | 3 |
| random-2000-s2 | 44.5 (41.9 to 47.2) | +7.0 (+4.5 to +9.5) | <0.001 | 12% (8% to 16%) | 3 |
| middle-2000-s3 | 44.8 (42.1 to 47.4) | +7.3 (+4.9 to +9.7) | <0.001 | 12% (8% to 16%) | 3 |
| random-2000-s3 | 44.6 (41.9 to 47.2) | +7.1 (+4.6 to +9.5) | <0.001 | 12% (8% to 16%) | 3 |
| shuffled-labels-2000 | 0.0 (0.0 to 0.0) | -37.5 (-40.1 to -34.8) | <0.001 | -63% (-71% to -56%) | 4 |
| easy-2000 | 42.1 (39.5 to 44.7) | +4.6 (+2.3 to +7.0) | <0.001 | 8% (4% to 12%) | 3 |
| hard-2000 | 37.2 (34.6 to 39.9) | -0.2 (-2.6 to +2.0) | 0.893 | -0% (-4% to 3%) | 3 |
| length-matched-random-2000 | 43.4 (40.8 to 46.1) | +5.9 (+3.6 to +8.3) | <0.001 | 10% (6% to 14%) | 3 |
| middle-250 | 41.7 (39.0 to 44.4) | +4.2 (+2.1 to +6.5) | <0.001 | 7% (4% to 11%) | 3 |
| random-250 | 40.5 (37.9 to 43.2) | +3.0 (+0.7 to +5.3) | 0.010 | 5% (1% to 9%) | 3 |
| middle-500 | 41.4 (38.8 to 44.0) | +3.9 (+1.7 to +6.2) | <0.001 | 7% (3% to 10%) | 4 |
| random-500 | 39.5 (36.8 to 42.1) | +2.1 (-0.2 to +4.3) | 0.088 | 3% (-0% to 7%) | 3 |
| middle-1000 | 42.8 (40.1 to 45.4) | +5.3 (+3.0 to +7.7) | <0.001 | 9% (5% to 13%) | 3 |
| random-1000 | 41.2 (38.6 to 43.8) | +3.7 (+1.3 to +6.1) | 0.002 | 6% (2% to 10%) | 3 |
| middle-4000 | 45.0 (42.3 to 47.5) | +7.5 (+5.0 to +9.9) | <0.001 | 13% (9% to 16%) | 3 |
| random-4000 | 46.2 (43.5 to 48.8) | +8.8 (+6.4 to +11.1) | <0.001 | 15% (11% to 19%) | 3 |
| prompt-only | 35.9 (33.5 to 38.5) | -1.5 (-3.9 to +1.1) | 0.269 | -3% (-7% to 2%) | 46 |

MBPP (sanitized test split):

| state | score | minus base | McNemar p | share of the gap closed | completion tokens per answer |
|---|---|---|---|---|---|
| base | 82.1 (77.4 to 86.8) |  |  |  | 73 |
| middle-2000-s1 | 85.6 (81.3 to 89.9) | +3.5 (+0.8 to +6.2) | 0.022 | 28% (8% to 47%) | 61 |
| random-2000-s1 | 83.7 (79.0 to 87.9) | +1.6 (-1.6 to +4.7) | 0.454 | 12% (-13% to 35%) | 63 |
| middle-2000-s2 | 85.6 (80.9 to 89.9) | +3.5 (+0.4 to +7.0) | 0.064 | 28% (3% to 51%) | 62 |
| random-2000-s2 | 83.7 (79.0 to 87.9) | +1.6 (-1.6 to +4.7) | 0.454 | 12% (-14% to 33%) | 62 |
| middle-2000-s3 | 83.7 (79.0 to 87.9) | +1.6 (-1.6 to +4.7) | 0.481 | 12% (-15% to 36%) | 60 |
| random-2000-s3 | 84.8 (80.2 to 89.1) | +2.7 (-0.8 to +6.2) | 0.210 | 22% (-11% to 46%) | 64 |
| shuffled-labels-2000 | 0.0 (0.0 to 0.0) | -82.1 (-86.8 to -77.4) | <0.001 | -659% (-1095% to -457%) | 4 |
| easy-2000 | 84.0 (79.4 to 88.7) | +1.9 (-1.2 to +5.1) | 0.332 | 16% (-11% to 38%) | 70 |
| hard-2000 | 84.0 (79.4 to 88.3) | +1.9 (-1.2 to +5.1) | 0.332 | 16% (-12% to 37%) | 68 |
| length-matched-random-2000 | 83.3 (78.6 to 87.5) | +1.2 (-1.2 to +3.5) | 0.508 | 9% (-10% to 26%) | 69 |
| middle-250 | 84.8 (80.2 to 89.1) | +2.7 (-0.4 to +5.8) | 0.143 | 22% (-4% to 43%) | 67 |
| random-250 | 82.9 (78.2 to 87.5) | +0.8 (-1.9 to +3.5) | 0.791 | 6% (-21% to 26%) | 66 |
| middle-500 | 84.4 (79.8 to 88.7) | +2.3 (-0.4 to +5.1) | 0.180 | 19% (-4% to 38%) | 70 |
| random-500 | 84.4 (79.8 to 88.7) | +2.3 (-0.4 to +5.1) | 0.180 | 19% (-4% to 39%) | 63 |
| middle-1000 | 82.5 (77.8 to 87.2) | +0.4 (-3.1 to +3.9) | 1.000 | 3% (-28% to 27%) | 69 |
| random-1000 | 84.0 (79.8 to 88.3) | +1.9 (-1.2 to +5.5) | 0.359 | 16% (-13% to 38%) | 65 |
| middle-4000 | 86.0 (81.3 to 89.9) | +3.9 (+0.4 to +7.4) | 0.041 | 31% (5% to 55%) | 61 |
| random-4000 | 85.2 (80.9 to 89.5) | +3.1 (+0.0 to +6.2) | 0.077 | 25% (3% to 47%) | 60 |
| prompt-only | 84.4 (79.8 to 88.7) | +2.3 (-0.4 to +5.5) | 0.180 | 19% (-4% to 40%) | 69 |

## The main comparison: middle band against random

Both arms train on the same number of training tokens with the same number of updates. Each of the three seeds has its own draw of both sets, its own starting weights and its own batch order. The pooled difference averages each item's score over an arm's three adapters and resamples benchmark items.

| benchmark | middle band, by seed | random, by seed | middle minus random, pooled | seed by seed |
|---|---|---|---|---|
| GSM8K | 42.1, 43.0, 44.8 | 42.8, 44.5, 44.6 | -0.7 (-1.8 to +0.5) | -1.5 to +0.2 |
| MBPP (sanitized test split) | 85.6, 85.6, 83.7 | 83.7, 83.7, 84.8 | +0.9 (-0.8 to +2.6) | -1.2 to +1.9 |

The controls are single runs with seed 1, compared with `middle-2000-s1` item by item.

| benchmark | control | control score | middle band score | middle band minus control | McNemar p |
|---|---|---|---|---|---|
| GSM8K | easy only | 42.1 (39.5 to 44.7) | 42.1 (39.3 to 44.7) | +0.0 (-2.0 to +2.1) | 1.000 |
| GSM8K | hard only | 37.2 (34.6 to 39.9) | 42.1 (39.3 to 44.7) | +4.9 (+2.6 to +7.2) | <0.001 |
| GSM8K | shuffled labels | 0.0 (0.0 to 0.0) | 42.1 (39.3 to 44.7) | +42.1 (+39.3 to +44.7) | <0.001 |
| GSM8K | length-matched random | 43.4 (40.8 to 46.1) | 42.1 (39.3 to 44.7) | -1.3 (-3.3 to +0.5) | 0.226 |
| MBPP (sanitized test split) | easy only | 84.0 (79.4 to 88.7) | 85.6 (81.3 to 89.9) | +1.6 (-1.2 to +4.3) | 0.424 |
| MBPP (sanitized test split) | hard only | 84.0 (79.4 to 88.3) | 85.6 (81.3 to 89.9) | +1.6 (-1.2 to +4.3) | 0.424 |
| MBPP (sanitized test split) | shuffled labels | 0.0 (0.0 to 0.0) | 85.6 (81.3 to 89.9) | +85.6 (+81.3 to +89.9) | <0.001 |
| MBPP (sanitized test split) | length-matched random | 83.3 (78.6 to 87.5) | 85.6 (81.3 to 89.9) | +2.3 (+0.0 to +5.1) | 0.109 |

## The size sweep

One adapter per size and arm, seed 1. The sets of each arm are nested, and the two arms have the same training tokens at every size.

GSM8K:

| items in the middle-band set | middle band | minus base | random | minus base |
|---|---|---|---|---|
| 250 | 41.7 (39.0 to 44.4) | +4.2 (+2.1 to +6.5) | 40.5 (37.9 to 43.2) | +3.0 (+0.7 to +5.3) |
| 500 | 41.4 (38.8 to 44.0) | +3.9 (+1.7 to +6.2) | 39.5 (36.8 to 42.1) | +2.1 (-0.2 to +4.3) |
| 1,000 | 42.8 (40.1 to 45.4) | +5.3 (+3.0 to +7.7) | 41.2 (38.6 to 43.8) | +3.7 (+1.3 to +6.1) |
| 2,000 | 42.1 (39.3 to 44.7) | +4.6 (+2.3 to +7.0) | 42.8 (40.1 to 45.3) | +5.3 (+2.9 to +7.7) |
| 4,000 | 45.0 (42.3 to 47.5) | +7.5 (+5.0 to +9.9) | 46.2 (43.5 to 48.8) | +8.8 (+6.4 to +11.1) |

MBPP (sanitized test split):

| items in the middle-band set | middle band | minus base | random | minus base |
|---|---|---|---|---|
| 250 | 84.8 (80.2 to 89.1) | +2.7 (-0.4 to +5.8) | 82.9 (78.2 to 87.5) | +0.8 (-1.9 to +3.5) |
| 500 | 84.4 (79.8 to 88.7) | +2.3 (-0.4 to +5.1) | 84.4 (79.8 to 88.7) | +2.3 (-0.4 to +5.1) |
| 1,000 | 82.5 (77.8 to 87.2) | +0.4 (-3.1 to +3.9) | 84.0 (79.8 to 88.3) | +1.9 (-1.2 to +5.5) |
| 2,000 | 85.6 (81.3 to 89.9) | +3.5 (+0.8 to +6.2) | 83.7 (79.0 to 87.9) | +1.6 (-1.6 to +4.7) |
| 4,000 | 86.0 (81.3 to 89.9) | +3.9 (+0.4 to +7.4) | 85.2 (80.9 to 89.5) | +3.1 (+0.0 to +6.2) |

## Eight examples in the prompt, with no training

The prompt-only state is the base model in the fast setting with eight held-out middle-band problems and their answers placed before the question.

| benchmark | base | prompt-only | prompt-only minus base | McNemar p | main adapter | main adapter minus prompt-only | McNemar p |
|---|---|---|---|---|---|---|---|
| GSM8K | 37.5 (34.8 to 40.1) | 35.9 (33.5 to 38.5) | -1.5 (-3.9 to +1.1) | 0.269 | 42.1 (39.3 to 44.7) | +6.1 (+3.7 to +8.4) | <0.001 |
| MBPP (sanitized test split) | 82.1 (77.4 to 86.8) | 84.4 (79.8 to 88.7) | +2.3 (-0.4 to +5.5) | 0.180 | 85.6 (81.3 to 89.9) | +1.2 (-1.2 to +3.9) | 0.549 |
| JudgeBench, pairwise | 71.0 (66.2 to 75.4) | 71.6 (66.8 to 76.2) | +0.6 (-2.2 to +3.4) | 0.801 | 72.2 (67.4 to 76.6) | +0.6 (-2.4 to +3.6) | 0.798 |
| JudgeBench, one response at a time | 63.4 (60.2 to 66.4) | 63.4 (60.0 to 66.8) | +0.0 (-2.6 to +2.4) | 1.000 | 67.0 (63.4 to 70.2) | +3.6 (+0.8 to +6.4) | 0.025 |
| RewardBench 2, best of four | 70.1 (65.8 to 74.4) | 72.0 (67.5 to 76.2) | +2.0 (-2.0 to +5.4) | 0.193 | 70.3 (65.9 to 74.5) | -1.7 (-4.9 to +1.5) | 0.200 |
| LLMBar | 72.4 (67.8 to 77.2) | 73.4 (68.6 to 78.2) | +1.0 (-2.6 to +4.4) | 0.620 | 78.6 (74.0 to 83.0) | +5.2 (+1.8 to +8.8) | 0.003 |
| BFCL V4, single turn | 77.6 (72.0 to 82.8) | 72.3 (66.7 to 78.0) | -5.3 (-10.1 to -0.6) | 0.343 | 78.3 (72.8 to 83.4) | +6.0 (+1.2 to +10.9) | 0.184 |
| JSONSchemaBench (sample of 1,000) | 96.2 (94.4 to 97.8) | 95.8 (94.0 to 97.4) | -0.4 (-2.0 to +1.2) | 0.804 | 96.2 (94.4 to 97.8) | +0.4 (-1.0 to +1.8) | 0.791 |
| IFEval | 67.8 (63.6 to 71.8) | 62.0 (57.8 to 66.2) | -5.8 (-10.2 to -1.0) | 0.021 | 73.6 (69.8 to 77.4) | +11.6 (+7.0 to +15.8) | <0.001 |
| MMLU-Pro (sample of 500) | 56.4 (52.0 to 61.0) | 55.8 (51.6 to 60.2) | -0.6 (-4.0 to +3.0) | 0.824 | 57.8 (53.4 to 62.2) | +2.0 (-1.4 to +5.2) | 0.275 |
| BFCL V4, multi turn (sample of 200) | 28.5 (22.0 to 35.0) | 16.0 (11.0 to 21.0) | -12.5 (-19.5 to -6.0) | <0.001 | 25.5 (20.0 to 32.0) | +9.5 (+3.5 to +15.5) | 0.004 |

## The watch cells under the main adapter

These benchmarks were never trained. A fall of more than 2 points is flagged.

| benchmark | before | after | difference | McNemar p | flag |
|---|---|---|---|---|---|
| JudgeBench, pairwise | 71.0 (66.2 to 75.4) | 72.2 (67.4 to 76.6) | +1.2 (-1.2 to +3.6) | 0.461 |  |
| JudgeBench, one response at a time | 63.4 (60.2 to 66.4) | 67.0 (63.4 to 70.2) | +3.6 (+0.8 to +6.4) | 0.030 |  |
| RewardBench 2, best of four | 70.1 (65.8 to 74.4) | 70.3 (65.9 to 74.5) | +0.2 (-2.9 to +3.1) | 1.000 |  |
| LLMBar | 72.4 (67.8 to 77.2) | 78.6 (74.0 to 83.0) | +6.2 (+3.6 to +9.0) | <0.001 |  |
| BFCL V4, single turn | 77.6 (72.0 to 82.8) | 78.3 (72.8 to 83.4) | +0.7 (-2.0 to +4.3) | 0.694 |  |
| JSONSchemaBench (sample of 1,000) | 96.2 (94.4 to 97.8) | 96.2 (94.4 to 97.8) | +0.0 (-1.6 to +1.6) | 1.000 |  |
| IFEval | 67.8 (63.6 to 71.8) | 73.6 (69.8 to 77.4) | +5.8 (+1.6 to +9.8) | 0.009 |  |
| MMLU-Pro (sample of 500) | 56.4 (52.0 to 61.0) | 57.8 (53.4 to 62.2) | +1.4 (-1.6 to +4.6) | 0.470 |  |
| BFCL V4, multi turn (sample of 200) | 28.5 (22.0 to 35.0) | 25.5 (20.0 to 32.0) | -3.0 (-9.0 to +3.0) | 0.418 | fell by more than 2 points |

## The main adapter when it is left free to reason

The adapter was trained to give the answer alone. Here it runs in the reasoning setting (free-running effort 0.0 with the 8,192-token limit, the study's reference), on the same items, against the base model in the same setting.

| benchmark | items | base | adapter | difference | McNemar p | thinking tokens per answer, base | adapter | replies at the limit, base | adapter |
|---|---|---|---|---|---|---|---|---|---|
| GSM8K | 1,319 | 96.7 (95.8 to 97.7) | 84.3 (82.3 to 86.2) | -12.4 (-14.2 to -10.7) | <0.001 | 226 | 162 | 1 | 0 |
| MBPP (sanitized test split) | 257 | 94.5 (91.4 to 97.3) | 94.2 (91.0 to 96.9) | -0.4 (-2.3 to +1.2) | 1.000 | 831 | 747 | 12 | 13 |
| JudgeBench, pairwise | 500 | 75.6 (70.4 to 80.0) | 80.0 (75.4 to 84.2) | +4.4 (+1.4 to +7.6) | 0.005 | 2,458 | 2,137 | 70 | 52 |
| JudgeBench, one response at a time | 500 | 75.0 (70.6 to 79.0) | 77.2 (73.0 to 81.0) | +2.2 (-0.6 to +5.0) | 0.185 | 2,188 | 1,750 | 46 | 32 |
| RewardBench 2, best of four | 500 | 79.5 (75.3 to 83.6) | 81.7 (77.6 to 85.6) | +2.2 (-0.7 to +5.5) | 0.533 | 1,034 | 765 | 27 | 14 |
| LLMBar | 500 | 85.0 (81.0 to 88.8) | 40.8 (35.8 to 45.8) | -44.2 (-49.6 to -38.8) | <0.001 | 262 | 215 | 0 | 0 |
| BFCL V4, single turn | 500 | 79.3 (73.5 to 84.8) | 80.2 (74.6 to 85.2) | +0.8 (-2.2 to +4.3) | 0.212 | 145 | 98 | 0 | 0 |
| JSONSchemaBench (sample of 1,000) | 500 | 98.4 (97.2 to 99.4) | 96.8 (95.2 to 98.4) | -1.6 (-3.2 to -0.2) | 0.077 | 312 | 203 | 1 | 0 |
| IFEval | 500 | 87.0 (84.0 to 89.8) | 85.2 (82.2 to 88.2) | -1.8 (-4.6 to +0.8) | 0.233 | 1,091 | 907 | 37 | 32 |
| MMLU-Pro (sample of 500) | 500 | 80.2 (76.6 to 83.4) | 77.4 (73.6 to 80.6) | -2.8 (-5.4 to -0.2) | 0.044 | 1,154 | 1,149 | 32 | 37 |
| BFCL V4, multi turn (sample of 200) | 200 | 37.0 (30.5 to 43.5) | 33.5 (27.5 to 40.5) | -3.5 (-10.0 to +3.0) | 0.371 | 1,728 | 1,172 | 1 | 1 |

## Every state on every benchmark, fast setting

GSM8K (target):

| state | score | minus base | McNemar p | prompt tokens per answer | completion tokens per answer | flag |
|---|---|---|---|---|---|---|
| base | 37.5 (34.8 to 40.1) |  |  | 99 | 18 |  |
| middle-2000-s1 | 42.1 (39.3 to 44.7) | +4.6 (+2.3 to +7.0) | <0.001 | 99 | 3 |  |
| random-2000-s1 | 42.8 (40.1 to 45.3) | +5.3 (+2.9 to +7.7) | <0.001 | 99 | 3 |  |
| middle-2000-s2 | 43.0 (40.3 to 45.6) | +5.5 (+3.3 to +7.7) | <0.001 | 99 | 3 |  |
| random-2000-s2 | 44.5 (41.9 to 47.2) | +7.0 (+4.5 to +9.5) | <0.001 | 99 | 3 |  |
| middle-2000-s3 | 44.8 (42.1 to 47.4) | +7.3 (+4.9 to +9.7) | <0.001 | 99 | 3 |  |
| random-2000-s3 | 44.6 (41.9 to 47.2) | +7.1 (+4.6 to +9.5) | <0.001 | 99 | 3 |  |
| shuffled-labels-2000 | 0.0 (0.0 to 0.0) | -37.5 (-40.1 to -34.8) | <0.001 | 99 | 4 |  |
| easy-2000 | 42.1 (39.5 to 44.7) | +4.6 (+2.3 to +7.0) | <0.001 | 99 | 3 |  |
| hard-2000 | 37.2 (34.6 to 39.9) | -0.2 (-2.6 to +2.0) | 0.893 | 99 | 3 |  |
| length-matched-random-2000 | 43.4 (40.8 to 46.1) | +5.9 (+3.6 to +8.3) | <0.001 | 99 | 3 |  |
| middle-250 | 41.7 (39.0 to 44.4) | +4.2 (+2.1 to +6.5) | <0.001 | 99 | 3 |  |
| random-250 | 40.5 (37.9 to 43.2) | +3.0 (+0.7 to +5.3) | 0.010 | 99 | 3 |  |
| middle-500 | 41.4 (38.8 to 44.0) | +3.9 (+1.7 to +6.2) | <0.001 | 99 | 4 |  |
| random-500 | 39.5 (36.8 to 42.1) | +2.1 (-0.2 to +4.3) | 0.088 | 99 | 3 |  |
| middle-1000 | 42.8 (40.1 to 45.4) | +5.3 (+3.0 to +7.7) | <0.001 | 99 | 3 |  |
| random-1000 | 41.2 (38.6 to 43.8) | +3.7 (+1.3 to +6.1) | 0.002 | 99 | 3 |  |
| middle-4000 | 45.0 (42.3 to 47.5) | +7.5 (+5.0 to +9.9) | <0.001 | 99 | 3 |  |
| random-4000 | 46.2 (43.5 to 48.8) | +8.8 (+6.4 to +11.1) | <0.001 | 99 | 3 |  |
| prompt-only | 35.9 (33.5 to 38.5) | -1.5 (-3.9 to +1.1) | 0.269 | 830 | 46 |  |

MBPP (sanitized test split) (target):

| state | score | minus base | McNemar p | prompt tokens per answer | completion tokens per answer | flag |
|---|---|---|---|---|---|---|
| base | 82.1 (77.4 to 86.8) |  |  | 160 | 73 |  |
| middle-2000-s1 | 85.6 (81.3 to 89.9) | +3.5 (+0.8 to +6.2) | 0.022 | 160 | 61 |  |
| random-2000-s1 | 83.7 (79.0 to 87.9) | +1.6 (-1.6 to +4.7) | 0.454 | 160 | 63 |  |
| middle-2000-s2 | 85.6 (80.9 to 89.9) | +3.5 (+0.4 to +7.0) | 0.064 | 160 | 62 |  |
| random-2000-s2 | 83.7 (79.0 to 87.9) | +1.6 (-1.6 to +4.7) | 0.454 | 160 | 62 |  |
| middle-2000-s3 | 83.7 (79.0 to 87.9) | +1.6 (-1.6 to +4.7) | 0.481 | 160 | 60 |  |
| random-2000-s3 | 84.8 (80.2 to 89.1) | +2.7 (-0.8 to +6.2) | 0.210 | 160 | 64 |  |
| shuffled-labels-2000 | 0.0 (0.0 to 0.0) | -82.1 (-86.8 to -77.4) | <0.001 | 160 | 4 |  |
| easy-2000 | 84.0 (79.4 to 88.7) | +1.9 (-1.2 to +5.1) | 0.332 | 160 | 70 |  |
| hard-2000 | 84.0 (79.4 to 88.3) | +1.9 (-1.2 to +5.1) | 0.332 | 160 | 68 |  |
| length-matched-random-2000 | 83.3 (78.6 to 87.5) | +1.2 (-1.2 to +3.5) | 0.508 | 160 | 69 |  |
| middle-250 | 84.8 (80.2 to 89.1) | +2.7 (-0.4 to +5.8) | 0.143 | 160 | 67 |  |
| random-250 | 82.9 (78.2 to 87.5) | +0.8 (-1.9 to +3.5) | 0.791 | 160 | 66 |  |
| middle-500 | 84.4 (79.8 to 88.7) | +2.3 (-0.4 to +5.1) | 0.180 | 160 | 70 |  |
| random-500 | 84.4 (79.8 to 88.7) | +2.3 (-0.4 to +5.1) | 0.180 | 160 | 63 |  |
| middle-1000 | 82.5 (77.8 to 87.2) | +0.4 (-3.1 to +3.9) | 1.000 | 160 | 69 |  |
| random-1000 | 84.0 (79.8 to 88.3) | +1.9 (-1.2 to +5.5) | 0.359 | 160 | 65 |  |
| middle-4000 | 86.0 (81.3 to 89.9) | +3.9 (+0.4 to +7.4) | 0.041 | 160 | 61 |  |
| random-4000 | 85.2 (80.9 to 89.5) | +3.1 (+0.0 to +6.2) | 0.077 | 160 | 60 |  |
| prompt-only | 84.4 (79.8 to 88.7) | +2.3 (-0.4 to +5.5) | 0.180 | 891 | 69 |  |

JudgeBench, pairwise (watch):

| state | score | minus base | McNemar p | prompt tokens per answer | completion tokens per answer | flag |
|---|---|---|---|---|---|---|
| base | 71.0 (66.2 to 75.4) |  |  | 1,395 | 4 |  |
| middle-2000-s1 | 72.2 (67.4 to 76.6) | +1.2 (-1.2 to +3.6) | 0.461 | 1,395 | 3 |  |
| random-2000-s1 | 71.6 (66.8 to 76.0) | +0.6 (-1.8 to +3.2) | 0.766 | 1,395 | 3 |  |
| middle-2000-s2 | 73.8 (69.2 to 78.0) | +2.8 (+0.4 to +5.2) | 0.059 | 1,395 | 3 |  |
| random-2000-s2 | 74.4 (69.8 to 78.6) | +3.4 (+0.2 to +6.6) | 0.050 | 1,395 | 3 |  |
| middle-2000-s3 | 74.8 (70.2 to 79.0) | +3.8 (+1.0 to +6.6) | 0.016 | 1,395 | 3 |  |
| random-2000-s3 | 70.2 (65.4 to 74.8) | -0.8 (-4.4 to +2.8) | 0.731 | 1,395 | 3 |  |
| shuffled-labels-2000 | 0.0 (0.0 to 0.0) | -71.0 (-75.4 to -66.2) | <0.001 | 1,395 | 4 | regression |
| easy-2000 | 72.6 (68.0 to 77.0) | +1.6 (-0.8 to +4.0) | 0.256 | 1,395 | 3 |  |
| hard-2000 | 74.0 (69.4 to 78.0) | +3.0 (+0.2 to +5.8) | 0.053 | 1,395 | 3 |  |
| length-matched-random-2000 | 73.6 (69.2 to 77.8) | +2.6 (+0.0 to +5.2) | 0.092 | 1,395 | 3 |  |
| middle-250 | 72.6 (68.0 to 76.8) | +1.6 (-1.2 to +4.4) | 0.322 | 1,395 | 3 |  |
| random-250 | 71.0 (66.6 to 75.2) | +0.0 (-2.4 to +2.6) | 1.000 | 1,395 | 3 |  |
| middle-500 | 73.2 (69.0 to 77.4) | +2.2 (-0.4 to +5.0) | 0.144 | 1,395 | 3 |  |
| random-500 | 72.6 (67.6 to 76.8) | +1.6 (-1.4 to +4.8) | 0.350 | 1,395 | 3 |  |
| middle-1000 | 73.6 (69.0 to 77.8) | +2.6 (+0.2 to +5.2) | 0.092 | 1,395 | 3 |  |
| random-1000 | 70.6 (66.2 to 74.8) | -0.4 (-3.2 to +2.4) | 0.896 | 1,395 | 3 |  |
| middle-4000 | 74.8 (70.0 to 79.0) | +3.8 (+1.0 to +6.8) | 0.016 | 1,395 | 3 |  |
| random-4000 | 72.6 (67.8 to 76.6) | +1.6 (-1.0 to +4.2) | 0.332 | 1,395 | 3 |  |
| prompt-only | 71.6 (66.8 to 76.2) | +0.6 (-2.2 to +3.4) | 0.801 | 2,126 | 3 |  |

JudgeBench, one response at a time (watch):

| state | score | minus base | McNemar p | prompt tokens per answer | completion tokens per answer | flag |
|---|---|---|---|---|---|---|
| base | 63.4 (60.2 to 66.4) |  |  | 856 | 4 |  |
| middle-2000-s1 | 67.0 (63.4 to 70.2) | +3.6 (+0.8 to +6.4) | 0.030 | 856 | 3 |  |
| random-2000-s1 | 67.4 (63.6 to 70.8) | +4.0 (+0.8 to +7.0) | 0.029 | 856 | 3 |  |
| middle-2000-s2 | 66.2 (63.0 to 69.4) | +2.8 (-0.0 to +5.6) | 0.146 | 856 | 3 |  |
| random-2000-s2 | 68.6 (65.2 to 71.8) | +5.2 (+2.0 to +8.4) | 0.011 | 856 | 3 |  |
| middle-2000-s3 | 68.0 (64.2 to 71.4) | +4.6 (+1.6 to +7.6) | 0.014 | 856 | 3 |  |
| random-2000-s3 | 66.2 (62.4 to 69.6) | +2.8 (-0.4 to +6.0) | 0.193 | 856 | 3 |  |
| shuffled-labels-2000 | 0.0 (0.0 to 0.0) | -63.4 (-66.4 to -60.2) | <0.001 | 856 | 4 | regression |
| easy-2000 | 66.0 (62.6 to 69.2) | +2.6 (-0.4 to +5.6) | 0.171 | 856 | 3 |  |
| hard-2000 | 68.4 (64.8 to 71.6) | +5.0 (+2.0 to +7.8) | 0.005 | 856 | 3 |  |
| length-matched-random-2000 | 67.6 (63.8 to 71.0) | +4.2 (+1.0 to +7.0) | 0.026 | 856 | 3 |  |
| middle-250 | 66.0 (62.4 to 69.2) | +2.6 (+0.2 to +5.0) | 0.060 | 856 | 3 |  |
| random-250 | 65.8 (62.2 to 69.4) | +2.4 (-0.2 to +5.0) | 0.126 | 856 | 3 |  |
| middle-500 | 66.2 (62.6 to 69.4) | +2.8 (+0.0 to +5.4) | 0.076 | 856 | 3 |  |
| random-500 | 65.4 (61.8 to 68.8) | +2.0 (-1.0 to +4.8) | 0.320 | 856 | 3 |  |
| middle-1000 | 66.6 (62.8 to 70.0) | +3.2 (+0.2 to +6.2) | 0.060 | 856 | 3 |  |
| random-1000 | 67.4 (64.0 to 70.6) | +4.0 (+1.2 to +7.0) | 0.022 | 856 | 3 |  |
| middle-4000 | 67.4 (63.8 to 70.8) | +4.0 (+1.0 to +7.0) | 0.033 | 856 | 3 |  |
| random-4000 | 65.8 (61.8 to 69.4) | +2.4 (-0.8 to +5.6) | 0.235 | 856 | 3 |  |
| prompt-only | 63.4 (60.0 to 66.8) | +0.0 (-2.6 to +2.4) | 1.000 | 1,587 | 3 |  |

RewardBench 2, best of four (watch):

| state | score | minus base | McNemar p | prompt tokens per answer | completion tokens per answer | flag |
|---|---|---|---|---|---|---|
| base | 70.1 (65.8 to 74.4) |  |  | 1,845 | 3 |  |
| middle-2000-s1 | 70.3 (65.9 to 74.5) | +0.2 (-2.9 to +3.1) | 1.000 | 1,845 | 3 |  |
| random-2000-s1 | 67.5 (62.8 to 71.8) | -2.5 (-5.8 to +0.5) | 0.023 | 1,845 | 3 | regression |
| middle-2000-s2 | 70.7 (66.1 to 75.1) | +0.7 (-2.7 to +3.9) | 0.500 | 1,845 | 3 |  |
| random-2000-s2 | 69.7 (65.2 to 74.1) | -0.4 (-3.5 to +2.8) | 0.104 | 1,845 | 3 |  |
| middle-2000-s3 | 68.9 (64.3 to 73.2) | -1.2 (-4.5 to +2.1) | 0.152 | 1,845 | 3 |  |
| random-2000-s3 | 69.7 (65.2 to 74.2) | -0.4 (-3.5 to +2.6) | 0.430 | 1,845 | 3 |  |
| shuffled-labels-2000 | 0.0 (0.0 to 0.0) | -70.1 (-74.4 to -65.8) | <0.001 | 1,845 | 4 | regression |
| easy-2000 | 69.9 (65.5 to 74.4) | -0.2 (-3.0 to +2.8) | 0.585 | 1,845 | 3 |  |
| hard-2000 | 68.7 (64.2 to 73.3) | -1.3 (-4.5 to +1.7) | 0.066 | 1,845 | 3 |  |
| length-matched-random-2000 | 67.2 (63.0 to 71.4) | -2.9 (-6.2 to +0.3) | 0.049 | 1,845 | 3 | regression |
| middle-250 | 71.6 (67.1 to 76.0) | +1.6 (-1.2 to +4.4) | 1.000 | 1,845 | 3 |  |
| random-250 | 71.5 (66.9 to 75.9) | +1.4 (-1.4 to +4.4) | 1.000 | 1,845 | 3 |  |
| middle-500 | 69.7 (65.1 to 74.2) | -0.3 (-3.3 to +2.5) | 0.500 | 1,845 | 3 |  |
| random-500 | 70.4 (65.9 to 74.6) | +0.4 (-2.4 to +3.1) | 1.000 | 1,845 | 3 |  |
| middle-1000 | 70.2 (65.8 to 74.5) | +0.1 (-2.8 to +3.1) | 0.878 | 1,845 | 3 |  |
| random-1000 | 69.0 (64.8 to 73.6) | -1.0 (-4.1 to +2.0) | 0.126 | 1,845 | 3 |  |
| middle-4000 | 67.0 (62.4 to 71.6) | -3.0 (-6.0 to -0.3) | 0.003 | 1,845 | 3 | regression |
| random-4000 | 68.2 (63.8 to 72.7) | -1.8 (-4.9 to +1.2) | 0.117 | 1,845 | 3 |  |
| prompt-only | 72.0 (67.5 to 76.2) | +2.0 (-2.0 to +5.4) | 0.193 | 2,576 | 3 |  |

LLMBar (watch):

| state | score | minus base | McNemar p | prompt tokens per answer | completion tokens per answer | flag |
|---|---|---|---|---|---|---|
| base | 72.4 (67.8 to 77.2) |  |  | 334 | 3 |  |
| middle-2000-s1 | 78.6 (74.0 to 83.0) | +6.2 (+3.6 to +9.0) | <0.001 | 334 | 3 |  |
| random-2000-s1 | 71.8 (67.0 to 76.6) | -0.6 (-4.0 to +2.6) | 0.795 | 334 | 3 |  |
| middle-2000-s2 | 74.0 (69.0 to 78.6) | +1.6 (-1.2 to +4.6) | 0.312 | 334 | 3 |  |
| random-2000-s2 | 73.8 (69.0 to 78.6) | +1.4 (-1.6 to +4.4) | 0.392 | 334 | 3 |  |
| middle-2000-s3 | 75.0 (70.2 to 79.4) | +2.6 (-0.4 to +5.6) | 0.117 | 334 | 3 |  |
| random-2000-s3 | 74.4 (69.8 to 79.0) | +2.0 (-0.8 to +4.8) | 0.174 | 334 | 3 |  |
| shuffled-labels-2000 | 0.6 (0.0 to 1.4) | -71.8 (-76.6 to -67.0) | <0.001 | 334 | 4 | regression |
| easy-2000 | 73.4 (68.4 to 78.2) | +1.0 (-2.0 to +3.8) | 0.551 | 334 | 3 |  |
| hard-2000 | 73.0 (68.2 to 77.8) | +0.6 (-2.2 to +3.2) | 0.749 | 334 | 3 |  |
| length-matched-random-2000 | 76.2 (71.6 to 80.8) | +3.8 (+1.4 to +6.4) | 0.003 | 334 | 3 |  |
| middle-250 | 74.6 (70.0 to 79.2) | +2.2 (-0.4 to +5.0) | 0.135 | 334 | 3 |  |
| random-250 | 74.6 (69.8 to 79.2) | +2.2 (-0.2 to +4.6) | 0.080 | 334 | 3 |  |
| middle-500 | 73.8 (69.4 to 78.4) | +1.4 (-1.2 to +4.0) | 0.360 | 334 | 3 |  |
| random-500 | 75.2 (70.4 to 80.0) | +2.8 (+0.0 to +5.6) | 0.059 | 334 | 3 |  |
| middle-1000 | 76.0 (71.6 to 80.4) | +3.6 (+1.4 to +6.0) | 0.004 | 334 | 3 |  |
| random-1000 | 74.6 (69.8 to 79.4) | +2.2 (-0.6 to +5.2) | 0.144 | 334 | 3 |  |
| middle-4000 | 75.6 (71.0 to 80.2) | +3.2 (+0.4 to +6.2) | 0.029 | 334 | 3 |  |
| random-4000 | 77.0 (72.4 to 81.6) | +4.6 (+1.8 to +7.6) | 0.002 | 334 | 3 |  |
| prompt-only | 73.4 (68.6 to 78.2) | +1.0 (-2.6 to +4.4) | 0.620 | 1,065 | 3 |  |

BFCL V4, single turn (watch):

| state | score | minus base | McNemar p | prompt tokens per answer | completion tokens per answer | flag |
|---|---|---|---|---|---|---|
| base | 77.6 (72.0 to 82.8) |  |  | 1,278 | 59 |  |
| middle-2000-s1 | 78.3 (72.8 to 83.4) | +0.7 (-2.0 to +4.3) | 0.694 | 1,278 | 46 |  |
| random-2000-s1 | 78.7 (73.2 to 83.8) | +1.1 (-1.8 to +4.6) | 0.512 | 1,278 | 50 |  |
| middle-2000-s2 | 78.8 (73.3 to 83.8) | +1.2 (-2.0 to +5.2) | 0.464 | 1,278 | 46 |  |
| random-2000-s2 | 78.8 (73.3 to 83.8) | +1.3 (-1.8 to +4.9) | 0.712 | 1,278 | 45 |  |
| middle-2000-s3 | 80.5 (74.9 to 85.4) | +2.9 (-0.0 to +6.5) | 0.114 | 1,278 | 44 |  |
| random-2000-s3 | 80.0 (74.6 to 85.1) | +2.5 (-0.6 to +6.0) | 0.349 | 1,278 | 50 |  |
| shuffled-labels-2000 | 18.2 (18.2 to 18.2) | -59.4 (-64.6 to -53.8) | <0.001 | 1,278 | 4 | regression |
| easy-2000 | 79.0 (73.3 to 84.4) | +1.4 (-0.5 to +3.6) | 0.119 | 1,278 | 50 |  |
| hard-2000 | 73.0 (66.9 to 78.7) | -4.6 (-9.3 to -0.2) | 0.001 | 1,278 | 52 | regression |
| length-matched-random-2000 | 79.1 (73.9 to 83.5) | +1.6 (-1.8 to +5.3) | 0.603 | 1,278 | 48 |  |
| middle-250 | 75.5 (69.4 to 81.3) | -2.0 (-7.3 to +2.4) | 0.207 | 1,278 | 63 | regression |
| random-250 | 73.8 (67.5 to 79.8) | -3.8 (-10.2 to +1.8) | 0.747 | 1,278 | 64 | regression |
| middle-500 | 77.8 (72.2 to 83.2) | +0.3 (-3.1 to +3.4) | 0.165 | 1,278 | 58 |  |
| random-500 | 78.1 (73.0 to 82.6) | +0.5 (-3.3 to +4.8) | 0.897 | 1,278 | 50 |  |
| middle-1000 | 81.4 (75.9 to 86.7) | +3.9 (+1.7 to +6.1) | <0.001 | 1,278 | 54 |  |
| random-1000 | 79.1 (73.5 to 84.4) | +1.6 (-0.8 to +4.1) | 0.012 | 1,278 | 53 |  |
| middle-4000 | 80.6 (75.1 to 85.4) | +3.0 (+0.1 to +6.6) | 0.550 | 1,278 | 42 |  |
| random-4000 | 77.8 (72.3 to 82.8) | +0.2 (-3.1 to +4.0) | 0.362 | 1,278 | 40 |  |
| prompt-only | 72.3 (66.7 to 78.0) | -5.3 (-10.1 to -0.6) | 0.343 | 2,009 | 56 | regression |

JSONSchemaBench (sample of 1,000) (watch):

| state | score | minus base | McNemar p | prompt tokens per answer | completion tokens per answer | flag |
|---|---|---|---|---|---|---|
| base | 96.2 (94.4 to 97.8) |  |  | 2,038 | 151 |  |
| middle-2000-s1 | 96.2 (94.4 to 97.8) | +0.0 (-1.6 to +1.6) | 1.000 | 2,038 | 116 |  |
| random-2000-s1 | 95.2 (93.4 to 97.0) | -1.0 (-2.6 to +0.6) | 0.332 | 2,038 | 169 |  |
| middle-2000-s2 | 96.4 (94.6 to 97.8) | +0.2 (-1.0 to +1.4) | 1.000 | 2,038 | 119 |  |
| random-2000-s2 | 95.4 (93.6 to 97.2) | -0.8 (-2.2 to +0.4) | 0.388 | 2,038 | 132 |  |
| middle-2000-s3 | 96.6 (95.0 to 98.0) | +0.4 (-1.2 to +2.0) | 0.804 | 2,038 | 116 |  |
| random-2000-s3 | 95.8 (94.0 to 97.4) | -0.4 (-1.6 to +0.8) | 0.754 | 2,038 | 137 |  |
| shuffled-labels-2000 | 2.6 (1.4 to 4.0) | -93.6 (-95.6 to -91.2) | <0.001 | 2,038 | 4 | regression |
| easy-2000 | 96.4 (94.8 to 98.0) | +0.2 (-1.2 to +1.4) | 1.000 | 2,038 | 122 |  |
| hard-2000 | 93.8 (91.8 to 95.8) | -2.4 (-4.2 to -0.6) | 0.012 | 2,038 | 249 | regression |
| length-matched-random-2000 | 96.8 (95.2 to 98.4) | +0.6 (-0.4 to +1.8) | 0.508 | 2,038 | 146 |  |
| middle-250 | 95.6 (93.8 to 97.4) | -0.6 (-2.0 to +0.6) | 0.549 | 2,038 | 151 |  |
| random-250 | 95.4 (93.6 to 97.2) | -0.8 (-2.0 to +0.6) | 0.388 | 2,038 | 202 |  |
| middle-500 | 96.2 (94.6 to 97.8) | +0.0 (-1.4 to +1.4) | 1.000 | 2,038 | 153 |  |
| random-500 | 95.4 (93.4 to 97.2) | -0.8 (-2.2 to +0.4) | 0.388 | 2,038 | 117 |  |
| middle-1000 | 95.8 (94.0 to 97.4) | -0.4 (-2.0 to +1.2) | 0.804 | 2,038 | 125 |  |
| random-1000 | 94.6 (92.4 to 96.4) | -1.6 (-3.2 to -0.2) | 0.057 | 2,038 | 155 |  |
| middle-4000 | 95.2 (93.2 to 97.0) | -1.0 (-2.4 to +0.4) | 0.267 | 2,038 | 118 |  |
| random-4000 | 95.8 (94.0 to 97.4) | -0.4 (-1.6 to +0.8) | 0.754 | 2,038 | 132 |  |
| prompt-only | 95.8 (94.0 to 97.4) | -0.4 (-2.0 to +1.2) | 0.804 | 2,769 | 111 |  |

IFEval (watch):

| state | score | minus base | McNemar p | prompt tokens per answer | completion tokens per answer | flag |
|---|---|---|---|---|---|---|
| base | 67.8 (63.6 to 71.8) |  |  | 64 | 296 |  |
| middle-2000-s1 | 73.6 (69.8 to 77.4) | +5.8 (+1.6 to +9.8) | 0.009 | 64 | 208 |  |
| random-2000-s1 | 72.0 (68.0 to 75.8) | +4.2 (-0.2 to +8.6) | 0.064 | 64 | 244 |  |
| middle-2000-s2 | 71.8 (67.8 to 75.6) | +4.0 (-0.4 to +8.2) | 0.072 | 64 | 177 |  |
| random-2000-s2 | 71.6 (67.6 to 75.4) | +3.8 (-0.6 to +8.0) | 0.107 | 64 | 196 |  |
| middle-2000-s3 | 75.6 (72.0 to 79.4) | +7.8 (+3.6 to +12.2) | <0.001 | 64 | 170 |  |
| random-2000-s3 | 69.2 (65.2 to 73.2) | +1.4 (-3.2 to +5.8) | 0.600 | 64 | 189 |  |
| shuffled-labels-2000 | 20.6 (17.0 to 24.2) | -47.2 (-52.4 to -42.0) | <0.001 | 64 | 22 | regression |
| easy-2000 | 72.8 (69.0 to 76.6) | +5.0 (+1.0 to +9.0) | 0.019 | 64 | 194 |  |
| hard-2000 | 73.6 (69.4 to 77.6) | +5.8 (+1.6 to +10.0) | 0.008 | 64 | 306 |  |
| length-matched-random-2000 | 72.2 (68.4 to 76.0) | +4.4 (+0.0 to +9.0) | 0.063 | 64 | 250 |  |
| middle-250 | 67.8 (63.6 to 71.8) | +0.0 (-4.0 to +4.0) | 1.000 | 64 | 242 |  |
| random-250 | 66.8 (62.6 to 71.0) | -1.0 (-5.2 to +3.0) | 0.699 | 64 | 292 |  |
| middle-500 | 69.8 (65.8 to 73.6) | +2.0 (-2.2 to +5.8) | 0.387 | 64 | 267 |  |
| random-500 | 69.2 (65.0 to 73.2) | +1.4 (-3.0 to +5.6) | 0.579 | 64 | 155 |  |
| middle-1000 | 71.6 (67.8 to 75.4) | +3.8 (-0.4 to +8.0) | 0.099 | 64 | 221 |  |
| random-1000 | 72.4 (68.4 to 76.2) | +4.6 (+0.6 to +8.6) | 0.035 | 64 | 216 |  |
| middle-4000 | 74.0 (70.0 to 77.8) | +6.2 (+2.2 to +10.2) | 0.004 | 64 | 191 |  |
| random-4000 | 75.6 (71.6 to 79.2) | +7.8 (+3.6 to +12.0) | <0.001 | 64 | 180 |  |
| prompt-only | 62.0 (57.8 to 66.2) | -5.8 (-10.2 to -1.0) | 0.021 | 795 | 182 | regression |

MMLU-Pro (sample of 500) (watch):

| state | score | minus base | McNemar p | prompt tokens per answer | completion tokens per answer | flag |
|---|---|---|---|---|---|---|
| base | 56.4 (52.0 to 61.0) |  |  | 224 | 4 |  |
| middle-2000-s1 | 57.8 (53.4 to 62.2) | +1.4 (-1.6 to +4.6) | 0.470 | 224 | 3 |  |
| random-2000-s1 | 59.0 (54.8 to 63.4) | +2.6 (-0.8 to +6.0) | 0.177 | 224 | 3 |  |
| middle-2000-s2 | 59.0 (54.8 to 63.2) | +2.6 (-0.8 to +6.0) | 0.171 | 224 | 3 |  |
| random-2000-s2 | 56.8 (52.6 to 61.2) | +0.4 (-3.2 to +3.8) | 0.910 | 224 | 3 |  |
| middle-2000-s3 | 58.6 (54.2 to 63.0) | +2.2 (-1.2 to +5.6) | 0.254 | 224 | 3 |  |
| random-2000-s3 | 57.4 (53.0 to 61.8) | +1.0 (-2.4 to +4.2) | 0.644 | 224 | 3 |  |
| shuffled-labels-2000 | 0.0 (0.0 to 0.0) | -56.4 (-61.0 to -52.0) | <0.001 | 224 | 4 | regression |
| easy-2000 | 57.2 (52.8 to 61.6) | +0.8 (-2.2 to +3.8) | 0.699 | 224 | 3 |  |
| hard-2000 | 59.6 (55.4 to 64.0) | +3.2 (+0.0 to +6.4) | 0.072 | 224 | 3 |  |
| length-matched-random-2000 | 58.0 (53.6 to 62.4) | +1.6 (-1.6 to +5.0) | 0.428 | 224 | 3 |  |
| middle-250 | 55.8 (51.4 to 60.2) | -0.6 (-3.8 to +2.6) | 0.804 | 224 | 3 |  |
| random-250 | 57.2 (52.8 to 61.6) | +0.8 (-2.2 to +3.8) | 0.694 | 224 | 3 |  |
| middle-500 | 57.2 (53.0 to 61.6) | +0.8 (-2.2 to +3.8) | 0.694 | 224 | 3 |  |
| random-500 | 55.8 (51.4 to 60.2) | -0.6 (-4.0 to +2.6) | 0.820 | 224 | 3 |  |
| middle-1000 | 56.2 (52.0 to 60.8) | -0.2 (-3.2 to +2.8) | 1.000 | 224 | 3 |  |
| random-1000 | 58.4 (54.0 to 62.8) | +2.0 (-1.2 to +5.4) | 0.282 | 224 | 3 |  |
| middle-4000 | 55.6 (51.2 to 60.0) | -0.8 (-4.0 to +2.6) | 0.728 | 224 | 3 |  |
| random-4000 | 57.8 (53.4 to 62.2) | +1.4 (-2.2 to +5.2) | 0.534 | 224 | 3 |  |
| prompt-only | 55.8 (51.6 to 60.2) | -0.6 (-4.0 to +3.0) | 0.824 | 955 | 3 |  |

BFCL V4, multi turn (sample of 200) (watch):

| state | score | minus base | McNemar p | prompt tokens per answer | completion tokens per answer | flag |
|---|---|---|---|---|---|---|
| base | 28.5 (22.0 to 35.0) |  |  | 182,346 | 608 |  |
| middle-2000-s1 | 25.5 (20.0 to 32.0) | -3.0 (-9.0 to +3.0) | 0.418 | 177,622 | 520 | regression |
| random-2000-s1 | 30.0 (24.0 to 36.5) | +1.5 (-5.5 to +8.5) | 0.775 | 148,239 | 480 |  |
| middle-2000-s2 | 25.5 (19.5 to 31.5) | -3.0 (-10.0 to +3.5) | 0.480 | 168,158 | 481 | regression |
| random-2000-s2 | 36.5 (30.0 to 43.5) | +8.0 (+1.5 to +14.5) | 0.026 | 180,585 | 526 |  |
| middle-2000-s3 | 28.0 (22.0 to 34.5) | -0.5 (-6.5 to +6.0) | 1.000 | 161,448 | 471 |  |
| random-2000-s3 | 36.0 (29.5 to 42.5) | +7.5 (+0.0 to +14.5) | 0.044 | 153,414 | 472 |  |
| shuffled-labels-2000 | 0.0 (0.0 to 0.0) | -28.5 (-35.0 to -22.0) | <0.001 | 44,725 | 18 | regression |
| easy-2000 | 23.0 (17.5 to 29.0) | -5.5 (-11.0 to +0.5) | 0.090 | 154,982 | 498 | regression |
| hard-2000 | 23.5 (18.0 to 29.5) | -5.0 (-10.5 to +1.0) | 0.143 | 165,611 | 548 | regression |
| length-matched-random-2000 | 24.5 (18.5 to 30.5) | -4.0 (-10.5 to +3.0) | 0.302 | 164,996 | 475 | regression |
| middle-250 | 11.0 (7.0 to 15.5) | -17.5 (-24.0 to -11.0) | <0.001 | 110,939 | 382 | regression |
| random-250 | 11.0 (6.5 to 15.5) | -17.5 (-24.0 to -11.0) | <0.001 | 91,008 | 284 | regression |
| middle-500 | 8.5 (5.0 to 12.5) | -20.0 (-27.0 to -13.0) | <0.001 | 95,871 | 282 | regression |
| random-500 | 19.0 (13.5 to 24.5) | -9.5 (-15.5 to -3.0) | 0.005 | 139,682 | 424 | regression |
| middle-1000 | 18.0 (13.0 to 23.5) | -10.5 (-17.0 to -4.5) | 0.002 | 107,770 | 326 | regression |
| random-1000 | 18.5 (13.5 to 24.0) | -10.0 (-16.5 to -3.5) | 0.005 | 134,046 | 394 | regression |
| middle-4000 | 27.5 (21.5 to 34.0) | -1.0 (-8.0 to +6.0) | 0.890 | 179,769 | 499 |  |
| random-4000 | 25.5 (20.0 to 31.5) | -3.0 (-9.5 to +3.5) | 0.471 | 201,894 | 533 | regression |
| prompt-only | 16.0 (11.0 to 21.0) | -12.5 (-19.5 to -6.0) | <0.001 | 131,447 | 454 | regression |

## Cost

The evaluation made 152,189 sampling calls and read 25,349 replies from the cache: 571,394,850 prompt tokens and 10,236,499 completion tokens. 0 calls failed, 0 were retried and 0 saw a throttling event.
A fast-setting pass over one adapter took between 17.4 and 29.5 minutes of wall time.
