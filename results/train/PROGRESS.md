# Training progress

Updated 2026-11-27 04:51 UTC. 1 of 18 runs are done, 0.66 hours after the first one started. Training now: random-2000-s1, since 04:51 UTC.

| run | updates | minutes | s per update | final loss | dev middle | dev easy | dev hard | stalls and pauses |
|---|---|---|---|---|---|---|---|---|
| middle-2000-s1 | 252 | 39 | 8.52 | 1.709 | 19.3% | 88.0% | 0.7% | 9 updates of 30 s or more, 9.8 min in all, longest 106 s |

Still to run: random-2000-s1, middle-2000-s2, random-2000-s2, middle-2000-s3, random-2000-s3, shuffled-labels-2000, easy-2000, hard-2000, length-matched-random-2000, middle-250, random-250, middle-500, random-500, middle-1000, random-1000, middle-4000, random-4000.

Anything unusual: updates took a median of 5.55 s while adapters were being evaluated (112 updates) and 5.96 s otherwise (140 updates); the means were 9.93 s and 7.4 s.

Evaluation, as of 2026-11-27 04:50 UTC: 3 states are complete (base, prompt-only, base-reasoning).

Evaluation is running in series, after training. It overlapped training from 04:27 to 05:10 UTC at 32 requests in flight (the prompt-only state and the main adapter's fast pass). In that stretch updates averaged 9.9 s against 7.2 s alone, 37% slower, because stalls were more frequent and longer (the median update was unchanged at 5.6 s against 5.8 s, and the service reported no rate-limit pause). That is past the quarter that was allowed, so the evaluation was stopped at 05:10 UTC and will start again when the last run has finished. The main adapter's fast pass is complete: GSM8K 42.1 against 37.5 for the base model, MBPP 85.6 against 82.1.
