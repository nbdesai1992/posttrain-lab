# Training progress

Updated 2026-11-27 15:06 UTC. 18 of 18 runs are done, 10.91 hours after the first one started. All runs are done.

| run | updates | minutes | s per update | final loss | dev middle | dev easy | dev hard | stalls and pauses |
|---|---|---|---|---|---|---|---|---|
| middle-2000-s1 | 252 | 39 | 8.52 | 1.709 | 19.3% | 88.0% | 0.7% | 9 updates of 30 s or more, 9.8 min in all, longest 106 s |
| random-2000-s1 | 252 | 42 | 8.91 | 2.090 | 14.0% | 88.0% | 0.0% | 8 updates of 30 s or more, 10.7 min in all, longest 215 s |
| middle-2000-s2 | 252 | 42 | 8.83 | 1.678 | 14.0% | 88.0% | 0.0% | 5 updates of 30 s or more, 7.2 min in all, longest 165 s; paused_rate_limit reported 2 times |
| random-2000-s2 | 252 | 70 | 15.69 | 1.997 | 12.7% | 90.0% | 0.0% | 20 updates of 30 s or more, 36.7 min in all, longest 436 s |
| middle-2000-s3 | 252 | 36 | 7.84 | 1.563 | 20.7% | 89.3% | 0.7% | 6 updates of 30 s or more, 8.2 min in all, longest 148 s |
| random-2000-s3 | 252 | 33 | 6.97 | 2.067 | 13.3% | 88.0% | 0.0% | 3 updates of 30 s or more, 3.6 min in all, longest 95 s |
| shuffled-labels-2000 | 252 | 50 | 10.44 | 2.594 | 0.0% | 0.0% | 0.0% | 10 updates of 30 s or more, 15.4 min in all, longest 222 s |
| easy-2000 | 252 | 38 | 8.19 | 0.114 | 10.7% | 90.0% | 0.0% | 6 updates of 30 s or more, 6.2 min in all, longest 90 s |
| hard-2000 | 252 | 35 | 7.5 | 3.965 | 6.0% | 87.3% | 0.0% | 5 updates of 30 s or more, 4.4 min in all, longest 84 s |
| length-matched-random-2000 | 252 | 46 | 10.06 | 1.883 | 12.0% | 89.3% | 0.7% | 13 updates of 30 s or more, 15.0 min in all, longest 184 s |
| middle-250 | 32 | 7 | 7.27 | 2.061 | 7.3% | 90.0% | 0.0% | 1 updates of 30 s or more, 1.0 min in all, longest 57 s |
| random-250 | 32 | 7 | 6.74 | 2.494 | 8.0% | 89.3% | 0.0% | none |
| middle-500 | 64 | 11 | 6.95 | 1.958 | 11.3% | 90.0% | 0.0% | 1 updates of 30 s or more, 0.9 min in all, longest 54 s |
| random-500 | 64 | 13 | 8.39 | 2.422 | 8.7% | 86.7% | 0.0% | 2 updates of 30 s or more, 1.1 min in all, longest 34 s |
| middle-1000 | 128 | 16 | 6.05 | 1.864 | 16.0% | 89.3% | 0.0% | 1 updates of 30 s or more, 0.6 min in all, longest 35 s |
| random-1000 | 128 | 19 | 7.55 | 2.096 | 10.7% | 88.7% | 0.0% | 3 updates of 30 s or more, 3.2 min in all, longest 94 s |
| middle-4000 | 496 | 62 | 7.11 | 1.664 | 19.3% | 88.7% | 0.7% | 10 updates of 30 s or more, 9.2 min in all, longest 124 s |
| random-4000 | 496 | 75 | 8.57 | 2.015 | 17.3% | 89.3% | 0.0% | 13 updates of 30 s or more, 14.2 min in all, longest 111 s |

Still to run: nothing.

Anything unusual: updates took a median of 5.66 s while adapters were being evaluated (3264 updates) and 5.92 s otherwise (696 updates); the means were 8.79 s and 8.14 s.

Adapters are evaluated while later runs train, at 16 requests in flight. The evaluation first ran at 32 in flight, from 04:27 to 05:10 UTC on 2026-11-27, for the prompt-only state and the main adapter's fast pass. Over that stretch an update took 9.9 s on average, against 8.0 s for the 344 updates made without it up to 05:53 UTC. That is 24% slower, with an interval from 8% faster to 67% slower when groups of 16 updates are resampled. The median update did not change, and the service reported no rate-limit pause during updates. The evaluation was stopped from 05:10 to 05:56 UTC to measure training alone, and then started again at half the concurrency. It goes back to after training if 500 or more updates made beside it average more than a quarter above 8.0 s, or if the service reports a rate-limit pause during updates.

Every run trains in its own process. The third run started in the first process sat in the service's queue for seven minutes, with `paused_rate_limit` reported 13 times and no update made, as the third pilot run had. The process was stopped at 05:41 UTC and the run began two minutes into a fresh one. A run still waits about two minutes in the queue before its first update.

Evaluation, as of 2026-11-27 15:05 UTC: 21 states are complete (base, middle-2000-s1, random-2000-s1, middle-2000-s2, random-2000-s2, middle-2000-s3, random-2000-s3, shuffled-labels-2000, easy-2000, hard-2000, length-matched-random-2000, middle-250, random-250, middle-500, random-500, middle-1000, random-1000, middle-4000, prompt-only, base-reasoning, middle-2000-s1-reasoning).
