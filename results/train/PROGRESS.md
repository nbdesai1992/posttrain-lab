# Training progress

Updated 2026-11-27 09:33 UTC. 7 of 18 runs are done, 5.36 hours after the first one started. Training now: easy-2000, since 09:33 UTC.

| run | updates | minutes | s per update | final loss | dev middle | dev easy | dev hard | stalls and pauses |
|---|---|---|---|---|---|---|---|---|
| middle-2000-s1 | 252 | 39 | 8.52 | 1.709 | 19.3% | 88.0% | 0.7% | 9 updates of 30 s or more, 9.8 min in all, longest 106 s |
| random-2000-s1 | 252 | 42 | 8.91 | 2.090 | 14.0% | 88.0% | 0.0% | 8 updates of 30 s or more, 10.7 min in all, longest 215 s |
| middle-2000-s2 | 252 | 42 | 8.83 | 1.678 | 14.0% | 88.0% | 0.0% | 5 updates of 30 s or more, 7.2 min in all, longest 165 s; paused_rate_limit reported 2 times |
| random-2000-s2 | 252 | 70 | 15.69 | 1.997 | 12.7% | 90.0% | 0.0% | 20 updates of 30 s or more, 36.7 min in all, longest 436 s |
| middle-2000-s3 | 252 | 36 | 7.84 | 1.563 | 20.7% | 89.3% | 0.7% | 6 updates of 30 s or more, 8.2 min in all, longest 148 s |
| random-2000-s3 | 252 | 33 | 6.97 | 2.067 | 13.3% | 88.0% | 0.0% | 3 updates of 30 s or more, 3.6 min in all, longest 95 s |
| shuffled-labels-2000 | 252 | 50 | 10.44 | 2.594 | 0.0% | 0.0% | 0.0% | 10 updates of 30 s or more, 15.4 min in all, longest 222 s |

Still to run: easy-2000, hard-2000, length-matched-random-2000, middle-250, random-250, middle-500, random-500, middle-1000, random-1000, middle-4000, random-4000.

Anything unusual: updates took a median of 5.69 s while adapters were being evaluated (1344 updates) and 5.93 s otherwise (420 updates); the means were 10.07 s and 8.09 s.

Adapters are evaluated while later runs train, at 16 requests in flight. The evaluation first ran at 32 in flight, from 04:27 to 05:10 UTC on 2026-11-27, for the prompt-only state and the main adapter's fast pass. Over that stretch an update took 9.9 s on average, against 8.0 s for the 344 updates made without it up to 05:53 UTC. That is 24% slower, with an interval from 8% faster to 67% slower when groups of 16 updates are resampled. The median update did not change, and the service reported no rate-limit pause during updates. The evaluation was stopped from 05:10 to 05:56 UTC to measure training alone, and then started again at half the concurrency. It goes back to after training if 500 or more updates made beside it average more than a quarter above 8.0 s, or if the service reports a rate-limit pause during updates.

Every run trains in its own process. The third run started in the first process sat in the service's queue for seven minutes, with `paused_rate_limit` reported 13 times and no update made, as the third pilot run had. The process was stopped at 05:41 UTC and the run began two minutes into a fresh one. A run still waits about two minutes in the queue before its first update.

Evaluation, as of 2026-11-27 09:32 UTC: 10 states are complete (base, middle-2000-s1, random-2000-s1, middle-2000-s2, random-2000-s2, middle-2000-s3, random-2000-s3, prompt-only, base-reasoning, middle-2000-s1-reasoning).
