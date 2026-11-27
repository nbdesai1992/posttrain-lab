# Training progress

Updated 2026-11-27 06:23 UTC. 3 of 18 runs are done, 2.2 hours after the first one started. Training now: random-2000-s2, since 06:23 UTC.

| run | updates | minutes | s per update | final loss | dev middle | dev easy | dev hard | stalls and pauses |
|---|---|---|---|---|---|---|---|---|
| middle-2000-s1 | 252 | 39 | 8.52 | 1.709 | 19.3% | 88.0% | 0.7% | 9 updates of 30 s or more, 9.8 min in all, longest 106 s |
| random-2000-s1 | 252 | 42 | 8.91 | 2.090 | 14.0% | 88.0% | 0.0% | 8 updates of 30 s or more, 10.7 min in all, longest 215 s |
| middle-2000-s2 | 252 | 42 | 8.83 | 1.678 | 14.0% | 88.0% | 0.0% | 5 updates of 30 s or more, 7.2 min in all, longest 165 s; paused_rate_limit reported 2 times |

Still to run: random-2000-s2, middle-2000-s3, random-2000-s3, shuffled-labels-2000, easy-2000, hard-2000, length-matched-random-2000, middle-250, random-250, middle-500, random-500, middle-1000, random-1000, middle-4000, random-4000.

Anything unusual: updates took a median of 5.77 s while adapters were being evaluated (412 updates) and 5.89 s otherwise (344 updates); the means were 9.42 s and 7.96 s.

Adapters are evaluated while later runs train, at 16 requests in flight. The evaluation first ran at 32 in flight, from 04:27 to 05:10 UTC on 2026-11-27, for the prompt-only state and the main adapter's fast pass. Over that stretch an update took 9.9 s on average, against 8.0 s for the 344 updates made without it up to 05:53 UTC. That is 24% slower, with an interval from 8% faster to 67% slower when groups of 16 updates are resampled. The median update did not change, and the service reported no rate-limit pause during updates. The evaluation was stopped from 05:10 to 05:56 UTC to measure training alone, and then started again at half the concurrency. It goes back to after training if 500 or more updates made beside it average more than a quarter above 8.0 s, or if the service reports a rate-limit pause during updates.

Every run trains in its own process. The third run started in the first process sat in the service's queue for seven minutes, with `paused_rate_limit` reported 13 times and no update made, as the third pilot run had. The process was stopped at 05:41 UTC and the run began two minutes into a fresh one. A run still waits about two minutes in the queue before its first update.

Evaluation, as of 2026-11-27 04:50 UTC: 3 states are complete (base, prompt-only, base-reasoning).
