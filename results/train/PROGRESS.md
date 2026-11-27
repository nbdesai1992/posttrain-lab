# Training progress

Updated 2026-11-27 04:51 UTC. 1 of 18 runs are done, 0.66 hours after the first one started. Training now: random-2000-s1, since 04:51 UTC.

| run | updates | minutes | s per update | final loss | dev middle | dev easy | dev hard | stalls and pauses |
|---|---|---|---|---|---|---|---|---|
| middle-2000-s1 | 252 | 39 | 8.52 | 1.709 | 19.3% | 88.0% | 0.7% | 9 updates of 30 s or more, 9.8 min in all, longest 106 s |

Still to run: random-2000-s1, middle-2000-s2, random-2000-s2, middle-2000-s3, random-2000-s3, shuffled-labels-2000, easy-2000, hard-2000, length-matched-random-2000, middle-250, random-250, middle-500, random-500, middle-1000, random-1000, middle-4000, random-4000.

Anything unusual: updates took a median of 5.55 s while adapters were being evaluated (112 updates) and 5.96 s otherwise (140 updates); the means were 9.93 s and 7.4 s.

Evaluation, as of 2026-11-27 04:50 UTC: 3 states are complete (base, prompt-only, base-reasoning).
