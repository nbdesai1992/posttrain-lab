# Training runs

One LoRA adapter per run on `thinkingmachines/Inkling-2-Mini:peft:65536:free`, from a saved set in `results/data/sets.json`. A training example is the prompt followed by the answer alone, rendered for the fast setting, and only the answer and the two closing tokens carry loss. Across the runs here the rank is 16, the learning rate 0.00018 (falling linearly to zero), and the passes over a set 4. An update is about 3,500 training tokens. 3 of 18 planned runs are done. Generated 2026-11-27T06:23:33+00:00.

## Runs

| run | set | seed | items | training tokens | updates | wall time (min) | s per update | first loss | final loss | stalls and pauses | checkpoint |
|---|---|---|---|---|---|---|---|---|---|---|---|
| middle-2000-s1 | middle-2000 | 1 | 2,000 | 884,244 | 252 | 39.5 | 8.52 | 3.180 | 1.709 | 9 updates of 30 s or more, 9.8 min in all, longest 106 s | `tinker://537aecc0-ddff-5c3e-888d-d9a0ca10388d:train:0/sampler_weights/middle-2000-s1-20261127-044907` |
| random-2000-s1 | random-2000 | 1 | 2,164 | 884,028 | 252 | 41.9 | 8.91 | 3.663 | 2.090 | 8 updates of 30 s or more, 10.7 min in all, longest 215 s | `tinker://cc1123cc-16c4-5c3e-a884-48f9bd38aa5f:train:0/sampler_weights/random-2000-s1-20261127-053112` |
| middle-2000-s2 | middle-2000-draw2 | 2 | 2,000 | 882,724 | 252 | 42.0 | 8.83 | 3.510 | 1.678 | 5 updates of 30 s or more, 7.2 min in all, longest 165 s; paused_rate_limit reported 2 times | `tinker://fad373a6-d32c-506f-b552-e69d707d0a98:train:0/sampler_weights/middle-2000-s2-20261127-062144` |

Training tokens count every pass. The final loss is the mean over the last tenth of a run's updates, per target token. Wall time includes setting up, the held-out loss, saving the adapter and scoring the dev items.

## Held-out generated items

A sanity check and never a headline number: scores come only from public benchmarks. The dev items are 150 from each cell, answered in the fast setting. By construction the base model gets every easy one right and none of the others.

| run | dev loss before, middle | after | dev middle | dev easy | dev hard | unfinished replies |
|---|---|---|---|---|---|---|
| middle-2000-s1 | 2.981 | 2.302 | 19.3% | 88.0% | 0.7% | 4 |
| random-2000-s1 | 2.981 | 2.476 | 14.0% | 88.0% | 0.0% | 5 |
| middle-2000-s2 | 2.981 | 2.406 | 14.0% | 88.0% | 0.0% | 0 |

## Time, stalls and pauses

The 3 runs made 756 updates on 2,650,996 training tokens in 1.84 hours of updates and 2.06 hours of wall time, 8.76 s per update on average. 22 updates took 30 s or more, 27.7 minutes in all. Queue states the service reported: paused_rate_limit (2). The checkpoints take 2.2 GB.

Adapters were evaluated by a second process while later runs trained. Updates took a median of 5.77 s (mean 9.42 s, 412 updates) while that process was working and 5.89 s (mean 7.96 s, 344 updates) while it was not.

Adapters are evaluated while later runs train, at 16 requests in flight. The evaluation first ran at 32 in flight, from 04:27 to 05:10 UTC on 2026-11-27, for the prompt-only state and the main adapter's fast pass. Over that stretch an update took 9.9 s on average, against 8.0 s for the 344 updates made without it up to 05:53 UTC. That is 24% slower, with an interval from 8% faster to 67% slower when groups of 16 updates are resampled. The median update did not change, and the service reported no rate-limit pause during updates. The evaluation was stopped from 05:10 to 05:56 UTC to measure training alone, and then started again at half the concurrency. It goes back to after training if 500 or more updates made beside it average more than a quarter above 8.0 s, or if the service reports a rate-limit pause during updates.

Every run trains in its own process. The third run started in the first process sat in the service's queue for seven minutes, with `paused_rate_limit` reported 13 times and no update made, as the third pilot run had. The process was stopped at 05:41 UTC and the run began two minutes into a fresh one. A run still waits about two minutes in the queue before its first update.
