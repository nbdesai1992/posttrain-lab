# Training runs

One LoRA adapter per run on `thinkingmachines/Inkling-2-Mini:peft:65536:free`, from a saved set in `results/data/sets.json`. A training example is the prompt followed by the answer alone, rendered for the fast setting, and only the answer and the two closing tokens carry loss. Across the runs here the rank is 16, the learning rate 0.00018 (falling linearly to zero), and the passes over a set 4. An update is about 3,500 training tokens. 18 of 18 planned runs are done. Generated 2026-11-27T15:06:14+00:00.

## Runs

| run | set | seed | items | training tokens | updates | wall time (min) | s per update | first loss | final loss | stalls and pauses | checkpoint |
|---|---|---|---|---|---|---|---|---|---|---|---|
| middle-2000-s1 | middle-2000 | 1 | 2,000 | 884,244 | 252 | 39.5 | 8.52 | 3.180 | 1.709 | 9 updates of 30 s or more, 9.8 min in all, longest 106 s | `tinker://537aecc0-ddff-5c3e-888d-d9a0ca10388d:train:0/sampler_weights/middle-2000-s1-20261127-044907` |
| random-2000-s1 | random-2000 | 1 | 2,164 | 884,028 | 252 | 41.9 | 8.91 | 3.663 | 2.090 | 8 updates of 30 s or more, 10.7 min in all, longest 215 s | `tinker://cc1123cc-16c4-5c3e-a884-48f9bd38aa5f:train:0/sampler_weights/random-2000-s1-20261127-053112` |
| middle-2000-s2 | middle-2000-draw2 | 2 | 2,000 | 882,724 | 252 | 42.0 | 8.83 | 3.510 | 1.678 | 5 updates of 30 s or more, 7.2 min in all, longest 165 s; paused_rate_limit reported 2 times | `tinker://fad373a6-d32c-506f-b552-e69d707d0a98:train:0/sampler_weights/middle-2000-s2-20261127-062144` |
| random-2000-s2 | random-2000-draw2 | 2 | 2,211 | 882,700 | 252 | 69.6 | 15.69 | 2.614 | 1.997 | 20 updates of 30 s or more, 36.7 min in all, longest 436 s | `tinker://e515c5e2-eaf0-5775-86d2-873586f4de9e:train:0/sampler_weights/random-2000-s2-20261127-073133` |
| middle-2000-s3 | middle-2000-draw3 | 3 | 2,000 | 875,668 | 252 | 36.4 | 7.84 | 3.291 | 1.563 | 6 updates of 30 s or more, 8.2 min in all, longest 148 s | `tinker://b08574f6-fcb7-5893-be81-3d624f88b3ff:train:0/sampler_weights/middle-2000-s3-20261127-080756` |
| random-2000-s3 | random-2000-draw3 | 3 | 2,143 | 875,656 | 252 | 32.6 | 6.97 | 2.802 | 2.067 | 3 updates of 30 s or more, 3.6 min in all, longest 95 s | `tinker://0292900b-87db-54ef-a7ca-d55a791004f4:train:0/sampler_weights/random-2000-s3-20261127-084113` |
| shuffled-labels-2000 | shuffled-labels-2000 | 1 | 2,000 | 884,244 | 252 | 50.1 | 10.44 | 5.910 | 2.594 | 10 updates of 30 s or more, 15.4 min in all, longest 222 s | `tinker://70eb0c6c-cd36-5dbe-99ff-07c3360500ab:train:0/sampler_weights/shuffled-labels-2000-20261127-093130` |
| easy-2000 | easy-2000 | 1 | 2,865 | 884,324 | 252 | 37.7 | 8.19 | 0.408 | 0.114 | 6 updates of 30 s or more, 6.2 min in all, longest 90 s | `tinker://12fa294b-a279-53a0-8bb7-c591cb63506b:train:0/sampler_weights/easy-2000-20261127-100921` |
| hard-2000 | hard-2000 | 1 | 1,949 | 884,348 | 252 | 34.8 | 7.5 | 5.126 | 3.965 | 5 updates of 30 s or more, 4.4 min in all, longest 84 s | `tinker://db75f59d-c34f-5c26-9dd9-d23a876586e7:train:0/sampler_weights/hard-2000-20261127-104428` |
| length-matched-random-2000 | length-matched-random-2000 | 1 | 2,000 | 884,244 | 252 | 46.2 | 10.06 | 3.097 | 1.883 | 13 updates of 30 s or more, 15.0 min in all, longest 184 s | `tinker://5d89ebd1-7b1d-5db8-973b-e1e80460b283:train:0/sampler_weights/length-matched-random-2000-20261127-113016` |
| middle-250 | middle-250 | 1 | 250 | 116,992 | 32 | 7.2 | 7.27 | 3.097 | 2.061 | 1 updates of 30 s or more, 1.0 min in all, longest 57 s | `tinker://3149eac2-a4a1-5fbf-abae-b7ae7aba532a:train:0/sampler_weights/middle-250-20261127-113830` |
| random-250 | random-250 | 1 | 274 | 116,856 | 32 | 6.7 | 6.74 | 3.802 | 2.494 | none | `tinker://f43f5aab-a78a-5c82-9d82-ac916e750389:train:0/sampler_weights/random-250-20261127-114531` |
| middle-500 | middle-500 | 1 | 500 | 226,380 | 64 | 10.9 | 6.95 | 3.394 | 1.958 | 1 updates of 30 s or more, 0.9 min in all, longest 54 s | `tinker://fd7bab27-678d-5350-ac83-54bc84df5b39:train:0/sampler_weights/middle-500-20261127-115623` |
| random-500 | random-500 | 1 | 537 | 226,404 | 64 | 13.3 | 8.39 | 2.554 | 2.422 | 2 updates of 30 s or more, 1.1 min in all, longest 34 s | `tinker://b2e85d01-300e-5fc5-b7da-876c79db3a06:train:0/sampler_weights/random-500-20261127-120932` |
| middle-1000 | middle-1000 | 1 | 1,000 | 448,556 | 128 | 16.4 | 6.05 | 3.179 | 1.864 | 1 updates of 30 s or more, 0.6 min in all, longest 35 s | `tinker://42f2e86b-3caf-5201-a05e-f4cbb12e2e22:train:0/sampler_weights/middle-1000-20261127-122649` |
| random-1000 | random-1000 | 1 | 1,096 | 448,316 | 128 | 19.4 | 7.55 | 3.431 | 2.096 | 3 updates of 30 s or more, 3.2 min in all, longest 94 s | `tinker://555877ee-034a-5e87-91bb-9548ac4f2535:train:0/sampler_weights/random-1000-20261127-124627` |
| middle-4000 | middle-4000 | 1 | 4,000 | 1,740,548 | 496 | 62.2 | 7.11 | 3.425 | 1.664 | 10 updates of 30 s or more, 9.2 min in all, longest 124 s | `tinker://668f7c57-905e-5651-b859-036095f4cc08:train:0/sampler_weights/middle-4000-20261127-134903` |
| random-4000 | random-4000 | 1 | 4,301 | 1,740,360 | 496 | 75.1 | 8.57 | 3.195 | 2.015 | 13 updates of 30 s or more, 14.2 min in all, longest 111 s | `tinker://268cb477-1776-5757-972a-1f10bc34dea7:train:0/sampler_weights/random-4000-20261127-150417` |

Training tokens count every pass. The final loss is the mean over the last tenth of a run's updates, per target token. Wall time includes setting up, the held-out loss, saving the adapter and scoring the dev items.

## Held-out generated items

A sanity check and never a headline number: scores come only from public benchmarks. The dev items are 150 from each cell, answered in the fast setting. By construction the base model gets every easy one right and none of the others.

| run | dev loss before, middle | after | dev middle | dev easy | dev hard | unfinished replies |
|---|---|---|---|---|---|---|
| middle-2000-s1 | 2.981 | 2.302 | 19.3% | 88.0% | 0.7% | 4 |
| random-2000-s1 | 2.981 | 2.476 | 14.0% | 88.0% | 0.0% | 5 |
| middle-2000-s2 | 2.981 | 2.406 | 14.0% | 88.0% | 0.0% | 0 |
| random-2000-s2 | 2.981 | 2.461 | 12.7% | 90.0% | 0.0% | 0 |
| middle-2000-s3 | 2.981 | 2.405 | 20.7% | 89.3% | 0.7% | 3 |
| random-2000-s3 | 2.981 | 2.335 | 13.3% | 88.0% | 0.0% | 2 |
| shuffled-labels-2000 | 2.981 | 3.565 | 0.0% | 0.0% | 0.0% | 0 |
| easy-2000 | 2.981 | 3.598 | 10.7% | 90.0% | 0.0% | 4 |
| hard-2000 | 2.981 | 2.731 | 6.0% | 87.3% | 0.0% | 3 |
| length-matched-random-2000 | 2.981 | 2.421 | 12.0% | 89.3% | 0.7% | 5 |
| middle-250 | 2.981 | 2.258 | 7.3% | 90.0% | 0.0% | 3 |
| random-250 | 2.981 | 2.325 | 8.0% | 89.3% | 0.0% | 15 |
| middle-500 | 2.981 | 2.369 | 11.3% | 90.0% | 0.0% | 24 |
| random-500 | 2.981 | 2.381 | 8.7% | 86.7% | 0.0% | 19 |
| middle-1000 | 2.981 | 2.333 | 16.0% | 89.3% | 0.0% | 3 |
| random-1000 | 2.981 | 2.555 | 10.7% | 88.7% | 0.0% | 12 |
| middle-4000 | 2.981 | 2.259 | 19.3% | 88.7% | 0.7% | 6 |
| random-4000 | 2.981 | 2.385 | 17.3% | 89.3% | 0.0% | 0 |

## Time, stalls and pauses

The 18 runs made 3,960 updates on 13,886,592 training tokens in 9.55 hours of updates and 10.7 hours of wall time, 8.68 s per update on average. 116 updates took 30 s or more, 147.4 minutes in all. Queue states the service reported: paused_rate_limit (2). The checkpoints take 13.3 GB.

Adapters were evaluated by a second process while later runs trained. Updates took a median of 5.66 s (mean 8.79 s, 3,264 updates) while that process was working and 5.92 s (mean 8.14 s, 696 updates) while it was not.

Adapters are evaluated while later runs train, at 16 requests in flight. The evaluation first ran at 32 in flight, from 04:27 to 05:10 UTC on 2026-11-27, for the prompt-only state and the main adapter's fast pass. Over that stretch an update took 9.9 s on average, against 8.0 s for the 344 updates made without it up to 05:53 UTC. That is 24% slower, with an interval from 8% faster to 67% slower when groups of 16 updates are resampled. The median update did not change, and the service reported no rate-limit pause during updates. The evaluation was stopped from 05:10 to 05:56 UTC to measure training alone, and then started again at half the concurrency. It goes back to after training if 500 or more updates made beside it average more than a quarter above 8.0 s, or if the service reports a rate-limit pause during updates.

Every run trains in its own process. The third run started in the first process sat in the service's queue for seven minutes, with `paused_rate_limit` reported 13 times and no update made, as the third pilot run had. The process was stopped at 05:41 UTC and the run began two minutes into a fresh one. A run still waits about two minutes in the queue before its first update.
