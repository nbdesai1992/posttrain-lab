# Training runs

One LoRA adapter per run on `thinkingmachines/Inkling-2-Mini:peft:65536:free`, from a saved set in `results/data/sets.json`. A training example is the prompt followed by the answer alone, rendered for the fast setting, and only the answer and the two closing tokens carry loss. Across the runs here the rank is 16, the learning rate 0.00018 (falling linearly to zero), and the passes over a set 4. An update is about 3,500 training tokens. 2 of 18 planned runs are done. Generated 2026-11-27T05:33:17+00:00.

## Runs

| run | set | seed | items | training tokens | updates | wall time (min) | s per update | first loss | final loss | stalls and pauses | checkpoint |
|---|---|---|---|---|---|---|---|---|---|---|---|
| middle-2000-s1 | middle-2000 | 1 | 2,000 | 884,244 | 252 | 39.5 | 8.52 | 3.180 | 1.709 | 9 updates of 30 s or more, 9.8 min in all, longest 106 s | `tinker://537aecc0-ddff-5c3e-888d-d9a0ca10388d:train:0/sampler_weights/middle-2000-s1-20261127-044907` |
| random-2000-s1 | random-2000 | 1 | 2,164 | 884,028 | 252 | 41.9 | 8.91 | 3.663 | 2.090 | 8 updates of 30 s or more, 10.7 min in all, longest 215 s | `tinker://cc1123cc-16c4-5c3e-a884-48f9bd38aa5f:train:0/sampler_weights/random-2000-s1-20261127-053112` |

Training tokens count every pass. The final loss is the mean over the last tenth of a run's updates, per target token. Wall time includes setting up, the held-out loss, saving the adapter and scoring the dev items.

## Held-out generated items

A sanity check and never a headline number: scores come only from public benchmarks. The dev items are 150 from each cell, answered in the fast setting. By construction the base model gets every easy one right and none of the others.

| run | dev loss before, middle | after | dev middle | dev easy | dev hard | unfinished replies |
|---|---|---|---|---|---|---|
| middle-2000-s1 | 2.981 | 2.302 | 19.3% | 88.0% | 0.7% | 4 |
| random-2000-s1 | 2.981 | 2.476 | 14.0% | 88.0% | 0.0% | 5 |

## Time, stalls and pauses

The 2 runs made 504 updates on 1,768,272 training tokens in 1.22 hours of updates and 1.36 hours of wall time, 8.72 s per update on average. 17 updates took 30 s or more, 20.5 minutes in all. Queue states the service reported: none. The checkpoints take 1.5 GB.

Adapters were evaluated by a second process while later runs trained. Updates took a median of 5.6 s (mean 9.92 s, 224 updates) while that process was working and 5.89 s (mean 7.76 s, 280 updates) while it was not.
