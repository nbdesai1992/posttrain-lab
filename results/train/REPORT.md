# Training runs

One LoRA adapter per run on `thinkingmachines/Inkling-2-Mini:peft:65536:free`, from a saved set in `results/data/sets.json`. A training example is the prompt followed by the answer alone, rendered for the fast setting, and only the answer and the two closing tokens carry loss. Across the runs here the rank is 16, the learning rate 0.00018 (falling linearly to zero), and the passes over a set 4. An update is about 3,500 training tokens. 1 of 18 planned runs are done. Generated 2026-11-27T04:51:13+00:00.

## Runs

| run | set | seed | items | training tokens | updates | wall time (min) | s per update | first loss | final loss | stalls and pauses | checkpoint |
|---|---|---|---|---|---|---|---|---|---|---|---|
| middle-2000-s1 | middle-2000 | 1 | 2,000 | 884,244 | 252 | 39.5 | 8.52 | 3.180 | 1.709 | 9 updates of 30 s or more, 9.8 min in all, longest 106 s | `tinker://537aecc0-ddff-5c3e-888d-d9a0ca10388d:train:0/sampler_weights/middle-2000-s1-20261127-044907` |

Training tokens count every pass. The final loss is the mean over the last tenth of a run's updates, per target token. Wall time includes setting up, the held-out loss, saving the adapter and scoring the dev items.

## Held-out generated items

A sanity check and never a headline number: scores come only from public benchmarks. The dev items are 150 from each cell, answered in the fast setting. By construction the base model gets every easy one right and none of the others.

| run | dev loss before, middle | after | dev middle | dev easy | dev hard | unfinished replies |
|---|---|---|---|---|---|---|
| middle-2000-s1 | 2.981 | 2.302 | 19.3% | 88.0% | 0.7% | 4 |

## Time, stalls and pauses

The 1 runs made 252 updates on 884,244 training tokens in 0.6 hours of updates and 0.66 hours of wall time, 8.52 s per update on average. 9 updates took 30 s or more, 9.8 minutes in all. Queue states the service reported: none. The checkpoints take 0.7 GB.

Adapters were evaluated by a second process while later runs trained. Updates took a median of 5.55 s (mean 9.93 s, 112 updates) while that process was working and 5.96 s (mean 7.4 s, 140 updates) while it was not.
