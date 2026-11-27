# Training runs

One adapter per run. Dev accuracy is on held-out generated items in the fast setting and is a sanity check only.

| run | set | seed | rank | learning rate | items | updates | training tokens | s per update | loss first | loss last | dev loss, middle | dev middle | dev easy | dev hard | checkpoint |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pilot-r16-lr1.8e-4 | pilot-train | 20261126 | 16 | 0.00018 | 500 | 64 | 219,700 | 12.2 | 3.345 | 1.752 | 2.39967 | 0.1267 | 0.8933 | 0.0 | `tinker://de2c0255-a0fa-5492-80b9-461292ce49b4:train:0/sampler_weights/pilot-r16-lr1.8e-4-20261127-022755` |
| pilot-r16-lr5.4e-4 | pilot-train | 20261126 | 16 | 0.00054 | 500 | 64 | 219,700 | 9.09 | 3.345 | 1.136 | 3.32704 | 0.1267 | 0.8667 | 0.0067 | `tinker://de2c0255-a0fa-5492-80b9-461292ce49b4:train:1/sampler_weights/pilot-r16-lr5.4e-4-20261127-024254` |
| pilot-r16-lr1.6e-3 | pilot-train | 20261126 | 16 | 0.0016 | 500 | 64 | 219,700 | 11.87 | 3.345 | 0.827 | 4.22493 | 0.0333 | 0.7 | 0.0 | `tinker://11fe5372-045c-5cea-81fa-58192b799fa7:train:0/sampler_weights/pilot-r16-lr1.6e-3-20261127-031537` |
| pilot-r8-lr5.4e-4 | pilot-train | 20261126 | 8 | 0.00054 | 500 | 64 | 219,700 | 7.29 | 3.345 | 1.171 | 3.19308 | 0.1 | 0.8467 | 0.0067 | `tinker://f9bdc096-f9f4-5993-a87b-cd6952bf1e51:train:0/sampler_weights/pilot-r8-lr5.4e-4-20261127-032810` |
| pilot-r32-lr5.4e-4 | pilot-train | 20261126 | 32 | 0.00054 | 500 | 64 | 219,700 | 6.8 | 3.345 | 1.075 | 3.40162 | 0.12 | 0.8733 | 0.0 | `tinker://98e03b10-a9b4-56e5-8d3f-01437b4dd741:train:0/sampler_weights/pilot-r32-lr5.4e-4-20261127-034134` |
| pilot-r16-lr6e-5 | pilot-train | 20261126 | 16 | 6e-05 | 500 | 64 | 219,700 | 7.43 | 3.345 | 2.105 | 2.23638 | 0.0867 | 0.8933 | 0.0 | `tinker://aeff8b75-29c5-508a-8f9d-249cba827fb5:train:0/sampler_weights/pilot-r16-lr6e-5-20261127-035437` |
