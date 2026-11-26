# Smoke test report

Model ID `thinkingmachines/Inkling-2-Mini:peft:65536:free`. Generated 2026-11-26T19:10:50+00:00 with tinker 0.40.1, tml-renderers 0.5.0 and tinker-cookbook 0.7.1. Every number below was measured by the smoke step unless it says otherwise. The full data is in `summary.json` next to this file, and the raw rows are in `runs/smoke/`.

## Sampling throughput

Short arithmetic prompts, temperature 0, closed loop at a fixed number of requests in flight.

| effort | in flight | requests | req/min | req/min to 95% done | prompt tok/min | completion tok/min | latency p50 (s) | p95 (s) | max (s) | tokens per reply | throttled | failed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.0 | 1 | 30 | 17.0 | 16.9 | 492 | 911 | 3.48 | 4.48 | 7.44 | 53.8 | 0 | 0 |
| 0.0 | 4 | 40 | 71.5 | 72.6 | 2,073 | 3,974 | 3.45 | 3.92 | 3.96 | 55.6 | 0 | 0 |
| 0.0 | 16 | 48 | 290.7 | 311.5 | 8,431 | 16,068 | 2.5 | 3.34 | 6.36 | 55.3 | 0 | 0 |
| 0.0 | 32 | 64 | 475.9 | 521.6 | 13,801 | 26,071 | 2.75 | 5.1 | 5.63 | 54.8 | 0 | 0 |
| 0.0 | 64 | 192 | 941.2 | 1037.7 | 27,295 | 52,708 | 2.54 | 5.48 | 6.54 | 56.0 | 0 | 0 |
| 0.9 | 1 | 30 | 17.9 | 18.0 | 556 | 1,303 | 3.44 | 3.73 | 5.07 | 72.6 | 0 | 0 |
| 0.9 | 4 | 40 | 71.6 | 68.1 | 2,221 | 5,167 | 3.3 | 3.94 | 8.06 | 72.1 | 0 | 0 |
| 0.9 | 16 | 48 | 271.0 | 255.2 | 8,400 | 19,572 | 3.27 | 3.91 | 5.56 | 72.2 | 0 | 0 |
| 0.9 | 32 | 64 | 566.8 | 668.8 | 17,570 | 40,711 | 2.45 | 3.08 | 4.25 | 71.8 | 0 | 0 |
| 0.9 | 64 | 192 | 381.4 | 1260.1 | 11,825 | 27,632 | 2.65 | 3.7 | 24.91 | 72.4 | 0 | 0 |

The rates divide by the wall time of the whole level, so a single slow request at the end pulls them down. The column to 95% done stops the clock when 95% of the level's requests had returned.

Longer replies, to separate the fixed time per request from the time per token:

| words asked for | requests | completion tokens mean | latency p50 (s) | latency max (s) | cut off |
|---|---|---|---|---|---|
| 150 | 8 | 188 | 6.01 | 6.46 | 0 |
| 500 | 8 | 475 | 8.37 | 11.29 | 0 |
| 1200 | 8 | 609 | 11.5 | 16.73 | 0 |

A straight-line fit over these 24 replies, sent 8 at a time, gives 2.82 s of fixed time per request plus 13.68 ms per completion token, which is about 73.1 tokens per second inside one request.

Reading long prompts: 24 prompts of about 2,379 tokens, one generated token each, 16 in flight, took 5.09 s. That is 672,476 prompt tokens per minute, with latency p50 2.49 s and 0 throttled requests.

## Throttling

Throttling events during the run, by kind.

| event | count |
|---|---|
| queue paused_rate_limit (training) | 12 |

The first few, verbatim, are under `service_events.examples` in `summary.json`.

From reading the SDK source (tinker 0.40.1), which is not the same as observing it: an HTTP 429 on submit is swallowed by the SDK and the request is sent again after one second, a queued request reports `paused_rate_limit` or `paused_capacity` with a reason through a queue-state observer and is logged only after a long grace period, and an HTTP 402 is retried silently for up to an hour behind the log line "The job is paused due to billing status". None of these raises an exception to the caller, which is why the client in this repository records them itself and stops on a billing pause.

Other errors and warnings seen during the run.

| event | count |
|---|---|
| http 400 BadRequestError at sample_submit | 1 |
| log ERROR: Request failed with non-retryable error: BadRequestError: Er | 1 |

None of the error responses carried a `Retry-After` header or a rate-limit header.

## What effort 0.0 does

The same prompts (arithmetic, yes/no judgments, short extractions) in three settings.

| setting | prompts | replies with thinking | completion tokens mean | p50 | p90 | max | thinking tokens mean | strict format | correct | cut off |
|---|---|---|---|---|---|---|---|---|---|---|
| effort_0.0 | 51 | 51 (100%) | 62.6 | 62.0 | 77.0 | 90.0 | 53.6 | 100% | 100% | 0 |
| effort_0.9 | 51 | 51 (100%) | 70.7 | 69.0 | 89.0 | 136.0 | 61.7 | 100% | 100% | 0 |
| effort_0.0_forced_answer | 51 | 0 (0%) | 4.0 | 3.0 | 6.0 | 7.0 | 0.0 | 100% | 96% | 0 |

Completion tokens include the framing tokens around the content: seven for a reply with thinking, four for a reply without, and two when the answer opening is forced. The forced-answer setting appends the two tokens that open a text answer to the prompt, so the model cannot reason first.

## Reading a verdict probability

On 12 yes/no judging prompts at effort 0.0, 12 replies opened with thinking. The model's own probability of opening with thinking averaged 1.0000.

| how the verdict was read | mean probability of the top verdict | accuracy on these prompts |
|---|---|---|
| first answer token after thinking, effort 0.0 | 1.0 | 100% |
| first answer token after thinking, effort 0.9 | 1.0 | 100% |
| answer position scored directly, no thinking | 0.99183 | 92% |

Scored directly, yes averaged 0.9845 when the label was yes and 0.1668 when it was no, and the four tokens yes, no, Yes and No held 100.0% of the probability. The direct score and a sampled reply with a forced answer opening agreed to within 1e-05. A direct score took 2.28 s on average.

## Training throughput

LoRA rank 16, learning rate 0.00054, one training client. Creating the client took 1.02 s.

| phase | batches | tokens per batch | seconds for each batch | mean seconds | examples/min | tokens/min |
|---|---|---|---|---|---|---|
| forward only | 2 | 450 | 26.95, 7 | 16.98 | 28.3 | 1,589 |
| separate calls | 2 | 450 | 16.91, 18.7 | 17.8 | 27.0 | 1,518 |
| 8 short examples | 3 | 450 | 9.49, 9.44, 9.74 | 9.56 | 50.2 | 2,823 |
| 32 short examples | 3 | 1,800 | 30.22, 8.52, 24.04 | 20.93 | 91.7 | 5,161 |
| 16 long examples | 3 | 12,087 | 8.62, 8.72, 113.37 | 43.57 | 22.0 | 16,645 |

Each batch here was awaited before the next was sent. The fastest batch of 450 tokens took 9.44 s and the fastest of 12,087 tokens took 8.62 s, while the slowest batch of all took 113.37 s. The time a batch takes follows the service's cycle and its load more than the batch's size. The first phase is a forward pass only, and the second awaits the gradient and optimizer calls one after the other.

Saving sampler weights took 11.93 s and creating a sampler from them took 0.48 s. The checkpoint is 738.8 MB. It was saved to `tinker://94200f79-ae45-5147-ae57-194cb55dd646:train:0/sampler_weights/smoke-20261126-185705` and expires at 2026-12-03T18:57:16.395900+00:00.

Samples from the adapter:

| expected | reply | thinking tokens | completion tokens | latency (s) |
|---|---|---|---|---|
| no | 'yes' | 0 | 5 | 3.38 |
| no | 'no' | 0 | 5 | 2.77 |
| yes | 'no' | 0 | 5 | 2.6 |

## Training with updates sent back to back

Each update was submitted before the previous one had returned, which the Tinker docs recommend so that a request is always waiting when the service starts a cycle. Seconds per update leave out the first update of each phase.

| phase | updates | tokens per update | seconds per update | examples/min | tokens/min |
|---|---|---|---|---|---|
| 32 short examples | 4 | 1,800 | 4.7 | 408.8 | 22,995 |
| 16 long examples | 4 | 12,087 | 5.93 | 162.0 | 122,366 |

Then 2 training clients were started on the last phase at once. Their first updates took 16.18 s, 142.89 s. The slower client reported the queue state `paused_rate_limit` 12 times while it waited, and by the time it began the other had finished, so the two never trained at the same time. Once it began it ran at the usual speed. This run therefore shows that a second training client is rate limited at the start. It does not show whether two can train side by side.

## Context limit

| probe | prompt tokens | max tokens | outcome | seconds | detail |
|---|---|---|---|---|---|
| over the limit | 66,032 | 8 | rejected | 0.88 | Error code: 400 - {'detail': "Prompt length plus max_tokens exceeds the model's context window: 66032 prompt tokens + 8 max_tokens > 65536."} |
| just under the limit | 65,471 | 8 | accepted | 5.01 | stop reason length, 8 tokens generated |
| prompt fits, prompt plus max_tokens does not | 65,471 | 256 | accepted | 3.86 | stop reason stop, 48 tokens generated |

A prompt longer than the window is refused at once. When the prompt fits, a `max_tokens` that would pass the window is still accepted, so the service appears to cap generation at the room that is left. The reply in the last row ended on its own before reaching that cap, so the cut-off itself was not observed. The second full-length prompt had 65,408 of its tokens served from the prefix cache.

## Spend

Checkpoint storage rate on this account: $0.0 per GB-month. Checkpoints stored before the run: 0. Checkpoints listed after it: 1, holding 739 MB. Estimated cost of all metered usage in the 24 hours up to the end of the run: $0.0. The service's usage records can lag by some minutes.
