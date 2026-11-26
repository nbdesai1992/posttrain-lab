"""Smoke test: what the free tier can do, measured in a few minutes of calls.

It answers the questions the rest of the plan is sized on.

* How fast is sampling at a few concurrency levels, and what does throttling look like?
* How fast does the service read long prompts?
* What does effort 0.0 actually do to the reply, and are answers parseable?
* Can the probability of a yes or no verdict be read cleanly?
* How fast is LoRA training, and does a trained adapter load and sample?
* What happens at the 64K context limit?

Raw rows go to ``runs/smoke/``. ``results/smoke/summary.json`` and ``REPORT.md`` are
rebuilt from those rows at the end of every run.
"""

from __future__ import annotations

import asyncio
import json
import math
import random
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from ..client import AccountBlockedError, CallRecord, ModelClient
from ..render import DEFAULT_EFFORT, FAST_EFFORT
from ..stats import describe, fit_line
from ..store import append_jsonl, read_jsonl
from . import smoke_prompts as prompts
from .base import Step, StepContext, Unit

VERDICT_TOKENS = ["yes", "no", "Yes", "No"]
MODES = {
    # name: (effort, force the reply to open as an answer)
    "effort_0.0": (FAST_EFFORT, False),
    "effort_0.9": (DEFAULT_EFFORT, False),
    "effort_0.0_forced_answer": (FAST_EFFORT, True),
}


def _per_minute(count: float, seconds: float) -> float:
    return round(count / seconds * 60.0, 1) if seconds > 0 else 0.0


def _summarize_events(events: list) -> dict:
    """Count service events by what they say, and keep a few verbatim."""
    counts: Counter = Counter()
    examples: dict[str, dict] = {}
    for event in events:
        detail = event.detail
        if event.type == "http_error":
            label = f"http {detail.get('status')} {detail.get('exception')} at {detail.get('where')}"
        elif event.type == "queue_state":
            label = f"queue {detail.get('state')} ({detail.get('where')})"
        else:
            label = f"log {detail.get('level')}: {str(detail.get('message'))[:60]}"
        counts[label] += 1
        examples.setdefault(label, event.to_dict())
    return {"counts": dict(counts), "examples": list(examples.values())[:8]}


def _batch_seconds(steps: list[dict]) -> list[float]:
    """Seconds each batch took. An optimizer call awaited on its own is added to the
    gradient call before it."""
    seconds: list[float] = []
    for step in steps:
        if step["kind"] == "optim_step" and seconds:
            seconds[-1] = round(seconds[-1] + step["latency_s"], 2)
        else:
            seconds.append(step["latency_s"])
    return seconds


def _is_throttle(label: str) -> bool:
    return label.startswith(("http 429", "http 402", "queue ")) or "paused" in label


def _verdict_mass(top: list[tuple[int, float]], decode) -> dict:
    """Probability of yes and of no among the most likely tokens at one position."""
    mass = {"yes": 0.0, "no": 0.0}
    for token, logprob in top:
        word = decode([token]).strip().lower()
        if word in mass:
            mass[word] += math.exp(logprob)
    return mass


class Smoke(Step):
    name = "smoke"
    summary = "Measure free-tier throughput, effort 0.0 behaviour, verdict logprobs and a tiny training loop"
    default_config = "configs/smoke.toml"

    # ------------------------------------------------------------------ plan

    def plan(self, ctx: StepContext) -> list[Unit]:
        cfg = ctx.config
        seed = cfg.get("seed", 0)
        est = cfg.get("estimates", {})
        est_tokens = {
            "effort_0.0": est.get("completion_tokens_effort_0", 60),
            "effort_0.9": est.get("completion_tokens_effort_default", 90),
            "effort_0.0_forced_answer": est.get("completion_tokens_forced", 8),
        }
        renderer = ctx.renderer
        units: list[Unit] = [Unit("account", {"when": "before"}, label="before")]

        tp = cfg.get("throughput", {})
        sample_prompt = len(renderer.build_prompt(prompts.throughput_prompt(random.Random(seed)), DEFAULT_EFFORT))
        for effort in tp.get("efforts", []):
            mode = "effort_0.0" if effort == FAST_EFFORT else "effort_0.9"
            for concurrency, requests in zip(tp.get("concurrency", []), tp.get("requests", [])):
                units.append(
                    Unit(
                        "throughput",
                        {
                            "effort": effort,
                            "concurrency": concurrency,
                            "requests": requests,
                            "max_tokens": tp["max_tokens"],
                            "max_seconds": tp["max_seconds"],
                            "seed": seed,
                        },
                        calls=requests,
                        prompt_tokens=requests * sample_prompt,
                        completion_tokens=requests * est_tokens[mode],
                        max_completion_tokens=requests * tp["max_tokens"],
                        label=f"effort {effort} concurrency {concurrency}",
                    )
                )

        pf = cfg.get("prefill", {})
        if pf.get("requests", 0) > 0:
            item = prompts.long_judgment_item(random.Random(seed), pf["source_sentences"])
            long_prompt = len(renderer.build_prompt(item["messages"], FAST_EFFORT))
            units.append(
                Unit(
                    "prefill",
                    {**pf, "seed": seed},
                    calls=pf["requests"],
                    prompt_tokens=pf["requests"] * long_prompt,
                    completion_tokens=pf["requests"],
                    max_completion_tokens=pf["requests"],
                    label=f"{pf['requests']} prompts of about {long_prompt} tokens",
                )
            )

        lo = cfg.get("long_output", {})
        if lo.get("requests_per_length", 0) > 0:
            calls = lo["requests_per_length"] * len(lo["words"])
            expected = int(lo["requests_per_length"] * sum(lo["words"]) * 1.4)  # about 1.4 tokens per word
            units.append(
                Unit(
                    "long_output",
                    {**lo, "seed": seed},
                    calls=calls,
                    prompt_tokens=calls * 45,
                    completion_tokens=expected,
                    max_completion_tokens=calls * lo["max_tokens"],
                    label=f"replies of about {', '.join(str(w) for w in lo['words'])} words",
                )
            )

        bh = cfg.get("behaviour", {})
        if bh.get("per_kind", 0) > 0:
            items = prompts.behaviour_items(seed, bh["per_kind"])
            for mode in bh.get("modes", list(MODES)):
                effort, _ = MODES[mode]
                prompt_tokens = sum(len(renderer.build_prompt(i["messages"], effort)) for i in items)
                units.append(
                    Unit(
                        "behaviour",
                        {"mode": mode, "per_kind": bh["per_kind"], "max_tokens": bh["max_tokens"], "seed": seed},
                        calls=len(items),
                        prompt_tokens=prompt_tokens,
                        completion_tokens=len(items) * est_tokens[mode],
                        max_completion_tokens=len(items) * bh["max_tokens"],
                        label=mode,
                    )
                )

        vd = cfg.get("verdict", {})
        if vd.get("items", 0) > 0:
            items = prompts.judgment_items(seed + 1, vd["items"])
            prompt_tokens = sum(len(renderer.build_prompt(i["messages"], FAST_EFFORT)) for i in items)
            units.append(
                Unit(
                    "verdict",
                    {"items": vd["items"], "max_tokens": vd["max_tokens"], "seed": seed + 1},
                    calls=4 * len(items),  # free at 0.0, free at 0.9, forced sample, forced score
                    prompt_tokens=4 * prompt_tokens,
                    completion_tokens=len(items) * (est_tokens["effort_0.0"] + est_tokens["effort_0.9"] + 4),
                    max_completion_tokens=len(items) * (2 * vd["max_tokens"] + 9),
                    label=f"{vd['items']} yes/no judgments",
                )
            )

        tr = cfg.get("training", {})
        if tr.get("enabled", False):
            short, long = self._training_pools(ctx, seed)
            short_tokens = sum(len(t) for t, _ in short) / len(short)
            long_tokens = sum(len(t) for t, _ in long) / len(long) if long else 0
            tokens = 0
            calls = 3  # create the run, save the weights, create the sampler
            calls += tr["forward_calls"] + 2 * tr["separate_steps"] + tr["adapter_samples"]
            tokens += (tr["forward_calls"] + tr["separate_steps"]) * tr["small_batch"] * short_tokens
            for phase in tr["phases"]:
                per_example = long_tokens if phase["pool"] == "long" else short_tokens
                tokens += phase["steps"] * phase["batch"] * per_example
                calls += phase["steps"]
            units.append(
                Unit("training", {**tr, "seed": seed}, calls=calls, training_tokens=int(tokens), label=f"rank {tr['rank']}")
            )

        pt = cfg.get("pipelined_training", {})
        if pt.get("enabled", False):
            short, long = self._training_pools(ctx, seed)
            per_example = {
                "short": sum(len(t) for t, _ in short) / len(short),
                "long": sum(len(t) for t, _ in long) / len(long),
            }
            tokens = sum(pt["steps"] * ph["batch"] * per_example[ph["pool"]] for ph in pt["phases"])
            last = pt["phases"][-1]
            overlap = pt.get("parallel_clients", 0)
            tokens += overlap * pt["steps"] * last["batch"] * per_example[last["pool"]]
            units.append(
                Unit(
                    "pipelined_training",
                    {**pt, "seed": seed},
                    calls=1 + len(pt["phases"]) + (2 * overlap - 1 if overlap else 0),
                    training_tokens=int(tokens),
                    label=f"{pt['steps']} updates per phase",
                )
            )

        cx = cfg.get("context", {})
        if cx.get("enabled", False):
            near = cx["limit_tokens"] - cx["margin_tokens"]
            calls = 1 + int(cx.get("near_limit", False)) + int(cx.get("sum_over_limit", False))
            units.append(
                Unit(
                    "context",
                    dict(cx),
                    calls=calls,
                    prompt_tokens=cx["over_limit_tokens"] + (calls - 1) * near,
                    completion_tokens=8 * calls,
                    max_completion_tokens=8 + cx.get("sum_over_max_tokens", 0),
                    label="64K limit",
                )
            )

        ctx.scratch["units"] = units
        return units

    def _training_pools(self, ctx: StepContext, seed: int) -> tuple[list, list]:
        """Synthetic training examples: a judgment and its one-word answer, rendered
        at effort 0.0 with an answer-only target, the format the study will train on."""
        tr = ctx.config["training"]
        rng = random.Random(seed + 2)
        short_items = [prompts.judgment_item(rng) for _ in range(tr["short_examples"])]
        long_items = [prompts.long_judgment_item(rng, tr["long_sentences"]) for _ in range(tr["long_examples"])]

        def render(item: dict):
            return ctx.renderer.build_example(
                [*item["messages"], {"role": "assistant", "content": item["answer"]}], FAST_EFFORT
            )

        return [render(i) for i in short_items], [render(i) for i in long_items]

    # ------------------------------------------------------------------- run

    async def run_unit(self, unit: Unit, ctx: StepContext) -> dict:
        assert ctx.client is not None
        if "base_model_reported" not in ctx.scratch:
            # Create the sampler first, so no timing probe pays for it.
            ctx.scratch["base_model_reported"] = await ctx.client.warm_up()
        events_before = len(ctx.client.backend.events)
        started = time.time()
        row = await getattr(self, f"_run_{unit.kind}")(unit, ctx)
        new_events = ctx.client.backend.events[events_before:]
        for event in new_events:
            append_jsonl(ctx.runs_dir / "events.jsonl", {"unit": unit.kind, "label": unit.label, **event.to_dict()})
        row["service_events"] = _summarize_events(new_events)
        row["started_at"] = datetime.fromtimestamp(started, timezone.utc).isoformat(timespec="seconds")
        row["unit_seconds"] = round(time.time() - started, 1)
        return row

    async def _run_account(self, unit: Unit, ctx: StepContext) -> dict:
        backend = ctx.client.backend
        row = {"snapshot": await backend.account_snapshot()}
        if unit.params["when"] == "before":
            row["client_config"] = await backend.client_config()
            row["base_model_reported"] = ctx.scratch["base_model_reported"]
            rate = row["snapshot"].get("storage_rate_usd_per_gb_month")
            if rate:
                raise AccountBlockedError(
                    f"Checkpoint storage is billed on this account (${rate}/GB-month), so saving "
                    "an adapter would cost money. Switch the account to the free storage plan first."
                )
        return row

    async def _run_throughput(self, unit: Unit, ctx: StepContext) -> dict:
        p = unit.params
        client: ModelClient = ctx.client
        stop_fraction = ctx.config["throughput"].get("throttled_fraction_stop", 0.5)

        # Stop climbing if the level below was mostly throttled.
        levels = [u for u in ctx.scratch["units"] if u.kind == "throughput" and u.params["effort"] == p["effort"]]
        below = [u for u in levels if u.params["concurrency"] < p["concurrency"]]
        if below:
            previous = ctx.result(below[-1])
            if previous and (previous.get("throttle_dominated") or previous.get("skipped")):
                reason = f"throttling dominated at concurrency {below[-1].params['concurrency']}"
                ctx.out(f"    skipped: {reason}")
                return {"skipped": True, "reason": reason}

        rng = random.Random(p["seed"] * 1000 + p["concurrency"])
        conversations = [prompts.throughput_prompt(rng) for _ in range(p["requests"])]
        salt = f"throughput-{uuid.uuid4().hex}"  # timing probes must always reach the service
        client.set_concurrency(p["concurrency"])
        records: list[CallRecord] = []
        errors: Counter = Counter()
        next_index = 0
        started = time.monotonic()
        deadline = started + p["max_seconds"]

        async def worker() -> None:
            nonlocal next_index
            while next_index < len(conversations) and time.monotonic() < deadline:
                index = next_index
                next_index += 1
                try:
                    result = await client.sample(
                        conversations[index],
                        effort=p["effort"],
                        max_tokens=p["max_tokens"],
                        seed=index,
                        salt=salt,
                        tag=f"throughput-e{p['effort']}-c{p['concurrency']}",
                    )
                    records.append(result.record)
                except AccountBlockedError:
                    raise
                except Exception as exc:
                    errors[type(exc).__name__] += 1
                    ctx.out(f"    request failed: {type(exc).__name__}: {str(exc)[:200]}")

        await asyncio.gather(*(worker() for _ in range(p["concurrency"])))
        wall = time.monotonic() - started
        throttled = sum(1 for r in records if r.throttle_events > 0)
        done = len(records)
        row = {
            "effort": p["effort"],
            "concurrency": p["concurrency"],
            "requests_planned": p["requests"],
            "requests_completed": done,
            "requests_failed": sum(errors.values()),
            "stopped_by_time_budget": next_index < len(conversations),
            "wall_s": round(wall, 2),
            "requests_per_min": _per_minute(done, wall),
            "prompt_tokens_per_min": _per_minute(sum(r.prompt_tokens for r in records), wall),
            "completion_tokens_per_min": _per_minute(sum(r.completion_tokens for r in records), wall),
            "prompt_tokens": sum(r.prompt_tokens for r in records),
            "completion_tokens": sum(r.completion_tokens for r in records),
            "thinking_tokens": sum(r.thinking_tokens for r in records),
            "latency_s": describe(r.latency_s for r in records),
            "completion_tokens_per_request": describe((r.completion_tokens for r in records), 1),
            "tokens_per_second_per_request": describe(
                r.completion_tokens / r.latency_s for r in records if r.latency_s > 0
            ),
            "truncated": sum(1 for r in records if r.stop_reason == "length"),
            "retries": sum(r.attempts - 1 for r in records),
            "throttled_requests": throttled,
            "throttle_events": sum(r.throttle_events for r in records),
            "errors": dict(errors),
        }
        row["throttle_dominated"] = done == 0 or throttled / done >= stop_fraction or sum(errors.values()) > done
        ctx.out(
            f"    {done}/{p['requests']} requests in {wall:.0f} s: {row['requests_per_min']} req/min, "
            f"{row['completion_tokens_per_min']:,.0f} completion tok/min, "
            f"latency p50 {row['latency_s'].get('p50')} s p95 {row['latency_s'].get('p95')} s, "
            f"{throttled} throttled, {sum(errors.values())} failed"
        )
        return row

    async def _run_prefill(self, unit: Unit, ctx: StepContext) -> dict:
        """Long prompts, one generated token each: how fast the service reads input."""
        p = unit.params
        client: ModelClient = ctx.client
        rng = random.Random(p["seed"] + 3)
        items = [prompts.long_judgment_item(rng, p["source_sentences"]) for _ in range(p["requests"])]
        salt = f"prefill-{uuid.uuid4().hex}"
        client.set_concurrency(p["concurrency"])
        started = time.monotonic()
        scores = await asyncio.gather(
            *(client.score_choices(i["messages"], VERDICT_TOKENS, salt=salt, tag="prefill") for i in items)
        )
        wall = time.monotonic() - started
        records = [s.record for s in scores]
        correct = 0
        for item, score in zip(items, scores):
            p_yes = score.probs["yes"] + score.probs["Yes"]
            correct += int(("yes" if p_yes >= 0.5 else "no") == item["answer"])
        row = {
            "concurrency": p["concurrency"],
            "requests_completed": len(records),
            "wall_s": round(wall, 2),
            "requests_per_min": _per_minute(len(records), wall),
            "prompt_tokens": sum(r.prompt_tokens for r in records),
            "prompt_tokens_per_request": describe((r.prompt_tokens for r in records), 0),
            "prompt_tokens_per_min": _per_minute(sum(r.prompt_tokens for r in records), wall),
            "cached_prompt_tokens": sum(r.cached_prompt_tokens for r in records),
            "latency_s": describe(r.latency_s for r in records),
            "throttled_requests": sum(1 for r in records if r.throttle_events > 0),
            "forced_verdict_accuracy": round(correct / len(items), 3),
        }
        ctx.out(
            f"    {len(records)} prompts in {wall:.0f} s: {row['prompt_tokens_per_min']:,.0f} prompt tok/min, "
            f"latency p50 {row['latency_s'].get('p50')} s"
        )
        return row

    async def _run_long_output(self, unit: Unit, ctx: StepContext) -> dict:
        """Replies of a few hundred to a couple of thousand tokens, to separate the
        fixed time per request from the time per generated token."""
        p = unit.params
        client: ModelClient = ctx.client
        client.set_concurrency(p["concurrency"])
        salt = f"long-output-{uuid.uuid4().hex}"
        jobs = [(words, index) for words in p["words"] for index in range(p["requests_per_length"])]
        results = await asyncio.gather(
            *(
                client.sample(
                    prompts.essay_prompt(index, words),
                    effort=FAST_EFFORT,
                    max_tokens=p["max_tokens"],
                    force_answer=True,
                    seed=index,
                    salt=salt,
                    tag=f"long-output-{words}",
                )
                for words, index in jobs
            )
        )
        by_length = {}
        for words in p["words"]:
            records = [r.record for (w, _), r in zip(jobs, results) if w == words]
            by_length[str(words)] = {
                "requests": len(records),
                "completion_tokens": describe((r.completion_tokens for r in records), 0),
                "latency_s": describe(r.latency_s for r in records),
                "cut_off_by_max_tokens": sum(1 for r in records if r.stop_reason == "length"),
            }
        records = [r.record for r in results]
        fixed, per_token = fit_line([r.completion_tokens for r in records], [r.latency_s for r in records])
        row = {
            "concurrency": p["concurrency"],
            "requests": len(records),
            "by_requested_words": by_length,
            "fit": {
                "fixed_seconds_per_request": round(fixed, 2),
                "milliseconds_per_completion_token": round(per_token * 1000, 2),
                "completion_tokens_per_second_in_one_request": round(1 / per_token, 1) if per_token > 0 else None,
            },
            "throttled_requests": sum(1 for r in records if r.throttle_events > 0),
        }
        ctx.out(
            f"    {len(records)} replies: {row['fit']['fixed_seconds_per_request']} s fixed plus "
            f"{row['fit']['milliseconds_per_completion_token']} ms per token "
            f"({row['fit']['completion_tokens_per_second_in_one_request']} tok/s in one request)"
        )
        return row

    async def _run_behaviour(self, unit: Unit, ctx: StepContext) -> dict:
        p = unit.params
        client: ModelClient = ctx.client
        effort, forced = MODES[p["mode"]]
        items = prompts.behaviour_items(p["seed"], p["per_kind"])
        client.set_concurrency(ctx.config["behaviour"].get("concurrency", 8))
        results = await asyncio.gather(
            *(
                client.sample(
                    item["messages"],
                    effort=effort,
                    max_tokens=p["max_tokens"],
                    force_answer=forced,
                    tag=f"behaviour-{p['mode']}",
                )
                for item in items
            )
        )
        rows = []
        for item, result in zip(items, results):
            check = prompts.check_answer(item, result.text)
            rows.append(
                {
                    "id": item["id"],
                    "kind": item["kind"],
                    "expected": item["answer"],
                    "text": result.text[:300],
                    "first_kind": result.parsed.first_kind,
                    "completion_tokens": result.record.completion_tokens,
                    "thinking_tokens": result.record.thinking_tokens,
                    "answer_tokens": result.record.answer_tokens,
                    "ended": result.parsed.ended,
                    "well_formed": result.parsed.well_formed,
                    **check,
                }
            )

        def summarize(subset: list[dict]) -> dict:
            n = len(subset)
            return {
                "n": n,
                "with_thinking": sum(1 for r in subset if r["thinking_tokens"] > 0),
                "share_with_thinking": round(sum(1 for r in subset if r["thinking_tokens"] > 0) / n, 3),
                "completion_tokens": describe((r["completion_tokens"] for r in subset), 1),
                "thinking_tokens": describe((r["thinking_tokens"] for r in subset), 1),
                "answer_tokens": describe((r["answer_tokens"] for r in subset), 1),
                "parseable": round(sum(r["parseable"] for r in subset) / n, 3),
                "correct": round(sum(r["correct"] for r in subset) / n, 3),
                "cut_off_by_max_tokens": sum(1 for r in subset if not r["ended"]),
                "malformed": sum(1 for r in subset if not r["well_formed"]),
            }

        row = {"mode": p["mode"], "effort": effort, "forced_answer": forced, "overall": summarize(rows)}
        row["by_kind"] = {
            kind: summarize([r for r in rows if r["kind"] == kind]) for kind in sorted({r["kind"] for r in rows})
        }
        row["items"] = rows
        o = row["overall"]
        ctx.out(
            f"    {p['mode']}: {o['with_thinking']}/{o['n']} replies contain thinking, "
            f"completion tokens mean {o['completion_tokens']['mean']} p90 {o['completion_tokens']['p90']}, "
            f"parseable {o['parseable']:.0%}, correct {o['correct']:.0%}"
        )
        return row

    async def _run_verdict(self, unit: Unit, ctx: StepContext) -> dict:
        """Read P(yes) and P(no) three ways and show which of them is usable."""
        p = unit.params
        client: ModelClient = ctx.client
        decode = ctx.renderer.decode
        items = prompts.judgment_items(p["seed"], p["items"])
        client.set_concurrency(ctx.config["verdict"].get("concurrency", 6))

        async def one(item: dict) -> dict:
            messages = item["messages"]
            free_fast, free_slow, forced_sample, forced_score = await asyncio.gather(
                client.sample(messages, effort=FAST_EFFORT, max_tokens=p["max_tokens"], top_logprobs=5, tag="verdict-free-0.0"),
                client.sample(messages, effort=DEFAULT_EFFORT, max_tokens=p["max_tokens"], top_logprobs=5, tag="verdict-free-0.9"),
                client.sample(messages, effort=FAST_EFFORT, max_tokens=8, top_logprobs=5, force_answer=True, tag="verdict-forced-sample"),
                client.score_choices(messages, VERDICT_TOKENS, effort=FAST_EFFORT, tag="verdict-forced-score"),
            )
            row = {"id": item["id"], "label": item["answer"]}
            for name, result in (("free_0.0", free_fast), ("free_0.9", free_slow), ("forced_sample", forced_sample)):
                index = result.parsed.first_text_index
                entry = {
                    "first_kind": result.parsed.first_kind,
                    "thinking_tokens": result.record.thinking_tokens,
                    "first_answer_token_index": index,
                    "text": result.text[:80],
                    # What a naive reader gets from the first sampled position.
                    "top_at_first_position": [[decode([t]), round(lp, 3)] for t, lp in (result.top_logprobs or [[]])[0]],
                }
                if index is not None and result.top_logprobs and index < len(result.top_logprobs):
                    mass = _verdict_mass(result.top_logprobs[index], decode)
                    entry["p_yes_at_answer"] = round(mass["yes"], 5)
                    entry["p_no_at_answer"] = round(mass["no"], 5)
                    entry["top_at_answer"] = [[decode([t]), round(lp, 3)] for t, lp in result.top_logprobs[index]]
                row[name] = entry
            p_yes = forced_score.probs["yes"] + forced_score.probs["Yes"]
            row["forced_score"] = {
                "p_yes": round(p_yes, 5),
                "p_no": round(1 - p_yes, 5),
                "by_token": {k: round(v, 5) for k, v in forced_score.probs.items()},
                "mass_on_candidates": round(forced_score.mass, 5),
                "p_reply_opens_with_thinking": round(forced_score.p_thinking_first, 6),
                "p_reply_opens_with_answer": round(forced_score.p_text_first, 6),
                "latency_s": round(forced_score.record.latency_s, 2),
            }
            return row

        rows = await asyncio.gather(*(one(item) for item in items))

        def accuracy(pick) -> float:
            return round(sum(1 for r in rows if pick(r) == r["label"]) / len(rows), 3)

        def saturation(name: str) -> float:
            values = [max(r[name].get("p_yes_at_answer", 0), r[name].get("p_no_at_answer", 0)) for r in rows]
            return round(sum(values) / len(values), 5)

        # The same position read two ways should give the same number.
        consistency = [
            abs(r["forced_sample"]["p_yes_at_answer"] - r["forced_score"]["p_yes"] * r["forced_score"]["mass_on_candidates"])
            for r in rows
            if "p_yes_at_answer" in r["forced_sample"]
        ]
        summary = {
            "n": len(rows),
            "free_0.0_opens_with_thinking": sum(1 for r in rows if r["free_0.0"]["first_kind"] == "thinking"),
            "free_0.9_opens_with_thinking": sum(1 for r in rows if r["free_0.9"]["first_kind"] == "thinking"),
            "mean_p_reply_opens_with_thinking_at_0.0": round(
                sum(r["forced_score"]["p_reply_opens_with_thinking"] for r in rows) / len(rows), 6
            ),
            "mean_top_verdict_probability_after_thinking_0.0": saturation("free_0.0"),
            "mean_top_verdict_probability_after_thinking_0.9": saturation("free_0.9"),
            "mean_top_verdict_probability_forced": round(
                sum(max(r["forced_score"]["p_yes"], r["forced_score"]["p_no"]) for r in rows) / len(rows), 5
            ),
            "mean_forced_p_yes_when_label_yes": round(
                sum(r["forced_score"]["p_yes"] for r in rows if r["label"] == "yes") / max(1, sum(r["label"] == "yes" for r in rows)), 4
            ),
            "mean_forced_p_yes_when_label_no": round(
                sum(r["forced_score"]["p_yes"] for r in rows if r["label"] == "no") / max(1, sum(r["label"] == "no" for r in rows)), 4
            ),
            "mean_mass_on_yes_no_tokens_forced": round(
                sum(r["forced_score"]["mass_on_candidates"] for r in rows) / len(rows), 4
            ),
            "accuracy_free_0.0": accuracy(lambda r: r["free_0.0"]["text"].strip().lower().rstrip(".")),
            "accuracy_free_0.9": accuracy(lambda r: r["free_0.9"]["text"].strip().lower().rstrip(".")),
            "accuracy_forced_score": accuracy(lambda r: "yes" if r["forced_score"]["p_yes"] >= 0.5 else "no"),
            "max_gap_between_forced_sample_and_forced_score": round(max(consistency), 5) if consistency else None,
            "mean_forced_score_latency_s": round(sum(r["forced_score"]["latency_s"] for r in rows) / len(rows), 2),
        }
        ctx.out(
            f"    {summary['free_0.0_opens_with_thinking']}/{len(rows)} replies at effort 0.0 open with thinking; "
            f"verdict probability after thinking averages {summary['mean_top_verdict_probability_after_thinking_0.0']}, "
            f"forced read averages {summary['mean_top_verdict_probability_forced']}"
        )
        return {"summary": summary, "items": rows}

    async def _run_training(self, unit: Unit, ctx: StepContext) -> dict:
        p = unit.params
        client: ModelClient = ctx.client
        backend = client.backend
        short, long = self._training_pools(ctx, p["seed"])
        pools = {"short": short, "long": long}
        cursor = {"short": 0, "long": 0}
        lr = p["learning_rate"]
        client.set_concurrency(2)

        def batch(pool: str, size: int) -> list:
            examples = [pools[pool][(cursor[pool] + i) % len(pools[pool])] for i in range(size)]
            cursor[pool] += size
            return examples

        def loss_per_token(result: dict, examples: list) -> float | None:
            total = result.get("metrics", {}).get("loss:sum")
            weight = sum(sum(w) for _, w in examples)
            return round(total / weight, 4) if total is not None and weight else None

        steps: list[dict] = []

        def log(phase: str, record: CallRecord, examples: list | None, result: dict | None = None, **extra) -> None:
            steps.append(
                {
                    "phase": phase,
                    "kind": record.kind,
                    "examples": record.examples,
                    "tokens": record.prompt_tokens,
                    "latency_s": round(record.latency_s, 2),
                    "throttle_events": record.throttle_events,
                    "loss_per_target_token": loss_per_token(result, examples) if result and examples else None,
                    **extra,
                }
            )

        run = await client.create_training_run(rank=p["rank"], seed=p["seed"], tag="smoke")
        create_s = run.created.latency_s
        ctx.out(f"    training client created in {create_s:.1f} s")

        for _ in range(p["forward_calls"]):
            examples = batch("short", p["small_batch"])
            result, record = await run.forward(examples, tag="forward")
            log("forward only", record, examples, result)
        for _ in range(p["separate_steps"]):
            examples = batch("short", p["small_batch"])
            result, record = await run.forward_backward(examples, tag="separate")
            log("separate calls", record, examples, result)
            _, record = await run.optim_step(lr, tag="separate")
            log("separate calls", record, None)
        for phase in p["phases"]:
            for _ in range(phase["steps"]):
                examples = batch(phase["pool"], phase["batch"])
                result, record = await run.train_step(examples, lr, tag=phase["name"])
                log(phase["name"], record, examples, result, forward_backward_s=round(result.get("forward_backward_s", 0.0), 2))
            done = [s for s in steps if s["phase"] == phase["name"]]
            ctx.out(
                f"    {phase['name']}: {len(done)} steps, mean {sum(s['latency_s'] for s in done) / len(done):.1f} s, "
                f"{_per_minute(sum(s['tokens'] for s in done), sum(s['latency_s'] for s in done)):,.0f} tok/min"
            )

        name = f"smoke-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
        path, save_record = await run.save_sampler_weights(name, p["checkpoint_ttl_seconds"], tag="save")
        ctx.out(f"    saved sampler weights in {save_record.latency_s:.1f} s: {path}")
        info = await backend.checkpoint_info(path)
        load_record = await client.load_adapter(path, tag="load")
        ctx.out(f"    sampler created from the saved weights in {load_record.latency_s:.1f} s")

        rng = random.Random(p["seed"] + 4)
        samples = []
        for index in range(p["adapter_samples"]):
            item = prompts.judgment_item(rng)
            result = await client.sample(
                item["messages"], effort=FAST_EFFORT, max_tokens=256, model_path=path, salt=name, tag="adapter-sample"
            )
            samples.append(
                {
                    "expected": item["answer"],
                    "text": result.text[:120],
                    "first_kind": result.parsed.first_kind,
                    "completion_tokens": result.record.completion_tokens,
                    "thinking_tokens": result.record.thinking_tokens,
                    "latency_s": round(result.record.latency_s, 2),
                }
            )
        ctx.out(f"    adapter sample: {samples[0]['text']!r} ({samples[0]['thinking_tokens']} thinking tokens)")

        phases = {}
        for phase_name in dict.fromkeys(s["phase"] for s in steps):
            subset = [s for s in steps if s["phase"] == phase_name]
            gradient = [s for s in subset if s["kind"] != "optim_step"]
            seconds = sum(s["latency_s"] for s in subset)
            phases[phase_name] = {
                "calls": len(subset),
                "batches": len(gradient),
                "examples": sum(s["examples"] or 0 for s in gradient),
                "tokens": sum(s["tokens"] for s in gradient),
                "tokens_per_batch": round(sum(s["tokens"] for s in gradient) / len(gradient)),
                "seconds": round(seconds, 2),
                "mean_seconds_per_batch": round(seconds / len(gradient), 2),
                "seconds_per_batch": _batch_seconds(subset),
                "examples_per_min": _per_minute(sum(s["examples"] or 0 for s in gradient), seconds),
                "tokens_per_min": _per_minute(sum(s["tokens"] for s in gradient), seconds),
                "loss_per_target_token": [s["loss_per_target_token"] for s in gradient],
            }
        return {
            "rank": p["rank"],
            "learning_rate": lr,
            "create_training_client_s": round(create_s, 2),
            "phases": phases,
            "save_sampler_weights_s": round(save_record.latency_s, 2),
            "create_sampler_from_weights_s": round(load_record.latency_s, 2),
            "tinker_path": path,
            "checkpoint": info,
            "checkpoint_ttl_seconds": p["checkpoint_ttl_seconds"],
            "adapter_samples": samples,
            "steps": steps,
        }

    async def _run_pipelined_training(self, unit: Unit, ctx: StepContext) -> dict:
        """Updates sent back to back, which is how a real run should train, and then
        two training clients at once to see whether runs can overlap."""
        p = unit.params
        client: ModelClient = ctx.client
        short, long = self._training_pools(ctx, p["seed"])
        pools = {"short": short, "long": long}
        lr = p["learning_rate"]
        client.set_concurrency(4)

        def batches(phase: dict) -> list[list]:
            pool = pools[phase["pool"]]
            return [[pool[(step * phase["batch"] + i) % len(pool)] for i in range(phase["batch"])] for step in range(p["steps"])]

        def describe_updates(results: list[dict], record: CallRecord) -> dict:
            seconds = [round(r["seconds"], 2) for r in results]
            steady = seconds[1:] or seconds  # the first update also waits for a cycle to open
            tokens_per_update = record.prompt_tokens / len(results)
            mean_steady = sum(steady) / len(steady)
            return {
                "updates": len(results),
                "tokens_per_update": round(tokens_per_update),
                "seconds_per_update": seconds,
                "steady_seconds_per_update": round(mean_steady, 2),
                "tokens_per_min": _per_minute(tokens_per_update, mean_steady),
                "examples_per_min": _per_minute(record.examples / len(results), mean_steady),
                "total_seconds": round(record.latency_s, 2),
                "throttle_events": record.throttle_events,
            }

        run = await client.create_training_run(rank=p["rank"], seed=p["seed"], tag="pipelined")
        row: dict = {"rank": p["rank"], "create_training_client_s": round(run.created.latency_s, 2), "phases": {}}
        for phase in p["phases"]:
            results, record = await run.train_pipelined(batches(phase), lr, tag=phase["name"])
            row["phases"][phase["name"]] = describe_updates(results, record)
            d = row["phases"][phase["name"]]
            ctx.out(
                f"    {phase['name']}: {d['tokens_per_update']:,} tokens per update, "
                f"{d['steady_seconds_per_update']} s per update, {d['tokens_per_min']:,.0f} tok/min"
            )

        overlap = p.get("parallel_clients", 0)
        if overlap >= 2:
            phase = p["phases"][-1]
            try:
                others = await asyncio.wait_for(
                    asyncio.gather(
                        *(client.create_training_run(rank=p["rank"], seed=p["seed"] + n, tag="pipelined-extra") for n in range(1, overlap))
                    ),
                    timeout=p.get("parallel_timeout_s", 180),
                )
                outcomes = await asyncio.wait_for(
                    asyncio.gather(*(r.train_pipelined(batches(phase), lr, tag=f"overlap-{n}") for n, r in enumerate([run, *others]))),
                    timeout=p.get("parallel_timeout_s", 180) * 2,
                )
                per_client = [describe_updates(results, record) for results, record in outcomes]
                alone = row["phases"][phase["name"]]["steady_seconds_per_update"]
                together = sum(d["steady_seconds_per_update"] for d in per_client) / len(per_client)
                row["overlap"] = {
                    "clients": overlap,
                    "phase": phase["name"],
                    "create_extra_clients_s": [round(r.created.latency_s, 2) for r in others],
                    "per_client": per_client,
                    "steady_seconds_per_update_alone": alone,
                    "steady_seconds_per_update_together": round(together, 2),
                    "combined_tokens_per_min": round(sum(d["tokens_per_min"] for d in per_client), 1),
                }
                ctx.out(
                    f"    {overlap} clients at once: {together:.2f} s per update each (alone: {alone} s), "
                    f"{row['overlap']['combined_tokens_per_min']:,.0f} tok/min combined"
                )
            except AccountBlockedError:
                raise
            except Exception as exc:
                row["overlap"] = {"clients": overlap, "failed": f"{type(exc).__name__}: {str(exc)[:300]}"}
                ctx.out(f"    {overlap} clients at once did not complete: {row['overlap']['failed']}")
        return row

    async def _run_context(self, unit: Unit, ctx: StepContext) -> dict:
        p = unit.params
        client: ModelClient = ctx.client
        renderer = ctx.renderer
        client.set_concurrency(1)
        base = len(renderer.build_prompt(prompts.filler_message(0), FAST_EFFORT))
        per_sentence = len(renderer.build_prompt(prompts.filler_message(100), FAST_EFFORT)) - base
        per_sentence /= 100

        def prompt_near(target_tokens: int, over: bool) -> list[int]:
            """A prompt just over the target length, or the longest one that fits under it."""
            sentences = int((target_tokens - base) / per_sentence) + (2 if over else 0)
            tokens = renderer.build_prompt(prompts.filler_message(sentences), FAST_EFFORT)
            while not over and len(tokens) > target_tokens and sentences > 0:
                sentences -= 1
                tokens = renderer.build_prompt(prompts.filler_message(sentences), FAST_EFFORT)
            return tokens

        async def probe(name: str, target_tokens: int, max_tokens: int, over: bool = False) -> dict:
            tokens = prompt_near(target_tokens, over)
            started = time.monotonic()
            row = {"probe": name, "prompt_tokens": len(tokens), "max_tokens": max_tokens}
            try:
                result = await client.sample_tokens(
                    tokens, effort=FAST_EFFORT, max_tokens=max_tokens, salt=f"context-{uuid.uuid4().hex}", tag=f"context-{name}"
                )
                row.update(
                    outcome="accepted",
                    completion_tokens=result.record.completion_tokens,
                    stop_reason=result.stop_reason,
                    cached_prompt_tokens=result.record.cached_prompt_tokens,
                    text=result.text[:80],
                )
            except AccountBlockedError:
                raise
            except Exception as exc:
                row.update(outcome="rejected", error_class=type(exc).__name__, error_message=str(exc)[:400])
            row["seconds"] = round(time.monotonic() - started, 2)
            ctx.out(f"    {name}: {row['prompt_tokens']:,} prompt tokens + {max_tokens} max tokens -> {row['outcome']} in {row['seconds']} s")
            return row

        limit = p["limit_tokens"]
        probes = [await probe("over the limit", p["over_limit_tokens"], 8, over=True)]
        # The two probes below each send a prompt of about 64K tokens. Leave them out
        # if the service was already throttling long prompts.
        prefill = [ctx.result(u) for u in ctx.scratch["units"] if u.kind == "prefill"]
        if any(row and row.get("throttled_requests") for row in prefill):
            ctx.out("    long prompts were throttled earlier, so the two full-length probes are skipped")
            return {"limit_tokens": limit, "probes": probes, "full_length_probes_skipped": True}
        if p.get("near_limit"):
            probes.append(await probe("just under the limit", limit - p["margin_tokens"], 8))
        if p.get("sum_over_limit"):
            probes.append(await probe("prompt fits, prompt plus max_tokens does not", limit - p["margin_tokens"], p["sum_over_max_tokens"]))
        return {"limit_tokens": limit, "probes": probes}

    # ---------------------------------------------------------------- report

    async def finish(self, ctx: StepContext, rows: list[dict]) -> None:
        # Taken at the end of every run, so it always reflects the latest calls.
        account_after = await ctx.client.backend.account_snapshot()
        calls = list(read_jsonl(ctx.runs_dir / "calls.jsonl"))
        summary = build_summary(ctx.config, rows, account_after, calls)
        ctx.results_dir.mkdir(parents=True, exist_ok=True)
        (ctx.results_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        (ctx.results_dir / "REPORT.md").write_text(render_report(summary), encoding="utf-8")
        ctx.out(f"Wrote {ctx.results_dir / 'summary.json'} and {ctx.results_dir / 'REPORT.md'}")


def _version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "not installed"


def _training_summary(row: dict, trimmed) -> dict:
    out = trimmed(row, ("steps", "update_latency_fit"))
    for name, phase in out["phases"].items():
        if "seconds_per_batch" not in phase:
            phase["seconds_per_batch"] = _batch_seconds([s for s in row.get("steps", []) if s["phase"] == name])
    return out


def _steady_rates(row: dict, calls: list[dict]) -> dict:
    """Throughput up to the moment 95% of a level's requests were done. One slow
    request at the end can stretch the wall time of a short probe a great deal."""
    if row.get("skipped"):
        return {}
    tag = f"throughput-e{row['effort']}-c{row['concurrency']}"
    records = [c for c in calls if c.get("tag") == tag and not c.get("error_class")][-row["requests_completed"] :]
    if len(records) < 20:
        return {}
    start = min(c["started_at"] for c in records)
    done = sorted(records, key=lambda c: c["ended_at"])[: max(1, int(len(records) * 0.95))]
    seconds = done[-1]["ended_at"] - start
    return {
        "requests_per_min_to_95pct_done": _per_minute(len(done), seconds),
        "completion_tokens_per_min_to_95pct_done": _per_minute(sum(c["completion_tokens"] for c in done), seconds),
        "prompt_tokens_per_min_to_95pct_done": _per_minute(sum(c["prompt_tokens"] for c in done), seconds),
    }


def _overlap_summary(row: dict) -> dict:
    """Whether the training clients in the overlap check really trained at the same
    time. A client's later updates overlap another's only if its first update
    finished before the other client was done."""
    overlap = row.get("overlap")
    if not overlap or "per_client" not in overlap:
        return row
    clients = overlap["per_client"]
    first_done = [c["seconds_per_update"][0] for c in clients]
    all_done = [c["total_seconds"] for c in clients]
    overlap["trained_at_the_same_time"] = all(
        first_done[i] < all_done[j] for i in range(len(clients)) for j in range(len(clients)) if i != j
    )
    overlap["longest_wait_for_first_update_s"] = max(first_done)
    return row


def build_summary(config: dict, rows: list[dict], account_after: dict | None = None, calls: list[dict] | None = None) -> dict:
    """Collect the stored rows into one document. Long per-item lists stay in runs/."""

    def of(kind: str) -> list[dict]:
        return [r for r in rows if r.get("kind") == kind]

    def trimmed(row: dict, drop: tuple[str, ...]) -> dict:
        return {k: v for k, v in row.items() if k not in ("key", "kind", "params", *drop)}

    account = {r["params"]["when"]: r for r in of("account")}
    events: Counter = Counter()
    examples: list[dict] = []
    for row in rows:
        observed = row.get("service_events", {})
        events.update(observed.get("counts", {}))
        examples.extend(observed.get("examples", []))
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_id": config.get("model_id"),
        "versions": {p: _version(p) for p in ("tinker", "tml-renderers", "tinker-cookbook")},
        "config": {k: v for k, v in config.items() if k != "model_id"},
        "account_before": account.get("before", {}).get("snapshot"),
        "account_after": account_after,
        "client_config": account.get("before", {}).get("client_config"),
        "throughput": [{**trimmed(r, ()), **_steady_rates(r, calls or [])} for r in of("throughput")],
        "prefill": [trimmed(r, ()) for r in of("prefill")],
        "long_output": [trimmed(r, ()) for r in of("long_output")],
        "behaviour": {r["mode"]: trimmed(r, ("items",)) for r in of("behaviour")},
        "verdict": [trimmed(r, ()) for r in of("verdict")],
        "training": [_training_summary(r, trimmed) for r in of("training")],
        "pipelined_training": [_overlap_summary(trimmed(r, ())) for r in of("pipelined_training")],
        "context": [trimmed(r, ()) for r in of("context")],
        "service_events": {"counts": dict(events), "examples": examples[:12]},
    }
    return summary


def render_report(summary: dict) -> str:
    """A short Markdown report of the measured numbers."""
    lines: list[str] = []
    add = lines.append
    versions = summary["versions"]
    add("# Smoke test report")
    add("")
    add(
        f"Model ID `{summary['model_id']}`. Generated {summary['generated_at']} with tinker "
        f"{versions['tinker']}, tml-renderers {versions['tml-renderers']} and tinker-cookbook "
        f"{versions['tinker-cookbook']}. Every number below was measured by the smoke step unless it says otherwise. "
        "The full data is in `summary.json` next to this file, and the raw rows are in `runs/smoke/`."
    )

    add("")
    add("## Sampling throughput")
    add("")
    add("Short arithmetic prompts, temperature 0, closed loop at a fixed number of requests in flight.")
    add("")
    add("| effort | in flight | requests | req/min | req/min to 95% done | prompt tok/min | completion tok/min | latency p50 (s) | p95 (s) | max (s) | tokens per reply | throttled | failed |")
    add("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for row in summary["throughput"]:
        if row.get("skipped"):
            add(f"| | | skipped | | | | | | | | | {row.get('reason')} | |")
            continue
        add(
            f"| {row['effort']} | {row['concurrency']} | {row['requests_completed']} | {row['requests_per_min']:,} | "
            f"{row.get('requests_per_min_to_95pct_done', '')} | "
            f"{row['prompt_tokens_per_min']:,.0f} | {row['completion_tokens_per_min']:,.0f} | "
            f"{row['latency_s'].get('p50')} | {row['latency_s'].get('p95')} | {row['latency_s'].get('max')} | "
            f"{row['completion_tokens_per_request'].get('mean')} | {row['throttled_requests']} | {row['requests_failed']} |"
        )
    add("")
    add(
        "The rates divide by the wall time of the whole level, so a single slow request at the end pulls them down. "
        "The column to 95% done stops the clock when 95% of the level's requests had returned."
    )
    for row in summary.get("long_output", []):
        fit = row["fit"]
        add("")
        add("Longer replies, to separate the fixed time per request from the time per token:")
        add("")
        add("| words asked for | requests | completion tokens mean | latency p50 (s) | latency max (s) | cut off |")
        add("|---|---|---|---|---|---|")
        for words, group in row["by_requested_words"].items():
            add(
                f"| {words} | {group['requests']} | {group['completion_tokens'].get('mean'):,.0f} | "
                f"{group['latency_s'].get('p50')} | {group['latency_s'].get('max')} | {group['cut_off_by_max_tokens']} |"
            )
        add("")
        add(
            f"A straight-line fit over these {row['requests']} replies, sent {row['concurrency']} at a time, gives "
            f"{fit['fixed_seconds_per_request']} s of fixed time per request plus "
            f"{fit['milliseconds_per_completion_token']} ms per completion token, which is about "
            f"{fit['completion_tokens_per_second_in_one_request']} tokens per second inside one request."
        )
    for row in summary["prefill"]:
        add("")
        add(
            f"Reading long prompts: {row['requests_completed']} prompts of about "
            f"{row['prompt_tokens_per_request'].get('mean'):,.0f} tokens, one generated token each, "
            f"{row['concurrency']} in flight, took {row['wall_s']} s. That is {row['prompt_tokens_per_min']:,.0f} "
            f"prompt tokens per minute, with latency p50 {row['latency_s'].get('p50')} s and "
            f"{row['throttled_requests']} throttled requests."
        )

    add("")
    add("## Throttling")
    add("")
    counts = summary["service_events"]["counts"]
    throttles = {label: n for label, n in counts.items() if _is_throttle(label)}
    others = {label: n for label, n in counts.items() if not _is_throttle(label)}
    levels = [row["concurrency"] for row in summary["throughput"] if not row.get("skipped")]
    if throttles:
        add("Throttling events during the run, by kind.")
        add("")
        add("| event | count |")
        add("|---|---|")
        for label, count in sorted(throttles.items(), key=lambda kv: -kv[1]):
            add(f"| {label} | {count} |")
        add("")
        add("The first few, verbatim, are under `service_events.examples` in `summary.json`.")
    else:
        add(
            f"No throttling was seen at any level, up to {max(levels) if levels else 0} requests in flight. "
            "There was no HTTP 429, no paused queue state, no billing pause and no retry. "
            "So this run cannot say what a throttled request looks like on the free tier, or where the limit is."
        )
    add("")
    add(
        "From reading the SDK source (tinker 0.40.1), which is not the same as observing it: "
        "an HTTP 429 on submit is swallowed by the SDK and the request is sent again after one second, "
        "a queued request reports `paused_rate_limit` or `paused_capacity` with a reason through a queue-state "
        "observer and is logged only after a long grace period, and an HTTP 402 is retried silently for up to an "
        "hour behind the log line \"The job is paused due to billing status\". None of these raises an exception "
        "to the caller, which is why the client in this repository records them itself and stops on a billing pause."
    )
    if others:
        add("")
        add("Other errors and warnings seen during the run.")
        add("")
        add("| event | count |")
        add("|---|---|")
        for label, count in sorted(others.items(), key=lambda kv: -kv[1]):
            add(f"| {label} | {count} |")
        http_errors = [e for e in summary["service_events"]["examples"] if e.get("type") == "http_error"]
        if http_errors and not any(e.get("rate_headers") for e in http_errors):
            add("")
            add("None of the error responses carried a `Retry-After` header or a rate-limit header.")

    if summary["behaviour"]:
        add("")
        add("## What effort 0.0 does")
        add("")
        add("The same prompts (arithmetic, yes/no judgments, short extractions) in three settings.")
        add("")
        add("| setting | prompts | replies with thinking | completion tokens mean | p50 | p90 | max | thinking tokens mean | strict format | correct | cut off |")
        add("|---|---|---|---|---|---|---|---|---|---|---|")
        for mode, row in summary["behaviour"].items():
            o = row["overall"]
            c, t = o["completion_tokens"], o["thinking_tokens"]
            add(
                f"| {mode} | {o['n']} | {o['with_thinking']} ({o['share_with_thinking']:.0%}) | {c['mean']} | {c['p50']} | "
                f"{c['p90']} | {c['max']} | {t['mean']} | {o['parseable']:.0%} | {o['correct']:.0%} | {o['cut_off_by_max_tokens']} |"
            )
        add("")
        add(
            "Completion tokens include the framing tokens around the content: seven for a reply with thinking, "
            "four for a reply without, and two when the answer opening is forced. "
            "The forced-answer setting appends the two tokens that open a text answer to the prompt, "
            "so the model cannot reason first."
        )

    for row in summary["verdict"]:
        s = row["summary"]
        add("")
        add("## Reading a verdict probability")
        add("")
        add(
            f"On {s['n']} yes/no judging prompts at effort 0.0, {s['free_0.0_opens_with_thinking']} replies opened "
            f"with thinking. The model's own probability of opening with thinking averaged "
            f"{s['mean_p_reply_opens_with_thinking_at_0.0']:.4f}."
        )
        add("")
        add("| how the verdict was read | mean probability of the top verdict | accuracy on these prompts |")
        add("|---|---|---|")
        add(f"| first answer token after thinking, effort 0.0 | {s['mean_top_verdict_probability_after_thinking_0.0']} | {s['accuracy_free_0.0']:.0%} |")
        add(f"| first answer token after thinking, effort 0.9 | {s['mean_top_verdict_probability_after_thinking_0.9']} | {s['accuracy_free_0.9']:.0%} |")
        add(f"| answer position scored directly, no thinking | {s['mean_top_verdict_probability_forced']} | {s['accuracy_forced_score']:.0%} |")
        add("")
        add(
            f"Scored directly, yes averaged {s['mean_forced_p_yes_when_label_yes']} when the label was yes and "
            f"{s['mean_forced_p_yes_when_label_no']} when it was no, and the four tokens yes, no, Yes and No held "
            f"{s['mean_mass_on_yes_no_tokens_forced']:.1%} of the probability. The direct score and a sampled reply "
            f"with a forced answer opening agreed to within {s['max_gap_between_forced_sample_and_forced_score']}. "
            f"A direct score took {s['mean_forced_score_latency_s']} s on average."
        )

    for row in summary["training"]:
        add("")
        add("## Training throughput")
        add("")
        add(
            f"LoRA rank {row['rank']}, learning rate {row['learning_rate']}, one training client. "
            f"Creating the client took {row['create_training_client_s']} s."
        )
        add("")
        add("| phase | batches | tokens per batch | seconds for each batch | mean seconds | examples/min | tokens/min |")
        add("|---|---|---|---|---|---|---|")
        for name, phase in row["phases"].items():
            each = ", ".join(f"{x:g}" for x in phase.get("seconds_per_batch", []))
            add(
                f"| {name} | {phase['batches']} | {phase['tokens_per_batch']:,} | {each} | {phase['mean_seconds_per_batch']} | "
                f"{phase['examples_per_min']:,} | {phase['tokens_per_min']:,.0f} |"
            )
        info = row.get("checkpoint") or {}
        size = info.get("size_bytes")
        add("")
        updates = [
            ph
            for name, ph in row["phases"].items()
            if ph.get("seconds_per_batch") and name not in ("forward only", "separate calls")
        ]
        if len(updates) >= 2:
            small = min(updates, key=lambda ph: ph["tokens_per_batch"])
            large = max(updates, key=lambda ph: ph["tokens_per_batch"])
            slowest = max(max(ph["seconds_per_batch"]) for ph in updates)
            add(
                "Each batch here was awaited before the next was sent. "
                f"The fastest batch of {small['tokens_per_batch']:,} tokens took {min(small['seconds_per_batch']):g} s and "
                f"the fastest of {large['tokens_per_batch']:,} tokens took {min(large['seconds_per_batch']):g} s, while "
                f"the slowest batch of all took {slowest:g} s. The time a batch takes follows the service's cycle and "
                "its load more than the batch's size. "
                "The first phase is a forward pass only, and the second awaits the gradient and optimizer calls one after the other."
            )
        add("")
        add(
            f"Saving sampler weights took {row['save_sampler_weights_s']} s and creating a sampler from them took "
            f"{row['create_sampler_from_weights_s']} s. "
            + (f"The checkpoint is {size / 1e6:,.1f} MB. " if size else "")
            + f"It was saved to `{row['tinker_path']}`"
            + (f" and expires at {info['expires_at']}." if info.get("expires_at") else ".")
        )
        add("")
        add("Samples from the adapter:")
        add("")
        add("| expected | reply | thinking tokens | completion tokens | latency (s) |")
        add("|---|---|---|---|---|")
        for sample in row["adapter_samples"]:
            add(
                f"| {sample['expected']} | {sample['text']!r} | {sample['thinking_tokens']} | "
                f"{sample['completion_tokens']} | {sample['latency_s']} |"
            )

    for row in summary.get("pipelined_training", []):
        add("")
        add("## Training with updates sent back to back")
        add("")
        add(
            "Each update was submitted before the previous one had returned, which the Tinker docs "
            "recommend so that a request is always waiting when the service starts a cycle. "
            "Seconds per update leave out the first update of each phase."
        )
        add("")
        add("| phase | updates | tokens per update | seconds per update | examples/min | tokens/min |")
        add("|---|---|---|---|---|---|")
        for name, phase in row["phases"].items():
            add(
                f"| {name} | {phase['updates']} | {phase['tokens_per_update']:,} | {phase['steady_seconds_per_update']} | "
                f"{phase['examples_per_min']:,} | {phase['tokens_per_min']:,.0f} |"
            )
        overlap = row.get("overlap")
        if overlap and "failed" not in overlap:
            waits = ", ".join(f"{c['seconds_per_update'][0]:g} s" for c in overlap["per_client"])
            pauses = sum(c["throttle_events"] for c in overlap["per_client"])
            add("")
            if overlap.get("trained_at_the_same_time"):
                add(
                    f"With {overlap['clients']} training clients running the last phase at the same time, each took "
                    f"{overlap['steady_seconds_per_update_together']} s per update against "
                    f"{overlap['steady_seconds_per_update_alone']} s alone. Their first updates took {waits}."
                )
            else:
                add(
                    f"Then {overlap['clients']} training clients were started on the last phase at once. Their first "
                    f"updates took {waits}. The slower client reported the queue state `paused_rate_limit` {pauses} "
                    "times while it waited, and by the time it began the other had finished, so the two never "
                    "trained at the same time. Once it began it ran at the usual speed. This run therefore shows that "
                    "a second training client is rate limited at the start. It does not show whether two can train side by side."
                )
        elif overlap:
            add("")
            add(f"Running {overlap['clients']} training clients at once did not complete: {overlap['failed']}")

    for row in summary["context"]:
        add("")
        add("## Context limit")
        add("")
        add("| probe | prompt tokens | max tokens | outcome | seconds | detail |")
        add("|---|---|---|---|---|---|")
        for probe in row["probes"]:
            detail = probe.get("error_message") or f"stop reason {probe.get('stop_reason')}, {probe.get('completion_tokens')} tokens generated"
            add(
                f"| {probe['probe']} | {probe['prompt_tokens']:,} | {probe['max_tokens']} | {probe['outcome']} | "
                f"{probe['seconds']} | {str(detail)[:160]} |"
            )

        accepted = [probe for probe in row["probes"] if probe["outcome"] == "accepted"]
        if len(accepted) == 2:
            add("")
            add(
                "A prompt longer than the window is refused at once. When the prompt fits, a `max_tokens` that would "
                "pass the window is still accepted, so the service appears to cap generation at the room that is left. "
                "The reply in the last row ended on its own before reaching that cap, so the cut-off itself was not "
                f"observed. The second full-length prompt had {accepted[1].get('cached_prompt_tokens', 0):,} of its tokens "
                "served from the prefix cache."
            )

    before, after = summary.get("account_before") or {}, summary.get("account_after") or {}
    add("")
    add("## Spend")
    add("")
    add(
        f"Checkpoint storage rate on this account: ${before.get('storage_rate_usd_per_gb_month')} per GB-month. "
        f"Checkpoints stored before the run: {before.get('checkpoint_count')}. Checkpoints listed after it: "
        f"{after.get('checkpoints_listed')}, holding {(after.get('checkpoints_listed_bytes') or 0) / 1e6:,.0f} MB. "
        f"Estimated cost of all metered usage in the 24 hours up to the end of the run: "
        f"${after.get('usage_estimated_cost_usd')}. The service's usage records can lag by some minutes."
    )
    add("")
    return "\n".join(lines)
