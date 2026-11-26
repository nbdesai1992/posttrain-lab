"""Survey: the base model on public benchmarks, in each setting.

Three settings, each applied the same way to every benchmark and, later, to every
adapter.

* ``fast`` is a forced direct answer at effort 0.0. The two tokens that open a text
  answer are appended to the prompt, so there is no reasoning block.
* ``slow`` is effort 0.9, free-running.
* ``effort_0`` is effort 0.0, free-running: the model's shipped no-effort setting,
  which still reasons. It runs on a fixed subsample as a reference line.

A unit is one item in one setting. Units only sample. Grading happens at the end of
every run from the stored replies, so a grader can be fixed and rerun offline.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from collections import Counter
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from ..benchmarks import BENCHMARKS, NOT_RUN, Benchmark, Item, sources
from ..benchmarks.base import fixed_sample
from ..client import AccountBlockedError, ModelClient
from ..render import DEFAULT_EFFORT, FAST_EFFORT
from ..stats import bootstrap, describe, paired_difference, percentile
from ..store import JsonlStore, append_jsonl, read_jsonl, stable_hash
from .base import Step, StepContext, Unit

SETTINGS = {
    # name: (effort, force the reply to open as an answer)
    "fast": (FAST_EFFORT, True),
    "slow": (DEFAULT_EFFORT, False),
    "effort_0": (FAST_EFFORT, False),
}
SETTING_NOTES = {
    "fast": "forced direct answer at effort 0.0 (no reasoning block)",
    "slow": "effort 0.9, free-running",
    "effort_0": "effort 0.0, free-running (the shipped no-effort setting), on a fixed subsample",
}
GRADER_VERSION = 2  # bump to regrade everything after a grader change
CONTEXT_WINDOW = 65536
TARGET_GAP, USABLE_GAP = 10.0, 3.0  # points, from the study design


def reference_subsample(items: list[Item], size: int, seed: int) -> set[str]:
    """IDs of a fixed subsample of about ``size`` items that keeps paired items together."""
    clusters = list(dict.fromkeys(item.cluster for item in items))
    per_cluster = len(items) / len(clusters)
    chosen = set(fixed_sample(clusters, max(1, round(size / per_cluster)), seed))
    return {item.id for item in items if item.cluster in chosen}


def classify(benchmark: Benchmark, gap_points: float | None) -> str:
    """The target rule: fast trails slow by 10 points or more, or is within 3."""
    if gap_points is None:
        return "not measured"
    if gap_points >= TARGET_GAP:
        band = "training target"
    elif gap_points <= USABLE_GAP:
        band = "usable as is"
    else:
        band = "in between"
    return f"watch only ({band})" if benchmark.role == "watch" else band


class Survey(Step):
    name = "survey"
    summary = "Score the base model on the public benchmarks in the fast and slow settings"
    default_config = "configs/survey.toml"
    parallel_units = True

    # ------------------------------------------------------------------ plan

    def _root(self, ctx: StepContext) -> Path:
        return ctx.runs_dir / "sources"

    def _load(self, ctx: StepContext) -> dict[str, dict]:
        """Every configured benchmark with its items and reference subsample."""
        if "benchmarks" in ctx.scratch:
            return ctx.scratch["benchmarks"]
        seed = ctx.config.get("seed", 0)
        root = self._root(ctx)
        names = ctx.config.get("benchmarks", list(BENCHMARKS))
        registry = ctx.config.get("_registry", BENCHMARKS)  # tests pass their own
        if not ctx.config.get("_skip_download"):
            wanted = {registry[name].source for name in names} | {"ifeval_checker", "nltk_data"}
            sources.ensure(root, wanted, out=ctx.out)
        loaded = {}
        for name in names:
            benchmark = registry[name]
            items = benchmark.load(root, seed)
            if ctx.config.get("limit"):  # a trial run on a few items per benchmark
                keep = set(fixed_sample(list(dict.fromkeys(i.cluster for i in items)), ctx.config["limit"], seed))
                items = [item for item in items if item.cluster in keep]
            loaded[name] = {
                "benchmark": benchmark,
                "items": {item.id: item for item in items},
                "reference": reference_subsample(
                    items, min(benchmark.reference_size or 10**9, ctx.config.get("reference_sample", 200)), seed
                ),
            }
        ctx.scratch["benchmarks"] = loaded
        return loaded

    def plan(self, ctx: StepContext) -> list[Unit]:
        estimates = ctx.config.get("estimates", {})
        units = []
        for name, entry in self._load(ctx).items():
            benchmark: Benchmark = entry["benchmark"]
            for setting in ctx.config.get("settings", list(SETTINGS)):
                for item in entry["items"].values():
                    if setting == "effort_0" and item.id not in entry["reference"]:
                        continue
                    calls = 1
                    chars = sum(len(m["content"]) for m in item.messages)
                    if benchmark.multi_step:
                        calls = int(len(item.extra["question"]) * estimates.get("steps_per_turn", 2.5))
                        chars = calls * (len(json.dumps(item.extra["function"])) + 2000)
                    limit = benchmark.max_tokens_fast if setting == "fast" else benchmark.max_tokens_slow
                    expected = min(limit, estimates.get(setting, {}).get(name, estimates.get("completion_tokens", 300)))
                    units.append(
                        Unit(
                            name,
                            {"benchmark": name, "setting": setting, "item": item.id, "prompt": item.fingerprint()},
                            calls=calls,
                            prompt_tokens=int(chars / estimates.get("chars_per_token", 3.5)),
                            completion_tokens=calls * expected,
                            max_completion_tokens=calls * limit,
                        )
                    )
        ctx.scratch["planned"] = len(units)
        return units

    # ------------------------------------------------------------------- run

    async def run_unit(self, unit: Unit, ctx: StepContext) -> dict:
        client: ModelClient = ctx.client
        entry = self._load(ctx)[unit.params["benchmark"]]
        benchmark: Benchmark = entry["benchmark"]
        item: Item = entry["items"][unit.params["item"]]
        setting = unit.params["setting"]
        effort, forced = SETTINGS[setting]
        special = ctx.renderer.special
        opening = [special.message_model, special.content_text] if forced else []
        max_tokens = benchmark.max_tokens_fast if setting == "fast" else benchmark.max_tokens_slow
        totals = Counter()
        last: dict = {}

        async def ask(messages: list[dict]) -> str | None:
            prompt = ctx.renderer.build_prompt(messages, effort)
            if len(prompt) + len(opening) + 16 > CONTEXT_WINDOW:
                totals["prompt_tokens"] += len(prompt)
                return None  # the prompt does not fit the model's context window
            result = await client.sample_tokens(
                [*prompt, *opening],
                forced_opening=opening,
                effort=effort,
                max_tokens=max_tokens,
                top_logprobs=5 if (benchmark.choices and forced) else 0,
                tag=f"{benchmark.name}:{setting}",
            )
            record = result.record
            totals.update(
                calls=1, prompt_tokens=record.prompt_tokens, completion_tokens=record.completion_tokens,
                thinking_tokens=record.thinking_tokens, answer_tokens=record.answer_tokens,
                cut_off=int(not result.parsed.ended),
            )
            totals["latency_s"] += record.latency_s
            last.update(result=result)
            return result.text

        if benchmark.multi_step:
            overflow = False

            async def ask_or_stop(messages: list[dict]) -> str:
                nonlocal overflow
                reply = None if overflow else await ask(messages)
                if reply is None:
                    overflow = True
                    return ""  # an empty reply ends the turn
                return reply

            row = await benchmark.converse(item, ask_or_stop)
            if overflow:
                row["forced_stop"] = True
                row["failed"] = "context_overflow"
        else:
            text = await ask(item.messages)
            if text is None:
                row = {"text": "", "failed": "context_overflow"}
            else:
                result = last["result"]
                row = {"text": text, "stop_reason": result.stop_reason, "first_kind": result.parsed.first_kind}
                if benchmark.choices and forced and result.top_logprobs:
                    # The model's own probabilities for the first answer token.
                    row["first_token"] = [
                        [ctx.renderer.decode([token]), round(math.exp(logprob), 5)] for token, logprob in result.top_logprobs[0]
                    ]
        row.update({key: totals[key] for key in ("calls", "prompt_tokens", "completion_tokens", "thinking_tokens", "answer_tokens", "cut_off")})
        row["latency_s"] = round(totals["latency_s"], 3)
        row.update(benchmark=benchmark.name, setting=setting, item=item.id)
        self._progress(ctx)
        return row

    def _progress(self, ctx: StepContext) -> None:
        state = ctx.scratch.setdefault("progress", {"done": 0, "started": time.monotonic(), "last": time.monotonic()})
        state["done"] += 1
        now = time.monotonic()
        if now - state["last"] >= ctx.config.get("progress_seconds", 60):
            state["last"] = now
            minutes = (now - state["started"]) / 60
            totals = ctx.client.totals
            ctx.out(
                f"  {state['done']:,} units in {minutes:.1f} min ({state['done'] / minutes:,.0f}/min), "
                f"{totals['api_calls']:,} API calls, {totals['cache_hits']:,} cache hits, "
                f"{totals['completion_tokens']:,} completion tokens, {len(ctx.client.backend.events)} service events"
            )

    # ----------------------------------------------------------------- grade

    async def _grade(self, ctx: StepContext, rows: list[dict]) -> dict[tuple[str, str, str], dict]:
        """Grade every stored reply, reusing grades that were already computed."""
        loaded = self._load(ctx)
        store = JsonlStore(ctx.runs_dir / "grades.jsonl")
        root = self._root(ctx)
        grades: dict[tuple[str, str, str], dict] = {}
        async def grade(benchmark: Benchmark, item: Item, row: dict, slots: asyncio.Semaphore) -> None:
            reply = {key: row.get(key) for key in ("text", "turns", "forced_stop")}
            key = stable_hash([GRADER_VERSION, benchmark.name, item.id, item.fingerprint(), reply])
            cached = store.get(key)
            if cached is None:
                if row.get("failed"):
                    result = {"score": 0.0, "failed": row["failed"]}
                else:
                    async with slots:
                        result = await asyncio.to_thread(benchmark.grade, item, reply)
                cached = store.put(key, {"benchmark": benchmark.name, "item": item.id, "grade": result})
            grades[(row["benchmark"], row["setting"], row["item"])] = cached["grade"]

        for name, entry in loaded.items():
            mine = [row for row in rows if row.get("benchmark") == name and row["item"] in entry["items"]]
            if not mine:
                continue
            await asyncio.to_thread(entry["benchmark"].prepare, root)
            started = time.monotonic()
            # Graders run one at a time unless the benchmark says its grader can overlap.
            workers = ctx.config.get("grading_workers", 4) if entry["benchmark"].parallel_grading else 1
            slots = asyncio.Semaphore(workers)
            await asyncio.gather(*(grade(entry["benchmark"], entry["items"][row["item"]], row, slots) for row in mine))
            ctx.out(f"  graded {name}: {len(mine):,} replies in {time.monotonic() - started:.0f} s")
        return grades

    async def finish(self, ctx: StepContext, rows: list[dict]) -> None:
        if ctx.client is not None:
            for event in ctx.client.backend.events[ctx.scratch.get("events_written", 0) :]:
                append_jsonl(ctx.runs_dir / "events.jsonl", event.to_dict())
            ctx.scratch["events_written"] = len(ctx.client.backend.events)
        grades = await self._grade(ctx, rows)
        calls = list(read_jsonl(ctx.runs_dir / "calls.jsonl"))
        events = list(read_jsonl(ctx.runs_dir / "events.jsonl"))
        account = await ctx.client.backend.account_snapshot() if ctx.client is not None else None
        summary = build_summary(ctx.config, self._load(ctx), rows, grades, calls, events, account, self._root(ctx))
        ctx.results_dir.mkdir(parents=True, exist_ok=True)
        (ctx.results_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
        (ctx.results_dir / "figure-data.json").write_text(json.dumps(figure_data(summary), indent=1) + "\n", encoding="utf-8")
        (ctx.results_dir / "REPORT.md").write_text(render_report(summary), encoding="utf-8")
        # Item IDs per benchmark, so a later run can be checked against the same items.
        (ctx.results_dir / "item-ids.json").write_text(
            json.dumps({name: {"items": sorted(e["items"]), "effort_0_subsample": sorted(e["reference"])} for name, e in self._load(ctx).items()}) + "\n",
            encoding="utf-8",
        )
        ctx.out(f"Wrote summary.json, figure-data.json, item-ids.json and REPORT.md to {ctx.results_dir}")


# ------------------------------------------------------------------ summary


def _version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "not installed"


def _setting_summary(benchmark: Benchmark, items: list[Item], rows: list[dict], grades: list[dict], seed: int) -> dict:
    scores = [g["score"] for g in grades]
    clusters = [item.cluster for item in items]
    groups = [item.group for item in items]
    ran = [row for row in rows if not row.get("failed")]
    out = {
        "n": len(rows),
        "score": bootstrap(scores, clusters=clusters, groups=groups, macro=benchmark.macro, seed=seed),
        "tokens_per_answer": {
            "prompt": round(sum(r["prompt_tokens"] for r in ran) / max(len(ran), 1), 1),
            "completion": describe((r["completion_tokens"] for r in ran), 1),
            "thinking": describe((r["thinking_tokens"] for r in ran), 1),
            "answer": round(sum(r["answer_tokens"] for r in ran) / max(len(ran), 1), 1),
        },
        "latency_s": describe(r["latency_s"] for r in ran),
        "calls_per_item": round(sum(r["calls"] for r in ran) / max(len(ran), 1), 2),
        "cut_off_by_max_tokens": sum(1 for r in ran if r.get("cut_off")),
        "context_overflow": sum(1 for r in rows if r.get("failed") == "context_overflow"),
    }
    strict = [g["strict"] for g in grades if "strict" in g]
    if strict:
        out["bare_answer_format"] = round(sum(strict) / len(strict), 4)
    by_group: dict[str, list[float]] = {}
    for group, score in zip(groups, scores):
        by_group.setdefault(group, []).append(score)
    if len(by_group) > 1:
        out["by_group"] = {g: {"n": len(v), "score": round(sum(v) / len(v), 4)} for g, v in sorted(by_group.items())}
    out.update(benchmark.extra_metrics(items, grades))
    return out


def _paired(benchmark: Benchmark, items: dict[str, Item], first: dict[str, dict], second: dict[str, dict], seed: int) -> dict | None:
    common = [item_id for item_id in first if item_id in second]
    if not common:
        return None
    return paired_difference(
        [first[i]["score"] for i in common],
        [second[i]["score"] for i in common],
        clusters=[items[i].cluster for i in common],
        groups=[items[i].group for i in common],
        macro=benchmark.macro,
        seed=seed,
    )


def soak_timeline(calls: list[dict], events: list[dict], bucket_minutes: int = 10) -> list[dict]:
    """Throughput, latency and service events in fixed windows of wall-clock time."""
    done = [c for c in calls if c.get("kind") == "sample" and not c.get("cache_hit") and c.get("ended_at")]
    if not done:
        return []
    width = bucket_minutes * 60
    buckets: dict[int, list[dict]] = {}
    for call in done:
        buckets.setdefault(int(call["ended_at"] // width), []).append(call)
    event_counts: dict[int, Counter] = {}
    for event in events:
        label = event.get("state") or (f"http {event.get('status')}" if event.get("status") else event.get("type"))
        event_counts.setdefault(int(event["ts"] // width), Counter())[label] += 1
    timeline = []
    for bucket, members in sorted(buckets.items()):
        good = [c for c in members if not c.get("error_class")]
        span = max(c["ended_at"] for c in members) - min(c["started_at"] for c in members)
        seconds = min(max(span, 1.0), width + 600)
        timeline.append(
            {
                "window_start": datetime.fromtimestamp(bucket * width, timezone.utc).isoformat(timespec="minutes"),
                "requests": len(good),
                "errors": len(members) - len(good),
                "requests_per_min": round(len(good) / seconds * 60, 1),
                "completion_tokens_per_min": round(sum(c["completion_tokens"] for c in good) / seconds * 60),
                "prompt_tokens_per_min": round(sum(c["prompt_tokens"] for c in good) / seconds * 60),
                "latency_p50_s": round(percentile([c["latency_s"] for c in good], 50), 2) if good else None,
                "latency_p95_s": round(percentile([c["latency_s"] for c in good], 95), 2) if good else None,
                "latency_max_s": round(max((c["latency_s"] for c in good), default=0), 1),
                "retries": sum(c["attempts"] - 1 for c in members),
                "throttle_events": sum(c.get("throttle_events", 0) for c in members),
                "service_events": dict(event_counts.get(bucket, {})),
            }
        )
    return timeline


def build_summary(config, loaded, rows, grades, calls, events, account, root) -> dict:
    seed = config.get("seed", 0)
    licences = sources.licences()
    benchmarks = {}
    for name, entry in loaded.items():
        benchmark: Benchmark = entry["benchmark"]
        items: dict[str, Item] = entry["items"]
        per_setting: dict[str, dict[str, dict]] = {}
        row_by: dict[str, dict[str, dict]] = {}
        for row in rows:
            if row.get("benchmark") == name and row["item"] in items and (name, row["setting"], row["item"]) in grades:
                per_setting.setdefault(row["setting"], {})[row["item"]] = grades[(name, row["setting"], row["item"])]
                row_by.setdefault(row["setting"], {})[row["item"]] = row
        settings = {}
        for setting, graded in per_setting.items():
            ids = [i for i in items if i in graded]
            settings[setting] = _setting_summary(
                benchmark, [items[i] for i in ids], [row_by[setting][i] for i in ids], [graded[i] for i in ids], seed
            )
            settings[setting]["complete"] = len(ids) == (len(entry["reference"]) if setting == "effort_0" else len(items))
        fast, slow, ref = per_setting.get("fast", {}), per_setting.get("slow", {}), per_setting.get("effort_0", {})
        gap = _paired(benchmark, items, fast, slow, seed)
        gap_points = round(gap["estimate"] * 100, 1) if gap and gap["estimate"] is not None else None
        on_subsample = {}
        for setting, graded in (("fast", fast), ("slow", slow)):
            ids = [i for i in ref if i in graded]
            if ids:
                on_subsample[setting] = bootstrap(
                    [graded[i]["score"] for i in ids], clusters=[items[i].cluster for i in ids],
                    groups=[items[i].group for i in ids], macro=benchmark.macro, seed=seed,
                )
        benchmarks[name] = {
            "title": benchmark.title,
            "job": benchmark.job,
            "role": benchmark.role,
            "graded_by": benchmark.graded_by,
            "licence": licences.get(benchmark.source),
            "chance": benchmark.chance,
            "score_is": "mean over groups" if benchmark.macro else "mean over items",
            "items": len(items),
            "settings": settings,
            "slow_minus_fast": gap,
            "gap_points": gap_points,
            "classification": classify(benchmark, gap_points),
            "effort_0_subsample": {
                "items": len(entry["reference"]),
                "fast_on_subsample": on_subsample.get("fast"),
                "slow_on_subsample": on_subsample.get("slow"),
                "effort_0_minus_fast": _paired(benchmark, items, {i: fast[i] for i in ref if i in fast}, ref, seed),
                "slow_minus_effort_0": _paired(benchmark, items, ref, {i: slow[i] for i in ref if i in slow}, seed),
            },
        }
    sampled = [c for c in calls if c.get("kind") == "sample" and not c.get("cache_hit") and not c.get("error_class")]
    event_counts = Counter(
        e.get("state") or (f"http {e.get('status')}" if e.get("status") else f"{e.get('type')}: {str(e.get('message'))[:50]}") for e in events
    )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_id": config.get("model_id"),
        "versions": {p: _version(p) for p in ("tinker", "tml-renderers", "tinker-cookbook", "jsonschema", "nltk")},
        "seed": seed,
        "settings": SETTING_NOTES,
        "target_rule": f"training target if slow minus fast is at least {TARGET_GAP:g} points, usable as is if it is at most {USABLE_GAP:g}",
        "benchmarks": benchmarks,
        "not_run": NOT_RUN,
        "pins": {"sources": len(sources.load_pins()), "bfcl_revision": sources.BFCL_REV, "ifeval_checker_revision": sources.IFEVAL_REV},
        "usage": {
            "api_calls": len(sampled),
            "prompt_tokens": sum(c["prompt_tokens"] for c in sampled),
            "completion_tokens": sum(c["completion_tokens"] for c in sampled),
            "thinking_tokens": sum(c["thinking_tokens"] for c in sampled),
            "failed_calls": sum(1 for c in calls if c.get("error_class")),
            "retries": sum(c.get("attempts", 1) - 1 for c in calls),
            "throttled_calls": sum(1 for c in calls if c.get("throttle_events")),
        },
        "soak": soak_timeline(calls, events),
        "service_events": dict(event_counts),
        "account_after": account,
        "max_concurrency": config.get("client", {}).get("max_concurrency"),
    }


def figure_data(summary: dict) -> dict:
    """Exactly the numbers the capability map and the cost figure need."""
    capability, cost = [], []
    for name, b in summary["benchmarks"].items():
        row = {"benchmark": name, "title": b["title"], "job": b["job"], "role": b["role"], "chance": b["chance"],
               "gap_points": b["gap_points"], "classification": b["classification"]}
        for setting in SETTINGS:
            s = b["settings"].get(setting)
            if not s:
                continue
            row[setting] = {"score": s["score"]["estimate"], "low": s["score"]["low"], "high": s["score"]["high"], "n": s["n"]}
            cost.append(
                {
                    "benchmark": name, "title": b["title"], "setting": setting, "score": s["score"]["estimate"],
                    "completion_tokens_per_answer": s["tokens_per_answer"]["completion"].get("mean"),
                    "thinking_tokens_per_answer": s["tokens_per_answer"]["thinking"].get("mean"),
                    "prompt_tokens_per_answer": s["tokens_per_answer"]["prompt"],
                    "latency_p50_s": s["latency_s"].get("p50"),
                }
            )
        capability.append(row)
    return {
        "generated_at": summary["generated_at"],
        "settings": summary["settings"],
        "capability_map": capability,
        "tokens_per_answer_against_score": cost,
        "not_run": [{"benchmark": k, **v} for k, v in summary["not_run"].items()],
    }


# ------------------------------------------------------------------- report


def _pct(value: float | None) -> str:
    return "" if value is None else f"{value * 100:.1f}"


def _score(setting: dict | None) -> str:
    if not setting:
        return ""
    s = setting["score"]
    return f"{_pct(s['estimate'])} ({_pct(s['low'])} to {_pct(s['high'])})"


def render_report(summary: dict) -> str:
    lines: list[str] = []
    add = lines.append
    v = summary["versions"]
    add("# Survey report")
    add("")
    add(
        f"The base model `{summary['model_id']}` on public benchmarks, generated {summary['generated_at']} with tinker "
        f"{v['tinker']}, tml-renderers {v['tml-renderers']} and tinker-cookbook {v['tinker-cookbook']}. "
        "Every score comes from this repository's harness at temperature 0 with one prompt per benchmark, "
        "and every reply was graded by a program. Scores are percentages with 95% bootstrap intervals. "
        "The full data is in `summary.json`, and the raw replies are in `runs/survey/`."
    )
    add("")
    add("The settings:")
    add("")
    for name, note in summary["settings"].items():
        add(f"- `{name}` is {note}.")

    add("")
    add("## Capability map")
    add("")
    add("| benchmark | job | items | fast | slow | slow minus fast (points) | by the target rule |")
    add("|---|---|---|---|---|---|---|")
    for name, b in summary["benchmarks"].items():
        gap = b["slow_minus_fast"]
        gap_text = "" if not gap or gap["estimate"] is None else f"{gap['estimate'] * 100:+.1f} ({gap['low'] * 100:+.1f} to {gap['high'] * 100:+.1f})"
        add(
            f"| {b['title']} | {b['job']} | {b['items']:,} | {_score(b['settings'].get('fast'))} | "
            f"{_score(b['settings'].get('slow'))} | {gap_text} | {b['classification']} |"
        )
    for name, b in summary["not_run"].items():
        add(f"| {b['title']} | {b['job']} | | | | | {b['reason']} |")
    add("")
    add(
        f"The rule, fixed before the run: {summary['target_rule']}. Jobs marked watch only are never trained. "
        "A score that is a mean over groups (RewardBench 2 subsets, BFCL categories) weights every group equally. "
        "Published leaderboard numbers are left out of this report, so the rule is applied to the gap between the two settings only."
    )
    incomplete = [f"{b['title']} ({s})" for b in summary["benchmarks"].values() for s, d in b["settings"].items() if not d.get("complete")]
    if incomplete:
        add("")
        add("Not every planned item has a reply yet in these cells, so their numbers are provisional: " + ", ".join(incomplete) + ".")

    add("")
    add("## The shipped no-effort setting, on a subsample")
    add("")
    add("Free-running effort 0.0 still reasons before it answers. It ran on a fixed subsample of each benchmark, shown next to the other two settings on the same items.")
    add("")
    add("| benchmark | items | fast | effort_0 | slow | reasoning at effort_0 |")
    add("|---|---|---|---|---|---|")
    for name, b in summary["benchmarks"].items():
        e = b["settings"].get("effort_0")
        sub = b["effort_0_subsample"]
        if not e:
            continue
        thinking = e["tokens_per_answer"]["thinking"]
        share = "" if not thinking.get("n") else f"mean {thinking['mean']:,.0f} thinking tokens"
        add(
            f"| {b['title']} | {e['n']} | {_score({'score': sub['fast_on_subsample']}) if sub['fast_on_subsample'] else ''} | {_score(e)} | "
            f"{_score({'score': sub['slow_on_subsample']}) if sub['slow_on_subsample'] else ''} | {share} |"
        )

    add("")
    add("## Tokens and time per answer")
    add("")
    add("| benchmark | setting | prompt tokens | completion tokens mean | p50 | p90 | max | thinking tokens mean | p90 | latency p50 (s) | p95 (s) | cut off | too long for the context |")
    add("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, b in summary["benchmarks"].items():
        for setting, s in b["settings"].items():
            t = s["tokens_per_answer"]
            c, th, lat = t["completion"], t["thinking"], s["latency_s"]
            add(
                f"| {b['title']} | {setting} | {t['prompt']:,.0f} | {c.get('mean', 0):,.0f} | {c.get('p50', 0):,.0f} | {c.get('p90', 0):,.0f} | "
                f"{c.get('max', 0):,.0f} | {th.get('mean', 0):,.0f} | {th.get('p90', 0):,.0f} | {lat.get('p50', '')} | {lat.get('p95', '')} | "
                f"{s['cut_off_by_max_tokens']} | {s['context_overflow']} |"
            )
    add("")
    add("For the multi-turn benchmark the figures are totals over all the model calls an item took.")

    judging = [(n, b) for n, b in summary["benchmarks"].items() if b["job"] == "Judging answers"]
    if judging:
        add("")
        add("## Judging in detail")
        add("")
        add("| benchmark | setting | score | false accepts | false rejects | same verdict after swapping | right in both orders | picked the first position | unparsed |")
        add("|---|---|---|---|---|---|---|---|---|")
        for name, b in judging:
            for setting, s in b["settings"].items():
                add(
                    f"| {b['title']} | {setting} | {_pct(s['score']['estimate'])} | {_pct(s.get('false_accept_rate'))} | {_pct(s.get('false_reject_rate'))} | "
                    f"{_pct(s.get('swap_consistency'))} | {_pct(s.get('both_orders_correct'))} | {_pct(s.get('picked_first_position'))} | {s.get('unparsed', '')} |"
                )
        add("")
        add(
            "False accepts and false rejects need a yes or no verdict on one response, so they come from the pointwise form of JudgeBench: "
            "a false accept is a yes to an incorrect response and a false reject is a no to a correct one. "
            "The pairwise benchmarks show each pair in both orders, and their intervals resample whole pairs."
        )

    grouped = [(n, b) for n, b in summary["benchmarks"].items() if any("by_group" in s for s in b["settings"].values())]
    if grouped:
        add("")
        add("## Scores by group")
        for name, b in grouped:
            fast, slow = b["settings"].get("fast", {}).get("by_group", {}), b["settings"].get("slow", {}).get("by_group", {})
            add("")
            add(f"{b['title']}:")
            add("")
            add("| group | items | fast | slow |")
            add("|---|---|---|---|")
            for group in sorted(set(fast) | set(slow)):
                n = (fast.get(group) or slow.get(group))["n"]
                add(f"| {group} | {n} | {_pct(fast.get(group, {}).get('score'))} | {_pct(slow.get(group, {}).get('score'))} |")

    add("")
    add("## Throughput over the run")
    add("")
    usage = summary["usage"]
    add(
        f"The run kept up to {summary.get('max_concurrency')} requests in flight. It made {usage['api_calls']:,} sampling calls with "
        f"{usage['prompt_tokens']:,} prompt tokens and {usage['completion_tokens']:,} completion tokens, of which "
        f"{usage['thinking_tokens']:,} were thinking tokens. {usage['failed_calls']} calls failed, {usage['retries']} were retried "
        f"and {usage['throttled_calls']} saw a throttling event."
    )
    add("")
    add("| window start (UTC) | requests | req/min | prompt tok/min | completion tok/min | latency p50 (s) | p95 (s) | max (s) | errors | throttle events |")
    add("|---|---|---|---|---|---|---|---|---|---|")
    for w in summary["soak"]:
        add(
            f"| {w['window_start'][5:16].replace('T', ' ')} | {w['requests']:,} | {w['requests_per_min']:,} | {w['prompt_tokens_per_min']:,} | "
            f"{w['completion_tokens_per_min']:,} | {w['latency_p50_s']} | {w['latency_p95_s']} | {w['latency_max_s']} | {w['errors']} | {w['throttle_events']} |"
        )
    if summary["service_events"]:
        add("")
        add("Service events recorded during the run: " + ", ".join(f"{k} ({n})" for k, n in sorted(summary["service_events"].items(), key=lambda kv: -kv[1])) + ".")
    else:
        add("")
        add("The service raised no throttling event, no paused queue state and no billing pause during the run.")
    account = summary.get("account_after") or {}
    if account:
        add("")
        add(
            f"Estimated cost of all metered usage in the 24 hours up to the end of the run, as reported by the service: "
            f"${account.get('usage_estimated_cost_usd')}. Its usage records lag behind the calls."
        )

    add("")
    add("## What was run, and under which licence")
    add("")
    add("| benchmark | items | graded by | licence |")
    add("|---|---|---|---|")
    for name, b in summary["benchmarks"].items():
        add(f"| {b['title']} | {b['items']:,} | {b['graded_by']} | {b['licence']} |")
    for name, b in summary["not_run"].items():
        add(f"| {b['title']} | | | {b['reason']} |")
    add("")
    add(
        f"Every file is downloaded without a login, pinned to a revision and checked against a SHA-256 value "
        f"({summary['pins']['sources']} files). BFCL is revision `{summary['pins']['bfcl_revision'][:12]}` of the gorilla repository "
        f"and the IFEval checker is revision `{summary['pins']['ifeval_checker_revision'][:12]}` of google-research. "
        "From BFCL V4 the survey runs the Python single-turn categories and the four multi-turn categories. "
        "It leaves out `simple_java` and `simple_javascript`, whose decoders need tree-sitter, and the agentic `web_search` and `memory` "
        "categories, which need an outside search service or an embedding model. "
        "BFCL's prompt text, AST checker, simulated back ends and state checks are used unchanged, while the reply decoding and the call "
        "execution are this repository's own, because BFCL's run model output through `eval`. "
        "From RewardBench 2 the Ties subset is left out: it has several right answers per prompt and its own rating metric, which a pick-one task cannot score. "
        "JSONSchemaBench items whose schema does not fit the 64K context are counted as failures."
    )
    add("")
    return "\n".join(lines)
