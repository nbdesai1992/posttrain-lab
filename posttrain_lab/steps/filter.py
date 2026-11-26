"""Filter: sort every candidate by whether the fast and the reasoning setting solve it.

Each candidate from build-data is run twice, graded by program, and put in a cell:
easy (both right), middle band (fast wrong, reasoning right), hard (both wrong) or
rare (fast right, reasoning wrong). The end of the run builds the training sets the
arms need and writes them, with the labeled items, to ``results/data/``.

The sampling, caching and grading are the survey's, with the candidates standing in
as one more benchmark.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from ..benchmarks.base import Benchmark, Item, user
from ..data import generators
from ..data.sets import CELLS, MAIN_SIZE, SIZES, build_sets, cell_of
from ..stats import describe
from ..store import read_jsonl
from .base import StepContext
from .survey import Survey, soak_timeline


class Candidates(Benchmark):
    name = "synthetic"
    title = "Generated reasoning problems"
    job = "Reasoning without the wait"
    graded_by = "the answer the generator computed"
    max_tokens_fast = 256
    max_tokens_slow = 8192

    def __init__(self, path: Path):
        self.path = Path(path)

    def load(self, root: Path, seed: int) -> list[Item]:
        return [Item(id=row["id"], messages=user(row["prompt"]), answer=row, group=row["family"]) for row in read_jsonl(self.path)]

    def grade(self, item: Item, reply: dict) -> dict:
        return generators.grade_answer(item.answer, reply.get("text", ""))


class Filter(Survey):
    name = "filter"
    summary = "Sort candidate items by whether the fast and the reasoning setting get them right, and build the training sets"
    default_config = "configs/filter.toml"

    def _load(self, ctx: StepContext) -> dict[str, dict]:
        if "benchmarks" not in ctx.scratch:
            path = Path(ctx.config.get("candidates") or ctx.runs_dir.parent / "build-data" / "candidates.jsonl")
            benchmark = Candidates(path)
            items = benchmark.load(path.parent, 0)
            ctx.scratch["benchmarks"] = {"synthetic": {"benchmark": benchmark, "items": {item.id: item for item in items}, "reference": set()}}
            ctx.scratch["build_summary"] = json.loads((path.parent / "summary.json").read_text()) if (path.parent / "summary.json").exists() else {}
        return ctx.scratch["benchmarks"]

    async def finish(self, ctx: StepContext, rows: list[dict]) -> None:
        grades = await self._grade(ctx, rows)
        items = self._load(ctx)["synthetic"]["items"]
        by_setting: dict[str, dict[str, dict]] = {"fast": {}, "reasoning": {}}
        for row in rows:
            if row.get("setting") in by_setting and row["item"] in items:
                by_setting[row["setting"]][row["item"]] = {**row, "score": grades[("synthetic", row["setting"], row["item"])]["score"]}
        labeled = []
        for item_id, item in items.items():
            fast, reasoning = by_setting["fast"].get(item_id), by_setting["reasoning"].get(item_id)
            if fast is None or reasoning is None:
                continue  # not run in both settings yet
            labeled.append(
                {
                    **{k: item.answer[k] for k in ("id", "family", "level", "params", "question", "answer", "answer_type", "prompt", "seed", "generator_version", "prompt_tokens", "train_tokens")},
                    "cell": cell_of(fast["score"] == 1.0, reasoning["score"] == 1.0),
                    "fast_right": fast["score"] == 1.0,
                    "reasoning_right": reasoning["score"] == 1.0,
                    "fast_completion_tokens": fast["completion_tokens"],
                    "reasoning_completion_tokens": reasoning["completion_tokens"],
                    "reasoning_thinking_tokens": reasoning["thinking_tokens"],
                    "reasoning_cut_off": bool(reasoning.get("cut_off")),
                    "reasoning_latency_s": reasoning["latency_s"],
                }
            )
        (ctx.runs_dir / "items.jsonl").write_text("".join(json.dumps(row) + "\n" for row in labeled), encoding="utf-8")
        built = build_sets(labeled, seed=ctx.config.get("seed", 0), dev_per_cell=ctx.config.get("dev_per_cell", 100), sizes=tuple(ctx.config.get("sizes", SIZES)), main_size=ctx.config.get("main_size", MAIN_SIZE))
        calls = list(read_jsonl(ctx.runs_dir / "calls.jsonl"))
        summary = build_summary(ctx, labeled, built, calls, len(items))
        write_outputs(ctx.results_dir.parent / "data", labeled, built, summary)
        ctx.out(f"{len(labeled):,} items labeled: " + ", ".join(f"{cell} {summary['cells']['overall'][cell]['count']:,}" for cell in CELLS))
        ctx.out(f"Wrote the report, summary, figure data, items and sets to {ctx.results_dir.parent / 'data'}")


def _cells(rows: list[dict]) -> dict:
    counts = Counter(row["cell"] for row in rows)
    return {cell: {"count": counts.get(cell, 0), "share": round(counts.get(cell, 0) / max(len(rows), 1), 4)} for cell in CELLS}


def build_summary(ctx: StepContext, labeled: list[dict], built: dict, calls: list[dict], candidates: int) -> dict:
    by_id = {row["id"]: row for row in labeled}
    families = sorted({row["family"] for row in labeled})
    sampled = [c for c in calls if c.get("kind") == "sample" and not c.get("cache_hit") and not c.get("error_class")]
    sets = {}
    for name, entry in built["sets"].items():
        rows = [by_id[i] for i in entry["items"]]
        sets[name] = {
            "n_items": entry["n_items"],
            "train_tokens": entry["train_tokens"],
            "train_tokens_per_item": describe((r["train_tokens"] for r in rows), 1),
            "cells": {cell: sum(1 for r in rows if r["cell"] == cell) for cell in CELLS},
            "families": dict(Counter(r["family"] for r in rows).most_common()),
            **{k: entry[k] for k in ("token_budget", "short", "unchanged_answers") if k in entry},
        }
    reasoning_tokens = [r["reasoning_completion_tokens"] for r in labeled]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_id": ctx.config.get("model_id"),
        "seed": ctx.config.get("seed", 0),
        "generator_version": generators.GENERATOR_VERSION,
        "settings": {
            "fast": "a forced direct answer at effort 0.0, limited to 256 tokens",
            "reasoning": "effort 0.0, free-running, limited to 8,192 tokens",
        },
        "build": ctx.scratch.get("build_summary", {}),
        "candidates": candidates,
        "labeled": len(labeled),
        "cells": {
            "overall": _cells(labeled),
            "by_family": {f: _cells([r for r in labeled if r["family"] == f]) for f in families},
            "by_level": {f"L{level}": _cells([r for r in labeled if r["level"] == level]) for level in generators.LEVELS},
            "by_family_and_level": {
                f: {f"L{level}": {cell: sum(1 for r in labeled if r["family"] == f and r["level"] == level and r["cell"] == cell) for cell in CELLS} for level in generators.LEVELS}
                for f in families
            },
        },
        "accuracy": {
            "fast": round(sum(r["fast_right"] for r in labeled) / max(len(labeled), 1), 4),
            "reasoning": round(sum(r["reasoning_right"] for r in labeled) / max(len(labeled), 1), 4),
        },
        "tokens_per_call": {
            "prompt": describe((r["prompt_tokens"] for r in labeled), 1),
            "fast_completion": describe((r["fast_completion_tokens"] for r in labeled), 1),
            "reasoning_completion": describe(reasoning_tokens, 1),
            "reasoning_thinking": describe((r["reasoning_thinking_tokens"] for r in labeled), 1),
            "reasoning_cut_off": sum(1 for r in labeled if r["reasoning_cut_off"]),
            "train_tokens_per_item": describe((r["train_tokens"] for r in labeled), 1),
        },
        "sets": sets,
        "dev": {cell: len(ids) for cell, ids in built["dev"].items()},
        "prompt_examples": [{"id": i, "family": by_id[i]["family"], "level": by_id[i]["level"]} for i in built["prompt_examples"]],
        "available_after_holding_out": built["available"],
        "targets": {"middle": max(ctx.config.get("sizes", SIZES)), "easy": ctx.config.get("main_size", MAIN_SIZE), "hard": ctx.config.get("main_size", MAIN_SIZE)},
        "usage": {
            "api_calls": len(sampled),
            "prompt_tokens": sum(c["prompt_tokens"] for c in sampled),
            "completion_tokens": sum(c["completion_tokens"] for c in sampled),
            "thinking_tokens": sum(c["thinking_tokens"] for c in sampled),
            "failed_calls": sum(1 for c in calls if c.get("error_class")),
            "retries": sum(max(c.get("attempts", 1) - 1, 0) for c in calls),
            "throttled_calls": sum(1 for c in calls if c.get("throttle_events")),
        },
        "soak": soak_timeline(calls, []),
        "max_concurrency": ctx.config.get("client", {}).get("max_concurrency"),
        "canary": built["canary"],
    }


def figure_data(summary: dict) -> dict:
    cells = summary["cells"]
    return {
        "generated_at": summary["generated_at"],
        "four_cells": [{"cell": cell, **cells["overall"][cell]} for cell in CELLS],
        "by_family": [{"family": family, **{cell: row[cell]["count"] for cell in CELLS}} for family, row in cells["by_family"].items()],
        "by_level": [{"level": level, **{cell: row[cell]["count"] for cell in CELLS}} for level, row in cells["by_level"].items()],
        "sets": [{"set": name, "n_items": s["n_items"], "train_tokens": s["train_tokens"], **{f"from_{cell}": s["cells"][cell] for cell in CELLS}} for name, s in summary["sets"].items()],
    }


def render_report(summary: dict) -> str:
    lines: list[str] = []
    add = lines.append
    build, cells, tokens = summary["build"], summary["cells"], summary["tokens_per_call"]
    leak = build.get("leak_check", {})
    add("# Training data for the reasoning target")
    add("")
    add(
        f"Generated problems whose answers were fixed by a program before any model saw them, sorted by whether `{summary['model_id']}` "
        f"solves them in the fast setting and in the reasoning setting. Generated {summary['generated_at']} with generator version "
        f"{summary['generator_version']} and seed {summary['seed']}. None of these numbers is a headline result: scores come only from public benchmarks."
    )
    add("")
    add(f"- `fast` is {summary['settings']['fast']}.")
    add(f"- `reasoning` is {summary['settings']['reasoning']}.")

    add("")
    add("## Candidates and the leak check")
    add("")
    add(
        f"The generators produced {build.get('generated', 0):,} distinct problems in {len(build.get('by_family', {}))} families at five levels of difficulty, "
        f"after skipping {build.get('repeats_skipped', 0):,} repeats. Each was compared with the question text of {leak.get('references', 0):,} survey benchmark items: "
        f"an exact match, a match after normalizing case, punctuation and spacing, and a Jaccard similarity of character {leak.get('ngram')}-grams "
        f"of {leak.get('threshold')} or more with digits masked. The check dropped {sum(leak.get('dropped', {}).values())} "
        f"({leak.get('dropped', {}) or 'none of any kind'}), leaving {build.get('kept', 0):,}. "
        f"{leak.get('references_too_long_or_short_to_match', 0):,} benchmark texts are too long or too short to reach the threshold with any candidate, "
        f"and {leak.get('pairs_compared_in_full', 0):,} pairs shared enough rare n-grams to be compared in full."
    )
    add("")
    add("| family | level 1 | level 2 | level 3 | level 4 | level 5 |")
    add("|---|---|---|---|---|---|")
    for family, row in build.get("by_family", {}).items():
        add(f"| {family} | " + " | ".join(f"{row.get(f'L{level}', 0):,}" for level in generators.LEVELS) + " |")

    add("")
    add("## The four cells")
    add("")
    add(
        f"{summary['labeled']:,} of {summary['candidates']:,} candidates have a reply in both settings. "
        f"The fast setting got {summary['accuracy']['fast']:.1%} right and the reasoning setting {summary['accuracy']['reasoning']:.1%}."
    )
    add("")
    add("| | reasoning right | reasoning wrong |")
    add("|---|---|---|")
    o = cells["overall"]
    add(f"| fast right | easy: {o['easy']['count']:,} ({o['easy']['share']:.1%}) | rare: {o['rare']['count']:,} ({o['rare']['share']:.1%}) |")
    add(f"| fast wrong | middle band: {o['middle']['count']:,} ({o['middle']['share']:.1%}) | hard: {o['hard']['count']:,} ({o['hard']['share']:.1%}) |")
    add("")
    add("| family | items | easy | middle band | hard | rare |")
    add("|---|---|---|---|---|---|")
    for family, row in cells["by_family"].items():
        total = sum(row[cell]["count"] for cell in CELLS)
        add(f"| {family} | {total:,} | " + " | ".join(f"{row[cell]['count']:,} ({row[cell]['share']:.0%})" for cell in CELLS) + " |")
    add("")
    add("| level | items | easy | middle band | hard | rare |")
    add("|---|---|---|---|---|---|")
    for level, row in cells["by_level"].items():
        total = sum(row[cell]["count"] for cell in CELLS)
        add(f"| {level[1:]} | {total:,} | " + " | ".join(f"{row[cell]['count']:,} ({row[cell]['share']:.0%})" for cell in CELLS) + " |")
    want, have = summary["targets"], summary["available_after_holding_out"]
    short = [f"{cell} has {have[cell]:,} of the {want[cell]:,} wanted" for cell in want if have[cell] < want[cell]]
    add("")
    add(
        "After the dev items and the prompt-only examples are held out, the cells hold "
        + ", ".join(f"{have[cell]:,} {cell}" for cell in CELLS)
        + ". "
        + ("Short of its size: " + "; ".join(short) + "." if short else "Every cell reaches the size the arms need.")
    )

    add("")
    add("## Tokens per call")
    add("")
    add("| | mean | p50 | p90 | max |")
    add("|---|---|---|---|---|")
    for label, key in (("prompt tokens", "prompt"), ("fast reply tokens", "fast_completion"), ("reasoning reply tokens", "reasoning_completion"), ("of which thinking", "reasoning_thinking"), ("training tokens per item", "train_tokens_per_item")):
        d = tokens[key]
        add(f"| {label} | {d.get('mean', 0):,.0f} | {d.get('p50', 0):,.0f} | {d.get('p90', 0):,.0f} | {d.get('max', 0):,.0f} |")
    add("")
    add(f"{tokens['reasoning_cut_off']:,} reasoning replies reached the 8,192-token limit. A training item is the prompt followed by the answer alone, rendered for the fast setting.")

    add("")
    add("## Training sets")
    add("")
    add("Sets that are compared with each other are matched on training tokens. The middle-band sets are nested, and so are the random sets.")
    add("")
    add("| set | items | training tokens | easy | middle band | hard | rare | note |")
    add("|---|---|---|---|---|---|---|---|")
    for name, s in summary["sets"].items():
        note = []
        if s.get("short"):
            note.append("fewer items than asked for")
        if "unchanged_answers" in s:
            note.append(f"answers permuted among items, {s['unchanged_answers']} kept a right answer")
        if "token_budget" in s:
            note.append(f"matched to {s['token_budget']:,} tokens")
        add(f"| {name} | {s['n_items']:,} | {s['train_tokens']:,} | " + " | ".join(f"{s['cells'][cell]:,}" for cell in CELLS) + f" | {'; '.join(note)} |")
    add("")
    add(
        "Held out of every training set: a dev set of "
        + ", ".join(f"{n} {cell}" for cell, n in summary["dev"].items())
        + f" items for sanity checks, and {len(summary['prompt_examples'])} middle-band examples for the prompt-only arm "
        f"(from the families {', '.join(sorted({e['family'] for e in summary['prompt_examples']}))})."
    )

    add("")
    add("## Cost of this stage")
    add("")
    usage = summary["usage"]
    add(
        f"The filter made {usage['api_calls']:,} sampling calls with up to {summary.get('max_concurrency')} in flight: {usage['prompt_tokens']:,} prompt tokens and "
        f"{usage['completion_tokens']:,} completion tokens, of which {usage['thinking_tokens']:,} were thinking tokens. "
        f"{usage['failed_calls']} calls failed, {usage['retries']} were retried and {usage['throttled_calls']} saw a throttling event."
    )
    add("")
    add("| window start (UTC) | requests | req/min | completion tok/min | latency p50 (s) | p95 (s) |")
    add("|---|---|---|---|---|---|")
    for w in summary["soak"]:
        add(f"| {w['window_start'][5:16].replace('T', ' ')} | {w['requests']:,} | {w['requests_per_min']:,} | {w['completion_tokens_per_min']:,} | {w['latency_p50_s']} | {w['latency_p95_s']} |")

    add("")
    add("## Files")
    add("")
    add(
        "`items.jsonl.gz` holds every labeled item: the problem, the program's answer, the family, level and parameters, token counts, the cell, "
        "and what each setting did. `sets.json` holds the item ids of every training set, the dev sets, the prompt-only examples and the permuted answers. "
        "`manifest.json` lists a SHA-256 value for each file. The items can be regenerated from the seed with the `build-data` step. "
        f"Every file carries the marker `{summary['canary']}`."
    )
    add("")
    return "\n".join(lines)


def write_outputs(results_dir: Path, labeled: list[dict], built: dict, summary: dict) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "summary.json": json.dumps(summary, indent=1) + "\n",
        "figure-data.json": json.dumps(figure_data(summary), indent=1) + "\n",
        "REPORT.md": render_report(summary),
        "sets.json": json.dumps({k: built[k] for k in ("canary", "seed", "sets", "dev", "pilot", "prompt_examples")}) + "\n",
    }
    for name, text in files.items():
        (results_dir / name).write_text(text, encoding="utf-8")
    body = json.dumps({"canary": built["canary"]}) + "\n" + "".join(json.dumps(row) + "\n" for row in labeled)
    with open(results_dir / "items.jsonl.gz", "wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as handle:  # a fixed timestamp keeps the bytes the same on every build
            handle.write(body.encode("utf-8"))
    manifest = {
        "canary": built["canary"],
        "files": {name: {"sha256": hashlib.sha256((results_dir / name).read_bytes()).hexdigest(), "bytes": (results_dir / name).stat().st_size} for name in ("items.jsonl.gz", "sets.json")},
        "items": len(labeled),
    }
    (results_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
