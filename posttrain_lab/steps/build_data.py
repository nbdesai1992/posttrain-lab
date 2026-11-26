"""Build-data: generate candidate training problems and check them for leaks.

No model is called. Each unit generates the problems of one family and level from the
seed, and the end of the run compares every candidate with every survey benchmark
item and writes the survivors to ``runs/build-data/candidates.jsonl``.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from ..benchmarks import BENCHMARKS
from ..data import generators
from ..data.leak import NGRAM, THRESHOLD, leak_check, normalize
from ..render import FAST_EFFORT
from ..stats import describe
from ..store import read_jsonl
from .base import Step, StepContext, Unit


class BuildData(Step):
    name = "build-data"
    summary = "Generate synthetic training items whose labels come from programs, and check them for leaks"
    default_config = "configs/build-data.toml"

    def plan(self, ctx: StepContext) -> list[Unit]:
        seed = ctx.config.get("seed", 0)
        units = []
        for family, counts in ctx.config.get("counts", {}).items():
            for level, count in zip(generators.LEVELS, counts):
                if count > 0:
                    units.append(
                        Unit(
                            family,
                            {"family": family, "level": level, "count": count, "seed": seed, "generator_version": generators.GENERATOR_VERSION},
                            label=f"level {level}: {count}",
                        )
                    )
        ctx.scratch["units"] = units
        return units

    def _file(self, ctx: StepContext, params: dict) -> Path:
        return ctx.runs_dir / "candidates" / f"{params['family']}-L{params['level']}.jsonl"

    async def run_unit(self, unit: Unit, ctx: StepContext) -> dict:
        p = unit.params
        items, seen, index, skipped = [], set(), 0, 0
        while len(items) < p["count"] and index < p["count"] * 4:
            item = generators.make_item(p["family"], p["level"], index, p["seed"])
            index += 1
            key = normalize(item["question"])
            if key in seen:
                skipped += 1  # the same problem came up again; keep the first
                continue
            seen.add(key)
            messages = [{"role": "user", "content": item["prompt"]}]
            # Tokens of the fast-setting prompt (with the forced answer opening) and of the
            # training example: the same prompt followed by the answer and nothing else.
            item["prompt_tokens"] = len(ctx.renderer.build_prompt(messages, FAST_EFFORT)) + 2
            item["train_tokens"] = len(ctx.renderer.build_example([*messages, {"role": "assistant", "content": item["answer"]}], FAST_EFFORT)[0])
            items.append(item)
        path = self._file(ctx, p)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(item) + "\n" for item in items), encoding="utf-8")
        return {"generated": len(items), "repeats_skipped": skipped, "short_by": p["count"] - len(items)}

    def references(self, ctx: StepContext) -> list[tuple[str, str]]:
        """The question text of every survey benchmark item."""
        root = Path(ctx.config.get("survey_sources") or ctx.runs_dir.parent / "survey" / "sources")
        registry = ctx.config.get("_registry", BENCHMARKS)
        seen: dict[str, str] = {}
        for name, benchmark in registry.items():
            for item in benchmark.load(root, ctx.config.get("survey_seed", ctx.config.get("seed", 0))):
                seen.setdefault(item.leak_text(), f"{name}:{item.id}")
        return [(label, text) for text, label in seen.items()]

    async def finish(self, ctx: StepContext, rows: list[dict]) -> None:
        candidates, seen = [], set()
        for unit in ctx.scratch["units"]:
            for item in read_jsonl(self._file(ctx, unit.params)):
                if normalize(item["question"]) not in seen:
                    seen.add(normalize(item["question"]))
                    candidates.append(item)
        references = self.references(ctx)
        kept, dropped, stats = leak_check(candidates, references, n=ctx.config.get("ngram", NGRAM), threshold=ctx.config.get("threshold", THRESHOLD))
        (ctx.runs_dir / "candidates.jsonl").write_text("".join(json.dumps(item) + "\n" for item in kept), encoding="utf-8")
        (ctx.runs_dir / "dropped.jsonl").write_text("".join(json.dumps(item) + "\n" for item in dropped), encoding="utf-8")
        by_cell = Counter((item["family"], item["level"]) for item in kept)
        by_id = {item["id"]: item for item in candidates}
        templates: dict[tuple, dict] = {}
        for row in dropped:  # only the benchmark item's id is recorded, never its text
            item = by_id[row["id"]]
            entry = templates.setdefault(
                (item["family"], item["level"], row["reason"], row.get("reference")),
                {"family": item["family"], "level": item["level"], "reason": row["reason"], "nearest_benchmark_item": row.get("reference"), "count": 0, "similarities": [], "example": item["question"]},
            )
            entry["count"] += 1
            entry["similarities"].append(row.get("similarity", 1.0))
        for entry in templates.values():
            values = entry.pop("similarities")
            entry["similarity_low"], entry["similarity_high"] = min(values), max(values)
        stats["dropped_by_template"] = sorted(templates.values(), key=lambda e: (e["family"], e["level"]))
        stats["references_by_benchmark"] = dict(Counter(label.split(":", 1)[0] for label, _ in references))
        summary = {
            "seed": ctx.config.get("seed", 0),
            "generator_version": generators.GENERATOR_VERSION,
            "generated": len(candidates),
            "repeats_skipped": sum(row.get("repeats_skipped", 0) for row in rows),
            "leak_check": stats,
            "kept": len(kept),
            "requested": {family: list(counts) for family, counts in ctx.config.get("counts", {}).items()},
            "by_family": {family: {f"L{level}": by_cell.get((family, level), 0) for level in generators.LEVELS} for family in ctx.config.get("counts", {})},
            "prompt_tokens": describe((item["prompt_tokens"] for item in kept), 1),
            "train_tokens": describe((item["train_tokens"] for item in kept), 1),
        }
        (ctx.runs_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
        ctx.out(
            f"{len(candidates):,} candidates, {len(dropped)} dropped by the leak check against {stats['references']:,} benchmark texts "
            f"({stats['dropped']}), {len(kept):,} kept in {ctx.runs_dir / 'candidates.jsonl'}"
        )
