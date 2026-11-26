"""Eval: an adapter in the fast setting on the survey benchmarks, against the base model.

A model state is the base model, an adapter, or the base model with worked examples
in the prompt. Every state answers the same items with the same prompt in the fast
setting. Each state is then compared with the base model on the same items: a paired
difference with a bootstrap interval and an exact McNemar test. The benchmarks a
state was trained for are its target cells and every other one is a watch cell, where
a fall of more than two points is reported as a regression.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from ..benchmarks import BENCHMARKS
from ..benchmarks import sources
from .base import StepContext, Unit
from .survey import Survey, _paired, _setting_summary, reference_subsample
from .train import REPO_ROOT, load_data

REGRESSION_POINTS = 2.0


class Eval(Survey):
    name = "eval"
    summary = "Rerun the survey benchmarks for every adapter in the fast setting, paired with the base model"
    default_config = "configs/eval.toml"

    def _root(self, ctx: StepContext) -> Path:
        return Path(ctx.config.get("survey_sources") or ctx.runs_dir.parent / "survey" / "sources")

    def states(self, ctx: StepContext) -> list[dict]:
        """The base model first, then whatever the config and the training results name."""
        states = [{"name": "base"}]
        states += [dict(s) for s in ctx.config.get("states", [])]
        runs_file = Path(ctx.config.get("runs_file") or REPO_ROOT / "results" / "train" / "runs.json")
        if ctx.config.get("use_training_runs", True) and runs_file.exists():
            wanted = ctx.config.get("runs")  # None means every run
            for run in json.loads(runs_file.read_text())["runs"]:
                if wanted is None or run["name"] in wanted:
                    states.append({"name": run["name"], "model_path": run["tinker_path"], "set": run["set"]})
        return states

    def _load(self, ctx: StepContext) -> dict[str, dict]:
        if "benchmarks" in ctx.scratch:
            return ctx.scratch["benchmarks"]
        seed = ctx.config.get("seed", 0)
        registry = ctx.config.get("_registry", BENCHMARKS)
        names = ctx.config.get("benchmarks", list(registry))
        if not ctx.config.get("_skip_download"):
            sources.ensure(self._root(ctx), {registry[n].source for n in names} | {"ifeval_checker", "nltk_data"}, out=ctx.out)
        sizes = ctx.config.get("items", {})
        loaded = {}
        for name in names:
            benchmark = registry[name]
            items = benchmark.load(self._root(ctx), seed)
            size = sizes.get(name, sizes.get("default", 500))
            keep = {i.id for i in items} if not size or size >= len(items) else reference_subsample(items, size, seed)
            loaded[name] = {"benchmark": benchmark, "items": {i.id: i for i in items if i.id in keep}, "reference": set()}
        ctx.scratch["benchmarks"] = loaded
        if any(s.get("few_shot") for s in self.states(ctx)):
            items, sets = load_data(ctx.config)
            ctx.scratch["few_shot"] = [
                message
                for item_id in sets["prompt_examples"]
                for message in ({"role": "user", "content": items[item_id]["prompt"]}, {"role": "assistant", "content": items[item_id]["answer"]})
            ]
        return loaded

    def plan(self, ctx: StepContext) -> list[Unit]:
        loaded = self._load(ctx)
        units = []
        for state in self.states(ctx):
            for name, entry in loaded.items():
                benchmark = entry["benchmark"]
                for item in entry["items"].values():
                    params = {"state": state["name"], "benchmark": name, "setting": "fast", "item": item.id, "prompt": item.fingerprint()}
                    if state.get("model_path"):
                        params["model_path"] = state["model_path"]
                    if state.get("few_shot"):
                        params["few_shot"] = True
                    calls = 12 if benchmark.multi_step else 1
                    chars = sum(len(m["content"]) for m in item.messages) or 40000 * calls
                    units.append(Unit(state["name"], params, calls=calls, prompt_tokens=int(chars / 3.5), completion_tokens=40 * calls, max_completion_tokens=benchmark.max_tokens_fast * calls))
        return units

    async def finish(self, ctx: StepContext, rows: list[dict]) -> None:
        loaded = self._load(ctx)
        states = self.states(ctx)
        seed = ctx.config.get("seed", 0)
        targets = set(ctx.config.get("target_benchmarks", ["gsm8k", "mbpp"]))
        graded: dict[str, dict] = {}
        for state in states:
            mine = [row for row in rows if row.get("state") == state["name"]]
            grades = await self._grade(ctx, mine)
            graded[state["name"]] = {(row["benchmark"], row["item"]): (row, grades[(row["benchmark"], "fast", row["item"])]) for row in mine if (row["benchmark"], "fast", row["item"]) in grades}
        summary = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "model_id": ctx.config.get("model_id"), "setting": "fast", "regression_points": REGRESSION_POINTS, "states": {}}
        for state in states:
            cells = {}
            for name, entry in loaded.items():
                benchmark, items = entry["benchmark"], entry["items"]
                have = {item_id: graded[state["name"]].get((name, item_id)) for item_id in items}
                ids = [i for i in items if have[i]]
                if not ids:
                    continue
                cell = _setting_summary(benchmark, [items[i] for i in ids], [have[i][0] for i in ids], [have[i][1] for i in ids], seed)
                cell["complete"] = len(ids) == len(items)
                cell["role"] = "target" if name in targets else "watch"
                if state["name"] != "base":
                    base = {i: graded["base"][(name, i)][1] for i in ids if (name, i) in graded["base"]}
                    paired = _paired(benchmark, items, base, {i: have[i][1] for i in ids}, seed)
                    cell["minus_base"] = paired
                    if paired and paired["estimate"] is not None:
                        cell["regression"] = cell["role"] == "watch" and paired["estimate"] * 100 < -REGRESSION_POINTS
                cells[name] = cell
            summary["states"][state["name"]] = {**{k: v for k, v in state.items() if k != "name"}, "benchmarks": cells}
        ctx.results_dir.mkdir(parents=True, exist_ok=True)
        (ctx.results_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
        (ctx.results_dir / "REPORT.md").write_text(render_report(summary, {n: e["benchmark"].title for n, e in loaded.items()}), encoding="utf-8")
        ctx.out(f"Wrote {ctx.results_dir / 'summary.json'} and REPORT.md")


def render_report(summary: dict, titles: dict[str, str]) -> str:
    lines = ["# Evaluation of adapters", ""]
    lines.append(
        f"Every model state answered the same items with the same prompt in the fast setting (a forced direct answer at effort 0.0). "
        f"Each cell shows the score in percent and, under it in the same cell, the paired difference from the base model in points with its 95% interval. "
        f"A watch cell that falls by more than {summary['regression_points']:g} points is marked as a regression. Generated {summary['generated_at']}."
    )
    names = [n for n in titles if any(n in s["benchmarks"] for s in summary["states"].values())]
    lines += ["", "| state | " + " | ".join(titles[n] for n in names) + " |", "|---|" + "---|" * len(names)]
    for state, entry in summary["states"].items():
        cells = []
        for name in names:
            cell = entry["benchmarks"].get(name)
            if not cell:
                cells.append("")
                continue
            text = f"{cell['score']['estimate'] * 100:.1f}"
            paired = cell.get("minus_base")
            if paired and paired["estimate"] is not None:
                text += f"; {paired['estimate'] * 100:+.1f} ({paired['low'] * 100:+.1f} to {paired['high'] * 100:+.1f})"
                text += " regression" if cell.get("regression") else ""
            cells.append(text + (" (target)" if cell["role"] == "target" else ""))
        lines.append(f"| {state} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
