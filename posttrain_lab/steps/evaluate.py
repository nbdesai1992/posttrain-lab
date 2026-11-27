"""Eval: every adapter on the survey benchmarks, against the base model.

A model state is the base model, an adapter, or the base model with worked examples
in the prompt, in one setting. Every state answers the same items with the same
prompt. Each fast-setting state is compared with the base model on the same items: a
paired difference with a bootstrap interval and an exact McNemar test. The benchmarks
a state was trained for are its target cells and every other one is a watch cell,
where a fall of more than two points is reported as a regression.

The step can run while adapters are still being trained. It evaluates whatever
adapters the training summary lists, and with ``--no-report`` it stops there. Without
it, it also writes the comparisons between arms and applies the decision rules.
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path

from ..benchmarks import BENCHMARKS
from ..benchmarks import sources
from ..stats import paired_difference, percentile
from ..store import read_jsonl
from .base import StepContext, Unit
from .survey import SETTING_LABELS, SETTINGS, Survey, _paired, _setting_summary, reference_subsample
from .train import REPO_ROOT, load_data, write_atomic

REGRESSION_POINTS = 2.0
ALPHA = 0.05
SIZES = (250, 500, 1000, 2000, 4000)
CONTROLS = {
    "easy-2000": "easy only",
    "hard-2000": "hard only",
    "shuffled-labels-2000": "shuffled labels",
    "length-matched-random-2000": "length-matched random",
}


class Eval(Survey):
    name = "eval"
    summary = "Rerun the survey benchmarks for every adapter, paired with the base model"
    default_config = "configs/eval.toml"

    def _root(self, ctx: StepContext) -> Path:
        return Path(ctx.config.get("survey_sources") or ctx.runs_dir.parent / "survey" / "sources")

    def _runs(self, ctx: StepContext) -> list[dict]:
        """The finished training runs, from the training summary."""
        if not ctx.config.get("use_training_runs", True):
            return []
        runs_file = Path(ctx.config.get("runs_file") or REPO_ROOT / "results" / "train" / "summary.json")
        if not runs_file.exists():
            return []
        wanted = ctx.config.get("runs")  # None means every run
        return [run for run in json.loads(runs_file.read_text())["runs"] if wanted is None or run["name"] in wanted]

    def states(self, ctx: StepContext) -> list[dict]:
        """The base model, then every trained adapter, then the states the config names.

        A state the config builds on an adapter that is not trained yet is left out
        until it is.
        """
        if "states" in ctx.scratch:
            return ctx.scratch["states"]
        runs = self._runs(ctx)
        by_name = {run["name"]: run for run in runs}
        states = [{"name": "base", "setting": "fast"}]
        states += [{"name": run["name"], "setting": "fast", "model_path": run["tinker_path"], "set": run.get("set"), "seed": run.get("seed")} for run in runs]
        for extra in ctx.config.get("states", []):
            state = {"setting": "fast", **extra}
            if state["setting"] not in SETTINGS:
                raise ValueError(f"State {state['name']!r} names an unknown setting {state['setting']!r}")
            if "adapter" in state:
                if state["adapter"] not in by_name:
                    continue
                state["model_path"] = by_name[state["adapter"]]["tinker_path"]
            states.append(state)
        if len({state["name"] for state in states}) != len(states):
            raise ValueError("Two model states have the same name")
        ctx.scratch["states"] = states
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
            fast = state["setting"] == "fast"
            for name, entry in loaded.items():
                benchmark = entry["benchmark"]
                for item in entry["items"].values():
                    params = {"state": state["name"], "benchmark": name, "setting": state["setting"], "item": item.id, "prompt": item.fingerprint()}
                    if state.get("model_path"):
                        params["model_path"] = state["model_path"]
                    if state.get("few_shot"):
                        params["few_shot"] = True
                    calls = 12 if benchmark.multi_step else 1
                    chars = sum(len(m["content"]) for m in item.messages) or 40000 * calls
                    limit = benchmark.max_tokens_fast if fast else benchmark.max_tokens_slow
                    units.append(Unit(state["name"], params, calls=calls, prompt_tokens=int(chars / 3.5), completion_tokens=(40 if fast else 1000) * calls, max_completion_tokens=limit * calls))
        ctx.scratch["planned"] = units
        return units

    async def run_unit(self, unit: Unit, ctx: StepContext) -> dict:
        if not ctx.scratch.get("flagged"):  # tells the train step that adapters are being evaluated
            ctx.runs_dir.mkdir(parents=True, exist_ok=True)
            (ctx.runs_dir / "ACTIVE").write_text(datetime.now(timezone.utc).isoformat(timespec="seconds") + "\n", encoding="utf-8")
            ctx.scratch["flagged"] = True
        return await super().run_unit(unit, ctx)

    def _status(self, ctx: StepContext, rows: list[dict]) -> dict:
        planned, done = {}, {}
        for unit in ctx.scratch.get("planned", []):
            planned[unit.kind] = planned.get(unit.kind, 0) + 1
        for row in rows:
            done[row.get("state")] = done.get(row.get("state"), 0) + 1
        return {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "states": {name: {"done": done.get(name, 0), "of": count} for name, count in planned.items()}}

    async def finish(self, ctx: StepContext, rows: list[dict]) -> None:
        (ctx.runs_dir / "ACTIVE").unlink(missing_ok=True)
        status = self._status(ctx, rows)
        write_atomic(ctx.runs_dir / "status.json", json.dumps(status, indent=1) + "\n")
        if ctx.config.get("no_report"):
            complete = sum(1 for state in status["states"].values() if state["done"] >= state["of"])
            ctx.out(f"{complete} of {len(status['states'])} states are complete. No report was written.")
            return
        loaded = self._load(ctx)
        states = self.states(ctx)
        seed = ctx.config.get("seed", 0)
        data: dict[str, dict[str, dict]] = {}
        for state in states:
            mine = [row for row in rows if row.get("state") == state["name"]]
            grades = await self._grade(ctx, mine)
            cells: dict[str, dict] = {}
            for row in mine:
                key = (row["benchmark"], state["setting"], row["item"])
                if key in grades and row["item"] in loaded[row["benchmark"]]["items"]:
                    cell = cells.setdefault(row["benchmark"], {"rows": {}, "grades": {}})
                    cell["rows"][row["item"]], cell["grades"][row["item"]] = row, grades[key]
            data[state["name"]] = cells
        calls = list(read_jsonl(ctx.runs_dir / "calls.jsonl"))
        summary = build_summary(ctx.config, states, loaded, data, calls, seed)
        write_atomic(ctx.results_dir / "summary.json", json.dumps(summary, indent=1) + "\n")
        write_atomic(ctx.results_dir / "figure-data.json", json.dumps(figure_data(summary), indent=1) + "\n")
        write_atomic(ctx.results_dir / "REPORT.md", render_report(summary))
        ctx.out(f"Wrote summary.json, figure-data.json and REPORT.md to {ctx.results_dir}")


# ------------------------------------------------------------------ analysis


def share_of_gap(fast: list[float], reference: list[float], state: list[float], seed: int, resamples: int = 2000) -> dict | None:
    """How much of the way from the base model's fast score to its reference score a
    state's score goes, with a percentile bootstrap over items."""
    n = len(fast)
    gap = sum(reference) - sum(fast)
    if not n or gap <= 0:
        return None
    rng = random.Random(seed)
    draws = []
    for _ in range(resamples):
        picks = rng.choices(range(n), k=n)
        low = sum(fast[i] for i in picks)
        width = sum(reference[i] for i in picks) - low
        if width > 0:
            draws.append((sum(state[i] for i in picks) - low) / width)
    return {
        "estimate": round((sum(state) - sum(fast)) / gap, 4),
        "low": round(percentile(draws, 2.5), 4),
        "high": round(percentile(draws, 97.5), 4),
    }


def includes_zero(difference: dict | None) -> bool | None:
    if not difference or difference.get("estimate") is None:
        return None
    return difference["low"] <= 0 <= difference["high"]


def build_summary(config: dict, states: list[dict], loaded: dict, data: dict, calls: list[dict], seed: int) -> dict:
    targets = [name for name in config.get("target_benchmarks", ["gsm8k", "mbpp"]) if name in loaded]
    main = config.get("main_adapter", "middle-2000-s1")
    titles = {name: entry["benchmark"].title for name, entry in loaded.items()}
    by_name = {state["name"]: state for state in states}

    def scores(state: str, bench: str) -> dict[str, dict]:
        return data.get(state, {}).get(bench, {}).get("grades", {})

    def paired(first: str, second: str, bench: str) -> dict | None:
        """Second minus first on the items both answered."""
        if not scores(first, bench) or not scores(second, bench):
            return None
        return _paired(loaded[bench]["benchmark"], loaded[bench]["items"], scores(first, bench), scores(second, bench), seed)

    def pooled(first: list[str], second: list[str], bench: str) -> dict | None:
        """The mean over the second group of states minus the mean over the first, item by item."""
        first = [s for s in first if scores(s, bench)]
        second = [s for s in second if scores(s, bench)]
        if not first or not second:
            return None
        items = loaded[bench]["items"]
        common = [i for i in items if all(i in scores(s, bench) for s in first + second)]
        if not common:
            return None

        def mean(group: list[str], item: str) -> float:
            return sum(scores(s, bench)[item]["score"] for s in group) / len(group)

        result = paired_difference(
            [mean(first, i) for i in common], [mean(second, i) for i in common],
            clusters=[items[i].cluster for i in common], groups=[items[i].group for i in common], macro=loaded[bench]["benchmark"].macro, seed=seed,
        )
        return {key: result[key] for key in ("n", "estimate", "low", "high") if key in result} | {"first": first, "second": second}

    # The fast-to-reference gap of the base model on the same items, for the targets.
    reference_state = next((s["name"] for s in states if s["setting"] == "reasoning" and not s.get("model_path") and not s.get("few_shot")), None)
    gaps = {}
    for bench in targets:
        if reference_state and scores("base", bench) and scores(reference_state, bench):
            difference = paired("base", reference_state, bench)
            common = [i for i in scores("base", bench) if i in scores(reference_state, bench)]
            gaps[bench] = {
                "base_fast": round(sum(scores("base", bench)[i]["score"] for i in common) / len(common), 4),
                "base_reasoning": round(sum(scores(reference_state, bench)[i]["score"] for i in common) / len(common), 4),
                "gap": difference,
            }

    out_states = {}
    for state in states:
        cells = {}
        for bench, entry in loaded.items():
            cell_data = data.get(state["name"], {}).get(bench)
            if not cell_data:
                continue
            benchmark, items = entry["benchmark"], entry["items"]
            ids = [i for i in items if i in cell_data["grades"]]
            cell = _setting_summary(benchmark, [items[i] for i in ids], [cell_data["rows"][i] for i in ids], [cell_data["grades"][i] for i in ids], seed)
            cell["complete"] = len(ids) == len(items)
            cell["role"] = "target" if bench in targets else "watch"
            if state["setting"] == "fast" and state["name"] != "base":
                difference = paired("base", state["name"], bench)
                cell["minus_base"] = difference
                if difference and difference["estimate"] is not None:
                    cell["regression"] = cell["role"] == "watch" and difference["estimate"] * 100 < -REGRESSION_POINTS
                if bench in gaps:
                    common = [i for i in ids if i in scores("base", bench) and i in scores(reference_state, bench)]
                    cell["gap_closed"] = share_of_gap(
                        [scores("base", bench)[i]["score"] for i in common], [scores(reference_state, bench)[i]["score"] for i in common],
                        [cell_data["grades"][i]["score"] for i in common], seed,
                    )
            cells[bench] = cell
        out_states[state["name"]] = {**{k: v for k, v in state.items() if k != "name"}, "benchmarks": cells}

    def score_of(state: str, bench: str) -> dict | None:
        cell = out_states.get(state, {}).get("benchmarks", {}).get(bench)
        return cell["score"] if cell else None

    # The main comparison: middle band against random at 2,000, pooled over seeds.
    names = {s["name"] for s in states if s["setting"] == "fast"}
    seeds = [k for k in range(1, 10) if f"middle-2000-s{k}" in names and f"random-2000-s{k}" in names]
    comparison = {}
    for bench in targets:
        middle, rand = [f"middle-2000-s{k}" for k in seeds], [f"random-2000-s{k}" for k in seeds]
        per_seed = []
        for k in seeds:
            difference = paired(f"random-2000-s{k}", f"middle-2000-s{k}", bench)
            per_seed.append({"seed": k, "middle": (score_of(f"middle-2000-s{k}", bench) or {}).get("estimate"), "random": (score_of(f"random-2000-s{k}", bench) or {}).get("estimate"), "middle_minus_random": difference})
        estimates = [row["middle_minus_random"]["estimate"] for row in per_seed if row["middle_minus_random"]]
        controls = {}
        for control, label in CONTROLS.items():
            controls[control] = {"label": label, "middle_minus_control": paired(control, main, bench), "control_score": score_of(control, bench), "middle_score": score_of(main, bench)}
        comparison[bench] = {
            "seeds": seeds,
            "middle_minus_random": pooled(rand, middle, bench),
            "per_seed": per_seed,
            "seed_range": {"low": min(estimates), "high": max(estimates)} if estimates else None,
            "middle_scores": [row["middle"] for row in per_seed],
            "random_scores": [row["random"] for row in per_seed],
            "controls": controls,
        }

    # The size sweep, as a curve for each arm.
    sweep = {}
    for bench in targets:
        sweep[bench] = {}
        for arm in ("middle", "random"):
            points = []
            for size in SIZES:
                name = f"{arm}-{size}-s1" if size == 2000 else f"{arm}-{size}"
                cell = out_states.get(name, {}).get("benchmarks", {}).get(bench)
                if cell:
                    points.append({"size": size, "state": name, "score": cell["score"], "minus_base": cell.get("minus_base"), "gap_closed": cell.get("gap_closed")})
            sweep[bench][arm] = points

    # The prompt-only state against the base model and against the main adapter.
    prompt_state = next((s["name"] for s in states if s.get("few_shot") and s["setting"] == "fast"), None)
    prompt_only = {}
    if prompt_state:
        for bench in loaded:
            if score_of(prompt_state, bench):
                prompt_only[bench] = {
                    "state": prompt_state, "score": score_of(prompt_state, bench), "base_score": score_of("base", bench), "minus_base": paired("base", prompt_state, bench),
                    "main_score": score_of(main, bench), "main_minus_prompt_only": paired(prompt_state, main, bench),
                }

    # Every watch cell under the main adapter.
    watch = []
    for bench in loaded:
        cell = out_states.get(main, {}).get("benchmarks", {}).get(bench)
        if cell and cell["role"] == "watch":
            watch.append({"benchmark": bench, "before": score_of("base", bench), "after": cell["score"], "difference": cell.get("minus_base"), "regression": bool(cell.get("regression"))})

    # The reasoning setting: the base model against the main adapter, when both were run.
    slow_state = next((s["name"] for s in states if s["setting"] == "reasoning" and s.get("adapter") == main), None)
    slow = {}
    if slow_state and reference_state:
        for bench in loaded:
            before, after = out_states.get(reference_state, {}).get("benchmarks", {}).get(bench), out_states.get(slow_state, {}).get("benchmarks", {}).get(bench)
            if before and after:
                slow[bench] = {
                    "base": before["score"], "adapter": after["score"], "difference": paired(reference_state, slow_state, bench), "complete": after["complete"],
                    "base_thinking_tokens": before["tokens_per_answer"]["thinking"].get("mean"), "adapter_thinking_tokens": after["tokens_per_answer"]["thinking"].get("mean"),
                    "base_cut_off": before["cut_off_by_max_tokens"], "adapter_cut_off": after["cut_off_by_max_tokens"], "n": after["n"],
                }

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_id": config.get("model_id"),
        "settings": {name: SETTING_LABELS.get(name, name) for name in sorted({s["setting"] for s in states})},
        "regression_points": REGRESSION_POINTS,
        "alpha": ALPHA,
        "main_adapter": main,
        "targets": targets,
        "titles": titles,
        "items": {name: len(entry["items"]) for name, entry in loaded.items()},
        "states": out_states,
        "base_gap": gaps,
        "main_comparison": comparison,
        "size_sweep": sweep,
        "prompt_only": prompt_only,
        "watch_cells": watch,
        "reasoning_pass": {"base_state": reference_state, "adapter_state": slow_state, "benchmarks": slow},
        "usage": usage(calls, states),
    }
    summary["decisions"] = decide(summary)
    summary["missing"] = missing(config, summary)
    return summary


def usage(calls: list[dict], states: list[dict]) -> dict:
    """Calls and tokens of the evaluation, in all and by adapter."""
    sampled = [c for c in calls if c.get("kind") == "sample" and not c.get("error_class")]
    made = [c for c in sampled if not c.get("cache_hit")]
    by_path = {state.get("model_path"): state["name"] for state in states if state.get("model_path") and state["setting"] == "fast"}
    per_adapter = {}
    for call in made:
        name = by_path.get(call.get("model_path"))
        if name and call.get("tag", "").endswith(":fast"):
            entry = per_adapter.setdefault(name, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "first": call["started_at"], "last": call["ended_at"]})
            entry["calls"] += 1
            entry["prompt_tokens"] += call["prompt_tokens"]
            entry["completion_tokens"] += call["completion_tokens"]
            entry["first"], entry["last"] = min(entry["first"], call["started_at"]), max(entry["last"], call["ended_at"])
    for entry in per_adapter.values():
        entry["minutes"] = round((entry.pop("last") - entry.pop("first")) / 60, 1)
    return {
        "api_calls": len(made),
        "cache_hits": len(sampled) - len(made),
        "prompt_tokens": sum(c["prompt_tokens"] for c in made),
        "completion_tokens": sum(c["completion_tokens"] for c in made),
        "failed_calls": sum(1 for c in calls if c.get("error_class")),
        "retries": sum(max(c.get("attempts", 1) - 1, 0) for c in calls),
        "throttled_calls": sum(1 for c in calls if c.get("throttle_events")),
        "fast_pass_by_adapter": per_adapter,
    }


def decide(summary: dict) -> list[dict]:
    """The decision rules, applied as written. One entry per rule and benchmark."""
    out = []
    main, titles = summary["main_adapter"], summary["titles"]
    main_cells = summary["states"].get(main, {}).get("benchmarks", {})

    def points(difference: dict) -> str:
        return f"{difference['estimate'] * 100:+.1f} points ({difference['low'] * 100:+.1f} to {difference['high'] * 100:+.1f})"

    def add(rule: str, bench: str | None, outcome: str, line: str) -> None:
        out.append({"rule": rule, "benchmark": bench, "outcome": outcome, "line": line})

    for bench in summary["targets"]:
        title = titles[bench]
        # P2, first part: the main adapter raises the fast score.
        difference = main_cells.get(bench, {}).get("minus_base")
        if not difference or difference["estimate"] is None:
            add("P2", bench, "not decided", f"P2 on {title}: not decided, the main adapter has no score.")
        else:
            rises, separated = difference["estimate"] > 0, difference["mcnemar_p"] < ALPHA
            outcome = "holds" if rises and separated else "wrong"
            reason = "the score rises and McNemar's test separates before from after" if outcome == "holds" else ("the score does not rise" if not rises else "McNemar's test does not separate before from after")
            add("P2", bench, outcome, f"P2 on {title}: {outcome}. The main adapter changes the fast score by {points(difference)}, McNemar p = {difference['mcnemar_p']:.4g}; {reason}.")
        # P2, second part: the adapter beats eight examples in the prompt.
        versus = summary["prompt_only"].get(bench, {}).get("main_minus_prompt_only")
        if not versus or versus["estimate"] is None:
            add("P2, against prompt-only", bench, "not decided", f"P2 against the prompt-only state on {title}: not decided, one of the two has no score.")
        else:
            outcome = "holds" if versus["estimate"] > 0 and versus["mcnemar_p"] < ALPHA else "wrong"
            add("P2, against prompt-only", bench, outcome, f"P2 against the prompt-only state on {title}: {outcome}. The main adapter minus eight examples in the prompt is {points(versus)}, McNemar p = {versus['mcnemar_p']:.4g}.")
        # P3: middle band against random, pooled over seeds; then against easy only and hard only.
        comparison = summary["main_comparison"].get(bench, {})
        pooled = comparison.get("middle_minus_random")
        if not pooled or pooled.get("estimate") is None:
            add("P3", bench, "not decided", f"P3 on {title}: not decided, the two arms have no scores.")
        else:
            outcome = "wrong" if includes_zero(pooled) or pooled["estimate"] < 0 else "holds"
            why = "the interval includes zero" if includes_zero(pooled) else ("random is ahead" if pooled["estimate"] < 0 else "the interval is above zero")
            spread = comparison.get("seed_range")
            beside = f", seed by seed from {spread['low'] * 100:+.1f} to {spread['high'] * 100:+.1f}" if spread else ""
            add("P3", bench, outcome, f"P3 on {title}: {outcome}. Middle band minus random over {len(comparison['seeds'])} seeds is {points(pooled)}{beside}; {why}.")
        for control in ("easy-2000", "hard-2000"):
            entry = comparison.get("controls", {}).get(control, {})
            difference = entry.get("middle_minus_control")
            label = entry.get("label", control)
            if not difference or difference["estimate"] is None:
                add(f"P3, against {label}", bench, "not decided", f"P3 against {label} on {title}: not decided, one of the two has no score.")
            else:
                against = includes_zero(difference)
                outcome = "counts against" if against else ("supports" if difference["estimate"] > 0 else "counts against")
                why = "the interval includes zero" if against else ("the middle band is ahead" if difference["estimate"] > 0 else f"{label} is ahead")
                add(f"P3, against {label}", bench, outcome, f"P3 against {label} on {title}: {outcome}. Middle band minus {label} is {points(difference)}; {why}.")
        # P4: real labels against shuffled labels.
        difference = comparison.get("controls", {}).get("shuffled-labels-2000", {}).get("middle_minus_control")
        if not difference or difference["estimate"] is None:
            add("P4", bench, "not decided", f"P4 on {title}: not decided, one of the two has no score.")
        elif includes_zero(difference):
            add("P4", bench, "the two match", f"P4 on {title}: real labels minus shuffled labels is {points(difference)}. The interval includes zero, so the two match and the gain is the model getting used to the format.")
        else:
            ahead = "real labels are ahead, so the gain is more than the format" if difference["estimate"] > 0 else "shuffled labels are ahead"
            add("P4", bench, "the two differ", f"P4 on {title}: real labels minus shuffled labels is {points(difference)}. The interval excludes zero: {ahead}.")
    # P5: no watch cell falls by more than two points under the main adapter.
    if not summary["watch_cells"]:
        add("P5", None, "not decided", "P5: not decided, the main adapter has no watch cells.")
    else:
        fallen = [cell for cell in summary["watch_cells"] if cell["regression"]]
        if fallen:
            listed = "; ".join(f"{titles[cell['benchmark']]} {points(cell['difference'])}" for cell in fallen)
            add("P5", None, "wrong", f"P5: wrong. {len(fallen)} of {len(summary['watch_cells'])} watch cells fall by more than {REGRESSION_POINTS:g} points under the main adapter: {listed}.")
        else:
            add("P5", None, "holds", f"P5: holds. None of the {len(summary['watch_cells'])} watch cells falls by more than {REGRESSION_POINTS:g} points under the main adapter.")
    return out


def missing(config: dict, summary: dict) -> list[str]:
    """States the plan names that have no complete set of replies."""
    wanted = list(config.get("expected_states", []))
    out = []
    for name in wanted:
        cells = summary["states"].get(name, {}).get("benchmarks", {})
        if not cells:
            out.append(f"{name}: not evaluated")
        elif len(cells) < len(summary["items"]) or not all(cell["complete"] for cell in cells.values()):
            out.append(f"{name}: incomplete")
    return out


# ------------------------------------------------------------------ outputs


def pct(score: dict | None) -> str:
    return "" if not score or score.get("estimate") is None else f"{score['estimate'] * 100:.1f} ({score['low'] * 100:.1f} to {score['high'] * 100:.1f})"


def pts(difference: dict | None) -> str:
    return "" if not difference or difference.get("estimate") is None else f"{difference['estimate'] * 100:+.1f} ({difference['low'] * 100:+.1f} to {difference['high'] * 100:+.1f})"


def pvalue(difference: dict | None) -> str:
    return "" if not difference or difference.get("mcnemar_p") is None else (f"{difference['mcnemar_p']:.3f}" if difference["mcnemar_p"] >= 0.001 else "<0.001")


def share(value: dict | None) -> str:
    return "" if not value else f"{value['estimate']:.0%} ({value['low']:.0%} to {value['high']:.0%})"


def figure_data(summary: dict) -> dict:
    targets = summary["targets"]
    return {
        "generated_at": summary["generated_at"],
        "main_adapter": summary["main_adapter"],
        "base_gap": {bench: {"base_fast": gap["base_fast"], "base_reasoning": gap["base_reasoning"]} for bench, gap in summary["base_gap"].items()},
        "targets_by_state": [
            {"state": name, "setting": state["setting"], "set": state.get("set"), "seed": state.get("seed"),
             **{bench: {"score": cell["score"]["estimate"], "low": cell["score"]["low"], "high": cell["score"]["high"],
                        "minus_base": (cell.get("minus_base") or {}).get("estimate"), "gap_closed": (cell.get("gap_closed") or {}).get("estimate")}
                for bench in targets for cell in [state["benchmarks"].get(bench)] if cell}}
            for name, state in summary["states"].items() if state["setting"] == "fast"
        ],
        "main_comparison": {
            bench: {
                "middle_minus_random": entry["middle_minus_random"], "seed_range": entry["seed_range"], "middle_scores": entry["middle_scores"], "random_scores": entry["random_scores"],
                "controls": {name: control["middle_minus_control"] for name, control in entry["controls"].items()},
            }
            for bench, entry in summary["main_comparison"].items()
        },
        "size_sweep": {
            bench: {arm: [{"size": point["size"], "score": point["score"]["estimate"], "low": point["score"]["low"], "high": point["score"]["high"]} for point in points] for arm, points in arms.items()}
            for bench, arms in summary["size_sweep"].items()
        },
        "watch_cells": [
            {"benchmark": cell["benchmark"], "title": summary["titles"][cell["benchmark"]], "before": (cell["before"] or {}).get("estimate"), "after": cell["after"]["estimate"],
             "difference": (cell["difference"] or {}).get("estimate"), "low": (cell["difference"] or {}).get("low"), "high": (cell["difference"] or {}).get("high"), "regression": cell["regression"]}
            for cell in summary["watch_cells"]
        ],
        "reasoning_pass": [
            {"benchmark": bench, "title": summary["titles"][bench], "base": entry["base"]["estimate"], "adapter": entry["adapter"]["estimate"], "difference": (entry["difference"] or {}).get("estimate"),
             "low": (entry["difference"] or {}).get("low"), "high": (entry["difference"] or {}).get("high")}
            for bench, entry in summary["reasoning_pass"]["benchmarks"].items()
        ],
        "decisions": [{"rule": d["rule"], "benchmark": d["benchmark"], "outcome": d["outcome"]} for d in summary["decisions"]],
    }


def render_report(summary: dict) -> str:
    lines: list[str] = []
    add = lines.append
    titles, targets, main = summary["titles"], summary["targets"], summary["main_adapter"]
    fast_states = {name: state for name, state in summary["states"].items() if state["setting"] == "fast"}
    add("# Evaluation of the adapters")
    add("")
    add(
        f"Every model state answered the same benchmark items with the same prompt, and a program graded every reply. The fast setting is a forced direct answer at effort 0.0. "
        f"Scores are percentages with 95% bootstrap intervals over items. A difference is paired on the same items, in points, with its 95% bootstrap interval, and the p-value is "
        f"from an exact McNemar test on the items where the two states disagree. The main adapter is `{main}`. Everything in this report is measured unless it says otherwise. "
        f"Generated {summary['generated_at']}."
    )
    add("")
    add("Items per benchmark: " + ", ".join(f"{titles[name]} {count:,}" for name, count in summary["items"].items()) + ".")
    if summary["missing"]:
        add("")
        add("Not complete at the time of this report: " + "; ".join(summary["missing"]) + ".")

    add("")
    add("## The decision rules")
    add("")
    add("The rules were fixed before any adapter was trained. They are applied as written, on GSM8K and on MBPP separately.")
    add("")
    for decision in summary["decisions"]:
        add(f"- {decision['line']}")

    add("")
    add("## The target benchmarks")
    add("")
    for bench in targets:
        gap = summary["base_gap"].get(bench)
        if gap:
            add(f"On {titles[bench]} the base model scores {gap['base_fast'] * 100:.1f} in the fast setting and {gap['base_reasoning'] * 100:.1f} in the reasoning setting on the same items, a gap of {pts(gap['gap'])} points.")
    add("The share of the gap closed is a state's gain over the base model's fast score, divided by that gap.")
    for bench in targets:
        add("")
        add(f"{titles[bench]}:")
        add("")
        add("| state | score | minus base | McNemar p | share of the gap closed | completion tokens per answer |")
        add("|---|---|---|---|---|---|")
        for name, state in fast_states.items():
            cell = state["benchmarks"].get(bench)
            if cell:
                add(f"| {name} | {pct(cell['score'])} | {pts(cell.get('minus_base'))} | {pvalue(cell.get('minus_base'))} | {share(cell.get('gap_closed'))} | {cell['tokens_per_answer']['completion'].get('mean', 0):,.0f} |")

    add("")
    add("## The main comparison: middle band against random")
    add("")
    add(
        "Both arms train on the same number of training tokens with the same number of updates. Each of the three seeds has its own draw of both sets, its own starting weights "
        "and its own batch order. The pooled difference averages each item's score over an arm's three adapters and resamples benchmark items."
    )
    add("")
    add("| benchmark | middle band, by seed | random, by seed | middle minus random, pooled | seed by seed |")
    add("|---|---|---|---|---|")
    for bench in targets:
        entry = summary["main_comparison"].get(bench, {})
        spread = entry.get("seed_range")

        def by_seed(values: list) -> str:
            return ", ".join("" if value is None else f"{value * 100:.1f}" for value in values)

        add(f"| {titles[bench]} | {by_seed(entry.get('middle_scores', []))} | {by_seed(entry.get('random_scores', []))} | {pts(entry.get('middle_minus_random'))} | "
            + (f"{spread['low'] * 100:+.1f} to {spread['high'] * 100:+.1f}" if spread else "") + " |")
    add("")
    add(f"The controls are single runs with seed 1, compared with `{main}` item by item.")
    add("")
    add("| benchmark | control | control score | middle band score | middle band minus control | McNemar p |")
    add("|---|---|---|---|---|---|")
    for bench in targets:
        for control in summary["main_comparison"].get(bench, {}).get("controls", {}).values():
            add(f"| {titles[bench]} | {control['label']} | {pct(control['control_score'])} | {pct(control['middle_score'])} | {pts(control['middle_minus_control'])} | {pvalue(control['middle_minus_control'])} |")

    add("")
    add("## The size sweep")
    add("")
    add("One adapter per size and arm, seed 1. The sets of each arm are nested, and the two arms have the same training tokens at every size.")
    for bench in targets:
        add("")
        add(f"{titles[bench]}:")
        add("")
        add("| items in the middle-band set | middle band | minus base | random | minus base |")
        add("|---|---|---|---|---|")
        arms = summary["size_sweep"].get(bench, {})
        for size in SIZES:
            middle = next((p for p in arms.get("middle", []) if p["size"] == size), None)
            rand = next((p for p in arms.get("random", []) if p["size"] == size), None)
            if middle or rand:
                add(f"| {size:,} | {pct(middle and middle['score'])} | {pts(middle and middle['minus_base'])} | {pct(rand and rand['score'])} | {pts(rand and rand['minus_base'])} |")

    if summary["prompt_only"]:
        add("")
        add("## Eight examples in the prompt, with no training")
        add("")
        add("The prompt-only state is the base model in the fast setting with eight held-out middle-band problems and their answers placed before the question.")
        add("")
        add("| benchmark | base | prompt-only | prompt-only minus base | McNemar p | main adapter | main adapter minus prompt-only | McNemar p |")
        add("|---|---|---|---|---|---|---|---|")
        for bench, entry in summary["prompt_only"].items():
            add(f"| {titles[bench]} | {pct(entry['base_score'])} | {pct(entry['score'])} | {pts(entry['minus_base'])} | {pvalue(entry['minus_base'])} | {pct(entry['main_score'])} | "
                f"{pts(entry['main_minus_prompt_only'])} | {pvalue(entry['main_minus_prompt_only'])} |")

    add("")
    add("## The watch cells under the main adapter")
    add("")
    add(f"These benchmarks were never trained. A fall of more than {summary['regression_points']:g} points is flagged.")
    add("")
    add("| benchmark | before | after | difference | McNemar p | flag |")
    add("|---|---|---|---|---|---|")
    for cell in summary["watch_cells"]:
        add(f"| {titles[cell['benchmark']]} | {pct(cell['before'])} | {pct(cell['after'])} | {pts(cell['difference'])} | {pvalue(cell['difference'])} | {'fell by more than ' + format(summary['regression_points'], 'g') + ' points' if cell['regression'] else ''} |")

    slow = summary["reasoning_pass"]["benchmarks"]
    if slow:
        add("")
        add("## The main adapter when it is left free to reason")
        add("")
        add(
            "The adapter was trained to give the answer alone. Here it runs in the reasoning setting (free-running effort 0.0 with the 8,192-token limit, the study's reference), "
            "on the same items, against the base model in the same setting."
        )
        add("")
        add("| benchmark | items | base | adapter | difference | McNemar p | thinking tokens per answer, base | adapter | replies at the limit, base | adapter |")
        add("|---|---|---|---|---|---|---|---|---|---|")
        for bench, entry in slow.items():
            add(f"| {titles[bench]} | {entry['n']:,}{'' if entry['complete'] else ' (incomplete)'} | {pct(entry['base'])} | {pct(entry['adapter'])} | {pts(entry['difference'])} | {pvalue(entry['difference'])} | "
                f"{entry['base_thinking_tokens'] or 0:,.0f} | {entry['adapter_thinking_tokens'] or 0:,.0f} | {entry['base_cut_off']} | {entry['adapter_cut_off']} |")

    add("")
    add("## Every state on every benchmark, fast setting")
    for bench in summary["items"]:
        add("")
        add(f"{titles[bench]}" + (" (target)" if bench in targets else " (watch)") + ":")
        add("")
        add("| state | score | minus base | McNemar p | prompt tokens per answer | completion tokens per answer | flag |")
        add("|---|---|---|---|---|---|---|")
        for name, state in fast_states.items():
            cell = state["benchmarks"].get(bench)
            if cell:
                flag = "regression" if cell.get("regression") else ("" if cell["complete"] else "incomplete")
                add(f"| {name} | {pct(cell['score'])} | {pts(cell.get('minus_base'))} | {pvalue(cell.get('minus_base'))} | {cell['tokens_per_answer']['prompt']:,.0f} | {cell['tokens_per_answer']['completion'].get('mean', 0):,.0f} | {flag} |")

    add("")
    add("## Cost")
    add("")
    cost = summary["usage"]
    add(
        f"The evaluation made {cost['api_calls']:,} sampling calls and read {cost['cache_hits']:,} replies from the cache: {cost['prompt_tokens']:,} prompt tokens and "
        f"{cost['completion_tokens']:,} completion tokens. {cost['failed_calls']} calls failed, {cost['retries']} were retried and {cost['throttled_calls']} saw a throttling event."
    )
    if cost["fast_pass_by_adapter"]:
        minutes = [entry["minutes"] for entry in cost["fast_pass_by_adapter"].values()]
        add(f"A fast-setting pass over one adapter took between {min(minutes)} and {max(minutes)} minutes of wall time.")
    add("")
    return "\n".join(lines)
