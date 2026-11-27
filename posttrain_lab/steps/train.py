"""Train: one LoRA adapter per run, on a saved training set.

A training example is an item's prompt followed by its answer and nothing else,
rendered for the fast setting, so the adapter learns to answer without a reasoning
block. Every run records its config, seed, set, token counts and losses, saves one
sampler checkpoint with an expiry, and is scored on held-out synthetic dev items as a
sanity check. A finished run is skipped on rerun; an interrupted one starts over.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import math
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

from ..client import AccountBlockedError, ModelClient
from ..data import generators
from ..render import FAST_EFFORT
from ..stats import percentile
from ..store import append_jsonl, read_jsonl, stable_hash
from .base import Step, StepContext, Unit

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULTS = {"rank": 16, "learning_rate": 5.4e-4, "batch_size": 32, "epochs": 4}


def load_data(config: dict) -> tuple[dict[str, dict], dict]:
    """The labeled items by id, and the saved sets."""
    data_dir = Path(config.get("data_dir") or REPO_ROOT / "results" / "data")
    with gzip.open(data_dir / "items.jsonl.gz", "rt", encoding="utf-8") as handle:
        items = {row["id"]: row for row in map(json.loads, handle) if "id" in row}
    return items, json.loads((data_dir / "sets.json").read_text())


def set_items(name: str, items: dict[str, dict], sets: dict) -> list[dict]:
    """The items of a named set, with the answers the set trains on."""
    if name == "pilot-train":
        return [items[i] for i in sets["pilot"]["train"]]
    entry = sets["sets"][name]
    moved = entry.get("answers", {})  # the shuffled-label set trains on permuted answers
    return [{**items[i], "answer": moved.get(i, items[i]["answer"])} for i in entry["items"]]


def fast_example(renderer, prompt: str, answer: str) -> tuple[list[int], list[float]]:
    """A training example for the fast setting, as tokens and loss weights.

    The prompt and the forced answer opening carry no loss, as they are given at
    sampling time. The answer and the tokens that end the reply carry all of it.
    """
    messages = [{"role": "user", "content": prompt}]
    special = renderer.special
    given = [*renderer.build_prompt(messages, FAST_EFFORT), special.message_model, special.content_text]
    tokens, weights = renderer.build_example([*messages, {"role": "assistant", "content": answer}], FAST_EFFORT)
    if list(tokens[: len(given)]) != given:
        raise ValueError("A training example does not start with the fast-setting prompt")
    return list(tokens), [0.0] * len(given) + [float(w) for w in weights[len(given) :]]


def schedule(n_items: int, batch_size: int, epochs: int, seed: int, per_epoch: int | None = None) -> list[list[int]]:
    """Batches of item positions: a fresh shuffle for every pass over the set.

    With ``per_epoch`` each pass is cut into that many batches of nearly equal size,
    so sets with the same training tokens get the same number of updates whatever
    their item counts. Without it a batch is ``batch_size`` items.
    """
    batches = []
    for epoch in range(epochs):
        order = list(range(n_items))
        random.Random(f"{seed}:{epoch}").shuffle(order)
        if per_epoch:
            count = min(per_epoch, n_items)
            edges = [round(index * n_items / count) for index in range(count + 1)]
            batches += [order[edges[index] : edges[index + 1]] for index in range(count)]
        else:
            batches += [order[start : start + batch_size] for start in range(0, n_items, batch_size)]
    return batches


def updates_per_epoch(set_tokens: int, tokens_per_update: int | None) -> int | None:
    """How many updates one pass over a set gets when updates are sized in tokens."""
    return max(1, round(set_tokens / tokens_per_update)) if tokens_per_update else None


STALL_SECONDS = 30.0  # an update that takes this long counts as a stall; the usual one takes about five


def write_atomic(path: Path, text: str) -> None:
    """Replace a file in one step, so a reader in another process never sees half of it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def now() -> datetime:
    return datetime.now(timezone.utc)


class Train(Step):
    name = "train"
    summary = "Train LoRA adapters on the free model ID"
    default_config = "configs/train.toml"
    parallel_units = True  # a run that fails stays pending and the next one starts
    unit_workers = 1  # one run at a time: a second training client waits in the service's queue

    def _data(self, ctx: StepContext) -> tuple[dict[str, dict], dict]:
        if "data" not in ctx.scratch:
            ctx.scratch["data"] = load_data(ctx.config)
        return ctx.scratch["data"]

    def plan(self, ctx: StepContext) -> list[Unit]:
        items, sets = self._data(ctx)
        units = []
        for run in ctx.config.get("runs", []):
            settings = {key: run.get(key, ctx.config.get(key, default)) for key, default in DEFAULTS.items()}
            rows = set_items(run["set"], items, sets)
            tokens = sum(row["train_tokens"] for row in rows)
            params = {
                "name": run["name"], "set": run["set"], "seed": run.get("seed", ctx.config.get("seed", 0)), **settings,
                "set_fingerprint": stable_hash([[row["id"], row["answer"]] for row in rows])[:12],
            }
            sized = run.get("tokens_per_update", ctx.config.get("tokens_per_update"))
            if sized:  # updates sized in tokens, so token-matched sets get the same number of them
                params["tokens_per_update"] = sized
            updates = len(schedule(len(rows), settings["batch_size"], settings["epochs"], params["seed"], updates_per_epoch(tokens, sized)))
            units.append(Unit("run", params, calls=updates, training_tokens=tokens * settings["epochs"], label=f"{run['name']} ({len(rows):,} items, {updates} updates)"))
        ctx.scratch["plan"] = units
        return units

    def _stored(self, ctx: StepContext) -> list[dict]:
        rows = (ctx.store.get(unit.key(ctx.step)) for unit in ctx.scratch.get("plan", []))
        return [row for row in rows if row is not None]

    async def run_unit(self, unit: Unit, ctx: StepContext) -> dict:
        p = unit.params
        ctx.out(f"run {unit.label}")
        ctx.scratch["now"] = {"name": p["name"], "started_at": now().isoformat(timespec="seconds")}
        write_outputs(ctx, self._stored(ctx))
        try:
            return await self._run(unit, ctx)
        except AccountBlockedError:
            raise
        except Exception as exc:
            append_jsonl(ctx.runs_dir / "failures.jsonl", {"run": p["name"], "at": now().isoformat(timespec="seconds"), "error": f"{type(exc).__name__}: {str(exc)[:300]}"})
            ctx.out(f"    {p['name']} failed and stays pending: {type(exc).__name__}: {str(exc)[:200]}")
            raise
        finally:
            ctx.scratch["now"] = None

    async def _run(self, unit: Unit, ctx: StepContext) -> dict:
        p = unit.params
        client: ModelClient = ctx.client
        backend = client.backend
        items, sets = self._data(ctx)
        rows = set_items(p["set"], items, sets)

        # Stay under the service's cap on stored checkpoints.
        stored = (await backend.account_snapshot()).get("checkpoints_listed", 0)
        if stored >= ctx.config.get("max_checkpoints", 90):
            raise AccountBlockedError(f"{stored} checkpoints are stored, at or over the limit this step allows itself. Let some expire or delete them.")

        examples = [fast_example(ctx.renderer, row["prompt"], row["answer"]) for row in rows]
        per_epoch = updates_per_epoch(sum(row["train_tokens"] for row in rows), p.get("tokens_per_update"))
        batches = schedule(len(rows), p["batch_size"], p["epochs"], p["seed"], per_epoch)
        rates = [p["learning_rate"] * (1 - step / len(batches)) for step in range(len(batches))]  # linear decay to zero
        started_at = now().isoformat(timespec="seconds")
        started = time.monotonic()
        training = await client.create_training_run(rank=p["rank"], seed=p["seed"], tag=p["name"])
        held = {cell: [fast_example(ctx.renderer, items[i]["prompt"], items[i]["answer"]) for i in ids] for cell, ids in sets["dev"].items() if ids}

        async def dev_loss() -> dict[str, float]:
            """Loss per target token on the held-out generated items, without an update."""
            out = {}
            for cell, batch in held.items():
                result, _ = await training.forward(batch, tag=f"{p['name']}:dev-loss")
                out[cell] = round(result.get("metrics", {}).get("loss:sum", 0.0) / max(sum(sum(w) for _, w in batch), 1.0), 5)
            return out

        events_seen = len(getattr(backend, "events", []))
        states: dict[str, int] = {}

        def note_events() -> None:
            """Write down what the service said about its queue since the last look."""
            nonlocal events_seen
            for event in getattr(backend, "events", [])[events_seen:]:
                record = event.to_dict()
                append_jsonl(ctx.runs_dir / "events.jsonl", {"run": p["name"], **record})
                label = record.get("state") or record.get("type")
                states[label] = states.get(label, 0) + 1
            events_seen = len(getattr(backend, "events", []))

        async def watch_queue() -> None:
            """Save queue events as they come, so a long wait can be read while it lasts."""
            while True:
                await asyncio.sleep(10)
                note_events()

        watcher = asyncio.ensure_future(watch_queue())
        try:
            row = await self._train(ctx, p, client, backend, training, rows, items, sets, examples, batches, rates, dev_loss, note_events, started)
        finally:
            watcher.cancel()
            note_events()
            # A fresh session for the next run: a second training client in one session waits in the queue.
            if hasattr(backend, "new_session"):
                await backend.new_session()
        row["service_events"] = states
        row["started_at"] = started_at
        return row

    async def _train(self, ctx, p, client, backend, training, rows, items, sets, examples, batches, rates, dev_loss, note_events, started) -> dict:
        dev_loss_before = await dev_loss()
        loop_started = time.monotonic()
        losses: list[float] = []
        update_seconds: list[float] = []
        group_losses: list[float] = []
        chunk = ctx.config.get("updates_per_call", 16)
        evaluating = ctx.runs_dir.parent / "eval" / "ACTIVE"  # there while another process evaluates adapters
        for start in range(0, len(batches), chunk):
            group = batches[start : start + chunk]
            overlapped = evaluating.exists()
            results, record = await training.train_pipelined([[examples[i] for i in batch] for batch in group], rates[start : start + chunk], tag=p["name"])
            if record.attempts > 1:
                raise RuntimeError("A group of updates was sent more than once after an error, so the weights cannot be trusted. The run starts over.")
            took = []
            for batch, result in zip(group, results):
                weight = sum(sum(examples[i][1]) for i in batch)
                losses.append(round(result.get("metrics", {}).get("loss:sum", 0.0) / max(weight, 1.0), 5))
                took.append(round(result.get("seconds", 0.0), 2))
            update_seconds += took
            group_losses.append(round(sum(losses[-len(group) :]) / len(group), 5))
            note_events()
            append_jsonl(
                ctx.runs_dir / "progress.jsonl",
                {"run": p["name"], "at": now().isoformat(timespec="seconds"), "updates": start + len(group), "of": len(batches), "loss": group_losses[-1],
                 "seconds": round(time.monotonic() - started, 1), "update_seconds": took, "eval_active": overlapped or evaluating.exists()},
            )
            ctx.out(f"    {p['name']}: {start + len(group)}/{len(batches)} updates, loss {group_losses[-1]:.4f}, {time.monotonic() - started:.0f} s")
            if not math.isfinite(group_losses[-1]) or group_losses[-1] > 2 * group_losses[0]:
                raise RuntimeError(f"The run diverged: loss {group_losses[-1]} after {start + len(group)} updates, from {group_losses[0]}.")
        loop_seconds = time.monotonic() - loop_started
        dev_loss_after = await dev_loss()

        stamp = now().strftime("%Y%m%d-%H%M%S")
        ttl = int(ctx.config.get("checkpoint_ttl_days", 30) * 86400)
        path, _ = await training.save_sampler_weights(f"{p['name']}-{stamp}", ttl, tag=p["name"])
        info = await backend.checkpoint_info(path)
        await client.load_adapter(path, tag=p["name"])  # refuses an adapter whose base model is not free

        # A sanity check on held-out generated items. Never a headline number.
        score_ids = [i for ids in sets["dev"].values() for i in ids]
        client.set_concurrency(ctx.config.get("client", {}).get("max_concurrency", 16))
        dev = {}
        for item_id in score_ids:
            dev[item_id] = client.sample([{"role": "user", "content": items[item_id]["prompt"]}], effort=FAST_EFFORT, max_tokens=256, force_answer=True, model_path=path, tag=f"{p['name']}:dev")
        replies = dict(zip(dev, await asyncio.gather(*dev.values())))
        by_cell: dict[str, list[float]] = {}
        thinking = 0
        unfinished = 0
        for item_id, reply in replies.items():
            by_cell.setdefault(items[item_id]["cell"], []).append(generators.grade_answer(items[item_id], reply.text)["score"])
            thinking += reply.record.thinking_tokens
            unfinished += int(not reply.parsed.ended)
        tail = losses[-max(1, len(losses) // 10) :]
        slow = [seconds for seconds in update_seconds if seconds >= STALL_SECONDS]
        return {
            **{k: p[k] for k in ("name", "set", "seed", "rank", "learning_rate", "batch_size", "epochs", "set_fingerprint")},
            "items": len(rows),
            "set_train_tokens": sum(len(tokens) for tokens, _ in examples),
            "train_tokens": sum(len(examples[i][0]) for batch in batches for i in batch),
            "updates": len(batches),
            "tokens_per_update": p.get("tokens_per_update"),
            "items_per_update": round(len(rows) * p["epochs"] / len(batches), 1),
            "train_seconds": round(loop_seconds, 1),
            "seconds_per_update": round(loop_seconds / len(batches), 2),
            "median_seconds_per_update": round(percentile(update_seconds, 50), 2),
            "run_seconds": round(time.monotonic() - started, 1),  # with set-up, the save, the load and the dev replies
            "stalls": {"updates": len(slow), "seconds": round(sum(slow), 1), "longest_s": round(max(slow), 1) if slow else 0.0},
            "update_seconds": update_seconds,
            "loss_first": losses[0],
            "loss_last": round(sum(tail) / len(tail), 5),
            "losses": losses,
            "group_losses": group_losses,
            "tinker_path": path,
            "checkpoint": info,
            "checkpoint_ttl_days": ctx.config.get("checkpoint_ttl_days", 30),
            "dev_loss_before": dev_loss_before,
            "dev_loss": dev_loss_after,
            "dev_accuracy": {cell: round(sum(scores) / len(scores), 4) for cell, scores in sorted(by_cell.items())},
            "dev_items": len(replies),
            "dev_thinking_tokens": thinking,
            "dev_unfinished_replies": unfinished,
            "loss_weighted_tokens": int(sum(sum(weights) for _, weights in examples)),
            "finished_at": now().isoformat(timespec="seconds"),
        }

    async def finish(self, ctx: StepContext, rows: list[dict]) -> None:
        write_outputs(ctx, rows)
        ctx.out(f"Wrote the summary, the report, the figure data and PROGRESS.md to {ctx.results_dir}")


# ------------------------------------------------------------------ outputs


def overlap_check(progress: list[dict]) -> dict:
    """Update times while another process was evaluating adapters, against times while none was."""
    out = {}
    for label, wanted in (("alone", False), ("while_evaluating", True)):
        seconds = [s for row in progress if bool(row.get("eval_active")) == wanted for s in row.get("update_seconds", [])]
        out[label] = {"updates": len(seconds), "median_s": round(percentile(seconds, 50), 2) if seconds else None, "mean_s": round(sum(seconds) / len(seconds), 2) if seconds else None}
    both = out["alone"]["updates"] and out["while_evaluating"]["updates"]
    out["median_ratio"] = round(out["while_evaluating"]["median_s"] / out["alone"]["median_s"], 2) if both else None
    out["mean_ratio"] = round(out["while_evaluating"]["mean_s"] / out["alone"]["mean_s"], 2) if both else None
    return out


def build_summary(ctx: StepContext, rows: list[dict]) -> dict:
    planned = [unit.params["name"] for unit in ctx.scratch.get("plan", [])]
    order = {name: index for index, name in enumerate(planned)}
    runs = sorted(
        ({k: v for k, v in row.items() if k not in ("key", "kind", "params", "losses", "update_seconds")} for row in rows),
        key=lambda run: order.get(run["name"], len(order)),
    )
    names = set(planned)
    progress = [row for row in read_jsonl(ctx.runs_dir / "progress.jsonl") if row.get("run") in names]
    failures = [row for row in read_jsonl(ctx.runs_dir / "failures.jsonl") if row.get("run") in names]
    starts = [run["started_at"] for run in runs if run.get("started_at")]
    current = ctx.scratch.get("now")
    if current:
        starts.append(current["started_at"])
    first = min(starts) if starts else None
    status_file = ctx.runs_dir.parent / "eval" / "status.json"
    return {
        "generated_at": now().isoformat(timespec="seconds"),
        "model_id": ctx.config.get("model_id"),
        "settings": {key: ctx.config.get(key) for key in ("rank", "learning_rate", "tokens_per_update", "batch_size", "epochs", "checkpoint_ttl_days") if ctx.config.get(key) is not None},
        "planned_runs": planned,
        "complete": len(runs) == len(planned),
        "first_run_started_at": first,
        "hours_since_first_start": round((now() - datetime.fromisoformat(first)).total_seconds() / 3600, 2) if first else 0.0,
        "now_training": current,
        "runs": runs,
        "totals": {
            "runs": len(runs),
            "updates": sum(run["updates"] for run in runs),
            "train_tokens": sum(run["train_tokens"] for run in runs),
            "update_hours": round(sum(run["train_seconds"] for run in runs) / 3600, 2),
            "run_hours": round(sum(run["run_seconds"] for run in runs) / 3600, 2),
            "seconds_per_update": round(sum(run["train_seconds"] for run in runs) / max(sum(run["updates"] for run in runs), 1), 2),
            "stalled_updates": sum(run.get("stalls", {}).get("updates", 0) for run in runs),
            "stalled_seconds": round(sum(run.get("stalls", {}).get("seconds", 0) for run in runs), 1),
            "queue_events": {state: sum(run.get("service_events", {}).get(state, 0) for run in runs) for state in sorted({s for run in runs for s in run.get("service_events", {})})},
            "checkpoint_bytes": sum((run.get("checkpoint") or {}).get("size_bytes", 0) or 0 for run in runs),
        },
        "overlap_with_evaluation": overlap_check(progress),
        "failures": failures,
        "evaluation_status": json.loads(status_file.read_text()) if status_file.exists() else None,
    }


def trouble(run: dict) -> str:
    """What went wrong during a run, in a few words."""
    notes = []
    stalls = run.get("stalls", {})
    if stalls.get("updates"):
        notes.append(f"{stalls['updates']} updates of {STALL_SECONDS:.0f} s or more, {stalls['seconds'] / 60:.1f} min in all, longest {stalls['longest_s']:.0f} s")
    for state, count in sorted(run.get("service_events", {}).items()):
        notes.append(f"{state} reported {count} times")
    return "; ".join(notes) or "none"


def render_progress(summary: dict) -> str:
    runs, planned = summary["runs"], summary["planned_runs"]
    lines = ["# Training progress", ""]
    current = summary.get("now_training")
    lines.append(
        f"Updated {summary['generated_at'][:16].replace('T', ' ')} UTC. {len(runs)} of {len(planned)} runs are done, "
        f"{summary['hours_since_first_start']} hours after the first one started. "
        + (f"Training now: {current['name']}, since {current['started_at'][11:16]} UTC." if current else ("All runs are done." if summary["complete"] else "No run is training at the moment."))
    )
    lines += ["", "| run | updates | minutes | s per update | final loss | dev middle | dev easy | dev hard | stalls and pauses |", "|---|---|---|---|---|---|---|---|---|"]
    for run in runs:
        dev = run["dev_accuracy"]
        lines.append(
            f"| {run['name']} | {run['updates']} | {run['run_seconds'] / 60:.0f} | {run['seconds_per_update']} | {run['loss_last']:.3f} | "
            f"{dev.get('middle', 0):.1%} | {dev.get('easy', 0):.1%} | {dev.get('hard', 0):.1%} | {trouble(run)} |"
        )
    left = [name for name in planned if name not in {run["name"] for run in runs}]
    lines += ["", "Still to run: " + (", ".join(left) if left else "nothing") + "."]
    unusual = []
    for failure in summary["failures"]:
        unusual.append(f"{failure['run']} failed at {failure['at'][11:16]} UTC and was left pending ({failure['error']})")
    overlap = summary["overlap_with_evaluation"]
    if overlap["median_ratio"] is not None:
        unusual.append(
            f"updates took a median of {overlap['while_evaluating']['median_s']} s while adapters were being evaluated ({overlap['while_evaluating']['updates']} updates) "
            f"and {overlap['alone']['median_s']} s otherwise ({overlap['alone']['updates']} updates); the means were {overlap['while_evaluating']['mean_s']} s and {overlap['alone']['mean_s']} s"
        )
    lines += ["", "Anything unusual: " + ("; ".join(unusual) + "." if unusual else "nothing beyond the stalls and pauses in the table.")]
    status = summary.get("evaluation_status")
    if status:
        done = [name for name, state in status.get("states", {}).items() if state["done"] >= state["of"]]
        partly = [f"{name} ({state['done']:,} of {state['of']:,})" for name, state in status.get("states", {}).items() if 0 < state["done"] < state["of"]]
        lines += ["", f"Evaluation, as of {status.get('at', '')[:16].replace('T', ' ')} UTC: {len(done)} states are complete ({', '.join(done) or 'none'})" + (f", and in progress: {', '.join(partly)}." if partly else ".")]
    return "\n".join(lines) + "\n"


def render_report(summary: dict) -> str:
    runs, totals = summary["runs"], summary["totals"]

    def values(key: str, form: str = "{}") -> str:
        found = sorted({run[key] for run in runs if run.get(key) is not None})
        return ", ".join(form.format(value) for value in found) or "not set"

    sized = values("tokens_per_update", "{:,}")
    lines = ["# Training runs", ""]
    lines.append(
        f"One LoRA adapter per run on `{summary['model_id']}`, from a saved set in `results/data/sets.json`. A training example is the prompt followed by the answer alone, "
        f"rendered for the fast setting, and only the answer and the two closing tokens carry loss. Across the runs here the rank is {values('rank')}, the learning rate {values('learning_rate', '{:g}')} "
        f"(falling linearly to zero), and the passes over a set {values('epochs')}. "
        + (f"An update is about {sized} training tokens. " if sized != "not set" else f"An update is {values('batch_size')} items. ")
        + f"{len(runs)} of {len(summary['planned_runs'])} planned runs are done. Generated {summary['generated_at']}."
    )
    lines += ["", "## Runs", "",
              "| run | set | seed | items | training tokens | updates | wall time (min) | s per update | first loss | final loss | stalls and pauses | checkpoint |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for run in runs:
        lines.append(
            f"| {run['name']} | {run['set']} | {run['seed']} | {run['items']:,} | {run['train_tokens']:,} | {run['updates']} | {run['run_seconds'] / 60:.1f} | "
            f"{run['seconds_per_update']} | {run['loss_first']:.3f} | {run['loss_last']:.3f} | {trouble(run)} | `{run['tinker_path']}` |"
        )
    lines += ["", "Training tokens count every pass. The final loss is the mean over the last tenth of a run's updates, per target token. "
              "Wall time includes setting up, the held-out loss, saving the adapter and scoring the dev items."]
    lines += ["", "## Held-out generated items", "",
              "A sanity check and never a headline number: scores come only from public benchmarks. The dev items are 150 from each cell, answered in the fast setting. "
              "By construction the base model gets every easy one right and none of the others.", "",
              "| run | dev loss before, middle | after | dev middle | dev easy | dev hard | unfinished replies |", "|---|---|---|---|---|---|---|"]
    for run in runs:
        dev = run["dev_accuracy"]
        lines.append(
            f"| {run['name']} | {run['dev_loss_before'].get('middle', 0):.3f} | {run['dev_loss'].get('middle', 0):.3f} | {dev.get('middle', 0):.1%} | {dev.get('easy', 0):.1%} | "
            f"{dev.get('hard', 0):.1%} | {run['dev_unfinished_replies']} |"
        )
    lines += ["", "## Time, stalls and pauses", ""]
    lines.append(
        f"The {totals['runs']} runs made {totals['updates']:,} updates on {totals['train_tokens']:,} training tokens in {totals['update_hours']} hours of updates and "
        f"{totals['run_hours']} hours of wall time, {totals['seconds_per_update']} s per update on average. {totals['stalled_updates']} updates took {STALL_SECONDS:.0f} s or more, "
        f"{totals['stalled_seconds'] / 60:.1f} minutes in all. Queue states the service reported: "
        + (", ".join(f"{state} ({count})" for state, count in totals["queue_events"].items()) or "none")
        + f". The checkpoints take {totals['checkpoint_bytes'] / 1e9:.1f} GB."
    )
    overlap = summary["overlap_with_evaluation"]
    if overlap["median_ratio"] is not None:
        lines += ["", f"Adapters were evaluated by a second process while later runs trained. Updates took a median of {overlap['while_evaluating']['median_s']} s "
                  f"(mean {overlap['while_evaluating']['mean_s']} s, {overlap['while_evaluating']['updates']:,} updates) while that process was working and {overlap['alone']['median_s']} s "
                  f"(mean {overlap['alone']['mean_s']} s, {overlap['alone']['updates']:,} updates) while it was not."]
    if summary["failures"]:
        lines += ["", "Runs that failed and were started again:", ""]
        lines += [f"- {failure['run']} at {failure['at']}: {failure['error']}" for failure in summary["failures"]]
    return "\n".join(lines) + "\n"


def figure_data(summary: dict) -> dict:
    return {
        "generated_at": summary["generated_at"],
        "runs": [
            {"run": run["name"], "set": run["set"], "seed": run["seed"], "items": run["items"], "train_tokens": run["train_tokens"], "updates": run["updates"],
             "minutes": round(run["run_seconds"] / 60, 1), "seconds_per_update": run["seconds_per_update"], "loss_by_group_of_updates": run.get("group_losses", []),
             "loss_first": run["loss_first"], "loss_last": run["loss_last"], "dev_accuracy": run["dev_accuracy"], "stalled_seconds": run.get("stalls", {}).get("seconds", 0)}
            for run in summary["runs"]
        ],
        "overlap_with_evaluation": summary["overlap_with_evaluation"],
    }


def write_outputs(ctx: StepContext, rows: list[dict]) -> None:
    summary = build_summary(ctx, rows)
    write_atomic(ctx.results_dir / "summary.json", json.dumps(summary, indent=1) + "\n")
    write_atomic(ctx.results_dir / "PROGRESS.md", render_progress(summary))
    write_atomic(ctx.results_dir / "REPORT.md", render_report(summary))
    write_atomic(ctx.results_dir / "figure-data.json", json.dumps(figure_data(summary), indent=1) + "\n")
