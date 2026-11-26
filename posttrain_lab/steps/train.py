"""Train: one LoRA adapter per run, on a saved training set.

A training example is an item's prompt followed by its answer and nothing else,
rendered for the fast setting, so the adapter learns to answer without a reasoning
block. Every run records its config, seed, set, token counts and losses, saves one
sampler checkpoint with an expiry, and is scored on held-out synthetic dev items as a
sanity check. A finished run is skipped on rerun; an interrupted one starts over.
"""

from __future__ import annotations

import gzip
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

from ..client import AccountBlockedError, ModelClient
from ..data import generators
from ..render import FAST_EFFORT
from ..store import append_jsonl, stable_hash
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


def schedule(n_items: int, batch_size: int, epochs: int, seed: int) -> list[list[int]]:
    """Batches of item positions: a fresh shuffle for every pass over the set."""
    batches = []
    for epoch in range(epochs):
        order = list(range(n_items))
        random.Random(f"{seed}:{epoch}").shuffle(order)
        batches += [order[start : start + batch_size] for start in range(0, n_items, batch_size)]
    return batches


class Train(Step):
    name = "train"
    summary = "Train LoRA adapters on the free model ID"
    default_config = "configs/train.toml"

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
            updates = len(schedule(len(rows), settings["batch_size"], settings["epochs"], params["seed"]))
            units.append(Unit("run", params, calls=updates, training_tokens=tokens * settings["epochs"], label=f"{run['name']} ({len(rows):,} items, {updates} updates)"))
        return units

    async def run_unit(self, unit: Unit, ctx: StepContext) -> dict:
        p = unit.params
        client: ModelClient = ctx.client
        backend = client.backend
        items, sets = self._data(ctx)
        rows = set_items(p["set"], items, sets)

        # Stay under the service's cap on stored checkpoints.
        stored = (await backend.account_snapshot()).get("checkpoints_listed", 0)
        if stored >= ctx.config.get("max_checkpoints", 90):
            raise AccountBlockedError(f"{stored} checkpoints are stored, at or over the limit this step allows itself. Let some expire or delete them.")

        examples = [
            ctx.renderer.build_example([{"role": "user", "content": row["prompt"]}, {"role": "assistant", "content": row["answer"]}], FAST_EFFORT)
            for row in rows
        ]
        batches = schedule(len(rows), p["batch_size"], p["epochs"], p["seed"])
        rates = [p["learning_rate"] * (1 - step / len(batches)) for step in range(len(batches))]  # linear decay to zero
        started = time.monotonic()
        training = await client.create_training_run(rank=p["rank"], seed=p["seed"], tag=p["name"])
        losses: list[float] = []
        chunk = ctx.config.get("updates_per_call", 16)
        for start in range(0, len(batches), chunk):
            group = batches[start : start + chunk]
            results, record = await training.train_pipelined([[examples[i] for i in batch] for batch in group], rates[start : start + chunk], tag=p["name"])
            for batch, result in zip(group, results):
                weight = sum(sum(examples[i][1]) for i in batch)
                losses.append(round(result.get("metrics", {}).get("loss:sum", 0.0) / max(weight, 1.0), 5))
            append_jsonl(ctx.runs_dir / "progress.jsonl", {"run": p["name"], "updates": start + len(group), "of": len(batches), "loss": losses[-1], "seconds": round(time.monotonic() - started, 1)})
            ctx.out(f"    {p['name']}: {start + len(group)}/{len(batches)} updates, loss {losses[-1]:.4f}, {time.monotonic() - started:.0f} s")
        train_seconds = time.monotonic() - started

        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        ttl = int(ctx.config.get("checkpoint_ttl_days", 30) * 86400)
        path, _ = await training.save_sampler_weights(f"{p['name']}-{stamp}", ttl, tag=p["name"])
        info = await backend.checkpoint_info(path)
        await client.load_adapter(path, tag=p["name"])  # refuses an adapter whose base model is not free

        # A sanity check on held-out generated items. Never a headline number.
        score_ids = sets["pilot"]["score"] if p["set"] == "pilot-train" else [i for ids in sets["dev"].values() for i in ids]
        client.set_concurrency(ctx.config.get("client", {}).get("max_concurrency", 16))
        dev = {}
        for item_id in score_ids:
            dev[item_id] = client.sample([{"role": "user", "content": items[item_id]["prompt"]}], effort=FAST_EFFORT, max_tokens=256, force_answer=True, model_path=path, tag=f"{p['name']}:dev")
        import asyncio

        replies = dict(zip(dev, await asyncio.gather(*dev.values())))
        by_cell: dict[str, list[float]] = {}
        thinking = 0
        for item_id, reply in replies.items():
            by_cell.setdefault(items[item_id]["cell"], []).append(generators.grade_answer(items[item_id], reply.text)["score"])
            thinking += reply.record.thinking_tokens
        tail = losses[-max(1, len(losses) // 10) :]
        return {
            **{k: p[k] for k in ("name", "set", "seed", "rank", "learning_rate", "batch_size", "epochs", "set_fingerprint")},
            "items": len(rows),
            "set_train_tokens": sum(len(tokens) for tokens, _ in examples),
            "train_tokens": sum(len(examples[i][0]) for batch in batches for i in batch),
            "updates": len(batches),
            "train_seconds": round(train_seconds, 1),
            "seconds_per_update": round(train_seconds / len(batches), 2),
            "loss_first": losses[0],
            "loss_last": round(sum(tail) / len(tail), 5),
            "losses": losses,
            "tinker_path": path,
            "checkpoint": info,
            "checkpoint_ttl_days": ctx.config.get("checkpoint_ttl_days", 30),
            "dev_accuracy": {cell: round(sum(scores) / len(scores), 4) for cell, scores in sorted(by_cell.items())},
            "dev_items": len(replies),
            "dev_thinking_tokens": thinking,
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    async def finish(self, ctx: StepContext, rows: list[dict]) -> None:
        runs = [{k: v for k, v in row.items() if k not in ("key", "kind", "params", "losses")} for row in rows]
        ctx.results_dir.mkdir(parents=True, exist_ok=True)
        (ctx.results_dir / "runs.json").write_text(json.dumps({"model_id": ctx.config.get("model_id"), "runs": runs}, indent=1) + "\n", encoding="utf-8")
        lines = ["# Training runs", "", "One adapter per run. Dev accuracy is on held-out generated items in the fast setting and is a sanity check only.", "",
                 "| run | set | seed | rank | learning rate | items | updates | training tokens | s per update | loss first | loss last | dev middle | dev easy | dev hard | checkpoint |",
                 "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in runs:
            dev = r["dev_accuracy"]
            lines.append(
                f"| {r['name']} | {r['set']} | {r['seed']} | {r['rank']} | {r['learning_rate']:g} | {r['items']:,} | {r['updates']} | {r['train_tokens']:,} | "
                f"{r['seconds_per_update']} | {r['loss_first']:.3f} | {r['loss_last']:.3f} | {dev.get('middle', '')} | {dev.get('easy', '')} | {dev.get('hard', '')} | `{r['tinker_path']}` |"
            )
        (ctx.results_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        ctx.out(f"Wrote {ctx.results_dir / 'runs.json'}")
