"""The convention every step follows.

A step is a subcommand. It reads a config, plans its work as a list of units, and
runs the units that are not done yet. Each finished unit appends one row to
``runs/<step>/results.jsonl``, keyed by a hash of the unit's inputs, so a rerun picks
up where the last one stopped. ``--dry-run`` prints the plan (units, calls, a token
estimate) and makes no API call.

To add a step: subclass ``Step``, implement ``plan`` and ``run_unit`` (and ``finish``
if it writes a summary), and register it in ``steps/__init__.py``.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..client import AccountBlockedError, ModelClient
from ..render import Renderer
from ..store import JsonlStore, stable_hash


@dataclass
class Unit:
    """One resumable piece of work.

    ``params`` must hold everything that determines the result, because the unit's
    key is a hash of it. The counts are estimates for the dry run.
    """

    kind: str
    params: dict
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0  # expected
    max_completion_tokens: int = 0  # upper bound, if every call ran to max_tokens
    training_tokens: int = 0
    label: str = ""

    def key(self, step: str) -> str:
        return stable_hash({"step": step, "kind": self.kind, "params": self.params})


@dataclass
class StepContext:
    step: str
    config: dict
    renderer: Renderer
    runs_dir: Path  # runs/<step>: raw rows, call log
    results_dir: Path  # results/<step>: small summaries that are committed
    store: JsonlStore
    client: ModelClient | None = None  # None during a dry run
    out: Callable[[str], None] = print
    scratch: dict = field(default_factory=dict)  # shared between a step's units

    def result(self, unit: Unit) -> dict | None:
        """The stored row for a unit, if it has been run."""
        return self.store.get(unit.key(self.step))


class Step:
    name: str = ""
    summary: str = ""
    default_config: str | None = None  # path relative to the repository root
    implemented: bool = True
    parallel_units: bool = False  # run pending units concurrently (the client bounds the calls)
    unit_workers: int | None = None  # how many units run at once; defaults to the client's concurrency

    def plan(self, ctx: StepContext) -> list[Unit]:
        raise NotImplementedError

    async def run_unit(self, unit: Unit, ctx: StepContext) -> dict:
        raise NotImplementedError

    async def finish(self, ctx: StepContext, rows: list[dict]) -> None:
        """Called at the end of every full run, with the stored row of each planned
        unit in plan order. The place to write a summary into ``ctx.results_dir``."""


class NotImplementedStep(Step):
    """A step that is planned but not written yet."""

    implemented = False

    def plan(self, ctx: StepContext) -> list[Unit]:
        return []


def format_plan(step: Step, units: list[Unit], pending: list[Unit]) -> str:
    """The dry-run table: per kind of unit, how much work is planned and how much is left."""
    pending_ids = {id(u) for u in pending}
    kinds: dict[str, dict[str, int]] = {}
    for unit in units:
        row = kinds.setdefault(
            unit.kind,
            {"units": 0, "pending": 0, "calls": 0, "prompt": 0, "completion": 0, "max": 0, "training": 0},
        )
        row["units"] += 1
        if id(unit) in pending_ids:
            row["pending"] += 1
            row["calls"] += unit.calls
            row["prompt"] += unit.prompt_tokens
            row["completion"] += unit.completion_tokens
            row["max"] += unit.max_completion_tokens
            row["training"] += unit.training_tokens
    total = {k: sum(row[k] for row in kinds.values()) for k in ("units", "pending", "calls", "prompt", "completion", "max", "training")}
    header = (
        f"{'kind':<22}{'units':>6}{'pending':>8}{'calls':>7}"
        f"{'prompt tok':>13}{'completion tok':>16}{'(max)':>14}{'training tok':>14}"
    )
    lines = [f"{step.name}: plan for the pending work (token counts are estimates)", header, "-" * len(header)]
    for kind, row in [*kinds.items(), ("total", total)]:
        lines.append(
            f"{kind:<22}{row['units']:>6}{row['pending']:>8}{row['calls']:>7}"
            f"{row['prompt']:>13,}{row['completion']:>16,}{row['max']:>14,}{row['training']:>14,}"
        )
    return "\n".join(lines)


async def run_step(
    step: Step, ctx: StepContext, *, dry_run: bool = False, fresh: bool = False, only: str | None = None
) -> int:
    """Plan a step and run whatever is pending. Returns a process exit code.

    ``only`` limits the run to units of one kind. Together with ``fresh`` it reruns
    just those units."""
    if not step.implemented:
        ctx.out(f"{step.name}: not implemented yet")
        return 2
    units = step.plan(ctx)
    keys = [unit.key(ctx.step) for unit in units]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{step.name}: two planned units have the same inputs")
    if only is not None and only not in {u.kind for u in units}:
        raise ValueError(f"{step.name} plans no unit of kind {only!r}")
    pending = [
        u
        for u, key in zip(units, keys)
        if (only is None or u.kind == only) and (fresh or key not in ctx.store)
    ]
    for line in format_plan(step, units, pending).splitlines():
        ctx.out(line)
    if dry_run:
        ctx.out("Dry run: no API calls were made.")
        return 0
    if ctx.client is None:
        raise RuntimeError("A model client is required outside a dry run")

    started = time.monotonic()

    async def run_one(unit: Unit) -> None:
        row = await step.run_unit(unit, ctx)
        ctx.store.put(unit.key(ctx.step), {"kind": unit.kind, "params": unit.params, **row})

    failures: list[BaseException] = []
    try:
        if step.parallel_units:
            # A fixed number of workers, so that only the units in progress hold their
            # prompts in memory. Each worker makes one call at a time.
            queue = list(reversed(pending))

            async def worker() -> None:
                while queue and not any(isinstance(f, AccountBlockedError) for f in failures):
                    unit = queue.pop()
                    try:
                        await run_one(unit)
                    except AccountBlockedError as exc:
                        failures.append(exc)
                    except Exception as exc:  # the unit stays pending for the next run
                        failures.append(exc)

            workers = step.unit_workers or ctx.client.max_concurrency
            await asyncio.gather(*(worker() for _ in range(min(workers, max(len(pending), 1)))))
            for failure in failures:
                if isinstance(failure, AccountBlockedError):
                    raise failure
        else:
            for index, unit in enumerate(pending, start=1):
                ctx.out(f"[{index}/{len(pending)}] {unit.kind} {unit.label}".rstrip())
                await run_one(unit)
    except AccountBlockedError as exc:
        ctx.out(f"Stopped on an account problem: {exc}")
        ctx.out("Finished units are saved. Once the account is sorted out, rerun the same command.")
        return 3
    if failures:
        kinds: dict[str, int] = {}
        for failure in failures:
            kinds[type(failure).__name__] = kinds.get(type(failure).__name__, 0) + 1
        ctx.out(f"{len(failures)} units failed and stay pending for the next run: {kinds}")
        ctx.out(f"first failure: {type(failures[0]).__name__}: {str(failures[0])[:300]}")

    rows = [ctx.store.get(key) for key in keys]
    await step.finish(ctx, [row for row in rows if row is not None])
    totals = ctx.client.totals
    ctx.out(
        f"{step.name}: done in {time.monotonic() - started:.0f} s. "
        f"{totals['api_calls']} API calls, {totals['cache_hits']} cache hits, "
        f"{totals['prompt_tokens']:,} prompt tokens, {totals['completion_tokens']:,} completion tokens, "
        f"{totals['training_tokens']:,} training tokens."
    )
    return 1 if failures else 0
