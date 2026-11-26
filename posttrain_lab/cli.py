"""Command line entry point: ``python -m posttrain_lab <step> [options]``."""

from __future__ import annotations

import argparse
import asyncio
import sys
import tomllib
from pathlib import Path
from typing import Callable, Sequence

from .client import ModelClient, RetryPolicy
from .guard import FREE_MODEL_ID, PaidModelError, check_model_id
from .steps import STEPS
from .steps.base import StepContext, run_step
from .store import JsonlStore

REPO_ROOT = Path(__file__).resolve().parent.parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="posttrain-lab",
        description="Post-training experiments on Inkling 2 Mini through Tinker's free model ID.",
    )
    subparsers = parser.add_subparsers(dest="step", metavar="<step>", required=True)
    for step in STEPS.values():
        note = "" if step.implemented else " (not implemented yet)"
        sub = subparsers.add_parser(step.name, help=step.summary + note, description=step.summary + note)
        sub.add_argument("--config", type=Path, help="TOML config file (default: the step's file in configs/)")
        sub.add_argument("--dry-run", action="store_true", help="print planned calls and a token estimate, call nothing")
        sub.add_argument("--fresh", action="store_true", help="rerun units even if their results are stored")
        sub.add_argument("--only", metavar="KIND", help="run only the units of this kind (see the kind column of --dry-run)")
        sub.add_argument("--runs-dir", type=Path, default=REPO_ROOT / "runs", help="raw rows, call logs and the response cache")
        sub.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results", help="small summaries that are committed")
        sub.add_argument("--model-id", default=None, help=f"model ID to call (default: the config's, else {FREE_MODEL_ID})")
        sub.add_argument("--concurrency", type=int, default=None, help="maximum calls in flight")
        sub.add_argument(
            "--allow-paid",
            action="store_true",
            help="permit a model ID that does not end in :free. Calls to it are billed.",
        )
    return parser


def load_config(step, path: Path | None) -> dict:
    if path is None and step.default_config:
        path = REPO_ROOT / step.default_config
    if path is None:
        return {}
    with Path(path).open("rb") as handle:
        return tomllib.load(handle)


def _real_renderer(model_id: str):
    from .backends.tinker_backend import InklingRenderer

    return InklingRenderer(model_id)


def _real_backend(allow_paid: bool):
    from .backends.tinker_backend import TinkerBackend

    return TinkerBackend(allow_paid=allow_paid)


async def _run(args: argparse.Namespace, make_renderer: Callable, make_backend: Callable, out: Callable) -> int:
    step = STEPS[args.step]
    if not step.implemented:
        out(f"{step.name}: not implemented yet")
        return 2
    config = load_config(step, args.config)
    model_id = args.model_id or config.get("model_id") or FREE_MODEL_ID
    try:
        check_model_id(model_id, allow_paid=args.allow_paid)
    except PaidModelError as exc:
        out(f"error: {exc}")
        return 2
    config["model_id"] = model_id

    runs_dir = Path(args.runs_dir) / step.name
    ctx = StepContext(
        step=step.name,
        config=config,
        renderer=make_renderer(model_id),
        runs_dir=runs_dir,
        results_dir=Path(args.results_dir) / step.name,
        store=JsonlStore(runs_dir / "results.jsonl"),
        out=out,
    )
    if args.dry_run:
        return await run_step(step, ctx, dry_run=True, fresh=args.fresh, only=args.only)

    backend = make_backend(args.allow_paid)
    client_config = config.get("client", {})
    ctx.client = ModelClient(
        backend,
        ctx.renderer,
        model_id=model_id,
        allow_paid=args.allow_paid,
        cache_dir=Path(args.runs_dir) / "cache",
        call_log=runs_dir / "calls.jsonl",
        max_concurrency=args.concurrency or client_config.get("max_concurrency", 8),
        retry=RetryPolicy(
            max_attempts=client_config.get("max_attempts", 5),
            call_timeout_s=client_config.get("call_timeout_s", 1800.0),
        ),
        step=step.name,
    )
    try:
        return await run_step(step, ctx, dry_run=False, fresh=args.fresh, only=args.only)
    finally:
        await backend.close()


def main(
    argv: Sequence[str] | None = None,
    *,
    make_renderer: Callable = _real_renderer,
    make_backend: Callable = _real_backend,
    out: Callable[[str], None] = print,
) -> int:
    args = build_parser().parse_args(argv)
    return asyncio.run(_run(args, make_renderer, make_backend, out))


if __name__ == "__main__":
    sys.exit(main())
