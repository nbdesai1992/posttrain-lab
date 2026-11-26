"""Steps that are planned but not written yet. Each one names what it will do."""

from __future__ import annotations

from .base import NotImplementedStep


class Train(NotImplementedStep):
    name = "train"
    summary = "Train LoRA adapters on the free model ID"


class Eval(NotImplementedStep):
    name = "eval"
    summary = "Rerun every survey benchmark for every adapter"


class Report(NotImplementedStep):
    name = "report"
    summary = "Build the results tables and figures from stored rows"
