"""Steps that are planned but not written yet. Each one names what it will do."""

from __future__ import annotations

from .base import NotImplementedStep


class Report(NotImplementedStep):
    name = "report"
    summary = "Build the results tables and figures from stored rows"
