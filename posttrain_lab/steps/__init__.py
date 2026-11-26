"""The registry of steps, in the order they run."""

from __future__ import annotations

from .base import Step
from .smoke import Smoke
from .stubs import BuildData, Eval, Filter, Report, Survey, Train

STEPS: dict[str, Step] = {
    step.name: step for step in (Smoke(), Survey(), BuildData(), Filter(), Train(), Eval(), Report())
}
