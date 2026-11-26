"""The registry of steps, in the order they run."""

from __future__ import annotations

from .base import Step
from .build_data import BuildData
from .filter import Filter
from .smoke import Smoke
from .stubs import Eval, Report, Train
from .survey import Survey

STEPS: dict[str, Step] = {
    step.name: step for step in (Smoke(), Survey(), BuildData(), Filter(), Train(), Eval(), Report())
}
