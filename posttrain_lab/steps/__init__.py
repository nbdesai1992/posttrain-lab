"""The registry of steps, in the order they run."""

from __future__ import annotations

from .base import Step
from .build_data import BuildData
from .filter import Filter
from .smoke import Smoke
from .evaluate import Eval
from .stubs import Report
from .survey import Survey
from .train import Train

STEPS: dict[str, Step] = {
    step.name: step for step in (Smoke(), Survey(), BuildData(), Filter(), Train(), Eval(), Report())
}
