"""Public benchmarks the study is scored on. Each is graded by a program."""

from __future__ import annotations

from .base import Benchmark, Item
from .bfcl import BFCL, BFCLMultiTurn
from .code import MBPP
from .ifeval import IFEval
from .simple import GSM8K, JSONSchemaBench, JudgeBench, JudgeBenchPointwise, LLMBar, MMLUPro, RewardBench2

BENCHMARKS: dict[str, Benchmark] = {
    benchmark.name: benchmark
    for benchmark in (
        JudgeBench(), JudgeBenchPointwise(), RewardBench2(), LLMBar(), BFCL(), BFCLMultiTurn(),
        JSONSchemaBench(), GSM8K(), MBPP(), IFEval(), MMLUPro(),
    )
}

# Benchmarks in the study design that this repository cannot run without a login.
NOT_RUN = {
    "llm_aggrefact": {
        "title": "LLM-AggreFact", "job": "Checking claims against a source",
        "reason": "not run, needs a token: the dataset is gated on Hugging Face (licence CC-BY-ND-4.0)",
    },
    "wildguardtest": {
        "title": "WildGuardTest", "job": "Screening for harm",
        "reason": "not run, needs a token: the dataset is gated on Hugging Face (licence ODC-BY)",
    },
}

__all__ = ["BENCHMARKS", "NOT_RUN", "Benchmark", "Item"]
