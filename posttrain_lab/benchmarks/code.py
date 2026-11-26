"""MBPP: short Python tasks graded by running the benchmark's own assertions."""

from __future__ import annotations

from pathlib import Path

from . import sandbox
from .base import Benchmark, Item, read_parquet, strip_code_fence, user


class MBPP(Benchmark):
    name = "mbpp"
    title = "MBPP (sanitized test split)"
    job = "Reasoning without the wait"
    role = "candidate"
    graded_by = "the benchmark's unit tests, run in a container with no network"
    source = "mbpp"
    max_tokens_fast = 1024
    parallel_grading = True  # each reply runs in its own container

    PROMPT = (
        "{task}\n\nYour code must pass these tests:\n{tests}\n\n"
        "Reply with the complete Python code in a single ```python code block and nothing else."
    )

    def load(self, root: Path, seed: int) -> list[Item]:
        items = []
        for row in read_parquet(root / self.source / "sanitized/test-00000-of-00001.parquet"):
            tests = list(row["test_list"])
            items.append(
                Item(
                    id=f"mbpp-{row['task_id']}",
                    messages=user(self.PROMPT.format(task=row["prompt"], tests="\n".join(tests))),
                    answer={"tests": tests, "imports": list(row["test_imports"] or [])},
                )
            )
        return items

    def program(self, item: Item, reply: dict) -> str:
        code = strip_code_fence(reply.get("text", ""), "python")
        return "\n".join([*item.answer["imports"], code, "", *item.answer["tests"], ""])

    def prepare(self, root: Path) -> None:
        if not sandbox.docker_available():
            raise RuntimeError("MBPP is graded in Docker, and the Docker daemon is not reachable")

    def grade(self, item: Item, reply: dict) -> dict:
        if not reply.get("text", "").strip():
            return {"score": 0.0, "status": "empty"}
        result = sandbox.run_python(self.program(item, reply))
        if result.status == "sandbox_error":
            raise RuntimeError(f"The sandbox failed, so this reply was not graded: {result.detail}")
        return {"score": float(result.passed), "status": result.status, "detail": result.detail[-200:]}

    def extra_metrics(self, items: list[Item], grades: list[dict]) -> dict:
        statuses: dict[str, int] = {}
        for grade in grades:
            statuses[grade.get("status", "?")] = statuses.get(grade.get("status", "?"), 0) + 1
        return {"outcomes": statuses}
