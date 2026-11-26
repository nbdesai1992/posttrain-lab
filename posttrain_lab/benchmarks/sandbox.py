"""Run untrusted Python in a throwaway Docker container.

Model-written code never runs on the host. Each run gets a fresh container with no
network, a read-only root, no capabilities, an unprivileged user, and limits on
memory, processes and time.
"""

from __future__ import annotations

import subprocess
import uuid
from dataclasses import dataclass

IMAGE = "python:3.11-slim"


@dataclass
class RunResult:
    passed: bool
    status: str  # "ok", "failed", "timeout" or "sandbox_error"
    detail: str = ""


def docker_available() -> bool:
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=20).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def run_python(code: str, *, timeout_s: int = 15, memory: str = "512m", image: str = IMAGE) -> RunResult:
    """Run ``code`` with ``python -`` inside a locked-down container. Passes if it exits 0."""
    name = f"posttrain-lab-{uuid.uuid4().hex[:12]}"
    command = [
        "docker", "run", "--rm", "-i", "--name", name,
        "--network", "none", "--read-only", "--tmpfs", "/tmp:rw,size=64m",
        "--memory", memory, "--memory-swap", memory, "--cpus", "1", "--pids-limit", "128",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--user", "65534:65534",
        "--workdir", "/tmp", "--env", "PYTHONDONTWRITEBYTECODE=1",
        image, "timeout", "--signal=KILL", str(timeout_s), "python", "-I", "-",
    ]
    try:
        done = subprocess.run(command, input=code.encode("utf-8"), capture_output=True, timeout=timeout_s + 30)
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "kill", name], capture_output=True)
        return RunResult(False, "timeout", "the container did not finish in time")
    except OSError as exc:
        return RunResult(False, "sandbox_error", str(exc))
    stderr = done.stderr.decode("utf-8", "replace")[-400:]
    if done.returncode == 0:
        return RunResult(True, "ok")
    if done.returncode == 137:
        return RunResult(False, "timeout", "killed after the time or memory limit")
    if done.returncode in (125, 126, 127):
        return RunResult(False, "sandbox_error", stderr)
    return RunResult(False, "failed", stderr)
