"""Shared fixtures. Every test runs offline against the fake renderer and backend."""

from __future__ import annotations

import asyncio

import pytest

from posttrain_lab.client import ModelClient, RetryPolicy
from posttrain_lab.testing import FakeBackend, FakeRenderer


async def _no_sleep(seconds: float) -> None:
    await asyncio.sleep(0)


@pytest.fixture
def renderer() -> FakeRenderer:
    return FakeRenderer()


@pytest.fixture
def make_client(tmp_path, renderer):
    """Build a client on a fake backend. Returns (client, backend)."""

    def build(backend: FakeBackend | None = None, **kwargs):
        backend = backend or FakeBackend(renderer)
        kwargs.setdefault("cache_dir", tmp_path / "cache")
        kwargs.setdefault("call_log", tmp_path / "calls.jsonl")
        kwargs.setdefault("retry", RetryPolicy(max_attempts=4, call_timeout_s=5))
        kwargs.setdefault("sleep", _no_sleep)
        kwargs.setdefault("poll_s", 0.01)
        return ModelClient(backend, renderer, **kwargs), backend

    return build


def run(coroutine):
    return asyncio.run(coroutine)
