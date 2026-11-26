"""The spend guard: only free model IDs, and no other provider's SDK."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import posttrain_lab
from posttrain_lab import cli
from posttrain_lab.guard import (
    FORBIDDEN_PROVIDER_MODULES,
    FREE_MODEL_ID,
    PaidModelError,
    check_model_id,
)
from posttrain_lab.testing import FakeBackend

from conftest import run

PAID_ID = "thinkingmachines/Inkling-2-Mini"
QUESTION = [{"role": "user", "content": "is two even"}]


def test_free_id_is_accepted():
    assert check_model_id(FREE_MODEL_ID) == FREE_MODEL_ID


@pytest.mark.parametrize("model_id", [PAID_ID, "thinkingmachines/Inkling-2-Mini:peft:262144", "", "free", ":free:paid"])
def test_other_ids_are_refused(model_id):
    with pytest.raises(PaidModelError):
        check_model_id(model_id)


def test_flag_overrides_the_guard():
    assert check_model_id(PAID_ID, allow_paid=True) == PAID_ID


def test_client_refuses_a_paid_id_before_any_call(make_client, renderer):
    backend = FakeBackend(renderer)
    with pytest.raises(PaidModelError):
        make_client(backend, model_id=PAID_ID)
    assert backend.sample_calls == []


def test_client_accepts_a_paid_id_only_with_the_flag(make_client):
    client, _ = make_client(model_id=PAID_ID, allow_paid=True)
    assert client.model_id == PAID_ID


def test_adapter_on_a_paid_base_model_is_refused(make_client, renderer):
    backend = FakeBackend(renderer)
    backend.adapter_base_model = PAID_ID
    client, _ = make_client(backend)
    with pytest.raises(PaidModelError):
        run(client.sample(QUESTION, max_tokens=8, model_path="tinker://run/sampler_weights/x"))
    assert backend.sample_calls == []


def test_adapter_on_the_free_base_model_is_sampled(make_client):
    client, backend = make_client()
    result = run(client.sample(QUESTION, max_tokens=64, model_path="tinker://run/sampler_weights/x"))
    assert result.text == "yes"
    assert backend.sample_calls[0]["model_path"] == "tinker://run/sampler_weights/x"


def test_cli_refuses_a_paid_id(tmp_path, renderer):
    lines: list[str] = []
    made: list[str] = []

    def make_backend(allow_paid):
        made.append("backend")
        return FakeBackend(renderer)

    code = cli.main(
        ["smoke", "--model-id", PAID_ID, "--runs-dir", str(tmp_path / "runs"), "--results-dir", str(tmp_path / "results")],
        make_renderer=lambda model_id: renderer,
        make_backend=make_backend,
        out=lines.append,
    )
    assert code == 2
    assert made == []
    assert any("Refusing model ID" in line for line in lines)


def test_no_other_provider_sdk_is_imported():
    package_dir = Path(posttrain_lab.__file__).parent
    offenders = []
    for path in package_dir.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module]
            for name in names:
                if any(name == banned or name.startswith(banned + ".") for banned in FORBIDDEN_PROVIDER_MODULES):
                    offenders.append(f"{path.name}: {name}")
    assert offenders == []


def test_only_the_tinker_backend_imports_the_sdk():
    package_dir = Path(posttrain_lab.__file__).parent
    importers = set()
    for path in package_dir.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module]
            if any(n.split(".")[0] in ("tinker", "tinker_cookbook", "tml_renderers") for n in names):
                importers.add(path.name)
    assert importers == {"tinker_backend.py"}
