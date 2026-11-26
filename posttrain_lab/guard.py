"""Spend guard.

Every model call in this repository goes through ``check_model_id`` first. Tinker's
free model IDs end in ``:free``; anything else costs money and is refused unless the
caller passes the explicit override.
"""

from __future__ import annotations

FREE_MODEL_ID = "thinkingmachines/Inkling-2-Mini:peft:65536:free"
FREE_SUFFIX = ":free"

# SDKs for other model providers. The study runs on one free key, so nothing in the
# package may import these. tests/test_guard.py enforces it.
FORBIDDEN_PROVIDER_MODULES = (
    "anthropic",
    "openai",
    "google.generativeai",
    "google.genai",
    "vertexai",
    "cohere",
    "mistralai",
    "together",
    "groq",
    "litellm",
)


class PaidModelError(RuntimeError):
    """Raised when a call would use a model ID that is not on the free tier."""


def is_free(model_id: str) -> bool:
    return isinstance(model_id, str) and model_id.endswith(FREE_SUFFIX)


def check_model_id(model_id: str, *, allow_paid: bool = False) -> str:
    """Return ``model_id`` if it may be called, otherwise raise ``PaidModelError``."""
    if is_free(model_id):
        return model_id
    if allow_paid:
        return model_id
    raise PaidModelError(
        f"Refusing model ID {model_id!r}: it does not end in {FREE_SUFFIX!r}, so calls to it "
        f"are billed. The free ID is {FREE_MODEL_ID!r}. Pass --allow-paid to override."
    )
