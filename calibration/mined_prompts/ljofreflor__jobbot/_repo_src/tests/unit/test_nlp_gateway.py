"""Tests for NL offload gateway — deterministic first, LLM optional."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from jobbot.models.candidate import Candidate
from jobbot.nlp.gateway import (
    LlmOutcome,
    llm_extras_available,
    run_optional_llm,
)
from jobbot.nlp.refine import refine_permanent_profile
from tests.fixtures.profile import sample_profile_dict


def test_run_optional_llm_skips_model_when_disabled() -> None:
    calls = {"llm": 0}

    def det() -> str:
        return "heuristic"

    def llm() -> str:
        calls["llm"] += 1
        return "model"

    out = run_optional_llm(deterministic=det, llm=llm, use_llm=False, task_name="t")
    assert out == LlmOutcome(value="heuristic", mode="deterministic")
    assert calls["llm"] == 0


def test_run_optional_llm_uses_model_when_enabled() -> None:
    out = run_optional_llm(
        deterministic=lambda: "heuristic",
        llm=lambda: "model",
        use_llm=True,
        task_name="t",
    )
    assert out.mode == "llm"
    assert out.value == "model"


def test_run_optional_llm_falls_back_on_failure() -> None:
    def boom() -> str:
        raise RuntimeError("no key")

    out = run_optional_llm(
        deterministic=lambda: "heuristic",
        llm=boom,
        use_llm=True,
        task_name="cover",
    )
    assert out.mode == "llm_fallback"
    assert out.value == "heuristic"
    assert "cover" in out.detail


def test_llm_extras_available_false_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert llm_extras_available() is False


def test_refine_use_llm_falls_back_without_extras() -> None:
    """Promotion path: --llm without extras must not crash; heuristics win."""
    candidate = Candidate.model_validate(sample_profile_dict())
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("OPENAI_API_KEY", None)
        result = refine_permanent_profile(candidate, None, use_llm=True)
    assert result.mode in {"cold", "cumulative"}
    assert result.fields.experiencia_y_perfil
