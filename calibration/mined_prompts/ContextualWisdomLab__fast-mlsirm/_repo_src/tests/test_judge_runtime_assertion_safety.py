"""Regression tests for optimization-safe judge runtime validation."""

from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys

import pytest

import fast_mlsirm.judge_calibration as judge_calibration
import fast_mlsirm.llm_judge as llm_judge


@pytest.mark.parametrize("module", [llm_judge, judge_calibration])
def test_judge_production_modules_do_not_use_runtime_assertions(module: object) -> None:
    """Production judge validation cannot disappear under ``python -O``."""
    source_path = Path(module.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))

    assert not [node for node in ast.walk(tree) if isinstance(node, ast.Assert)]


def test_response_schema_validation_survives_python_optimization() -> None:
    """An invalid threshold schema remains fail-closed when assertions are stripped."""
    code = """
from fast_mlsirm.llm_judge import ContextualOrchestratorJudge

try:
    ContextualOrchestratorJudge._response_format(
        "cumulative_threshold",
        ["criterion"],
        None,
    )
except ValueError as exc:
    if str(exc) != "cumulative_threshold requires an explicit category_count":
        raise SystemExit(f"unexpected error: {exc}")
else:
    raise SystemExit("invalid response schema was accepted under python -O")
"""

    completed = subprocess.run(
        [sys.executable, "-O", "-c", code],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
