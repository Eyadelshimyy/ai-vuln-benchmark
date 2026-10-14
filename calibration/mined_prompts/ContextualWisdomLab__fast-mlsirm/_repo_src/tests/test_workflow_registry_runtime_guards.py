"""Regression guards for workflow-registry runtime integrity checks."""

from __future__ import annotations

import ast
from pathlib import Path


def test_workflow_registry_runtime_integrity_does_not_depend_on_assert() -> None:
    """Runtime fail-closed guards must survive ``python -O`` optimization."""
    source_path = (
        Path(__file__).parents[1] / "scripts" / "audit_workflow_registry.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))

    assert not any(isinstance(node, ast.Assert) for node in ast.walk(tree)), (
        "workflow-registry runtime integrity must not depend on assert statements"
    )
