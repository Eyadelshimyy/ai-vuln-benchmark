"""Security regressions for the sales-readiness CLI output boundary."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_sales_readiness():
    """Load the repository script as a module for boundary tests."""
    script = Path(__file__).resolve().parents[1] / "scripts" / "sales_readiness.py"
    spec = importlib.util.spec_from_file_location("sales_readiness_output_boundary", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_cli_output_path_rejects_repository_escape(tmp_path: Path) -> None:
    """An absolute CLI output path cannot escape the declared repository root."""
    module = _load_sales_readiness()
    repo_root = tmp_path / "repository"
    repo_root.mkdir()
    outside = tmp_path / "outside" / "sales_readiness_manifest.json"

    with pytest.raises(ValueError, match="must stay within --repo-root"):
        module._resolve_cli_output_path(str(outside), repo_root)


def test_cli_output_path_anchors_relative_path_to_repository(tmp_path: Path) -> None:
    """Relative CLI output paths are anchored to the trusted repository root."""
    module = _load_sales_readiness()
    repo_root = tmp_path / "repository"
    repo_root.mkdir()

    resolved = module._resolve_cli_output_path(
        "release-acceptance/sales_readiness_manifest.json",
        repo_root,
    )

    assert resolved == repo_root / "release-acceptance" / "sales_readiness_manifest.json"
