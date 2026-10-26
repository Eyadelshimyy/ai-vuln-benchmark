"""Committed architecture diagrams stay present for the README."""

from __future__ import annotations

from pathlib import Path


def test_architecture_pngs_are_committed(project_root: Path) -> None:
    """README embeds these; regenerating needs Graphviz, but the PNGs must ship."""
    images = project_root / "docs" / "images"
    for name in ("architecture.png", "architecture-detail.png"):
        path = images / name
        assert path.is_file(), f"missing {path} — run `make architecture`"
        assert path.stat().st_size > 10_000, f"{path} looks empty"


def test_render_architecture_script_exists(project_root: Path) -> None:
    script = project_root / "scripts" / "render_architecture.py"
    assert script.is_file()
    text = script.read_text(encoding="utf-8")
    assert "from diagrams import" in text
    assert "ops_failures" in text
