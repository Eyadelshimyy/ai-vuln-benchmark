from __future__ import annotations

from pathlib import Path

import pytest

from fpga.workspace import workspace_path, workspace_root


def test_workspace_root_uses_project_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OPENHANDS_PROJECT_DIR", str(tmp_path))
    assert workspace_root() == tmp_path.resolve()


def test_workspace_path_accepts_relative_path(tmp_path: Path) -> None:
    assert workspace_path("reports/result.json", tmp_path) == (tmp_path / "reports" / "result.json")


def test_workspace_path_accepts_absolute_path_inside_workspace(tmp_path: Path) -> None:
    candidate = tmp_path / "contracts" / "design.fpga.json"
    assert workspace_path(candidate, tmp_path) == candidate


def test_workspace_path_rejects_parent_traversal(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="path is outside the workspace"):
        workspace_path("../outside.fpga.json", tmp_path)


def test_workspace_path_rejects_absolute_path_outside_workspace(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="path is outside the workspace"):
        workspace_path(tmp_path.parent / "outside.fpga.json", tmp_path)


def test_workspace_path_rejects_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    (tmp_path / "linked").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="workspace path contains a symlink"):
        workspace_path("linked/contract.fpga.json", tmp_path)
