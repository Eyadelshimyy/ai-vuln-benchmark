"""Cold-install workspace: discovery, init, .local paths."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from jobbot.config import CONFIG_FILENAME, load_config, resolve_workspace
from jobbot.workspace import (
    GITIGNORE_FILENAME,
    INIT_GITIGNORE_LINES,
    WorkspaceExistsError,
    init_workspace,
)
from tests.fixtures.cv_pdf import write_sample_cv


def test_resolve_workspace_walks_up_to_jobbot_toml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "postulaciones"
    nested = root / "notas" / "hoy"
    nested.mkdir(parents=True)
    (root / CONFIG_FILENAME).write_text("[search]\ncountries = [\"CL\"]\n", encoding="utf-8")
    monkeypatch.delenv("JOBBOT_ROOT", raising=False)
    monkeypatch.chdir(nested)
    assert resolve_workspace() == root.resolve()


def test_resolve_workspace_honours_jobbot_root_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.setenv("JOBBOT_ROOT", str(elsewhere))
    monkeypatch.chdir(tmp_path)
    assert resolve_workspace() == elsewhere.resolve()


def test_init_creates_local_layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "postulaciones"
    result = init_workspace(target)
    assert result.root == target.resolve()
    assert (target / CONFIG_FILENAME).is_file()
    assert (target / ".local" / "profile.yaml").is_file()
    assert (target / ".local" / "portals.yaml").is_file()
    assert (target / ".local" / "companies.yaml").is_file()
    assert (target / ".local" / "templates" / "cv_ats.txt.j2").is_file()
    assert (target / ".local" / "output").is_dir()
    assert (target / ".local" / "browser-data").is_dir()

    monkeypatch.delenv("JOBBOT_ROOT", raising=False)
    monkeypatch.chdir(target)
    config = load_config()
    assert config.root == target.resolve()
    assert config.profile_path == (target / ".local" / "profile.yaml").resolve()
    assert config.database_path == (target / ".local" / "jobbot.sqlite").resolve()
    assert config.output_dir == (target / ".local" / "output").resolve()
    assert config.portals_path == (target / ".local" / "portals.yaml").resolve()
    assert config.companies_path == (target / ".local" / "companies.yaml").resolve()
    assert config.browser_data_dir == (target / ".local" / "browser-data").resolve()


def test_init_writes_gitignore_with_local_and_tmp(tmp_path: Path) -> None:
    """#212: private state and disposable scaffolding must be ignored by default."""
    target = tmp_path / "postulaciones"
    result = init_workspace(target)
    gitignore = target / GITIGNORE_FILENAME
    assert gitignore.is_file()
    assert gitignore in result.created
    lines = {line.strip() for line in gitignore.read_text(encoding="utf-8").splitlines()}
    assert set(INIT_GITIGNORE_LINES) <= lines


def test_init_merges_gitignore_without_duplicating(tmp_path: Path) -> None:
    target = tmp_path / "ws"
    target.mkdir()
    existing = target / GITIGNORE_FILENAME
    existing.write_text("# mine\n*.pdf\n.local/\n", encoding="utf-8")
    init_workspace(target)
    text = existing.read_text(encoding="utf-8")
    assert text.startswith("# mine\n*.pdf\n.local/\n")
    assert text.count(".local/") == 1
    assert text.count("tmp/") == 1
    assert "tmp/" in text.splitlines()

    before = existing.read_text(encoding="utf-8")
    init_workspace(target, force=True)
    assert existing.read_text(encoding="utf-8") == before


def test_init_gitignore_is_honoured_by_git(tmp_path: Path) -> None:
    target = tmp_path / "ws"
    init_workspace(target)
    subprocess.run(["git", "init"], cwd=target, check=True, capture_output=True)
    (target / "tmp").mkdir()
    (target / "tmp" / "x").write_text("probe", encoding="utf-8")
    (target / ".local" / "x").write_text("probe", encoding="utf-8")
    checked = subprocess.run(
        ["git", "check-ignore", "tmp/x", ".local/x"],
        cwd=target,
        check=True,
        capture_output=True,
        text=True,
    )
    ignored = {line.strip() for line in checked.stdout.splitlines() if line.strip()}
    assert ignored == {"tmp/x", ".local/x"}


def test_init_local_under_absolute_target_when_cwd_differs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``.local`` must land in DIR even when cwd is somewhere else."""
    cwd = tmp_path / "elsewhere"
    target = tmp_path / "postulaciones" / "ws"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.delenv("JOBBOT_ROOT", raising=False)

    result = init_workspace(target)

    assert result.root == target.resolve()
    assert (target / CONFIG_FILENAME).is_file()
    assert (target / ".local" / "profile.yaml").is_file()
    assert not (cwd / ".local").exists()
    assert not (cwd / CONFIG_FILENAME).exists()
    assert not (tmp_path / ".local").exists()


def test_init_dot_uses_cwd_not_home_or_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "invoked-here"
    decoy_home = tmp_path / "fake-home"
    decoy_root = tmp_path / "jobbot-root-env"
    workspace.mkdir()
    decoy_home.mkdir()
    decoy_root.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.setenv("HOME", str(decoy_home))
    monkeypatch.setenv("JOBBOT_ROOT", str(decoy_root))

    result = init_workspace(Path("."))

    assert result.root == workspace.resolve()
    assert (workspace / ".local" / "profile.yaml").is_file()
    assert (workspace / CONFIG_FILENAME).is_file()
    assert not (decoy_home / ".local").exists()
    assert not (decoy_root / ".local").exists()
    assert not (decoy_root / CONFIG_FILENAME).exists()


def test_cli_init_writes_local_inside_directory_arg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from typer.testing import CliRunner

    from jobbot.cli import app

    cwd = tmp_path / "cwd"
    target = tmp_path / "foo"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.delenv("JOBBOT_ROOT", raising=False)

    runner = CliRunner()
    outcome = runner.invoke(app, ["init", str(target)])
    assert outcome.exit_code == 0, outcome.stdout
    assert (target / CONFIG_FILENAME).is_file()
    assert (target / ".local" / "profile.yaml").is_file()
    assert not (cwd / ".local").exists()


def test_init_refuses_overwrite_without_force(tmp_path: Path) -> None:
    target = tmp_path / "ws"
    init_workspace(target)
    with pytest.raises(WorkspaceExistsError):
        init_workspace(target)


def test_legacy_cwd_without_toml_keeps_data_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("JOBBOT_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    config = load_config()
    assert config.root == tmp_path.resolve()
    assert config.profile_path == (tmp_path / "data" / "profile.yaml").resolve()
    assert config.output_dir == (tmp_path / "output").resolve()
    assert "JOBBOT_ROOT" not in os.environ


def test_import_pdf_promote_restamps_owner_for_cold_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """init seeds Ana Ejemplo; --promote must re-stamp so validate can load the new CV."""
    from typer.testing import CliRunner

    from jobbot.cli import app
    from jobbot.workspace import STAMP_NAME, owner_fingerprint, read_stamp

    target = tmp_path / "postulaciones"
    init_workspace(target)
    pdf = write_sample_cv(tmp_path / "cv.pdf")
    monkeypatch.delenv("JOBBOT_WORKSPACE", raising=False)
    monkeypatch.delenv("JOBBOT_ROOT", raising=False)
    monkeypatch.setenv("JOBBOT_ROOT", str(target))
    monkeypatch.chdir(target)

    # First command stamps .local for the seed profile (Ana Ejemplo).
    load_config()
    seed_stamp = read_stamp(target / ".local" / STAMP_NAME)
    assert seed_stamp is not None
    assert seed_stamp.fingerprint == owner_fingerprint("Ana Ejemplo")
    runner = CliRunner()
    outcome = runner.invoke(app, ["profile", "import-pdf", str(pdf), "--promote"])
    assert outcome.exit_code == 0, outcome.output

    stamp = read_stamp(target / ".local" / STAMP_NAME)
    assert stamp is not None
    assert stamp.fingerprint != owner_fingerprint("Ana Ejemplo")

    validate = runner.invoke(app, ["profile", "validate"])
    assert validate.exit_code == 0, validate.output
    assert "Wrong workspace" not in validate.output
