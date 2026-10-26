"""Several candidates must coexist in one checkout without ever bleeding into each other.

Isolation used to rest on the current directory alone: running a command from the wrong
folder silently wrote one person's CV over another's `output/jobs/J0001/`, since every
workspace numbers its jobs from J0001. A workspace is now picked explicitly and every
`data/` + `output/` pair is stamped with a fingerprint of its owner, so a profile can
never be used against artifacts that belong to somebody else.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import jobbot.config as config_module
from jobbot.config import load_config
from jobbot.exit_codes import SUCCESS
from jobbot.workspace import (
    STAMP_NAME,
    WorkspaceOwnerError,
    owner_fingerprint,
    read_stamp,
    write_stamp,
)


def _profile(path: Path, name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"personal:\n  name: {name}\n  email: alguien@example.com\n",
        encoding="utf-8",
    )


@pytest.fixture
def sandboxes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Workspaces live under a directory we can point somewhere safe in tests."""
    root = tmp_path / "sandboxes"
    root.mkdir()
    monkeypatch.setenv("JOBBOT_SANDBOXES", str(root))
    return root


def test_selected_workspace_keeps_every_path_inside_its_own_dir(
    sandboxes: Path, project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The workspace wins over the current directory, which is the repo root here."""
    _profile(sandboxes / "rocio" / "data" / "profile.yaml", "Rocío Paredes")
    monkeypatch.setenv("JOBBOT_WORKSPACE", "rocio")
    monkeypatch.chdir(project_root)

    config = load_config()

    root = sandboxes / "rocio"
    assert config.profile_path == root / "data" / "profile.yaml"
    assert config.output_dir == root / "output"
    assert config.database_path == root / "data" / "jobbot.sqlite"
    # templates are code, not data: a workspace borrows the checkout's own
    assert config.templates_dir == project_root / "templates"


def test_selected_workspace_ignores_the_users_global_config(
    sandboxes: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A global config with absolute paths used to reach into the real profile."""
    other = tmp_path / "elsewhere" / "profile.yaml"
    _profile(other, "Otra Persona")
    global_config = tmp_path / "config.toml"
    global_config.write_text(f'[paths]\nprofile = "{other}"\n', encoding="utf-8")
    monkeypatch.setattr(config_module, "_user_config_path", lambda: global_config)

    _profile(sandboxes / "rocio" / "data" / "profile.yaml", "Rocío Paredes")
    monkeypatch.setenv("JOBBOT_WORKSPACE", "rocio")

    config = load_config()

    assert config.profile_path == sandboxes / "rocio" / "data" / "profile.yaml"


def test_first_run_stamps_the_workspace_without_writing_a_name(
    sandboxes: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stamp carries a fingerprint, never the person it belongs to."""
    _profile(sandboxes / "rocio" / "data" / "profile.yaml", "Rocío Paredes")
    monkeypatch.setenv("JOBBOT_WORKSPACE", "rocio")

    load_config()

    stamp_path = sandboxes / "rocio" / "data" / STAMP_NAME
    stamp = read_stamp(stamp_path)
    assert stamp is not None
    assert stamp.label == "rocio"
    assert stamp.fingerprint == owner_fingerprint("Rocío Paredes")
    assert "Rocío" not in stamp_path.read_text(encoding="utf-8")
    assert "Paredes" not in stamp_path.read_text(encoding="utf-8")


def test_another_persons_profile_cannot_reuse_a_workspace(
    sandboxes: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dropping a test CV over an existing profile.yaml must fail loudly."""
    root = sandboxes / "rocio"
    _profile(root / "data" / "profile.yaml", "Rocío Paredes")
    monkeypatch.setenv("JOBBOT_WORKSPACE", "rocio")
    load_config()

    _profile(root / "data" / "profile.yaml", "Jorge Pinto")

    with pytest.raises(WorkspaceOwnerError):
        load_config()


def test_output_belonging_to_someone_else_blocks_the_run(
    sandboxes: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Copying artifacts between workspaces is the other way they mix."""
    root = sandboxes / "rocio"
    _profile(root / "data" / "profile.yaml", "Rocío Paredes")
    monkeypatch.setenv("JOBBOT_WORKSPACE", "rocio")
    load_config()

    output_stamp = root / "output" / STAMP_NAME
    output_stamp.parent.mkdir(parents=True, exist_ok=True)
    write_stamp(output_stamp, label="jorge", fingerprint=owner_fingerprint("Jorge Pinto"))

    with pytest.raises(WorkspaceOwnerError):
        load_config()


def test_a_directory_without_a_workspace_still_works(
    sandboxes: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No workspace selected keeps the old behaviour: the current directory."""
    plain = tmp_path / "plain"
    _profile(plain / "data" / "profile.yaml", "Rocío Paredes")
    monkeypatch.chdir(plain)

    config = load_config()

    assert config.profile_path == plain / "data" / "profile.yaml"


def test_workspace_new_and_list_from_the_cli(
    sandboxes: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`workspace new` prepares an isolated home for a test CV; `list` shows it."""
    from jobbot.cli import run_cli

    monkeypatch.chdir(project_root)

    assert run_cli(["workspace", "new", "rocio"], standalone_mode=False) == SUCCESS
    assert (sandboxes / "rocio" / "data").is_dir()

    capsys.readouterr()
    assert run_cli(["workspace", "list"], standalone_mode=False) == SUCCESS
    assert "rocio" in capsys.readouterr().out


def test_workspace_flag_runs_a_command_against_the_test_cv(
    sandboxes: Path, project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """From the repo root, `--workspace` must validate the test CV, not the real one."""
    from jobbot.cli import run_cli

    monkeypatch.chdir(project_root)
    assert run_cli(["workspace", "new", "rocio"], standalone_mode=False) == SUCCESS
    (sandboxes / "rocio" / "data" / "profile.yaml").write_text(
        (project_root / "data" / "profile.example.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    assert (
        run_cli(["--workspace", "rocio", "profile", "validate"], standalone_mode=False) == SUCCESS
    )
