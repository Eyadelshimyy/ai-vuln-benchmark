"""chrome-debug must refuse to launch over a profile another Chrome already holds."""

from __future__ import annotations

from pathlib import Path

import pytest

from jobbot.browser.sessions import ChromeProcess, ProfileBusyError
from jobbot.exit_codes import GENERIC_FAILURE, SUCCESS


def _workspace(tmp_path: Path, project_root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "data" / "profile.yaml").write_text(
        (project_root / "data" / "profile.example.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    return tmp_path / "browser-data" / "indeed-cdp"


def test_chrome_debug_print_only_does_not_need_a_free_profile(
    tmp_path: Path, project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from jobbot.cli import run_cli

    _workspace(tmp_path, project_root, monkeypatch)
    launched: list[object] = []
    monkeypatch.setattr("subprocess.Popen", lambda *a, **k: launched.append((a, k)))

    def boom(profile_dir: Path, **_: object) -> None:
        raise ProfileBusyError(f"{profile_dir} busy")

    monkeypatch.setattr("jobbot.browser.sessions.ensure_profile_free", boom)

    code = run_cli(
        ["browser", "chrome-debug", "--site", "indeed", "--print-only"],
        standalone_mode=False,
    )
    assert code == SUCCESS
    assert launched == []


def test_chrome_debug_launch_refuses_a_busy_profile(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from jobbot.browser.sessions import ensure_profile_free
    from jobbot.cli import run_cli

    profile_dir = _workspace(tmp_path, project_root, monkeypatch)
    profile_dir.mkdir(parents=True, exist_ok=True)
    launched: list[object] = []
    monkeypatch.setattr("subprocess.Popen", lambda *a, **k: launched.append((a, k)))

    busy = [
        ChromeProcess(
            pid=4242,
            profile_dir=profile_dir,
            cdp_port=None,
        )
    ]
    # Prove the shared guard sees the holder, then wire it through the CLI path.
    ensure_profile_free(profile_dir, processes=[])
    with pytest.raises(ProfileBusyError, match="4242"):
        ensure_profile_free(profile_dir, processes=busy)

    monkeypatch.setattr(
        "jobbot.browser.sessions.list_chrome_processes",
        lambda: busy,
    )

    code = run_cli(
        ["browser", "chrome-debug", "--site", "indeed", "--port", "9224"],
        standalone_mode=False,
    )
    err = capsys.readouterr().err
    assert code == GENERIC_FAILURE
    assert launched == [], "must not Popen a second Chrome on the same profile"
    assert "4242" in err or "already open" in err.casefold() or "busy" in err.casefold()
