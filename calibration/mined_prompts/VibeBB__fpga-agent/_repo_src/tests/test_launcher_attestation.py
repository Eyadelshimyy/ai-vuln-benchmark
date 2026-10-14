"""Launcher-side image provenance verification tests."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_PATH = next((REPO_ROOT / "plugins").rglob("*_launcher.py"))
IMAGE_NAMES = {
    "wire-agent": "wire",
    "firmware-agent": "firmware",
    "UX-creator-agent": "ux",
    "mechanical-agent": "mech",
    "simulation-agent": "sim",
    "electrical-circuit-agent": "circuit",
    "production-engineering-agent": "prodeng",
    "fpga-agent": "fpga",
    "dashboard-agent": "dashboard",
}


@pytest.fixture()
def launcher() -> Any:
    spec = importlib.util.spec_from_file_location("launcher_under_test", LAUNCHER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["launcher_under_test"] = module
    spec.loader.exec_module(module)
    return module


def _pin(module: Any, *, attestation: str | None = None) -> dict[str, str | None]:
    repo = module._REPOSITORY
    name = IMAGE_NAMES[repo.removeprefix("VibeBB/")]
    image = f"ghcr.io/vibebb/{name}-tools"
    digest = "sha256:" + "a" * 64
    return {
        "ref": f"{image}@{digest}",
        "image": image,
        "digest": digest,
        "attestation": attestation
        if attestation is not None
        else f"https://github.com/{repo}/attestations/1",
    }


def _tool_path(name: str) -> str:
    return f"/usr/bin/{name}"


def _without_gh(name: str) -> str | None:
    return None if name == "gh" else _tool_path(name)


def _unexpected_tool_lookup(_name: str) -> str | None:
    pytest.fail("unexpected tool lookup")


def _unexpected_subprocess(*_args: Any, **_kwargs: Any) -> Any:
    pytest.fail("unexpected subprocess call")


def _runner(
    module: Any,
    monkeypatch: pytest.MonkeyPatch,
    *,
    inspect_code: int = 0,
    auth_code: int = 0,
    verify_code: int = 0,
    timeout_verify: bool = False,
) -> list[tuple[list[str], dict[str, Any]]]:
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def run(command: list[str], **kwargs: Any) -> Any:
        calls.append((command, kwargs))
        if command[0].endswith("/gh"):
            if command[1:3] == ["auth", "status"]:
                return module.subprocess.CompletedProcess(command, auth_code, "", "")
            if command[1:3] == ["attestation", "verify"]:
                if timeout_verify:
                    raise module.subprocess.TimeoutExpired(command, kwargs.get("timeout", 0))
                return module.subprocess.CompletedProcess(command, verify_code, "", "")
        if command[0].endswith("/docker"):
            if command[1:3] == ["image", "inspect"]:
                return module.subprocess.CompletedProcess(command, inspect_code, "", "")
            if command[1:2] == ["pull"]:
                return module.subprocess.CompletedProcess(command, 0, "", "")
        raise AssertionError(f"unexpected subprocess command: {command}")

    monkeypatch.setattr(
        module.shutil,
        "which",
        _tool_path,
    )
    monkeypatch.setattr(module.subprocess, "run", run)
    return calls


def _ensure_image(
    module: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    pin: dict[str, str | None],
    *,
    prewarm: bool,
    pull: bool = True,
) -> Any:
    override_env = module._VERIFY_ENV.removesuffix("_VERIFY_ATTESTATION")
    monkeypatch.delenv(f"{override_env}_TOOLS_IMAGE", raising=False)
    if module._REPOSITORY in {
        "VibeBB/wire-agent",
        "VibeBB/UX-creator-agent",
        "VibeBB/mechanical-agent",
        "VibeBB/electrical-circuit-agent",
        "VibeBB/production-engineering-agent",
    }:

        def image_from_lock(_root: Path) -> Any:
            return pin

        def docker_path() -> str:
            return "/usr/bin/docker"

        monkeypatch.setattr(module, "_image_from_lock", image_from_lock)
        monkeypatch.setattr(module, "_docker", docker_path)
        return module._ensure_image(tmp_path, pull=pull, prewarm=prewarm)
    if module._REPOSITORY == "VibeBB/dashboard-agent":
        return module._image_ready(pin, pull=pull, prewarm=prewarm)
    return module._ensure_image(pin, pull=pull, prewarm=prewarm)


def test_off_does_not_call_gh(launcher: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(launcher._VERIFY_ENV, "off")
    monkeypatch.setattr(
        launcher.shutil,
        "which",
        _unexpected_tool_lookup,
    )
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        _unexpected_subprocess,
    )
    launcher._verify_attestation(_pin(launcher), override=False)


def test_auto_skips_when_gh_is_missing(
    launcher: Any, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    monkeypatch.setenv(launcher._VERIFY_ENV, "auto")
    monkeypatch.setattr(launcher.shutil, "which", _without_gh)
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        _unexpected_subprocess,
    )
    launcher._verify_attestation(_pin(launcher), override=False)
    assert "gh is not on PATH" in capsys.readouterr().err


def test_auto_skips_when_auth_fails(
    launcher: Any, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    monkeypatch.setenv(launcher._VERIFY_ENV, "auto")
    calls = _runner(launcher, monkeypatch, auth_code=1)
    launcher._verify_attestation(_pin(launcher), override=False)
    assert calls[0][0] == ["/usr/bin/gh", "auth", "status"]
    assert calls[0][1]["timeout"] == launcher._GH_AUTH_TIMEOUT_S
    assert "gh auth status failed" in capsys.readouterr().err


def test_auto_skips_without_lock_attestation(
    launcher: Any, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    monkeypatch.setenv(launcher._VERIFY_ENV, "auto")
    pin = _pin(launcher)
    pin["attestation"] = None
    monkeypatch.setattr(launcher.shutil, "which", _tool_path)
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        _unexpected_subprocess,
    )
    launcher._verify_attestation(pin, override=False)
    assert "lock entry has no attestation" in capsys.readouterr().err


def test_auto_skips_image_override(
    launcher: Any, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    monkeypatch.setenv(launcher._VERIFY_ENV, "auto")
    monkeypatch.setattr(launcher.shutil, "which", _tool_path)
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        _unexpected_subprocess,
    )
    launcher._verify_attestation(_pin(launcher), override=True)
    assert "override has no lock attestation context" in capsys.readouterr().err


@pytest.mark.parametrize("missing_context", ["attestation", "gh"])
def test_require_errors_for_missing_context(
    launcher: Any,
    monkeypatch: pytest.MonkeyPatch,
    missing_context: str,
) -> None:
    monkeypatch.setenv(launcher._VERIFY_ENV, "require")
    pin = _pin(launcher)
    if missing_context == "attestation":
        pin["attestation"] = None

    def which(name: str) -> str | None:
        if missing_context == "gh" and name == "gh":
            return None
        return _tool_path(name)

    monkeypatch.setattr(launcher.shutil, "which", which)
    with pytest.raises(RuntimeError, match="verification required"):
        launcher._verify_attestation(pin, override=False)


def test_auto_verification_failure_raises(launcher: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(launcher._VERIFY_ENV, "auto")
    calls = _runner(launcher, monkeypatch, verify_code=1)
    with pytest.raises(RuntimeError, match="verification failed"):
        launcher._verify_attestation(_pin(launcher), override=False)
    assert calls[-1][0][1:3] == ["attestation", "verify"]


def test_auto_verification_success_uses_exact_signer(
    launcher: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(launcher._VERIFY_ENV, "require")
    calls = _runner(launcher, monkeypatch)
    pin = _pin(launcher)
    launcher._verify_attestation(pin, override=False)
    verify_command, verify_kwargs = calls[-1]
    assert verify_command == [
        "/usr/bin/gh",
        "attestation",
        "verify",
        f"oci://{pin['image']}@{pin['digest']}",
        "--repo",
        launcher._REPOSITORY,
        "--signer-workflow",
        f"{launcher._REPOSITORY}/{launcher._PUBLISH_FILE}",
    ]
    assert verify_kwargs["timeout"] == launcher._ATTEST_TIMEOUT_S


def test_verification_timeout_raises(launcher: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(launcher._VERIFY_ENV, "require")
    _runner(launcher, monkeypatch, timeout_verify=True)
    with pytest.raises(RuntimeError, match="timed out"):
        launcher._verify_attestation(_pin(launcher), override=False)


def test_prewarm_verifies_even_when_image_is_local(
    launcher: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(launcher._VERIFY_ENV, "require")
    calls = _runner(launcher, monkeypatch)
    _ensure_image(launcher, monkeypatch, tmp_path, _pin(launcher), prewarm=True)
    assert [call[0][1:3] for call in calls] == [
        ["auth", "status"],
        ["attestation", "verify"],
        ["image", "inspect"],
    ]


def test_ordinary_local_image_is_not_reverified(
    launcher: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(launcher._VERIFY_ENV, "require")
    calls = _runner(launcher, monkeypatch)
    _ensure_image(launcher, monkeypatch, tmp_path, _pin(launcher), prewarm=False)
    assert [call[0][1:3] for call in calls] == [["image", "inspect"]]


def test_verification_failure_prevents_pull(
    launcher: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(launcher._VERIFY_ENV, "auto")
    calls = _runner(launcher, monkeypatch, inspect_code=1, verify_code=1)
    with pytest.raises(RuntimeError, match="verification failed"):
        _ensure_image(launcher, monkeypatch, tmp_path, _pin(launcher), prewarm=False)
    assert not any(call[0][1:2] == ["pull"] for call in calls)


def test_successful_verification_allows_pull(
    launcher: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(launcher._VERIFY_ENV, "require")
    calls = _runner(launcher, monkeypatch, inspect_code=1)
    _ensure_image(launcher, monkeypatch, tmp_path, _pin(launcher), prewarm=False)
    assert calls[-1][0][1:2] == ["pull"]


def test_invalid_mode_returns_exit_code_two(launcher: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(launcher._VERIFY_ENV, "invalid")
    monkeypatch.setattr(launcher.sys, "argv", ["launcher.py", "doctor"])
    if launcher._REPOSITORY == "VibeBB/simulation-agent":
        assert launcher.main(["doctor"]) == 2
    else:
        assert launcher.main() == 2
