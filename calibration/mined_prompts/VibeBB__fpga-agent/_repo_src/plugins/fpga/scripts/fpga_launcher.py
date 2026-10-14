#!/usr/bin/env python3
"""Resolve the fpga package and tools image, then exec the entry point.

Plugin installs update the assets under plugins/fpga but do not install
the ``fpga`` Python package or the embedded toolchain. The launcher runs
every entry point inside the pinned fpga-tools image, mounting the
resolved source tree and the workspace so paths stay identical inside the
container. The container runs with ``--network none``: the image carries the open-source
FPGA toolchain (OSS CAD Suite, NVC) the gates need. ``program`` always runs
on the host, because it needs the USB programmer; it is human-only and is
never forwarded to the image.

Source resolution order (first directory containing fpga/__init__.py wins):
  1. $FPGA_SRC
  2. newest ~/.openhands/cache/extensions/fpga-agent-*/src
  3. /opt/fpga/src (fpga-tools image)
  4. <repo>/src when running from a repository checkout

Image resolution order (first hit wins):
  1. $FPGA_TOOLS_IMAGE (full ref, e.g. ghcr.io/.../fpga-tools@sha256:...)
  2. <plugin>/tools-image.json or repo docker/image-digests.json
     (image + digest, falling back to image + tag)
  3. none resolvable -> fail closed: every command except ``program``
     fails (``--warn`` prints the warn JSON and exits 0). The only
     exception is when the launcher itself already runs inside the
     fpga-tools image (``/opt/fpga/src/fpga/__init__.py`` and
     ``/opt/oss-cad-suite/bin/yosys`` both exist), where it execs the
     package in-process.

Usage: mcp_server | prewarm | <fpga cli args...>. When ``--warn`` is
present (SessionStart doctor mode), launcher failures print a warning and
exit 0.

Launcher-side verification uses FPGA_VERIFY_ATTESTATION=auto|require|off.
It verifies lock provenance before pulls and on every prewarm; normal use
does not re-verify an image that is already present locally.
"""

from __future__ import annotations

import contextlib
import json
import os
import pwd
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, TypedDict, cast

_MODULES = {"mcp_server": "fpga.mcp_server"}
_CONTAINER_SRC = "/plugin-src"
_ENV_PREFIXES = ("OPENHANDS_", "FPGA_")
_ENV_KEYS = ("TMPDIR",)
_CONTAINER_ENV = {
    "HOME": "/tmp",
    "TMPDIR": "/tmp",
    "XDG_CACHE_HOME": "/tmp/.cache",
    "XDG_CONFIG_HOME": "/tmp/.config",
}
_LOCK_KEY = "fpga_tools"
_INSPECT_TIMEOUT_S = 30
_DOCKER_INFO_TIMEOUT_S = 10
_PULL_TIMEOUT_S = 900
_ATTEST_TIMEOUT_S = 120
_GH_AUTH_TIMEOUT_S = 15
_VERIFY_ENV = "FPGA_VERIFY_ATTESTATION"
_REPOSITORY = "VibeBB/fpga-agent"
_PUBLISH_FILE = ".github/workflows/publish-fpga-images.yml"
_IMAGE_SRC = Path("/opt/fpga/src/fpga/__init__.py")
_IMAGE_YOSYS = Path("/opt/oss-cad-suite/bin/yosys")


def running_inside_tools_image() -> bool:
    """True when the launcher already runs inside the fpga-tools image."""
    return _IMAGE_SRC.is_file() and _IMAGE_YOSYS.is_file()


class ImagePin(TypedDict):
    ref: str
    image: str | None
    digest: str | None
    attestation: str | None


def _homes() -> list[Path]:
    homes = [Path.home()]
    try:
        real = Path(pwd.getpwuid(os.getuid()).pw_dir)
    except (KeyError, OSError):
        return homes
    if real != homes[0]:
        homes.append(real)
    return homes


def _cache_dirs(pattern: str) -> list[Path]:
    found: list[Path] = []
    for home in _homes():
        cache = home / ".openhands" / "cache" / "extensions"
        try:
            if cache.is_dir():
                found.extend(
                    sorted(cache.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
                )
        except OSError:
            pass
    return found


def _candidates(plugin_root: Path) -> list[Path]:
    candidates: list[Path] = []
    env_src = os.environ.get("FPGA_SRC")
    if env_src:
        candidates.append(Path(env_src))
    candidates += _cache_dirs("fpga-agent-*/src")
    candidates.append(Path("/opt/fpga/src"))
    candidates.append(plugin_root.parent.parent / "src")
    return candidates


def resolve_source(plugin_root: Path) -> Path | None:
    for candidate in _candidates(plugin_root):
        try:
            if (candidate / "fpga" / "__init__.py").is_file():
                return candidate.resolve()
        except OSError:
            continue
    return None


def _lock_entry_ref(lock_path: Path, key: str | None) -> ImagePin | None:
    try:
        data = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    data = cast(dict[str, Any], data)
    entry = data.get(key) if key else data
    if not isinstance(entry, dict):
        return None
    entry = cast(dict[str, Any], entry)
    image = entry.get("image")
    if not isinstance(image, str) or not image:
        return None
    digest = entry.get("digest")
    digest = digest if isinstance(digest, str) and digest else None
    tag = entry.get("tag")
    tag = tag if isinstance(tag, str) and tag else None
    if digest is None and tag is None:
        return None
    attestation = entry.get("attestation")
    return {
        "ref": f"{image}@{digest}" if digest else f"{image}:{tag}",
        "image": image,
        "digest": digest,
        "attestation": attestation if isinstance(attestation, str) and attestation else None,
    }


def image_pin(plugin_root: Path) -> ImagePin | None:
    explicit = os.environ.get("FPGA_TOOLS_IMAGE")
    if explicit:
        return {"ref": explicit, "image": None, "digest": None, "attestation": None}
    pin = _lock_entry_ref(plugin_root / "tools-image.json", None)
    if pin:
        return pin
    repo_dirs = [plugin_root.parent.parent, *_cache_dirs("fpga-agent-*")]
    for repo_dir in repo_dirs:
        pin = _lock_entry_ref(repo_dir / "docker" / "image-digests.json", _LOCK_KEY)
        if pin:
            return pin
    return None


def image_ref(plugin_root: Path) -> str | None:
    pin = image_pin(plugin_root)
    return pin["ref"] if pin is not None else None


def _attestation_mode() -> str:
    mode = os.environ.get(_VERIFY_ENV, "auto")
    if mode not in {"auto", "require", "off"}:
        raise ValueError(
            f"{_VERIFY_ENV} must be auto, require, or off (got {mode!r}); "
            f"usage: {_VERIFY_ENV}=auto|require|off"
        )
    return mode


def _run_timed(
    command: list[str],
    operation: str,
    timeout: int,
    **kwargs: Any,
) -> subprocess.CompletedProcess[str]:
    try:
        return cast(
            subprocess.CompletedProcess[str],
            subprocess.run(command, timeout=timeout, **kwargs),
        )
    except OSError as exc:
        raise RuntimeError(f"{operation} failed: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{operation} timed out after {timeout}s") from exc


def _verify_attestation(pin: ImagePin, *, override: bool) -> None:
    mode = _attestation_mode()
    if mode == "off":
        return
    reason: str | None = None
    gh = shutil.which("gh")
    if override:
        reason = "tools image override has no lock attestation context"
    elif not pin["attestation"]:
        reason = "lock entry has no attestation"
    elif not pin["image"] or not pin["digest"]:
        reason = "lock entry has no digest"
    elif gh is None:
        reason = "gh is not on PATH"
    else:
        try:
            auth = _run_timed(
                [gh, "auth", "status"],
                "gh auth status",
                _GH_AUTH_TIMEOUT_S,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except RuntimeError:
            reason = "gh auth status failed"
        else:
            if auth.returncode != 0:
                reason = "gh auth status failed"
    if reason is not None:
        if mode == "require":
            raise RuntimeError(f"attestation verification required but {reason}")
        print(f"fpga_launcher: attestation verification skipped: {reason}", file=sys.stderr)
        return
    assert gh is not None
    assert pin["image"] is not None and pin["digest"] is not None
    result = _run_timed(
        [
            gh,
            "attestation",
            "verify",
            f"oci://{pin['image']}@{pin['digest']}",
            "--repo",
            _REPOSITORY,
            "--signer-workflow",
            f"{_REPOSITORY}/{_PUBLISH_FILE}",
        ],
        "gh attestation verify",
        _ATTEST_TIMEOUT_S,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"attestation verification failed for {pin['image']}@{pin['digest']}")


def _inside_conversation_container() -> bool:
    """True when running inside an OpenHands docker conversation runtime.

    The runtime injects ``OH_PERSISTENCE_DIR``/``OH_RUNTIME_LAUNCHED_PROFILE``
    into each ``agent-server-conversation-*`` container, which carries no
    docker client — fpga tools then have nowhere to launch the pinned
    tools image. ``OH_CONVERSATION_RUNTIME`` is not usable as the signal:
    the runtime sets it to ``local`` inside the container itself.
    """
    if os.environ.get("OH_PERSISTENCE_DIR") or os.environ.get("OH_RUNTIME_LAUNCHED_PROFILE"):
        return True
    with contextlib.suppress(OSError):
        return Path.home() == Path("/var/openhands/.openhands")
    return False


def _ensure_image(
    image: ImagePin | str,
    *,
    pull: bool,
    prewarm: bool = False,
    override: bool = False,
) -> None:
    if isinstance(image, str):
        pin: ImagePin = {"ref": image, "image": None, "digest": None, "attestation": None}
        override = True
    else:
        pin = image
    ref = pin["ref"]
    docker = shutil.which("docker")
    if docker is None:
        detail = f"docker not found on PATH (tools image {ref} is pinned)"
        if _inside_conversation_container():
            detail += (
                " — this appears to be an OpenHands docker conversation "
                "container, which cannot launch tool containers; set the "
                "conversation runtime to local (Agent Canvas -> Settings -> "
                "Application) and start a new conversation"
            )
        raise RuntimeError(detail)
    if prewarm:
        _verify_attestation(pin, override=override)
    try:
        inspect = subprocess.run(
            [docker, "image", "inspect", ref],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=_INSPECT_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"docker image inspect timed out after {_INSPECT_TIMEOUT_S}s") from exc
    if inspect.returncode == 0:
        return
    if not pull:
        raise RuntimeError(
            f"fpga tools image {ref} not pulled locally; run 'fpga_launcher.py prewarm' to fetch it"
        )
    if not prewarm:
        _verify_attestation(pin, override=override)
    print(f"fpga_launcher: pulling tools image {ref}", file=sys.stderr)
    try:
        pulled = subprocess.run(
            [docker, "pull", ref],
            check=False,
            stdout=subprocess.DEVNULL,
            timeout=_PULL_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"docker pull timed out after {_PULL_TIMEOUT_S}s") from exc
    if pulled.returncode != 0:
        raise RuntimeError(f"fpga tools image {ref} not present locally and pull failed")


def _docker_info_security_options() -> str | None:
    """Return `docker info` security options, or None when unavailable."""
    docker = shutil.which("docker")
    if docker is None:
        return None
    try:
        result = subprocess.run(
            [docker, "info", "-f", "{{json .SecurityOptions}}"],
            capture_output=True,
            text=True,
            timeout=_DOCKER_INFO_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout if result.returncode == 0 else None


def _container_user() -> str:
    """uid:gid to run the tools container as.

    On rootless Docker the host uid maps to an unmapped subuid inside the
    container user namespace, so bind-mounted workspace writes fail. There
    container root (0:0) maps back to the daemon's owner — the invoking
    user — so 0:0 keeps writes working without weakening isolation (the
    container stays read-only/cap-dropped). On rootful Docker keep the host
    uid so artifacts stay user-owned.
    """
    if "name=rootless" in (_docker_info_security_options() or ""):
        return "0:0"
    return f"{os.getuid()}:{os.getgid()}"


def docker_argv(image: str, source: Path | None, inner_argv: list[str]) -> list[str]:
    workdir = os.environ.get("OPENHANDS_PROJECT_DIR") or os.getcwd()
    cwd = Path.cwd()
    inside = cwd == Path(workdir) or Path(workdir) in cwd.parents
    argv = [
        "docker",
        "run",
        "--rm",
        "-i",
        "--network",
        "none",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--user",
        _container_user(),
        "-v",
        f"{workdir}:{workdir}",
        "-w",
        str(cwd) if inside else workdir,
    ]
    if source is not None:
        argv += ["-v", f"{source}:{_CONTAINER_SRC}:ro", "-e", f"PYTHONPATH={_CONTAINER_SRC}"]
    for key, value in os.environ.items():
        if key in _ENV_KEYS or any(key.startswith(p) for p in _ENV_PREFIXES):
            argv += ["-e", f"{key}={value}"]
    for key, value in _CONTAINER_ENV.items():
        argv += ["-e", f"{key}={value}"]
    return [*argv, image, *inner_argv]


def inner_argv(argv: list[str], python: str = "python") -> list[str]:
    if argv[0] in _MODULES:
        return [python, "-m", _MODULES[argv[0]], *argv[1:]]
    return [python, "-m", "fpga.cli", *argv]


def _warn_or_die(message: str, argv: list[str]) -> int:
    if "--warn" in argv:
        print(f"warn: {message}", file=sys.stderr)
        print(json.dumps({"verdict": "fail", "detail": message}))
        return 0
    print(f"fpga_launcher: {message}", file=sys.stderr)
    return 1


def main() -> int:
    argv = sys.argv[1:]
    if not argv:
        print(
            "usage: fpga_launcher.py {mcp_server|prewarm|<fpga cli args...>}",
            file=sys.stderr,
        )
        return 2
    try:
        _attestation_mode()
    except ValueError as exc:
        print(f"fpga_launcher: {exc}", file=sys.stderr)
        return 2
    plugin_root = Path(__file__).resolve().parents[1]
    source = resolve_source(plugin_root)
    if argv[0] == "fetch":
        # Fetch contract-declared HDL libraries on the host; the tools
        # container runs with --network none so cloning cannot happen
        # inside it. Usage: fpga_launcher.py fetch <contract.fpga.json>
        if len(argv) != 2:
            print("usage: fpga_launcher.py fetch <contract.fpga.json>", file=sys.stderr)
            return 2
        candidates = [plugin_root.parents[1]]
        if source is not None:
            candidates.append(source.parent)
        script = next(
            (
                c / "scripts" / "fetch_hdl_libraries.py"
                for c in candidates
                if (c / "scripts" / "fetch_hdl_libraries.py").is_file()
            ),
            None,
        )
        if script is None:
            return _warn_or_die("scripts/fetch_hdl_libraries.py not found", argv)
        return subprocess.call(
            [sys.executable, str(script), "--contract", argv[1]],
            cwd=script.parents[1],
        )
    if argv[0] == "program":
        # Programming is human-only and needs the USB programmer: it always
        # runs on the host interpreter, never inside the tools image.
        if source is None:
            return _warn_or_die("fpga package source not found", argv)
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            p for p in (str(source), env.get("PYTHONPATH", "")) if p
        )
        command = inner_argv(argv, sys.executable)
        os.execvpe(command[0], command, env)
        return 0
    pin = image_pin(plugin_root)
    ref = pin["ref"] if pin is not None else None
    if pin is None or ref is None:
        if argv[0] == "prewarm":
            return _warn_or_die("no fpga tools image pinned; cannot prewarm", argv)
        if running_inside_tools_image():
            if source is None:
                return _warn_or_die("fpga package source not found in tools image", argv)
            env = dict(os.environ)
            env["PYTHONPATH"] = os.pathsep.join(
                p for p in (str(source), env.get("PYTHONPATH", "")) if p
            )
            command = inner_argv(argv, sys.executable)
            os.execvpe(command[0], command, env)
            return 0
        return _warn_or_die(
            "no fpga tools image resolvable; fpga tools run only inside the "
            "pinned fpga-tools image (set FPGA_TOOLS_IMAGE or pull the pinned "
            "image with 'fpga_launcher.py prewarm')",
            argv,
        )
    try:
        _ensure_image(
            pin,
            pull="--warn" not in argv,
            prewarm=argv[0] == "prewarm",
            override=bool(os.environ.get("FPGA_TOOLS_IMAGE")),
        )
    except RuntimeError as exc:
        return _warn_or_die(str(exc), argv)
    if argv[0] == "prewarm":
        print(f"fpga_launcher: tools image ready: {ref}")
        return 0
    os.execvp("docker", docker_argv(ref, source, inner_argv(argv)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
