"""Resolve a pinned Solidity compiler without trusting source text as a path.

The campaign names a version. This module looks only in operator-controlled
locations and accepts a binary when ``solc --version`` reports that version.
It never reads an executable path out of a Solidity file, and it never
overwrites the ``solc`` already on ``PATH``.
"""

from __future__ import annotations

import hashlib
import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

_VERSION = re.compile(r"^\d+\.\d+\.\d+$")
_EVM_WORD = re.compile(r"[A-Za-z][A-Za-z0-9]*")


@dataclass(frozen=True)
class ResolvedCompiler:
    """A local compiler whose reported version matches the request."""

    version: str
    path: str
    reported_version: str
    origin: str  # bugforge | svm | path


def exact_version(value: str) -> str:
    """An exact ``major.minor.patch`` version, or empty when it is not one.

    A range, a path, or any other string is rejected. Callers must not treat
    rejection as permission to use a different compiler.
    """
    text = value.strip()
    if not _VERSION.fullmatch(text):
        return ""
    return text


def distribution_platforms(system: str, machine: str) -> tuple[str, ...]:
    """Official Solidity distribution ids for this OS and CPU, preferred first.

    Apple Silicon has no separate 0.8.15 archive. The official macOS build for
    that release is ``macosx-amd64``, which is not a Linux binary. A Linux
    aarch64 host does not fall back to amd64.
    """
    system = system.lower()
    machine = machine.lower()
    if system == "darwin" and machine in {"arm64", "aarch64"}:
        return ("macosx-aarch64", "macosx-amd64")
    if system == "darwin":
        return ("macosx-amd64",)
    if system == "linux" and machine in {"aarch64", "arm64"}:
        return ("linux-aarch64",)
    if system == "linux" and machine in {"x86_64", "amd64"}:
        return ("linux-amd64",)
    return ()


def verify_sha256(data: bytes, expected: str) -> bool:
    """True when ``data`` matches a hex SHA-256 from the official version list."""
    digest = expected.strip().lower().removeprefix("0x")
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        return False
    return hashlib.sha256(data).hexdigest() == digest


def resolve_solc(version: str) -> ResolvedCompiler | None:
    """The local binary for ``version``, or None when that version is not installed.

    Search order: ``BUGFORGE_SOLC_ROOT``, ``~/.bugforge/solc``, svm, a
    ``solc-<version>`` name on ``PATH``, then ``solc`` on ``PATH``. A hit is
    accepted only when the executable prints that exact version. A different
    installed compiler is not a fallback.
    """
    wanted = exact_version(version)
    if not wanted:
        return None
    for path, origin, rooted in _candidates(wanted):
        if not _allowed(path, rooted):
            continue
        reported = reported_version(path)
        if reported == wanted:
            return ResolvedCompiler(wanted, str(path), reported, origin)
    return None


def reported_version(path: Path) -> str:
    """The ``Version:`` line of ``path --version``, or empty when it cannot be read."""
    try:
        proc = subprocess.run(
            [str(path), "--version"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    match = re.search(r"Version:\s*(\d+\.\d+\.\d+)", proc.stdout + proc.stderr)
    return match.group(1) if match else ""


def supported_evm_versions(path: str) -> tuple[str, ...]:
    """EVM names this compiler documents, in the order ``solc --help`` lists them."""
    try:
        proc = subprocess.run(
            [path, "--help"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ()
    text = proc.stdout + proc.stderr
    for line in text.splitlines():
        if "--evm-version" not in line:
            continue
        # The option description continues on following lines until a blank one.
        start = text.find(line)
        block = text[start : start + 500]
        words = _EVM_WORD.findall(block)
        skip = {"evm", "version", "or", "Select", "the", "desired"}
        found = tuple(word for word in words if word not in skip and word[:1].islower())
        if found:
            return found
    return ()


def evm_version_for(path: str, requested: str) -> tuple[str, bool]:
    """``(applied, exact)``.

    ``applied`` is ``requested`` when this compiler accepts it. Otherwise it is
    empty: the caller must not write an unsupported EVM into the harness and
    must not describe the compile as an exact match for the requested setting.
    """
    name = requested.strip()
    if not name or name == "unknown":
        return "", True
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9]{0,31}", name):
        return "", False
    supported = supported_evm_versions(path)
    if name in supported:
        return name, True
    return "", False


def _candidates(version: str) -> tuple[tuple[Path, str, bool], ...]:
    found: list[tuple[Path, str, bool]] = []
    for root in _roots():
        found.append((root / version / "solc", "bugforge", True))
        found.append((root / version / f"solc-{version}", "svm", True))
    named = shutil.which(f"solc-{version}")
    if named:
        found.append((Path(named), "path", False))
    ambient = shutil.which("solc")
    if ambient:
        found.append((Path(ambient), "path", False))
    return tuple(found)


def _roots() -> tuple[Path, ...]:
    roots: list[Path] = []
    override = os.environ.get("BUGFORGE_SOLC_ROOT", "").strip()
    if override:
        roots.append(Path(override).expanduser())
    roots.append(Path.home() / ".bugforge" / "solc")
    roots.append(Path.home() / ".local" / "share" / "svm")
    roots.append(Path.home() / ".svm")
    return tuple(roots)


def _allowed(path: Path, rooted: bool) -> bool:
    try:
        resolved = path.resolve()
    except OSError:
        return False
    if not resolved.is_file():
        return False
    if not rooted:
        return True
    for root in _roots():
        try:
            resolved.relative_to(root.resolve())
            return True
        except ValueError:
            continue
    return False


def current_platforms() -> tuple[str, ...]:
    return distribution_platforms(platform.system(), platform.machine())
