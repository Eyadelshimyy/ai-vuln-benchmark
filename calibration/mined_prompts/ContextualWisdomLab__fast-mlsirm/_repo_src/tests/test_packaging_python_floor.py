"""Packaging-contract tests for the advertised Python floor.

The project's supported interpreter floor lives in exactly one authoritative
place, ``pyproject.toml`` (``project.requires-python``), and the committed
lockfile ``uv.lock`` must advertise the identical floor. A drift between the
two means the lock resolves — and therefore reproduces builds for —
interpreter ranges the package no longer supports, which silently breaks the
"rebuild from immutable provenance" guarantee documented in
``docs/PRD.md`` (PRD-PRN-006).
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
from packaging.markers import Marker, default_environment
from packaging.version import Version

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PYPROJECT = _REPO_ROOT / "pyproject.toml"
_UV_LOCK = _REPO_ROOT / "uv.lock"
_PYTHON_FLOOR = Version("3.12")
_BASE_VERSION_WITNESSES = (
    "0.0.0",
    "3.10.0",
    "3.10.14",
    "3.11.0",
    "3.11.9",
    "3.12.0",
    "3.12.9",
    "3.13.0",
    "3.14.0",
)
_VERSION_LITERAL_RE = re.compile(r"(?<!\d)(\d+)\.(\d+)(?:\.(\d+|\*))?")
_PLATFORM_ENVIRONMENTS = (
    {"sys_platform": "linux", "platform_system": "Linux", "os_name": "posix"},
    {"sys_platform": "darwin", "platform_system": "Darwin", "os_name": "posix"},
    {"sys_platform": "win32", "platform_system": "Windows", "os_name": "nt"},
)


def _read_toml(path: Path) -> dict[str, object]:
    """Return parsed TOML from a required packaging contract file."""
    assert path.exists(), f"missing packaging file: {path}"
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _read_floor(path: Path) -> str:
    """Return the authoritative ``requires-python`` value from *path*."""
    document = _read_toml(path)
    if path == _PYPROJECT:
        project = document.get("project")
        assert isinstance(project, dict), "pyproject.toml has no [project] table"
        floor = project.get("requires-python")
    else:
        floor = document.get("requires-python")
    assert isinstance(floor, str), f"{path.name} declares no requires-python floor"
    return floor


def _marker_matches(marker: Marker, version: str, platform: dict[str, str]) -> bool:
    """Evaluate one lock marker under an explicit Python/platform environment."""
    environment = default_environment()
    environment.update(platform)
    environment["python_full_version"] = version
    environment["python_version"] = ".".join(version.split(".")[:2])
    return marker.evaluate(environment)


def _minor_neighbor(major: int, minor: int, direction: int) -> str | None:
    """Return a nearby minor-version witness without invoking marker code."""
    if direction < 0:
        if minor > 0:
            return f"{major}.{minor - 1}.999"
        if major > 0:
            return f"{major - 1}.999.999"
        return None
    return f"{major}.{minor + 1}.0"


def _version_witnesses(marker_text: str) -> tuple[str, ...]:
    """Return boundary-aware Python-version witnesses for one marker.

    Fixed witnesses cover the current support boundary and a global lower
    sentinel. Every explicit version literal contributes exact and adjacent
    witnesses in patch or minor space so strict inequalities and bounded ranges
    cannot hide unsupported intervals between the fixed samples.
    """
    witnesses = set(_BASE_VERSION_WITNESSES)
    for match in _VERSION_LITERAL_RE.finditer(marker_text):
        major = int(match.group(1))
        minor = int(match.group(2))
        patch_text = match.group(3)
        if patch_text in (None, "*"):
            witnesses.add(f"{major}.{minor}.0")
            witnesses.add(f"{major}.{minor}.999")
            previous_minor = _minor_neighbor(major, minor, -1)
            if previous_minor is not None:
                witnesses.add(previous_minor)
            witnesses.add(_minor_neighbor(major, minor, 1))
            continue
        patch = int(patch_text)
        witnesses.add(f"{major}.{minor}.{patch}")
        if patch > 0:
            witnesses.add(f"{major}.{minor}.{patch - 1}")
        else:
            previous_minor = _minor_neighbor(major, minor, -1)
            if previous_minor is not None:
                witnesses.add(previous_minor)
        witnesses.add(f"{major}.{minor}.{patch + 1}")
    return tuple(sorted(witnesses, key=Version))


def _targets_only_dropped_interpreters(marker_text: str) -> bool:
    """Return whether a resolution marker selects only pre-3.12 interpreters.

    PEP 508 marker parsing normalizes quote style, whitespace, and the
    ``python_version`` versus ``python_full_version`` spelling. Version
    witnesses include a global lower sentinel plus every explicit version
    literal and adjacent patch/minor boundaries, so exact pins, strict bounds,
    and unsupported ranges below the historical 3.10 samples cannot evade the
    pre-3.12 contract.
    """
    marker = Marker(marker_text)
    witnesses = _version_witnesses(marker_text)
    matches_unsupported = any(
        Version(version) < _PYTHON_FLOOR
        and _marker_matches(marker, version, platform)
        for version in witnesses
        for platform in _PLATFORM_ENVIRONMENTS
    )
    matches_supported = any(
        Version(version) >= _PYTHON_FLOOR
        and _marker_matches(marker, version, platform)
        for version in witnesses
        for platform in _PLATFORM_ENVIRONMENTS
    )
    return matches_unsupported and not matches_supported


def _lock_marker_strings(lock: dict[str, object]) -> tuple[str, ...]:
    """Collect top-level partitions and nested package/dependency markers."""
    markers: list[str] = []
    stack: list[object] = [lock]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "marker":
                    assert isinstance(child, str), "uv.lock marker fields must be strings"
                    markers.append(child)
                elif key == "resolution-markers":
                    assert isinstance(child, list), (
                        "uv.lock resolution-markers must be a list"
                    )
                    assert all(isinstance(marker, str) for marker in child), (
                        "uv.lock resolution-markers must contain only strings"
                    )
                    markers.extend(child)
                else:
                    stack.append(child)
        elif isinstance(value, list):
            stack.extend(value)
    return tuple(markers)


def test_uv_lock_floor_matches_pyproject_floor() -> None:
    """The lockfile must advertise the exact pyproject interpreter floor."""
    assert _read_floor(_UV_LOCK) == _read_floor(_PYPROJECT)


def test_uv_lock_has_no_stale_pre_3_12_resolution_markers() -> None:
    """The lock must not retain partitions targeting only dropped Pythons."""
    lock = _read_toml(_UV_LOCK)
    stale = [
        marker
        for marker in _lock_marker_strings(lock)
        if _targets_only_dropped_interpreters(marker)
    ]
    assert not stale, (
        "uv.lock still contains resolution partitions that target only dropped "
        f"pre-3.12 interpreters: {stale!r}; regenerate with `uv lock` after "
        "confirming the pyproject floor"
    )


def test_uv_lock_stale_guard_inspects_nested_package_markers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Package/dependency marker fields must not bypass the floor contract."""
    synthetic_lock: dict[str, object] = {
        "resolution-markers": ["python_version >= '3.12'"],
        "package": [
            {
                "name": "legacy-backport",
                "marker": "python_full_version == '3.11.5'",
            }
        ],
    }
    monkeypatch.setitem(globals(), "_read_toml", lambda _path: synthetic_lock)

    with pytest.raises(AssertionError, match="pre-3.12"):
        test_uv_lock_has_no_stale_pre_3_12_resolution_markers()


@pytest.mark.parametrize(
    "marker",
    (
        "python_full_version == '3.11.*'",
        "python_full_version == '3.11.5'",
        "python_full_version == '3.10.42'",
        "python_full_version > '3.11.100' and python_version < '3.12'",
        "python_full_version < '3.10'",
        "python_version > '3.8' and python_version < '3.10'",
        'python_version   <   "3.12"',
        "python_version <= '3.11'",
        'python_full_version == "3.10.*" and sys_platform == "win32"',
    ),
)
def test_stale_marker_detection_is_rendering_independent(marker: str) -> None:
    """Rendering, exact patch pins, and patch-bound inequalities stay covered."""
    assert _targets_only_dropped_interpreters(marker)


@pytest.mark.parametrize(
    "marker",
    (
        "python_version >= '3.12'",
        "python_version < '3.13'",
        "python_full_version == '3.12.*'",
        "python_full_version == '3.12.5'",
    ),
)
def test_supported_resolution_markers_are_not_misclassified(marker: str) -> None:
    """Partitions that include supported interpreters remain permitted."""
    assert not _targets_only_dropped_interpreters(marker)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
