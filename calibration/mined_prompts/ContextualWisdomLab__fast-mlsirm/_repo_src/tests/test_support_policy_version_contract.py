"""Repository contracts for externally visible support-version policy."""

from __future__ import annotations

import re
from pathlib import Path


_ROOT = Path(__file__).parents[1]


def _supported_minor_line() -> str:
    """Return the package's current ``major.minor.x`` support line."""
    pyproject = (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "(\d+)\.(\d+)\.\d+(?:[^\"]*)?"$', pyproject, re.MULTILINE)
    assert match is not None, "pyproject.toml must declare a parseable project version"
    return f"{match.group(1)}.{match.group(2)}.x"


def test_security_and_support_track_current_package_minor() -> None:
    """Public support policies must identify the package's current minor line."""
    supported_line = _supported_minor_line()
    security = (_ROOT / "SECURITY.md").read_text(encoding="utf-8")
    support = (_ROOT / "SUPPORT.md").read_text(encoding="utf-8")

    assert f"| {supported_line} | Yes |" in security
    assert supported_line in support
    assert "0.1.x" not in security
    assert "0.1.x" not in support


def test_support_policy_does_not_restate_obsolete_beta_backend_contract() -> None:
    """Support wording must not freeze obsolete maturity or backend ownership claims."""
    support = (_ROOT / "SUPPORT.md").read_text(encoding="utf-8")

    assert "Commercial Beta Support Scope" not in support
    assert "NumPy or Rust backend" not in support
    assert "documented public API" in support
    assert "high-stakes" in support
