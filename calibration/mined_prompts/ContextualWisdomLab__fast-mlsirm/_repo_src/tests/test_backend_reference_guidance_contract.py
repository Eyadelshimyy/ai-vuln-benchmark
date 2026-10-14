"""Regression contract for production/reference backend guidance."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from fast_mlsirm import backend
from fast_mlsirm.cli import main


ROOT = Path(__file__).resolve().parents[1]
CHANGELOG_PATH = ROOT / "CHANGELOG.md"


def _changelog_section(heading: str) -> str:
    """Return one released or unreleased CHANGELOG.md subsection's exact text."""
    text = CHANGELOG_PATH.read_text(encoding="utf-8")
    start = text.index(heading)
    next_heading = text.find("\n#### ", start + len(heading))
    end = next_heading if next_heading != -1 else len(text)
    return text[start:end]


def test_auto_backend_failure_names_the_public_reference_api() -> None:
    """The stable runtime error must point to the named Python reference API."""
    message = backend.AUTO_BACKEND_UNAVAILABLE_MESSAGE
    assert "fast_mlsirm.fit_reference" in message
    assert "backend='numpy'" in message


def test_cli_backend_help_does_not_advertise_an_invalid_numpy_choice(capsys) -> None:
    """CLI guidance must use ``--reference`` rather than an invalid backend value."""
    with patch.object(sys, "argv", ["fast-mlsirm", "fit", "--help"]), pytest.raises(
        SystemExit
    ) as stopped:
        main()

    assert stopped.value.code == 0
    help_text = capsys.readouterr().out
    assert "--reference" in help_text
    assert "pass numpy only for the explicit reference/parity path" not in help_text


def test_accepted_backend_adr_keeps_the_protected_main_decision() -> None:
    """A feature PR must not silently rewrite the accepted production backend ADR."""
    text = (ROOT / "docs/adr/0002-rust-first-numerical-ownership.md").read_text(
        encoding="utf-8"
    )

    assert "public production backend architecture exposes only `auto` and `rust`" in text
    assert "`fit_reference` API" in text
    assert "`fit --reference` CLI mode" in text
    assert "not a `FitConfig` production choice" in text


def test_runtime_changelog_distinguishes_cli_and_python_reference_paths() -> None:
    """Release guidance must name the actual CLI and Python reference entry points.

    The originating change-fragment is deleted once its release cut consumes
    it (``docs/changelog.d/`` fragments are transient); the durable record of
    this guidance is the rendered ``CHANGELOG.md`` section it was folded into.
    """
    text = _changelog_section("#### Runtime contract buyer-facing ownership")

    assert "`fast-mlsirm fit --reference`" in text
    assert "`fast_mlsirm.fit_reference`" in text
    assert "pass explicit `backend=\"numpy\"`" not in text
