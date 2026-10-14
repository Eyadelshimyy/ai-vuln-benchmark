"""Regression tests for essay HTML report title trust boundaries."""

from __future__ import annotations

import runpy
from collections.abc import Callable
from pathlib import Path

import pytest

from fast_mlsirm.scoring.essay import (
    render_essay_facets_calibration_report_html,
    render_essay_score_report_html,
    render_essay_validation_evidence_report_html,
)

_SCORE_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("test_scoring_essay_report_html.py"))
)
_VALIDATION_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("test_scoring_essay_validation_reporting.py"))
)
_FACETS_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("test_scoring_essay_facets_reporting.py"))
)
clean_score_report = _SCORE_FIXTURES["clean_report"]
build_validation_report = _VALIDATION_FIXTURES["build_report"]
build_facets_report = _FACETS_FIXTURES["build_report"]


class _HostileTitle(str):
    """String subclass whose text callbacks must never cross a report boundary."""

    def strip(self, *args: object, **kwargs: object) -> str:
        """Fail if validation invokes caller-controlled string behavior."""
        raise AssertionError("hostile title strip callback executed")

    def replace(self, *args: object, **kwargs: object) -> str:
        """Fail if HTML escaping invokes caller-controlled string behavior."""
        raise AssertionError("hostile title replace callback executed")


@pytest.mark.parametrize(
    ("renderer", "report_factory", "message"),
    (
        (
            render_essay_score_report_html,
            clean_score_report,
            "essay score report title must be a non-empty string",
        ),
        (
            render_essay_validation_evidence_report_html,
            build_validation_report,
            "essay validation evidence title must be a non-empty string",
        ),
        (
            render_essay_facets_calibration_report_html,
            build_facets_report,
            "essay facets calibration title must be a non-empty string",
        ),
    ),
)
def test_renderers_reject_string_subclasses_without_callbacks(
    tmp_path: Path,
    renderer: Callable[..., Path],
    report_factory: Callable[[], object],
    message: str,
) -> None:
    """All essay HTML titles reject subclasses before caller callbacks or writes."""
    output = tmp_path / f"{report_factory.__name__}.html"

    with pytest.raises(ValueError, match=message):
        renderer(
            report_factory(),
            output,
            title=_HostileTitle("audit title"),
        )

    assert not output.exists()
