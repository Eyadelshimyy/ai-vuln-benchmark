"""Security regressions for bounded facets-report artifact publication."""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

from fast_mlsirm.scoring.essay import render_essay_facets_calibration_report_html

_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("test_scoring_essay_facets_reporting.py"))
)
build_report = _FIXTURES["build_report"]


def test_renderer_confines_nested_output_to_explicit_root(tmp_path: Path) -> None:
    """An approved root permits nested reports and returns the canonical path."""
    output_root = tmp_path / "approved_reports"
    expected = output_root / "nested_reports" / "calibration_report.html"

    returned = render_essay_facets_calibration_report_html(
        build_report(),
        "nested_reports/calibration_report.html",
        output_root=output_root,
    )

    assert returned == expected.resolve()
    assert returned.is_absolute()
    assert returned.is_relative_to(output_root.resolve())
    assert expected.is_file()


def test_renderer_rejects_relative_and_absolute_output_escape(tmp_path: Path) -> None:
    """Traversal and absolute paths outside the approved root fail before writes."""
    output_root = tmp_path / "approved_reports"
    relative_escape = Path("..") / "relative_escape.html"
    absolute_escape = tmp_path / "absolute_escape.html"

    for output_path in (relative_escape, absolute_escape):
        with pytest.raises(ValueError, match="approved output directory"):
            render_essay_facets_calibration_report_html(
                build_report(),
                output_path,
                output_root=output_root,
            )

    assert not (tmp_path / "relative_escape.html").exists()
    assert not absolute_escape.exists()


def test_renderer_defaults_to_current_working_directory_boundary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Without an explicit root, only paths inside the current directory are valid."""
    monkeypatch.chdir(tmp_path)

    returned = render_essay_facets_calibration_report_html(
        build_report(),
        "local_report.html",
    )

    assert returned == (tmp_path / "local_report.html").resolve()
    with pytest.raises(ValueError, match="approved output directory"):
        render_essay_facets_calibration_report_html(
            build_report(),
            "../outside_report.html",
        )
    assert not (tmp_path.parent / "outside_report.html").exists()


def test_renderer_rejects_symlink_parent_escape(tmp_path: Path) -> None:
    """An existing symlink cannot redirect a bounded report outside its root."""
    output_root = tmp_path / "approved_reports"
    outside_root = tmp_path / "outside_reports"
    output_root.mkdir()
    outside_root.mkdir()
    linked_directory = output_root / "linked_reports"
    try:
        linked_directory.symlink_to(outside_root, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable on this platform")

    with pytest.raises(ValueError, match="approved output directory"):
        render_essay_facets_calibration_report_html(
            build_report(),
            "linked_reports/escaped_report.html",
            output_root=output_root,
        )

    assert not (outside_root / "escaped_report.html").exists()


def test_renderer_rejects_non_directory_output_root(tmp_path: Path) -> None:
    """An existing regular file cannot be treated as an approved directory."""
    output_root = tmp_path / "not_a_directory"
    output_root.write_text("occupied", encoding="utf-8")

    with pytest.raises(ValueError, match="output root must be a directory"):
        render_essay_facets_calibration_report_html(
            build_report(),
            "calibration_report.html",
            output_root=output_root,
        )