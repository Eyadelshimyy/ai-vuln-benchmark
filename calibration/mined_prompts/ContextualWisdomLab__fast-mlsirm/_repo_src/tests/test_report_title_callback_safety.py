"""Regression tests for the generic diagnostics-report title trust boundary."""

from __future__ import annotations

import json

import pytest

from fast_mlsirm.report import render_diagnostics_report


class _HostileTitle(str):
    """String subclass whose title callbacks must never be dispatched."""

    calls = 0

    def __bool__(self) -> bool:
        """Record forbidden truth-value dispatch during title resolution."""
        type(self).calls += 1
        raise RuntimeError("private title truth callback")

    def replace(self, *args: object, **kwargs: object) -> str:
        """Record forbidden replacement dispatch during HTML escaping."""
        type(self).calls += 1
        raise RuntimeError("private title replacement callback")


def _write_fit_payload(path) -> None:
    """Write one minimal valid fit-diagnostics payload."""
    path.write_text(json.dumps({"model_fit": {"loglik": -1.0}}), encoding="utf-8")


def test_generic_report_rejects_title_subclass_without_callbacks(tmp_path) -> None:
    """A caller string subclass cannot execute while a report title is admitted."""
    source = tmp_path / "fit.json"
    output = tmp_path / "fit.html"
    _write_fit_payload(source)
    _HostileTitle.calls = 0

    with pytest.raises(ValueError, match="title must be a string or None"):
        render_diagnostics_report(source, output, title=_HostileTitle("Private title"))

    assert _HostileTitle.calls == 0
    assert not output.exists()


def test_generic_report_empty_builtin_title_keeps_default(tmp_path) -> None:
    """An empty exact built-in title preserves the existing default-title contract."""
    source = tmp_path / "fit.json"
    output = tmp_path / "fit.html"
    _write_fit_payload(source)

    render_diagnostics_report(source, output, title="")

    html = output.read_text(encoding="utf-8")
    assert "<title>Fit Diagnostics Report</title>" in html
    assert '<h1 id="hero-heading">Fit Diagnostics Report</h1>' in html


def test_generic_report_builtin_title_remains_html_escaped(tmp_path) -> None:
    """An admitted exact built-in title remains escaped in every HTML title surface."""
    source = tmp_path / "fit.json"
    output = tmp_path / "fit.html"
    _write_fit_payload(source)

    render_diagnostics_report(source, output, title='<Admin & "Ops">')

    html = output.read_text(encoding="utf-8")
    assert "<title>&lt;Admin &amp; &quot;Ops&quot;&gt;</title>" in html
    assert '<h1 id="hero-heading">&lt;Admin &amp; &quot;Ops&quot;&gt;</h1>' in html
    assert '<Admin & "Ops">' not in html
