"""Unit-test pass-rate gate used by CI for develop→main."""

from __future__ import annotations

from pathlib import Path

import pytest

from jobbot.ops.test_gate import enforce_pass_rate, pass_rate, read_junit_counts


def test_pass_rate_counts_only_real_failures() -> None:
    assert pass_rate(tests=100, failures=5, errors=0) == pytest.approx(0.95)
    assert pass_rate(tests=0, failures=0, errors=0) == 0.0


def test_junit_reader_and_gate(tmp_path: Path) -> None:
    report = tmp_path / "junit.xml"
    report.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<testsuites>
  <testsuite name="pytest" tests="20" failures="1" errors="0" skipped="0">
    <testcase classname="t" name="ok"/>
  </testsuite>
</testsuites>
""",
        encoding="utf-8",
    )
    assert read_junit_counts(report) == (20, 1, 0)
    assert enforce_pass_rate(report, min_rate=0.95) == 0
    assert enforce_pass_rate(report, min_rate=0.96) == 1
