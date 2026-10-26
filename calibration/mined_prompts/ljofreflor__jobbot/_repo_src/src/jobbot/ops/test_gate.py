"""CI gates for unit-test pass rate (develop→main requires ≥95%)."""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def pass_rate(*, tests: int, failures: int, errors: int) -> float:
    """Fraction of collected tests that passed."""
    if tests <= 0:
        return 0.0
    passed = tests - failures - errors
    return passed / tests


def read_junit_counts(path: Path) -> tuple[int, int, int]:
    """Return (tests, failures, errors) from a pytest JUnit XML report."""
    root = ET.parse(path).getroot()  # noqa: S314 — local CI artefact
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    if not suites and root.tag == "testsuites":
        suites = list(root.findall("testsuite"))
    tests = failures = errors = 0
    for suite in suites:
        tests += int(suite.attrib.get("tests") or 0)
        failures += int(suite.attrib.get("failures") or 0)
        errors += int(suite.attrib.get("errors") or 0)
    return tests, failures, errors


def enforce_pass_rate(path: Path, *, min_rate: float) -> int:
    """Exit 0 when the JUnit report meets ``min_rate``, else 1."""
    tests, failures, errors = read_junit_counts(path)
    rate = pass_rate(tests=tests, failures=failures, errors=errors)
    passed = tests - failures - errors
    print(
        f"unit pass rate: {passed}/{tests} = {rate:.1%} "
        f"(required ≥ {min_rate:.0%}; failures={failures}, errors={errors})"
    )
    if rate + 1e-12 < min_rate:
        print(
            f"FAIL: pass rate {rate:.1%} is below the {min_rate:.0%} gate "
            f"for this branch.",
            file=sys.stderr,
        )
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("junit_xml", type=Path, help="pytest --junitxml path")
    parser.add_argument(
        "--min-pass-rate",
        type=float,
        default=0.95,
        help="Minimum fraction of unit tests that must pass (default: 0.95)",
    )
    args = parser.parse_args(argv)
    if not args.junit_xml.is_file():
        print(f"missing junit report: {args.junit_xml}", file=sys.stderr)
        return 1
    if not 0 < args.min_pass_rate <= 1:
        print("--min-pass-rate must be in (0, 1]", file=sys.stderr)
        return 1
    return enforce_pass_rate(args.junit_xml, min_rate=args.min_pass_rate)


if __name__ == "__main__":
    raise SystemExit(main())
