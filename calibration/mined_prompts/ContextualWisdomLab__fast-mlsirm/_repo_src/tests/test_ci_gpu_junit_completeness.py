"""Source-only GPU JUnit admission controls; no GPU or fitting is executed."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path
import re
import textwrap
from xml.etree import ElementTree as ET

import pytest


_WORKFLOW = Path(__file__).resolve().parents[1] / ".github/workflows/ci.yml"
_SELECTORS = (
    "tests/test_marginal_parity.py::test_marginal_gpu_agrees_with_cpu_loosely",
    "tests/test_bifactor_gpu_high_q.py::test_bifactor_gpu_parity_q121",
    "tests/test_bifactor_gpu_high_q.py::test_bifactor_gpu_parity_q241",
    "tests/test_bifactor_gpu_high_q.py::test_bifactor_gpu_parity_q481",
    "tests/test_bifactor_gpu_high_q.py::test_bifactor_gpu_parity_q241_wide_items_metal_workgroups",
    "tests/test_bifactor_gpu_high_q.py::test_bifactor_cpu_q121_vs_q241_agree",
    "tests/test_bifactor_bootstrap_benchmark.py::test_joint_bootstrap_cpu_vs_gpu_wall_time_q121",
)
_IDENTITIES = tuple(
    (path[:-3].replace("/", "."), name)
    for path, name in (selector.split("::") for selector in _SELECTORS)
)


def _guard() -> str:
    """Extract only the real workflow guard, not a fixture implementation."""
    workflow = _WORKFLOW.read_text(encoding="utf-8")
    step = workflow.split("      - name: Reject skipped GPU evidence\n", 1)[1]
    script = step.split("python - <<'PY'\n", 1)[1].split("          PY\n", 1)[0]
    return textwrap.dedent(script)


def _suite(identities: tuple = _IDENTITIES) -> ET.Element:
    """Create inert pytest-shaped transport rows, not scientific results."""
    suite = ET.Element(
        "testsuite", tests=str(len(identities)), failures="0", errors="0", skipped="0"
    )
    for classname, name in identities:
        ET.SubElement(suite, "testcase", classname=classname, name=name, time="0")
    return suite


def _execute(xml: ET.Element, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the exact guard in a private XML namespace without native imports."""
    (tmp_path / "gpu-junit.xml").write_bytes(ET.tostring(xml))
    monkeypatch.chdir(tmp_path)
    exec(compile(_guard(), str(_WORKFLOW) + "::gpu-junit-guard", "exec"), {})


def _reject(xml: ET.Element, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Require an actual admission refusal, not a harness setup error."""
    with pytest.raises(SystemExit):
        _execute(xml, tmp_path, monkeypatch)


@pytest.mark.parametrize("wrapper", [False, True], ids=["suite", "suites"])
@pytest.mark.parametrize("family", ["xunit1", "xunit2"])
def test_complete_exact_seven_is_accepted(tmp_path, monkeypatch, wrapper, family):
    """Both installed pytest JUnit families preserve classname/name identities."""
    from _pytest.junitxml import mangle_test_address

    assert tuple(
        (".".join(parts[:-1]), parts[-1])
        for parts in map(mangle_test_address, _SELECTORS)
    ) == _IDENTITIES
    suite = _suite()
    if family == "xunit1":
        for case, selector in zip(suite, _SELECTORS, strict=True):
            case.set("file", selector.split("::")[0])
            case.set("line", "1")
    ET.SubElement(suite, "properties")
    if wrapper:
        xml = ET.Element("testsuites", name="pytest tests")
        xml.append(suite)
    else:
        xml = suite
    _execute(xml, tmp_path, monkeypatch)


@pytest.mark.parametrize("case", ["subset", "unrelated", "header_only"])
def test_incomplete_evidence_is_rejected(tmp_path, monkeypatch, case):
    """Reproduce the three retained false admissions against real guard bytes."""
    if case == "subset":
        xml = _suite(_IDENTITIES[:1])
    elif case == "unrelated":
        xml = _suite((("tests.test_unrelated", "test_other"),))
    else:
        xml = _suite(())
        xml.set("tests", "7")
    _reject(xml, tmp_path, monkeypatch)


@pytest.mark.parametrize("case", ["missing", "extra", "duplicate", "duplicate_replaces_missing", "wrong_classname", "missing_name", "parameterized_name"])
def test_exact_testcase_identity_is_required(tmp_path, monkeypatch, case):
    """Reject missing, extra, repeated or only name-matching selector evidence."""
    xml = _suite()
    if case == "missing":
        xml.remove(xml[-1])
    elif case == "extra":
        ET.SubElement(xml, "testcase", classname="tests.test_other", name="test_other")
    elif case == "duplicate":
        xml.append(deepcopy(xml[0]))
    elif case == "duplicate_replaces_missing":
        xml.remove(xml[-1])
        xml.append(deepcopy(xml[0]))
    elif case == "wrong_classname":
        xml[0].set("classname", "tests.test_other")
    elif case == "missing_name":
        del xml[0].attrib["name"]
    else:
        xml[0].set("name", xml[0].attrib["name"] + "[unexpected]")
    xml.set("tests", str(len(xml)))
    _reject(xml, tmp_path, monkeypatch)


@pytest.mark.parametrize("field", ["tests", "failures", "errors", "skipped"])
@pytest.mark.parametrize("value", [None, "-1", "bogus", "1"])
def test_headers_are_required_and_bound_to_rows(tmp_path, monkeypatch, field, value):
    """No missing, malformed, nonzero-outcome or mismatched count is evidence."""
    xml = _suite()
    if value is None:
        del xml.attrib[field]
    else:
        xml.set(field, value)
    _reject(xml, tmp_path, monkeypatch)


@pytest.mark.parametrize("outcome", ["failure", "error", "skipped"])
def test_hidden_row_outcomes_are_rejected(tmp_path, monkeypatch, outcome):
    """Even zero outcome headers cannot hide an actual failed/error/skip row."""
    xml = _suite()
    ET.SubElement(xml[0], outcome, message="inert negative control")
    _reject(xml, tmp_path, monkeypatch)


def test_multiple_complete_suites_are_counted(tmp_path, monkeypatch):
    """Support direct pytest-shaped suites without treating headers as rows."""
    xml = ET.Element("testsuites")
    xml.extend((_suite(_IDENTITIES[:3]), _suite(_IDENTITIES[3:])))
    _execute(xml, tmp_path, monkeypatch)


@pytest.mark.parametrize("case", ["empty", "nested", "wrong_root", "stray_row"])
def test_unsupported_or_incomplete_layout_is_rejected(tmp_path, monkeypatch, case):
    """Reject ignored testcase containers and empty or non-JUnit roots."""
    xml = ET.Element("testsuites")
    if case == "nested":
        outer = ET.SubElement(xml, "testsuite", tests="7", failures="0", errors="0", skipped="0")
        outer.append(_suite())
    elif case == "wrong_root":
        xml.tag = "report"
        xml.append(_suite())
    elif case == "stray_row":
        xml.append(_suite())
        xml.append(ET.Element("testcase", classname="tests.other", name="test_other"))
    _reject(xml, tmp_path, monkeypatch)


@pytest.mark.parametrize("field", ["tests", "failures", "errors", "skipped"])
@pytest.mark.parametrize("value", ["0", "8", "-1", "invalid"])
def test_present_aggregate_headers_match_rows(tmp_path, monkeypatch, field, value):
    """Optional wrapper totals must agree with actual successful testcase rows."""
    xml = ET.Element("testsuites", {field: value})
    xml.append(_suite())
    if field != "tests" and value == "0":
        _execute(xml, tmp_path, monkeypatch)
    else:
        _reject(xml, tmp_path, monkeypatch)


@pytest.mark.parametrize("outcome", ["failure", "error", "skipped"])
def test_matching_nonzero_headers_are_rejected(tmp_path, monkeypatch, outcome):
    """Truthful failure counts still cannot admit unsuccessful evidence."""
    xml = _suite()
    ET.SubElement(xml[0], outcome)
    xml.set({"failure": "failures", "error": "errors", "skipped": "skipped"}[outcome], "1")
    _reject(xml, tmp_path, monkeypatch)


def test_nested_extra_testcase_is_rejected(tmp_path, monkeypatch):
    """An ignored nested testcase cannot hide an additional executed identity."""
    xml = _suite()
    ET.SubElement(xml[0], "testcase", classname="tests.other", name="test_nested")
    _reject(xml, tmp_path, monkeypatch)


def test_guard_manifest_matches_unchanged_positional_science_command():
    """Bind the expected tuple inventory to the original command's selectors."""
    workflow = _WORKFLOW.read_text(encoding="utf-8")
    command = workflow.split("pytest -q --junitxml=gpu-junit.xml", 1)[1].split(
        "      - name: Reject skipped GPU evidence", 1
    )[0]
    selectors = re.findall(r"tests/[^\s\\]+::[^\s\\]+", command)
    assert Counter(selectors) == Counter(_SELECTORS)
    assert len(selectors) == 7
