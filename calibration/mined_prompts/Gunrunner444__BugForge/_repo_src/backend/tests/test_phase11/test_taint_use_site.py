"""Use-site flow-sensitive taint: a sink sees the definition reaching that call.

The final per-symbol map must not make an earlier sink inherit a later
definition of the same name. Taint is not sticky: a later safe assignment
clears later uses without erasing earlier tainted uses.
"""

from __future__ import annotations

from pathlib import Path

from app.domain.security import VulnerabilityClass
from tests.language_support import analyze_source


def _lines(result, vuln: VulnerabilityClass) -> list[int]:
    return sorted(obs.line for obs in result.observations if obs.vulnerability_class is vuln)


def test_earlier_safe_eval_does_not_see_later_taint(tmp_path: Path) -> None:
    src = """
q = "safe"
eval(q)

q = request.args.get("q")
"""
    result = analyze_source(tmp_path, "a.py", src)
    assert VulnerabilityClass.DYNAMIC_EXECUTION not in {
        obs.vulnerability_class for obs in result.observations
    }


def test_earlier_tainted_eval_not_cleared_by_later_safe(tmp_path: Path) -> None:
    src = """
q = request.args.get("q")
eval(q)

q = "safe"
"""
    result = analyze_source(tmp_path, "b.py", src)
    assert _lines(result, VulnerabilityClass.DYNAMIC_EXECUTION)


def test_taint_between_two_safe_uses(tmp_path: Path) -> None:
    src = """
q = "safe"
eval(q)

q = request.args.get("q")
eval(q)

q = "safe"
eval(q)
"""
    result = analyze_source(tmp_path, "c.py", src)
    lines = _lines(result, VulnerabilityClass.DYNAMIC_EXECUTION)
    assert lines == [6], lines


def test_safe_then_tainted_then_sink(tmp_path: Path) -> None:
    src = """
q = "safe"
q = request.args.get("q")
eval(q)
"""
    result = analyze_source(tmp_path, "d.py", src)
    assert _lines(result, VulnerabilityClass.DYNAMIC_EXECUTION)


def test_tainted_then_safe_then_sink(tmp_path: Path) -> None:
    src = """
q = request.args.get("q")
q = "safe"
eval(q)
"""
    result = analyze_source(tmp_path, "e.py", src)
    assert not _lines(result, VulnerabilityClass.DYNAMIC_EXECUTION)


def test_sql_command_path_ssrf_xss_redirect_use_sites(tmp_path: Path) -> None:
    src = """
from flask import redirect

q = "safe"
cursor.execute(q)
os.system(q)
open(q)
requests.get(q)
markupsafe.Markup(q)
redirect(q)

q = request.args.get("q")
cursor.execute("SELECT " + q)
os.system(q)
open(q)
requests.get(q)
markupsafe.Markup(q)
redirect(q)

q = "safe"
cursor.execute(q)
os.system(q)
open(q)
requests.get(q)
markupsafe.Markup(q)
redirect(q)
"""
    result = analyze_source(tmp_path, "multi.py", src)
    expected = {
        VulnerabilityClass.SQL_INJECTION,
        VulnerabilityClass.COMMAND_INJECTION,
        VulnerabilityClass.POTENTIAL_PATH_TRAVERSAL,
        VulnerabilityClass.SSRF,
        VulnerabilityClass.XSS,
        VulnerabilityClass.UNSAFE_REDIRECT,
    }
    by_class: dict[VulnerabilityClass, list[int]] = {cls: [] for cls in expected}
    for obs in result.observations:
        if obs.vulnerability_class in by_class:
            by_class[obs.vulnerability_class].append(obs.line)
    for cls, lines in by_class.items():
        assert lines, (cls, result.observations)
        assert min(lines) >= 13, (cls, lines)
        assert max(lines) <= 18, (cls, lines)


def test_forward_return_use_site_and_later_clear(tmp_path: Path) -> None:
    src = """
def forward(x):
    return x

a = "safe"
b = forward(request.args.get("q"))
eval(a)
eval(b)
b = "safe"
eval(b)
"""
    result = analyze_source(tmp_path, "fwd.py", src)
    lines = _lines(result, VulnerabilityClass.DYNAMIC_EXECUTION)
    assert lines == [8], lines


def test_parameter_reassignment_use_sites(tmp_path: Path) -> None:
    src = """
def wrap(value):
    eval(value)
    value = "safe"
    eval(value)

wrap(request.args.get("q"))
"""
    result = analyze_source(tmp_path, "param.py", src)
    lines = _lines(result, VulnerabilityClass.DYNAMIC_EXECUTION)
    assert lines == [3], lines


def test_nested_scope_does_not_taint_outer_safe_use(tmp_path: Path) -> None:
    src = """
function f() {
    let q = "safe";
    eval(q);
    if (true) {
        let q = req.query.q;
        eval(q);
    }
    eval(q);
}
"""
    result = analyze_source(tmp_path, "nest.js", src)
    lines = _lines(result, VulnerabilityClass.DYNAMIC_EXECUTION)
    assert lines == [7], lines


def test_sibling_scope_sinks(tmp_path: Path) -> None:
    src = """
function A() {
    let q = req.query.q;
    eval(q);
}
function B() {
    let q = "safe";
    eval(q);
}
"""
    result = analyze_source(tmp_path, "sib.js", src)
    lines = _lines(result, VulnerabilityClass.DYNAMIC_EXECUTION)
    assert lines == [4], lines


def test_sanitizer_does_not_apply_to_later_raw_use(tmp_path: Path) -> None:
    src = """
q = request.query.q;
el.innerHTML = escapeHtml(q);
q = request.query.q;
el.innerHTML = q;
"""
    result = analyze_source(tmp_path, "san.js", src)
    lines = _lines(result, VulnerabilityClass.XSS)
    assert lines, lines
    assert min(lines) >= 5, lines


def test_sanitizer_on_sink_argument_only(tmp_path: Path) -> None:
    src = """
q = request.query.q;
el.innerHTML = escapeHtml(q);
el.innerHTML = q;
"""
    result = analyze_source(tmp_path, "san2.js", src)
    lines = _lines(result, VulnerabilityClass.XSS)
    assert 4 in lines, lines
    assert 3 not in lines, lines


def test_safe_then_tainted_then_safe_then_sink(tmp_path: Path) -> None:
    src = """
q = "safe"
q = request.args.get("q")
q = "safe"
eval(q)
"""
    result = analyze_source(tmp_path, "sts.py", src)
    assert not _lines(result, VulnerabilityClass.DYNAMIC_EXECUTION)


def test_conditional_merge_taints_later_sink(tmp_path: Path) -> None:
    src = """
q = "safe"
if cond:
    q = request.args.get("q")
eval(q)
"""
    result = analyze_source(tmp_path, "cond.py", src)
    assert _lines(result, VulnerabilityClass.DYNAMIC_EXECUTION)


def test_nested_python_inner_does_not_taint_outer_safe(tmp_path: Path) -> None:
    src = """
def outer():
    q = "safe"
    eval(q)
    def inner():
        q = request.args.get("q")
        eval(q)
    eval(q)
"""
    result = analyze_source(tmp_path, "nest.py", src)
    lines = _lines(result, VulnerabilityClass.DYNAMIC_EXECUTION)
    assert lines == [7], lines
