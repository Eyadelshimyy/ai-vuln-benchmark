"""Phase 14 bounded field-sensitive dataflow.

Constant attributes, keys, and indexes flow. Dynamic keys, reassigned aliases,
and unrelated instances stay unresolved. Static findings stay potential.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.config import Settings, settings
from app.domain.findings import FindingStatus
from app.domain.security import VulnerabilityClass
from app.parsing.engine import parse_source
from app.security.engine import SecurityAnalysisEngine
from app.security.language_vocab import vocab_for
from app.security.taint import analyze_taint, vocab_sources

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "phase14_repo"
EVAL = VulnerabilityClass.DYNAMIC_EXECUTION
XSS = VulnerabilityClass.XSS


def _scan(root: Path):
    paths = sorted(path for path in root.rglob("*") if path.is_file())
    return SecurityAnalysisEngine().analyze_repository(root, paths)


def _lines(result, vuln: VulnerabilityClass, filename: str) -> list[int]:
    return sorted(
        obs.line
        for obs in result.observations
        if obs.vulnerability_class is vuln and str(obs.file_path).endswith(filename)
    )


def _obs(result, vuln: VulnerabilityClass, filename: str, line: int):
    matches = [
        obs
        for obs in result.observations
        if obs.vulnerability_class is vuln
        and str(obs.file_path).endswith(filename)
        and obs.line == line
    ]
    assert len(matches) == 1
    return matches[0]


def test_constant_dict_key_and_attribute_flow() -> None:
    result = _scan(FIXTURE / "dict_field")
    assert _lines(result, EVAL, "app.py") == [3]
    assert _obs(result, EVAL, "app.py", 3).metadata["field_path"] == 'obj["payload"]'

    dotted = _scan(FIXTURE / "object_field")
    assert _lines(dotted, EVAL, "app.py") == [3]
    assert _obs(dotted, EVAL, "app.py", 3).metadata["field_path"] == "obj.payload"


def test_constant_index_flows_and_other_index_does_not() -> None:
    result = _scan(FIXTURE / "index_field")
    assert _lines(result, EVAL, "app.py") == [3]
    assert _obs(result, EVAL, "app.py", 3).metadata["field_path"] == "items[0]"


def test_dynamic_index_does_not_invent_a_key() -> None:
    result = _scan(FIXTURE / "dynamic_index")
    assert _lines(result, EVAL, "app.py") == []
    script = _scan(FIXTURE / "js_dynamic")
    assert _lines(script, EVAL, "app.js") == []


def test_field_reassignment_is_use_site_sensitive() -> None:
    cleared = _scan(FIXTURE / "reassign_clear")
    assert _lines(cleared, EVAL, "app.py") == [3]
    tainted = _scan(FIXTURE / "reassign_taint")
    assert _lines(tainted, EVAL, "app.py") == [5]


def test_field_sanitizer_is_vulnerability_specific() -> None:
    result = _scan(FIXTURE / "sanitizer")
    assert _lines(result, XSS, "app.py") == [8]
    assert _lines(result, EVAL, "app.py") == [9]
    assert "field_path" in _obs(result, EVAL, "app.py", 9).metadata
    assert all(item.status is not FindingStatus.VERIFIED for item in result.findings)


def test_proven_alias_flows_and_broken_alias_does_not() -> None:
    assert _lines(_scan(FIXTURE / "alias_write"), EVAL, "app.py") == [4]
    assert _lines(_scan(FIXTURE / "alias_read"), EVAL, "app.py") == [4]
    conditional = _scan(FIXTURE / "conditional_alias")
    assert _lines(conditional, EVAL, "app.py") == [7]
    reassigned = _scan(FIXTURE / "reassigned_alias")
    assert _lines(reassigned, EVAL, "app.py") == [7]


def test_distinct_instances_do_not_share_fields() -> None:
    result = _scan(FIXTURE / "two_instances")
    assert _lines(result, EVAL, "app.py") == [8]
    methods = _scan(FIXTURE / "class_method")
    assert _lines(methods, EVAL, "app.py") == [4]


def test_nested_depth_keeps_shallower_findings_and_reports_the_limit() -> None:
    result = _scan(FIXTURE / "nested")
    assert _lines(result, EVAL, "app.py") == [5]
    obs = _obs(result, EVAL, "app.py", 5)
    assert obs.metadata["field_path"] == "obj.payload"
    assert "field depth limit" in obs.metadata["analysis_incomplete"]


def test_whole_value_container_still_reaches_a_field_use() -> None:
    result = _scan(FIXTURE / "whole_value")
    assert _lines(result, EVAL, "app.py") == [2]


def test_cross_file_field_expression_flows_and_whole_object_does_not() -> None:
    argument = _scan(FIXTURE / "cross_arg")
    assert _lines(argument, EVAL, "app.py") == [5]
    assert _obs(argument, EVAL, "app.py", 5).metadata["field_path"] == "obj.payload"
    returned = _scan(FIXTURE / "cross_return")
    assert _lines(returned, EVAL, "app.py") == [4]
    shared = _scan(FIXTURE / "cross_object")
    assert _lines(shared, EVAL, "app.py") == []
    assert _lines(shared, EVAL, "helpers.py") == []


def test_javascript_constant_field_and_index() -> None:
    dotted = _scan(FIXTURE / "js_object")
    assert _lines(dotted, EVAL, "app.js") == [3]
    assert _obs(dotted, EVAL, "app.js", 3).metadata["field_path"] == "obj.payload"
    indexed = _scan(FIXTURE / "js_index")
    assert _lines(indexed, EVAL, "app.js") == [3]
    assert _obs(indexed, EVAL, "app.js", 3).metadata["field_path"] == "items[0]"


def test_zero_depth_and_binding_cap_are_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "taint_max_field_depth", 0)
    disabled = _scan(FIXTURE / "object_field")
    assert _lines(disabled, EVAL, "app.py") == []

    monkeypatch.setattr(settings, "taint_max_field_depth", 4)
    monkeypatch.setattr(settings, "taint_max_field_bindings", 1)
    capped = _scan(FIXTURE / "sanitizer")
    # Only the first field write is kept. The later escape and the other field are dropped.
    assert _lines(capped, EVAL, "app.py") == [9]
    assert _lines(capped, XSS, "app.py") == [7]
    assert "field binding limit" in _obs(capped, EVAL, "app.py", 9).metadata["analysis_incomplete"]

    monkeypatch.setattr(settings, "taint_max_field_bindings", 2000)
    monkeypatch.setattr(settings, "taint_max_alias_edges", 0)
    no_alias = _scan(FIXTURE / "alias_write")
    assert _lines(no_alias, EVAL, "app.py") == []


def test_negative_field_limits_are_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(taint_max_field_depth=-1)
    with pytest.raises(ValidationError):
        Settings(taint_max_field_bindings=-1)
    with pytest.raises(ValidationError):
        Settings(taint_max_alias_edges=-1)


def test_field_paths_are_recorded_only_for_constant_keys() -> None:
    python = parse_source(
        "python",
        Path("fields.py"),
        'obj["payload"] = q\nitems[0] = q\ndata[key] = q\nobj.payload = q\n',
    )
    names = {binding.name for binding in python.bindings}
    assert 'obj["payload"]' in names
    assert "items[0]" in names
    assert "obj.payload" in names
    assert not any(name.startswith("data[") for name in names)

    script = parse_source(
        "javascript",
        Path("fields.js"),
        'obj["payload"] = q;\nitems[0] = q;\nobj[key] = q;\nobj.payload = q;\n',
    )
    script_names = {binding.name for binding in script.bindings}
    assert 'obj["payload"]' in script_names
    assert "items[0]" in script_names
    assert "obj.payload" in script_names
    assert "obj" not in script_names
    assert not any(name.startswith("obj[key]") or name == "obj.key" for name in script_names)


def test_repeated_scans_match() -> None:
    first = _scan(FIXTURE / "cross_arg")
    second = _scan(FIXTURE / "cross_arg")

    def canon(result) -> list[tuple[str, int, str, str]]:
        return sorted(
            (
                obs.file_path,
                obs.line,
                obs.vulnerability_class.value,
                obs.metadata.get("field_path", ""),
            )
            for obs in result.observations
        )

    assert canon(first) == canon(second)
    graph = parse_source(
        "python",
        Path("fields.py"),
        'obj["payload"] = request.args.get("q")\neval(obj["payload"])\n',
    )
    vocab = vocab_for("python")
    assert vocab is not None
    sources = vocab_sources(vocab, ())
    assert analyze_taint(graph, sources).limit_reason == analyze_taint(graph, sources).limit_reason


def test_same_file_return_of_a_field(tmp_path: Path) -> None:
    root = tmp_path / "ret"
    root.mkdir()
    (root / "app.py").write_text(
        "def load():\n"
        "    obj = {}\n"
        "    obj.payload = request.args.get('q')\n"
        "    return obj.payload\n"
        "value = load()\n"
        "eval(value)\n"
    )
    result = _scan(root)
    assert _lines(result, EVAL, "app.py") == [6]
