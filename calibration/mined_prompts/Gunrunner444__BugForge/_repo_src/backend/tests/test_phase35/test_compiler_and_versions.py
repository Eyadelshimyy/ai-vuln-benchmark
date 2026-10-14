"""Phase 35 project compiler identity, paths, and version semantics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.parsing.engine import parse_source
from app.parsing.solidity_compiler import (
    CompilerSemantics,
    compiler_semantics,
    compiler_semantics_for_scan,
    interpret_standard_json,
)
from app.parsing.solidity_links import overload_selectors
from app.parsing.solidity_project import (
    build_compiler_project,
    current_compiler_project,
    project_standard_json,
    reset_compiler_project,
    reset_compiler_project_cache,
    resolve_analysis_source,
    set_compiler_project,
    set_compiler_project_cache,
)
from app.parsing.solidity_storage import analyze_storage, apply_compiler_layout
from app.parsing.solidity_version import solidity_language_facts
from app.plugins import reset_plugin_catalog
from app.security.engine import SecurityAnalysisEngine
from app.security.rules.base import SecurityRule


def _graph(tmp_path: Path, source: str, name: str):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return parse_source("solidity", path, source)


def test_relative_graph_path_uses_parsed_bytes(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    body = "pragma solidity ^0.8.20;\ncontract Token { uint256 value; }\n"
    graph = _graph(repo, body, "src/Token.sol")
    graph.file_path = "src/Token.sol"
    (repo / "src" / "Token.sol").write_text(
        "pragma solidity ^0.8.20;\ncontract Other {}\n", encoding="utf-8"
    )
    seen: dict[str, str] = {}

    def runner(payload: str) -> str:
        seen["payload"] = payload
        return '{"version":"0.8.26","contracts":{"src/Token.sol":{"Token":{}}}}'

    project = build_compiler_project(repo, {"src/Token.sol": graph}, runner=runner)
    assert project.status == "AVAILABLE"
    payload = json.loads(seen["payload"])
    assert payload["sources"]["src/Token.sol"]["content"] == body
    assert "BugForge.sol" not in payload["sources"]


def test_escaping_path_is_not_read(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "outside.sol"
    outside.write_text("contract Secret { uint256 hidden; }", encoding="utf-8")
    status, relative, text = resolve_analysis_source(repo, str(outside), "contract Secret {}")
    assert status == "rejected"
    assert relative == ""
    assert text == ""
    status, _, text = resolve_analysis_source(repo, "../outside.sol", "contract Secret {}")
    assert status == "rejected"
    assert text == ""
    called = False

    def runner(_payload: str) -> str:
        nonlocal called
        called = True
        return '{"version":"0.8.26"}'

    outside_graph = _graph(tmp_path, "contract Secret {}", "outside.sol")
    project = build_compiler_project(repo, {str(outside): outside_graph}, runner=runner)
    assert project.status != "AVAILABLE"
    assert any("rejected" in item for item in project.diagnostics)
    assert called is False


def test_project_bundle_keeps_import_paths(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    token = "pragma solidity ^0.8.20;\ninterface Token { function balanceOf(address) external view returns (uint256); }\n"
    vault = 'pragma solidity ^0.8.20;\nimport "./Token.sol";\ncontract Vault { Token token; }\n'
    graphs = {
        "src/Token.sol": _graph(repo, token, "src/Token.sol"),
        "src/Vault.sol": _graph(repo, vault, "src/Vault.sol"),
    }
    for graph, public in zip(graphs.values(), graphs, strict=True):
        graph.file_path = public
    payload = project_standard_json(
        {"src/Token.sol": token, "src/Vault.sol": vault}, ["oz/=lib/oz/"]
    )
    sources = payload["sources"]
    assert isinstance(sources, dict)
    assert sources["src/Token.sol"]["content"] == token
    assert 'import "./Token.sol"' in sources["src/Vault.sol"]["content"]
    project = build_compiler_project(
        repo, graphs, runner=lambda _payload: '{"errors":[{"severity":"error"}]}'
    )
    assert project.status == "FAILED"
    assert project.complete is False
    assert "src/Token.sol" in project.sources
    assert "src/Vault.sol" in project.sources


def test_repeated_contract_names_stay_distinct(tmp_path: Path) -> None:
    left = "pragma solidity ^0.8.20;\ncontract C { address owner; }\n"
    right = "pragma solidity ^0.8.20;\ncontract C { address owner; }\n"
    repo = tmp_path / "repo"
    left_graph = _graph(repo, left, "src/a/C.sol")
    right_graph = _graph(repo, right, "src/b/C.sol")
    compiler = CompilerSemantics(
        "AVAILABLE",
        "solc",
        "test",
        layouts=[
            {
                "contract": "C",
                "label": "owner",
                "slot": "0",
                "source": "src/a/C.sol",
                "type": "address",
            },
            {
                "contract": "C",
                "label": "owner",
                "slot": "3",
                "source": "src/b/C.sol",
                "type": "address",
            },
        ],
    )
    left_model = apply_compiler_layout(
        analyze_storage(left_graph), compiler, source_path="src/a/C.sol"
    )
    right_model = apply_compiler_layout(
        analyze_storage(right_graph), compiler, source_path="src/b/C.sol"
    )
    assert left_model.compiler_matches == 1
    assert right_model.compiler_matches == 1
    assert any(item.compiler_slot == "3" for item in right_model.disagreements)
    assert all(item.compiler_slot != "3" for item in left_model.disagreements)


def test_scan_compiler_context_is_cleared(tmp_path: Path) -> None:
    seen: list[object] = []

    class Probe(SecurityRule):
        rule_id = "probe"
        vulnerability_class = None  # type: ignore[assignment]

        def check(self, graph, *, frameworks=()):  # type: ignore[no-untyped-def]
            del graph, frameworks
            seen.append(current_compiler_project())
            return []

    repo = tmp_path / "repo"
    source = "pragma solidity ^0.8.20;\ncontract C { uint256 value; }\n"
    path = repo / "C.sol"
    path.parent.mkdir(parents=True)
    path.write_text(source, encoding="utf-8")
    SecurityAnalysisEngine(rules=[Probe()]).analyze_repository(repo, [path])
    assert seen and seen[0] is not None
    assert current_compiler_project() is None
    other = tmp_path / "other"
    other_path = other / "D.sol"
    other_path.parent.mkdir(parents=True)
    other_path.write_text(
        "pragma solidity ^0.8.20;\ncontract D { uint256 other; }\n", encoding="utf-8"
    )
    SecurityAnalysisEngine(rules=[Probe()]).analyze_repository(other, [other_path])
    assert seen[0] is not seen[1]
    assert getattr(seen[0], "source_identity", "") != getattr(seen[1], "source_identity", "")
    assert current_compiler_project() is None
    reset_plugin_catalog()


def test_project_cache_is_scan_local(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "repo"
    source = "pragma solidity ^0.8.20;\ncontract C { uint256 value; }\n"
    graph = _graph(repo, source, "C.sol")
    graph.file_path = "C.sol"
    monkeypatch.setattr("app.parsing.solidity_project._compiler_tool", lambda: ("solc", False))
    token = set_compiler_project_cache()
    try:
        first = build_compiler_project(repo, {"C.sol": graph})
        second = build_compiler_project(repo, {"C.sol": graph})
        assert first is second
        changed = _graph(repo, source + "\n", "C2.sol")
        changed.file_path = "C2.sol"
        other = build_compiler_project(repo, {"C2.sol": changed})
        assert other is not first
        monkeypatch.setattr("app.parsing.solidity_project._compiler_tool", lambda: ("forge", False))
        shifted = build_compiler_project(repo, {"C.sol": graph})
        assert shifted is not first
    finally:
        reset_compiler_project_cache(token)
    fresh = set_compiler_project_cache()
    try:
        again = build_compiler_project(repo, {"C.sol": graph})
        assert again is not first
    finally:
        reset_compiler_project_cache(fresh)


def test_host_compiler_is_disabled_and_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.parsing.solidity_compiler.shutil.which",
        lambda name: "/usr/bin/solc" if name == "solc" else None,
    )
    absent = compiler_semantics_for_scan("contract C {}")
    assert absent.status == "UNAVAILABLE"
    assert "disabled" in absent.detail
    assert absent.storage == []

    class _Settings:
        solidity_host_compiler = True

    monkeypatch.setattr("app.core.config.get_settings", lambda: _Settings())

    class _Completed:
        stdout = "x" * 1_000_001
        stderr = ""

    def _run(*_args: object, **kwargs: object) -> _Completed:
        assert kwargs.get("timeout") == 20
        env = kwargs.get("env")
        assert isinstance(env, dict)
        assert set(env) == {"PATH"}
        return _Completed()

    monkeypatch.setattr("subprocess.run", _run)
    with pytest.raises(ValueError, match="bound"):
        from app.parsing.solidity_compiler import run_solc_standard_json

        run_solc_standard_json("{}")

    def explode(_source: str) -> str:
        raise TimeoutError("slow")

    failed = compiler_semantics("contract C {}", runner=explode)
    assert failed.status == "FAILED"
    assert failed.storage == []


def test_compiler_json_reports_ast_ir_and_truncation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.parsing.solidity_compiler.shutil.which",
        lambda name: "/usr/bin/solc" if name == "solc" else None,
    )
    ir = "Yul " + ("a" * 70_000)
    raw = json.dumps(
        {
            "version": "0.8.26+commit.abc",
            "sources": {"src/C.sol": {"ast": {"nodeType": "SourceUnit"}}},
            "contracts": {
                "src/C.sol": {
                    "C": {
                        "ir": ir,
                        "evm": {"methodIdentifiers": {"value()": "3fa4f245"}},
                        "storageLayout": {
                            "storage": [{"label": "value", "slot": "0", "offset": 0, "type": "t"}]
                        },
                    }
                }
            },
        }
    )
    parsed = interpret_standard_json(raw, tool="solc")
    assert parsed.ast_available is True
    assert parsed.ir_available is True
    assert parsed.ir_truncated is True
    assert len(parsed.ir) <= 65_536
    assert parsed.selectors[0]["source"] == "src/C.sol"
    missing = compiler_semantics("contract C {}", runner=lambda _source: "")
    assert missing.status in {"FAILED", "UNAVAILABLE"}


def test_version_semantics_are_conservative() -> None:
    assert solidity_language_facts("pragma solidity 0.8.20;").arithmetic == "checked"
    assert solidity_language_facts("pragma solidity ^0.8.20;").arithmetic == "checked"
    assert solidity_language_facts("pragma solidity ^0.7.5;").arithmetic == "wrapping"
    assert solidity_language_facts("pragma solidity >=0.8.0 <0.9.0;").arithmetic == "checked"
    ranged = solidity_language_facts("pragma solidity >=0.7.0 <0.9.0;")
    assert ranged.arithmetic == "unknown"
    assert solidity_language_facts("pragma solidity 0.8.20 || 0.7.0;").arithmetic == "unknown"
    assert solidity_language_facts("pragma solidity not-a-version;").arithmetic == "unknown"
    commented = solidity_language_facts("// pragma solidity 0.4.0\ncontract C {}")
    assert commented.arithmetic == "unknown"
    quoted = solidity_language_facts('contract C { string s = "pragma solidity 0.4.0"; }')
    assert quoted.arithmetic == "unknown"
    both = solidity_language_facts("pragma solidity ^0.8.0;\npragma solidity >=0.8.20 <0.8.30;")
    assert both.arithmetic == "checked"
    assert both.floor == "0.8.20"
    compiler = solidity_language_facts("", "0.8.26+commit.abc")
    assert compiler.source == "compiler"
    assert compiler.selfdestruct == "deprecated"
    disagree = solidity_language_facts("pragma solidity 0.7.6;", "0.8.26+commit.abc")
    assert disagree.arithmetic == "unknown"


def test_overloads_keep_distinct_selectors(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    contract Token {
        function transfer(address to) external {}
        function transfer(address to, uint256 amount) external {}
        function broken(Unknown value) external {}
    }
    """
    rows = overload_selectors(_graph(tmp_path, source, "Token.sol"))
    symbols = {row["symbol"] for row in rows}
    assert "transfer(address)" in symbols
    assert "transfer(address,uint256)" in symbols
    assert len([row for row in rows if row["symbol"].startswith("transfer(")]) == 2
    assert all("broken(" not in row["symbol"] for row in rows)


def test_context_reset_helper_clears_a_model() -> None:
    from app.parsing.solidity_project import CompilerProjectModel

    token = set_compiler_project(CompilerProjectModel(status="UNAVAILABLE", detail="scan"))
    assert current_compiler_project() is not None
    reset_compiler_project(token)
    assert current_compiler_project() is None
