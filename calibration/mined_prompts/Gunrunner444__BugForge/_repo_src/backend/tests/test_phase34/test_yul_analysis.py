"""Phase 34 bounded Yul structure and compiler/IR reconciliation."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.domain.evidence import VERIFICATION_PROVENANCE, EvidenceProvenance
from app.parsing.engine import parse_source, reset_syntax_registry
from app.parsing.solidity_compiler import (
    CompilerSemantics,
    compiler_cache_key,
    compiler_semantics,
    compiler_semantics_for_scan,
    interpret_standard_json,
    language_semantics,
    pragma_solidity_version,
    reconcile_selectors,
    reconcile_semantics,
    record_compiler_type_hints,
    reset_compiler_cache,
    reset_compiler_type_hints,
    set_compiler_cache,
    set_compiler_type_hints,
)
from app.parsing.solidity_defi import analyze_defi
from app.parsing.solidity_proxy import analyze_proxy
from app.parsing.solidity_storage import (
    analyze_storage,
    apply_compiler_layout,
    reset_storage_context,
    set_storage_context,
)
from app.parsing.solidity_yul import analyze_yul
from app.plugins import reset_plugin_catalog
from app.security.engine import SecurityAnalysisEngine

_IMPL = "0x360894a13ba1a3210667c828492db98dca3e2076cc3735a920a3ca505d382bbc"


@pytest.fixture(autouse=True)
def _fresh() -> None:
    reset_syntax_registry()
    reset_plugin_catalog()


def _graph(tmp_path: Path, source: str, name: str = "Yul.sol"):
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return parse_source("solidity", path, source)


def _model(tmp_path: Path, source: str):
    return analyze_yul(_graph(tmp_path, source))


def test_literal_and_constant_sload_are_known(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    contract A {
        uint256 constant SLOT = 7;
        function read() external {
            assembly {
                let literal := sload(0)
                let named := sload(SLOT)
            }
        }
    }
    """
    model = _model(tmp_path, source)
    summaries = {item.summary: item.known for item in model.traces}
    assert summaries["literal <- storage[0]"] is True
    assert summaries["named <- storage[SLOT]"] is True
    named = next(item for item in model.loads if item.variable == "named")
    assert named.slot.slot_class == "known"
    assert model.hostile == []
    assert model.incomplete is False


def test_sload_flows_into_delegatecall(tmp_path: Path) -> None:
    source = f"""
    pragma solidity ^0.8.20;
    contract Proxy {{
        bytes32 constant SLOT = {_IMPL};
        fallback() external payable {{
            assembly {{
                let impl := sload(SLOT)
                let ok := delegatecall(gas(), impl, 0, 0, 0, 0)
                if ok {{ stop() }}
            }}
        }}
    }}
    """
    model = _model(tmp_path, source)
    summaries = [item.summary for item in model.traces]
    assert "impl <- storage[SLOT]" in summaries
    assert "delegatecall.target <- impl" in summaries
    assert all(item.known for item in model.traces if "impl" in item.summary)
    assert "dynamic delegatecall target" not in model.hostile
    stored = analyze_storage(_graph(tmp_path, source, "Flow.sol"))
    assert stored.yul_detail is not None
    assert stored.yul_detail.calls
    assert isinstance(stored.yul, list)


def test_critical_sstore_and_dynamic_delegatecall(tmp_path: Path) -> None:
    source = f"""
    pragma solidity ^0.8.20;
    contract Proxy {{
        bytes32 constant SLOT = {_IMPL};
        function write(uint256 base) external {{
            assembly {{
                sstore(SLOT, 1)
                let target := add(base, 1)
                let ok := delegatecall(gas(), target, 0, 0, 0, 0)
                if ok {{ stop() }}
            }}
        }}
    }}
    """
    model = _model(tmp_path, source)
    assert "sstore into implementation slot" in model.hostile
    assert "dynamic delegatecall target" in model.hostile
    assert "ignored delegatecall success" not in model.hostile


def test_call_after_store_is_not_a_write_after_call(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    contract A {
        function go(address addr) external {
            assembly {
                sstore(0, 1)
                let ok := call(gas(), addr, 0, 0, 0, 0, 0)
                if ok { stop() }
            }
        }
    }
    """
    model = _model(tmp_path, source)
    assert any(item.summary == "call after storage write" for item in model.traces)
    assert "storage write after external control transfer" not in model.hostile


def test_store_after_call_is_hostile(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    contract A {
        function go(address addr) external {
            assembly {
                let ok := call(gas(), addr, 0, 0, 0, 0, 0)
                if ok { sstore(0, 1) }
            }
        }
    }
    """
    model = _model(tmp_path, source)
    assert "storage write after external control transfer" in model.hostile
    assert model.stores[0].branch_dependent is True


def test_nested_assembly_is_walked(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    contract A {
        function go() external {
            assembly {
                assembly {
                    let x := sload(1)
                }
            }
        }
    }
    """
    model = _model(tmp_path, source)
    assert any(item.summary == "nested assembly" for item in model.traces)
    assert any(item.variable == "x" and item.slot.slot_class == "known" for item in model.loads)


def test_comments_strings_and_unrelated_arithmetic_are_not_hostile(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    contract A {
        uint256 constant OTHER = 4;
        function go() external {
            // sstore(0, 1) delegatecall(gas(), impl, 0, 0, 0, 0)
            string memory note = "sstore delegatecall";
            assembly {
                let x := add(1, 2)
                let y := sload(OTHER)
            }
        }
    }
    """
    model = _model(tmp_path, source)
    assert model.calls == []
    assert model.stores == []
    assert model.hostile == []
    assert any(
        item.slot_class == "computed" for item in (var.expression for var in model.variables)
    )


def test_computed_slot_stays_unknown(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    contract A {
        function go() external {
            assembly { let x := sload(add(1, 2)) }
        }
    }
    """
    model = _model(tmp_path, source)
    assert model.loads[0].slot.slot_class == "computed"
    assert model.loads[0].slot.slot_class != "known"
    assert model.hostile == []


def test_duplicate_constants_and_unknown_names_stay_ambiguous(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    contract One {
        uint256 constant SLOT = 1;
        function read() external { assembly { let x := sload(SLOT) } }
    }
    contract Two {
        uint256 constant SLOT = 2;
        function read() external { assembly { let y := sload(mystery) } }
    }
    """
    model = _model(tmp_path, source)
    classes = {item.variable: item.slot.slot_class for item in model.loads}
    assert classes["x"] == "unknown"
    assert classes["y"] == "unknown"
    assert "dynamic delegatecall target" not in model.hostile


def test_caller_delegatecall_and_ignored_success(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    contract A {
        function go() external {
            assembly {
                let target := calldataload(0)
                delegatecall(gas(), target, 0, 0, 0, 0)
                returndatacopy(0, 0, returndatasize())
            }
        }
    }
    """
    model = _model(tmp_path, source)
    assert "delegatecall to a caller-derived value" in model.hostile
    assert "ignored delegatecall success" in model.hostile
    assert "return-data confusion" in model.hostile
    assert model.calls[0].branch_dependent is False


def test_branches_mark_stores_and_calls(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    contract A {
        function go(uint256 value, address impl) external {
            assembly {
                if gt(value, 0) { sstore(0, 1) }
                else { sstore(1, 2) }
                switch value
                case 0 { let ok := delegatecall(gas(), impl, 0, 0, 0, 0) if ok { stop() } }
                default { sstore(2, 3) }
            }
        }
    }
    """
    model = _model(tmp_path, source)
    kinds = [item.kind for item in model.branches]
    assert "if" in kinds and "else" in kinds and "switch" in kinds
    assert all(item.branch_dependent for item in model.stores)
    assert model.calls and model.calls[0].branch_dependent is True


def test_node_limit_is_incomplete_not_safe(tmp_path: Path) -> None:
    body = " ".join(f"let a{index} := {index};" for index in range(250))
    source = f"""
    pragma solidity ^0.8.20;
    contract A {{
        function go() external {{ assembly {{ {body} }} }}
    }}
    """
    model = _model(tmp_path, source)
    assert model.incomplete is True
    assert "unknown" in model.limit_reason
    assert "safe" not in model.limit_reason.lower()


def test_benign_yul_does_not_gain_a_yul_finding(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    contract A {
        function touch() external { assembly { let x := sload(0) mstore(0, x) } }
    }
    """
    path = tmp_path / "A.sol"
    path.write_text(source, encoding="utf-8")
    result = SecurityAnalysisEngine().analyze_repository(tmp_path, [path])
    ids = {item.rule_id for item in result.observations}
    assert "sol.assembly_sensitive" not in ids
    assert "sol.yul" not in ids
    assert all(item.metadata.get("status") == "potential" for item in result.observations)


def test_compiler_json_paths_do_not_require_solc() -> None:
    absent = interpret_standard_json("not-json")
    assert absent.status == "FAILED" and absent.storage == []
    errors = interpret_standard_json('{"errors":[{"severity":"error","message":"nope"}]}')
    assert errors.status == "FAILED" and errors.storage == []
    empty = interpret_standard_json("{}")
    assert empty.status == "FAILED"
    assert "requested output" in empty.detail
    ast_only = interpret_standard_json('{"sources":{"C.sol":{"ast":{"nodeType":"SourceUnit"}}}}')
    assert ast_only.status == "AVAILABLE" and ast_only.ast_available is True
    assert ast_only.storage == []
    ir_only = interpret_standard_json('{"contracts":{"C.sol":{"C":{"ir":"object \\"C\\" {}"}}}}')
    assert ir_only.ir_available is True and ir_only.storage == []
    layout = interpret_standard_json(
        '{"contracts":{"C.sol":{"C":{"storageLayout":{"storage":[{"label":"owner","slot":"0"}]}}}},'
        '"version":"0.8.20+commit.aaaa"}'
    )
    assert layout.storage == [{"label": "owner", "slot": "0"}]
    assert layout.compiler_version.startswith("0.8.20")
    huge = "x" * 70_000
    truncated = interpret_standard_json(
        '{"contracts":{"C.sol":{"C":{"ir":"' + huge + '"}}},"version":"0.8.20"}'
    )
    assert truncated.ir_truncated is True and len(truncated.ir) == 65_536


def test_compiler_invocation_failures_are_distinct(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.parsing.solidity_compiler.shutil.which", lambda _name: None)
    missing = compiler_semantics("contract C {}")
    assert missing.status == "UNAVAILABLE" and missing.storage == []
    monkeypatch.setattr("app.parsing.solidity_compiler.shutil.which", lambda _name: "/usr/bin/solc")

    def explode(_source: str) -> str:
        raise OSError("compiler executable unavailable")

    failed = compiler_semantics("contract C {}", runner=explode)
    assert failed.status == "FAILED" and failed.storage == []


def test_pragma_version_ignores_comments_and_ranges() -> None:
    commented = "// pragma solidity 0.4.0\ncontract C { uint256 a; }"
    assert pragma_solidity_version(commented) == ""
    assert language_semantics(pragma_solidity_version(commented), "")["arithmetic"] == "unknown"
    exact = pragma_solidity_version("pragma solidity 0.8.20;\ncontract C {}")
    assert language_semantics(exact, "")["arithmetic"] == "checked"
    assert language_semantics(exact, "")["source"] == "pragma"
    ranged = pragma_solidity_version("pragma solidity >=0.4.0 <0.9.0;\ncontract C {}")
    assert language_semantics(ranged, "")["arithmetic"] == "unknown"
    compiler = language_semantics("", "0.8.26+commit.abc")
    assert compiler["source"] == "compiler" and compiler["selfdestruct"] == "deprecated"


def test_reconciliation_keeps_both_values(tmp_path: Path) -> None:
    disagreements = reconcile_semantics(
        [{"contract": "Impl", "symbol": "value", "slot": "0", "offset": "0", "type": "uint256"}],
        [{"contract": "Impl", "label": "value", "slot": "4", "offset": "1", "type": "uint128"}],
    )
    categories = {item.category for item in disagreements}
    assert categories == {"storage_slot", "offset", "type"}
    assert all(item.confidence == "unresolved" for item in disagreements)
    selectors = reconcile_selectors(
        [{"contract": "Impl", "symbol": "value()", "selector": "3fa4f245"}],
        [{"contract": "Impl", "name": "value()", "selector": "0xdeadbeef"}],
    )
    assert selectors[0].category == "selector"
    assert selectors[0].parser_value != selectors[0].compiler_value
    source = "pragma solidity ^0.8.20; contract Impl { uint256 value; }"
    model = analyze_storage(_graph(tmp_path, source, "Impl.sol"))
    apply_compiler_layout(
        model,
        CompilerSemantics(
            "AVAILABLE",
            "solc",
            "test",
            layouts=[{"contract": "Impl", "label": "value", "slot": "4", "type": "uint128"}],
        ),
    )
    assert model.variables[0].slot == 0
    assert any(item.category == "storage_slot" for item in model.semantic_disagreements)


def test_compiler_cache_is_snapshot_scoped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.parsing.solidity_compiler.shutil.which", lambda _name: None)
    token = set_compiler_cache()
    try:
        first = compiler_semantics_for_scan("contract A {}", snapshot="snap-a")
        second = compiler_semantics_for_scan("contract A {}", snapshot="snap-a")
        other = compiler_semantics_for_scan("contract A {}", snapshot="snap-b")
        assert first is second
        assert other is not first
        assert compiler_cache_key(
            snapshot="snap-a", compiler="", version="", config="x", sources="s"
        ) != compiler_cache_key(snapshot="snap-b", compiler="", version="", config="x", sources="s")
    finally:
        reset_compiler_cache(token)
    fresh = set_compiler_cache()
    try:
        again = compiler_semantics_for_scan("contract A {}", snapshot="snap-a")
        assert again is not first
    finally:
        reset_compiler_cache(fresh)


def test_compiler_type_hint_tightens_address_without_being_required(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    contract Vault {
        address token;
        function pull(address to, uint256 amount) external {
            token.transfer(to, amount);
            token.balanceOf(address(this));
        }
    }
    """
    graph = _graph(tmp_path, source, "Vault.sol")
    plain = analyze_defi(graph)
    assert any(item.classification == "erc20" for item in plain.interactions)
    token = set_compiler_type_hints()
    try:
        record_compiler_type_hints(
            CompilerSemantics(
                "AVAILABLE",
                "solc",
                "test",
                layouts=[{"contract": "Vault", "label": "token", "type": "uint256", "slot": "0"}],
            )
        )
        tightened = analyze_defi(graph)
        assert all(item.classification != "erc20" for item in tightened.interactions)
    finally:
        reset_compiler_type_hints(token)
    again = set_compiler_type_hints()
    try:
        record_compiler_type_hints(
            CompilerSemantics(
                "AVAILABLE",
                "solc",
                "test",
                layouts=[{"contract": "Vault", "label": "token", "type": "IERC20", "slot": "0"}],
            )
        )
        typed = analyze_defi(graph)
        match = next(item for item in typed.interactions if item.method == "transfer")
        assert match.classification == "erc20"
        assert match.confidence == "strong"
    finally:
        reset_compiler_type_hints(again)


def test_compiler_agreement_strengthens_proxy_confidence_only(tmp_path: Path) -> None:
    source = f"""
    pragma solidity ^0.8.20;
    contract Proxy {{
        uint256 gap;
        bytes32 constant SLOT = {_IMPL};
        fallback() external payable {{
            assembly {{
                let impl := sload(SLOT)
                let ok := delegatecall(gas(), impl, 0, 0, 0, 0)
                if ok {{ stop() }}
            }}
        }}
    }}
    """
    graph = _graph(tmp_path, source, "Proxy.sol")
    token = set_storage_context({str(tmp_path / "Proxy.sol"): graph})
    try:
        model = analyze_proxy(graph)
        assert model.storage is not None
        apply_compiler_layout(
            model.storage,
            CompilerSemantics(
                "AVAILABLE",
                "solc",
                "test",
                layouts=[{"contract": "Proxy", "label": "gap", "slot": "0", "type": "uint256"}],
            ),
        )
        notes = " ".join(analyze_proxy(graph).notes)
    finally:
        reset_storage_context(token)
    assert "does not verify the proxy" in notes
    assert EvidenceProvenance.SANDBOX_EXECUTION not in VERIFICATION_PROVENANCE
