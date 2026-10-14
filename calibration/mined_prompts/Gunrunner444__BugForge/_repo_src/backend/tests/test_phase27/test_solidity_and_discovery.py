"""Phase 27: Solidity analysis and the multi-engine discovery loop."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.adapters.discovery.external import SlitherEngine, normalize_slither
from app.adapters.languages.capabilities import capability_matrix, promotion_stage
from app.analysis.quality import quality_rules_for
from app.discovery.capabilities import EngineAvailability, EngineCapability, ResultStatus
from app.discovery.corpus import DiscoveryCorpus, SeedSource
from app.discovery.correlation import correlate_results
from app.discovery.engine import AnalysisRequest, DiscoveryEngine
from app.discovery.graph import record_discovery_chain
from app.discovery.harness import candidate_foundry_test, validate_harness, write_candidate
from app.discovery.invariants import suggest_invariants
from app.discovery.oracle import OracleKind, evaluate_oracle
from app.discovery.results import DynamicFinding, DynamicResult
from app.discovery.scheduler import DiscoveryScheduler
from app.domain.language import ParserTier
from app.parsing.engine import parse_source, reset_syntax_registry
from app.parsing.keccak import function_selector, keccak256
from app.plugins import get_plugin_catalog, reset_plugin_catalog
from app.security.engine import SecurityAnalysisEngine
from app.security_agent.evidence_graph import EvidenceGraph


@pytest.fixture(autouse=True)
def _fresh() -> None:
    reset_syntax_registry()
    reset_plugin_catalog()


def _scan(tmp_path: Path, source: str) -> set[str]:
    path = tmp_path / "Contract.sol"
    path.write_text(source, encoding="utf-8")
    result = SecurityAnalysisEngine().analyze_repository(tmp_path, [path])
    return {item.rule_id for item in result.observations}


def test_keccak_and_selector() -> None:
    assert (
        keccak256(b"").hex() == "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"
    )
    assert function_selector("withdraw(uint256)") == "0x2e1a7d4d"


def test_parser_reports_full_ast(tmp_path: Path) -> None:
    source = """
pragma solidity ^0.8.20;
import {Ownable} from "./Ownable.sol";
interface IAuth { function owner() external view returns (address); }
library Math { function add(uint a, uint b) internal pure returns (uint) { return a + b; } }
struct Point { uint x; }
enum Kind { A, B }
error Boom(uint x);
contract Vault is IAuth {
    using Math for uint;
    uint public value;
    mapping(address => uint) balances;
    event Deposit(address indexed from, uint amount);
    modifier onlyOwner() { require(msg.sender == value); _; }
    constructor() { value = 1; }
    function owner() external view override returns (address) { return address(this); }
    fallback() external payable {}
    receive() external payable { value = 1; }
    function asm() external { assembly { let x := 1 } }
}
"""
    graph = parse_source("solidity", tmp_path / "Vault.sol", source)
    assert graph.parser_tier is ParserTier.FULL_AST
    assert graph.parser_backend == "tree_sitter"
    kinds = {entity.entity_type for entity in graph.entities}
    for expected in {"contract", "interface", "library", "struct", "enum", "function", "modifier"}:
        assert expected in kinds
    assert any(item.module.endswith("Ownable.sol") for item in graph.imports)
    assert any(
        "selector=0x" in event.extra for event in graph.events if event.kind == "sol_function"
    )
    assert "checked_arithmetic=true" in graph.semantic_context


def test_profile_fallback_is_labeled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import app.parsing.solidity_graph as solidity_graph

    monkeypatch.setattr(solidity_graph, "_from_treesitter", lambda *_args, **_kwargs: None)
    graph = solidity_graph.parse_solidity_source(
        tmp_path / "A.sol", "contract A { function f() public {} }"
    )
    assert graph.parser_tier is ParserTier.PROFILE_FALLBACK
    assert graph.parser_backend == "profile"


@pytest.mark.parametrize(
    ("source", "rule_id"),
    [
        (
            """
            pragma solidity ^0.8.20;
            contract Vault {
                mapping(address => uint) balances;
                function withdraw(uint amount) external {
                    (bool ok,) = msg.sender.call{value: amount}("");
                    require(ok);
                    balances[msg.sender] -= amount;
                }
            }
            """,
            "sol.reentrancy",
        ),
        (
            """
            pragma solidity ^0.8.20;
            contract Auth {
                address owner;
                function enter() external { require(tx.origin == owner); }
            }
            """,
            "sol.tx_origin",
        ),
        (
            """
            pragma solidity ^0.8.20;
            contract Pay {
                function pay() external { msg.sender.call{value: 1}(""); }
            }
            """,
            "sol.unchecked_call",
        ),
        (
            """
            pragma solidity ^0.8.20;
            contract Fwd {
                function exec(address target) external { target.delegatecall(""); }
            }
            """,
            "sol.arbitrary_delegatecall",
        ),
        (
            """
            pragma solidity ^0.8.20;
            contract Own {
                address owner;
                function setOwner(address next) external { owner = next; }
            }
            """,
            "sol.missing_authorization",
        ),
        (
            """
            pragma solidity ^0.8.20;
            contract Cast {
                function cut(uint value) external pure returns (uint8) { return uint8(value); }
            }
            """,
            "sol.downcast",
        ),
        (
            """
            pragma solidity ^0.8.20;
            contract Math {
                uint x;
                function bump() external { unchecked { x = x + 1; } }
            }
            """,
            "sol.unchecked_arithmetic",
        ),
        (
            """
            pragma solidity 0.7.6;
            contract Old {
                uint x;
                function bump() external { x = x + 1; }
            }
            """,
            "sol.pre_0_8_arithmetic",
        ),
        (
            """
            pragma solidity ^0.8.20;
            contract Round {
                function shares(uint a, uint b, uint c) external pure returns (uint) { return a / b * c; }
            }
            """,
            "sol.rounding",
        ),
        (
            """
            pragma solidity ^0.8.20;
            contract Sig {
                function recover(bytes32 h, uint8 v, bytes32 r, bytes32 s) external pure returns (address) {
                    return ecrecover(h, v, r, s);
                }
            }
            """,
            "sol.signature_replay",
        ),
        (
            """
            pragma solidity ^0.8.20;
            contract Boot {
                address owner;
                function initialize(address next) external { owner = next; }
            }
            """,
            "sol.initializer",
        ),
        (
            """
            pragma solidity ^0.8.20;
            contract Proxy {
                address public implementation;
                fallback() external payable { implementation.delegatecall(msg.data); }
            }
            """,
            "sol.storage_collision",
        ),
        (
            """
            pragma solidity ^0.8.20;
            contract Loop {
                function pay(address[] calldata users) external {
                    for (uint i; i < users.length; i++) { users[i].call(""); }
                }
            }
            """,
            "sol.unbounded_loop",
        ),
        (
            """
            pragma solidity ^0.8.20;
            contract Token {
                function send(address token, address to, uint amount) external {
                    token.transfer(to, amount);
                }
            }
            """,
            "sol.erc20_unchecked_return",
        ),
        (
            """
            pragma solidity ^0.8.20;
            contract Hook {
                mapping(address => uint) balances;
                function tokensReceived() external {
                    msg.sender.call("");
                    balances[msg.sender] += 1;
                }
            }
            """,
            "sol.callback_reentrancy",
        ),
        (
            """
            pragma solidity ^0.8.20;
            contract Proxy {
                address implementation;
                function upgradeTo(address next) external { implementation = next; }
            }
            """,
            "sol.upgrade_auth",
        ),
    ],
)
def test_solidity_detector_fires(tmp_path: Path, source: str, rule_id: str) -> None:
    assert rule_id in _scan(tmp_path, source)


def test_safe_solidity_contract_has_no_solidity_findings(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    contract Safe {
        address public owner;
        mapping(address => uint) balances;
        constructor() { owner = msg.sender; }
        modifier onlyOwner() { require(msg.sender == owner); _; }
        function withdraw(uint amount) external onlyOwner {
            uint balance = balances[msg.sender];
            require(balance >= amount);
            balances[msg.sender] = balance - amount;
            (bool ok,) = msg.sender.call{value: amount}("");
            require(ok);
        }
        function cut(uint value) external pure returns (uint8) {
            require(value <= type(uint8).max);
            return uint8(value);
        }
        function grow() external { unchecked { } }
    }
    """
    rules = {item for item in _scan(tmp_path, source) if item.startswith("sol.")}
    assert rules == set()


@pytest.mark.parametrize(
    ("source", "absent"),
    [
        (
            """
            pragma solidity ^0.8.20;
            contract Safe {
                mapping(address => uint) balances;
                function withdraw(uint amount) external {
                    balances[msg.sender] -= amount;
                    (bool ok,) = msg.sender.call{value: amount}("");
                    require(ok);
                }
            }
            """,
            "sol.reentrancy",
        ),
        (
            """
            pragma solidity ^0.8.20;
            contract Safe {
                address immutable implementation;
                function exec(bytes calldata data) external {
                    (bool ok,) = implementation.delegatecall(data);
                    require(ok);
                }
            }
            """,
            "sol.arbitrary_delegatecall",
        ),
        (
            """
            pragma solidity ^0.8.20;
            contract Safe {
                address owner;
                modifier onlyOwner() { require(msg.sender == owner); _; }
                function setOwner(address next) external onlyOwner { owner = next; }
            }
            """,
            "sol.missing_authorization",
        ),
        (
            """
            pragma solidity ^0.8.20;
            contract Safe {
                uint x;
                function bump() external { x = x + 1; }
            }
            """,
            "sol.unchecked_arithmetic",
        ),
        (
            """
            pragma solidity ^0.8.20;
            contract Safe {
                uint nonce;
                bytes32 DOMAIN_SEPARATOR;
                function recover(bytes32 h, uint8 v, bytes32 r, bytes32 s) external returns (address) {
                    nonce += 1;
                    h = keccak256(abi.encode(DOMAIN_SEPARATOR, address(this), nonce));
                    return ecrecover(h, v, r, s);
                }
            }
            """,
            "sol.signature_replay",
        ),
        (
            """
            pragma solidity ^0.8.20;
            contract Safe {
                address owner;
                bool initialized;
                modifier initializer() { require(!initialized); initialized = true; _; }
                function initialize(address next) external initializer { owner = next; }
            }
            """,
            "sol.initializer",
        ),
        (
            """
            pragma solidity ^0.8.20;
            contract Safe {
                function pay(address[] calldata users) external {
                    for (uint i; i < 10 && i < users.length; i++) { users[i].call(""); }
                }
            }
            """,
            "sol.unbounded_loop",
        ),
        (
            """
            pragma solidity ^0.8.20;
            contract Safe {
                function send(address token, address to, uint amount) external {
                    require(token.transfer(to, amount));
                }
            }
            """,
            "sol.erc20_unchecked_return",
        ),
    ],
)
def test_solidity_negative_cases(tmp_path: Path, source: str, absent: str) -> None:
    assert absent not in _scan(tmp_path, source)


def test_solidity_quality_is_separate(tmp_path: Path) -> None:
    source = """
    pragma solidity >=0.4.0;
    contract Q {
        uint x;
        fallback() external payable {}
        function set(uint x) external { throw; assembly { let y := 1 } }
    }
    """
    graph = parse_source("solidity", tmp_path / "Q.sol", source)
    findings = []
    for rule in quality_rules_for("solidity"):
        findings.extend(rule.check_graph(graph))
    categories = {item.category for item in findings}
    assert "sol_quality_empty_handler" in categories
    assert "sol_quality_shadowing" in categories
    assert "sol_quality_unsafe_pragma" in categories
    assert "sol_quality_assembly" in categories
    assert "sol_quality_deprecated" in categories
    assert all(item.catalog == "code_quality" for item in findings)


def test_capability_matrix_uses_explicit_levels() -> None:
    matrix = capability_matrix("solidity")
    assert matrix["security_rules"] == "YES"
    assert matrix["data_flow"] == "LIMITED"
    assert promotion_stage("solidity") == "FULL_ANALYSIS"
    assert promotion_stage("elixir") == "DETECTION_ONLY"
    for value in matrix.values():
        if value in {"tree_sitter", "full_ast", "profile", "profile_fallback", "none"}:
            continue
        assert value in {"YES", "LIMITED", "UNSUPPORTED", "UNAVAILABLE_AT_RUNTIME"}


class _Fake(DiscoveryEngine):
    def __init__(
        self, engine_id: str, caps: frozenset[EngineCapability], available: bool = True
    ) -> None:
        self._id = engine_id
        self._caps = caps
        self._available = available

    @property
    def engine_id(self) -> str:
        return self._id

    @property
    def display_name(self) -> str:
        return self._id

    @property
    def supported_languages(self) -> frozenset[str]:
        return frozenset({"solidity"})

    def capabilities(self) -> frozenset[EngineCapability]:
        return self._caps

    def availability(self) -> EngineAvailability:
        return EngineAvailability.AVAILABLE if self._available else EngineAvailability.UNAVAILABLE

    def _analyze_target(self, request: AnalysisRequest) -> DynamicResult:
        return DynamicResult(
            engine=self.engine_id,
            language=request.language,
            target=request.target,
            status=ResultStatus.INGESTED,
            executed=True,
            findings=(
                DynamicFinding(
                    "sol.reentrancy", "reentrancy", function=request.function, contract="Vault"
                ),
            ),
        )

    def _start_campaign(self, request: AnalysisRequest) -> DynamicResult:
        return DynamicResult(
            engine=self.engine_id,
            language=request.language,
            target=request.target,
            status=ResultStatus.EXECUTED,
            executed=True,
            coverage={"new": "false"},
        )


def test_scheduler_skips_unavailable_and_defers_halmos(tmp_path: Path) -> None:
    engines = (
        _Fake("bugforge-static", frozenset({EngineCapability.STATIC_ANALYSIS})),
        _Fake("slither", frozenset({EngineCapability.STATIC_ANALYSIS}), available=False),
        _Fake("foundry", frozenset({EngineCapability.TEST_EXECUTION, EngineCapability.FUZZING})),
        _Fake("halmos", frozenset({EngineCapability.SYMBOLIC_EXECUTION})),
    )
    scheduler = DiscoveryScheduler(engines, max_engines=2)
    request = AnalysisRequest(
        tmp_path, "solidity", target="withdraw", framework="foundry", has_harness=True
    )
    decisions = {item.engine_id: item for item in scheduler.select(request)}
    assert decisions["bugforge-static"].action == "run"
    assert decisions["slither"].action == "skip"
    assert decisions["foundry"].action == "run"
    assert decisions["halmos"].action == "skip"
    results = scheduler.run_selected(request)
    assert all(item.executed for item in results)
    assert all(item.metadata.get("verified", "false") != "true" for item in results)


def test_static_target_becomes_difficult_after_stagnation(tmp_path: Path) -> None:
    engine = _Fake("foundry", frozenset({EngineCapability.FUZZING}))
    scheduler = DiscoveryScheduler((engine,), max_rounds=1)
    request = AnalysisRequest(
        tmp_path, "solidity", target="withdraw", framework="foundry", has_harness=True
    )
    scheduler.note_result(
        DynamicResult(
            engine="foundry",
            language="solidity",
            target="withdraw",
            status=ResultStatus.EXECUTED,
            executed=True,
            coverage={"new": "false"},
        ),
        request,
    )
    nxt = scheduler.target_from_static(
        repo_root=request,
        file_path="Vault.sol",
        function="withdraw",
        contract="Vault",
        rule_id="sol.reentrancy",
    )
    assert nxt.difficult is True
    assert nxt.extra["source"] == "static_finding"


def test_oracle_does_not_treat_exit_code_as_a_bug() -> None:
    verdict = evaluate_oracle(OracleKind.EXIT_STATUS, exit_code=1, actual="error")
    assert verdict.meaningful is False
    crash = evaluate_oracle(OracleKind.SANITIZER, sanitizer="stack-buffer-overflow")
    assert crash.meaningful is True
    assert "Sanitizer" in crash.explanation


def test_corpus_tracks_provenance_and_redacts() -> None:
    corpus = DiscoveryCorpus()
    seed = corpus.add(
        "bearer abcdefghijklmnop",
        source=SeedSource.STATIC_FINDING,
        reason="reach_reentrancy_sink",
    )
    assert seed.source is SeedSource.STATIC_FINDING
    assert "abcdefghijklmnop" not in seed.preview
    assert seed.content_sha256


def test_slither_unavailable_does_not_invent_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("app.adapters.discovery.external.tool_path", lambda _name: None)
    engine = SlitherEngine()
    assert engine.availability() is EngineAvailability.UNAVAILABLE
    result = engine.analyze_target(AnalysisRequest(tmp_path, "solidity", target="Vault"))
    assert result.executed is False
    assert result.status is ResultStatus.UNAVAILABLE
    assert result.findings == ()


def test_slither_json_normalization() -> None:
    payload = """
    {"results": {"detectors": [{"check": "reentrancy-eth", "impact": "High",
      "confidence": "Medium", "description": "reentrant",
      "elements": [{"name": "withdraw", "source_mapping": {"filename": "A.sol", "lines": [12]}}]}]}}
    """
    findings = normalize_slither(payload)
    assert findings[0].detector_id == "reentrancy-eth"
    assert findings[0].function == "withdraw"
    assert findings[0].status == "potential"
    assert findings[0].line == 12


def test_correlation_does_not_count_duplicate_scanner_hits() -> None:
    finding = DynamicFinding("reentrancy-eth", "reentrancy", function="withdraw", contract="Vault")
    first = DynamicResult(
        engine="slither",
        language="solidity",
        target="withdraw",
        status=ResultStatus.INGESTED,
        executed=True,
        findings=(finding, finding),
    )
    second = DynamicResult(
        engine="bugforge-static",
        language="solidity",
        target="withdraw",
        status=ResultStatus.INGESTED,
        executed=True,
        findings=(
            DynamicFinding("sol.reentrancy", "reentrancy", function="withdraw", contract="Vault"),
        ),
    )
    groups = correlate_results([first, second])
    assert groups[0].independent_engines == 2
    assert groups[0].verified is False


def test_evidence_chain_and_invariants(tmp_path: Path) -> None:
    graph = EvidenceGraph()
    result = DynamicResult(
        engine="halmos",
        language="solidity",
        target="withdraw",
        status=ResultStatus.EXECUTED,
        executed=True,
        minimized_input="call withdraw",
        oracle_explanation="counterexample reaches the branch",
        provenance="halmos",
    )
    record_discovery_chain(graph, summary="possible reentrancy", result=result)
    kinds = {node.kind for node in graph.nodes.values()}
    assert {"hypothesis", "fuzz_target", "fuzz_campaign", "counterexample", "oracle"} <= kinds
    source = "pragma solidity ^0.8.20; contract T { uint totalSupply; address owner; function initialize() external {} }"
    syntax = parse_source("solidity", tmp_path / "T.sol", source)
    names = {item.name for item in suggest_invariants(syntax)}
    assert "supply_conservation" not in names
    assert "initialize_once" in names
    assert all(
        item.valid is False and item.status == "candidate" for item in suggest_invariants(syntax)
    )


def test_harness_is_not_written_into_the_repository(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "disposable" / "Candidate.t.sol"
    request = AnalysisRequest(repo, "solidity", function="withdraw", contract="Vault")
    source = candidate_foundry_test(request)
    assert validate_harness(source, outside)
    write_candidate(source, outside, repo_root=repo)
    assert outside.is_file()
    with pytest.raises(ValueError):
        write_candidate(source, repo / "src" / "Hack.t.sol", repo_root=repo)


def test_discovery_engines_are_registered() -> None:
    catalog = get_plugin_catalog()
    ids = set(catalog.discovery_engines.available_ids())
    assert {"bugforge-static", "slither", "foundry", "echidna", "medusa", "halmos", "wake"} <= ids
    slither = catalog.discovery_engines.create("slither")
    assert EngineCapability.STATIC_ANALYSIS in slither.capabilities()
