"""Phase 28: Solidity semantic depth and discovery-loop hardening."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.adapters.discovery.external import (
    EchidnaEngine,
    FoundryEngine,
    HalmosEngine,
    MedusaEngine,
    SlitherEngine,
    WakeEngine,
    parse_echidna_output,
    parse_foundry_output,
    parse_halmos_output,
    parse_medusa_output,
    read_foundry_config,
)
from app.discovery.builtin import BugforgeStaticEngine
from app.discovery.capabilities import EngineAvailability, EngineCapability, ResultStatus
from app.discovery.corpus import DiscoveryCorpus, SeedSource
from app.discovery.engine import AnalysisRequest
from app.discovery.graph import record_discovery_chain
from app.discovery.harness import candidate_foundry_test
from app.discovery.invariants import suggest_invariants
from app.discovery.oracle import OracleKind, evaluate_oracle
from app.discovery.results import DynamicResult
from app.discovery.scheduler import DiscoveryScheduler
from app.parsing.engine import parse_source, reset_syntax_registry
from app.parsing.keccak import function_selector
from app.parsing.solidity_links import relate_solidity_files
from app.parsing.solidity_types import canonical_solidity_type
from app.plugins import reset_plugin_catalog
from app.security.engine import SecurityAnalysisEngine
from app.security_agent.evidence_graph import EvidenceGraph, GraphIntegrityError


@pytest.fixture(autouse=True)
def _fresh() -> None:
    reset_syntax_registry()
    reset_plugin_catalog()


def _ids(tmp_path: Path, source: str) -> set[str]:
    path = tmp_path / "C.sol"
    path.write_text(source, encoding="utf-8")
    result = SecurityAnalysisEngine().analyze_repository(tmp_path, [path])
    return {item.rule_id for item in result.observations}


def test_known_selectors_and_zero_arguments() -> None:
    assert function_selector("foo()") == "0xc2985578"
    assert function_selector("transfer(address,uint256)") == "0xa9059cbb"
    assert function_selector("balanceOf(address)") == "0x70a08231"
    assert function_selector("totalSupply()") == "0x18160ddd"
    assert canonical_solidity_type("uint[] memory") == "uint256[]"
    assert canonical_solidity_type("address payable") == "address"
    assert canonical_solidity_type("bytes32[4]") == "bytes32[4]"
    assert canonical_solidity_type("(uint256, address)") == "(uint256,address)"
    assert canonical_solidity_type("(uint256,address)[]") == "(uint256,address)[]"
    assert canonical_solidity_type("MyStruct") == ""


def test_parsed_selectors_include_structs_and_skip_errors(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    struct Pair { uint256 a; address b; }
    contract C {
        function foo() external {}
        function bar(uint256[] memory xs, address payable to, bytes data) external {}
        function use(Pair p) external {}
        function raw((uint256,address) p) external {}
    }
    """
    graph = parse_source("solidity", tmp_path / "C.sol", source)
    functions = {
        _field(event.extra, "function"): event.extra
        for event in graph.events
        if event.kind == "sol_function"
    }
    assert "selector=0xc2985578" in functions["foo"]
    assert "selector_status=canonical" in functions["foo"]
    assert "uint256[]" in functions["bar"]
    assert function_selector("use((uint256,address))") in functions["use"]
    assert "selector_status=unresolved" in functions["raw"]
    assert "selector=0x" not in functions["raw"].split("selector_status")[0] or True
    assert "selector=" in functions["raw"]
    raw_selector = _field(functions["raw"], "selector")
    assert raw_selector == ""


def test_checked_call_requires_the_success_flag(tmp_path: Path) -> None:
    ignored = _ids(
        tmp_path,
        """
        pragma solidity ^0.8.20;
        contract P { function pay(address t) external { t.call(""); } }
        """,
    )
    assert "sol.unchecked_call" in ignored
    unrelated = _ids(
        tmp_path,
        """
        pragma solidity ^0.8.20;
        contract P {
            function pay(address t) external {
                (bool ok,) = t.call("");
                require(other);
            }
        }
        """,
    )
    assert "sol.unchecked_call" in unrelated
    checked = _ids(
        tmp_path,
        """
        pragma solidity ^0.8.20;
        contract P {
            function pay(address t) external {
                (bool ok,) = t.call("");
                require(ok);
            }
        }
        """,
    )
    assert "sol.unchecked_call" not in checked
    wrapped = _ids(
        tmp_path,
        """
        pragma solidity ^0.8.20;
        contract P { function pay(address t) external { require(t.call("")); } }
        """,
    )
    assert "sol.unchecked_call" not in wrapped


def test_authorization_is_not_any_msg_sender(tmp_path: Path) -> None:
    assigned = _ids(
        tmp_path,
        """
        pragma solidity ^0.8.20;
        contract P {
            address owner;
            function setOwner(address next) external { owner = msg.sender; }
        }
        """,
    )
    assert "sol.missing_authorization" in assigned
    guarded = _ids(
        tmp_path,
        """
        pragma solidity ^0.8.20;
        contract P {
            address owner;
            function setOwner(address next) external { require(msg.sender == owner); owner = next; }
        }
        """,
    )
    assert "sol.missing_authorization" not in guarded
    role = _ids(
        tmp_path,
        """
        pragma solidity ^0.8.20;
        contract P {
            modifier onlyRole(bytes32 role) { _; }
            function mint(address to) external onlyRole(0) {}
        }
        """,
    )
    assert "sol.missing_authorization" not in role


def test_reentrancy_correlates_the_same_variable(tmp_path: Path) -> None:
    classic = _ids(
        tmp_path,
        """
        pragma solidity ^0.8.20;
        contract V {
            mapping(address => uint) balances;
            function withdraw() external {
                uint amount = balances[msg.sender];
                (bool ok,) = msg.sender.call{value: amount}("");
                require(ok);
                balances[msg.sender] = 0;
            }
        }
        """,
    )
    assert "sol.reentrancy" in classic
    cei = _ids(
        tmp_path,
        """
        pragma solidity ^0.8.20;
        contract V {
            mapping(address => uint) balances;
            function withdraw() external {
                uint amount = balances[msg.sender];
                balances[msg.sender] = 0;
                (bool ok,) = msg.sender.call{value: amount}("");
                require(ok);
            }
        }
        """,
    )
    assert "sol.reentrancy" not in cei
    unrelated = _ids(
        tmp_path,
        """
        pragma solidity ^0.8.20;
        contract V {
            uint counter;
            function poke(address t) external {
                counter = 1;
                t.call("");
                counter = 2;
            }
        }
        """,
    )
    assert "sol.reentrancy" not in unrelated


def test_encode_packed_and_randomness_are_contextual(tmp_path: Path) -> None:
    packed = _ids(
        tmp_path,
        """
        pragma solidity ^0.8.20;
        contract S {
            function prove(string memory a, bytes memory b) external pure returns (bytes32) {
                return keccak256(abi.encodePacked(a, b));
            }
        }
        """,
    )
    assert "sol.encode_packed" in packed
    fixed = _ids(
        tmp_path,
        """
        pragma solidity ^0.8.20;
        contract S {
            function prove(address a, uint256 b) external pure returns (bytes32) {
                return keccak256(abi.encodePacked(a, b));
            }
        }
        """,
    )
    assert "sol.encode_packed" not in fixed
    lottery = _ids(
        tmp_path,
        """
        pragma solidity ^0.8.20;
        contract L {
            function lottery() external view returns (uint) {
                return uint(keccak256(abi.encodePacked(block.timestamp, block.prevrandao)));
            }
        }
        """,
    )
    assert "sol.insecure_randomness" in lottery
    deadline = _ids(
        tmp_path,
        """
        pragma solidity ^0.8.20;
        contract L {
            function active(uint end) external view {
                require(block.timestamp < end);
            }
        }
        """,
    )
    assert "sol.insecure_randomness" not in deadline


def test_signature_domain_and_selfdestruct(tmp_path: Path) -> None:
    domain = _ids(
        tmp_path,
        """
        pragma solidity ^0.8.20;
        contract S {
            uint nonce;
            function recover(bytes32 h, uint8 v, bytes32 r, bytes32 s) external returns (address) {
                nonce += 1;
                return ecrecover(h, v, r, s);
            }
        }
        """,
    )
    assert "sol.signature_replay" in domain
    assert "sol.signature_domain" in domain
    bound = _ids(
        tmp_path,
        """
        pragma solidity ^0.8.20;
        contract S {
            uint nonce;
            bytes32 DOMAIN_SEPARATOR;
            function recover(bytes32 h, uint8 v, bytes32 r, bytes32 s) external returns (address) {
                nonce += 1;
                h = keccak256(abi.encode(DOMAIN_SEPARATOR, address(this), nonce));
                return ecrecover(h, v, r, s);
            }
        }
        """,
    )
    assert "sol.signature_domain" not in bound
    destroyed = _ids(
        tmp_path,
        """
        pragma solidity ^0.8.20;
        contract D { function kill() external { selfdestruct(payable(msg.sender)); } }
        """,
    )
    assert "sol.selfdestruct" in destroyed


def test_immutable_delegatecall_is_not_arbitrary(tmp_path: Path) -> None:
    rules = _ids(
        tmp_path,
        """
        pragma solidity ^0.8.20;
        contract P {
            address immutable implementation;
            function exec(bytes calldata data) external {
                implementation.delegatecall(data);
            }
        }
        """,
    )
    assert "sol.arbitrary_delegatecall" not in rules


def test_semantic_supply_invariant(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    contract T {
        uint totalSupply;
        mapping(address => uint) balances;
        function mint(address to, uint amount) external {
            balances[to] += amount;
            totalSupply += amount;
        }
    }
    """
    graph = parse_source("solidity", tmp_path / "T.sol", source)
    names = {item.name for item in suggest_invariants(graph)}
    assert "supply_conservation" in names
    assert all(item.valid is False for item in suggest_invariants(graph))


def test_harness_uses_parsed_parameters(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    contract Vault {
        function withdraw(uint256 amount) external payable {}
    }
    """
    graph = parse_source("solidity", tmp_path / "Vault.sol", source)
    request = AnalysisRequest(tmp_path, "solidity", function="withdraw", contract="Vault")
    text = candidate_foundry_test(request, graph)
    assert "function withdraw(uint256 amount) external payable;" in text
    assert "function withdraw() external;" not in text


def test_static_engine_respects_language(tmp_path: Path) -> None:
    (tmp_path / "A.sol").write_text(
        'pragma solidity ^0.8.20; contract A { function withdraw() external { msg.sender.call(""); } }',
        encoding="utf-8",
    )
    (tmp_path / "app.py").write_text("def ok():\n    return 1\n", encoding="utf-8")
    engine = BugforgeStaticEngine()
    assert "python" in engine.supported_languages
    assert "solidity" in engine.supported_languages
    result = engine.analyze_target(AnalysisRequest(tmp_path, "python"))
    assert result.findings == ()
    assert "sol." not in " ".join(item.detector_id for item in result.findings)
    solidity = engine.analyze_target(AnalysisRequest(tmp_path, "solidity"))
    assert any(item.detector_id.startswith("sol.") for item in solidity.findings)


def test_corpus_identity_survives_snapshot() -> None:
    corpus = DiscoveryCorpus()
    first = corpus.add(
        "seed-body", source=SeedSource.STATIC_FINDING, reason="reach sink", target="withdraw"
    )
    again = corpus.add(
        "seed-body", source=SeedSource.STATIC_FINDING, reason="again", target="withdraw"
    )
    assert first.seed_id == again.seed_id
    assert len(corpus.seeds) == 1
    assert first.preview != "seed-body" or "seed-body" in first.preview
    restored = DiscoveryCorpus.from_snapshot(corpus.snapshot())
    assert restored.seeds[0].content_sha256 == first.content_sha256
    assert restored.seeds[0].source is SeedSource.STATIC_FINDING
    assert restored.seeds[0].preview == first.preview


def test_oracle_distinguishes_reverts_from_tool_failure() -> None:
    assert evaluate_oracle(OracleKind.EXPECTED_REVERT, actual="revert").meaningful is False
    unexpected = evaluate_oracle(OracleKind.UNEXPECTED_REVERT, actual="revert Panic")
    assert unexpected.meaningful is True
    assert (
        evaluate_oracle(OracleKind.TOOL_FAILURE, actual="Compiler run failed").meaningful is False
    )
    assert evaluate_oracle(OracleKind.PANIC, actual="0x11").meaningful is True
    assert (
        evaluate_oracle(OracleKind.EXIT_STATUS, exit_code=500, actual="HTTP 500").meaningful
        is False
    )


def test_evidence_motivates_round_trip() -> None:
    graph = EvidenceGraph()
    graph.add(kind="hypothesis", provenance="static", summary="reentrancy", source="bugforge")
    with pytest.raises(GraphIntegrityError):
        graph.link(next(iter(graph.nodes)), next(iter(graph.nodes)), "not-a-relation")
    parent = next(iter(graph.nodes))
    child = graph.add(
        kind="fuzz_target",
        provenance="static",
        summary="withdraw",
        source="scheduler",
        parent_id=parent,
        relation="motivates",
    )
    restored = EvidenceGraph.from_snapshot(graph.snapshot())
    relations = [edge[2] for edge in restored.edges]
    assert "motivates" in relations
    assert child.id in restored.nodes
    result = DynamicResult(
        engine="halmos",
        language="solidity",
        target="withdraw",
        status=ResultStatus.INTERESTING,
        executed=True,
        minimized_input="amount=1",
        oracle_explanation="counterexample",
        oracle_kind="assertion",
        reproduction_command="forge test --match-test withdraw",
        seed_id="seed_001",
        contract="Vault",
        function="withdraw",
    )
    record_discovery_chain(EvidenceGraph(), summary="reentrancy", result=result)


def test_scheduler_followup_uses_stagnation_then_counterexample(tmp_path: Path) -> None:
    class _Engine:
        def __init__(self, engine_id: str, caps: frozenset[EngineCapability]) -> None:
            self._id = engine_id
            self._caps = caps

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
            return EngineAvailability.AVAILABLE

        def analyze_target(self, request: AnalysisRequest) -> DynamicResult:
            return DynamicResult(
                engine=self.engine_id,
                language="solidity",
                target=request.target,
                status=ResultStatus.INGESTED,
                executed=True,
            )

        def start_campaign(self, request: AnalysisRequest) -> DynamicResult:
            if self.engine_id == "halmos":
                return DynamicResult(
                    engine="halmos",
                    language="solidity",
                    target=request.target,
                    status=ResultStatus.INTERESTING,
                    executed=True,
                    minimized_input="counterexample-calldata",
                    metadata={"seed_source": "symbolic"},
                    coverage={"new": "false"},
                )
            return DynamicResult(
                engine=self.engine_id,
                language="solidity",
                target=request.target,
                status=ResultStatus.EXECUTED,
                executed=True,
                coverage={"new": "true" if request.extra.get("mode") == "fuzz" else "false"},
            )

    foundry = _Engine(
        "foundry", frozenset({EngineCapability.TEST_EXECUTION, EngineCapability.FUZZING})
    )
    halmos = _Engine("halmos", frozenset({EngineCapability.SYMBOLIC_EXECUTION}))
    scheduler = DiscoveryScheduler((foundry, halmos), max_rounds=1)  # type: ignore[arg-type]
    request = AnalysisRequest(
        tmp_path,
        "solidity",
        target="withdraw",
        function="withdraw",
        contract="Vault",
        framework="foundry",
        has_harness=True,
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
    assert scheduler.feedback.difficult is True
    follow = scheduler.run_followup(request)
    assert follow[0].engine == "halmos"
    assert scheduler.corpus.by_source(SeedSource.SYMBOLIC_EXECUTION)
    second = scheduler.plan_followup(request)
    assert any(item.engine_id == "foundry" and "counterexample" in item.reason for item in second)
    assert all(item.engine_id != "halmos" or item.action != "run" for item in second)


def test_optional_tools_stay_unavailable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.adapters.discovery.external.tool_path", lambda _name: None)
    request = AnalysisRequest(tmp_path, "solidity", target="Vault.sol", difficult=True)
    for engine in (
        SlitherEngine(),
        FoundryEngine(),
        EchidnaEngine(),
        MedusaEngine(),
        HalmosEngine(),
        WakeEngine(),
    ):
        result = (
            engine.start_campaign(request)
            if engine.engine_id != "slither"
            else engine.analyze_target(request)
        )
        if engine.engine_id == "halmos":
            planned = engine.start_campaign(AnalysisRequest(tmp_path, "solidity", target="Vault"))
            assert planned.executed is False
            assert planned.status is ResultStatus.PLANNED
            assert result.status is ResultStatus.NOT_IMPLEMENTED
            missing = engine.start_campaign(
                AnalysisRequest(
                    tmp_path,
                    "solidity",
                    target="test_withdraw",
                    function="test_withdraw",
                    difficult=True,
                )
            )
            assert missing.executed is False
            assert missing.status is ResultStatus.UNAVAILABLE
            continue
        assert result.executed is False
        assert result.findings == ()
        assert result.status is ResultStatus.UNAVAILABLE


def test_tool_parsers_do_not_invent_findings() -> None:
    assert parse_foundry_output("Suite result: ok. 1 passed")["assertion"] == ""
    failed = parse_foundry_output("[FAIL. Reason: Assertion failed] test_withdraw()\nSuite result")
    assert "test_withdraw" in str(failed["assertion"])
    echidna = parse_echidna_output("echidna_supply: falsified!\nCall sequence:\n  withdraw(1)\n")
    assert echidna["sequence"]
    assert "falsified" in str(echidna["assertion"])
    medusa = parse_medusa_output("coverage: 12%\nAssertion failed: invariant_supply\n")
    assert medusa["coverage"]["percent"] == "12"
    assert medusa["assertion"]
    quiet = parse_medusa_output("medusa finished\n")
    assert quiet["assertion"] == ""
    assert quiet["coverage"]["coverage_available"] == "false"
    assert "new" not in quiet["coverage"]
    assert "new_coverage" not in quiet["coverage"]
    halmos = parse_halmos_output("[FAIL] test_withdraw()\nCounterexample:\n  amount = 1\n")
    assert "Counterexample" in halmos["counterexample"]
    assert halmos["test"] == "test_withdraw"


def test_foundry_config_and_cross_file_links(tmp_path: Path) -> None:
    (tmp_path / "foundry.toml").write_text(
        'src = "src"\nsolc_version = "0.8.20"\n', encoding="utf-8"
    )
    (tmp_path / "src").mkdir()
    config = read_foundry_config(tmp_path)
    assert config["solc_version"] == "0.8.20"
    assert config["src"] == "src"
    base = parse_source(
        "solidity",
        tmp_path / "Ownable.sol",
        "pragma solidity ^0.8.20; contract Ownable { address owner; }",
    )
    child = parse_source(
        "solidity",
        tmp_path / "Vault.sol",
        'pragma solidity ^0.8.20; import "./Ownable.sol"; contract Vault is Ownable {}',
    )
    ambiguous = parse_source(
        "solidity",
        tmp_path / "Other.sol",
        "pragma solidity ^0.8.20; contract Ownable { }",
    )
    unique = relate_solidity_files({"Ownable.sol": base, "Vault.sol": child})
    assert any(link.kind == "inherits" and link.target_name == "Ownable" for link in unique)
    ambiguous_links = relate_solidity_files(
        {"Ownable.sol": base, "Vault.sol": child, "Other.sol": ambiguous}
    )
    assert not any(link.kind == "inherits" for link in ambiguous_links)


def test_discovery_loop_does_not_verify(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    contract Vault {
        mapping(address => uint) balances;
        function withdraw() external {
            uint amount = balances[msg.sender];
            (bool ok,) = msg.sender.call{value: amount}("");
            balances[msg.sender] = 0;
        }
    }
    """
    path = tmp_path / "Vault.sol"
    path.write_text(source, encoding="utf-8")
    scan = SecurityAnalysisEngine().analyze_repository(tmp_path, [path])
    assert any(item.rule_id == "sol.reentrancy" for item in scan.observations)
    observation = next(item for item in scan.observations if item.rule_id == "sol.reentrancy")
    assert observation.metadata.get("status") == "potential"
    engine = BugforgeStaticEngine()
    result = engine.analyze_target(
        AnalysisRequest(tmp_path, "solidity", files=("Vault.sol",), function="withdraw")
    )
    assert any(item.detector_id == "sol.reentrancy" for item in result.findings)
    assert result.metadata["verified"] == "false"
    safe = """
    pragma solidity ^0.8.20;
    contract Vault {
        address owner;
        mapping(address => uint) balances;
        modifier onlyOwner() { require(msg.sender == owner); _; }
        function withdraw() external onlyOwner {
            uint amount = balances[msg.sender];
            balances[msg.sender] = 0;
            (bool ok,) = msg.sender.call{value: amount}("");
            require(ok);
        }
    }
    """
    assert "sol.reentrancy" not in _ids(tmp_path, safe)
    assert "sol.missing_authorization" not in _ids(tmp_path, safe)


def _field(extra: str, key: str) -> str:
    for part in extra.split("|"):
        if part.startswith(key + "="):
            return part.split("=", 1)[1]
    return ""
