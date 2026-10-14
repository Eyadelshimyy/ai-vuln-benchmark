"""Phase 48 runtime validation and the Phase 47 protocol-engine hardening."""

from __future__ import annotations

import json
from pathlib import Path

from app.adapters.discovery.protocol import ProtocolEngine
from app.adapters.discovery.runtime import RuntimeEngine
from app.discovery.capabilities import EngineAvailability, EngineCapability, ResultStatus
from app.discovery.engine import AnalysisRequest, DiscoveryEngine
from app.discovery.loop import ResearchObservation, choose_next
from app.discovery.results import DynamicResult
from app.discovery.scheduler import DiscoveryScheduler, missing_capability
from app.discovery.sequences import MAX_EXECUTIONS
from app.domain.evidence import EvidenceKind
from app.execution.base import ExecutionResult
from app.parsing.solidity_economics import AssetDelta, OracleStatus
from app.parsing.solidity_ir import ContractFact, SemanticCall, SemanticFunction, SemanticProgram
from app.parsing.solidity_protocol import (
    MAX_PROTOCOL_EXPANSION,
    ContractNode,
    FunctionNode,
    InteractionEdge,
    PriceLink,
    ProtocolBounds,
    ProtocolGraph,
    TrustBoundary,
    build_protocol_graph,
    clamp_protocol_bounds,
    correlate_modalities,
    expand_paths,
    plan_protocol_calls,
    protocol_domain_evidence,
    protocol_evidence,
    separate_asset_deltas,
)
from app.parsing.solidity_runtime import (
    NormalizedEvent,
    RuntimeObservation,
    StorageChange,
    bind_protocol_edge,
    bind_state_transition,
    clamp_runtime_executions,
    classify_replay,
    compare_executions,
    event_proves_asset,
    failure_is_vulnerability,
    fork_identity,
    metamorphic_expectation,
    network_allowed,
    parse_runtime_document,
    runtime_balance,
    runtime_command,
    same_state,
    slots_equivalent,
    state_of,
)

_RUNTIME = Path(__file__).resolve().parents[2] / "app" / "adapters" / "discovery" / "runtime.py"


def _span(line: int = 1):
    from app.parsing.solidity_ir import SourceSpanRef

    return SourceSpanRef("A.sol", line, line + 1, line, line)


def _call(callee: str, target: str, call_type: str, call_id: str) -> SemanticCall:
    return SemanticCall(
        "caller",
        callee,
        "resolved",
        target,
        "",
        "",
        call_type,
        True,
        False,
        False,
        False,
        _span(),
        call_id,
    )


def _function(
    contract: str,
    name: str,
    line: int,
    calls: tuple[SemanticCall, ...] = (),
    *,
    source: str = "",
    authorization: str = "unknown",
    span: tuple[int, int, int, int] | None = None,
) -> SemanticFunction:
    return SemanticFunction(
        contract,
        name,
        line,
        "external",
        "",
        (),
        (),
        (),
        tuple(item.callee for item in calls),
        authorization,
        span=span or (line, 0, line, 20),
        call_sites=calls,
        source=source,
    )


def _program(
    file: str, functions: tuple[SemanticFunction, ...], contracts: tuple[ContractFact, ...]
) -> SemanticProgram:
    return SemanticProgram(
        "phase39.1", file, "partial", functions, (), "", "unavailable", contracts=contracts
    )


def _node(node_id: str, contract: str, source_file: str) -> ContractNode:
    return ContractNode(
        node_id, "p", "snap", "cfg", source_file, contract, "contract", (1, 2, 3, 4), "", "parser"
    )


def _document(transaction: dict[str, object], **shared: str) -> str:
    payload: dict[str, object] = {
        "schema": "bugforge-runtime-v1",
        "project": "p",
        "source_snapshot": "snap",
        "compiler_configuration": "cfg",
        "runtime_environment": "local",
        "runtime_configuration": "pinned-local",
        "chain_id": "31337",
        "block_number": "1",
        "block_timestamp": "0",
        "state_snapshot": "genesis",
        "tool": "bugforge-runtime",
        "tool_version": "bugforge-runtime-v1",
        "duration": "0.01",
        "sequence_id": "seq",
        "deployments": [
            {"address": "0x1", "contract": "Vault", "identity": "Vault", "source": "Vault.sol"}
        ],
        "transactions": [transaction],
    }
    payload.update(shared)
    return json.dumps(payload)


def _transaction(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "index": "0",
        "actor": "user",
        "contract": "Vault",
        "address": "0x1",
        "function_identity": "Vault.sol::Vault.deposit(uint256)",
        "identity_established": True,
        "selector": "0xb6b55f25",
        "success": "true",
        "return_data": "0x",
        "gas_used": "21000",
        "call_type": "CALL",
        "execution_id": "run-1",
        "trace": [
            {
                "call_type": "CALL",
                "caller": "0xuser",
                "callee": "0x1",
                "selector": "0xb6b55f25",
                "children": [
                    {
                        "call_type": "STATICCALL",
                        "caller": "0x1",
                        "callee": "0x2",
                        "selector": "0x11111111",
                        "children": [
                            {
                                "call_type": "DELEGATECALL",
                                "caller": "0x2",
                                "callee": "0x3",
                                "selector": "0x22222222",
                            }
                        ],
                    }
                ],
            }
        ],
        "events": [
            {
                "address": "0x1",
                "name": "Transfer",
                "topic": "0xddf252ad",
                "transaction_index": "0",
                "log_index": "0",
                "raw": "0x",
                "contract": "Vault",
                "identity_established": True,
                "source_identity": "Vault.sol",
            }
        ],
        "storage": [
            {
                "address": "0x1",
                "slot": "0x3",
                "before": "0x1",
                "after": "0x2",
                "transaction_index": "0",
                "execution_context": "seq",
                "layout_established": True,
                "source_identity": "Vault.sol::balance",
            }
        ],
    }
    row.update(overrides)
    return row


def _state(address: str = "0x1", block: str = "1", actor: str = "user") -> RuntimeObservation:
    parsed = parse_runtime_document(
        _document(_transaction(address=address, actor=actor), block_number=block)
    )
    assert parsed is not None
    observed = parsed[0]
    if address != "0x1":
        return RuntimeObservation(
            project=observed.project,
            source_snapshot=observed.source_snapshot,
            compiler_configuration=observed.compiler_configuration,
            runtime_environment=observed.runtime_environment,
            sequence_id=observed.sequence_id,
            transaction_index=observed.transaction_index,
            actor=actor,
            contract="",
            function_identity=observed.function_identity,
            success=observed.success,
            return_data=observed.return_data,
            gas_used=observed.gas_used,
            block_number=block,
            chain_id=observed.chain_id,
            state_snapshot=observed.state_snapshot,
            runtime_configuration=observed.runtime_configuration,
            deployment_address=address,
            execution_id=f"{block}:{address}",
            status="observed",
        )
    return observed


class _Fake(DiscoveryEngine):
    def __init__(self, engine_id: str, capabilities: frozenset[EngineCapability]) -> None:
        self._id = engine_id
        self._capabilities = capabilities

    @property
    def engine_id(self) -> str:
        return self._id

    @property
    def display_name(self) -> str:
        return self._id

    def capabilities(self) -> frozenset[EngineCapability]:
        return self._capabilities

    def availability(self) -> EngineAvailability:
        return EngineAvailability.AVAILABLE

    def _start_campaign(self, request: AnalysisRequest) -> DynamicResult:
        return DynamicResult(
            engine=self.engine_id,
            language=request.language,
            target=request.target,
            status=ResultStatus.EXECUTED,
            executed=True,
        )


def test_local_runtime_is_sandboxed_and_has_no_host_fallback(tmp_path: Path, monkeypatch) -> None:
    text = _RUNTIME.read_text(encoding="utf-8")
    assert "LocalTestExecutor" not in text
    assert "shell=True" not in text
    assert "subprocess" not in text
    engine = RuntimeEngine()
    assert engine.availability() is EngineAvailability.UNAVAILABLE
    missing = engine.start_campaign(AnalysisRequest(tmp_path, "solidity", target="Vault"))
    assert missing.status is ResultStatus.UNAVAILABLE
    assert missing.executed is False
    assert missing.metadata["verified"] == "false"
    assert "host execution is not started" in missing.oracle_explanation
    captured: dict[str, object] = {}

    def fake_execute(config):
        captured["config"] = config
        manifest = json.loads(Path(config.command[2]).read_text(encoding="utf-8"))
        document = json.loads(
            _document(
                _transaction(),
                request_identity=manifest["request_identity"],
                execution_id=manifest["execution_id"],
                mode=manifest["mode"],
                target=manifest["target"],
            )
        )
        document["transactions"][0].pop("execution_id")
        return ExecutionResult(
            exit_code=0,
            stdout="",
            stderr="",
            duration_seconds=0.1,
            artifact_contents={"runtime.json": json.dumps(document)},
        )

    monkeypatch.setattr(
        "app.adapters.discovery.runtime._runtime_image", lambda: "local/runtime:pinned"
    )
    monkeypatch.setattr("app.adapters.discovery.runtime._docker_present", lambda: True)
    monkeypatch.setattr("app.adapters.discovery.runtime._local_image", lambda: True)
    monkeypatch.setattr("app.adapters.discovery.runtime._execute", fake_execute)
    result = engine.start_campaign(
        AnalysisRequest(
            tmp_path,
            "solidity",
            target="Vault",
            extra={"network": "true", "https://rpc.example": "1"},
        )
    )
    assert result.executed is False
    assert result.status is ResultStatus.FAILED
    assert "config" not in captured
    clean = engine.start_campaign(
        AnalysisRequest(
            tmp_path,
            "solidity",
            target="Vault",
            extra={"sequence_id": "seq", "transaction_index": "0"},
        )
    )
    config = captured["config"]
    assert config.allow_network is False
    assert config.memory_limit_mb == 512
    assert config.cpu_limit == 1.0
    assert config.command[:2] == runtime_command(config.command[2])[:2]
    assert config.command == runtime_command(config.command[2])
    assert "docker.sock" not in str(config.read_only_volumes)
    assert "http" not in " ".join(config.command)
    assert clean.executed is True
    assert clean.metadata["verified"] == "false"
    assert clean.metadata["network"] == "none"
    assert clean.to_evidence().kind is EvidenceKind.RUNTIME_OBSERVATION
    assert clean.to_evidence().contributes_to_verification is False
    assert (
        network_allowed(
            mode="local", fork_enabled=True, deterministic=True, requested_network="true"
        )
        is False
    )


def test_fork_requires_operator_configuration_and_a_pinned_block(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        "app.adapters.discovery.runtime._execute",
        lambda config: calls.append("ran") or ExecutionResult(0, "", "", 0.0),
    )
    engine = RuntimeEngine()
    request = AnalysisRequest(
        tmp_path,
        "solidity",
        target="Vault",
        extra={"mode": "fork", "fork_url": "https://public.example", "fork_block": "latest"},
    )
    blocked = engine.start_campaign(request)
    assert blocked.status is ResultStatus.UNAVAILABLE
    assert blocked.executed is False
    assert calls == []
    monkeypatch.setattr("app.adapters.discovery.runtime._fork_enabled", lambda: True)
    monkeypatch.setattr("app.adapters.discovery.runtime._fork_source", lambda: "operator-fork")
    unpinned = engine.start_campaign(request)
    assert unpinned.executed is False
    assert unpinned.metadata["deterministic"] == "false"
    assert unpinned.metadata["observation_status"] == "unknown"
    assert calls == []
    identity = fork_identity(
        configured=True,
        fork_source="operator-fork",
        chain_id="1",
        block_number="latest",
        state_snapshot="root",
    )
    assert identity.deterministic is False
    pinned = fork_identity(
        configured=True,
        fork_source="operator-fork",
        chain_id="1",
        block_number="100",
        state_snapshot="root",
    )
    assert pinned.deterministic is True
    assert (
        network_allowed(
            mode="fork", fork_enabled=False, deterministic=True, requested_network="true"
        )
        is False
    )


def test_runtime_identity_trace_events_and_storage_stay_exact() -> None:
    parsed = parse_runtime_document(_document(_transaction()))
    assert parsed is not None
    observed = parsed[0]
    assert observed.contract == "Vault"
    assert observed.function_identity == "Vault.sol::Vault.deposit(uint256)"
    assert observed.trace[0].call_type == "CALL"
    assert observed.trace[0].children[0].call_type == "STATICCALL"
    assert observed.trace[0].children[0].children[0].call_type == "DELEGATECALL"
    assert observed.trace[0].function_identity == ""
    unresolved = parse_runtime_document(
        _document(_transaction(identity_established=False, function_identity="Vault.deposit"))
    )
    assert unresolved is not None
    assert unresolved[0].function_identity == ""
    assert unresolved[0].selector == "0xb6b55f25"
    event = observed.events[0]
    assert event.contract_identity == "Vault"
    assert event.topic == "0xddf252ad"
    assert event.transaction_index == "0"
    assert event_proves_asset(NormalizedEvent(name="Transfer")) is False
    assert event_proves_asset(NormalizedEvent(name="Swap")) is False
    stored = observed.storage_changes[0]
    assert stored.contract_identity == "Vault"
    assert stored.slot == "0x3"
    other = StorageChange(
        address="0x9",
        contract_identity="Proxy",
        slot="0x3",
        layout_established=True,
    )
    assert slots_equivalent(stored, other) is False
    assert slots_equivalent(stored, stored) is True
    assert parse_runtime_document("not json") is None
    assert parse_runtime_document("{}") is None


def test_runtime_binds_only_matching_phase41_and_phase46_identity() -> None:
    assert (
        bind_state_transition(
            runtime_contract="Vault",
            runtime_function="Vault.sol::Vault.deposit(uint256)",
            runtime_storage="Vault.sol::balance",
            transition_contract="Vault",
            transition_function="Vault.sol::Vault.deposit(uint256)",
            transition_storage="Vault.sol::balance",
            transition_id="transition-1",
        )
        == "transition-1"
    )
    assert (
        bind_state_transition(
            runtime_contract="Vault",
            runtime_function="deposit",
            runtime_storage="balance",
            transition_contract="Vault",
            transition_function="Vault.sol::Vault.deposit(uint256)",
            transition_storage="Vault.sol::balance",
            transition_id="transition-1",
        )
        == ""
    )
    attacker = runtime_balance(
        actor="attacker",
        token="USDC",
        kind="erc20",
        before=1,
        after=2,
        transaction_index=0,
        snapshot="genesis",
        contract="Vault",
        source="Vault.sol",
        provenance="runtime-execution",
    )
    victim = runtime_balance(
        actor="victim",
        token="USDC",
        kind="erc20",
        before=2,
        after=1,
        transaction_index=0,
        snapshot="genesis",
        contract="Vault",
        source="Vault.sol",
        provenance="runtime-execution",
    )
    other_token = runtime_balance(
        actor="attacker",
        token="WETH",
        kind="erc20",
        before=1,
        after=2,
        transaction_index=0,
        snapshot="genesis",
        contract="Vault",
        source="Vault.sol",
        provenance="runtime-execution",
    )
    assert attacker is not None and victim is not None and other_token is not None
    assert separate_asset_deltas((attacker, victim)).status == OracleStatus.INCOMPLETE.value
    assert separate_asset_deltas((attacker, other_token)).status == OracleStatus.UNKNOWN.value
    assert (
        runtime_balance(
            actor="attacker",
            token="USDC",
            kind="erc20",
            before=1,
            after=5,
            transaction_index=1,
            snapshot="later",
            contract="Vault",
            source="Vault.sol",
            provenance="externally-supplied",
        )
        is None
    )
    later = AssetDelta(
        "erc20", "USDC", "token", "attacker", 1, 2, 1, "runtime", 1, "later", "Vault", "Vault.sol"
    )
    assert separate_asset_deltas((attacker, later)).status == OracleStatus.INCOMPLETE.value


def test_differential_comparison_does_not_merge_states_or_verify(tmp_path: Path) -> None:
    left = _state()
    right = _state()
    same = compare_executions(left, right, reset=True)
    assert same.status == "candidate"
    assert same.classification == "deterministic same result"
    assert same.vulnerability == "unknown"
    assert "safety" in same.assumptions[1]
    diverged = compare_executions(
        left,
        RuntimeObservation(
            project=left.project,
            source_snapshot=left.source_snapshot,
            compiler_configuration=left.compiler_configuration,
            runtime_configuration=left.runtime_configuration,
            sequence_id=left.sequence_id,
            transaction_index=left.transaction_index,
            contract=left.contract,
            function_identity=left.function_identity,
            actor=left.actor,
            success="false",
            block_number=left.block_number,
            chain_id=left.chain_id,
            state_snapshot=left.state_snapshot,
            deployment_address=left.deployment_address,
            execution_id="run-2",
            status="observed",
        ),
        reset=True,
    )
    assert diverged.status == "candidate"
    assert diverged.classification == "deterministic divergence"
    assert failure_is_vulnerability("false") is False
    other_block = _state(block="2")
    assert same_state(state_of(left), state_of(other_block)) is False
    assert compare_executions(left, other_block, reset=True).status == "unknown"
    other_target = _state(address="0x99")
    assert same_state(state_of(left), state_of(other_target)) is False
    assert classify_replay(left, right, reset=False).classification == "state-dependent divergence"
    assert metamorphic_expectation("reorder-independent", established=False) == "unknown"
    assert metamorphic_expectation("same-snapshot", established=True) == "expected-equivalent"
    incomplete = parse_runtime_document('{"schema": "bugforge-runtime-v1", "transactions": []}')
    assert incomplete is not None
    assert incomplete[0].status == "incomplete"
    assert clamp_runtime_executions(100) == MAX_EXECUTIONS
    assert clamp_protocol_bounds(ProtocolBounds(max_expansion=100)).max_expansion == (
        MAX_PROTOCOL_EXPANSION
    )
    engine = RuntimeEngine()
    failed = engine.start_campaign(AnalysisRequest(tmp_path, "solidity", target="Vault"))
    assert failed.executed is False
    assert failed.metadata["verified"] == "false"
    assert failed.to_evidence().contributes_to_verification is False


def test_scheduler_budget_covers_protocol_runtime_and_followup(tmp_path: Path) -> None:
    protocol = AnalysisRequest(tmp_path, "solidity", target="Vault", extra={"protocol": "true"})
    runtime = AnalysisRequest(tmp_path, "solidity", target="Vault", extra={"runtime": "true"})
    fork = AnalysisRequest(tmp_path, "solidity", target="Vault", extra={"fork": "true"})
    differential = AnalysisRequest(
        tmp_path, "solidity", target="Vault", extra={"differential": "true"}
    )
    feedback = DiscoveryScheduler(engines=()).feedback
    assert missing_capability(protocol, feedback) == "cross_contract_analysis"
    assert missing_capability(runtime, feedback) == "runtime_validation"
    assert missing_capability(fork, feedback) == "fork_validation"
    assert missing_capability(differential, feedback) == "differential_validation"
    assert (
        choose_next(
            ResearchObservation("runtime", "missing"),
            stalled=False,
            has_property=False,
            static_known=False,
        ).capability
        == "runtime_validation"
    )
    engines = (
        ProtocolEngine(),
        RuntimeEngine(),
        _Fake("bugforge-static", frozenset({EngineCapability.STATIC_ANALYSIS})),
        _Fake("halmos", frozenset({EngineCapability.SYMBOLIC_EXECUTION})),
    )
    scheduler = DiscoveryScheduler(engines, max_engines=1)
    assert [item.engine_id for item in scheduler.select(protocol) if item.action == "run"] == [
        "bugforge-protocol"
    ]
    unavailable = scheduler.run_selected(protocol)
    assert unavailable[0].status is ResultStatus.UNSUPPORTED
    assert "cross_contract_analysis" not in scheduler.feedback.exercised
    budget = DiscoveryScheduler(engines, max_engines=1)
    budget.feedback.difficult = True
    budget.engines_started = 1
    assert (
        budget.run_followup(AnalysisRequest(tmp_path, "solidity", target="Vault", difficult=True))
        == []
    )
    assert budget.run_selected(runtime) == []


def test_protocol_graph_keeps_edge_identity_and_rejects_false_dependencies() -> None:
    left = InteractionEdge("edge-17", "A", "B", "external_call", "established", "fn", "c1")
    right = InteractionEdge("edge-29", "A", "B", "staticcall", "established", "fn", "c2")
    graph = ProtocolGraph(
        (_node("A", "Router", "Router.sol"), _node("B", "Vault", "Vault.sol")),
        (),
        (left, right),
        (),
        (),
        False,
        "",
        "p",
        "snap",
        "cfg",
    )
    paths = expand_paths(graph)
    assert {item.edges for item in paths} >= {("edge-17",), ("edge-29",)}
    assert any("edge-17" in item.path_id and "edge-29" not in item.path_id for item in paths)
    oracle = _call("latestAnswer", "Oracle", "oracle", "o1")
    token = _call("transfer", "Token", "token", "t1")
    program = _program(
        "A.sol",
        (
            _function("Router", "swap", 1, (oracle, token)),
            _function("Oracle", "latestAnswer", 2),
            _function("Token", "transfer", 3),
        ),
        (
            ContractFact("Router", "contract", (), (10, 40, 1, 1, 1, 20)),
            ContractFact("Oracle", "contract", ()),
            ContractFact("Token", "contract", ()),
        ),
    )
    built = build_protocol_graph(
        (program,), project="p", source_snapshot="snap", compiler_configuration="cfg"
    )
    assert all(item.kind != "price_dependency" for item in built.edges)
    assert built.nodes[0].contract == "Router"
    router = next(item for item in built.nodes if item.contract == "Router")
    assert router.span == (10, 40, 1, 1, 1, 20)
    assert router.span != (1, 0, 1, 20)
    oracle_edge = next(item.edge_id for item in built.edges if item.kind == "oracle_read")
    asset_edge = next(item.edge_id for item in built.edges if item.kind == "token_transfer")
    coincidence = build_protocol_graph(
        (program,),
        project="p",
        source_snapshot="snap",
        compiler_configuration="cfg",
        price_links=(
            PriceLink(oracle_edge, "", "", asset_edge, "same-function oracle read and asset call"),
        ),
    )
    assert all(item.kind != "price_dependency" for item in coincidence.edges)
    linked = build_protocol_graph(
        (program,),
        project="p",
        source_snapshot="snap",
        compiler_configuration="cfg",
        price_links=(
            PriceLink(
                oracle_edge,
                "compute-1",
                "valuation-1",
                asset_edge,
                "oracle value flows through valuation into the transfer",
                (4, 8, 2, 1, 2, 20),
            ),
        ),
    )
    assert any(item.kind == "price_dependency" for item in linked.edges)
    sink = _function("Sink", "onTokenReceived", 3, authorization="established")
    caller = _function("Router", "fill", 1, (_call("onTokenReceived", "Sink", "external", "cb"),))
    auth_program = _program(
        "A.sol",
        (caller, sink),
        (ContractFact("Router", "contract", ()), ContractFact("Sink", "contract", ())),
    )
    denied = build_protocol_graph(
        (auth_program,), project="p", source_snapshot="snap", compiler_configuration="cfg"
    )
    assert all(item.kind != "authorization_dependency" for item in denied.edges)
    caller_id = next(item.function_id for item in denied.functions if item.name == "fill")
    callee_id = next(
        item.function_id for item in denied.functions if item.name == "onTokenReceived"
    )
    allowed = build_protocol_graph(
        (auth_program,),
        project="p",
        source_snapshot="snap",
        compiler_configuration="cfg",
        trust_boundaries=(
            TrustBoundary(
                caller_id,
                callee_id,
                "cb",
                "caller-controlled-amount",
                "established",
                "established",
                "caller-controlled identity crosses the call",
                "privileged sink can be triggered by that input",
                "trust-boundary",
            ),
        ),
    )
    assert any(item.kind == "authorization_dependency" for item in allowed.edges)


def test_protocol_sequences_files_overloads_and_evidence(tmp_path: Path) -> None:
    router = _program(
        "Router.sol",
        (_function("Router", "swap", 1, source="function swap(uint256 amount) external { }"),),
        (ContractFact("Router", "contract", (), (1, 2, 1, 1, 1, 8)),),
    )
    vault = _program(
        "Vault.sol",
        (_function("Vault", "deposit", 2, source="function deposit(uint256 assets) external { }"),),
        (ContractFact("Vault", "contract", ()),),
    )
    token = _program(
        "Token.sol",
        (
            _function(
                "Token",
                "transferFrom",
                3,
                source="function transferFrom(address from, address to, uint256 amount) external { }",
            ),
        ),
        (ContractFact("Token", "contract", ()),),
    )
    oracle = _program(
        "Oracle.sol",
        (_function("Oracle", "read", 4, source="function read() external view { }"),),
        (ContractFact("Oracle", "contract", ()),),
    )
    calls = plan_protocol_calls(
        (
            ("Router.sol", "Router", "swap"),
            ("Vault.sol", "Vault", "deposit"),
            ("Token.sol", "Token", "transferFrom"),
            ("Oracle.sol", "Oracle", "read"),
        ),
        None,
        "",
        programs=(router, vault, token, oracle),
    )
    assert calls is not None
    assert [item.function for item in calls] == ["swap", "deposit", "transferFrom", "read"]
    assert calls[0].arguments
    assert calls[3].arguments == ()
    assert (
        plan_protocol_calls((("Vault.sol", "Vault", "missing"),), None, "", programs=(vault,))
        is None
    )
    left = _function("Vault", "foo", 1, source="function foo(uint256 amount) external { }")
    right = _function(
        "Vault", "foo", 2, source="function foo(address account) external { }", span=(2, 0, 2, 40)
    )
    overloaded = _program("Vault.sol", (left, right), (ContractFact("Vault", "contract", ()),))
    graph = build_protocol_graph(
        (overloaded,), project="p", source_snapshot="snap", compiler_configuration="cfg"
    )
    signatures = sorted(item.signature for item in graph.functions)
    assert signatures == ["foo(address)", "foo(uint256)"]
    assert graph.functions[0].function_id != graph.functions[1].function_id
    assert plan_protocol_calls((("Vault", "foo"),), overloaded, "") is None
    selected = plan_protocol_calls((("Vault", "foo(uint256)"),), overloaded, "")
    assert selected is not None
    assert selected[0].function == "foo"
    assert selected[0].function_identity.endswith(":1")
    evidence = protocol_domain_evidence(
        protocol_evidence(
            graph, engines=("bugforge-protocol",), environment="local", uncertainty="candidate"
        )
    )
    assert evidence.kind is EvidenceKind.PROTOCOL_OBSERVATION
    assert evidence.contributes_to_verification is False
    assert evidence.metadata["verified"] == "false"
    matched = correlate_modalities(
        static_id="op",
        runtime_event="Transfer",
        transition_id="op",
        economic_status="verified",
        trace="",
        static_operation="Vault.sol::Vault.deposit(uint256)",
        runtime_operation="Vault.sol::Vault.deposit(uint256)",
        transition_operation="Vault.sol::Vault.deposit(uint256)",
        economic_operation="Vault.sol::Vault.deposit(uint256)",
    )
    assert matched.status == "candidate"
    assert matched.economic_status == "candidate"
    assert matched.executable_input == ""
    named = correlate_modalities(
        static_id="",
        runtime_event="Transfer",
        transition_id="",
        economic_status="verified",
        trace="",
        static_operation="deposit",
        runtime_operation="deposit",
        transition_operation="deposit",
    )
    assert named.status == "incomplete"
    ambiguous = correlate_modalities(
        static_id="op",
        runtime_event="",
        transition_id="op",
        economic_status="candidate",
        trace="",
        static_operation="Vault.sol::Vault.foo(uint256)",
        runtime_operation="Vault.sol::Vault.foo(uint256)",
        transition_operation="Vault.sol::Vault.foo(uint256)",
        overload_ambiguous=True,
    )
    assert ambiguous.status == "incomplete"
    source = tmp_path / "Router.sol"
    source.write_text(
        "contract Router { function swap(uint256 amount) external {} }\n", encoding="utf-8"
    )
    engine = ProtocolEngine()
    result = engine.start_campaign(
        AnalysisRequest(
            tmp_path,
            "solidity",
            target="Router",
            files=("Router.sol",),
            extra={"project_id": "p", "source_snapshot": "snap", "compiler_configuration": "cfg"},
        )
    )
    assert result.status is ResultStatus.INGESTED
    assert result.executed is False
    assert result.metadata["verified"] == "false"
    assert result.metadata["llm_invoked"] == "false"
    assert result.to_evidence().kind is EvidenceKind.PROTOCOL_OBSERVATION
    assert result.to_evidence().contributes_to_verification is False
    edge = InteractionEdge("edge-17", "A", "B", "external_call", "established", "fn-1", "c1")
    function = FunctionNode(
        "fn-1", "A", "Router", "deposit", "deposit(uint256)", "Router.sol", 1, (1, 0, 1, 10)
    )
    named = FunctionNode(
        "fn-name", "A", "Router", "deposit", "unresolved", "Router.sol", 2, (2, 0, 2, 10)
    )
    bound = ProtocolGraph(
        (_node("A", "Router", "Router.sol"), _node("B", "Vault", "Vault.sol")),
        (function, named),
        (edge, InteractionEdge("edge-name", "A", "B", "external_call", "name", "fn-name", "c2")),
        (),
        (),
        False,
        "",
        "p",
        "snap",
        "cfg",
    )
    assert (
        bind_protocol_edge(
            bound, contract_identity="Router", function_identity="fn-1", edge_id="edge-17"
        )
        == "edge-17"
    )
    assert (
        bind_protocol_edge(
            bound, contract_identity="Router", function_identity="deposit", edge_id="edge-17"
        )
        == ""
    )
    assert (
        bind_protocol_edge(
            bound, contract_identity="Router", function_identity="fn-name", edge_id="edge-name"
        )
        == ""
    )
