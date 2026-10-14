"""Phase 47 protocol graph, hardened economics, and scheduler bounds."""

from __future__ import annotations

from pathlib import Path

from app.adapters.discovery.economic import EconomicEngine
from app.adapters.discovery.ityfuzz import (
    ItyFuzzEngine,
    parse_ityfuzz_output,
    parse_ityfuzz_streams,
)
from app.discovery.capabilities import EngineAvailability, EngineCapability, ResultStatus
from app.discovery.corpus import SeedSource
from app.discovery.engine import AnalysisRequest, DiscoveryEngine
from app.discovery.loop import ResearchObservation, choose_next
from app.discovery.results import DynamicResult
from app.discovery.scheduler import DiscoveryScheduler, coverage_key, missing_capability
from app.discovery.sequences import (
    MAX_EXECUTIONS,
    ExplorationBounds,
    PlannedCall,
    PlannedSequence,
    action_hypothesis,
    clamp_bounds,
    economic_mutations,
)
from app.domain.evidence import EvidenceKind
from app.parsing.solidity_arguments import ArgumentCandidate
from app.parsing.solidity_economics import (
    AssetDelta,
    OracleStatus,
    fee_on_transfer,
    lending_transition,
    parse_integer,
    preview_discrepancy,
)
from app.parsing.solidity_ir import (
    ContractFact,
    SemanticCall,
    SemanticFunction,
    SemanticProgram,
    SourceSpanRef,
)
from app.parsing.solidity_protocol import (
    MAX_PROTOCOL_CONTRACTS,
    MAX_PROTOCOL_EDGES,
    EventLink,
    ImplementationLink,
    ProtocolBounds,
    StorageLink,
    actor_privilege,
    build_protocol_graph,
    candidate_status,
    clamp_protocol_bounds,
    correlate_modalities,
    expand_paths,
    plan_protocol_calls,
    protocol_evidence,
    protocol_sequences,
    same_storage,
    separate_asset_deltas,
)


def _span(line: int = 1) -> SourceSpanRef:
    return SourceSpanRef("A.sol", line, line + 1, line, line)


def _call(callee: str, target: str, call_type: str, resolution: str, call_id: str) -> SemanticCall:
    return SemanticCall(
        "caller",
        callee,
        resolution,
        target,
        "",
        "",
        call_type,
        call_type != "internal",
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
    )


def _program(
    file: str, functions: tuple[SemanticFunction, ...], contracts: tuple[ContractFact, ...]
) -> SemanticProgram:
    return SemanticProgram(
        "phase39.1",
        file,
        "partial",
        functions,
        (),
        "",
        "unavailable",
        contracts=contracts,
    )


def _graph(*programs: SemanticProgram, **kwargs: object):
    return build_protocol_graph(
        programs,
        project=str(kwargs.get("project", "project")),
        source_snapshot=str(kwargs.get("source_snapshot", "snap")),
        compiler_configuration=str(kwargs.get("compiler_configuration", "solc-0.8.20")),
    )


def test_similar_names_do_not_connect_contracts() -> None:
    vault = _function("Vault", "deposit", 1)
    other = _function("VaultV2", "deposit", 2)
    program = _program(
        "A.sol",
        (vault, other),
        (ContractFact("Vault", "contract", ()), ContractFact("VaultV2", "contract", ())),
    )
    graph = _graph(program)
    assert graph.edges == ()
    assert {item.contract for item in graph.nodes} == {"Vault", "VaultV2"}


def test_same_function_names_keep_contract_identity() -> None:
    program = _program(
        "A.sol",
        (_function("Router", "swap", 1), _function("Pool", "swap", 4, span=(4, 0, 4, 30))),
        (ContractFact("Router", "contract", ()), ContractFact("Pool", "contract", ())),
    )
    graph = _graph(program)
    ids = [item.function_id for item in graph.functions]
    assert len(ids) == 2
    assert ids[0] != ids[1]
    assert "Router" in ids[0]
    assert "Pool" in ids[1]


def test_overloads_remain_distinct() -> None:
    first = _function("Vault", "deposit", 1, span=(1, 0, 1, 10))
    second = _function("Vault", "deposit", 8, span=(8, 0, 8, 40))
    program = _program("A.sol", (first, second), (ContractFact("Vault", "contract", ()),))
    graph = _graph(program)
    assert len(graph.functions) == 2
    assert graph.functions[0].function_id != graph.functions[1].function_id
    assert graph.functions[0].line != graph.functions[1].line


def test_unresolved_dynamic_calls_stay_unknown() -> None:
    site = _call("call", "target", "low-level-call", "low-level", "dyn")
    caller = _function("Router", "run", 1, (site,))
    token = _function("Token", "transfer", 3)
    program = _program(
        "A.sol",
        (caller, token),
        (ContractFact("Router", "contract", ()), ContractFact("Token", "contract", ())),
    )
    graph = _graph(program)
    assert graph.edges == ()
    assert graph.unresolved
    assert graph.unresolved[0].reason == "unresolved"


def test_delegatecall_is_not_an_ordinary_call() -> None:
    delegated = _call("delegatecall", "Impl", "delegatecall", "resolved", "d1")
    ordinary = _call("run", "Impl", "external", "resolved", "e1")
    proxy = _function("Proxy", "forward", 1, (delegated,))
    user = _function("Proxy", "use", 2, (ordinary,), span=(2, 0, 2, 20))
    impl = _function("Impl", "run", 4)
    program = _program(
        "A.sol",
        (proxy, user, impl),
        (ContractFact("Proxy", "contract", ()), ContractFact("Impl", "contract", ())),
    )
    graph = _graph(program)
    kinds = {item.kind for item in graph.edges}
    assert "delegatecall" in kinds
    assert "external_call" in kinds
    assert "delegatecall" != "external_call"


def test_proxy_relationship_requires_evidence() -> None:
    proxy = _function("Proxy", "fallback", 1)
    impl = _function("Impl", "run", 2)
    program = _program(
        "A.sol",
        (proxy, impl),
        (ContractFact("Proxy", "contract", ()), ContractFact("Impl", "contract", ())),
    )
    bare = build_protocol_graph(
        (program,), project="p", source_snapshot="s", compiler_configuration="c"
    )
    assert all(item.kind != "proxy_implementation" for item in bare.edges)
    linked = build_protocol_graph(
        (program,),
        project="p",
        source_snapshot="s",
        compiler_configuration="c",
        implementation_links=(
            ImplementationLink("Proxy", "Impl", "established-layout", "A.sol", "A.sol", "c"),
        ),
    )
    assert any(item.kind == "proxy_implementation" for item in linked.edges)
    unknown = build_protocol_graph(
        (program,),
        project="p",
        source_snapshot="s",
        compiler_configuration="c",
        implementation_links=(ImplementationLink("Proxy", "", "guess", "A.sol", "A.sol", "c"),),
    )
    assert all(item.kind != "proxy_implementation" for item in unknown.edges)


def test_storage_dependencies_are_declaration_specific() -> None:
    assert same_storage("decl-a", "decl-b", "balance", "balance") is False
    assert same_storage("decl-a", "decl-a", "balance", "reserves") is True
    writer = _function("Market", "borrow", 1)
    reader = _function("Oracle", "peek", 2)
    program = _program(
        "A.sol",
        (writer, reader),
        (ContractFact("Market", "contract", ()), ContractFact("Oracle", "contract", ())),
    )
    graph = build_protocol_graph(
        (program,),
        project="p",
        source_snapshot="s",
        compiler_configuration="c",
    )
    writer_id = next(item.function_id for item in graph.functions if item.contract == "Market")
    reader_id = next(item.function_id for item in graph.functions if item.contract == "Oracle")
    linked = build_protocol_graph(
        (program,),
        project="p",
        source_snapshot="s",
        compiler_configuration="c",
        storage_links=(StorageLink("decl-a", writer_id, reader_id, "same-declaration", "A.sol"),),
    )
    assert any(item.kind == "storage_dependency" for item in linked.edges)
    named = build_protocol_graph(
        (program,),
        project="p",
        source_snapshot="s",
        compiler_configuration="c",
        storage_links=(StorageLink("", writer_id, reader_id, "name-only", "A.sol"),),
    )
    assert all(item.kind != "storage_dependency" for item in named.edges)


def test_asset_and_token_identity_stay_separated() -> None:
    attacker = AssetDelta("erc20", "USDC", "token", "attacker", 1, 2, 1, "positive")
    victim = AssetDelta("erc20", "USDC", "token", "victim", 2, 1, -1, "negative")
    other = AssetDelta("erc20", "WETH", "token", "attacker", 1, 2, 1, "positive")
    separated = separate_asset_deltas((attacker, victim))
    assert separated.status == OracleStatus.INCOMPLETE.value
    assert "profit" in separated.assumptions[0]
    mixed = separate_asset_deltas((attacker, other))
    assert mixed.status == OracleStatus.UNKNOWN.value
    assert "conversion" in mixed.explanation


def test_callback_and_authorization_are_not_labels() -> None:
    assert actor_privilege("attacker", "unknown") == "unknown"
    assert actor_privilege("admin", "") == "unknown"
    assert actor_privilege("callback", "unknown") == "unknown"
    assert actor_privilege("user", "established") == "established"
    missing = _call("onTokenReceived", "Sink", "external", "unresolved", "cb")
    present = _call("onTokenReceived", "Sink", "external", "resolved", "cb2")
    source = _function("Router", "fill", 1, (missing,))
    sink = _function("Sink", "onTokenReceived", 3, authorization="established")
    program = _program(
        "A.sol",
        (source, sink),
        (ContractFact("Router", "contract", ()), ContractFact("Sink", "contract", ())),
    )
    graph = _graph(program)
    assert all(item.kind != "callback" for item in graph.edges)
    source = _function("Router", "fill", 1, (present,))
    program = _program(
        "A.sol",
        (source, sink),
        (ContractFact("Router", "contract", ()), ContractFact("Sink", "contract", ())),
    )
    graph = _graph(program)
    assert any(item.kind == "callback" for item in graph.edges)
    assert all(item.kind != "authorization_dependency" for item in graph.edges)
    assert all(item.status == "candidate" for item in graph.hypotheses)
    assert candidate_status("verified") == "candidate"
    assert candidate_status("exploited") == "candidate"


def test_expansion_and_planner_caps_cannot_be_raised() -> None:
    raised = clamp_protocol_bounds(
        ProtocolBounds(max_contracts=100, max_edges=100, max_expansion=100, max_callback_depth=9)
    )
    assert raised.max_contracts == MAX_PROTOCOL_CONTRACTS
    assert raised.max_edges == MAX_PROTOCOL_EDGES
    assert raised.max_expansion == 16
    assert raised.max_callback_depth == 2
    tightened = clamp_protocol_bounds(ProtocolBounds(max_contracts=2, max_edges=3))
    assert tightened.max_contracts == 2
    assert clamp_bounds(ExplorationBounds(max_executions=99)).max_executions == MAX_EXECUTIONS
    steps = tuple((f"C{index}", "run") for index in range(5))
    assert plan_protocol_calls(steps, None, "") is None
    assert len(steps) > 4
    token = _call("transfer", "Token", "token", "resolved", "t1")
    vault_call = _call("deposit", "Vault", "external", "resolved", "v1")
    program = _program(
        "A.sol",
        (
            _function("Router", "swap", 1, (vault_call,)),
            _function("Vault", "deposit", 2, (token,), span=(2, 0, 2, 30)),
            _function("Token", "transfer", 3, span=(3, 0, 3, 20)),
        ),
        (
            ContractFact("Router", "contract", ()),
            ContractFact("Vault", "contract", ()),
            ContractFact("Token", "contract", ()),
        ),
    )
    graph = _graph(program)
    assert any(item.kind == "external_call" for item in graph.edges)
    assert any(item.kind == "token_transfer" for item in graph.edges)
    assert len(expand_paths(graph, ProtocolBounds(max_expansion=1))) == 1


def test_invalid_action_is_not_an_executable_call() -> None:
    hypothesis = action_hypothesis("donate", source="", contract="", program=None)
    assert hypothesis.executable is False
    call = PlannedCall(
        "deposit", "user", (ArgumentCandidate("1", "one", "uint256"),), "0", "in", (), 1
    )
    sequence = PlannedSequence("id", "path", (call,), "planned", (), "base")
    mutated = economic_mutations(
        sequence,
        bounds=ExplorationBounds(max_mutations=4),
        established=frozenset({"donate", "swap", "liquidate"}),
    )
    assert all(item.calls[0].function != "donate" for item in mutated)
    assert all(item.calls[0].arguments for item in mutated)


def test_malformed_numbers_fee_preview_and_lending() -> None:
    assert parse_integer("abc") is None
    assert parse_integer("-5") is None
    assert parse_integer("-5", signed=True) == -5
    assert parse_integer("9" * 80) is None
    assert parse_integer("1.5") is None
    engine = EconomicEngine()
    result = engine.start_campaign(
        AnalysisRequest(
            Path("/tmp/bugforge-phase47"),
            "solidity",
            target="Vault",
            extra={"case": "eth", "before": "nope", "after": "1"},
        )
    )
    assert result.executed is False
    assert result.status is not ResultStatus.FAILED
    assert result.to_evidence().kind is EvidenceKind.ECONOMIC_OBSERVATION
    assert result.to_evidence().contributes_to_verification is False
    reversed_fee = fee_on_transfer(
        requested=10, sender_delta=10, receiver_delta=-2, accounted=10, semantics="exact"
    )
    assert reversed_fee.status == OracleStatus.UNKNOWN.value
    assert reversed_fee.status != OracleStatus.BALANCED.value
    mint = preview_discrepancy(
        preview=9,
        executed=10,
        established=frozenset({"erc4626-preview"}),
        operation="previewMint",
    )
    assert mint.status == OracleStatus.INVARIANT_VIOLATION.value
    shares = preview_discrepancy(
        preview=11,
        executed=10,
        established=frozenset({"erc4626-preview"}),
        operation="convertToShares",
    )
    assert shares.status == OracleStatus.INVARIANT_VIOLATION.value
    assets = preview_discrepancy(
        preview=9,
        executed=10,
        established=frozenset({"erc4626-preview"}),
        operation="convertToAssets",
    )
    assert assets.status == OracleStatus.INCOMPLETE.value
    unpaid = lending_transition(
        debt_before=10,
        debt_after=4,
        repayment=None,
        collateral_before=None,
        collateral_after=None,
        health="unknown",
        liquidated=False,
    )
    assert unpaid.status == OracleStatus.INCOMPLETE.value


def test_ityfuzz_trace_is_not_a_corpus_seed(tmp_path: Path) -> None:
    banner = """
    Found vulnerabilities!
    ================ Description ================
    [Fund Loss]: example
    ================ Trace ================
    [Sender] 0xabc
       └─[1] 0xdef.sync()
    """
    parsed = parse_ityfuzz_output(banner)
    assert parsed.minimized == ""
    assert "[Sender]" in parsed.diagnostic
    streams = parse_ityfuzz_streams("not a finding", banner)
    assert streams.findings
    assert streams.minimized == ""
    scheduler = DiscoveryScheduler(engines=(), max_engines=1)
    scheduler.note_result(
        DynamicResult(
            engine="ityfuzz",
            language="solidity",
            target="Vault",
            status=ResultStatus.INTERESTING,
            executed=True,
            minimized_input=banner,
            metadata={"executable_input": "false"},
        ),
        AnalysisRequest(tmp_path, "solidity", target="Vault"),
    )
    assert scheduler.corpus.by_source(SeedSource.ITYFUZZ) == ()
    source = Path(__file__).resolve().parents[2] / "app" / "adapters" / "discovery" / "ityfuzz.py"
    text = source.read_text(encoding="utf-8")
    assert "shell=True" not in text
    assert "subprocess" not in text
    assert ItyFuzzEngine().availability() is EngineAvailability.UNAVAILABLE


def test_economic_observation_cannot_verify(tmp_path: Path) -> None:
    result = EconomicEngine().start_campaign(
        AnalysisRequest(
            tmp_path,
            "solidity",
            target="Vault",
            contract="Vault",
            function="Vault.deposit:1",
            source_file="Vault.sol",
            extra={
                "case": "eth",
                "before": "1",
                "after": "2",
                "token": "ETH",
                "actor": "attacker",
                "sequence_id": "seq",
                "source_snapshot": "snap",
                "compiler_configuration": "cfg",
                "initial_state": "before",
                "final_state": "after",
                "transaction_index": "0",
                "provenance": "runtime-observation",
            },
        )
    )
    assert result.executed is False
    assert result.provenance == "economic-calculation"
    evidence = result.to_evidence()
    assert evidence.kind is EvidenceKind.ECONOMIC_OBSERVATION
    assert evidence.contributes_to_verification is False
    assert evidence.metadata["verified"] == "false"
    assert result.metadata["input_provenance"] == "externally-supplied"
    assert result.metadata["bound"] == "true"
    economic = result.economic_evidence
    assert economic is not None
    assert economic.bound is True
    assert economic.observation_class != "runtime-observation"


def test_protocol_evidence_keeps_source_binding() -> None:
    program = _program(
        "A.sol", (_function("Vault", "deposit", 1),), (ContractFact("Vault", "contract", ()),)
    )
    graph = build_protocol_graph(
        (program,), project="repo", source_snapshot="snap-1", compiler_configuration="cfg-1"
    )
    evidence = protocol_evidence(graph, engines=("bugforge-static",), environment="local")
    assert evidence.project == "repo"
    assert evidence.source_snapshot == "snap-1"
    assert evidence.compiler_configuration == "cfg-1"
    assert evidence.status == "candidate"
    other = build_protocol_graph(
        (program,), project="repo", source_snapshot="snap-2", compiler_configuration="cfg-2"
    )
    assert other.functions[0].function_id != graph.functions[0].function_id
    correlation = correlate_modalities(
        static_id="",
        runtime_event="Transfer",
        transition_id="",
        economic_status="verified",
        trace="[Sender] trace",
    )
    assert correlation.executable_input == ""
    assert correlation.economic_status == "candidate"
    assert correlation.status == "incomplete"
    event = EventLink("Transfer", graph.functions[0].function_id, "", "name-only")
    named = build_protocol_graph(
        (program,),
        project="repo",
        source_snapshot="snap-1",
        compiler_configuration="cfg-1",
        event_links=(event,),
    )
    assert all(item.kind != "event_state" for item in named.edges)


def test_coverage_and_followup_and_static_binding(tmp_path: Path) -> None:
    result = DynamicResult(
        engine="foundry",
        language="solidity",
        target="Vault",
        status=ResultStatus.EXECUTED,
        executed=True,
        coverage={"percent": "10"},
    )
    left = AnalysisRequest(
        tmp_path,
        "solidity",
        target="Vault",
        extra={"compiler_configuration": "a", "source_snapshot": "s", "project_id": "p"},
    )
    right = AnalysisRequest(
        tmp_path,
        "solidity",
        target="Vault",
        extra={"compiler_configuration": "b", "source_snapshot": "s", "project_id": "p"},
    )
    assert coverage_key(result, left) != coverage_key(result, right)
    scheduler = DiscoveryScheduler((_Fake("halmos"), _Fake("foundry")), max_engines=1)
    scheduler.feedback.difficult = True
    scheduler.corpus.add(
        "0x",
        source=SeedSource.SYMBOLIC_EXECUTION,
        reason="seed",
        language="solidity",
        target="Vault",
    )
    request = AnalysisRequest(
        tmp_path,
        "solidity",
        target="Vault",
        function="withdraw",
        contract="Vault",
        framework="foundry",
        has_harness=True,
        campaign_id="camp",
        source_file="Vault.sol",
        extra={"source_snapshot": "snap", "compiler_configuration": "cfg", "mode": "fuzz"},
    )
    follow = scheduler.run_followup(request)
    assert len(follow) == 1
    assert scheduler.engines_started == 1
    again = scheduler.run_followup(request)
    assert again == []
    nxt = scheduler.target_from_static(
        repo_root=request,
        file_path="Vault.sol",
        function="withdraw",
        contract="Vault",
        rule_id="sol.reentrancy",
    )
    assert nxt.extra["source_snapshot"] == "snap"
    assert nxt.extra["compiler_configuration"] == "cfg"
    assert nxt.extra["mode"] == "fuzz"
    assert nxt.extra["source"] == "static_finding"
    assert nxt.campaign_id == "camp"
    assert nxt.contract == "Vault"
    plain = AnalysisRequest(tmp_path, "solidity", target="Vault", extra={"protocol": "true"})
    assert missing_capability(plain, DiscoveryScheduler(engines=()).feedback) == (
        "cross_contract_analysis"
    )
    action = choose_next(
        ResearchObservation("cross-contract", "no graph"),
        stalled=False,
        has_property=False,
        static_known=False,
    )
    assert action.capability == "cross_contract_analysis"


def test_executable_protocol_sequence_rejects_missing_arguments() -> None:
    source = "contract Vault { function deposit(uint256 assets) external {} }"
    assert protocol_sequences((("Vault", "missing"),), None, source) == ()
    long = tuple(("Vault", "deposit") for _ in range(MAX_EXECUTIONS))
    assert plan_protocol_calls(long, None, source) is None


class _Fake(DiscoveryEngine):
    def __init__(self, engine_id: str) -> None:
        self._id = engine_id

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
        if self._id == "halmos":
            return frozenset({EngineCapability.SYMBOLIC_EXECUTION})
        return frozenset({EngineCapability.FUZZING, EngineCapability.TEST_EXECUTION})

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

    def _analyze_target(self, request: AnalysisRequest) -> DynamicResult:
        return self._start_campaign(request)
