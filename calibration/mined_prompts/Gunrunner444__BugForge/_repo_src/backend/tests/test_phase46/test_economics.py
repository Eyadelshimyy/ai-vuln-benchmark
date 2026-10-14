"""Phase 46 economic observations and the Phase 45 execution corrections."""

from __future__ import annotations

import subprocess
from dataclasses import replace
from pathlib import Path

from app.adapters.discovery.economic import EconomicEngine
from app.adapters.discovery.ityfuzz import (
    ItyFuzzEngine,
    ityfuzz_command,
    parse_ityfuzz_output,
)
from app.benchmarks.catalog import ECONOMIC_BENCHMARKS, load_local
from app.discovery.capabilities import EngineAvailability, EngineCapability, ResultStatus
from app.discovery.corpus import DiscoveryCorpus, SeedSource
from app.discovery.engine import AnalysisRequest, DiscoveryEngine
from app.discovery.loop import ResearchObservation, choose_next
from app.discovery.results import DynamicResult
from app.discovery.scheduler import DiscoveryScheduler, missing_capability
from app.discovery.sequences import (
    ExplorationBounds,
    PlannedCall,
    PlannedSequence,
    bind_sequence,
    economic_mutations,
    established_actions,
    refine_actions,
    sequence_bound,
)
from app.domain.evidence import EvidenceKind
from app.parsing.engine import parse_source, reset_syntax_registry
from app.parsing.solidity_arguments import ArgumentCandidate
from app.parsing.solidity_economics import (
    Conversion,
    ConversionKind,
    EconomicEvidence,
    OracleStatus,
    Quantity,
    amm_invariant,
    canonical_status,
    correlate_event,
    debt_delta,
    donation_inflation,
    erc20_delta,
    erc4626_rounding,
    fee_delta,
    fee_on_transfer,
    flash_position,
    lending_transition,
    native_eth_delta,
    net_change,
    oracle_dependency,
    preview_discrepancy,
    quantity_delta,
    reserve_delta,
    share_delta,
    snapshot_of,
)
from app.parsing.solidity_ir import build_semantic_program
from app.parsing.solidity_replay import prepare_replay, promote, replay_evidence, run_replay
from app.parsing.solidity_spec import specify
from app.parsing.solidity_state_transitions import analyze_state_transitions
from app.plugins import reset_plugin_catalog

_FAILING = """
pragma solidity ^0.8.20;
contract Vault {
    uint256 public totalSupply;
    uint256 public balances;
    function mint() public { totalSupply = totalSupply + 1; }
}
"""

_BANNER = """
Found vulnerabilities!
================ Description ================
[Fund Loss]: Anyone can earn 8.254 ETH by interacting with the provided contracts
================ Trace ================
[Sender] 0xe1A425f1AC34A8a441566f93c82dD730639c8510
   └─[1] 0x17269a3CACB6eA16FE5137eC3ccBde00A6A97668.sync()
"""

_NOT_A_CONCLUSION = frozenset(
    {"safe", "verified", "proved", "exploited", "confirmed", "profitable"}
)


def _qty(
    token: str, actor: str, amount: int | None, snapshot: str, kind: str = "balance"
) -> Quantity:
    return Quantity(
        token,
        "wei" if token == "ETH" else "token",
        actor,
        amount,
        "observation",
        0,
        snapshot,
        "runtime-observation",
        "observed" if amount is not None else "unknown",
        kind=kind,
    )


def test_economic_state_snapshot_keeps_identity() -> None:
    item = _qty("USDC", "attacker", 5, "before", "erc20")
    snap = snapshot_of(item, snapshot_id="s0", caller="attacker")
    assert snap.snapshot_id == "s0"
    assert snap.quantities[0].token == "USDC"
    assert snap.quantities[0].actor == "attacker"
    assert snap.quantities[0].provenance == "runtime-observation"
    assert snap.caller == "attacker"


def test_native_eth_delta() -> None:
    delta = native_eth_delta(_qty("ETH", "user", 1, "before"), _qty("ETH", "user", 4, "after"))
    assert delta.kind == "eth"
    assert delta.delta == 3
    assert delta.known == "positive"
    assert delta.token == "ETH"


def test_erc20_delta() -> None:
    delta = erc20_delta(_qty("USDC", "user", 10, "before"), _qty("USDC", "user", 7, "after"))
    assert delta.kind == "erc20"
    assert delta.delta == -3
    assert delta.known == "negative"


def test_share_delta() -> None:
    delta = share_delta(_qty("SHARE", "user", 2, "before"), _qty("SHARE", "user", 2, "after"))
    assert delta.kind == "share"
    assert delta.known == "zero"


def test_debt_delta() -> None:
    delta = debt_delta(_qty("DAI", "user", 9, "before"), _qty("DAI", "user", 4, "after"))
    assert delta.kind == "debt"
    assert delta.delta == -5


def test_reserve_delta() -> None:
    delta = reserve_delta(_qty("WETH", "pool", 10, "before"), _qty("WETH", "pool", None, "after"))
    assert delta.kind == "reserve"
    assert delta.known == "unknown"
    assert delta.delta is None


def test_fee_delta() -> None:
    delta = fee_delta(_qty("USDC", "protocol", 1, "before"), _qty("USDC", "protocol", 3, "after"))
    assert delta.kind == "fee"
    assert delta.delta == 2


def test_actor_specific_delta_does_not_cross_actors() -> None:
    delta = quantity_delta(
        _qty("USDC", "attacker", 1, "before"),
        _qty("USDC", "user", 5, "after"),
        kind="actor",
    )
    assert delta.known == "unknown"
    assert delta.delta is None


def test_token_identity_mismatch_stays_unknown() -> None:
    delta = erc20_delta(_qty("USDC", "user", 1, "before"), _qty("ETH", "user", 1, "after"))
    assert delta.known == "unknown"
    assert delta.delta is None


def test_no_invented_usd_conversion() -> None:
    result = net_change(
        final=5,
        initial=0,
        costs=0,
        fees=0,
        repayments=0,
        token="USDC",
        other_token="ETH",
        other_amount=1,
        conversion=None,
    )
    assert result.status == OracleStatus.UNKNOWN.value
    assert result.status not in _NOT_A_CONCLUSION


def test_explicit_conversion_source_works() -> None:
    source = Conversion(ConversionKind.EXACT_LOCAL.value, "SHARE", "USDC", 2, 1, "local accounting")
    result = net_change(
        final=0,
        initial=0,
        costs=0,
        fees=0,
        repayments=0,
        token="USDC",
        other_token="SHARE",
        other_amount=5,
        conversion=source,
    )
    assert result.status == OracleStatus.POTENTIAL_POSITIVE_DELTA.value
    assert result.conversion == ConversionKind.EXACT_LOCAL.value
    assert "10" in result.explanation


def test_unsupported_conversion_stays_unknown() -> None:
    source = Conversion(ConversionKind.UNSUPPORTED.value, "ETH", "USDC", 2000, 1, "guess")
    result = net_change(
        final=0,
        initial=0,
        costs=0,
        fees=0,
        repayments=0,
        token="USDC",
        other_token="ETH",
        other_amount=1,
        conversion=source,
    )
    assert result.status == OracleStatus.UNKNOWN.value


def test_borrowed_principal_is_not_attacker_profit() -> None:
    result = flash_position(
        owned_initial=0,
        owned_final=1000,
        borrowed=1000,
        repayment=1000,
        fee=0,
        token="USDC",
    )
    assert result.status == OracleStatus.NON_PROFITABLE.value
    assert "not profit" in result.assumptions[0]


def test_flash_loan_repayment_is_modeled() -> None:
    result = flash_position(
        owned_initial=0,
        owned_final=1006,
        borrowed=1000,
        repayment=1000,
        fee=1,
        token="USDC",
    )
    assert result.status == OracleStatus.POTENTIAL_POSITIVE_DELTA.value
    assert result.relation == "flash"
    assert result.status not in _NOT_A_CONCLUSION


def test_unpaid_borrow_becomes_unsettled_debt() -> None:
    result = flash_position(
        owned_initial=0,
        owned_final=1000,
        borrowed=1000,
        repayment=None,
        fee=0,
        token="USDC",
    )
    assert result.status == OracleStatus.INCOMPLETE.value
    assert result.relation == "debt"


def test_donation_inflation_sequence_needs_an_observation() -> None:
    silent = donation_inflation(
        assets_before=100,
        shares_before=100,
        donated=100,
        deposit_assets=100,
        shares_minted=50,
        established=frozenset({"donation", "erc4626-deposit"}),
        observed=False,
    )
    assert silent.status == OracleStatus.INCOMPLETE.value
    observed = donation_inflation(
        assets_before=100,
        shares_before=100,
        donated=100,
        deposit_assets=100,
        shares_minted=50,
        established=frozenset({"donation"}),
        observed=True,
    )
    assert observed.status == OracleStatus.POTENTIAL_LOSS.value
    assert observed.relation == "donation/share-conversion"
    assert "exploit" in observed.assumptions[0]


def test_erc4626_rounding_relationship() -> None:
    missing = erc4626_rounding(
        assets=10, total_assets=100, supply=100, direction="down", established=frozenset()
    )
    assert missing.status == OracleStatus.UNSUPPORTED.value
    rounded = erc4626_rounding(
        assets=10,
        total_assets=100,
        supply=100,
        direction="down",
        established=frozenset({"erc4626-deposit"}),
    )
    assert rounded.status == OracleStatus.BALANCED.value
    assert "10" in rounded.explanation


def test_erc4626_preview_execution_discrepancy() -> None:
    unspecified = preview_discrepancy(
        preview=10, executed=9, established=frozenset({"erc4626-preview"})
    )
    assert unspecified.status == OracleStatus.UNKNOWN.value
    deposit = preview_discrepancy(
        preview=10,
        executed=9,
        established=frozenset({"erc4626-preview"}),
        operation="previewDeposit",
    )
    assert deposit.status == OracleStatus.INVARIANT_VIOLATION.value
    assert "not automatically exploitable" in deposit.assumptions[0]
    conservative = preview_discrepancy(
        preview=9,
        executed=10,
        established=frozenset({"erc4626-preview"}),
        operation="previewDeposit",
    )
    assert conservative.status == OracleStatus.INCOMPLETE.value


def test_fee_on_transfer_discrepancy() -> None:
    unknown = fee_on_transfer(
        requested=100, sender_delta=-100, receiver_delta=98, accounted=100, semantics="unknown"
    )
    assert unknown.status == OracleStatus.UNKNOWN.value
    found = fee_on_transfer(
        requested=100,
        sender_delta=-100,
        receiver_delta=98,
        accounted=100,
        semantics="fee-on-transfer",
    )
    assert found.status == OracleStatus.INVARIANT_VIOLATION.value
    assert found.relation == "fee-on-transfer"


def test_oracle_to_valuation_dependency() -> None:
    bare = oracle_dependency(
        state_changed=True,
        freshness="fresh",
        design="spot",
        input_relation=False,
        valued=True,
        sensitive=True,
    )
    assert bare.status == OracleStatus.UNKNOWN.value
    linked = oracle_dependency(
        state_changed=True,
        freshness="fresh",
        design="spot",
        input_relation=True,
        valued=True,
        sensitive=True,
    )
    assert linked.relation == "spot:state->oracle->valuation"
    assert linked.status == OracleStatus.INCOMPLETE.value


def test_stale_oracle_is_not_a_current_price() -> None:
    result = oracle_dependency(
        state_changed=True,
        freshness="stale",
        design="twap",
        input_relation=True,
        valued=True,
        sensitive=True,
    )
    assert result.status == OracleStatus.INCOMPLETE.value
    assert "stale" in result.explanation


def test_amm_reserve_relation_requires_an_established_model() -> None:
    unknown = amm_invariant(reserve0=10, reserve1=10, model="pair")
    assert unknown.status == OracleStatus.UNSUPPORTED.value
    product = amm_invariant(reserve0=4, reserve1=5, model="constant-product")
    assert product.status == OracleStatus.BALANCED.value
    assert "20" in product.explanation


def test_lending_debt_matches_repayment() -> None:
    result = lending_transition(
        debt_before=10,
        debt_after=7,
        repayment=3,
        collateral_before=20,
        collateral_after=20,
        health="healthy",
        liquidated=False,
        debt_semantics="repayment-only",
    )
    assert result.status == OracleStatus.BALANCED.value
    unpaid = lending_transition(
        debt_before=10,
        debt_after=7,
        repayment=None,
        collateral_before=20,
        collateral_after=20,
        health="healthy",
        liquidated=False,
    )
    assert unpaid.status == OracleStatus.INCOMPLETE.value


def test_liquidation_state_transition() -> None:
    result = lending_transition(
        debt_before=10,
        debt_after=10,
        repayment=None,
        collateral_before=5,
        collateral_after=0,
        health="healthy",
        liquidated=True,
    )
    assert result.status == OracleStatus.INVARIANT_VIOLATION.value
    assert result.relation == "health/liquidation"


def test_economic_mutation_respects_centralized_bounds() -> None:
    call = PlannedCall(
        "deposit",
        "user",
        (ArgumentCandidate("5", "one", "uint256"),),
        "0",
        "inbound",
        (),
        1,
    )
    sequence = PlannedSequence("id", "path", (call,), "planned", (), "base", "project", "Vault")
    mutated = economic_mutations(
        sequence,
        bounds=ExplorationBounds(max_mutations=99, max_sequence_length=99),
        established=frozenset(
            {"donate", "swap", "borrow", "deposit", "withdraw", "approve", "permit"}
        ),
    )
    assert len(mutated) <= 4
    assert all(len(item.calls) <= 4 for item in mutated)
    scheduler = DiscoveryScheduler(engines=(), max_engines=2)
    assert scheduler.max_engines == 2


def test_economic_sequence_stays_target_and_project_bound() -> None:
    source = "contract Vault { function deposit(uint256 assets) external {} function nickname() external {} }"
    assert established_actions(source, "Vault", None, frozenset({"deposit"})) == ("deposit",)
    assert established_actions(source, "Vault", None, frozenset({"swap"})) == ()
    call = PlannedCall("deposit", "user", (), "0", "inbound", (), 1)
    sequence = PlannedSequence("id", "path", (call,), "planned", (), "base")
    bound = bind_sequence(sequence, project="repo", target="Vault")
    assert sequence_bound(bound, project="repo", target="Vault")
    assert not sequence_bound(bound, project="other", target="Vault")
    assert refine_actions("share_balance_changed", frozenset({"withdraw"})) == ("withdraw",)
    assert refine_actions("reserve_changed", frozenset()) == ()
    assert refine_actions("attacker_balance_increased", frozenset({"swap"})) == ("swap",)


def test_engine_selection_is_capability_aware(tmp_path: Path) -> None:
    engines = (
        _Fake("bugforge-static", frozenset({EngineCapability.STATIC_ANALYSIS})),
        _Fake("echidna", frozenset({EngineCapability.FUZZING})),
        _Fake("foundry", frozenset({EngineCapability.TEST_EXECUTION})),
        EconomicEngine(),
    )
    scheduler = DiscoveryScheduler(engines, max_engines=1)
    economic = AnalysisRequest(
        tmp_path,
        "solidity",
        target="Vault",
        extra={
            "economic": "true",
            "case": "eth",
            "before": "1",
            "after": "2",
            "token": "ETH",
            "actor": "user",
        },
    )
    runs = [item.engine_id for item in scheduler.select(economic) if item.action == "run"]
    assert runs == ["bugforge-economic"]
    plain = AnalysisRequest(
        tmp_path, "solidity", target="Vault", framework="foundry", has_harness=True
    )
    plain_runs = [item.engine_id for item in scheduler.select(plain) if item.action == "run"]
    assert plain_runs == ["bugforge-static"]
    assert "bugforge-economic" not in plain_runs
    feedback = scheduler.feedback
    feedback.exercised.append("fuzzing")
    stalled = AnalysisRequest(
        tmp_path, "solidity", target="Vault", extra={"source": "static_finding"}
    )
    assert missing_capability(stalled, feedback) == "static_analysis"
    encoded = AnalysisRequest(tmp_path, "solidity", target="Vault", extra={"property": "encoded"})
    assert missing_capability(encoded, DiscoveryScheduler(engines).feedback) == "test_execution"
    action = choose_next(
        ResearchObservation("economic", "no delta"),
        stalled=False,
        has_property=False,
        static_known=True,
    )
    assert action.capability == "economic_simulation"


def test_ityfuzz_runs_only_in_the_controlled_sandbox(tmp_path: Path, monkeypatch) -> None:
    source = Path(__file__).resolve().parents[2] / "app" / "adapters" / "discovery" / "ityfuzz.py"
    text = source.read_text(encoding="utf-8")
    assert "run_command" not in text
    assert "shell=True" not in text
    assert "subprocess" not in text
    monkeypatch.setattr(
        subprocess, "run", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("host"))
    )
    monkeypatch.setattr("app.adapters.discovery.ityfuzz._sandbox_image", lambda: "")
    assert ItyFuzzEngine().availability() is EngineAvailability.UNAVAILABLE
    monkeypatch.setattr("app.adapters.discovery.ityfuzz._sandbox_image", lambda: "ityfuzz:local")
    monkeypatch.setattr("app.adapters.discovery.ityfuzz._docker_present", lambda: True)
    monkeypatch.setattr("app.adapters.discovery.ityfuzz._local_image", lambda: True)
    from app.discovery.process import ProcessResult

    monkeypatch.setattr(
        "app.adapters.discovery.ityfuzz._execute_campaign",
        lambda *_args: ProcessResult(True, False, 0, _BANNER, "", True),
    )
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "Vault.abi").write_text("[]", encoding="utf-8")
    (tmp_path / "out" / "Vault.bin").write_text("00", encoding="utf-8")
    result = ItyFuzzEngine().start_campaign(
        AnalysisRequest(
            tmp_path, "solidity", target="Vault", campaign_id="c1", extra={"artifact_dir": "out"}
        )
    )
    assert result.metadata["environment"] == "docker-sandbox"
    assert result.to_evidence().kind is EvidenceKind.FUZZING
    assert result.metadata["verified"] == "false"
    command = ityfuzz_command()
    assert command == ["ityfuzz", "evm", "-t", "/bugforge-output/artifacts/*"]
    outside = ItyFuzzEngine().start_campaign(
        AnalysisRequest(tmp_path, "solidity", target="Vault", extra={"artifact_dir": "/etc"})
    )
    assert outside.status is ResultStatus.UNSUPPORTED
    assert outside.executed is False


def test_ityfuzz_output_is_never_fabricated() -> None:
    invented = parse_ityfuzz_output(
        '{"vulnerabilities":[{"title":"x"}],"transactions":[{"data":"0x"}]}'
    )
    assert invented.findings == ()
    assert invented.status is ResultStatus.UNSUPPORTED
    malformed = parse_ityfuzz_output("not json {oops")
    assert malformed.findings == ()
    empty = parse_ityfuzz_output("")
    assert empty.findings == ()
    assert empty.status is ResultStatus.EXECUTED
    stats = parse_ityfuzz_output(
        "[Stats #0] executions: 10\n============= Coverage Summary =============\nInstruction Covered: 12.5%\n"
    )
    assert stats.findings == ()
    assert stats.coverage["percent"] == "12.5"
    real = parse_ityfuzz_output(_BANNER)
    assert real.findings[0].description.startswith("[Fund Loss]")
    assert real.minimized == ""
    assert "[Sender]" in real.diagnostic


def test_ityfuzz_output_is_fuzzing_evidence(tmp_path: Path) -> None:
    result = ItyFuzzEngine().collect_results(
        AnalysisRequest(tmp_path, "solidity", target="Vault", extra={"ityfuzz_output": _BANNER})
    )
    evidence = result.to_evidence()
    assert evidence.kind is EvidenceKind.FUZZING
    assert evidence.kind is not EvidenceKind.REPRODUCTION
    assert evidence.metadata["verified"] == "false"
    assert result.metadata["authority"] == "interesting"


def _replay(tmp_path: Path):
    reset_syntax_registry()
    reset_plugin_catalog()
    path = tmp_path / "Vault.sol"
    path.write_text(_FAILING, encoding="utf-8")
    program = build_semantic_program(parse_source("solidity", path, _FAILING))
    model = analyze_state_transitions(program)
    candidate = next(item for item in model.paths if item.path_id.startswith("accounting:"))
    spec = specify(candidate, model, program)
    return candidate, spec, _FAILING


def test_sandboxed_foundry_can_reach_reproduced(tmp_path: Path, monkeypatch) -> None:
    candidate, spec, source = _replay(tmp_path)
    _sequence, artifact = prepare_replay(candidate, spec, source)

    def execute(harness: str, planned, source_id: str) -> tuple[int, str, str]:
        assert "fork-url" not in harness
        return 1, f"Suite result: FAILED\n[FAIL] {planned.fail_token}\n", ""

    monkeypatch.setattr("app.parsing.solidity_replay.sandbox_requirement", lambda: "")
    monkeypatch.setattr("app.parsing.solidity_replay._execute", execute)
    result = run_replay(candidate, spec, source)
    assert result.execution == "forge-sandbox"
    assert result.status == "reproduced"
    assert replay_evidence(result).kind is EvidenceKind.REPRODUCTION
    assert (
        promote("candidate_violation", execution="simulated", environment="real-target", bound=True)
        == "candidate_violation"
    )
    assert (
        promote("candidate_violation", execution="sandbox", environment="real-target", bound=True)
        == "candidate_violation"
    )


def test_simulated_execution_cannot_become_reproduced(tmp_path: Path) -> None:
    candidate, spec, source = _replay(tmp_path)
    _sequence, artifact = prepare_replay(candidate, spec, source)
    result = run_replay(
        candidate,
        spec,
        source,
        runner=lambda _harness: (1, f"Suite result: FAILED\n[FAIL] {artifact.fail_token}\n", ""),
    )
    assert result.execution == "simulated"
    assert result.status == "candidate_violation"
    assert replay_evidence(result).kind is not EvidenceKind.REPRODUCTION


def test_unbound_inputs_never_reproduce(tmp_path: Path, monkeypatch) -> None:
    candidate, spec, source = _replay(tmp_path)
    sequence, artifact = prepare_replay(candidate, spec, source)

    def refused(*_args, **_kwargs):
        raise AssertionError("unbound replay executed")

    monkeypatch.setattr("app.parsing.solidity_replay._execute", refused)
    cases = (
        replace(artifact, harness=artifact.harness + "\n"),
        replace(artifact, dependency_id="wrong"),
        replace(artifact, compiler_config="wrong"),
        artifact,
    )
    sequences = (
        sequence,
        sequence,
        sequence,
        replace(sequence, sequence_id=sequence.sequence_id + "-edited"),
    )
    for stale, planned in zip(cases, sequences, strict=True):
        monkeypatch.setattr(
            "app.parsing.solidity_replay.prepare_replay",
            lambda *_a, _stale=stale, _planned=planned, **_k: (_planned, _stale),
        )
        result = run_replay(candidate, spec, source)
        assert result.status != "reproduced"
        assert result.execution == "not_attempted"


def test_corpus_binding_includes_compiler_and_engine() -> None:
    corpus = DiscoveryCorpus()
    corpus.add(
        "seed",
        source=SeedSource.ITYFUZZ,
        reason="campaign",
        target="Vault",
        project="repo",
        source_snapshot="snap",
        compiler_configuration="0.8.20",
        engine="ityfuzz",
        engine_version="ityfuzz-stdout-v1",
        campaign="one",
    )
    corpus.add(
        "seed",
        source=SeedSource.ITYFUZZ,
        reason="other compiler",
        target="Vault",
        project="repo",
        source_snapshot="snap",
        compiler_configuration="0.8.19",
        engine="ityfuzz",
        engine_version="ityfuzz-stdout-v1",
        campaign="one",
    )
    assert len(corpus.seeds) == 2
    assert (
        corpus.for_binding(
            project="repo",
            target="Vault",
            source_snapshot="snap",
            compiler_configuration="0.8.20",
            engine="ityfuzz",
            engine_version="other",
        )
        == ()
    )
    matched = corpus.for_binding(
        project="repo",
        target="Vault",
        source_snapshot="snap",
        compiler_configuration="0.8.20",
        engine="ityfuzz",
        engine_version="ityfuzz-stdout-v1",
    )
    assert len(matched) == 1
    assert corpus.for_binding(project="repo", target="Other", source_snapshot="snap") == ()


def test_no_host_execution_or_network_in_economic_analysis() -> None:
    root = Path(__file__).resolve().parents[2] / "app"
    for relative in ("parsing/solidity_economics.py", "adapters/discovery/economic.py"):
        text = (root / relative).read_text(encoding="utf-8")
        assert "subprocess" not in text
        assert "socket" not in text
        assert "requests" not in text
        assert "httpx" not in text
        assert "openai" not in text
        assert "anthropic" not in text
        assert "xai" not in text
        assert "grok" not in text
    cursor = (root / "cursor_control" / "analysis.py").read_text(encoding="utf-8")
    assert '"llm_invoked": False' in cursor


def test_economic_evidence_is_bound_and_not_a_conclusion() -> None:
    assert canonical_status("exploited") == OracleStatus.UNKNOWN.value
    evidence = EconomicEvidence(
        "repo",
        "Vault",
        "seq",
        (0, 1),
        ("attacker",),
        ("USDC",),
        "before",
        "after",
        (),
        "exact_local",
        "",
        "assets/shares",
        "exploited",
        ("not confirmation",),
        ("Vault.sol:1",),
        "bugforge-economic",
        "phase46",
        "in-process",
        "snap",
        "0.8.20",
    )
    assert evidence.status == OracleStatus.UNKNOWN.value
    linked = correlate_event("Transfer", established=frozenset({"balance"}))
    assert linked.relation == "event->balance->shares"
    assert correlate_event("Swap", established=frozenset()).status == OracleStatus.UNSUPPORTED.value
    assert all(item.local_only for item in ECONOMIC_BENCHMARKS)
    assert load_local(Path("/tmp/bugforge-missing-benchmarks")) == ()


class _Fake(DiscoveryEngine):
    def __init__(self, engine_id: str, caps: frozenset[EngineCapability]) -> None:
        self._id = engine_id
        self._caps = caps

    @property
    def engine_id(self) -> str:
        return self._id

    @property
    def display_name(self) -> str:
        return self._id

    def capabilities(self) -> frozenset[EngineCapability]:
        return self._caps

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
