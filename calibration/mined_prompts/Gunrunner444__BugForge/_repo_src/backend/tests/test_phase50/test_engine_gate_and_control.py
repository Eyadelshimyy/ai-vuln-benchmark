"""Research engines, the program gate, planner integration, and Cursor control."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from unittest.mock import patch

import pytest

import app.discovery.bounty as bounty_package
from app.discovery.bounty.campaign import BountyManifest
from app.discovery.bounty.compiler_diff import NoCompilerBackend
from app.discovery.bounty.engine import (
    BugforgeCompilerDifferentialEngine,
    BugforgeResearchEngine,
    build_bounty_engines,
    load_sources,
)
from app.discovery.bounty.gate import BountyGate
from app.discovery.capabilities import EngineAvailability, EngineCapability, ResultStatus
from app.discovery.engine import AnalysisRequest
from app.discovery.orchestration import MemoryStore, Orchestrator, StopReason, Suggestion
from app.discovery.orchestration.model import CampaignIdentity, Candidate, ResearchState
from app.discovery.orchestration.planner import (
    CAPABILITY_MODE,
    ORCHESTRATABLE,
    prerequisite,
    validate_suggestions,
)
from tests.test_phase49.phase49_support import make_scheduler
from tests.test_phase50.phase50_support import FIXTURES, manifest, request_for

CALLER = "caller_context_vulnerable.sol"
PACKAGE = Path(bounty_package.__file__).parent
NEW = (
    "bounty_context_analysis",
    "caller_context_analysis",
    "oracle_quality_analysis",
    "proof_binding_analysis",
    "account_abstraction_analysis",
    "accounting_analysis",
    "compiler_advisory_analysis",
    "compiler_differential_validation",
    "vfcs_generation",
)


def _engine_request(mani: BountyManifest, capability: str, files=(CALLER,), **extra: str):
    request, identity = request_for(
        FIXTURES, mani, files=files, contract="SelfTrustMulticall", mode=capability, **extra
    )
    return request, identity


# ---- capabilities and engines ---------------------------------------------------------------


def test_new_capabilities_are_orchestratable_with_an_explicit_mode() -> None:
    for name in NEW:
        assert EngineCapability(name)
        assert name in ORCHESTRATABLE
        assert CAPABILITY_MODE[name] == name


def test_research_engine_runs_each_capability_without_verifying() -> None:
    mani = manifest()
    engine = BugforgeResearchEngine(mani)
    assert engine.availability() is EngineAvailability.AVAILABLE
    assert EngineCapability.STATIC_ANALYSIS not in engine.capabilities()
    for name in NEW:
        if name == "compiler_differential_validation":
            continue
        request, _ = _engine_request(mani, name)
        assert engine.selected_capability(request).value == name
        result = engine.start_campaign(request)
        assert result.executed and result.status is ResultStatus.INGESTED, name
        assert result.metadata["capability"] == name
        assert result.metadata["verified"] == "false"
        assert result.metadata["program_context"] == mani.identity_digest()
        assert all(f.status == "potential" for f in result.findings)


def test_caller_context_capability_reports_the_candidates() -> None:
    mani = manifest()
    request, _ = _engine_request(mani, "caller_context_analysis")
    result = BugforgeResearchEngine(mani).start_campaign(request)
    assert {f.detector_id for f in result.findings} == {
        "caller_context.self_call_elevation",
        "caller_context.unrestricted_dispatch_with_approval_authority",
        "caller_context.trusted_intermediary_actor_parameter",
        "caller_context.forwarded_sender_delegatecall",
    }


def test_inapplicable_family_is_reported_as_not_applicable_not_clean() -> None:
    mani = manifest()
    request, _ = _engine_request(mani, "account_abstraction_analysis", files=("no_aa.sol",))
    result = BugforgeResearchEngine(mani).start_campaign(request)
    assert result.executed and result.findings == ()
    assert result.metadata["applicable"] == "false"
    assert "account_abstraction" in result.metadata["families_skipped"]


def test_engine_refuses_a_foreign_program_context() -> None:
    mani = manifest()
    request, _ = _engine_request(mani, "caller_context_analysis", program_context="pc_other")
    result = BugforgeResearchEngine(mani).start_campaign(request)
    assert not result.executed and result.findings == ()
    assert result.status is ResultStatus.UNSUPPORTED


def test_engine_does_not_fabricate_results_without_sources(tmp_path) -> None:
    mani = manifest()
    request, _ = request_for(
        tmp_path, mani, files=("missing.sol",), contract="X", mode="caller_context_analysis"
    )
    result = BugforgeResearchEngine(mani).start_campaign(request)
    assert not result.executed and result.findings == ()


def test_source_loading_stays_under_the_repository_root(tmp_path) -> None:
    (tmp_path / "ok.sol").write_text("contract A {}")
    outside = tmp_path.parent / "outside.sol"
    outside.write_text("contract B {}")
    request = AnalysisRequest(
        repo_root=tmp_path, language="solidity", files=("ok.sol", "../outside.sol", "/etc/passwd")
    )
    assert set(load_sources(request)) == {"ok.sol"}


def test_vfcs_capability_exposes_a_sequence_bound_to_the_campaign() -> None:
    mani = manifest()
    request, _ = _engine_request(mani, "vfcs_generation")
    result = BugforgeResearchEngine(mani).start_campaign(request)
    ids = json.loads(result.metadata["vfcs_ids"])
    assert ids and result.metadata["sequence_id"] == ids[0]
    assert result.metadata["vfcs_count"] == str(len(ids))


def test_compiler_differential_engine_is_unavailable_without_a_compiler() -> None:
    mani = manifest()
    engine = BugforgeCompilerDifferentialEngine(mani, NoCompilerBackend())
    assert engine.availability() is EngineAvailability.UNAVAILABLE
    request, _ = _engine_request(mani, "compiler_differential_validation")
    result = engine.start_campaign(request)
    assert result.status is ResultStatus.UNAVAILABLE and not result.executed
    assert result.findings == ()


def test_default_engines_do_not_pretend_a_compiler_exists() -> None:
    engines = {e.engine_id: e for e in build_bounty_engines(manifest())}
    assert set(engines) == {"bugforge-research", "bugforge-compiler-diff"}
    assert engines["bugforge-compiler-diff"].availability() is EngineAvailability.UNAVAILABLE


# ---- planner prerequisites and suggestions --------------------------------------------------


def test_prerequisites_name_what_is_missing() -> None:
    mani = manifest()
    identity = mani.to_campaign_identity()
    bare = AnalysisRequest(
        repo_root=Path("."), language="solidity", extra={"program_context": "pc"}
    )
    assert prerequisite("caller_context_analysis", bare, identity) == "identity:target"
    with_files = AnalysisRequest(
        repo_root=Path("."), language="solidity", files=("a.sol",), extra={}
    )
    assert (
        prerequisite("caller_context_analysis", with_files, identity) == "identity:program_context"
    )
    ok = AnalysisRequest(
        repo_root=Path("."), language="solidity", files=("a.sol",), extra={"program_context": "pc"}
    )
    assert prerequisite("caller_context_analysis", ok, identity) == ""
    empty = CampaignIdentity("c")
    assert prerequisite("compiler_differential_validation", ok, empty) == "identity:source_snapshot"


def test_cursor_suggestions_are_validated_not_trusted() -> None:
    accepted, rejected = validate_suggestions(
        (
            Suggestion("caller_context_analysis"),
            Suggestion("submit_report"),
            Suggestion("vfcs_generation", engine="unregistered"),
        ),
        frozenset({"bugforge-research"}),
    )
    assert [s.capability for s in accepted] == ["caller_context_analysis"]
    assert len(rejected) == 2


# ---- gate -----------------------------------------------------------------------------------


def _candidate(capability: str) -> Candidate:
    return Candidate(engine="e", capability=capability, eligible=True)


def _check(gate: BountyGate, request: AnalysisRequest, capability: str):
    state = ResearchState(identity=CampaignIdentity("c"))
    return gate.check(candidate=_candidate(capability), request=request, state=state)


def test_gate_requires_the_program_context() -> None:
    mani = manifest()
    request, _ = _engine_request(mani, "caller_context_analysis")
    assert _check(BountyGate(mani), request, "caller_context_analysis").allowed
    foreign = AnalysisRequest(
        **{**request.__dict__, "extra": {**request.extra, "program_context": "pc_x"}}
    )
    verdict = _check(BountyGate(mani), foreign, "caller_context_analysis")
    assert not verdict.allowed and verdict.stop is StopReason.SAFETY_BLOCKED
    missing = AnalysisRequest(**{**request.__dict__, "extra": {}})
    assert not _check(BountyGate(mani), missing, "caller_context_analysis").allowed


def test_out_of_scope_target_is_blocked() -> None:
    mani = manifest()
    request, _ = request_for(FIXTURES, mani, files=(CALLER,), contract="BalanceAsDeposit")
    verdict = _check(BountyGate(mani), request, "caller_context_analysis")
    assert not verdict.allowed and verdict.stop is StopReason.SCOPE_BLOCKED


def test_unknown_scope_allows_local_analysis_but_never_fork_testing() -> None:
    mani = manifest()
    request, _ = request_for(
        FIXTURES,
        mani,
        files=("x.sol",),
        contract="Unlisted",
        mode="fork",
        **{"chain_id": "1", "fork_block": "19000000", "state_snapshot": "snap-19000000"},
    )
    gate = BountyGate(mani, approvals=frozenset({"fork_validation"}))
    assert _check(gate, request, "caller_context_analysis").allowed
    verdict = _check(gate, request, "fork_validation")
    assert not verdict.allowed and verdict.stop is StopReason.SCOPE_BLOCKED


def test_fork_needs_scope_testing_mode_pinned_identity_and_approval() -> None:
    mani = manifest()
    request, _ = request_for(FIXTURES, mani, files=(CALLER,), contract="SelfTrustMulticall")
    unapproved = _check(BountyGate(mani), request, "fork_validation")
    assert unapproved.stop is StopReason.APPROVAL_REQUIRED
    approved = BountyGate(mani, approvals=frozenset({"fork_validation"}))
    assert _check(approved, request, "fork_validation").allowed
    drifted = AnalysisRequest(
        **{**request.__dict__, "extra": {**request.extra, "fork_block": "19000001"}}
    )
    assert _check(approved, drifted, "fork_validation").stop is StopReason.SAFETY_BLOCKED
    local = manifest(testing_mode="local_only")
    request_local, _ = request_for(FIXTURES, local, files=(CALLER,), contract="SelfTrustMulticall")
    blocked = _check(
        BountyGate(local, approvals=frozenset({"fork_validation"})),
        request_local,
        "fork_validation",
    )
    assert not blocked.allowed and blocked.stop is StopReason.SAFETY_BLOCKED
    unpinned = manifest(fork=None, testing_mode="fork_allowed")
    request_unpinned, _ = request_for(
        FIXTURES, unpinned, files=(CALLER,), contract="SelfTrustMulticall"
    )
    assert not _check(
        BountyGate(unpinned, approvals=frozenset({"fork_validation"})),
        request_unpinned,
        "fork_validation",
    ).allowed


def test_a_scope_claim_in_the_request_grants_nothing() -> None:
    mani = manifest()
    request, _ = request_for(
        FIXTURES,
        mani,
        files=("x.sol",),
        contract="Unlisted",
        scope="in_scope",
        chain_id="1",
        fork_block="19000000",
        state_snapshot="snap-19000000",
    )
    gate = BountyGate(mani, approvals=frozenset({"fork_validation"}))
    assert not _check(gate, request, "fork_validation").allowed
    blocked, _ = request_for(
        FIXTURES, mani, files=(CALLER,), contract="SelfTrustMulticall", scope="out_of_scope"
    )
    assert not _check(gate, blocked, "caller_context_analysis").allowed


def test_network_outside_fork_validation_is_refused() -> None:
    mani = manifest()
    request, _ = request_for(
        FIXTURES, mani, files=(CALLER,), contract="SelfTrustMulticall", network="mainnet"
    )
    verdict = _check(BountyGate(mani), request, "caller_context_analysis")
    assert not verdict.allowed and verdict.stop is StopReason.SAFETY_BLOCKED


# ---- Cursor control -------------------------------------------------------------------------


def test_cursor_cannot_bypass_the_gate_or_the_budget(tmp_path) -> None:
    mani = manifest()
    request, identity = request_for(FIXTURES, mani, files=(CALLER,), contract="BalanceAsDeposit")
    engines = list(build_bounty_engines(mani))
    scheduler = make_scheduler(engines, max_engines=2, max_rounds=2)
    orch = Orchestrator(
        scheduler,
        request,
        identity=mani.to_campaign_identity(contract="BalanceAsDeposit"),
        gate=BountyGate(mani),
        store=MemoryStore(),
    )
    for name in NEW:
        orch.suggest(name)
    orch.suggest("fork_validation")
    state = orch.run()
    assert state.stop_reason == StopReason.SCOPE_BLOCKED.value
    assert not any(e.phase.value == "completed" for e in state.executions)
    assert state.budget.limits["engines"] == 2 and state.budget.limits["rounds"] <= 2


def test_no_model_is_called_and_nothing_is_submitted() -> None:
    from app.ai.anthropic_provider import AnthropicProvider
    from app.ai.openai_provider import OpenAIProvider

    mani = manifest()
    request, identity = _engine_request(mani, "caller_context_analysis")
    scheduler = make_scheduler(list(build_bounty_engines(mani)), max_engines=4, max_rounds=4)
    with (
        patch.object(OpenAIProvider, "complete", side_effect=AssertionError("openai")),
        patch.object(AnthropicProvider, "complete", side_effect=AssertionError("anthropic")),
    ):
        state = Orchestrator(
            scheduler, request, identity=identity, gate=BountyGate(mani), store=MemoryStore()
        ).run()
    assert state.executions
    assert all(item.attrs.get("verified") != "true" for item in state.evidence)


def test_bounty_package_has_no_network_model_or_submission_code() -> None:
    forbidden = {
        "requests",
        "httpx",
        "aiohttp",
        "urllib3",
        "socket",
        "ssl",
        "http",
        "asyncio",
        "openai",
        "anthropic",
        "litellm",
        "langchain",
        "app.ai",
        "ftplib",
        "smtplib",
    }
    for path in PACKAGE.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                assert not any(name == bad or name.startswith(f"{bad}.") for bad in forbidden), (
                    path.name,
                    name,
                )
                if name == "subprocess" or name.startswith("subprocess."):
                    # compiler_diff.py shells out to solc; stateful.py (Phase 52)
                    # shells out to forge for local stateful execution;
                    # engine_adapters.py runs echidna/medusa offline on the same plan;
                    # solc_resolve.py asks a local solc binary for --version and --help.
                    assert path.name in {
                        "compiler_diff.py",
                        "stateful.py",
                        "engine_adapters.py",
                        "solc_resolve.py",
                    }
    # the manifest legitimately names credential words in order to refuse them
    text = "\n".join(
        p.read_text(encoding="utf-8") for p in PACKAGE.glob("*.py") if p.name != "campaign.py"
    ).lower()
    for needle in (
        "api_key",
        "apikey",
        "bearer",
        "mnemonic",
        "chat/completions",
        "hackerone.com",
        "immunefi.com",
        "requests.post",
        "urlopen",
        "pip install",
        "curl ",
    ):
        assert needle not in text, needle
    for path in PACKAGE.glob("*.py"):
        body = path.read_text(encoding="utf-8")
        assert "verified=True" not in body and '"verified": "true"' not in body
        assert "def submit" not in body and "def auto_submit" not in body


@pytest.mark.parametrize("name", ["solidity_advisories.json"])
def test_corpus_file_ships_with_the_package(name: str) -> None:
    assert (PACKAGE / "data" / name).is_file()
