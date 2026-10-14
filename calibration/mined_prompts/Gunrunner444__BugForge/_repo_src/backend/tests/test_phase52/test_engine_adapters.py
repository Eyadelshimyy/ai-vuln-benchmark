"""Phase 52 Slice B: second-engine property adapters and the honest engine registry."""

from __future__ import annotations

import json
from typing import Any

import pytest

from app.discovery.bounty import engine_adapters as ea
from app.discovery.bounty.engine_adapters import (
    FLAG_RAN,
    FLAG_UNJUDGED,
    FLAG_VIOLATED,
    EngineRun,
    EngineStatus,
    PropertyEngine,
    engine_harness_source,
    engine_registry,
    parse_echidna,
    parse_medusa,
    usable_property_engines,
    verdict_from_flags,
)
from app.discovery.bounty.properties import (
    OracleKind,
    OracleSpec,
    PropertyEvaluation,
    PropertyVerdict,
    build_property,
)
from app.discovery.bounty.stateful import (
    IdentityContext,
    Outcome,
    StatefulExecutor,
    ToolStatus,
    _Base,
    bind_identity,
    build_harness,
    independent_check,
)
from tests.test_phase52.phase52_support import (
    DELTA_VAULT,
    RAW_VAULT,
    SAFE_INIT,
    VULNERABLE_INIT,
    deposit_sequence,
    init_sequence,
    model_of,
    requires_engine,
)

NO_TOOLS = ToolStatus("unavailable", "unavailable")


def _init_harness(text: str = VULNERABLE_INIT):  # type: ignore[no-untyped-def]
    sources = {"Acct.sol": text}
    model = model_of(sources)
    seq = init_sequence()
    spec = build_property(seq, model)
    return build_harness(seq, model, oracle=spec.oracle, sources=sources), seq, sources, model, spec


# ---- in-EVM judge generation ------------------------------------------------------------------


def test_engine_harness_hosts_the_same_plan_with_an_independent_judge() -> None:
    harness, seq, _, _, _ = _init_harness()
    source = engine_harness_source(harness, seq)
    assert "contract BugForgeEngineHarness" in source
    assert "BugForgeObservation" not in source  # observations become storage, not events
    assert "forge-std" not in source and 'import "./src/Acct.sol";' in source
    # the planned calls are carried over verbatim
    assert 'abi.encodeWithSignature("initialize(address)", address(0xA11CE))' in source
    # second_call_must_fail: precondition call 0 must succeed, probe is call 1
    assert "if (!_bfOk(0)) return false;" in source and "return _bfOk(1);" in source
    for flag in (FLAG_RAN, FLAG_UNJUDGED, FLAG_VIOLATED):
        assert f"function {flag}()" in source


def test_credit_oracles_get_overflow_safe_signed_delta_judges() -> None:
    sources = {"Vault.sol": RAW_VAULT}
    model = model_of(sources)
    seq = deposit_sequence()
    spec = build_property(seq, model)
    assert spec.oracle.kind is OracleKind.CREDIT_LE_RECEIVED
    harness = build_harness(seq, model, oracle=spec.oracle, sources=sources)
    source = engine_harness_source(harness, seq)
    assert "_bfGreaterDelta(bfVal[7][0], bfVal[7][1], bfVal[6][0], bfVal[6][1])" in source
    assert "contract BugForgeFixtureToken" in source
    holdings = next(a for a in spec.alternates if a.kind is OracleKind.CREDIT_LE_HOLDINGS)
    alt = build_harness(seq, model, oracle=holdings, sources=sources)
    assert "return bfVal[7][1] > bfVal[6][1];" in engine_harness_source(alt, seq)


def test_non_executable_or_unbuildable_plans_get_no_engine_harness() -> None:
    harness, seq, sources, model, _ = _init_harness()
    none = build_harness(seq, model, oracle=OracleSpec(), sources=sources)
    assert engine_harness_source(none, seq) == ""
    lost = OracleSpec(OracleKind.CALL_MUST_FAIL, probe_role="nope", probe_function="x()")
    assert engine_harness_source(build_harness(seq, model, oracle=lost, sources=sources), seq) == ""


# ---- structured verdicts ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("flags", "outcome"),
    [
        ({FLAG_RAN: "failed", FLAG_UNJUDGED: "passed", FLAG_VIOLATED: "failed"}, "violated"),
        ({FLAG_RAN: "failed", FLAG_UNJUDGED: "passed", FLAG_VIOLATED: "passed"}, "held"),
        ({FLAG_RAN: "failed", FLAG_UNJUDGED: "failed", FLAG_VIOLATED: "passed"}, "inconclusive"),
        ({FLAG_RAN: "passed", FLAG_UNJUDGED: "passed", FLAG_VIOLATED: "passed"}, "inconclusive"),
        ({FLAG_RAN: "failed", FLAG_VIOLATED: "failed"}, "inconclusive"),  # a flag is missing
        ({}, "inconclusive"),
    ],
)
def test_flags_map_to_structured_verdicts_and_missing_never_passes(
    flags: dict[str, str], outcome: str
) -> None:
    result, verdict, _ = verdict_from_flags(flags)
    expected = {
        "violated": Outcome.PROPERTY_VIOLATED,
        "held": Outcome.PROPERTY_HELD,
        "inconclusive": Outcome.INCONCLUSIVE,
    }[outcome]
    assert result is expected
    if outcome == "inconclusive":
        assert verdict == PropertyVerdict.NOT_EVALUATED.value


def test_engine_reports_are_parsed_by_exact_flag_names() -> None:
    echidna = (
        "echidna_bf_flag_ran: failed!\U0001f4a5\n  Call sequence:\n"
        "echidna_bf_flag_unjudged: passing\necho_unrelated: failed!\n"
        "echidna_bf_flag_violated: failed!\n"
    )
    assert parse_echidna(echidna) == {
        FLAG_RAN: "failed",
        FLAG_UNJUDGED: "passed",
        FLAG_VIOLATED: "failed",
    }
    medusa = (
        "\x1b[1m[FAILED]\x1b[0m Property Test: BugForgeEngineHarness.echidna_bf_flag_ran()\n"
        "[PASSED] Property Test: BugForgeEngineHarness.echidna_bf_flag_unjudged()\n"
        "[NOT STARTED] Property Test: BugForgeEngineHarness.echidna_bf_flag_violated()\n"
    )
    assert parse_medusa(medusa) == {FLAG_RAN: "failed", FLAG_UNJUDGED: "passed"}


def test_engine_configs_are_server_owned_and_offline() -> None:
    medusa = json.loads(ea.medusa_config())
    fork = medusa["fuzzing"]["chainConfig"]["forkConfig"]
    assert fork == {"forkModeEnabled": False, "rpcUrl": "", "rpcBlock": 1}
    assert medusa["fuzzing"]["chainConfig"]["cheatCodes"]["enableFFI"] is False
    assert medusa["fuzzing"]["corpusDirectory"] == ""
    echidna = ea.echidna_config()
    assert "rpc" not in echidna.lower() and "allowFFI: false" in echidna
    assert "--disable-onchain-sources" in ea._specs()["echidna"].argv


# ---- registry ---------------------------------------------------------------------------------


def test_registry_without_tools_is_honest(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ea, "tool_path", lambda name: None)
    monkeypatch.setattr(ea, "tool_status", lambda: NO_TOOLS)
    entries = {e.name: e for e in engine_registry()}
    assert entries["bugforge-static"].status is EngineStatus.USABLE
    assert entries["foundry"].status is EngineStatus.UNAVAILABLE
    assert entries["echidna"].status is EngineStatus.UNAVAILABLE
    assert entries["medusa"].status is EngineStatus.UNAVAILABLE
    assert entries["ityfuzz"].status is EngineStatus.UNAVAILABLE
    assert entries["fork_replay"].status is EngineStatus.BLOCKED_BY_POLICY
    assert usable_property_engines(tuple(entries.values())) == ()


def test_a_binary_alone_is_installed_not_usable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ea, "tool_path", lambda name: f"/opt/{name}")
    monkeypatch.setattr(ea, "_engine_version", lambda name: "9.9.9")
    monkeypatch.setattr(ea, "tool_status", lambda: ToolStatus("available", "available"))
    monkeypatch.setattr(ea, "execution_enabled", lambda: True)
    monkeypatch.setattr(
        ea,
        "smoke_check",
        lambda name: ea.SmokeResult(name, name == "echidna", "smoke", (("vulnerable", "x"),)),
    )
    entries = {e.name: e for e in engine_registry()}
    assert entries["echidna"].status is EngineStatus.USABLE
    assert entries["medusa"].status is EngineStatus.INSTALLED
    # an installed tool without a campaign adapter is unsupported, never usable
    assert entries["ityfuzz"].status is EngineStatus.UNSUPPORTED
    assert entries["halmos"].status is EngineStatus.UNSUPPORTED
    assert usable_property_engines(tuple(entries.values())) == ("echidna",)
    assert {e.name: e.status for e in engine_registry(smoke=False)}["echidna"] is (
        EngineStatus.INSTALLED
    )


def test_disabled_execution_is_blocked_by_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ea, "execution_enabled", lambda: False)
    entries = {e.name: e for e in engine_registry(smoke=False)}
    for name in ("foundry", "echidna", "medusa"):
        assert entries[name].status is EngineStatus.BLOCKED_BY_POLICY


# ---- independent check wiring -----------------------------------------------------------------


class _FakeEngine:
    def __init__(self, name: str, outcome: Outcome, verdict: str) -> None:
        self.name = name
        self.outcome = outcome
        self.verdict = verdict
        self.calls = 0

    def run(self, harness: Any, sequence: Any, sources: Any) -> EngineRun:
        self.calls += 1
        return EngineRun(self.name, "completed", self.outcome, self.verdict, "fake")


def _violated_primary():  # type: ignore[no-untyped-def]
    harness, seq, sources, model, spec = _init_harness()
    identity = bind_identity(seq, model, sources, IdentityContext(), property_id=spec.property_id)
    base = _Base(seq, spec, spec.oracle, "default", identity, NO_TOOLS)
    evaluation = PropertyEvaluation(PropertyVerdict.PROPERTY_VIOLATED, "x", 1, True, True)
    primary = base.done(Outcome.PROPERTY_VIOLATED, "x", harness=harness, evaluation=evaluation)
    return primary, seq, sources, model, spec


def _check(engines: list[Any], spec_alternates: bool = False):  # type: ignore[no-untyped-def]
    primary, seq, sources, model, spec = _violated_primary()
    from dataclasses import replace

    spec = replace(spec, alternates=()) if not spec_alternates else spec
    executor = StatefulExecutor(tools=NO_TOOLS)
    return independent_check(executor, seq, model, sources, spec, primary, property_engines=engines)


def test_engine_agreement_is_material_corroboration_never_verification() -> None:
    engine = _FakeEngine("echidna", Outcome.PROPERTY_VIOLATED, "property_violated")
    corroboration, _ = _check([engine])
    assert engine.calls == 1
    assert corroboration.status == "corroborated_candidate"
    assert corroboration.independence == "second_engine"
    assert "echidna:property" in corroboration.agreeing
    path = next(p for p in corroboration.paths if p.path == "echidna:property")
    assert path.material and path.engine == "echidna"
    assert corroboration.engine_runs and corroboration.as_dict()["verified"] is False
    # medusa was not passed in: recorded, not run, never counted
    medusa = next(p for p in corroboration.paths if p.path == "medusa:property")
    assert medusa.outcome == "unavailable" and medusa.verdict == "not_evaluated"


def test_an_engine_that_holds_the_property_makes_the_contradiction_visible() -> None:
    engine = _FakeEngine("medusa", Outcome.PROPERTY_HELD, "property_held")
    corroboration, _ = _check([engine])
    assert corroboration.status == "disagreement"


def test_an_inconclusive_engine_does_not_corroborate() -> None:
    engine = _FakeEngine("echidna", Outcome.EXECUTION_FAILED, "not_evaluated")
    corroboration, _ = _check([engine])
    assert corroboration.status == "single_path"


# ---- real engines (skipped when absent) -------------------------------------------------------


@pytest.mark.parametrize("name", ["echidna", "medusa"])
def test_real_engine_judges_violation_and_held(name: str) -> None:
    if not all(ea.tool_path(t) for t in ("solc", "crytic-compile", name)):
        pytest.skip(f"{name} not installed")
    engine = PropertyEngine(name)
    harness, seq, sources, _, _ = _init_harness(VULNERABLE_INIT)
    run = engine.run(harness, seq, sources)
    assert run.outcome is Outcome.PROPERTY_VIOLATED, run.output_tail
    assert dict(run.flags)[FLAG_RAN] == "failed" and run.verified is False
    assert run.harness_hash and run.config_hash and run.version
    harness, seq, sources, _, _ = _init_harness(SAFE_INIT)
    assert engine.run(harness, seq, sources).outcome is Outcome.PROPERTY_HELD


@requires_engine("echidna")
def test_echidna_judges_credit_oracles_on_fixture_tokens() -> None:
    engine = PropertyEngine("echidna")
    for text, want in (
        (RAW_VAULT, Outcome.PROPERTY_VIOLATED),
        (DELTA_VAULT, Outcome.PROPERTY_HELD),
    ):
        sources = {"Vault.sol": text}
        model = model_of(sources)
        seq = deposit_sequence()
        spec = build_property(seq, model)
        harness = build_harness(seq, model, oracle=spec.oracle, sources=sources)
        run = engine.run(harness, seq, sources)
        assert run.outcome is want, run.output_tail


@requires_engine("echidna")
def test_smoke_check_marks_echidna_usable() -> None:
    result = ea.smoke_check("echidna")
    assert result.usable, result.reason
    assert dict(result.observed) == {"vulnerable": "property_violated", "safe": "property_held"}
