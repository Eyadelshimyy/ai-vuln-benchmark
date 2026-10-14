"""Phase 52 hardening, Slice A: real local execution with forge (skipped when absent)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.discovery.bounty.properties import build_property
from app.discovery.bounty.stateful import (
    BundleIntegrityError,
    IdentityContext,
    Outcome,
    StatefulExecutor,
    build_bundle,
    independent_check,
    materialize_bundle,
    replay_bundle,
    run_feedback_loop,
    sha256_text,
)
from app.discovery.bounty.vfcs import stamp_source_identity
from tests.test_phase52.phase52_support import (
    DELTA_VAULT,
    RAW_VAULT,
    SAFE_INIT,
    VULNERABLE_INIT,
    deposit_sequence,
    init_sequence,
    model_of,
    requires_forge,
)

pytestmark = requires_forge


def _run(sources: dict[str, str], seq, **kwargs):  # type: ignore[no-untyped-def]
    model = model_of(sources)
    seq = stamp_source_identity(seq, sources)
    spec = build_property(seq, model)
    executor = StatefulExecutor()
    return seq, executor, model, spec, executor.execute(seq, model, sources, spec=spec, **kwargs)


def test_unguarded_initializer_violates_its_property_with_structured_result() -> None:
    sources = {"Acct.sol": VULNERABLE_INIT}
    _seq, executor, model, spec, obs = _run(sources, init_sequence())
    assert obs.outcome is Outcome.PROPERTY_VIOLATED, obs.reason
    result = obs.result
    assert result is not None
    assert result.process_status == "completed" and result.compiler_status == "ok"
    assert result.test_status == "passed" and result.property_status == "violated"
    assert result.call_results == (1, 1)
    assert result.assertion_id == spec.property_id
    assert dict(result.source_hashes) == {"Acct.sol": sha256_text(VULNERABLE_INIT)}
    assert result.harness_hash == sha256_text(obs.harness.source)  # type: ignore[union-attr]
    assert result.solc_version and result.forge_version
    assert obs.identity is not None and obs.identity.bound
    assert obs.identity.replay_mode.value == "local_source_replay"
    assert obs.strong_candidate and obs.verified is False


def test_guarded_initializer_holds() -> None:
    _seq, _executor, _model, _spec, obs = _run({"Acct.sol": SAFE_INIT}, init_sequence())
    assert obs.outcome is Outcome.PROPERTY_HELD, obs.reason
    assert obs.reverted_index == 1  # the revert is the property holding, not a finding
    assert not obs.strong_candidate


def test_fee_on_transfer_credit_violation_and_safe_delta_accounting() -> None:
    _seq, _e, _m, _s, bad = _run({"V.sol": RAW_VAULT}, deposit_sequence())
    assert bad.outcome is Outcome.PROPERTY_VIOLATED, bad.reason
    assert "exceeds received" in bad.reason
    assert any(p.name == "test_deploy:fixture_token" for p in bad.harness.primitives)  # type: ignore[union-attr]
    _seq, _e, _m, _s, good = _run({"V.sol": DELTA_VAULT}, deposit_sequence())
    assert good.outcome is Outcome.PROPERTY_HELD, good.reason
    _seq, _e, _m, _s, false_return = _run(
        {"V.sol": DELTA_VAULT}, deposit_sequence("accounting.unchecked_token_return")
    )
    # the safe vault requires the transfer to succeed: the credit call reverts, so the
    # credit property cannot be judged -- inconclusive, not held and not violated
    assert false_return.outcome is Outcome.INCONCLUSIVE


def test_no_oracle_sequence_is_execution_only() -> None:
    sources = {"Acct.sol": VULNERABLE_INIT}
    model = model_of(sources)
    from dataclasses import replace

    seq = stamp_source_identity(replace(init_sequence(), template="boundary-batch"), sources)
    obs = StatefulExecutor().execute(seq, model, sources)
    assert obs.outcome is Outcome.SEQUENCE_EXECUTED_NO_ORACLE
    assert obs.declaration == "no_property_available" and not obs.strong_candidate


def test_compile_failure_is_distinct() -> None:
    broken = VULNERABLE_INIT.replace("owner = newOwner;", "owner = newOwner")
    _seq, _e, _m, _s, obs = _run({"Acct.sol": broken}, init_sequence())
    assert obs.outcome is Outcome.COMPILE_FAILED
    assert obs.result is not None and obs.result.compiler_status == "failed"


def test_independent_check_uses_a_materially_different_oracle() -> None:
    sources = {"Acct.sol": VULNERABLE_INIT}
    seq, executor, model, spec, obs = _run(sources, init_sequence())
    corroboration, runs = independent_check(executor, seq, model, sources, spec, obs)
    assert corroboration.status == "corroborated_candidate"
    assert corroboration.independence == "same_toolchain_alternate"
    assert corroboration.verified is False
    paths = {p.path: p for p in corroboration.paths}
    assert paths["foundry:pipeline:no_optimizer"].material is False
    assert paths["foundry:alternate:state_unchanged:owner()"].verdict == "property_violated"
    assert paths["echidna:property"].outcome == "unavailable"  # never fabricated
    assert corroboration.as_dict()["verified"] is False and len(runs) == 2


def test_feedback_loop_minimizes_and_bundles_reproduce(tmp_path: Path) -> None:
    sources = {"Acct.sol": VULNERABLE_INIT}
    model = model_of(sources)
    seq = stamp_source_identity(init_sequence(), sources)
    executor = StatefulExecutor()
    result = run_feedback_loop(
        executor, (seq,), {seq.sequence_id: model}, {seq.sequence_id: sources}, rounds=2
    )
    assert [o.outcome for o in result.observations] == [Outcome.PROPERTY_VIOLATED]
    minimized = result.minimized[seq.sequence_id]
    assert minimized.completed and len(minimized.minimized) == 2  # both calls are needed
    assert minimized.preserved_property == seq.property_under_test
    corroboration = result.independent[seq.sequence_id]
    assert corroboration.independence == "same_toolchain_alternate"
    assert corroboration.verified is False
    assert result.mutated == ()  # a violation is not a mutation signal
    assert (
        seq.identity.source_hashes
        == stamp_source_identity(init_sequence(), sources).identity.source_hashes
    )
    obs = result.observations[0]
    bundle = build_bundle(
        obs,
        seq,
        result.specs[seq.sequence_id],
        sources,
        minimized=minimized,
        corroboration=result.independent[seq.sequence_id],
    )
    blobs = {sha256_text(VULNERABLE_INIT): VULNERABLE_INIT}
    assert bundle["schema"] == "bugforge.repro_bundle/1" and bundle["verified"] is False
    assert "api_key" not in str(bundle).lower()
    again = build_bundle(
        obs,
        seq,
        result.specs[seq.sequence_id],
        sources,
        minimized=minimized,
        corroboration=result.independent[seq.sequence_id],
    )
    assert again["bundle_id"] == bundle["bundle_id"]  # content-addressed
    replay = replay_bundle(bundle, blobs)
    assert replay["status"] == "replayed" and replay["events_match"] is True
    with pytest.raises(BundleIntegrityError):
        materialize_bundle(bundle, {sha256_text(VULNERABLE_INIT): "tampered"}, tmp_path / "x")


def test_proxy_target_is_inconclusive_not_replayed_as_the_implementation() -> None:
    sources = {"Acct.sol": VULNERABLE_INIT}
    model = model_of(sources)
    obs = StatefulExecutor().execute(
        stamp_source_identity(init_sequence(), sources),
        model,
        sources,
        context=IdentityContext(proxy_kind="uups"),
    )
    assert obs.outcome is Outcome.INCONCLUSIVE
    assert obs.reason_code == "proxy_requires_compatible_harness"
