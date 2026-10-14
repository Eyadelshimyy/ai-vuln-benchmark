"""Deployment, implementation, and compiler identity fail closed before Forge."""

from __future__ import annotations

from app.discovery.bounty.stateful import (
    IdentityContext,
    Outcome,
    ReplayMode,
    StatefulExecutor,
    bind_identity,
)
from app.discovery.bounty.vfcs import SequenceIdentity, stamp_source_identity
from tests.test_phase52.phase52_support import VULNERABLE_INIT, init_sequence, model_of


def _bound(**identity: str):
    sources = {"Acct.sol": VULNERABLE_INIT}
    sequence = stamp_source_identity(init_sequence(identity=SequenceIdentity(**identity)), sources)
    return sequence, model_of(sources), sources


def _refuse(obs, executor) -> None:  # type: ignore[no-untyped-def]
    assert obs.outcome is Outcome.IDENTITY_MISMATCH, obs.reason
    assert obs.executed is False and obs.verified is False
    assert executor.runs == 0


def test_source_only_context_rejects_a_pinned_sequence() -> None:
    sequence, model, sources = _bound(deployment="1:0xabc")
    executor = StatefulExecutor()
    obs = executor.execute(sequence, model, sources, context=IdentityContext(deployment="source"))
    _refuse(obs, executor)
    assert "source-only" in obs.reason
    assert obs.identity is not None
    assert obs.identity.replay_mode is ReplayMode.LOCAL_SOURCE_REPLAY
    assert obs.identity.deployment == "source"


def test_pinned_context_rejects_a_source_only_sequence() -> None:
    sequence, model, sources = _bound()
    executor = StatefulExecutor()
    obs = executor.execute(sequence, model, sources, context=IdentityContext(deployment="1:0xabc"))
    _refuse(obs, executor)
    assert "deployment-specific" in obs.reason
    assert obs.identity is not None and obs.identity.deployment == "1:0xabc"


def test_unspecified_context_rejects_a_pinned_sequence() -> None:
    sequence, model, sources = _bound(deployment="1:0xabc")
    executor = StatefulExecutor()
    obs = executor.execute(sequence, model, sources, context=IdentityContext())
    _refuse(obs, executor)
    assert "does not establish that deployment" in obs.reason


def test_source_oriented_pairs_stay_bound_without_a_deployment() -> None:
    sequence, model, sources = _bound()
    unspecified = bind_identity(sequence, model, sources, IdentityContext())
    explicit = bind_identity(sequence, model, sources, IdentityContext(deployment="source"))
    assert unspecified.bound and unspecified.deployment == "source"
    assert explicit.bound and explicit.deployment == "source"
    assert unspecified.replay_mode is ReplayMode.LOCAL_SOURCE_REPLAY
    pinned = stamp_source_identity(
        init_sequence(identity=SequenceIdentity(deployment="1:0xabc")), sources
    )
    same = bind_identity(pinned, model, sources, IdentityContext(deployment="1:0xabc"))
    assert same.bound and same.deployment == "1:0xabc"
    assert same.replay_mode is ReplayMode.LOCAL_SOURCE_REPLAY


def test_a_required_implementation_is_not_optional() -> None:
    sequence, model, sources = _bound(implementation="impl_a")
    executor = StatefulExecutor()
    missing = executor.execute(sequence, model, sources, context=IdentityContext())
    _refuse(missing, executor)
    assert "implementation" in missing.reason and "does not establish" in missing.reason
    extra = executor.execute(
        _bound()[0], model, sources, context=IdentityContext(implementation="impl_a")
    )
    assert extra.outcome is Outcome.IDENTITY_MISMATCH
    assert "does not record" in extra.reason
    assert executor.runs == 0


def test_a_required_compiler_configuration_is_not_optional() -> None:
    sequence, model, sources = _bound(compiler_configuration="solc=0.8.20;optimizer=disabled")
    executor = StatefulExecutor()
    missing = executor.execute(sequence, model, sources, context=IdentityContext())
    _refuse(missing, executor)
    assert "compiler configuration" in missing.reason
    bare, bare_model, bare_sources = _bound()
    supplied = executor.execute(
        bare,
        bare_model,
        bare_sources,
        context=IdentityContext(compiler_configuration="solc=0.8.28;optimizer=enabled"),
    )
    assert supplied.outcome is Outcome.IDENTITY_MISMATCH
    assert "does not record" in supplied.reason
    assert executor.runs == 0


def test_matching_compiler_and_implementation_stay_bound() -> None:
    sequence, model, sources = _bound(
        compiler_configuration="solc=0.8.20;optimizer=disabled", implementation="impl_a"
    )
    identity = bind_identity(
        sequence,
        model,
        sources,
        IdentityContext(
            compiler_configuration="solc=0.8.20;optimizer=disabled",
            implementation="impl_a",
            deployment="source",
        ),
    )
    assert identity.bound
    assert identity.compiler_configuration == "solc=0.8.20;optimizer=disabled"
    assert identity.implementation == "impl_a"
    assert identity.deployment == "source"
