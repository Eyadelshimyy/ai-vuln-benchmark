"""Source identity is part of every executed sequence. Mismatches never reach Forge."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from app.discovery.bounty.stateful import IdentityContext, Outcome, StatefulExecutor, tool_status
from app.discovery.bounty.vfcs import (
    FeedbackSignal,
    SequenceIdentity,
    mutate,
    stamp_source_identity,
)
from app.discovery.process import tool_path
from tests.test_phase52.phase52_support import VULNERABLE_INIT, init_sequence, model_of

BASE = """// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract Ownable {
    address public owner;
}
"""

CHILD = """// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "./Ownable.sol";

contract Acct is Ownable {
    function initialize(address newOwner) external {
        owner = newOwner;
    }
}
"""


def _require_toolchain() -> None:
    forge = tool_path("forge")
    solc = tool_path("solc")
    tools = tool_status()
    ready = bool(
        forge and solc and Path(forge).is_file() and Path(solc).is_file() and tools.available
    )
    if ready:
        return
    message = f"forge/solc not usable: forge={forge!r} solc={solc!r}"
    if os.environ.get("BUGFORGE_REQUIRE_FOUNDRY") == "1":
        pytest.fail(message)
    pytest.skip(message)


def _sources() -> dict[str, str]:
    return {"Acct.sol": VULNERABLE_INIT}


def _bound(**identity: str):
    sources = _sources()
    sequence = stamp_source_identity(init_sequence(identity=SequenceIdentity(**identity)), sources)
    return sequence, model_of(sources), sources


def _refuse(obs, executor) -> None:  # type: ignore[no-untyped-def]
    assert obs.outcome is Outcome.IDENTITY_MISMATCH, obs.reason
    assert obs.verified is False and obs.strong_candidate is False
    assert obs.executed is False
    assert executor.runs == 0


def test_exact_original_source_executes() -> None:
    _require_toolchain()
    sequence, model, sources = _bound(source_snapshot="fixture-init")
    executor = StatefulExecutor()
    obs = executor.execute(sequence, model, sources)
    assert obs.outcome is Outcome.PROPERTY_VIOLATED, obs.reason
    assert executor.runs == 1
    assert obs.identity is not None and obs.identity.bound
    assert dict(obs.identity.source_hashes) == dict(sequence.identity.source_hashes)


def test_changed_body_with_the_same_name_does_not_run() -> None:
    sequence, model, sources = _bound()
    changed = {
        "Acct.sol": sources["Acct.sol"].replace("address public owner;", "address public admin;")
    }
    executor = StatefulExecutor()
    obs = executor.execute(sequence, model_of(changed), changed)
    _refuse(obs, executor)
    assert "changed since the sequence was created" in obs.reason


def test_same_signature_with_a_changed_body_does_not_run() -> None:
    sequence, _model, sources = _bound()
    changed = {
        "Acct.sol": sources["Acct.sol"].replace(
            "owner = newOwner;", "owner = newOwner; owner = newOwner;"
        )
    }
    executor = StatefulExecutor()
    obs = executor.execute(sequence, model_of(changed), changed)
    _refuse(obs, executor)
    assert "function initialize(address" in changed["Acct.sol"]
    assert "changed since the sequence was created" in obs.reason


def test_a_missing_base_contract_does_not_run() -> None:
    sources = {"Ownable.sol": BASE, "Acct.sol": CHILD}
    sequence = stamp_source_identity(init_sequence(), sources)
    executor = StatefulExecutor()
    obs = executor.execute(sequence, model_of(sources), {"Acct.sol": CHILD})
    _refuse(obs, executor)
    assert "Ownable.sol" in obs.reason


def test_an_extra_or_missing_file_does_not_run() -> None:
    sequence, model, sources = _bound()
    executor = StatefulExecutor()
    extra = {**sources, "Extra.sol": "contract Extra {}\n"}
    added = executor.execute(sequence, model, extra)
    _refuse(added, executor)
    assert "Extra.sol is not part of the sequence's source identity" in added.reason
    missing = executor.execute(sequence, model, {})
    assert missing.outcome is Outcome.IDENTITY_MISMATCH
    assert executor.runs == 0


def test_another_campaign_deployment_or_implementation_does_not_run() -> None:
    sequence, model, sources = _bound(
        campaign_id="cp_a", deployment="1:0xabc", implementation="impl_a"
    )
    executor = StatefulExecutor()
    other = executor.execute(sequence, model, sources, context=IdentityContext(campaign_id="cp_b"))
    _refuse(other, executor)
    assert "another campaign" in other.reason
    elsewhere = executor.execute(
        sequence, model, sources, context=IdentityContext(deployment="1:0xdef")
    )
    assert elsewhere.outcome is Outcome.IDENTITY_MISMATCH
    assert "1:0xdef" in elsewhere.reason
    other_impl = executor.execute(
        sequence, model, sources, context=IdentityContext(implementation="impl_b")
    )
    assert other_impl.outcome is Outcome.IDENTITY_MISMATCH
    assert "impl_b" in other_impl.reason
    assert executor.runs == 0


def test_a_changed_compiler_configuration_does_not_run() -> None:
    sequence, model, sources = _bound(compiler_configuration="solc=0.8.20;optimizer=disabled")
    executor = StatefulExecutor()
    obs = executor.execute(
        sequence,
        model,
        sources,
        context=IdentityContext(compiler_configuration="solc=0.8.28;optimizer=enabled"),
    )
    _refuse(obs, executor)
    assert "compiler configuration" in obs.reason


def test_a_restored_sequence_with_matching_hashes_executes() -> None:
    _require_toolchain()
    sequence, model, sources = _bound(campaign_id="cp_restore", source_snapshot="snap-1")
    stored = json.loads(
        json.dumps(
            {
                "campaign_id": sequence.identity.campaign_id,
                "source_snapshot": sequence.identity.source_snapshot,
                "source_hashes": [list(item) for item in sequence.identity.source_hashes],
            }
        )
    )
    restored = replace(
        sequence,
        identity=SequenceIdentity(
            campaign_id=stored["campaign_id"],
            source_snapshot=stored["source_snapshot"],
            source_hashes=tuple(tuple(item) for item in stored["source_hashes"]),
        ),
    )
    executor = StatefulExecutor()
    obs = executor.execute(restored, model, sources)
    assert obs.outcome is Outcome.PROPERTY_VIOLATED, obs.reason
    assert executor.runs == 1
    assert dict(obs.identity.source_hashes) == dict(restored.identity.source_hashes)  # type: ignore[union-attr]


def test_a_restored_sequence_whose_hash_no_longer_matches_does_not_run() -> None:
    sequence, _model, sources = _bound()
    changed = {"Acct.sol": sources["Acct.sol"] + "\n"}
    executor = StatefulExecutor()
    obs = executor.execute(sequence, model_of(changed), changed)
    _refuse(obs, executor)
    assert "changed since the sequence was created" in obs.reason


def test_missing_identity_cannot_be_supplied_by_the_caller() -> None:
    sources = _sources()
    model = model_of(sources)
    executor = StatefulExecutor()
    expected = {path: hashlib.sha256(text.encode()).hexdigest() for path, text in sources.items()}
    forged = executor.execute(
        init_sequence(),
        model,
        sources,
        context=IdentityContext(expected_hashes=expected),
    )
    _refuse(forged, executor)
    assert "not recorded" in forged.reason
    omitted = executor.execute(init_sequence(), model, sources)
    assert omitted.outcome is Outcome.IDENTITY_MISMATCH
    assert "not recorded" in omitted.reason
    assert executor.runs == 0


def test_mutation_keeps_the_parent_source_identity() -> None:
    sequence, _model, _sources_ = _bound(campaign_id="cp_a", deployment="1:0xabc")
    children = mutate(
        [sequence],
        [FeedbackSignal("caller_context_mismatch", sequence.sequence_id, call_index=0)],
    )
    assert children
    child = children[0]
    assert child.identity.source_hashes == sequence.identity.source_hashes
    assert child.identity.campaign_id == sequence.identity.campaign_id
    assert child.identity.deployment == sequence.identity.deployment
    assert child.property_under_test == sequence.property_under_test
    assert child.sequence_id != sequence.sequence_id
