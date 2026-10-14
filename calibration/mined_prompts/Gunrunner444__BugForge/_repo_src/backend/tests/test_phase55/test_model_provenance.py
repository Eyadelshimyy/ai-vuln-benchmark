"""A parsed model is attested to the source text it was built from."""

from __future__ import annotations

from app.discovery.bounty.stateful import Outcome, StatefulExecutor
from app.discovery.bounty.vfcs import generate, stamp_source_identity
from app.parsing.solidity_research import ResearchModel, SemanticCandidate, build_research_model
from tests.test_phase52.phase52_support import VULNERABLE_INIT, init_sequence, model_of

SOURCE_A = VULNERABLE_INIT
SOURCE_B = VULNERABLE_INIT.replace("owner = newOwner;", "owner = newOwner; owner = newOwner;")

BASE = """// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract Ownable {
    address public owner;
    function _check() internal view {
        require(msg.sender == owner);
    }
    modifier onlyOwner() {
        _check();
        _;
    }
}
"""

CHILD = """// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "./Ownable.sol";

contract Acct is Ownable {
    function initialize(address newOwner) external {
        owner = newOwner;
    }
    function poke() external onlyOwner {
        owner = msg.sender;
    }
}
"""


def _candidate() -> SemanticCandidate:
    return SemanticCandidate(
        detector="aa.unprotected_account_initializer",
        family="account_abstraction",
        title="initializer",
        summary="",
        file="Acct.sol",
        line=1,
        contract="Acct",
        function="initialize(address)",
    )


def test_model_records_file_hashes_for_the_text_it_parsed() -> None:
    sources = {"Ownable.sol": BASE, "Acct.sol": CHILD}
    model = build_research_model(sources)
    assert model.source_hashes
    assert dict(model.source_hashes)["Acct.sol"]
    assert dict(model.source_hashes)["Ownable.sol"]
    assert model.source_set_digest
    assert "Acct" in model.contracts and "Ownable" in model.contracts
    assert model.contracts["Acct"].bases == ("Ownable",)


def test_generation_refuses_a_model_parsed_from_different_text() -> None:
    model_a = model_of({"Acct.sol": SOURCE_A})
    sources_b = {"Acct.sol": SOURCE_B}
    built = generate(model_a, [_candidate()], sources=sources_b)
    assert built.sequences == ()
    assert built.skipped
    assert "model source attestation" in built.skipped[0][1]
    matched = generate(model_a, [_candidate()], sources={"Acct.sol": SOURCE_A})
    assert matched.sequences
    assert (
        dict(matched.sequences[0].identity.source_hashes)["Acct.sol"]
        == dict(model_a.source_hashes)["Acct.sol"]
    )


def test_execution_refuses_model_a_with_source_b_before_forge() -> None:
    model_a = model_of({"Acct.sol": SOURCE_A})
    sources_b = {"Acct.sol": SOURCE_B}
    sequence = stamp_source_identity(init_sequence(), sources_b)
    assert "function initialize(address" in SOURCE_B
    executor = StatefulExecutor()
    obs = executor.execute(sequence, model_a, sources_b)
    assert obs.outcome is Outcome.IDENTITY_MISMATCH, obs.reason
    assert "model" in obs.reason
    assert obs.executed is False
    assert executor.runs == 0


def test_partial_analysis_stays_truncated_and_is_not_relabeled() -> None:
    sources = {"A.sol": "contract A { function f() external {} }", "B.sol": SOURCE_A}
    model = build_research_model(sources, max_files=1)
    assert model.truncated
    assert any("only 1 of 2" in note for note in model.notes)
    assert len(model.source_hashes) == 1
    refused = generate(model, [_candidate()], sources=sources)
    assert refused.sequences == ()
    assert "model source attestation" in refused.skipped[0][1]
    considered = dict(model.source_hashes)
    same = {path: sources[path] for path in considered}
    # The partial model is only consistent with the files it actually read.
    again = build_research_model(same, max_files=1)
    assert again.source_set_digest == model.source_set_digest
    assert again.truncated is False or again.source_hashes == model.source_hashes


def test_a_model_without_attestation_is_not_treated_as_bound() -> None:
    bare = ResearchModel(contracts=model_of({"Acct.sol": SOURCE_A}).contracts)
    executor = StatefulExecutor()
    obs = executor.execute(
        stamp_source_identity(init_sequence(), {"Acct.sol": SOURCE_A}),
        bare,
        {"Acct.sol": SOURCE_A},
    )
    assert obs.outcome is Outcome.IDENTITY_MISMATCH
    assert "no source attestation" in obs.reason
    assert executor.runs == 0
