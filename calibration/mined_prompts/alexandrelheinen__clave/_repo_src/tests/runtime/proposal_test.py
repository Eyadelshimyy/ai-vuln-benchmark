"""Tests for the Python half of the inference boundary."""

import json

import pytest

from clave.runtime.proposal import PROPOSAL_VERSION, Outcome, Proposal, ProposalError


def proposal(**overrides: object) -> Proposal:
    """Build a valid proposal, with fields replaced for the case under test."""
    fields: dict[str, object] = {
        "object_id": 7,
        "material_class": "M-01",
        "confidence": 0.9,
        "point": (0.1, 0.0, 0.36),
        "yaw_radians": 0.0,
        "reference_time_nanos": 1_000,
        "window_start_nanos": 1_000,
        "window_end_nanos": 2_000,
    }
    fields.update(overrides)
    return Proposal(**fields)  # type: ignore[arg-type]


def test_an_encoded_proposal_carries_the_version_first() -> None:
    """A runtime reads the version before it interprets anything else."""
    read = json.loads(proposal().encode())
    assert read["version"] == PROPOSAL_VERSION
    assert read["material_class"] == "M-01"
    assert read["x_meters"] == 0.1


def test_a_class_outside_the_taxonomy_is_refused_before_it_is_sent() -> None:
    """Shipping garbage to read the complaint back teaches nothing."""
    with pytest.raises(ProposalError, match="M-99"):
        proposal(material_class="M-99")


def test_a_confidence_outside_the_unit_interval_is_refused() -> None:
    """The contract validates this too; catching it here names the field."""
    with pytest.raises(ProposalError, match="confidence"):
        proposal(confidence=1.4)


def test_a_window_that_ends_before_it_starts_is_refused() -> None:
    """Such a window names no reachable interval."""
    with pytest.raises(ProposalError, match="ends before it starts"):
        proposal(window_start_nanos=2_000, window_end_nanos=1_000)


def test_a_pose_timed_outside_its_window_is_refused() -> None:
    """The pose says where the object will be when the effector arrives."""
    with pytest.raises(ProposalError, match="outside the window"):
        proposal(reference_time_nanos=5_000)


def test_each_verdict_the_runtime_can_report_is_readable() -> None:
    """The three verdicts are what a run counts."""
    accepted = Outcome.decode(
        b'{"verdict":"accepted","object_id":7,"channel":1,"reject_reason":null}'
    )
    assert accepted.accepted
    assert accepted.channel == 1
    overridden = Outcome.decode(b'{"verdict":"overridden","check":"reach"}')
    assert overridden.overridden
    assert overridden.check == "reach"
    refused = Outcome.decode(b'{"verdict":"refused","reason":"bad version"}')
    assert refused.reason == "bad version"
    assert not refused.accepted


def test_an_unreadable_answer_is_reported_rather_than_guessed() -> None:
    """A producer that guesses what the runtime meant reports fiction."""
    with pytest.raises(ProposalError):
        Outcome.decode(b"not json")
    with pytest.raises(ProposalError, match="unknown verdict"):
        Outcome.decode(b'{"verdict":"maybe"}')
