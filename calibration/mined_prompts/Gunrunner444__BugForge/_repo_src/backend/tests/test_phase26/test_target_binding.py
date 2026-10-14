"""Target-bound corroboration, reproduction, and lifecycle invariants."""

from __future__ import annotations

from dataclasses import replace
from uuid import uuid4

import pytest

from app.domain.evidence import Evidence, EvidenceBundle, EvidenceKind, observation_identity
from app.domain.findings import FindingStatus, SecurityFinding, SourceLocation
from app.domain.lifecycle_policy import (
    can_corroborate,
    positive_reproduction,
    reproduction_for_target,
)
from app.domain.target_identity import semantic_target_identity
from app.domain.trusted_evidence import is_trusted_observation, issue_for_finding
from app.repositories.security_finding_repo import merge_lifecycle_state

EVAL = "dangerous_dynamic_execution"


def _static(
    *,
    scope: str = "fn",
    argument: str = "0",
    occurrence: str = "0",
    sink_id: str = "py.eval",
    call: str = "eval(a)",
    summary: str = "eval at 1",
    target: str = "",
) -> Evidence:
    metadata = {
        "line": "1",
        "vulnerability_class": EVAL,
        "sink": "eval",
        "scope_id": scope,
        "argument_index": argument,
        "sink_occurrence": occurrence,
        "sink_id": sink_id,
        "call_identity": call,
    }
    if target:
        metadata["observed_target"] = target
    return Evidence(
        kind=EvidenceKind.STATIC_ANALYSIS,
        source="sec.taint.dynamic_execution",
        summary=summary,
        artifact_path="app.py",
        metadata=metadata,
    )


def _finding(*items: Evidence, project_id: str = "project-a", **kwargs: object) -> SecurityFinding:
    finding_key = str(kwargs.pop("finding_key", "same"))
    return SecurityFinding.potential(
        "Potential dynamic execution",
        evidence=EvidenceBundle.from_items(items or (_static(),)),
        vulnerability_class=EVAL,
        source_location=SourceLocation(file_path="app.py", line=1),
        finding_key=finding_key,
        project_id=project_id,
        flow_summary="flow",
        flow_source="request.args",
        flow_sink="eval",
        **kwargs,  # type: ignore[arg-type]
    )


def _http(**metadata: str) -> Evidence:
    data = {"method": "GET", "url": "https://app.example/search", "status": "200"}
    data.update(metadata)
    return Evidence(
        kind=EvidenceKind.HTTP_RESPONSE,
        source="lab-http",
        summary="response reflected payload",
        details="uid=0",
        metadata=data,
    )


def _repro(**metadata: str) -> Evidence:
    data = {"outcome": "reproduced", "reproduced": "true", "execution_id": "exec-a"}
    data.update(metadata)
    return Evidence(
        kind=EvidenceKind.REPRODUCTION,
        source="reproduction_engine",
        summary="exploit ran",
        details="payload",
        metadata=data,
    )


def test_reproduction_does_not_survive_target_drift() -> None:
    original = _finding(_static())
    reproduced = original.reproduce([_repro()])
    assert reproduced.status is FindingStatus.REPRODUCED
    historical = next(item for item in reproduced.evidence.items if positive_reproduction(item))
    old_target = semantic_target_identity(reproduced)
    assert historical.metadata["observed_target"] == old_target

    drifted = replace(
        reproduced,
        status=FindingStatus.POTENTIAL,
        flow_sink="exec",
    )
    assert semantic_target_identity(drifted) != old_target
    assert any(positive_reproduction(item) for item in drifted.evidence.items)
    with pytest.raises(ValueError, match="semantic target"):
        drifted.reproduce()
    with pytest.raises(ValueError, match="Human acceptance"):
        drifted.human_accept()
    with pytest.raises(ValueError, match="reproduction|verification"):
        replace(drifted, status=FindingStatus.HUMAN_ACCEPTED)

    fresh = drifted.reproduce([_repro(execution_id="exec-b")])
    assert fresh.status is FindingStatus.REPRODUCED
    assert reproduction_for_target(fresh.evidence.items[-1], semantic_target_identity(fresh))
    assert fresh.human_accept().status is FindingStatus.HUMAN_ACCEPTED
    assert any(item.metadata.get("observed_target") == old_target for item in fresh.evidence.items)


def test_static_stamp_does_not_corroborate_a_new_target() -> None:
    left = _static(summary="first", call="eval(a)")
    right = _static(summary="second", call="eval(b)")
    finding = _finding(left, right)
    target = semantic_target_identity(finding)
    stamped = replace(
        finding,
        evidence=EvidenceBundle.from_items(
            [
                replace(left, metadata={**left.metadata, "observed_target": target}),
                replace(right, metadata={**right.metadata, "observed_target": target}),
            ]
        ),
    )
    assert can_corroborate(
        stamped.evidence,
        target_id=target,
        finding_id=str(stamped.id),
        finding_key=stamped.finding_key,
        project_id=stamped.project_id,
    )
    drifted = replace(stamped, flow_source="request.form")
    assert not can_corroborate(
        drifted.evidence,
        target_id=semantic_target_identity(drifted),
        finding_id=str(drifted.id),
        finding_key=drifted.finding_key,
        project_id=drifted.project_id,
    )
    again = _static(summary="first", call="eval(a)")
    duplicate = replace(again, id=uuid4())
    assert observation_identity(again) == observation_identity(duplicate)


@pytest.mark.parametrize(
    ("mutate",),
    [
        (lambda finding: replace(finding, flow_sink="exec"),),
        (lambda finding: replace(finding, flow_source="request.form"),),
        (
            lambda finding: replace(
                finding, source_location=SourceLocation(file_path="other.py", line=9)
            ),
        ),
        (lambda finding: replace(finding, project_id="project-b"),),
        (lambda finding: replace(finding, field_path="user.id"),),
    ],
)
def test_runtime_observation_does_not_transfer(mutate) -> None:
    finding = _finding(_static(), _static(summary="other", call="eval(b)"))
    issued = issue_for_finding(finding, _http(), "exec-http")
    assert is_trusted_observation(issued)
    finding.verify([issued])
    drifted = mutate(finding)
    assert semantic_target_identity(drifted) != semantic_target_identity(finding)
    with pytest.raises(ValueError, match="independent|server-issued|finding"):
        drifted.verify([issued])
    assert not can_corroborate(
        EvidenceBundle.from_items([issued]),
        target_id=semantic_target_identity(drifted),
        finding_id=str(drifted.id),
        finding_key=drifted.finding_key,
        project_id=drifted.project_id,
    )


def test_static_identity_fields_change_the_target() -> None:
    base = _finding(_static())
    origin = semantic_target_identity(base)
    variants = [
        _finding(_static(argument="1")),
        _finding(_static(scope="other")),
        _finding(_static(occurrence="1")),
        _finding(_static(sink_id="py.exec")),
        _finding(_static(call="exec(a)")),
    ]
    assert len({semantic_target_identity(item) for item in variants}) == len(variants)
    assert all(semantic_target_identity(item) != origin for item in variants)


def test_project_mismatch_blocks_verification_and_corroboration() -> None:
    finding = _finding()
    other = replace(finding, project_id="project-b")
    issued = issue_for_finding(finding, _http(), "exec-p")
    with pytest.raises(ValueError, match="independent|server-issued|finding"):
        other.verify([issued])
    assert not can_corroborate(
        EvidenceBundle.from_items([issued]),
        target_id=semantic_target_identity(other),
        finding_id=str(other.id),
        finding_key=other.finding_key,
        project_id=other.project_id,
    )


def test_unsigned_note_does_not_change_trust_and_signed_metadata_does() -> None:
    finding = _finding()
    issued = issue_for_finding(finding, _http(), "exec-n")
    noted = replace(issued, metadata={**dict(issued.metadata), "note": "operator note"})
    assert is_trusted_observation(noted)
    finding.verify([noted])
    with pytest.raises(ValueError, match="signature"):
        replace(
            issued,
            metadata={**dict(issued.metadata), "observed_target": "different-target"},
        )


def test_new_uuid_is_not_independence_and_rejected_is_terminal() -> None:
    finding = _finding()
    issued = issue_for_finding(finding, _http(), "exec-d")
    copied = replace(issued, id=uuid4())
    assert observation_identity(issued) == observation_identity(copied)
    verified = finding.verify([issued])
    rejected = verified.reject()
    assert rejected.status is FindingStatus.REJECTED
    with pytest.raises(ValueError, match="Rejected"):
        rejected.verify([issue_for_finding(finding, _http(), "exec-e")])
    with pytest.raises(ValueError, match="Rejected"):
        rejected.reproduce([_repro()])
    with pytest.raises(ValueError, match="Rejected"):
        rejected.corroborate()


def test_static_ingress_cannot_verify() -> None:
    static = _static(summary="only static")
    with pytest.raises(ValueError, match="independent|server-issued"):
        SecurityFinding.verified(
            "Potential dynamic execution",
            evidence=EvidenceBundle.from_items([static, replace(static, summary="copy")]),
            vulnerability_class=EVAL,
            source_location=SourceLocation(file_path="app.py", line=1),
            finding_key="same",
            project_id="project-a",
            flow_sink="eval",
            flow_source="request.args",
        )


def test_stale_writer_cannot_replace_stronger_status_or_prove_a_new_target() -> None:
    current = _finding()
    issued = issue_for_finding(current, _http(), "exec-v")
    current = current.verify([issued])
    stale = replace(current, status=FindingStatus.POTENTIAL, flow_summary="")
    merged = merge_lifecycle_state(current, stale)
    assert merged.status is FindingStatus.VERIFIED
    assert any(is_trusted_observation(item) for item in merged.evidence.items)

    drifted_writer = replace(_finding(), flow_sink="exec").reproduce([_repro()])
    with pytest.raises(ValueError, match="reproduction|semantic target"):
        merge_lifecycle_state(_finding(), drifted_writer)
