"""Buyer-facing report regressions for the governed item-bank lifecycle."""

from __future__ import annotations

import json
from pathlib import Path
import runpy

import pytest

from fast_mlsirm.rubric import (
    ItemBankEvidenceKind,
    ItemBankEvidenceReference,
    ItemBankLifecycleState,
    PolicyCriticality,
    audit_generated_item_candidate,
    build_item_bank_pilot_record,
    build_pilot_candidate_record,
    transition_item_bank_record,
)
from fast_mlsirm.rubric.item_bank_report import (
    ItemBankReportError,
    build_item_bank_report,
    render_item_bank_report_html,
    render_item_bank_report_json,
)

_AUDIT_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("test_rubric_candidate_audit.py"))
)


def _fingerprint(character: str) -> str:
    """Return one deterministic complete SHA-256-style fixture value."""
    return character * 64


def _evidence(
    kind: ItemBankEvidenceKind,
    suffix: str,
    character: str,
) -> ItemBankEvidenceReference:
    """Return one source-text-free evidence reference."""
    return ItemBankEvidenceReference(
        evidence_kind=kind,
        evidence_id=f"{suffix}_evidence",
        evidence_fingerprint=_fingerprint(character),
    )


def _lifecycle(*, include_linking: bool = False):
    """Return one complete pilot-to-active lifecycle fixture."""
    candidate = _AUDIT_FIXTURES["_candidate"]()
    audit_report = audit_generated_item_candidate(candidate)
    pilot_candidate = build_pilot_candidate_record(
        candidate,
        audit_report,
        **_AUDIT_FIXTURES["_pilot_kwargs"](),
    )
    pilot = build_item_bank_pilot_record(
        pilot_candidate,
        item_version="1.0.0",
        policy_criticality=PolicyCriticality.CONJUNCTIVE_GATE,
    )
    calibrated = transition_item_bank_record(
        pilot,
        ItemBankLifecycleState.CALIBRATED,
        evidence_references=(
            _evidence(ItemBankEvidenceKind.CALIBRATION, "calibration", "a"),
            _evidence(ItemBankEvidenceKind.ITEM_FIT, "item_fit", "b"),
            _evidence(ItemBankEvidenceKind.DIF, "dif", "c"),
            _evidence(
                ItemBankEvidenceKind.ITEM_INFORMATION,
                "item_information",
                "d",
            ),
        ),
        transition_reason_id="calibration_completed",
    )
    approval_evidence = [
        _evidence(ItemBankEvidenceKind.APPROVAL, "approval", "e"),
    ]
    if include_linking:
        approval_evidence.append(
            _evidence(ItemBankEvidenceKind.LINKING, "linking", "f")
        )
    approved = transition_item_bank_record(
        calibrated,
        ItemBankLifecycleState.APPROVED,
        evidence_references=tuple(approval_evidence),
        transition_reason_id="governance_approval",
        approved_use_ids=("production_scoring",),
    )
    active = transition_item_bank_record(
        approved,
        ItemBankLifecycleState.ACTIVE,
        evidence_references=(),
        transition_reason_id="release_activation",
    )
    return pilot, calibrated, approved, active


def test_machine_report_preserves_provenance_and_explicit_limitations() -> None:
    """The report exposes governed evidence without inventing missing claims."""
    records = _lifecycle()

    report = build_item_bank_report(records)

    assert report["current_state"] == "active"
    assert report["item_id"] == records[-1].item_id
    assert report["item_version"] == "1.0.0"
    assert report["rubric_id"] == records[-1].rubric_id
    assert report["rubric_version"] == records[-1].rubric_version
    assert report["policy_criticality"] == "conjunctive_gate"
    assert report["approved_use_ids"] == ["production_scoring"]
    assert report["cross_version_comparability"] == "not_demonstrated"
    assert "linking_evidence_not_present" in report["limitations"]
    assert "exposure_evidence_not_present" in report["limitations"]
    assert "drift_evidence_not_present" in report["limitations"]
    assert report["evidence_status"]["calibration"] == "present"
    assert report["evidence_status"]["item_fit"] == "present"
    assert report["evidence_status"]["dif"] == "present"
    assert report["evidence_status"]["item_information"] == "present"
    assert [step["state"] for step in report["timeline"]] == [
        "piloting",
        "calibrated",
        "approved",
        "active",
    ]


def test_linking_evidence_changes_only_the_comparability_evidence_status() -> None:
    """A linked report may state support without recomputing linking arithmetic."""
    records = _lifecycle(include_linking=True)

    report = build_item_bank_report(records)

    assert report["cross_version_comparability"] == "supported_by_linking_evidence"
    assert report["evidence_status"]["linking"] == "present"
    assert "linking_evidence_not_present" not in report["limitations"]


def test_json_report_is_deterministic_machine_readable_and_source_text_free() -> None:
    """JSON output is stable and contains only governed record metadata."""
    records = _lifecycle(include_linking=True)

    first = render_item_bank_report_json(records)
    second = render_item_bank_report_json(records)
    parsed = json.loads(first)

    assert first == second
    assert parsed == build_item_bank_report(records)
    assert "source_text" not in first
    assert "prompt_text" not in first
    assert "response_text" not in first


def test_html_report_is_standalone_accessible_and_escapes_title() -> None:
    """The human report has semantic landmarks and a visible focus contract."""
    records = _lifecycle()

    rendered = render_item_bank_report_html(
        records,
        title="Item bank <release> & review",
    )
    piloting_rendered = render_item_bank_report_html((records[0],))

    assert rendered.startswith("<!doctype html>")
    assert '<html lang="en">' in rendered
    assert '<meta charset="utf-8">' in rendered
    assert '<main id="main-content">' in rendered
    assert '<a class="skip-link" href="#main-content">Skip to report</a>' in rendered
    assert "Item bank &lt;release&gt; &amp; review" in rendered
    assert f"<dt>Blueprint</dt><dd>{records[-1].blueprint_id}</dd>" in rendered
    assert "<dt>Approved-use scope</dt><dd>production_scoring</dd>" in rendered
    assert "<dt>Approved-use scope</dt><dd>none</dd>" in piloting_rendered
    assert "<caption>Lifecycle timeline</caption>" in rendered
    assert "<caption>Evidence inventory</caption>" in rendered
    assert ":focus-visible" in rendered
    assert "outline: none" not in rendered


def test_report_rejects_partial_or_forged_lifecycle_lineage() -> None:
    """A report cannot silently present a disconnected successor as history."""
    records = _lifecycle()

    with pytest.raises(ItemBankReportError, match="must start at piloting"):
        build_item_bank_report(records[1:])

    with pytest.raises(ItemBankReportError, match="lineage is not contiguous"):
        build_item_bank_report((records[0], records[2], records[3]))


def test_report_replays_creation_fingerprint_before_trusting_record_fields() -> None:
    """Frozen-record bypasses cannot forge approved-use metadata under an old hash."""
    records = _lifecycle()
    object.__setattr__(records[-1], "approved_use_ids", ("forged_use",))

    with pytest.raises(ItemBankReportError, match="creation-time identity"):
        build_item_bank_report(records)


def test_report_rejects_hostile_container_and_title_subclasses() -> None:
    """Rendering rejects caller subclasses before invoking their callbacks."""
    records = _lifecycle()

    class HostileTuple(tuple):
        def __iter__(self):
            raise AssertionError("caller iterator executed")

    class HostileString(str):
        def strip(self) -> str:
            raise AssertionError("caller string callback executed")

    with pytest.raises(TypeError, match="records must be a built-in tuple"):
        build_item_bank_report(HostileTuple(records))

    with pytest.raises(TypeError, match="title must be a built-in str"):
        render_item_bank_report_html(records, title=HostileString("Item bank"))
