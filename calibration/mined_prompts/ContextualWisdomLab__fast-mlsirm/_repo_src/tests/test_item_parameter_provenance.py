"""Governed provisional/calibrated parameter-provenance regressions."""

from __future__ import annotations

import json
from pathlib import Path
import runpy

import pytest

from fast_mlsirm.rubric.item_bank_report import (
    ItemBankReportError,
    build_item_bank_report,
    render_item_bank_report_html,
    render_item_bank_report_json,
)
from fast_mlsirm.rubric.item_parameter_evidence import (
    ItemParameterStatus,
    ProvisionalParameterMethod,
    build_item_parameter_evidence,
)

_REPORT_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("test_rubric_item_bank_report.py"))
)


def _fingerprint(character: str) -> str:
    """Return one deterministic complete SHA-256-style fixture value."""

    return character * 64


def _provisional(item_id: str, item_version: str):
    """Return one explicit provisional parameter fixture."""

    return build_item_parameter_evidence(
        item_id=item_id,
        item_version=item_version,
        response_model_id="rasch_model",
        parameter_artifact_fingerprint=_fingerprint("a"),
        status=ItemParameterStatus.PROVISIONAL,
        provisional_method=ProvisionalParameterMethod.RASCH_COMMON_DISCRIMINATION,
        provisional_basis_fingerprint=_fingerprint("b"),
    )


def _calibrated(item_id: str, item_version: str):
    """Return one explicit calibrated parameter fixture."""

    return build_item_parameter_evidence(
        item_id=item_id,
        item_version=item_version,
        response_model_id="two_pl_model",
        parameter_artifact_fingerprint=_fingerprint("c"),
        status=ItemParameterStatus.CALIBRATED,
        calibration_evidence_fingerprint=_fingerprint("d"),
    )


def test_provisional_parameter_evidence_is_explicit_and_content_addressed() -> None:
    """Cold-start parameters cannot masquerade as empirical calibration."""

    evidence = _provisional("generated_item", "1.0.0")

    payload = evidence.to_dict()
    assert payload["status"] == "provisional"
    assert payload["provisional_method"] == "rasch_common_discrimination"
    assert payload["provisional_basis_fingerprint"] == _fingerprint("b")
    assert payload["calibration_evidence_fingerprint"] is None
    assert payload["parameter_artifact_fingerprint"] == _fingerprint("a")
    assert payload["evidence_fingerprint"] == evidence.evidence_fingerprint


def test_calibrated_parameter_evidence_requires_calibration_provenance() -> None:
    """A calibrated claim is bound to exact empirical calibration evidence."""

    evidence = _calibrated("generated_item", "1.0.0")

    payload = evidence.to_dict()
    assert payload["status"] == "calibrated"
    assert payload["provisional_method"] is None
    assert payload["provisional_basis_fingerprint"] is None
    assert payload["calibration_evidence_fingerprint"] == _fingerprint("d")


@pytest.mark.parametrize(
    "kwargs, message",
    [
        pytest.param(
            {
                "status": ItemParameterStatus.PROVISIONAL,
                "provisional_method": ProvisionalParameterMethod.TEMPLATE_PRIOR,
                "provisional_basis_fingerprint": _fingerprint("e"),
                "calibration_evidence_fingerprint": _fingerprint("f"),
            },
            "provisional parameter evidence must not carry calibration evidence",
            id="provisional-with-calibration",
        ),
        pytest.param(
            {
                "status": ItemParameterStatus.PROVISIONAL,
                "provisional_method": None,
                "provisional_basis_fingerprint": _fingerprint("e"),
            },
            "provisional parameter evidence requires a provisional method",
            id="provisional-without-method",
        ),
        pytest.param(
            {
                "status": ItemParameterStatus.CALIBRATED,
                "provisional_method": ProvisionalParameterMethod.LLTM_PREDICTED,
                "provisional_basis_fingerprint": _fingerprint("e"),
                "calibration_evidence_fingerprint": _fingerprint("f"),
            },
            "calibrated parameter evidence must not carry provisional provenance",
            id="calibrated-with-provisional-provenance",
        ),
        pytest.param(
            {
                "status": ItemParameterStatus.CALIBRATED,
                "calibration_evidence_fingerprint": None,
            },
            "calibrated parameter evidence requires calibration evidence",
            id="calibrated-without-calibration",
        ),
    ],
)
def test_parameter_evidence_rejects_mixed_or_missing_provenance(
    kwargs: dict[str, object],
    message: str,
) -> None:
    """Status and provenance cannot be combined into an ambiguous claim."""

    with pytest.raises(ValueError, match=message):
        build_item_parameter_evidence(
            item_id="generated_item",
            item_version="1.0.0",
            response_model_id="rasch_model",
            parameter_artifact_fingerprint=_fingerprint("a"),
            **kwargs,
        )


@pytest.mark.parametrize(
    "status, kwargs, message",
    [
        pytest.param(
            ItemParameterStatus.PROVISIONAL,
            {
                "provisional_method": ProvisionalParameterMethod.TEMPLATE_PRIOR,
                "provisional_basis_fingerprint": _fingerprint("a"),
            },
            "provisional basis must not be the parameter artifact itself",
            id="provisional-self-basis",
        ),
        pytest.param(
            ItemParameterStatus.CALIBRATED,
            {"calibration_evidence_fingerprint": _fingerprint("a")},
            "calibration evidence must not be the parameter artifact itself",
            id="calibrated-self-evidence",
        ),
    ],
)
def test_parameter_evidence_rejects_self_referential_provenance(
    status: ItemParameterStatus,
    kwargs: dict[str, object],
    message: str,
) -> None:
    """A parameter artifact cannot serve as its own provenance source."""

    with pytest.raises(ValueError, match=message):
        build_item_parameter_evidence(
            item_id="generated_item",
            item_version="1.0.0",
            response_model_id="rasch_model",
            parameter_artifact_fingerprint=_fingerprint("a"),
            status=status,
            **kwargs,
        )


def test_parameter_evidence_replays_creation_identity_before_serialization() -> None:
    """Frozen-record bypasses cannot relabel provisional parameters as calibrated."""

    evidence = _provisional("generated_item", "1.0.0")
    object.__setattr__(evidence, "status", ItemParameterStatus.CALIBRATED)

    with pytest.raises(ValueError, match="creation-time identity"):
        evidence.to_dict()


def test_parameter_evidence_rejects_caller_string_subclass_without_callback() -> None:
    """Identifier admission never executes caller-controlled string normalization."""

    calls: list[str] = []

    class HostileString(str):
        def strip(self) -> str:
            calls.append("strip")
            raise AssertionError("caller string callback executed")

    with pytest.raises(ValueError, match="item_id must be a string"):
        build_item_parameter_evidence(
            item_id=HostileString("generated_item"),
            item_version="1.0.0",
            response_model_id="rasch_model",
            parameter_artifact_fingerprint=_fingerprint("a"),
            status=ItemParameterStatus.PROVISIONAL,
            provisional_method=ProvisionalParameterMethod.TEMPLATE_PRIOR,
            provisional_basis_fingerprint=_fingerprint("b"),
        )

    assert calls == []


def test_item_bank_report_preserves_explicit_parameter_status_without_inference() -> None:
    """Reports show supplied provenance and never infer calibration from lifecycle alone."""

    pilot, calibrated, approved, active = _REPORT_FIXTURES["_lifecycle"]()
    provisional = _provisional(pilot.item_id, pilot.item_version)
    calibrated_evidence = _calibrated(active.item_id, active.item_version)

    pilot_report = build_item_bank_report((pilot,), parameter_evidence=provisional)
    active_report = build_item_bank_report(
        (pilot, calibrated, approved, active),
        parameter_evidence=calibrated_evidence,
    )
    unsupplied_report = build_item_bank_report((pilot,))

    assert pilot_report["parameter_status"] == "provisional"
    assert pilot_report["parameter_evidence"] == provisional.to_dict()
    assert active_report["parameter_status"] == "calibrated"
    assert active_report["parameter_evidence"] == calibrated_evidence.to_dict()
    assert unsupplied_report["parameter_status"] == "not_supplied"
    assert unsupplied_report["parameter_evidence"] is None

    rendered_json = json.loads(
        render_item_bank_report_json((pilot,), parameter_evidence=provisional)
    )
    rendered_html = render_item_bank_report_html(
        (pilot,),
        parameter_evidence=provisional,
    )
    assert rendered_json["parameter_status"] == "provisional"
    assert "<dt>Parameter status</dt><dd>provisional</dd>" in rendered_html
    assert provisional.evidence_fingerprint in rendered_html


def test_item_bank_report_rejects_parameter_identity_and_lifecycle_mismatch() -> None:
    """A report cannot relabel another item or a lifecycle-incompatible parameter claim."""

    pilot, calibrated, approved, active = _REPORT_FIXTURES["_lifecycle"]()

    with pytest.raises(ItemBankReportError, match="item identity does not match"):
        build_item_bank_report(
            (pilot,),
            parameter_evidence=_provisional("other_item", pilot.item_version),
        )
    with pytest.raises(ItemBankReportError, match="piloting cannot claim calibrated"):
        build_item_bank_report(
            (pilot,),
            parameter_evidence=_calibrated(pilot.item_id, pilot.item_version),
        )
    with pytest.raises(ItemBankReportError, match="calibrated lifecycle requires calibrated"):
        build_item_bank_report(
            (pilot, calibrated, approved, active),
            parameter_evidence=_provisional(active.item_id, active.item_version),
        )
