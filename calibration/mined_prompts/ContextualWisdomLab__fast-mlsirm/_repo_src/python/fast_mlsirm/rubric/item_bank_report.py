"""Machine-readable and accessible reports for governed item-bank lifecycles.

The report layer summarizes immutable lifecycle records and evidence identities.
It deliberately does not calculate calibration, fit, DIF, information, linking,
exposure, drift, or uncertainty; those numerical quantities remain Rust-owned.
"""

from __future__ import annotations

from html import escape
import json
from typing import Any

from .item_bank import (
    ItemBankEvidenceKind,
    ItemBankEvidenceReference,
    ItemBankLifecycleError,
    ItemBankLifecycleRecord,
    ItemBankLifecycleState,
    PolicyCriticality,
    _verify_current_record,
)
from .item_parameter_evidence import (
    ItemParameterEvidence,
    ItemParameterStatus,
    _verify_item_parameter_evidence,
)

_MAX_REPORT_RECORDS = 256
_MAX_TITLE_CHARACTERS = 160
_REPORT_SCHEMA_VERSION = "fast-mlsirm-item-bank-report-v2"
_RECORD_INSTANCE_FIELDS = frozenset(
    {
        "item_id",
        "item_version",
        "candidate_fingerprint",
        "pilot_record_fingerprint",
        "audit_report_fingerprint",
        "blueprint_id",
        "rubric_id",
        "rubric_version",
        "lifecycle_state",
        "policy_criticality",
        "approved_use_ids",
        "evidence_references",
        "previous_record_fingerprint",
        "transition_reason_id",
        "suspension_concern_kinds",
        "schema_version",
        "_record_fingerprint",
    }
)
_EVIDENCE_INSTANCE_FIELDS = frozenset(
    {"evidence_kind", "evidence_id", "evidence_fingerprint"}
)


class ItemBankReportError(ValueError):
    """Raised when a lifecycle cannot be represented as a trustworthy report."""


def _has_exact_instance_fields(value: object, expected: frozenset[str]) -> bool:
    """Return whether an exact package object retains only declared state fields."""
    field_names = tuple(vars(value))
    if any(type(name) is not str for name in field_names):
        return False
    return frozenset(field_names) == expected


def _preflight_record_identity(record: ItemBankLifecycleRecord) -> None:
    """Reject mutated record fields before replay can invoke caller callbacks."""
    if not _has_exact_instance_fields(record, _RECORD_INSTANCE_FIELDS):
        raise ItemBankReportError(
            "lifecycle record no longer matches its creation-time identity"
        )
    string_fields = (
        "item_id",
        "item_version",
        "candidate_fingerprint",
        "pilot_record_fingerprint",
        "audit_report_fingerprint",
        "blueprint_id",
        "rubric_id",
        "rubric_version",
        "transition_reason_id",
        "schema_version",
    )
    if any(type(getattr(record, name)) is not str for name in string_fields):
        raise ItemBankReportError(
            "lifecycle record no longer matches its creation-time identity"
        )
    if type(vars(record)["_record_fingerprint"]) is not str:
        raise ItemBankReportError(
            "lifecycle record no longer matches its creation-time identity"
        )
    if type(record.lifecycle_state) is not ItemBankLifecycleState:
        raise ItemBankReportError(
            "lifecycle record no longer matches its creation-time identity"
        )
    if type(record.policy_criticality) is not PolicyCriticality:
        raise ItemBankReportError(
            "lifecycle record no longer matches its creation-time identity"
        )
    if (
        record.previous_record_fingerprint is not None
        and type(record.previous_record_fingerprint) is not str
    ):
        raise ItemBankReportError(
            "lifecycle record no longer matches its creation-time identity"
        )
    if type(record.approved_use_ids) is not tuple or any(
        type(value) is not str for value in record.approved_use_ids
    ):
        raise ItemBankReportError(
            "lifecycle record no longer matches its creation-time identity"
        )
    if type(record.evidence_references) is not tuple:
        raise ItemBankReportError(
            "lifecycle record no longer matches its creation-time identity"
        )
    for reference in record.evidence_references:
        if type(reference) is not ItemBankEvidenceReference:
            raise ItemBankReportError(
                "lifecycle record no longer matches its creation-time identity"
            )
        if not _has_exact_instance_fields(reference, _EVIDENCE_INSTANCE_FIELDS):
            raise ItemBankReportError(
                "lifecycle record no longer matches its creation-time identity"
            )
        if type(reference.evidence_kind) is not ItemBankEvidenceKind or any(
            type(value) is not str
            for value in (reference.evidence_id, reference.evidence_fingerprint)
        ):
            raise ItemBankReportError(
                "lifecycle record no longer matches its creation-time identity"
            )
    if type(record.suspension_concern_kinds) is not tuple or any(
        type(kind) is not ItemBankEvidenceKind
        for kind in record.suspension_concern_kinds
    ):
        raise ItemBankReportError(
            "lifecycle record no longer matches its creation-time identity"
        )


def _normalize_records(records: object) -> tuple[ItemBankLifecycleRecord, ...]:
    """Validate one complete contiguous package-owned lifecycle timeline."""
    if type(records) is not tuple:
        raise TypeError("records must be a built-in tuple")
    if not records:
        raise ItemBankReportError("records must contain at least one lifecycle record")
    if len(records) > _MAX_REPORT_RECORDS:
        raise ItemBankReportError(
            f"records must contain at most {_MAX_REPORT_RECORDS} lifecycle records"
        )

    verified_records: list[ItemBankLifecycleRecord] = []
    for record in records:
        if type(record) is not ItemBankLifecycleRecord:
            raise TypeError("records must contain exact ItemBankLifecycleRecord values")
        _preflight_record_identity(record)
        try:
            verified_records.append(_verify_current_record(record))
        except ItemBankLifecycleError:
            raise ItemBankReportError(
                "lifecycle record no longer matches its creation-time identity"
            ) from None
    normalized = tuple(verified_records)

    first = normalized[0]
    if (
        first.lifecycle_state is not ItemBankLifecycleState.PILOTING
        or first.previous_record_fingerprint is not None
    ):
        raise ItemBankReportError("lifecycle report must start at piloting")

    stable_identity = (
        first.item_id,
        first.item_version,
        first.candidate_fingerprint,
        first.pilot_record_fingerprint,
        first.audit_report_fingerprint,
        first.blueprint_id,
        first.rubric_id,
        first.rubric_version,
        first.policy_criticality,
    )
    previous = first
    for record in normalized[1:]:
        current_identity = (
            record.item_id,
            record.item_version,
            record.candidate_fingerprint,
            record.pilot_record_fingerprint,
            record.audit_report_fingerprint,
            record.blueprint_id,
            record.rubric_id,
            record.rubric_version,
            record.policy_criticality,
        )
        if current_identity != stable_identity:
            raise ItemBankReportError("lifecycle identity changed across report records")
        if record.previous_record_fingerprint != previous.record_fingerprint:
            raise ItemBankReportError("lifecycle lineage is not contiguous")
        previous = record
    return normalized


def _present_evidence_kinds(
    record: ItemBankLifecycleRecord,
) -> frozenset[ItemBankEvidenceKind]:
    """Return the evidence classes accumulated by the current lifecycle record."""
    return frozenset(reference.evidence_kind for reference in record.evidence_references)


def _evidence_status(
    present: frozenset[ItemBankEvidenceKind],
) -> dict[str, str]:
    """Return an explicit present/not-present status for every governed class."""
    return {
        kind.value: "present" if kind in present else "not_present"
        for kind in ItemBankEvidenceKind
    }


def _limitations(
    current: ItemBankLifecycleRecord,
    present: frozenset[ItemBankEvidenceKind],
) -> list[str]:
    """Return explicit non-claims implied by absent lifecycle evidence."""
    limitations: list[str] = []
    if current.lifecycle_state is ItemBankLifecycleState.PILOTING:
        limitations.append("calibration_not_completed")
    if ItemBankEvidenceKind.LINKING not in present:
        limitations.append("linking_evidence_not_present")
    if ItemBankEvidenceKind.EXPOSURE not in present:
        limitations.append("exposure_evidence_not_present")
    if ItemBankEvidenceKind.DRIFT not in present:
        limitations.append("drift_evidence_not_present")
    return limitations


def _parameter_report_payload(
    current: ItemBankLifecycleRecord,
    parameter_evidence: ItemParameterEvidence | None,
) -> tuple[str, dict[str, Any] | None]:
    """Validate optional parameter provenance against the current lifecycle state."""
    if parameter_evidence is None:
        return "not_supplied", None
    if type(parameter_evidence) is not ItemParameterEvidence:
        raise ItemBankReportError(
            "parameter evidence must be an exact ItemParameterEvidence"
        )
    try:
        verified = _verify_item_parameter_evidence(parameter_evidence)
    except ValueError:
        raise ItemBankReportError(
            "parameter evidence no longer matches its creation-time identity"
        ) from None
    if (verified.item_id, verified.item_version) != (
        current.item_id,
        current.item_version,
    ):
        raise ItemBankReportError(
            "parameter evidence item identity does not match lifecycle record"
        )
    if (
        current.lifecycle_state is ItemBankLifecycleState.PILOTING
        and verified.status is ItemParameterStatus.CALIBRATED
    ):
        raise ItemBankReportError("piloting cannot claim calibrated parameter evidence")
    if (
        current.lifecycle_state is not ItemBankLifecycleState.PILOTING
        and verified.status is not ItemParameterStatus.CALIBRATED
    ):
        raise ItemBankReportError(
            "calibrated lifecycle requires calibrated parameter evidence"
        )
    return verified.status.value, verified.to_dict()


def build_item_bank_report(
    records: tuple[ItemBankLifecycleRecord, ...],
    *,
    parameter_evidence: ItemParameterEvidence | None = None,
) -> dict[str, Any]:
    """Build a source-text-free report from one complete lifecycle lineage.

    Parameters
    ----------
    records:
        Exact package-owned lifecycle records ordered from the initial piloting
        record through the current state. The chain must be contiguous.
    parameter_evidence:
        Optional explicit parameter-provenance record. When omitted, the report
        says ``not_supplied`` and never infers calibration merely from lifecycle
        state. When supplied, item/version and lifecycle/status compatibility are
        replayed before any parameter claim is rendered.

    Returns
    -------
    dict
        JSON-compatible lifecycle, provenance, evidence, and limitation data.
    """
    normalized = _normalize_records(records)
    current = normalized[-1]
    parameter_status, parameter_payload = _parameter_report_payload(
        current,
        parameter_evidence,
    )
    present = _present_evidence_kinds(current)
    comparability = (
        "supported_by_linking_evidence"
        if ItemBankEvidenceKind.LINKING in present
        else "not_demonstrated"
    )
    timeline = [
        {
            "state": record.lifecycle_state.value,
            "transition_reason_id": record.transition_reason_id,
            "record_id": record.record_id,
            "record_fingerprint": record.record_fingerprint,
            "previous_record_fingerprint": record.previous_record_fingerprint,
            "approved_use_ids": list(record.approved_use_ids),
        }
        for record in normalized
    ]
    return {
        "schema_version": _REPORT_SCHEMA_VERSION,
        "item_id": current.item_id,
        "item_version": current.item_version,
        "current_state": current.lifecycle_state.value,
        "parameter_status": parameter_status,
        "parameter_evidence": parameter_payload,
        "policy_criticality": current.policy_criticality.value,
        "blueprint_id": current.blueprint_id,
        "rubric_id": current.rubric_id,
        "rubric_version": current.rubric_version,
        "candidate_fingerprint": current.candidate_fingerprint,
        "pilot_record_fingerprint": current.pilot_record_fingerprint,
        "audit_report_fingerprint": current.audit_report_fingerprint,
        "current_record_id": current.record_id,
        "current_record_fingerprint": current.record_fingerprint,
        "approved_use_ids": list(current.approved_use_ids),
        "cross_version_comparability": comparability,
        "evidence_status": _evidence_status(present),
        "evidence_references": [
            reference.to_dict() for reference in current.evidence_references
        ],
        "timeline": timeline,
        "limitations": _limitations(current, present),
        "numerical_evidence_policy": (
            "report_only; numerical evidence is referenced, not recomputed"
        ),
    }


def render_item_bank_report_json(
    records: tuple[ItemBankLifecycleRecord, ...],
    *,
    parameter_evidence: ItemParameterEvidence | None = None,
) -> str:
    """Render one deterministic UTF-8 JSON representation of the bank report."""
    report = build_item_bank_report(records, parameter_evidence=parameter_evidence)
    return json.dumps(
        report,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n"


def _normalize_title(title: object) -> str:
    """Validate a caller-supplied HTML title before string callbacks execute."""
    if type(title) is not str:
        raise TypeError("title must be a built-in str")
    normalized = title.strip()
    if not normalized:
        raise ValueError("title must not be empty")
    if len(normalized) > _MAX_TITLE_CHARACTERS:
        raise ValueError(
            f"title must be at most {_MAX_TITLE_CHARACTERS} characters"
        )
    if "\n" in normalized or "\r" in normalized:
        raise ValueError("title must be a single line")
    return normalized


def _table_row(label: str, value: str) -> str:
    """Render one escaped definition-list row for the report summary."""
    return f"<dt>{escape(label)}</dt><dd>{escape(value)}</dd>"


def render_item_bank_report_html(
    records: tuple[ItemBankLifecycleRecord, ...],
    *,
    title: str = "Item bank lifecycle report",
    parameter_evidence: ItemParameterEvidence | None = None,
) -> str:
    """Render a standalone accessible HTML lifecycle report.

    The HTML includes semantic landmarks, table captions, an explicit skip link,
    and a visible keyboard-focus treatment. It contains only governed IDs,
    fingerprints, state, and evidence metadata from the lifecycle records.
    """
    normalized_title = _normalize_title(title)
    report = build_item_bank_report(records, parameter_evidence=parameter_evidence)
    escaped_title = escape(normalized_title)
    parameter_payload = report["parameter_evidence"]
    parameter_fingerprint = (
        "not supplied"
        if parameter_payload is None
        else str(parameter_payload["evidence_fingerprint"])
    )

    summary = "".join(
        (
            _table_row("Item", str(report["item_id"])),
            _table_row("Version", str(report["item_version"])),
            _table_row("Blueprint", str(report["blueprint_id"])),
            _table_row("Current state", str(report["current_state"])),
            _table_row("Parameter status", str(report["parameter_status"])),
            _table_row("Parameter evidence", parameter_fingerprint),
            _table_row("Rubric", str(report["rubric_id"])),
            _table_row("Rubric version", str(report["rubric_version"])),
            _table_row("Policy criticality", str(report["policy_criticality"])),
            _table_row(
                "Approved-use scope",
                ", ".join(str(value) for value in report["approved_use_ids"])
                or "none",
            ),
            _table_row(
                "Cross-version comparability",
                str(report["cross_version_comparability"]),
            ),
        )
    )

    evidence_rows = "".join(
        "<tr><th scope=\"row\">"
        + escape(kind)
        + "</th><td>"
        + escape(status)
        + "</td></tr>"
        for kind, status in report["evidence_status"].items()
    )
    timeline_rows = "".join(
        "<tr><td>"
        + escape(str(step["state"]))
        + "</td><td>"
        + escape(str(step["transition_reason_id"]))
        + "</td><td><code>"
        + escape(str(step["record_fingerprint"]))
        + "</code></td></tr>"
        for step in report["timeline"]
    )
    limitations = report["limitations"]
    limitations_html = (
        "<p>No report-layer limitations were inferred from the governed "
        "evidence inventory.</p>"
        if not limitations
        else "<ul>"
        + "".join(f"<li>{escape(str(value))}</li>" for value in limitations)
        + "</ul>"
    )

    return (
        "<!doctype html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{escaped_title}</title>\n"
        "<style>"
        "body{font-family:system-ui,sans-serif;line-height:1.5;margin:0;}"
        "main{max-width:72rem;margin:auto;padding:1.25rem;}"
        ".skip-link{position:absolute;left:.5rem;top:.5rem;padding:.5rem;}"
        ":focus-visible{outline:3px solid currentColor;outline-offset:2px;}"
        "table{border-collapse:collapse;width:100%;margin-block:1rem;}"
        "th,td{border:1px solid currentColor;padding:.5rem;text-align:left;}"
        "dt{font-weight:700;margin-top:.5rem;}dd{margin-left:0;}"
        "code{overflow-wrap:anywhere;}"
        "</style>\n</head>\n<body>\n"
        '<a class="skip-link" href="#main-content">Skip to report</a>\n'
        '<main id="main-content">\n'
        f"<h1>{escaped_title}</h1>\n"
        '<section aria-labelledby="summary-heading"><h2 id="summary-heading">'
        "Summary</h2><dl>"
        f"{summary}</dl></section>\n"
        '<section aria-labelledby="evidence-heading"><h2 id="evidence-heading">'
        "Evidence</h2><table><caption>Evidence inventory</caption>"
        "<thead><tr><th scope=\"col\">Evidence class</th>"
        "<th scope=\"col\">Status</th></tr></thead><tbody>"
        f"{evidence_rows}</tbody></table></section>\n"
        '<section aria-labelledby="timeline-heading"><h2 id="timeline-heading">'
        "Timeline</h2><table><caption>Lifecycle timeline</caption>"
        "<thead><tr><th scope=\"col\">State</th>"
        "<th scope=\"col\">Reason</th><th scope=\"col\">Record fingerprint</th>"
        f"</tr></thead><tbody>{timeline_rows}</tbody></table></section>\n"
        '<section aria-labelledby="limitations-heading">'
        '<h2 id="limitations-heading">Limitations</h2>'
        f"{limitations_html}</section>\n"
        "<p>Numerical evidence is referenced by governed identity and is not "
        "recomputed by this report.</p>\n"
        "</main>\n</body>\n</html>\n"
    )


__all__ = [
    "ItemBankReportError",
    "build_item_bank_report",
    "render_item_bank_report_html",
    "render_item_bank_report_json",
]
