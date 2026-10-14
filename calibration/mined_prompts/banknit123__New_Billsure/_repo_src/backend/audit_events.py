"""
backend/audit_events.py
=========================
Unified append-only audit trail for the ASIC ERS pilot (task section
12), covering the event categories the task spec requires that don't
already have a dedicated audit table elsewhere in this codebase.

Context: several modules already have their own append-only audit
tables (onboarding_audit_log, launch_gate_audit_log, complaint_audit_
log, plus the pre-existing generic `audit_log` table driven by
audit_trigger_func() on the ledger/credit tables since migration 002).
This module does NOT duplicate or replace those — it exists for the
event categories that had nowhere to go: login/security events,
administrative access, and data exports specifically. It also defines
the canonical field shape (actor, role, timestamp, action, object_type,
object_id, previous_state, new_state, reason, correlation_id, source)
that the task spec asks every audit event to carry, for any future
module that wants a shared, consistent audit table rather than
inventing its own.

KNOWN GAP, stated plainly rather than glossed over: consolidating every
existing per-module audit table into this single schema would be a
genuine, non-trivial refactor across ~8 files built across sessions 1-11
— out of scope for this session. `get_events_for_object()` below only
searches this table, not the others; a caller wanting a truly complete
history of an object today still needs to separately check
onboarding_audit_log / launch_gate_audit_log / complaint_audit_log /
audit_log as well. Recorded as a real gap in the evidence pack, not
hidden.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import supabase_db as sdb

logger = logging.getLogger(__name__)

EVENT_CATEGORIES = (
    "login", "security", "onboarding", "consent", "credit_assessment",
    "override", "limit_change", "bill_approval", "payment", "ledger_posting",
    "reconciliation", "complaint", "hardship", "document_change",
    "administrative_access", "launch_gate_change", "configuration_change",
    "data_export",
)

# Field names that must never appear in an exported report's payload in
# unredacted form — a denylist applied by redact_for_export(). This is
# deliberately conservative (better to over-redact than leak); it is
# not a substitute for each report-generating function also choosing
# not to include raw PII in the first place (see regulatory_reports.py).
SENSITIVE_FIELD_NAMES = frozenset({
    "full_name", "first_name", "last_name", "date_of_birth", "dob",
    "email", "phone", "phone_number", "address", "residential_address",
    "bank_account_number", "account_number", "bsb", "tax_file_number", "tfn",
    "medicare_number", "passport_number", "drivers_licence_number",
    "government_id_number", "government_id",
})


class AuditEventError(Exception):
    """Raised for any invalid audit-event operation. Every raise path is
    a refusal to record a malformed event — there is no partial/best-
    effort audit entry."""


@dataclass
class AuditEvent:
    category: str
    action: str
    actor: str
    role: str
    object_type: str
    object_id: str
    previous_state: Optional[dict]
    new_state: Optional[dict]
    reason: Optional[str]
    correlation_id: Optional[str]
    source: str
    timestamp: str


async def record_event(
    category: str, action: str, actor: str, role: str, object_type: str, object_id: str,
    source: str, previous_state: Optional[dict] = None, new_state: Optional[dict] = None,
    reason: Optional[str] = None, correlation_id: Optional[str] = None,
) -> dict:
    """Records one audit event. Every field the task spec requires
    (actor, role, timestamp, action, object, previous state, new state,
    reason, correlation ID, source) is a required or explicitly-optional
    parameter here — there is no way to call this function and end up
    with an event missing actor/role/action/object/source, the fields
    that make an audit trail actually useful for investigation."""
    if category not in EVENT_CATEGORIES:
        raise AuditEventError(f"unknown audit event category: {category}")
    if not actor or not role or not action or not object_type or not object_id or not source:
        raise AuditEventError("actor, role, action, object_type, object_id, and source are all required")

    row = {
        "category": category, "action": action, "actor": actor, "role": role,
        "object_type": object_type, "object_id": object_id,
        "previous_state": previous_state, "new_state": new_state,
        "reason": reason, "correlation_id": correlation_id, "source": source,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return await sdb.insert_one("audit_events", row)


async def get_events_for_object(object_type: str, object_id: str) -> list:
    """Returns every audit_events row for a given object, in this
    table only — see module docstring's KNOWN GAP note about other
    per-module audit tables not being covered here."""
    return await sdb.find_many("audit_events", {"object_type": object_type, "object_id": object_id})


async def get_events_by_correlation(correlation_id: str) -> list:
    """Returns every event sharing a correlation_id — e.g. every audit
    event that resulted from one customer-facing request, across
    however many objects/categories it touched."""
    return await sdb.find_many("audit_events", {"correlation_id": correlation_id})


def redact_for_export(event: dict) -> dict:
    """Returns a copy of an audit event with any SENSITIVE_FIELD_NAMES
    key, anywhere inside previous_state or new_state, replaced with a
    redaction marker rather than the real value. Applied at export time
    (not at record time) because the audit trail itself needs the real
    values for investigation — only the EXPORTED/reported version needs
    to avoid unnecessary personal information, per the task spec's
    explicit requirement for exports specifically."""
    def _redact_dict(d):
        if not isinstance(d, dict):
            return d
        return {k: ("[REDACTED]" if k in SENSITIVE_FIELD_NAMES else _redact_dict(v) if isinstance(v, dict) else v)
                for k, v in d.items()}

    redacted = dict(event)
    if "previous_state" in redacted:
        redacted["previous_state"] = _redact_dict(redacted["previous_state"])
    if "new_state" in redacted:
        redacted["new_state"] = _redact_dict(redacted["new_state"])
    return redacted
