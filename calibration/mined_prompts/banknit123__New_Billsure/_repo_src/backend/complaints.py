"""
backend/complaints.py
=======================
Internal dispute resolution (IDR) case management and AFCA escalation
tracking for the ASIC ERS pilot.

IMPORTANT — AFCA has no public API. Checked directly against AFCA's
member portal documentation before building this: complaint management
for AFCA members is entirely through a web UI (the member portal,
launched 17 June 2024) — there is no REST API, webhook, or any other
programmatic integration surface documented anywhere. This module does
NOT call AFCA. `escalate_to_afca()` only records that a human has
escalated a complaint, with an optional AFCA case reference number
entered manually once AFCA issues one (visible in the member portal) —
there is no code path anywhere in this codebase that talks to AFCA's
servers, and there shouldn't be one, because the integration point
doesn't exist for anyone to build against.

Regulatory timeframes are NOT hard-coded arbitrary numbers — they come
from `DEFAULT_IDR_TIMEFRAME_POLICY`, sourced from ASIC Regulatory Guide
271 (RG 271), effective 5 October 2021:
- Acknowledgement: within 1 business day (RG 271's "generally within 24
  hours" guidance).
- Standard complaints: 30 calendar days (RG 271.65).
- Credit default notice-related complaints: 21 calendar days.
- Superannuation trustee complaints: 45 calendar days (kept for
  completeness; not expected to be relevant to this pilot's utility-bill
  credit product, but included so the policy object is honest about
  what RG 271 actually specifies rather than silently omitting a
  category).
Every timeframe carries its policy_version/source/effective_date, and
`propose_config_version`-style versioning (a new policy is a new
object, never a silent mutation) — see IDRTimeframePolicy.

Complaint status follows AFCA's own model directly (confirmed from
their member portal documentation): status is only ever 'open' or
'closed'; 'stage' separately tracks where in the process a complaint
currently sits. Conflating the two loses information AFCA itself
considers meaningfully different.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Optional

import supabase_db as sdb

logger = logging.getLogger(__name__)

COMPLAINT_STATUSES = ("open", "closed")
COMPLAINT_STAGES = (
    "received", "acknowledged", "investigating", "awaiting_customer",
    "remedy_proposed", "resolved", "escalated_to_afca", "closed",
)
COMPLAINT_CHANNELS = ("phone", "email", "web_form", "in_person", "mail", "social_media")
SEVERITY_LEVELS = ("low", "medium", "high", "critical")
ROOT_CAUSE_CATEGORIES = (
    "process_failure", "system_error", "staff_conduct", "communication_failure",
    "product_design", "third_party_provider", "policy_or_pricing", "other",
)
COMPLAINT_CATEGORIES = ("standard", "credit_default_notice", "superannuation_trustee")


class ComplaintsError(Exception):
    """Raised for any invalid complaints-workflow operation. Every raise
    path here is a refusal, never a silent skip of a required step."""


@dataclass(frozen=True)
class IDRTimeframePolicy:
    policy_version: str
    effective_date: str
    source: str
    acknowledgement_business_days: int
    standard_response_days: int              # calendar days
    credit_default_notice_response_days: int  # calendar days
    superannuation_trustee_response_days: int  # calendar days


# Sourced from ASIC RG 271, effective 5 October 2021 — see module
# docstring. This is the pilot's default; a firm could adopt a stricter
# (lower) internal target, but never a looser one than RG 271 permits.
# Changing this requires a new IDRTimeframePolicy object with its own
# version/effective_date, never editing this one in place.
DEFAULT_IDR_TIMEFRAME_POLICY = IDRTimeframePolicy(
    policy_version="idr-policy-rg271-v1",
    effective_date="2021-10-05",
    source="ASIC Regulatory Guide 271 (RG 271), effective 5 October 2021",
    acknowledgement_business_days=1,
    standard_response_days=30,
    credit_default_notice_response_days=21,
    superannuation_trustee_response_days=45,
)


def _response_days_for_category(category: str, policy: IDRTimeframePolicy) -> int:
    if category == "credit_default_notice":
        return policy.credit_default_notice_response_days
    if category == "superannuation_trustee":
        return policy.superannuation_trustee_response_days
    return policy.standard_response_days


def _add_business_days(start: datetime, business_days: int) -> datetime:
    """Simple Mon-Fri business-day addition, no public-holiday calendar
    (documented limitation — a real deployment should plug in an AU
    public-holiday calendar; this is conservative in the sense that it
    never UNDER-counts a deadline, since ignoring holidays only makes
    the computed deadline earlier than the true regulatory deadline,
    which is the safe direction to be wrong in)."""
    d = start
    added = 0
    while added < business_days:
        d += timedelta(days=1)
        if d.weekday() < 5:  # Mon-Fri
            added += 1
    return d


def compute_due_date(received_at: datetime, category: str, policy: IDRTimeframePolicy = DEFAULT_IDR_TIMEFRAME_POLICY) -> str:
    days = _response_days_for_category(category, policy)
    return (received_at + timedelta(days=days)).isoformat()


def compute_acknowledgement_due(received_at: datetime, policy: IDRTimeframePolicy = DEFAULT_IDR_TIMEFRAME_POLICY) -> str:
    return _add_business_days(received_at, policy.acknowledgement_business_days).isoformat()


async def intake_complaint(
    customer_id: str, channel: str, description: str, category: str,
    severity: str, vulnerability_indicators: list, received_by: str,
    credit_decision_id: Optional[str] = None, bill_id: Optional[str] = None,
    disbursement_id: Optional[str] = None, application_id: Optional[str] = None,
    policy: IDRTimeframePolicy = DEFAULT_IDR_TIMEFRAME_POLICY,
) -> dict:
    """Intake through any accessible channel. Every complaint gets a
    computed response due-date and acknowledgement-due date from the
    policy at intake time — not left to be worked out later — and the
    policy version used is recorded on the complaint itself, so a
    later dispute about "was this handled on time" can always be
    checked against exactly what was in force when the complaint came
    in, even if the policy changes afterward."""
    if channel not in COMPLAINT_CHANNELS:
        raise ComplaintsError(f"unknown channel: {channel}")
    if severity not in SEVERITY_LEVELS:
        raise ComplaintsError(f"unknown severity: {severity}")
    if category not in COMPLAINT_CATEGORIES:
        raise ComplaintsError(f"unknown category: {category}")

    now = datetime.now(timezone.utc)
    row = {
        "customer_id": customer_id, "channel": channel, "description": description,
        "category": category, "severity": severity,
        "vulnerability_indicators": list(vulnerability_indicators),
        "status": "open", "stage": "received",
        "credit_decision_id": credit_decision_id, "bill_id": bill_id,
        "disbursement_id": disbursement_id, "application_id": application_id,
        "received_by": received_by, "received_at": now.isoformat(),
        "acknowledgement_due_at": compute_acknowledgement_due(now, policy),
        "response_due_at": compute_due_date(now, category, policy),
        "policy_version": policy.policy_version,
        "root_cause_category": None, "afca_reference_number": None,
        "escalated_to_afca_at": None,
    }
    complaint = await sdb.insert_one("complaints", row)
    await _append_audit(complaint["id"], "intake", received_by, f"channel={channel} category={category}", {}, row)
    return complaint


async def _append_audit(complaint_id: str, action: str, actor: str, reason: str, previous_state: dict, new_state: dict) -> None:
    await sdb.insert_one("complaint_audit_log", {
        "complaint_id": complaint_id, "action": action, "actor": actor, "reason": reason,
        "previous_state": previous_state, "new_state": new_state, "timestamp": datetime.now(timezone.utc).isoformat(),
    })


async def acknowledge_complaint(complaint_id: str, acknowledged_by: str) -> dict:
    complaint = await sdb.find_one("complaints", {"id": complaint_id})
    if not complaint:
        raise ComplaintsError(f"no complaint {complaint_id}")
    now = datetime.now(timezone.utc)
    due = datetime.fromisoformat(complaint["acknowledgement_due_at"])
    late = now > due
    updates = {"stage": "acknowledged", "acknowledged_by": acknowledged_by, "acknowledged_at": now.isoformat(), "acknowledgement_late": late}
    await sdb.update_one("complaints", {"id": complaint_id}, updates)
    await _append_audit(complaint_id, "acknowledged", acknowledged_by, f"late={late}", complaint, updates)
    return {**complaint, **updates}


async def assign_owner(complaint_id: str, owner: str, assigned_by: str) -> dict:
    complaint = await sdb.find_one("complaints", {"id": complaint_id})
    if not complaint:
        raise ComplaintsError(f"no complaint {complaint_id}")
    updates = {"owner": owner, "stage": "investigating" if complaint.get("stage") in ("received", "acknowledged") else complaint.get("stage")}
    await sdb.update_one("complaints", {"id": complaint_id}, updates)
    await _append_audit(complaint_id, "owner_assigned", assigned_by, f"owner={owner}", complaint, updates)
    return {**complaint, **updates}


async def add_investigation_note(complaint_id: str, note: str, added_by: str) -> dict:
    complaint = await sdb.find_one("complaints", {"id": complaint_id})
    if not complaint:
        raise ComplaintsError(f"no complaint {complaint_id}")
    notes = list(complaint.get("investigation_notes", []))
    notes.append({"note": note, "added_by": added_by, "timestamp": datetime.now(timezone.utc).isoformat()})
    updates = {"investigation_notes": notes}
    await sdb.update_one("complaints", {"id": complaint_id}, updates)
    return {**complaint, **updates}


async def record_customer_communication(complaint_id: str, direction: str, summary: str, communicated_by: str) -> dict:
    if direction not in ("inbound", "outbound"):
        raise ComplaintsError("direction must be 'inbound' or 'outbound'")
    complaint = await sdb.find_one("complaints", {"id": complaint_id})
    if not complaint:
        raise ComplaintsError(f"no complaint {complaint_id}")
    comms = list(complaint.get("communications", []))
    comms.append({"direction": direction, "summary": summary, "communicated_by": communicated_by,
                   "timestamp": datetime.now(timezone.utc).isoformat()})
    updates = {"communications": comms}
    await sdb.update_one("complaints", {"id": complaint_id}, updates)
    return {**complaint, **updates}


async def propose_remedy(complaint_id: str, remedy_description: str, compensation_amount: Decimal, proposed_by: str) -> dict:
    complaint = await sdb.find_one("complaints", {"id": complaint_id})
    if not complaint:
        raise ComplaintsError(f"no complaint {complaint_id}")
    remedy = await sdb.insert_one("complaint_remedies", {
        "complaint_id": complaint_id, "description": remedy_description, "compensation_amount": str(compensation_amount),
        "proposed_by": proposed_by, "status": "proposed", "created_at": datetime.now(timezone.utc).isoformat(),
    })
    await sdb.update_one("complaints", {"id": complaint_id}, {"stage": "remedy_proposed"})
    return remedy


async def approve_remedy(remedy_id: str, approved_by: str) -> dict:
    """A remedy involving real compensation is a real-money decision —
    requires an approver distinct from the proposer, same maker-checker
    discipline as every other financial decision point in this
    codebase (config changes, credit activation, hardship arrangements,
    etc.)."""
    remedy = await sdb.find_one("complaint_remedies", {"id": remedy_id})
    if not remedy:
        raise ComplaintsError(f"no remedy {remedy_id}")
    if remedy.get("proposed_by") == approved_by:
        raise ComplaintsError("remedy approval requires a distinct approver from the proposer (maker-checker)")
    if remedy.get("status") != "proposed":
        raise ComplaintsError(f"remedy {remedy_id} is not awaiting approval (status={remedy.get('status')})")
    updates = {"status": "approved", "approved_by": approved_by, "approved_at": datetime.now(timezone.utc).isoformat()}
    await sdb.update_one("complaint_remedies", {"id": remedy_id}, updates)
    return {**remedy, **updates}


async def resolve_complaint(complaint_id: str, outcome: str, root_cause_category: str, resolved_by: str, resolution_notes: str) -> dict:
    if root_cause_category not in ROOT_CAUSE_CATEGORIES:
        raise ComplaintsError(f"unknown root_cause_category: {root_cause_category}")
    complaint = await sdb.find_one("complaints", {"id": complaint_id})
    if not complaint:
        raise ComplaintsError(f"no complaint {complaint_id}")

    now = datetime.now(timezone.utc)
    due = datetime.fromisoformat(complaint["response_due_at"])
    updates = {
        "status": "closed", "stage": "resolved", "outcome": outcome, "root_cause_category": root_cause_category,
        "resolution_notes": resolution_notes, "resolved_by": resolved_by, "resolved_at": now.isoformat(),
        "resolved_late": now > due,
    }
    await sdb.update_one("complaints", {"id": complaint_id}, updates)
    await _append_audit(complaint_id, "resolved", resolved_by, f"outcome={outcome} root_cause={root_cause_category}", complaint, updates)
    return {**complaint, **updates}


def needs_delay_notification(complaint: dict, now: Optional[datetime] = None) -> bool:
    """RG 271 requires an 'IDR delay notification' to the complainant,
    explaining the right to escalate to AFCA, if the response deadline
    is going to be (or has been) missed. This function only detects the
    condition — it does not send anything (no notification channel is
    wired up in this codebase yet); a caller/scheduled job would use
    this to decide when a human needs to send one."""
    now = now or datetime.now(timezone.utc)
    if complaint.get("status") == "closed":
        return False
    due = datetime.fromisoformat(complaint["response_due_at"])
    return now > due


async def escalate_to_afca(complaint_id: str, escalated_by: str, reason: str, afca_reference_number: Optional[str] = None) -> dict:
    """Records that a complaint has been escalated to AFCA. Does NOT
    call any AFCA system — see module docstring for why no such
    integration exists to call. `afca_reference_number` is optional at
    the time of escalation (AFCA issues it after the complaint is
    lodged in their consumer portal, visible to the firm in the member
    portal afterward) — pass it in a follow-up update once known via
    a direct database update or a future
    `record_afca_reference_number()` helper, not required here."""
    complaint = await sdb.find_one("complaints", {"id": complaint_id})
    if not complaint:
        raise ComplaintsError(f"no complaint {complaint_id}")
    if not reason or not reason.strip():
        raise ComplaintsError("AFCA escalation requires a documented reason")

    updates = {
        "stage": "escalated_to_afca", "escalated_to_afca_at": datetime.now(timezone.utc).isoformat(),
        "escalated_by": escalated_by, "afca_reference_number": afca_reference_number,
    }
    await sdb.update_one("complaints", {"id": complaint_id}, updates)
    await _append_audit(complaint_id, "escalated_to_afca", escalated_by, reason, complaint, updates)
    return {**complaint, **updates}


def root_cause_report(complaints: list) -> dict:
    """Aggregate root-cause counts across a set of resolved complaints,
    for the management/regulatory reporting task section 10 asks for.
    Pure function over already-fetched data — no DB access itself, so
    it's trivially testable and reusable for any complaint subset a
    caller wants to report on (a date range, a product line, etc.)."""
    counts = {c: 0 for c in ROOT_CAUSE_CATEGORIES}
    for complaint in complaints:
        cause = complaint.get("root_cause_category")
        if cause in counts:
            counts[cause] += 1
    return counts
