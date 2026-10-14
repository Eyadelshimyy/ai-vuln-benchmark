"""
backend/launch_gates.py
=========================
Fail-closed regulatory launch-gate service for the ASIC ERS pilot.

Production credit and payments (real money moving for a real customer)
must remain DISABLED unless every mandatory gate below is recorded,
evidenced, reviewed by someone other than its creator, and currently
unexpired. This module is the single source of truth for that decision —
every place in the app that would move real money or activate a real
customer must call `is_production_authorized()` and refuse to proceed on
anything other than an explicit True.

Fail-closed rules enforced here (do not weaken these without a compliance
sign-off, and even then not by editing this file quietly):

- Default state for a gate that doesn't exist yet is CLOSED. Absence of a
  row is not treated as "not applicable" — it blocks production the same
  as an explicit "failed" status.
- Evidence with an expiry_date in the past auto-closes the gate the next
  time it's read, without waiting for anyone to notice. There is no
  "still probably fine" grace period.
- The same person can never both create (submit evidence for) and approve
  the same gate.
- Full production activation requires TWO distinct authorised approvers,
  neither of whom is the person who prepared the activation request.
- If ANY mandatory gate is not APPROVED (or has just auto-expired), new
  lending/new customer activation must stop immediately — this module
  does not wait for a cron job or a human to flip a switch.
- Existing customers are never simply cut off — `is_production_authorized
  () == False` blocks NEW lending only. Statements, complaints, and
  hardship support must remain reachable via the wind-down path
  regardless of gate status; enforcing that at the UI/API layer is the
  caller's job, but this module never returns a signal that would justify
  blocking those specifically.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import supabase_db as sdb

logger = logging.getLogger(__name__)


# The 22 mandatory gates from the ERS readiness scope. Keys are stable
# identifiers used as `gate_key` in the DB — do not rename without a
# migration, audit trails reference these.
MANDATORY_GATES = {
    "asic_ers_financial_services_exemption": "ASIC ERS financial-services exemption available",
    "asic_ers_credit_exemption": "ASIC ERS credit exemption available",
    "ers_commencement_date_reached": "Applicable ERS commencement date reached",
    "afca_membership_active": "AFCA membership active",
    "pi_insurance_active": "Professional indemnity insurance active",
    "pi_runoff_cover_confirmed": "Required 12-month run-off cover confirmed",
    "customer_funds_account_established": "Separate customer-funds account established with an ADI",
    "credit_funding_account_established": "Credit-funding account established",
    "operating_account_established": "Operating account established",
    "au_regulatory_legal_opinion_received": "Australian regulatory legal opinion received",
    "customer_agreements_approved": "Customer agreements approved",
    "credit_and_disclosures_approved": "Credit and financial-product disclosures approved",
    "target_market_determinations_approved": "Target Market Determinations approved",
    "privacy_and_consent_docs_approved": "Privacy and consent documents approved",
    "production_security_assessment_passed": "Production security assessment passed",
    "penetration_testing_passed": "Penetration testing passed",
    "bank_feed_integration_verified": "Bank-feed integration verified",
    "reconciliation_testing_passed": "Reconciliation testing passed",
    "responsible_lending_workflow_approved": "Responsible-lending workflow approved",
    "incident_response_test_passed": "Incident-response test passed",
    "business_continuity_test_passed": "Business-continuity test passed",
    "wind_down_test_passed": "Wind-down test passed",
}

GATE_STATUSES = ("not_started", "evidence_submitted", "approved", "expired", "failed")


class LaunchGateError(Exception):
    """Raised for any invalid gate operation. Every raise path here is a
    refusal, never a partial success — there is no code path in this
    module that activates production and also raises."""


@dataclass
class GateRecord:
    gate_key: str
    description: str
    status: str
    owner: Optional[str] = None
    evidence_reference: Optional[str] = None
    reviewer: Optional[str] = None
    approval_date: Optional[str] = None
    expiry_date: Optional[str] = None
    id: Optional[str] = None


def _effective_status(row: dict, now: Optional[datetime] = None) -> str:
    """Applies auto-expiry on read: a row whose expiry_date has passed is
    treated as 'expired' regardless of what its stored status says. This
    keeps expiry enforcement correct even if a background job that would
    otherwise flip the stored status hasn't run yet."""
    now = now or datetime.now(timezone.utc)
    expiry = row.get("expiry_date")
    if expiry:
        expiry_dt = expiry if isinstance(expiry, datetime) else datetime.fromisoformat(str(expiry))
        if expiry_dt.tzinfo is None:
            expiry_dt = expiry_dt.replace(tzinfo=timezone.utc)
        if expiry_dt <= now and row.get("status") == "approved":
            return "expired"
    return row.get("status", "not_started")


async def _append_audit(gate_key: str, action: str, actor: str, reason: str, previous_state: dict, new_state: dict) -> None:
    await sdb.insert_one("launch_gate_audit_log", {
        "gate_key": gate_key,
        "action": action,
        "actor": actor,
        "reason": reason,
        "previous_state": previous_state,
        "new_state": new_state,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


async def submit_gate_evidence(gate_key: str, owner: str, evidence_reference: str, expiry_date: Optional[str] = None) -> dict:
    """Records that evidence has been prepared/submitted for a gate. Does
    NOT approve it — approval is a separate, distinct-person step."""
    if gate_key not in MANDATORY_GATES:
        raise LaunchGateError(f"unknown gate_key: {gate_key}")
    existing = await sdb.find_one("launch_gates", {"gate_key": gate_key})
    new_row = {
        "gate_key": gate_key,
        "description": MANDATORY_GATES[gate_key],
        "status": "evidence_submitted",
        "owner": owner,
        "evidence_reference": evidence_reference,
        "reviewer": None,
        "approval_date": None,
        "expiry_date": expiry_date,
    }
    if existing:
        await sdb.update_one("launch_gates", {"gate_key": gate_key}, new_row)
        result = {**existing, **new_row}
    else:
        result = await sdb.insert_one("launch_gates", new_row)
    await _append_audit(gate_key, "evidence_submitted", owner, "evidence submitted", existing or {}, new_row)
    return result


async def approve_gate(gate_key: str, reviewer: str, approval_date: Optional[str] = None) -> dict:
    """Approves a gate. Fails closed if the reviewer is the same person
    who submitted the evidence (maker-checker) or if there is no evidence
    to review yet."""
    if gate_key not in MANDATORY_GATES:
        raise LaunchGateError(f"unknown gate_key: {gate_key}")
    existing = await sdb.find_one("launch_gates", {"gate_key": gate_key})
    if not existing:
        raise LaunchGateError(f"gate {gate_key} has no submitted evidence yet — cannot approve")
    if existing.get("status") not in ("evidence_submitted",):
        raise LaunchGateError(f"gate {gate_key} is not awaiting approval (status={existing.get('status')})")
    if existing.get("owner") == reviewer:
        raise LaunchGateError("maker-checker violation: the reviewer must differ from the person who submitted evidence")

    approval_date = approval_date or datetime.now(timezone.utc).isoformat()
    updates = {"status": "approved", "reviewer": reviewer, "approval_date": approval_date}
    await sdb.update_one("launch_gates", {"gate_key": gate_key}, updates)
    new_state = {**existing, **updates}
    await _append_audit(gate_key, "approved", reviewer, "gate approved", existing, new_state)
    return new_state


async def fail_gate(gate_key: str, actor: str, reason: str) -> dict:
    if gate_key not in MANDATORY_GATES:
        raise LaunchGateError(f"unknown gate_key: {gate_key}")
    existing = await sdb.find_one("launch_gates", {"gate_key": gate_key}) or {"gate_key": gate_key}
    updates = {"status": "failed"}
    if existing.get("id"):
        await sdb.update_one("launch_gates", {"gate_key": gate_key}, updates)
    new_state = {**existing, **updates}
    await _append_audit(gate_key, "failed", actor, reason, existing, new_state)
    return new_state


async def get_all_gate_statuses() -> dict:
    """Returns {gate_key: effective_status} for every mandatory gate,
    defaulting missing gates to 'not_started' (i.e. closed). This is the
    view a readiness dashboard/scorecard should read from."""
    rows = await sdb.find_many("launch_gates", {})
    by_key = {r["gate_key"]: r for r in rows}
    result = {}
    for key in MANDATORY_GATES:
        row = by_key.get(key)
        result[key] = _effective_status(row) if row else "not_started"
    return result


async def is_production_authorized() -> bool:
    """The single fail-closed check every real-money code path must call.
    Returns True only if every mandatory gate is currently 'approved'
    (i.e. not expired, not failed, not missing). Any other condition
    returns False — there is no ambiguous/partial-True state."""
    statuses = await get_all_gate_statuses()
    return all(status == "approved" for status in statuses.values())


async def activate_production(requested_by: str, approver_1: str, approver_2: str, reason: str) -> bool:
    """Two-person activation of production/real-money functionality.
    Refuses (raises LaunchGateError) unless: all gates are approved,
    the two approvers are distinct from each other, and neither approver
    is the person who requested activation. On success, records an
    audit event and returns True — callers must still gate all real-money
    code paths on is_production_authorized(), this function does not
    itself flip any other state; it exists to force the two-person check
    to happen as one atomic, audited action rather than being assumed."""
    if approver_1 == approver_2:
        raise LaunchGateError("production activation requires two DISTINCT approvers")
    if requested_by in (approver_1, approver_2):
        raise LaunchGateError("the person requesting activation cannot also be one of the two approvers")

    authorized = await is_production_authorized()
    if not authorized:
        statuses = await get_all_gate_statuses()
        blocking = [k for k, v in statuses.items() if v != "approved"]
        raise LaunchGateError(f"cannot activate production — gates not approved: {blocking}")

    await sdb.insert_one("production_activation_events", {
        "requested_by": requested_by,
        "approver_1": approver_1,
        "approver_2": approver_2,
        "reason": reason,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    await _append_audit("__production_activation__", "activated", requested_by,
                         reason, {}, {"approver_1": approver_1, "approver_2": approver_2})
    logger.warning("PRODUCTION ACTIVATED by %s, approved by %s and %s", requested_by, approver_1, approver_2)
    return True


async def new_lending_allowed() -> bool:
    """Convenience alias for the specific decision of whether NEW credit
    can be extended or a new customer activated right now. Identical to
    is_production_authorized() today, kept as a separate name because the
    two questions (can we move money at all vs. can we grow exposure)
    are allowed to diverge in future without an API change here."""
    return await is_production_authorized()


async def existing_customers_route() -> str:
    """Existing customers must always be able to reach statements,
    complaints, and hardship support — this never returns anything that
    would justify blocking those. Returns 'normal' when gates are healthy,
    'wind_down' otherwise; both routes keep those three surfaces live."""
    return "normal" if await is_production_authorized() else "wind_down"
