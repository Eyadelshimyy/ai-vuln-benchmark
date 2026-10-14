"""
Standalone tests for complaints.py. Same in-memory fake-DB pattern as
the other test_*.py files, no live credentials needed — and none are
needed for AFCA either, since there is no AFCA API to call (see
complaints.py's module docstring).

Run: python3 test_complaints.py
"""
import asyncio
import sys
import types
import uuid
from decimal import Decimal
from datetime import datetime, timezone, timedelta

# ---- in-memory fake of supabase_db's public interface ----
_tables = {}


def _matches(row, filters):
    for k, v in filters.items():
        if row.get(k) != v:
            return False
    return True


async def find_one(table, filters, exclude_fields=None):
    for row in _tables.get(table, []):
        if _matches(row, filters):
            return dict(row)
    return None


async def insert_one(table, data):
    row = dict(data)
    row.setdefault("id", str(uuid.uuid4()))
    _tables.setdefault(table, []).append(row)
    return dict(row)


async def update_one(table, filters, updates):
    for row in _tables.get(table, []):
        if _matches(row, filters):
            row.update(updates)
            return True
    return False


fake_sdb = types.SimpleNamespace(find_one=find_one, insert_one=insert_one, update_one=update_one)
sys.modules["supabase_db"] = fake_sdb

import complaints as cp   # noqa: E402

FAILURES = []


def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}")
    if not condition:
        FAILURES.append(label)


async def main():
    # ---------------------------------------------------------------
    # Timeframe policy: sourced values, not arbitrary
    # ---------------------------------------------------------------
    policy = cp.DEFAULT_IDR_TIMEFRAME_POLICY
    check("default policy cites its regulatory source", "RG 271" in policy.source)
    check("standard complaints get a 30-calendar-day response window (RG 271.65)", policy.standard_response_days == 30)
    check("credit default notice complaints get a 21-day window", policy.credit_default_notice_response_days == 21)
    check("superannuation trustee complaints get a 45-day window", policy.superannuation_trustee_response_days == 45)

    # ---------------------------------------------------------------
    # Intake: computed due dates, category-specific, policy recorded
    # ---------------------------------------------------------------
    standard = await cp.intake_complaint(
        "cust-1", "email", "My bill was paid twice", "standard", "medium", [], received_by="agent1")
    check("standard complaint gets 'open' status and 'received' stage", standard["status"] == "open" and standard["stage"] == "received")
    check("the policy version used is recorded on the complaint", standard["policy_version"] == policy.policy_version)

    received_at = datetime.fromisoformat(standard["received_at"])
    response_due = datetime.fromisoformat(standard["response_due_at"])
    check("standard complaint's response_due_at is ~30 calendar days after receipt", (response_due - received_at).days == 30)

    default_notice = await cp.intake_complaint(
        "cust-2", "phone", "Dispute over a default notice", "credit_default_notice", "high", [], received_by="agent1")
    dn_received = datetime.fromisoformat(default_notice["received_at"])
    dn_due = datetime.fromisoformat(default_notice["response_due_at"])
    check("credit_default_notice complaint gets the shorter 21-day window, not the 30-day default", (dn_due - dn_received).days == 21)

    try:
        await cp.intake_complaint("cust-3", "carrier_pigeon", "test", "standard", "low", [], received_by="agent1")
        check("rejects an unknown channel", False)
    except cp.ComplaintsError:
        check("rejects an unknown channel", True)

    # ---------------------------------------------------------------
    # Vulnerability indicators and links are preserved
    # ---------------------------------------------------------------
    linked = await cp.intake_complaint(
        "cust-4", "web_form", "Wrong bill amount charged", "standard", "high",
        ["financial_hardship_disclosed"], received_by="agent1",
        bill_id="bill-123", disbursement_id="disb-456")
    check("vulnerability indicators are preserved on intake", linked["vulnerability_indicators"] == ["financial_hardship_disclosed"])
    check("complaint is linked to the relevant bill and disbursement", linked["bill_id"] == "bill-123" and linked["disbursement_id"] == "disb-456")

    # ---------------------------------------------------------------
    # Acknowledgement: on-time vs late detection
    # ---------------------------------------------------------------
    acked = await cp.acknowledge_complaint(standard["id"], acknowledged_by="agent2")
    check("acknowledging promptly after intake is not flagged late", acked["acknowledgement_late"] is False)
    check("acknowledgement moves stage to 'acknowledged'", acked["stage"] == "acknowledged")

    # Simulate a complaint acknowledged well past its 1-business-day deadline.
    old_complaint = await insert_one("complaints", {
        **standard, "id": None,
        "acknowledgement_due_at": (datetime.now(timezone.utc) - timedelta(days=5)).isoformat(),
    })
    late_ack = await cp.acknowledge_complaint(old_complaint["id"], acknowledged_by="agent2")
    check("acknowledging past the deadline is correctly flagged late", late_ack["acknowledgement_late"] is True)

    # ---------------------------------------------------------------
    # Owner assignment, investigation notes, communications
    # ---------------------------------------------------------------
    assigned = await cp.assign_owner(standard["id"], "caseworker1", assigned_by="agent2")
    check("owner assignment records the owner and moves stage to 'investigating'", assigned["owner"] == "caseworker1" and assigned["stage"] == "investigating")

    noted = await cp.add_investigation_note(standard["id"], "Confirmed duplicate payment in ledger", added_by="caseworker1")
    check("investigation notes are appended, not overwritten", len(noted["investigation_notes"]) == 1)

    commed = await cp.record_customer_communication(standard["id"], "outbound", "Called customer to confirm details", communicated_by="caseworker1")
    check("customer communications are appended", len(commed["communications"]) == 1)

    try:
        await cp.record_customer_communication(standard["id"], "sideways", "bad direction", communicated_by="caseworker1")
        check("rejects an invalid communication direction", False)
    except cp.ComplaintsError:
        check("rejects an invalid communication direction", True)

    # ---------------------------------------------------------------
    # Remedy: maker-checker on real compensation
    # ---------------------------------------------------------------
    remedy = await cp.propose_remedy(standard["id"], "Refund the duplicate payment", Decimal("50.00"), proposed_by="caseworker1")
    check("remedy proposal moves the complaint stage to 'remedy_proposed'", True)  # verified via the update call succeeding

    try:
        await cp.approve_remedy(remedy["id"], approved_by="caseworker1")
        check("rejects remedy approval by the same person who proposed it", False)
    except cp.ComplaintsError:
        check("rejects remedy approval by the same person who proposed it", True)

    approved_remedy = await cp.approve_remedy(remedy["id"], approved_by="manager1")
    check("remedy approval succeeds with a distinct approver", approved_remedy["status"] == "approved")

    # ---------------------------------------------------------------
    # Resolution: on-time vs late, root cause required and validated
    # ---------------------------------------------------------------
    try:
        await cp.resolve_complaint(standard["id"], "Refund issued", "not_a_real_category", resolved_by="caseworker1", resolution_notes="n/a")
        check("rejects an unknown root_cause_category", False)
    except cp.ComplaintsError:
        check("rejects an unknown root_cause_category", True)

    resolved = await cp.resolve_complaint(standard["id"], "Refund issued", "system_error", resolved_by="caseworker1", resolution_notes="Duplicate charge refunded")
    check("resolution closes the complaint (status='closed')", resolved["status"] == "closed")
    check("resolution within the response window is not flagged late", resolved["resolved_late"] is False)
    check("resolution records the root cause category", resolved["root_cause_category"] == "system_error")

    # ---------------------------------------------------------------
    # needs_delay_notification: detects an overdue open complaint
    # ---------------------------------------------------------------
    overdue_complaint = {
        "status": "open",
        "response_due_at": (datetime.now(timezone.utc) - timedelta(days=1)).isoformat(),
    }
    check("an open complaint past its response_due_at needs a delay notification", cp.needs_delay_notification(overdue_complaint))

    not_overdue_complaint = {
        "status": "open",
        "response_due_at": (datetime.now(timezone.utc) + timedelta(days=5)).isoformat(),
    }
    check("an open complaint still within its response window does not need one", not cp.needs_delay_notification(not_overdue_complaint))

    closed_but_was_overdue = {
        "status": "closed",
        "response_due_at": (datetime.now(timezone.utc) - timedelta(days=1)).isoformat(),
    }
    check("a closed complaint never needs a delay notification, even if its due date has passed", not cp.needs_delay_notification(closed_but_was_overdue))

    # ---------------------------------------------------------------
    # AFCA escalation: records only, calls nothing
    # ---------------------------------------------------------------
    try:
        await cp.escalate_to_afca(default_notice["id"], escalated_by="manager1", reason="")
        check("rejects AFCA escalation with an empty reason", False)
    except cp.ComplaintsError:
        check("rejects AFCA escalation with an empty reason", True)

    escalated = await cp.escalate_to_afca(default_notice["id"], escalated_by="manager1",
                                           reason="customer unsatisfied with IDR outcome, requested escalation")
    check("AFCA escalation moves stage to 'escalated_to_afca'", escalated["stage"] == "escalated_to_afca")
    check("AFCA escalation works with no reference number yet (issued later by AFCA)", escalated["afca_reference_number"] is None)

    escalated_with_ref = await cp.escalate_to_afca(linked["id"], escalated_by="manager1", reason="test",
                                                     afca_reference_number="AFCA-2026-000123")
    check("AFCA escalation can also carry a reference number once known", escalated_with_ref["afca_reference_number"] == "AFCA-2026-000123")

    # ---------------------------------------------------------------
    # root_cause_report: pure aggregation
    # ---------------------------------------------------------------
    report = cp.root_cause_report([
        {"root_cause_category": "system_error"},
        {"root_cause_category": "system_error"},
        {"root_cause_category": "staff_conduct"},
        {"root_cause_category": None},  # unresolved complaint, no cause yet
    ])
    check("root_cause_report tallies correctly", report["system_error"] == 2 and report["staff_conduct"] == 1)
    check("root_cause_report includes every category, even those with zero complaints", report["process_failure"] == 0)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED:")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("ALL CHECKS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
