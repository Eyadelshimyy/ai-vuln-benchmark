"""
Standalone tests for hardship_collections.py. Exercises it against real
pilot_config + credit_ledger, same as test_credit_ledger.py, so
record_repayment()'s ledger integration is genuinely proven rather than
assumed. Same in-memory fake-DB pattern as the other test_*.py files.

Run: python3 test_hardship_collections.py
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


async def find_many(table, filters=None, exclude_fields=None, limit=10000):
    filters = filters or {}
    return [dict(r) for r in _tables.get(table, []) if _matches(r, filters)][:limit]


async def insert_one(table, data):
    row = dict(data)
    row.setdefault("id", str(uuid.uuid4()))
    _tables.setdefault(table, []).append(row)
    return dict(row)


async def insert_many(table, rows):
    return [await insert_one(table, r) for r in rows]


async def update_one(table, filters, updates):
    for row in _tables.get(table, []):
        if _matches(row, filters):
            row.update(updates)
            return True
    return False


async def update_many(table, filters, updates):
    n = 0
    for row in _tables.get(table, []):
        if _matches(row, filters):
            row.update(updates)
            n += 1
    return n


fake_sdb = types.SimpleNamespace(
    find_one=find_one, find_many=find_many, insert_one=insert_one,
    insert_many=insert_many, update_one=update_one, update_many=update_many,
)
sys.modules["supabase_db"] = fake_sdb

import pilot_config as pc            # noqa: E402
import credit_ledger as cl           # noqa: E402
import hardship_collections as hc    # noqa: E402

FAILURES = []


def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}")
    if not condition:
        FAILURES.append(label)


async def main():
    await pc.propose_config_version(pc.PilotConfig(), proposed_by="alice", approved_by="bob", activate=True)
    account = await cl.activate_customer_credit_account("cust-1", Decimal("2500.00"), 0, Decimal("0"), "admin1", "admin2")
    # Draw the full 2500 so there's a real outstanding balance to build a schedule against.
    await cl.draw_credit("cust-1", Decimal("500.00"), "bill-1", "admin1")
    await cl.draw_credit("cust-1", Decimal("500.00"), "bill-2", "admin1")
    outstanding = await cl.get_outstanding_principal("cust-1")
    check("cust-1 has $1000 outstanding to build a schedule against", outstanding == Decimal("1000.00"))

    # ---------------------------------------------------------------
    # Schedule generation: exact sum, no rounding drift
    # ---------------------------------------------------------------
    schedule = await hc.generate_repayment_schedule("cust-1", Decimal("1000.00"), 3, "2026-09-01", created_by="admin1")
    check("schedule has 3 installments for a 3-month term", len(schedule["installments"]) == 3)
    total = sum(Decimal(i["scheduled_amount"]) for i in schedule["installments"])
    check("installments sum exactly to the principal (no rounding drift)", total == Decimal("1000.00"))
    check("the first two installments are equal thirds", schedule["installments"][0]["scheduled_amount"] == schedule["installments"][1]["scheduled_amount"])

    # ---------------------------------------------------------------
    # record_repayment: real ledger integration, classification
    # ---------------------------------------------------------------
    first_installment = schedule["installments"][0]
    scheduled_amount = Decimal(first_installment["scheduled_amount"])  # 333.33
    half_1 = Decimal("150.00")
    half_2 = scheduled_amount - half_1  # exact remainder, avoids any rounding ambiguity

    partial_result = await hc.record_repayment(first_installment["id"], "cust-1", half_1, recorded_by="admin1")
    check("a partial payment is classified as 'partial'", partial_result["status"] == "partial")
    outstanding_after_partial = await cl.get_outstanding_principal("cust-1")
    check("a partial repayment actually reduced ledger outstanding principal (real money moved)",
          outstanding_after_partial == Decimal("1000.00") - half_1)

    top_up = await hc.record_repayment(first_installment["id"], "cust-1", half_2, recorded_by="admin1")
    check("topping up to the full scheduled amount (cumulative) moves status to 'paid'", top_up["status"] == "paid")
    check("amount_paid reflects the CUMULATIVE total across both partial payments, not just the last one",
          Decimal(top_up["amount_paid"]) == scheduled_amount)

    try:
        await hc.record_repayment(first_installment["id"], "cust-1", Decimal("1.00"), recorded_by="admin1")
        check("rejects a repayment against an already-fully-paid installment", False)
    except hc.HardshipCollectionsError:
        check("rejects a repayment against an already-fully-paid installment", True)

    # ---------------------------------------------------------------
    # record_repayment propagates a real ledger overpayment error and
    # leaves the installment untouched
    # ---------------------------------------------------------------
    second_installment = schedule["installments"][1]
    outstanding_before_overpay_attempt = await cl.get_outstanding_principal("cust-1")
    try:
        await hc.record_repayment(second_installment["id"], "cust-1", Decimal("999999.00"), recorded_by="admin1")
        check("propagates a real credit_ledger overpayment error rather than swallowing it", False)
    except cl.CreditLedgerError:
        check("propagates a real credit_ledger overpayment error rather than swallowing it", True)

    still_scheduled = await find_one("repayment_installments", {"id": second_installment["id"]})
    check("a rejected overpayment attempt leaves the installment status untouched ('scheduled')", still_scheduled["status"] == "scheduled")
    outstanding_unchanged = await cl.get_outstanding_principal("cust-1")
    check("a rejected overpayment attempt moves zero additional money", outstanding_unchanged == outstanding_before_overpay_attempt)

    # ---------------------------------------------------------------
    # record_failed_repayment: no fee, no auto-escalation
    # ---------------------------------------------------------------
    third_installment = schedule["installments"][2]
    failed = await hc.record_failed_repayment(third_installment["id"], "card declined", recorded_by="admin1")
    check("a failed repayment is marked 'failed'", failed["status"] == "failed")
    check("a failed installment record has no fee field at all (structurally cannot carry one)", "fee_amount" not in failed and "late_fee" not in failed)

    # ---------------------------------------------------------------
    # reschedule_installment: policy-bounded, no charge
    # ---------------------------------------------------------------
    resched = await hc.reschedule_installment(third_installment["id"], "2026-12-01", "customer requested", requested_by="admin1")
    check("first reschedule succeeds", resched["reschedule_count"] == 1)
    for _ in range(hc.MAX_RESCHEDULES_PER_INSTALLMENT - 1):
        resched = await hc.reschedule_installment(third_installment["id"], "2027-01-01", "again", requested_by="admin1")
    check(f"reschedule count reaches the policy limit ({hc.MAX_RESCHEDULES_PER_INSTALLMENT})", resched["reschedule_count"] == hc.MAX_RESCHEDULES_PER_INSTALLMENT)

    try:
        await hc.reschedule_installment(third_installment["id"], "2027-02-01", "one more", requested_by="admin1")
        check("blocks rescheduling beyond the policy limit", False)
    except hc.HardshipCollectionsError:
        check("blocks rescheduling beyond the policy limit", True)

    # ---------------------------------------------------------------
    # request_hardship: works with ZERO payment-status gating
    # ---------------------------------------------------------------
    # cust-2 has never made any payment at all, has no credit account,
    # no schedule -- confirm hardship intake still succeeds.
    hardship_case = await hc.request_hardship("cust-2-never-paid-anything", "job loss", ["financial_hardship_disclosed"], requested_by="cust-2-never-paid-anything")
    check("hardship request succeeds for a customer with zero payment history / no schedule at all", hardship_case["status"] == "open")

    # Even a customer with an active failed installment can request hardship.
    hardship_for_failed = await hc.request_hardship("cust-1", "reduced income", [], requested_by="cust-1")
    check("hardship request succeeds even for a customer with a currently-failed installment", hardship_for_failed["status"] == "open")

    # ---------------------------------------------------------------
    # pause_collections: maker-checker
    # ---------------------------------------------------------------
    future = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
    try:
        await hc.pause_collections(hardship_for_failed["id"], paused_by="agent1", approved_by="agent1", pause_until=future)
        check("rejects a collection pause approved by the same person who requested it", False)
    except hc.HardshipCollectionsError:
        check("rejects a collection pause approved by the same person who requested it", True)

    pause = await hc.pause_collections(hardship_for_failed["id"], paused_by="agent1", approved_by="agent2", pause_until=future)
    check("collection pause succeeds with a distinct approver", pause["active"] is True)

    is_paused = await hc.is_collection_paused("cust-1")
    check("is_collection_paused reports True while an active pause is in effect", is_paused is True)

    is_not_paused_other_customer = await hc.is_collection_paused("cust-with-no-pause")
    check("is_collection_paused reports False for a customer with no pause", is_not_paused_other_customer is False)

    # ---------------------------------------------------------------
    # Hardship arrangements: maker-checker + fee/interest suppression
    # ---------------------------------------------------------------
    try:
        await hc.propose_hardship_arrangement(hardship_for_failed["id"],
                                               [{"due_date": "2027-01-01", "amount": "100.00", "fee_amount": "5.00"}],
                                               proposed_by="agent1")
        check("rejects a proposed arrangement carrying a nonzero fee", False)
    except hc.HardshipCollectionsError:
        check("rejects a proposed arrangement carrying a nonzero fee", True)

    try:
        await hc.propose_hardship_arrangement(hardship_for_failed["id"],
                                               [{"due_date": "2027-01-01", "amount": "100.00", "interest_amount": "2.00"}],
                                               proposed_by="agent1")
        check("rejects a proposed arrangement carrying nonzero interest", False)
    except hc.HardshipCollectionsError:
        check("rejects a proposed arrangement carrying nonzero interest", True)

    clean_arrangement = await hc.propose_hardship_arrangement(
        hardship_for_failed["id"], [{"due_date": "2027-01-01", "amount": "100.00"}], proposed_by="agent1")
    check("a clean (no fee/interest) arrangement proposal succeeds", clean_arrangement["status"] == "proposed")

    try:
        await hc.approve_hardship_arrangement(clean_arrangement["id"], approved_by="agent1")
        check("rejects arrangement approval by the same person who proposed it", False)
    except hc.HardshipCollectionsError:
        check("rejects arrangement approval by the same person who proposed it", True)

    approved_arrangement = await hc.approve_hardship_arrangement(clean_arrangement["id"], approved_by="agent2")
    check("arrangement approval succeeds with a distinct approver", approved_arrangement["status"] == "approved")

    updated_case = await find_one("hardship_cases", {"id": hardship_for_failed["id"]})
    check("the hardship case moves to 'arrangement_active' once approved", updated_case["status"] == "arrangement_active")

    # ---------------------------------------------------------------
    # escalate_hardship_case: always requires a human + a documented reason
    # ---------------------------------------------------------------
    try:
        await hc.escalate_hardship_case(hardship_case["id"], escalated_by="agent1", reason="")
        check("rejects escalation with an empty reason", False)
    except hc.HardshipCollectionsError:
        check("rejects escalation with an empty reason", True)

    escalated = await hc.escalate_hardship_case(hardship_case["id"], escalated_by="agent1", reason="repeated missed contact")
    check("escalation with a documented reason succeeds", escalated["status"] == "escalated")
    check("escalation is recorded in the case's audit history", len(escalated["escalation_history"]) == 1)

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
