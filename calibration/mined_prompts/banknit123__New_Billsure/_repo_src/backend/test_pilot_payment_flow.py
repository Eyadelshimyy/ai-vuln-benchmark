"""
Integration test for pilot_payment_flow.py — exercises
bill_verification.py, payment_permitted_use.py, credit_ledger.py, and
pilot_config.py TOGETHER, proving the whole real-money path actually
works end to end, not just that each module works alone. Same
in-memory fake-DB pattern as the other test_*.py files.

Run: python3 test_pilot_payment_flow.py
"""
import asyncio
import sys
import types
import uuid
from decimal import Decimal
from datetime import datetime, timezone

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
    out = []
    for data in rows:
        out.append(await insert_one(table, data))
    return out


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

import pilot_config as pc              # noqa: E402
import credit_ledger as cl             # noqa: E402
import bill_verification as bv         # noqa: E402
import payment_permitted_use as ppu    # noqa: E402
import pilot_payment_flow as flow      # noqa: E402

FAILURES = []


def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}")
    if not condition:
        FAILURES.append(label)


BILLER_ALLOWLIST = {"AusNet Electricity"}
APPROVED_CATEGORIES = {"electricity", "gas", "water", "telecommunications"}


async def main():
    await pc.propose_config_version(pc.PilotConfig(), proposed_by="alice", approved_by="bob", activate=True)

    # ---------------------------------------------------------------
    # Full happy path: verify a bill, activate credit, pay it
    # ---------------------------------------------------------------
    submission = bv.BillSubmission(
        customer_id="cust-1", file_bytes=b"bill A bytes",
        customer_name_on_account="Jane Citizen", customer_name_on_bill="Jane Citizen",
        biller_name_extracted="AusNet Electricity", biller_reference="REF-1",
        category="electricity", amount=Decimal("300.00"), due_date="2026-09-01",
        extraction_confidence=0.98, fraud_indicators=[],
    )
    bill = await bv.submit_and_verify_bill(submission, BILLER_ALLOWLIST, APPROVED_CATEGORIES)
    check("bill verifies cleanly", bill["verification_status"] == "verified")

    account = await cl.activate_customer_credit_account("cust-1", Decimal("2500.00"), 0, Decimal("0"), "admin1", "admin2")
    check("credit account activated for cust-1", account["status"] == "active")

    disbursement = await flow.pay_verified_bill(bill["id"], "cust-1", requested_by="admin3")
    check("pay_verified_bill returns a disbursement", disbursement["status"] == "queued")
    check("disbursement amount matches the bill amount", disbursement["amount"] == "300.00")
    check("disbursement is linked to a real credit journal id", disbursement.get("credit_journal_id") is not None)

    outstanding = await cl.get_outstanding_principal("cust-1")
    check("credit ledger outstanding principal actually increased by the payment (proves money moved, not just a queued row)",
          outstanding == Decimal("300.00"))

    bill_row = await find_one("pilot_bill_submissions", {"id": bill["id"]})
    check("the bill is marked with its disbursement id", bill_row["disbursement_id"] == disbursement["id"])

    # ---------------------------------------------------------------
    # Paying the same bill twice is blocked
    # ---------------------------------------------------------------
    try:
        await flow.pay_verified_bill(bill["id"], "cust-1", requested_by="admin3")
        check("rejects paying the same bill a second time", False)
    except flow.PaymentFlowError:
        check("rejects paying the same bill a second time", True)

    outstanding_after_retry = await cl.get_outstanding_principal("cust-1")
    check("a rejected second-payment attempt does NOT move any additional money", outstanding_after_retry == Decimal("300.00"))

    # ---------------------------------------------------------------
    # Paying an unverified bill is blocked before touching the ledger
    # ---------------------------------------------------------------
    unverified_submission = bv.BillSubmission(
        customer_id="cust-1", file_bytes=b"bill B bytes, low confidence",
        customer_name_on_account="Jane Citizen", customer_name_on_bill="Jane Citizen",
        biller_name_extracted="AusNet Electricity", biller_reference="REF-2",
        category="electricity", amount=Decimal("100.00"), due_date="2026-09-05",
        extraction_confidence=0.3, fraud_indicators=[],
    )
    unverified_bill = await bv.submit_and_verify_bill(unverified_submission, BILLER_ALLOWLIST, APPROVED_CATEGORIES)
    check("low-confidence bill goes to manual_review, not verified", unverified_bill["verification_status"] == "manual_review")

    outstanding_before_attempt = await cl.get_outstanding_principal("cust-1")
    try:
        await flow.pay_verified_bill(unverified_bill["id"], "cust-1", requested_by="admin3")
        check("rejects paying a bill that isn't verified", False)
    except flow.PaymentFlowError:
        check("rejects paying a bill that isn't verified", True)
    outstanding_after_attempt = await cl.get_outstanding_principal("cust-1")
    check("attempting to pay an unverified bill does not move any money", outstanding_before_attempt == outstanding_after_attempt)

    # ---------------------------------------------------------------
    # Insufficient available credit blocks payment AND leaves the bill
    # still verified/undisbursed (retryable later, e.g. after a repayment)
    # ---------------------------------------------------------------
    big_submission = bv.BillSubmission(
        customer_id="cust-1", file_bytes=b"bill C bytes",
        customer_name_on_account="Jane Citizen", customer_name_on_bill="Jane Citizen",
        biller_name_extracted="AusNet Electricity", biller_reference="REF-3",
        category="electricity", amount=Decimal("2300.00"), due_date="2026-09-10",
        extraction_confidence=0.99, fraud_indicators=[],
    )
    # 2300 exceeds the $500 single-bill limit too, but let's specifically
    # prove the ledger-level available-credit check by using an amount
    # under the single-bill limit that would still overrun what's left
    # of the $2,500 contractual limit after the first $300 draw... note
    # max_single_bill_payment is $500, so instead exercise this via many
    # bills up to the outstanding cap, then confirm a further bill is
    # blocked at the ledger stage specifically.
    remaining_capacity = Decimal("2500.00") - Decimal("300.00")  # 2200 left of contractual limit
    # Draw in $500 chunks (the single-bill cap) until close to the limit.
    draws_needed = int(remaining_capacity // Decimal("500.00"))  # 4 more $500 bills = 2000, total 2300
    for i in range(draws_needed):
        sub = bv.BillSubmission(
            customer_id="cust-1", file_bytes=f"bill chunk {i}".encode(),
            customer_name_on_account="Jane Citizen", customer_name_on_bill="Jane Citizen",
            biller_name_extracted="AusNet Electricity", biller_reference=f"REF-CHUNK-{i}",
            category="electricity", amount=Decimal("500.00"), due_date="2026-09-15",
            extraction_confidence=0.99, fraud_indicators=[],
        )
        b = await bv.submit_and_verify_bill(sub, BILLER_ALLOWLIST, APPROVED_CATEGORIES)
        await flow.pay_verified_bill(b["id"], "cust-1", requested_by="admin3")

    outstanding_now = await cl.get_outstanding_principal("cust-1")
    check(f"outstanding is now {outstanding_now} after chunked draws (expected 2300.00)", outstanding_now == Decimal("2300.00"))

    over_submission = bv.BillSubmission(
        customer_id="cust-1", file_bytes=b"bill D bytes, would overrun available credit",
        customer_name_on_account="Jane Citizen", customer_name_on_bill="Jane Citizen",
        biller_name_extracted="AusNet Electricity", biller_reference="REF-4",
        category="electricity", amount=Decimal("300.00"), due_date="2026-09-20",
        extraction_confidence=0.99, fraud_indicators=[],
    )
    over_bill = await bv.submit_and_verify_bill(over_submission, BILLER_ALLOWLIST, APPROVED_CATEGORIES)
    check("the $300 top-up bill itself verifies fine (only 200 of contractual limit left)", over_bill["verification_status"] == "verified")

    try:
        await flow.pay_verified_bill(over_bill["id"], "cust-1", requested_by="admin3")
        check("rejects payment that would exceed available credit, even though the bill itself verified", False)
    except flow.PaymentFlowError:
        check("rejects payment that would exceed available credit, even though the bill itself verified", True)

    still_undisbursed = await find_one("pilot_bill_submissions", {"id": over_bill["id"]})
    check("the over-limit bill is left verified and undisbursed (not silently marked failed/paid) after the blocked attempt",
          still_undisbursed["verification_status"] == "verified" and still_undisbursed.get("disbursement_id") is None)

    outstanding_unchanged = await cl.get_outstanding_principal("cust-1")
    check("a blocked payment attempt does not move any money", outstanding_unchanged == Decimal("2300.00"))

    # After a repayment frees up room, the same bill can now be paid.
    await cl.repay_credit("cust-1", Decimal("300.00"), created_by="admin1")
    retried = await flow.pay_verified_bill(over_bill["id"], "cust-1", requested_by="admin3")
    check("the previously-blocked bill can be paid after a repayment frees up available credit", retried["status"] == "queued")

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
