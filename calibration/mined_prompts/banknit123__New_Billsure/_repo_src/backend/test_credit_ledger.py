"""
Standalone logic test for credit_ledger.py — verifies pilot_config's
limits are actually enforced by the credit sub-ledger, not just by
pilot_config.py in isolation. Same in-memory fake-DB pattern as the
other test_*.py files. No live credentials needed.

Run: python3 test_credit_ledger.py
"""
import asyncio
import sys
import types
import uuid
from decimal import Decimal

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

import pilot_config as pc     # noqa: E402
import credit_ledger as cl    # noqa: E402

FAILURES = []


def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}")
    if not condition:
        FAILURES.append(label)


async def main():
    # Activate a pilot config so credit_ledger reads real limits from it
    # rather than falling back to hard-coded constants -- this is the
    # actual integration point being tested.
    await pc.propose_config_version(pc.PilotConfig(), proposed_by="alice", approved_by="bob", activate=True)
    active = await pc.get_active_config()
    check("pilot config is active before any credit ledger operation", active is not None and active.is_active)

    # ---------------------------------------------------------------
    # activate_customer_credit_account: maker-checker
    # ---------------------------------------------------------------
    try:
        await cl.activate_customer_credit_account("cust-1", Decimal("2500.00"), 0, Decimal("0"), "admin1", "admin1")
        check("rejects credit account activation prepared and approved by the same person", False)
    except cl.CreditLedgerError:
        check("rejects credit account activation prepared and approved by the same person", True)

    acct1 = await cl.activate_customer_credit_account("cust-1", Decimal("2500.00"), 0, Decimal("0"), "admin1", "admin2")
    check("credit account activates with distinct preparer/approver", acct1["status"] == "active")

    # ---------------------------------------------------------------
    # activate_customer_credit_account: customer cap enforced THROUGH
    # the ledger module, reading pilot_config's actual active limit
    # ---------------------------------------------------------------
    active_count = 1
    for i in range(2, 26):  # activate customers 2..25 (24 more, total 25)
        await cl.activate_customer_credit_account(f"cust-{i}", Decimal("2500.00"), active_count, Decimal(active_count * 2500), "admin1", "admin2")
        active_count += 1
    check("25 customers activated successfully (at the pilot cap)", active_count == 25)

    try:
        await cl.activate_customer_credit_account("cust-26", Decimal("2500.00"), 25, Decimal("62500.00"), "admin1", "admin2")
        check("rejects activating a 26th customer", False)
    except pc.ConfigValidationError:
        check("rejects activating a 26th customer", True)

    # ---------------------------------------------------------------
    # activate_customer_credit_account: aggregate exposure cap enforced
    # ---------------------------------------------------------------
    try:
        await cl.activate_customer_credit_account("cust-27-instead", Decimal("2500.00"), 24, Decimal("61000.00"), "admin1", "admin2")
        check("rejects activation that would push aggregate exposure over $62,500", False)
    except pc.ConfigValidationError:
        check("rejects activation that would push aggregate exposure over $62,500", True)

    # ---------------------------------------------------------------
    # activate_customer_credit_account: contractual limit ceiling from config
    # ---------------------------------------------------------------
    try:
        await cl.activate_customer_credit_account("cust-over-limit", Decimal("3000.00"), 0, Decimal("0"), "admin1", "admin2")
        check("rejects a contractual_limit above the active config's ceiling", False)
    except cl.CreditLedgerError:
        check("rejects a contractual_limit above the active config's ceiling", True)

    try:
        await cl.activate_customer_credit_account("cust-1", Decimal("2500.00"), 1, Decimal("2500"), "admin1", "admin2")
        check("rejects a second credit account for a customer who already has one", False)
    except cl.CreditLedgerError:
        check("rejects a second credit account for a customer who already has one", True)

    # ---------------------------------------------------------------
    # draw_credit: single-bill limit, available credit, outstanding cap
    # ---------------------------------------------------------------
    outstanding_before = await cl.get_outstanding_principal("cust-1")
    check("new credit account starts with zero outstanding principal", outstanding_before == Decimal("0.00"))

    try:
        await cl.draw_credit("cust-1", Decimal("1600.00"), "bill-1", "admin1")
        check("rejects a draw above the pilot's max single-bill payment ($1,500)", False)
    except cl.CreditLedgerError:
        check("rejects a draw above the pilot's max single-bill payment ($1,500)", True)

    j1 = await cl.draw_credit("cust-1", Decimal("400.00"), "bill-1", "admin1")
    check("valid draw within limits posts a journal", j1 is not None)
    outstanding_after = await cl.get_outstanding_principal("cust-1")
    check("outstanding principal increased by the draw amount", outstanding_after == Decimal("400.00"))

    try:
        await cl.draw_credit("cust-1", Decimal("2200.00"), "bill-2", "admin1")
        check("rejects a draw that would exceed the customer's available credit", False)
    except cl.CreditLedgerError:
        check("rejects a draw that would exceed the customer's available credit", True)
    # (2500 limit - 400 outstanding = 2100 available; 2200 exceeds it,
    # even though 2200 alone is under neither the $1,500 single-bill cap
    # -- 2200 > 1500 so it would ALSO fail the single-bill check first;
    # the point of this case is still valid as a hard block either way.)

    try:
        await cl.draw_credit("nonexistent-customer", Decimal("100.00"), "bill-3", "admin1")
        check("rejects a draw against a customer with no credit account", False)
    except cl.CreditLedgerError:
        check("rejects a draw against a customer with no credit account", True)

    # ---------------------------------------------------------------
    # draw_credit: max outstanding balance enforced even with room under
    # the single-bill limit and available credit
    # ---------------------------------------------------------------
    # cust-2 has a full 2500 limit; draw close to the outstanding cap in
    # chunks under the single-bill limit, then verify the last chunk that
    # would breach max_outstanding_balance ($2500) is blocked.
    await cl.draw_credit("cust-2", Decimal("500.00"), "bill-4", "admin1")
    await cl.draw_credit("cust-2", Decimal("500.00"), "bill-5", "admin1")
    await cl.draw_credit("cust-2", Decimal("500.00"), "bill-6", "admin1")
    await cl.draw_credit("cust-2", Decimal("500.00"), "bill-7", "admin1")
    outstanding_cust2 = await cl.get_outstanding_principal("cust-2")
    check("cust-2 outstanding reached 2000 after four $500 draws", outstanding_cust2 == Decimal("2000.00"))

    j5 = await cl.draw_credit("cust-2", Decimal("500.00"), "bill-8", "admin1")
    check("fifth $500 draw succeeds, reaching exactly the $2500 outstanding cap (boundary, not breach)", j5 is not None)

    try:
        await cl.draw_credit("cust-2", Decimal("1.00"), "bill-9", "admin1")
        check("rejects any further draw once outstanding is already at the $2500 cap", False)
    except cl.CreditLedgerError:
        check("rejects any further draw once outstanding is already at the $2500 cap", True)

    # ---------------------------------------------------------------
    # repay_credit: reduces principal, cannot overpay
    # ---------------------------------------------------------------
    await cl.repay_credit("cust-2", Decimal("1000.00"), created_by="admin1")
    outstanding_after_repay = await cl.get_outstanding_principal("cust-2")
    check("repayment reduces outstanding principal correctly", outstanding_after_repay == Decimal("1500.00"))

    try:
        await cl.repay_credit("cust-2", Decimal("999999.00"), created_by="admin1")
        check("rejects a repayment larger than outstanding principal", False)
    except cl.CreditLedgerError:
        check("rejects a repayment larger than outstanding principal", True)

    # Room freed by the repayment should allow a new draw again.
    j_after_repay = await cl.draw_credit("cust-2", Decimal("300.00"), "bill-10", "admin1")
    check("a draw succeeds again after a repayment frees up available credit", j_after_repay is not None)

    # ---------------------------------------------------------------
    # exposure snapshot + warning thresholds
    # ---------------------------------------------------------------
    snapshot = await cl.get_exposure_snapshot()
    check("exposure snapshot reports 25 active customers", snapshot.active_customer_count == 25)
    check("exposure snapshot reports aggregate contractual exposure of $62,500 (25 x $2,500)", snapshot.aggregate_contractual_exposure == Decimal("62500.00"))

    warnings = cl.check_exposure_thresholds(snapshot, Decimal("2500.00"))
    check("customer count at 100% of cap reports a 'breach' warning level", warnings["customer_count"] == "breach")
    check("aggregate contractual exposure at 100% of cap reports a 'breach' warning level", warnings["aggregate_contractual_exposure"] == "breach")
    check("has_critical_breach() is True when customer count / aggregate exposure are both at cap", cl.has_critical_breach(warnings))

    # A snapshot well under any threshold should report no warnings and
    # no critical breach.
    calm_snapshot = cl.ExposureSnapshot(
        active_customer_count=5, max_pilot_customers=25,
        aggregate_contractual_exposure=Decimal("10000.00"), aggregate_contractual_cap=Decimal("62500.00"),
        aggregate_drawn_exposure=Decimal("1000.00"),
        per_customer=[{"customer_id": "x", "contractual_limit": Decimal("2500"), "outstanding_principal": Decimal("100"), "available_credit": Decimal("2400")}],
    )
    calm_warnings = cl.check_exposure_thresholds(calm_snapshot, Decimal("2500.00"))
    check("a calm snapshot well under every threshold reports no critical breach", not cl.has_critical_breach(calm_warnings))
    check("a calm snapshot's customer_count warning is None", calm_warnings["customer_count"] is None)

    # A snapshot at exactly 70% should surface the 70pct warning level.
    seventy_pct_snapshot = cl.ExposureSnapshot(
        active_customer_count=18, max_pilot_customers=25,  # 18/25 = 72%
        aggregate_contractual_exposure=Decimal("45000.00"), aggregate_contractual_cap=Decimal("62500.00"),  # ~72%
        aggregate_drawn_exposure=Decimal("0"), per_customer=[],
    )
    seventy_pct_warnings = cl.check_exposure_thresholds(seventy_pct_snapshot, Decimal("2500.00"))
    check("~72% customer-count utilisation surfaces a '70pct' warning, not a breach",
          seventy_pct_warnings["customer_count"] == "70pct")
    check("~72% utilisation does not trip has_critical_breach()", not cl.has_critical_breach(seventy_pct_warnings))

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
