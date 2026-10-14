"""
Standalone logic test — an in-memory fake of supabase_db's interface so the
ledger/payment_runs/reconciliation flow can be exercised without a real
Postgres instance. This is NOT a replacement for testing against actual
Supabase (in particular it can't verify the deferred trigger or RLS), but
it does prove the Python-level accounting logic is internally consistent
before you point it at a real database.

Run: python3 test_ledger_flow.py
"""
import asyncio
import sys
import uuid
from datetime import datetime, timezone, timedelta
from decimal import Decimal

# ---- in-memory fake of supabase_db's public interface ----
_tables = {}


def _matches(row, filters):
    for k, v in filters.items():
        if isinstance(v, dict):
            if "$in" in v and row.get(k) not in v["$in"]:
                return False
            if "$ne" in v and row.get(k) == v["$ne"]:
                return False
        elif row.get(k) != v:
            return False
    return True


def _compute_ledger_account_balances():
    """Emulates the ledger_account_balances SQL view over the fake tables.
    Sign convention must match the real view exactly: 'customer' accounts
    are credit-normal (liabilities); everything else is debit-normal
    (assets) — see the comment above the view definition in
    002_ledger_and_reconciliation.sql."""
    postings_by_account = {}
    account_type_by_id = {la["id"]: la["account_type"] for la in _tables.get("ledger_accounts", [])}
    for p in _tables.get("ledger_postings", []):
        acc = p["ledger_account_id"]
        amt = Decimal(str(p["amount"]))
        if account_type_by_id.get(acc) == "customer":
            signed = amt if p["direction"] == "credit" else -amt
        else:
            signed = amt if p["direction"] == "debit" else -amt
        postings_by_account[acc] = postings_by_account.get(acc, Decimal("0")) + signed

    rows = []
    for la in _tables.get("ledger_accounts", []):
        rows.append({
            "ledger_account_id": la["id"],
            "account_type": la["account_type"],
            "user_id": la.get("user_id"),
            "code": la.get("code"),
            "balance": str(postings_by_account.get(la["id"], Decimal("0"))),
        })
    return rows


async def find_one(table, filters, exclude_fields=None):
    if table == "ledger_account_balances":
        for row in _compute_ledger_account_balances():
            if _matches(row, filters):
                return row
        return None
    if table == "customer_balances":
        for row in _compute_ledger_account_balances():
            if row["account_type"] == "customer" and _matches({"user_id": row["user_id"]}, filters):
                return {"user_id": row["user_id"], "ledger_balance": row["balance"]}
        return None
    for row in _tables.get(table, []):
        if _matches(row, filters):
            return dict(row)
    return None


async def find_many(table, filters=None, exclude_fields=None, order_by=None, order_desc=False, limit=10000):
    if table == "ledger_account_balances":
        rows = [r for r in _compute_ledger_account_balances() if not filters or _matches(r, filters)]
        return rows[:limit]
    if table == "customer_balances":
        rows = [{"user_id": r["user_id"], "ledger_balance": r["balance"]}
                for r in _compute_ledger_account_balances() if r["account_type"] == "customer"]
        rows = [r for r in rows if not filters or _matches(r, filters)]
        return rows[:limit]
    rows = [dict(r) for r in _tables.get(table, []) if not filters or _matches(r, filters)]
    if order_by:
        rows.sort(key=lambda r: r.get(order_by), reverse=order_desc)
    return rows[:limit]


async def insert_one(table, data):
    row = dict(data)
    row.setdefault("id", str(uuid.uuid4()))
    row.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    if table == "fund_holds":
        row.setdefault("released", False)  # mirrors `released BOOLEAN DEFAULT FALSE` in the real schema
    _tables.setdefault(table, []).append(row)
    return dict(row)


async def insert_many(table, rows):
    out = []
    for data in rows:
        out.append(await insert_one(table, data))
    return out


async def update_one(table, filters, updates):
    set_data = updates.get("$set", updates)
    for row in _tables.get(table, []):
        if _matches(row, filters):
            row.update(set_data)
            return True
    return False


# make this module importable as "supabase_db" and "ledger"/"payment_runs"/
# "reconciliation" importable normally, by inserting this directory's fake
# module into sys.modules before the real ones (which import supabase_db).
import types
fake_sdb = types.ModuleType("supabase_db")
fake_sdb.find_one = find_one
fake_sdb.find_many = find_many
fake_sdb.insert_one = insert_one
fake_sdb.insert_many = insert_many
fake_sdb.update_one = update_one
sys.modules["supabase_db"] = fake_sdb

import ledger          # noqa: E402
import payment_runs    # noqa: E402
import reconciliation  # noqa: E402


async def main():
    await ledger.ensure_system_accounts()

    alice, bob = "user-alice", "user-bob"

    # Alice tops up $200 via card (no hold), Bob tops up $150 via BECS DD (held).
    await ledger.record_contribution_cleared(alice, "200.00", "payment_transactions", "tx-1",
                                              payment_method_type="card")
    await ledger.record_contribution_cleared(bob, "150.00", "payment_transactions", "tx-2",
                                              payment_method_type="au_becs_debit")

    assert await ledger.get_customer_balance(alice) == Decimal("200.00")
    assert await ledger.get_customer_balance(bob) == Decimal("150.00")

    # Available balance: Alice's card funds are immediately available.
    # Bob's BECS funds are NOT yet available (still inside the hold window).
    alice_avail = await payment_runs.get_available_balance(alice)
    bob_avail = await payment_runs.get_available_balance(bob)
    assert alice_avail == Decimal("200.00"), f"alice_avail={alice_avail}"
    assert bob_avail == Decimal("0.00"), f"bob_avail={bob_avail} (should be held)"
    print(f"[ok] available balances correct: alice={alice_avail} bob={bob_avail} (bob's BECS funds correctly held)")

    # Two bills: Alice owes $80 (due sooner), Bob owes $50 (due later, but funds held anyway).
    now = datetime.now(timezone.utc)
    await insert_one("bills", {
        "user_id": alice, "status": "pending", "amount": 80.0,
        "due_date": (now + timedelta(days=1)).isoformat(),
        "provider": "EnergyCo", "category": "electricity", "biller_code": "1111", "reference_number": "AAA",
    })
    await insert_one("bills", {
        "user_id": bob, "status": "pending", "amount": 50.0,
        "due_date": (now + timedelta(days=2)).isoformat(),
        "provider": "WaterCo", "category": "water", "biller_code": "2222", "reference_number": "BBB",
    })

    run = await payment_runs.build_payment_run(horizon_days=3, created_by="system")
    items = await find_many("payment_run_items", {"payment_run_id": run["id"]})
    assert len(items) == 1 and items[0]["user_id"] == alice, f"expected only alice's bill queued, got {items}"
    print(f"[ok] payment run queued only alice's bill (bob's held funds correctly excluded): {[ (i['user_id'], i['amount']) for i in items ]}")

    # Reconciliation must be run and clean before a run can be approved.
    try:
        await payment_runs.approve_payment_run(run["id"], approver_user_id="admin-1")
        raise AssertionError("expected approval to be blocked before reconciliation has ever run")
    except ValueError as e:
        print(f"[ok] approval correctly blocked before first reconciliation: {e}")

    recon = await reconciliation.run_trust_reconciliation()
    assert recon["status"] == "ok", f"expected clean reconciliation, got {recon}"
    print(f"[ok] reconciliation clean: trust_ledger={recon['trust_ledger_balance']} sum_customers={recon['sum_customer_balances']}")

    approved = await payment_runs.approve_payment_run(run["id"], approver_user_id="admin-1")
    assert approved["status"] == "approved"

    # Maker == checker should be rejected.
    run2 = await payment_runs.build_payment_run(horizon_days=3, created_by="admin-1")
    try:
        await payment_runs.approve_payment_run(run2["id"], approver_user_id="admin-1")
        raise AssertionError("expected maker==checker to be rejected")
    except ValueError as e:
        print(f"[ok] maker-checker correctly enforced: {e}")

    # Execute alice's item through to cleared.
    to_execute = await payment_runs.get_run_for_execution(run["id"])
    item = to_execute[0]
    await payment_runs.mark_item_submitted(item["id"], provider_payment_reference="BPAY-REF-1")
    await payment_runs.mark_item_cleared(item["id"], provider_payment_reference="BPAY-REF-1", cleared_by="admin-2")

    alice_balance_after = await ledger.get_customer_balance(alice)
    assert alice_balance_after == Decimal("120.00"), f"expected 200-80=120, got {alice_balance_after}"
    bill = await find_one("bills", {"id": item["bill_id"]})
    assert bill["status"] == "paid"
    print(f"[ok] bill cleared correctly: alice balance now {alice_balance_after}, bill status={bill['status']}")

    recon2 = await reconciliation.run_trust_reconciliation()
    assert recon2["status"] == "ok", f"expected clean reconciliation after payment, got {recon2}"
    print(f"[ok] reconciliation still clean after payment: trust_ledger={recon2['trust_ledger_balance']} sum_customers={recon2['sum_customer_balances']}")

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
