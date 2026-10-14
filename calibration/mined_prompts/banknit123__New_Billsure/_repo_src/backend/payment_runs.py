"""
backend/payment_runs.py
========================
Turns "which bills are due" + "which customers can actually afford them
right now" into a prioritised, maker-checker-approved batch of biller
payments, and tracks each one through to confirmed execution before the
ledger is ever debited.

State machine per payment_run_items row:
    queued -> submitted -> cleared   (money actually left the trust account)
                         -> failed   (reservation released, bill stays 'pending')

State machine per payment_runs row:
    draft -> approved -> (items get executed and cleared one by one)

Nothing in this file calls a bank or BPAY API. Wire mark_item_submitted /
mark_item_cleared / mark_item_failed to whatever actually executes the
payment — either a real BPAY/NPP API integration, or an admin UI where
staff confirm a manually-executed BPAY payment against a bank statement
line. See INTEGRATION_NOTES.md.
"""

import logging
from decimal import Decimal
from datetime import datetime, timezone, timedelta
from typing import Optional

import supabase_db as sdb
import ledger
import reconciliation

logger = logging.getLogger(__name__)


async def get_available_balance(user_id: str) -> Decimal:
    """
    The ONLY number that should ever be checked before queuing a bill for
    payment — never the raw ledger balance. Cleared ledger balance, minus:
      - funds still inside their settlement hold window (fund_holds), and
      - amounts already reserved by this customer's other in-flight
        (queued/submitted) payment run items, so two concurrent runs can
        never both claim the same dollar.
    """
    ledger_balance = await ledger.get_customer_balance(user_id)

    holds = await sdb.find_many("fund_holds", {"user_id": user_id, "released": False})
    now = datetime.now(timezone.utc)
    held = sum(
        Decimal(str(h["amount"])) for h in holds
        if datetime.fromisoformat(h["available_at"].replace("Z", "+00:00")) > now
    )

    in_flight = await sdb.find_many(
        "payment_run_items", {"user_id": user_id, "status": {"$in": ["queued", "submitted"]}}
    )
    reserved = sum(Decimal(str(i["amount"])) for i in in_flight)

    return ledger_balance - held - reserved


async def build_payment_run(horizon_days: int = 3, created_by: Optional[str] = None,
                             priority: str = "due_date") -> dict:
    """
    Select pending bills due within `horizon_days`, check each against the
    OWNING customer's available balance (never another customer's), and
    queue whatever can safely be paid — ordered by priority so the
    tightest-deadline bills get first claim on a customer's available funds
    if there isn't enough for everything that customer owes.

    Never moves money and never touches the ledger — it only decides what
    SHOULD be paid. Approval and execution are separate steps below.
    """
    now = datetime.now(timezone.utc)
    horizon = now + timedelta(days=horizon_days)

    bills = await sdb.find_many("bills", {"status": "pending"})
    due_soon = [b for b in bills if datetime.fromisoformat(b["due_date"].replace("Z", "+00:00")) <= horizon]

    if priority == "due_date":
        due_soon.sort(key=lambda b: b["due_date"])
    elif priority == "amount_asc":
        due_soon.sort(key=lambda b: b["amount"])
    # else: leave in whatever order the DB returned — callers can plug in
    # other priority schemes (e.g. disconnection-risk category) here.

    run = await sdb.insert_one("payment_runs", {
        "status": "draft", "created_by": created_by, "run_date": now.date().isoformat(),
    })

    committed_this_run = {}  # user_id -> Decimal, running total already queued in THIS run
    rank = 0
    total_amount = Decimal("0")
    item_count = 0

    for bill in due_soon:
        uid = bill["user_id"]
        available = await get_available_balance(uid)
        already = committed_this_run.get(uid, Decimal("0"))
        remaining = available - already
        amount = Decimal(str(bill["amount"]))

        if remaining < amount:
            continue  # not enough available balance right now — bill stays 'pending', picked up next run

        rank += 1
        committed_this_run[uid] = already + amount
        total_amount += amount
        item_count += 1

        await sdb.insert_one("payment_run_items", {
            "payment_run_id": run["id"],
            "bill_id": bill["id"],
            "user_id": uid,
            "amount": str(amount),
            "priority_rank": rank,
            "status": "queued",
            "biller_code": bill.get("biller_code") or bill.get("bpay_code") or "",
            "reference_number": bill.get("reference_number") or "",
        })

    await sdb.update_one("payment_runs", {"id": run["id"]},
                          {"$set": {"total_amount": str(total_amount), "item_count": item_count}})
    return await sdb.find_one("payment_runs", {"id": run["id"]})


async def approve_payment_run(run_id: str, approver_user_id: str) -> dict:
    """
    Maker-checker: the approver must be a different person from whoever
    triggered build_payment_run (for system-triggered runs, created_by is
    null, so any admin may approve). Also refuses to approve if the most
    recent trust-account reconciliation has an open, unresolved exception —
    a payment run should never go out while the books don't balance.
    """
    run = await sdb.find_one("payment_runs", {"id": run_id})
    if not run:
        raise ValueError("Payment run not found")
    if run["status"] != "draft":
        raise ValueError(f"Payment run is '{run['status']}', not 'draft' — cannot approve")
    if run.get("created_by") and run["created_by"] == approver_user_id:
        raise ValueError("Approver must differ from the run's creator (maker-checker)")
    if not await reconciliation.is_safe_to_process_payments():
        raise ValueError("Blocked: trust account has an open reconciliation exception — resolve before approving payments")

    await sdb.update_one("payment_runs", {"id": run_id}, {"$set": {
        "status": "approved", "approved_by": approver_user_id,
        "approved_at": datetime.now(timezone.utc).isoformat(),
    }})
    return await sdb.find_one("payment_runs", {"id": run_id})


async def get_run_for_execution(run_id: str) -> list:
    """Queued items ready to be submitted for payment, in priority order.
    Export this to whatever executes BPAY payments — an API integration, or
    a report for staff to work through, calling mark_item_submitted for
    each as they go."""
    run = await sdb.find_one("payment_runs", {"id": run_id})
    if not run:
        raise ValueError("Payment run not found")
    if run["status"] != "approved":
        raise ValueError("Payment run must be approved before execution")
    return await sdb.find_many("payment_run_items", {"payment_run_id": run_id, "status": "queued"},
                                order_by="priority_rank")


async def mark_item_submitted(item_id: str, provider_payment_reference: str = "") -> None:
    """Payment has been SENT (BPAY submitted / transfer initiated) but not
    yet confirmed cleared. Funds stay reserved (get_available_balance
    already excludes 'submitted' items) — the ledger is NOT touched yet."""
    await sdb.update_one("payment_run_items", {"id": item_id, "status": "queued"}, {"$set": {
        "status": "submitted", "provider_payment_reference": provider_payment_reference,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }})


async def mark_item_cleared(item_id: str, provider_payment_reference: str = "",
                             cleared_by: Optional[str] = None) -> str:
    """
    Call ONLY once the biller payment is confirmed executed — a BPAY
    receipt, or a bank statement line matched during reconciliation. This
    is the single place the ledger is actually debited for a bill payment,
    and the single place bills.status flips to 'paid'.
    """
    item = await sdb.find_one("payment_run_items", {"id": item_id})
    if not item:
        raise ValueError("Payment run item not found")
    if item["status"] not in ("queued", "submitted"):
        raise ValueError(f"Item is '{item['status']}' — cannot clear")

    journal_id = await ledger.record_bill_payment_cleared(
        user_id=item["user_id"], amount=item["amount"],
        reference_type="payment_run_items", reference_id=item_id,
        description=f"Bill payment cleared (run item {item_id})",
        created_by=cleared_by,
    )

    await sdb.update_one("payment_run_items", {"id": item_id}, {"$set": {
        "status": "cleared", "journal_id": journal_id,
        "provider_payment_reference": provider_payment_reference or item.get("provider_payment_reference", ""),
        "cleared_at": datetime.now(timezone.utc).isoformat(),
    }})
    await sdb.update_one("bills", {"id": item["bill_id"]}, {"$set": {
        "status": "paid", "paid_at": datetime.now(timezone.utc).isoformat(),
        "payment_reference": provider_payment_reference,
    }})
    return journal_id


async def mark_item_failed(item_id: str, reason: str) -> None:
    """Payment didn't go through (BPAY rejected, biller details invalid,
    etc). Releases the reservation — bill stays 'pending' and is picked up
    by the next build_payment_run()."""
    await sdb.update_one("payment_run_items", {"id": item_id, "status": {"$ne": "cleared"}}, {"$set": {
        "status": "failed", "failure_reason": reason,
    }})
