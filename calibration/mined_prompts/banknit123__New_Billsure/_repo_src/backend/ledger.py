"""
backend/ledger.py
==================
Double-entry ledger for BillSure's pooled customer trust account.

Design principles this module enforces:

- No balance is ever stored and mutated directly (no more
  `users.wallet_balance += x`, which is what the current codebase does).
  Every balance is the SUM of ledger_postings, computed on read. This makes
  the ledger self-auditing — there is nothing to race, because you only
  ever INSERT immutable postings, never UPDATE a running total.

- Every movement of money is a *journal*: two or more postings whose debits
  equal their credits. A customer's balance only ever moves in the same
  journal that moves the TRUST_BANK system account by the same amount in
  the opposite direction. That is what makes "sum of customer balances ==
  trust bank ledger balance" a structural guarantee rather than something
  reconciliation merely hopes is true — see reconciliation.py, which
  verifies it continuously against the real bank account too.

- This module never talks to Stripe, BPAY, or a bank API directly. It only
  records the accounting effect of events that happened elsewhere (a Stripe
  payment cleared, a BPAY payment was confirmed executed). Callers are
  responsible for confirming the real-world event before calling in —
  see payment_runs.py for the bill-payment side of that discipline.
"""

import logging
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, timezone, timedelta
from typing import Optional

import supabase_db as sdb

logger = logging.getLogger(__name__)

TRUST_BANK = "TRUST_BANK"
FEES_RECEIVABLE = "FEES_RECEIVABLE"
OPERATING = "OPERATING"
SUSPENSE = "SUSPENSE"
SYSTEM_ACCOUNT_CODES = [TRUST_BANK, FEES_RECEIVABLE, OPERATING, SUSPENSE]

# Conservative placeholder — CONFIRM actual BECS Direct Entry dishonour /
# clearing timing for your Stripe account and adjust. Card-funded top-ups
# carry a different (chargeback) risk profile; if you want a hold on those
# too, add a second constant and branch on payment_method_type below.
BECS_HOLD_DAYS = 4


def _q(amount) -> Decimal:
    return Decimal(str(amount)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


async def ensure_system_accounts() -> None:
    """Idempotently create the fixed system ledger accounts. Call once at app startup."""
    type_by_code = {
        TRUST_BANK: "trust_bank",
        FEES_RECEIVABLE: "fees_receivable",
        OPERATING: "operating",
        SUSPENSE: "suspense",
    }
    for code in SYSTEM_ACCOUNT_CODES:
        if not await sdb.find_one("ledger_accounts", {"code": code}):
            await sdb.insert_one("ledger_accounts", {"account_type": type_by_code[code], "code": code})


async def get_or_create_customer_account(user_id: str) -> dict:
    acct = await sdb.find_one("ledger_accounts", {"account_type": "customer", "user_id": user_id})
    if acct:
        return acct
    await sdb.insert_one("ledger_accounts", {"account_type": "customer", "user_id": user_id})
    return await sdb.find_one("ledger_accounts", {"account_type": "customer", "user_id": user_id})


async def _system_account(code: str) -> dict:
    acct = await sdb.find_one("ledger_accounts", {"code": code})
    if not acct:
        raise RuntimeError(f"System ledger account {code} missing — call ensure_system_accounts() at startup")
    return acct


async def post_journal(
    entry_type: str,
    postings: list,
    reference_type: Optional[str] = None,
    reference_id: Optional[str] = None,
    description: Optional[str] = None,
    created_by: Optional[str] = None,
) -> str:
    """
    Post a balanced journal. `postings` is a list of dicts:
    {"ledger_account_id": ..., "direction": "debit"|"credit", "amount": ...}

    Raises if postings don't balance — this Python-side check is
    belt-and-braces; the real enforcement is the deferred DB trigger, which
    fires even if a caller bypasses this function.
    """
    total_debit = sum(_q(p["amount"]) for p in postings if p["direction"] == "debit")
    total_credit = sum(_q(p["amount"]) for p in postings if p["direction"] == "credit")
    if total_debit != total_credit:
        raise ValueError(f"Journal does not balance: debits={total_debit} credits={total_credit}")
    if len(postings) < 2:
        raise ValueError("A journal needs at least two postings")

    journal = await sdb.insert_one("journal_entries", {
        "entry_type": entry_type,
        "reference_type": reference_type,
        "reference_id": reference_id,
        "description": description,
        "created_by": created_by,
    })
    journal_id = journal["id"]

    rows = [{
        "journal_id": journal_id,
        "ledger_account_id": p["ledger_account_id"],
        "direction": p["direction"],
        "amount": str(_q(p["amount"])),
    } for p in postings]

    # Single multi-row insert => one Postgres statement => the deferred
    # balance-check trigger evaluates all postings of this journal together.
    await sdb.insert_many("ledger_postings", rows)

    return journal_id


async def get_customer_balance(user_id: str) -> Decimal:
    row = await sdb.find_one("customer_balances", {"user_id": user_id})
    return _q(row["ledger_balance"]) if row else Decimal("0.00")


async def get_system_balance(code: str) -> Decimal:
    row = await sdb.find_one("ledger_account_balances", {"code": code})
    return _q(row["balance"]) if row else Decimal("0.00")


# ---------------------------------------------------------------
# High-level events — call these instead of touching wallet_balance.
# Wire these into server.py in place of increment_wallet_balance() calls;
# see INTEGRATION_NOTES.md for exactly which endpoints to change.
# ---------------------------------------------------------------

async def record_contribution_cleared(
    user_id: str, amount, reference_type: str, reference_id: str,
    payment_method_type: str = "card", description: str = "",
) -> str:
    """
    Customer's payment has cleared per Stripe (webhook/poll reported
    'paid'). Posts the journal immediately so the ledger matches Stripe in
    real time — but if this was a direct debit, also opens a fund_hold so
    the funds are NOT treated as available for outbound bill payment until
    BECS_HOLD_DAYS has passed. See payment_runs.get_available_balance(),
    which is what actually gates bill payment — never raw ledger balance.
    """
    customer_acct = await get_or_create_customer_account(user_id)
    trust = await _system_account(TRUST_BANK)

    journal_id = await post_journal(
        entry_type="contribution_cleared",
        postings=[
            {"ledger_account_id": trust["id"], "direction": "debit", "amount": amount},
            {"ledger_account_id": customer_acct["id"], "direction": "credit", "amount": amount},
        ],
        reference_type=reference_type, reference_id=reference_id,
        description=description or "Customer contribution cleared into trust account",
    )

    if payment_method_type in ("au_becs_debit", "both"):
        await sdb.insert_one("fund_holds", {
            "journal_id": journal_id,
            "user_id": user_id,
            "amount": str(_q(amount)),
            "hold_reason": "becs_dd_clearing",
            "available_at": (datetime.now(timezone.utc) + timedelta(days=BECS_HOLD_DAYS)).isoformat(),
        })

    return journal_id


async def record_bill_payment_cleared(
    user_id: str, amount, reference_type: str, reference_id: str,
    description: str = "", created_by: Optional[str] = None,
) -> str:
    """A biller payment has been CONFIRMED executed (BPAY receipt / bank
    debit confirmed against a statement). Only call this once execution is
    confirmed — see payment_runs.mark_item_cleared(), the only caller this
    should have."""
    customer_acct = await get_or_create_customer_account(user_id)
    trust = await _system_account(TRUST_BANK)
    return await post_journal(
        entry_type="bill_payment",
        postings=[
            {"ledger_account_id": customer_acct["id"], "direction": "debit", "amount": amount},
            {"ledger_account_id": trust["id"], "direction": "credit", "amount": amount},
        ],
        reference_type=reference_type, reference_id=reference_id,
        description=description or "Bill paid to biller from customer trust balance",
        created_by=created_by,
    )


async def record_refund_to_customer(
    user_id: str, amount, reference_type: str, reference_id: str,
    description: str = "", created_by: Optional[str] = None,
) -> str:
    customer_acct = await get_or_create_customer_account(user_id)
    trust = await _system_account(TRUST_BANK)
    return await post_journal(
        entry_type="refund",
        postings=[
            {"ledger_account_id": trust["id"], "direction": "debit", "amount": amount},
            {"ledger_account_id": customer_acct["id"], "direction": "credit", "amount": amount},
        ],
        reference_type=reference_type, reference_id=reference_id,
        description=description or "Refund to customer",
        created_by=created_by,
    )
