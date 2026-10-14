"""
backend/credit_ledger.py
==========================
Double-entry credit sub-ledger for the ASIC ERS pilot credit facility,
and the real-time credit-limit/exposure monitoring built on top of it.

This is a SEPARATE ledger from `ledger.py` (the existing customer trust/
contribution ledger for the bill-smoothing product), on purpose:

- `ledger.py` tracks money customers have contributed and BillSure holds
  in trust on their behalf — a liability to the customer.
- `credit_ledger.py` tracks money BillSure has lent a customer to pay a
  bill — an asset (a receivable) owed BY the customer TO BillSure,
  funded from a distinct CREDIT_FUNDING capital pool, never from trust
  funds. Mixing the two into one ledger_accounts table (even though the
  underlying table would technically allow it with a constraint change)
  would make "customer funds cannot finance credit advances" and
  "credit-funding money cannot be recorded as a customer contribution"
  harder to verify by inspection, not easier — segregation was worth a
  second table, not a shared one with a type flag.

Every limit enforced here reads from `pilot_config` (or a snapshot of
it passed in) — never a hard-coded number — so this module is the
concrete integration point between the versioned pilot configuration
and real money movement:

- `activate_customer_credit_account()` enforces the pilot customer cap
  (`pilot_config.check_customer_cap`) and the aggregate contractual
  exposure cap (`pilot_config.check_aggregate_exposure`) before a 26th
  customer or an over-cap aggregate limit can ever be created.
- `draw_credit()` enforces the single-bill limit, the per-customer
  outstanding-balance limit, and available credit (contractual limit
  minus current outstanding) before any journal is posted.
- `get_exposure_snapshot()` / `check_exposure_thresholds()` implement
  the real-time monitoring and 70/80/90% warning thresholds required by
  the task's credit-limit and exposure monitoring section, reading
  current state from postings the same "sum of immutable postings, never
  a mutated running total" way `ledger.py` does.

Follows `ledger.py`'s double-entry discipline: no balance is ever stored
and mutated directly; every balance is the sum of immutable postings in
a balanced journal.
"""

import logging
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, timezone
from typing import Optional

import supabase_db as sdb
import pilot_config as pc

logger = logging.getLogger(__name__)

CREDIT_FUNDING = "CREDIT_FUNDING"


def _q(amount) -> Decimal:
    return Decimal(str(amount)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


class CreditLedgerError(Exception):
    """Raised for any invalid credit-ledger operation. Every raise path
    is a refusal to post — there is no partial/best-effort posting."""


async def ensure_credit_funding_account() -> dict:
    acct = await sdb.find_one("credit_ledger_accounts", {"code": CREDIT_FUNDING})
    if acct:
        return acct
    return await sdb.insert_one("credit_ledger_accounts", {
        "account_type": "credit_funding", "code": CREDIT_FUNDING, "customer_id": None, "contractual_limit": None,
    })


async def get_customer_credit_account(customer_id: str) -> Optional[dict]:
    return await sdb.find_one("credit_ledger_accounts", {"account_type": "customer_credit", "customer_id": customer_id})


async def _post_credit_journal(entry_type: str, postings: list, reference_type: Optional[str] = None,
                                reference_id: Optional[str] = None, description: Optional[str] = None,
                                created_by: Optional[str] = None) -> str:
    """Same balanced-journal discipline as ledger.post_journal(), applied
    to the separate credit ledger tables."""
    total_debit = sum(_q(p["amount"]) for p in postings if p["direction"] == "debit")
    total_credit = sum(_q(p["amount"]) for p in postings if p["direction"] == "credit")
    if total_debit != total_credit:
        raise CreditLedgerError(f"credit journal does not balance: debits={total_debit} credits={total_credit}")
    if len(postings) < 2:
        raise CreditLedgerError("a credit journal needs at least two postings")

    journal = await sdb.insert_one("credit_journal_entries", {
        "entry_type": entry_type, "reference_type": reference_type, "reference_id": reference_id,
        "description": description, "created_by": created_by,
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    rows = [{
        "journal_id": journal["id"], "credit_ledger_account_id": p["credit_ledger_account_id"],
        "direction": p["direction"], "amount": str(_q(p["amount"])),
        "created_at": datetime.now(timezone.utc).isoformat(),
    } for p in postings]
    # Single multi-row insert => one Postgres statement => migration
    # 015's deferred balance-check trigger evaluates all of this
    # journal's postings together. Matches ledger.py's post_journal()
    # discipline exactly, for the same reason: separate insert_one calls
    # would each be their own statement/transaction against PostgREST,
    # and the deferred trigger can only see postings committed in the
    # same transaction.
    await sdb.insert_many("credit_ledger_postings", rows)
    return journal["id"]


async def get_outstanding_principal(customer_id: str) -> Decimal:
    """Customer credit accounts are debit-normal (an asset/receivable to
    BillSure): balance = sum(debits) - sum(credits). Computed from
    postings every time, never from a stored/mutated column."""
    acct = await get_customer_credit_account(customer_id)
    if not acct:
        return Decimal("0.00")
    postings = await sdb.find_many("credit_ledger_postings", {"credit_ledger_account_id": acct["id"]})
    debit = sum(_q(p["amount"]) for p in postings if p["direction"] == "debit")
    credit = sum(_q(p["amount"]) for p in postings if p["direction"] == "credit")
    return _q(debit - credit)


async def activate_customer_credit_account(
    customer_id: str, contractual_limit: Decimal, active_customer_count: int,
    current_aggregate_contractual_exposure: Decimal, proposed_by: str, approved_by: str,
) -> dict:
    """Creates a customer's credit account. Enforces, in this order:
    1. maker-checker on the activation itself (mirrors
       onboarding.approve_credit_activation — this is the ledger-side
       half of that same event, not a replacement for it),
    2. the pilot customer cap (blocks a 26th customer),
    3. the aggregate contractual exposure cap (blocks exceeding $62,500),
    4. the per-customer contractual limit ceiling from pilot_config.
    No credit account is created if any check fails."""
    if proposed_by == approved_by:
        raise CreditLedgerError("credit account activation requires a distinct approver (maker-checker)")

    await pc.check_customer_cap(active_customer_count)
    await pc.check_aggregate_exposure(current_aggregate_contractual_exposure, contractual_limit)

    cfg = await pc.get_active_config()
    max_limit = cfg.contractual_credit_limit if cfg else pc.HARD_MAX_CONTRACTUAL_CREDIT_LIMIT
    if contractual_limit > max_limit:
        raise CreditLedgerError(f"contractual_limit {contractual_limit} exceeds active config ceiling {max_limit}")

    existing = await get_customer_credit_account(customer_id)
    if existing:
        raise CreditLedgerError(f"customer {customer_id} already has a credit account")

    account = await sdb.insert_one("credit_ledger_accounts", {
        "account_type": "customer_credit", "code": None, "customer_id": customer_id,
        "contractual_limit": str(_q(contractual_limit)), "status": "active",
        "activated_by": approved_by, "activated_at": datetime.now(timezone.utc).isoformat(),
    })
    return account


async def draw_credit(customer_id: str, amount: Decimal, bill_id: str, requested_by: str) -> str:
    """Disburses credit to pay a verified bill (called only after
    payment_permitted_use.validate_disbursement() has already passed —
    this function re-checks the ledger-side limits independently rather
    than trusting that caller, since those are two different concerns:
    permitted-use is about WHAT can be paid, this is about HOW MUCH
    credit room the customer actually has left).

    Enforces, in order: account is active, amount <= pilot's
    max_single_bill_payment, amount <= available credit (contractual
    limit - current outstanding), and resulting outstanding <= pilot's
    max_outstanding_balance. Posts a balanced journal only if every
    check passes."""
    account = await get_customer_credit_account(customer_id)
    if not account or account.get("status") != "active":
        raise CreditLedgerError(f"no active credit account for customer {customer_id}")

    cfg = await pc.get_active_config()
    max_single_bill = cfg.max_single_bill_payment if cfg else pc.HARD_MAX_SINGLE_BILL_PAYMENT
    max_outstanding = cfg.max_outstanding_balance if cfg else pc.HARD_MAX_OUTSTANDING_BALANCE

    amount = _q(amount)
    if amount <= 0:
        raise CreditLedgerError("draw amount must be positive")
    if amount > max_single_bill:
        raise CreditLedgerError(f"draw amount {amount} exceeds pilot max single-bill payment {max_single_bill}")

    outstanding = await get_outstanding_principal(customer_id)
    contractual_limit = _q(account["contractual_limit"])
    available = contractual_limit - outstanding
    if amount > available:
        raise CreditLedgerError(f"draw amount {amount} exceeds available credit {available} (limit {contractual_limit}, outstanding {outstanding})")

    projected_outstanding = outstanding + amount
    if projected_outstanding > max_outstanding:
        raise CreditLedgerError(f"resulting outstanding {projected_outstanding} would exceed pilot max outstanding balance {max_outstanding}")

    funding = await ensure_credit_funding_account()
    return await _post_credit_journal(
        entry_type="credit_draw",
        postings=[
            {"credit_ledger_account_id": account["id"], "direction": "debit", "amount": amount},
            {"credit_ledger_account_id": funding["id"], "direction": "credit", "amount": amount},
        ],
        reference_type="bill", reference_id=bill_id, description=f"Credit draw to pay bill {bill_id}",
        created_by=requested_by,
    )


async def repay_credit(customer_id: str, amount: Decimal, reference_id: Optional[str] = None,
                        created_by: Optional[str] = None) -> str:
    """Records a repayment reducing outstanding principal. Refuses to
    post a repayment larger than the current outstanding balance —
    outstanding principal is never allowed to go negative, mirroring
    ledger.py's 'negative customer-funds balances are prohibited'
    invariant applied to this ledger instead."""
    account = await get_customer_credit_account(customer_id)
    if not account:
        raise CreditLedgerError(f"no credit account for customer {customer_id}")

    amount = _q(amount)
    if amount <= 0:
        raise CreditLedgerError("repayment amount must be positive")

    outstanding = await get_outstanding_principal(customer_id)
    if amount > outstanding:
        raise CreditLedgerError(f"repayment {amount} exceeds outstanding principal {outstanding} — cannot overpay into a negative balance")

    funding = await ensure_credit_funding_account()
    return await _post_credit_journal(
        entry_type="credit_repayment",
        postings=[
            {"credit_ledger_account_id": funding["id"], "direction": "debit", "amount": amount},
            {"credit_ledger_account_id": account["id"], "direction": "credit", "amount": amount},
        ],
        reference_type="repayment", reference_id=reference_id, description="Credit repayment",
        created_by=created_by,
    )


@dataclass
class ExposureSnapshot:
    active_customer_count: int
    max_pilot_customers: int
    aggregate_contractual_exposure: Decimal
    aggregate_contractual_cap: Decimal
    aggregate_drawn_exposure: Decimal
    per_customer: list = field(default_factory=list)   # [{customer_id, contractual_limit, outstanding_principal, available_credit}]


async def get_exposure_snapshot() -> ExposureSnapshot:
    accounts = await sdb.find_many("credit_ledger_accounts", {"account_type": "customer_credit"})
    active_accounts = [a for a in accounts if a.get("status") == "active"]
    cfg = await pc.get_active_config()
    max_customers = cfg.max_pilot_customers if cfg else pc.HARD_MAX_PILOT_CUSTOMERS
    aggregate_cap = cfg.aggregate_contractual_exposure_cap if cfg else pc.HARD_MAX_AGGREGATE_EXPOSURE

    per_customer = []
    aggregate_contractual = Decimal("0.00")
    aggregate_drawn = Decimal("0.00")
    for acct in active_accounts:
        limit = _q(acct["contractual_limit"])
        outstanding = await get_outstanding_principal(acct["customer_id"])
        aggregate_contractual += limit
        aggregate_drawn += outstanding
        per_customer.append({
            "customer_id": acct["customer_id"],
            "contractual_limit": limit,
            "outstanding_principal": outstanding,
            "available_credit": limit - outstanding,
        })

    return ExposureSnapshot(
        active_customer_count=len(active_accounts),
        max_pilot_customers=max_customers,
        aggregate_contractual_exposure=aggregate_contractual,
        aggregate_contractual_cap=aggregate_cap,
        aggregate_drawn_exposure=aggregate_drawn,
        per_customer=per_customer,
    )


def check_exposure_thresholds(snapshot: ExposureSnapshot, max_outstanding_balance: Decimal) -> dict:
    """Returns a dict of warning levels ('70pct'/'80pct'/'90pct'/'breach'/
    None) for every monitored metric, using pilot_config.warning_level so
    the same 70/80/90% thresholds apply everywhere consistently. A
    'breach' anywhere here means enforcement upstream (activate_customer_
    credit_account / draw_credit) should already have prevented it — this
    function exists to catch and surface that as a critical alert if it
    ever happens anyway (e.g. a config was tightened after accounts were
    already activated), not to be the primary control."""
    warnings = {
        "customer_count": pc.warning_level(Decimal(snapshot.active_customer_count), Decimal(snapshot.max_pilot_customers)),
        "aggregate_contractual_exposure": pc.warning_level(snapshot.aggregate_contractual_exposure, snapshot.aggregate_contractual_cap),
        "per_customer_outstanding": {},
    }
    for c in snapshot.per_customer:
        level = pc.warning_level(c["outstanding_principal"], max_outstanding_balance)
        if level:
            warnings["per_customer_outstanding"][c["customer_id"]] = level
    return warnings


def has_critical_breach(warnings: dict) -> bool:
    """True if any monitored metric is at or beyond 'breach' — callers
    (e.g. a scheduled monitoring job) should block further new-lending
    activity and notify authorised administrators when this is true,
    mirroring launch_gates' fail-closed posture but for exposure limits
    specifically rather than regulatory gates."""
    if warnings.get("customer_count") == "breach":
        return True
    if warnings.get("aggregate_contractual_exposure") == "breach":
        return True
    if any(level == "breach" for level in warnings.get("per_customer_outstanding", {}).values()):
        return True
    return False
