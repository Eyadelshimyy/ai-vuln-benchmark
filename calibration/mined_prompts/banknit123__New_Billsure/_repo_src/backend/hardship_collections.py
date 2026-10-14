"""
backend/hardship_collections.py
==================================
Repayments, hardship, and collections for the ASIC ERS pilot credit
facility. Implements task section 9's explicit requirement that this
pilot must NOT use aggressive collections, default fees, or automated
adverse action — every function here either records a human decision or
performs a mechanical, policy-bounded action; nothing in this module
escalates a customer's situation, applies a fee, or takes an adverse
action without a person choosing to do so.

Relationship to other modules:
- Repayments are recorded here (schedule/installment bookkeeping) AND
  posted to the ledger via `credit_ledger.repay_credit()` in the same
  call — mirroring `pilot_payment_flow.py`'s discipline of never letting
  bookkeeping state and ledger state drift apart. A failed ledger call
  (e.g. attempted overpayment) means the installment is NOT marked paid.
- Fee/interest suppression isn't optional configuration here — pilot_
  config.py already enforces 0% interest and $0 fees pilot-wide, and
  this module additionally asserts that no hardship arrangement can
  introduce a nonzero fee or interest rate, as a second, independent
  check specific to hardship arrangements (defence in depth: two
  different modules refusing the same wrong thing for two different
  reasons is more robust than either alone).
- A hardship request never checks payment history, outstanding balance,
  or account standing before being accepted — see
  `request_hardship()`'s docstring. This is deliberate, not an
  oversight: "hardship customers must access support without first
  making a payment" is a hard requirement, not a nice-to-have.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional

import supabase_db as sdb
import credit_ledger as cl

logger = logging.getLogger(__name__)

INSTALLMENT_STATUSES = ("scheduled", "paid", "partial", "failed", "skipped")
HARDSHIP_CASE_STATUSES = ("open", "arrangement_proposed", "arrangement_active", "escalated", "closed")
MAX_RESCHEDULES_PER_INSTALLMENT = 3


def _q(amount) -> Decimal:
    return Decimal(str(amount)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


class HardshipCollectionsError(Exception):
    """Raised for any invalid operation. Every raise path here is a
    refusal, never a fallback to an automated adverse action."""


# ---------------------------------------------------------------
# Repayment schedules
# ---------------------------------------------------------------

async def generate_repayment_schedule(customer_id: str, principal: Decimal, term_months: int,
                                       first_due_date: str, created_by: str) -> dict:
    """Splits `principal` into `term_months` equal monthly installments
    (the final installment absorbs any rounding remainder, so the sum
    always equals `principal` exactly — no installment is ever silently
    off by a cent in a way that compounds). Pure schedule bookkeeping;
    does not touch the ledger — the schedule describes what SHOULD be
    paid, actual payments are recorded separately via record_repayment()."""
    if term_months <= 0:
        raise HardshipCollectionsError("term_months must be positive")
    principal = _q(principal)
    base_installment = _q(principal / term_months)
    installments = []
    running_total = Decimal("0.00")
    due = datetime.fromisoformat(first_due_date)
    for i in range(term_months):
        if i == term_months - 1:
            amount = principal - running_total
        else:
            amount = base_installment
            running_total += amount
        installments.append({
            "sequence": i + 1,
            "due_date": (due + timedelta(days=30 * i)).date().isoformat(),
            "scheduled_amount": str(_q(amount)),
            "status": "scheduled",
            "reschedule_count": 0,
        })

    schedule = await sdb.insert_one("repayment_schedules", {
        "customer_id": customer_id, "principal": str(principal), "term_months": term_months,
        "created_by": created_by, "created_at": datetime.now(timezone.utc).isoformat(),
    })
    stored_installments = []
    for inst in installments:
        stored_installments.append(await sdb.insert_one("repayment_installments", {**inst, "schedule_id": schedule["id"]}))

    total_scheduled = sum(Decimal(i["scheduled_amount"]) for i in stored_installments)
    if total_scheduled != principal:
        raise HardshipCollectionsError(f"schedule generation rounding error: installments sum to {total_scheduled}, expected {principal}")

    return {**schedule, "installments": stored_installments}


async def record_repayment(installment_id: str, customer_id: str, amount: Decimal, payment_date: Optional[str] = None,
                            recorded_by: Optional[str] = None) -> dict:
    """Records a repayment against a specific installment AND posts it
    to the credit ledger in the same call. If credit_ledger.repay_credit
    raises (e.g. amount exceeds outstanding principal), this function
    propagates that error and does NOT mark the installment paid —
    bookkeeping state never drifts ahead of what the ledger actually
    recorded.

    Classification (advance / on-time partial / on-time full) is
    informational only — it does not change how the ledger is updated,
    only how the installment's status field reads afterward."""
    installment = await sdb.find_one("repayment_installments", {"id": installment_id})
    if not installment:
        raise HardshipCollectionsError(f"no installment {installment_id}")
    if installment.get("status") == "paid":
        raise HardshipCollectionsError(f"installment {installment_id} is already fully paid")

    amount = _q(amount)
    if amount <= 0:
        raise HardshipCollectionsError("repayment amount must be positive")

    # This raises (and installment stays untouched) if amount exceeds
    # the customer's actual outstanding principal — see credit_ledger.
    journal_id = await cl.repay_credit(customer_id, amount, reference_id=installment_id, created_by=recorded_by)

    scheduled = Decimal(installment["scheduled_amount"])
    # Compare CUMULATIVE amount paid across every repayment recorded
    # against this installment, not just this single payment — a
    # customer paying an installment off in two partial payments must
    # end up 'paid' once the total reaches the scheduled amount, not
    # stay 'partial' forever because neither individual payment alone
    # reached it.
    previously_paid = Decimal(str(installment.get("amount_paid") or "0"))
    cumulative_paid = _q(previously_paid + amount)
    new_status = "paid" if cumulative_paid >= scheduled else "partial"

    updates = {
        "status": new_status,
        "amount_paid": str(cumulative_paid),
        "last_payment_amount": str(amount),
        "payment_date": payment_date or datetime.now(timezone.utc).isoformat(),
        "credit_journal_id": journal_id,
        "is_advance": (payment_date or "") < installment["due_date"] if payment_date else False,
    }
    await sdb.update_one("repayment_installments", {"id": installment_id}, updates)
    return {**installment, **updates}


async def record_failed_repayment(installment_id: str, reason: str, recorded_by: str) -> dict:
    """Marks an installment's most recent payment attempt as failed.
    Deliberately does NOTHING else: no fee is applied (pilot fees are
    always $0, see pilot_config.py, and this function has no fee field
    to set even if that changed), no automatic escalation, no automatic
    adverse action of any kind. A human must separately choose to
    escalate (escalate_hardship_case()) or propose an arrangement — this
    function only records the fact that a payment attempt failed."""
    installment = await sdb.find_one("repayment_installments", {"id": installment_id})
    if not installment:
        raise HardshipCollectionsError(f"no installment {installment_id}")
    updates = {"status": "failed", "failure_reason": reason, "failed_recorded_by": recorded_by,
               "failed_at": datetime.now(timezone.utc).isoformat()}
    await sdb.update_one("repayment_installments", {"id": installment_id}, updates)
    return {**installment, **updates}


async def reschedule_installment(installment_id: str, new_due_date: str, reason: str, requested_by: str) -> dict:
    """Reschedules a single installment's due date. Bounded by
    MAX_RESCHEDULES_PER_INSTALLMENT — a policy limit, not a punitive
    fee; exceeding it routes to manual review (raises, caller should
    escalate to a hardship arrangement instead of a simple reschedule)
    rather than silently allowing unlimited pushes or applying any
    charge."""
    installment = await sdb.find_one("repayment_installments", {"id": installment_id})
    if not installment:
        raise HardshipCollectionsError(f"no installment {installment_id}")
    count = installment.get("reschedule_count", 0)
    if count >= MAX_RESCHEDULES_PER_INSTALLMENT:
        raise HardshipCollectionsError(
            f"installment {installment_id} has already been rescheduled {count} times "
            f"(policy limit {MAX_RESCHEDULES_PER_INSTALLMENT}) — propose a hardship arrangement instead"
        )
    updates = {"due_date": new_due_date, "reschedule_count": count + 1, "reschedule_reason": reason, "rescheduled_by": requested_by}
    await sdb.update_one("repayment_installments", {"id": installment_id}, updates)
    return {**installment, **updates}


# ---------------------------------------------------------------
# Hardship intake — no payment-status gate, ever
# ---------------------------------------------------------------

async def request_hardship(customer_id: str, reason: str, vulnerability_indicators: list, requested_by: str) -> dict:
    """Opens a hardship case. THIS FUNCTION DOES NOT CHECK the
    customer's outstanding balance, payment history, or account
    standing before accepting the request — 'hardship customers must be
    able to access support without first making a payment' is a hard
    requirement (task section 9), not a default this function happens
    to satisfy. There is no code path here that could reject a hardship
    request because of an unpaid installment, a failed payment, or
    zero payment history."""
    return await sdb.insert_one("hardship_cases", {
        "customer_id": customer_id, "reason": reason, "vulnerability_indicators": list(vulnerability_indicators),
        "status": "open", "requested_by": requested_by, "created_at": datetime.now(timezone.utc).isoformat(),
        "escalation_history": [],
    })


async def pause_collections(hardship_case_id: str, paused_by: str, approved_by: str, pause_until: str) -> dict:
    """Temporarily suspends collection activity for the customer tied to
    this hardship case. Maker-checker: approved_by must differ from
    paused_by, same discipline as every other approval gate in this
    codebase."""
    if paused_by == approved_by:
        raise HardshipCollectionsError("collection pause requires a distinct approver (maker-checker)")
    case = await sdb.find_one("hardship_cases", {"id": hardship_case_id})
    if not case:
        raise HardshipCollectionsError(f"no hardship case {hardship_case_id}")

    return await sdb.insert_one("collection_pauses", {
        "hardship_case_id": hardship_case_id, "customer_id": case["customer_id"],
        "paused_by": paused_by, "approved_by": approved_by, "pause_until": pause_until,
        "created_at": datetime.now(timezone.utc).isoformat(), "active": True,
    })


async def is_collection_paused(customer_id: str, now: Optional[datetime] = None) -> bool:
    """True if the customer has any active, unexpired collection pause.
    Other modules (a future collections/reminder job) should consult
    this before sending any repayment reminder or escalation — not
    wired into anything yet, this is the check function such a job
    would call."""
    now = now or datetime.now(timezone.utc)
    pauses = await sdb.find_many("collection_pauses", {"customer_id": customer_id, "active": True})
    for p in pauses:
        until = datetime.fromisoformat(p["pause_until"])
        if until.tzinfo is None:
            until = until.replace(tzinfo=timezone.utc)
        if until > now:
            return True
    return False


# ---------------------------------------------------------------
# Hardship arrangements — maker-checker, fee/interest suppression
# ---------------------------------------------------------------

async def propose_hardship_arrangement(hardship_case_id: str, new_installments: list, proposed_by: str) -> dict:
    """Proposes a revised repayment arrangement (e.g. reduced instalment
    amounts, extended term) for a hardship case. Refuses outright if any
    proposed installment carries a nonzero fee or implies interest —
    this is an independent check from pilot_config's pilot-wide 0%/$0
    enforcement, specific to hardship arrangements, so a bug or future
    change in one doesn't silently let a fee through the other."""
    for inst in new_installments:
        if Decimal(str(inst.get("fee_amount", "0"))) != Decimal("0"):
            raise HardshipCollectionsError("hardship arrangements cannot include any fee — pilot fees are always $0")
        if Decimal(str(inst.get("interest_amount", "0"))) != Decimal("0"):
            raise HardshipCollectionsError("hardship arrangements cannot include any interest — pilot interest is always 0%")

    case = await sdb.find_one("hardship_cases", {"id": hardship_case_id})
    if not case:
        raise HardshipCollectionsError(f"no hardship case {hardship_case_id}")

    arrangement = await sdb.insert_one("hardship_arrangements", {
        "hardship_case_id": hardship_case_id, "customer_id": case["customer_id"],
        "proposed_installments": new_installments, "proposed_by": proposed_by,
        "status": "proposed", "created_at": datetime.now(timezone.utc).isoformat(),
    })
    await sdb.update_one("hardship_cases", {"id": hardship_case_id}, {"status": "arrangement_proposed"})
    return arrangement


async def approve_hardship_arrangement(arrangement_id: str, approved_by: str) -> dict:
    """Maker-checker: approver must differ from the person who proposed
    the arrangement. Only on approval does the case move to
    'arrangement_active' — a proposed-but-unapproved arrangement has no
    effect on the customer's actual schedule."""
    arrangement = await sdb.find_one("hardship_arrangements", {"id": arrangement_id})
    if not arrangement:
        raise HardshipCollectionsError(f"no arrangement {arrangement_id}")
    if arrangement.get("proposed_by") == approved_by:
        raise HardshipCollectionsError("arrangement approval requires a distinct approver from the proposer (maker-checker)")
    if arrangement.get("status") != "proposed":
        raise HardshipCollectionsError(f"arrangement {arrangement_id} is not awaiting approval (status={arrangement.get('status')})")

    updates = {"status": "approved", "approved_by": approved_by, "approved_at": datetime.now(timezone.utc).isoformat()}
    await sdb.update_one("hardship_arrangements", {"id": arrangement_id}, updates)
    await sdb.update_one("hardship_cases", {"id": arrangement["hardship_case_id"]}, {"status": "arrangement_active"})
    return {**arrangement, **updates}


# ---------------------------------------------------------------
# Escalation — always a human decision, always audited
# ---------------------------------------------------------------

async def escalate_hardship_case(hardship_case_id: str, escalated_by: str, reason: str) -> dict:
    """Records an escalation. This is the ONLY function in this module
    that moves a case toward 'escalated' status, and it always requires
    a human caller and a documented reason — there is no automatic
    trigger anywhere in this module (not on a failed repayment, not on
    a missed reschedule deadline, not on anything) that calls this."""
    case = await sdb.find_one("hardship_cases", {"id": hardship_case_id})
    if not case:
        raise HardshipCollectionsError(f"no hardship case {hardship_case_id}")
    if not reason or not reason.strip():
        raise HardshipCollectionsError("escalation requires a documented reason")

    history = list(case.get("escalation_history", []))
    history.append({"escalated_by": escalated_by, "reason": reason, "timestamp": datetime.now(timezone.utc).isoformat()})
    updates = {"status": "escalated", "escalation_history": history}
    await sdb.update_one("hardship_cases", {"id": hardship_case_id}, updates)
    return {**case, **updates}
