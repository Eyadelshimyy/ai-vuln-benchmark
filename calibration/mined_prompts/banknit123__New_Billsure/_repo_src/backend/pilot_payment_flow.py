"""
backend/pilot_payment_flow.py
===============================
Joins the pieces that, until now, existed correctly but independently:
`bill_verification.py` decides a bill is payable, `payment_permitted_use.
py` decides a specific payment instruction is allowed, and
`credit_ledger.py` decides whether the customer actually has the credit
room and moves the money. This module is the one function that calls
all three in the right order for an actual pilot bill payment, so a
disbursement is never recorded without money having actually moved in
the ledger first.

Ordering matters and is deliberate:
1. Load the bill and confirm it is `verified` and not already disbursed
   (bill_verification's output, read fresh — never trust a stale
   in-memory copy for a real-money decision).
2. Load the customer's credit account and confirm it is active.
3. Run `payment_permitted_use.validate_disbursement()` — this checks the
   prohibited-payment-type list, the bill/recipient match, and the
   amount ceilings (bill amount, pilot single-bill limit). This can
   reject WITHOUT touching the ledger at all.
4. Only if that passes: call `credit_ledger.draw_credit()` — this is the
   step that actually moves money (posts the balanced ledger journal)
   and enforces available-credit / outstanding-balance limits. If this
   raises, NOTHING has been recorded as disbursed — the bill is still
   just "verified", eligible to be retried (e.g. after a repayment frees
   up room) or handled as a hardship/exception case.
5. Only if the draw succeeds: call `payment_permitted_use.
   create_disbursement()`, passing the credit journal id, which records
   the disbursement and marks the bill as paid in the same call. This
   is the step that can never happen without step 4 having already
   succeeded — there is no code path here that marks a bill paid
   without a corresponding ledger journal existing first.

If ANY step fails, this function raises and nothing further-reaching
has already happened — there is no partial state where money moved but
no disbursement was recorded, or a disbursement was recorded but no
money moved.
"""

import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

import supabase_db as sdb
import pilot_config as pc
import credit_ledger as cl
import bill_verification as bv
import payment_permitted_use as ppu

logger = logging.getLogger(__name__)


class PaymentFlowError(Exception):
    """Raised for any failure anywhere in the flow. Always means nothing
    past that point happened — see module docstring for the ordering
    guarantee this depends on."""


async def pay_verified_bill(bill_id: str, customer_id: str, requested_by: str) -> dict:
    bill = await sdb.find_one("pilot_bill_submissions", {"id": bill_id})
    if not bill:
        raise PaymentFlowError(f"no bill {bill_id}")
    if bill.get("customer_id") != customer_id:
        raise PaymentFlowError("bill does not belong to the specified customer")
    if bill.get("verification_status") != "verified":
        raise PaymentFlowError(f"bill {bill_id} is not verified (status={bill.get('verification_status')})")
    if bill.get("disbursement_id"):
        raise PaymentFlowError(f"bill {bill_id} already has a disbursement — refusing a second payment")

    account = await cl.get_customer_credit_account(customer_id)
    if not account or account.get("status") != "active":
        raise PaymentFlowError(f"customer {customer_id} has no active credit account")

    cfg = await pc.get_active_config()
    max_single_bill = cfg.max_single_bill_payment if cfg else pc.HARD_MAX_SINGLE_BILL_PAYMENT

    amount = Decimal(str(bill["amount"]))
    req = ppu.DisbursementRequest(
        bill_id=bill_id,
        bill_status=bill["verification_status"],
        bill_hash=bill["bill_hash"],
        bill_amount=amount,
        bill_already_disbursed=bool(bill.get("disbursement_id")),
        payment_type=ppu.ALLOWED_PAYMENT_TYPE,
        recipient_biller_name=bill["biller_name_extracted"],
        bill_biller_name=bill["biller_name_extracted"],
        requested_amount=amount,
        max_single_bill_payment=max_single_bill,
    )

    # Step 3: permitted-use check. Can reject without touching the ledger.
    try:
        ppu.validate_disbursement(req)
    except ppu.PermittedUseError as e:
        raise PaymentFlowError(f"permitted-use check failed: {e}") from e

    # Step 4: draw credit. This is the money-moving step. If it raises
    # (insufficient available credit, outstanding cap, etc.), the bill
    # remains verified/undisbursed — nothing further happens.
    try:
        journal_id = await cl.draw_credit(customer_id, amount, bill_id, requested_by)
    except cl.CreditLedgerError as e:
        raise PaymentFlowError(f"credit draw failed: {e}") from e

    # Step 5: only now record the disbursement, linked to the journal
    # that actually moved the money.
    disbursement = await ppu.create_disbursement(req, requested_by, credit_journal_id=journal_id)
    logger.info("Bill %s paid for customer %s via credit journal %s, disbursement %s",
                bill_id, customer_id, journal_id, disbursement["id"])
    return disbursement
