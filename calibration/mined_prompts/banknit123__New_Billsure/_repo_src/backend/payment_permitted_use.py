"""
backend/payment_permitted_use.py
==================================
Permitted-use controls for pilot credit disbursements. This module is
the last line of defence before money moves: even a bill that passed
`bill_verification.py` cannot be paid unless the disbursement itself
also passes every check here.

Hard rule: a disbursement may ONLY pay a verified, approved bill, to the
exact biller and reference on that bill, for an amount that does not
exceed the bill's own amount. There is no code path in this module that
authorises a cash payment, a payment to a customer, a payment to a
personal account, a credit-card/loan/BNPL repayment, a gambling payment,
rent, a fine, a tax liability, or a payment to any merchant not on the
verified-biller allowlist — these are enumerated explicitly below and
checked first, before any amount/limit logic runs, so no combination of
otherwise-valid amounts can smuggle a prohibited payment through.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

import supabase_db as sdb

logger = logging.getLogger(__name__)

# Recipient/payment types that are NEVER permitted for this pilot,
# regardless of amount, approver, or override — there is deliberately no
# override mechanism in this module for this list.
PROHIBITED_PAYMENT_TYPES = frozenset({
    "cash_advance",
    "payment_to_customer",
    "payment_to_personal_account",
    "credit_card_repayment",
    "loan_or_bnpl_repayment",
    "gambling",
    "rent",
    "fine",
    "tax_liability",
    "unsupported_merchant",
})

ALLOWED_PAYMENT_TYPE = "verified_utility_biller"


class PermittedUseError(Exception):
    """Raised for any blocked or invalid disbursement request. Every
    raise path is a refusal; there is no partial-disbursement result."""


@dataclass
class DisbursementRequest:
    bill_id: str
    bill_status: str                 # verified | manual_review | rejected | pending
    bill_hash: str
    bill_amount: Decimal
    bill_already_disbursed: bool
    payment_type: str                # must equal ALLOWED_PAYMENT_TYPE
    recipient_biller_name: str
    bill_biller_name: str
    requested_amount: Decimal
    max_single_bill_payment: Decimal   # from pilot_config, passed in, never hard-coded here


def validate_disbursement(req: DisbursementRequest) -> None:
    """Raises PermittedUseError on any violation. Returns None (does not
    return a 'maybe') on success — there is no ambiguous outcome for a
    real-money instruction."""
    if req.payment_type in PROHIBITED_PAYMENT_TYPES:
        raise PermittedUseError(f"payment_type '{req.payment_type}' is prohibited for this pilot")
    if req.payment_type != ALLOWED_PAYMENT_TYPE:
        raise PermittedUseError(f"payment_type must be '{ALLOWED_PAYMENT_TYPE}', got '{req.payment_type}'")

    if req.bill_status != "verified":
        raise PermittedUseError(f"cannot disburse against a bill that is not verified (status={req.bill_status})")
    if req.bill_already_disbursed:
        raise PermittedUseError("this bill already has a disbursement linked — duplicate payment blocked")
    if req.recipient_biller_name != req.bill_biller_name:
        raise PermittedUseError("disbursement recipient does not match the verified bill's biller — blocked")

    if req.requested_amount <= 0:
        raise PermittedUseError("requested_amount must be positive")
    if req.requested_amount > req.bill_amount:
        raise PermittedUseError("requested_amount exceeds the approved bill's amount — blocked")
    if req.requested_amount > req.max_single_bill_payment:
        raise PermittedUseError("requested_amount exceeds the pilot's max single-bill payment limit — blocked")


async def create_disbursement(req: DisbursementRequest, requested_by: str, credit_journal_id: Optional[str] = None) -> dict:
    """Validates and, only on success, records a disbursement AND
    immediately links it back to the bill (marking it as paid) in the
    same call — there is deliberately no window where a disbursement
    exists but the bill isn't yet marked disbursed, which is exactly the
    kind of gap a duplicate-payment race would exploit.

    `credit_journal_id` is optional so this module still works standalone
    (as it did before pilot_payment_flow.py existed) — but any caller
    funding the disbursement from the credit ledger should pass the
    journal id returned by credit_ledger.draw_credit(), so a
    disbursement's funding source is always traceable back to the exact
    ledger journal that moved the money, not just asserted."""
    validate_disbursement(req)

    disbursement = await sdb.insert_one("pilot_bill_disbursements", {
        "bill_id": req.bill_id,
        "bill_hash": req.bill_hash,
        "amount": str(req.requested_amount),
        "recipient_biller_name": req.recipient_biller_name,
        "payment_type": req.payment_type,
        "requested_by": requested_by,
        "status": "queued",
        "credit_journal_id": credit_journal_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    await sdb.update_one("pilot_bill_submissions", {"id": req.bill_id}, {"disbursement_id": disbursement["id"]})
    return disbursement
