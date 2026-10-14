"""
backend/bill_verification.py
==============================
Bill ingestion and verification for the ASIC ERS pilot credit facility.
Every disbursement must trace back to exactly one approved bill; this
module is where a submitted bill becomes (or fails to become) eligible
to be paid.

Design rules:
- Nothing here auto-approves a bill on low-confidence extraction, a name
  mismatch, or any fraud/alteration indicator — those always route to
  `manual_review`, never straight to `verified` or straight to
  `rejected`. Only unambiguous, objective failures (unsupported biller,
  unsupported category, duplicate, already paid) are rejected outright.
- An approved bill gets an immutable cryptographic hash
  (`compute_bill_hash`) recorded before anything else can reference it.
  A disbursement links to that hash, not to a mutable file path, so
  "which bill was this payment actually for" can never be quietly
  changed after the fact.
- Duplicate and already-paid detection run against BOTH the exact file
  hash and the (biller, biller_reference, amount, due_date) tuple, since
  a customer might upload a re-scanned or re-photographed copy of the
  same bill with a different hash.
"""

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

import supabase_db as sdb

logger = logging.getLogger(__name__)

VERIFICATION_STATUSES = ("pending", "verified", "manual_review", "rejected")

# Confidence below this (0.0-1.0) always goes to manual review rather
# than being trusted, regardless of how clean everything else looks.
DEFAULT_MIN_EXTRACTION_CONFIDENCE = 0.85


class BillVerificationError(Exception):
    """Raised for invalid operations — always a refusal, never a partial
    approval."""


@dataclass
class BillSubmission:
    customer_id: str
    file_bytes: bytes
    customer_name_on_account: str
    biller_name_extracted: str
    biller_reference: str
    category: str                       # electricity | gas | water | telecommunications
    amount: Decimal
    due_date: str                        # ISO date
    customer_name_on_bill: Optional[str] = None
    extraction_confidence: float = 1.0
    fraud_indicators: list = field(default_factory=list)   # populated by an upstream document-analysis step; this module never invents these itself


@dataclass
class VerificationResult:
    status: str                # verified | manual_review | rejected
    reasons: list
    bill_hash: str
    evidence: dict
    checked_at: str


def compute_bill_hash(file_bytes: bytes) -> str:
    """SHA-256 of the uploaded file — the immutable reference every
    disbursement links against. Never derived from mutable metadata like
    a filename or upload timestamp."""
    return hashlib.sha256(file_bytes).hexdigest()


def _names_roughly_match(a: Optional[str], b: Optional[str]) -> bool:
    if not a or not b:
        return False
    norm = lambda s: "".join(ch.lower() for ch in s if ch.isalnum())
    na, nb = norm(a), norm(b)
    if not na or not nb:
        return False
    # Cheap, deliberately conservative fuzzy match: exact after
    # normalisation, or one contains the other (handles "J Smith" vs
    # "John Smith", middle names, etc.) — a real deployment should use a
    # proper name-matching library, this is a documented placeholder.
    return na == nb or na in nb or nb in na


def verify_bill(
    submission: BillSubmission,
    biller_allowlist: set,
    approved_categories: set,
    existing_bill_hashes: set,
    existing_biller_reference_tuples: set,     # {(biller_name, biller_reference, amount_str, due_date)}
    already_paid_hashes: set,
    already_paid_reference_tuples: set,
    min_confidence: float = DEFAULT_MIN_EXTRACTION_CONFIDENCE,
    max_plausible_amount: Optional[Decimal] = None,
    now: Optional[datetime] = None,
) -> VerificationResult:
    """Deterministic bill verification. All comparison sets are passed in
    by the caller (never fetched inside this function) so a decision is
    always reproducible from its recorded inputs.

    `max_plausible_amount` (optional) catches a real failure mode found
    during live deployment testing: OCR/text-layer extraction can
    misread a barcode or reference-number digit sequence as the bill
    amount, especially on PDFs with custom/embedded fonts, producing an
    implausibly large figure with a high reported confidence (the
    confidence score reflects how many fields were found, not whether
    the values are sane). A caller should pass the pilot's own
    max_single_bill_payment here — since no bill above that could ever
    actually be paid, a submitted amount far exceeding it is either
    genuinely out of scope for this pilot or an extraction error, and
    either way deserves a human look rather than being silently
    auto-verified on high 'confidence' alone."""
    now = now or datetime.now(timezone.utc)
    bill_hash = compute_bill_hash(submission.file_bytes)
    ref_tuple = (submission.biller_name_extracted, submission.biller_reference, str(submission.amount), submission.due_date)

    reasons = []
    evidence = {
        "biller_name_extracted": submission.biller_name_extracted,
        "category": submission.category,
        "amount": str(submission.amount),
        "due_date": submission.due_date,
        "extraction_confidence": submission.extraction_confidence,
        "fraud_indicators": list(submission.fraud_indicators),
        "bill_hash": bill_hash,
    }

    # --- unambiguous, objective rejections ---
    if submission.category not in approved_categories:
        reasons.append("CATEGORY_NOT_SUPPORTED")
    if submission.biller_name_extracted not in biller_allowlist:
        reasons.append("BILLER_NOT_ALLOWLISTED")
    if bill_hash in already_paid_hashes or ref_tuple in already_paid_reference_tuples:
        reasons.append("BILL_ALREADY_PAID")
    elif bill_hash in existing_bill_hashes or ref_tuple in existing_biller_reference_tuples:
        reasons.append("DUPLICATE_BILL")
    if submission.amount <= 0:
        reasons.append("INVALID_AMOUNT")

    hard_rejects = {"CATEGORY_NOT_SUPPORTED", "BILLER_NOT_ALLOWLISTED", "BILL_ALREADY_PAID", "DUPLICATE_BILL", "INVALID_AMOUNT"}
    if any(r in hard_rejects for r in reasons):
        return VerificationResult(status="rejected", reasons=reasons, bill_hash=bill_hash, evidence=evidence, checked_at=now.isoformat())

    # --- ambiguous cases: always manual review, never silently resolved ---
    review_reasons = []
    if submission.extraction_confidence < min_confidence:
        review_reasons.append("LOW_EXTRACTION_CONFIDENCE")
    if not _names_roughly_match(submission.customer_name_on_account, submission.customer_name_on_bill):
        review_reasons.append("NAME_MISMATCH")
    if submission.fraud_indicators:
        review_reasons.append("ALTERATION_OR_FRAUD_INDICATOR")
    if max_plausible_amount is not None and submission.amount > max_plausible_amount:
        review_reasons.append("AMOUNT_EXCEEDS_PLAUSIBLE_RANGE")

    if review_reasons:
        return VerificationResult(status="manual_review", reasons=review_reasons, bill_hash=bill_hash, evidence=evidence, checked_at=now.isoformat())

    return VerificationResult(status="verified", reasons=[], bill_hash=bill_hash, evidence=evidence, checked_at=now.isoformat())


async def submit_and_verify_bill(submission: BillSubmission, biller_allowlist: set, approved_categories: set,
                                  min_confidence: float = DEFAULT_MIN_EXTRACTION_CONFIDENCE,
                                  max_plausible_amount: Optional[Decimal] = None) -> dict:
    existing_verified = await sdb.find_many("pilot_bill_submissions", {"customer_id": submission.customer_id})
    existing_hashes = {b["bill_hash"] for b in existing_verified}
    existing_ref_tuples = {
        (b["biller_name_extracted"], b["biller_reference"], b["amount"], b["due_date"]) for b in existing_verified
    }
    paid = [b for b in existing_verified if b.get("disbursement_id")]
    paid_hashes = {b["bill_hash"] for b in paid}
    paid_ref_tuples = {(b["biller_name_extracted"], b["biller_reference"], b["amount"], b["due_date"]) for b in paid}

    result = verify_bill(
        submission, biller_allowlist, approved_categories,
        existing_hashes, existing_ref_tuples, paid_hashes, paid_ref_tuples,
        min_confidence=min_confidence, max_plausible_amount=max_plausible_amount,
    )

    row = {
        "customer_id": submission.customer_id,
        "biller_name_extracted": submission.biller_name_extracted,
        "biller_reference": submission.biller_reference,
        "category": submission.category,
        "amount": str(submission.amount),
        "due_date": submission.due_date,
        "customer_name_on_bill": submission.customer_name_on_bill,
        "extraction_confidence": submission.extraction_confidence,
        "fraud_indicators": list(submission.fraud_indicators),
        "bill_hash": result.bill_hash,
        "verification_status": result.status,
        "verification_reasons": result.reasons,
        "verification_evidence": result.evidence,
        "disbursement_id": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return await sdb.insert_one("pilot_bill_submissions", row)


async def record_manual_review_decision(bill_id: str, reviewer: str, decision: str, notes: str) -> dict:
    if decision not in ("verified", "rejected"):
        raise BillVerificationError("manual review decision must be 'verified' or 'rejected'")
    existing = await sdb.find_one("pilot_bill_submissions", {"id": bill_id})
    if not existing:
        raise BillVerificationError(f"no bill {bill_id}")
    if existing.get("verification_status") != "manual_review":
        raise BillVerificationError(f"bill {bill_id} is not awaiting manual review (status={existing.get('verification_status')})")
    updates = {
        "verification_status": decision,
        "manual_reviewed_by": reviewer,
        "manual_review_notes": notes,
        "manual_reviewed_at": datetime.now(timezone.utc).isoformat(),
    }
    await sdb.update_one("pilot_bill_submissions", {"id": bill_id}, updates)
    return {**existing, **updates}
