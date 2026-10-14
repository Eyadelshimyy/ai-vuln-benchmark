"""
Standalone logic test for bill_verification.py and
payment_permitted_use.py — same in-memory fake-DB pattern as the other
test_*.py files in this repo. No live credentials needed.

Run: python3 test_bill_verification_and_permitted_use.py
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


async def update_one(table, filters, updates):
    for row in _tables.get(table, []):
        if _matches(row, filters):
            row.update(updates)
            return True
    return False


fake_sdb = types.SimpleNamespace(find_one=find_one, find_many=find_many, insert_one=insert_one, update_one=update_one)
sys.modules["supabase_db"] = fake_sdb

import bill_verification as bv           # noqa: E402
import payment_permitted_use as ppu      # noqa: E402

FAILURES = []


def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}")
    if not condition:
        FAILURES.append(label)


BILLER_ALLOWLIST = {"AusNet Electricity", "Origin Energy", "Yarra Valley Water"}
APPROVED_CATEGORIES = {"electricity", "gas", "water", "telecommunications"}


def _submission(**overrides):
    defaults = dict(
        customer_id="cust-1",
        file_bytes=b"fake pdf bytes for bill A",
        customer_name_on_account="Jane Citizen",
        customer_name_on_bill="Jane Citizen",
        biller_name_extracted="AusNet Electricity",
        biller_reference="REF-001",
        category="electricity",
        amount=Decimal("150.00"),
        due_date="2026-09-01",
        extraction_confidence=0.97,
        fraud_indicators=[],
    )
    defaults.update(overrides)
    return bv.BillSubmission(**defaults)


async def main():
    # ---------------------------------------------------------------
    # bill_verification: clean bill verifies
    # ---------------------------------------------------------------
    clean = await bv.submit_and_verify_bill(_submission(), BILLER_ALLOWLIST, APPROVED_CATEGORIES)
    check("clean, allowlisted, matching-name bill is 'verified'", clean["verification_status"] == "verified")
    check("verified bill has no reasons attached", clean["verification_reasons"] == [])
    check("bill_hash was computed and stored", len(clean["bill_hash"]) == 64)

    # ---------------------------------------------------------------
    # bill_verification: hard rejects (never manual_review for these)
    # ---------------------------------------------------------------
    not_allowlisted = await bv.submit_and_verify_bill(
        _submission(customer_id="cust-2", biller_name_extracted="Random Merchant Pty Ltd"),
        BILLER_ALLOWLIST, APPROVED_CATEGORIES)
    check("non-allowlisted biller is rejected outright", not_allowlisted["verification_status"] == "rejected")
    check("rejection carries BILLER_NOT_ALLOWLISTED", "BILLER_NOT_ALLOWLISTED" in not_allowlisted["verification_reasons"])

    bad_category = await bv.submit_and_verify_bill(
        _submission(customer_id="cust-3", category="rent"), BILLER_ALLOWLIST, APPROVED_CATEGORIES)
    check("unsupported category (rent) is rejected outright", bad_category["verification_status"] == "rejected")

    bad_amount = await bv.submit_and_verify_bill(
        _submission(customer_id="cust-4", amount=Decimal("-10.00")), BILLER_ALLOWLIST, APPROVED_CATEGORIES)
    check("non-positive amount is rejected", bad_amount["verification_status"] == "rejected")

    # ---------------------------------------------------------------
    # bill_verification: duplicate + already-paid detection
    # ---------------------------------------------------------------
    dup = await bv.submit_and_verify_bill(_submission(), BILLER_ALLOWLIST, APPROVED_CATEGORIES)  # cust-1 again, identical
    check("resubmitting the identical bill for the same customer is rejected as DUPLICATE_BILL",
          dup["verification_status"] == "rejected" and "DUPLICATE_BILL" in dup["verification_reasons"])

    # Mark the original as disbursed, then resubmitting must say ALREADY_PAID, not just DUPLICATE.
    await update_one("pilot_bill_submissions", {"id": clean["id"]}, {"disbursement_id": "disb-1"})
    already_paid = await bv.submit_and_verify_bill(_submission(), BILLER_ALLOWLIST, APPROVED_CATEGORIES)
    check("resubmitting an already-paid bill is rejected as BILL_ALREADY_PAID",
          already_paid["verification_status"] == "rejected" and "BILL_ALREADY_PAID" in already_paid["verification_reasons"])

    # ---------------------------------------------------------------
    # bill_verification: ambiguous cases -> manual_review, never silently resolved
    # ---------------------------------------------------------------
    low_confidence = await bv.submit_and_verify_bill(
        _submission(customer_id="cust-5", file_bytes=b"unique bytes 5", extraction_confidence=0.5),
        BILLER_ALLOWLIST, APPROVED_CATEGORIES)
    check("low-confidence extraction goes to manual_review, not auto-verified",
          low_confidence["verification_status"] == "manual_review" and "LOW_EXTRACTION_CONFIDENCE" in low_confidence["verification_reasons"])

    name_mismatch = await bv.submit_and_verify_bill(
        _submission(customer_id="cust-6", file_bytes=b"unique bytes 6", customer_name_on_bill="Someone Else Entirely"),
        BILLER_ALLOWLIST, APPROVED_CATEGORIES)
    check("name mismatch goes to manual_review, not auto-rejected or auto-verified",
          name_mismatch["verification_status"] == "manual_review" and "NAME_MISMATCH" in name_mismatch["verification_reasons"])

    fraud_flagged = await bv.submit_and_verify_bill(
        _submission(customer_id="cust-7", file_bytes=b"unique bytes 7", fraud_indicators=["inconsistent_font_detected"]),
        BILLER_ALLOWLIST, APPROVED_CATEGORIES)
    check("fraud indicator goes to manual_review, never auto-rejected without a human",
          fraud_flagged["verification_status"] == "manual_review" and "ALTERATION_OR_FRAUD_INDICATOR" in fraud_flagged["verification_reasons"])

    # ---------------------------------------------------------------
    # bill_verification: manual review decision workflow
    # ---------------------------------------------------------------
    try:
        await bv.record_manual_review_decision(clean["id"], "reviewer1", "verified", "n/a")
        check("rejects manual review on a bill that isn't in manual_review status", False)
    except bv.BillVerificationError:
        check("rejects manual review on a bill that isn't in manual_review status", True)

    resolved = await bv.record_manual_review_decision(low_confidence["id"], "reviewer1", "verified", "confirmed against original PDF")
    check("manual review can move a low-confidence bill to verified", resolved["verification_status"] == "verified")

    try:
        await bv.record_manual_review_decision(name_mismatch["id"], "reviewer1", "declined", "not a valid decision string")
        check("rejects an invalid manual review decision value", False)
    except bv.BillVerificationError:
        check("rejects an invalid manual review decision value", True)

    # ---------------------------------------------------------------
    # max_plausible_amount: found live during deployment testing --
    # a real bill with a custom/embedded PDF font produced a garbled
    # text layer, and the amount-extraction heuristic misread a
    # barcode/reference digit sequence as a $10,000 bill amount, with
    # extraction_confidence still reporting 1.0 (confidence reflects
    # how many fields were found, not whether the values are sane).
    # This would have been auto-verified without this check.
    # ---------------------------------------------------------------
    implausible = await bv.submit_and_verify_bill(
        _submission(customer_id="cust-8", file_bytes=b"unique bytes 8", amount=Decimal("10000.00")),
        BILLER_ALLOWLIST, APPROVED_CATEGORIES, max_plausible_amount=Decimal("500.00"))
    check("an amount far exceeding max_plausible_amount goes to manual_review, not auto-verified on high confidence alone",
          implausible["verification_status"] == "manual_review" and "AMOUNT_EXCEEDS_PLAUSIBLE_RANGE" in implausible["verification_reasons"])

    within_range = await bv.submit_and_verify_bill(
        _submission(customer_id="cust-9", file_bytes=b"unique bytes 9", amount=Decimal("150.00")),
        BILLER_ALLOWLIST, APPROVED_CATEGORIES, max_plausible_amount=Decimal("500.00"))
    check("an amount within max_plausible_amount is unaffected by the check and verifies normally",
          within_range["verification_status"] == "verified")

    no_bound_set = await bv.submit_and_verify_bill(
        _submission(customer_id="cust-10", file_bytes=b"unique bytes 10", amount=Decimal("10000.00")),
        BILLER_ALLOWLIST, APPROVED_CATEGORIES)  # max_plausible_amount not passed -- defaults to None, check disabled
    check("with no max_plausible_amount passed at all, the check is simply skipped (backward compatible, no crash)",
          no_bound_set["verification_status"] == "verified")

    # ---------------------------------------------------------------
    # payment_permitted_use: prohibited payment types always blocked
    # ---------------------------------------------------------------
    base_req = dict(
        bill_id=clean["id"], bill_status="verified", bill_hash=clean["bill_hash"],
        bill_amount=Decimal("150.00"), bill_already_disbursed=False,
        payment_type="verified_utility_biller", recipient_biller_name="AusNet Electricity",
        bill_biller_name="AusNet Electricity", requested_amount=Decimal("150.00"),
        max_single_bill_payment=Decimal("500.00"),
    )

    for prohibited in ("cash_advance", "payment_to_customer", "payment_to_personal_account",
                        "credit_card_repayment", "loan_or_bnpl_repayment", "gambling", "rent", "fine", "tax_liability"):
        req = ppu.DisbursementRequest(**{**base_req, "payment_type": prohibited})
        try:
            ppu.validate_disbursement(req)
            check(f"blocks prohibited payment_type '{prohibited}'", False)
        except ppu.PermittedUseError:
            check(f"blocks prohibited payment_type '{prohibited}'", True)

    # ---------------------------------------------------------------
    # payment_permitted_use: bill-status / linkage checks
    # ---------------------------------------------------------------
    not_verified = ppu.DisbursementRequest(**{**base_req, "bill_status": "pending"})
    try:
        ppu.validate_disbursement(not_verified)
        check("blocks disbursement against a bill that isn't verified", False)
    except ppu.PermittedUseError:
        check("blocks disbursement against a bill that isn't verified", True)

    already_disbursed = ppu.DisbursementRequest(**{**base_req, "bill_already_disbursed": True})
    try:
        ppu.validate_disbursement(already_disbursed)
        check("blocks a duplicate disbursement against an already-paid bill", False)
    except ppu.PermittedUseError:
        check("blocks a duplicate disbursement against an already-paid bill", True)

    wrong_recipient = ppu.DisbursementRequest(**{**base_req, "recipient_biller_name": "A Different Biller"})
    try:
        ppu.validate_disbursement(wrong_recipient)
        check("blocks a disbursement whose recipient doesn't match the verified bill's biller", False)
    except ppu.PermittedUseError:
        check("blocks a disbursement whose recipient doesn't match the verified bill's biller", True)

    # ---------------------------------------------------------------
    # payment_permitted_use: amount limits
    # ---------------------------------------------------------------
    over_bill_amount = ppu.DisbursementRequest(**{**base_req, "requested_amount": Decimal("999.00")})
    try:
        ppu.validate_disbursement(over_bill_amount)
        check("blocks a disbursement amount exceeding the approved bill's amount", False)
    except ppu.PermittedUseError:
        check("blocks a disbursement amount exceeding the approved bill's amount", True)

    over_pilot_limit = ppu.DisbursementRequest(**{**base_req, "bill_amount": Decimal("600.00"), "requested_amount": Decimal("600.00")})
    try:
        ppu.validate_disbursement(over_pilot_limit)
        check("blocks a disbursement amount exceeding the pilot's max single-bill limit even if the bill itself is bigger", False)
    except ppu.PermittedUseError:
        check("blocks a disbursement amount exceeding the pilot's max single-bill limit even if the bill itself is bigger", True)

    zero_amount = ppu.DisbursementRequest(**{**base_req, "requested_amount": Decimal("0.00")})
    try:
        ppu.validate_disbursement(zero_amount)
        check("blocks a zero-amount disbursement", False)
    except ppu.PermittedUseError:
        check("blocks a zero-amount disbursement", True)

    # ---------------------------------------------------------------
    # payment_permitted_use: happy path creates and links a disbursement
    # ---------------------------------------------------------------
    good_req = ppu.DisbursementRequest(**base_req)
    disbursement = await ppu.create_disbursement(good_req, requested_by="admin1")
    check("valid disbursement is created", disbursement["status"] == "queued" and disbursement["amount"] == "150.00")

    bill_row = await find_one("pilot_bill_submissions", {"id": clean["id"]})
    check("the underlying bill is linked to its disbursement immediately", bill_row["disbursement_id"] == disbursement["id"])

    # A second attempt against the same (now-linked) bill must fail even
    # though `base_req` still says bill_already_disbursed=False -- this
    # simulates the caller re-checking against updated state, since
    # validate_disbursement itself takes bill_already_disbursed as an
    # input rather than re-querying (kept pure/deterministic).
    second_attempt = ppu.DisbursementRequest(**{**base_req, "bill_already_disbursed": True})
    try:
        await ppu.create_disbursement(second_attempt, requested_by="admin2")
        check("blocks a second disbursement against an already-linked bill", False)
    except ppu.PermittedUseError:
        check("blocks a second disbursement against an already-linked bill", True)

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
