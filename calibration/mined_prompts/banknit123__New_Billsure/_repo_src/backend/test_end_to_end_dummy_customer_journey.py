"""
End-to-end integration test: a single dummy customer's full journey
through the ASIC ERS pilot system, exercising every major module built
across all 13 sessions of this workstream in one continuous, narrated
run.

This is NOT a mocked walkthrough — the bill photo step runs REAL
Tesseract OCR against a genuinely generated image (same technique as
test_bill_ocr.py), and every money movement goes through the real
credit_ledger double-entry postings, not a stub. Everything else uses
the same in-memory fake supabase_db pattern as the other test_*.py
files (no live database/network access), so this remains a "genuinely
free, network-free, real-crypto/real-OCR" test in the tradition
established across this workstream.

This corresponds directly to the "end_to_end_bill_payment" synthetic
ASIC-review demonstration scenario already defined in
operational_readiness.SYNTHETIC_ASIC_REVIEW_SCENARIOS.

Journey covered:
1. Identity verification (Didit adapter, explicitly-gated mock mode —
   real network calls still aren't possible from this sandbox).
2. Onboarding application + eligibility (deterministic, no scoring).
3. Responsible-lending affordability assessment.
4. Document acceptance (credit guide).
5. Maker-checker credit activation (onboarding.py AND credit_ledger.py
   both require it).
6. A dummy customer photographs a real bill -> real Tesseract OCR
   extraction -> bill verification against the biller allowlist.
7. The verified bill is paid via the full pilot_payment_flow
   (permitted-use check -> real credit draw -> disbursement).
8. A repayment schedule is generated and a repayment is recorded,
   reducing real ledger outstanding principal.
9. Regulatory reports are generated summarising the dummy customer's
   activity — confirming no raw PII appears in the output.

Run: python3 test_end_to_end_dummy_customer_journey.py
"""
import asyncio
import io
import sys
import types
import uuid
from decimal import Decimal
from datetime import datetime, timezone

from PIL import Image, ImageDraw, ImageFont

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
    return [await insert_one(table, r) for r in rows]


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

import pilot_config as pc                  # noqa: E402
import onboarding as ob                    # noqa: E402
import responsible_lending as rl           # noqa: E402
import identity_verification as idv        # noqa: E402
import document_versioning as dv           # noqa: E402
import credit_ledger as cl                 # noqa: E402
import bill_ocr as ocr                     # noqa: E402
import biller_allowlist as ba              # noqa: E402
import bill_verification as bv             # noqa: E402
import pilot_payment_flow as flow          # noqa: E402
import hardship_collections as hc          # noqa: E402
import regulatory_reports as rr            # noqa: E402

FAILURES = []
STEP = [0]


def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")
    if not condition:
        FAILURES.append(label)


def narrate(title):
    STEP[0] += 1
    print(f"\n=== Step {STEP[0]}: {title} ===")


def make_bill_photo(biller_name, account_name, reference, amount, due_date) -> bytes:
    """Generates a real synthetic bill photo -- same technique as
    test_bill_ocr.py -- for a genuine Tesseract OCR pass, not a
    pre-canned extraction result."""
    img = Image.new("RGB", (900, 500), color="white")
    draw = ImageDraw.Draw(img)
    try:
        font_big = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    except Exception:
        font_big = ImageFont.load_default()
        font_small = font_big
    draw.text((30, 30), biller_name, fill="black", font=font_big)
    draw.text((30, 100), f"Account Name: {account_name}", fill="black", font=font_small)
    draw.text((30, 150), f"Reference Number: {reference}", fill="black", font=font_small)
    draw.text((30, 200), f"Amount Due: ${amount}", fill="black", font=font_small)
    draw.text((30, 250), f"Due Date: {due_date}", fill="black", font=font_small)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


async def main():
    CUSTOMER_NAME = "Jane Dummy"
    CUSTOMER_ID = "user-jane-dummy-001"
    APPROVED_AREAS = {"VIC"}
    APPROVED_CATEGORIES = {"electricity", "gas", "water", "telecommunications"}

    # ---------------------------------------------------------------
    narrate("Pilot configuration is active")
    # ---------------------------------------------------------------
    await pc.propose_config_version(pc.PilotConfig(), proposed_by="ops_lead", approved_by="compliance_lead", activate=True)
    active_config = await pc.get_active_config()
    check("pilot config is active before onboarding anyone", active_config is not None and active_config.is_active)

    # ---------------------------------------------------------------
    narrate("Jane's identity is verified (Didit adapter, gated mock mode)")
    # ---------------------------------------------------------------
    import os
    os.environ["ALLOW_MOCK_IDENTITY_VERIFICATION"] = "true"
    import importlib
    importlib.reload(idv)
    session = await idv.start_verification_session(applicant_reference=CUSTOMER_ID)
    check("identity verification session started", session.provider == "mock" and session.session_id.startswith("mock-session-"))
    identity_result = await idv.get_verification_result(session.session_id)
    check("identity verification result is 'verified'", identity_result.status == "verified")

    # ---------------------------------------------------------------
    narrate("Jane applies -- deterministic eligibility, no opaque scoring")
    # ---------------------------------------------------------------
    consents = {}
    for c in ob.REQUIRED_CONSENTS:
        consents = ob.record_consent(consents, c, "v1")

    application = ob.OnboardingApplication(
        user_id=CUSTOMER_ID,
        identity_verification_status=identity_result.status,
        age_confirmed=True,
        residential_state="VIC",
        residential_country="AU",
        bank_account_verified=True,
        income_amount="5200",
        income_frequency="monthly",
        employment_status="full_time",
        recurring_living_expenses="2800",
        existing_debts_and_bnpl="0",
        requested_credit_purpose="electricity",
        requirements_and_objectives="Smooth out quarterly electricity bills",
        vulnerability_indicators=[],
        bankruptcy_status="none",
        utility_bill_ownership_verified=True,
        consents=consents,
    )
    app_row = await ob.submit_application(application, APPROVED_AREAS, APPROVED_CATEGORIES, policy_version="onboarding-policy-v1")
    check("Jane's application is 'eligible' (deterministic rules, not a black-box score)", app_row["eligibility_outcome"] == "eligible")
    check("no reason codes on a clean eligible application", app_row["reason_codes"] == [])

    # ---------------------------------------------------------------
    narrate("Responsible-lending assessment -- deterministic affordability")
    # ---------------------------------------------------------------
    now = datetime.now(timezone.utc)
    affordability = rl.AffordabilityInputs(
        gross_income_amount=Decimal("5200"), income_frequency="monthly",
        essential_expenditure_monthly=Decimal("2800"),
        discretionary_expenditure_monthly=Decimal("400"),
        existing_credit_repayments_monthly=Decimal("0"),
        bnpl_repayments_monthly=Decimal("0"),
        proposed_billsure_repayment_monthly=Decimal("150"),
        evidence_as_of=now.isoformat(),
    )
    assessment_result = rl.run_assessment(affordability, now=now)
    check("affordability assessment recommends 'approve'", assessment_result.recommendation == "approve")
    assessment_row = await rl.persist_assessment(app_row["id"], affordability, assessed_by="credit_assessor_1")
    check("assessment is persisted", assessment_row["recommendation"] == "approve")

    # ---------------------------------------------------------------
    narrate("Jane accepts the (template) credit guide -- reproducible acceptance")
    # ---------------------------------------------------------------
    credit_guide_v1 = await dv.create_document_version(
        "credit_guide", b"Credit Guide v1 -- pilot terms", "2026-01-01", created_by="legal_team", is_material_change=False)
    approved_guide = await dv.approve_document_version(credit_guide_v1["id"], approved_by="compliance_lead")
    check("credit guide v1 is approved and active", approved_guide["status"] == "approved")
    acceptance = await dv.record_customer_acceptance(CUSTOMER_ID, "credit_guide", approved_guide["id"], ip_address="203.0.113.9")
    check("Jane's acceptance of the credit guide is recorded", acceptance["version_number"] == 1)

    # ---------------------------------------------------------------
    narrate("Credit activation -- maker-checker in BOTH onboarding.py and credit_ledger.py")
    # ---------------------------------------------------------------
    onboarding_activation = await ob.approve_credit_activation(app_row["id"], prepared_by="credit_assessor_1", approved_by="compliance_lead")
    check("onboarding-side credit activation approved (distinct preparer/approver)", onboarding_activation["approved_by"] == "compliance_lead")

    credit_account = await cl.activate_customer_credit_account(
        CUSTOMER_ID, contractual_limit=Decimal("2500.00"), active_customer_count=0,
        current_aggregate_contractual_exposure=Decimal("0"), proposed_by="credit_assessor_1", approved_by="compliance_lead",
    )
    check("Jane's credit account is active with a $2,500 contractual limit", credit_account["status"] == "active")
    outstanding = await cl.get_outstanding_principal(CUSTOMER_ID)
    check("Jane's credit account starts at zero outstanding principal", outstanding == Decimal("0.00"))

    # ---------------------------------------------------------------
    narrate("Jane photographs a real electricity bill -- genuine Tesseract OCR, not mocked")
    # ---------------------------------------------------------------
    bill_photo_bytes = make_bill_photo("Origin Energy", CUSTOMER_NAME, "REF-JANE-99001", "145.00", "20/09/2026")
    known_billers = ba.allowlist_names()
    extraction = ocr.extract_bill_data(bill_photo_bytes, known_billers=known_billers, is_pdf=False)
    check("OCR used the real tesseract_ocr method (not a stub)", extraction.extraction_method == "tesseract_ocr")
    print(f"    OCR extracted: amount={extraction.guessed_amount}, due_date={extraction.guessed_due_date}, "
          f"reference={extraction.guessed_biller_reference}, biller_candidates={extraction.biller_name_candidates}, "
          f"confidence={extraction.extraction_confidence:.2f}")
    check("OCR correctly extracted the amount from the photographed bill", extraction.guessed_amount == Decimal("145.00"))
    check("OCR correctly identified Origin Energy as a biller candidate", "Origin Energy" in extraction.biller_name_candidates)

    # ---------------------------------------------------------------
    narrate("The OCR'd bill is verified against the biller allowlist")
    # ---------------------------------------------------------------
    submission = bv.BillSubmission(
        customer_id=CUSTOMER_ID, file_bytes=bill_photo_bytes,
        customer_name_on_account=CUSTOMER_NAME, customer_name_on_bill=CUSTOMER_NAME,
        biller_name_extracted="Origin Energy", biller_reference=extraction.guessed_biller_reference or "REF-JANE-99001",
        category="electricity", amount=extraction.guessed_amount or Decimal("145.00"),
        due_date=extraction.guessed_due_date or "20/09/2026",
        extraction_confidence=extraction.extraction_confidence, fraud_indicators=[],
    )
    bill_row = await bv.submit_and_verify_bill(submission, known_billers, APPROVED_CATEGORIES,
                                                min_confidence=bv.DEFAULT_MIN_EXTRACTION_CONFIDENCE)
    print(f"    Bill verification status: {bill_row['verification_status']} (reasons: {bill_row['verification_reasons']})")

    if bill_row["verification_status"] == "manual_review":
        # A real OCR pass on a synthetic image may legitimately land
        # just under the confidence threshold -- exercise the manual
        # review path for real rather than treating this as a failure.
        bill_row = await bv.record_manual_review_decision(bill_row["id"], reviewer="case_worker_1", decision="verified",
                                                            notes="confirmed against the original photo, all fields correct")
        check("bill moved to 'verified' after manual review", bill_row["verification_status"] == "verified")
    else:
        check("bill was verified automatically (high-confidence OCR)", bill_row["verification_status"] == "verified")

    # ---------------------------------------------------------------
    narrate("The verified bill is paid -- full flow, real ledger movement")
    # ---------------------------------------------------------------
    disbursement = await flow.pay_verified_bill(bill_row["id"], CUSTOMER_ID, requested_by="payments_admin_1")
    check("disbursement was created", disbursement["status"] == "queued")
    check("disbursement amount matches the bill", disbursement["amount"] == "145.00")
    check("disbursement is linked to a real credit ledger journal", disbursement.get("credit_journal_id") is not None)

    outstanding_after_payment = await cl.get_outstanding_principal(CUSTOMER_ID)
    check("Jane's outstanding principal genuinely increased by the bill amount (real money moved)",
          outstanding_after_payment == Decimal("145.00"))

    # ---------------------------------------------------------------
    narrate("Repayment schedule + a real repayment reducing the ledger balance")
    # ---------------------------------------------------------------
    schedule = await hc.generate_repayment_schedule(CUSTOMER_ID, Decimal("145.00"), term_months=1, first_due_date="2026-10-01",
                                                       created_by="ops_lead")
    check("a 1-installment schedule was generated for the $145 balance", len(schedule["installments"]) == 1)
    installment = schedule["installments"][0]

    repayment = await hc.record_repayment(installment["id"], CUSTOMER_ID, Decimal("145.00"), recorded_by="ops_lead")
    check("the installment is fully paid", repayment["status"] == "paid")

    outstanding_after_repayment = await cl.get_outstanding_principal(CUSTOMER_ID)
    check("Jane's outstanding principal is back to zero after the real repayment", outstanding_after_repayment == Decimal("0.00"))

    # ---------------------------------------------------------------
    narrate("Regulatory reports summarise Jane's activity -- no raw PII")
    # ---------------------------------------------------------------
    exposure_snapshot = await cl.get_exposure_snapshot()
    exposure_report = rr.credit_exposure_report(exposure_snapshot)
    check("credit exposure report shows exactly 1 active customer (Jane)", exposure_report["active_customer_count"] == 1)

    all_disbursements = await find_many("pilot_bill_disbursements", {})
    payment_report = rr.payment_activity_report(all_disbursements)
    check("payment activity report shows Jane's $145 disbursement", payment_report["total_amount_disbursed"] == "145.00")

    demo_report = rr.customer_numbers_and_demographics_report([credit_account], [app_row])
    check("demographics report shows 1 active credit customer", demo_report["active_credit_customers"] == 1)
    check("demographics report never contains Jane's actual name anywhere", CUSTOMER_NAME not in str(demo_report))

    print(f"\n  Full journey summary for {CUSTOMER_ID}:")
    print(f"    - Identity verified: {identity_result.status}")
    print(f"    - Eligibility: {app_row['eligibility_outcome']}")
    print(f"    - Affordability: {assessment_result.recommendation} (surplus ${assessment_result.surplus_monthly}/mo)")
    print(f"    - Credit account: active, $2,500 limit")
    print(f"    - Bill paid: $145.00 to Origin Energy (OCR confidence {extraction.extraction_confidence:.2f})")
    print(f"    - Final outstanding balance: ${outstanding_after_repayment}")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED:")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("ALL CHECKS PASSED -- full dummy customer journey completed end to end")


if __name__ == "__main__":
    asyncio.run(main())
