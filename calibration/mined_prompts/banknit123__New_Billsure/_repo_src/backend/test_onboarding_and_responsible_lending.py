"""
Standalone logic test for onboarding.py and responsible_lending.py — same
in-memory fake-DB pattern as test_ledger_flow.py and
test_pilot_config_and_launch_gates.py, no live credentials needed.

Run: python3 test_onboarding_and_responsible_lending.py
"""
import asyncio
import sys
import types
import uuid
from decimal import Decimal
from datetime import datetime, timezone, timedelta

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

import onboarding as ob            # noqa: E402
import responsible_lending as rl   # noqa: E402

FAILURES = []


def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}")
    if not condition:
        FAILURES.append(label)


def _complete_consents():
    consents = {}
    for c in ob.REQUIRED_CONSENTS:
        consents = ob.record_consent(consents, c, "v1")
    return consents


def _base_eligible_app(**overrides):
    defaults = dict(
        user_id="user-1",
        identity_verification_status="verified",
        age_confirmed=True,
        residential_state="VIC",
        residential_country="AU",
        bank_account_verified=True,
        income_amount="1000",
        income_frequency="monthly",
        employment_status="full_time",
        recurring_living_expenses="500",
        existing_debts_and_bnpl="0",
        requested_credit_purpose="electricity",
        requirements_and_objectives="Smooth out quarterly electricity bills",
        vulnerability_indicators=[],
        bankruptcy_status="none",
        utility_bill_ownership_verified=True,
        consents=_complete_consents(),
    )
    defaults.update(overrides)
    return ob.OnboardingApplication(**defaults)


async def main():
    approved_areas = {"VIC"}
    approved_categories = {"electricity", "gas", "water", "telecommunications"}

    # ---------------------------------------------------------------
    # onboarding: deterministic eligibility, no opaque scoring
    # ---------------------------------------------------------------
    eligible_result = ob.evaluate_eligibility(_base_eligible_app(), approved_areas, approved_categories, "policy-v1")
    check("complete, compliant application is 'eligible'", eligible_result.outcome == "eligible")
    check("eligible outcome carries no reason codes", eligible_result.reason_codes == [])

    under_18 = ob.evaluate_eligibility(_base_eligible_app(age_confirmed=False), approved_areas, approved_categories, "policy-v1")
    check("unconfirmed age is declined with AGE_UNCONFIRMED", under_18.outcome == "declined" and "AGE_UNCONFIRMED" in under_18.reason_codes)

    not_identity = ob.evaluate_eligibility(_base_eligible_app(identity_verification_status="failed"), approved_areas, approved_categories, "policy-v1")
    check("failed identity check is declined", not_identity.outcome == "declined" and "IDENTITY_NOT_VERIFIED" in not_identity.reason_codes)

    outside_vic = ob.evaluate_eligibility(_base_eligible_app(residential_state="NSW"), approved_areas, approved_categories, "policy-v1")
    check("non-VIC applicant is declined with OUTSIDE_PILOT_AREA", outside_vic.outcome == "declined" and "OUTSIDE_PILOT_AREA" in outside_vic.reason_codes)

    bad_purpose = ob.evaluate_eligibility(_base_eligible_app(requested_credit_purpose="rent"), approved_areas, approved_categories, "policy-v1")
    check("prohibited purpose (rent) is declined", bad_purpose.outcome == "declined" and "PURPOSE_NOT_APPROVED" in bad_purpose.reason_codes)

    bankrupt = ob.evaluate_eligibility(_base_eligible_app(bankruptcy_status="undischarged"), approved_areas, approved_categories, "policy-v1")
    check("undischarged bankruptcy is declined", bankrupt.outcome == "declined" and "BANKRUPTCY_UNDISCHARGED" in bankrupt.reason_codes)

    # Vulnerability indicators are REFERRED, never auto-declined or auto-approved.
    vulnerable = ob.evaluate_eligibility(_base_eligible_app(vulnerability_indicators=["financial_hardship_disclosed"]),
                                          approved_areas, approved_categories, "policy-v1")
    check("vulnerability indicator produces 'referred', not 'declined' or 'eligible'", vulnerable.outcome == "referred")
    check("referred-for-vulnerability carries VULNERABILITY_INDICATOR code", "VULNERABILITY_INDICATOR" in vulnerable.reason_codes)

    incomplete = ob.evaluate_eligibility(_base_eligible_app(income_amount=None), approved_areas, approved_categories, "policy-v1")
    check("incomplete evidence produces 'referred', not an auto-decline", incomplete.outcome == "referred" and "INCOMPLETE_EVIDENCE" in incomplete.reason_codes)

    missing_consent = ob.evaluate_eligibility(_base_eligible_app(consents={}), approved_areas, approved_categories, "policy-v1")
    check("missing consent is referred, not silently proceeded with", missing_consent.outcome == "referred" and "MISSING_CONSENT" in missing_consent.reason_codes)

    # ---------------------------------------------------------------
    # onboarding: manual review requires documented reasons
    # ---------------------------------------------------------------
    app_row = await ob.submit_application(_base_eligible_app(vulnerability_indicators=["x"]), approved_areas, approved_categories, "policy-v1")
    try:
        await ob.record_manual_review_outcome(app_row["id"], "reviewer1", "declined", [], "no reason given")
        check("rejects a non-approved manual outcome with zero reason codes", False)
    except ob.OnboardingError:
        check("rejects a non-approved manual outcome with zero reason codes", True)

    try:
        await ob.record_manual_review_outcome(app_row["id"], "reviewer1", "declined", ["NOT_A_REAL_CODE"], "bad code")
        check("rejects an unknown reason code", False)
    except ob.OnboardingError:
        check("rejects an unknown reason code", True)

    reviewed = await ob.record_manual_review_outcome(app_row["id"], "reviewer1", "approved", [], "vulnerability addressed, affordability confirmed separately")
    check("manual approval recorded", reviewed["final_outcome"] == "approved")

    # ---------------------------------------------------------------
    # onboarding: maker-checker before credit activation
    # ---------------------------------------------------------------
    try:
        await ob.approve_credit_activation(app_row["id"], prepared_by="reviewer1", approved_by="reviewer1")
        check("rejects credit activation prepared and approved by the same person", False)
    except ob.OnboardingError:
        check("rejects credit activation prepared and approved by the same person", True)

    try:
        await ob.approve_credit_activation(app_row["id"], prepared_by="reviewer1", approved_by="reviewer2")
        check("rejects activation prepared by the same person who did the manual review", False)
    except ob.OnboardingError:
        check("rejects activation prepared by the same person who did the manual review", True)

    activated = await ob.approve_credit_activation(app_row["id"], prepared_by="preparer3", approved_by="reviewer2")
    check("credit activation succeeds with three distinct people across the workflow", activated["approved_by"] == "reviewer2")

    clean_app = await ob.submit_application(_base_eligible_app(user_id="user-2"), approved_areas, approved_categories, "policy-v1")
    try:
        await ob.approve_credit_activation(clean_app["id"], prepared_by="a", approved_by="a")
        check("rejects activation of an eligible (never-reviewed) app when preparer==approver", False)
    except ob.OnboardingError:
        check("rejects activation of an eligible (never-reviewed) app when preparer==approver", True)
    clean_activation = await ob.approve_credit_activation(clean_app["id"], prepared_by="a", approved_by="b")
    check("activation of a directly-eligible (no manual review needed) app succeeds with distinct approver", clean_activation["approved_by"] == "b")

    declined_app = await ob.submit_application(_base_eligible_app(user_id="user-3", age_confirmed=False), approved_areas, approved_categories, "policy-v1")
    try:
        await ob.approve_credit_activation(declined_app["id"], prepared_by="a", approved_by="b")
        check("rejects activation of a declined application", False)
    except ob.OnboardingError:
        check("rejects activation of a declined application", True)

    # ---------------------------------------------------------------
    # responsible_lending: deterministic affordability
    # ---------------------------------------------------------------
    now = datetime.now(timezone.utc)
    good_inputs = rl.AffordabilityInputs(
        gross_income_amount=Decimal("6000"), income_frequency="monthly",
        essential_expenditure_monthly=Decimal("3000"),
        discretionary_expenditure_monthly=Decimal("500"),
        existing_credit_repayments_monthly=Decimal("400"),
        bnpl_repayments_monthly=Decimal("100"),
        proposed_billsure_repayment_monthly=Decimal("200"),
        evidence_as_of=now.isoformat(),
    )
    good_result = rl.run_assessment(good_inputs, now=now)
    check("affordable scenario recommends 'approve'", good_result.recommendation == "approve" and good_result.affordability_pass)
    check("surplus is calculated correctly (6000-3000-400-100-200=2300)", good_result.surplus_monthly == Decimal("2300.00"))

    tight_inputs = rl.AffordabilityInputs(
        gross_income_amount=Decimal("1000"), income_frequency="monthly",
        essential_expenditure_monthly=Decimal("900"),
        discretionary_expenditure_monthly=Decimal("0"),
        existing_credit_repayments_monthly=Decimal("50"),
        bnpl_repayments_monthly=Decimal("50"),
        proposed_billsure_repayment_monthly=Decimal("100"),
        evidence_as_of=now.isoformat(),
    )
    tight_result = rl.run_assessment(tight_inputs, now=now)
    check("unaffordable scenario recommends 'decline', not 'approve'", tight_result.recommendation == "decline" and not tight_result.affordability_pass)

    vulnerable_inputs = rl.AffordabilityInputs(
        gross_income_amount=Decimal("6000"), income_frequency="monthly",
        essential_expenditure_monthly=Decimal("1000"),
        discretionary_expenditure_monthly=Decimal("0"),
        existing_credit_repayments_monthly=Decimal("0"),
        bnpl_repayments_monthly=Decimal("0"),
        proposed_billsure_repayment_monthly=Decimal("100"),
        evidence_as_of=now.isoformat(),
        hardship_flag=True,
    )
    vulnerable_result = rl.run_assessment(vulnerable_inputs, now=now)
    check("hardship flag forces 'refer' even though arithmetic would pass", vulnerable_result.recommendation == "refer" and vulnerable_result.affordability_pass)

    stale_inputs = rl.AffordabilityInputs(
        gross_income_amount=Decimal("6000"), income_frequency="monthly",
        essential_expenditure_monthly=Decimal("1000"),
        discretionary_expenditure_monthly=Decimal("0"),
        existing_credit_repayments_monthly=Decimal("0"),
        bnpl_repayments_monthly=Decimal("0"),
        proposed_billsure_repayment_monthly=Decimal("100"),
        evidence_as_of=(now - timedelta(days=120)).isoformat(),
    )
    stale_result = rl.run_assessment(stale_inputs, now=now)
    check("stale evidence (>90 days) is referred, not approved on old numbers", stale_result.recommendation == "refer" and any("STALE" in i for i in stale_result.evidence_issues))

    incomplete_inputs = rl.AffordabilityInputs(
        gross_income_amount=None, income_frequency="monthly",
        essential_expenditure_monthly=Decimal("1000"),
        discretionary_expenditure_monthly=Decimal("0"),
        existing_credit_repayments_monthly=Decimal("0"),
        bnpl_repayments_monthly=Decimal("0"),
        proposed_billsure_repayment_monthly=Decimal("100"),
        evidence_as_of=now.isoformat(),
    )
    incomplete_result = rl.run_assessment(incomplete_inputs, now=now)
    check("missing income figure is referred, never treated as zero/approved", incomplete_result.recommendation == "refer")

    # ---------------------------------------------------------------
    # responsible_lending: override requires reason + independent approval
    # ---------------------------------------------------------------
    persisted = await rl.persist_assessment("app-1", tight_inputs, assessed_by="assessor1")
    try:
        await rl.override_recommendation(persisted["id"], "approve", "", overridden_by="reviewer1", approved_by="reviewer2")
        check("rejects an override with an empty reason", False)
    except rl.ResponsibleLendingError:
        check("rejects an override with an empty reason", True)

    try:
        await rl.override_recommendation(persisted["id"], "approve", "compensating factors documented", overridden_by="reviewer1", approved_by="reviewer1")
        check("rejects a self-approved override", False)
    except rl.ResponsibleLendingError:
        check("rejects a self-approved override", True)

    override = await rl.override_recommendation(persisted["id"], "approve", "compensating factors documented",
                                                  overridden_by="reviewer1", approved_by="reviewer2")
    check("override with reason + independent approver succeeds", override["override_to"] == "approve")

    # ---------------------------------------------------------------
    # responsible_lending: limit increases always require a fresh,
    # passing assessment -- never automatic
    # ---------------------------------------------------------------
    try:
        await rl.can_increase_limit("cust-1", Decimal("300"), Decimal("500"), latest_assessment=None)
        check("rejects a limit increase with no assessment on file at all", False)
    except rl.ResponsibleLendingError:
        check("rejects a limit increase with no assessment on file at all", True)

    declining_assessment = {"recommendation": "decline", "superseded": False, "assessed_at": now.isoformat()}
    try:
        await rl.can_increase_limit("cust-1", Decimal("300"), Decimal("500"), latest_assessment=declining_assessment)
        check("rejects a limit increase backed by a declining assessment", False)
    except rl.ResponsibleLendingError:
        check("rejects a limit increase backed by a declining assessment", True)

    superseded_assessment = {"recommendation": "approve", "superseded": True, "assessed_at": now.isoformat()}
    try:
        await rl.can_increase_limit("cust-1", Decimal("300"), Decimal("500"), latest_assessment=superseded_assessment)
        check("rejects a limit increase backed by a superseded assessment", False)
    except rl.ResponsibleLendingError:
        check("rejects a limit increase backed by a superseded assessment", True)

    stale_assessment = {"recommendation": "approve", "superseded": False, "assessed_at": (now - timedelta(days=120)).isoformat()}
    try:
        await rl.can_increase_limit("cust-1", Decimal("300"), Decimal("500"), latest_assessment=stale_assessment)
        check("rejects a limit increase backed by a stale (>90d) assessment", False)
    except rl.ResponsibleLendingError:
        check("rejects a limit increase backed by a stale (>90d) assessment", True)

    fresh_passing_assessment = {"recommendation": "approve", "superseded": False, "assessed_at": now.isoformat()}
    try:
        await rl.can_increase_limit("cust-1", Decimal("300"), Decimal("500"), latest_assessment=fresh_passing_assessment)
        check("allows a limit increase backed by a fresh, passing, non-superseded assessment", True)
    except rl.ResponsibleLendingError:
        check("allows a limit increase backed by a fresh, passing, non-superseded assessment", False)

    try:
        await rl.can_increase_limit("cust-1", Decimal("500"), Decimal("300"), latest_assessment=fresh_passing_assessment)
        check("rejects a 'decrease' framed as an increase check", False)
    except rl.ResponsibleLendingError:
        check("rejects a 'decrease' framed as an increase check", True)

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
