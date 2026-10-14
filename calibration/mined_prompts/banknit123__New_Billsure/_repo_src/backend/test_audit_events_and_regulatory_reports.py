"""
Standalone tests for audit_events.py and regulatory_reports.py. Same
in-memory fake-DB pattern as the other test_*.py files, no live
credentials needed.

Run: python3 test_audit_events_and_regulatory_reports.py
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


fake_sdb = types.SimpleNamespace(find_one=find_one, find_many=find_many, insert_one=insert_one)
sys.modules["supabase_db"] = fake_sdb

import audit_events as ae          # noqa: E402
import regulatory_reports as rr    # noqa: E402

FAILURES = []


def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}")
    if not condition:
        FAILURES.append(label)


async def main():
    # ---------------------------------------------------------------
    # audit_events.record_event: required fields, category validation
    # ---------------------------------------------------------------
    try:
        await ae.record_event("not_a_real_category", "test", "admin1", "compliance", "customer", "cust-1", "test_source")
        check("rejects an unknown category", False)
    except ae.AuditEventError:
        check("rejects an unknown category", True)

    try:
        await ae.record_event("login", "test", "", "compliance", "customer", "cust-1", "test_source")
        check("rejects a missing actor", False)
    except ae.AuditEventError:
        check("rejects a missing actor", True)

    event = await ae.record_event(
        "administrative_access", "viewed_customer_record", actor="admin1", role="compliance",
        object_type="customer", object_id="cust-1", source="admin_portal",
        previous_state=None, new_state={"viewed_fields": ["balance"]},
        reason="routine compliance review", correlation_id="corr-abc",
    )
    check("a fully-specified event is recorded with every required field", event["actor"] == "admin1" and event["category"] == "administrative_access")

    # ---------------------------------------------------------------
    # Retrieval by object and by correlation
    # ---------------------------------------------------------------
    await ae.record_event("data_export", "exported_report", actor="admin2", role="compliance",
                           object_type="customer", object_id="cust-1", source="reporting_module",
                           correlation_id="corr-abc")
    by_object = await ae.get_events_for_object("customer", "cust-1")
    check("get_events_for_object returns every event for that object", len(by_object) == 2)

    by_correlation = await ae.get_events_by_correlation("corr-abc")
    check("get_events_by_correlation ties related events together across categories", len(by_correlation) == 2)

    # ---------------------------------------------------------------
    # redact_for_export: sensitive fields stripped, non-sensitive kept
    # ---------------------------------------------------------------
    sensitive_event = {
        "actor": "admin1", "category": "onboarding",
        "previous_state": {"full_name": "Jane Citizen", "employment_status": "full_time"},
        "new_state": {"email": "jane@example.com", "eligibility_outcome": "eligible", "nested": {"date_of_birth": "1990-01-01", "state": "VIC"}},
    }
    redacted = ae.redact_for_export(sensitive_event)
    check("redact_for_export strips full_name from previous_state", redacted["previous_state"]["full_name"] == "[REDACTED]")
    check("redact_for_export keeps non-sensitive fields in previous_state", redacted["previous_state"]["employment_status"] == "full_time")
    check("redact_for_export strips email from new_state", redacted["new_state"]["email"] == "[REDACTED]")
    check("redact_for_export keeps non-sensitive top-level fields in new_state", redacted["new_state"]["eligibility_outcome"] == "eligible")
    check("redact_for_export strips sensitive fields nested inside new_state", redacted["new_state"]["nested"]["date_of_birth"] == "[REDACTED]")
    check("redact_for_export keeps non-sensitive nested fields", redacted["new_state"]["nested"]["state"] == "VIC")
    check("redact_for_export does not mutate the original event", sensitive_event["previous_state"]["full_name"] == "Jane Citizen")

    # ---------------------------------------------------------------
    # regulatory_reports: customer numbers and demographics
    # ---------------------------------------------------------------
    accounts = [{"status": "active"}, {"status": "active"}, {"status": "closed"}]
    applications = [
        {"residential_state": "VIC", "employment_status": "full_time", "eligibility_outcome": "eligible"},
        {"residential_state": "VIC", "employment_status": "part_time", "eligibility_outcome": "referred"},
        {"residential_state": "NSW", "employment_status": "full_time", "eligibility_outcome": "declined"},
    ]
    demo_report = rr.customer_numbers_and_demographics_report(accounts, applications)
    check("demographics report counts only active credit customers", demo_report["active_credit_customers"] == 2)
    check("demographics report counts all applications regardless of status", demo_report["total_applications"] == 3)
    check("demographics report aggregates by state without any name/address field appearing", demo_report["by_residential_state"] == {"VIC": 2, "NSW": 1})
    check("no raw PII key appears anywhere in the demographics report", "full_name" not in str(demo_report) and "date_of_birth" not in str(demo_report))

    # ---------------------------------------------------------------
    # regulatory_reports: credit exposure
    # ---------------------------------------------------------------
    class FakeSnapshot:
        active_customer_count = 10
        max_pilot_customers = 25
        aggregate_contractual_exposure = Decimal("25000.00")
        aggregate_contractual_cap = Decimal("62500.00")
        aggregate_drawn_exposure = Decimal("8000.00")

    exposure_report = rr.credit_exposure_report(FakeSnapshot())
    check("credit exposure report computes utilisation percentage correctly", exposure_report["utilisation_pct"] == "40.0")

    # ---------------------------------------------------------------
    # regulatory_reports: payment activity
    # ---------------------------------------------------------------
    disbursements = [
        {"amount": "300.00", "status": "cleared"},
        {"amount": "150.00", "status": "cleared"},
        {"amount": "50.00", "status": "failed"},
    ]
    payment_report = rr.payment_activity_report(disbursements)
    check("payment activity report totals amounts correctly", payment_report["total_amount_disbursed"] == "500.00")
    check("payment activity report computes the average correctly", payment_report["average_disbursement"] == "166.67")
    check("payment activity report breaks down by status", payment_report["by_status"]["failed"] == 1)

    empty_payment_report = rr.payment_activity_report([])
    check("payment activity report handles zero disbursements without dividing by zero", empty_payment_report["average_disbursement"] == "0.00")

    # ---------------------------------------------------------------
    # regulatory_reports: arrears and hardship
    # ---------------------------------------------------------------
    installments = [{"status": "paid"}, {"status": "failed"}, {"status": "partial"}, {"status": "failed"}]
    hardship_cases = [{"status": "open"}, {"status": "arrangement_active"}, {"status": "open"}]
    arrears_report = rr.arrears_and_hardship_report(installments, hardship_cases)
    check("arrears report counts failed installments correctly", arrears_report["installments_failed"] == 2)
    check("arrears report counts hardship cases by status", arrears_report["hardship_cases_by_status"]["open"] == 2)

    # ---------------------------------------------------------------
    # regulatory_reports: complaints and AFCA (uses real complaints.py logic)
    # ---------------------------------------------------------------
    complaints_data = [
        {"status": "closed", "stage": "resolved", "resolved_late": False, "root_cause_category": "system_error"},
        {"status": "open", "stage": "escalated_to_afca", "resolved_late": None, "root_cause_category": None},
        {"status": "closed", "stage": "resolved", "resolved_late": True, "root_cause_category": "process_failure"},
    ]
    complaints_report = rr.complaints_and_afca_report(complaints_data)
    check("complaints report counts escalations correctly", complaints_report["escalated_to_afca_count"] == 1)
    check("complaints report counts late resolutions correctly", complaints_report["resolved_late_count"] == 1)
    check("complaints report's root cause breakdown only considers resolved complaints", complaints_report["root_cause_breakdown"]["system_error"] == 1)

    # ---------------------------------------------------------------
    # regulatory_reports: consumer losses and remediation
    # ---------------------------------------------------------------
    remedies = [
        {"status": "approved", "compensation_amount": "50.00"},
        {"status": "approved", "compensation_amount": "25.50"},
        {"status": "proposed", "compensation_amount": "100.00"},  # not yet approved -- must not count
    ]
    losses_report = rr.consumer_losses_and_remediation_report(remedies)
    check("consumer losses report only totals APPROVED compensation, not merely proposed", losses_report["total_compensation_approved"] == "75.50")
    check("consumer losses report counts total proposals separately from approved", losses_report["remedies_proposed"] == 3 and losses_report["remedies_approved"] == 2)

    # ---------------------------------------------------------------
    # regulatory_reports: reconciliation exceptions
    # ---------------------------------------------------------------
    exceptions = [{"status": "open"}, {"status": "resolved"}, {"status": "open"}, {"status": "investigating"}]
    recon_report = rr.reconciliation_exceptions_report(exceptions)
    check("reconciliation report counts currently-open exceptions correctly (excludes resolved/matched)", recon_report["currently_open"] == 3)

    # ---------------------------------------------------------------
    # regulatory_reports: security incidents (category-filtered input)
    # ---------------------------------------------------------------
    security_events = [{"action": "failed_login"}, {"action": "failed_login"}, {"action": "admin_password_reset"}]
    security_report = rr.security_incidents_report(security_events)
    check("security incidents report tallies by action", security_report["by_action"]["failed_login"] == 2)

    # ---------------------------------------------------------------
    # regulatory_reports: ERS end-of-test report assembles every piece
    # ---------------------------------------------------------------
    pilot_outcomes = rr.pilot_outcomes_and_public_benefit_report(
        total_bills_paid=42, total_amount_disbursed=Decimal("12500.00"),
        hardship_cases_resolved=3, complaints_resolved=2, average_days_to_resolve_complaint=12.5,
    )
    end_of_test = rr.ers_end_of_test_report(
        demo_report, exposure_report, payment_report, arrears_report, complaints_report,
        losses_report, recon_report, security_report, pilot_outcomes,
        period_start="2026-07-01", period_end="2026-12-31",
    )
    check("ERS end-of-test report assembles every sub-report", end_of_test["credit_exposure"]["utilisation_pct"] == "40.0")
    check("ERS end-of-test report carries the reporting period", end_of_test["metadata"]["period_start"] == "2026-07-01")

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
