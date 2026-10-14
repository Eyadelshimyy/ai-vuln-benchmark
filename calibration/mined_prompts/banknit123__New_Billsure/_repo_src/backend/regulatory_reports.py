"""
backend/regulatory_reports.py
================================
Exportable reports for the ASIC ERS pilot (task section 12's ten
required report types). Every function here is a PURE aggregation over
already-fetched data — no database access of its own, same pattern as
complaints.root_cause_report() — so each is trivially testable and
composable, and so a caller controls exactly what raw data it fetches
and passes in (which is itself part of avoiding unnecessary personal
information in an export: these functions were designed against inputs
that never carried a customer's name, DOB, or government ID in the
first place, because none of pilot_config/credit_ledger/onboarding/
bill_verification/hardship_collections/complaints ever stored those
fields on the objects these reports read).

Every report's output uses customer_id (an opaque identifier) where a
per-customer figure is needed, never a name or other directly-
identifying field. This is a design property of the input shapes these
functions expect, not a redaction step bolted on afterward — see
audit_events.redact_for_export() for the separate belt-and-braces
redaction applied specifically to raw audit event exports, which DO
carry arbitrary previous_state/new_state payloads that could contain
more than these purpose-built reports do.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

import complaints as cp


@dataclass
class ReportMetadata:
    report_type: str
    generated_at: str
    period_start: Optional[str] = None
    period_end: Optional[str] = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------
# 1. Customer numbers and demographics
# ---------------------------------------------------------------

def customer_numbers_and_demographics_report(credit_accounts: list, onboarding_applications: list) -> dict:
    """No name, DOB, or address ever appears in the input rows this
    function expects (onboarding_applications only carries
    residential_state, not a street address — see onboarding.py) —
    aggregated by state and employment_status only, both of which are
    coarse enough not to be identifying at pilot scale on their own."""
    active_accounts = [a for a in credit_accounts if a.get("status") == "active"]
    by_state = Counter(a.get("residential_state") for a in onboarding_applications)
    by_employment_status = Counter(a.get("employment_status") for a in onboarding_applications)
    by_eligibility_outcome = Counter(a.get("eligibility_outcome") for a in onboarding_applications)

    return {
        "metadata": ReportMetadata("customer_numbers_and_demographics", _now()).__dict__,
        "total_applications": len(onboarding_applications),
        "active_credit_customers": len(active_accounts),
        "by_residential_state": dict(by_state),
        "by_employment_status": dict(by_employment_status),
        "by_eligibility_outcome": dict(by_eligibility_outcome),
    }


# ---------------------------------------------------------------
# 2. Credit exposure
# ---------------------------------------------------------------

def credit_exposure_report(exposure_snapshot) -> dict:
    """Takes a credit_ledger.ExposureSnapshot (or any object/dict with
    the same shape) — already aggregate, never per-customer names."""
    return {
        "metadata": ReportMetadata("credit_exposure", _now()).__dict__,
        "active_customer_count": exposure_snapshot.active_customer_count,
        "max_pilot_customers": exposure_snapshot.max_pilot_customers,
        "aggregate_contractual_exposure": str(exposure_snapshot.aggregate_contractual_exposure),
        "aggregate_contractual_cap": str(exposure_snapshot.aggregate_contractual_cap),
        "aggregate_drawn_exposure": str(exposure_snapshot.aggregate_drawn_exposure),
        "utilisation_pct": str(
            (exposure_snapshot.aggregate_contractual_exposure / exposure_snapshot.aggregate_contractual_cap * 100)
            .quantize(Decimal("0.1"))
        ) if exposure_snapshot.aggregate_contractual_cap else "0.0",
    }


# ---------------------------------------------------------------
# 3. Payment activity
# ---------------------------------------------------------------

def payment_activity_report(disbursements: list) -> dict:
    total_amount = sum(Decimal(str(d["amount"])) for d in disbursements)
    by_status = Counter(d.get("status") for d in disbursements)
    return {
        "metadata": ReportMetadata("payment_activity", _now()).__dict__,
        "total_disbursements": len(disbursements),
        "total_amount_disbursed": str(total_amount),
        "average_disbursement": str((total_amount / len(disbursements)).quantize(Decimal("0.01"))) if disbursements else "0.00",
        "by_status": dict(by_status),
    }


# ---------------------------------------------------------------
# 4. Arrears and hardship
# ---------------------------------------------------------------

def arrears_and_hardship_report(installments: list, hardship_cases: list) -> dict:
    failed = [i for i in installments if i.get("status") == "failed"]
    partial = [i for i in installments if i.get("status") == "partial"]
    by_hardship_status = Counter(h.get("status") for h in hardship_cases)
    return {
        "metadata": ReportMetadata("arrears_and_hardship", _now()).__dict__,
        "installments_failed": len(failed),
        "installments_partial": len(partial),
        "hardship_cases_total": len(hardship_cases),
        "hardship_cases_by_status": dict(by_hardship_status),
    }


# ---------------------------------------------------------------
# 5. Complaints and AFCA escalation
# ---------------------------------------------------------------

def complaints_and_afca_report(complaints: list) -> dict:
    by_status = Counter(c.get("status") for c in complaints)
    by_stage = Counter(c.get("stage") for c in complaints)
    escalated = [c for c in complaints if c.get("stage") == "escalated_to_afca"]
    late_resolutions = [c for c in complaints if c.get("resolved_late")]
    resolved_complaints = [c for c in complaints if c.get("status") == "closed"]
    return {
        "metadata": ReportMetadata("complaints_and_afca", _now()).__dict__,
        "total_complaints": len(complaints),
        "by_status": dict(by_status),
        "by_stage": dict(by_stage),
        "escalated_to_afca_count": len(escalated),
        "resolved_late_count": len(late_resolutions),
        "root_cause_breakdown": cp.root_cause_report(resolved_complaints),
    }


# ---------------------------------------------------------------
# 6. Consumer losses and remediation
# ---------------------------------------------------------------

def consumer_losses_and_remediation_report(complaint_remedies: list) -> dict:
    approved = [r for r in complaint_remedies if r.get("status") == "approved"]
    total_compensation = sum(Decimal(str(r["compensation_amount"])) for r in approved)
    return {
        "metadata": ReportMetadata("consumer_losses_and_remediation", _now()).__dict__,
        "remedies_proposed": len(complaint_remedies),
        "remedies_approved": len(approved),
        "total_compensation_approved": str(total_compensation),
    }


# ---------------------------------------------------------------
# 7. Reconciliation exceptions
# ---------------------------------------------------------------

def reconciliation_exceptions_report(exceptions: list) -> dict:
    by_status = Counter(e.get("status") for e in exceptions)
    open_exceptions = [e for e in exceptions if e.get("status") not in ("resolved", "matched")]
    return {
        "metadata": ReportMetadata("reconciliation_exceptions", _now()).__dict__,
        "total_exceptions": len(exceptions),
        "by_status": dict(by_status),
        "currently_open": len(open_exceptions),
    }


# ---------------------------------------------------------------
# 8. Security incidents
# ---------------------------------------------------------------

def security_incidents_report(security_events: list) -> dict:
    """Expects a list of audit_events rows already filtered to
    category='security' — this function doesn't filter itself, so a
    caller must deliberately select the security category, making it
    harder to accidentally include unrelated audit categories in a
    'security incidents' report."""
    by_action = Counter(e.get("action") for e in security_events)
    return {
        "metadata": ReportMetadata("security_incidents", _now()).__dict__,
        "total_security_events": len(security_events),
        "by_action": dict(by_action),
    }


# ---------------------------------------------------------------
# 9. Pilot outcomes and public-benefit measures
# ---------------------------------------------------------------

def pilot_outcomes_and_public_benefit_report(
    total_bills_paid: int, total_amount_disbursed: Decimal,
    hardship_cases_resolved: int, complaints_resolved: int,
    average_days_to_resolve_complaint: Optional[float] = None,
) -> dict:
    """A narrower, qualitative-leaning summary of pilot outcomes for
    reporting the pilot's public-benefit case — distinct from the raw
    operational reports above, this is closer to what would go in an
    ERS progress update to ASIC. Deliberately takes pre-computed
    figures rather than raw rows, since 'public benefit' framing
    figures are usually a summary layer on top of the operational
    reports, not a separate raw dataset."""
    return {
        "metadata": ReportMetadata("pilot_outcomes_and_public_benefit", _now()).__dict__,
        "total_bills_paid": total_bills_paid,
        "total_amount_disbursed": str(total_amount_disbursed),
        "hardship_cases_resolved": hardship_cases_resolved,
        "complaints_resolved": complaints_resolved,
        "average_days_to_resolve_complaint": average_days_to_resolve_complaint,
    }


# ---------------------------------------------------------------
# 10. ERS end-of-test reporting
# ---------------------------------------------------------------

def ers_end_of_test_report(
    customer_demographics: dict, credit_exposure: dict, payment_activity: dict,
    arrears_and_hardship: dict, complaints_and_afca: dict, consumer_losses: dict,
    reconciliation_exceptions: dict, security_incidents: dict, pilot_outcomes: dict,
    period_start: str, period_end: str,
) -> dict:
    """Combines every other report into the single comprehensive
    end-of-test submission ASIC's ERS process expects. Takes the
    already-generated reports (each independently testable above)
    rather than raw data, so this function's own correctness is just
    'did it assemble the pieces correctly,' not a re-implementation of
    every aggregation above."""
    return {
        "metadata": ReportMetadata("ers_end_of_test", _now(), period_start, period_end).__dict__,
        "customer_demographics": customer_demographics,
        "credit_exposure": credit_exposure,
        "payment_activity": payment_activity,
        "arrears_and_hardship": arrears_and_hardship,
        "complaints_and_afca": complaints_and_afca,
        "consumer_losses_and_remediation": consumer_losses,
        "reconciliation_exceptions": reconciliation_exceptions,
        "security_incidents": security_incidents,
        "pilot_outcomes_and_public_benefit": pilot_outcomes,
    }
