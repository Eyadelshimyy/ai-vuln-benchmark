"""
backend/responsible_lending.py
================================
Deterministic responsible-lending assessment for the ASIC ERS pilot
credit facility. No machine-learning credit scoring — every number here
is a plain arithmetic rule against verified inputs, and every rejection
or referral is explainable in one sentence.

This module answers one question: "can this proposed BillSure repayment
be made without substantial hardship, given verified income and
expenses?" It does not decide whether to onboard the applicant (see
onboarding.py) and it does not activate credit (see
onboarding.approve_credit_activation(), a separate maker-checker step).

Design rules enforced here:
- Affordability can only be demonstrated with complete, internally
  consistent, and reasonably current evidence. Missing, stale, or
  inconsistent inputs block approval outright — they do not get
  "assumed favourable".
- Any hardship or vulnerability indicator forces referral, regardless of
  what the arithmetic says.
- A recommendation from this module is never final. A human can only
  override it with a documented reason AND a second, independent
  approval (override author != approver) — recorded, not silent.
- Limit increases always require a fresh assessment. There is no
  function anywhere in this module that increases a limit based on an
  old assessment or with no assessment at all.
"""

import logging
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from datetime import datetime, timezone, timedelta
from typing import Optional

import supabase_db as sdb

logger = logging.getLogger(__name__)

POLICY_VERSION = "rl-pilot-v1"

# Evidence older than this is treated as stale and blocks approval —
# affordability decisions must be based on reasonably current figures.
MAX_EVIDENCE_AGE_DAYS = 90

RECOMMENDATIONS = ("approve", "decline", "refer")


class ResponsibleLendingError(Exception):
    """Raised for invalid assessment operations — always a refusal to
    proceed, never a partial or assumed-favourable result."""


@dataclass
class AffordabilityInputs:
    gross_income_amount: Decimal
    income_frequency: str                    # weekly | fortnightly | monthly
    essential_expenditure_monthly: Decimal
    discretionary_expenditure_monthly: Decimal
    existing_credit_repayments_monthly: Decimal
    bnpl_repayments_monthly: Decimal
    proposed_billsure_repayment_monthly: Decimal
    evidence_as_of: str                        # ISO date the income/expense evidence was collected
    vulnerability_indicators: list = field(default_factory=list)
    hardship_flag: bool = False


@dataclass
class AssessmentResult:
    verified_net_income_monthly: Decimal
    essential_expenditure_monthly: Decimal
    existing_repayments_monthly: Decimal
    proposed_repayment_monthly: Decimal
    surplus_monthly: Decimal
    affordability_pass: bool
    evidence_issues: list
    referral_required: bool
    recommendation: str          # approve | decline | refer
    reasons: list
    policy_version: str
    human_readable_report: str
    assessed_at: str


_FREQUENCY_TO_MONTHLY = {
    "weekly": Decimal("52") / Decimal("12"),
    "fortnightly": Decimal("26") / Decimal("12"),
    "monthly": Decimal("1"),
}


def _to_monthly(amount: Decimal, frequency: str) -> Decimal:
    factor = _FREQUENCY_TO_MONTHLY.get(frequency)
    if factor is None:
        raise ResponsibleLendingError(f"unknown income_frequency: {frequency}")
    return (amount * factor).quantize(Decimal("0.01"))


def _detect_evidence_issues(inputs: AffordabilityInputs, now: datetime) -> list:
    issues = []
    for label, value in (
        ("gross_income_amount", inputs.gross_income_amount),
        ("essential_expenditure_monthly", inputs.essential_expenditure_monthly),
        ("existing_credit_repayments_monthly", inputs.existing_credit_repayments_monthly),
        ("bnpl_repayments_monthly", inputs.bnpl_repayments_monthly),
        ("proposed_billsure_repayment_monthly", inputs.proposed_billsure_repayment_monthly),
    ):
        if value is None:
            issues.append(f"INCOMPLETE:{label}")
        elif value < 0:
            issues.append(f"INCONSISTENT:{label} is negative")

    try:
        evidence_date = datetime.fromisoformat(inputs.evidence_as_of)
        if evidence_date.tzinfo is None:
            evidence_date = evidence_date.replace(tzinfo=timezone.utc)
        if (now - evidence_date) > timedelta(days=MAX_EVIDENCE_AGE_DAYS):
            issues.append("STALE:evidence_as_of older than 90 days")
    except (ValueError, TypeError):
        issues.append("INCOMPLETE:evidence_as_of missing or unparsable")

    if inputs.gross_income_amount is not None and inputs.gross_income_amount == 0:
        issues.append("INCONSISTENT:gross_income_amount is zero")

    return issues


def run_assessment(inputs: AffordabilityInputs, now: Optional[datetime] = None) -> AssessmentResult:
    """Deterministic affordability assessment. Never raises on a bad
    outcome — a decline or referral is a valid, expected result, not an
    error. Raises ResponsibleLendingError only for malformed input types
    that make the calculation itself impossible (e.g. unknown frequency)."""
    now = now or datetime.now(timezone.utc)
    issues = _detect_evidence_issues(inputs, now)
    reasons = []

    if issues:
        # Cannot demonstrate affordability on incomplete/stale/inconsistent
        # evidence — block outright rather than proceeding on partial data.
        report = (
            "Assessment could not be completed: evidence issues found "
            f"({'; '.join(issues)}). Affordability has not been demonstrated; "
            "referred for manual evidence collection before any decision."
        )
        return AssessmentResult(
            verified_net_income_monthly=Decimal("0"),
            essential_expenditure_monthly=inputs.essential_expenditure_monthly or Decimal("0"),
            existing_repayments_monthly=(inputs.existing_credit_repayments_monthly or Decimal("0"))
            + (inputs.bnpl_repayments_monthly or Decimal("0")),
            proposed_repayment_monthly=inputs.proposed_billsure_repayment_monthly or Decimal("0"),
            surplus_monthly=Decimal("0"),
            affordability_pass=False,
            evidence_issues=issues,
            referral_required=True,
            recommendation="refer",
            reasons=["EVIDENCE_ISSUES"] + issues,
            policy_version=POLICY_VERSION,
            human_readable_report=report,
            assessed_at=now.isoformat(),
        )

    net_income_monthly = _to_monthly(inputs.gross_income_amount, inputs.income_frequency)
    existing_repayments = inputs.existing_credit_repayments_monthly + inputs.bnpl_repayments_monthly
    surplus = (
        net_income_monthly
        - inputs.essential_expenditure_monthly
        - existing_repayments
        - inputs.proposed_billsure_repayment_monthly
    )
    affordability_pass = surplus >= Decimal("0.00")

    referral_required = bool(inputs.vulnerability_indicators) or inputs.hardship_flag

    if referral_required:
        recommendation = "refer"
        reasons.append("VULNERABILITY_OR_HARDSHIP_INDICATOR")
    elif not affordability_pass:
        recommendation = "decline"
        reasons.append("AFFORDABILITY_NOT_DEMONSTRATED")
    else:
        recommendation = "approve"

    report = (
        f"Verified net income: ${net_income_monthly}/month. "
        f"Essential expenditure: ${inputs.essential_expenditure_monthly}/month. "
        f"Existing credit + BNPL repayments: ${existing_repayments}/month. "
        f"Proposed BillSure repayment: ${inputs.proposed_billsure_repayment_monthly}/month. "
        f"Resulting surplus: ${surplus}/month — "
        f"{'affordable without substantial hardship' if affordability_pass else 'insufficient surplus to demonstrate affordability'}. "
        f"Recommendation: {recommendation}"
        + (" (referred due to a disclosed vulnerability or hardship indicator, overriding the arithmetic result)."
           if referral_required else ".")
    )

    return AssessmentResult(
        verified_net_income_monthly=net_income_monthly,
        essential_expenditure_monthly=inputs.essential_expenditure_monthly,
        existing_repayments_monthly=existing_repayments,
        proposed_repayment_monthly=inputs.proposed_billsure_repayment_monthly,
        surplus_monthly=surplus,
        affordability_pass=affordability_pass,
        evidence_issues=[],
        referral_required=referral_required,
        recommendation=recommendation,
        reasons=reasons,
        policy_version=POLICY_VERSION,
        human_readable_report=report,
        assessed_at=now.isoformat(),
    )


async def persist_assessment(application_id: str, inputs: AffordabilityInputs, assessed_by: str) -> dict:
    result = run_assessment(inputs)
    row = {
        "application_id": application_id,
        "assessed_by": assessed_by,
        "verified_net_income_monthly": str(result.verified_net_income_monthly),
        "essential_expenditure_monthly": str(result.essential_expenditure_monthly),
        "existing_repayments_monthly": str(result.existing_repayments_monthly),
        "proposed_repayment_monthly": str(result.proposed_repayment_monthly),
        "surplus_monthly": str(result.surplus_monthly),
        "affordability_pass": result.affordability_pass,
        "evidence_issues": result.evidence_issues,
        "referral_required": result.referral_required,
        "recommendation": result.recommendation,
        "reasons": result.reasons,
        "policy_version": result.policy_version,
        "human_readable_report": result.human_readable_report,
        "assessed_at": result.assessed_at,
        "superseded": False,
    }
    return await sdb.insert_one("responsible_lending_assessments", row)


async def override_recommendation(assessment_id: str, override_to: str, reason: str,
                                   overridden_by: str, approved_by: str) -> dict:
    """A compliance reviewer can only override an automated recommendation
    with a documented reason AND independent approval — approved_by must
    differ from overridden_by. Refuses silently-approved or self-approved
    overrides outright."""
    if override_to not in RECOMMENDATIONS:
        raise ResponsibleLendingError(f"invalid override_to: {override_to}")
    if not reason or not reason.strip():
        raise ResponsibleLendingError("an override requires a documented, non-empty reason")
    if overridden_by == approved_by:
        raise ResponsibleLendingError("override approval must be independent of the person requesting the override")

    assessment = await sdb.find_one("responsible_lending_assessments", {"id": assessment_id})
    if not assessment:
        raise ResponsibleLendingError(f"no assessment {assessment_id}")

    record = await sdb.insert_one("responsible_lending_overrides", {
        "assessment_id": assessment_id,
        "original_recommendation": assessment["recommendation"],
        "override_to": override_to,
        "reason": reason,
        "overridden_by": overridden_by,
        "approved_by": approved_by,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    return record


async def can_increase_limit(customer_id: str, current_limit: Decimal, requested_limit: Decimal,
                              latest_assessment: Optional[dict]) -> None:
    """Raises ResponsibleLendingError unless a FRESH (not superseded,
    passing) responsible-lending assessment exists to support the
    increase. There is deliberately no path in this module that
    increases a limit automatically off an old assessment, a decline, a
    referral, or no assessment at all — this is the enforcement point
    for 'prohibits automatic limit increases' and 'supports reassessment
    before any limit increase'."""
    if requested_limit <= current_limit:
        raise ResponsibleLendingError("not a limit increase — nothing to reassess")
    if latest_assessment is None:
        raise ResponsibleLendingError("no responsible-lending assessment on file — a fresh assessment is required before any limit increase")
    if latest_assessment.get("superseded"):
        raise ResponsibleLendingError("the latest assessment on file has been superseded — reassessment required")
    if latest_assessment.get("recommendation") != "approve":
        raise ResponsibleLendingError(
            f"latest assessment recommendation is '{latest_assessment.get('recommendation')}', not 'approve' — "
            "cannot increase limit without a passing reassessment (or a documented, independently-approved override)"
        )
    assessed_at = latest_assessment.get("assessed_at")
    if assessed_at:
        assessed_dt = datetime.fromisoformat(assessed_at)
        if assessed_dt.tzinfo is None:
            assessed_dt = assessed_dt.replace(tzinfo=timezone.utc)
        if (datetime.now(timezone.utc) - assessed_dt) > timedelta(days=MAX_EVIDENCE_AGE_DAYS):
            raise ResponsibleLendingError("latest assessment is older than the evidence-freshness window — reassessment required")
