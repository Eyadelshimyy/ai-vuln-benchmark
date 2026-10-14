"""
backend/onboarding.py
======================
Structured customer onboarding and eligibility workflow for the ASIC ERS
pilot credit facility.

Hard rule this module exists to enforce: **no applicant is automatically
approved by an opaque score.** Every decision path here is a named,
inspectable rule (age, identity, residency, bank verification, bill
ownership, bankruptcy status, requested purpose) producing an explicit
outcome and reason codes — never a single numeric "risk score" that
approves or declines on its own. Vulnerability or incomplete-evidence
cases are always routed to manual review, never auto-declined or
auto-approved.

Relationship to other modules:
- `pilot_config.py` supplies the geographic and bill-category boundaries
  this module checks eligibility against (VIC-only, approved utility
  categories) — never hard-coded here.
- `responsible_lending.py` performs the affordability assessment for
  applicants who pass eligibility; this module does not decide
  affordability itself.
- `launch_gates.py` is not called from here directly — whether a
  positive onboarding outcome can actually result in a live credit
  facility is a separate, fail-closed decision made at the real-money
  code path, not assumed by onboarding passing.
- Final credit activation additionally requires
  `approve_credit_activation()` below: a maker-checker step distinct
  from whoever ran the eligibility check or the responsible-lending
  assessment.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional

import supabase_db as sdb

logger = logging.getLogger(__name__)

ELIGIBILITY_OUTCOMES = ("eligible", "declined", "referred", "withdrawn")
FINAL_OUTCOMES = ("approved", "declined", "referred", "withdrawn")

REQUIRED_CONSENTS = ("privacy", "identity_check", "affordability_check", "fraud_check")

# Adverse/decline/referral reason codes. Every non-approval must carry at
# least one of these, never a bare "declined" with no explanation.
REASON_CODES = {
    "AGE_UNCONFIRMED": "Age could not be confirmed as 18 or over",
    "IDENTITY_NOT_VERIFIED": "Identity verification did not pass",
    "ADDRESS_NOT_AUSTRALIAN": "Residential address is not an Australian address",
    "OUTSIDE_PILOT_AREA": "Residential address is outside the current pilot geographic area",
    "BANK_ACCOUNT_NOT_VERIFIED": "Australian bank account could not be verified",
    "UTILITY_OWNERSHIP_NOT_VERIFIED": "Utility bill ownership could not be verified",
    "PURPOSE_NOT_APPROVED": "Requested credit purpose is outside approved bill categories",
    "BANKRUPTCY_UNDISCHARGED": "Applicant has an undischarged bankruptcy",
    "MISSING_CONSENT": "One or more required consents were not recorded",
    "VULNERABILITY_INDICATOR": "Financial vulnerability indicator present — referred for manual review",
    "INCOMPLETE_EVIDENCE": "Application evidence is incomplete — referred for manual review",
    "PILOT_CAPACITY_REACHED": "Pilot customer capacity has been reached",
    "APPLICANT_WITHDREW": "Applicant withdrew the application",
    "AFFORDABILITY_NOT_DEMONSTRATED": "Responsible-lending assessment could not demonstrate affordability",
}


class OnboardingError(Exception):
    """Raised for invalid onboarding operations. Every raise path here is
    a refusal to proceed, not a partial approval."""


@dataclass
class OnboardingApplication:
    user_id: str
    identity_verification_status: str = "pending"      # pending | verified | failed
    age_confirmed: bool = False
    residential_state: Optional[str] = None             # e.g. "VIC"
    residential_country: str = "AU"
    bank_account_verified: bool = False
    income_amount: Optional[str] = None                  # stored as string; encrypted at rest by caller
    income_frequency: Optional[str] = None               # weekly | fortnightly | monthly
    employment_status: Optional[str] = None
    recurring_living_expenses: Optional[str] = None
    existing_debts_and_bnpl: Optional[str] = None
    requested_credit_purpose: Optional[str] = None        # electricity | gas | water | telecommunications
    requirements_and_objectives: Optional[str] = None
    vulnerability_indicators: list = field(default_factory=list)
    bankruptcy_status: str = "unknown"                    # none | undischarged | discharged | unknown
    utility_bill_ownership_verified: bool = False
    consents: dict = field(default_factory=dict)          # {consent_type: {"version": ..., "accepted_at": ...}}

    id: Optional[str] = None
    created_at: Optional[str] = None
    eligibility_outcome: Optional[str] = None
    final_outcome: Optional[str] = None
    reason_codes: list = field(default_factory=list)
    policy_version: Optional[str] = None


@dataclass
class EligibilityResult:
    outcome: str            # eligible | declined | referred | withdrawn
    reason_codes: list
    evidence: dict           # reproducible snapshot of what was checked
    policy_version: str
    checked_at: str


def _has_all_required_consents(consents: dict) -> bool:
    return all(c in consents and consents[c].get("version") and consents[c].get("accepted_at") for c in REQUIRED_CONSENTS)


def evaluate_eligibility(app: OnboardingApplication, approved_geographic_areas, approved_bill_categories,
                          policy_version: str, now: Optional[datetime] = None) -> EligibilityResult:
    """Deterministic, fully reproducible eligibility check. Every input is
    read from the application snapshot passed in — never re-fetched or
    inferred — so the same (app, config, policy_version) always produces
    the same result, satisfying the "reproduce the information and
    policy version used in each decision" requirement.

    Returns 'referred' (never 'declined') for vulnerability indicators or
    incomplete evidence — those are exceptions requiring a human, not
    automatic rejections."""
    now = now or datetime.now(timezone.utc)
    codes = []
    evidence = {
        "identity_verification_status": app.identity_verification_status,
        "age_confirmed": app.age_confirmed,
        "residential_state": app.residential_state,
        "residential_country": app.residential_country,
        "bank_account_verified": app.bank_account_verified,
        "utility_bill_ownership_verified": app.utility_bill_ownership_verified,
        "requested_credit_purpose": app.requested_credit_purpose,
        "bankruptcy_status": app.bankruptcy_status,
        "consents_present": list(app.consents.keys()),
        "vulnerability_indicators": list(app.vulnerability_indicators),
    }

    if not app.age_confirmed:
        codes.append("AGE_UNCONFIRMED")
    if app.identity_verification_status != "verified":
        codes.append("IDENTITY_NOT_VERIFIED")
    if app.residential_country != "AU":
        codes.append("ADDRESS_NOT_AUSTRALIAN")
    elif app.residential_state not in set(approved_geographic_areas):
        codes.append("OUTSIDE_PILOT_AREA")
    if not app.bank_account_verified:
        codes.append("BANK_ACCOUNT_NOT_VERIFIED")
    if not app.utility_bill_ownership_verified:
        codes.append("UTILITY_OWNERSHIP_NOT_VERIFIED")
    if app.requested_credit_purpose not in set(approved_bill_categories):
        codes.append("PURPOSE_NOT_APPROVED")
    if app.bankruptcy_status == "undischarged":
        codes.append("BANKRUPTCY_UNDISCHARGED")
    if not _has_all_required_consents(app.consents):
        codes.append("MISSING_CONSENT")

    # Hard declines: objective, unambiguous disqualifiers.
    hard_decline_codes = {c for c in codes if c not in ("MISSING_CONSENT",)}

    incomplete_evidence = (
        app.income_amount is None
        or app.income_frequency is None
        or app.employment_status is None
        or app.recurring_living_expenses is None
        or not app.requirements_and_objectives
    )

    if hard_decline_codes:
        outcome = "declined"
        final_codes = codes
    elif "MISSING_CONSENT" in codes:
        # Cannot proceed at all without consent, but this is a stop, not
        # a punitive decline — treat as referred so it can be resolved by
        # obtaining consent rather than a permanent adverse record.
        outcome = "referred"
        final_codes = codes
    elif app.vulnerability_indicators:
        outcome = "referred"
        final_codes = codes + ["VULNERABILITY_INDICATOR"]
    elif incomplete_evidence:
        outcome = "referred"
        final_codes = codes + ["INCOMPLETE_EVIDENCE"]
    else:
        outcome = "eligible"
        final_codes = []

    return EligibilityResult(
        outcome=outcome,
        reason_codes=final_codes,
        evidence=evidence,
        policy_version=policy_version,
        checked_at=now.isoformat(),
    )


async def submit_application(app: OnboardingApplication, approved_geographic_areas, approved_bill_categories,
                              policy_version: str) -> dict:
    """Persists the application and its eligibility outcome. Does NOT
    activate any credit — see approve_credit_activation() for the
    separate maker-checker step required after this."""
    result = evaluate_eligibility(app, approved_geographic_areas, approved_bill_categories, policy_version)
    row = {
        "user_id": app.user_id,
        "identity_verification_status": app.identity_verification_status,
        "age_confirmed": app.age_confirmed,
        "residential_state": app.residential_state,
        "residential_country": app.residential_country,
        "bank_account_verified": app.bank_account_verified,
        "employment_status": app.employment_status,
        "requested_credit_purpose": app.requested_credit_purpose,
        "requirements_and_objectives": app.requirements_and_objectives,
        "vulnerability_indicators": list(app.vulnerability_indicators),
        "bankruptcy_status": app.bankruptcy_status,
        "utility_bill_ownership_verified": app.utility_bill_ownership_verified,
        "consents": app.consents,
        "eligibility_outcome": result.outcome,
        "final_outcome": None,
        "reason_codes": result.reason_codes,
        "policy_version": policy_version,
        "eligibility_evidence": result.evidence,
        "created_at": datetime.now(timezone.utc).isoformat(),
        # Sensitive financial fields stored only as caller-encrypted
        # strings — this module never stores plaintext income/expense
        # figures; encryption is the caller's responsibility via
        # utils.auth.encrypt_field before calling submit_application, to
        # keep this module testable without a live ENCRYPTION_KEY.
        "income_amount_encrypted": app.income_amount,
        "income_frequency": app.income_frequency,
        "recurring_living_expenses_encrypted": app.recurring_living_expenses,
        "existing_debts_and_bnpl_encrypted": app.existing_debts_and_bnpl,
    }
    created = await sdb.insert_one("onboarding_applications", row)
    await _append_audit(created["id"], "eligibility_evaluated", app.user_id, result.outcome, {}, row)
    return created


async def _append_audit(application_id: str, action: str, actor: str, reason: str, previous_state: dict, new_state: dict) -> None:
    await sdb.insert_one("onboarding_audit_log", {
        "application_id": application_id,
        "action": action,
        "actor": actor,
        "reason": reason,
        "previous_state": previous_state,
        "new_state": new_state,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


async def record_manual_review_outcome(application_id: str, reviewer: str, outcome: str, reason_codes: list, notes: str) -> dict:
    """Records a human decision on a referred application. `outcome`
    must be one of FINAL_OUTCOMES. Every non-approved outcome must carry
    at least one reason code — this function refuses to record a bare
    decline/referral with no documented reason."""
    if outcome not in FINAL_OUTCOMES:
        raise OnboardingError(f"invalid outcome: {outcome}")
    if outcome != "approved" and not reason_codes:
        raise OnboardingError("a non-approved outcome must carry at least one documented reason code")
    unknown = [c for c in reason_codes if c not in REASON_CODES]
    if unknown:
        raise OnboardingError(f"unknown reason code(s): {unknown}")

    existing = await sdb.find_one("onboarding_applications", {"id": application_id})
    if not existing:
        raise OnboardingError(f"no application {application_id}")

    updates = {
        "final_outcome": outcome,
        "reason_codes": reason_codes,
        "manual_review_notes": notes,
        "manual_reviewed_by": reviewer,
        "manual_reviewed_at": datetime.now(timezone.utc).isoformat(),
    }
    await sdb.update_one("onboarding_applications", {"id": application_id}, updates)
    await _append_audit(application_id, "manual_review_recorded", reviewer, ",".join(reason_codes) or "approved", existing, updates)
    return {**existing, **updates}


async def approve_credit_activation(application_id: str, prepared_by: str, approved_by: str) -> dict:
    """The maker-checker gate between "this applicant is eligible /
    affordability was demonstrated" and an actual credit facility being
    activated for them. `approved_by` must differ from `prepared_by` and
    from whoever ran the original eligibility/manual-review decision.
    Refuses unless the application's final_outcome (or eligibility
    outcome, if no manual review was needed) is 'approved'/'eligible'."""
    app = await sdb.find_one("onboarding_applications", {"id": application_id})
    if not app:
        raise OnboardingError(f"no application {application_id}")
    if prepared_by == approved_by:
        raise OnboardingError("credit activation requires a distinct approver from the preparer (maker-checker)")

    effectively_approved = app.get("final_outcome") == "approved" or (
        app.get("final_outcome") is None and app.get("eligibility_outcome") == "eligible"
    )
    if not effectively_approved:
        raise OnboardingError(
            f"application {application_id} is not in an approved state "
            f"(eligibility_outcome={app.get('eligibility_outcome')}, final_outcome={app.get('final_outcome')})"
        )
    if prepared_by in (app.get("manual_reviewed_by"),):
        raise OnboardingError("the person who made the manual review decision cannot also prepare credit activation")

    activation = await sdb.insert_one("credit_activation_events", {
        "application_id": application_id,
        "prepared_by": prepared_by,
        "approved_by": approved_by,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    await _append_audit(application_id, "credit_activated", approved_by, "maker-checker credit activation", app, activation)
    return activation


def record_consent(consents: dict, consent_type: str, version: str, accepted_at: Optional[str] = None) -> dict:
    """Returns an updated consents dict with a new, versioned,
    timestamped consent entry. Re-accepting an already-recorded consent
    type overwrites only that entry — each entry itself is a snapshot
    (version + timestamp), so history is reconstructable from the
    onboarding_audit_log even though the current `consents` field only
    holds the latest per type."""
    updated = dict(consents)
    updated[consent_type] = {
        "version": version,
        "accepted_at": accepted_at or datetime.now(timezone.utc).isoformat(),
    }
    return updated


# ---------------------------------------------------------------
# Real identity-verification provider wiring (identity_verification.py).
# This is the call site that turns identity_verification_status from a
# caller-supplied field into something backed by an actual check.
# Deliberately a thin wrapper: onboarding.py stays the source of truth
# for the application record; identity_verification.py stays entirely
# unaware that "onboarding applications" exist. Neither module imports
# the other's data model, only this glue function does.
# ---------------------------------------------------------------

async def start_identity_verification(application_id: str, workflow_id: Optional[str] = None,
                                       callback: Optional[str] = None) -> "identity_verification.VerificationSession":
    """Starts a real (or, in explicitly-gated dev mode, mock) identity
    verification session for an application and records the session
    reference on it. Does NOT change identity_verification_status yet —
    that only happens once a result comes back, via a webhook (see
    apply_identity_verification_webhook(), the authoritative path) or a
    manual poll (apply_identity_verification_result(), secondary).
    Propagates identity_verification.IdentityVerificationError untouched
    on failure (e.g. no provider configured) rather than swallowing it —
    this function must never leave an application looking verified when
    the provider call actually failed. Returns the full
    VerificationSession (including the `url` the applicant needs to
    open) rather than just a session id, since a caller building the
    actual onboarding UI needs that URL."""
    import identity_verification as idv

    app = await sdb.find_one("onboarding_applications", {"id": application_id})
    if not app:
        raise OnboardingError(f"no application {application_id}")

    session = await idv.start_verification_session(applicant_reference=application_id, workflow_id=workflow_id, callback=callback)
    await sdb.update_one("onboarding_applications", {"id": application_id}, {"identity_verification_session_id": session.session_id})
    await _append_audit(application_id, "identity_verification_started", app.get("user_id", "system"),
                         f"provider={session.provider} session={session.session_id}", app,
                         {"identity_verification_session_id": session.session_id})
    return session


async def apply_identity_verification_result(application_id: str) -> dict:
    """Secondary path: polls the provider for the application's stored
    session and updates identity_verification_status to match. Prefer
    apply_identity_verification_webhook() in production — per
    identity_verification.py's own docstring, the webhook is Didit's
    authoritative source, this poll is a fallback (admin 'check now'
    button, backfill) not the primary flow."""
    import identity_verification as idv

    app = await sdb.find_one("onboarding_applications", {"id": application_id})
    if not app:
        raise OnboardingError(f"no application {application_id}")
    session_id = app.get("identity_verification_session_id")
    if not session_id:
        raise OnboardingError(f"application {application_id} has no identity verification session started yet")

    result = await idv.get_verification_result(session_id)
    updates = {"identity_verification_status": result.status}
    await sdb.update_one("onboarding_applications", {"id": application_id}, updates)
    await _append_audit(application_id, "identity_verification_result_applied", "system",
                         f"provider={result.provider} status={result.status}", app, updates)
    return {**app, **updates}


async def apply_identity_verification_webhook(raw_body: bytes, signature: str, timestamp: str) -> Optional[dict]:
    """The authoritative path: verifies a Didit webhook delivery, then
    applies its decision to the application whose id matches the
    webhook's `vendor_data` field (onboarding.py's convention is to pass
    application_id as vendor_data at session-creation time — see
    start_identity_verification()).

    Idempotent on `event_id`: a duplicate delivery (Didit retries up to
    twice on 5xx/404) is detected and returns None without reapplying —
    callers should still return 2xx to Didit for a duplicate, since the
    correct response is 'already handled', not an error.

    Raises identity_verification.IdentityVerificationError if the
    signature doesn't verify — callers should return 401 in that case,
    never process the payload."""
    import identity_verification as idv

    if not idv.verify_webhook_signature(raw_body, signature, timestamp):
        raise idv.IdentityVerificationError("webhook signature verification failed — refusing to process this delivery")

    event = idv.parse_webhook_event(raw_body)

    already_processed = await sdb.find_one("didit_webhook_events", {"event_id": event.event_id})
    if already_processed:
        return None
    await sdb.insert_one("didit_webhook_events", {
        "event_id": event.event_id, "session_id": event.session_id, "application_id": event.vendor_data,
        "status": event.status, "received_at": datetime.now(timezone.utc).isoformat(),
    })

    application_id = event.vendor_data
    app = await sdb.find_one("onboarding_applications", {"id": application_id})
    if not app:
        # A webhook for an application_id we don't recognise is logged,
        # not silently dropped and not raised as if it were our error —
        # the event is still recorded above for audit/debugging.
        logger.warning("Didit webhook for unknown application_id=%s (session=%s)", application_id, event.session_id)
        return None

    updates = {"identity_verification_status": event.onboarding_status}
    await sdb.update_one("onboarding_applications", {"id": application_id}, updates)
    await _append_audit(application_id, "identity_verification_webhook_applied", "didit_webhook",
                         f"status={event.status} -> {event.onboarding_status}", app, updates)
    return {**app, **updates}
