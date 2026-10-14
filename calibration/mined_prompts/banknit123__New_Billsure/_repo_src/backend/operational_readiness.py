"""
backend/operational_readiness.py
===================================
Operational readiness (task section 14) — the parts genuinely
implementable as code: health/readiness aggregation, feature flags,
background-job heartbeat monitoring, backup-verification record-
keeping, and synthetic ASIC-review demonstration scenarios.

Like security_controls.py, several section-14 items aren't code this
module can provide: actual database migration execution/checking is a
deployment-pipeline concern (the migrations themselves, 012-022, are
the artefact), and backup/restore/rollback PROCEDURES are
organisational runbooks — see `docs/asic-ers-readiness/runbooks/`, not
this module. What this module provides is the bookkeeping/verification
layer around those procedures: recording that a backup was taken and
verified, not performing the backup itself (no storage backend is
integrated here).
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional

import supabase_db as sdb

logger = logging.getLogger(__name__)


class OperationalReadinessError(Exception):
    """Raised for any invalid operational-readiness check. Every raise
    path here means the system should be reported as NOT ready, never
    silently reported as healthy."""


# ---------------------------------------------------------------
# Health / readiness
# ---------------------------------------------------------------

@dataclass
class ComponentHealth:
    name: str
    healthy: bool
    detail: str = ""


@dataclass
class HealthReport:
    overall_healthy: bool
    checked_at: str
    components: list = field(default_factory=list)


async def check_health(component_checks: dict) -> HealthReport:
    """Aggregates a set of already-run component checks (name ->
    ComponentHealth) into one report. Deliberately takes pre-computed
    results rather than running checks itself, so this function stays
    infrastructure-agnostic — a caller decides what "database
    connectivity," "background job heartbeat," etc. actually mean for
    its deployment, and this function only aggregates. overall_healthy
    is False if ANY component is unhealthy — there is no averaging or
    partial-credit toward a healthy overall status."""
    components = list(component_checks.values())
    overall = all(c.healthy for c in components) if components else False
    report = HealthReport(overall_healthy=overall, checked_at=datetime.now(timezone.utc).isoformat(), components=components)
    return report


# ---------------------------------------------------------------
# Feature flags
# ---------------------------------------------------------------

# Every real-money or customer-facing pilot capability should be gated
# behind an explicit flag, defaulting OFF, mirroring the fail-closed
# posture already established by ALLOW_MOCK_PAYMENTS / ALLOW_MOCK_
# IDENTITY_VERIFICATION / ALLOW_MOCK_BANK_VERIFICATION elsewhere in this
# codebase. This registry doesn't replace those env-var flags — it's a
# structured place to track flag state and history when a flag isn't
# just a simple env var (e.g. a flag that needs an audit trail of who
# toggled it and when, unlike a plain os.environ read).
async def set_feature_flag(flag_name: str, enabled: bool, changed_by: str, reason: str) -> dict:
    if not reason or not reason.strip():
        raise OperationalReadinessError("changing a feature flag requires a documented reason")
    return await sdb.insert_one("feature_flag_changes", {
        "flag_name": flag_name, "enabled": enabled, "changed_by": changed_by,
        "reason": reason, "changed_at": datetime.now(timezone.utc).isoformat(),
    })


async def get_feature_flag(flag_name: str) -> bool:
    """Returns the most recent value set for this flag, or False
    (disabled) if it has never been explicitly set — fails closed:
    an unrecognised or never-configured flag is OFF, never ON by
    default."""
    changes = await sdb.find_many("feature_flag_changes", {"flag_name": flag_name})
    if not changes:
        return False
    latest = max(changes, key=lambda c: c["changed_at"])
    return latest["enabled"]


# ---------------------------------------------------------------
# Background-job monitoring
# ---------------------------------------------------------------

# A job is considered stalled if it hasn't reported a heartbeat within
# this window. Conservative default; a real deployment should tune this
# per job based on its actual expected cadence (e.g. the existing
# scheduler.py's reconciliation_loop runs daily, so its own threshold
# would be longer than this generic default).
DEFAULT_STALL_THRESHOLD_MINUTES = 15


async def record_job_heartbeat(job_name: str, status: str = "ok", detail: Optional[str] = None) -> dict:
    return await sdb.insert_one("job_heartbeats", {
        "job_name": job_name, "status": status, "detail": detail,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    })


async def is_job_stalled(job_name: str, threshold_minutes: int = DEFAULT_STALL_THRESHOLD_MINUTES,
                          now: Optional[datetime] = None) -> bool:
    """True if the job has never reported a heartbeat, OR its most
    recent heartbeat is older than the threshold. A job that has never
    run at all is treated as stalled, not as 'unknown, assume fine' —
    fail closed."""
    now = now or datetime.now(timezone.utc)
    heartbeats = await sdb.find_many("job_heartbeats", {"job_name": job_name})
    if not heartbeats:
        return True
    latest = max(heartbeats, key=lambda h: h["recorded_at"])
    recorded_at = datetime.fromisoformat(latest["recorded_at"])
    if recorded_at.tzinfo is None:
        recorded_at = recorded_at.replace(tzinfo=timezone.utc)
    return (now - recorded_at) > timedelta(minutes=threshold_minutes)


# ---------------------------------------------------------------
# Backup verification
# ---------------------------------------------------------------

async def record_backup_verification(backup_reference: str, verified_by: str, restore_tested: bool, notes: str = "") -> dict:
    """Records that a backup was checked. `restore_tested=True` should
    only ever be set after an ACTUAL restore was performed into a
    non-production environment and confirmed to produce usable data —
    this function trusts the caller's flag rather than performing any
    restore itself (no storage backend is integrated here); the
    distinction between 'a backup file exists' and 'we proved it
    restores' matters enough that this field is separate from just
    recording the backup happened."""
    return await sdb.insert_one("backup_verifications", {
        "backup_reference": backup_reference, "verified_by": verified_by,
        "restore_tested": restore_tested, "notes": notes,
        "verified_at": datetime.now(timezone.utc).isoformat(),
    })


# ---------------------------------------------------------------
# Synthetic ASIC-review demonstration scenarios
# ---------------------------------------------------------------

@dataclass(frozen=True)
class DemoScenario:
    key: str
    title: str
    description: str
    demonstrates: list


# Each scenario names the specific control(s) it's meant to demonstrate
# to a reviewer, and is built entirely from synthetic data across
# sessions 1-12's modules — no real customer data anywhere in any of
# these.
SYNTHETIC_ASIC_REVIEW_SCENARIOS = (
    DemoScenario(
        "customer_cap_enforcement", "26th customer activation is blocked",
        "Activate 25 synthetic customer credit accounts, then attempt a 26th and show it is rejected by pilot_config.check_customer_cap() / credit_ledger's DB-level trigger.",
        ["pilot_config.py", "credit_ledger.py", "migration 015"],
    ),
    DemoScenario(
        "end_to_end_bill_payment", "A verified bill is paid via the full flow, blocked payments leave zero side effects",
        "Submit a synthetic bill, verify it, activate credit, pay it via pilot_payment_flow.pay_verified_bill(), then attempt a second payment against the same bill and show it is refused with no ledger change.",
        ["bill_verification.py", "payment_permitted_use.py", "credit_ledger.py", "pilot_payment_flow.py"],
    ),
    DemoScenario(
        "hardship_without_prior_payment", "A customer with zero payment history can request hardship support",
        "Create a synthetic customer with no credit account and no payment history, call hardship_collections.request_hardship(), and show it succeeds with no payment-status check.",
        ["hardship_collections.py"],
    ),
    DemoScenario(
        "launch_gate_fail_closed", "Production stays blocked until every regulatory gate is approved",
        "Show launch_gates.is_production_authorized() returning False with zero gates recorded, approve 21 of 22 gates, show it is still False, approve the 22nd, show it becomes True, then expire one gate and show it immediately reverts to False.",
        ["launch_gates.py", "migration 012"],
    ),
    DemoScenario(
        "maker_checker_everywhere", "No single person can both propose and approve a financial or compliance decision",
        "Walk through pilot_config version activation, credit account activation, responsible-lending overrides, hardship arrangement approval, document version approval, and complaint remedy approval, showing each refuses a same-person maker-checker attempt.",
        ["pilot_config.py", "credit_ledger.py", "responsible_lending.py", "hardship_collections.py", "document_versioning.py", "complaints.py"],
    ),
    DemoScenario(
        "document_acceptance_reproducibility", "Exactly what a customer accepted can be reproduced after the document changes",
        "Approve a document version, record a synthetic customer's acceptance, publish a new material version, and show document_versioning.reproduce_accepted_document() still returns the original content the customer actually saw.",
        ["document_versioning.py"],
    ),
)


def get_demo_scenario(key: str) -> Optional[DemoScenario]:
    for s in SYNTHETIC_ASIC_REVIEW_SCENARIOS:
        if s.key == key:
            return s
    return None
