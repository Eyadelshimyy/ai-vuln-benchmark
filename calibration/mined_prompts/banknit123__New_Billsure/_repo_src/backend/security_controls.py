"""
backend/security_controls.py
==============================
Security and privacy controls for the ASIC ERS pilot (task section 13)
that are genuinely implementable as code — role-based access control,
least privilege, MFA requirements, PII-safe logging, secure file-upload
validation with a malware-scan integration point, and the account-
deletion/data-retention workflow.

Several section-13 items are NOT code this module can provide —
security headers and CSRF/XSS/injection protections are properties of
how `server.py`'s FastAPI app is actually configured (out of scope for
this module to retrofit unattended into a 150KB+ file it didn't
author), dependency/secret scanning is a CI pipeline concern, and
backup/restore/incident-response/data-breach procedures are
organisational runbooks, not functions — see
`docs/asic-ers-readiness/` for those (incident-response.md,
security-and-privacy.md, and the runbooks/ directory).

Data-retention figures are sourced, not invented: the AML/CTF Act 2006
(Cth) requires reporting entities to retain customer due-diligence and
transaction records for at least 7 years from the date the transaction
completed or the business relationship ended (AUSTRAC record-keeping
guidance, s.106/s.111 of the Act); ASIC additionally expects AFSL
holders to retain financial and compliance records for at least 7
years. `MINIMUM_RETENTION_YEARS` uses that figure. Confirm the exact
applicable period with legal counsel before relying on this for a real
pilot — this is a sourced default, not itself legal advice.

This module never logs, and structurally cannot log, the categories the
task spec explicitly prohibits: government identifiers, full bank
details, credentials, or sensitive affordability information —
`redact_sensitive(...)` is designed to catch these even in free-text
log messages, not just structured fields.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional

import supabase_db as sdb

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------
# Role-based access control / least privilege
# ---------------------------------------------------------------

ROLES = ("customer", "case_worker", "compliance_reviewer", "admin", "system")

# Least-privilege matrix: each role gets only the permissions it needs,
# not a superset. 'admin' is deliberately NOT granted every permission
# automatically — each permission is listed explicitly per role, so
# adding a new permission requires a conscious decision about which
# roles get it, rather than every existing admin silently inheriting it.
ROLE_PERMISSIONS = {
    "customer": frozenset({"view_own_data", "submit_bill", "submit_complaint", "request_hardship", "accept_document",
                            "submit_application", "start_identity_verification", "view_own_balance"}),
    "case_worker": frozenset({"view_own_data", "view_assigned_complaints", "add_investigation_note",
                               "record_customer_communication", "acknowledge_complaint",
                               "submit_application", "manual_review_application", "manual_review_bill",
                               "view_customer_balances", "submit_bill", "request_hardship", "submit_complaint",
                               "start_identity_verification"}),
    "compliance_reviewer": frozenset({"view_own_data", "view_assigned_complaints", "add_investigation_note",
                                       "record_customer_communication", "acknowledge_complaint",
                                       "approve_remedy", "resolve_complaint", "approve_document_version",
                                       "approve_credit_activation", "approve_config_change", "approve_launch_gate",
                                       "manual_review_application", "manual_review_bill", "view_customer_balances",
                                       "process_payment", "export_reports"}),
    "admin": frozenset({"view_own_data", "manage_users", "view_audit_log", "export_reports",
                         "manage_biller_allowlist", "propose_config_change",
                         "submit_application", "manual_review_application", "manual_review_bill",
                         "view_customer_balances", "process_payment", "approve_credit_activation",
                         "submit_bill", "request_hardship", "submit_complaint", "accept_document",
                         "start_identity_verification"}),
    "system": frozenset({"post_ledger_entry", "record_webhook_event", "run_reconciliation",
                          "process_payment", "view_customer_balances"}),
}


class SecurityControlError(Exception):
    """Raised for any access-control, retention, or upload-validation
    violation. Every raise path here is a refusal."""


def has_permission(role: str, permission: str) -> bool:
    if role not in ROLE_PERMISSIONS:
        return False
    return permission in ROLE_PERMISSIONS[role]


def require_permission(role: str, permission: str) -> None:
    """Raises unless the role has the permission. Callers use this as a
    guard at the top of any privileged operation — fails closed on an
    unknown role or unknown permission alike (an unknown role has an
    empty permission set by construction, via has_permission's False
    default)."""
    if not has_permission(role, permission):
        raise SecurityControlError(f"role '{role}' does not have permission '{permission}'")


# Roles that must have MFA verified before performing any privileged
# action — administrators and anyone who can approve a financial or
# compliance decision, per the task spec's explicit MFA-for-
# administrators requirement, extended here to every role that can
# approve something (not just the literal 'admin' role) since
# compliance_reviewer holds equivalent approval power in this system.
MFA_REQUIRED_ROLES = frozenset({"admin", "compliance_reviewer"})


def requires_mfa(role: str) -> bool:
    return role in MFA_REQUIRED_ROLES


async def record_mfa_verification(user_id: str, role: str, method: str) -> dict:
    """Records that MFA was verified for this session/user. Does not
    itself perform MFA (no TOTP/SMS provider integrated) — this is the
    bookkeeping side; a real login flow would call this only after an
    actual second factor genuinely succeeded."""
    return await sdb.insert_one("mfa_verifications", {
        "user_id": user_id, "role": role, "method": method,
        "verified_at": datetime.now(timezone.utc).isoformat(),
    })


def require_mfa_verified(role: str, mfa_verified: bool) -> None:
    """Guard for any action a MFA_REQUIRED_ROLES role is about to take.
    Fails closed: if the role requires MFA and it wasn't verified this
    session, the action is refused — there is no code path that lets an
    admin/compliance action proceed on an unverified session."""
    if requires_mfa(role) and not mfa_verified:
        raise SecurityControlError(f"role '{role}' requires MFA verification before this action, but none was recorded for this session")


# ---------------------------------------------------------------
# PII-safe logging
# ---------------------------------------------------------------

# Patterns matched against free-text log messages, not just structured
# field names (see audit_events.SENSITIVE_FIELD_NAMES for the
# structured-field denylist used at export time) -- catches a
# government ID or bank detail that ends up embedded in a log message
# string rather than a dict key.
_TFN_PATTERN = re.compile(r"\b\d{3}[\s-]?\d{3}[\s-]?\d{3}\b")            # AU Tax File Number: 9 digits
_MEDICARE_PATTERN = re.compile(r"\b\d{4}[\s-]?\d{5}[\s-]?\d{1}\b")        # AU Medicare number: 10-11 digits
_BSB_PATTERN = re.compile(r"\b\d{3}-\d{3}\b")                            # AU BSB: 6 digits, XXX-XXX
_ACCOUNT_NUMBER_PATTERN = re.compile(r"(?i)\baccount\s*(?:number|no\.?)?\s*[:#]?\s*(\d{6,10})\b")
_CREDENTIAL_PATTERN = re.compile(r"(?i)(password|api[_-]?key|secret|token)\s*[:=]\s*\S+")


def redact_sensitive(message: str) -> str:
    """Returns `message` with any government-identifier-, bank-detail-,
    or credential-shaped substring replaced with a redaction marker.
    Intended as a defence-in-depth layer applied at the logging
    boundary (e.g. a logging.Filter) — not a substitute for simply not
    putting sensitive data into log messages in the first place, but a
    real safety net for the case where a developer does anyway."""
    redacted = _TFN_PATTERN.sub("[REDACTED-ID]", message)
    redacted = _MEDICARE_PATTERN.sub("[REDACTED-ID]", redacted)
    redacted = _BSB_PATTERN.sub("[REDACTED-BANK-DETAILS]", redacted)
    redacted = _ACCOUNT_NUMBER_PATTERN.sub("account [REDACTED-BANK-DETAILS]", redacted)
    redacted = _CREDENTIAL_PATTERN.sub(lambda m: f"{m.group(1)}=[REDACTED-CREDENTIAL]", redacted)
    return redacted


class PiiRedactingLogFilter(logging.Filter):
    """A logging.Filter that applies redact_sensitive() to every log
    record's message before it's emitted. Attach to any logger/handler
    that might receive free-text messages containing customer data:
    `logger.addFilter(PiiRedactingLogFilter())`."""
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            record.msg = redact_sensitive(str(record.msg))
        except Exception:
            pass  # never let a redaction bug block logging entirely
        return True


# ---------------------------------------------------------------
# Secure file-upload validation
# ---------------------------------------------------------------

ALLOWED_UPLOAD_EXTENSIONS = frozenset({".pdf", ".jpg", ".jpeg", ".png"})
MAX_UPLOAD_SIZE_BYTES = 15 * 1024 * 1024  # 15MB

# Magic-byte signatures for the allowed types, checked against actual
# file content -- not just the extension, which is trivially spoofable.
_MAGIC_BYTES = {
    ".pdf": (b"%PDF-",),
    ".jpg": (b"\xff\xd8\xff",),
    ".jpeg": (b"\xff\xd8\xff",),
    ".png": (b"\x89PNG\r\n\x1a\n",),
}


@dataclass
class UploadValidationResult:
    filename: str
    size_bytes: int
    extension: str
    malware_scan_status: str   # 'pending' | 'clean' | 'infected' -- see note below


def validate_file_upload(filename: str, content: bytes, allowed_extensions=ALLOWED_UPLOAD_EXTENSIONS,
                          max_size_bytes: int = MAX_UPLOAD_SIZE_BYTES) -> UploadValidationResult:
    """Validates size, extension, and actual file-content magic bytes
    (not just the claimed extension). Returns malware_scan_status=
    'pending' always — NO malware scanner is integrated in this
    codebase; this is the documented integration point (per the task
    spec's 'malware-scanning integration point' requirement), not a
    working scanner. A caller MUST treat 'pending' as 'not yet safe to
    process' and must not advance a file to actual use (e.g. bill
    verification, disbursement) until a real scan result exists —
    there is no code path in this module that reports 'clean' on its
    own, since nothing here can actually determine that."""
    if len(content) == 0:
        raise SecurityControlError("uploaded file is empty")
    if len(content) > max_size_bytes:
        raise SecurityControlError(f"uploaded file ({len(content)} bytes) exceeds the maximum allowed size ({max_size_bytes} bytes)")

    ext = None
    lower = filename.lower()
    for candidate in allowed_extensions:
        if lower.endswith(candidate):
            ext = candidate
            break
    if ext is None:
        raise SecurityControlError(f"file extension not allowed (permitted: {sorted(allowed_extensions)})")

    signatures = _MAGIC_BYTES.get(ext, ())
    if signatures and not any(content.startswith(sig) for sig in signatures):
        raise SecurityControlError(
            f"file content does not match its extension ({ext}) -- possible spoofed/renamed file, refusing to accept it"
        )

    return UploadValidationResult(filename=filename, size_bytes=len(content), extension=ext, malware_scan_status="pending")


# ---------------------------------------------------------------
# Data retention / account deletion
# ---------------------------------------------------------------

# Sourced from AML/CTF Act 2006 (Cth) record-keeping requirements and
# ASIC's AFSL record-keeping expectations -- see module docstring.
MINIMUM_RETENTION_YEARS = 7
RETENTION_SOURCE = "AML/CTF Act 2006 (Cth) record-keeping requirements (AUSTRAC guidance); ASIC AFSL record-keeping expectations"


def compute_retention_until(last_relevant_date: datetime, retention_years: int = MINIMUM_RETENTION_YEARS) -> str:
    return last_relevant_date.replace(year=last_relevant_date.year + retention_years).isoformat()


async def request_account_deletion(customer_id: str, requested_by: str, reason: str,
                                    last_transaction_date: Optional[datetime] = None) -> dict:
    """Records a deletion request. Does NOT delete anything immediately
    — computes retention_until per MINIMUM_RETENTION_YEARS from the
    customer's last relevant transaction date (or now, if they never
    transacted) and stores the request as 'pending_retention_period'.
    A separate process (not built in this codebase — no scheduled job
    exists yet) would need to actually purge data once retention_until
    passes; this function's job is to make the retention boundary
    explicit and auditable, not to perform deletion itself."""
    last_relevant = last_transaction_date or datetime.now(timezone.utc)
    retention_until = compute_retention_until(last_relevant)

    return await sdb.insert_one("account_deletion_requests", {
        "customer_id": customer_id, "requested_by": requested_by, "reason": reason,
        "requested_at": datetime.now(timezone.utc).isoformat(),
        "retention_until": retention_until, "retention_source": RETENTION_SOURCE,
        "status": "pending_retention_period",
    })


async def can_delete_now(deletion_request_id: str, now: Optional[datetime] = None) -> bool:
    """True only once the retention period has actually passed. Never
    returns True based on the customer's wishes alone, however
    reasonable — lawful record-keeping obligations override an
    individual deletion request until the retention period elapses;
    this is the enforcement point for that."""
    now = now or datetime.now(timezone.utc)
    request = await sdb.find_one("account_deletion_requests", {"id": deletion_request_id})
    if not request:
        raise SecurityControlError(f"no deletion request {deletion_request_id}")
    retention_until = datetime.fromisoformat(request["retention_until"])
    if retention_until.tzinfo is None:
        retention_until = retention_until.replace(tzinfo=timezone.utc)
    return now >= retention_until


# ---------------------------------------------------------------
# Data-breach assessment intake
# ---------------------------------------------------------------

BREACH_SEVERITY_LEVELS = ("low", "medium", "high", "critical")


async def record_data_breach_assessment(description: str, affected_data_categories: list, severity: str,
                                         assessed_by: str, notifiable: Optional[bool] = None) -> dict:
    """Records the START of a data-breach assessment (task section 13's
    'data-breach assessment workflow'). `notifiable` (whether this
    triggers a mandatory notification under the Notifiable Data
    Breaches scheme) is left as an explicit Optional[bool], defaulting
    to None ('not yet determined') rather than False — a breach's
    notifiability must be a deliberate human determination, never an
    implicit default that could under-report a breach that should have
    been escalated."""
    if severity not in BREACH_SEVERITY_LEVELS:
        raise SecurityControlError(f"unknown severity: {severity}")
    return await sdb.insert_one("data_breach_assessments", {
        "description": description, "affected_data_categories": list(affected_data_categories),
        "severity": severity, "assessed_by": assessed_by, "notifiable": notifiable,
        "assessed_at": datetime.now(timezone.utc).isoformat(), "status": "under_assessment",
    })
