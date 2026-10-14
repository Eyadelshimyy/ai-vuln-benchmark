"""
Standalone tests for security_controls.py. Same in-memory fake-DB
pattern as the other test_*.py files, no live credentials needed.

Run: python3 test_security_controls.py
"""
import asyncio
import sys
import types
import uuid
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


fake_sdb = types.SimpleNamespace(find_one=find_one, find_many=find_many, insert_one=insert_one)
sys.modules["supabase_db"] = fake_sdb

import security_controls as sc   # noqa: E402

FAILURES = []


def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}")
    if not condition:
        FAILURES.append(label)


async def main():
    # ---------------------------------------------------------------
    # RBAC / least privilege
    # ---------------------------------------------------------------
    check("customer role can view own data", sc.has_permission("customer", "view_own_data"))
    check("customer role CANNOT approve a remedy (least privilege)", not sc.has_permission("customer", "approve_remedy"))
    check("compliance_reviewer can approve a remedy", sc.has_permission("compliance_reviewer", "approve_remedy"))
    check("an unknown role has zero permissions (fails closed)", not sc.has_permission("some_made_up_role", "view_own_data"))

    try:
        sc.require_permission("customer", "manage_users")
        check("require_permission raises for a role lacking the permission", False)
    except sc.SecurityControlError:
        check("require_permission raises for a role lacking the permission", True)

    sc.require_permission("admin", "manage_users")  # should not raise
    check("require_permission does not raise when the role has the permission", True)

    # 'admin' does NOT automatically get every permission -- explicit
    # least-privilege check that admin lacks something case_worker has,
    # proving there's no blanket admin-gets-everything shortcut.
    check("admin does not automatically inherit case_worker-specific permissions (no blanket superuser shortcut)",
          not sc.has_permission("admin", "acknowledge_complaint"))

    # ---------------------------------------------------------------
    # MFA gating
    # ---------------------------------------------------------------
    check("admin role requires MFA", sc.requires_mfa("admin"))
    check("compliance_reviewer role requires MFA", sc.requires_mfa("compliance_reviewer"))
    check("customer role does not require MFA", not sc.requires_mfa("customer"))

    try:
        sc.require_mfa_verified("admin", mfa_verified=False)
        check("blocks an admin action when MFA was not verified this session", False)
    except sc.SecurityControlError:
        check("blocks an admin action when MFA was not verified this session", True)

    sc.require_mfa_verified("admin", mfa_verified=True)  # should not raise
    check("allows an admin action once MFA is verified", True)

    sc.require_mfa_verified("customer", mfa_verified=False)  # should not raise -- customer doesn't need MFA
    check("does not require MFA for a role that isn't in MFA_REQUIRED_ROLES", True)

    await sc.record_mfa_verification("admin1", "admin", "totp")
    check("MFA verification is recordable", True)

    # ---------------------------------------------------------------
    # PII-safe logging
    # ---------------------------------------------------------------
    tfn_message = "Processing application for customer, TFN 123 456 789 on file"
    redacted_tfn = sc.redact_sensitive(tfn_message)
    check("redact_sensitive strips a TFN-shaped number from a free-text message", "123 456 789" not in redacted_tfn and "[REDACTED-ID]" in redacted_tfn)

    bank_message = "Bank details confirmed: BSB 063-000 account 12345678"
    redacted_bank = sc.redact_sensitive(bank_message)
    check("redact_sensitive strips a BSB-account combination", "12345678" not in redacted_bank)

    credential_message = "Retrying with api_key=sk_live_abcdef123456 after timeout"
    redacted_cred = sc.redact_sensitive(credential_message)
    check("redact_sensitive strips a credential-shaped value", "sk_live_abcdef123456" not in redacted_cred and "[REDACTED-CREDENTIAL]" in redacted_cred)

    clean_message = "Bill payment of $150.00 processed for bill bill-123"
    check("redact_sensitive leaves ordinary, non-sensitive log messages unchanged", sc.redact_sensitive(clean_message) == clean_message)

    # PiiRedactingLogFilter integration: attach and confirm it mutates the record.
    import logging
    import io
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.addFilter(sc.PiiRedactingLogFilter())
    test_logger = logging.getLogger("test_pii_redaction_logger")
    test_logger.addHandler(handler)
    test_logger.setLevel(logging.INFO)
    test_logger.info("customer TFN 987 654 321 submitted")
    stream.flush()
    check("PiiRedactingLogFilter actually redacts a real emitted log record", "987 654 321" not in stream.getvalue())

    # ---------------------------------------------------------------
    # File upload validation
    # ---------------------------------------------------------------
    real_pdf_bytes = b"%PDF-1.4\n%fake pdf content for testing"
    result = sc.validate_file_upload("bill.pdf", real_pdf_bytes)
    check("a real PDF with correct magic bytes and extension validates", result.extension == ".pdf")
    check("a freshly-validated upload has malware_scan_status='pending', never 'clean'", result.malware_scan_status == "pending")

    try:
        sc.validate_file_upload("empty.pdf", b"")
        check("rejects an empty file", False)
    except sc.SecurityControlError:
        check("rejects an empty file", True)

    try:
        sc.validate_file_upload("bill.exe", real_pdf_bytes)
        check("rejects a disallowed file extension", False)
    except sc.SecurityControlError:
        check("rejects a disallowed file extension", True)

    spoofed = b"MZ\x90\x00this is actually a Windows executable renamed to .pdf"
    try:
        sc.validate_file_upload("bill.pdf", spoofed)
        check("rejects a file whose content doesn't match its claimed extension (spoofing detection)", False)
    except sc.SecurityControlError:
        check("rejects a file whose content doesn't match its claimed extension (spoofing detection)", True)

    oversized = b"%PDF-1.4\n" + (b"0" * (sc.MAX_UPLOAD_SIZE_BYTES + 1))
    try:
        sc.validate_file_upload("huge.pdf", oversized)
        check("rejects a file exceeding the maximum size", False)
    except sc.SecurityControlError:
        check("rejects a file exceeding the maximum size", True)

    real_png_bytes = b"\x89PNG\r\n\x1a\n" + b"fake png data"
    png_result = sc.validate_file_upload("photo.png", real_png_bytes)
    check("a real PNG validates correctly", png_result.extension == ".png")

    # ---------------------------------------------------------------
    # Data retention / account deletion
    # ---------------------------------------------------------------
    check("MINIMUM_RETENTION_YEARS is sourced at 7 years (AML/CTF Act 2006 (Cth) / ASIC AFSL expectations)", sc.MINIMUM_RETENTION_YEARS == 7)

    last_transaction = datetime(2026, 1, 1, tzinfo=timezone.utc)
    deletion_request = await sc.request_account_deletion("cust-1", requested_by="cust-1", reason="closing account",
                                                            last_transaction_date=last_transaction)
    check("deletion request is NOT immediately deletable", deletion_request["status"] == "pending_retention_period")
    retention_until = datetime.fromisoformat(deletion_request["retention_until"])
    check("retention_until is exactly 7 years after the last relevant transaction", retention_until.year == 2033)

    not_yet = await sc.can_delete_now(deletion_request["id"], now=datetime(2030, 1, 1, tzinfo=timezone.utc))
    check("cannot delete before the retention period has elapsed, even years later", not_yet is False)

    now_ok = await sc.can_delete_now(deletion_request["id"], now=datetime(2033, 6, 1, tzinfo=timezone.utc))
    check("can delete once the retention period has genuinely elapsed", now_ok is True)

    # ---------------------------------------------------------------
    # Data breach assessment
    # ---------------------------------------------------------------
    try:
        await sc.record_data_breach_assessment("test", [], "not_a_real_severity", assessed_by="security1")
        check("rejects an unknown breach severity", False)
    except sc.SecurityControlError:
        check("rejects an unknown breach severity", True)

    breach = await sc.record_data_breach_assessment(
        "Suspected unauthorised access to admin panel", ["credit_assessment_data"], "high", assessed_by="security1")
    check("a data breach assessment defaults notifiable to None (undetermined), never False by default", breach["notifiable"] is None)
    check("breach assessment starts in 'under_assessment' status", breach["status"] == "under_assessment")

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
