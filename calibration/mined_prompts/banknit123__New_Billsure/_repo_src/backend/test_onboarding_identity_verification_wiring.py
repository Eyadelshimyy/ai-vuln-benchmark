"""
Test for onboarding.py's identity_verification.py wiring
(start_identity_verification / apply_identity_verification_result).
Same in-memory fake-DB pattern as the other test_*.py files. Exercises
the wiring against identity_verification.py's explicitly-gated mock
path — the real Didit endpoint still can't be reached from this
sandbox's restricted network egress (see identity_verification.py's
module docstring), so this proves the GLUE code is correct, not the
live provider call.

Run: python3 test_onboarding_identity_verification_wiring.py
"""
import asyncio
import importlib
import os
import sys
import types
import uuid
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


fake_sdb = types.SimpleNamespace(find_one=find_one, insert_one=insert_one, update_one=update_one)
sys.modules["supabase_db"] = fake_sdb

import onboarding as ob   # noqa: E402

FAILURES = []


def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}")
    if not condition:
        FAILURES.append(label)


async def main():
    # ---------------------------------------------------------------
    # Fail-closed: no identity_verification provider configured
    # ---------------------------------------------------------------
    for k in ("DIDIT_API_KEY", "DIDIT_WORKFLOW_ID", "ALLOW_MOCK_IDENTITY_VERIFICATION"):
        os.environ.pop(k, None)
    if "identity_verification" in sys.modules:
        importlib.reload(sys.modules["identity_verification"])

    app_row = await insert_one("onboarding_applications", {
        "user_id": "user-1", "identity_verification_status": "pending", "created_at": datetime.now(timezone.utc).isoformat(),
    })

    try:
        await ob.start_identity_verification(app_row["id"])
        check("propagates identity_verification's fail-closed error when no provider/mock is configured", False)
    except Exception as e:
        check("propagates identity_verification's fail-closed error when no provider/mock is configured",
              type(e).__name__ == "IdentityVerificationError")

    unchanged = await find_one("onboarding_applications", {"id": app_row["id"]})
    check("a failed verification start leaves identity_verification_status untouched (still 'pending')",
          unchanged["identity_verification_status"] == "pending")

    # ---------------------------------------------------------------
    # Wiring works end to end against the explicitly-gated mock path
    # ---------------------------------------------------------------
    os.environ["ALLOW_MOCK_IDENTITY_VERIFICATION"] = "true"
    importlib.reload(sys.modules["identity_verification"])

    session_id = await ob.start_identity_verification(app_row["id"])
    check("start_identity_verification returns a VerificationSession with a session id",
          hasattr(session_id, "session_id") and bool(session_id.session_id))

    with_session = await find_one("onboarding_applications", {"id": app_row["id"]})
    check("the session id is persisted on the application row", with_session["identity_verification_session_id"] == session_id.session_id)
    check("identity_verification_status is still 'pending' immediately after starting (result not applied yet)",
          with_session["identity_verification_status"] == "pending")

    updated = await ob.apply_identity_verification_result(app_row["id"])
    check("apply_identity_verification_result moves status to 'verified' from the (mock) provider result",
          updated["identity_verification_status"] == "verified")

    persisted = await find_one("onboarding_applications", {"id": app_row["id"]})
    check("the status change is actually persisted, not just returned", persisted["identity_verification_status"] == "verified")

    # ---------------------------------------------------------------
    # apply_identity_verification_result without a session started first
    # ---------------------------------------------------------------
    no_session_app = await insert_one("onboarding_applications", {
        "user_id": "user-2", "identity_verification_status": "pending", "created_at": datetime.now(timezone.utc).isoformat(),
    })
    try:
        await ob.apply_identity_verification_result(no_session_app["id"])
        check("refuses to apply a result for an application with no session started", False)
    except ob.OnboardingError:
        check("refuses to apply a result for an application with no session started", True)

    os.environ.pop("ALLOW_MOCK_IDENTITY_VERIFICATION", None)
    importlib.reload(sys.modules["identity_verification"])

    # ---------------------------------------------------------------
    # The authoritative webhook path: real HMAC signature, real
    # canonicalisation, idempotent on event_id
    # ---------------------------------------------------------------
    import hashlib
    import hmac
    import json
    import time

    os.environ["DIDIT_WEBHOOK_SECRET"] = "test-secret"
    importlib.reload(sys.modules["identity_verification"])

    webhook_app = await insert_one("onboarding_applications", {
        "user_id": "user-3", "identity_verification_status": "pending", "created_at": datetime.now(timezone.utc).isoformat(),
    })

    def sign(secret, body_dict, ts):
        canonical = json.dumps(body_dict, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        return hmac.new(secret.encode(), canonical.encode(), hashlib.sha256).hexdigest()

    now = str(time.time())
    payload = {
        "event_id": "evt-real-001", "webhook_type": "status.updated", "session_id": "sess-xyz",
        "status": "Approved", "vendor_data": webhook_app["id"], "decision": {}, "metadata": {},
        "timestamp": int(float(now)),
    }
    raw_body = json.dumps(payload, sort_keys=True).encode("utf-8")
    # Sign the EXACT bytes we're about to send, sorted the same way, so
    # this matches what verify_webhook_signature's own canonicalisation
    # produces (its sort is recursive-key, not just top-level, but this
    # payload's nested objects are empty/flat so top-level sort_keys is
    # equivalent here).
    signature = sign("test-secret", payload, now)

    applied = await ob.apply_identity_verification_webhook(raw_body, signature, now)
    check("a correctly-signed webhook updates the correct application", applied is not None and applied["id"] == webhook_app["id"])
    check("Approved webhook moves identity_verification_status to 'verified'", applied["identity_verification_status"] == "verified")

    persisted_webhook = await find_one("onboarding_applications", {"id": webhook_app["id"]})
    check("the webhook-driven status change is actually persisted", persisted_webhook["identity_verification_status"] == "verified")

    # Duplicate delivery (same event_id) is a no-op, not reapplied/errored.
    duplicate_result = await ob.apply_identity_verification_webhook(raw_body, signature, now)
    check("a duplicate webhook delivery (same event_id) is a no-op, returns None", duplicate_result is None)

    # A tampered payload with a stale/mismatched signature is rejected.
    bad_signature = "0" * len(signature)
    try:
        await ob.apply_identity_verification_webhook(raw_body, bad_signature, now)
        check("rejects a webhook with an invalid signature", False)
    except Exception as e:
        check("rejects a webhook with an invalid signature", "signature" in str(e).lower())

    os.environ.pop("DIDIT_WEBHOOK_SECRET", None)
    importlib.reload(sys.modules["identity_verification"])

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
