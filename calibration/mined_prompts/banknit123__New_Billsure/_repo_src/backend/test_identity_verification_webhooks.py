"""
Test for identity_verification.py's webhook verification/parsing and
the full 10-literal status mapping. Builds real webhook payloads,
signs them with a real HMAC exactly the way this module's own
verify_webhook_signature() does, and confirms the round trip — this is
genuinely testable without network access, unlike the live API calls,
since it's pure local cryptography and parsing.

Run: python3 test_identity_verification_webhooks.py
"""
import hashlib
import hmac
import importlib
import json
import os
import sys
import time

FAILURES = []


def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}")
    if not condition:
        FAILURES.append(label)


def _reload_with_env(**env):
    for k in ("DIDIT_API_KEY", "DIDIT_WEBHOOK_SECRET", "DIDIT_WORKFLOW_ID", "ALLOW_MOCK_IDENTITY_VERIFICATION"):
        os.environ.pop(k, None)
    os.environ.update(env)
    if "identity_verification" in sys.modules:
        return importlib.reload(sys.modules["identity_verification"])
    import identity_verification
    return identity_verification


def _sign(secret: str, body: dict, timestamp: str) -> str:
    """Independent reference implementation of Didit's canonicalisation,
    written separately from identity_verification.py's own
    _canonicalize()/_shorten_floats()/_sort_keys() so this test isn't
    just checking the module against itself."""
    def shorten(v):
        if isinstance(v, list):
            return [shorten(x) for x in v]
        if isinstance(v, dict):
            return {k: shorten(x) for k, x in v.items()}
        if isinstance(v, float) and v.is_integer():
            return int(v)
        return v

    def sort_keys(v):
        if isinstance(v, list):
            return [sort_keys(x) for x in v]
        if isinstance(v, dict):
            return {k: sort_keys(v[k]) for k in sorted(v.keys())}
        return v

    canonical = json.dumps(sort_keys(shorten(body)), ensure_ascii=False, separators=(",", ":"))
    return hmac.new(secret.encode("utf-8"), canonical.encode("utf-8"), hashlib.sha256).hexdigest()


def main():
    idv = _reload_with_env(DIDIT_WEBHOOK_SECRET="test-webhook-secret-value")

    now = str(time.time())
    payload = {
        "event_id": "evt-123",
        "webhook_type": "status.updated",
        "session_id": "sess-abc",
        "status": "Approved",
        "vendor_data": "application-42",
        "decision": {"id_verifications": [{"status": "Approved", "score": 99.0}]},
        "metadata": {"plan": "premium"},
        "timestamp": int(float(now)),
    }
    raw_body = json.dumps(payload).encode("utf-8")
    correct_signature = _sign("test-webhook-secret-value", payload, now)

    # ---------------------------------------------------------------
    # Signature verification: real HMAC round trip
    # ---------------------------------------------------------------
    check("a correctly-signed, fresh webhook verifies successfully",
          idv.verify_webhook_signature(raw_body, correct_signature, now))

    check("a tampered body fails verification even with the 'correct' signature",
          not idv.verify_webhook_signature(json.dumps({**payload, "status": "Declined"}).encode(), correct_signature, now))

    wrong_secret_sig = _sign("a-different-secret-entirely", payload, now)
    check("a signature computed with the wrong secret fails verification",
          not idv.verify_webhook_signature(raw_body, wrong_secret_sig, now))

    stale_timestamp = str(time.time() - 400)  # >300s old
    stale_signature = _sign("test-webhook-secret-value", payload, stale_timestamp)
    check("a stale timestamp (>300s old) fails verification even with a correct signature",
          not idv.verify_webhook_signature(raw_body, stale_signature, stale_timestamp))

    check("an empty/missing signature fails verification", not idv.verify_webhook_signature(raw_body, "", now))
    check("garbage body fails verification instead of raising", not idv.verify_webhook_signature(b"not json at all", "somesig", now))

    # ---------------------------------------------------------------
    # No webhook secret configured -> raises, doesn't silently accept
    # or silently reject-forever without saying why
    # ---------------------------------------------------------------
    idv_no_secret = _reload_with_env()
    try:
        idv_no_secret.verify_webhook_signature(raw_body, correct_signature, now)
        check("raises when DIDIT_WEBHOOK_SECRET is not configured, rather than silently rejecting", False)
    except idv_no_secret.IdentityVerificationError:
        check("raises when DIDIT_WEBHOOK_SECRET is not configured, rather than silently rejecting", True)

    # ---------------------------------------------------------------
    # Event parsing + the full 10-literal status mapping
    # ---------------------------------------------------------------
    idv2 = _reload_with_env(DIDIT_WEBHOOK_SECRET="test-webhook-secret-value")
    event = idv2.parse_webhook_event(raw_body)
    check("parsed event carries the correct event_id", event.event_id == "evt-123")
    check("parsed event carries the correct vendor_data (application id)", event.vendor_data == "application-42")
    check("Approved maps to onboarding_status='verified'", event.onboarding_status == "verified")

    expected_map = {
        "Not Started": "pending", "In Progress": "pending", "Awaiting User": "pending",
        "In Review": "pending", "Approved": "verified", "Declined": "failed",
        "Resubmitted": "pending", "Abandoned": "failed", "Expired": "failed", "Kyc Expired": "failed",
    }
    all_correct = True
    for status, expected in expected_map.items():
        p = {**payload, "status": status}
        e = idv2.parse_webhook_event(json.dumps(p).encode("utf-8"))
        if e.onboarding_status != expected:
            all_correct = False
            print(f"    mismatch: {status} -> {e.onboarding_status}, expected {expected}")
    check("all 10 literal Didit statuses map to the correct onboarding pending/verified/failed value", all_correct)

    check("every literal in identity_verification.SESSION_STATUSES has a mapping entry",
          all(s in expected_map for s in idv2.SESSION_STATUSES) and len(idv2.SESSION_STATUSES) == 10)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED:")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
