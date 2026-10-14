"""
Test for identity_verification.py. Exercises the fail-closed default
(no provider configured, mock not enabled -> refuses) and the explicitly
gated mock path. Does NOT call the real Didit API (no credentials
available in this environment) — that code path is implemented but
untested here; see the module docstring.

Run: python3 test_identity_verification.py
"""
import asyncio
import importlib
import os
import sys

FAILURES = []


def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}")
    if not condition:
        FAILURES.append(label)


def _reload_with_env(**env):
    """Set/clear env vars, then reload the module so its module-level
    DIDIT_API_KEY / ALLOW_MOCK_IDENTITY_VERIFICATION constants pick up
    the change — those are read once at import time, matching how the
    rest of this codebase (e.g. ALLOW_MOCK_PAYMENTS in server.py) reads
    feature flags."""
    for k in ("DIDIT_API_KEY", "DIDIT_BASE_URL", "DIDIT_WORKFLOW_ID", "DIDIT_WEBHOOK_SECRET", "ALLOW_MOCK_IDENTITY_VERIFICATION"):
        os.environ.pop(k, None)
    os.environ.update(env)
    if "identity_verification" in sys.modules:
        return importlib.reload(sys.modules["identity_verification"])
    import identity_verification
    return identity_verification


async def main():
    # ---------------------------------------------------------------
    # Fail-closed default: no provider configured, mock not enabled
    # ---------------------------------------------------------------
    iv = _reload_with_env()
    check("no DIDIT_API_KEY and mock disabled by default", not iv.DIDIT_API_KEY and not iv.ALLOW_MOCK_IDENTITY_VERIFICATION)

    try:
        await iv.start_verification_session("applicant-1")
        check("refuses to start a verification session with no provider configured and mock disabled", False)
    except iv.IdentityVerificationError:
        check("refuses to start a verification session with no provider configured and mock disabled", True)

    # ---------------------------------------------------------------
    # Mock mode must be explicitly enabled — never a default
    # ---------------------------------------------------------------
    iv2 = _reload_with_env(ALLOW_MOCK_IDENTITY_VERIFICATION="true")
    session = await iv2.start_verification_session("applicant-2")
    check("mock session starts once ALLOW_MOCK_IDENTITY_VERIFICATION=true is explicitly set",
          session.session_id.startswith("mock-session-") and session.provider == "mock")

    result = await iv2.get_verification_result(session.session_id)
    check("mock verification result reports status='verified'", result.status == "verified")
    check("mock verification result is honestly labelled provider='mock', not 'didit'", result.provider == "mock")
    check("mock result's raw_response makes clear this is not a real check", "MOCK" in result.raw_response.get("note", ""))

    # ---------------------------------------------------------------
    # A mock session id is refused if mock mode gets disabled in between
    # (e.g. config drift between starting and checking a session)
    # ---------------------------------------------------------------
    iv3 = _reload_with_env()  # mock disabled again, no key
    try:
        await iv3.get_verification_result(session.session_id)
        check("refuses to trust a mock session id if mock mode has since been disabled", False)
    except iv3.IdentityVerificationError:
        check("refuses to trust a mock session id if mock mode has since been disabled", True)

    # ---------------------------------------------------------------
    # A real (non-mock) session id with no API key configured is refused
    # ---------------------------------------------------------------
    try:
        await iv3.get_verification_result("some-real-looking-session-id")
        check("refuses to check a real-looking session id with no DIDIT_API_KEY configured", False)
    except iv3.IdentityVerificationError:
        check("refuses to check a real-looking session id with no DIDIT_API_KEY configured", True)

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
