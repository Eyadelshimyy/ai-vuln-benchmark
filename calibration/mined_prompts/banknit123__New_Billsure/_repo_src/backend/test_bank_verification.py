"""
Test for bank_verification.py. Same structure as
test_identity_verification.py — exercises the fail-closed default and
the explicitly gated mock path. Does NOT call the real Open Bank Project
sandbox (no credentials in this environment).

Run: python3 test_bank_verification.py
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
    for k in ("OBP_API_KEY", "OBP_BASE_URL", "ALLOW_MOCK_BANK_VERIFICATION"):
        os.environ.pop(k, None)
    os.environ.update(env)
    if "bank_verification" in sys.modules:
        return importlib.reload(sys.modules["bank_verification"])
    import bank_verification
    return bank_verification


async def main():
    # ---------------------------------------------------------------
    # Fail-closed default
    # ---------------------------------------------------------------
    bvmod = _reload_with_env()
    check("no OBP_API_KEY and mock disabled by default", not bvmod.OBP_API_KEY and not bvmod.ALLOW_MOCK_BANK_VERIFICATION)

    try:
        await bvmod.verify_account("063000", "12345678", "Jane Citizen")
        check("refuses to verify a bank account with no provider configured and mock disabled", False)
    except bvmod.BankVerificationError:
        check("refuses to verify a bank account with no provider configured and mock disabled", True)

    # ---------------------------------------------------------------
    # Input validation independent of provider configuration
    # ---------------------------------------------------------------
    bvmod2 = _reload_with_env(ALLOW_MOCK_BANK_VERIFICATION="true")
    try:
        await bvmod2.verify_account("", "12345678", "Jane Citizen")
        check("rejects a missing bsb even with mock mode enabled", False)
    except bvmod2.BankVerificationError:
        check("rejects a missing bsb even with mock mode enabled", True)

    # ---------------------------------------------------------------
    # Mock mode must be explicitly enabled
    # ---------------------------------------------------------------
    result = await bvmod2.verify_account("063000", "12345678", "Jane Citizen")
    check("mock verification succeeds once ALLOW_MOCK_BANK_VERIFICATION=true is explicitly set", result.verified is True)
    check("mock result is honestly labelled provider='mock'", result.provider == "mock")
    check("mock result's raw_response makes clear this is not a real check", "MOCK" in result.raw_response.get("note", ""))

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
