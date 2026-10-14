"""
backend/bank_verification.py
==============================
Bank account verification adapter for `onboarding.bank_account_verified`.

Real provider (sandbox stage): Open Bank Project's free public sandbox
(https://apisandbox.openbankproject.com) — chosen because it's free with
no accreditation step to START building against, unlike real Australian
CDR data which requires ACCC accreditation regardless of vendor. This is
explicitly a SANDBOX-STAGE integration, not a path to real bank-verified
customers — see the module-level warning below and
docs/asic-ers-readiness/external-dependencies.md.

THIS SESSION DID NOT HAVE OPEN BANK PROJECT SANDBOX CREDENTIALS. Same
posture as identity_verification.py: implements the real integration
shape, fails closed with no configuration, and provides an explicitly
gated mock path for local development. The real-endpoint code path is
implemented but unverified against a live sandbox account.

IMPORTANT — this does not become a path to real CDR-accredited bank
verification just by swapping in production credentials. Real Australian
Open Banking data requires ACCC accreditation as a Data Recipient (or
sponsorship through an accredited intermediary) — a regulatory process,
not a config value. Treat OBP_API_KEY as sandbox-only; a real pilot
needs a genuinely accredited pathway, tracked as an external dependency.
"""

import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

OBP_BASE_URL = os.environ.get("OBP_BASE_URL", "https://apisandbox.openbankproject.com")
OBP_API_KEY = os.environ.get("OBP_API_KEY", "")
ALLOW_MOCK_BANK_VERIFICATION = os.environ.get("ALLOW_MOCK_BANK_VERIFICATION", "false").lower() == "true"


class BankVerificationError(Exception):
    """Raised for configuration or provider failures. Every raise path
    means 'do not treat this account as verified'."""


@dataclass
class BankVerificationResult:
    verified: bool
    provider: str                  # 'open_bank_project_sandbox' | 'mock'
    account_reference: str
    checked_at: str
    raw_response: Optional[dict] = None


async def verify_account(bsb: str, account_number: str, account_holder_name: str) -> BankVerificationResult:
    """Fails closed if no real provider is configured and mock mode
    isn't explicitly enabled — mirrors identity_verification.py exactly,
    same reasoning: silently treating an unverified account as verified
    is the one outcome this function must never produce by accident."""
    if not bsb or not account_number:
        raise BankVerificationError("bsb and account_number are both required")

    if OBP_API_KEY:
        return await _verify_via_open_bank_project(bsb, account_number, account_holder_name)
    if ALLOW_MOCK_BANK_VERIFICATION:
        logger.warning("ALLOW_MOCK_BANK_VERIFICATION is enabled — using a MOCK bank verification result, "
                        "not a real provider. This must never be enabled outside local development/testing.")
        return BankVerificationResult(
            verified=True, provider="mock", account_reference=f"mock-{uuid.uuid4()}",
            checked_at=datetime.now(timezone.utc).isoformat(),
            raw_response={"note": "MOCK result — ALLOW_MOCK_BANK_VERIFICATION is enabled, this is not a real check"},
        )
    raise BankVerificationError(
        "no bank verification provider is configured (OBP_API_KEY unset) and "
        "ALLOW_MOCK_BANK_VERIFICATION is not enabled — refusing to proceed rather than silently skip verification"
    )


async def _verify_via_open_bank_project(bsb: str, account_number: str, account_holder_name: str) -> BankVerificationResult:
    """Real Open Bank Project sandbox integration. NOT tested against a
    live sandbox account in this session (no credentials available) —
    implemented against OBP's documented account-lookup contract; verify
    against a real sandbox account before relying on this.

    NOTE: this queries the sandbox's static test dataset, which will
    never contain a real customer's real BSB/account combination — this
    function is only meaningful as a proof-of-integration exercise
    against OBP's test accounts, not as a way to verify a real pilot
    customer's real bank account. Real verification needs an
    accredited CDR pathway (see module docstring)."""
    import httpx

    async with httpx.AsyncClient(base_url=OBP_BASE_URL, timeout=15.0) as client:
        try:
            response = await client.get(
                f"/obp/v5.1.0/banks/AU_SANDBOX/accounts/{bsb}-{account_number}/account",
                headers={"Authorization": f"DirectLogin token={OBP_API_KEY}"},
            )
        except httpx.HTTPError as e:
            raise BankVerificationError(f"Open Bank Project lookup failed: {e}") from e

    if response.status_code == 404:
        return BankVerificationResult(
            verified=False, provider="open_bank_project_sandbox", account_reference=f"{bsb}-{account_number}",
            checked_at=datetime.now(timezone.utc).isoformat(), raw_response={"status_code": 404},
        )
    try:
        response.raise_for_status()
    except httpx.HTTPError as e:
        raise BankVerificationError(f"Open Bank Project lookup failed: {e}") from e

    data = response.json()
    holder_matches = account_holder_name.strip().lower() in str(data.get("owners", "")).lower()

    return BankVerificationResult(
        verified=holder_matches, provider="open_bank_project_sandbox", account_reference=f"{bsb}-{account_number}",
        checked_at=datetime.now(timezone.utc).isoformat(), raw_response=data,
    )
