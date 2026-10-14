"""
backend/identity_verification.py
===================================
Identity verification adapter for `onboarding.identity_verification_status`,
built against Didit's real, confirmed API contract.

Two sessions of work went into getting this contract right:
- Session 6 built a first version guessed from web-search summaries.
- Session 7 corrected the auth header and added `workflow_id` after
  checking Didit's published docs, but still had the response field
  name wrong (`session_url` instead of `url`) and only handled 5 of the
  10 real session-status literals.
- This session (8) had the actual Didit integration reference doc
  pasted in, confirming the full contract precisely — including that
  **webhooks, not polling, are the authoritative source of truth** for
  a verification decision. This version is a full rebuild against that
  confirmed contract, not an incremental patch.

THIS SANDBOX STILL CANNOT REACH verification.didit.me — network egress
here is restricted to package-registry domains. So even now, with a
correct contract, nothing in this module has been called against a live
Didit endpoint. What changed is the CONFIDENCE level: the request/
response shapes and the webhook signature scheme below are copied from
an authoritative integration reference, not inferred from search
snippets. Treat this as "implemented against a verified contract, still
network-untested" — not "implemented and tested" — until it's actually
run from an environment with real internet access.

## Two ways a verification result reaches this module

1. **Webhook (authoritative).** Didit POSTs a signed event to your
   registered webhook destination when a session's status changes.
   `verify_webhook_signature()` + `parse_webhook_event()` below implement
   the X-Signature-V2 HMAC scheme exactly as documented: canonicalise
   (shorten whole-number floats, sort keys recursively, JSON-encode with
   unescaped Unicode) then HMAC-SHA256 with `DIDIT_WEBHOOK_SECRET`,
   constant-time compare. This is the ONLY source this module trusts for
   a final Approved/Declined decision.
2. **Polling (`get_verification_result`, secondary).** `GET /v3/session/
   {id}/decision/` — useful for an admin "check current status" button
   or a backfill, but per Didit's own integration guidance the webhook
   is authoritative; don't build a flow that only polls and never
   registers a webhook.
"""

import hashlib
import hmac
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

DIDIT_BASE_URL = os.environ.get("DIDIT_BASE_URL", "https://verification.didit.me")
DIDIT_API_KEY = os.environ.get("DIDIT_API_KEY", "")
DIDIT_WEBHOOK_SECRET = os.environ.get("DIDIT_WEBHOOK_SECRET", "")
# workflow_id is per-session config, not a secret (per Didit's own
# integration guidance -- "NEVER put workflow_id in .env or treat it as
# a secret"). DIDIT_WORKFLOW_ID is kept as an optional convenience
# default for this deployment's standard KYC workflow; callers can
# still pass a different workflow_id per call for e.g. a KYB flow.
DIDIT_WORKFLOW_ID = os.environ.get("DIDIT_WORKFLOW_ID", "")
ALLOW_MOCK_IDENTITY_VERIFICATION = os.environ.get("ALLOW_MOCK_IDENTITY_VERIFICATION", "false").lower() == "true"

# Exact, case-sensitive literal session statuses per Didit's V3 contract.
SESSION_STATUSES = (
    "Not Started", "In Progress", "Awaiting User", "In Review", "Approved",
    "Declined", "Resubmitted", "Abandoned", "Expired", "Kyc Expired",
)

# Mapped onto onboarding.py's 3-state model. Only an explicit "Approved"
# is ever treated as verified -- every other literal, including the
# ambiguous/in-flight ones, maps to something that keeps the applicant
# blocked from credit activation. "Kyc Expired" (a previously-approved
# user whose verification has aged out) maps to 'failed' rather than
# quietly staying 'verified', since the whole point of that status is
# that the old approval should no longer be trusted.
_STATUS_TO_ONBOARDING = {
    "Approved": "verified",
    "Declined": "failed",
    "Abandoned": "failed",
    "Expired": "failed",
    "Kyc Expired": "failed",
    "Not Started": "pending",
    "In Progress": "pending",
    "Awaiting User": "pending",
    "In Review": "pending",
    "Resubmitted": "pending",
}


class IdentityVerificationError(Exception):
    """Raised for configuration, provider, or signature-verification
    failures. Every raise path here means 'do not proceed' -- nothing in
    this module has a code path that treats a failed check as success."""


@dataclass
class VerificationSession:
    session_id: str
    url: str
    session_token: Optional[str]
    status: str
    provider: str        # 'didit' | 'mock'


@dataclass
class IdentityVerificationResult:
    status: str                  # onboarding.py's pending | verified | failed
    didit_status: str             # the real literal status, kept for audit/debugging
    provider: str
    provider_reference: str
    checked_at: str
    decision: Optional[dict] = None
    raw_response: Optional[dict] = None


@dataclass
class WebhookEvent:
    event_id: str
    webhook_type: str
    session_id: str
    status: str                    # real Didit literal
    onboarding_status: str          # mapped onto pending/verified/failed
    vendor_data: str                 # this is the application_id, by convention (see onboarding.py)
    decision: Optional[dict]
    metadata: Optional[dict]
    timestamp: int


def _onboarding_status(didit_status: str) -> str:
    return _STATUS_TO_ONBOARDING.get(didit_status, "pending")


async def start_verification_session(
    applicant_reference: str,
    workflow_id: Optional[str] = None,
    callback: Optional[str] = None,
    metadata: Optional[dict] = None,
    language: Optional[str] = None,
    contact_details: Optional[dict] = None,
    expected_details: Optional[dict] = None,
) -> VerificationSession:
    """Creates a verification session. `applicant_reference` is passed
    as `vendor_data` -- onboarding.py's convention is to pass the
    application_id itself, so a webhook's `vendor_data` field can be
    used directly to look the application back up (see
    onboarding.apply_identity_verification_webhook()). Fails closed if
    no real provider is configured and mock mode isn't explicitly
    enabled."""
    workflow_id = workflow_id or DIDIT_WORKFLOW_ID
    if DIDIT_API_KEY:
        if not workflow_id:
            raise IdentityVerificationError(
                "no workflow_id provided and DIDIT_WORKFLOW_ID is not configured -- a Didit workflow must exist first "
                "(create one in the console or via POST /v3/workflows/) before a session can be created"
            )
        return await _start_didit_session(applicant_reference, workflow_id, callback, metadata, language, contact_details, expected_details)
    if ALLOW_MOCK_IDENTITY_VERIFICATION:
        logger.warning("ALLOW_MOCK_IDENTITY_VERIFICATION is enabled -- using a MOCK identity verification session, "
                        "not a real provider. This must never be enabled outside local development/testing.")
        session_id = f"mock-session-{uuid.uuid4()}"
        return VerificationSession(session_id=session_id, url=f"https://mock.local/verify/{session_id}",
                                    session_token=None, status="Not Started", provider="mock")
    raise IdentityVerificationError(
        "no identity verification provider is configured (DIDIT_API_KEY unset) and "
        "ALLOW_MOCK_IDENTITY_VERIFICATION is not enabled -- refusing to proceed rather than silently skip verification"
    )


async def _start_didit_session(applicant_reference: str, workflow_id: str, callback: Optional[str],
                                metadata: Optional[dict], language: Optional[str],
                                contact_details: Optional[dict], expected_details: Optional[dict]) -> VerificationSession:
    """Real Didit integration against the confirmed POST /v3/session/
    contract. NOT tested against a live endpoint -- see module docstring
    for why this sandbox can't reach Didit's host."""
    import httpx

    body = {"workflow_id": workflow_id, "vendor_data": applicant_reference}
    if callback:
        body["callback"] = callback
    if metadata:
        body["metadata"] = metadata
    if language:
        body["language"] = language
    if contact_details:
        body["contact_details"] = contact_details
    if expected_details:
        body["expected_details"] = expected_details

    async with httpx.AsyncClient(base_url=DIDIT_BASE_URL, timeout=15.0) as client:
        try:
            response = await client.post(
                "/v3/session/",
                headers={"x-api-key": DIDIT_API_KEY, "Content-Type": "application/json"},
                json=body,
            )
        except httpx.HTTPError as e:
            raise IdentityVerificationError(f"Didit session creation failed: {e}") from e

    if response.status_code == 403:
        # Documented failure shape for a missing/invalid/revoked key:
        # {"detail": "You do not have permission to perform this action."}
        # No machine-readable discriminator between "missing" / "expired"
        # / "wrong app" -- surface the raw detail rather than guessing.
        raise IdentityVerificationError(f"Didit rejected the API key (403): {response.text}")
    try:
        response.raise_for_status()
    except httpx.HTTPError as e:
        raise IdentityVerificationError(f"Didit session creation failed: {e}") from e

    data = response.json()
    session_id = data.get("session_id")
    url = data.get("url")
    if not session_id or not url:
        raise IdentityVerificationError(f"Didit response missing session_id/url: {data}")

    return VerificationSession(
        session_id=session_id, url=url, session_token=data.get("session_token"),
        status=data.get("status", "Not Started"), provider="didit",
    )


async def get_verification_result(session_id: str) -> IdentityVerificationResult:
    """Polls GET /v3/session/{id}/decision/ for the current result.
    Secondary to webhooks -- see module docstring. Same fail-closed
    posture as start_verification_session()."""
    if session_id.startswith("mock-session-"):
        if not ALLOW_MOCK_IDENTITY_VERIFICATION:
            raise IdentityVerificationError(
                "received a mock session id but ALLOW_MOCK_IDENTITY_VERIFICATION is not enabled -- refusing to trust it"
            )
        return _mock_result(session_id)

    if not DIDIT_API_KEY:
        raise IdentityVerificationError("DIDIT_API_KEY unset -- cannot check a real Didit session result")
    return await _get_didit_result(session_id)


def _mock_result(session_id: str) -> IdentityVerificationResult:
    """Deterministic local mock for development/testing ONLY, gated
    behind ALLOW_MOCK_IDENTITY_VERIFICATION. Always returns Approved --
    this exists to exercise onboarding.py's downstream logic without a
    real provider, not to simulate realistic KYC failure modes."""
    return IdentityVerificationResult(
        status="verified", didit_status="Approved", provider="mock", provider_reference=session_id,
        checked_at=datetime.now(timezone.utc).isoformat(),
        raw_response={"note": "MOCK result -- ALLOW_MOCK_IDENTITY_VERIFICATION is enabled, this is not a real check"},
    )


async def _get_didit_result(session_id: str) -> IdentityVerificationResult:
    import httpx

    async with httpx.AsyncClient(base_url=DIDIT_BASE_URL, timeout=15.0) as client:
        try:
            response = await client.get(f"/v3/session/{session_id}/decision/", headers={"x-api-key": DIDIT_API_KEY})
        except httpx.HTTPError as e:
            raise IdentityVerificationError(f"Didit result lookup failed: {e}") from e

    if response.status_code == 403:
        raise IdentityVerificationError(f"Didit rejected the API key (403): {response.text}")
    try:
        response.raise_for_status()
    except httpx.HTTPError as e:
        raise IdentityVerificationError(f"Didit result lookup failed: {e}") from e

    data = response.json()
    didit_status = data.get("status", "Not Started")

    return IdentityVerificationResult(
        status=_onboarding_status(didit_status), didit_status=didit_status, provider="didit",
        provider_reference=session_id, checked_at=datetime.now(timezone.utc).isoformat(),
        decision=data.get("decision"), raw_response=data,
    )


# ---------------------------------------------------------------
# Webhooks -- the authoritative decision source. Exact canonicalisation
# per Didit's documented X-Signature-V2 scheme: shorten whole-number
# floats to ints (recursively), sort object keys (recursively, arrays
# keep their order), JSON-encode with unescaped Unicode and no extra
# whitespace, then HMAC-SHA256 with DIDIT_WEBHOOK_SECRET.
# ---------------------------------------------------------------

def _shorten_floats(value):
    if isinstance(value, list):
        return [_shorten_floats(v) for v in value]
    if isinstance(value, dict):
        return {k: _shorten_floats(v) for k, v in value.items()}
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _sort_keys(value):
    if isinstance(value, list):
        return [_sort_keys(v) for v in value]
    if isinstance(value, dict):
        return {k: _sort_keys(value[k]) for k in sorted(value.keys())}
    return value


def _canonicalize(parsed_body) -> str:
    return json.dumps(_sort_keys(_shorten_floats(parsed_body)), ensure_ascii=False, separators=(",", ":"))


def verify_webhook_signature(raw_body: bytes, signature: str, timestamp: str, secret: Optional[str] = None) -> bool:
    """Returns True only if the signature is valid AND the timestamp is
    within the 300-second freshness window (replay protection). Never
    raises for a bad signature/stale timestamp -- those are normal
    'reject this delivery' outcomes, not errors. DOES raise if no
    webhook secret is configured at all, since silently returning False
    forever in that case would look identical to 'every webhook is
    fraudulent' rather than 'this deployment isn't set up yet' --
    surfacing that distinction matters operationally."""
    secret = secret or DIDIT_WEBHOOK_SECRET
    if not secret:
        raise IdentityVerificationError("DIDIT_WEBHOOK_SECRET is not configured -- cannot verify webhook signatures")

    try:
        ts = float(timestamp)
    except (TypeError, ValueError):
        return False
    if abs(time.time() - ts) > 300:
        return False

    try:
        parsed = json.loads(raw_body)
    except (TypeError, ValueError, json.JSONDecodeError):
        return False

    canonical = _canonicalize(parsed)
    expected = hmac.new(secret.encode("utf-8"), canonical.encode("utf-8"), hashlib.sha256).hexdigest()
    if not signature or len(signature) != len(expected):
        return False
    return hmac.compare_digest(expected, signature)


def parse_webhook_event(raw_body: bytes) -> WebhookEvent:
    """Parses an ALREADY-VERIFIED webhook body into a structured event.
    Callers must call verify_webhook_signature() first and only proceed
    to this on a True result -- this function does not itself verify
    anything, to keep 'verify' and 'parse' as two separately-testable,
    separately-callable steps (mirroring the reference integration's
    explicit ordering: verify, THEN dispatch)."""
    data = json.loads(raw_body)
    status = data.get("status", "Not Started")
    return WebhookEvent(
        event_id=data.get("event_id", ""),
        webhook_type=data.get("webhook_type", ""),
        session_id=data.get("session_id", ""),
        status=status,
        onboarding_status=_onboarding_status(status),
        vendor_data=data.get("vendor_data", ""),
        decision=data.get("decision"),
        metadata=data.get("metadata"),
        timestamp=data.get("timestamp", 0),
    )
