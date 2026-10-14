"""
backend/pilot_auth.py
=======================
API-key authentication for `pilot_api.py`. This is deliberately a
simple, operator-issued API key scheme — not a full username/password/
session login system — because `pilot_api.py` is a private testing
deployment for pilot staff and internal systems right now (per its own
docstring and the deployment guide's recommendation to keep it behind
an unpublicised subdomain), not a public consumer-facing product yet.
Building real customer self-service authentication (signup, password
reset, magic links, etc.) is a separate, later piece of work — this
module solves "who is allowed to call this API at all" for the current
testing phase, using the RBAC roles already defined and tested in
`security_controls.py`.

Security properties:
- The raw API key is returned to the caller of `issue_api_key()`
  EXACTLY ONCE, at issuance time, and is never stored or logged in
  plaintext anywhere after that — only its SHA-256 hash is persisted.
  This mirrors how real API-key systems (Stripe, GitHub, etc.) work:
  losing the raw key means generating a new one, not "looking it up
  again."
- `verify_api_key()` looks up by hash and refuses a revoked key even if
  the hash still matches — revocation is permanent, not a soft flag
  callers could accidentally bypass.
- A key issued for an `MFA_REQUIRED_ROLE` (admin, compliance_reviewer —
  see `security_controls.MFA_REQUIRED_ROLES`) carries its own
  `mfa_verified` flag, set at issuance time by whoever confirmed the
  operator's identity out-of-band (no real MFA provider is integrated
  in this codebase — see `security_controls.py`'s own documented gap).
  A key without `mfa_verified=True` for an MFA-required role will be
  rejected at request time by `pilot_api.py`'s auth dependency, via
  `security_controls.require_mfa_verified()`.
"""

import hashlib
import logging
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import supabase_db as sdb
import security_controls as sc

logger = logging.getLogger(__name__)


class PilotAuthError(Exception):
    """Raised for any invalid auth operation. Every raise path here
    means 'this caller is not authenticated/authorized' — there is no
    code path that treats an invalid or revoked key as valid."""


@dataclass
class IssuedApiKey:
    raw_key: str          # shown to the caller ONCE; never persisted or logged
    key_id: str
    actor_id: str
    role: str


def _hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


async def issue_api_key(actor_id: str, role: str, issued_by: str, mfa_verified: bool = False, notes: str = "") -> IssuedApiKey:
    """Issues a new API key for `actor_id` with `role`. Only the hash
    is stored — the raw key is returned once and the caller is
    responsible for delivering it to the actual operator securely
    (this function does not email/message it anywhere). Refuses an
    unknown role outright rather than issuing a key that would fail
    every permission check later."""
    if role not in sc.ROLES:
        raise PilotAuthError(f"unknown role: {role}")

    raw_key = f"bsp_{secrets.token_urlsafe(32)}"  # 'bsp' = BillSure Pilot, a recognisable prefix, not a secret itself
    row = await sdb.insert_one("pilot_api_keys", {
        "key_hash": _hash_key(raw_key), "actor_id": actor_id, "role": role,
        "mfa_verified": mfa_verified, "issued_by": issued_by, "notes": notes,
        "issued_at": datetime.now(timezone.utc).isoformat(), "revoked": False,
        "revoked_by": None, "revoked_at": None,
    })
    return IssuedApiKey(raw_key=raw_key, key_id=row["id"], actor_id=actor_id, role=role)


async def verify_api_key(raw_key: str) -> Optional[dict]:
    """Returns the key record (actor_id, role, mfa_verified) if valid
    and not revoked, else None. Never raises on an invalid key — an
    invalid key is a normal 'not authenticated' outcome for the caller
    (pilot_api.py's dependency) to turn into a 401, not an exception to
    propagate as a 500."""
    if not raw_key:
        return None
    record = await sdb.find_one("pilot_api_keys", {"key_hash": _hash_key(raw_key)})
    if not record:
        return None
    if record.get("revoked"):
        return None
    return record


async def revoke_api_key(key_id: str, revoked_by: str, reason: str) -> dict:
    if not reason or not reason.strip():
        raise PilotAuthError("revoking an API key requires a documented reason")
    existing = await sdb.find_one("pilot_api_keys", {"id": key_id})
    if not existing:
        raise PilotAuthError(f"no API key {key_id}")
    updates = {"revoked": True, "revoked_by": revoked_by, "revoked_reason": reason,
               "revoked_at": datetime.now(timezone.utc).isoformat()}
    await sdb.update_one("pilot_api_keys", {"id": key_id}, updates)
    return {**existing, **updates}
