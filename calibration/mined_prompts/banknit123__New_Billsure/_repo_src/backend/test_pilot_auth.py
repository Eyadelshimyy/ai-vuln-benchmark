"""
Standalone tests for pilot_auth.py. Same in-memory fake-DB pattern as
the other test_*.py files, no live credentials needed.

Run: python3 test_pilot_auth.py
"""
import asyncio
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

import pilot_auth as pa   # noqa: E402

FAILURES = []


def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}")
    if not condition:
        FAILURES.append(label)


async def main():
    # ---------------------------------------------------------------
    # Issuance
    # ---------------------------------------------------------------
    try:
        await pa.issue_api_key("actor-1", "not_a_real_role", issued_by="admin1")
        check("rejects issuing a key for an unknown role", False)
    except pa.PilotAuthError:
        check("rejects issuing a key for an unknown role", True)

    issued = await pa.issue_api_key("customer-jane", "customer", issued_by="ops_lead")
    check("issued key has the 'bsp_' prefix", issued.raw_key.startswith("bsp_"))
    check("issued key is reasonably long (not a guessable short string)", len(issued.raw_key) > 30)

    # ---------------------------------------------------------------
    # Storage never contains the raw key
    # ---------------------------------------------------------------
    stored = await find_one("pilot_api_keys", {"id": issued.key_id})
    check("the stored record does NOT contain the raw key anywhere", issued.raw_key not in str(stored))
    check("the stored record contains a hash, not the plaintext key", stored["key_hash"] != issued.raw_key)

    # ---------------------------------------------------------------
    # Verification
    # ---------------------------------------------------------------
    verified = await pa.verify_api_key(issued.raw_key)
    check("a freshly-issued key verifies successfully", verified is not None and verified["actor_id"] == "customer-jane")
    check("verified key carries the correct role", verified["role"] == "customer")

    check("an empty key verifies to None, not an error", await pa.verify_api_key("") is None)
    check("a made-up/garbage key verifies to None", await pa.verify_api_key("bsp_totally_made_up_garbage") is None)
    check("a key with the right prefix but wrong random part still fails (not fooled by the prefix alone)",
          await pa.verify_api_key("bsp_" + issued.raw_key[4:-1] + "X") is None)

    # ---------------------------------------------------------------
    # Revocation is permanent
    # ---------------------------------------------------------------
    try:
        await pa.revoke_api_key(issued.key_id, revoked_by="admin1", reason="")
        check("rejects revocation with no documented reason", False)
    except pa.PilotAuthError:
        check("rejects revocation with no documented reason", True)

    await pa.revoke_api_key(issued.key_id, revoked_by="admin1", reason="customer requested account closure")
    revoked_check = await pa.verify_api_key(issued.raw_key)
    check("a revoked key no longer verifies, even though the hash still matches exactly", revoked_check is None)

    # ---------------------------------------------------------------
    # MFA-required roles carry the mfa_verified flag correctly
    # ---------------------------------------------------------------
    admin_key_no_mfa = await pa.issue_api_key("admin-bob", "admin", issued_by="ops_lead", mfa_verified=False)
    admin_verified_no_mfa = await pa.verify_api_key(admin_key_no_mfa.raw_key)
    check("an admin key issued without MFA confirmation carries mfa_verified=False", admin_verified_no_mfa["mfa_verified"] is False)

    admin_key_with_mfa = await pa.issue_api_key("admin-carol", "admin", issued_by="ops_lead", mfa_verified=True,
                                                  notes="MFA confirmed via 1:1 video call before issuance")
    admin_verified_with_mfa = await pa.verify_api_key(admin_key_with_mfa.raw_key)
    check("an admin key issued after MFA confirmation carries mfa_verified=True", admin_verified_with_mfa["mfa_verified"] is True)

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
