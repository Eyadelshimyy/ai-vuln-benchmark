"""
backend/issue_pilot_api_key.py
=================================
Command-line tool for an operator to issue a pilot API key. Deliberately
NOT an HTTP endpoint — exposing "create an API key" over HTTP creates a
chicken-and-egg authentication problem (what credential authorizes the
call that creates the first credential?) that a CLI run by someone with
direct database/deployment access sidesteps entirely. This is how the
first admin key gets created; after that, an admin could reasonably
build an authenticated HTTP endpoint for issuing further keys — that
isn't built here.

Usage (run from wherever SUPABASE_URL/SUPABASE_SERVICE_KEY are set,
e.g. locally with the pilot sandbox project's credentials, or via
Render's shell):

    python3 issue_pilot_api_key.py --actor-id admin_bob --role admin \\
        --issued-by ops_lead --mfa-verified --notes "confirmed via video call"

Prints the raw key ONCE. It is not recoverable afterward — only its
hash is stored (see pilot_auth.py). Save it somewhere secure
immediately (a password manager, not a chat message or a file
committed to git).
"""
import argparse
import asyncio

import pilot_auth as pa
import security_controls as sc


async def main():
    parser = argparse.ArgumentParser(description="Issue a pilot API key")
    parser.add_argument("--actor-id", required=True, help="stable identifier for the person/system this key represents")
    parser.add_argument("--role", required=True, choices=sc.ROLES, help="role to grant")
    parser.add_argument("--issued-by", required=True, help="who is issuing this key (for audit)")
    parser.add_argument("--mfa-verified", action="store_true",
                         help="set only after genuinely confirming this operator's identity out-of-band "
                              "(no real MFA provider is integrated -- see security_controls.py)")
    parser.add_argument("--notes", default="", help="free-text note, e.g. how MFA was confirmed")
    args = parser.parse_args()

    if args.role in sc.MFA_REQUIRED_ROLES and not args.mfa_verified:
        print(f"WARNING: role '{args.role}' requires MFA (security_controls.MFA_REQUIRED_ROLES). "
              f"This key will be rejected by pilot_api.py for any privileged action until reissued "
              f"with --mfa-verified after genuinely confirming the operator's identity.")

    issued = await pa.issue_api_key(args.actor_id, args.role, issued_by=args.issued_by,
                                     mfa_verified=args.mfa_verified, notes=args.notes)
    print(f"\nIssued API key for actor_id='{issued.actor_id}' role='{issued.role}':\n")
    print(f"  {issued.raw_key}\n")
    print("This is shown ONCE. Save it now -- it cannot be retrieved again (only its hash is stored).")


if __name__ == "__main__":
    asyncio.run(main())
