# Disclosure dossier: _write_bytes (CWE-22)

- repo: ProtossBlaster/leapmotor-mate  (https://github.com/ProtossBlaster/leapmotor-mate.git)
- file: web/session_share.py
- tainted parameter: path
- Tier-2 sink: open
- Tier-2 exploit proof: traversal write escaped to /tmp/tier2_proof_pz5181an/pwned_d09444654167

## Reachability (what makes this disclosable)
- reachable from a real entry point: **True**
- entry-point type: **http**
- call path to verify by hand:

    refresh_now -> get_fresh_signals -> _connect -> ensure_account_cert -> _write_bytes

- dynamic reproduce verdict: NEEDS_MANUAL

## Manual verification checklist
1. Open the entry point (`refresh_now`) and confirm it takes external/attacker input.
2. Follow the call path above and confirm the input reaches `_write_bytes` without being sanitized.
3. Reproduce the exploit (payload below) through that entry point against a local run.
4. Only then prepare a responsible-disclosure report.

- payload: `/tmp/tier2_proof_pz5181an/pwned_d09444654167`

NOTE: reachability here is a NAME-BASED static call graph (conservative, may over-connect). Step 1-3 above are the human confirmation that closes that gap before any disclosure.
