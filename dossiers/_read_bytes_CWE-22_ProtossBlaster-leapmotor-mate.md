# Disclosure dossier: _read_bytes (CWE-22)

- repo: ProtossBlaster/leapmotor-mate  (https://github.com/ProtossBlaster/leapmotor-mate.git)
- file: web/session_share.py
- tainted parameter: path
- Tier-2 sink: open
- Tier-2 exploit proof: attacker-controlled input opened our out-of-tree planted file /tmp/tier2_proof_eje5knp8/secret_17ecae1f1bee.txt (proven by live execution trace; read-traversal, no output echo required)

## Reachability (what makes this disclosable)
- reachable from a real entry point: **True**
- entry-point type: **api**
- call path to verify by hand:

    test_switching_the_schedule_off_shows_at_once -> _save -> _read_bytes

- dynamic reproduce verdict: NEEDS_MANUAL

## Manual verification checklist
1. Open the entry point (`test_switching_the_schedule_off_shows_at_once`) and confirm it takes external/attacker input.
2. Follow the call path above and confirm the input reaches `_read_bytes` without being sanitized.
3. Reproduce the exploit (payload below) through that entry point against a local run.
4. Only then prepare a responsible-disclosure report.

- payload: `/tmp/tier2_proof_eje5knp8/secret_17ecae1f1bee.txt`

NOTE: reachability here is a NAME-BASED static call graph (conservative, may over-connect). Step 1-3 above are the human confirmation that closes that gap before any disclosure.
