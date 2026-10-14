# Disclosure dossier: tool_write (CWE-22)

- repo: loveRhythm1990/mini-pi  (https://github.com/loveRhythm1990/mini-pi.git)
- file: 03-tools/src/tools.py
- tainted parameter: path
- Tier-2 sink: open
- Tier-2 exploit proof: traversal write escaped to /tmp/tier2_proof_eyy529bc/pwned_b3c8d0753e66

## Reachability (what makes this disclosable)
- reachable from a real entry point: **True**
- entry-point type: **api**
- call path to verify by hand:

    tool_write

- dynamic reproduce verdict: SETUP_FAILED

## Manual verification checklist
1. Open the entry point (`tool_write`) and confirm it takes external/attacker input.
2. Follow the call path above and confirm the input reaches `tool_write` without being sanitized.
3. Reproduce the exploit (payload below) through that entry point against a local run.
4. Only then prepare a responsible-disclosure report.

- payload: `/tmp/tier2_proof_eyy529bc/pwned_b3c8d0753e66`

NOTE: reachability here is a NAME-BASED static call graph (conservative, may over-connect). Step 1-3 above are the human confirmation that closes that gap before any disclosure.
