# Disclosure dossier: submit (CWE-22)

- repo: danielreuter/research-notes  (https://github.com/danielreuter/research-notes.git)
- file: lanes/proofs-n2-guest/tools/refill.py
- tainted parameter: name
- Tier-2 sink: Path.write_text
- Tier-2 exploit proof: traversal write escaped to /tmp/tier2_proof_6uimd_ab/pwned_7bad822784ca

## Reachability (what makes this disclosable)
- reachable from a real entry point: **True**
- entry-point type: **api**
- call path to verify by hand:

    submit

- dynamic reproduce verdict: SETUP_FAILED

## Manual verification checklist
1. Open the entry point (`submit`) and confirm it takes external/attacker input.
2. Follow the call path above and confirm the input reaches `submit` without being sanitized.
3. Reproduce the exploit (payload below) through that entry point against a local run.
4. Only then prepare a responsible-disclosure report.

- payload: `/tmp/tier2_proof_6uimd_ab/pwned_7bad822784ca`

NOTE: reachability here is a NAME-BASED static call graph (conservative, may over-connect). Step 1-3 above are the human confirmation that closes that gap before any disclosure.
