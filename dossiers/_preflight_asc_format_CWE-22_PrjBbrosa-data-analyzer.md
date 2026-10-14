# Disclosure dossier: _preflight_asc_format (CWE-22)

- repo: PrjBbrosa/data-analyzer  (https://github.com/PrjBbrosa/data-analyzer.git)
- file: mf4_analyzer/io/asc_can_format.py
- tainted parameter: fp
- Tier-2 sink: open
- Tier-2 exploit proof: attacker-controlled input opened our out-of-tree planted file /tmp/tier2_proof_ye93ahxn/secret_6a114bc0f497.txt (proven by live execution trace; read-traversal, no output echo required)

## Reachability (what makes this disclosable)
- reachable from a real entry point: **True**
- entry-point type: **api**
- call path to verify by hand:

    read_asc_outcome -> _preflight_asc_format

- dynamic reproduce verdict: NEEDS_MANUAL

## Manual verification checklist
1. Open the entry point (`read_asc_outcome`) and confirm it takes external/attacker input.
2. Follow the call path above and confirm the input reaches `_preflight_asc_format` without being sanitized.
3. Reproduce the exploit (payload below) through that entry point against a local run.
4. Only then prepare a responsible-disclosure report.

- payload: `/tmp/tier2_proof_ye93ahxn/secret_6a114bc0f497.txt`

NOTE: reachability here is a NAME-BASED static call graph (conservative, may over-connect). Step 1-3 above are the human confirmation that closes that gap before any disclosure.
