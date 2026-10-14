# Disclosure dossier: load_county_names (CWE-22)

- repo: kamoras/civitas  (https://github.com/kamoras/civitas.git)
- file: backend/scripts/build_county_district_crosswalk.py
- tainted parameter: path
- Tier-2 sink: open
- Tier-2 exploit proof: attacker-controlled input opened our out-of-tree planted file /tmp/tier2_proof_rvffrr65/secret_2150b910ee54.txt (proven by live execution trace; read-traversal, no output echo required)

## Reachability (what makes this disclosable)
- reachable from a real entry point: **True**
- entry-point type: **api**
- call path to verify by hand:

    load_county_names

- dynamic reproduce verdict: REPRODUCED_PUBLIC_API

## Manual verification checklist
1. Open the entry point (`load_county_names`) and confirm it takes external/attacker input.
2. Follow the call path above and confirm the input reaches `load_county_names` without being sanitized.
3. Reproduce the exploit (payload below) through that entry point against a local run.
4. Only then prepare a responsible-disclosure report.

- payload: `/tmp/tier2_proof_rvffrr65/secret_2150b910ee54.txt`

NOTE: reachability here is a NAME-BASED static call graph (conservative, may over-connect). Step 1-3 above are the human confirmation that closes that gap before any disclosure.
