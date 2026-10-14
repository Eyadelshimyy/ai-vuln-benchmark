# Disclosure dossier: read_record (CWE-22)

- repo: kamoras/civitas  (https://github.com/kamoras/civitas.git)
- file: backend/app/shared_state.py
- tainted parameter: path
- Tier-2 sink: open
- Tier-2 exploit proof: attacker-controlled input opened our out-of-tree planted file /tmp/tier2_proof_r20_8wo5/secret_c84afa1b9228.txt (proven by live execution trace; read-traversal, no output echo required)

## Reachability (what makes this disclosable)
- reachable from a real entry point: **True**
- entry-point type: **http**
- call path to verify by hand:

    pipeline_status -> get_next_run_time -> read_heartbeat -> read_record

- dynamic reproduce verdict: REPRODUCED_PUBLIC_API

## Manual verification checklist
1. Open the entry point (`pipeline_status`) and confirm it takes external/attacker input.
2. Follow the call path above and confirm the input reaches `read_record` without being sanitized.
3. Reproduce the exploit (payload below) through that entry point against a local run.
4. Only then prepare a responsible-disclosure report.

- payload: `/tmp/tier2_proof_r20_8wo5/secret_c84afa1b9228.txt`

NOTE: reachability here is a NAME-BASED static call graph (conservative, may over-connect). Step 1-3 above are the human confirmation that closes that gap before any disclosure.
