# Disclosure dossier: _read_node_ids (CWE-22)

- repo: mikelane/pytest-gremlins  (https://github.com/mikelane/pytest-gremlins.git)
- file: src/pytest_gremlins/node_id_args.py
- tainted parameter: path
- Tier-2 sink: open
- Tier-2 exploit proof: attacker-controlled input opened our out-of-tree planted file /tmp/tier2_proof_dipza1pb/secret_880510c19e19.txt (proven by live execution trace; read-traversal, no output echo required)

## Reachability (what makes this disclosable)
- reachable from a real entry point: **True**
- entry-point type: **api**
- call path to verify by hand:

    args_with_node_ids_from_file -> _read_node_ids

- dynamic reproduce verdict: NEEDS_MANUAL

## Manual verification checklist
1. Open the entry point (`args_with_node_ids_from_file`) and confirm it takes external/attacker input.
2. Follow the call path above and confirm the input reaches `_read_node_ids` without being sanitized.
3. Reproduce the exploit (payload below) through that entry point against a local run.
4. Only then prepare a responsible-disclosure report.

- payload: `/tmp/tier2_proof_dipza1pb/secret_880510c19e19.txt`

NOTE: reachability here is a NAME-BASED static call graph (conservative, may over-connect). Step 1-3 above are the human confirmation that closes that gap before any disclosure.
