# Disclosure dossier: _read_model_json (CWE-22)

- repo: bramstroker/homeassistant-powercalc  (https://github.com/bramstroker/homeassistant-powercalc.git)
- file: utils/library/common.py
- tainted parameter: file_name
- Tier-2 sink: open
- Tier-2 exploit proof: attacker-controlled input opened our out-of-tree planted file /tmp/tier2_proof_zigilrs0/secret_7f9877f260b7.txt (proven by live execution trace; read-traversal, no output echo required)

## Reachability (what makes this disclosable)
- reachable from a real entry point: **True**
- entry-point type: **api**
- call path to verify by hand:

    find_model_json_files -> _read_model_json

- dynamic reproduce verdict: NEEDS_MANUAL

## Manual verification checklist
1. Open the entry point (`find_model_json_files`) and confirm it takes external/attacker input.
2. Follow the call path above and confirm the input reaches `_read_model_json` without being sanitized.
3. Reproduce the exploit (payload below) through that entry point against a local run.
4. Only then prepare a responsible-disclosure report.

- payload: `/tmp/tier2_proof_zigilrs0/secret_7f9877f260b7.txt`

NOTE: reachability here is a NAME-BASED static call graph (conservative, may over-connect). Step 1-3 above are the human confirmation that closes that gap before any disclosure.
