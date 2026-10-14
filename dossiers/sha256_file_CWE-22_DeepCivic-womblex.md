# Disclosure dossier: sha256_file (CWE-22)

- repo: DeepCivic/womblex  (https://github.com/DeepCivic/womblex.git)
- file: src/womblex/utils/checksum.py
- tainted parameter: path
- Tier-2 sink: open
- Tier-2 exploit proof: attacker-controlled input opened our out-of-tree planted file /tmp/tier2_proof_mcykxwfl/secret_275e29d86a63.txt (proven by live execution trace; read-traversal, no output echo required)

## Reachability (what makes this disclosable)
- reachable from a real entry point: **True**
- entry-point type: **api**
- call path to verify by hand:

    sha256_file

- dynamic reproduce verdict: REPRODUCED_PUBLIC_API

## Manual verification checklist
1. Open the entry point (`sha256_file`) and confirm it takes external/attacker input.
2. Follow the call path above and confirm the input reaches `sha256_file` without being sanitized.
3. Reproduce the exploit (payload below) through that entry point against a local run.
4. Only then prepare a responsible-disclosure report.

- payload: `/tmp/tier2_proof_mcykxwfl/secret_275e29d86a63.txt`

NOTE: reachability here is a NAME-BASED static call graph (conservative, may over-connect). Step 1-3 above are the human confirmation that closes that gap before any disclosure.
