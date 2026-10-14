# Disclosure dossier: tool_edit (CWE-22)

- repo: loveRhythm1990/mini-pi  (https://github.com/loveRhythm1990/mini-pi.git)
- file: 03-tools/src/tools.py
- tainted parameter: path
- Tier-2 sink: open
- Tier-2 exploit proof: attacker-controlled input opened our out-of-tree planted file /tmp/tier2_proof_7embeael/secret_77b973e1a88d.txt (proven by live execution trace; read-traversal, no output echo required)

## Reachability (what makes this disclosable)
- reachable from a real entry point: **True**
- entry-point type: **api**
- call path to verify by hand:

    tool_edit

- dynamic reproduce verdict: NEEDS_MANUAL

## Manual verification checklist
1. Open the entry point (`tool_edit`) and confirm it takes external/attacker input.
2. Follow the call path above and confirm the input reaches `tool_edit` without being sanitized.
3. Reproduce the exploit (payload below) through that entry point against a local run.
4. Only then prepare a responsible-disclosure report.

- payload: `/tmp/tier2_proof_7embeael/secret_77b973e1a88d.txt`

NOTE: reachability here is a NAME-BASED static call graph (conservative, may over-connect). Step 1-3 above are the human confirmation that closes that gap before any disclosure.
