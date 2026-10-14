# Disclosure dossier: load_overrides (CWE-22)

- repo: nthplusio/skills  (https://github.com/nthplusio/skills.git)
- file: skills/scoussens-skills/pr-review-round/scripts/queue.py
- tainted parameter: path
- Tier-2 sink: open
- Tier-2 exploit proof: attacker-controlled input opened our out-of-tree planted file /tmp/tier2_proof_ld7f4s2a/secret_1a055c5af94c.txt (proven by live execution trace; read-traversal, no output echo required)

## Reachability (what makes this disclosable)
- reachable from a real entry point: **True**
- entry-point type: **api**
- call path to verify by hand:

    load_overrides

- dynamic reproduce verdict: SETUP_FAILED

## Manual verification checklist
1. Open the entry point (`load_overrides`) and confirm it takes external/attacker input.
2. Follow the call path above and confirm the input reaches `load_overrides` without being sanitized.
3. Reproduce the exploit (payload below) through that entry point against a local run.
4. Only then prepare a responsible-disclosure report.

- payload: `/tmp/tier2_proof_ld7f4s2a/secret_1a055c5af94c.txt`

NOTE: reachability here is a NAME-BASED static call graph (conservative, may over-connect). Step 1-3 above are the human confirmation that closes that gap before any disclosure.
