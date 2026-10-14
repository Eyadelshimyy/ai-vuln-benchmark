# Disclosure dossier: _load_csv (CWE-22)

- repo: alexandrelheinen/fret  (https://github.com/alexandrelheinen/fret.git)
- file: scripts/plot_telemetry.py
- tainted parameter: path
- Tier-2 sink: open
- Tier-2 exploit proof: attacker-controlled input opened our out-of-tree planted file /tmp/tier2_proof_umckf23r/secret_a32450e2a9e1.txt (proven by live execution trace; read-traversal, no output echo required)

## Reachability (what makes this disclosable)
- reachable from a real entry point: **True**
- entry-point type: **api**
- call path to verify by hand:

    plot_telemetry -> _load_csv

- dynamic reproduce verdict: NEEDS_MANUAL

## Manual verification checklist
1. Open the entry point (`plot_telemetry`) and confirm it takes external/attacker input.
2. Follow the call path above and confirm the input reaches `_load_csv` without being sanitized.
3. Reproduce the exploit (payload below) through that entry point against a local run.
4. Only then prepare a responsible-disclosure report.

- payload: `/tmp/tier2_proof_umckf23r/secret_a32450e2a9e1.txt`

NOTE: reachability here is a NAME-BASED static call graph (conservative, may over-connect). Step 1-3 above are the human confirmation that closes that gap before any disclosure.
