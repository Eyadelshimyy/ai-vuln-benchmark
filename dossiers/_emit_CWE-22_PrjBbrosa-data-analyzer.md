# Disclosure dossier: _emit (CWE-22)

- repo: PrjBbrosa/data-analyzer  (https://github.com/PrjBbrosa/data-analyzer.git)
- file: scripts/probe_switch_smoothness.py
- tainted parameter: out
- Tier-2 sink: None
- Tier-2 exploit proof: traversal write escaped to /tmp/tier2_proof_clwkdlzo/pwned_98ca3dc16264

## Reachability (what makes this disclosable)
- reachable from a real entry point: **True**
- entry-point type: **api**
- call path to verify by hand:

    run_section_scenario -> _emit

- dynamic reproduce verdict: NEEDS_MANUAL

## Manual verification checklist
1. Open the entry point (`run_section_scenario`) and confirm it takes external/attacker input.
2. Follow the call path above and confirm the input reaches `_emit` without being sanitized.
3. Reproduce the exploit (payload below) through that entry point against a local run.
4. Only then prepare a responsible-disclosure report.

- payload: `/tmp/tier2_proof_clwkdlzo/pwned_98ca3dc16264`

NOTE: reachability here is a NAME-BASED static call graph (conservative, may over-connect). Step 1-3 above are the human confirmation that closes that gap before any disclosure.
