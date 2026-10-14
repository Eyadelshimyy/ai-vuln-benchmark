# Disclosure dossier: save_day_store (CWE-22)

- repo: 2100preet/stockscanner  (https://github.com/2100preet/stockscanner.git)
- file: odte_scanner/signals/sure_winners.py
- tainted parameter: path
- Tier-2 sink: Path.write_text
- Tier-2 exploit proof: traversal write escaped to /tmp/tier2_proof_k2e1b0vi/pwned_490d41199377

## Reachability (what makes this disclosable)
- reachable from a real entry point: **True**
- entry-point type: **http**
- call path to verify by hand:

    snapshot -> build_sure_winner_board -> save_day_store

- dynamic reproduce verdict: NEEDS_MANUAL

## Manual verification checklist
1. Open the entry point (`snapshot`) and confirm it takes external/attacker input.
2. Follow the call path above and confirm the input reaches `save_day_store` without being sanitized.
3. Reproduce the exploit (payload below) through that entry point against a local run.
4. Only then prepare a responsible-disclosure report.

- payload: `/tmp/tier2_proof_k2e1b0vi/pwned_490d41199377`

NOTE: reachability here is a NAME-BASED static call graph (conservative, may over-connect). Step 1-3 above are the human confirmation that closes that gap before any disclosure.
