# Disclosure dossier: _get (CWE-918)

- repo: kamoras/civitas  (https://github.com/kamoras/civitas.git)
- file: backend/scripts/calibrate_lda_bill_match.py
- tainted parameter: url
- Tier-2 sink: urllib.request.urlopen
- Tier-2 exploit proof: SSRF: program fetched the attacker-controlled URL http://127.0.0.1:40935/5b0165870487

## Reachability (what makes this disclosable)
- reachable from a real entry point: **True**
- entry-point type: **http**
- call path to verify by hand:

    refresh_action_center -> refresh_action_issues -> _run_refresh -> calibrate -> lda_descriptions -> _get

- dynamic reproduce verdict: NEEDS_MANUAL

## Manual verification checklist
1. Open the entry point (`refresh_action_center`) and confirm it takes external/attacker input.
2. Follow the call path above and confirm the input reaches `_get` without being sanitized.
3. Reproduce the exploit (payload below) through that entry point against a local run.
4. Only then prepare a responsible-disclosure report.

- payload: `http://127.0.0.1:40935/5b0165870487`

NOTE: reachability here is a NAME-BASED static call graph (conservative, may over-connect). Step 1-3 above are the human confirmation that closes that gap before any disclosure.
