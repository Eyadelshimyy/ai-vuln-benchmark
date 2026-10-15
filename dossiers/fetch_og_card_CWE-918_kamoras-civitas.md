# Disclosure dossier: fetch_og_card (CWE-918)

- repo: kamoras/civitas  (https://github.com/kamoras/civitas.git)
- file: backend/app/pipeline/analyze/bluesky_utils.py
- tainted parameter: url
- Tier-2 sink: httpx.Client.send
- Tier-2 exploit proof: SSRF: program fetched the attacker-controlled URL http://127.0.0.1:43555/aecdb2a6ec3f

## Reachability (what makes this disclosable)
- reachable from a real entry point: **True**
- entry-point type: **http**
- call path to verify by hand:

    refresh_action_center -> refresh_action_issues -> _run_refresh -> _run_periodic_bluesky_posts -> fill_missing_cards -> capture_card -> fetch_og_card

- dynamic reproduce verdict: SETUP_FAILED

## Manual verification checklist
1. Open the entry point (`refresh_action_center`) and confirm it takes external/attacker input.
2. Follow the call path above and confirm the input reaches `fetch_og_card` without being sanitized.
3. Reproduce the exploit (payload below) through that entry point against a local run.
4. Only then prepare a responsible-disclosure report.

- payload: `http://127.0.0.1:43555/aecdb2a6ec3f`

NOTE: reachability here is a NAME-BASED static call graph (conservative, may over-connect). Step 1-3 above are the human confirmation that closes that gap before any disclosure.
