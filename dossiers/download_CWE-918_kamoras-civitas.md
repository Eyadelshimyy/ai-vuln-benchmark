# Disclosure dossier: download (CWE-918)

- repo: kamoras/civitas  (https://github.com/kamoras/civitas.git)
- file: backend/scripts/fetch_state_leg_crosswalk.py
- tainted parameter: url
- Tier-2 sink: urllib.request.urlopen
- Tier-2 exploit proof: SSRF: program fetched the attacker-controlled URL http://127.0.0.1:39313/159ca65fe23d

## Reachability (what makes this disclosable)
- reachable from a real entry point: **True**
- entry-point type: **api**
- call path to verify by hand:

    download

- dynamic reproduce verdict: NEEDS_MANUAL

## Manual verification checklist
1. Open the entry point (`download`) and confirm it takes external/attacker input.
2. Follow the call path above and confirm the input reaches `download` without being sanitized.
3. Reproduce the exploit (payload below) through that entry point against a local run.
4. Only then prepare a responsible-disclosure report.

- payload: `http://127.0.0.1:39313/159ca65fe23d`

NOTE: reachability here is a NAME-BASED static call graph (conservative, may over-connect). Step 1-3 above are the human confirmation that closes that gap before any disclosure.
