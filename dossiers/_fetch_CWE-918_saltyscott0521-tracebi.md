# Disclosure dossier: _fetch (CWE-918)

- repo: saltyscott0521/tracebi  (https://github.com/saltyscott0521/tracebi.git)
- file: tracebi/_updates.py
- tainted parameter: url
- Tier-2 sink: urllib.request.urlopen
- Tier-2 exploit proof: SSRF: program fetched the attacker-controlled URL http://127.0.0.1:44289/4667341692a7

## Reachability (what makes this disclosable)
- reachable from a real entry point: **True**
- entry-point type: **api**
- call path to verify by hand:

    latest_release -> _fetch

- dynamic reproduce verdict: NEEDS_MANUAL

## Manual verification checklist
1. Open the entry point (`latest_release`) and confirm it takes external/attacker input.
2. Follow the call path above and confirm the input reaches `_fetch` without being sanitized.
3. Reproduce the exploit (payload below) through that entry point against a local run.
4. Only then prepare a responsible-disclosure report.

- payload: `http://127.0.0.1:44289/4667341692a7`

NOTE: reachability here is a NAME-BASED static call graph (conservative, may over-connect). Step 1-3 above are the human confirmation that closes that gap before any disclosure.
