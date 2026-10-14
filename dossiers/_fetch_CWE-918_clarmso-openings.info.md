# Disclosure dossier: _fetch (CWE-918)

- repo: clarmso/openings.info  (https://github.com/clarmso/openings.info.git)
- file: src/http.py
- tainted parameter: url
- Tier-2 sink: urllib.request.urlopen
- Tier-2 exploit proof: SSRF: program fetched the attacker-controlled URL http://127.0.0.1:38197/9b2792b2e48a

## Reachability (what makes this disclosable)
- reachable from a real entry point: **True**
- entry-point type: **api**
- call path to verify by hand:

    fetch_json -> _fetch

- dynamic reproduce verdict: NEEDS_MANUAL

## Manual verification checklist
1. Open the entry point (`fetch_json`) and confirm it takes external/attacker input.
2. Follow the call path above and confirm the input reaches `_fetch` without being sanitized.
3. Reproduce the exploit (payload below) through that entry point against a local run.
4. Only then prepare a responsible-disclosure report.

- payload: `http://127.0.0.1:38197/9b2792b2e48a`

NOTE: reachability here is a NAME-BASED static call graph (conservative, may over-connect). Step 1-3 above are the human confirmation that closes that gap before any disclosure.
