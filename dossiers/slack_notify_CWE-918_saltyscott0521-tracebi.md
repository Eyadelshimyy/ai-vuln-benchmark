# Disclosure dossier: slack_notify (CWE-918)

- repo: saltyscott0521/tracebi  (https://github.com/saltyscott0521/tracebi.git)
- file: tracebi/_delivery.py
- tainted parameter: webhook_url
- Tier-2 sink: urllib.request.urlopen
- Tier-2 exploit proof: SSRF: program fetched the attacker-controlled URL http://127.0.0.1:37743/db1f8b2f2917

## Reachability (what makes this disclosable)
- reachable from a real entry point: **True**
- entry-point type: **api**
- call path to verify by hand:

    slack_notify

- dynamic reproduce verdict: NEEDS_MANUAL

## Manual verification checklist
1. Open the entry point (`slack_notify`) and confirm it takes external/attacker input.
2. Follow the call path above and confirm the input reaches `slack_notify` without being sanitized.
3. Reproduce the exploit (payload below) through that entry point against a local run.
4. Only then prepare a responsible-disclosure report.

- payload: `http://127.0.0.1:37743/db1f8b2f2917`

NOTE: reachability here is a NAME-BASED static call graph (conservative, may over-connect). Step 1-3 above are the human confirmation that closes that gap before any disclosure.
