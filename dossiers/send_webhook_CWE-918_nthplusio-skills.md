# Disclosure dossier: send_webhook (CWE-918)

- repo: nthplusio/skills  (https://github.com/nthplusio/skills.git)
- file: skills/scoussens-skills/visualize-spec/evals/fixtures/order-hooks/src/order_hooks/webhooks.py
- tainted parameter: url
- Tier-2 sink: requests.Session.request
- Tier-2 exploit proof: SSRF: program fetched the attacker-controlled URL http://127.0.0.1:41611/721d19e955d0

## Reachability (what makes this disclosable)
- reachable from a real entry point: **True**
- entry-point type: **http**
- call path to verify by hand:

    create_order -> send_webhook

- dynamic reproduce verdict: NEEDS_MANUAL

## Manual verification checklist
1. Open the entry point (`create_order`) and confirm it takes external/attacker input.
2. Follow the call path above and confirm the input reaches `send_webhook` without being sanitized.
3. Reproduce the exploit (payload below) through that entry point against a local run.
4. Only then prepare a responsible-disclosure report.

- payload: `http://127.0.0.1:41611/721d19e955d0`

NOTE: reachability here is a NAME-BASED static call graph (conservative, may over-connect). Step 1-3 above are the human confirmation that closes that gap before any disclosure.
