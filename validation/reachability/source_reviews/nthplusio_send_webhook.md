# Source review — nthplusio/skills `send_webhook` (CWE-918 SSRF)

**Verdict (source-only): REACHABILITY_SUPPORTED.** Route→sink dataflow is real; no persisted
end-to-end runtime effect artifact is in the repo, so not APP_CONFIRMED on this evidence.

## Chain (verified in source)
- `app.py` (Flask): `app = Flask(__name__)`; `@app.post("/subscriptions")` stores subscriber
  URLs; `@app.post("/orders")` → `create_order()` → `for url in subscriptions.urls_for("order.created"): send_webhook(url, {...})`.
- `webhooks.py:5` `def send_webhook(url, payload): requests.post(url, json=payload)` — no scheme
  check, no host allowlist, no timeout.
- Attacker controls `url`: `POST /subscriptions` registers an arbitrary URL, then any
  `POST /orders` makes the server fetch it. Stored SSRF.

## Origin / controls
- Input origin: request body (`/subscriptions`). No validation or egress control before the sink.
- Auth: **no** auth on the routes (plain Flask fixture app, no auth middleware). source-derived.

## Scope / limits
- This is **eval-fixture code** (`.../visualize-spec/evals/fixtures/order-hooks/`), not a deployed
  service — so not a disclosure case.
- Historical label conflict: `TAXONOMY.md` lists APP_CONFIRMED; `REACHABILITY_MANUAL_VALIDATION.md`
  lists REACHABILITY_SUPPORTED with the end-to-end run "deferred". No saved end-to-end artifact
  (no callback log / outbound-fetch capture) found in the repo. Pending a saved artifact, the
  defensible label is REACHABILITY_SUPPORTED.
- Warrants a future dynamic test by the researcher (boot fixture + loopback listener) to reach APP_CONFIRMED.
