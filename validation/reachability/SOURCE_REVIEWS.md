# V1 attacker-flagged candidates — Level-1 source reviews (static only)

The static screen flagged 3 v1 confirmed functions as `attacker`-origin + route-reachable.
Level-1 source inspection (no boot, no requests) resolves them:

| candidate | CWE | source-only verdict | basis |
|---|---|---|---|
| nthplusio/skills `send_webhook` | 918 | **REACHABILITY_SUPPORTED** | real Flask `/subscriptions`→`/orders`→`requests.post(url)`, attacker url, no egress control; fixture-scope; no saved end-to-end artifact |
| quantam101/profitenginev5 `digest` | 22 | **REFUTED (false edge)** | screened chain merges `stable_embedding`'s local var `digest` with the module fn `digest(path)`; the webhook never reaches the file hasher |
| trsdn/macos-notarization-broker `write_json` | 22 | **REFUTED (no HTTP entrypoint)** | argparse CLI tool; screen's "req-arg" matched a param coincidentally named `request` |

**Result:** of 3 attacker-flags, 2 are static-screen false positives; 1 (nthplusio) is a genuine
route→sink SSRF, fixture-scope, pending a saved end-to-end artifact for APP_CONFIRMED.
Per-candidate detail in `source_reviews/`.
