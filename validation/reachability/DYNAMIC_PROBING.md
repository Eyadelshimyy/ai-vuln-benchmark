# Dynamic app-level reachability probing (execution-adjudicated)

Method: start a canary HTTP listener, enumerate a RUNNING app's routes (OpenAPI), fire
unique per-(route,param) canary payloads at every route, and observe which canaries fire
-> which sinks are attacker-reachable END-TO-END via a real route+param. Complements static
reachability (over-reports) and manual full-source tracing. Tool: dyn_prober_live.py (v2).

## Prober validation (positive control)
- **nthplusio/skills order-hooks** (known stored SSRF): prober auto-discovered the routes,
  fired canary payloads, canary listener HIT -> SSRF fired end-to-end. VALIDATED (catches a
  real reachable sink with zero hand-tracing).

## Applied datapoints
| app | boot | routes probed | canary reachable? | SSRF callbacks | result |
|---|---|---|---|---|---|
| nthplusio/skills | clean (Flask) | 2 | n/a (in-process) | **1** | SSRF APP-REACHABLE (positive control) |
| **civitas** | **FAITHFUL (docker compose, real build+DB+pipeline)** | **114** | **VERIFIED** (container->host /REACHTEST received) | **0** | **no reachable SSRF across 114 routes** |
| leapmotor-mate | stubbed/approximate | 112 | n/a | 0 | no reachable SSRF (sink uses fixed host) |
| civitas (dyn) — reachability proof | container reached host canary (`listener received: ['/REACHTEST']`) | — | yes | — | negative is real, not a blocked canary |

## Why the negatives are real (characterized, not "prober missed it")
- **civitas**: faithful boot, 114 routes, canary reachability PROVEN; 0 SSRF callbacks. Its confirmed
  URL-fetch sinks (e.g. `_get`->SCDB, `fetch_og_card`->bluesky) pull URLs from internal data/config,
  not request params -> no attacker-controlled host to redirect. Matches the manual full-source verdict (REFUTED).
- **leapmotor**: URL-fetch sink `charger_locator` uses a HARDCODED host (`overpass-api.de`); attacker
  controls only lat/lon query data, never the target host -> structurally not SSRF.

## Honest scope / limits
- Prober tests **SSRF (URL-fetch) only**. It does NOT dynamically exercise CWE-22 (path traversal) or
  CWE-78 (command injection) — those sinks' reachability rests on the full-source hand-tracing.
- civitas also has confirmed **CWE-22** sinks (read_json etc.); their reachability = REFUTED by manual
  tracing (allowlist + constant paths), NOT by this SSRF run.
- leapmotor's boot was stubbed (approximate); civitas's boot was FAITHFUL (the strong datapoint).
- A dynamic negative is only trusted after a canary-reachability proof (done for civitas).
- Heavy apps stockscanner/BugForge/womblex not booted; civitas (largest) + nthplusio (positive) cover
  the strong cases.

## v3 upgrade — three sink classes, validated, applied to civitas
Prober v3 adds CWE-22 (path traversal, sentinel-file canary) and CWE-78 (command injection,
command-callback canary) alongside SSRF, with PER-PARAMETER isolation (each string param tested
alone -> a hit names the exact route+param).

Validation (positive controls, live HTTP, all RECOVERED):
- SSRF  CWE-918  -> POST /subscriptions param='url'
- CWE-78         -> POST /ping param='host'
- CWE-22         -> POST /read param='filename'

Applied to faithful civitas (114 routes, canary reachability proven), sentinel planted in-container:
- SSRF   CWE-918 : 0 reachable
- CWE-78         : 0 reachable
- CWE-22         : 0 reachable  <- dynamic confirmation of the manual REFUTED verdict
=> civitas, the largest confirmed contributor, is COMPREHENSIVELY 0-reachable across all three
   execution-confirmed sink classes, by a prober validated to catch all three.

Honest caveats (unchanged): per-parameter single-request coverage (not full multi-step workflows);
the CWE-22 canary fires only if a sink does open(attacker_path) reaching the sentinel; 0 is evidence
for the probed parameters, not proof every input path was exhausted.

## Bottom line — attacker-reachability triangulated THREE ways
Static call-graph (90 gold, over-reports) + manual full-source tracing (12 HTTP gold -> 1 reachable,
rest attrited with documented reasons) + dynamic probing (nthplusio reachable [positive control];
civitas FAITHFUL 114 routes -> 0, reachability-proven; leapmotor 0, characterized) ALL converge:
**1 attacker-reachable confirmed vulnerability (the SSRF) corpus-wide; near-total attrition with
documented, execution-adjudicated causes.** The near-zero is the finding, now confirmed by three
independent methods including a faithful boot of the largest app.
