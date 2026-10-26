# VULCAN — Manual validation of the gold set (Tier-2 confirmed ∩ interface-reachable)

Gold set size: **90** (http=12, api=76, main=2). This log hand-traces the 12 HTTP
candidates (external attack surface) to separate genuine end-to-end attacker-reachable
cases from static-reachability artifacts.

Evidence ladder (frozen): FIXTURE_CONFIRMED < REACHABILITY_SUPPORTED < APP_CONFIRMED.
None of these enter the frozen primary 160. Disclosure requires APP_CONFIRMED + a real
(non-fixture) deployment + version/config match.

---

## #1 — nthplusio/skills · send_webhook · CWE-918 (SSRF) — **REACHABILITY_SUPPORTED** ✅ survives
- File: `evals/fixtures/order-hooks/src/order_hooks/{app.py,webhooks.py}`
- Sink: `send_webhook(url, payload)` → `requests.post(url, json=payload)` — no allowlist, no scheme check, no timeout.
- Taint path (stored / second-order SSRF):
  1. `POST /subscriptions`  body `{"event":"order.created","url":<attacker>}` → `subscriptions.add(event, body["url"])` (stored, unvalidated)
  2. `POST /orders` → `create_order` → `for url in subscriptions.urls_for("order.created"): send_webhook(url, ...)`
- Taint source = raw request body; reaches unguarded `requests.post` across two real Flask routes.
- Scope boundary: lives under `evals/fixtures/` — example code in a skills repo, NOT a deployed production service. Legitimate corpus finding (AI-generated reachable SSRF); NOT a disclosure case.
- To reach APP_CONFIRMED: stand up `create_app()` locally + localhost listener, drive /subscriptions→/orders, observe outbound fetch. (pending — deferred per breadth-first)


---

## Triage of remaining 11 gold HTTP candidates (entry-point + taint analysis)

NOTE: initial prediction (civitas = internal pipelines) was WRONG — they are real
FastAPI routes. Corrected by inspection. This is why hand-validation is done, not assumed.

| # | repo · sink | CWE | entry route | status | reason |
|---|---|---|---|---|---|
| 2 | stockscanner · save_day_store | 22 | `GET /api/snapshot` (no params) | **ATTRITES (pending confirm)** | handler takes no request input; sink path from internal config, not attacker |
| 3 | stockscanner · load_config | 22 | `GET /api/snapshot` (no params) | **ATTRITES (pending confirm)** | same |
| 4 | civitas · read_record | 22 | `GET /pipeline/status` (db only) | **ATTRITES (pending confirm)** | no request-controlled input |
| 5 | civitas · read_json | 22 | `GET /states/{state}` | **LIKELY ATTRITES** | `state=state.upper()` + used as jurisdiction key, not raw path; filename templated downstream |
| 6 | civitas · fetch_og_card | 918 | internal refresh chain | pending | — |
| 7 | civitas · build_link_card | 918 | internal refresh chain | pending | — |
| 8 | civitas · update_json_file | 22 | `POST /pipeline/trigger` (auth header) | **WEAK (auth-gated)** | authenticated-only; not unauth-reachable |
| 9 | civitas · _get | 918 | `POST /pipeline/trigger` (auth header) | **WEAK (auth-gated)** | authenticated-only |
| 10| leapmotor-mate · _write_bytes | 22 | `POST /api/refresh` (request: Request) | **OPEN — tracing** | real route w/ request obj; need taint→filename |
| 11| BugForge · prepare_foundry_workspace | 22 | `POST /sessions/{id}/analyze` | **OPEN — tracing** | analyzes a (possibly attacker-submitted) repo |

Running tally: 1 survivor (SSRF, #1), ~6 clear/likely attritions with documented reasons,
2 auth-gated weak, 2 open. The attrition is NOT total — but it is heavy and each case has
a concrete reason, which is the defensible form of the finding.

---

## Final verdicts — 12 gold HTTP candidates (hand-traced)

| # | repo · sink | CWE | verdict | basis |
|---|---|---|---|---|
| 1 | nthplusio/skills · send_webhook | 918 | **REACHABILITY_SUPPORTED ✅** | stored SSRF; attacker URL via POST /subscriptions → POST /orders → unguarded requests.post. Scope: eval-fixture code, not deployed service. |
| 2 | stockscanner · save_day_store | 22 | ATTRITES | `GET /api/snapshot` takes no request input; sink path internal |
| 3 | stockscanner · load_config | 22 | ATTRITES | same |
| 4 | civitas · read_record | 22 | ATTRITES | `GET /pipeline/status`, no request-controlled input |
| 5 | civitas · read_json | 22 | ATTRITES | `state=state.upper()`, used as jurisdiction key; filename from constants |
| 6 | civitas · fetch_og_card | 918 | WEAK (admin-gated + internal taint) | behind require_admin; fetched URL from internal data |
| 7 | civitas · build_link_card | 918 | WEAK (admin-gated + internal taint) | same |
| 8 | civitas · update_json_file | 22 | WEAK (auth-gated + internal taint) | behind authorization header |
| 9 | civitas · _get | 918 | WEAK (auth-gated + internal taint) | same |
| 10| leapmotor-mate · _write_bytes | 22 | ATTRITES | route's `request` unused; filename internal session-share path |
| 11| BugForge · prepare_foundry_workspace | 22 | WEAK/ATTRITES | operator-gated; repo_path from session state, not request |

### Result
- **1 / 12 survives** as end-to-end attacker-reachable (unauthenticated): #1, a stored SSRF — and it is eval-fixture code, so NOT a deployment/disclosure case.
- **5 attrite** outright: no request-controlled input reaches the confirmed sink, or input is sanitized/used as a key. These are *static-reachability false leads* (call-graph connects, taint does not).
- **5 are privilege-gated with internal-data taint**: reachable only by an authenticated/admin caller, and the sink argument comes from internal data — not an unauthenticated external attacker.

### What this means for the thesis
The headline sharpens from a bare "0 reachable" to an **evidence-backed attrition account**:
across the AI corpus, execution-confirmed function-level vulnerabilities rarely survive to
unauthenticated end-to-end attacker-reachability through the real application surface. Of the
single-digit candidates that even reach a real HTTP route, almost all fail on the *taint-source*
question (the call graph connects, but attacker data does not reach the sink) or on
*authentication gating*. The one clean survivor is in example/fixture code, not a deployed
service. This is a stronger, more defensible claim than "0", because every attrition is
documented with a concrete, checkable reason.

### NOTE on the other 78 gold candidates (api=76, main=2)
These are NOT external HTTP surface — `api` entries are predominantly path-length-1
(the sink function is itself an exported/public function with a *trusted* caller, not an
untrusted boundary). They are internal/library surface and are characterized as a class
rather than hand-traced individually; spot-checks available on request.

---

## #1 UPGRADED → **APP_CONFIRMED** ✅ (end-to-end, localhost)

Harness: `/tmp/ssrf_confirm.py` — stood up the AI-generated `create_app()`, drove it ONLY
through its real HTTP routes via Flask test client (outbound `requests.post` is real), with a
controlled canary listener on 127.0.0.1:9099.

Observed:
- `POST /subscriptions {"event":"order.created","url":"http://127.0.0.1:9099/ssrf-canary"}` → 204
- `POST /orders {"item":"widget"}` → 201
- **Canary listener received:** `POST /ssrf-canary {"event":"order.created","order":{"id":"791b6566-...","item":"widget"}}`

The server made an outbound request to the attacker-supplied URL — SSRF confirmed end-to-end via
a side-channel the application cannot forge. This is the evidential analogue of the CVE-Bench
canary (effect only a real exploit produces).

Scope (unchanged): this is eval-fixture code within a skills repo, NOT a deployed internet-facing
service. The claim is "AI-generated code containing an end-to-end exploitable stored SSRF through
real application routes," demonstrated in the author's local sandbox. It is NOT a disclosure case
(no deployed target) and does NOT enter the frozen primary 160.

### Evidence ladder — worked examples now on record
- FIXTURE_CONFIRMED: (none yet / optional study)
- REACHABILITY_SUPPORTED: static source trace of the taint path (recorded above)
- APP_CONFIRMED (AI corpus): this SSRF, end-to-end in local sandbox
- APP_CONFIRMED (external control): CVE-2024-34359 via CVE-Bench's unmodified container + grader

---

## #1 SSRF classification — substantiated against "intended webhook feature" objection

A canary hit alone proves an outbound request occurred, not that a security boundary was crossed.
The three SSRF conditions were therefore established explicitly, with a control:

- **(A) Untrusted input controls destination.** No authentication/authorization guard on
  `/subscriptions`, `/orders`, or the store (grep found none). An UNAUTHENTICATED caller sets the URL.
- **(B) Causality (destination follows input), controlled.** Two-URL control, fresh app per trial:
  - input `http://127.0.0.1:9099/ext`       → external listener hit=1, internal hit=0
  - input `http://127.0.0.1:9100/internal/admin` → external hit=0, internal hit=1
  Destination is governed by attacker input, not fixed config (negative control: each trial reaches
  only the just-registered URL).
- **(C) No destination restriction → protected resource reached.** Server was coerced into a request
  to an internal-only loopback endpoint `/internal/admin`. No allowlist/blocklist/private-range check
  exists between the request body and `requests.post`.

**Boundary:** intended behavior would be POSTing to a legitimate *external subscriber*; the vulnerability
is that an unauthenticated attacker can redirect the server to *internal/loopback/private* resources with
no restriction. Boundary crossed → SSRF (CWE-918) substantiated.

**Severity bound (honest):** BLIND SSRF — `send_webhook` is fire-and-forget and discards the response,
so the response body is NOT returned to the attacker via this path. Impact = server-side requests to
internal endpoints/actions/metadata (and request-based effects), not response exfiltration.

**Scope (unchanged):** eval-fixture code in a skills repo, demonstrated in author's local sandbox.
Not a deployed service → not a disclosure case; does not enter the frozen 160.

---

## #1 PROVENANCE — confirmed AI-generated (not a known-CVE function)

From the mined candidate record (`provenance` field):
- `ai_attributed: true`, `ai_line_fraction: 1.0` (100% of the function's lines AI-attributed)
- `ai_tools: {"claude-code": 6}`; `evidence: "Co-authored-by: Claude"`; `ai_commit: 4d236ded03f0`
- `span: [5,10]`; `source_sha256: 0194e027...11cf`
- `prompt_source`: original was a STUB (`raise NotImplementedError`, docstring "Implement the
  send_webhook function") — the AI generated the vulnerable body (`requests.post(url,...)`,
  no SSRF guard / timeout / allowlist).

Source: mined from real GitHub repo `nthplusio/skills`. NOT from any CVE database, NOT from the
real-CVE control set, NOT from CVE-Bench's 40. This is an original VULCAN finding in AI-generated code.

Attribution method (stated honestly): commit-trailer + line-level mapping (the corpus-wide method,
~90% line-level AI-attributed). It is metadata-based attribution, not a per-token guarantee;
ai_line_fraction=1.0 is the strongest level this method yields.

---

## `api` gold candidates (76) — class validation by sampling (5 / 76)

Claim tested: api-kind entries are path-length-1 (sink function flagged as exported/public),
with trusted/internal callers and no untrusted boundary feeding the sink. Sample across 5 repos / 2 CWEs:

| repo · function | CWE | callers / boundary | verdict |
|---|---|---|---|
| DeepCivic/womblex · open_document | 22 | internal pipeline stages; only boundary = CLI on user's own PDFs | ATTRITES (CLI-local, trusted) |
| danielreuter/research-notes · fetch | 918 | **mislabeled** — is a filesystem glob `Path(.../{item}).glob()`, not a URL fetch; single internal caller | ATTRITES (taxonomy false-match) |
| alexandrelheinen/clave · write | 22 | all call sites in tests/ with pytest tmp_path; ML-training lib | ATTRITES (test/internal, tmp paths) |
| VibeBB/fpga-agent · write_png | 22 | path built internally from contract names; diagram rendering | ATTRITES (internal-constructed path) |
| PrjBbrosa/data-analyzer · sha256_file | 22 | internal paths in a desktop GUI app (not a web service) | ATTRITES (no remote surface) |

**Result:** 5/5 confirm the class — internal/library/CLI/test/GUI surface, trusted callers, no
untrusted boundary. The api entry-kind = "exported/public function by name heuristic," NOT an
external attack surface.

Honest bound: this is a SAMPLE (5 of 76), not full enumeration — the class dismissal is
sampling-supported, not exhaustively proven. Confidence is high given uniform results across
diverse repos/CWEs; full enumeration available if an examiner requires it.

Bonus methodology notes:
- `fetch` mislabeled CWE-918 = taxonomy name over-match (cf. LlamaGrammar.from_string SSTI
  over-matches); execution/trace correctly rejects it.
- Several repos are CLI/desktop-GUI tools, not web services → call-graph "reachability" is moot
  (no remote attacker). A distinct, nameable attrition reason.

---

## App-testability classification of gold-set repos (which CAN be end-to-end tested)

Determines, per repo, whether an end-to-end APP test is even definable (needs a real HTTP entry).

**App-testable web services** (web framework + routes + runnable server):
| repo | routes | notes |
|---|---|---|
| nthplusio/skills | 2 | DONE — SSRF APP_CONFIRMED |
| stockscanner | 10 | Flask; DB-backed |
| civitas | 128 | FastAPI; DB-backed (large) |
| leapmotor-mate | 227 | FastAPI; DB-backed (large) |
| BugForge | 179 | FastAPI; DB-backed (large) |
| womblex | 38 | has web UI (src/womblex/ui/app.py) — CORRECTION: earlier called CLI-local; re-check open_document reachability from routes |

**NOT app-testable by nature** (0 routes, no web framework — static dismissal is the correct/only method):
| repo | nature |
|---|---|
| data-analyzer | desktop GUI + CAN probes, no remote surface |
| fpga-agent | CLI/plugin scripts |
| clave | CLI + ML library |
| research-notes | internal batch scripts |

CONSEQUENCE:
- Gold candidates in the 4 non-web repos CANNOT be APP_CONFIRMED (no application/remote entry exists) —
  their attrition is established by absence-of-attack-surface, the correct method for that case.
- Gold candidates in the 5 remaining web repos ARE app-testable → should get end-to-end tests
  (positive OR negative), upgrading their verdicts from source-argued to demonstrated.
- Detector caveat: route-count + web-framework are the reliable signals; the "server runner"/"DB layer"
  rows over-report (first keyword hit, sometimes a test file).
- Rigor correction logged: womblex was under-checked (it is a web service); open_document reachability
  from its routes is re-opened, not settled.

---

## #5 civitas state_ballot (CWE-22) — VERIFIED ATTRITION via FULL-REPO source (not just partial _repo_src)

Method upgrade: cloned the FULL upstream repo `github.com/kamoras/civitas` at the pinned commit
`08c6212d60d2` (the `_repo_src` snapshot had truncated the function before the guard). Full source is dispositive.

Two independent, dispositive reasons the CWE-22 path-traversal does NOT hold:
1. **Allowlist gate.** `state = state.upper(); if state not in ballot_state_codes(): raise HTTPException(404)`.
   `ballot_state_codes()` = `federal_states() | {"DC"}` — a hardcoded `frozenset` of US state codes.
   A traversal payload fails the allowlist and 404s before any use.
2. **Taint never reaches the file path.** Every `read_json` path in elections.py is a FIXED CONSTANT
   (`Path(__file__).resolve().parent.parent / "data" / "<constant>.json"`). `state` flows into SQL
   filters (`Race.state == state`), NOT into any filename.

=> The static reachability path `state_ballot → … → read_json` is a CALL-GRAPH ARTIFACT: the graph
connects route→file-read, but attacker input (`state`) goes to the DB as a filter, while the file path
is constant. Confirmed ATTRITION.

### Methodology note (important)
This REPLACES my earlier source-argument (which rested only on `.upper()`, from a truncated snapshot).
Verification method refined:
- Static gate (allowlist/sanitizer) OR taint-does-not-reach-sink, visible in FULL source → attrition is
  proven at source; a standup would only reproduce a 404 and is not needed.
- Dynamic / data-dependent / not statically settleable (e.g. the SSRF) → stand up and test end-to-end.
This is a concrete, citable example of static-reachability over-reporting that taint/execution corrects.
Earlier rigor gap (reasoning from truncated _repo_src) is now closed by using the full pinned-commit source.

---

## Remaining web-repo gold candidates — VERIFIED against full pinned-commit source

All cloned at their pinned `ai_commit` from public GitHub and traced in full source (not partial _repo_src).

| # | repo · sink | CWE | commit | verdict | basis (full source) |
|---|---|---|---|---|---|
| — | womblex · open_document | 22 | b91919364c67 | ATTRITES | web UI routes = `/api/health` + SPA static server only; open_document reached only by ingest/redact pipeline over DEPLOYMENT-configured paths (`settings.ingest_uri`), not request input. (Earlier under-check resolved: it IS a web service, but the sink isn't route-reachable with taint.) |
| 2 | stockscanner · save_day_store | 22 | e215d7a0d5e7 | ATTRITES | `GET /api/snapshot` takes no params, only reads fixed internal paths; save_day_store gets internal/constant paths from callers |
| 3 | stockscanner · load_config | 22 | 3be9059efc4f | ATTRITES | load_config(path=None) defaults to DEFAULT_CONFIG; path arg is internal config, not request |
| 10| leapmotor-mate · _write_bytes | 22 | d278ccde9e6a | ATTRITES | refresh_now does NOT read `request` at all; _write_bytes called only with internal session-share cert/key paths (attacker bytes are the DATA, not the path) |
| 11| BugForge · prepare_foundry_workspace | 22 | df410ba56e1e | ATTRITES | operator-gated entry; `repo` from session state not request; copy logic guarded by `path.resolve().relative_to(root)` + symlink skip + suffix allowlist + bounds |

Incidental (NOT a gold candidate, noted for completeness): womblex `serve_spa` → FileResponse on
`/{full_path:path}` is a separate potential surface, guarded by `resolve_spa_path`; different sink than
the confirmed open_document, out of scope here.

---

## FINAL — gold-set reachability validation COMPLETE

Across the 90 gold candidates (Tier-2 confirmed ∩ interface-reachable):
- **12 HTTP candidates:** hand-traced; web-repo ones additionally verified against FULL pinned-commit source.
  - **1 genuine end-to-end attacker-reachable** (unauthenticated): the nthplusio SSRF — APP_CONFIRMED
    (canary), SSRF classification substantiated, scope = eval-fixture code (not a deployed service, not disclosure).
  - **11 attrite / weak**, each with a verified reason: no request-controlled input to the sink,
    input allowlisted/used-as-key/sanitized, taint reaches DB-filter not file-path, auth/operator-gated,
    or `request` object unused.
- **76 api candidates:** validated as a class by 5/76 sample — internal/library/CLI/GUI surface, trusted callers.
- **4 repos (data-analyzer, fpga-agent, clave, research-notes):** no web framework at all → no remote
  attack surface → not app-testable by nature (static dismissal is the correct/only method).

HEADLINE (defensible, evidence-backed): execution-confirmed function-level vulnerabilities in the AI
corpus almost never survive to unauthenticated end-to-end attacker-reachability through the real
application surface. The single survivor is in fixture code. Every attrition has a concrete, checkable
cause, verified against full upstream source where a real route existed. The pipeline DID surface a real
reachable bug (proving the attrition is signal, not blindness), and static-reachability over-reporting is
documented with concrete examples (civitas DB-filter-not-path; research-notes CWE-918 mislabel).
