# Corpus v2 — attacker-reachability log

Axis B (`validation/TAXONOMY.md`). Labels: APP_CONFIRMED (end-to-end effect observed) /
REFUTED (full-source trace shows that attacker input cannot reach the sink) /
STATIC_LEAD / NOT_TESTED. An app that was not booted is NOT_TESTED, never "unreachable".

## Status (2026-10-27)

**0 APP_CONFIRMED in v2.** No v2 app was booted, so there is no dynamic evidence in either direction.

Static triage of all 37 confirmations (`analysis/reachability.py --depth 4` over each
repo's vendored source):
- **22** are not call-reachable from any HTTP route.
- **15** are call-reachable from a route, but the screen classifies the input reaching
  them as not attacker-controlled: constant 7, operator-cli 2, internal 1, unclassified 5.
  Several of these chains are artifacts of the name-based call graph (generic names
  such as `execute` or `main`).
- **0** are screened as attacker-reachable.

Manual follow-up on the route-reachable SQL helpers:
- `foodies execute(sql)` and `hacky-movies fetch_all` / `fetch_one(query)`: **REFUTED (static, full source).**
- Every route-reachable caller passes CONSTANT SQL with `?` placeholders (for example
  `activate_user`: `... WHERE activation_token_hash = ? AND activation_expires_at > ?`).
  User data is only ever a bound parameter.
- The only f-string query (`PRAGMA table_info({table_name})`) takes an internal table name.

Labels:
- REFUTED (manual, static): `ai_cover` (below) and the 3 SQL helpers.
- Static screen negatives (22 not route-reachable): screen evidence only. The screen
  over-approximates call reachability, so "not reachable" here is a static negative,
  not a demonstrated one.
- NOT_TESTED dynamically: all 37 (no app booted).

## ayushdad609-code/autoclip — `publish_kit.ai_cover` (CWE-22, COMPOSED via helper) — REFUTED (static, full source)

- **Sink chain:** `ai_cover(project_id, job_id, strategy_id)` → `frame_path()` →
  `cover.cover_dir(project_id, …)` → `get_project_directory(project_id)/output/covers/…`.
  Tier-2 confirmed a traversal write through `project_id`.
- **Entry:** `POST /{project_id}/output-variants/{variant_id}/cover/ai`
  (`backend/api/v1/studio.py`) → `jobs.request_ai_cover` → executor → `publish_kit.ai_cover`.
- **Why not reachable:**
  1. The route calls `project_or_404(project_id, db)` first. It requires a `Project` row
     with exactly that id and raises 404 otherwise, so a traversal string never reaches the job.
  2. A FastAPI `{project_id}` path parameter cannot contain `/`.
  3. `job_id` and `strategy_id` come from the stored variant record, not from the request.
- **Residual:** if an attacker could create a project whose id is a traversal string, the
  guard would pass. Project creation was not traced. Label stays REFUTED for the traced
  entry, with this caveat. Not dynamically tested.

## autoclip project_id path-traversal (CWE-22) — APP_CONFIRMED, bounded (2026-10-27, supersedes earlier REFUTED)

CORRECTION: an earlier entry marked this REFUTED based on a normalizing HTTP client
(python-requests collapses `..` before sending). A raw client (curl --path-as-is) does
NOT normalize, so `..` reaches the handler. Demonstrated END-TO-END in the real AutoClip
app at its pinned commit, in an ISOLATED temp data dir + temp SQLite DB:

  Request:  POST /api/v1/projects/../sync-data   (curl --path-as-is)
  Response: 200 OK  {"result":{"success":true,"project_id":".."}}
  Escape:   data_sync_service operated on .../data/projects/../...  = .../data/... (one level
            outside the intended data/projects/ root; confirmed in server logs)
  Effect:   a DB row was created in `projects` (id="..", name="项目_..") — read-only query confirmed.

Levels: (1) validation-inconsistency YES; (2) HTTP reachability YES (raw client);
(3) path escape YES — but CAPPED AT ONE LEVEL (single `{project_id}` str segment; deep
`../../`, encoded `%2f`, and absolute paths all 404, verified); (4) impact: observable but
BOUNDED — out-of-projects DB-record creation + metadata-path probing one level up into the
app's own data dir. NOT arbitrary filesystem access, NOT sensitive-file disclosure, NOT RCE.
Authorization/exposure of the endpoint not characterized.

VULCAN classification: CONFIRMED (exploitability) × APP_CONFIRMED (reachability), demonstrated,
with impact narrowly bounded to a one-level escape into the parent data directory. This is the
SECOND demonstrated attacker-reachable confirmed vulnerability corpus-wide (with the v1 SSRF).
Shares the project_id->path mechanism with the frozen confirmed sinks ai_cover/_write.

Evidence bundle to preserve: pinned repo commit; sync-data route + data_sync_service source;
get_project_directory() (unified_storage.py:65); the HTTP response; the server-log lines
showing the data/projects/../ paths; the read-only DB row. Isolated temp env only; not run
against normal data.
