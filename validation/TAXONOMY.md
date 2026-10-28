# VULCAN — Canonical label taxonomy

Every function in the corpus carries **two orthogonal labels**: one for
**Exploitability** (a function-level property, decided by the execution oracle)
and one for **Reachability** (an application-level property, decided by static
call-graph analysis and, for a small verified set, manual tracing / end-to-end
demonstration).

Keeping these on separate axes is deliberate: the distinction between
*conditional exploitability* and *attacker reachability* is VULCAN's central
contribution, so it is the structure of the labels, not a footnote.

---

## Axis A — Exploitability (function-level; from the execution oracle)

| Label | Meaning | Oracle signal |
|---|---|---|
| `CONFIRMED` | Deterministic Tier-2 exploit bank fired against the real code | `confirmed = true` |
| `TRIGGERED_ONLY` | Tier-1 taint reached the hooked sink, but Tier-2 did not confirm | `triggered = true, confirmed = false` |
| `DETECTION_ONLY` | *(legacy label; superseded)* a class with 0 confirmations in a corpus. Do NOT use as an umbrella — distinguish **exercised-zero** vs **not-exercised** via the four coverage states below and `TIER2_COVERAGE_AUDIT.md` | see four-state coverage audit |
| `SAFE` | Executed, no taint trigger (a proven negative under tested conditions) | `executable = true, triggered = false` |
| `NOT_EXECUTABLE` | Could not be executed in the harness (excluded; characterized separately) | `executable = false` |

**Qualifier (not a competing label):** `FIXTURE_CONFIRMED` — exploitation observed
ONLY under synthetic preconditions (e.g. a hand-built DB fixture). It is a weaker
form of evidence and is **never merged into `CONFIRMED`**; it is reported as a
separate study with its assumptions documented.

`CONFIRMED` is the frozen primary ground truth. `NOT_EXECUTABLE` is excluded from
the ground-truth label set (not counted as safe).

**Counting confirmed-by-class — use `exec_cwe`, not candidate tags.** A function may
carry several candidate CWE tags but is confirmed only via the ONE class that actually
fired (`exec_cwe`). Grouping confirmations by raw tags double-counts and can falsely
attribute a confirmation to a detection-only class the function merely carries as a tag
(e.g. a CWE-22-confirmed function also tagged CWE-502 must NOT be counted as a CWE-502
confirmation).

> **COUNTING-UNIT KEY (read before quoting any by-class split).** An `exec_cwe` breakdown
> depends on the unit. **Name-deduped functions (149, v1):** CWE-22=122, CWE-918=24, CWE-78=3;
> CWE-89/502/611 = 0. **Candidate rows (160, v1; = MANIFEST.json):** CWE-22=130, CWE-918=26,
> CWE-78=4; CWE-89/502/611 = 0. **Candidate rows (37, v2):** CWE-22=30, CWE-918=4, **CWE-89=3**.
> Units: candidate row = one (function × injection-point) census row; name-deduped function =
> one (repo, function name); source-unique function = one (repo, sha256(source)).

**Three confirmability states (verified against `tier2/strategies.py`, 2026-10-28).** Do not
describe a class's 0 as a tested negative unless a strategy for it exists. `STRATEGIES` registers
CWE-22/78/89/94/502/611/918/1336 — so every class below has a strategy:

- **(a) confirmable AND confirmed** — strategy exists and fired: CWE-22, CWE-78, CWE-918 (v1+v2); **CWE-89 (v2, 3 cases)**.
- **(b) exercised, zero confirmed** — strategy **invoked** on N identified candidates, 0 confirmed
  (state the tested denominator N; do NOT generalize to all tagged candidates):
  CWE-89 v1 (N=3), CWE-611 v1 (N=3, all diag `EXECUTED_NO_SINK`), CWE-78 v2 (N=28).
- **(c) registered but NOT exercised** — strategy exists but **no candidate invoked it** (not a tested
  negative): CWE-502 (v1 and v2; 0 triggered the deserialization sink), CWE-611 v2 (0 invoked),
  CWE-94 and CWE-1336 (0 tagged candidates in the main census; CWE-1336's SSTI sub-study is excluded
  by `corpus_census.py`, so 1336 is UNKNOWN/out-of-scope for the frozen main counts).
- **(d) no strategy** — none of the mined classes (all 8 are registered in `STRATEGIES`).

Full matrix per CWE × corpus: `validation/TIER2_COVERAGE_AUDIT.json` / `.md`. **Do not describe
CWE-502's 0 as a tested negative** (it was never exercised), and do not describe CWE-611 as a tested
negative beyond its 3 v1 attempts.

CWE-89 is therefore NOT a detection-only/unconfirmable class; it is (a) in v2 and (b) in v1.

**CWE-89 is confirmable (reconciled 2026-10-27).** The earlier "detection-only, as designed"
wording was v1-scoped and is corrected here. The Tier-2 bank *does* carry a SQL strategy
(`tier2/strategies.py::_SqlInjection`, `STRATEGIES["CWE-89"]`), and it **fired on 3 v2
candidates** (foodies `execute(sql)`, hacky-movies `fetch_all(query)`/`fetch_one(query)`),
each proven by injected SQL invoking the seeded `vulcan_proof()` sentinel. v1 has **0** CWE-89
confirmations (verified against `corpus_census_pinned.json`). **Behaviour-coupled limitation:**
all three are *whole-statement raw-SQL execution sinks* — the parameter IS the entire SQL
statement (DIRECT_CALLER_ARG for SQL, analogous to `execute_bash` for CWE-78). The oracle
confirms that attacker-supplied SQL executes; whether a given caller passes attacker-controlled
SQL is the Axis-B reachability question (the 3 are REFUTED on reachability — callers pass
constant SQL with `?` placeholders; see `corpus_v2/REACHABILITY_V2.md`). CWE-502 was never
exercised (0 invocations); CWE-611 was exercised only on 3 v1 candidates (0 confirmed). See
`TIER2_COVERAGE_AUDIT.md`.

---

## Axis B — Reachability (application-level)

Reachability is meaningful only where there is something to reach — i.e. for
`CONFIRMED` (and optionally `TRIGGERED_ONLY`) functions. For `SAFE` /
`NOT_EXECUTABLE` it is recorded but not interpreted.

| Label | Meaning | Source |
|---|---|---|
| `APP_CONFIRMED` | Demonstrated end-to-end: attacker input reaches the sink through a real entry point, impact observed via a side-channel the app can't forge | **manual** |
| `REFUTED` | A real entry point exists, but full-source tracing shows attacker input does NOT reach the sink (allowlist/sanitizer/trusted-caller/taint-to-other-param) | **manual** |
| `STATIC_LEAD` | Static call-graph connects a real entry point to the sink — **UNVERIFIED**. Static reachability over-reports heavily (see note), so this is a lead, not a finding | auto |
| `UNREACHABLE` | No static path from any real entry point to the sink | auto |

**Over-reporting note:** in the hand-validated gold set, almost every
`STATIC_LEAD` collapsed to `REFUTED` on inspection (taint went to a DB filter not
a file path, input was allowlisted, the entry's request object was unused, or the
"entry" was a trusted internal caller). Therefore an unverified `STATIC_LEAD` is
**never** reported as real attacker-reachability.

---

## Verification provenance

Every row also carries `source ∈ {manual, auto}`:
- `manual` — a human (this session) traced the full upstream source at the pinned
  commit and/or demonstrated it end-to-end. Only the gold HTTP candidates so far.
- `auto` — label derived purely from pipeline output. All `STATIC_LEAD` /
  `UNREACHABLE` labels are `auto` unless explicitly upgraded.

---

## Derived sets

- **Gold set** = `CONFIRMED` × {`STATIC_LEAD` | `REFUTED` | `APP_CONFIRMED`}
  (confirmed AND has some reachability signal). This is the hand-validation worklist.
- **Headline reachable** = `CONFIRMED` × `APP_CONFIRMED` (the only cell that claims a
  real, demonstrated attacker-reachable vulnerability).
- **Attrition evidence** = `CONFIRMED` × `REFUTED` (confirmed but demonstrably not
  reachable), each with a documented reason.

---

## Units (read this before quoting any count)

The same corpus yields different "confirmed" numbers depending on the UNIT of
counting. All are legitimate; the thesis must state which it means each time.

| Unit | What one row is | Total | "Confirmed" count | Used for |
|---|---|---|---|---|
| **Candidate** | a (function × injection-point/param) attempt | 2032 | 160 | execution-coverage %, oracle stats |
| **Function** (canonical) | a distinct (repo, function); labels = its strongest candidate | ~1216 | 149 | headline "N confirmed vulnerabilities", the two-axis table |
| **Label-set** | the RQ2 frozen, balanced, deduped subset | 153 (51 pos / 102 neg) | 51 | detector evaluation (RQ2) ONLY |

**Canonical headline unit = Function.** A vulnerability is a property of a function,
not of a single injection-point, so a function with three taintable params is ONE
confirmed vulnerability, not three. `classify_corpus.py` reports function-level as the
headline and candidate-level as a secondary column.

**Correction (unit audit, 2026-10-26, verified against `corpus_census_reproduced.json`).**
The census writes ONE row per mined function record and tries parameters until one
confirms. The 160 confirmed rows are therefore 160 distinct (repo, file, function)
records, not injection points. The 160 → 149 step is a (repo, function-NAME) key, and
it merges 7 genuinely different functions that share a name (for example 4 different
civitas `fetch` functions and 3 different `_sha256` functions). The counts are:

| dedup key | confirmed |
|---|---|
| (repo, file, function), i.e. census rows | 160 |
| (repo, sha256(source)): source-unique, merges only true copies | **156** |
| (repo, function name) | 149 |

The RQ2 label set (`validation/rq2/`) uses the source-unique unit (156). The frozen 149
is kept as-is for traceability. When quoting it, call it "name-deduplicated", not
"distinct functions".

**CWE-22 stratification.** By `analysis/cwe22_subtype.py`, 125 of the 130 CWE-22
confirmed rows are `DIRECT_CALLER_PATH`: the function opens the path its caller passes.
Only 5 are `BASE_JOIN_OVERRIDE`, where an intended base directory is escaped. Every
CWE-22 proof uses an attacker-chosen absolute path; the `../` templates cannot satisfy
any verifier. The label stays CONFIRMED (conditional exploitability). For DIRECT rows,
whether this is a vulnerability is an Axis-B trust-boundary question. See
`validation/rq1/RQ1_BY_GENERATOR.md`.

Conversion is lossless and reproducible: candidate→function dedup keeps the strongest
exploitability label per function (CONFIRMED > TRIGGERED_ONLY > DETECTION_ONLY > SAFE >
NOT_EXECUTABLE) and the strongest/most-verified reachability label. The RQ2 label-set is
a further-filtered subset (balancing + exclusions) documented in the RQ2 section.

---

## Worked examples (this session)

| Function | Exploitability | Reachability | source |
|---|---|---|---|
| nthplusio/skills `send_webhook` (SSRF) | `CONFIRMED` | `REACHABILITY_SUPPORTED`¹ | manual |
| civitas `read_json` | `CONFIRMED` | `REFUTED` | manual (full-source) |
| leapmotor `_write_bytes` | `CONFIRMED` | `REFUTED` | manual (full-source) |
| BugForge `prepare_foundry_workspace` | `CONFIRMED` | `REFUTED` | manual (full-source) |
| CVE-2024-34359 (external control) | `CONFIRMED` | `APP_CONFIRMED` | manual (CVE-Bench grader) |
| typical confirmed internal utility | `CONFIRMED` | `UNREACHABLE` | auto |

¹ **Corrected 2026-10-27.** Earlier rows here read `APP_CONFIRMED`, but no saved end-to-end
artifact (callback/outbound-fetch capture) exists in the repo, and `REACHABILITY_MANUAL_VALIDATION.md`
describes the end-to-end run as deferred. Route→sink reachability is source-supported; the demonstrated
effect is not persisted. Defensible label: REACHABILITY_SUPPORTED. See
`validation/reachability/source_reviews/nthplusio_send_webhook.md` and `REACH_LEDGER.md`.
