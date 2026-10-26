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
| `DETECTION_ONLY` | CWE class the oracle cannot confirm by construction (CWE-89/502/611) | class ∈ detection-only set |
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
confirmation). Confirmed-by-class breakdown (by `exec_cwe`): CWE-22=122, CWE-918=24,
CWE-78=3 (total 149); CWE-89/502/611 = 0 confirmations (detection-only, as designed).

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

Conversion is lossless and reproducible: candidate→function dedup keeps the strongest
exploitability label per function (CONFIRMED > TRIGGERED_ONLY > DETECTION_ONLY > SAFE >
NOT_EXECUTABLE) and the strongest/most-verified reachability label. The RQ2 label-set is
a further-filtered subset (balancing + exclusions) documented in the RQ2 section.

---

## Worked examples (this session)

| Function | Exploitability | Reachability | source |
|---|---|---|---|
| nthplusio/skills `send_webhook` (SSRF) | `CONFIRMED` | `APP_CONFIRMED` | manual |
| civitas `read_json` | `CONFIRMED` | `REFUTED` | manual (full-source) |
| leapmotor `_write_bytes` | `CONFIRMED` | `REFUTED` | manual (full-source) |
| BugForge `prepare_foundry_workspace` | `CONFIRMED` | `REFUTED` | manual (full-source) |
| CVE-2024-34359 (external control) | `CONFIRMED` | `APP_CONFIRMED` | manual (CVE-Bench grader) |
| typical confirmed internal utility | `CONFIRMED` | `UNREACHABLE` | auto |
