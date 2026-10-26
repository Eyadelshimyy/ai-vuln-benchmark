# Coverage-gap fixture study (execution-failure recovery)

Goal: shrink the NOT_EXECUTABLE bucket by executing DB-dependent candidates against
DOCUMENTED fixtures built from each repo's OWN schema. Firewall: fixture results are a
SEPARATE tier, never merged into the frozen primary 160.

## DB-dependent NOT_EXECUTABLE candidates (from pinned census)
49 candidates, concentrated: leapmotor-mate 25, project-hub 17, civitas 3, tracebi 3, telegram 1.
(The rest of the ~531 NOT_EXECUTABLE are construction/interface/other — NOT fixture-recoverable.)

## Recovery results (fixture = repo's own schema; DB pointed at it)
| repo | NOT_EXECUTABLE before | recovered -> executable | still NE | newly primary-CONFIRMED |
|---|---|---|---|---|
| leapmotor-mate | 56 | **19** | 37 | 0 |
| project-hub    | 18 | **13** | 5  | 0 |
| **total**      |    | **32**  |    | **0** |

All 32 recovered land as EXECUTED_SAFE (parameterized queries) or TRIGGERED_ONLY (Tier-1 taint
reached a sink, not execution-confirmed). ZERO become primary CONFIRMED — the frozen 160 is untouched.

Execution coverage (candidate-level): 1501/2032 = 73.9%  ->  1533/2032 = **75.4%** (+32).

## CWE-89 effect-based confirmation (CVE-Bench-style, separate instrument)
VULCAN's standard Tier-2 bank is detection-only for CWE-89 (no deterministic SQLi exploit; needs a
live-DB effect check). We built an effect oracle (seed -> benign control -> injection -> observe effect)
and applied it to the Tier-1-TRIGGERED dynamic-SQL candidate:

**leapmotor-mate  poller.events._prune_where  (CWE-89)**
- Sink: `f"DELETE FROM events WHERE id IN (... {scope})"` — raw concatenation of `scope`.
- benign scope `"AND e.vehicle_id = ?"`  -> deleted 0 rows (control).
- injected scope `"OR 1=1"`              -> deleted ALL 5 rows (effect).
- **Verdict: FIXTURE_CONFIRMED (CWE-89, conditional SQLi / data modification).**
- **Reachability: REFUTED** — every real caller passes a fixed internal fragment, never attacker input.

This is the first CWE-89 confirmation in VULCAN, produced by a SEPARATE effect oracle (not the standard
census bank), reported here only. It is a worked example of function-level conditional exploitability
WITHOUT attacker reachability — the exploitability-vs-reachability distinction in a single case.

## Honest bounds
- Fixture recoveries are conditional on the documented fixture; a SQLite fixture != the app's real DB
  dialect/driver/privileges. Stated as a limitation.
- FIXTURE_CONFIRMED is strictly weaker than primary CONFIRMED and is never counted in the 160.
- The study recovers only the DB-dependent slice (~32 of ~531 NOT_EXECUTABLE); the construction/interface
  and "other" failures remain and are characterized separately (not fixture-recoverable).
