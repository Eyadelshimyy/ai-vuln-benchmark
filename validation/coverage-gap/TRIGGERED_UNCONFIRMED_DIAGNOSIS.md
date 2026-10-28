# Why 117 v1 functions TRIGGERED but did not CONFIRM (diagnosis, 2026-10-26)

**Method.** I re-ran the frozen oracle on exactly these 117 rows:

    python3 analysis/corpus_census.py --tier2 --only-files <list> --out <scratch>

This records the per-parameter Tier-2 diagnosis (`tier2_evidence`). It is a diagnostic
run; the frozen census is unchanged.

**Stability.** All 117 are TRIGGERED again and 0 confirm, the same as the frozen census.

## Root causes (one function can show several buckets, one per parameter)

| exec class | n | bucket | reading |
|---|---|---|---|
| CWE-78 | 55 | CMD_ARGV_NOT_SHELL | Attacker input reaches `subprocess` only as an **argument in a list**, with no shell. Not OS command injection. Possible CWE-88 (argument injection), which is program-specific and not tested by the oracle. |
| CWE-78 | 8 / 7 | EXECUTED_NO_SINK / SINK_REACHED_INPUT_ABSENT | Input never reached a command sink. Genuine negatives. |
| CWE-918 | 22 (+3 LIVE_RUN_FAILED) | EXECUTED_NO_SINK | No request carrying our URL was made. Genuine negatives; the 3 run failures are inconclusive. |
| CWE-22 | 13 (+2, +1) | SINK_REACHED_INPUT_ABSENT / EXECUTED_NO_SINK / INPUT_NOT_IN_SQL | Input never reached the path sink. Genuine negatives. |
| CWE-22 | 29 | PATH_TRAVERSAL_NO_OBSERVABLE | See below. |
| CWE-89 | 3 / 2 / 1 | INPUT_NOT_IN_SQL / PAYLOAD_MISMATCH / NO_PROOF | Too few to generalise. |
| CWE-611 | 3 | EXECUTED_NO_SINK | Genuine negatives. |

## The 29 ambiguous CWE-22 rows

These were checked by replaying each one and comparing the resolved path that was
actually opened with the payload.

| pattern | repos | what the function does |
|---|---|---|
| payload + fixed suffix | 23 | uses the parameter as a **directory or prefix**, e.g. `out_dir/index.json`, `directory/input.dat` |
| payload embedded in a filename | 3 | `f"players_raw_{stem}.csv"` inside a fixed directory |
| exact path, open failed | 3 | no observable effect possible |

**Decision: no oracle change.** An oracle extension that confirms "the program wrote into an
attacker-chosen out-of-tree directory" would be sound, and it would raise the CWE-22 count. But
every such case is a `DIRECT_CALLER_PATH` finding: the caller chooses the directory. That
is the conditional class that already makes up 125 of the 130 CWE-22 confirmations. It would
add volume to the weakest construct, not catch hidden attacker-reachable vulnerabilities.
Recorded here instead, so that a reviewer can see the gap was examined.
