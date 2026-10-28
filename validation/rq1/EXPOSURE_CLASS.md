# What VULCAN's confirmations actually are: exposure class (v1 + v2)

Status: verified 2026-10-27.
- CWE-22 rows: classified by `analysis/cwe22_subtype.py`, with 10 random DIRECT rows hand-checked (all correct).
- Every non-CWE-22 confirmation (37 rows: v1 30, v2 7): **read by hand**.

## Two kinds of confirmed weakness

- **DIRECT_CALLER_ARG:** the parameter *is* the whole resource the dangerous operation
  acts on. Examples: `open(path)`, `urlopen(url)`, `subprocess.run(command)`, and
  `connection.execute(query)` in a DB helper. The oracle proves that whatever the caller
  passes is acted on without restriction. It is a vulnerability **only if an untrusted
  caller controls that argument**, which is the reachability axis.
- **COMPOSED (classic injection):** attacker data is spliced into a resource the function
  otherwise fixes, such as `os.path.join(BASE, name)` or `f"cmd {x}"` run by a shell.
  The author intended a restriction and the attacker escapes it.

## Counts (confirmed)

| class | v1 | v2* | DIRECT_CALLER_ARG | COMPOSED |
|---|---|---|---|---|
| CWE-22 path | 130 rows | 30 | ≤127 + ≤29 | **3 + 1**, hand-reviewed (`cwe22_composed_manual.json`) |
| CWE-918 SSRF | 26 | 4 | all 30: `url` parameter passed whole to `urlopen`/`requests`/`httpx` | 0 |
| CWE-78 command | 4 | 0 | all 4: `run(command)`, `execute_bash(command)` ×2, `tmux_new_session(..., shell_command)` | 0 |
| CWE-89 SQL | 0 | 3 | all 3: DB helpers `execute(sql)`, `fetch_all(query)`, `fetch_one(query)` | 0 |

\* v2 = canonical census `corpus_census_v2.json` (writable corpus copy; protocol deviation 5).

**Hand review of the heuristic's BASE_JOIN rows (2026-10-27).** 3 of the 6 rows it marked BASE_JOIN were confirmed through a parameter used as a WHOLE path, so they are DIRECT. The heuristic flags any base composition in the function, not the composition on the confirmed parameter. `autoclip ai_cover` is COMPOSED but was marked DIRECT (helper). Composed cases are therefore taken from the hand-reviewed list, not the heuristic.

**Heuristic limit.** `cwe22_subtype.py` is intraprocedural. A base path composed inside a
helper (as in `ai_cover`) is counted as DIRECT. So the DIRECT counts are an UPPER BOUND. The
10-row hand check found no such miss, but 10 rows bound the error rate only loosely
(95% upper bound ≈ 26%).

CWE-22 has 8 UNKNOWN rows in v1 (unparsable before the dedent fix); after the fix, all
v1 rows classify.

## Reading

- Execution-confirmed exploitability in this AI-authored code is almost entirely
  **capability exposure through a parameter**: helpers that open, fetch or execute whatever
  they are given. Classic composed injection is rare: 4 path cases, and 0 command, SQL or
  SSRF cases.
- This is not an oracle artifact. The soundness battery shows that composed-but-guarded
  code is correctly NOT confirmed, and composed-unguarded code IS confirmed (`v_read`,
  `v_write`, `v_os`). The corpus simply contains few composed sinks reachable from a parameter.
- **Consequence for RQ2:** a detector asked "is this exploitable *as written*?" may
  reasonably answer "only if the caller is untrusted". This is why the pre-registered
  primary condition states the oracle's threat model (`v2-threat`).
- **Consequence for the paper:** report confirmations as conditional exploitability, and
  let the reachability axis decide which ones are attacker-reachable vulnerabilities.
