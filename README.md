# VULCAN: A Benchmark for Execution-Verified AI-Generated Vulnerabilities

**VULCAN — VULnerability Confirmation via Actual executioN.** Confirms
vulnerabilities in AI-generated code by *executing* it and watching the
real dangerous operation fire at runtime, rather than pattern-matching the
source.

Companion code for the bachelor's thesis **"An Empirical Evaluation and
Benchmark of AI-Generated Vulnerabilities in Code"** (Eyad Hassan
Abdelfatah, GIU Cairo, Cybersecurity B.Sc., supervised by Dr. Marwa Zamzam
and Dr. Ahmed Maghawry).

## The gap this project addresses

The literature review behind this thesis (`docs/Literature_Review_Sheet.*`)
covers four recent papers benchmarking security in LLM-generated code
(Sabra et al. 2025; Shahid et al./LLM-CSEC 2025; Li et al./SafeGenBench
2025; Yoo & Kim 2025). All of them stop at **detection**: static analysis,
multi-tool SAST, SAST+LLM-judge, or CWE-to-CVE historical mapping flags
code as "containing a vulnerability pattern." None of them confirm that a
specific flagged snippet is **actually exploitable**.

This matters because detection tools disagree with each other constantly
(LLM-CSEC: three SAST tools found zero common CWEs across their sample;
SafeGenBench: SAST and an LLM judge only agreed 61% of the time), which
means a meaningful fraction of what gets reported as "vulnerabilities
found" in prior benchmarks could be false positives — patterns that look
dangerous but are not, in that specific piece of code, actually
attackable.

**This project adds the missing confirmation stage, two different ways,**
and compares them:

1. **Static detection (SAST)** — `taint_scan.py`, a hand-built AST-based
   taint tracker scoped to the three CWEs in this thesis, used as the
   first-pass detector and as the baseline every later stage is measured
   against.
2. **Dynamic exploit confirmation (DAST)** — `exploit_confirmation/`, a
   harness per CWE that actually runs flagged code against a crafted
   malicious input in a sandboxed environment and checks for an
   unambiguous, mechanically-verifiable side effect.
3. **Interactive/runtime verification (IAST)** — `analysis/iast/`, a
   newer, more general runtime layer that monkey-patches the real
   dangerous stdlib calls themselves (`subprocess.Popen`, `os.system`,
   `sqlite3.Connection.execute`, `open`, `Path.write_text`, process
   creation at the OS-audit-hook level) so that ANY code path reaching
   them with attacker-shaped input is caught, no matter how many layers
   of wrapper functions sit in between — including cases SAST is
   *structurally* unable to see at all (see "What IAST catches that SAST
   can't," below).

Scope: **CWE-89 (SQL Injection), CWE-78 (OS Command Injection), CWE-22
(Path Traversal)**, in Python, tested against both a hand-built
calibration set, 6 real disclosed CVEs, and real, unreviewed functions
mined directly out of popular open-source Python packages.

**Explicit scope note (methodology):** this project measures whether a
given function, if deployed as part of a real application, would be
exploitable — not full production reachability from a live network entry
point. No server, no HTTP listener, no CLI front-end is stood up; each
candidate function is called directly, in a sandboxed process, with a
value standing in for "whatever an attacker could get into this
parameter." This keeps the thesis question answerable at the scale a
benchmark needs (dozens to hundreds of mined functions, and eventually
many LLM completions per prompt) rather than requiring a bespoke running
application and attack chain be built for every single sample.

## Three detection strategies, compared head-to-head

| Layer | File(s) | How it decides | Catches | Misses |
|---|---|---|---|---|
| **SAST** | `analysis/taint_scan.py` | Static AST walk: seeds parameters as tainted, propagates through assignment/string-building, flags tainted values reaching a named sink call | Fast, no execution needed, works on code that can't even run | Cross-file/parent-class taint (e.g. a field set by `load_conf()` in a class not in the file), `*args`/`**kwargs` signatures, anything requiring actual runtime values |
| **DAST** | `exploit_confirmation/harness_*.py` | Runs a hand-adapted sample against a real payload, checks for a real side effect | Zero false positives by construction — if it fires, it's real | Needs each sample adapted to a fixed function contract; doesn't scale to arbitrary, unseen signatures |
| **IAST** | `analysis/iast/*.py` | Patches the real stdlib sink functions; calls the REAL, unmodified function-under-test with a marker value in one parameter; watches whether the marker reaches a real dangerous call | Generalizes to arbitrary, unseen real-world functions (see mined results below); catches wrapper-function chains and cross-method/cross-class taint for free, because it doesn't trace dataflow at all — it just watches what actually executes | "Taint by value, not by type" (see `tainted.py`) — if an input is transformed (hashed, encrypted) before reaching a sink, the marker doesn't survive and a real vulnerability could be missed; a same-user sandboxed child process, not a full OS jail |

### What IAST catches that SAST structurally can't

Verified directly against real, disclosed CVEs this session (not a
hypothetical):

- **invoke's `Runner.local()`** — signature is `def local(self, *args,
  **kwargs)`. `taint_scan.py`'s seeding logic only reads named
  positional/keyword parameters, so it silently skips this function
  entirely — 0 findings, not because it's safe, but because the scanner
  can't even look. IAST doesn't care about the static signature shape at
  all; it just watches whether the marker you passed reaches
  `subprocess.Popen` for real. (`test_against_real_misses.py`)
- **GitPython's `clone()`/`clone_from()`** missing `--separate-git-dir`
  from its unsafe-options denylist (GHSA-8mcc-hrx5-hvxc, CVE-2026-78677)
  — there is no `open()`/`write_text()`/`Popen()` call anywhere in the
  *visible* vulnerable code path; the danger is an argument-injection
  effect several layers downstream inside GitPython's own internals.
  `taint_scan.py` finds nothing. IAST catches it because it watches the
  real `Popen` call that eventually happens, regardless of how many real
  internal layers sit in between. (`test_cwe22.py`, TEST 3)
- **Glances' Cassandra exporter** (CVE-2026-35588) — `self.table`/
  `self.keyspace` are populated by `load_conf()`, a method on a parent
  class not even present in the file, almost certainly via a dynamic
  `setattr()`-style mechanism invisible to any single-file AST walker.
  `taint_scan.py` is left as a documented, unfixed miss here. IAST
  sidesteps the whole problem by construction: it doesn't care *how*
  `self.table` got its value, only what's actually sitting in it when
  the real `execute()` call fires. (`test_cwe89.py`, TEST 2)

Where SAST already works correctly (PraisonAI's SQL store,
`StreamBackedCorpusView._open()`, khoj's path traversal, invoke's
`start()`), IAST agrees with it — these are included as sanity checks,
not just wins.

## IAST architecture

```
analysis/iast/tainted.py   The taint primitive: a random per-run marker
                            string, tracked by VALUE CONTENT (not type).
                            Deliberately NOT a str subclass — verified
                            directly that CPython's f-string bytecode
                            (BUILD_STRING) silently discards subclass
                            identity, which would cause false "safe"
                            verdicts on the single most common
                            string-building pattern in real CVEs. Trades
                            true dataflow tracking for something that
                            actually survives f-strings, %-formatting,
                            and os.path.join.

analysis/iast/sinks.py     Monkey-patches the real stdlib call sites
                            behind all three CWEs, at the LOWEST common
                            implementation (patching subprocess.Popen.
                            __init__ once covers run/call/check_call/
                            check_output and any direct Popen(...) for
                            free). A sink hit WITH tainted input records
                            the hit and raises SinkTriggered instead of
                            performing the real dangerous action; a hit
                            WITHOUT tainted input lets the real call
                            through, so legitimate control flow still
                            works. A parallel, lower-level layer uses
                            sys.addaudithook() to watch 'os.exec' and
                            'os.posix_spawn' directly at the C level —
                            this catches process creation that bypasses
                            every Python-level patch entirely, verified
                            against invoke's own pty-based execution path
                            (pty.fork() + os.execve()), including relaying
                            hits across a real fork() boundary via a
                            scratch-dir JSONL file, since a hook firing
                            inside a forked child lives in separate
                            memory from the parent.

analysis/iast/harness.py   run_with_taint(func, args, kwargs) — calls a
                            function-under-test for real, inside the
                            patched-sink context, and reports what
                            happened. You choose which parameter carries
                            the taint marker, the same way a real
                            attacker targets one specific input — the
                            harness never guesses.

analysis/iast/test_cwe22.py, test_cwe89.py, test_against_real_misses.py
                            Hand-built, human-reviewed reconstructions of
                            6 REAL, disclosed CVEs (GHSA-62mm-xwmv-crhg,
                            CVE-2026-63312, CVE-2026-78677, CVE-2026-40315,
                            CVE-2026-35588, plus invoke's start()/local())
                            — every one of these a human read line-by-line
                            before deciding how to call it safely. All 6
                            substantive assertions TRIGGERED correctly;
                            both negative controls (GitPython clone_from
                            with no dangerous kwarg; a safe Cassandra table
                            name) correctly did NOT trigger.

analysis/mine_prompts.py   Mines real, documented functions straight out
                            of a real GitHub repo (no human curation) that
                            touch one of the three CWE sink taxonomies,
                            strips each to a bare signature+docstring
                            prompt (for the eventual LLM-generation
                            experiment), and — new — mines a
                            "construction recipe" for the function's
                            enclosing class straight out of the repo's OWN
                            test suite (e.g. finds Database(memory=True)
                            in sqlite-utils's own tests/conftest.py),
                            restricted to call sites where every argument
                            is a literal or a tmp_path-style fixture, so
                            the harness can construct a REAL instance via
                            the REAL __init__ instead of guessing.

analysis/iast/auto_harness.py
                            Runs the IAST harness AUTOMATICALLY against
                            every function mine_prompts.py finds — code
                            nobody has read first, unlike the 6 hand-built
                            CVE tests above. This is structurally riskier
                            (see the file's own docstring for the full
                            reasoning) and reports one of three honest
                            outcomes per candidate: TRIGGERED,
                            NOT_TRIGGERED, or COULD_NOT_EXECUTE (a crash
                            is NOT the same as "safe" — collapsing it into
                            NOT_TRIGGERED would be a false-safe verdict).
                            Runs every candidate inside a disposable,
                            resource-limited child process with its own
                            scratch cwd (--install-missing optionally
                            pip-installs the real target package first,
                            to run the real, unmodified code instead of a
                            best-effort reconstruction).

                            --live-demo mode: a deliberately separate,
                            single-candidate, human-confirmed path
                            (--live-demo --target-param <name> --payload
                            '<real value>' --confirm-live) that lets a
                            confirmed hit's real dangerous action actually
                            execute (a real file write, a real shell
                            command, real SQL) instead of being
                            intercepted, so you can see the concrete,
                            physical proof on disk — used for the live
                            demo runs below.
```

### Honest limits of the IAST sandbox

Each candidate runs in a same-user, same-kernel child process with
resource limits and a scratch working directory — not a container, a VM,
or a seccomp/gVisor-style jail. This raises the bar against *accidental*
damage from a merely-buggy generated completion (the realistic threat
model for this thesis — running an unreviewed LLM completion, not a
deliberately adversarial payload designed to escape the sandbox itself).
It is not a security boundary against a kernel-level exploit. The
backstop is that this whole pipeline already runs on disposable
infrastructure, not production systems.

The taint marker is tracked by *value content*, not dataflow: if a
function hashes, encrypts, or otherwise transforms an input before it
reaches a sink, the marker substring won't survive and a real
vulnerability could be missed (a false negative). This is a real,
different failure mode from `taint_scan.py`'s limitations, not a
strictly-better replacement for it — both are reported honestly as
tradeoffs, not papered over.

## Results so far

### Hand-built calibration set (SAST + DAST, 12 samples, 2 per CWE × vulnerable/safe)

```
Exploit-confirmation harness accuracy against hand-built ground truth: 12/12 (100.0%)
```

Static detection alone flagged 8/12 samples as suspicious; only 6/12 were
actually exploitable. `cmdi/safe_2.py` and `path_traversal/safe_2.py`
were pattern-matched by Semgrep (an allowlist-validated `shell=True`
call; an `os.path.basename` call that also happens to look like unsafe
path joining) despite being genuinely safe — the exploit-confirmation
stage correctly cleared both, and correctly confirmed every vulnerable
sample. 100% agreement with ground truth on the exact question
detection-only prior work cannot answer.

### Hand-built real-CVE reconstructions (IAST, 6 disclosed CVEs)

All 6 substantive assertions TRIGGERED correctly; both negative controls
correctly did NOT trigger. 3 of the 6 (GitPython CWE-22, Glances
Cassandra CWE-89, invoke's `*args`/`**kwargs` `local()`) are cases
`taint_scan.py` cannot detect at all — see "What IAST catches that SAST
structurally can't," above.

### Automated mining + IAST, run against real, unreviewed code from 4 real packages

No human read any of this code before the harness called it. Numbers are
per mined candidate × injection point tried (`ground_truth` mode — the
real, original function body, not an LLM completion yet):

| Repo | TRIGGERED | NOT_TRIGGERED | COULD_NOT_EXECUTE | Total |
|---|---|---|---|---|
| sqlite-utils | 9 | 15 | 18 | 42 |
| invoke | 1 | 3 | 7 | 11 |
| fabric | 0 | 5 | 2 | 7 |
| tinydb | 2 | 0 | 0 | 2 |
| **Total** | **12** | **23** | **27** | **62** |

The construction-recipe mining improvement (mining a real,
safe-to-call-literally construction call for a candidate's enclosing
class straight out of the target repo's own test suite) was validated to
generalize beyond its original target: tested fresh against `peewee` (a
repo never tested before this validation), it correctly built a real
`DataSet('sqlite:///:memory:')` instance from a recipe mined directly out
of peewee's own `tests/dataset.py`, converting what would otherwise be a
`COULD_NOT_EXECUTE` into a genuine `TRIGGERED`.

### Live demos (real side effects, human-confirmed)

Run with `--live-demo`, which lets a confirmed hit's real dangerous
action actually complete instead of being intercepted, so the physical
evidence can be inspected on disk afterward:

- **tinydb, CWE-22 (path traversal)**: TRIGGERED, confirmed via the real
  write landing outside the intended scratch directory.
- **invoke, CWE-78 (command injection)**: TRIGGERED via the
  `sys.addaudithook()` layer, specifically through invoke's own
  `pty.fork()` + `os.execve()` execution path — the exact real-world
  mechanism documented in invoke's own source, not a synthetic stand-in.

## Repository layout

```
analysis/
  taint_scan.py                 SAST: the hand-built AST taint tracker.
  mine_prompts.py                Mines real functions + construction
                                  recipes out of a real GitHub repo.
  fetch_real_cves.py              Pulls real, CVE-confirmed vulnerable
                                  Python code from GitHub Security
                                  Advisories (run on your own machine —
                                  api.github.com isn't reachable from the
                                  sandboxed analysis environment).
  triage_real_cves.py              Ranks fetched CVE candidates by how
                                    usable they are for ground truth.
  extract_real_cve_functions.py    Pulls the top-ranked candidates' real
                                    pre-fix source into standalone .py
                                    files, with provenance headers.
  evaluate_real_cve_accuracy.py    SAST accuracy against the extracted
                                    real-CVE ground truth.
  scan_mined_baseline.py           SAST run against the REAL (human-
                                    written) code behind every mined
                                    prompt — a baseline rate to compare
                                    the eventual AI-generated rate
                                    against.
  iast/
    tainted.py                    The taint marker primitive.
    sinks.py                       Monkey-patched real sink functions +
                                    the sys.addaudithook() layer.
    harness.py                     run_with_taint() — the core call+watch
                                    primitive.
    auto_harness.py                Automated IAST runner against mined
                                    candidates, plus --live-demo mode.
    test_cwe22.py, test_cwe89.py,
    test_against_real_misses.py     Hand-built, human-reviewed real-CVE
                                     reconstructions (6 CVEs total).

calibration/
  python_handbuilt/              Hand-built vulnerable/safe pairs, 2 per
                                   CWE — ground truth for the SAST+DAST
                                   pipeline.
  mined_prompts/<repo>/          Output of mine_prompts.py: one JSON per
                                   mined candidate function, a SUMMARY.md,
                                   and auto_harness_results.json per repo
                                   (currently: tinydb, sqlite-utils,
                                   invoke, fabric).
  real_cve/                       Shortlist output of triage_real_cves.py.
  real_cve_extracted/<cwe>/      Extracted real pre-fix vulnerable source
                                   per CVE, with provenance headers.
  juliet/                         Notes + plan for extending Java
                                   calibration with NIST's Juliet Test
                                   Suite (not vendored — public domain,
                                   distributed separately by NIST).

detection/
  semgrep_rules/                 Local, offline Semgrep rules per CWE.
  run_detection.py                Runs the rules against a directory of
                                   samples.

exploit_confirmation/
  harness_sqli.py, harness_cmdi.py, harness_path_traversal.py
                                  DAST harnesses: fire canonical payloads
                                   at a sample in a sandboxed environment,
                                   report confirmed_exploitable based on
                                   an unambiguous side effect only.

generation/
  generate.py                    Calls OpenAI / Anthropic / any OpenAI-
                                   compatible open-source endpoint for
                                   every prompt; skips (doesn't crash) any
                                   model with no API key configured.
  samples/                        Where generated code lands.

prompts/                         JSON prompt sets (one per CWE) asking
                                   for the relevant functionality WITHOUT
                                   mentioning security, so results reflect
                                   default LLM behavior.

scripts/
  run_calibration.py              Runs SAST+DAST against the hand-built
                                   set, prints the comparison table above.
  run_taint_calibration.py        SAST-only calibration run.

docs/                             Literature review (xlsx/md/tex/pdf) and
                                   the one-page initial plan from the
                                   thesis-planning phase.

results/
  raw/                            Detection-stage output.
  confirmed/                      Exploit-confirmation-stage output.

thesis_pipeline_diagram.html      Visual diagram of the full pipeline.
```

## Running it

```bash
pip install -r requirements.txt

# --- SAST + DAST pipeline ---

# 1. Prove the SAST+DAST pipeline works against hand-built ground truth
#    (no API keys, no network needed):
python3 scripts/run_calibration.py

# 2. Pull real, disclosed CVEs for ground truth (run on your own machine,
#    not the sandbox -- needs api.github.com):
export GITHUB_TOKEN=ghp_xxx          # optional, raises rate limit 60 -> 5000/hr
python3 analysis/fetch_real_cves.py
python3 analysis/triage_real_cves.py
python3 analysis/extract_real_cve_functions.py --top 20
python3 analysis/evaluate_real_cve_accuracy.py

# --- IAST pipeline ---

# 3. Run the 6 hand-built real-CVE reconstructions:
python3 analysis/iast/test_cwe22.py
python3 analysis/iast/test_cwe89.py
python3 analysis/iast/test_against_real_misses.py

# 4. Mine real, unreviewed candidate functions out of any real repo:
python3 analysis/mine_prompts.py --repo-url https://github.com/<org>/<repo>.git --repo-name <repo> --limit 25

# 5. Run the automated IAST harness against everything just mined:
python3 analysis/iast/auto_harness.py --candidates calibration/mined_prompts/<repo>/ --use ground_truth --install-missing

# 6. (Optional) see one confirmed hit's real side effect on disk:
python3 analysis/iast/auto_harness.py --candidates calibration/mined_prompts/<repo>/<file>.json \
    --live-demo --target-param <param_name> --payload '<real malicious value>' --confirm-live

# --- Original SAST-only baseline + LLM generation wiring ---

# 7. Generate real LLM code (needs at least one of these env vars set):
export OPENAI_API_KEY=...        # for GPT
export ANTHROPIC_API_KEY=...     # for Claude
export OPENSOURCE_BASE_URL=...   # e.g. a local vLLM / Together / Groq endpoint
python3 generation/generate.py --all

# 8. Detect candidate vulnerabilities in what was generated:
python3 detection/run_detection.py --dir generation/samples --out results/raw/detection.json
```

**Note on `--use generated`:** `auto_harness.py` already supports running
an LLM-generated implementation in place of a mined function's real body
(`--use generated`, expecting a `generated_source` field in the candidate
JSON) — this is the actual thesis experiment. The wiring that feeds LLM
completions back into the mined candidate JSON files is the next piece of
engineering (see Roadmap).

## Roadmap

- [x] SAST stage (hand-built AST taint tracker, 3 CWEs, Python)
- [x] DAST exploit-confirmation harnesses (3 CWEs, Python) — validated
      100% against hand-built ground truth
- [x] Real-CVE ground truth pipeline (fetch → triage → extract → evaluate)
- [x] IAST runtime-verification layer (monkey-patched real sinks +
      `sys.addaudithook()` parallel coverage)
- [x] 6 hand-built, human-reviewed real-CVE reconstructions — 100% correct,
      including 3 cases SAST structurally cannot detect
- [x] Automated mining (real functions + construction recipes) from real
      GitHub repos, no human curation
- [x] Automated IAST harness run against 4 real repos (62 candidates:
      12 TRIGGERED / 23 NOT_TRIGGERED / 27 COULD_NOT_EXECUTE)
- [x] `--live-demo` mode for human-confirmed, real-side-effect proof runs
- [x] Generation stage wired up for GPT / Claude / open-source models
- [ ] Feed LLM-generated completions into mined candidates'
      `generated_source` field and run `--use generated` for real
- [ ] Full run across GPT / Claude / open-source × 3 CWEs, real mined
      prompts
- [ ] Results analysis: TRIGGERED-vs-flagged rate by model/CWE, SAST vs
      IAST agreement/disagreement breakdown
- [ ] Tier 2: real OS-level sandboxing (Docker/firejail/bubblewrap) for
      the auto-harness child process, beyond today's resource-limited
      same-kernel process
- [ ] Java detection + IAST (lower priority; Python is the primary
      language in scope)
- [ ] Write-up: final thesis chapters

## License

MIT for the code in this repository. NIST's Juliet Test Suite (referenced
but not vendored, see `calibration/juliet/README.md`) is public domain
and distributed separately by NIST. The real-world repos mined by
`mine_prompts.py` (tinydb, sqlite-utils, invoke, fabric, and others) and
the real CVE source extracted by `extract_real_cve_functions.py` remain
under their own original licenses — none of that code is redistributed
here for any purpose other than this thesis's own academic security
research.
