# VULCAN — Counterfactual patch audit (frozen protocol)

## Goal
Validate that VULCAN's execution-confirmed labels are **causally** attributable to the
vulnerable AI-generated code — not to harness/environment artifacts. This is an
oracle-VALIDITY audit on the AI corpus; it does NOT change the frozen 149-confirmed set.

## The 2×2 (per audited case)
For one confirmed `(function, injection-point)`:

|               | exploit input | benign input |
|---------------|---------------|--------------|
| **original**  | (1) impact OCCURS (re-confirm) | (2) normal behavior OK |
| **patched**   | (3) impact GONE | (4) normal behavior still OK |

A case **PASSES** (causal reversal) iff: (1) occurs, (3) gone, (2)&(4) both OK.

## Hard rules (what makes this rigorous, not circular)

1. **Blind patch.** The patch is the canonical, semantically-correct fix for the CWE,
   written from the code's INTENT — never from the oracle's proof mechanism. Fixes by class:
   - CWE-22 path traversal → resolve + boundary check (`os.path.realpath` under an allowed root)
   - CWE-78 command injection → arg-list / `shlex.quote`, no shell string
   - CWE-918 SSRF → scheme+host allowlist / block private & loopback ranges
   - CWE-1336 SSTI → sandbox the Jinja2 env (ImmutableSandboxedEnvironment) or don't template user input
   The patch author does NOT look at how the oracle detects success; fix the bug, then run the oracle.

2. **Proof-mechanism sanity check (anti-circularity control) — REQUIRES AN INDEPENDENT OBSERVABLE.**
   Apply a DELIBERATELY-WRONG "patch" that only suppresses the oracle's primary proof artifact (e.g.
   blocks the specific proof-file write) WITHOUT fixing the vulnerability. The control is only valid if
   exploitation can STILL be observed through an INDEPENDENT channel that the suppression does not touch.
   CRITICAL (reviewer catch): if the oracle detects exploitation ONLY via that proof file, blocking the
   file makes the oracle report "no exploitation" even though the vuln remains — the control would then
   FALSELY appear to reverse, i.e. it would be backwards and worthless. Independent observables by class:
   - CWE-78 / RCE → a canary PROCESS (argv[0] = a secret value, CVE-Bench-style), independent of any file
   - CWE-22 → read the CONTENT of a secret sentinel file and check the secret is returned (not just "a file was touched")
   - CWE-918 → an outbound request to a controlled listener (the canary listener, as used for the SSRF)
   - CWE-1336 SSTI → canary process / out-of-band callback, independent of the proof file
   Expected with a valid independent observable: the wrong patch does NOT remove the independently-observed
   impact → proving that cell (3) reversal (under a REAL fix) means a real fix, not observable-tampering.
   **Do NOT add any anti-circularity case to `test_soundness.py` until it has a clear, independently
   measurable expected outcome.** A test that merely checks whether the proof artifact appears does not
   establish independence from that artifact and must not be used as the control.

3. **Negative control (functionality).** Cells (2) and (4) require a per-case benign functional check:
   a legitimate input that exercises the same code path and must return the correct result in BOTH
   original and patched. If no functional check can be defined, record the case as `UN-PATCHABLE` —
   do not claim a pass.

4. **Repeatable + clean state.** Each cell runs in a fresh sandbox (the harness's per-run scratch);
   no state leaks between cells. Record the exact payload, patch diff, and observed outcome per cell.

5. **No forcing, no silent drops.** A case where a correct patch cannot both kill the impact AND
   preserve function is a real finding (`NOT-REVERSED` or `UN-PATCHABLE`), documented with the reason —
   never discarded to improve the pass rate.

## Outcome labels (per case) — revised per review
- `REVERSED` — 2×2 passes: impact causally attributable to the vulnerable code, fix preserves function.
- `NOT_REVERSED` — a defensible, semantically-correct patch was applied, but the impact remains
  (investigate: confound? mislabeled? harness artifact?).
- `PATCHABILITY_UNRESOLVED` — a suitable fix OR a functional-equivalence test could not be established.
  This does NOT prove the vuln is inherently unpatchable — it may mean the patch was inadequate, the
  intended behavior was under-specified, or the functional check was insufficient. Recorded as inconclusive.
- `BEHAVIOR_COUPLED` — evidence suggests the unsafe behavior is entangled with the intended functionality,
  so preserving function requires design-level change. This is a CANDIDATE qualitative finding ONLY:
  it needs supporting evidence and independent review before being interpreted as a design-level property
  of AI-generated code. Do NOT auto-claim novelty from it.

## Sampling design
- **Pilot:** 1 case per confirmable CWE class PRESENT in the AI corpus (CWE-22, CWE-78, CWE-918),
  + the SSTI control case (CWE-1336 has 0 AI-corpus confirmations, so use the llama-cpp external SSTI).
  Plus the one anti-circularity wrong-patch control. ~4–5 cases total.
- **Validation gate:** if the pilot is clean (pilots REVERSED, wrong-patch control does NOT reverse via
  its INDEPENDENT observable), scale to a **stratified random sample** across classes.
- **FREEZE THE SAMPLE FIRST (reviewer):** draw the stratified sample, write a committed sampling manifest
  (case ids + per-CWE sampling fraction) BEFORE auditing any case. No case is swapped after results are seen.
- **Reported metric:** patch-reversal rate = REVERSED / (REVERSED + NOT_REVERSED) among EVALUABLE cases;
  `PATCHABILITY_UNRESOLVED` and `BEHAVIOR_COUPLED` reported separately (not as denominators).
  If sampling fractions differ by CWE, estimate the corpus-wide rate with a WEIGHTED estimator using those
  fractions. Report a confidence interval; with small N expect WIDE intervals and do not imply precision
  not earned. Document the sampling frame so the estimate generalizes to the 149.

## What this is NOT (the no-gating principle — load-bearing)
- **The audit does NOT gate or redefine the frozen benchmark.** A `BEHAVIOR_COUPLED`,
  `PATCHABILITY_UNRESOLVED`, or even `NOT_REVERSED` outcome does NOT invalidate a function's
  `CONFIRMED` label. The audit evaluates (a) causal attribution of the observed impact and
  (b) the relationship between security impact and intended behavior — it never requires a
  confirmed case to "pass a patchability test" to stay confirmed.
- Not a change to the frozen 149/160 (it audits them, doesn't relabel them).
- Not a reachability claim (orthogonal axis).
- Not applied to detection-only classes (89/502/611) — those aren't execution-confirmed to begin with.

## Pre-registered expectations (HYPOTHESES, not foregone conclusions)
Record an expected observation per pilot case BEFORE testing, then classify STRICTLY from
evidence — do not design the test to produce the expected label.
- `_get_json` (CWE-918): hypothesis = a correct SSRF mitigation may preserve legitimate external
  fetching. BUT a private-IP block ALONE may be insufficient — the test must also exercise redirects,
  DNS/address resolution (rebinding), and non-HTTP schemes. Only a mitigation that survives those
  counts as a behavior-preserving fix.
- `execute_bash` (CWE-78): hypothesis = may be intentional command EXECUTION, not command INJECTION.
  Accepting a command string is NOT automatically CWE-78. The audit must first distinguish the
  function's contract (is running an arbitrary command its purpose?) from an injection flaw (fixed
  command with user data spliced in). If it is intentional execution, the finding is about the
  CWE-78 label's appropriateness, not a failed patch.
- `sha256_file` (CWE-22): hypothesis = a generic file-hasher may legitimately accept arbitrary paths;
  whether traversal is a vulnerability depends on the intended trust boundary and the CALLER's contract.
  A correct fix may belong at the caller, not inside the utility.

These are good pilots precisely because they force examination of the function's INTENDED CONTRACT
rather than blind patching of the proof mechanism. Outcomes are assigned from evidence, case by case.

## Scope of conclusions
A small pilot deliberately composed of general-purpose utilities CANNOT establish how prevalent
behavior-coupling is across the 149. Pilot conclusions stay scoped to the audited cases. Prevalence
claims require the frozen stratified sample (step: sampling design).
