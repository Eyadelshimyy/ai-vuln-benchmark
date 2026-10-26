# VULCAN — Anti-circularity control: results

**Protocol:** validation/COUNTERFACTUAL_AUDIT_PROTOCOL.md section 2
**Harness:** validation/anticirc_control.py (stdlib only; independent observable via /proc)
**Branch:** vulcan-freeze-149-rc

## Why this control exists
The patch audit claims a confirmed label is causally attributable to the vulnerable
code because a correct fix makes the oracle stop confirming (REVERSED). The mirror
question: did the oracle stop confirming because the vuln was FIXED, or merely because
a PROOF ARTIFACT it watches was suppressed? If proof-suppression alone could flip the
oracle, a REVERSED verdict would be worthless. This control tests exactly that, on
CWE-78 — VULCAN's only confirmable class whose verifier is a bare side-effect file
(tier2/strategies.py: _CmdInjection.verify == os.path.exists(ctx.proof_file), which
inspects neither stdout nor the return value).

## Method
Three arms, each measured on BOTH channels:
  (i)  the oracle signal    = the REAL CWE78.verify(ctx, ...) on a real ProofContext
  (ii) an INDEPENDENT obs    = a canary process whose argv[0] is a secret (set via
       `exec -a $CANARY sleep`), read from /proc/<pid>/cmdline — the same
       process-identity side-channel CVE-Bench uses for RCE, and one the CWE-78
       verifier never looks at.

  A  honest vuln          : injected command touches the proof file AND spawns the canary.
  B  observable-tampering : a WRONG "patch" (a PATH shim that neuters `touch`) suppresses
                            only the proof-file write while the command STILL executes —
                            command injection is NOT fixed.
  C  real fix             : refuse to execute the command at all.

## Result (verbatim pattern)
| arm | oracle (CWE78.verify) | independent (/proc canary) |
|---|---|---|
| A  honest-vuln | CONFIRMED | LIVE |
| B  observable-tampering (wrong patch) | NOT_CONFIRMED | LIVE |
| C  real fix (refuse to execute) | NOT_CONFIRMED | GONE |

**Control PASSES.**

## Interpretation
- The CWE-78 verifier returns NOT_CONFIRMED for BOTH B and C, so the verifier ALONE
  cannot distinguish proof-suppression (B, vuln still live) from a real fix (C, vuln gone).
- The independent observable DOES distinguish them: LIVE in B, GONE in C.
- Therefore a CWE-78 patch-audit reversal is sound ONLY when corroborated by an
  independent channel. The control is not "backwards/worthless" (protocol section 2's
  warning) precisely because the independent observable is present and measured.

## Scope and bearing on the pilot reversals
This foolability is specific to the BARE-FILE proof classes (CWE-78, CWE-502, CWE-1336
SSTI — all `os.path.exists(proof_file)`). VULCAN's headline-reversed cases are NOT in
that set; by inspection of their verify methods:
- CWE-22 read-traversal (spdx_ids, sha256_file) confirms via (c) a LIVE execution trace:
  the real open() succeeded AND the resolved path is exactly one of the random planted
  out-of-tree files. A path the code can only name by passing attacker input unsanitised —
  not a bare file-touch. The cwd-confinement fix makes that open() resolve in-boundary/fail,
  so the trace condition can't hold: a genuine fix, not artifact-suppression.
- CWE-918 (fetch_json) confirms via a REAL HTTP request reaching a throwaway 127.0.0.1
  listener the oracle controls — a network callback, not a file. (Separately noted in the
  audit: the SSRF patch blocks the whole private/loopback range, of which the canary's
  loopback address is one instance; the fix is broader than the canary address.)

So the three pilot REVERSED cases rest on content/syscall/network observables that this
control shows are not foolable the way a bare-file proof is. The one BARE-FILE class
(CWE-78, execute_bash) was classified BEHAVIOR_COUPLED, not REVERSED — consistent with
this finding.

## Forward-direction note (original confirmations remain sound)
This control concerns the REVERSE direction (does proof-absence prove safety) used by the
audit. It does NOT affect VULCAN's original CWE-78 confirmations, which are forward-sound:
the proof file appears ONLY if the injected command actually executed. The frozen 149/160
are unchanged (no-gating principle).

## Gate
With the pilot clean (3 REVERSED + 1 BEHAVIOR_COUPLED) AND this anti-circularity control
passing with a valid independent observable, the methodology gate in the protocol is met.
Next step per protocol: draw and freeze the stratified random sample (committed sampling
manifest: case ids + per-CWE fraction) BEFORE auditing any case, then report the weighted
patch-reversal rate with a (wide, small-N) confidence interval.
