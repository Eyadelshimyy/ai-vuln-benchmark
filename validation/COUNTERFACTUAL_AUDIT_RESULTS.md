# VULCAN — Counterfactual patch audit: pilot results

**Protocol:** validation/COUNTERFACTUAL_AUDIT_PROTOCOL.md (v2, frozen)
**Sample:** validation/COUNTERFACTUAL_pilot_manifest.json (seed 42, frozen before any case was run)
**Branch audited:** vulcan-freeze-149-rc (the frozen 149-function / 160-candidate release)
**Harness:** audit_case.py runs a single-candidate census on the ORIGINAL candidate vs a blind-patched variant (patch spliced into both ground_truth_source and the function body via AST replace_func), returning the oracle Tier-1 (triggered) / Tier-2 (confirmed) verdict for each. Baseline validated: each case reproduces confirmed=True on the unmodified candidate, matching its frozen census label.

NO-GATING REMINDER (load-bearing): these outcomes audit the causal attribution and the impact-vs-intent relationship of confirmed labels. They do NOT relabel, gate, or shrink the frozen 149/160. A BEHAVIOR_COUPLED or NOT_REVERSED case stays CONFIRMED.

## Result summary (4 pilot cases)

| Case | CWE | Baseline | Patched | triggered after patch | Outcome |
|---|---|---|---|---|---|
| waltfilipe__sada-test / fetch_json | 918 | confirmed=True | confirmed=False | - | REVERSED (clean) |
| VibeBB__fpga-agent / spdx_ids | 22 | confirmed=True | confirmed=False | True | REVERSED (caveat) |
| Cognition-Partner-Workshops__otterworks / sha256_file | 22 | confirmed=True | confirmed=False | True | REVERSED (caveat) |
| loveRhythm1990__mini-pi / execute_bash | 78 | confirmed=True | confirmed=True | True | BEHAVIOR_COUPLED |

Patch-reversal rate among evaluable cases = 3/3 = 100% (execute_bash reported separately as BEHAVIOR_COUPLED, not in the denominator, per protocol "Reported metric").

## Per-case findings

### 1. fetch_json (CWE-918, SSRF) - REVERSED (clean)
- Blind fix (patch_fetch_json.py): resolve the target host and block private/loopback/link-local/reserved/multicast ranges; scheme allowlist (http/https). Written from the function intent (fetch a remote JSON resource), not from the oracle proof mechanism.
- Oracle 2x2: baseline confirmed=True -> patched confirmed=False. The SSRF guard removes the confirmation.
- Benign preservation (cells 2 & 4): a legitimate external fetch (public host) is NOT blocked by the guard; the mitigation targets only non-routable/internal destinations, so normal remote fetching is preserved.
- Reading: the confirmed impact is causally attributable to the missing SSRF mitigation. Cleanest of the four: a correct, behavior-preserving fix kills the impact.
- Pre-registered caveat: a private-IP block alone is not a complete SSRF defense (redirect-following and DNS-rebinding remain). The reversal is valid for the oracle injection point; a production fix would additionally pin post-resolution and re-check on redirect.

### 2. spdx_ids (CWE-22, path traversal) - REVERSED (with caveat)
- Blind fix (patch_spdx_ids.py): confine the resolved path under Path(os.getcwd()).resolve(); return [] if the resolved target escapes that root.
- Oracle 2x2: baseline confirmed=True -> patched confirmed=False, triggered stays True. Tier-1 taint still reaches the open() sink, but Tier-2 no longer confirms: the fix blocks the out-of-boundary read (what the exploit bank actually proves) while the taint path itself is unchanged. This is the expected signature of a boundary fix and is strong evidence the confirmation was causally the traversal, not a harness artifact.
- Benign preservation: reading an in-project (in-cwd) file still succeeds under the patch.
- Honest caveat (pre-registered hypothesis, confirmed): spdx_ids is a general-purpose file reader. Whether reading an arbitrary path is a vulnerability depends on the CALLER trust boundary. The cwd-confinement fix reverses the oracle impact, but the semantically correct fix may belong at the caller, not inside the utility. Recorded as REVERSED (impact causally attributable) WITH the fix-belongs-at-caller note, not as proof the utility is intrinsically buggy.

### 3. sha256_file (CWE-22, path traversal) - REVERSED (with caveat)
- Blind fix (patch_sha256_file.py): same cwd-boundary confinement; raise PermissionError if the resolved path escapes the root.
- Oracle 2x2: identical signature to spdx_ids - baseline confirmed=True -> patched confirmed=False, triggered stays True.
- Benign preservation: hashing an in-cwd file still returns the correct digest under the patch.
- Same caveat: generic file-hasher; the trust boundary is the caller. REVERSED with fix-belongs-at-caller note.

### 4. execute_bash (CWE-78) - BEHAVIOR_COUPLED
- Blind fix (patch_execute_bash.py): shlex.split(command) + subprocess.run(args, shell=False, ...), the canonical anti-command-injection fix.
- Oracle 2x2: baseline confirmed=True -> patched confirmed=True (NOT_REVERSED).
- Investigation (why the defensible fix did not reverse): the function parameter IS the entire command. There is no fixed command with user data spliced in, so there is nothing for shell=False to protect. shlex.split("touch /tmp/pwn") -> ["touch","/tmp/pwn"] -> subprocess.run(..., shell=False) still executes touch. The oracle CWE-78 proof (a command runs) therefore still fires. This is not an oracle false-positive: attacker-controlled input does execute a command. It is intended arbitrary command execution, not command injection.
- Benign-regression evidence (the BEHAVIOR_COUPLED signature): the same fix breaks a legitimate shell feature. benign cmd "echo hello | tr a-z A-Z": original (shell=True) -> 'HELLO' (pipeline works); patched (shell=False) -> 'hello | tr a-z A-Z' (pipe treated as literal args, broken). So the only fix that could stop the exploit (stop executing the supplied command) would also destroy the function purpose. The unsafe capability is entangled with the intended functionality.
- Classification: BEHAVIOR_COUPLED. Reported as a finding about the appropriateness of the CWE-78 label (dangerous-by-design API vs injection flaw), NOT as a failed patch. CANDIDATE qualitative finding, scoped to this one case; NOT used to claim a design-level property of AI-generated code at large, and does NOT change execute_bash frozen CONFIRMED label.

## What the pilot establishes (and what it does not)

Establishes:
- For every evaluable confirmable-class case (3/3), a blind intent-derived fix causally reverses the oracle confirmation: the confirmed impact is attributable to the vulnerable operation, not to harness/environment artifacts. The triggered-stays-True / confirmed-flips signature on the two CWE-22 cases is especially clean: the taint path is untouched, only the proven impact is removed.
- The audit surfaces, rather than hides, the one case where "confirmed" means intended dangerous behavior (execute_bash). This feeds VULCAN exploitability-vs-reachability framing: function-level CONFIRMED is conditional exploitability, and execute_bash shows a case where the condition is "the caller is trusted to pass a command."

Does NOT establish (scope guard):
- Prevalence of behavior-coupling across the 149. The pilot is deliberately composed of general-purpose utilities; a prevalence claim requires the frozen stratified sample.
- That CWE-22 utilities are safe. The reversal shows the impact is real and fixable; the caveat only relocates where the correct fix belongs.

## Next step (validation gate)
Pilot is clean (3 REVERSED, the 4th an interpretable BEHAVIOR_COUPLED, no spurious non-reversals). The gate to scale is met EXCEPT the anti-circularity wrong-patch control with an INDEPENDENT observable (protocol section 2), which should be run before drawing the stratified sample. Until that control is run and the stratified sample is frozen, these four results stand as a pilot, scoped to the audited cases.
