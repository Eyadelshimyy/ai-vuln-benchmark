# CVE control set (RQ4 — instrument validation)

Real, human-authored **vulnerable + patched** pairs, run through VULCAN's own
harness to validate that the instrument distinguishes a known-vulnerable
implementation from its fix. See `docs/rq4_control_set.tex` for the full
protocol and terminology; this file is the practical schema + how-to.

Run: `python3 run_cve_controls.py --controls cve_controls/`

## What a "distinguish" means (and why it needs both tiers)

- **Parameterization fixes (SQL injection):** the vulnerable code splices data
  into the query string → the taint marker reaches the sink → **Tier-1
  TRIGGERED**. The patch binds a parameter → the marker leaves the SQL argument
  → **Tier-1 NOT_TRIGGERED**. **Tier 1 distinguishes.**
- **Validation / sanitisation fixes (path traversal, command injection):** the
  patch rejects malicious input but still calls the sink with benign input, so
  a benign marker reaches the sink in *both* → Tier-1 can't separate them. A
  *real* attack payload fires in the vulnerable version and is rejected by the
  patch → **Tier 2 distinguishes.**

A control is DISTINGUISHED if **either** tier separates vulnerable from patched.
The runner reports which tier did it.

## Inclusion criteria (pre-register BEFORE looking at the verdict)

A candidate CVE is **in-scope** only if all hold; record `in_scope` + reason:

1. Python, and the vulnerability is **function-local** (not a multi-function
   request flow or an external precondition).
2. The vulnerability class is one VULCAN models: CWE-89, CWE-78, CWE-22,
   CWE-94 (later: 502, 918, 611).
3. The affected function is **constructible/executable** in the harness.
4. The patch changes taint→sink reachability (Tier-1) **or** rejects a real
   payload (Tier-2) — i.e. the fix is one of the two shapes above.

Keep a few deliberately **out-of-scope** CVEs (`in_scope: false`) to show
VULCAN does not over-reach (it should *not* fire on those).

**Never** fabricate "human" code with an LLM or by hand-injecting a flaw — use
real historical pre/post-patch source only. The `EXAMPLE_synthetic_*.json`
files here are **self-tests of the runner**, clearly marked synthetic, not CVEs.

## Record schema (one JSON file per control)

```json
{
  "cve_id": "CVE-XXXX-NNNNN",
  "library": "name",
  "cwe": "CWE-89",
  "in_scope": true,
  "scope_reason": "function-local SQLi, library installs, fix is parameterization",
  "function": "get_user",
  "file": "pkg/module.py",
  "line": 6,
  "param": "name",
  "construction_recipe": null,
  "module_source": "…surrounding module, with the function present as a STUB at `line`…",
  "vulnerable_source": "    def get_user(self, name):\n        …pre-patch body…",
  "patched_source":    "    def get_user(self, name):\n        …post-patch body…"
}
```

Notes:
- `module_source` supplies imports / class / `__init__` so the receiver can be
  built; put the target function there as a one-line stub at `line` — the
  runner splices `vulnerable_source` / `patched_source` in its place (same
  mechanism used for AI candidates).
- `param` + an in-model `cwe` enable the **Tier-2** check; omit `param` to run
  Tier-1 only.
- `construction_recipe` is optional (same format as mined candidates) for
  functions whose receiver needs specific constructor arguments.
