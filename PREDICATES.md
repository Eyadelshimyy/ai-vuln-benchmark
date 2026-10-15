# VULCAN Tier-2 confirmation predicates (per CWE)

**Purpose.** This table makes explicit *what* the Tier-2 exploit bank proves, so
"confirmed" is not taken on faith. For each class, Tier-2 fires **only** when an
adversarial, attacker-controlled input produces the security *failure* named in
the "Observable" column — not merely because the sink function executed. The
"Safe (NOT confirmed)" column lists coding patterns that reach the same sink but
are correctly rejected, which is what distinguishes **execution-confirmed
exploitation** from **mere sink activation**.

> Verify each row against the actual implementation in `tier2/engine.py`,
> `analysis/iast/sinks.py`, and the cases in `tier2/test_soundness.py` before
> putting this in the thesis. The rows below describe the observed behaviour of
> the bank in this project.

| CWE | Adversarial input (payload) | Observable that proves the *failure* | Safe pattern → NOT confirmed |
|---|---|---|---|
| **CWE-22** path traversal | a marker path that escapes the intended base (`../…/<nonce>`) | a file is read/written **outside** the intended directory (the marker lands/opens out of bounds) | base confined via `basename`/`realpath` containment check |
| **CWE-78** OS command injection | shell metacharacter payload, e.g. `$(touch <marker>)`, `; touch <marker>` | the **injected** command executes — the marker file is created | `subprocess(..., shell=False)` with an argv list, or `shlex.quote` |
| **CWE-89** SQL injection | SQL-breaking payload that changes the query's meaning | the query returns attacker-chosen rows / the injected clause takes effect | parameterized query (`?`/`%s` placeholders) or integer coercion |
| **CWE-611** XXE | XML declaring an external entity pointing at a local marker | the entity is **resolved** — marker content appears in the parse result | `resolve_entities=False`, `no_network=True`, or an HTML (non-DTD) parser |
| **CWE-918** SSRF | an attacker-controlled URL to a loopback listener (`http://127.0.0.1:<port>/<nonce>`) | the program **issues the request** to that host — the listener receives the nonce | host allow-list / scheme+host validation that blocks the target |

**Why this matters for the thesis wording.** Because the "Safe pattern" column
is correctly **not confirmed** (verified in the soundness battery), Tier-2 is
**execution-confirmed exploitation under the bank's adversarial inputs**, not
"a dangerous API was invoked." State it that way, and cite this table.

**What Tier-2 does *not* establish on its own:**
- that the vulnerable parameter is reachable from a real interface (that is the
  reachability stage), or
- that attacker-controlled input actually reaches it on a real deployment (that
  is the manual trust-boundary / end-to-end stage).

So the evidence ladder is: **static trigger (candidate) → execution-confirmed
exploitation (this table) → reachable from an entry point → end-to-end
demonstrated.** Each rung is a strictly stronger, separately reported claim.
