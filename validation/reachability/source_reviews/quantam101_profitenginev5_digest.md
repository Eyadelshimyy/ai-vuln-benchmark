# Source review — quantam101/profitenginev5 `digest` (CWE-22)

**Verdict (source-only): REFUTED (the attacker-flagged chain is a false call-graph edge).**

## The screened chain is spurious
Screen flagged: `telegram_webhook → stable_embedding → digest` [attacker].
- The confirmed `digest(path)` is in `master_distillation/dataset.py:18`:
  `def digest(path): h = hashlib.sha256(); with open(path,"rb") as f: ...` — a file hasher.
- `runtime/telegram_gateway.py:60` `stable_embedding(text)` contains a **local variable**
  `digest = hashlib.sha256(text.encode()).digest()` (bytes of a hash of the text). It never calls
  `dataset.digest(path)`. The name-based call graph merged the local name `digest` with the
  module-level function `digest`.
- So the telegram webhook does **not** reach the file-hashing `digest(path)`. No request-controlled
  path flows to `open(path)` via this route.

## Actual reachability of `digest(path)`
- `digest(path)` is called from dataset-loading code (CLI/offline), with `path` from local dataset
  files, not an HTTP request. No HTTP entrypoint controls `path` in the source.

## Conclusion
- The function remains function-level CONFIRMED (it opens whatever path it is given — DIRECT_CALLER_PATH).
- App-level attacker reachability via the telegram route is **REFUTED** (false edge). No dynamic test warranted.
