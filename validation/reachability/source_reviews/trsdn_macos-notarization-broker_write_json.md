# Source review — trsdn/macos-notarization-broker `write_json` (CWE-22)

**Verdict (source-only): REFUTED (no HTTP entrypoint; screen "req-arg" was a false handler match).**

## Why the screen flagged it
The screen's `_is_handler` returns "req-arg" when a function's first parameter is named `request`.
`scripts/local_handoff.py:166` `def validate_archive(request: dict, archive: Path, work: Path, ...)`
has a first parameter literally named `request` — but it is a **notarization-request dict**, not an
HTTP request object, and there is no web framework here.

## Actual entrypoint
- `scripts/broker.py:2278` `def command_validate(args: argparse.Namespace)` — an **argparse CLI
  command**. The whole project is a command-line macOS notarization broker; no Flask/FastAPI/Django
  app object exists.
- `write_json(path, data)` (`local_handoff.py:50`) even checks `not path.is_symlink()` and chmods 0o600;
  `path` originates from CLI args / computed work dirs, not an HTTP request.

## Conclusion
- Function-level CONFIRMED stands (it writes to the path it is given).
- App-level attacker reachability is **REFUTED**: there is no HTTP route, and the flagged handler was
  a coincidental parameter-name match. No dynamic test warranted.
