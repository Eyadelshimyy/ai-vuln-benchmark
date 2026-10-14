"""Opt-in local log of gateway tool calls, and its summary.

``TRACEBI_MCP_LOG=1`` makes the MCP gateway append one JSON line per tool
call to ``.tracebi/gateway_log.jsonl`` in the project. Nothing leaves the
machine, and nothing in a line is customer data: the tool name, ok or
error, the duration, the actor, the NAMES of the arguments passed, and on
error the exception type plus its first line. Argument values, results,
filter values and tokens are never written, and any argument value an
error message echoes is masked before the line is written.

``tracebi agent log`` reads the file back as a summary.
"""

import functools
import inspect
import json
import os
import re
import time
import uuid
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Iterable, Optional

LOG_PATH = os.path.join(".tracebi", "gateway_log.jsonl")

# One id per gateway process: "first attempt" is per report, per session.
_SESSION = uuid.uuid4().hex[:12]
_build_attempts: Counter = Counter()


def enabled() -> bool:
    return os.environ.get("TRACEBI_MCP_LOG", "") == "1"


def _argument_values(value: Any) -> Iterable[str]:
    """Every scalar inside an argument, as text, for masking."""
    if isinstance(value, dict):
        for k, v in value.items():
            yield from _argument_values(k)
            yield from _argument_values(v)
    elif isinstance(value, (list, tuple, set)):
        for v in value:
            yield from _argument_values(v)
    elif value is not None and not isinstance(value, bool):
        yield str(value)


def _mask(message: str, arguments: dict) -> str:
    # Longest first, so a value that contains another is masked whole.
    # Whole tokens only: "report" must not redact the "report" inside
    # "reports". Values under three characters are left alone.
    for text in sorted(set(_argument_values(arguments)), key=len, reverse=True):
        if len(text) >= 3:
            message = re.sub(
                rf"(?<!\w){re.escape(text)}(?!\w)", "<value>", message)
    return message


def _attempt_key(arguments: dict) -> str:
    return str(arguments.get("report") or arguments.get("name") or "")


def _first_line(text: str) -> str:
    return text.splitlines()[0] if text else ""


def _write(line: dict) -> None:
    try:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(line) + "\n")
    except OSError:
        pass  # a log that cannot be written must never fail the tool call


def logged(tool: str, fn: Callable, actor: Callable[[], str]) -> Callable:
    """Wrap one gateway tool so each call appends a line when enabled."""
    signature = inspect.signature(fn)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if not enabled():
            return fn(*args, **kwargs)
        try:
            arguments = dict(signature.bind_partial(*args, **kwargs).arguments)
        except TypeError:
            arguments = dict(kwargs)
        # The MCP layer passes every parameter, defaults included; record
        # the ones the agent actually set.
        params = signature.parameters
        arguments = {
            k: v for k, v in arguments.items()
            if k not in params or params[k].default is inspect.Parameter.empty
            or v != params[k].default
        }
        line: dict = {
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "session": _SESSION,
            "tool": tool,
            "actor": actor(),
            "arguments": sorted(arguments),
        }
        if tool == "build_report":
            key = _attempt_key(arguments)
            _build_attempts[key] += 1
            line["attempt"] = _build_attempts[key]
        start = time.perf_counter()
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:
            line.update(ok=False, error_type=type(exc).__name__,
                        error=_mask(_first_line(str(exc)), arguments))
            line["ms"] = round((time.perf_counter() - start) * 1000, 1)
            _write(line)
            raise
        line["ms"] = round((time.perf_counter() - start) * 1000, 1)
        if isinstance(result, dict) and result.get("ok") is False:
            errors = result.get("errors") or [result.get("error") or ""]
            line.update(ok=False, error_type="refused",
                        error=_mask(_first_line(str(errors[0])), arguments))
        else:
            line["ok"] = True
        _write(line)
        return result

    return wrapper


def record_rejected_arguments(tool: str, arguments: Any, exc: BaseException,
                              actor: str) -> None:
    """Log a call the protocol rejected before the tool function ran.

    Argument validation (a wrong name, a missing required argument) never
    enters :func:`logged`. Record the names the client passed and a masked
    first line. A ``build_report`` rejection counts as an attempt, keyed the
    same way as a call that used ``report``.
    """
    if not enabled():
        return
    raw = arguments if isinstance(arguments, dict) else {}
    line: dict = {
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "session": _SESSION,
        "tool": tool,
        "actor": actor,
        "arguments": sorted(str(k) for k in raw),
        "ok": False,
        "error_type": type(exc).__name__,
        "error": _mask(_first_line(str(exc)), raw),
    }
    if tool == "build_report":
        key = _attempt_key(raw)
        _build_attempts[key] += 1
        line["attempt"] = _build_attempts[key]
    _write(line)


# ── Summary ─────────────────────────────────────────────────────────────────

_SINCE = re.compile(r"^(\d+)([dhm])$")


def parse_since(text: str) -> timedelta:
    m = _SINCE.match(text.strip())
    if not m:
        raise ValueError(f"--since takes a number and d, h or m (like 7d), not {text!r}")
    n, unit = int(m.group(1)), m.group(2)
    return {"d": timedelta(days=n), "h": timedelta(hours=n),
            "m": timedelta(minutes=n)}[unit]


def read_lines(path: str, since: Optional[timedelta] = None) -> list[dict]:
    if not os.path.isfile(path):
        return []
    cutoff = datetime.now(timezone.utc) - since if since else None
    out = []
    with open(path, encoding="utf-8") as f:
        for raw in f:
            try:
                line = json.loads(raw)
            except ValueError:
                continue
            if not isinstance(line, dict):
                continue
            if cutoff is not None:
                try:
                    if datetime.fromisoformat(line.get("at", "")) < cutoff:
                        continue
                except ValueError:
                    continue
            out.append(line)
    return out


def summarize(lines: list[dict]) -> dict:
    per_tool: dict = {}
    errors: Counter = Counter()
    for line in lines:
        tool = line.get("tool") or "?"
        entry = per_tool.setdefault(tool, {"calls": 0, "errors": 0})
        entry["calls"] += 1
        if line.get("ok") is False:
            entry["errors"] += 1
            errors[(tool, f"{line.get('error_type')}: {line.get('error')}")] += 1
    for entry in per_tool.values():
        entry["error_rate"] = round(entry["errors"] / entry["calls"], 3)
    firsts = [ln for ln in lines
              if ln.get("tool") == "build_report" and ln.get("attempt") == 1]
    first_ok = sum(1 for ln in firsts if ln.get("ok") is True)
    return {
        "calls": len(lines),
        "tools": dict(sorted(per_tool.items())),
        "top_errors": [
            {"tool": tool, "error": message, "count": count}
            for (tool, message), count in errors.most_common(10)
        ],
        "first_build": {
            "first_attempts": len(firsts),
            "succeeded": first_ok,
            "rate": round(first_ok / len(firsts), 3) if firsts else None,
        },
    }


def format_summary(summary: dict) -> str:
    out = [f"{summary['calls']} gateway calls"]
    if not summary["calls"]:
        return out[0]
    out.append("")
    out.append(f"{'tool':<24}{'calls':>7}{'errors':>8}{'rate':>8}")
    for tool, e in summary["tools"].items():
        out.append(f"{tool:<24}{e['calls']:>7}{e['errors']:>8}"
                   f"{e['error_rate'] * 100:>7.1f}%")
    fb = summary["first_build"]
    out.append("")
    if fb["first_attempts"]:
        out.append(f"first-build success: {fb['succeeded']} of "
                   f"{fb['first_attempts']} ({fb['rate'] * 100:.0f}%)")
    else:
        out.append("first-build success: no build_report calls")
    if summary["top_errors"]:
        out.append("")
        out.append("top errors:")
        for e in summary["top_errors"]:
            out.append(f"  {e['count']:>4}  {e['tool']}  {e['error']}")
    return "\n".join(out)
