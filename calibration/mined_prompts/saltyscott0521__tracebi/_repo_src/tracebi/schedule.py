"""
Scheduled reports — a report that runs and delivers itself.

A package opts in with a ``schedule`` block in its ``report.json``::

    "schedule": {
      "cron": "0 9 * * MON",
      "timezone": "America/New_York",
      "to": ["cfo@example.com", "ops@example.com"]
    }

``cron`` is required (five fields). ``timezone`` is an IANA name, default
UTC. ``to`` is optional: without it a run sends no email. When
``TRACEBI_SLACK_BOT_TOKEN`` and ``TRACEBI_SLACK_CHANNEL`` are both set,
a successful run still uploads the report file to that channel.
``refresh`` is optional::

    "refresh": {"transforms": ["holdings_transform"], "pipelines": ["sales_etl"]}

names what to run BEFORE the build so the report shows fresh data: each
transform (``tracebi run-transform``), then each pipeline (``tracebi
run-pipeline``), in order, each in a fresh process. A step that fails —
including a sink contract that refuses the new data — fails the run, and
nothing is built or sent.

A failed refresh or build is retried before that failure is recorded.
The default is two retries, waiting 60 seconds and then 300 (any further
retry waits 300 again). ``"retries"`` is an integer from 0 to 5; 0
disables them. A receipt that does not verify, and a delivery failure,
are not retried — a failed send is not repeated. The recorded run
includes ``attempts`` (1 when the first try succeeded).

``owner`` is an optional email address. A run that ends ``failed``,
``refused``, or ``empty`` emails that address a plain-text alert. When
``TRACEBI_PUBLIC_URL`` is set, the alert links to that report on the Runs
page. When ``TRACEBI_SLACK_WEBHOOK`` is set, the same text is posted
there. Without an owner, nothing is alerted. ``--no-send`` records the
alert and does not email or ping it. An alert that fails to send is
stored on the run and does not change the run's status. A failed Slack
ping is ``alert.slack_error`` and leaves a sent email sent.

``burst`` fans that one run out into one build per filter value::

    "burst": {
      "filter": {"dim_region.region": {"in": ["EMEA", "APAC"]}},
      "to": {"EMEA": ["emea@example.com"], "APAC": ["apac@example.com"]}
    }

``filter`` is one dimension key mapped to a list, or to ``{"in": [...]}``.
Each value is its own build, with that filter applied on every binding
whose model has the dimension — the same ``query_under_selection`` path a
kept selection uses. Recipients come from ``burst.to[value]``. A value
missing there is skipped and recorded; it is not sent to the top-level
``to``. ``burst`` without a ``to`` object is refused when the package
loads. Each slice is verified on its own. A refused, empty, or failed
slice is not sent; the others still are. The run is one parent record
whose ``slices`` list names each value, status, recipients, and output
path. If any slice failed, came back empty, or was refused, one owner
alert names those slices.

When ``TRACEBI_SLACK_BOT_TOKEN`` and ``TRACEBI_SLACK_CHANNEL`` are both
set, a successful send also uploads the ``.html`` and the
``.manifest.json`` to that channel, with a short summary: report name,
``rendered_at``, the verify verdict, and up to five headline value
figures already recorded on the manifest or written into the page.
``TRACEBI_SLACK_WEBHOOK`` is not this path — an incoming webhook cannot
upload a file, and it stays a text ping. A failed upload is stored as
``delivery.slack.error`` and is not retried. ``--no-send`` records the
channel and does not call Slack. With no email recipients, Slack alone
still delivers.

After a receipt verifies, a figure binding with zero rows (that section's
``dataset_shape`` in the manifest) is recorded ``empty`` and the report is
not sent. A binding no figure uses does not count. A receipt that does not
verify stays ``refused``.

The schedule lives in the repo beside the bindings, so a reviewer approves
*when* and *to whom* in the same pull request as *what*.

One run is: refresh → build → verify → empty check → deliver → record, then
an owner alert when one is due. It is ``tracebi report
send`` with the recipients read from the package, and the same rule —
distribution never outruns verification: a receipt that does not verify is
recorded as ``refused`` and nothing is sent. Every run is a row in
``tracebi_runs`` (kind ``schedule``). An existing
``output/schedule_runs.jsonl`` is imported once.

``tracebi schedule serve`` runs every schedule in one foreground process
(needs ``tracebi[pipeline]`` for APScheduler). Any external scheduler — cron,
a Kubernetes CronJob, CI — can instead call ``tracebi schedule run <name>``.
"""

from __future__ import annotations

import json
import os
import re
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Union
from urllib.parse import quote

#: Run outcomes. ``delivered`` and ``built`` are successes.
DELIVERED = "delivered"   # built, verified, sent
BUILT = "built"           # built and verified; nothing to deliver, or --no-send
REFUSED = "refused"       # built; the receipt did not verify, nothing sent
EMPTY = "empty"           # verified, but a figure's binding had no rows; nothing sent
FAILED = "failed"         # a refresh step failed, or the build, verify or send raised
SKIPPED = "skipped"       # a burst value with no recipients; not built, not sent

RUN_LOG = "schedule_runs.jsonl"

_ALLOWED_KEYS = {"cron", "timezone", "to", "refresh", "retries", "owner", "burst"}
_ALERT_STATUSES = {FAILED, REFUSED, EMPTY}
_REFRESH_KINDS = ("transforms", "pipelines")
_DEFAULT_RETRIES = 2
_MAX_RETRIES = 5
# First retry waits a minute; every retry after that waits five minutes.
_RETRY_DELAYS = (60, 300)

#: Patched in tests so a retry never calls ``time.sleep``.
_sleep = time.sleep


def parse_schedule_block(raw, *, path: str) -> dict:
    """Validate a package's ``schedule`` block; return it normalized.

    Returns ``{"cron": str, "timezone": str | None, "to": list[str],
    "refresh": {"transforms": list[str], "pipelines": list[str]},
    "retries": int, "owner": str | None, "burst": dict | None}``.
    ``retries`` defaults to 2. ``owner`` and ``burst`` default to ``None``.
    Raises ``ValueError`` naming *path* and the fix.
    """
    if not isinstance(raw, dict):
        raise ValueError(
            f"{path}: 'schedule' must be an object like "
            f"{{\"cron\": \"0 9 * * MON\", \"to\": [\"a@example.com\"]}}."
        )
    unknown = set(raw) - _ALLOWED_KEYS
    if unknown:
        raise ValueError(
            f"{path}: unknown schedule field(s): {sorted(unknown)}. "
            f"Allowed: {sorted(_ALLOWED_KEYS)}."
        )

    cron = raw.get("cron")
    if not isinstance(cron, str) or len(cron.split()) != 5:
        raise ValueError(
            f"{path}: schedule 'cron' must be a five-field cron expression "
            f"(minute hour day month day_of_week), e.g. \"0 9 * * MON\"; "
            f"got {cron!r}."
        )

    tz = raw.get("timezone")
    if tz is not None:
        if not isinstance(tz, str) or not tz:
            raise ValueError(f"{path}: schedule 'timezone' must be an IANA "
                             f"name such as \"Europe/London\".")
        from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
        try:
            ZoneInfo(tz)
        except (ZoneInfoNotFoundError, ValueError):
            raise ValueError(
                f"{path}: unknown schedule timezone {tz!r}. Use an IANA name "
                f"such as \"America/New_York\" (on a system with no time zone "
                f"database, `pip install tzdata`)."
            ) from None

    to = _email_list(raw.get("to", []), path=path, what="'to'")
    refresh = raw.get("refresh", {})
    if not isinstance(refresh, dict) or set(refresh) - set(_REFRESH_KINDS) or any(
            not isinstance(refresh.get(k, []), list)
            or not all(isinstance(n, str) and n for n in refresh.get(k, []))
            for k in _REFRESH_KINDS):
        raise ValueError(
            f"{path}: schedule 'refresh' must be an object like "
            f"{{\"transforms\": [\"holdings_transform\"], "
            f"\"pipelines\": [\"sales_etl\"]}}.")
    retries = raw.get("retries", _DEFAULT_RETRIES)
    # bool is an int subclass; True would otherwise retry once.
    if (isinstance(retries, bool) or not isinstance(retries, int)
            or not 0 <= retries <= _MAX_RETRIES):
        raise ValueError(
            f"{path}: schedule 'retries' must be an integer from 0 to "
            f"{_MAX_RETRIES} (0 disables retries); got {retries!r}."
        )
    owner = raw.get("owner")
    if owner is not None and (not isinstance(owner, str) or "@" not in owner):
        raise ValueError(
            f"{path}: schedule 'owner' must be one email address."
        )
    burst = _parse_burst(raw["burst"], path=path) if "burst" in raw else None
    return {"cron": " ".join(cron.split()), "timezone": tz, "to": list(to),
            "refresh": {k: list(refresh.get(k, [])) for k in _REFRESH_KINDS},
            "retries": retries, "owner": owner, "burst": burst}


def _email_list(raw, *, path: str, what: str) -> list[str]:
    if isinstance(raw, str):
        raw = [a.strip() for a in raw.split(",") if a.strip()]
    if not isinstance(raw, list) or any(
            not isinstance(a, str) or "@" not in a for a in raw):
        raise ValueError(
            f"{path}: schedule {what} must be a list of email addresses."
        )
    return list(raw)


def _parse_burst(raw, *, path: str) -> dict:
    """The ``burst`` object: one dimension filter, and who receives each value."""
    if not isinstance(raw, dict):
        raise ValueError(
            f"{path}: schedule 'burst' must be an object with 'filter' and "
            f"'to', e.g. {{\"filter\": {{\"dim_region.region\": "
            f"{{\"in\": [\"EMEA\"]}}}}, \"to\": {{\"EMEA\": "
            f"[\"emea@example.com\"]}}}}."
        )
    unknown = set(raw) - {"filter", "to"}
    if unknown:
        raise ValueError(
            f"{path}: unknown burst field(s): {sorted(unknown)}. "
            f"Allowed: filter, to."
        )
    if "to" not in raw:
        raise ValueError(
            f"{path}: schedule 'burst' needs a 'to' object mapping each "
            f"filter value to its recipients. A value missing from 'to' is "
            f"skipped at run time; omitting 'to' entirely is not."
        )
    filt = raw.get("filter")
    if not isinstance(filt, dict) or len(filt) != 1:
        raise ValueError(
            f"{path}: schedule 'burst.filter' must be one dimension key "
            f"mapped to a list or {{\"in\": [...]}} of values."
        )
    key, spec = next(iter(filt.items()))
    head, dot, attr = str(key).partition(".")
    if not isinstance(key, str) or not dot or not head or not attr:
        raise ValueError(
            f"{path}: schedule 'burst.filter' key must be a dimension "
            f"reference like \"dim_region.region\"; got {key!r}."
        )
    values = _burst_values(spec, path)
    to_raw = raw["to"]
    if not isinstance(to_raw, dict):
        raise ValueError(
            f"{path}: schedule 'burst.to' must be an object mapping each "
            f"filter value to a list of email addresses."
        )
    extra = sorted(set(to_raw) - set(values))
    if extra:
        raise ValueError(
            f"{path}: schedule 'burst.to' names {extra}, which are not "
            f"values of burst.filter ({values})."
        )
    to = {val: _email_list(addrs, path=path, what=f"'burst.to' for {val!r}")
          for val, addrs in to_raw.items()}
    return {"filter": {key: {"in": values}}, "to": to}


def _burst_values(spec, path: str) -> list[str]:
    if isinstance(spec, list):
        values = list(spec)
    elif (isinstance(spec, dict) and set(spec) == {"in"}
          and isinstance(spec.get("in"), list)):
        values = list(spec["in"])
    else:
        raise ValueError(
            f"{path}: schedule 'burst.filter' must map its dimension to a "
            f"list of values or {{\"in\": [\"EMEA\", \"APAC\"]}}."
        )
    if not values:
        raise ValueError(
            f"{path}: schedule 'burst.filter' needs at least one value."
        )
    if any(not isinstance(v, str) or not v for v in values):
        raise ValueError(
            f"{path}: schedule 'burst.filter' values must be non-empty strings."
        )
    if len(values) != len(set(values)):
        raise ValueError(
            f"{path}: schedule 'burst.filter' values must be unique; "
            f"got {values}."
        )
    return values


def discover_library_schedules() -> tuple[list[dict], list[dict]]:
    """Every scheduled package under the configured library roots.

    Honours ``TRACEBI_LIBRARY_MOUNTS`` (and otherwise ``TRACEBI_REPORTS_DIR``).
    Mount labels prefix the report name (``finance/weekly``).
    """
    from tracebi.report_paths import library_roots

    schedules: list[dict] = []
    errors: list[dict] = []
    for label, root in library_roots():
        found, bad = discover_schedules(root)
        if label:
            for item in found:
                item["report"] = f"{label}/{item['report']}"
            for item in bad:
                item["report"] = f"{label}/{item['report']}"
        schedules.extend(found)
        errors.extend(bad)
    return schedules, errors


def discover_schedules(reports_dir: Union[str, Path]) -> tuple[list[dict], list[dict]]:
    """Every scheduled package under *reports_dir*.

    Returns ``(schedules, errors)``. Each schedule is the normalized block
    plus ``"report"`` (the package's path below *reports_dir*, e.g.
    ``finance/weekly``). A package that fails to
    load is an error entry ``{"report", "error"}``, not an exception — one
    broken package must not stop the others from running. Specs
    (``reports/<name>.json``) are not scheduled; migrate them to a package
    with ``tracebi migrate spec``.
    """
    from tracebi.reports.template_package import TemplatePackage

    reports_dir = Path(reports_dir)
    schedules: list[dict] = []
    errors: list[dict] = []
    if not reports_dir.is_dir():
        return schedules, errors
    for pkg in _package_dirs(reports_dir):
        name = pkg.relative_to(reports_dir).as_posix()   # "finance/weekly"
        try:
            package = TemplatePackage(str(pkg))
        except Exception as exc:  # noqa: BLE001 — report it, keep going
            errors.append({"report": name,
                           "error": f"{type(exc).__name__}: {exc}"})
            continue
        if package.schedule is not None:
            schedules.append({"report": name, **package.schedule})
    return schedules, errors


def _package_dirs(directory: Path) -> list[Path]:
    """Package directories below *directory*, in folders too: a directory
    with ``report.json`` is a package; any other is a folder to walk."""
    found: list[Path] = []
    for sub in sorted(p for p in directory.iterdir() if p.is_dir()):
        if sub.name.startswith(("_", ".")):
            continue
        if (sub / "report.json").is_file():
            found.append(sub)
        else:
            found.extend(_package_dirs(sub))
    return found


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def record_run(record: dict, output_dir: Union[str, Path]) -> None:
    """Record *record* in the state store. Imports a legacy log in this directory first."""
    from tracebi.state import record_schedule
    record_schedule(record, output_dir)


def last_runs(output_dir: Union[str, Path]) -> dict[str, dict]:
    """The most recent recorded run per report. A legacy log is imported once."""
    from tracebi.state import schedule_records
    latest: dict[str, dict] = {}
    for rec in schedule_records(output_dir):
        if rec.get("report"):
            latest[rec["report"]] = rec
    return latest


def run_schedule(schedule: dict, *, reports_dir: Union[str, Path],
                 output_dir: Union[str, Path],
                 models_dir: Union[str, Path, None] = None,
                 send: bool = True) -> dict:
    """Run one schedule now: refresh → build → verify → empty check → deliver → record.

    A failed refresh or build is retried (``retries``, default 2) before
    the failure is recorded. Verify, the empty check, and delivery are not
    retried. An owner alert, when one is due, is sent after the outcome is
    known and before the record is written. It never raises.

    A schedule with ``burst`` refreshes once, then repeats build → verify →
    deliver once per filter value. One slice that fails does not stop the
    others. The record's ``slices`` list is the per-value detail.

    Returns the run record (also stored in ``tracebi_runs``). Never raises for
    a failed run — the failure is the record's ``status`` and ``error``, so
    a scheduler loop survives one bad report.

    A Postgres advisory lock per report is held for the whole tick, so two
    workers do not send the same email. When the lock is already held this
    returns ``status="skipped"`` and writes nothing. On SQLite the lock is a
    no-op and every worker proceeds.
    """
    from tracebi.state import advisory_lock, ensure

    name = schedule["report"]
    with advisory_lock(ensure(), f"schedule:{name}") as got:
        if not got:
            now = _now()
            return {
                "report": name, "started_at": now, "finished_at": now,
                "status": "skipped", "verdict": None, "recipients": [],
                "output": None, "error": "already running in another process",
                "actor": None, "actor_role": None, "refresh": [], "attempts": 0,
            }
        return _run_schedule(schedule, reports_dir=reports_dir,
                             output_dir=output_dir, models_dir=models_dir,
                             send=send)


def _run_schedule(schedule: dict, *, reports_dir: Union[str, Path],
                  output_dir: Union[str, Path],
                  models_dir: Union[str, Path, None] = None,
                  send: bool = True) -> dict:
    from tracebi.audit import get_actor

    name = schedule["report"]
    user, role = get_actor()
    record: dict = {
        "report": name, "started_at": _now(), "finished_at": None,
        "status": FAILED, "verdict": None, "recipients": [],
        "output": None, "error": None, "actor": user, "actor_role": role,
        "refresh": [], "attempts": 1,
    }
    try:
        _run(schedule, record, Path(reports_dir), Path(output_dir),
             models_dir, send)
    except Exception as exc:  # noqa: BLE001 — recorded, not raised
        record["status"] = FAILED
        record["error"] = f"{type(exc).__name__}: {exc}"
    record["finished_at"] = _now()
    _maybe_alert(schedule, record, send)
    record_run(record, output_dir)
    return record


def _run(schedule: dict, record: dict, reports_dir: Path, output_dir: Path,
         models_dir, send: bool) -> None:
    if schedule.get("burst"):
        _run_burst(schedule, record, reports_dir, output_dir, models_dir, send)
        return
    from tracebi.cli import _build_report_target, _resolve_report_target

    name = schedule["report"]
    kind, path = _resolve_report_target(name, reports_dir, purpose="schedule")
    retries = schedule.get("retries", _DEFAULT_RETRIES)
    output = output_dir / f"{name}.html"
    built = False
    # Only a failed refresh or a failed build is retried. A receipt that
    # does not verify is a verdict, and a failed send must not be repeated.
    for attempt in range(retries + 1):
        record["attempts"] = attempt + 1
        record["refresh"] = []
        record["error"] = None
        try:
            if _refresh(schedule.get("refresh") or {}, record):
                _build_report_target(kind, path, output, report_name=name)
                record["output"] = str(output)
                built = True
                break
        except Exception as exc:  # noqa: BLE001 — retried, then recorded
            record["error"] = f"{type(exc).__name__}: {exc}"
        if attempt < retries:
            _sleep(_RETRY_DELAYS[min(attempt, len(_RETRY_DELAYS) - 1)])
    if not built:
        record["status"] = FAILED
        return
    _after_build(record, output, list(schedule.get("to") or []),
                 models_dir, send, name)


def _run_burst(schedule: dict, record: dict, reports_dir: Path,
               output_dir: Path, models_dir, send: bool) -> None:
    """Refresh once, then one build → verify → deliver per filter value.

    A value with no recipients in ``burst.to`` is recorded ``skipped`` and
    is not built. A slice that fails, comes back empty, or does not verify
    is not sent; later values still run. The parent status is the worst
    slice status, so one bad slice still alerts the owner.
    """
    from tracebi.cli import _build_report_target, _resolve_report_target

    name = schedule["report"]
    kind, path = _resolve_report_target(name, reports_dir, purpose="schedule")
    retries = schedule.get("retries", _DEFAULT_RETRIES)
    refreshed = False
    for attempt in range(retries + 1):
        record["attempts"] = attempt + 1
        record["refresh"] = []
        record["error"] = None
        try:
            if _refresh(schedule.get("refresh") or {}, record):
                refreshed = True
                break
        except Exception as exc:  # noqa: BLE001 — retried, then recorded
            record["error"] = f"{type(exc).__name__}: {exc}"
        if attempt < retries:
            _sleep(_RETRY_DELAYS[min(attempt, len(_RETRY_DELAYS) - 1)])
    if not refreshed:
        record["status"] = FAILED
        return

    burst = schedule["burst"]
    key, spec = next(iter(burst["filter"].items()))
    slices: list[dict] = []
    used: set[str] = set()
    for value in spec["in"]:
        sl: dict = {
            "value": value, "status": FAILED, "recipients": [],
            "output": None, "verdict": None, "error": None, "attempts": 1,
        }
        recipients = list(burst["to"].get(value) or [])
        if not recipients:
            sl["status"] = SKIPPED
            sl["error"] = (
                f"no recipients for {value!r} in burst.to; "
                f"this slice was not built or sent")
            slices.append(sl)
            continue
        output = _slice_output(output_dir, name, _slice_token(value, used))
        built = False
        for attempt in range(retries + 1):
            sl["attempts"] = attempt + 1
            sl["error"] = None
            try:
                _build_report_target(
                    kind, path, output, report_name=name,
                    filters={key: value})
                sl["output"] = str(output)
                built = True
                break
            except Exception as exc:  # noqa: BLE001 — this slice only
                sl["error"] = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                _sleep(_RETRY_DELAYS[min(attempt, len(_RETRY_DELAYS) - 1)])
        record["attempts"] = max(record["attempts"], sl["attempts"])
        if not built:
            sl["status"] = FAILED
            slices.append(sl)
            continue
        try:
            _after_build(sl, output, recipients, models_dir, send,
                         f"{name} [{value}]")
        except Exception as exc:  # noqa: BLE001 — one slice must not stop the rest
            sl["status"] = FAILED
            sl["error"] = f"{type(exc).__name__}: {exc}"
            sl["recipients"] = []
        slices.append(sl)
    _finish_burst(record, slices)


def _slice_token(value: str, used: set[str]) -> str:
    """A filename piece for one burst value, unique ignoring case."""
    raw = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "value"
    token = raw
    n = 2
    while token.lower() in used:
        token = f"{raw}-{n}"
        n += 1
    used.add(token.lower())
    return token


def _slice_output(output_dir: Path, report: str, token: str) -> Path:
    base = output_dir / f"{report}.html"
    return base.with_name(f"{base.stem}--{token}{base.suffix}")


def _burst_status(slices: list[dict]) -> str:
    """The parent status: the worst slice, ignoring values that were skipped."""
    statuses = [s["status"] for s in slices]
    if FAILED in statuses:
        return FAILED
    if REFUSED in statuses:
        return REFUSED
    if EMPTY in statuses:
        return EMPTY
    done = [s for s in statuses if s != SKIPPED]
    if not done or all(s == BUILT for s in done):
        return BUILT
    return DELIVERED


def _finish_burst(record: dict, slices: list[dict]) -> None:
    record["slices"] = slices
    record["recipients"] = [
        addr for sl in slices for addr in (sl.get("recipients") or [])]
    record["status"] = _burst_status(slices)
    record["output"] = None
    bad = [sl for sl in slices if sl["status"] in _ALERT_STATUSES]
    if bad:
        record["error"] = "; ".join(
            f"{sl['value']}: {sl.get('error') or sl['status']}" for sl in bad)
    elif slices and all(sl["status"] == SKIPPED for sl in slices):
        record["error"] = (
            "no burst value had recipients in burst.to; nothing was built")
    else:
        record["error"] = None
    verdicts = [sl.get("verdict") for sl in slices if sl.get("verdict")]
    if verdicts and all(v == verdicts[0] for v in verdicts):
        record["verdict"] = verdicts[0]
    elif record["status"] == REFUSED:
        refused = next(sl for sl in slices if sl["status"] == REFUSED)
        record["verdict"] = refused.get("verdict")


def _after_build(record: dict, output: Path, recipients: list, models_dir,
                 send: bool, label: str) -> None:
    """Verify one built artifact, then deliver it or record why not."""
    from tracebi.verify import load_models, verify_manifest

    manifest_path = output.with_name(output.name + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = verify_manifest(manifest, load_models(models_dir))
    record["verdict"] = result["verdict"]
    if result["exit_code"] != 0:
        record["status"] = REFUSED
        record["error"] = result["verdict_detail"]
        return

    empty = _empty_bindings(manifest)
    if empty:
        record["status"] = EMPTY
        record["empty_bindings"] = empty
        record["error"] = "no rows in " + ", ".join(empty)
        return

    _deliver(record, output=output, manifest_path=manifest_path,
             recipients=list(recipients), result=result, send=send, label=label)


def _deliver(record: dict, *, output: Path, manifest_path: Path,
             recipients: list, result: dict, send: bool, label: str) -> None:
    """Email *recipients* and, when configured, upload the file to Slack.

    Same rules as a single scheduled run: ``--no-send`` records the intent,
    a Slack file failure with no email recipients fails the record, and a
    Slack text-ping failure does not unsend the email.
    """
    slack = _slack_file_target()
    if not send:
        record["status"] = BUILT
        if slack:
            record["delivery"] = {
                "slack": {"channel": slack[1], "sent": False, "error": None}}
        return
    if not recipients and not slack:
        record["status"] = BUILT
        return

    if recipients:
        from tracebi._delivery import send_report, slack_notify
        send_report(output, manifest_path, recipients, verify_result=result)
        record["recipients"] = list(recipients)

        webhook = os.environ.get("TRACEBI_SLACK_WEBHOOK")
        if webhook:
            try:
                slack_notify(
                    webhook,
                    f"{label} delivered on schedule · "
                    f"{result['verdict'].upper().replace('_', ' ')}")
            except Exception as exc:  # noqa: BLE001 — the report already went out
                record["error"] = (
                    f"slack notify failed (report was sent): {exc}")

    if slack:
        token, channel = slack
        try:
            from tracebi._delivery import slack_send_report
            slack_send_report(
                output, manifest_path, channel=channel, token=token,
                verify_result=result)
        except Exception as exc:  # noqa: BLE001 — recorded, not retried
            record["delivery"] = {"slack": {
                "channel": channel, "sent": False,
                "error": f"{type(exc).__name__}: {exc}",
            }}
            if not recipients:
                record["status"] = FAILED
                record["error"] = record["delivery"]["slack"]["error"]
                return
        else:
            record["delivery"] = {
                "slack": {"channel": channel, "sent": True, "error": None}}

    record["status"] = DELIVERED


def _slack_file_target():
    """``(token, channel)`` when both Slack file-delivery variables are set.

    ``TRACEBI_SLACK_WEBHOOK`` is a text ping and is not this path. Either
    variable alone leaves file delivery off — a half-configured channel
    is not a destination.
    """
    token = (os.environ.get("TRACEBI_SLACK_BOT_TOKEN") or "").strip()
    channel = (os.environ.get("TRACEBI_SLACK_CHANNEL") or "").strip()
    if token and channel:
        return token, channel
    return None


def _empty_bindings(manifest: dict) -> list[str]:
    """Bindings a figure uses whose result has zero rows.

    Row counts are the manifest section's ``dataset_shape`` (rows, columns),
    written at build. A binding no figure names — including a totals binding
    a figure does not reference — is ignored. The same binding on several
    figures is listed once, in figure order.
    """
    used: list[str] = []
    seen: set[str] = set()
    for fig in manifest.get("figures") or []:
        if not isinstance(fig, dict):
            continue
        for key in ("binding", "totals"):
            name = fig.get(key)
            if isinstance(name, str) and name and name not in seen:
                seen.add(name)
                used.append(name)
    rows: dict[str, int] = {}

    def walk(sections) -> None:
        for section in sections or []:
            if not isinstance(section, dict):
                continue
            name = section.get("id") or section.get("title")
            shape = section.get("dataset_shape")
            if isinstance(name, str) and isinstance(shape, list) and shape:
                rows[name] = shape[0]
            walk(section.get("sections"))

    walk(manifest.get("sections"))
    return [name for name in used if rows.get(name) == 0]


def _what_happened(record: dict) -> str:
    bad = [s for s in (record.get("slices") or [])
           if s.get("status") in _ALERT_STATUSES]
    if bad:
        parts = []
        for s in bad:
            piece = f"{s.get('value')} ({s.get('status')})"
            if s.get("error"):
                piece = f"{piece}: {s['error']}"
            parts.append(piece)
        return "Slices: " + "; ".join(parts)
    status = record.get("status")
    if status == EMPTY:
        names = ", ".join(record.get("empty_bindings") or [])
        return f"These bindings returned no rows: {names}."
    if status == REFUSED:
        verdict = record.get("verdict") or "unknown"
        detail = record.get("error") or ""
        return f"The receipt did not verify ({verdict}). {detail}".strip()
    return record.get("error") or "The run failed."


def _alert_body(record: dict) -> str:
    lines = [
        f"Report: {record.get('report')}",
        f"Status: {record.get('status')}",
        f"What happened: {_what_happened(record)}",
        f"Attempts: {record.get('attempts')}",
        f"Started: {record.get('started_at')}",
        f"Finished: {record.get('finished_at')}",
        f"Recorded in: tracebi_runs (kind=schedule)",
    ]
    public = (os.environ.get("TRACEBI_PUBLIC_URL") or "").strip().rstrip("/")
    if public:
        target = quote(str(record.get("report") or ""), safe="")
        lines.append(f"See the run: {public}/runs?kind=schedule&target={target}")
    lines.append("")
    return "\n".join(lines)


def _maybe_alert(schedule: dict, record: dict, send: bool) -> None:
    """Email ``owner`` when the run ended failed, refused, or empty.

    Never raises. With ``send`` false, the alert is recorded and not sent.
    No owner, or a success, leaves ``alert`` off the record. A Slack ping,
    when ``TRACEBI_SLACK_WEBHOOK`` is set, uses the same text and does not
    change ``sent`` or the run status.
    """
    if record.get("status") not in _ALERT_STATUSES:
        return
    owner = schedule.get("owner")
    if not owner:
        return
    record["alert"] = {"to": owner, "sent": False, "error": None}
    if not send:
        return
    subject = (f"TraceBi: {record.get('report')} scheduled run "
               f"{record.get('status')}")
    body = _alert_body(record)
    try:
        from tracebi._delivery import send_alert
        send_alert(owner, subject, body)
    except Exception as exc:  # noqa: BLE001 — an alert must not change the run
        record["alert"]["error"] = f"{type(exc).__name__}: {exc}"
    else:
        record["alert"]["sent"] = True

    webhook = os.environ.get("TRACEBI_SLACK_WEBHOOK")
    if not webhook:
        return
    try:
        from tracebi._delivery import slack_notify
        slack_notify(webhook, body)
    except Exception as exc:  # noqa: BLE001 — a ping must not change the run
        record["alert"]["slack_error"] = f"{type(exc).__name__}: {exc}"


def _refresh(refresh: dict, record: dict) -> bool:
    """Run the schedule's refresh steps in order, each in a fresh process —
    the same commands a person runs, so a transform's sink contract still
    guards what lands. Returns False (with ``record["error"]`` set) at the
    first step that fails; later steps and the build do not run."""
    import subprocess
    import sys

    steps = [("transform", "run-transform", n) for n in refresh.get("transforms", [])]
    steps += [("pipeline", "run-pipeline", n) for n in refresh.get("pipelines", [])]
    for kind, command, name in steps:
        started = time.monotonic()
        proc = subprocess.run([sys.executable, "-m", "tracebi.cli", command, name],
                              capture_output=True, text=True)
        ok = proc.returncode == 0
        record["refresh"].append({"step": f"{kind}:{name}", "ok": ok,
                                  "seconds": round(time.monotonic() - started, 2)})
        if not ok:
            tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
            record["error"] = (f"refresh {kind} '{name}' failed, so nothing was "
                               f"built or sent: " + " | ".join(tail))
            return False
    return True


def make_job(reports_dir: Union[str, Path], output_dir: Union[str, Path],
             models_dir: Union[str, Path, None] = None):
    """The job ``tracebi schedule serve`` and the web server both run.

    APScheduler calls it on a worker thread, which does not inherit the
    caller's :class:`~contextvars.ContextVar`, so the actor is set inside.
    """
    def job(schedule: dict) -> dict:
        from tracebi.audit import actor
        with actor("scheduler", role="cli"):
            return run_schedule(schedule, reports_dir=reports_dir,
                                output_dir=output_dir, models_dir=models_dir)
    return job


def start_server_scheduler():
    """Start in-server schedules when ``TRACEBI_SCHEDULES_IN_SERVER=1``.

    Returns the running scheduler, or ``None`` when the switch is off or
    the reports directory has no schedule blocks. A missing APScheduler
    raises ``ImportError`` naming ``tracebi[pipeline]`` — startup must not
    continue as if the schedules were running.
    """
    import logging
    import os

    if os.environ.get("TRACEBI_SCHEDULES_IN_SERVER") != "1":
        return None
    log = logging.getLogger("tracebi.schedule")
    reports_dir = os.environ.get("TRACEBI_REPORTS_DIR", "reports")
    # Same default as `tracebi schedule serve`. TRACEBI_OUTPUT_ROOT is the
    # MCP gateway's confinement root, not where schedule runs are recorded.
    output_dir = "output"
    models_dir = os.environ.get("TRACEBI_MODELS_DIR", "models")
    schedules, errors = discover_library_schedules()
    for err in errors:
        log.warning("skipped %s: %s", err["report"], err["error"])
    if not schedules:
        log.info("TRACEBI_SCHEDULES_IN_SERVER is on; no schedules in the "
                 "report library")
        return None
    log.warning(
        "In-server schedules on SQLite can send the same email from each "
        "worker. Postgres takes one advisory lock per report.")
    scheduler = build_scheduler(
        schedules, make_job(reports_dir, output_dir, models_dir),
        blocking=False)
    for s in schedules:
        log.info("%s", describe_schedule(s))
    scheduler.start()
    return scheduler


@asynccontextmanager
async def server_lifespan(app):
    """Start in-server schedules with the process and stop them on shutdown.

    The web app passes this to ``FastAPI(lifespan=...)``. Tests use the same
    function on a tiny app so they do not import ``tracebi.web.api.main``
    (that import binds the routers to the global registry).
    """
    scheduler = start_server_scheduler()
    app.state.scheduler = scheduler
    try:
        yield
    finally:
        if scheduler is not None:
            scheduler.shutdown(wait=False)


def build_scheduler(schedules: list[dict], job: Callable[[dict], object],
                    blocking: bool = True):
    """An APScheduler instance with one cron job per schedule (not started).

    *job* is called with the schedule dict at each fire time.
    """
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.schedulers.blocking import BlockingScheduler
        from apscheduler.triggers.cron import CronTrigger
    except ImportError:
        raise ImportError(
            "tracebi schedule serve needs APScheduler. "
            "Install with: pip install 'tracebi[pipeline]'"
        ) from None

    scheduler = (BlockingScheduler if blocking else BackgroundScheduler)(
        timezone="UTC")
    for s in schedules:
        scheduler.add_job(
            func=job, args=[s],
            trigger=CronTrigger.from_crontab(s["cron"],
                                             timezone=s.get("timezone") or "UTC"),
            id=s["report"], name=s["report"],
            max_instances=1, coalesce=True, misfire_grace_time=300,
        )
    return scheduler


def describe_schedule(s: dict, last: Optional[dict] = None) -> str:
    """One human line for ``tracebi schedule list``."""
    to = ", ".join(s.get("to") or []) or "(build only)"
    tz = s.get("timezone") or "UTC"
    line = f"{s['report']:<28} {s['cron']:<16} {tz:<18} → {to}"
    if s.get("owner"):
        line += f"   owner: {s['owner']}"
    burst = s.get("burst")
    if burst:
        key, spec = next(iter(burst["filter"].items()))
        line += f"   burst: {key} × {len(spec['in'])}"
    retries = s.get("retries", _DEFAULT_RETRIES)
    if retries != _DEFAULT_RETRIES:
        line += f"   retries: {retries}"
    if last:
        line += f"   last: {last.get('status')} {last.get('finished_at') or ''}"
    return line
