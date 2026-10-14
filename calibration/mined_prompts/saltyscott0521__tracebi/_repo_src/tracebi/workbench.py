"""
The workbench — dev-state for the artifact authoring loop (architecture v2 §2.5).

Two shared primitives live here, deliberately outside the dev server:

* :func:`show` — the notebook-cell-output equivalent, callable from
  ``report.py`` or any exploration code. It appends an *exhibit* to the
  session's feed so the workbench becomes the agent's show surface ("steer
  from chat, see results in the portal"). The posting rule, exactly:
  ``TRACEBI_WORKBENCH_DIR`` always wins when set. When it is unset, show()
  posts to ``./.tracebi/workbench/_discovery`` in the CURRENT working
  directory **only if** that directory's ``.active`` heartbeat exists and
  was touched within the last :data:`HEARTBEAT_WINDOW` seconds — i.e. a
  dev server (``tracebi dev``, the app in dev mode) is running right now.
  Otherwise show() is a **no-op**, so a build or CI run (no live server →
  stale or absent heartbeat) ignores it entirely, and it never raises — a
  broken exhibit must not kill a build.

* :func:`collect_state` — THE one state builder. The app's Build panel, the
  classic dev server's ``/__workbench`` page, ``tracebi report status``, and
  the MCP ``workbench_state`` tool all consume this dict; factoring it here
  keeps the surfaces honest copies of one another. Per-binding resolution
  errors are captured *into* the state rather than raised — the workbench
  is where a broken working state becomes visible, so it must render one.

**Hard rule (v2 §2.5): everything here is dev-state.** Exhibits and pins
live under ``.tracebi/workbench/`` in the project, carry no receipts, and
never enter builds or manifests. Nothing in this module writes to a
warehouse or mints a fingerprint claim.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import math
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

EXHIBITS_FILE = "exhibits.jsonl"
PINS_FILE = "pins.json"
POINTING_FILE = "pointing.json"

#: The feed cap — the workbench is a lab log of the session, not an archive.
EXHIBIT_CAP = 100

#: The discovery session's workbench name — ``tracebi dev`` with no name.
DISCOVERY_NAME = "_discovery"

#: The discovery server's liveness marker inside its workbench dir, and how
#: fresh it must be (seconds) for show() to post without the env var set.
ACTIVE_FILE = ".active"
HEARTBEAT_WINDOW = 15.0


def workbench_dir(project_root: str, report_name: str) -> str:
    """``<project_root>/.tracebi/workbench/<report_name>``, created on demand."""
    path = os.path.join(project_root, ".tracebi", "workbench", report_name)
    os.makedirs(path, exist_ok=True)
    return path


# ── Pointing: what the person is pointing at right now ──────────────────────

#: Fields a pointing may carry, with the longest string kept for each. A
#: closed set on purpose: the page that sends one is not trusted with an
#: arbitrary document, and the agent that reads one gets a small, flat dict.
_POINTING_FIELDS = {
    "kind": 16, "figure_kind": 16, "id": 120, "binding": 120, "cell": 120,
    "selector": 300, "tag": 24, "text": 200, "section": 120,
}


def record_pointing(wb_dir: str, pointing: dict) -> dict:
    """Write what the person is pointing at and return what was kept.

    One target at a time: pointing somewhere else replaces it. ``kind`` is
    ``"figure"`` (a ``data-tb-figure`` element: ``figure_kind``, ``binding``,
    ``cell``, ``id``) or ``"element"`` (anything else: ``selector``, ``tag``,
    ``text``, ``section``). Unknown fields are dropped and strings capped, so
    whatever the page sends, the file stays small and flat. Dev-state only.
    """
    kept = {}
    for key, limit in _POINTING_FIELDS.items():
        value = (pointing or {}).get(key)
        if isinstance(value, str) and value.strip():
            kept[key] = " ".join(value.split())[:limit]
    if kept.get("kind") not in ("figure", "element"):
        raise ValueError("pointing needs kind 'figure' or 'element'")
    kept["at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    os.makedirs(wb_dir, exist_ok=True)
    with open(os.path.join(wb_dir, POINTING_FILE), "w", encoding="utf-8") as f:
        json.dump(kept, f, indent=2)
    return kept


def read_pointing(wb_dir: str) -> Optional[dict]:
    """What is being pointed at, or None (nothing, or an unreadable file)."""
    try:
        with open(os.path.join(wb_dir, POINTING_FILE), encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and data.get("kind") else None


def clear_pointing(wb_dir: str) -> None:
    """Stop pointing (the person pressed Escape, or switched report)."""
    try:
        os.remove(os.path.join(wb_dir, POINTING_FILE))
    except OSError:
        pass


def discovery_dir(project_root: str) -> str:
    """The project-level (discovery-mode) workbench directory."""
    return workbench_dir(project_root, DISCOVERY_NAME)


def heartbeat(wb_dir: str) -> None:
    """Touch ``.active`` in *wb_dir* — the discovery server's liveness marker.

    The dev app (and the classic discovery server) calls this on a timer;
    :func:`show` treats a fresh marker as permission to post without
    ``TRACEBI_WORKBENCH_DIR``. Same never-raise contract as the rest of the
    feed plumbing.
    """
    try:
        os.makedirs(wb_dir, exist_ok=True)
        path = os.path.join(wb_dir, ACTIVE_FILE)
        with open(path, "a", encoding="utf-8"):
            pass
        os.utime(path, None)
    except Exception as exc:  # noqa: BLE001 — liveness must not kill the loop
        print(f"[tracebi workbench] heartbeat dropped: "
              f"{type(exc).__name__}: {exc}", file=sys.stderr)


def _active_discovery_dir() -> Optional[str]:
    """cwd's ``.tracebi/workbench/_discovery`` — but only while a discovery
    server's heartbeat is fresh (``.active`` touched within
    :data:`HEARTBEAT_WINDOW` seconds). No live server means a stale or
    absent marker, which keeps the build/CI no-op guarantee."""
    wb = os.path.join(os.getcwd(), ".tracebi", "workbench", DISCOVERY_NAME)
    try:
        age = time.time() - os.path.getmtime(os.path.join(wb, ACTIVE_FILE))
    except OSError:
        return None
    return wb if age <= HEARTBEAT_WINDOW else None


# ── The exhibit feed ────────────────────────────────────────────────────────

#: The chart-sketch vocabulary — the runtime's (CHART_TYPES in
#: tracebi/reports/report.py), mirrored here so show() never has to import
#: the reports stack to validate a sketch.
_CHART_KINDS = ("bar", "barh", "line", "area", "pie", "scatter")


#: Set while the dev app computes state it only reads (collect_state runs
#: report.py): a show() there is not an exhibit anyone asked for, and with the
#: app's heartbeat live it would land in the project feed on every refresh.
_SHOW_OFF: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "tracebi_show_off", default=False)


@contextlib.contextmanager
def show_off():
    """Inside the block, :func:`show` posts nothing."""
    token = _SHOW_OFF.set(True)
    try:
        yield
    finally:
        _SHOW_OFF.reset(token)


def show(obj=None, note: Optional[str] = None, name: Optional[str] = None,
         chart: Optional[str] = None, x: Optional[str] = None,
         y=None) -> None:
    """Append one exhibit to the workbench feed — the notebook-cell output.

    Accepts a pandas DataFrame (excerpt + dtypes + shape recorded), a string
    (a markdown/code note), or — via ``show(name="binding")`` — a binding
    name the workbench renders from its own data panel. Zero ceremony::

        from tracebi.workbench import show
        show(df, note="after dropping the 9 null-mark funds")
        show(df, chart="bar", x="sector", y="fair_value")   # a chart sketch

    ``chart=`` with a DataFrame posts a **chart exhibit**: the same excerpt
    a frame exhibit carries, plus the recipe ``{chart, x, y}`` the workbench
    renders live (``chart`` is the runtime's vocabulary: bar, barh, line,
    area, pie, scatter; ``y`` is a column or a list of columns). Validation
    is soft by this function's contract — an unknown kind or a missing/
    unknown axis degrades to a plain frame exhibit with a stderr note.
    Frame and chart exhibits also carry a :func:`frame_profile` over the
    FULL frame (omitted, never fatal, when profiling fails).
    Every exhibit records its producing script as ``"source"``
    (``sys.argv[0]``, cwd-relative when derivable, absolute otherwise;
    omitted for interactive/no-file contexts).

    Where the exhibit lands, exactly: ``TRACEBI_WORKBENCH_DIR`` always wins
    when set. When it is unset, the exhibit posts to
    ``./.tracebi/workbench/_discovery`` in the CURRENT working directory
    **only if** its ``.active`` heartbeat was touched within the last
    :data:`HEARTBEAT_WINDOW` seconds — a dev server (``tracebi dev``) is
    live, so ANY script run in the project posts frames with zero
    configuration. Otherwise show() is a **no-op** (builds and CI have no
    live server, so a stale or absent heartbeat keeps them clean), and it
    never raises: a broken exhibit is dropped with a stderr note, never a
    dead build. Exhibits carry no receipts.
    """
    if _SHOW_OFF.get():
        return
    wb = os.environ.get("TRACEBI_WORKBENCH_DIR")
    if not wb:
        wb = _active_discovery_dir()
    if not wb:
        return
    try:
        import pandas as pd

        if isinstance(obj, pd.DataFrame):
            entry = {
                "kind": "frame",
                "name": name,
                "note": note,
                "columns": [str(c) for c in obj.columns],
                "dtypes": {str(c): str(t) for c, t in obj.dtypes.items()},
                "shape": [int(obj.shape[0]), int(obj.shape[1])],
                "rows": _json_safe_records(obj, 50),
            }
            if chart is not None:
                recipe, problem = _chart_recipe(chart, x, y, entry["columns"])
                if problem is None:
                    entry["kind"] = "chart"
                    entry["recipe"] = recipe
                else:
                    print(f"[tracebi workbench] chart fell back to frame: "
                          f"{problem}", file=sys.stderr)
            try:
                entry["profile"] = frame_profile(obj)
            except Exception as exc:  # noqa: BLE001 — omit the key, keep the exhibit
                print(f"[tracebi workbench] profile omitted: "
                      f"{type(exc).__name__}: {exc}", file=sys.stderr)
        elif isinstance(obj, str):
            entry = {"kind": "note", "text": obj, "note": note}
        elif obj is None and name:
            entry = {"kind": "binding", "name": name, "note": note}
        else:
            entry = {"kind": "note", "text": repr(obj), "note": note}
        entry["source"] = _script_source()
        origin = _caller_origin(sys._getframe(1))
        if origin:
            entry["origin"] = origin
        entry["step"] = _step_of((origin or {}).get("file") or entry["source"])
        if entry["kind"] in ("frame", "chart"):
            entry["digest"] = _digest(entry)
        _append_exhibit(wb, entry)
    except Exception as exc:  # noqa: BLE001 — by contract, show() never raises
        print(f"[tracebi workbench] exhibit dropped: "
              f"{type(exc).__name__}: {exc}", file=sys.stderr)


def post_agent_action(tool: str, text: str, frame: Optional[dict] = None) -> None:
    """Post what an MCP agent just did to the project feed (*From your agent*).

    The gateway calls this after a tool runs, so a person with the app open
    watches the agent work: a query lands as a frame (``frame`` holds its
    ``columns``, ``shape`` and ``rows``), a build or a refusal as one line.
    Where it lands follows :func:`show` exactly — ``TRACEBI_WORKBENCH_DIR``,
    else the discovery feed only while a dev server's heartbeat is fresh,
    else nothing — and it never raises. Exhibits carry no receipts.
    """
    if _SHOW_OFF.get():
        return
    wb = os.environ.get("TRACEBI_WORKBENCH_DIR") or _active_discovery_dir()
    if not wb:
        return
    try:
        entry = {"kind": "auto", "name": f"agent · {tool}", "text": text,
                 "source": "agent"}
        if frame:
            entry.update(kind="frame", note=text, text=None, **frame)
        _append_exhibit(wb, entry)
    except Exception as exc:  # noqa: BLE001 — same contract as show()
        print(f"[tracebi workbench] agent action dropped: "
              f"{type(exc).__name__}: {exc}", file=sys.stderr)


def _script_source() -> Optional[str]:
    """The running script's path (``sys.argv[0]``) — an exhibit's provenance.

    Relative to the cwd when derivable, absolute when the script lives
    outside it, None for interactive/no-file contexts (a REPL, ``-c``).
    Never raises, by :func:`show`'s contract.
    """
    try:
        argv0 = sys.argv[0] if sys.argv else ""
        if not argv0:
            return None
        path = os.path.abspath(argv0)
        if not os.path.isfile(path):
            return None
        try:
            rel = os.path.relpath(path, os.getcwd())
        except ValueError:      # e.g. a different drive on Windows
            return path
        if rel == os.pardir or rel.startswith(os.pardir + os.sep):
            return path
        return rel
    except Exception:  # noqa: BLE001 — provenance must never drop an exhibit
        return None


def _caller_origin(frame) -> Optional[dict]:
    """Where show() was called from: file (cwd-relative when inside it),
    line, and the few source lines leading up to the call — so an exhibit in
    the feed says which code produced it. Never raises."""
    try:
        import linecache

        path = frame.f_code.co_filename
        line = frame.f_lineno
        origin = {"file": path, "line": line}
        if os.path.isfile(path):
            try:
                rel = os.path.relpath(path, os.getcwd())
                if not rel.startswith(os.pardir):
                    origin["file"] = rel
            except ValueError:
                pass
            if not path.endswith(".ipynb"):     # exec'd cells: lines won't match
                lines = [linecache.getline(path, n).rstrip("\n")
                         for n in range(max(1, line - 3), line + 1)]
                origin["code"] = "\n".join(lines).strip("\n")
        return origin
    except Exception:  # noqa: BLE001 — provenance must never drop an exhibit
        return None


#: The workflow step an exhibit belongs to, from the folder its code lives in.
_STEP_FOLDERS = (("transforms", "transform"), ("models", "model"),
                 ("pipelines", "pipeline"), ("reports", "report"))


def _step_of(path: Optional[str]) -> str:
    """``transform`` / ``model`` / ``pipeline`` / ``report`` from the
    producing file's folder, else ``script``."""
    if not path:
        return "script"
    parts = os.path.normpath(path).split(os.sep)
    for folder, step in _STEP_FOLDERS:
        if folder in parts:
            return step
    return "script"


def _digest(entry: dict) -> str:
    """A short fingerprint of what an exhibit showed (columns, shape, rows),
    so the feed can say whether a re-run changed it. Presentation only."""
    import hashlib

    body = json.dumps([entry.get("columns"), entry.get("shape"),
                       entry.get("rows")], sort_keys=True, default=str)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:12]


def mark_changes(exhibits: list[dict]) -> list[dict]:
    """Label each named frame/chart exhibit ``new``, ``changed`` or ``same``
    against the previous exhibit with that name — did a re-run move it?
    Takes and returns the feed newest-first."""
    last: dict = {}
    marked = {}
    for ex in sorted(exhibits, key=lambda e: e.get("seq", 0)):
        name, digest = ex.get("name"), ex.get("digest")
        if name and digest:
            prev = last.get(name)
            marked[ex.get("seq")] = ("new" if prev is None
                                     else "same" if prev == digest else "changed")
            last[name] = digest
    return [{**ex, "change": marked[ex.get("seq")]} if ex.get("seq") in marked
            else ex for ex in exhibits]


def _display_rows(df, rows: list[dict], declared: Optional[dict] = None) -> list[dict]:
    """Each row of *rows* as the report would write it: the model's declared
    format, else the shape-derived one — the same formatter the built page
    uses. A parallel list; the raw rows stay as they are (for sorting, and
    for agents reading the state)."""
    try:
        from tracebi.reports.derive import derive_number_formats
        from tracebi.reports.template_package import _ssr_cell, _ssr_format

        formats = derive_number_formats(df)
        numeric = {str(c) for c in df.select_dtypes(include="number").columns}
        formats.update({c: f for c, f in (declared or {}).items() if c in numeric})
        out = []
        for r in rows:
            shown = {}
            for c, v in r.items():
                fmt = formats.get(c) if c in numeric else None
                shown[c] = _ssr_format(v, fmt) if (fmt and v is not None) else _ssr_cell(v)
            out.append(shown)
        return out
    except Exception:  # noqa: BLE001 — the page falls back to raw values
        return []


def _feed(wb_dir: str) -> list[dict]:
    """The exhibit feed as the workbench shows it: markdown rendered, re-runs
    marked, and table excerpts carrying display text."""
    import pandas as pd

    feed = mark_changes(render_note_markdown(read_exhibits(wb_dir)))
    out = []
    for ex in feed:
        if ex.get("kind") in ("frame", "chart") and ex.get("rows"):
            try:
                df = pd.DataFrame(ex["rows"], columns=ex.get("columns"))
                ex = {**ex, "display": _display_rows(df, ex["rows"])}
            except Exception:  # noqa: BLE001
                pass
        out.append(ex)
    return out


def _chart_recipe(chart, x, y, columns: list):
    """A chart sketch's recipe dict, or ``(None, why-not)``.

    Soft by the :func:`show` contract: a bad sketch degrades to a frame
    exhibit rather than raising or being dropped.
    """
    if chart not in _CHART_KINDS:
        return None, (f"unknown chart kind {chart!r} — "
                      f"one of {', '.join(_CHART_KINDS)}")
    if isinstance(y, (list, tuple)):
        ys = [str(c) for c in y]
    else:
        ys = [str(y)] if y is not None else []
    if x is None or not ys:
        return None, f"chart={chart!r} needs both x= and y="
    missing = [c for c in [str(x)] + ys if c not in columns]
    if missing:
        return None, f"column(s) not in the frame: {', '.join(missing)}"
    return {"chart": chart, "x": str(x), "y": ys}, None


def auto_entry(wb_dir: str, text: str) -> None:
    """A one-line dev-loop event ("binding X updated · …") for the feed.

    Same never-raise contract as :func:`show` — feed bookkeeping must not
    kill a rebuild.
    """
    try:
        _append_exhibit(wb_dir, {"kind": "auto", "text": text})
    except Exception as exc:  # noqa: BLE001 — same contract as show()
        print(f"[tracebi workbench] auto-entry dropped: "
              f"{type(exc).__name__}: {exc}", file=sys.stderr)


def _append_exhibit(wb_dir: str, entry: dict) -> None:
    """Append one JSON line with a monotonically increasing ``seq``."""
    os.makedirs(wb_dir, exist_ok=True)
    path = os.path.join(wb_dir, EXHIBITS_FILE)
    record = {"seq": _next_seq(path),
              "at": datetime.now().isoformat(timespec="seconds")}
    record.update({k: v for k, v in entry.items() if v is not None})
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def _next_seq(path: str) -> int:
    last = 0
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    last = max(last, int(json.loads(line).get("seq", 0)))
                except (ValueError, TypeError):
                    continue    # a torn line must not stop the feed
    return last + 1


def last_seq(wb_dir: str) -> int:
    """The newest exhibit's seq (0 when the feed is empty) — what a pin
    records as ``at_seq``."""
    return _next_seq(os.path.join(wb_dir, EXHIBITS_FILE)) - 1


def render_note_markdown(exhibits: list[dict]) -> list[dict]:
    """Decorate note exhibits with pre-escaped markdown HTML (``"html"``).

    The notebook-cell experience: ``show("## Approach\\n\\n...")`` renders as
    real markdown in the feed. Conversion happens at STATE time, never at
    write time — ``exhibits.jsonl`` keeps the raw text — and goes through
    the one escaped-first subset (:func:`tracebi.reports.compile_spec.
    md_to_html`), so content can never smuggle live markup; the page may
    insert this one field as HTML because it is safe by construction.
    Never raises: a note that will not convert stays plain text.
    """
    out = []
    for ex in exhibits:
        if ex.get("kind") == "note" and isinstance(ex.get("text"), str):
            try:
                from tracebi.reports.compile_spec import md_to_html
                ex = {**ex, "html": md_to_html(ex["text"])}
            except Exception:  # noqa: BLE001 — plain text is the fallback
                pass
        out.append(ex)
    return out


def read_exhibits(wb_dir: str, cap: int = EXHIBIT_CAP) -> list[dict]:
    """The feed, newest first, capped. Torn lines are skipped, not fatal."""
    path = os.path.join(wb_dir, EXHIBITS_FILE)
    if not os.path.isfile(path):
        return []
    entries: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                parsed = json.loads(line)
            except ValueError:
                continue
            if isinstance(parsed, dict):
                entries.append(parsed)
    entries.sort(key=lambda e: e.get("seq", 0), reverse=True)
    return entries[:cap]


# ── Pins — the one portal→chat gesture ──────────────────────────────────────


def _load_pins(wb_dir: str) -> tuple[list[dict], list[dict]]:
    """Open pins and the resolved list.

    A bare list is open pins only (the file written before resolution
    existed). The object form is ``{"open": [...], "resolved": [...]}``.
    """
    path = os.path.join(wb_dir, PINS_FILE)
    if not os.path.isfile(path):
        return [], []
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (ValueError, OSError):
        return [], []
    if isinstance(data, list):
        return [p for p in data if isinstance(p, dict)], []
    if isinstance(data, dict):
        open_pins = data.get("open")
        resolved = data.get("resolved")
        return (
            [p for p in open_pins if isinstance(p, dict)]
            if isinstance(open_pins, list) else [],
            [p for p in resolved if isinstance(p, dict)]
            if isinstance(resolved, list) else [],
        )
    return [], []


def read_pins(wb_dir: str) -> list[dict]:
    """Open pins as ``{id, note, at_seq}`` (empty when absent).

    Resolved pins are not in this list.
    """
    return _load_pins(wb_dir)[0]


def read_resolved(wb_dir: str) -> list[dict]:
    """Pins moved out of the open list. Empty when the file is a bare list."""
    return _load_pins(wb_dir)[1]


def _save_pins(wb_dir: str, open_pins: list[dict], resolved: list[dict]) -> None:
    os.makedirs(wb_dir, exist_ok=True)
    # A bare list when nothing is resolved, so older readers of open pins
    # keep working. The object form appears once a pin has been resolved.
    payload: object = (
        {"open": open_pins, "resolved": resolved} if resolved else open_pins
    )
    with open(os.path.join(wb_dir, PINS_FILE), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def resolve_pin(wb_dir: str, pin_id: str, note: str = "", by: str = "") -> dict:
    """Move one open pin into the resolved list in the same file.

    The pin is kept, with ``resolved_at``, ``resolved_by`` (``by`` when
    given, otherwise the current ``tracebi.audit`` actor when one is set)
    and ``resolved_note`` (the agent's note). Nothing is deleted. An
    unknown id raises ``ValueError``.
    """
    open_pins, resolved = _load_pins(wb_dir)
    match = next((p for p in open_pins if p.get("id") == pin_id), None)
    if match is None:
        raise ValueError(f"unknown pin id {pin_id!r}")
    if not by:
        from tracebi.audit import get_actor
        by = get_actor()[0] or ""
    moved = {
        **match,
        "resolved_at": datetime.now().isoformat(timespec="seconds"),
        "resolved_by": by,
        "resolved_note": note,
    }
    _save_pins(
        wb_dir,
        [p for p in open_pins if p.get("id") != pin_id],
        resolved + [moved],
    )
    return moved


def promote_request(report: Optional[str], pin: dict, exhibits: list[dict]) -> str:
    """The hand-off for a "Keep this" pin: a plain-language request an agent
    can act on, naming the exhibit, the code that produced it, and the one
    honest way to promote it. The workbench never edits the report itself."""
    seq = pin.get("exhibit")
    ex = next((e for e in exhibits if e.get("seq") == seq), {})
    what = f"workbench exhibit #{seq}" + (f" '{ex['name']}'" if ex.get("name") else "")
    where = f"report '{report}'" if report else "the report it belongs in"
    lines = [f"Keep {what} as a figure in {where}."]
    origin = ex.get("origin") or {}
    if origin.get("file"):
        lines.append(f"It was produced by {origin['file']}:{origin.get('line')}.")
    if ex.get("columns"):
        shape = ex.get("shape") or ["?", "?"]
        lines.append(f"Columns: {', '.join(ex['columns'])} ({shape[0]} rows).")
    lines.append(
        "If the model can express it, add a binding to report.json (a model "
        "query) and a figure in template.html, so it gets a receipt. If it "
        "can't, compute it in report.py, where it is marked python-derived. "
        "Then resolve the pin with a one-line note "
        "(`tracebi report pins <name> --resolve <id> --note ...`).")
    if pin.get("note"):
        lines.append(f"Author's note: {pin['note']}")
    return " ".join(lines)


def _pins_view(pins: list[dict], report: Optional[str], exhibits: list[dict]) -> list[dict]:
    """Pins as the workbench shows them: a promote pin carries its request,
    and a message typed in the timeline carries the author's words."""
    out = []
    for p in pins:
        if p.get("kind") == "promote":
            p = {**p, "request": promote_request(report, p, exhibits)}
        elif p.get("kind") == "message":
            p = {**p, "request": ("The author wrote in the workbench"
                                  + (f" for report '{report}'" if report else "")
                                  + f": {p.get('note', '')}")}
        out.append(p)
    return out


def write_pins(wb_dir: str, pins: list[dict]) -> None:
    """Replace the open list. Pins already resolved stay in the file."""
    _save_pins(wb_dir, pins, read_resolved(wb_dir))


#: Pins are read-modify-written, and two servers (the dev server's threads, the
#: web app's) may add one at once. One lock per process; the file is the truth.
_PINS_LOCK = threading.Lock()


def add_pin(wb_dir: str, pin_id: str, note: str = "", kind: Optional[str] = None,
            exhibit=None) -> list[dict]:
    """Add (or replace) one open pin and return the open list.

    ``kind`` is None (a figure pin: *pin_id* is the figure id), ``"promote"``
    (Keep this on an exhibit) or ``"message"`` (the author typing to the
    agent).
    """
    with _PINS_LOCK:
        pins = [p for p in read_pins(wb_dir) if p.get("id") != pin_id]
        pin = {"id": pin_id, "note": note or "", "at_seq": last_seq(wb_dir)}
        if kind == "promote":
            pin["kind"] = "promote"
            pin["exhibit"] = exhibit
        elif kind == "message":
            pin["kind"] = "message"
            pin["at"] = datetime.now().isoformat(timespec="seconds")
        pins.append(pin)
        write_pins(wb_dir, pins)
        return pins


def remove_pin(wb_dir: str, pin_id: str) -> list[dict]:
    """Drop one open pin (unpin) and return the open list. Resolved pins stay."""
    with _PINS_LOCK:
        pins = [p for p in read_pins(wb_dir) if p.get("id") != pin_id]
        write_pins(wb_dir, pins)
        return pins


#: The longest note kept. The page that sends one is not trusted with an
#: arbitrary document, same as :data:`_POINTING_FIELDS`.
NOTE_MAX = 2000


def leave_note(wb_dir: str, text: str) -> dict:
    """The person's note for the agent, left as a ``message`` pin; returns it.

    What the dev server's "Leave note" box and the app's note box both write:
    the agent reads it as an open pin (``workbench_state``, ``tracebi report
    pins``) and resolves it like any other. An empty note raises ``ValueError``.
    """
    text = (text or "").strip()[:NOTE_MAX]
    if not text:
        raise ValueError("a note needs some text")
    pin_id = f"msg-{int(time.time() * 1000)}"
    return next(p for p in add_pin(wb_dir, pin_id, note=text, kind="message")
                if p["id"] == pin_id)


def package_version(package_dir: str) -> str:
    """A cheap fingerprint of everything that changes what the workbench shows.

    The package's files, the models and transforms it reads, the shared theme,
    and the pins and feed: a path, mtime and size each. A poller compares two
    of these to decide whether to re-render; it renders nothing itself.
    ``pointing.json`` is left out on purpose, since the page that writes it
    would otherwise trigger its own refresh.
    """
    import hashlib

    from tracebi.report_paths import report_name_for_dir

    roots = [Path(package_dir),
             Path(os.environ.get("TRACEBI_MODELS_DIR", "models")),
             Path(os.environ.get("TRACEBI_TRANSFORMS_DIR", "transforms")),
             Path(os.environ.get("TRACEBI_REPORTS_DIR", "reports")) / "_theme.css"]
    wb = os.environ.get("TRACEBI_WORKBENCH_DIR") or workbench_dir(
        os.getcwd(), report_name_for_dir(str(package_dir)))
    roots += [Path(wb) / PINS_FILE, Path(wb) / EXHIBITS_FILE]
    digest = hashlib.sha1()
    for root in roots:
        files = [root] if root.is_file() else (
            sorted(p for p in root.rglob("*") if p.is_file()
                   and "__pycache__" not in p.parts) if root.is_dir() else [])
        for path in files:
            try:
                st = path.stat()
            except OSError:
                continue
            digest.update(f"{path}:{st.st_mtime_ns}:{st.st_size}\n".encode())
    return digest.hexdigest()[:16]


# ── JSON-safe previews ──────────────────────────────────────────────────────


def _json_safe(value):
    """One cell, made JSON-serialisable — NaN/NaT become null (bug #3)."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    try:
        import pandas as pd
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass                    # arrays and other non-scalars fall through
    if isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item"):  # numpy scalar → python scalar
        try:
            return _json_safe(value.item())
        except (ValueError, AttributeError):
            pass
    return str(value)


def _json_safe_records(df, limit: int) -> list[dict]:
    """First *limit* rows as JSON-safe records (presentation only — the
    numbers themselves live in the stamped triple, never here)."""
    return [
        {str(col): _json_safe(v) for col, v in row.items()}
        for _idx, row in df.head(limit).iterrows()
    ]


def _stat_float(value):
    """One summary stat as a JSON-safe float (NaN/NA/non-numeric → None)."""
    safe = _json_safe(value)
    if isinstance(safe, bool) or not isinstance(safe, (int, float)):
        return None
    return float(safe)


def frame_profile(df) -> dict:
    """Per-column stats over the FULL frame — dtype, null count, distinct
    count, min/max/mean for numeric columns, top-3 values for the rest.

    Presentation-grade only, like every excerpt here: the numbers live in
    the frame, never in the profile.
    """
    import pandas as pd

    profile: dict = {}
    for col in df.columns:
        s = df[col]
        stats = {"dtype": str(s.dtype),
                 "nulls": int(s.isna().sum()),
                 "distinct": int(s.nunique())}
        if (pd.api.types.is_numeric_dtype(s)
                and not pd.api.types.is_bool_dtype(s)):
            stats["min"] = _stat_float(s.min())
            stats["max"] = _stat_float(s.max())
            stats["mean"] = _stat_float(s.mean())
        else:
            stats["top"] = [str(v) for v in s.value_counts().head(3).index]
        profile[str(col)] = stats
    return profile


# ── The one state builder ───────────────────────────────────────────────────


def collect_state(package_dir: str, models: dict) -> dict:
    """Build the full workbench state for one artifact package (v2 §2.5).

    The dict every workbench surface renders from — the dev server's
    ``/__workbench`` page, ``tracebi report status``, and the MCP
    ``workbench_state`` tool. Resolves every binding and runs ``report.py``,
    capturing per-binding failures *into* the state; renders the page in
    memory (exploration kept, ids assigned) to extract the figure claims;
    joins the exhibit feed and pins. Nothing is written except the
    on-demand workbench directory itself — dev-state only, no receipts.
    """
    from tracebi.reports.embed import stamp
    from tracebi.reports.figures import (
        FigureError, extract_figures, lint_numeric_literals,
    )
    from tracebi.reports.report import Report, TableSection
    from tracebi.reports.stack import figures_config
    from tracebi.reports.template_package import REPORT_PY, TemplatePackage

    pkg = TemplatePackage(str(package_dir))

    # ── Resolve bindings; a failure is state, not an exception ──────────────
    report = Report(pkg.name)
    inputs = []
    errors: dict[str, str] = {}
    for bname, ref in pkg.bindings.items():
        model = (models or {}).get(ref.model)
        try:
            if model is None:
                raise ValueError(
                    f"model '{ref.model}' is not loaded. "
                    f"Available: {sorted(models or {})}."
                )
            sd = stamp(model, ref.query, name=bname)
        except Exception as exc:  # noqa: BLE001 — captured into the state
            errors[bname] = f"{type(exc).__name__}: {exc}"
            continue
        inputs.append(sd)
        report.add(TableSection(title=bname, dataset=sd.dataset, id=bname))

    outputs = []
    if pkg.report_py_path is not None:
        try:
            outputs = pkg.apply_report_py(report, inputs)
        except Exception as exc:  # noqa: BLE001 — captured into the state
            errors[REPORT_PY] = f"{type(exc).__name__}: {exc}"

    # ── Render in memory (snapshot-style) and extract the figure claims ─────
    figs = []
    page = None
    render_error = None
    try:
        # The one shared render wiring (TemplatePackage.render_page); the dev
        # view keeps exploration blocks — the final build strips them.
        page, _warnings, _unplaced = pkg.render_page(
            report, strip_exploration=False)
    except Exception as exc:  # noqa: BLE001 — captured into the state
        render_error = f"{type(exc).__name__}: {exc}"
        page = None
    if page is not None:
        try:
            figs = extract_figures(page)
        except FigureError as exc:
            render_error = f"{type(exc).__name__}: {exc}"

    # ── Figures: provenance (figures_config rules) + coverage ───────────────
    # Keyed by the DIRECTORY path below reports/ ("finance/weekly"), not the
    # declaration's "name": the address the dev server and discovery share.
    from tracebi.report_paths import report_name_for_dir
    wb = os.environ.get("TRACEBI_WORKBENCH_DIR") or workbench_dir(
        os.getcwd(), report_name_for_dir(str(package_dir)))
    pins = read_pins(wb)
    resolved_pins = read_resolved(wb)
    pinned_ids = {p.get("id") for p in pins}
    output_names = {sd.name for sd in outputs}
    resolved = {sd.name for sd in inputs} | output_names

    coverage = {"total": len(figs), "verified": 0, "derived": 0,
                "unverified": 0, "unbound_errors": 0}
    figures_state = []
    for f, cfg in zip(figs, figures_config(figs, output_names)):
        unbound = not f.unverified and (not f.binding or f.binding not in resolved)
        if f.unverified:
            coverage["unverified"] += 1
        elif unbound:
            coverage["unbound_errors"] += 1
        elif f.binding in output_names:
            coverage["derived"] += 1
        else:
            coverage["verified"] += 1
        entry = {"id": f.id, "kind": f.kind, "binding": f.binding,
                 "provenance": cfg["provenance"], "pinned": f.id in pinned_ids}
        if f.cell:
            entry["cell"] = f.cell
        if f.note:
            entry["note"] = f.note
        if unbound:
            entry["unbound"] = True
        figures_state.append(entry)

    # ── Data panel: one card per binding, declaration order then outputs ────
    used_by: dict[str, list[str]] = {}
    for f in figs:
        if f.binding:
            used_by.setdefault(f.binding, []).append(f.id)
        # A table's totals row reads its own one-row binding.
        totals = f.attrs.get("data-tb-totals")
        if totals:
            used_by.setdefault(totals, []).append(f.id)

    by_name = {sd.name: sd for sd in inputs}
    try:
        declared = pkg._declared_column_formats(
            models or {}, {sd.name: sd.dataset for sd in inputs})
    except Exception:  # noqa: BLE001 — formats are presentation only
        declared = {}
    bindings_state = []
    for bname, ref in pkg.bindings.items():
        if bname in errors:
            bindings_state.append({"name": bname, "source": "query",
                                   "model": ref.model, "error": errors[bname],
                                   "used_by": used_by.get(bname, [])})
        else:
            bindings_state.append({
                **_binding_state(by_name[bname], "query", used_by,
                                 declared.get(bname)),
                "model": ref.model})
    for sd in outputs:
        bindings_state.append(_binding_state(sd, "python", used_by))
    if REPORT_PY in errors:
        bindings_state.append({"name": REPORT_PY, "source": "python",
                               "error": errors[REPORT_PY], "used_by": []})

    unused = [b["name"] for b in bindings_state
              if not b["used_by"] and b["name"] != REPORT_PY]

    exhibits = _feed(wb)
    state = {
        "name": pkg.name,
        "figures": figures_state,
        "coverage": coverage,
        "bindings": bindings_state,
        "unused_bindings": unused,
        "lint": {
            "numeric_literals_outside_figures":
                lint_numeric_literals(page) if page is not None else 0,
        },
        "exhibits": exhibits,
        # What the person is pointing at right now ("this" in their message).
        "pointing": read_pointing(wb),
        "pins": _pins_view(pins, pkg.name, exhibits),
        "resolved": resolved_pins,
        "resolved_count": len(resolved_pins),
        "code": {
            "report.json": _read_optional(os.path.join(pkg.directory,
                                                       "report.json")),
            "report.py": _read_optional(pkg.report_py_path or ""),
            "script.js": pkg.script_js,
        },
    }
    if render_error:
        state["render_error"] = render_error
    return state


def _binding_state(sd, source: str, used_by: dict,
                   declared: Optional[dict] = None) -> dict:
    df = sd.dataset.to_pandas()
    preview = _json_safe_records(df, 25)
    return {
        "name": sd.name,
        "source": source,
        "rows": int(len(df)),
        "columns": [str(c) for c in df.columns],
        "dtypes": {str(c): str(t) for c, t in df.dtypes.items()},
        "fingerprint": sd.fingerprint[:12],
        "preview": preview,
        "display": _display_rows(df, preview, declared),
        "used_by": used_by.get(sd.name, []),
    }


def _read_optional(path: str) -> str:
    if path and os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return f.read()
    return ""


# ── The discovery state builder (no report anchored) ────────────────────────


def collect_discovery_state(project_root: str, models: dict) -> dict:
    """The project-level workbench state — ``tracebi dev`` with no name.

    Discovery mode is the live surface for phases ① and ②: the warehouse's
    tables and sink-contract summaries, every model's declared star schema,
    the report packages that exist, and the ``_discovery`` session's exhibit
    feed and pins. Same contract as :func:`collect_state`: every per-panel
    failure is captured *into* the state (an ``"error"`` string on that
    entry), never raised — the workbench renders broken state. Dev-state
    only; nothing is written except the on-demand workbench directory.
    """
    wb = discovery_dir(project_root)
    exhibits = _feed(wb)
    model_entries, loaded = _discovery_models(project_root, models)
    resolved = read_resolved(wb)
    return {
        "mode": "discovery",
        "name": DISCOVERY_NAME,
        "warehouse": _discovery_warehouse(project_root, loaded),
        "models": model_entries,
        "packages": _discovery_packages(project_root),
        "exhibits": exhibits,
        "pins": _pins_view(read_pins(wb), None, exhibits),
        "resolved": resolved,
        "resolved_count": len(resolved),
    }


def collect_feed(project_root: str) -> dict:
    """The project feed — the ``_discovery`` workbench's exhibits and pins.

    The part of :func:`collect_discovery_state` the app shows when no report is
    open. It leaves out the warehouse, models and packages (the app has pages
    for those), so polling it reads two small files and opens nothing.
    """
    wb = discovery_dir(project_root)
    exhibits = _feed(wb)
    resolved = read_resolved(wb)
    return {
        "exhibits": exhibits,
        "pins": _pins_view(read_pins(wb), None, exhibits),
        "resolved": resolved,
        "resolved_count": len(resolved),
    }


def _discovery_models(project_root: str, models: dict) -> tuple[list, list]:
    """One entry per model. The caller's loaded models render via the public
    ``info()``; every ``models/`` file the caller's loader skipped is
    re-attempted here so its failure lands in the state as an error entry,
    not a silent absence. Returns (entries, deduped loaded models)."""
    entries: list[dict] = []
    loaded: list = []
    seen: set[int] = set()
    for key, m in (models or {}).items():
        if id(m) in seen:
            continue                        # stem + .name index one object
        seen.add(id(m))
        loaded.append(m)
        entries.append(_model_entry(key, m))

    from tracebi.model_registry import ModelRegistry

    reg = ModelRegistry()                   # fresh — never pollutes the global
    models_dir = os.path.join(
        project_root, os.environ.get("TRACEBI_MODELS_DIR", "models"))
    for stem in reg.auto_discover(models_dir):
        if stem in (models or {}):
            continue
        try:
            m = reg.get(stem)
        except Exception as exc:  # noqa: BLE001 — captured into the state
            entries.append({"name": stem,
                            "error": f"{type(exc).__name__}: {exc}"})
            continue
        if id(m) in seen:
            continue
        seen.add(id(m))
        loaded.append(m)
        entries.append(_model_entry(stem, m))
    return entries, loaded


def _model_entry(key, m) -> dict:
    try:
        info = m.info()
        return {"name": info.get("name") or str(key),
                "tables": info.get("tables") or [],
                "facts": info.get("facts") or [],
                "dimensions": info.get("dimensions") or [],
                "measures": info.get("measures") or []}
    except Exception as exc:  # noqa: BLE001 — captured into the state
        return {"name": str(getattr(m, "name", key)),
                "error": f"{type(exc).__name__}: {exc}"}


def _discovery_warehouse(project_root: str, loaded: list) -> dict:
    """The warehouse panel: ``data/warehouse.duckdb`` under *project_root*
    plus any file-backed DuckDB connector database the loaded models declare
    (via the public ``describe()``), deduped by path — the first candidate
    that exists is the warehouse. Introspection uses a SHORT-LIVED read-only
    connection, closed deterministically: this process must never hold a
    handle a transform's ``write()`` would collide with (the same
    coexistence rule ``DuckDBConnector.disconnect`` exists for)."""
    default = os.path.join(project_root, "data", "warehouse.duckdb")
    candidates: list[str] = [default]
    for m in loaded:
        try:
            for c in m.connectors():
                d = c.describe()
                if d.get("type") != "DuckDBConnector":
                    continue
                db = d.get("database") or getattr(c, "database", None)
                if isinstance(db, str) and db and db != ":memory:":
                    candidates.append(db if os.path.isabs(db)
                                      else os.path.join(project_root, db))
        except Exception:  # noqa: BLE001 — a broken model already has its entry
            continue
    seen: set[str] = set()
    deduped: list[str] = []
    for p in candidates:
        ap = os.path.abspath(p)
        if ap not in seen:
            seen.add(ap)
            deduped.append(p)
    existing = [p for p in deduped if os.path.isfile(p)]
    path = existing[0] if existing else default

    wh: dict = {"path": path, "exists": bool(existing),
                "tables": [], "contracts": None}
    if not existing:
        return wh
    try:
        wh["tables"] = _warehouse_tables(path)
    except Exception as exc:  # noqa: BLE001 — captured into the state
        wh["error"] = f"{type(exc).__name__}: {exc}"
    wh["contracts"] = _contracts_summary(path)
    return wh


def warehouse_view(path: str) -> dict:
    """One file-backed warehouse for the app's Warehouse tab: every table with
    its row count, column profile and sink-contract status. The same
    introspection the classic discovery panel uses, on short-lived read-only
    connections closed before this returns. Failures land in ``"error"``."""
    from tracebi.contracts import table_statuses

    view: dict = {"path": path, "exists": os.path.isfile(path), "tables": []}
    if not view["exists"]:
        return view
    try:
        view["tables"] = _warehouse_tables(path)
        statuses = table_statuses(path, [t["name"] for t in view["tables"]])
        for t in view["tables"]:
            t["contract"] = statuses[t["name"]]
    except Exception as exc:  # noqa: BLE001 — a locked file is a result
        view["error"] = f"{type(exc).__name__}: {exc}"
    return view


#: Column cap for warehouse profiles — beyond it the per-column aggregate
#: scan costs more than a glance is worth, so the entry says why instead.
PROFILE_COLUMN_CAP = 50

#: DuckDB data_type prefixes profiled with MIN/MAX/AVG (prefix match, so
#: DECIMAL(18,3) counts and INTERVAL does not).
_NUMERIC_SQL = ("TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT",
                "UTINYINT", "USMALLINT", "UINTEGER", "UBIGINT",
                "FLOAT", "DOUBLE", "REAL", "DECIMAL", "NUMERIC")


def _warehouse_tables(path: str) -> list[dict]:
    """Tables, row counts, column dtypes, and per-column profiles (read-only
    SQL aggregates), on one short-lived read-only connection closed
    deterministically. A table's profile failure lands in that entry's
    ``"error"``, never raised."""
    import duckdb

    tables: list[dict] = []
    con = duckdb.connect(path, read_only=True)
    try:
        names = [r[0] for r in con.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main' ORDER BY table_name").fetchall()]
        for t in names:
            columns = {r[0]: r[1] for r in con.execute(
                "SELECT column_name, data_type "
                "FROM information_schema.columns "
                "WHERE table_schema = 'main' AND table_name = ? "
                "ORDER BY ordinal_position", [t]).fetchall()}
            n = con.execute(
                f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
            entry: dict = {"name": t, "rows": int(n), "columns": columns}
            if len(columns) > PROFILE_COLUMN_CAP:
                entry["profile"] = None
                entry["note"] = (f"profile skipped — {len(columns)} columns "
                                 f"(cap {PROFILE_COLUMN_CAP})")
            else:
                try:
                    entry["profile"] = _table_profile(con, t, columns, int(n))
                except Exception as exc:  # noqa: BLE001 — captured into the entry
                    entry["error"] = f"{type(exc).__name__}: {exc}"
            tables.append(entry)
    finally:
        con.close()
    return tables


def _table_profile(con, table: str, columns: dict, rows: int) -> dict:
    """The :func:`frame_profile` shape via read-only SQL aggregates — the
    frame itself never leaves the warehouse. Top-3 values are computed for
    VARCHAR columns only; other non-numerics keep dtype/nulls/distinct."""
    qt = table.replace('"', '""')
    profile: dict = {}
    for col, dtype in columns.items():
        qc = col.replace('"', '""')
        non_null, distinct = con.execute(
            f'SELECT COUNT("{qc}"), COUNT(DISTINCT "{qc}") FROM "{qt}"'
        ).fetchone()
        stats: dict = {"dtype": dtype,
                       "nulls": rows - int(non_null),
                       "distinct": int(distinct)}
        upper = str(dtype).upper()
        if any(upper.startswith(t) for t in _NUMERIC_SQL):
            lo, hi, mean = con.execute(
                f'SELECT MIN("{qc}"), MAX("{qc}"), AVG("{qc}") FROM "{qt}"'
            ).fetchone()
            # float() first: DuckDB hands DECIMAL aggregates back as
            # decimal.Decimal, which _json_safe would stringify.
            stats["min"] = _stat_float(None if lo is None else float(lo))
            stats["max"] = _stat_float(None if hi is None else float(hi))
            stats["mean"] = _stat_float(None if mean is None else float(mean))
        elif upper.startswith("VARCHAR"):
            stats["top"] = [str(r[0]) for r in con.execute(
                f'SELECT "{qc}" FROM "{qt}" WHERE "{qc}" IS NOT NULL '
                f'GROUP BY "{qc}" ORDER BY COUNT(*) DESC LIMIT 3'
            ).fetchall()]
        profile[col] = stats
    return profile


def _contracts_summary(warehouse: str) -> Optional[dict]:
    """``read_contracts`` parsed down to one line per transform:
    ``{checked_at, checks: "passed/total", tables: [names]}``. None when no
    certificate sits beside the warehouse."""
    from tracebi.contracts import read_contracts

    data = read_contracts(warehouse)
    if not data:
        return None
    out: dict[str, dict] = {}
    for tname, rec in (data.get("transforms") or {}).items():
        if not isinstance(rec, dict):
            continue
        checks = [c for c in (rec.get("checks") or []) if isinstance(c, dict)]
        passed = sum(1 for c in checks if c.get("passed"))
        out[tname] = {"checked_at": rec.get("checked_at"),
                      "checks": f"{passed}/{len(checks)}",
                      "tables": sorted(rec.get("tables") or {})}
    return out or None


def _discovery_packages(project_root: str) -> list[str]:
    """Package names in the library, folders included (``finance/weekly``).

    A directory with ``report.json`` or ``template.html`` is a package. One
    with neither and no subdirectories still counts, so a package-in-progress
    is visible; one with subdirectories is a folder, and is walked.
    Honours ``TRACEBI_LIBRARY_MOUNTS``.
    """
    from tracebi.report_paths import library_roots

    found: list[str] = []

    def walk(directory: str, prefix: str) -> None:
        try:
            entries = sorted(os.listdir(directory))
        except OSError:
            return
        for entry in entries:
            full = os.path.join(directory, entry)
            if entry.startswith((".", "_")) or not os.path.isdir(full):
                continue
            inside = os.listdir(full)
            if "report.json" in inside or "template.html" in inside or not any(
                    os.path.isdir(os.path.join(full, e)) for e in inside):
                found.append(prefix + entry)
            else:
                walk(full, f"{prefix}{entry}/")

    for label, root in library_roots():
        root_path = root if root.is_absolute() else Path(project_root) / root
        pfx = f"{label}/" if label else ""
        walk(str(root_path), pfx)
    return found
