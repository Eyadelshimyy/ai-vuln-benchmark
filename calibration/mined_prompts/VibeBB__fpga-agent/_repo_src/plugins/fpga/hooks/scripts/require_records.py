#!/usr/bin/env python3
"""Enforce the VibeBB Record Protocol at SessionStart and Stop.

SessionStart (argv ``session-start``) writes a per-session marker under
``<records_dir>/.sessions/``. Stop (argv ``stop``) refuses to let the
agent finish while this session still owes records:

* every vision tool event and every image the model viewed needs a
  ``vision_review`` with a long-form impression;
* every artifact matching ``artifact_globs`` that changed this session
  must be covered by a fresh ``stage_impression`` (its recorded sha256
  equals the current bytes, so the impression was written after the
  final regeneration);
* a session that changed artifacts must record at least one
  ``decision``;
* every log must keep an intact hash chain and every line must validate;
* every sister impression taken in with an import (``reads.jsonl``)
  must be answered in the ``upstream`` field of a later impression;
* images matching ``dual_review_globs`` need a blind second review and a
  reconcile record; a split needs one tiebreak, after which an
  unresolved split is reported as a warning (unknown), never re-asked;
* every bard song delivered to this plugin needs a song receipt.

SessionStart also prints a short recall of sister concerns aimed at this
plugin, unanswered imported impressions and songs waiting, so the agent
starts with them in context. Nothing here waits on another agent: only
records that already exist are checked, so two plugins reading each
other cannot deadlock.

The refusal is bounded by ``max_stop_denials`` per session so a model
that cannot comply still terminates; the last verdict is written to
``<records_dir>/records-status.json`` either way. Hook errors never
block (exit 0 with a stderr note).
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _records import (
    DEFAULT_MAX_STOP_DENIALS,
    IMAGE_OBSERVATIONS_FILE,
    LOG_FILES,
    VISION_EVENTS_FILE,
    acknowledged,
    changed_artifacts,
    covers,
    load_jsonl,
    load_policy,
    matches,
    parse_time,
    recall,
    received,
    sha256_file,
    song_deliveries,
    tree_sha256,
    verify_log,
)

PLUGIN_ROOT = Path(__file__).resolve().parents[2]
MAX_LISTED = 20


def _project_dir(payload: dict[str, Any]) -> Path:
    return Path(
        os.environ.get("OPENHANDS_PROJECT_DIR") or payload.get("working_dir") or "."
    ).resolve()


def _session_key(payload: dict[str, Any]) -> str:
    raw = str(payload.get("session_id") or "default")
    return re.sub(r"[^A-Za-z0-9._-]+", "-", raw)[:96] or "default"


def _in_session(record: dict[str, Any], session_id: str | None, since: float) -> bool:
    if session_id and record.get("session_id") == session_id:
        return True
    moment = parse_time(record.get("recorded_at"))
    return moment is not None and moment >= since


def _marker_path(records_dir: Path, payload: dict[str, Any]) -> Path:
    return records_dir / ".sessions" / f"{_session_key(payload)}.json"


def _read_marker(path: Path) -> dict[str, Any] | None:
    try:
        value: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return cast(dict[str, Any], value) if isinstance(value, dict) else None


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def session_start(payload: dict[str, Any], policy: dict[str, Any], root: Path) -> int:
    records_dir = root / str(policy["records_dir"])
    marker = _marker_path(records_dir, payload)
    if _read_marker(marker) is None:
        _write_json(
            marker,
            {"session_id": payload.get("session_id"), "started_at": time.time(), "denials": 0},
        )
    briefing = recall(root, str(policy["plugin"]))
    if briefing:
        print(json.dumps({"additionalContext": briefing}, ensure_ascii=False))
    return 0


def _dual_review_problems(
    policy: dict[str, Any], logs: dict[str, list[dict[str, Any]]], since: float
) -> tuple[list[str], list[str]]:
    patterns = cast(list[str], policy.get("dual_review_globs") or [])
    problems: list[str] = []
    warnings: list[str] = []
    if not patterns:
        return problems, warnings
    reviews = logs["vision_review"]
    reconciles = logs["review_reconcile"]
    primaries = [
        r
        for r in reviews
        if r.get("reviewer") == "primary"
        and _in_session(r, None, since)
        and matches(str(r.get("image_path") or ""), patterns)
    ]
    for primary in primaries:
        digest = primary.get("image_sha256")
        image = primary.get("image_path")
        blind = [
            r for r in reviews if r.get("reviewer") == "blind" and r.get("image_sha256") == digest
        ]
        if not blind:
            problems.append(
                f"image {image} needs a blind second review: delegate it with `task` to the"
                " blind vision sub-agent (it must not see the first review)"
            )
            continue
        mine = [
            c
            for c in reconciles
            if primary.get("event_id") in cast(list[str], c.get("reviews") or [])
        ]
        if not mine:
            problems.append(f"image {image}: reconcile the primary and blind reviews")
            continue
        last = max(mine, key=lambda c: int(c.get("round") or 0))
        if last.get("outcome") == "disagree":
            problems.append(
                f"image {image}: the reviews disagree; get one tiebreak review and reconcile"
                " again (round 2 is the last)"
            )
        elif last.get("outcome") == "unresolved":
            warnings.append(f"image {image}: independent reviews stay split; treat it as unknown")
    return problems, warnings


def problems_for(
    payload: dict[str, Any], policy: dict[str, Any], root: Path, since: float
) -> tuple[list[str], list[str]]:
    records_dir = root / str(policy["records_dir"])
    plugin = str(policy["plugin"])
    session_id = cast(str | None, payload.get("session_id"))
    problems: list[str] = []
    warnings: list[str] = []
    logs: dict[str, list[dict[str, Any]]] = {}
    for kind, filename in LOG_FILES.items():
        records, issues = verify_log(records_dir / filename, kind)
        problems.extend(issues)
        logs[kind] = records
    session = {
        kind: [r for r in items if _in_session(r, None, since)] for kind, items in logs.items()
    }

    reviews = session["vision_review"]
    reviewed_events = {str(r.get("source_event_id")) for r in reviews if r.get("source_event_id")}
    reviewed_images = {str(r.get("image_sha256")) for r in reviews if r.get("image_sha256")}
    events, _ = load_jsonl(records_dir / VISION_EVENTS_FILE)
    for event in events:
        if not _in_session(event, session_id, since):
            continue
        if str(event.get("event_id")) not in reviewed_events:
            problems.append(
                f"vision tool event {event.get('event_id')} (question: "
                f"{str(event.get('question'))[:80]!r}) has no vision_review impression"
            )
    observations, _ = load_jsonl(records_dir / IMAGE_OBSERVATIONS_FILE)
    seen: set[str] = set()
    for observation in observations:
        if not _in_session(observation, session_id, since):
            continue
        digest = str(observation.get("image_sha256"))
        if digest in seen:
            continue
        seen.add(digest)
        reviewed = digest in reviewed_images or str(observation.get("event_id")) in reviewed_events
        if not reviewed:
            problems.append(
                f"viewed image {observation.get('image_path')} (sha256 {digest[:12]}) has no"
                " vision_review impression"
            )

    for read in session["upstream_read"]:
        after = parse_time(read.get("recorded_at"))
        for ref in cast(list[dict[str, Any]], read.get("refs") or []):
            if not acknowledged(logs, ref, after):
                problems.append(
                    f"sister impression {ref.get('system')}:{str(ref.get('event_id'))[:12]} came"
                    f" with {cast(dict[str, Any], read.get('artifact') or {}).get('path')}; cite"
                    " it in the upstream field of an impression (adopted, deferred, disputed or"
                    " noted, and its effect)"
                )
        for issue in cast(list[str], read.get("unresolved") or []):
            warnings.append(f"import {read.get('producer')}: {issue}")

    dual, split = _dual_review_problems(policy, logs, since)
    problems.extend(dual)
    warnings.extend(split)

    for relative, doc, issues in song_deliveries(root):
        if doc is None or doc.get("to") != plugin:
            continue
        if issues:
            warnings.append(f"bard song {relative} is malformed: {issues[0]}")
            continue
        if not received(logs, relative, sha256_file(root / relative)):
            problems.append(
                f"bard sent you a song ({relative}); read it and record a song receipt: what"
                " you felt and whether it makes you look at the design again"
            )

    changed = changed_artifacts(root, policy, since)
    if changed:
        fresh: list[str] = []
        for impression in session["stage_impression"]:
            for entry in cast(list[dict[str, Any]], impression.get("artifacts") or []):
                path = str(entry.get("path") or "")
                target = (root / path).resolve()
                try:
                    current = tree_sha256(target) if target.exists() else None
                except OSError:
                    current = None
                if current is not None and current == entry.get("sha256"):
                    fresh.append(path)
        uncovered = [rel for rel in changed if not any(covers(p, rel) for p in fresh)]
        for relative in uncovered:
            problems.append(
                f"artifact {relative} changed this session but no fresh stage_impression covers it"
            )
        if not session["decision"]:
            problems.append(
                "artifacts changed this session but no decision record explains why"
                " (principles, options, evidence, unknowns, risks)"
            )
    return problems, warnings


def stop(payload: dict[str, Any], policy: dict[str, Any], root: Path) -> int:
    records_dir = root / str(policy["records_dir"])
    marker_path = _marker_path(records_dir, payload)
    marker = _read_marker(marker_path)
    if marker is None:
        session_start(payload, policy, root)
        return 0
    since = float(marker.get("started_at") or 0.0)
    problems, warnings = problems_for(payload, policy, root, since)
    status = {
        "plugin": policy["plugin"],
        "session_id": payload.get("session_id"),
        "checked_at": time.time(),
        "verdict": "fail" if problems else "pass",
        "problems": problems,
        "warnings": warnings,
    }
    _write_json(records_dir / "records-status.json", status)
    if not problems:
        return 0
    limit = int(policy.get("max_stop_denials", DEFAULT_MAX_STOP_DENIALS))
    denials = int(marker.get("denials") or 0)
    listed = problems[:MAX_LISTED]
    more = len(problems) - len(listed)
    body = "\n".join(f"- {item}" for item in listed)
    if more > 0:
        body += f"\n- ... and {more} more"
    hint = str(policy.get("record_hint") or "")
    if denials < limit:
        marker["denials"] = denials + 1
        _write_json(marker_path, marker)
        message = (
            f"[{policy['plugin']}] VibeBB Record Protocol: this session still owes records."
            f" Record them before finishing (refusal {denials + 1}/{limit}):\n{body}\n{hint}"
        ).strip()
        print(json.dumps({"decision": "deny", "reason": message, "additionalContext": message}))
        return 2
    message = (
        f"[{policy['plugin']}] VibeBB Record Protocol still unmet after {limit} refusals;"
        f" finishing with gaps recorded in {policy['records_dir']}/records-status.json."
        f" Tell the user exactly which records are missing:\n{body}"
    )
    print(json.dumps({"decision": "allow", "additionalContext": message}))
    return 0


def main(argv: list[str]) -> int:
    try:
        raw: Any = json.load(sys.stdin)
    except (json.JSONDecodeError, OSError):
        raw = {}
    payload = cast(dict[str, Any], raw) if isinstance(raw, dict) else {}
    mode = argv[0] if argv else str(payload.get("event_type") or "")
    try:
        policy = load_policy(PLUGIN_ROOT)
        root = _project_dir(payload)
        if mode in ("session-start", "SessionStart"):
            return session_start(payload, policy, root)
        if mode in ("stop", "Stop"):
            return stop(payload, policy, root)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"require-records hook skipped: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
