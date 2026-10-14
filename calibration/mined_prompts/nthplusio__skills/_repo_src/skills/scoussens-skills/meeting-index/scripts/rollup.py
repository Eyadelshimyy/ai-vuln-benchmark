#!/usr/bin/env python3
"""Roll up meeting records into local times, durations and totals.

Sources give you a UTC timestamp and a duration in milliseconds. Turning that
into "08:01, 43m" for forty rows plus per-day and overall totals is arithmetic,
and doing it by hand is slow and quietly error-prone. This does it once.

Input is a JSON file: either an array of records, or an object with the array
under `data` (which is what most listing endpoints return, so a raw response can
usually be saved and passed straight in).

Records use the normalised field names, and these aliases are accepted so raw
API responses work without conversion:

    id
    title        <- name
    start        <- start_at, created_at        (ISO 8601)
    duration_ms  <- duration                   (milliseconds)

Usage
-----
    python3 rollup.py records.json --offset-hours -5
    python3 rollup.py records.json --offset-hours -5 --min-seconds 60
    python3 rollup.py records.json --offset-hours -5 --json

Deriving --offset-hours
-----------------------
Timestamps are usually UTC and the user thinks in local time. Do not assume a
zone: take any record whose summary quotes a local time, subtract it from that
record's start, and pass the difference.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta

TITLE_KEYS = ("title", "name")
START_KEYS = ("start", "start_at", "created_at")
DURATION_KEYS = ("duration_ms", "duration")


def first(rec: dict, keys: tuple[str, ...]):
    for k in keys:
        if rec.get(k) not in (None, ""):
            return rec[k]
    return None


def load_records(path: str) -> list[dict]:
    """Accept a bare array or a listing response with a `data` array."""
    try:
        with open(path, encoding="utf-8") as fh:
            blob = json.load(fh)
    except FileNotFoundError:
        sys.exit(f"error: no such file: {path}")
    except json.JSONDecodeError as exc:
        sys.exit(f"error: {path} is not valid JSON ({exc})")

    if isinstance(blob, list):
        return blob
    if isinstance(blob, dict):
        records = blob.get("data")
        if records is None:
            sys.exit(
                "error: JSON object has no 'data' key. Pass a bare array of "
                "records, or a listing response containing one."
            )
        if blob.get("truncated"):
            print(
                f"warning: listing reports truncated=true after scanning "
                f"{blob.get('scanned', '?')} records. The window may be "
                "incomplete — narrow the range and list again in chunks.\n",
                file=sys.stderr,
            )
        if not isinstance(records, list):
            sys.exit("error: 'data' is not an array")
        return records
    sys.exit("error: expected a JSON object or array at the top level")


def parse_start(rec: dict) -> datetime | None:
    raw = first(rec, START_KEYS)
    if raw is None:
        return None
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def fmt_hm(minutes: float) -> str:
    """90.0 -> '1 h 30 min'; 43.0 -> '43 min'."""
    total = int(round(minutes))
    hours, mins = divmod(total, 60)
    if hours and mins:
        return f"{hours} h {mins} min"
    if hours:
        return f"{hours} h"
    return f"{mins} min"


def build(records: list[dict], offset_hours: float, min_seconds: float) -> dict:
    rows: list[dict] = []
    skipped: list[dict] = []

    for rec in records:
        start_utc = parse_start(rec)
        try:
            minutes = float(first(rec, DURATION_KEYS) or 0) / 60000.0
        except (TypeError, ValueError):
            minutes = 0.0

        row = {
            "id": rec.get("id", ""),
            "title": first(rec, TITLE_KEYS) or "(untitled)",
            "minutes": round(minutes, 1),
            "utc": start_utc.strftime("%H:%M") if start_utc else "??:??",
            "date": None,
            "local": "??:??",
            "_sort": datetime.max,
        }
        if start_utc is not None:
            local = start_utc + timedelta(hours=offset_hours)
            row.update(date=local.strftime("%Y-%m-%d"),
                       local=local.strftime("%H:%M"), _sort=local)

        (skipped if minutes * 60 < min_seconds else rows).append(row)

    rows.sort(key=lambda r: r["_sort"])
    clean = lambda r: {k: v for k, v in r.items() if not k.startswith("_")}

    by_day: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_day[row["date"] or "unknown"].append(row)

    days = []
    for date in sorted(by_day):
        entries = by_day[date]
        subtotal = sum(e["minutes"] for e in entries)
        days.append({
            "date": date,
            "weekday": (datetime.strptime(date, "%Y-%m-%d").strftime("%A")
                        if date != "unknown" else "unknown"),
            "count": len(entries),
            "minutes": round(subtotal, 1),
            "pretty": fmt_hm(subtotal),
            "entries": [clean(e) for e in entries],
        })

    grand = sum(r["minutes"] for r in rows)
    return {
        "offset_hours": offset_hours,
        "counted": len(rows),
        "skipped": [clean(s) for s in skipped],
        "total_minutes": round(grand, 1),
        "total_hours": round(grand / 60.0, 1),
        "total_pretty": fmt_hm(grand),
        "days": days,
    }


def emit_text(result: dict) -> None:
    off = result["offset_hours"]
    print(f"Local times shown at UTC{'+' if off >= 0 else '-'}{abs(off):g}\n")

    for day in result["days"]:
        header = f"{day['weekday']}, {day['date']}"
        print(header)
        print("-" * len(header))
        for e in day["entries"]:
            print(f"  {e['local']}  {e['minutes']:>4.0f}m  {e['title'][:74]}")
            if e["id"]:
                print(f"                  ref {str(e['id'])[:8]}  ({e['utc']} UTC)")
        print(f"  => {plural(day['count'], 'meeting')}, {day['pretty']}\n")

    print("=" * 62)
    print(f"TOTAL  {plural(result['counted'], 'meeting')}, "
          f"{result['total_pretty']} ({result['total_hours']} h)")

    if result["skipped"]:
        print(f"\nBelow the duration floor ({len(result['skipped'])}), not counted:")
        for s in result["skipped"]:
            print(f"  {s['minutes'] * 60:5.0f}s  {s['title'][:64]}")
        print("  (Short clips are usually accidental recordings. Confirm before")
        print("   excluding them, and say so in the output.)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Roll up meeting records into local times, durations and totals.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=("Derive --offset-hours by comparing one record's stated local time to\n"
                "its start timestamp. Do not assume a zone."),
    )
    ap.add_argument("records", help="JSON file: array of records, or a listing response")
    ap.add_argument("--offset-hours", type=float, default=0.0,
                    help="hours to add to the timestamp for local time, e.g. -5 (default: 0)")
    ap.add_argument("--min-seconds", type=float, default=0.0,
                    help="report records shorter than this separately, uncounted "
                         "(try 60 to surface accidental clips)")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of text")
    args = ap.parse_args()

    records = load_records(args.records)
    if not records:
        sys.exit("error: no records found in input")

    result = build(records, args.offset_hours, args.min_seconds)
    print(json.dumps(result, indent=2) if args.json else "", end="")
    if not args.json:
        emit_text(result)


if __name__ == "__main__":
    main()
