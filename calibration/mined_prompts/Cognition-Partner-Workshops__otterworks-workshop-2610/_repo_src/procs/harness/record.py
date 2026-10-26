from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any
from uuid import UUID

import psycopg
from psycopg import sql
import yaml
try:
    from procs.harness.fingerprints import fixture_sha, source_sha
    from procs.harness.ledger import scenario_rule_map
except ModuleNotFoundError:
    from fingerprints import fixture_sha, source_sha
    from ledger import scenario_rule_map

ROOT = Path(__file__).resolve().parents[2]
SCENARIOS = ROOT / "procs" / "scenarios"
TRANSCRIPTS = ROOT / "procs" / "transcripts"
DB_FILES = [
    ROOT / "services" / "legacy-billing" / "db" / "schema.sql",
    ROOT / "services" / "legacy-billing" / "db" / "procs" / "plans.sql",
    ROOT / "services" / "legacy-billing" / "db" / "procs" / "rating.sql",
    ROOT / "services" / "legacy-billing" / "db" / "procs" / "invoicing.sql",
    ROOT / "services" / "legacy-billing" / "db" / "procs" / "dunning.sql",
    ROOT / "services" / "legacy-billing" / "db" / "seed.sql",
]

WOULD_OVERWRITE = 2
STACK_UNREACHABLE = 3
SCENARIO_FAILED = 4


def connection():
    return psycopg.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "55432")),
        dbname=os.getenv("DB_NAME", f"billing_{os.getenv('NS', 'dev')}"),
        user=os.getenv("DB_USER", "billing"),
        password=os.getenv("DB_PASSWORD", "billing"),
        connect_timeout=5,
    )


def typed(value: Any, kind: str) -> Any:
    if kind == "uuid":
        return UUID(str(value))
    if kind == "date":
        return date.fromisoformat(str(value))
    if kind == "integer":
        return int(value)
    if kind == "decimal":
        return Decimal(str(value))
    if kind == "boolean":
        return bool(value)
    return value


def normalized(value: Any, kind: str | None = None) -> Any:
    if value is None:
        return None
    if kind == "decimal":
        return str(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
    if isinstance(value, Decimal):
        return str(value)
    if kind == "integer":
        return int(value)
    if kind == "date":
        return date.fromisoformat(str(value)).isoformat()
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, datetime):
        current = value
        if current.tzinfo is not None:
            current = current.astimezone(timezone.utc)
            return current.isoformat(timespec="seconds").replace("+00:00", "Z")
        return current.isoformat(timespec="seconds")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, (list, tuple)):
        return [normalized(item) for item in value]
    if isinstance(value, dict):
        return {str(key): normalized(item) for key, item in sorted(value.items())}
    return value


def query_rows(cursor, query: str, params=()) -> list[dict[str, Any]]:
    cursor.execute(query, params)
    names = [column.name for column in cursor.description] if cursor.description else []
    return [
        {name: normalized(value) for name, value in zip(names, row)}
        for row in cursor.fetchall()
    ]


def reset_database(connection_handle) -> None:
    with connection_handle.cursor() as cursor:
        cursor.execute("DROP SCHEMA IF EXISTS billing CASCADE")
        for path in DB_FILES:
            cursor.execute(path.read_text())
    connection_handle.commit()


def capture_fields(rows: list[dict[str, Any]], specs: list[dict[str, Any]]) -> dict[str, Any]:
    captured: dict[str, Any] = {}
    for spec in specs:
        source = str(spec["from"]) if "from" in spec else None
        values = [row.get(source) for row in rows] if source else []
        if spec.get("first"):
            values = values[:1]
        elif spec.get("last"):
            values = values[-1:]
        if spec.get("collect"):
            captured[spec["name"]] = [
                normalized(value, spec.get("type")) for value in values
            ]
        elif spec.get("collect_rows"):
            captured[spec["name"]] = [
                {
                    key: normalized(row.get(key), kind)
                    for key, kind in spec["columns"].items()
                }
                for row in rows
            ]
        else:
            captured[spec["name"]] = normalized(
                values[0] if values else None, spec.get("type")
            )
    return captured


def run_scenario(connection_handle, scenario: dict[str, Any]) -> dict[str, Any]:
    inputs = scenario.get("inputs", [])
    params = tuple(typed(item.get("value"), item["type"]) for item in inputs)
    entrypoint = scenario["entrypoint"]
    namespace, function = entrypoint.split(".", maxsplit=1)
    qualified_entrypoint = sql.Identifier(namespace, function)
    placeholders = sql.SQL(", ").join(sql.Placeholder() for _ in params)
    with connection_handle.cursor() as cursor:
        execute = cursor.execute
        if scenario["kind"] == "function":
            query = sql.SQL("SELECT * FROM {}({})").format(qualified_entrypoint, placeholders)
            execute(query, params)
            names = [column.name for column in cursor.description] if cursor.description else []
            result_rows = [
                {name: normalized(value) for name, value in zip(names, row)}
                for row in cursor.fetchall()
            ]
        else:
            execute(sql.SQL("CALL {}({})").format(qualified_entrypoint, placeholders), params)
            result_rows = []
        if scenario.get("after_sql"):
            cursor.execute(scenario["after_sql"])
        if scenario.get("capture_query"):
            result_rows = query_rows(cursor, scenario["capture_query"])
        probes = {}
        for probe in scenario.get("probes", []):
            probe_rows = query_rows(cursor, probe["query"])
            probes[probe["id"]] = (
                probe_rows
                if probe.get("collect_rows")
                else (probe_rows[0][next(iter(probe_rows[0]))] if probe_rows else None)
            )
    return {
        "scenario": scenario["id"],
        "module": scenario["module"],
        "entrypoint": entrypoint,
        "inputs": {
            str(item["name"]): normalized(item.get("value"), item.get("type"))
            for item in inputs
        },
        "business_fields": capture_fields(result_rows, scenario.get("fields", [])),
        "probes": probes,
        "rules": [],
    }


def scenario_files(module: str | None) -> list[Path]:
    paths = sorted(SCENARIOS.glob(f"{module}/*.yaml" if module else "*/*.yaml"))
    return paths


def load_scenarios(module: str | None) -> list[dict[str, Any]]:
    return [yaml.safe_load(path.read_text()) for path in scenario_files(module)]


def check_immutability(
    scenarios: list[dict[str, Any]],
    digest: str,
    allow: bool,
    rerecord_reason: str | None,
    transcript_root: Path,
    fixture_digest: str | None = None,
) -> None:
    existing = []
    for scenario in scenarios:
        path = transcript_root / scenario["module"] / f"{scenario['id']}.json"
        if path.exists():
            payload = json.loads(path.read_text())
            existing.append(
                (
                    path,
                    payload.get("source_sha"),
                    payload.get("fixture_sha"),
                )
            )
    if not existing:
        return
    if not allow or (
        all(old_sha == digest for _, old_sha, _ in existing)
        and not any(old_fixture != fixture_digest for _, _, old_fixture in existing)
        and rerecord_reason not in {"harness-change", "scenario-redesign"}
    ):
        names = ", ".join(
            str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
            for path, _, _ in existing
        )
        reason = (
            "unchanged procedure and fixture; pass --rerecord-reason harness-change "
            "or scenario-redesign for an audited non-procedure re-record"
            if allow
            else "pass --allow-rerecord only after procedure source changes"
        )
        raise RuntimeError(f"would overwrite immutable transcript(s): {names} ({reason})")


def write_transcripts(
    records: list[dict[str, Any]],
    digest: str,
    transcript_root: Path,
    fixture_digest: str | None = None,
) -> None:
    if not records:
        return
    transcript_root.mkdir(parents=True, exist_ok=True)
    existing_index_path = transcript_root / "index.json"
    if existing_index_path.exists():
        index_by_key = {
            (item["module"], item["scenario"]): item
            for item in json.loads(existing_index_path.read_text())
        }
    else:
        index_by_key = {}
    payloads = []
    for record in records:
        record["source_sha"] = digest
        if fixture_digest is not None:
            record["fixture_sha"] = fixture_digest
        destination = transcript_root / record["module"] / f"{record['scenario']}.json"
        payloads.append((destination, json.dumps(record, indent=2, sort_keys=True) + "\n"))
        index_by_key[(record["module"], record["scenario"])] = {
            "scenario": record["scenario"],
            "module": record["module"],
            "rules": record["rules"],
        }
    for destination, payload in payloads:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(payload)
    index = sorted(index_by_key.values(), key=lambda item: (item["module"], item["scenario"]))
    (transcript_root / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    (transcript_root / "SOURCE_SHA").write_text(digest + "\n")
    if fixture_digest is not None:
        (transcript_root / "FIXTURE_SHA").write_text(fixture_digest + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module")
    parser.add_argument("--allow-rerecord", action="store_true")
    parser.add_argument(
        "--rerecord-reason", choices=["harness-change", "scenario-redesign"]
    )
    parser.add_argument("--output-dir", type=Path, default=TRANSCRIPTS)
    args = parser.parse_args()
    scenarios = load_scenarios(args.module)
    if args.module and not scenarios:
        print(f"unknown scenario module: {args.module}", file=sys.stderr)
        return SCENARIO_FAILED
    digest = source_sha()
    current_fixture_sha = fixture_sha()
    rules_by_scenario = scenario_rule_map()
    try:
        check_immutability(
            scenarios,
            digest,
            args.allow_rerecord,
            args.rerecord_reason,
            args.output_dir,
            current_fixture_sha,
        )
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return WOULD_OVERWRITE
    try:
        connection_handle = connection()
    except psycopg.Error as error:
        print(f"legacy stack unreachable: {error}", file=sys.stderr)
        return STACK_UNREACHABLE
    records = []
    try:
        for scenario in scenarios:
            reset_database(connection_handle)
            try:
                records.append(run_scenario(connection_handle, scenario))
            except Exception as error:
                connection_handle.rollback()
                print(f"{scenario['id']}: scenario failed: {error}", file=sys.stderr)
                return SCENARIO_FAILED
    finally:
        connection_handle.close()
    if args.rerecord_reason:
        for record in records:
            record["rerecord_reason"] = args.rerecord_reason
    for record in records:
        record["rules"] = rules_by_scenario.get(record["scenario"], [])
    write_transcripts(records, digest, args.output_dir, current_fixture_sha)
    print(f"Recorded {len(records)} scenario(s), SOURCE_SHA={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
