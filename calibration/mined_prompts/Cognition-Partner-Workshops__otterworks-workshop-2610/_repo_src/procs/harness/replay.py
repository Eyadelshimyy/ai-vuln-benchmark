from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import subprocess
import sys
from datetime import date
from decimal import Decimal, ROUND_HALF_UP
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml
try:
    from procs.harness.fingerprints import fixture_sha, source_sha
    from procs.harness.status import EXTRACTED, PENDING, VALID_STATUSES, status_for
except ModuleNotFoundError:
    from fingerprints import fixture_sha, source_sha
    from status import EXTRACTED, PENDING, VALID_STATUSES, status_for

ROOT = Path(__file__).resolve().parents[2]
TRANSCRIPTS = ROOT / "procs" / "transcripts"
ROUTES = ROOT / "procs" / "routes.yaml"
REPORT_DIR = ROOT / "procs" / "reports"

FAIL = 1
TARGET_UNREACHABLE = 3
CONTRACT_MISSING = 6
SOURCE_MISMATCH = 7
RESET_FAILED = 8
SELECTION_EMPTY = 9


@dataclass(frozen=True)
class UnresolvedPath:
    path: str
    reason: str

    def describe(self) -> str:
        return f"<unresolvable {self.path}: {self.reason}>"


def source_matches(expected: str) -> bool:
    return expected == source_sha()


def normalize(value: Any, kind: str | None = None) -> Any:
    if value is None:
        return None
    if kind == "decimal":
        return str(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
    if kind == "integer":
        return int(value)
    if kind == "date":
        return date.fromisoformat(str(value)).isoformat()
    if isinstance(value, list):
        return [normalize(item) for item in value]
    if isinstance(value, dict):
        return {key: normalize(item) for key, item in sorted(value.items())}
    return value


def json_path(value: Any, path: str) -> Any:
    if path == "$":
        return value
    current: list[Any] = [value]
    tokens = re.findall(r"\[\*\]|\[\d+\]|[^.\[\]]+", path[1:])
    for token in tokens:
        if token == "[*]":
            if any(not isinstance(item, list) for item in current):
                return UnresolvedPath(path, "expected a list for [*]")
            current = [child for item in current for child in item]
        elif token.startswith("["):
            index = int(token[1:-1])
            if any(not isinstance(item, list) or index >= len(item) for item in current):
                return UnresolvedPath(path, f"missing list index {index}")
            current = [item[index] for item in current]
        else:
            if any(not isinstance(item, dict) or token not in item for item in current):
                return UnresolvedPath(path, f"missing field {token}")
            current = [item[token] for item in current]
    return current if "[*]" in path else (current[0] if current else None)


def extract(payload: Any, spec: dict[str, Any]) -> Any:
    value = json_path(payload, spec["json_path"])
    if isinstance(value, UnresolvedPath):
        return value
    if spec.get("type") == "rows":
        if not isinstance(value, list):
            return UnresolvedPath(spec["json_path"], "expected a list of rows")
        rows = [normalize(row) for row in value]
        if any(not isinstance(row, dict) for row in rows):
            return UnresolvedPath(spec["json_path"], "expected object rows")
        return rows
    if spec.get("collect"):
        if not isinstance(value, list):
            return UnresolvedPath(spec["json_path"], "expected a list to collect")
        return [normalize(item, spec.get("type")) for item in value]
    return normalize(value, spec.get("type"))


def request_json(
    base_url: str,
    method: str,
    path: str,
    body: dict[str, Any] | None = None,
) -> tuple[int, Any]:
    data = json.dumps(body).encode() if body is not None else None
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise OSError(f"unsupported target URL: {base_url}")
    connection_type = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    connection = connection_type(parsed.netloc, timeout=5)
    try:
        connection.request(
            method,
            f"{parsed.path.rstrip('/')}{path}",
            body=data,
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        raw = response.read()
        return response.status, json.loads(raw) if raw else None
    finally:
        connection.close()


def target_request(
    base_url: str, contract: dict[str, Any], inputs: dict[str, Any]
) -> tuple[int, Any]:
    path = contract["path"]
    for name, input_name in contract.get("inputs", {}).get("path", {}).items():
        path = path.replace("{" + name + "}", str(inputs[input_name]))
    query = contract.get("inputs", {}).get("query", {})
    if query:
        path += "?" + "&".join(f"{name}={inputs[input_name]}" for name, input_name in query.items())
    body = {
        name: inputs[input_name]
        for name, input_name in contract.get("inputs", {}).get("body", {}).items()
    }
    return request_json(base_url, contract["method"], path, body or None)


def load_transcripts(module: str | None) -> list[dict[str, Any]]:
    paths = sorted(TRANSCRIPTS.glob(f"{module}/*.json" if module else "*/*.json"))
    return [json.loads(path.read_text()) for path in paths]


def stale_transcripts(transcripts: list[dict[str, Any]], current_sha: str) -> list[str]:
    return [
        f"{item['scenario']}: transcript SOURCE_SHA {item.get('source_sha', '<missing>')} "
        f"does not match current procs {current_sha}"
        for item in transcripts
        if item.get("source_sha") != current_sha
    ]


def stale_fixture_transcripts(
    transcripts: list[dict[str, Any]], current_sha: str
) -> list[str]:
    return [
        f"{item['scenario']}: transcript FIXTURE_SHA {item.get('fixture_sha', '<missing>')} "
        f"does not match current fixture {current_sha}"
        for item in transcripts
        if item.get("fixture_sha") != current_sha
    ]


def compare(
    transcript: dict[str, Any], contract: dict[str, Any], payload: Any
) -> tuple[list[dict[str, Any]], list[str]]:
    failures = []
    contract_errors = []
    business_contract = contract["response"].get("business_fields", {})
    for field, expected in transcript["business_fields"].items():
        spec = business_contract.get(field)
        if spec is None:
            contract_errors.append(f"{transcript['scenario']}: business field {field} is unmapped")
            continue
        actual = extract(payload, spec)
        if isinstance(actual, UnresolvedPath):
            actual = actual.describe()
        if expected != actual:
            failures.append({"kind": "field", "name": field, "expected": expected, "actual": actual})
    probe_contract = contract["response"].get("probes", {})
    for probe_id, expected in transcript.get("probes", {}).items():
        spec = probe_contract.get(probe_id)
        if spec is None:
            contract_errors.append(f"{transcript['scenario']}: probe {probe_id} is unmapped")
            continue
        actual = extract(payload, spec)
        if isinstance(actual, UnresolvedPath):
            actual = actual.describe()
        if expected != actual:
            failures.append({"kind": "probe", "name": probe_id, "expected": expected, "actual": actual})
    return failures, contract_errors


def grade_response(
    transcript: dict[str, Any], contract: dict[str, Any], status: int, payload: Any
) -> tuple[list[dict[str, Any]], list[str]]:
    failures = [] if status == 200 else [
        {"kind": "status", "name": "status", "expected": 200, "actual": status}
    ]
    if status != 200:
        return failures, []
    compare_failures, contract_errors = compare(transcript, contract, payload)
    failures.extend(compare_failures)
    return failures, contract_errors


def classify_transcript(transcript: dict[str, Any], routes: dict[str, Any]) -> str:
    module = routes.get(transcript["module"])
    status = status_for(module)
    if status == PENDING:
        return "SKIP"
    if status == EXTRACTED:
        return "GRADE"
    return "CONTRACT_ERROR"


def reset_target(base_url: str) -> tuple[bool, str]:
    try:
        status, payload = request_json(base_url, "POST", "/internal/reset")
    except (http.client.HTTPException, OSError) as error:
        return False, f"target unreachable during reset: {error}"
    if status != 204:
        if status in {403, 404}:
            return False, (
                f"target reset refused with HTTP {status}: {payload}; "
                "enable BILLING_SVC_ALLOW_INTERNAL_RESET for disposable stacks"
            )
        return False, f"target reset returned HTTP {status}: {payload}"
    return True, ""


def write_report(
    current_sha: str,
    results: list[dict[str, Any]],
    contract_errors: list[str],
    current_fixture_sha: str | None = None,
) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = {"source_sha": current_sha, "results": results, "contract_errors": contract_errors}
    if current_fixture_sha is not None:
        report["fixture_sha"] = current_fixture_sha
    (REPORT_DIR / "parity.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    lines = ["# Billing parity report", "", f"Source SHA: `{current_sha}`"]
    if current_fixture_sha is not None:
        lines.append(f"Fixture SHA: `{current_fixture_sha}`")
    lines.append("")
    for item in results:
        lines.append(f"- **{item['status']}** `{item['module']}/{item['scenario']}`")
        for failure in item.get("failures", []):
            lines.append(
                f"  - {failure['kind']} `{failure['name']}`: expected `{failure['expected']}`, "
                f"actual `{failure['actual']}`"
            )
    lines.extend([f"- Contract error: {error}" for error in contract_errors])
    (REPORT_DIR / "parity.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module")
    parser.add_argument("--scenario")
    parser.add_argument("--base-url", default=os.getenv("BILLING_SVC_URL", "http://localhost:8097"))
    args = parser.parse_args()
    routes = yaml.safe_load(ROUTES.read_text())["modules"]
    expected_sha = (TRANSCRIPTS / "SOURCE_SHA").read_text().strip()
    expected_fixture_sha = (TRANSCRIPTS / "FIXTURE_SHA").read_text().strip()
    current_sha = source_sha()
    current_fixture_digest = fixture_sha()
    if not source_matches(expected_sha):
        print(f"transcript SOURCE_SHA {expected_sha} does not match current procs {current_sha}", file=sys.stderr)
        return SOURCE_MISMATCH
    if expected_fixture_sha != current_fixture_digest:
        print(
            f"transcript FIXTURE_SHA {expected_fixture_sha} does not match "
            f"current fixture {current_fixture_digest}",
            file=sys.stderr,
        )
        return SOURCE_MISMATCH
    selected = [
        item
        for item in load_transcripts(args.module)
        if args.scenario is None or item["scenario"] == args.scenario
    ]
    if not selected:
        selection = args.module or args.scenario or "requested selection"
        print(f"no transcripts matched {selection}", file=sys.stderr)
        return SELECTION_EMPTY
    stale = stale_transcripts(selected, current_sha)
    if stale:
        for error in stale:
            print(error, file=sys.stderr)
        return SOURCE_MISMATCH
    stale_fixtures = stale_fixture_transcripts(selected, current_fixture_digest)
    if stale_fixtures:
        for error in stale_fixtures:
            print(error, file=sys.stderr)
        return SOURCE_MISMATCH
    invalid_statuses = {
        (item["module"], status_for(routes.get(item["module"])))
        for item in selected
        if status_for(routes.get(item["module"])) not in VALID_STATUSES
    }
    if invalid_statuses:
        for module, status in sorted(invalid_statuses):
            print(f"{module}: invalid module status {status!r}", file=sys.stderr)
        return CONTRACT_MISSING
    extracted_modules = {
        item["module"] for item in selected if status_for(routes.get(item["module"])) == EXTRACTED
    }
    for module in sorted(extracted_modules):
        gate = subprocess.run(
            [sys.executable, str(ROOT / "procs" / "harness" / "rules_gate.py"), "--module", module],
            capture_output=True,
            text=True,
            check=False,
        )
        if gate.returncode:
            print(f"rules gate for {module} is not green:\n{gate.stderr or gate.stdout}", file=sys.stderr)
            return CONTRACT_MISSING
    results = []
    contract_errors = []
    ok, reset_error = reset_target(args.base_url)
    if not ok:
        print(reset_error, file=sys.stderr)
        write_report(current_sha, results, contract_errors, current_fixture_digest)
        return TARGET_UNREACHABLE if reset_error.startswith("target unreachable") else RESET_FAILED
    for transcript in selected:
        module = routes.get(transcript["module"])
        if module is None:
            contract_errors.append(f"{transcript['scenario']}: module missing from routes.yaml")
            continue
        if classify_transcript(transcript, routes) == "SKIP":
            results.append({"scenario": transcript["scenario"], "module": transcript["module"], "status": "SKIP"})
            continue
        contract = module.get("entrypoints", {}).get(transcript["entrypoint"])
        if contract is None:
            contract_errors.append(f"{transcript['scenario']}: entrypoint {transcript['entrypoint']} missing")
            continue
        ok, reset_error = reset_target(args.base_url)
        if not ok:
            print(f"{transcript['scenario']}: {reset_error}", file=sys.stderr)
            write_report(current_sha, results, contract_errors, current_fixture_digest)
            return TARGET_UNREACHABLE if reset_error.startswith("target unreachable") else RESET_FAILED
        try:
            status, payload = target_request(args.base_url, contract, transcript["inputs"])
        except (http.client.HTTPException, OSError) as error:
            print(f"target unreachable during {transcript['scenario']}: {error}", file=sys.stderr)
            write_report(current_sha, results, contract_errors, current_fixture_digest)
            return TARGET_UNREACHABLE
        failures, errors = grade_response(transcript, contract, status, payload)
        contract_errors.extend(errors)
        results.append({
            "scenario": transcript["scenario"],
            "module": transcript["module"],
            "status": "PASS" if not failures and not errors else "FAIL",
            "failures": failures,
        })
    if contract_errors:
        for error in contract_errors:
            print(error, file=sys.stderr)
        exit_code = CONTRACT_MISSING
    else:
        exit_code = FAIL if any(item["status"] == "FAIL" for item in results) else 0
    write_report(current_sha, results, contract_errors, current_fixture_digest)
    print(
        f"Parity PASS={sum(item['status'] == 'PASS' for item in results)} "
        f"FAIL={sum(item['status'] == 'FAIL' for item in results)} "
        f"SKIP={sum(item['status'] == 'SKIP' for item in results)}"
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
