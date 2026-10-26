#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx==0.27.2", "pyyaml==6.0.2", "PyJWT==2.9.0", "redis==5.2.1"]
# ///
"""Incident-responder harness for document-service.

    incident.py arm <scenario>          plant the flaw, seed, start load
    incident.py disarm                  reverse whatever is armed, stop load
    incident.py status                  firing alerts + the numbers behind them
    incident.py verify <scenario> --expect before|after
                                        fail-closed gate: fixture fingerprint,
                                        alert state, before/after thresholds
    incident.py load <scenario> [--duration S]   run the scenario's load in the foreground
    incident.py seed                    idempotent deterministic dataset
    incident.py simulate-alert          print what the Devin webhook / Slack received
    incident.py fingerprint             print the fixture + source fingerprints
    incident.py record --reason "..."   re-pin incident/expected.yaml (audited)

Every command reads incident/scenarios.yaml; targets are overridable with
INCIDENT_BASE_URL, INCIDENT_PROM_URL, INCIDENT_ALERTMANAGER_URL,
INCIDENT_SINK_URL, INCIDENT_REDIS_URL. INCIDENT_LOAD_SCALE (default 1) scales a
scenario's rps and concurrency for `arm` and `load` against targets with less
headroom than the local stack (a single-replica tenant on the shared cluster
runs one 500m-CPU worker); `verify` ignores it and always soaks at the catalog's
pinned profile, so a gate never passes under a lighter load than it recorded.
`verify` always writes a report to incident/reports/ (pass or fail) so a red
gate leaves evidence behind.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import math
import os
import random
import re
import signal
import statistics
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import jwt
import redis
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SCENARIOS_FILE = HERE / "scenarios.yaml"
EXPECTED_FILE = HERE / "expected.yaml"
STATE_DIR = HERE / ".state"
REPORT_DIR = HERE / "reports"
SERVICE_DIR = REPO / "services" / "document-service"

COMPOSE = os.environ.get(
    "INCIDENT_COMPOSE",
    "docker compose -f docker-compose.yml -f docker-compose.infra.yml "
    "-f docker-compose.incident.yml",
).split()

# Inputs that change what a recorded run looks like. Source and fixture are
# fingerprinted separately: the fixture must never drift; the source is
# expected to differ between the before-state and a fix. The migration history
# that exists on the before-state is fixture (rewriting 001-003 would change the
# schema the recording was made against); a fix is allowed to *add* a migration,
# so the versions directory as a whole is source.
MIGRATIONS = "services/document-service/alembic/versions"
FIXTURE_INPUTS = [
    "incident/scenarios.yaml",
    "incident/incident.py",
    "observability/prometheus/incident_alerts.yml",
    "observability/prometheus/prometheus.yml",
    "observability/alertmanager/alertmanager.yml.tmpl",
    "docker-compose.incident.yml",
    f"{MIGRATIONS}/001_initial_schema.py",
    f"{MIGRATIONS}/002_document_stats_rollups.py",
    f"{MIGRATIONS}/003_backfill_word_count.py",
]
SOURCE_INPUTS = [
    "services/document-service/app",
    MIGRATIONS,
    "services/document-service/Dockerfile",
    "services/document-service/pyproject.toml",
    "services/document-service/poetry.lock",
]

DEFAULT_SOAK_SECONDS = 150.0


# ---------------------------------------------------------------------------
# config / helpers
# ---------------------------------------------------------------------------


def load_catalog() -> dict[str, Any]:
    with SCENARIOS_FILE.open() as fh:
        return yaml.safe_load(fh)


def target(cat: dict[str, Any], key: str, env: str) -> str:
    return os.environ.get(env) or cat["targets"][key]


def base_url(cat: dict[str, Any]) -> str:
    return target(cat, "document_service", "INCIDENT_BASE_URL").rstrip("/")


def prom_url(cat: dict[str, Any]) -> str:
    return target(cat, "prometheus", "INCIDENT_PROM_URL").rstrip("/")


def sink_url(cat: dict[str, Any]) -> str:
    return target(cat, "alert_sink", "INCIDENT_SINK_URL").rstrip("/")


def redis_url(cat: dict[str, Any]) -> str:
    return target(cat, "redis", "INCIDENT_REDIS_URL")


def load_scale() -> float:
    raw = os.environ.get("INCIDENT_LOAD_SCALE", "1")
    try:
        scale = float(raw)
    except ValueError:
        scale = 0.0
    if not math.isfinite(scale) or scale <= 0:
        die(f"INCIDENT_LOAD_SCALE must be a positive finite number, got {raw!r}")
    return scale


def jwt_secret() -> str:
    return os.environ.get("JWT_SECRET", "otterworks-local-dev-jwt-secret-change-me-in-production")


def bearer(user_id: str) -> dict[str, str]:
    token = jwt.encode(
        {"user_id": user_id, "sub": user_id, "exp": int(time.time()) + 6 * 3600},
        jwt_secret(),
        algorithm="HS256",
    )
    return {"Authorization": f"Bearer {token}"}


def log(msg: str) -> None:
    print(f"[incident {datetime.now(UTC).strftime('%H:%M:%S')}] {msg}", flush=True)


def die(msg: str, code: int = 2) -> None:
    print(f"[incident] ERROR: {msg}", file=sys.stderr, flush=True)
    sys.exit(code)


def sh(
    *args: str,
    check: bool = True,
    capture: bool = False,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    log("$ " + " ".join(args))
    return subprocess.run(
        args,
        cwd=REPO,
        check=check,
        text=True,
        capture_output=capture,
        env={**os.environ, **env} if env else None,
    )


def compose(
    *args: str, check: bool = True, capture: bool = False, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return sh(*COMPOSE, *args, check=check, capture=capture, env=env)


def recreate_document_service(cat: dict[str, Any], memory_limit: str | None) -> None:
    """Recreate the document-service container with a scenario-specific memory ceiling.

    `None` restores the Compose default (`INCIDENT_MEMORY_LIMIT` or 512m).
    """
    env = {"INCIDENT_MEMORY_LIMIT": memory_limit} if memory_limit else None
    compose("up", "-d", "--no-build", "--force-recreate", "document-service", env=env)
    wait_healthy(cat)
    log(f"document-service memory limit -> {memory_limit or 'default'}")


def container_id(service: str) -> str | None:
    out = compose("ps", "-q", "-a", service, capture=True, check=False).stdout.strip()
    return out.splitlines()[0] if out else None


def container_inspect(service: str, fmt: str) -> str | None:
    cid = container_id(service)
    if not cid:
        return None
    out = sh("docker", "inspect", "--format", fmt, cid, capture=True, check=False)
    return out.stdout.strip() if out.returncode == 0 else None


def container_running(service: str) -> bool:
    return container_inspect(service, "{{.State.Running}}") == "true"


def container_memory_limit(service: str) -> int | None:
    raw = container_inspect(service, "{{.HostConfig.Memory}}")
    return int(raw) if raw and raw.isdigit() else None


_SIZE_UNITS = {"b": 1, "k": 1024, "m": 1024**2, "g": 1024**3}


def parse_size(spec: str) -> int:
    spec = str(spec).strip().lower()
    unit = spec[-1] if spec[-1] in _SIZE_UNITS else "b"
    digits = spec[:-1] if spec[-1] in _SIZE_UNITS else spec
    return int(digits) * _SIZE_UNITS[unit]


def _step_parts(step: Any) -> tuple[str, Any]:
    if isinstance(step, dict):
        ((kind, arg),) = step.items()
        return kind, arg
    return step, None


def arm_conditions(cat: dict[str, Any], sc: dict[str, Any]) -> list[tuple[str, bool]]:
    """(label, holds) for every incident-producing condition the scenario's arm steps create.

    The after gate re-checks these so a fix is only ever judged with the flaw's
    trigger still in place: flags set, memory ceiling applied, replica running.
    """
    conds: list[tuple[str, bool]] = []
    for step in sc["arm"]:
        kind, arg = _step_parts(step)
        match kind:
            case "flag":
                conds.append((f"chaos flag {arg} is set", flag_is_set(cat, str(arg))))
            case "memory-limit":
                want = parse_size(str(arg))
                have = container_memory_limit("document-service")
                conds.append(
                    (f"document-service memory limit is {arg} (have {have})", have == want)
                )
            case "start-replica":
                conds.append(
                    (
                        "document-service-replica is running",
                        container_running("document-service-replica"),
                    )
                )
    return conds


def rollup_soak_metrics(since: datetime) -> dict[str, float | None]:
    """Rollup rows written since `since`: how many distinct windows, and how many duplicates.

    Read straight from Postgres so the double-run after gate proves both that
    the job kept running under two replicas and that no window was computed twice.
    """
    sql = (
        "SELECT count(DISTINCT window_start), count(*) - count(DISTINCT window_start) "
        f"FROM document_stats_rollups WHERE created_at >= '{since.isoformat()}'"
    )
    out = compose(
        "exec",
        "-T",
        "postgres",
        "psql",
        "-U",
        "otterworks",
        "-d",
        "otterworks",
        "-qtAc",
        sql,
        capture=True,
        check=False,
    )
    try:
        windows, dupes = (int(x) for x in out.stdout.strip().split("|"))
    except ValueError:
        return {"rollup_windows_new": None, "rollup_duplicates_new": None}
    return {"rollup_windows_new": float(windows), "rollup_duplicates_new": float(dupes)}


REQUEST_LOG_DIR = "/var/log/otterworks/document-service"


def purge_request_log(cat: dict[str, Any]) -> None:
    """Delete the request-log files inside the container's tmpfs and prove the gauge fell to zero.

    A restart alone does not free the volume: the tmpfs is a container mount and
    outlives the process, so the next run would start with a full disk.
    """
    compose(
        "exec",
        "-T",
        "document-service",
        "sh",
        "-c",
        f"find {REQUEST_LOG_DIR} -mindepth 1 -type f -delete",
        check=False,
    )
    compose("restart", "document-service")
    wait_healthy(cat)
    remaining = compose(
        "exec",
        "-T",
        "document-service",
        "sh",
        "-c",
        f"find {REQUEST_LOG_DIR} -mindepth 1 -type f -printf '%s\\n' 2>/dev/null"
        " | awk '{s+=$1} END {print s+0}'",
        capture=True,
        check=False,
    ).stdout.strip()
    if remaining not in ("", "0"):
        die(f"purge-request-log: {remaining} bytes still in {REQUEST_LOG_DIR} after purge")
    ratio = None
    with contextlib.suppress(Exception):
        ratio = prom_scalar(cat, METRIC_EXPRS["request_log_ratio"])
    log(f"purge-request-log: {REQUEST_LOG_DIR} emptied (bytes=0, gauge ratio={_fmt(ratio)})")


def reset_fixture(cat: dict[str, Any]) -> None:
    """Delete the fixture owner's documents (versions/comments cascade) so `seed` starts clean."""
    owner = cat["seed"]["owner_id"]
    sql = (
        f"WITH gone AS (DELETE FROM documents WHERE owner_id = '{owner}' RETURNING 1) "
        "SELECT count(*) FROM gone"
    )
    out = compose(
        "exec",
        "-T",
        "postgres",
        "psql",
        "-U",
        "otterworks",
        "-d",
        "otterworks",
        "-qtAc",
        sql,
        capture=True,
    )
    log(f"reset-fixture: deleted {out.stdout.strip() or '0'} documents for owner {owner}")


def wait_healthy(cat: dict[str, Any], timeout: float = 90) -> None:
    url = base_url(cat) + "/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if httpx.get(url, timeout=2).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(1)
    die(f"document-service not healthy at {url} after {timeout:.0f}s")


def state_path(name: str) -> Path:
    STATE_DIR.mkdir(exist_ok=True)
    return STATE_DIR / name


def read_state() -> dict[str, Any]:
    p = state_path("armed.json")
    return json.loads(p.read_text()) if p.exists() else {}


def write_state(state: dict[str, Any]) -> None:
    state_path("armed.json").write_text(json.dumps(state, indent=2))


def git_sha() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=False
    )
    return out.stdout.strip() if out.returncode == 0 else "unknown"


# ---------------------------------------------------------------------------
# fingerprints
# ---------------------------------------------------------------------------


def _hash_paths(rel_paths: list[str]) -> str:
    h = hashlib.sha256()
    for rel in rel_paths:
        p = REPO / rel
        files = sorted(f for f in p.rglob("*") if f.is_file()) if p.is_dir() else [p]
        for f in files:
            if "__pycache__" in f.parts or f.suffix == ".pyc":
                continue
            h.update(str(f.relative_to(REPO)).encode())
            h.update(b"\0")
            h.update(f.read_bytes())
            h.update(b"\0")
    return h.hexdigest()


def fingerprints() -> dict[str, str]:
    return {"fixture": _hash_paths(FIXTURE_INPUTS), "source": _hash_paths(SOURCE_INPUTS)}


def load_expected() -> dict[str, Any]:
    if not EXPECTED_FILE.exists():
        die(f"{EXPECTED_FILE.relative_to(REPO)} missing; run `incident.py record --reason ...`")
    with EXPECTED_FILE.open() as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# prometheus
# ---------------------------------------------------------------------------


def prom_query(cat: dict[str, Any], expr: str) -> list[dict[str, Any]]:
    r = httpx.get(f"{prom_url(cat)}/api/v1/query", params={"query": expr}, timeout=10)
    r.raise_for_status()
    body = r.json()
    if body.get("status") != "success":
        raise RuntimeError(f"prometheus: {body}")
    return body["data"]["result"]


def prom_scalar(cat: dict[str, Any], expr: str) -> float | None:
    res = prom_query(cat, expr)
    if not res:
        return None
    try:
        return float(res[0]["value"][1])
    except (KeyError, ValueError, IndexError):
        return None


def prom_alerts(cat: dict[str, Any]) -> list[dict[str, Any]]:
    r = httpx.get(f"{prom_url(cat)}/api/v1/alerts", timeout=10)
    r.raise_for_status()
    return r.json()["data"]["alerts"]


def devin_page_for(
    cat: dict[str, Any], alertname: str, since: str | None = None
) -> dict[str, Any] | None:
    """Newest delivery to the Devin receiver that is a firing page for `alertname`.

    `since` (ISO-8601, UTC) drops captures from earlier runs of the same scenario.
    """
    try:
        r = httpx.get(f"{sink_url(cat)}/devin", timeout=5)
        deliveries = r.json() if r.status_code == 200 else []
    except (httpx.HTTPError, ValueError):
        return None
    for rec in reversed(deliveries or []):
        if since and rec.get("received_at", "") < since:
            continue
        payload = rec.get("payload") or {}
        if payload.get("status") != "firing":
            continue
        if any(
            a.get("labels", {}).get("alertname") == alertname for a in payload.get("alerts", [])
        ):
            return rec
    return None


def alert_state(cat: dict[str, Any], name: str) -> str:
    states = [a["state"] for a in prom_alerts(cat) if a["labels"].get("alertname") == name]
    if "firing" in states:
        return "firing"
    if "pending" in states:
        return "pending"
    return "inactive"


def wait_alert_inactive(cat: dict[str, Any], name: str, budget: float) -> None:
    """Block until `name` is inactive, so the next arm's page is provably fresh.

    A scenario's alert can keep firing for minutes after disarm (rate windows,
    `for`); the before gate rejects any alert that crossed into firing before the
    arm time, so arming under a still-firing alert can only produce a red gate.
    """
    deadline = time.monotonic() + budget
    while (state := alert_state(cat, name)) != "inactive":
        if time.monotonic() >= deadline:
            die(f"{name} still {state} from an earlier run after {budget:.0f}s; let it resolve")
        log(f"{name} is still {state} from an earlier run; waiting for it to resolve before arming")
        time.sleep(10)


def _parse_ts(raw: str) -> datetime:
    # Prometheus emits RFC 3339 with nanoseconds; datetime accepts at most 6 digits.
    raw = re.sub(r"\.(\d{6})\d+", r".\1", raw).replace("Z", "+00:00")
    ts = datetime.fromisoformat(raw)
    return ts if ts.tzinfo else ts.replace(tzinfo=UTC)


def alert_fired_at(cat: dict[str, Any], name: str) -> datetime | None:
    """When the firing alert `name` crossed into firing: activeAt plus the rule's `for`.

    Prometheus only records when the alert became pending (`activeAt`); the rule's
    `duration` is what it had to hold before firing. Returns None when not firing.
    """
    active = [
        a
        for a in prom_alerts(cat)
        if a["labels"].get("alertname") == name and a["state"] == "firing"
    ]
    if not active:
        return None
    active_at = min(_parse_ts(a["activeAt"]) for a in active)
    r = httpx.get(f"{prom_url(cat)}/api/v1/rules", params={"type": "alert"}, timeout=10)
    r.raise_for_status()
    hold = 0.0
    for group in r.json()["data"]["groups"]:
        for rule in group["rules"]:
            if rule.get("name") == name:
                hold = float(rule.get("duration", 0))
    return active_at + timedelta(seconds=hold)


LIST_HANDLER = "/api/v1/documents/"

_LIST = f'job="document-service",handler="{LIST_HANDLER}",method="GET"'
_LIST_H = f'handler="{LIST_HANDLER}"'

METRIC_EXPRS = {
    "request_rate": f"sum(rate(http_requests_total{{{_LIST}}}[2m]))",
    "p95_seconds": (
        "histogram_quantile(0.95, sum by (le) "
        f"(rate(http_request_duration_seconds_bucket{{{_LIST}}}[2m])))"
    ),
    "error_ratio": (
        'sum(rate(http_requests_total{job="document-service",status=~"5.."}[2m])) '
        '/ sum(rate(http_requests_total{job="document-service"}[2m]))'
    ),
    "queries_per_request": (
        f"sum(rate(otterworks_db_queries_per_request_sum{{{_LIST_H}}}[2m])) "
        f"/ sum(rate(otterworks_db_queries_per_request_count{{{_LIST_H}}}[2m]))"
    ),
    "request_log_ratio": (
        "max(otterworks_request_log_bytes / otterworks_request_log_capacity_bytes)"
    ),
    "memory_ratio": (
        "max(otterworks_process_resident_memory_bytes "
        "/ (otterworks_process_memory_limit_bytes > 0))"
    ),
    "cache_entries": "max(otterworks_render_cache_entries)",
    "duplicate_windows": "max(otterworks_rollup_duplicate_windows)",
}


def snapshot_metrics(cat: dict[str, Any]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for key, expr in METRIC_EXPRS.items():
        try:
            out[key] = prom_scalar(cat, expr)
        except Exception as exc:  # noqa: BLE001 - keep going, report what we can
            log(f"prometheus query {key} failed: {exc}")
            out[key] = None
    return out


# ---------------------------------------------------------------------------
# seed
# ---------------------------------------------------------------------------

WORDS = (
    "otter river bank reed willow current eddy stone moss lodge kit whisker "
    "paddle splash dive pool ripple shoal salmon trout fern drift dusk dawn"
).split()


def _lorem(rng: random.Random, words: int) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(words))


async def seed(cat: dict[str, Any]) -> dict[str, int]:
    cfg = cat["seed"]
    owner = cfg["owner_id"]
    rng = random.Random(cfg["random_seed"])
    headers = bearer(owner)
    url = base_url(cat)
    created = versions = 0
    async with httpx.AsyncClient(base_url=url, headers=headers, timeout=60) as client:
        # title -> (id, current version). A document whose version count is short
        # (an earlier seed was interrupted mid-history) is resumed, not skipped, so
        # the fixture is always the full configured history regardless of retries.
        existing: dict[str, tuple[str, int]] = {}
        page = 1
        while True:
            r = await client.get(
                LIST_HANDLER, params={"owner_id": owner, "page": page, "size": 100}
            )
            r.raise_for_status()
            body = r.json()
            for item in body["items"]:
                existing[item["title"]] = (item["id"], int(item["version"]))
            if page * 100 >= body["total"]:
                break
            page += 1
        want_versions = int(cfg["versions_per_document"])
        partial = sum(1 for _, v in existing.values() if v < want_versions)
        log(
            f"seed: {len(existing)} of {cfg['documents']} documents already present "
            f"for owner {owner} ({partial} with an incomplete version history)"
        )

        sem = asyncio.Semaphore(16)

        async def make(i: int) -> None:
            nonlocal created, versions
            title = f"Incident fixture doc {i:04d}"
            doc_rng = random.Random(cfg["random_seed"] * 1000 + i)
            doc_id, have = existing.get(title, (None, 0))
            if doc_id is not None and have >= want_versions:
                return
            async with sem:
                if doc_id is None:
                    r = await client.post(
                        LIST_HANDLER,
                        json={
                            "title": title,
                            "content": _lorem(doc_rng, cfg["words_per_document"]),
                            "content_type": "text/markdown",
                            "owner_id": owner,
                            "folder_id": cfg["folder_id"],
                        },
                    )
                    if r.status_code != 201:
                        raise RuntimeError(f"seed create {title}: {r.status_code} {r.text[:200]}")
                    doc_id = r.json()["id"]
                    created += 1
                    have = 1
                else:
                    # Replay the deterministic content stream up to the version the
                    # document already has so the resumed history is byte-identical.
                    for _ in range(have):
                        _lorem(doc_rng, cfg["words_per_document"])
                for v in range(have + 1, want_versions + 1):
                    r = await client.put(
                        f"{LIST_HANDLER}{doc_id}",
                        json={
                            "title": title,
                            "content": _lorem(doc_rng, cfg["words_per_document"]),
                            "content_type": "text/markdown",
                            "folder_id": cfg["folder_id"],
                        },
                    )
                    if r.status_code != 200:
                        raise RuntimeError(
                            f"seed version {title} v{v}: {r.status_code} {r.text[:200]}"
                        )
                    versions += 1

        await asyncio.gather(*(make(i) for i in range(cfg["documents"])))
    rng.random()  # keep the seed RNG referenced for future deterministic extensions
    log(f"seed: created {created} documents, {versions} extra versions")
    return {"created": created, "versions": versions}


# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------


class Stats:
    def __init__(self) -> None:
        self.latencies: list[float] = []
        self.queries: list[int] = []
        self.errors = 0
        self.total = 0
        self.shed = 0

    def add(self, latency: float, status: int, queries: int | None) -> None:
        self.total += 1
        self.latencies.append(latency)
        if status >= 500:
            self.errors += 1
        if queries is not None:
            self.queries.append(queries)

    def line(self) -> str:
        if not self.latencies:
            return "no requests"
        lat = sorted(self.latencies)
        p50 = lat[len(lat) // 2]
        p95 = lat[min(len(lat) - 1, int(len(lat) * 0.95))]
        q = statistics.mean(self.queries) if self.queries else float("nan")
        return (
            f"n={self.total} p50={p50 * 1000:.0f}ms p95={p95 * 1000:.0f}ms "
            f"queries/req={q:.1f} 5xx={self.errors / self.total:.1%} shed={self.shed}"
        )

    def reset(self) -> None:
        self.__init__()


async def _drive(
    cat: dict[str, Any], scenario: str, duration: float | None, scale: float = 1.0
) -> None:
    sc = cat["scenarios"][scenario]
    profiles = sc.get("load") or {}
    if not profiles:
        log(f"{scenario}: no load profile; nothing to drive")
        return
    name, prof = next(iter(profiles.items()))
    owner = cat["seed"]["owner_id"]
    headers = bearer(owner)
    rps = float(prof["rps"]) * scale
    concurrency = max(1, round(int(prof["concurrency"]) * scale))
    stats = Stats()
    stop = asyncio.Event()

    def _stop(*_: object) -> None:
        stop.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _stop)

    async with httpx.AsyncClient(base_url=base_url(cat), headers=headers, timeout=30) as client:
        # (id, title) so an edit keeps the document's own title; the seed is
        # idempotent on title and a renamed fixture would be re-created.
        docs: list[tuple[str, str]] = []
        if prof.get("flow") == "edit-then-export":
            r = await client.get(LIST_HANDLER, params={"owner_id": owner, "size": 100})
            r.raise_for_status()
            docs = [(d["id"], d["title"]) for d in r.json()["items"]]
            if not docs:
                die("edit-then-export load needs seeded documents; run `incident.py seed`")

        sem = asyncio.Semaphore(concurrency)
        rng = random.Random(cat["seed"]["random_seed"])

        async def one(i: int) -> None:
            async with sem:
                t0 = time.perf_counter()
                try:
                    if prof.get("flow") == "edit-then-export":
                        doc_id, title = docs[i % len(docs)]
                        r = await client.put(
                            f"{LIST_HANDLER}{doc_id}",
                            json={
                                "title": title,
                                "content": _lorem(rng, cat["seed"]["words_per_document"]),
                                "content_type": "text/markdown",
                                "folder_id": cat["seed"]["folder_id"],
                            },
                        )
                        if r.status_code == 200:
                            r = await client.get(
                                f"{LIST_HANDLER}{doc_id}/export",
                                params={"format": "html"},
                            )
                    else:
                        params = dict(prof.get("params") or {})
                        params["owner_id"] = owner
                        r = await client.request(prof["method"], prof["path"], params=params)
                    q = r.headers.get("X-DB-Queries")
                    stats.add(time.perf_counter() - t0, r.status_code, int(q) if q else None)
                except httpx.HTTPError:
                    stats.add(time.perf_counter() - t0, 599, None)

        log(
            f"load[{scenario}/{name}]: {rps:g} rps, concurrency {concurrency}"
            + (f" (INCIDENT_LOAD_SCALE={scale:g})" if scale != 1 else "")
            + f", target {base_url(cat)}"
        )
        started = time.monotonic()
        last_report = started
        i = 0
        tasks: set[asyncio.Task[None]] = set()
        while not stop.is_set():
            if duration and time.monotonic() - started >= duration:
                break
            if sem.locked():
                stats.shed += 1
            else:
                t = asyncio.create_task(one(i))
                tasks.add(t)
                t.add_done_callback(tasks.discard)
                i += 1
            await asyncio.sleep(1.0 / rps)
            if time.monotonic() - last_report >= 5:
                log(f"load[{scenario}]: {stats.line()}")
                stats.reset()
                last_report = time.monotonic()
        await asyncio.gather(*tasks, return_exceptions=True)
        log(f"load[{scenario}]: final {stats.line()}")


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def background_load_pid() -> int | None:
    """PID of the running background load generator, or None when none is alive."""
    p = state_path("load.pid")
    if not p.exists():
        return None
    pid = int(p.read_text().strip() or 0)
    return pid if _pid_alive(pid) else None


def start_background_load(scenario: str) -> int:
    # Re-arming must never strand the previous generator: one load process at a time.
    stop_background_load()
    logf = state_path(f"load-{scenario}.log").open("ab")
    proc = subprocess.Popen(
        [sys.executable, str(HERE / "incident.py"), "load", scenario],
        cwd=REPO,
        stdout=logf,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    state_path("load.pid").write_text(str(proc.pid))
    log(f"load: started pid {proc.pid} (log: {logf.name})")
    return proc.pid


def stop_background_load() -> None:
    p = state_path("load.pid")
    if not p.exists():
        return
    pid = int(p.read_text().strip() or 0)
    try:
        os.killpg(pid, signal.SIGTERM)
        log(f"load: stopped pid {pid}")
        deadline = time.monotonic() + 10
        while _pid_alive(pid) and time.monotonic() < deadline:
            time.sleep(0.2)
        if _pid_alive(pid):
            os.killpg(pid, signal.SIGKILL)
            log(f"load: pid {pid} did not exit on SIGTERM; killed")
    except ProcessLookupError:
        log(f"load: pid {pid} already gone")
    p.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# arm / disarm steps
# ---------------------------------------------------------------------------


def flag_is_set(cat: dict[str, Any], key: str) -> bool:
    try:
        return bool(redis.Redis.from_url(redis_url(cat), socket_timeout=2).exists(key))
    except redis.RedisError:
        return False


def set_flag(cat: dict[str, Any], key: str, on: bool) -> None:
    r = redis.Redis.from_url(redis_url(cat))
    if on:
        r.set(key, "1", ex=4 * 3600)
    else:
        r.delete(key)
    log(f"flag {key} -> {'on' if on else 'off'}")


def run_step(cat: dict[str, Any], scenario: str, step: Any) -> None:
    if isinstance(step, dict):
        ((kind, arg),) = step.items()
    else:
        kind, arg = step, None
    match kind:
        case "seed":
            asyncio.run(seed(cat))
        case "load":
            start_background_load(scenario)
        case "stop-load":
            stop_background_load()
        case "flag":
            set_flag(cat, arg, True)
        case "unflag":
            set_flag(cat, arg, False)
        case "restart-document-service":
            compose("restart", "document-service")
            wait_healthy(cat)
        case "purge-request-log":
            purge_request_log(cat)
        case "memory-limit":
            recreate_document_service(cat, str(arg))
        case "restore-memory-limit":
            recreate_document_service(cat, None)
        case "start-replica":
            compose(
                "--profile",
                "double-run",
                "up",
                "-d",
                "--no-build",
                "--force-recreate",
                "document-service-replica",
            )
        case "stop-replica":
            compose("--profile", "double-run", "rm", "-sf", "document-service-replica", check=False)
        case "purge-duplicate-rollups":
            compose(
                "exec",
                "-T",
                "postgres",
                "psql",
                "-U",
                "otterworks",
                "-d",
                "otterworks",
                "-c",
                "DELETE FROM document_stats_rollups a USING document_stats_rollups b "
                "WHERE a.id > b.id AND a.window_start = b.window_start;",
                check=False,
            )
            compose("restart", "document-service", check=False)
        case _:
            die(f"unknown step {kind!r} in scenario {scenario}")


def cmd_arm(args: argparse.Namespace) -> None:
    cat = load_catalog()
    if args.scenario not in cat["scenarios"]:
        die(f"unknown scenario {args.scenario}; choose from {', '.join(cat['scenarios'])}")
    state = read_state()
    if state.get("scenario") and state["scenario"] != args.scenario:
        die(f"{state['scenario']} is already armed; run `incident.py disarm` first")
    sc = cat["scenarios"][args.scenario]
    scale = load_scale()
    log(f"arming {args.scenario}: {sc['title']}")
    wait_healthy(cat)
    wait_alert_inactive(cat, sc["alert"]["name"], float(sc["alert"]["resolves_within_seconds"]))
    for step in sc["arm"]:
        run_step(cat, args.scenario, step)
    write_state(
        {
            "scenario": args.scenario,
            "armed_at": datetime.now(UTC).isoformat(),
            "git_sha": git_sha(),
            "load_scale": scale,
            "fingerprints": fingerprints(),
        }
    )
    alert = sc["alert"]
    log(
        f"armed. expect alert {alert['name']} to fire within {alert['fires_within_seconds']}s; "
        f"watch with `incident.py status`"
    )


def cmd_disarm(_: argparse.Namespace) -> None:
    cat = load_catalog()
    state = read_state()
    scenarios = [state["scenario"]] if state.get("scenario") else list(cat["scenarios"])
    stop_background_load()
    for name in scenarios:
        for step in cat["scenarios"][name]["disarm"]:
            run_step(cat, name, step)
    state_path("armed.json").unlink(missing_ok=True)
    try:
        httpx.delete(f"{sink_url(cat)}/", timeout=5)
        log("alert sink cleared")
    except httpx.HTTPError as exc:
        log(f"alert sink not reachable ({exc}); skipping")
    log("disarmed")


# ---------------------------------------------------------------------------
# status / simulate
# ---------------------------------------------------------------------------


def cmd_status(_: argparse.Namespace) -> None:
    cat = load_catalog()
    state = read_state()
    print(f"armed: {state.get('scenario') or '-'}  (since {state.get('armed_at', '-')})")
    try:
        alerts = prom_alerts(cat)
    except httpx.HTTPError as exc:
        die(f"prometheus unreachable at {prom_url(cat)}: {exc}")
    paging = [a for a in alerts if a["labels"].get("page") == "devin"]
    other = len(alerts) - len(paging)
    if paging:
        print("alerts (page=devin):")
        for a in paging:
            print(f"  {a['state']:8} {a['labels']['alertname']:32} since {a.get('activeAt', '')}")
    else:
        print("alerts (page=devin): none")
    if other:
        print(f"  ({other} other alert(s) active; only page=devin alerts route to the responder)")
    print("metrics (2m window):")
    for k, v in snapshot_metrics(cat).items():
        print(f"  {k:22} {'-' if v is None else f'{v:.3f}'}")


def cmd_simulate_alert(args: argparse.Namespace) -> None:
    cat = load_catalog()
    url = f"{sink_url(cat)}/{args.receiver}/latest"
    r = httpx.get(url, timeout=10)
    r.raise_for_status()
    record = r.json()
    if record is None:
        die(f"nothing captured yet on receiver {args.receiver!r} (is an alert firing?)", 1)
    print(json.dumps(record, indent=2))


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------


def _check(findings: list[str], ok: bool, msg: str) -> None:
    log(("PASS " if ok else "FAIL ") + msg)
    if not ok:
        findings.append(msg)


NO_SAMPLES = (
    "no samples in the 2m window (Prometheus returned nan/none): the load never "
    "reached the service or its scrape target is down"
)


def _threshold_checks(
    findings: list[str], metrics: dict[str, float | None], thresholds: dict[str, float]
) -> None:
    for key, threshold in thresholds.items():
        metric, bound = key.rsplit("_", 1)
        val = metrics.get(metric)
        op = ">=" if bound == "min" else "<="
        if val is None or math.isnan(val):
            _check(findings, False, f"{metric} {op} {threshold}: {NO_SAMPLES}")
            continue
        ok = val >= threshold if bound == "min" else val <= threshold
        _check(findings, ok, f"{metric}={_fmt(val)} {op} {threshold}")


# Below this fraction of the profile's rps the alert's own rate precondition
# (> 1 rps) can be unmet, so an inactive alert proves nothing about the fix.
MIN_OFFERED_LOAD_FRACTION = 0.5


def _check_load_reached(
    findings: list[str], metrics: dict[str, float | None], sc: dict[str, Any]
) -> None:
    prof = next(iter(sc["load"].values()))
    if prof.get("flow") == "edit-then-export":
        return  # request_rate only covers the list handler
    rps = float(prof["rps"])
    floor = rps * MIN_OFFERED_LOAD_FRACTION
    rate = metrics.get("request_rate")
    if rate is None or math.isnan(rate):
        _check(findings, False, f"offered load reached the service: {NO_SAMPLES}")
        return
    _check(
        findings,
        rate >= floor,
        f"offered load reached the service: request_rate={rate:.2f} >= {floor:g} rps "
        f"(half the pinned {rps:g} rps profile; below this the host cannot sustain the "
        f"workload and the result is inconclusive, not a verdict on the fix)",
    )


def cmd_verify(args: argparse.Namespace) -> None:
    findings: list[str] = []
    started = time.monotonic()
    report: dict[str, Any] = {
        "scenario": args.scenario,
        "expect": args.expect,
        "started_at": datetime.now(UTC).isoformat(),
    }
    # A gate that crashes must still leave its evidence: anything that escapes
    # the checks (Prometheus down, a preparation step failing) becomes a RED
    # finding with the traceback in the report rather than a bare exception.
    try:
        _verify(args, report, findings, started)
    except Exception as exc:
        findings.append(f"verify aborted: {type(exc).__name__}: {exc}")
        report["traceback"] = traceback.format_exc()
        _finish(report, findings, started)


def _verify(
    args: argparse.Namespace, report: dict[str, Any], findings: list[str], started: float
) -> None:
    cat = load_catalog()
    if args.scenario not in cat["scenarios"]:
        die(f"unknown scenario {args.scenario}")
    sc = cat["scenarios"][args.scenario]
    expected = load_expected()
    fp = fingerprints()
    report["git_sha"] = git_sha()
    report["fingerprints"] = fp
    report["expected_fingerprints"] = expected.get("fingerprints", {})

    # 1. Fixture must be byte-identical to what was recorded -- always.
    _check(
        findings,
        fp["fixture"] == expected["fingerprints"]["fixture"],
        f"fixture fingerprint matches expected.yaml ({fp['fixture'][:12]})",
    )
    # 2. Source must match in the before-state; must differ in the after-state.
    same_source = fp["source"] == expected["fingerprints"]["source"]
    if args.expect == "before":
        _check(findings, same_source, "source fingerprint matches the recorded before-state")
    else:
        _check(
            findings,
            not same_source,
            "source fingerprint differs from the before-state (a fix is present)",
        )
    _check(
        findings,
        args.scenario in expected.get("scenarios", {}),
        f"scenario {args.scenario} is pinned in expected.yaml",
    )

    if findings and not args.keep_going:
        _finish(report, findings, started)

    # 3. Runtime. Both gates are judged relative to the armed scenario: the
    # before-state's time-to-fire is measured from `armed_at`, not from whenever
    # verify happened to start, so a late page cannot pass by waiting.
    alert = sc["alert"]
    armed = read_state()
    _check(
        findings,
        armed.get("scenario") == args.scenario,
        f"scenario {args.scenario} is armed "
        f"(the {args.expect}-state is judged under its conditions)",
    )
    if findings and not args.keep_going:
        _finish(report, findings, started)
    armed_at = _parse_ts(armed["armed_at"]) if armed.get("armed_at") else datetime.now(UTC)
    report["armed_at"] = armed_at.isoformat()
    if args.expect == "before":
        budget = float(alert["fires_within_seconds"])
        fire_deadline = armed_at + timedelta(seconds=budget)
        deadline = time.monotonic() + max(0.0, (fire_deadline - datetime.now(UTC)).total_seconds())
        fired_at = alert_fired_at(cat, alert["name"])
        while fired_at is None and time.monotonic() < deadline:
            time.sleep(10)
            fired_at = alert_fired_at(cat, alert["name"])
            log(
                f"waiting for {alert['name']}: {alert_state(cat, alert['name'])} "
                f"({int(deadline - time.monotonic())}s left)"
            )
        time_to_fire = (fired_at - armed_at).total_seconds() if fired_at else None
        report["alert_fired_at"] = fired_at.isoformat() if fired_at else None
        report["time_to_fire_seconds"] = time_to_fire
        _check(
            findings,
            fired_at is not None and armed_at <= fired_at <= fire_deadline,
            f"alert {alert['name']} fired {_fmt(time_to_fire)}s after arming "
            f"(budget {budget:.0f}s; a stale or late alert does not count)",
        )
        metrics = snapshot_metrics(cat)
        report["metrics"] = metrics
        _threshold_checks(findings, metrics, sc.get("before", {}))
        # Alertmanager batches for group_wait before it posts, so allow it the
        # rest of the alert budget (at least 60s); only a *firing* delivery that
        # names this scenario's alert counts, never a stale capture.
        captured = None
        deadline = max(deadline, time.monotonic() + 60)
        armed_iso = armed_at.strftime("%Y-%m-%dT%H:%M:%SZ")
        while captured is None and time.monotonic() < deadline:
            captured = devin_page_for(cat, alert["name"], since=armed_iso)
            if captured is None:
                time.sleep(5)
        report["devin_webhook_captured"] = captured is not None
        report["devin_webhook_received_at"] = captured and captured.get("received_at")
        _check(
            findings,
            captured is not None,
            f"Devin webhook receiver captured a firing {alert['name']} page",
        )
    else:
        # After a fix: the scenario must still be armed, so the fix is judged
        # under the same incident-producing conditions (flags, memory limit,
        # replica) and the same load. Then the alert must be inactive and every
        # scenario-specific after-threshold met.
        # The armed background generator is paused for the whole after-gate: the
        # prepare steps (log purge, restart) must not race live requests, and the
        # gate drives the scenario's profile itself so the offered load is exactly
        # the scenario's, not the profile twice. It is resumed afterwards, pass or
        # fail, so the scenario stays armed for the next attempt.
        had_background = background_load_pid() is not None
        stop_background_load()
        try:
            for step in sc.get("after_prepare", []):
                run_step(cat, args.scenario, step)
            for label, holds in arm_conditions(cat, sc):
                _check(findings, holds, label)
            if findings and not args.keep_going:
                _finish(report, findings, started)
            soak = float(
                args.soak
                if args.soak is not None
                else sc.get("after_soak_seconds", DEFAULT_SOAK_SECONDS)
            )
            report["soak_seconds"] = soak
            report["load_scale"] = 1.0
            soak_started = datetime.now(UTC)
            if sc.get("load"):
                asyncio.run(_drive(cat, args.scenario, duration=soak))
            else:
                log(f"soaking {soak:.0f}s under the armed conditions (no load profile)")
                time.sleep(soak)
            deadline = time.monotonic() + alert["resolves_within_seconds"]
            state = alert_state(cat, alert["name"])
            while state != "inactive" and time.monotonic() < deadline:
                time.sleep(10)
                state = alert_state(cat, alert["name"])
                log(f"waiting for {alert['name']} to clear: {state}")
            _check(
                findings,
                state == "inactive",
                f"alert {alert['name']} is inactive under the same conditions",
            )
            # The trigger must have held for the whole soak, not just at the
            # start: a chaos flag that expired mid-run would make an unfixed
            # flaw look quiet.
            for label, holds in arm_conditions(cat, sc):
                _check(findings, holds, f"{label} (still, after the soak)")
            metrics = snapshot_metrics(cat)
            if any(key.startswith("rollup_") for key in sc.get("after", {})):
                metrics.update(rollup_soak_metrics(soak_started))
            report["metrics"] = metrics
            if sc.get("load"):
                _check_load_reached(findings, metrics, sc)
            _threshold_checks(findings, metrics, sc.get("after", {}))
        finally:
            if had_background:
                start_background_load(args.scenario)

    _finish(report, findings, started)


def _fmt(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def _finish(report: dict[str, Any], findings: list[str], started: float) -> None:
    report["findings"] = findings
    report["passed"] = not findings
    report["duration_seconds"] = round(time.monotonic() - started, 1)
    REPORT_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = REPORT_DIR / f"{report['scenario']}-{report['expect']}-{stamp}.json"
    out.write_text(json.dumps(report, indent=2))
    verdict = "GREEN" if not findings else "RED"
    print(
        f"\nverify {report['scenario']} --expect {report['expect']}: {verdict}  "
        f"(report: {out.relative_to(REPO)})"
    )
    for f in findings:
        print(f"  - {f}")
    sys.exit(0 if not findings else 1)


# ---------------------------------------------------------------------------
# fingerprint / record
# ---------------------------------------------------------------------------


def cmd_fingerprint(_: argparse.Namespace) -> None:
    fp = fingerprints()
    print(yaml.safe_dump({"fingerprints": fp}, sort_keys=False).rstrip())
    if not EXPECTED_FILE.exists():
        return
    exp = load_expected()["fingerprints"]
    drifted = [k for k in ("fixture", "source") if exp.get(k) != fp[k]]
    for k in ("fixture", "source"):
        print(f"{k}: {'DIFFERS from' if k in drifted else 'matches'} expected.yaml")
    if drifted:
        die(
            f"{', '.join(drifted)} fingerprint drifted from expected.yaml "
            "(expected in the after-state; in the before-state it means something "
            "other than the fix changed -- do not `record` past it)"
        )


def cmd_record(args: argparse.Namespace) -> None:
    if not args.reason or len(args.reason.strip()) < 10:
        die("--reason is required (>= 10 characters) and is written into expected.yaml")
    cat = load_catalog()
    doc = {
        "recorded": {
            "at": datetime.now(UTC).isoformat(),
            "git_sha": git_sha(),
            "reason": args.reason.strip(),
        },
        "fingerprints": fingerprints(),
        "scenarios": {
            name: {
                "alert": sc["alert"]["name"],
                "fires_within_seconds": sc["alert"]["fires_within_seconds"],
            }
            for name, sc in cat["scenarios"].items()
        },
    }
    header = (
        "# Recorded before-state for the incident-responder gate. Do not edit by hand;\n"
        '# re-pin with `make incident-record REASON="..."` -- the reason is kept here\n'
        "# so every re-record is auditable in git history.\n"
    )
    EXPECTED_FILE.write_text(header + yaml.safe_dump(doc, sort_keys=False))
    log(f"recorded {EXPECTED_FILE.relative_to(REPO)} ({doc['recorded']['reason']})")


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="incident.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("arm")
    p.add_argument("scenario")
    p.set_defaults(fn=cmd_arm)

    sub.add_parser("disarm").set_defaults(fn=cmd_disarm)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    sub.add_parser("fingerprint").set_defaults(fn=cmd_fingerprint)
    sub.add_parser("seed").set_defaults(fn=lambda _: asyncio.run(seed(load_catalog())))
    sub.add_parser("reset-fixture").set_defaults(fn=lambda _: reset_fixture(load_catalog()))
    sub.add_parser("stop-load").set_defaults(fn=lambda _: stop_background_load())

    p = sub.add_parser("load")
    p.add_argument("scenario")
    p.add_argument(
        "--duration", type=float, default=None, help="seconds; default runs until SIGTERM"
    )
    p.set_defaults(
        fn=lambda a: asyncio.run(_drive(load_catalog(), a.scenario, a.duration, load_scale()))
    )

    p = sub.add_parser("verify")
    p.add_argument("scenario")
    p.add_argument("--expect", choices=["before", "after"], required=True)
    p.add_argument(
        "--soak",
        type=float,
        default=None,
        help=(
            "seconds of load before judging the after-state (default: the scenario's "
            f"after_soak_seconds, else {DEFAULT_SOAK_SECONDS})"
        ),
    )
    p.add_argument(
        "--keep-going",
        action="store_true",
        help="continue past fingerprint failures (diagnostics only)",
    )
    p.set_defaults(fn=cmd_verify)

    p = sub.add_parser("simulate-alert")
    p.add_argument("--receiver", choices=["devin", "slack", "sink"], default="devin")
    p.set_defaults(fn=cmd_simulate_alert)

    p = sub.add_parser("record")
    p.add_argument("--reason", required=True)
    p.set_defaults(fn=cmd_record)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
