"""
Shared helpers for OtterWorks bulk seed data generation.

Provides deterministic UUID generation, shared constants, and DB connection
so all 10 data modules produce referentially-consistent data without needing
to query the database for parent IDs.
"""

import os
import uuid
from datetime import datetime, timedelta, timezone

import bcrypt
import psycopg2
from psycopg2.extras import execute_values, Json

# ── Deterministic UUID namespace ──────────────────────────────────────────────
# All modules use uuid5 with this namespace so IDs are stable across runs
# and cross-module references resolve correctly.

NS_UUID = uuid.UUID("d4e7f8a1-2b3c-4d5e-9f0a-1b2c3d4e5f6a")


def stable_id(category: str, index: int) -> str:
    return str(uuid.uuid5(NS_UUID, f"{category}-{index}"))


# ── Pre-computed ID pools ─────────────────────────────────────────────────────
# Each module imports these to build FK references without DB lookups.

NUM_AUTH_USERS = 250
NUM_ADMIN_USERS = 150

AUTH_USER_IDS = [stable_id("auth-user", i) for i in range(NUM_AUTH_USERS)]
ADMIN_USER_IDS = [stable_id("admin-user", i) for i in range(NUM_ADMIN_USERS)]

# ── Engineering org data ──────────────────────────────────────────────────────

DEPARTMENTS = [
    "Platform Engineering", "Backend", "Frontend", "Mobile", "DevOps",
    "SRE", "QA", "Data Engineering", "Data Science", "ML Engineering",
    "Security", "Infrastructure", "Product", "Design", "Developer Experience",
]

TEAMS = [
    "Core API", "Auth Team", "Search Team", "Billing", "Notifications",
    "CI/CD", "Observability", "Data Pipeline", "ML Platform", "Mobile iOS",
    "Mobile Android", "Web Platform", "Design Systems", "Security Ops",
    "Infra Automation", "Release Engineering", "Developer Tools",
    "Performance", "Reliability", "Growth",
]

# ── Time helpers ──────────────────────────────────────────────────────────────

def now() -> datetime:
    return datetime.now(timezone.utc)


def days_ago(n: int) -> datetime:
    return now() - timedelta(days=n)


def hours_ago(n: int) -> datetime:
    return now() - timedelta(hours=n)


def minutes_ago(n: int) -> datetime:
    return now() - timedelta(minutes=n)


# ── Password hashing ─────────────────────────────────────────────────────────

def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt(rounds=4)).decode()


# ── DB connection ─────────────────────────────────────────────────────────────

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "dbname": os.getenv("DB_NAME", "otterworks"),
    "user": os.getenv("DB_USER", "otterworks"),
    "password": os.getenv("DB_PASSWORD", "otterworks_dev"),
}


def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def set_search_path(cur, ns: str) -> None:
    schema = f"otterworks_{ns}"
    cur.execute(f'SET search_path TO "{schema}"')


# ── Insert helper ─────────────────────────────────────────────────────────────

def bulk_insert(cur, table: str, columns: list[str], rows: list[tuple],
                on_conflict: str = "DO NOTHING", template: str | None = None) -> int:
    if not rows:
        return 0
    cols = ", ".join(columns)
    sql = f"INSERT INTO {table} ({cols}) VALUES %s ON CONFLICT {on_conflict}"
    execute_values(cur, sql, rows, template=template)
    return len(rows)
