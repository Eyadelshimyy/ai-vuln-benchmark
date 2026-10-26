# /// script
# requires-python = ">=3.11"
# dependencies = ["psycopg2-binary", "bcrypt"]
# ///
"""
OtterWorks admin-service seed script.

Populates the admin-service PostgreSQL database with 10 realistic users,
storage quotas, audit log entries, feature flags, and announcements so the
admin dashboard shows real metrics instead of in-memory mock data.

Usage:
    uv run scripts/seed.py

Environment overrides (all optional — defaults match docker-compose):
    DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
"""

import os
import sys
import uuid
from datetime import datetime, timedelta, timezone

import bcrypt
import psycopg2
from psycopg2.extras import execute_values, Json

# ── Connection ────────────────────────────────────────────────────────────────

DB_CONFIG = {
    "host":     os.getenv("DB_HOST",     "localhost"),
    "port":     int(os.getenv("DB_PORT", "5432")),
    "dbname":   os.getenv("DB_NAME",     "otterworks"),
    "user":     os.getenv("DB_USER",     "otterworks"),
    "password": os.getenv("DB_PASSWORD", "otterworks_dev"),
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def now() -> datetime:
    return datetime.now(timezone.utc)

def days_ago(n: int) -> datetime:
    return now() - timedelta(days=n)

def hours_ago(n: int) -> datetime:
    return now() - timedelta(hours=n)

def uid() -> str:
    return str(uuid.uuid4())

def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt(rounds=10)).decode()

def log(msg: str) -> None:
    print(f"  {msg}")

# ── Seed data ─────────────────────────────────────────────────────────────────

# Fixed UUIDs so re-runs are idempotent
ADMIN_ID = "a0000000-0000-0000-0000-000000000001"  # pre-existing admin from auth migration

USERS = [
    {
        "id":           "5eed0001-0000-4000-a000-000000000001",
        "email":        "alice.johnson@otterworks.io",
        "display_name": "Alice Johnson",
        "role":         "super_admin",
        "status":       "active",
        "department":   "Engineering",
        "documents_count": 145,
        "last_login_at":   days_ago(1),
        "created_at":      days_ago(180),
        "tier":         "enterprise",
        "quota_gb":     1024,
        "used_gb":      38.4,
    },
    {
        "id":           "5eed0002-0000-4000-a000-000000000002",
        "email":        "bob.martinez@otterworks.io",
        "display_name": "Bob Martinez",
        "role":         "editor",
        "status":       "active",
        "department":   "Marketing",
        "documents_count": 87,
        "last_login_at":   hours_ago(3),
        "created_at":      days_ago(150),
        "tier":         "pro",
        "quota_gb":     200,
        "used_gb":      62.1,
    },
    {
        "id":           "5eed0003-0000-4000-a000-000000000003",
        "email":        "carol.chen@otterworks.io",
        "display_name": "Carol Chen",
        "role":         "editor",
        "status":       "active",
        "department":   "Design",
        "documents_count": 234,
        "last_login_at":   days_ago(2),
        "created_at":      days_ago(120),
        "tier":         "pro",
        "quota_gb":     200,
        "used_gb":      148.7,
    },
    {
        "id":           "5eed0004-0000-4000-a000-000000000004",
        "email":        "david.kim@otterworks.io",
        "display_name": "David Kim",
        "role":         "viewer",
        "status":       "suspended",
        "department":   "Sales",
        "documents_count": 12,
        "last_login_at":   days_ago(45),
        "created_at":      days_ago(90),
        "tier":         "basic",
        "quota_gb":     50,
        "used_gb":      3.2,
    },
    {
        "id":           "5eed0005-0000-4000-a000-000000000005",
        "email":        "emily.davis@otterworks.io",
        "display_name": "Emily Davis",
        "role":         "editor",
        "status":       "active",
        "department":   "Product",
        "documents_count": 198,
        "last_login_at":   hours_ago(1),
        "created_at":      days_ago(100),
        "tier":         "pro",
        "quota_gb":     200,
        "used_gb":      91.5,
    },
    {
        "id":           "5eed0006-0000-4000-a000-000000000006",
        "email":        "frank.wilson@otterworks.io",
        "display_name": "Frank Wilson",
        "role":         "admin",
        "status":       "active",
        "department":   "Operations",
        "documents_count": 67,
        "last_login_at":   hours_ago(5),
        "created_at":      days_ago(80),
        "tier":         "enterprise",
        "quota_gb":     1024,
        "used_gb":      210.0,
    },
    {
        "id":           "5eed0007-0000-4000-a000-000000000007",
        "email":        "grace.lee@otterworks.io",
        "display_name": "Grace Lee",
        "role":         "viewer",
        "status":       "active",
        "department":   "Finance",
        "documents_count": 34,
        "last_login_at":   days_ago(3),
        "created_at":      days_ago(60),
        "tier":         "basic",
        "quota_gb":     50,
        "used_gb":      18.9,
    },
    {
        "id":           "5eed0008-0000-4000-a000-000000000008",
        "email":        "henry.thompson@otterworks.io",
        "display_name": "Henry Thompson",
        "role":         "editor",
        "status":       "active",
        "department":   "Engineering",
        "documents_count": 312,
        "last_login_at":   hours_ago(2),
        "created_at":      days_ago(45),
        "tier":         "pro",
        "quota_gb":     200,
        "used_gb":      177.3,
    },
    {
        "id":           "5eed0009-0000-4000-a000-000000000009",
        "email":        "irene.garcia@otterworks.io",
        "display_name": "Irene Garcia",
        "role":         "viewer",
        "status":       "active",
        "department":   "Legal",
        "documents_count": 23,
        "last_login_at":   days_ago(7),
        "created_at":      days_ago(30),
        "tier":         "free",
        "quota_gb":     5,
        "used_gb":      2.1,
    },
    {
        "id":           "5eed0010-0000-4000-a000-000000000010",
        "email":        "james.park@otterworks.io",
        "display_name": "James Park",
        "role":         "editor",
        "status":       "active",
        "department":   "Data Science",
        "documents_count": 156,
        "last_login_at":   hours_ago(8),
        "created_at":      days_ago(15),
        "tier":         "pro",
        "quota_gb":     200,
        "used_gb":      44.8,
    },
]

FEATURE_FLAGS_EXTRA = [
    {
        "id":                  uid(),
        "name":                "beta_export_pipeline",
        "description":         "Enable the new async export pipeline for large reports",
        "enabled":             False,
        "rollout_percentage":  0,
    },
    {
        "id":                  uid(),
        "name":                "ai_document_summary",
        "description":         "AI-powered document summary on upload",
        "enabled":             True,
        "rollout_percentage":  25,
    },
    {
        "id":                  uid(),
        "name":                "advanced_search_filters",
        "description":         "Extended search filters powered by MeiliSearch facets",
        "enabled":             True,
        "rollout_percentage":  80,
    },
]

ANNOUNCEMENTS = [
    {
        "id":        "5eeda0a0-0000-4000-a000-000000000001",
        "title":     "Scheduled maintenance this Saturday",
        "body":      "The platform will be unavailable from 02:00–04:00 UTC on Saturday for database upgrades. "
                     "Please save your work beforehand.",
        "severity":  "maintenance",
        "status":    "published",
        "starts_at": days_ago(2),
        "ends_at":   days_ago(0),
    },
    {
        "id":        "5eeda0a0-0000-4000-a000-000000000002",
        "title":     "New AI document summary feature now in beta",
        "body":      "We've launched AI-powered document summaries for 25% of users. "
                     "Check the Feature Flags page to manage rollout.",
        "severity":  "info",
        "status":    "published",
        "starts_at": days_ago(5),
        "ends_at":   None,
    },
    {
        "id":        "5eeda0a0-0000-4000-a000-000000000003",
        "title":     "Q3 storage quota review",
        "body":      "Several enterprise accounts are approaching quota limits. "
                     "Admins should review the Storage Quotas page and upgrade affected users.",
        "severity":  "warning",
        "status":    "draft",
        "starts_at": None,
        "ends_at":   None,
    },
]

AUDIT_ACTIONS = [
    ("user.created",           "AdminUser"),
    ("user.updated",           "AdminUser"),
    ("user.suspended",         "AdminUser"),
    ("user.activated",         "AdminUser"),
    ("quota.updated",          "StorageQuota"),
    ("feature_flag.updated",   "FeatureFlag"),
    ("config.updated",         "SystemConfig"),
    ("announcement.created",   "Announcement"),
    ("announcement.updated",   "Announcement"),
]

# ── Seeding functions ─────────────────────────────────────────────────────────

def seed_users(cur) -> None:
    log("Inserting 10 admin_users ...")
    rows = []
    for u in USERS:
        metadata = {
            "department":      u["department"],
            "documents_count": u["documents_count"],
        }
        rows.append((
            u["id"], u["email"], u["display_name"], u["role"], u["status"],
            None,  # avatar_url
            Json(metadata),
            u["last_login_at"],
            u["created_at"], u["created_at"],
        ))

    execute_values(cur, """
        INSERT INTO admin_users
            (id, email, display_name, role, status, avatar_url, metadata,
             last_login_at, created_at, updated_at)
        VALUES %s
        ON CONFLICT (email) DO NOTHING
    """, rows, template="(%s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s)")

    # Mark the suspended user
    cur.execute("""
        UPDATE admin_users
        SET suspended_at = %s, suspended_reason = 'Policy violation — account under review'
        WHERE email = 'david.kim@otterworks.io' AND suspended_at IS NULL
    """, (days_ago(44),))
    log("  Done.")


def seed_quotas(cur) -> None:
    log("Inserting storage_quotas ...")
    gb = 1024 ** 3
    tier_map = {"free": 5, "basic": 50, "pro": 200, "enterprise": 1024}

    rows = []
    for u in USERS:
        quota_bytes = int(tier_map[u["tier"]] * gb)
        used_bytes  = int(u["used_gb"] * gb)
        rows.append((
            uid(), u["id"], quota_bytes, used_bytes, u["tier"],
            u["created_at"], u["created_at"],
        ))

    execute_values(cur, """
        INSERT INTO storage_quotas
            (id, user_id, quota_bytes, used_bytes, tier, created_at, updated_at)
        VALUES %s
        ON CONFLICT (user_id) DO NOTHING
    """, rows, template="(%s, %s::uuid, %s, %s, %s, %s, %s)")
    log("  Done.")


def seed_feature_flags(cur) -> None:
    log("Inserting extra feature_flags ...")
    rows = []
    for f in FEATURE_FLAGS_EXTRA:
        rows.append((
            f["id"], f["name"], f["description"], f["enabled"],
            f["rollout_percentage"], now(), now(),
        ))

    execute_values(cur, """
        INSERT INTO feature_flags
            (id, name, description, enabled, rollout_percentage, created_at, updated_at)
        VALUES %s
        ON CONFLICT (name) DO NOTHING
    """, rows, template="(%s, %s, %s, %s, %s, %s, %s)")
    log("  Done.")


def seed_announcements(cur) -> None:
    log("Inserting announcements ...")
    rows = []
    for a in ANNOUNCEMENTS:
        rows.append((
            a["id"], a["title"], a["body"], a["severity"], a["status"],
            a["starts_at"], a["ends_at"], now(), now(),
        ))

    execute_values(cur, """
        INSERT INTO announcements
            (id, title, body, severity, status, starts_at, ends_at, created_at, updated_at)
        VALUES %s
        ON CONFLICT DO NOTHING
    """, rows, template="(%s, %s, %s, %s, %s, %s, %s, %s, %s)")
    log("  Done.")


def seed_audit_logs(cur) -> None:
    log("Inserting ~80 audit_log entries spread over the past 30 days ...")

    # Gather valid user IDs that were actually inserted
    cur.execute("SELECT id, email FROM admin_users WHERE email LIKE '%@otterworks.io'")
    db_users = cur.fetchall()
    if not db_users:
        log("  No seeded users found — skipping audit logs.")
        return

    import random
    rng = random.Random(42)  # deterministic

    rows = []
    for i in range(80):
        user = rng.choice(db_users)
        actor_id, actor_email = user[0], user[1]
        action, resource_type = rng.choice(AUDIT_ACTIONS)

        # Pick a random resource_id — use another user's ID for user actions
        if resource_type == "AdminUser":
            resource_id = rng.choice(db_users)[0]
        else:
            resource_id = uid()

        age_hours = rng.randint(0, 30 * 24)
        created = hours_ago(age_hours)

        rows.append((
            uid(), actor_id, actor_email, action, resource_type, resource_id,
            '{}',  # changes_made
            f"192.168.{rng.randint(1,5)}.{rng.randint(1,254)}",
            "Mozilla/5.0 (seed)",
            created, created,
        ))

    cur.execute("DELETE FROM audit_logs WHERE user_agent = 'Mozilla/5.0 (seed)'")
    execute_values(cur, """
        INSERT INTO audit_logs
            (id, actor_id, actor_email, action, resource_type, resource_id,
             changes_made, ip_address, user_agent, created_at, updated_at)
        VALUES %s
    """, rows, template=(
        "(%s, %s::uuid, %s, %s, %s, %s::uuid, %s::jsonb, %s, %s, %s, %s)"
    ))
    log("  Done.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\nOtterWorks admin-service seed script")
    print(f"  Target: {DB_CONFIG['user']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}\n")

    try:
        conn = psycopg2.connect(**DB_CONFIG)
    except psycopg2.OperationalError as e:
        print(f"ERROR: Could not connect to database — {e}", file=sys.stderr)
        print("Make sure the stack is running: make infra-up && make up", file=sys.stderr)
        sys.exit(1)

    conn.autocommit = False
    cur = conn.cursor()

    try:
        seed_users(cur)
        seed_quotas(cur)
        seed_feature_flags(cur)
        seed_announcements(cur)
        seed_audit_logs(cur)
        conn.commit()
        print("\nSeed complete. Refresh the admin dashboard to see real data.")
    except Exception as e:
        conn.rollback()
        print(f"\nERROR during seed — rolled back: {e}", file=sys.stderr)
        raise
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    main()
