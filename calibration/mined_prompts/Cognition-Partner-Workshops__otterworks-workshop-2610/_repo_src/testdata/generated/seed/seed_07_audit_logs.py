"""Seed module 07: Audit logs (600+ entries)."""
import random
from datetime import timedelta
from helpers import (
    stable_id, ADMIN_USER_IDS, NUM_ADMIN_USERS,
    now, days_ago, hours_ago, bulk_insert, Json
)

# ── Action definitions keyed by resource_type ─────────────────────────────────

RESOURCE_ACTIONS: dict[str, list[str]] = {
    "AdminUser": [
        "user.created", "user.updated", "user.suspended",
        "user.activated", "user.deactivated", "user.role_changed",
        "user.password_reset",
    ],
    "StorageQuota": ["quota.updated", "quota.upgraded"],
    "FeatureFlag": [
        "feature_flag.created", "feature_flag.updated",
        "feature_flag.toggled",
    ],
    "SystemConfig": ["config.updated", "config.created"],
    "Announcement": [
        "announcement.created", "announcement.updated",
        "announcement.published",
    ],
    "Incident": [
        "incident.created", "incident.updated", "incident.resolved",
    ],
    "ApiKey": ["api_key.created", "api_key.revoked"],
    "Export": ["export.requested", "export.completed"],
}

SYSTEM_ACTIONS: list[tuple[str, str]] = [
    ("system.backup_completed", "SystemConfig"),
    ("system.maintenance_started", "SystemConfig"),
    ("system.maintenance_completed", "SystemConfig"),
]

# ── Changes templates per action ──────────────────────────────────────────────

ROLES = ["viewer", "editor", "admin", "super_admin"]
STATUSES = ["active", "suspended", "deactivated"]
TIERS = ["free", "basic", "pro", "enterprise"]
SEVERITIES = ["info", "warning", "critical"]
INCIDENT_SEVERITIES = ["low", "medium", "high", "critical"]
INCIDENT_STATUSES = ["open", "investigating", "resolved", "closed"]


def _changes_for_action(rng: random.Random, action: str) -> dict:
    """Return a realistic JSONB diff for the given action."""
    if action == "user.role_changed":
        old = rng.choice(ROLES)
        new = rng.choice([r for r in ROLES if r != old])
        return {"field": "role", "old": old, "new": new}
    if action == "user.suspended":
        return {"field": "status", "old": "active", "new": "suspended"}
    if action == "user.activated":
        return {"field": "status", "old": "suspended", "new": "active"}
    if action == "user.deactivated":
        return {"field": "status", "old": "active", "new": "deactivated"}
    if action == "user.password_reset":
        return {"field": "password_hash", "note": "password was reset"}
    if action == "user.updated":
        field = rng.choice(["display_name", "email", "avatar_url"])
        return {"field": field, "old": f"old_{field}_value", "new": f"new_{field}_value"}
    if action == "user.created":
        return {"field": "status", "new": "active"}
    if action == "quota.updated":
        old_bytes = rng.choice([1_073_741_824, 5_368_709_120, 10_737_418_240])
        new_bytes = old_bytes * 2
        return {"field": "quota_bytes", "old": old_bytes, "new": new_bytes}
    if action == "quota.upgraded":
        old_tier = rng.choice(TIERS[:3])
        idx = TIERS.index(old_tier)
        new_tier = TIERS[min(idx + 1, len(TIERS) - 1)]
        return {"field": "tier", "old": old_tier, "new": new_tier}
    if action == "feature_flag.toggled":
        enabled = rng.choice([True, False])
        return {"field": "enabled", "old": not enabled, "new": enabled}
    if action == "feature_flag.updated":
        return {"field": "rollout_percentage", "old": rng.randint(0, 50), "new": rng.randint(51, 100)}
    if action == "feature_flag.created":
        return {"field": "enabled", "new": False, "rollout_percentage": 0}
    if action == "config.updated":
        key = rng.choice(["max_upload_size_mb", "session_timeout_minutes", "rate_limit_rpm"])
        old_val = rng.randint(10, 100)
        return {"field": key, "old": old_val, "new": old_val + rng.randint(5, 50)}
    if action == "config.created":
        key = rng.choice(["maintenance_mode", "signup_enabled", "api_rate_limit"])
        return {"field": key, "new": rng.choice([True, False, 1000, 5000])}
    if action == "announcement.created":
        return {"field": "status", "new": "draft", "severity": rng.choice(SEVERITIES)}
    if action == "announcement.updated":
        return {"field": "body", "note": "content revised"}
    if action == "announcement.published":
        return {"field": "status", "old": "draft", "new": "active"}
    if action == "incident.created":
        sev = rng.choice(INCIDENT_SEVERITIES)
        return {"field": "status", "new": "open", "severity": sev}
    if action == "incident.updated":
        return {"field": "status", "old": "open", "new": "investigating"}
    if action == "incident.resolved":
        return {"field": "status", "old": "investigating", "new": "resolved"}
    if action == "api_key.created":
        prefix = "".join(rng.choices("abcdef0123456789", k=8))
        return {"field": "key_prefix", "new": f"ow_{prefix}"}
    if action == "api_key.revoked":
        return {"field": "status", "old": "active", "new": "revoked", "reason": "security rotation"}
    if action == "export.requested":
        fmt = rng.choice(["csv", "json", "parquet"])
        return {"field": "format", "new": fmt, "scope": rng.choice(["full", "incremental"])}
    if action == "export.completed":
        return {"field": "status", "old": "pending", "new": "completed", "rows_exported": rng.randint(100, 50000)}
    if action.startswith("system."):
        return {"note": action.replace("system.", "").replace("_", " ")}
    return {}


# ── User agents ───────────────────────────────────────────────────────────────

BROWSER_UAS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36 Edg/125.0.0.0",
]

SYSTEM_UA = "OtterWorks-API/1.0"


def seed(cur, ns: str) -> int:
    """Insert 600+ audit log entries. Returns row count."""
    rng = random.Random(42)
    total = 650

    # Load actual admin_users created_at to guarantee temporal consistency
    cur.execute("SELECT id, created_at FROM admin_users")
    admin_created = {str(row[0]): row[1] for row in cur.fetchall()}

    # Flatten resource_type -> actions for weighted selection
    resource_types = list(RESOURCE_ACTIONS.keys())
    weights = []
    for rt in resource_types:
        if rt in ("AdminUser", "FeatureFlag", "SystemConfig"):
            weights.append(3)
        elif rt in ("Incident", "Announcement"):
            weights.append(2)
        else:
            weights.append(1)

    columns = [
        "id", "actor_id", "actor_email", "action", "resource_type",
        "resource_id", "changes_made", "ip_address", "user_agent",
        "created_at", "updated_at",
    ]

    _now = now()
    rows: list[tuple] = []
    for i in range(total):
        log_id = stable_id("audit-log", i)

        # ~8% of entries are system actions (no actor)
        is_system = rng.random() < 0.08

        if is_system:
            actor_id = None
            actor_email = None
            action, resource_type = rng.choice(SYSTEM_ACTIONS)
            user_agent = SYSTEM_UA
            ip_addr = "127.0.0.1"
        else:
            admin_index = rng.randint(0, NUM_ADMIN_USERS - 1)
            actor_id = ADMIN_USER_IDS[admin_index]
            actor_email = f"admin{admin_index}@otterworks.io"
            resource_type = rng.choices(resource_types, weights=weights, k=1)[0]
            action = rng.choice(RESOURCE_ACTIONS[resource_type])
            user_agent = rng.choice(BROWSER_UAS)

            if rng.random() < 0.80:
                if rng.random() < 0.5:
                    ip_addr = f"10.{rng.randint(0,255)}.{rng.randint(0,255)}.{rng.randint(1,254)}"
                else:
                    ip_addr = f"192.168.{rng.randint(0,255)}.{rng.randint(1,254)}"
            else:
                ip_addr = f"{rng.randint(50,220)}.{rng.randint(0,255)}.{rng.randint(0,255)}.{rng.randint(1,254)}"

        resource_id = stable_id(resource_type.lower(), rng.randint(0, 499))
        changes = _changes_for_action(rng, action)

        # Timestamp within last 30 days, but always after the actor's created_at
        offset_minutes = rng.randint(0, 30 * 24 * 60)
        created_at = _now - timedelta(minutes=offset_minutes)
        if actor_id and actor_id in admin_created:
            earliest = admin_created[actor_id] + timedelta(minutes=1)
            if created_at < earliest:
                created_at = earliest
        updated_at = created_at

        rows.append((
            log_id,
            actor_id,
            actor_email,
            action,
            resource_type,
            resource_id,
            Json(changes),
            ip_addr,
            user_agent,
            created_at,
            updated_at,
        ))

    count = bulk_insert(
        cur, "audit_logs", columns, rows,
        template="(%s, %s::uuid, %s, %s, %s, %s::uuid, %s::jsonb, %s, %s, %s, %s)",
    )
    return count
