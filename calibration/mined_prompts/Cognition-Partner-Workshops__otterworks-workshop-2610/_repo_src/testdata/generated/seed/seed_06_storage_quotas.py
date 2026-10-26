"""Seed module 06: Storage quotas for admin users."""

import random
from datetime import timedelta
from helpers import (
    stable_id, ADMIN_USER_IDS, NUM_ADMIN_USERS,
    now, days_ago, bulk_insert
)

# Tier definitions: (name, quota in bytes, cumulative weight boundary)
_GB = 1024 ** 3
TIER_SPECS = {
    "free":       5 * _GB,
    "basic":      50 * _GB,
    "pro":        200 * _GB,
    "enterprise": 1024 * _GB,
}

# Weighted tier list matching ~20% free, ~30% basic, ~35% pro, ~15% enterprise
TIER_WEIGHTS = [
    ("free", 20),
    ("basic", 30),
    ("pro", 35),
    ("enterprise", 15),
]
_TIER_NAMES = [t for t, _ in TIER_WEIGHTS]
_TIER_CUM_WEIGHTS = [w for _, w in TIER_WEIGHTS]


def _pick_used_bytes(rng: random.Random, quota_bytes: int) -> int:
    """Return a realistic used_bytes value that never exceeds quota_bytes.

    Distribution shaped so most users land in the 20-60% range, with tails
    for near-empty (<5%) and near-full (80-95%) usage.
    """
    r = rng.random()
    if r < 0.10:
        # ~10% near-empty: 0-5%
        pct = rng.uniform(0.0, 0.05)
    elif r < 0.80:
        # ~70% mid-range: 20-60%
        pct = rng.uniform(0.20, 0.60)
    else:
        # ~20% near-full: 80-95%
        pct = rng.uniform(0.80, 0.95)
    return int(pct * quota_bytes)


def seed(cur, ns: str) -> int:
    """Insert storage quotas for all admin users. Returns row count."""
    rng = random.Random(42)
    current = now()

    columns = [
        "id", "user_id", "quota_bytes", "used_bytes",
        "tier", "created_at", "updated_at",
    ]

    rows = []
    for i in range(NUM_ADMIN_USERS):
        tier = rng.choices(_TIER_NAMES, weights=_TIER_CUM_WEIGHTS, k=1)[0]
        quota_bytes = TIER_SPECS[tier]
        used_bytes = _pick_used_bytes(rng, quota_bytes)

        created_at = days_ago(rng.randint(1, 365))
        # updated_at is between created_at and now
        delta_seconds = int((current - created_at).total_seconds())
        updated_at = created_at + timedelta(
            seconds=rng.randint(0, max(delta_seconds, 1))
        )

        rows.append((
            stable_id("storage-quota", i),
            ADMIN_USER_IDS[i],
            quota_bytes,
            used_bytes,
            tier,
            created_at,
            updated_at,
        ))

    count = bulk_insert(
        cur, "storage_quotas", columns, rows,
        on_conflict="(user_id) DO NOTHING",
    )
    return count
