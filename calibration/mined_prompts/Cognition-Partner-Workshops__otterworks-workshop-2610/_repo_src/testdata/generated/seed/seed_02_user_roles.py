"""Seed module 02: User role assignments."""
import random

from helpers import AUTH_USER_IDS, NUM_AUTH_USERS, bulk_insert

ROLES = ("USER", "EDITOR", "ADMIN", "OWNER")

# Base rates are set above the raw target percentages to compensate for the
# seniority dampening (average factor ≈ 0.8), yielding ≈ 93 EDITOR, 30 ADMIN,
# 4 OWNER assignments — within the ~100 / ~25 / ~5 targets.
_EXTRA_ROLE_BASE_RATES = {
    "EDITOR": 0.50,
    "ADMIN":  0.125,
    "OWNER":  0.04,
}


def seed(cur, ns: str) -> int:
    """Insert user role assignments. Returns row count."""
    rng = random.Random(42)
    rows: list[tuple[str, str]] = []

    for idx, user_id in enumerate(AUTH_USER_IDS):
        # Seniority factor: earlier hires (low index) are more senior and
        # more likely to accumulate extra roles.  Ranges from 1.0 (idx 0)
        # down to 0.6 (idx 249).
        seniority = 1.0 - 0.4 * (idx / (NUM_AUTH_USERS - 1))

        # Every user gets the base USER role.
        rows.append((user_id, "USER"))

        for role, base_rate in _EXTRA_ROLE_BASE_RATES.items():
            if rng.random() < base_rate * seniority:
                rows.append((user_id, role))

    count = bulk_insert(
        cur,
        "user_roles",
        ["user_id", "role"],
        rows,
        on_conflict="(user_id, role) DO NOTHING",
    )
    return count
