"""Seed module 04: Refresh tokens for auth users."""
import random
from datetime import timedelta

from helpers import (
    stable_id, AUTH_USER_IDS, NUM_AUTH_USERS,
    now, days_ago, hours_ago, bulk_insert
)


def seed(cur, ns: str) -> int:
    """Insert refresh tokens. Returns row count."""
    rng = random.Random(42)
    current = now()

    # Distribute ~400 tokens across 250 users.
    # Some users get multiple tokens (different devices/sessions).
    token_counts: list[int] = []
    for _ in range(NUM_AUTH_USERS):
        r = rng.random()
        if r < 0.55:
            token_counts.append(1)
        elif r < 0.85:
            token_counts.append(2)
        else:
            token_counts.append(3)

    columns = [
        "id", "user_id", "token_id", "expires_at", "revoked", "created_at",
    ]
    rows: list[tuple] = []
    idx = 0

    for user_idx, count in enumerate(token_counts):
        user_id = AUTH_USER_IDS[user_idx]
        for _ in range(count):
            row_id = stable_id("refresh-token", idx)
            token_id = f"rt_{rng.getrandbits(128):032x}"

            created_days_ago = rng.uniform(0, 90)
            created_at = current - timedelta(days=created_days_ago)

            lifetime_days = rng.uniform(7, 30)
            expires_at = created_at + timedelta(days=lifetime_days)

            # ~60% active, ~25% expired, ~15% revoked
            roll = rng.random()
            if roll < 0.15:
                revoked = True
            elif roll < 0.40:
                revoked = False
                # Force expiry into the past
                if expires_at > current:
                    past_offset = rng.uniform(0.01, min(created_days_ago + lifetime_days, 30))
                    expires_at = current - timedelta(days=past_offset)
                    if expires_at <= created_at:
                        expires_at = created_at + timedelta(hours=rng.uniform(1, 24))
            else:
                revoked = False
                # Ensure token is still valid
                if expires_at <= current:
                    expires_at = current + timedelta(days=rng.uniform(1, 14))

            rows.append((
                row_id, user_id, token_id, expires_at, revoked, created_at,
            ))
            idx += 1

    return bulk_insert(
        cur, "refresh_tokens", columns, rows,
        on_conflict="(token_id) DO NOTHING",
    )
