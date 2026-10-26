"""Seed module 03: User settings/preferences."""
import random
from helpers import AUTH_USER_IDS, NUM_AUTH_USERS, bulk_insert


def seed(cur, ns: str) -> int:
    """Insert user settings for all auth users. Returns row count."""
    rng = random.Random(42)

    themes = ['system'] * 50 + ['dark'] * 30 + ['light'] * 20
    languages = ['en'] * 70 + ['es'] * 10 + ['ja'] * 8 + ['de'] * 5 + ['fr'] * 4 + ['zh'] * 3

    columns = [
        "user_id",
        "notification_email",
        "notification_in_app",
        "notification_desktop",
        "theme",
        "language",
    ]

    rows = []
    for uid in AUTH_USER_IDS:
        rows.append((
            uid,
            rng.random() < 0.85,
            rng.random() < 0.90,
            rng.random() < 0.25,
            rng.choices(themes)[0],
            rng.choices(languages)[0],
        ))

    count = bulk_insert(cur, "user_settings", columns, rows,
                        on_conflict="(user_id) DO NOTHING")
    return count
