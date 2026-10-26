"""Seed module 05: Admin dashboard users (150 users)."""
import random
from helpers import (
    stable_id, ADMIN_USER_IDS, NUM_ADMIN_USERS,
    DEPARTMENTS, TEAMS, now, days_ago, hours_ago, bulk_insert, Json
)

FIRST_NAMES = [
    "Alden", "Brynn", "Cassidy", "Dario", "Elara", "Finnley", "Greta", "Hollis",
    "Ingrid", "Jasper", "Keira", "Lennox", "Maren", "Nico", "Orla", "Palmer",
    "Quinn", "Ronan", "Sable", "Tate", "Ursa", "Vesper", "Wren", "Xander",
    "Yara", "Zephyr", "Ansel", "Blair", "Colby", "Darcy", "Ember", "Flint",
    "Greer", "Haven", "Idris", "Jules", "Kade", "Linden", "Marlowe", "Nyla",
    "Onyx", "Pax", "Reeve", "Sage", "Thane", "Unity", "Voss", "Waverly",
    "Xyla", "York", "Zara", "Alder", "Briar", "Cove", "Drake", "Ellis",
    "Frost", "Gage", "Heath", "Isolde", "Juno", "Knox", "Lyric", "Milan",
    "Nieve", "Odin", "Pierce", "Quill", "Remy", "Storm", "Tova", "Upton",
    "Vale", "Winter", "Ximena", "Yael", "Zev", "Arden", "Bowie", "Cleo",
    "Dashiell", "Elowen", "Fable", "Gemma", "Huxley", "Ione", "Jett",
    "Kellan", "Lux", "Mercer", "Nash", "Octavia", "Phoebe", "Rune",
    "Soren", "Tiernan", "Ulric", "Vera", "Wilder", "Xeno", "Yves", "Zinnia",
    "Amias", "Beckett", "Callum", "Delphine", "Eamon", "Freya", "Gareth",
    "Hadley", "Ivar", "Jolie", "Kian", "Leona", "Magnus", "Niamh", "Oleander",
    "Perrin", "Questa", "Rhys", "Selene", "Torrin", "Una", "Viggo", "Winona",
    "Xavi", "Ysabel", "Zane", "Aurelia", "Bastian", "Cordelia", "Dante",
    "Eulalia", "Felix", "Gwendolyn", "Harris", "Ivy", "Joaquin", "Katya",
    "Lachlan", "Mira", "Niall", "Ophelia", "Preston",
]

LAST_NAMES = [
    "Ashford", "Bancroft", "Caldwell", "Davenport", "Everett", "Fairchild",
    "Gallagher", "Hartwell", "Iverson", "Jennings", "Kingsley", "Langford",
    "Merriweather", "Northcott", "Oakley", "Pemberton", "Quinlan", "Radcliffe",
    "Stirling", "Thornton", "Underwood", "Vaughn", "Whitmore", "Yardley",
    "Aldridge", "Blackwood", "Carrington", "Drummond", "Elsworth", "Fielding",
    "Gresham", "Holloway", "Irvine", "Jaspar", "Kenworth", "Lattimer",
    "Montague", "Norwood", "Osborne", "Prescott", "Ramsey", "Sheldon",
    "Trevelyan", "Upton", "Vickers", "Wainwright", "Yardborough", "Zelinsky",
    "Abernathy", "Beckenridge",
]

SUSPENDED_REASONS = [
    "Repeated policy violations",
    "Unauthorized data export attempt",
    "Account compromised — pending investigation",
    "Extended leave of absence",
    "Failed security audit review",
    "Pending HR disciplinary review",
    "Suspicious login activity detected",
    "Non-compliance with MFA requirement",
    "Excessive failed login attempts",
    "Reported by another admin for misconduct",
]


def seed(cur, ns: str) -> int:
    """Insert 150 admin users. Returns row count."""
    rng = random.Random(42)

    # Role distribution: ~50% viewer, ~30% editor, ~15% admin, ~5% super_admin
    roles = (
        ["viewer"] * 75
        + ["editor"] * 45
        + ["admin"] * 22
        + ["super_admin"] * 8
    )

    # Status distribution: ~85% active, ~10% suspended, ~5% deactivated
    statuses = (
        ["active"] * 128
        + ["suspended"] * 15
        + ["deactivated"] * 7
    )

    rng.shuffle(roles)
    rng.shuffle(statuses)

    # Build unique first+last name pairs
    first_pool = list(FIRST_NAMES)
    last_pool = list(LAST_NAMES)
    rng.shuffle(first_pool)

    names = []
    for i in range(NUM_ADMIN_USERS):
        first = first_pool[i % len(first_pool)]
        last = last_pool[i % len(last_pool)]
        names.append((first, last))

    current = now()
    columns = [
        "id", "email", "display_name", "role", "status",
        "avatar_url", "metadata", "last_login_at",
        "suspended_at", "suspended_reason", "created_at", "updated_at",
    ]

    rows = []
    for i in range(NUM_ADMIN_USERS):
        user_id = ADMIN_USER_IDS[i]
        first, last = names[i]
        email = f"{first.lower()}.{last.lower()}@otterworks.io"
        display_name = f"{first} {last}"
        role = roles[i]
        status = statuses[i]
        avatar_url = f"https://avatars.otterworks.io/admin/{user_id}.png"

        department = rng.choice(DEPARTMENTS)
        team = rng.choice(TEAMS)
        documents_count = rng.randint(0, 500)
        employee_id = f"EMP-{rng.randint(0, 9999):04d}"
        metadata = Json({
            "department": department,
            "team": team,
            "documents_count": documents_count,
            "employee_id": employee_id,
        })

        created_at = days_ago(rng.randint(1, 365))
        updated_at = created_at

        if status == "active":
            last_login_at = hours_ago(rng.randint(1, 336))  # within ~14 days
            suspended_at = None
            suspended_reason = None
        elif status == "suspended":
            last_login_at = days_ago(rng.randint(30, 180))
            suspended_at = days_ago(rng.randint(1, 29))
            suspended_reason = rng.choice(SUSPENDED_REASONS)
            updated_at = suspended_at
        else:  # deactivated
            last_login_at = None
            suspended_at = None
            suspended_reason = None

        rows.append((
            user_id, email, display_name, role, status,
            avatar_url, metadata, last_login_at,
            suspended_at, suspended_reason, created_at, updated_at,
        ))

    count = bulk_insert(
        cur, "admin_users", columns, rows,
        on_conflict="(email) DO NOTHING",
        template="(%s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s)",
    )
    return count
