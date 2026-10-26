"""Seed module 01: Auth service users (250 engineers)."""

import random
from datetime import timedelta

from helpers import (
    AUTH_USER_IDS,
    NUM_AUTH_USERS,
    DEPARTMENTS,
    now,
    days_ago,
    hash_password,
    bulk_insert,
)

# ── Name pools (diverse backgrounds) ─────────────────────────────────────────

FIRST_NAMES = [
    "Aarav", "Abigail", "Adaeze", "Aditya", "Akiko", "Alejandro", "Amara",
    "Amina", "Anastasia", "Andrei", "Ananya", "Antonio", "Arjun", "Arun",
    "Ayumi", "Benjamin", "Bianca", "Boris", "Camila", "Carlos", "Carmen",
    "Chandra", "Charlotte", "Chen", "Chidinma", "Chloe", "Daisuke", "Daniel",
    "Daniela", "David", "Deepika", "Diego", "Dmitri", "Elena", "Elif",
    "Emeka", "Emma", "Enrique", "Erik", "Esperanza", "Ethan", "Farah",
    "Fatima", "Felix", "Fernanda", "Gabriel", "Gabriela", "Gita", "Grace",
    "Haruki", "Hassan", "Helena", "Henrik", "Hugo", "Ibrahim", "Ingrid",
    "Isabella", "Javier", "Jia", "Jin", "Jonas", "Jorge", "Josephine",
    "Juan", "Julia", "Jun", "Kaia", "Kai", "Kamila", "Kaori", "Karthik",
    "Kavya", "Kenji", "Khadija", "Kofi", "Kwame", "Layla", "Leandro",
    "Lena", "Leo", "Li", "Liam", "Linh", "Lucia", "Luis", "Luna",
    "Magdalena", "Mai", "Malik", "Manish", "Mara", "Marco", "Maria",
    "Mariana", "Marina", "Mateo", "Maya", "Mei", "Michael", "Miguel",
    "Mika", "Min", "Miriam", "Mohammed", "Naomi", "Nadia", "Naveen",
    "Neha", "Nia", "Nikolai", "Nina", "Noah", "Nora", "Obinna", "Olga",
    "Oliver", "Omar", "Paloma", "Paolo", "Petra", "Priya", "Qiang",
    "Rafael", "Rahul", "Rania", "Raquel", "Rashid", "Ren", "Ricardo",
    "Rosa", "Rui", "Sakura", "Samuel", "Sandra", "Santiago", "Sara",
    "Sasha", "Satoshi", "Sebastian", "Shreya", "Sofia", "Soren", "Suki",
    "Sunita", "Takeshi", "Tanya", "Tariq", "Tatiana", "Thiago", "Tomoko",
    "Umar", "Valentina", "Viktor", "Wei", "Xiomara", "Yara", "Yuki",
    "Yusuf", "Zara", "Zhi",
]

LAST_NAMES = [
    "Abadi", "Achebe", "Adeyemi", "Aguilar", "Ahmed", "Akiyama", "Alvarez",
    "Andersen", "Antonov", "Arora", "Bautista", "Bergstrom", "Bhatt",
    "Bianchi", "Boateng", "Brennan", "Castillo", "Chakraborty", "Chan",
    "Chang", "Chen", "Cho", "Costa", "Cruz", "Dahl", "Das", "Delgado",
    "Desai", "Diallo", "Dimitrov", "Dubois", "Duong", "Eriksson", "Espinoza",
    "Fernandez", "Fischer", "Flores", "Fujimoto", "Garcia", "Gomes",
    "Gonzalez", "Gupta", "Gutierrez", "Hansen", "Hara", "Hayashi",
    "Hernandez", "Holm", "Huang", "Hussein", "Inoue", "Islam", "Ito",
    "Ivanov", "Jain", "Jensen", "Johansson", "Johnson", "Kang", "Kapoor",
    "Kawamura", "Khan", "Kim", "Kowalski", "Kumar", "Larsson", "Lee",
    "Li", "Lim", "Liu", "Lopez", "Lundgren", "Machado", "Malik", "Martinez",
    "Matsumoto", "Mendes", "Meyer", "Mishra", "Morales", "Moreau", "Mori",
    "Muller", "Nakamura", "Narang", "Navarro", "Nguyen", "Nielsen", "Nwosu",
    "Ochoa", "Okamoto", "Okafor", "Oliveira", "Ortega", "Ortiz", "Ota",
    "Owusu", "Ozturk", "Patel", "Park", "Perez", "Petrov", "Pham",
    "Popov", "Prasad", "Quispe", "Rahman", "Ramirez", "Rao", "Reddy",
    "Reyes", "Rivera", "Rodriguez", "Rossi", "Roy", "Ruiz", "Saarinen",
    "Saito", "Salazar", "Santos", "Sato", "Schmidt", "Shah", "Sharma",
    "Silva", "Singh", "Smirnov", "Sokolov", "Soto", "Suzuki", "Takahashi",
    "Tanaka", "Tran", "Torres", "Ueda", "Vargas", "Vasquez", "Volkov",
    "Wang", "Weber", "Williams", "Wong", "Wu", "Yamamoto", "Yang", "Yilmaz",
    "Yoshida", "Zhang", "Zhou",
]

# ── Department weight distribution ────────────────────────────────────────────
# More engineers than security/design/product — mirrors a real org.

_DEPT_WEIGHTS = {
    "Platform Engineering": 30,
    "Backend": 35,
    "Frontend": 25,
    "Mobile": 15,
    "DevOps": 18,
    "SRE": 14,
    "QA": 16,
    "Data Engineering": 18,
    "Data Science": 12,
    "ML Engineering": 12,
    "Security": 8,
    "Infrastructure": 14,
    "Product": 10,
    "Design": 8,
    "Developer Experience": 10,
}

_DEPT_POPULATION = []
for dept in DEPARTMENTS:
    _DEPT_POPULATION.extend([dept] * _DEPT_WEIGHTS.get(dept, 10))


def seed(cur, ns: str) -> int:
    """Insert 250 auth users. Returns row count."""
    rng = random.Random(42)

    pw_hash = hash_password("seed-password")
    ref_now = now()

    columns = [
        "id", "email", "password_hash", "display_name", "avatar_url",
        "email_verified", "mfa_enabled", "mfa_secret",
        "created_at", "updated_at", "last_login_at",
    ]

    used_emails: set[str] = set()
    rows: list[tuple] = []

    for i in range(NUM_AUTH_USERS):
        user_id = AUTH_USER_IDS[i]

        # Pick a unique first.last combo
        while True:
            first = rng.choice(FIRST_NAMES)
            last = rng.choice(LAST_NAMES)
            email = f"{first.lower()}.{last.lower()}@otterworks.io"
            if email not in used_emails:
                used_emails.add(email)
                break

        display_name = f"{first} {last}"
        avatar_url = f"https://avatars.otterworks.io/{user_id}.png"

        email_verified = rng.random() < 0.80
        mfa_enabled = rng.random() < 0.15
        mfa_secret = None
        if mfa_enabled:
            mfa_secret = "".join(rng.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ234567", k=32))

        # created_at spread over last 365 days
        created_at = ref_now - timedelta(days=rng.uniform(1, 365))
        updated_at = created_at + timedelta(days=rng.uniform(0, 30))
        if updated_at > ref_now:
            updated_at = ref_now

        # ~75% of users have a recent login; rest are inactive (NULL)
        last_login_at = None
        if rng.random() < 0.75:
            last_login_at = ref_now - timedelta(days=rng.uniform(0, 30))

        rows.append((
            user_id,
            email,
            pw_hash,
            display_name,
            avatar_url,
            email_verified,
            mfa_enabled,
            mfa_secret,
            created_at,
            updated_at,
            last_login_at,
        ))

    count = bulk_insert(
        cur, "users", columns, rows,
        on_conflict="(email) DO NOTHING",
    )
    return count
