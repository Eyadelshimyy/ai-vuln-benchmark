from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LEGACY = ROOT / "services" / "legacy-billing" / "db" / "seed.sql"
OUTPUT = ROOT / "services" / "billing-service" / "db" / "seed.sql"
TABLES = ("tenants", "plans", "subscriptions")


def generate() -> str:
    source = LEGACY.read_text()
    statements = []
    for table in TABLES:
        match = re.search(
            rf"INSERT INTO billing\.{table} .*?;\n",
            source,
            flags=re.DOTALL,
        )
        if match is None:
            raise RuntimeError(f"missing legacy seed section: {table}")
        statements.append(match.group(0).replace(f"billing.{table}", f"billing_svc.{table}"))
    return "\n".join(statements)


if __name__ == "__main__":
    OUTPUT.write_text(generate())
