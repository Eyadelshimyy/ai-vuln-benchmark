# /// script
# requires-python = ">=3.11"
# dependencies = ["psycopg2-binary", "bcrypt"]
# ///
"""
OtterWorks bulk seed data generator — master orchestrator.

Runs all 10 data modules in dependency order to populate a namespaced schema
with representative seed data for an engineering organization.

Usage:
    uv run testdata/generated/seed/generate.py --ns seed

Environment overrides:
    DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
"""

import argparse
import importlib
import sys
import time

# Modules in dependency order — later modules may reference IDs from earlier ones.
MODULES = [
    "seed_01_users",
    "seed_02_user_roles",
    "seed_03_user_settings",
    "seed_04_refresh_tokens",
    "seed_05_admin_users",
    "seed_06_storage_quotas",
    "seed_07_audit_logs",
    "seed_08_feature_flags",
    "seed_09_announcements",
    "seed_10_incidents_configs",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="OtterWorks bulk seed data generator")
    parser.add_argument("--ns", required=True, help="Namespace, e.g. 'seed'")
    args = parser.parse_args()

    ns = args.ns
    schema = f"otterworks_{ns}"

    print(f"\n{'=' * 60}")
    print(f"  OtterWorks Bulk Seed Data Generator")
    print(f"  Namespace: {ns}  |  Schema: {schema}")
    print(f"{'=' * 60}\n")

    # Import helpers to get DB connection
    from helpers import get_connection, set_search_path

    try:
        conn = get_connection()
    except Exception as e:
        print(f"ERROR: Cannot connect to database: {e}", file=sys.stderr)
        return 1

    conn.autocommit = False
    cur = conn.cursor()

    try:
        set_search_path(cur, ns)

        for mod_name in MODULES:
            print(f"  Running {mod_name} ...")
            t0 = time.time()
            mod = importlib.import_module(mod_name)
            count = mod.seed(cur, ns)
            elapsed = time.time() - t0
            print(f"    -> {count} rows in {elapsed:.1f}s")

        conn.commit()
        print(f"\nSeed complete. All modules ran successfully.")
        return 0

    except Exception as e:
        conn.rollback()
        print(f"\nERROR during seed — rolled back: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
