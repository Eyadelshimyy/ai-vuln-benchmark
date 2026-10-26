# /// script
# requires-python = ">=3.11"
# dependencies = ["psycopg2-binary", "tabulate"]
# ///
"""
OtterWorks test-data validation harness.

Validates that generated synthetic data satisfies:
  1. Schema conformance (FK, NOT NULL, types, unique constraints)
  2. Temporal consistency (no event references resources created after it)
  3. Referential integrity across services
  4. Acceptance-criteria coverage (row counts, distributions, field completeness)

Usage:
    uv run testdata/harness/validate.py --ns <namespace> [--criteria <file.json>]

Environment overrides:
    DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD

Exit codes:
    0 = all checks PASS
    1 = one or more checks FAIL
    2 = connection/config error
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import psycopg2
from tabulate import tabulate

# ── Configuration ─────────────────────────────────────────────────────────────

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "dbname": os.getenv("DB_NAME", "otterworks"),
    "user": os.getenv("DB_USER", "otterworks"),
    "password": os.getenv("DB_PASSWORD", "otterworks_dev"),
}

# ── Data Classes ──────────────────────────────────────────────────────────────


@dataclass
class CheckResult:
    name: str
    status: str  # PASS | FAIL
    detail: str = ""


@dataclass
class ValidationReport:
    namespace: str
    checks: list[CheckResult] = field(default_factory=list)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def passed(self) -> bool:
        return all(c.status == "PASS" for c in self.checks)

    def summary_table(self) -> str:
        rows = [[c.name, c.status, c.detail] for c in self.checks]
        return tabulate(rows, headers=["Check", "Status", "Detail"], tablefmt="simple")


# ── Core Checks ──────────────────────────────────────────────────────────────


def check_table_exists(cur, schema: str, table: str) -> CheckResult:
    """Verify a table exists in the given schema."""
    cur.execute(
        """
        SELECT EXISTS (
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = %s AND table_name = %s
        )
        """,
        (schema, table),
    )
    exists = cur.fetchone()[0]
    if exists:
        return CheckResult(f"table_exists:{schema}.{table}", "PASS")
    return CheckResult(
        f"table_exists:{schema}.{table}", "FAIL", "table does not exist"
    )


def check_row_count(
    cur, schema: str, table: str, expected_min: int, expected_max: int | None = None
) -> CheckResult:
    """Verify row count is within expected bounds."""
    cur.execute(f'SELECT count(*) FROM "{schema}"."{table}"')  # noqa: S608
    count = cur.fetchone()[0]
    if count < expected_min:
        return CheckResult(
            f"row_count:{schema}.{table}",
            "FAIL",
            f"got {count}, expected >= {expected_min}",
        )
    if expected_max is not None and count > expected_max:
        return CheckResult(
            f"row_count:{schema}.{table}",
            "FAIL",
            f"got {count}, expected <= {expected_max}",
        )
    return CheckResult(f"row_count:{schema}.{table}", "PASS", f"count={count}")


def check_not_null(cur, schema: str, table: str, columns: list[str]) -> CheckResult:
    """Verify specified columns have no NULL values."""
    failures = []
    for col in columns:
        cur.execute(
            f'SELECT count(*) FROM "{schema}"."{table}" WHERE "{col}" IS NULL'  # noqa: S608
        )
        null_count = cur.fetchone()[0]
        if null_count > 0:
            failures.append(f"{col}={null_count} nulls")
    if failures:
        return CheckResult(
            f"not_null:{schema}.{table}", "FAIL", "; ".join(failures)
        )
    return CheckResult(f"not_null:{schema}.{table}", "PASS", "all required columns populated")


def check_unique(cur, schema: str, table: str, columns: list[str]) -> CheckResult:
    """Verify uniqueness constraints hold."""
    failures = []
    for col in columns:
        cur.execute(
            f"""
            SELECT count(*) - count(DISTINCT "{col}")
            FROM "{schema}"."{table}"
            WHERE "{col}" IS NOT NULL
            """  # noqa: S608
        )
        dup_count = cur.fetchone()[0]
        if dup_count > 0:
            failures.append(f"{col}={dup_count} duplicates")
    if failures:
        return CheckResult(f"unique:{schema}.{table}", "FAIL", "; ".join(failures))
    return CheckResult(f"unique:{schema}.{table}", "PASS", "uniqueness satisfied")


def check_fk_integrity(
    cur, schema: str, child_table: str, child_col: str, parent_table: str, parent_col: str
) -> CheckResult:
    """Verify all FK references resolve to existing parent rows."""
    cur.execute(
        f"""
        SELECT count(*)
        FROM "{schema}"."{child_table}" c
        LEFT JOIN "{schema}"."{parent_table}" p ON c."{child_col}" = p."{parent_col}"
        WHERE c."{child_col}" IS NOT NULL AND p."{parent_col}" IS NULL
        """  # noqa: S608
    )
    orphan_count = cur.fetchone()[0]
    if orphan_count > 0:
        return CheckResult(
            f"fk_integrity:{child_table}.{child_col}->{parent_table}.{parent_col}",
            "FAIL",
            f"{orphan_count} orphaned references",
        )
    return CheckResult(
        f"fk_integrity:{child_table}.{child_col}->{parent_table}.{parent_col}",
        "PASS",
        "all references resolve",
    )


def check_temporal_consistency(
    cur,
    schema: str,
    event_table: str,
    event_ts_col: str,
    resource_table: str,
    resource_ts_col: str,
    join_col: str,
) -> CheckResult:
    """
    Verify events do not reference resources created after the event.

    Example: an audit_log entry should not have created_at < the resource's
    created_at (the event can't happen before the resource exists).
    """
    cur.execute(
        f"""
        SELECT count(*)
        FROM "{schema}"."{event_table}" e
        JOIN "{schema}"."{resource_table}" r ON e."{join_col}" = r."id"
        WHERE e."{event_ts_col}" < r."{resource_ts_col}"
        """  # noqa: S608
    )
    violations = cur.fetchone()[0]
    if violations > 0:
        return CheckResult(
            f"temporal_consistency:{event_table}.{event_ts_col}<{resource_table}.{resource_ts_col}",
            "FAIL",
            f"{violations} entries reference resources created after the event timestamp",
        )
    return CheckResult(
        f"temporal_consistency:{event_table}.{event_ts_col}<{resource_table}.{resource_ts_col}",
        "PASS",
        "all events respect causal ordering",
    )


def check_enum_values(
    cur, schema: str, table: str, column: str, allowed: list[str]
) -> CheckResult:
    """Verify a column only contains allowed enum values."""
    cur.execute(
        f'SELECT DISTINCT "{column}" FROM "{schema}"."{table}" WHERE "{column}" IS NOT NULL'  # noqa: S608
    )
    actual = {row[0] for row in cur.fetchall()}
    invalid = actual - set(allowed)
    if invalid:
        return CheckResult(
            f"enum_values:{schema}.{table}.{column}",
            "FAIL",
            f"invalid values: {sorted(invalid)}",
        )
    return CheckResult(
        f"enum_values:{schema}.{table}.{column}", "PASS", f"all in {allowed}"
    )


def check_quota_not_exceeded(cur, schema: str) -> CheckResult:
    """Verify used_bytes does not exceed quota_bytes in storage_quotas."""
    cur.execute(
        f"""
        SELECT count(*)
        FROM "{schema}"."storage_quotas"
        WHERE used_bytes > quota_bytes
        """  # noqa: S608
    )
    violations = cur.fetchone()[0]
    if violations > 0:
        return CheckResult(
            "business_rule:quota_not_exceeded",
            "FAIL",
            f"{violations} users have used_bytes > quota_bytes",
        )
    return CheckResult("business_rule:quota_not_exceeded", "PASS", "all within quota")


# ── Criteria-Based Validation ─────────────────────────────────────────────────


def validate_criteria(cur, schema: str, criteria_file: Path) -> list[CheckResult]:
    """
    Run acceptance-criteria checks from a JSON spec.

    The criteria file maps table names to constraints:
    {
      "users": {"min_rows": 100, "max_rows": 1000, "required_columns": ["email", "display_name"]},
      "audit_logs": {"min_rows": 500, "required_columns": ["action", "actor_id"]}
    }
    """
    results = []
    with open(criteria_file) as f:
        criteria = json.load(f)

    for table, constraints in criteria.items():
        if "min_rows" in constraints:
            results.append(
                check_row_count(
                    cur, schema, table,
                    constraints["min_rows"],
                    constraints.get("max_rows"),
                )
            )
        if "required_columns" in constraints:
            results.append(
                check_not_null(cur, schema, table, constraints["required_columns"])
            )
        if "unique_columns" in constraints:
            results.append(
                check_unique(cur, schema, table, constraints["unique_columns"])
            )
        if "enum_constraints" in constraints:
            for col, allowed in constraints["enum_constraints"].items():
                results.append(check_enum_values(cur, schema, table, col, allowed))

    return results


# ── Standard Validation Suite ─────────────────────────────────────────────────


def run_standard_checks(cur, schema: str) -> list[CheckResult]:
    """
    Run the standard OtterWorks test-data validation suite.

    These checks apply to any generated dataset regardless of the specific
    user story — they enforce the invariants of the OtterWorks schema.
    """
    results = []

    # Core tables must exist
    core_tables = [
        "users", "user_roles", "user_settings", "refresh_tokens",
        "admin_users", "storage_quotas", "audit_logs",
        "feature_flags", "announcements", "incidents",
    ]
    for table in core_tables:
        results.append(check_table_exists(cur, schema, table))

    # FK integrity: user_roles -> users
    results.append(
        check_fk_integrity(cur, schema, "user_roles", "user_id", "users", "id")
    )
    # FK integrity: user_settings -> users
    results.append(
        check_fk_integrity(cur, schema, "user_settings", "user_id", "users", "id")
    )
    # FK integrity: refresh_tokens -> users
    results.append(
        check_fk_integrity(cur, schema, "refresh_tokens", "user_id", "users", "id")
    )
    # FK integrity: storage_quotas -> admin_users (user_id)
    results.append(
        check_fk_integrity(cur, schema, "storage_quotas", "user_id", "admin_users", "id")
    )
    # FK integrity: audit_logs -> admin_users (actor_id, nullable)
    results.append(
        check_fk_integrity(cur, schema, "audit_logs", "actor_id", "admin_users", "id")
    )

    # Temporal consistency: audit_logs should not reference future resources
    results.append(
        check_temporal_consistency(
            cur, schema,
            "audit_logs", "created_at",
            "admin_users", "created_at",
            "actor_id",
        )
    )

    # Enum checks
    results.append(
        check_enum_values(
            cur, schema, "user_roles", "role",
            ["USER", "EDITOR", "ADMIN", "OWNER"],
        )
    )
    results.append(
        check_enum_values(
            cur, schema, "admin_users", "role",
            ["viewer", "editor", "admin", "super_admin"],
        )
    )
    results.append(
        check_enum_values(
            cur, schema, "admin_users", "status",
            ["active", "suspended", "deactivated"],
        )
    )
    results.append(
        check_enum_values(
            cur, schema, "incidents", "severity",
            ["low", "medium", "high", "critical"],
        )
    )
    results.append(
        check_enum_values(
            cur, schema, "incidents", "status",
            ["open", "investigating", "resolved", "closed"],
        )
    )
    results.append(
        check_enum_values(
            cur, schema, "storage_quotas", "tier",
            ["free", "basic", "pro", "enterprise"],
        )
    )
    results.append(
        check_enum_values(
            cur, schema, "announcements", "severity",
            ["info", "warning", "critical"],
        )
    )
    results.append(
        check_enum_values(
            cur, schema, "announcements", "status",
            ["draft", "active", "expired"],
        )
    )

    # Business rule: quota not exceeded
    results.append(check_quota_not_exceeded(cur, schema))

    # Uniqueness: emails must be unique
    results.append(check_unique(cur, schema, "users", ["email"]))
    results.append(check_unique(cur, schema, "admin_users", ["email"]))

    return results


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate OtterWorks synthetic test data"
    )
    parser.add_argument(
        "--ns", required=True,
        help="Namespace (schema prefix) to validate, e.g. 'dev' -> schema 'otterworks_dev'",
    )
    parser.add_argument(
        "--criteria", type=Path, default=None,
        help="Optional JSON file with acceptance-criteria checks (from GH Issue)",
    )
    parser.add_argument(
        "--schema-override", default=None,
        help="Override the schema name (default: otterworks_<ns>)",
    )
    args = parser.parse_args()

    schema = args.schema_override or f"otterworks_{args.ns}"

    try:
        conn = psycopg2.connect(**DB_CONFIG)
        conn.autocommit = True
    except psycopg2.OperationalError as e:
        print(f"ERROR: Cannot connect to database: {e}", file=sys.stderr)
        return 2

    cur = conn.cursor()
    report = ValidationReport(namespace=args.ns)

    # Check schema exists
    cur.execute(
        "SELECT EXISTS (SELECT 1 FROM information_schema.schemata WHERE schema_name = %s)",
        (schema,),
    )
    if not cur.fetchone()[0]:
        print(f"ERROR: Schema '{schema}' does not exist", file=sys.stderr)
        cur.close()
        conn.close()
        return 2

    # Run standard suite
    report.checks.extend(run_standard_checks(cur, schema))

    # Run criteria-based checks if provided
    if args.criteria:
        if not args.criteria.exists():
            print(f"ERROR: Criteria file '{args.criteria}' does not exist", file=sys.stderr)
            cur.close()
            conn.close()
            return 2
        report.checks.extend(validate_criteria(cur, schema, args.criteria))

    cur.close()
    conn.close()

    # Output report
    print(f"\n{'=' * 60}")
    print(f"  OtterWorks Test-Data Validation Report")
    print(f"  Namespace: {args.ns}  |  Schema: {schema}")
    print(f"  Ran at: {report.started_at.isoformat()}")
    print(f"{'=' * 60}\n")
    print(report.summary_table())
    print()

    pass_count = sum(1 for c in report.checks if c.status == "PASS")
    fail_count = sum(1 for c in report.checks if c.status == "FAIL")
    print(f"  PASS={pass_count}  FAIL={fail_count}")

    if report.passed:
        print("\n  ✓ All checks passed.\n")
        return 0
    else:
        print("\n  ✗ Validation FAILED. Fix the generator and re-run.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
