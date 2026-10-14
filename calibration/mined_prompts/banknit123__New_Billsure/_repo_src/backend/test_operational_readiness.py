"""
Standalone tests for operational_readiness.py. Same in-memory fake-DB
pattern as the other test_*.py files, no live credentials needed.

Run: python3 test_operational_readiness.py
"""
import asyncio
import sys
import types
import uuid
from datetime import datetime, timezone, timedelta

# ---- in-memory fake of supabase_db's public interface ----
_tables = {}


def _matches(row, filters):
    for k, v in filters.items():
        if row.get(k) != v:
            return False
    return True


async def find_one(table, filters, exclude_fields=None):
    for row in _tables.get(table, []):
        if _matches(row, filters):
            return dict(row)
    return None


async def find_many(table, filters=None, exclude_fields=None, limit=10000):
    filters = filters or {}
    return [dict(r) for r in _tables.get(table, []) if _matches(r, filters)][:limit]


async def insert_one(table, data):
    row = dict(data)
    row.setdefault("id", str(uuid.uuid4()))
    _tables.setdefault(table, []).append(row)
    return dict(row)


fake_sdb = types.SimpleNamespace(find_one=find_one, find_many=find_many, insert_one=insert_one)
sys.modules["supabase_db"] = fake_sdb

import operational_readiness as opr   # noqa: E402

FAILURES = []


def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}")
    if not condition:
        FAILURES.append(label)


async def main():
    # ---------------------------------------------------------------
    # Health aggregation
    # ---------------------------------------------------------------
    all_healthy = {
        "database": opr.ComponentHealth("database", True, "connected"),
        "reconciliation_job": opr.ComponentHealth("reconciliation_job", True, "last run 2h ago"),
    }
    healthy_report = await opr.check_health(all_healthy)
    check("overall health is True when every component is healthy", healthy_report.overall_healthy is True)

    one_unhealthy = {
        "database": opr.ComponentHealth("database", True, "connected"),
        "reconciliation_job": opr.ComponentHealth("reconciliation_job", False, "stalled for 3 days"),
    }
    unhealthy_report = await opr.check_health(one_unhealthy)
    check("overall health is False if even one component is unhealthy (no averaging/partial credit)", unhealthy_report.overall_healthy is False)

    empty_report = await opr.check_health({})
    check("an empty component set is reported unhealthy, not vacuously healthy", empty_report.overall_healthy is False)

    # ---------------------------------------------------------------
    # Feature flags: fail closed for unset flags
    # ---------------------------------------------------------------
    never_set = await opr.get_feature_flag("some_flag_never_configured")
    check("an unset feature flag defaults to False (fails closed)", never_set is False)

    try:
        await opr.set_feature_flag("real_money_enabled", True, changed_by="admin1", reason="")
        check("rejects a feature flag change with no documented reason", False)
    except opr.OperationalReadinessError:
        check("rejects a feature flag change with no documented reason", True)

    await opr.set_feature_flag("real_money_enabled", True, changed_by="admin1", reason="pilot launch approved")
    check("feature flag reads True after being explicitly enabled", await opr.get_feature_flag("real_money_enabled") is True)

    await opr.set_feature_flag("real_money_enabled", False, changed_by="admin2", reason="regulatory gate expired")
    check("the MOST RECENT flag change wins, not the first one set", await opr.get_feature_flag("real_money_enabled") is False)

    # ---------------------------------------------------------------
    # Background-job stall detection
    # ---------------------------------------------------------------
    never_run = await opr.is_job_stalled("never_run_job")
    check("a job that has never reported a heartbeat is treated as stalled, not 'unknown, assume fine'", never_run is True)

    await opr.record_job_heartbeat("reconciliation_loop", status="ok")
    fresh = await opr.is_job_stalled("reconciliation_loop", now=datetime.now(timezone.utc))
    check("a job with a fresh heartbeat is not stalled", fresh is False)

    stale_time = datetime.now(timezone.utc) + timedelta(minutes=opr.DEFAULT_STALL_THRESHOLD_MINUTES + 5)
    stalled = await opr.is_job_stalled("reconciliation_loop", now=stale_time)
    check("a job whose last heartbeat is older than the stall threshold is reported stalled", stalled is True)

    # ---------------------------------------------------------------
    # Backup verification
    # ---------------------------------------------------------------
    unverified_backup = await opr.record_backup_verification("backup-2026-08-01", verified_by="ops1", restore_tested=False)
    check("a backup can be recorded without an actual restore test (but the flag says so honestly)", unverified_backup["restore_tested"] is False)

    tested_backup = await opr.record_backup_verification("backup-2026-08-01", verified_by="ops1", restore_tested=True,
                                                            notes="restored into staging, confirmed row counts match")
    check("a genuinely restore-tested backup is recorded distinctly", tested_backup["restore_tested"] is True)

    # ---------------------------------------------------------------
    # Synthetic ASIC-review scenarios
    # ---------------------------------------------------------------
    check("at least one demo scenario is defined", len(opr.SYNTHETIC_ASIC_REVIEW_SCENARIOS) > 0)
    cap_scenario = opr.get_demo_scenario("customer_cap_enforcement")
    check("get_demo_scenario finds a known scenario by key", cap_scenario is not None and "pilot_config.py" in cap_scenario.demonstrates)
    check("get_demo_scenario returns None for an unknown key", opr.get_demo_scenario("not_a_real_scenario") is None)
    keys = [s.key for s in opr.SYNTHETIC_ASIC_REVIEW_SCENARIOS]
    check("every scenario has a unique key", len(keys) == len(set(keys)))

    print()
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED:")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("ALL CHECKS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
