"""
Standalone logic test for pilot_config.py and launch_gates.py — an
in-memory fake of supabase_db's interface, same pattern as
test_ledger_flow.py, so these ASIC ERS readiness controls can be
exercised without a real Postgres instance.

Run: python3 test_pilot_config_and_launch_gates.py
"""
import asyncio
import sys
import uuid
from decimal import Decimal
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


async def update_one(table, filters, updates):
    for row in _tables.get(table, []):
        if _matches(row, filters):
            row.update(updates)
            return True
    return False


async def update_many(table, filters, updates):
    n = 0
    for row in _tables.get(table, []):
        if _matches(row, filters):
            row.update(updates)
            n += 1
    return n


async def count_documents(table, filters=None):
    filters = filters or {}
    return len([r for r in _tables.get(table, []) if _matches(r, filters)])


# ---- monkeypatch both modules under test to use the fake DB ----
import types
fake_sdb = types.SimpleNamespace(
    find_one=find_one, find_many=find_many, insert_one=insert_one,
    update_one=update_one, update_many=update_many, count_documents=count_documents,
)
sys.modules["supabase_db"] = fake_sdb

import pilot_config as pc  # noqa: E402
import launch_gates as lg  # noqa: E402

FAILURES = []


def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}")
    if not condition:
        FAILURES.append(label)


async def expect_raises(coro, exc_type, label):
    try:
        await coro
        check(label, False)
    except exc_type:
        check(label, True)
    except Exception as e:
        check(f"{label} (wrong exception type: {type(e)})", False)


async def main():
    # ---------------------------------------------------------------
    # pilot_config: hard ceilings
    # ---------------------------------------------------------------
    good = pc.PilotConfig()
    pc.validate_config_change(good, None, proposed_by="alice", approved_by="bob")
    check("default config passes validation with distinct proposer/approver", True)

    try:
        pc.validate_config_change(pc.PilotConfig(max_pilot_customers=26), None, "alice", "bob")
        check("rejects max_pilot_customers > 25", False)
    except pc.ConfigValidationError:
        check("rejects max_pilot_customers > 25", True)

    over_limit = pc.PilotConfig(contractual_credit_limit=Decimal("3000.00"))
    try:
        pc.validate_config_change(over_limit, None, "alice", "bob")
        check("rejects contractual_credit_limit > 2500", False)
    except pc.ConfigValidationError:
        check("rejects contractual_credit_limit > 2500", True)

    cash = pc.PilotConfig(cash_withdrawals_enabled=True)
    try:
        pc.validate_config_change(cash, None, "alice", "bob")
        check("rejects cash_withdrawals_enabled=True", False)
    except pc.ConfigValidationError:
        check("rejects cash_withdrawals_enabled=True", True)

    transfers = pc.PilotConfig(customer_transfers_enabled=True)
    try:
        pc.validate_config_change(transfers, None, "alice", "bob")
        check("rejects customer_transfers_enabled=True", False)
    except pc.ConfigValidationError:
        check("rejects customer_transfers_enabled=True", True)

    bad_category = pc.PilotConfig(approved_bill_categories=frozenset({"electricity", "rent"}))
    try:
        pc.validate_config_change(bad_category, None, "alice", "bob")
        check("rejects prohibited bill category (rent)", False)
    except pc.ConfigValidationError:
        check("rejects prohibited bill category (rent)", True)

    aggregate_breach = pc.PilotConfig(max_pilot_customers=25, contractual_credit_limit=Decimal("2500.00"))
    # 25 * 2500 = 62500, exactly at cap -> should pass
    pc.validate_config_change(aggregate_breach, None, "alice", "bob")
    check("25 customers x 2500 limit == 62500 cap passes (boundary, not breach)", True)

    over_aggregate = pc.PilotConfig(max_pilot_customers=25, contractual_credit_limit=Decimal("2501.00"))
    try:
        pc.validate_config_change(over_aggregate, None, "alice", "bob")
        check("rejects implied aggregate exposure > 62500", False)
    except pc.ConfigValidationError:
        check("rejects implied aggregate exposure > 62500", True)

    # ---------------------------------------------------------------
    # pilot_config: maker-checker on config changes
    # ---------------------------------------------------------------
    try:
        pc.validate_config_change(pc.PilotConfig(), None, "alice", "alice")
        check("rejects same person as proposer and approver", False)
    except pc.ConfigValidationError:
        check("rejects same person as proposer and approver", True)

    try:
        pc.validate_config_change(pc.PilotConfig(), None, "alice", None)
        check("rejects first-ever config with no approver", False)
    except pc.ConfigValidationError:
        check("rejects first-ever config with no approver", True)

    # An increase over a previous version also needs a distinct approver.
    v1 = await pc.propose_config_version(
        pc.PilotConfig(max_pilot_customers=10), proposed_by="alice", approved_by="bob", activate=True)
    check("v1 activated", v1.is_active)

    try:
        await pc.propose_config_version(
            pc.PilotConfig(max_pilot_customers=15), proposed_by="alice", approved_by="alice", activate=True)
        check("rejects limit increase approved by the same person who proposed it", False)
    except pc.ConfigValidationError:
        check("rejects limit increase approved by the same person who proposed it", True)

    v2 = await pc.propose_config_version(
        pc.PilotConfig(max_pilot_customers=15), proposed_by="alice", approved_by="carol", activate=True)
    check("v2 (distinct approver) activates fine", v2.is_active and v2.max_pilot_customers == 15)

    active = await pc.get_active_config()
    check("get_active_config reflects the latest activated version", active.max_pilot_customers == 15)

    # ---------------------------------------------------------------
    # pilot_config: customer cap / aggregate exposure runtime checks
    # ---------------------------------------------------------------
    await pc.check_customer_cap(14)  # under v2's cap of 15, should not raise
    check("check_customer_cap allows activation under the cap", True)
    try:
        await pc.check_customer_cap(15)
        check("check_customer_cap blocks the (cap+1)th activation", False)
    except pc.ConfigValidationError:
        check("check_customer_cap blocks the (cap+1)th activation", True)

    try:
        await pc.check_aggregate_exposure(Decimal("60000.00"), Decimal("5000.00"))
        check("check_aggregate_exposure blocks exceeding the cap", False)
    except pc.ConfigValidationError:
        check("check_aggregate_exposure blocks exceeding the cap", True)

    # ---------------------------------------------------------------
    # launch_gates: fail-closed defaults
    # ---------------------------------------------------------------
    authorized = await lg.is_production_authorized()
    check("production NOT authorized with zero gates recorded (fail closed)", authorized is False)

    statuses = await lg.get_all_gate_statuses()
    check("all 22 mandatory gates default to not_started",
          len(statuses) == len(lg.MANDATORY_GATES) and all(v == "not_started" for v in statuses.values()))

    # ---------------------------------------------------------------
    # launch_gates: maker-checker per gate
    # ---------------------------------------------------------------
    await lg.submit_gate_evidence("afca_membership_active", owner="dave", evidence_reference="doc://afca-cert-1")
    try:
        await lg.approve_gate("afca_membership_active", reviewer="dave")
        check("rejects gate approval by the same person who submitted evidence", False)
    except lg.LaunchGateError:
        check("rejects gate approval by the same person who submitted evidence", True)

    approved = await lg.approve_gate("afca_membership_active", reviewer="erin")
    check("gate approval by a distinct reviewer succeeds", approved["status"] == "approved")

    try:
        await lg.approve_gate("pi_insurance_active", reviewer="erin")
        check("rejects approving a gate with no submitted evidence", False)
    except lg.LaunchGateError:
        check("rejects approving a gate with no submitted evidence", True)

    # ---------------------------------------------------------------
    # launch_gates: expiry auto-closes a gate
    # ---------------------------------------------------------------
    past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    await lg.submit_gate_evidence("pi_insurance_active", owner="dave", evidence_reference="doc://pi-2024", expiry_date=past)
    await lg.approve_gate("pi_insurance_active", reviewer="erin")
    statuses = await lg.get_all_gate_statuses()
    check("expired evidence is reported as 'expired', not 'approved'", statuses["pi_insurance_active"] == "expired")

    # ---------------------------------------------------------------
    # launch_gates: production activation requires all gates + 2 distinct
    # approvers, neither of whom is the requester
    # ---------------------------------------------------------------
    still_not_authorized = await lg.is_production_authorized()
    check("production still not authorized with only 1 of 22 gates truly approved", still_not_authorized is False)

    try:
        await lg.activate_production(requested_by="frank", approver_1="frank", approver_2="grace", reason="test")
        check("rejects activation where requester is also an approver", False)
    except lg.LaunchGateError:
        check("rejects activation where requester is also an approver", True)

    try:
        await lg.activate_production(requested_by="frank", approver_1="grace", approver_2="grace", reason="test")
        check("rejects activation with two identical approvers", False)
    except lg.LaunchGateError:
        check("rejects activation with two identical approvers", True)

    # Approve every remaining mandatory gate with distinct owner/reviewer,
    # non-expiring, to reach a fully-authorized state.
    remaining = [k for k in lg.MANDATORY_GATES if k not in ("afca_membership_active", "pi_insurance_active")]
    for key in remaining:
        await lg.submit_gate_evidence(key, owner="dave", evidence_reference=f"doc://{key}")
        await lg.approve_gate(key, reviewer="erin")
    # Re-approve the expired one with a future expiry.
    future = (datetime.now(timezone.utc) + timedelta(days=365)).isoformat()
    await lg.submit_gate_evidence("pi_insurance_active", owner="dave", evidence_reference="doc://pi-2026", expiry_date=future)
    await lg.approve_gate("pi_insurance_active", reviewer="erin")

    fully_authorized = await lg.is_production_authorized()
    check("production authorized once every mandatory gate is approved and unexpired", fully_authorized is True)

    activated = await lg.activate_production(requested_by="frank", approver_1="grace", approver_2="heidi", reason="pilot launch")
    check("two-person activation succeeds once all gates are genuinely approved", activated is True)

    route = await lg.existing_customers_route()
    check("existing customers routed to 'normal' once authorized", route == "normal")

    # Simulate one gate expiring after activation -> must immediately
    # fail closed again for NEW lending, without anyone flipping a switch.
    await sdb_expire_gate("afca_membership_active")
    now_blocked = await lg.is_production_authorized()
    check("a single expired gate immediately blocks new lending again", now_blocked is False)

    wind_down_route = await lg.existing_customers_route()
    check("existing customers still routed somewhere reachable (wind_down) after a gate expires",
          wind_down_route == "wind_down")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED:")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("ALL CHECKS PASSED")


async def sdb_expire_gate(gate_key):
    past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    await update_one("launch_gates", {"gate_key": gate_key}, {"expiry_date": past})


if __name__ == "__main__":
    asyncio.run(main())
