"""
EasyBillsPay — Supabase Database Adapter
=========================================
Drop-in replacement for MongoDB Motor operations.
All methods are async-compatible and return dicts without _id.
"""

import os
import logging
from supabase import create_client, Client
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

_client: Client = None


def get_supabase() -> Client:
    global _client
    if _client is None:
        url = os.environ.get("SUPABASE_URL", "")
        key = os.environ.get("SUPABASE_SERVICE_KEY", "")
        if not url or not key:
            raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")
        _client = create_client(url, key)
        logger.info("Supabase client initialized")
    return _client


def _apply_filters(query, filters: dict):
    """Apply MongoDB-style filter operators consistently across all query builders.

    Supports plain equality (`{"field": value}`) plus `$in` and `$ne` operators
    (`{"field": {"$in": [...]}}` / `{"field": {"$ne": value}}`). Previously each
    of find_many/update_one/update_many/delete_one/delete_many implemented its
    own ad-hoc subset of this (e.g. update_one only supported equality, so a
    caller could not express "update this row only if status != 'paid'" — the
    exact primitive needed for atomic idempotency checks like payment webhooks).
    """
    for k, v in filters.items():
        if isinstance(v, dict):
            if "$in" in v:
                query = query.in_(k, v["$in"])
            elif "$ne" in v:
                query = query.neq(k, v["$ne"])
            else:
                raise ValueError(f"Unsupported filter operator for field {k!r}: {v!r}")
        else:
            query = query.eq(k, v)
    return query


# ============================================================
# Generic CRUD helpers
# ============================================================

async def find_one(table: str, filters: dict, exclude_fields: list = None) -> dict | None:
    """Find a single record matching all filters."""
    sb = get_supabase()
    query = sb.table(table).select("*")
    query = _apply_filters(query, filters)
    result = query.limit(1).execute()
    if result.data:
        row = result.data[0]
        if exclude_fields:
            for f in exclude_fields:
                row.pop(f, None)
        return row
    return None


async def find_many(table: str, filters: dict = None, exclude_fields: list = None,
                    order_by: str = None, order_desc: bool = False, limit: int = 10000) -> list:
    """Find multiple records matching filters."""
    sb = get_supabase()
    query = sb.table(table).select("*")
    if filters:
        query = _apply_filters(query, filters)
    if order_by:
        query = query.order(order_by, desc=order_desc)
    query = query.limit(limit)
    result = query.execute()
    rows = result.data or []
    if exclude_fields:
        for row in rows:
            for f in exclude_fields:
                row.pop(f, None)
    return rows


async def insert_one(table: str, data: dict) -> dict:
    """Insert a single record. Returns the inserted row."""
    sb = get_supabase()
    # Remove None values to let DB defaults work
    clean = {k: v for k, v in data.items() if v is not None}
    # Convert datetime objects to ISO strings
    for k, v in clean.items():
        if isinstance(v, datetime):
            clean[k] = v.isoformat()
    result = sb.table(table).insert(clean).execute()
    if result.data:
        return result.data[0]
    return clean


async def insert_many(table: str, rows: list) -> list:
    """Insert multiple records in a single statement. Returns the inserted rows.

    Using one multi-row INSERT (rather than N calls to insert_one) matters
    beyond performance: several tables in the ledger migration rely on a
    deferred constraint trigger that checks invariants across *all* rows
    written in the same statement/transaction (e.g. that a journal's debits
    equal its credits). Calling insert_one twice would run each as its own
    transaction and the balance check would fail on the first row alone.
    """
    sb = get_supabase()
    clean_rows = []
    for data in rows:
        clean = {k: v for k, v in data.items() if v is not None}
        for k, v in clean.items():
            if isinstance(v, datetime):
                clean[k] = v.isoformat()
        clean_rows.append(clean)
    result = sb.table(table).insert(clean_rows).execute()
    return result.data or []


async def increment_wallet_balance(user_id: str, amount: float) -> bool:
    """Atomically increment (or decrement, for negative amount) a user's wallet_balance.

    Calls the increment_wallet_balance() Postgres function (schema.sql) which
    performs `wallet_balance = wallet_balance + amount` in a single statement,
    avoiding the read-then-write race condition of a plain update_one() $inc.
    """
    sb = get_supabase()
    result = sb.rpc("increment_wallet_balance", {"p_user_id": user_id, "p_amount": amount}).execute()
    return result.data is not None


async def increment_active_plan_totals(user_id: str, collected_delta: float = 0, paid_out_delta: float = 0) -> None:
    """Atomically increment total_collected / total_paid_out on a user's active payment plan."""
    sb = get_supabase()
    sb.rpc("increment_active_plan_totals", {
        "p_user_id": user_id,
        "p_collected_delta": collected_delta,
        "p_paid_out_delta": paid_out_delta,
    }).execute()


async def update_one(table: str, filters: dict, updates: dict) -> bool:
    """Update a single record matching filters.

    `filters` values may use `{"$ne": ...}` / `{"$in": [...]}` (see _apply_filters)
    which makes it possible to express atomic conditional transitions, e.g.
    only flip payment_status to "paid" if it isn't already "paid" — the DB
    performs this as one statement, so concurrent callers (a webhook and a
    status-poll request racing each other) can't both succeed.
    """
    sb = get_supabase()
    # Handle $set and $inc operators from MongoDB syntax
    set_data = {}
    inc_data = {}

    if "$set" in updates:
        set_data = updates["$set"]
    elif "$inc" in updates:
        inc_data = updates["$inc"]
    else:
        set_data = updates

    if inc_data:
        # NOTE: this read-then-write path is NOT atomic and can lose updates
        # under concurrent requests. The two known hot paths (users.wallet_balance
        # and payment_plans.total_collected/total_paid_out) have been migrated to
        # the atomic increment_wallet_balance()/increment_active_plan_totals()
        # helpers above — call those instead of update_one(..., {"$inc": ...})
        # for those fields. This fallback remains only for any other/future
        # non-monetary increment use case.
        current = await find_one(table, filters)
        if not current:
            return False
        for field, amount in inc_data.items():
            set_data[field] = (current.get(field, 0) or 0) + amount

    if not set_data:
        return False

    query = sb.table(table).update(set_data)
    query = _apply_filters(query, filters)
    result = query.execute()
    return bool(result.data)


async def update_many(table: str, filters: dict, updates: dict) -> int:
    """Update multiple records matching filters. Returns count."""
    sb = get_supabase()
    set_data = updates.get("$set", updates)

    query = sb.table(table).update(set_data)
    query = _apply_filters(query, filters)
    result = query.execute()
    return len(result.data) if result.data else 0


async def delete_one(table: str, filters: dict) -> bool:
    """Delete a single record matching filters."""
    sb = get_supabase()
    query = sb.table(table).delete()
    query = _apply_filters(query, filters)
    result = query.execute()
    return bool(result.data)


async def delete_many(table: str, filters: dict) -> int:
    """Delete multiple records matching filters. Returns count."""
    sb = get_supabase()
    query = sb.table(table).delete()
    query = _apply_filters(query, filters)
    result = query.execute()
    return len(result.data) if result.data else 0


async def count_documents(table: str, filters: dict = None) -> int:
    """Count records matching filters."""
    sb = get_supabase()
    query = sb.table(table).select("id", count="exact")
    if filters:
        query = _apply_filters(query, filters)
    result = query.execute()
    return result.count if result.count is not None else len(result.data or [])


async def find_with_fields(table: str, filters: dict = None, fields: list = None, limit: int = 10000) -> list:
    """Find records returning only specified fields."""
    sb = get_supabase()
    select_str = ",".join(fields) if fields else "*"
    query = sb.table(table).select(select_str)
    if filters:
        query = _apply_filters(query, filters)
    result = query.limit(limit).execute()
    return result.data or []
