"""Migration 0025 on PostgreSQL: card payments already settled are filled
into the ledger, a quote paid twice counts once, and open checkouts don't."""

import asyncio
import os
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import pytest
import sqlalchemy as sa
from alembic import command
from sqlalchemy.ext.asyncio import create_async_engine

from gvas.interfaces.migrate import build_config

BEFORE = "0024_quote_texted_at"
AFTER = "0025_payments_ledger"
NOW = datetime(2026, 1, 5, 12, 0, tzinfo=UTC)


async def _execute(database_url: str, statements: list[Any]) -> list[Any]:
    engine = create_async_engine(database_url)
    try:
        async with engine.begin() as connection:
            # Only the payment tables matter here; skip the quote's other links.
            await connection.execute(sa.text("SET LOCAL session_replication_role = replica"))
            result = None
            for statement in statements:
                result = await connection.execute(statement)
            return list(result.all()) if result is not None and result.returns_rows else []
    finally:
        await engine.dispose()


def _quote(business_id: UUID, quote_id: UUID) -> Any:
    return sa.text(
        "INSERT INTO quotes (id, business_id, conversation_id, external_conversation_id, "
        "status, revision, source_message_key, last_message_key, pending_request_text, "
        "billing, version, created_at, updated_at) VALUES (:id, :business, :conversation, "
        "'c', 'delivered', 1, 'k', 'k', '', 'one_time', 1, :now, :now)"
    ).bindparams(id=quote_id, business=business_id, conversation=uuid4(), now=NOW)


def _checkout(business_id: UUID, quote_id: UUID, session_id: str, status: str, at: datetime) -> Any:
    return sa.text(
        "INSERT INTO quote_payments (id, business_id, quote_id, provider, checkout_session_id, "
        "checkout_url, amount_cents, currency, status, created_at, updated_at) VALUES "
        "(:id, :business, :quote, 'stripe', :session, 'https://pay.test', 25000, 'usd', "
        ":status, :at, :at)"
    ).bindparams(
        id=uuid4(), business=business_id, quote=quote_id, session=session_id, status=status, at=at
    )


def _subscription(business_id: UUID, quote_id: UUID, ref: str, interval: str, at: datetime) -> Any:
    return sa.text(
        "INSERT INTO quote_subscriptions (id, business_id, quote_id, customer_id, provider, "
        "stripe_subscription_id, status, interval, amount_cents, currency, "
        "cancel_at_period_end, created_at, updated_at) VALUES (:id, :business, :quote, "
        ":customer, 'stripe', :ref, 'active', :interval, 25000, 'USD', false, :at, :at)"
    ).bindparams(
        id=uuid4(),
        business=business_id,
        quote=quote_id,
        customer=uuid4(),
        ref=f"{ref}-{business_id}",
        interval=interval,
        at=at,
    )


def test_settled_checkouts_are_filled_into_the_ledger(monkeypatch: pytest.MonkeyPatch) -> None:
    database_url = os.getenv("GVAS_TEST_DATABASE_URL")
    if database_url is None:
        pytest.skip("GVAS_TEST_DATABASE_URL is not set")
    monkeypatch.setenv("GVAS_DATABASE_URL", database_url)
    config = build_config()
    business_id, one_off, plan = uuid4(), uuid4(), uuid4()
    command.upgrade(config, BEFORE)
    try:
        asyncio.run(
            _execute(
                database_url,
                [
                    sa.text(
                        "INSERT INTO businesses (id, slug, name, created_at, updated_at) "
                        "VALUES (:id, :slug, 'Ledger Co', :now, :now)"
                    ).bindparams(id=business_id, slug=f"ledger-{business_id}", now=NOW),
                    _quote(business_id, one_off),
                    _quote(business_id, plan),
                    _checkout(business_id, one_off, "cs_first", "paid", NOW),
                    _checkout(business_id, one_off, "cs_again", "paid", NOW + timedelta(hours=1)),
                    _checkout(business_id, one_off, "cs_open", "open", NOW),
                    _checkout(business_id, plan, "cs_plan", "paid", NOW),
                    _subscription(business_id, plan, "sub_old", "month", NOW - timedelta(days=60)),
                    _subscription(business_id, plan, "sub_new", "year", NOW),
                ],
            )
        )

        command.upgrade(config, AFTER)
        rows = asyncio.run(
            _execute(
                database_url,
                [
                    sa.text(
                        "SELECT reference, kind, source, method, currency, months_covered, "
                        "paid_at, duplicate FROM payments WHERE business_id = :business "
                        "ORDER BY reference"
                    ).bindparams(business=business_id)
                ],
            )
        )
        assert [tuple(row) for row in rows] == [
            ("cs_again", "one_off", "stripe", "card", "USD", None, NOW + timedelta(hours=1), True),
            ("cs_first", "one_off", "stripe", "card", "USD", None, NOW, False),
            ("cs_plan", "plan", "stripe", "card", "USD", 12, NOW, False),
        ]
    finally:
        command.downgrade(config, "base")
