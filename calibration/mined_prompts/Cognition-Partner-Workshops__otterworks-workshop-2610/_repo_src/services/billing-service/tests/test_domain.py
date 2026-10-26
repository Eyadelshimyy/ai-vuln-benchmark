from __future__ import annotations

from dataclasses import replace
from datetime import date
from decimal import Decimal
from pathlib import Path
from uuid import UUID

import pytest

from app.domain import (
    EntitlementRow,
    PlanRow,
    SubscriptionRow,
    catalog,
    change_plan,
    entitlement,
)

TENANT = UUID("00000000-0000-0000-0000-000000000001")
STARTER = UUID("10000000-0000-0000-0000-000000000001")
GROWTH = UUID("10000000-0000-0000-0000-000000000002")
SUBSCRIPTION = UUID("20000000-0000-0000-0000-000000000001")


class FakeRepository:
    def __init__(self, subscriptions: list[SubscriptionRow]) -> None:
        self.subscriptions = subscriptions
        self.updates: list[tuple[UUID, date, str]] = []
        self.inserted: list[tuple[UUID, UUID, UUID, date, str]] = []

    def list_subscriptions(self, tenant_id: UUID) -> list[SubscriptionRow]:
        return list(self.subscriptions)

    def update_subscription(self, subscription_id: UUID, ends_on: date, status: str) -> None:
        self.updates.append((subscription_id, ends_on, status))
        self.subscriptions = [
            replace(item, ends_on=ends_on, status=status)
            if item.subscription_id == subscription_id
            else item
            for item in self.subscriptions
        ]

    def insert_subscription(
        self,
        subscription_id: UUID,
        tenant_id: UUID,
        plan_id: UUID,
        starts_on: date,
        status: str,
    ) -> None:
        self.inserted.append((subscription_id, tenant_id, plan_id, starts_on, status))
        self.subscriptions.append(
            SubscriptionRow(subscription_id, tenant_id, plan_id, starts_on, None, status, None)
        )


@pytest.mark.rule("PLANS-001")
def test_catalog_is_sorted_in_plain_python() -> None:
    plans = [
        PlanRow(GROWTH, "GROWTH", "growth", Decimal("149"), 500, Decimal("0.035"), True),
        PlanRow(STARTER, "STARTER", "starter", Decimal("49"), 100, Decimal("0.055"), True),
    ]
    assert [item.code for item in catalog(plans)] == ["STARTER", "GROWTH"]


@pytest.mark.rule("PLANS-002")
def test_entitlement_selects_effective_subscription() -> None:
    rows = [
        EntitlementRow(
            TENANT, "STARTER", "starter", Decimal("49"), 100, "active", None, date(2026, 1, 1)
        ),
        EntitlementRow(
            TENANT, "GROWTH", "growth", Decimal("149"), 500, "active", None, date(2026, 3, 1)
        ),
    ]
    assert entitlement(rows, TENANT, date(2026, 2, 28)).plan_code == "STARTER"


@pytest.mark.rule("PLANS-002")
def test_entitlement_preserves_suspended_status() -> None:
    row = EntitlementRow(
        TENANT, "GROWTH", "growth", Decimal("149"), 500, "suspended", None, date(2026, 1, 1)
    )
    assert entitlement([row], TENANT, date(2026, 2, 28)).subscription_status == "suspended"


@pytest.mark.rule("PLANS-003")
def test_change_plan_closes_prior_subscription() -> None:
    repository = FakeRepository(
        [SubscriptionRow(SUBSCRIPTION, TENANT, STARTER, date(2026, 1, 1), None, "active", None)]
    )
    change_plan(repository, TENANT, GROWTH, date(2026, 3, 1))
    assert repository.updates == [(SUBSCRIPTION, date(2026, 2, 28), "active")]


@pytest.mark.rule("PLANS-004")
def test_change_plan_uses_stable_uuid5_and_preserves_history() -> None:
    repository = FakeRepository(
        [SubscriptionRow(SUBSCRIPTION, TENANT, STARTER, date(2026, 1, 1), None, "active", None)]
    )
    result, created = change_plan(repository, TENANT, GROWTH, date(2026, 3, 1))
    assert len(result) == 2
    assert created.plan_id == GROWTH
    assert repository.inserted[0][0].version == 5


def test_change_plan_reports_new_subscription_with_later_dated_history() -> None:
    later = SubscriptionRow(
        UUID("20000000-0000-0000-0000-000000000002"),
        TENANT,
        GROWTH,
        date(2026, 6, 1),
        None,
        "active",
        None,
    )
    repository = FakeRepository(
        [
            SubscriptionRow(
                SUBSCRIPTION, TENANT, STARTER, date(2026, 1, 1), None, "active", None
            ),
            later,
        ]
    )
    subscriptions, created = change_plan(repository, TENANT, GROWTH, date(2026, 3, 1))
    assert subscriptions[-1].starts_on == date(2026, 6, 1)
    assert created.starts_on == date(2026, 3, 1)


def test_generated_seed_is_current() -> None:
    from scripts.generate_seed import generate

    seed_path = Path(__file__).parents[1] / "db" / "seed.sql"
    assert seed_path.read_text() == generate()
