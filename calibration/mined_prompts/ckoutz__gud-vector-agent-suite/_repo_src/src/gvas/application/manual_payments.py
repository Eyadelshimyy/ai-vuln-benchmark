"""Worker side of a manual payment: close the card checkout it replaced,
send the customer's receipt unless the payment was voided first, and nudge
the owner a week before a manual plan's paid-through date."""

from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from uuid import UUID

from gvas.domain.customer_linking import enqueue_owner_email_copy, enqueue_quote_owner_notice
from gvas.domain.identifiers import BusinessId, QuoteId, SubscriptionId
from gvas.domain.money import format_money
from gvas.domain.payments import PaymentCheckoutClosedError
from gvas.domain.ports import PaymentCheckoutPort
from gvas.domain.repositories import UnitOfWork
from gvas.domain.time_zones import business_zone


class ManualPaymentEffectsService:
    def __init__(
        self,
        unit_of_work_factory: Callable[[], UnitOfWork],
        *,
        checkout: PaymentCheckoutPort | None,
        receipts: Callable[[BusinessId, Mapping[str, object]], Awaitable[None]] | None,
    ) -> None:
        self._unit_of_work_factory = unit_of_work_factory
        self._checkout = checkout
        self._receipts = receipts

    async def expire_checkout(self, payload: Mapping[str, object]) -> str:
        session_id = payload.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("checkout expiry payload is incomplete")
        if self._checkout is None:
            return "unconfigured"
        try:
            await self._checkout.expire_checkout(session_id)
        except PaymentCheckoutClosedError:
            return "already closed"
        return "closed"

    async def send_receipt(self, business_id: BusinessId, payload: Mapping[str, object]) -> str:
        try:
            quote_id = QuoteId(UUID(str(payload.get("quote_id"))))
            payment_id = UUID(str(payload.get("payment_id")))
            raw_plan = payload.get("subscription_id")
            subscription_id = None if raw_plan is None else SubscriptionId(UUID(str(raw_plan)))
        except ValueError as error:
            raise ValueError("manual receipt payload is incomplete") from error
        async with self._unit_of_work_factory() as unit_of_work:
            payments = await unit_of_work.payments.list_for_quote(business_id, quote_id)
            plan = (
                None
                if subscription_id is None
                else await unit_of_work.quote_subscriptions.get(business_id, subscription_id)
            )
        if not any(p.payment_id == payment_id and p.voided_at is None for p in payments):
            return "voided"
        if self._receipts is None:
            raise RuntimeError("customer e-mail is not wired")
        # The plan's date as it is now: a payment undone since must not show.
        through = plan.paid_through if plan is not None and plan.is_live else None
        if through is not None:
            payload = {
                **payload,
                "body": f"{payload.get('body')}\n\nYour plan is paid through"
                f" {through:%B} {through.day}, {through.year}.",
            }
        await self._receipts(business_id, payload)
        return "sent"

    async def nudge_plan(self, business_id: BusinessId, payload: Mapping[str, object]) -> str:
        """Tell the owner a manual plan runs out in a week, unless it moved
        on since the nudge was queued: paid again, undone or ended."""

        try:
            subscription_id = SubscriptionId(UUID(str(payload.get("subscription_id"))))
        except ValueError as error:
            raise ValueError("plan nudge payload is incomplete") from error
        async with self._unit_of_work_factory() as unit_of_work:
            plan = await unit_of_work.quote_subscriptions.get(business_id, subscription_id)
            if (
                plan is None
                or not plan.is_manual
                or not plan.is_live
                or plan.paid_through is None
                or plan.paid_through.isoformat() != payload.get("paid_through")
            ):
                return "moved on"
            business = await unit_of_work.businesses.get(business_id)
            zone = business_zone(business.timezone if business is not None else None) or UTC
            if plan.paid_through < datetime.now(UTC).astimezone(zone).date():
                # Ran late: an advance reminder after the date would mislead.
                return "lapsed"
            quote = await unit_of_work.quotes.get(business_id, plan.quote_id)
            if quote is None:
                return "moved on"
            customer = (
                None
                if plan.customer_id is None
                else await unit_of_work.customers.get(business_id, plan.customer_id)
            )
            who: str | None
            if customer is not None:
                who = customer.display_name or customer.email
            else:
                who = quote.draft.recipient.display_name if quote.draft is not None else None
            through = plan.paid_through
            text = (
                f"{who or 'A customer'}'s plan"
                f" ({format_money(plan.amount_minor, plan.currency)}/{plan.interval.value})"
                f" is paid through {through:%b} {through.day}."
                " Record the next payment in the dashboard when it arrives."
            )
            correlation_id = f"plan-nudge:{plan.subscription_id}:{through.isoformat()}"
            if not await enqueue_quote_owner_notice(
                unit_of_work, quote, correlation_id=correlation_id, text=text
            ):
                await enqueue_owner_email_copy(
                    unit_of_work, business_id, correlation_id=correlation_id, text=text
                )
            await unit_of_work.commit()
        return "sent"
