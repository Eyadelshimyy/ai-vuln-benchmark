from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from gvas.domain.enums import BillingInterval, QuotePaymentStatus
from gvas.domain.identifiers import BusinessId, CustomerId, QuoteId, SubscriptionId
from gvas.domain.invoices import (
    FIRST_INVOICE_SEQUENCE,
    Invoice,
    InvoiceConflictError,
    InvoiceLine,
)
from gvas.domain.payments import (
    LedgerPayment,
    PaymentKind,
    PaymentMethod,
    PaymentSource,
    QuotePaymentConflictError,
    QuotePaymentRecord,
    QuoteSubscriptionRecord,
)
from gvas.infrastructure.models import Business
from gvas.infrastructure.payment_models import (
    InvoiceRow,
    LedgerPaymentRow,
    PaymentProviderEvent,
    QuotePayment,
    QuoteSubscription,
)


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _aware_or_none(value: datetime | None) -> datetime | None:
    return None if value is None else _aware(value)


class SqlQuotePaymentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    @staticmethod
    def _record(row: QuotePayment) -> QuotePaymentRecord:
        return QuotePaymentRecord(
            payment_id=row.id,
            business_id=BusinessId(row.business_id),
            quote_id=QuoteId(row.quote_id),
            provider=row.provider,
            checkout_session_id=row.checkout_session_id,
            checkout_url=row.checkout_url,
            payment_intent_id=row.payment_intent_id,
            expires_at=_aware_or_none(row.expires_at),
            amount_minor=row.amount_cents,
            currency=row.currency,
            status=QuotePaymentStatus(row.status),
            created_at=_aware(row.created_at),
            updated_at=_aware(row.updated_at),
        )

    async def find_open(
        self, business_id: BusinessId, quote_id: QuoteId
    ) -> QuotePaymentRecord | None:
        row = await self.session.scalar(
            select(QuotePayment)
            .where(
                QuotePayment.business_id == business_id,
                QuotePayment.quote_id == quote_id,
                QuotePayment.status.in_(
                    (
                        QuotePaymentStatus.OPEN.value,
                        QuotePaymentStatus.PENDING.value,
                    )
                ),
            )
            .order_by(QuotePayment.created_at.desc())
            .limit(1)
        )
        return None if row is None else self._record(row)

    async def count_for_quote(self, business_id: BusinessId, quote_id: QuoteId) -> int:
        return (
            await self.session.scalar(
                select(func.count())
                .select_from(QuotePayment)
                .where(
                    QuotePayment.business_id == business_id,
                    QuotePayment.quote_id == quote_id,
                )
            )
            or 0
        )

    async def find_by_checkout_session(self, checkout_session_id: str) -> QuotePaymentRecord | None:
        row = await self.session.scalar(
            select(QuotePayment).where(QuotePayment.checkout_session_id == checkout_session_id)
        )
        return None if row is None else self._record(row)

    async def create(self, record: QuotePaymentRecord) -> None:
        row = QuotePayment(
            id=record.payment_id,
            business_id=record.business_id,
            quote_id=record.quote_id,
            provider=record.provider,
            checkout_session_id=record.checkout_session_id,
            checkout_url=record.checkout_url,
            payment_intent_id=record.payment_intent_id,
            expires_at=record.expires_at,
            amount_cents=record.amount_minor,
            currency=record.currency,
            status=record.status.value,
            created_at=record.created_at,
            updated_at=record.updated_at,
        )
        try:
            async with self.session.begin_nested():
                self.session.add(row)
                await self.session.flush()
        except IntegrityError as error:
            raise QuotePaymentConflictError(
                "a payment for this checkout session already exists"
            ) from error

    async def save(
        self,
        record: QuotePaymentRecord,
        *,
        expected_from: QuotePaymentStatus,
    ) -> None:
        """Status-guarded write: one UPDATE conditioned on the row still
        holding ``expected_from``, so two concurrently processed events cannot
        overwrite each other — the loser raises and is retried against fresh
        state."""

        result = await self.session.execute(
            update(QuotePayment)
            .where(
                QuotePayment.id == record.payment_id,
                QuotePayment.status == expected_from.value,
            )
            .values(
                checkout_session_id=record.checkout_session_id,
                checkout_url=record.checkout_url,
                payment_intent_id=record.payment_intent_id,
                expires_at=record.expires_at,
                amount_cents=record.amount_minor,
                currency=record.currency,
                status=record.status.value,
                updated_at=record.updated_at,
            )
        )
        if self._rowcount(result) != 1:
            raise QuotePaymentConflictError(f"payment attempt already moved past {expected_from}")

    @staticmethod
    def _rowcount(result: object) -> int:
        rowcount = getattr(result, "rowcount", None)
        return -1 if rowcount is None else int(rowcount)


class SqlQuoteSubscriptionRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    @staticmethod
    def _record(row: QuoteSubscription) -> QuoteSubscriptionRecord:
        return QuoteSubscriptionRecord(
            subscription_id=SubscriptionId(row.id),
            business_id=BusinessId(row.business_id),
            quote_id=QuoteId(row.quote_id),
            customer_id=None if row.customer_id is None else CustomerId(row.customer_id),
            provider=row.provider,
            subscription_ref=row.stripe_subscription_id,
            status=row.status,
            interval=BillingInterval(row.interval),
            amount_minor=row.amount_cents,
            currency=row.currency,
            current_period_end=_aware_or_none(row.current_period_end),
            cancel_at_period_end=row.cancel_at_period_end,
            created_at=_aware(row.created_at),
            updated_at=_aware(row.updated_at),
            paid_from=row.paid_from,
            paid_through=row.paid_through,
        )

    async def list_for_quote(
        self, business_id: BusinessId, quote_id: QuoteId
    ) -> tuple[QuoteSubscriptionRecord, ...]:
        rows = await self.session.scalars(
            select(QuoteSubscription)
            .where(
                QuoteSubscription.business_id == business_id,
                QuoteSubscription.quote_id == quote_id,
            )
            .order_by(QuoteSubscription.created_at.desc())
        )
        return tuple(self._record(row) for row in rows)

    async def get(
        self,
        business_id: BusinessId,
        subscription_id: SubscriptionId,
        *,
        for_update: bool = False,
    ) -> QuoteSubscriptionRecord | None:
        query = select(QuoteSubscription).where(
            QuoteSubscription.business_id == business_id,
            QuoteSubscription.id == subscription_id,
        )
        if for_update:
            query = query.with_for_update().execution_options(populate_existing=True)
        row = await self.session.scalar(query)
        return None if row is None else self._record(row)

    async def find_by_subscription_ref(
        self, subscription_ref: str
    ) -> QuoteSubscriptionRecord | None:
        row = await self.session.scalar(
            select(QuoteSubscription).where(
                QuoteSubscription.stripe_subscription_id == subscription_ref
            )
        )
        return None if row is None else self._record(row)

    async def list_for_customer(
        self, business_id: BusinessId, customer_id: CustomerId
    ) -> tuple[QuoteSubscriptionRecord, ...]:
        rows = await self.session.scalars(
            select(QuoteSubscription)
            .where(
                QuoteSubscription.business_id == business_id,
                QuoteSubscription.customer_id == customer_id,
            )
            .order_by(QuoteSubscription.created_at.desc())
        )
        return tuple(self._record(row) for row in rows)

    async def list_for_business(
        self, business_id: BusinessId
    ) -> tuple[QuoteSubscriptionRecord, ...]:
        rows = await self.session.scalars(
            select(QuoteSubscription)
            .where(QuoteSubscription.business_id == business_id)
            .order_by(QuoteSubscription.created_at.desc())
        )
        return tuple(self._record(row) for row in rows)

    async def create(self, record: QuoteSubscriptionRecord) -> None:
        row = QuoteSubscription(
            id=record.subscription_id,
            business_id=record.business_id,
            quote_id=record.quote_id,
            customer_id=record.customer_id,
            provider=record.provider,
            stripe_subscription_id=record.subscription_ref,
            status=record.status,
            interval=record.interval.value,
            amount_cents=record.amount_minor,
            currency=record.currency,
            current_period_end=record.current_period_end,
            cancel_at_period_end=record.cancel_at_period_end,
            created_at=record.created_at,
            updated_at=record.updated_at,
            paid_from=record.paid_from,
            paid_through=record.paid_through,
        )
        try:
            async with self.session.begin_nested():
                self.session.add(row)
                await self.session.flush()
        except IntegrityError as error:
            raise QuotePaymentConflictError("this subscription is already recorded") from error

    async def save(self, record: QuoteSubscriptionRecord) -> None:
        await self.session.execute(
            update(QuoteSubscription)
            .where(
                QuoteSubscription.id == record.subscription_id,
                QuoteSubscription.business_id == record.business_id,
            )
            .values(
                customer_id=record.customer_id,
                paid_from=record.paid_from,
                paid_through=record.paid_through,
                status=record.status,
                interval=record.interval.value,
                amount_cents=record.amount_minor,
                currency=record.currency,
                current_period_end=record.current_period_end,
                cancel_at_period_end=record.cancel_at_period_end,
                updated_at=record.updated_at,
            )
        )


class SqlPaymentEventRepository:
    """First writer wins: the row lands inside the caller's transaction, so a
    failed effect un-records the event and the provider's retry replays it."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def try_record(self, provider: str, event_id: str, now: datetime) -> bool:
        row = PaymentProviderEvent(provider=provider, event_id=event_id, received_at=now)
        try:
            async with self.session.begin_nested():
                self.session.add(row)
                await self.session.flush()
        except IntegrityError:
            return False
        return True


class SqlPaymentLedgerRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    @staticmethod
    def _payment(row: LedgerPaymentRow) -> LedgerPayment:
        return LedgerPayment(
            payment_id=row.id,
            business_id=BusinessId(row.business_id),
            quote_id=QuoteId(row.quote_id),
            kind=PaymentKind(row.kind),
            source=PaymentSource(row.source),
            method=PaymentMethod(row.method),
            reference=row.reference,
            amount_minor=row.amount_cents,
            currency=row.currency,
            paid_at=_aware(row.paid_at),
            months_covered=row.months_covered,
            recorded_by=row.recorded_by,
            recorded_at=_aware(row.recorded_at),
            note=row.note,
            voided_at=_aware_or_none(row.voided_at),
            voided_by=row.voided_by,
            duplicate=row.duplicate,
            customer_status_before=row.customer_status_before,
        )

    async def _insert(self, payment: LedgerPayment) -> bool:
        row = LedgerPaymentRow(
            id=payment.payment_id,
            business_id=payment.business_id,
            quote_id=payment.quote_id,
            kind=payment.kind.value,
            source=payment.source.value,
            method=payment.method.value,
            reference=payment.reference,
            amount_cents=payment.amount_minor,
            currency=payment.currency,
            paid_at=payment.paid_at,
            months_covered=payment.months_covered,
            recorded_by=payment.recorded_by,
            recorded_at=payment.recorded_at,
            note=payment.note,
            voided_at=payment.voided_at,
            voided_by=payment.voided_by,
            duplicate=payment.duplicate,
            customer_status_before=payment.customer_status_before,
        )
        try:
            async with self.session.begin_nested():
                self.session.add(row)
                await self.session.flush()
        except IntegrityError:
            return False
        return True

    async def record(self, payment: LedgerPayment) -> LedgerPayment | None:
        if await self._insert(payment):
            return payment
        known = await self.session.scalar(
            select(LedgerPaymentRow.id).where(
                LedgerPaymentRow.source == payment.source.value,
                LedgerPaymentRow.reference == payment.reference,
            )
        )
        if known is not None or payment.kind is not PaymentKind.ONE_OFF or payment.duplicate:
            return None
        # The quote already has an active payment: keep the money on record
        # for the owner to refund, but never count it twice. The one that
        # settled first counts, whichever webhook arrived first.
        active = await self.session.scalar(
            select(LedgerPaymentRow)
            .where(
                LedgerPaymentRow.business_id == payment.business_id,
                LedgerPaymentRow.quote_id == payment.quote_id,
                LedgerPaymentRow.kind == PaymentKind.ONE_OFF.value,
                LedgerPaymentRow.voided_at.is_(None),
                LedgerPaymentRow.duplicate.is_(False),
            )
            .with_for_update()
        )
        if active is not None and payment.paid_at < _aware(active.paid_at):
            async with self.session.begin_nested():
                active.duplicate = True
                await self.session.flush()
            if await self._insert(payment):
                return payment
        flagged = payment.model_copy(update={"duplicate": True})
        return flagged if await self._insert(flagged) else None

    async def list_for_quote(
        self, business_id: BusinessId, quote_id: QuoteId
    ) -> tuple[LedgerPayment, ...]:
        rows = await self.session.scalars(
            select(LedgerPaymentRow)
            .where(
                LedgerPaymentRow.business_id == business_id,
                LedgerPaymentRow.quote_id == quote_id,
            )
            .order_by(LedgerPaymentRow.paid_at, LedgerPaymentRow.id)
        )
        return tuple(self._payment(row) for row in rows)

    async def void_manual(
        self, business_id: BusinessId, payment_id: UUID, *, by: str, at: datetime
    ) -> bool:
        voided = await self.session.scalar(
            update(LedgerPaymentRow)
            .where(
                LedgerPaymentRow.id == payment_id,
                LedgerPaymentRow.business_id == business_id,
                LedgerPaymentRow.source == PaymentSource.MANUAL.value,
                LedgerPaymentRow.voided_at.is_(None),
            )
            .values(voided_at=at, voided_by=by)
            .returning(LedgerPaymentRow.id)
            .execution_options(synchronize_session="fetch")
        )
        return voided is not None

    async def mark_duplicate(
        self, business_id: BusinessId, payment_id: UUID, duplicate: bool
    ) -> None:
        await self.session.execute(
            update(LedgerPaymentRow)
            .where(
                LedgerPaymentRow.id == payment_id,
                LedgerPaymentRow.business_id == business_id,
            )
            .values(duplicate=duplicate)
            .execution_options(synchronize_session="fetch")
        )

    async def list_for_business(
        self, business_id: BusinessId, *, since: datetime | None = None
    ) -> tuple[LedgerPayment, ...]:
        query = select(LedgerPaymentRow).where(LedgerPaymentRow.business_id == business_id)
        if since is not None:
            query = query.where(LedgerPaymentRow.paid_at >= since)
        rows = await self.session.scalars(
            query.order_by(LedgerPaymentRow.paid_at.desc(), LedgerPaymentRow.id)
        )
        return tuple(self._payment(row) for row in rows)


class SqlInvoiceRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    @staticmethod
    def _invoice(row: InvoiceRow) -> Invoice:
        return Invoice(
            invoice_id=row.id,
            business_id=BusinessId(row.business_id),
            quote_id=QuoteId(row.quote_id),
            sequence=row.sequence,
            number=row.number,
            customer_name=row.customer_name,
            customer_email=row.customer_email,
            customer_phone=row.customer_phone,
            service_address=row.service_address,
            line_items=tuple(InvoiceLine.model_validate(item) for item in row.line_items),
            total_minor=row.total_cents,
            currency=row.currency,
            sent_at=_aware(row.sent_at),
            due_on=row.due_on,
            sent_by=row.sent_by,
            created_at=_aware(row.created_at),
        )

    async def list_for_business(self, business_id: BusinessId) -> tuple[Invoice, ...]:
        rows = await self.session.scalars(
            select(InvoiceRow)
            .where(InvoiceRow.business_id == business_id)
            .order_by(InvoiceRow.sequence.desc())
        )
        return tuple(self._invoice(row) for row in rows)

    async def for_quote(self, business_id: BusinessId, quote_id: QuoteId) -> Invoice | None:
        row = await self.session.scalar(
            select(InvoiceRow).where(
                InvoiceRow.business_id == business_id, InvoiceRow.quote_id == quote_id
            )
        )
        return None if row is None else self._invoice(row)

    async def next_sequence(self, business_id: BusinessId) -> int:
        # Lock the business so two sends at once take consecutive numbers.
        await self.session.scalar(
            select(Business.id).where(Business.id == business_id).with_for_update()
        )
        highest = await self.session.scalar(
            select(func.max(InvoiceRow.sequence)).where(InvoiceRow.business_id == business_id)
        )
        return FIRST_INVOICE_SEQUENCE if highest is None else highest + 1

    async def create(self, invoice: Invoice) -> None:
        self.session.add(
            InvoiceRow(
                id=invoice.invoice_id,
                business_id=invoice.business_id,
                quote_id=invoice.quote_id,
                sequence=invoice.sequence,
                number=invoice.number,
                customer_name=invoice.customer_name,
                customer_email=invoice.customer_email,
                customer_phone=invoice.customer_phone,
                service_address=invoice.service_address,
                line_items=[item.model_dump() for item in invoice.line_items],
                total_cents=invoice.total_minor,
                currency=invoice.currency,
                sent_at=invoice.sent_at,
                due_on=invoice.due_on,
                sent_by=invoice.sent_by,
                created_at=invoice.created_at,
            )
        )
        try:
            async with self.session.begin_nested():
                await self.session.flush()
        except IntegrityError as error:
            raise InvoiceConflictError("the quote or number already has an invoice") from error
