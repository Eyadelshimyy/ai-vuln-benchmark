"""Invoices: what the owner bills once the work is done, kept apart from the
quote (the estimate). The small version: an invoice is an exact copy of an
accepted one-off quote's lines and total, numbered per business (LL-1042)
and due 14 days after it is sent. It is paid when its quote has a counted
payment in the ledger, so "Paid this month" stays the one source of truth.
Only businesses with an invoice prefix have invoices; sending one only
records it (no e-mail yet)."""

from datetime import date, datetime, timedelta, tzinfo
from enum import StrEnum
from typing import Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from gvas.domain.identifiers import BusinessId, QuoteId
from gvas.domain.payments import LedgerPayment

INVOICE_DUE_DAYS = 14
INVOICE_PREFIX_MAX_CHARS = 10
FIRST_INVOICE_SEQUENCE = 1001


class InvoiceStatus(StrEnum):
    SENT = "sent"
    PAID = "paid"
    OVERDUE = "overdue"


class InvoiceLine(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    description: str
    quantity: int
    unit_price_minor: int


class Invoice(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    invoice_id: UUID
    business_id: BusinessId
    quote_id: QuoteId
    sequence: int = Field(ge=1)
    number: str
    customer_name: str | None = None
    customer_email: str | None = None
    customer_phone: str | None = None
    service_address: str | None = None
    line_items: tuple[InvoiceLine, ...]
    total_minor: int = Field(ge=0)
    currency: str
    sent_at: datetime
    due_on: date
    sent_by: str | None = None
    created_at: datetime


class InvoiceConflictError(RuntimeError):
    """The quote already has an invoice, or the number was taken."""


def invoice_number(prefix: str, sequence: int) -> str:
    return f"{prefix}-{sequence}"


def due_on(sent_at: datetime, zone: tzinfo) -> date:
    return sent_at.astimezone(zone).date() + timedelta(days=INVOICE_DUE_DAYS)


def invoice_status(
    invoice: Invoice, payment: LedgerPayment | None, *, today: date
) -> InvoiceStatus:
    """Paid once its quote has a counted payment; overdue the day after it
    was due; sent until then."""

    if payment is not None and payment.counts:
        return InvoiceStatus.PAID
    return InvoiceStatus.OVERDUE if today > invoice.due_on else InvoiceStatus.SENT


class InvoiceRepository(Protocol):
    async def list_for_business(self, business_id: BusinessId) -> tuple[Invoice, ...]:
        """Newest first by number."""
        ...

    async def for_quote(self, business_id: BusinessId, quote_id: QuoteId) -> Invoice | None: ...

    async def next_sequence(self, business_id: BusinessId) -> int: ...

    async def create(self, invoice: Invoice) -> None:
        """Raises ``InvoiceConflictError`` when the quote or number is taken."""
        ...
