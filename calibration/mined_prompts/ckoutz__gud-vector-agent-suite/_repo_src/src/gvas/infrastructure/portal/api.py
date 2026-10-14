"""Customer portal adapter for approved quotes.

Only this module talks HTTP to the portal (``POST /api/quotes``). The portal
renders the quote and, when asked, emails the customer its link; it never
texts for bearer-token callers, so texting stays with the Telnyx adapter. The
returned claim token and link are kept in ``portal_quote_handoffs`` under the
delivery's idempotency key: a replayed command finds them there and does not
create a second portal quote. The token, request bodies and portal responses
never leave this module; errors carry the status code only.
"""

import logging
from datetime import UTC, datetime
from typing import Final, Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from gvas.domain.enums import DeliveryStatus
from gvas.domain.messages import CustomerDeliveryRequest, DeliveryReceipt
from gvas.infrastructure.portal.config import PortalSettings
from gvas.infrastructure.portal.models import PortalQuoteHandoff

logger = logging.getLogger(__name__)

QUOTES_PATH: Final = "/api/quotes"
PORTAL_CURRENCY: Final = "USD"


class PortalDeliveryError(RuntimeError):
    """Raised when a portal handoff attempt should be retried by the dispatcher."""


class PortalModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class PortalHandoff(PortalModel):
    portal_quote_id: str
    claim_token: str
    quote_url: str
    emailed: bool
    created_at: datetime


class PortalHandoffLedger(Protocol):
    """At-least-once: a recorded handoff suppresses a second create, a crash
    before ``record`` does not."""

    async def find(self, idempotency_key: str) -> PortalHandoff | None: ...

    async def record(self, idempotency_key: str, handoff: PortalHandoff) -> None: ...


class InMemoryPortalHandoffLedger:
    def __init__(self) -> None:
        self._handoffs: dict[str, PortalHandoff] = {}

    async def find(self, idempotency_key: str) -> PortalHandoff | None:
        return self._handoffs.get(idempotency_key)

    async def record(self, idempotency_key: str, handoff: PortalHandoff) -> None:
        self._handoffs.setdefault(idempotency_key, handoff)


class SqlPortalHandoffLedger:
    """Records handoffs in their own transaction so a caller rollback cannot lose one."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def find(self, idempotency_key: str) -> PortalHandoff | None:
        async with self._session_factory() as session:
            row = await session.get(PortalQuoteHandoff, idempotency_key)
            if row is None:
                return None
            return _handoff_of(row)

    async def record(self, idempotency_key: str, handoff: PortalHandoff) -> None:
        async with self._session_factory() as session:
            session.add(
                PortalQuoteHandoff(
                    idempotency_key=idempotency_key,
                    portal_quote_id=handoff.portal_quote_id,
                    claim_token=handoff.claim_token,
                    quote_url=handoff.quote_url,
                    emailed=handoff.emailed,
                    created_at=handoff.created_at,
                )
            )
            try:
                await session.commit()
            except IntegrityError:
                await session.rollback()
                existing = await session.scalar(
                    select(PortalQuoteHandoff).where(
                        PortalQuoteHandoff.idempotency_key == idempotency_key
                    )
                )
                if existing is None:
                    raise


def _handoff_of(row: PortalQuoteHandoff) -> PortalHandoff:
    created_at = row.created_at
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=UTC)
    return PortalHandoff(
        portal_quote_id=row.portal_quote_id,
        claim_token=row.claim_token,
        quote_url=row.quote_url,
        emailed=row.emailed,
        created_at=created_at,
    )


class PortalQuoteResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    ok: bool
    id: str
    claim_token: str = Field(alias="claimToken")
    quote_url: str = Field(alias="quoteUrl")
    emailed: bool = False


def _utc_now() -> datetime:
    return datetime.now(tz=UTC)


class PortalQuoteDelivery:
    """Implements ``CustomerQuoteDeliveryPort`` against the portal's quotes API."""

    def __init__(
        self,
        settings: PortalSettings,
        client: httpx.AsyncClient,
        ledger: PortalHandoffLedger,
    ) -> None:
        if not settings.is_configured:
            raise PortalDeliveryError("portal base url and api token are not configured")
        self._settings = settings
        self._client = client
        self._ledger = ledger

    async def deliver(self, request: CustomerDeliveryRequest) -> DeliveryReceipt:
        recorded = await self._ledger.find(request.idempotency_key)
        if recorded is not None:
            return _receipt(recorded)
        payload = portal_payload(request)
        try:
            response = await self._client.post(
                f"{self._settings.base_url.rstrip('/')}{QUOTES_PATH}",
                json=payload,
                headers={
                    "Authorization": f"Bearer {self._settings.api_token}",
                    "Content-Type": "application/json",
                },
                timeout=self._settings.timeout_seconds,
            )
        except httpx.HTTPError as error:
            logger.warning("portal request failed: %s", type(error).__name__)
            raise PortalDeliveryError("portal was unreachable") from error
        if response.status_code >= 400:
            logger.warning("portal returned http %s", response.status_code)
            raise PortalDeliveryError(f"portal returned http {response.status_code}")
        try:
            parsed = PortalQuoteResponse.model_validate(response.json())
        except (ValueError, ValidationError) as error:
            logger.warning("portal returned an unreadable response: %s", type(error).__name__)
            raise PortalDeliveryError("portal returned an unreadable response") from error
        if not parsed.ok:
            raise PortalDeliveryError("portal did not accept the quote")
        handoff = PortalHandoff(
            portal_quote_id=parsed.id,
            claim_token=parsed.claim_token,
            quote_url=parsed.quote_url,
            emailed=parsed.emailed,
            created_at=_utc_now(),
        )
        await self._ledger.record(request.idempotency_key, handoff)
        return _receipt(handoff)


def portal_payload(request: CustomerDeliveryRequest) -> dict[str, object]:
    """The portal's create body. Amounts are integer cents in USD; the request
    must carry structured items, and the customer needs an email or a phone."""

    if not request.line_items:
        raise PortalDeliveryError("portal quotes need structured line items")
    if (request.currency or PORTAL_CURRENCY).upper() != PORTAL_CURRENCY:
        raise PortalDeliveryError("portal accepts USD quotes only")
    recipient = request.recipient
    email = recipient.email_address
    phone = recipient.phone_number
    if email is None and phone is None:
        raise PortalDeliveryError("portal quotes need a customer email or phone")
    payload: dict[str, object] = {
        "customerName": recipient.display_name or email or phone,
        "items": [
            {
                "description": item.description,
                "quantity": item.quantity,
                "amountCents": item.unit_price_minor,
            }
            for item in request.line_items
        ],
        "billing": "one_time",
        "sendEmail": email is not None,
    }
    if email is not None:
        payload["customerEmail"] = email
    if phone is not None:
        payload["customerPhone"] = phone
    if recipient.service_address:
        payload["serviceAddress"] = recipient.service_address
    if request.note:
        payload["note"] = request.note
    return payload


def _receipt(handoff: PortalHandoff) -> DeliveryReceipt:
    return DeliveryReceipt(
        status=DeliveryStatus.ACCEPTED,
        provider_message_id=handoff.portal_quote_id,
        occurred_at=handoff.created_at,
        customer_link=handoff.quote_url,
        emailed=handoff.emailed,
    )
