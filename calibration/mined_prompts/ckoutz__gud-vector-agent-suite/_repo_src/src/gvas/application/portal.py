"""The customer portal: magic-link sign-in and a customer's own view of one
business.

Everything here is scoped twice — by the business a session was issued for
and by the customer inside it. A session is never looked up without both
predicates, so a customer of business A can neither see business B nor
another customer of A.

``request_login`` deliberately returns nothing: whether the address is known
is decided inside and only shows up as an e-mail (or its absence). It is
also the place older quotes are linked: any claimable quote addressed to the
e-mail that predates the customers table becomes theirs on first sign-in.
"""

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from uuid import uuid4

from gvas.domain.customer_linking import enqueue_quote_owner_notice
from gvas.domain.customers import (
    LOGIN_TOKEN_TTL,
    PORTAL_SESSION_TTL,
    CustomerRecord,
    PortalLoginEmailRequest,
    PortalLoginToken,
    PortalSession,
    ServiceRequest,
    hash_portal_token,
    new_portal_token,
    portal_login_email_command,
    portal_login_url,
    portal_token_matches,
)
from gvas.domain.identifiers import ServiceRequestId
from gvas.domain.owner_actions import issue_owner_login
from gvas.domain.payments import BillingPortalRequest, QuoteSubscriptionRecord
from gvas.domain.ports import BillingAccountPort
from gvas.domain.quotes import Quote, QuoteConcurrencyError, normalize_customer_email
from gvas.domain.repositories import BusinessRecord, UnitOfWork

logger = logging.getLogger(__name__)


class PortalAuthenticationError(ValueError):
    """Bad, expired, used or revoked token; shown as a generic 401."""


class BillingPortalUnavailableError(LookupError):
    """The customer has no provider-side billing account yet; shown as 404."""


class PortalContext:
    """The authenticated caller: the session, its customer and business."""

    def __init__(
        self, session: PortalSession, customer: CustomerRecord, business: BusinessRecord
    ) -> None:
        self.session = session
        self.customer = customer
        self.business = business


class PortalService:
    def __init__(
        self,
        unit_of_work_factory: Callable[[], UnitOfWork],
        billing_accounts: BillingAccountPort | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._unit_of_work_factory = unit_of_work_factory
        self._billing_accounts = billing_accounts
        self._now = now or (lambda: datetime.now(UTC))

    # -- magic link ---------------------------------------------------------

    async def request_login(self, public_key: str, email: str) -> None:
        """Issue a login link when — and only when — the address belongs to a
        customer of this business who holds at least one quote. Every other
        case returns silently."""

        address = normalize_customer_email(email)
        if not address or "@" not in address:
            return
        now = self._now()
        async with self._unit_of_work_factory() as unit_of_work:
            business = await unit_of_work.businesses.get_by_public_key(public_key)
            if business is None or not business.site_url:
                await unit_of_work.commit()
                return
            if business.owner_email and normalize_customer_email(business.owner_email) == address:
                # The owner signs in on the same page; their link opens the
                # owner dashboard. The 202 answer is identical either way.
                await issue_owner_login(unit_of_work, business, address, now)
                await unit_of_work.commit()
                return
            customer = await unit_of_work.customers.find_by_email(business.business_id, address)
            unlinked = await unit_of_work.quotes.list_unlinked_for_email(
                business.business_id, address
            )
            if customer is None and not unlinked:
                await unit_of_work.commit()
                return
            if customer is None:
                recipient = unlinked[0].draft.recipient if unlinked[0].draft is not None else None
                customer = await unit_of_work.customers.upsert(
                    business.business_id,
                    address,
                    display_name=recipient.display_name if recipient is not None else None,
                    phone=recipient.phone_number if recipient is not None else None,
                    now=now,
                )
            await _link_quotes(unit_of_work, unlinked, customer, now)
            quotes = await unit_of_work.quotes.list_for_customer(
                business.business_id, customer.customer_id
            )
            if not quotes:
                await unit_of_work.commit()
                return
            raw = new_portal_token()
            token_hash = hash_portal_token(raw)
            await unit_of_work.portal_login_tokens.add(
                PortalLoginToken(
                    token_hash=token_hash,
                    business_id=business.business_id,
                    customer_id=customer.customer_id,
                    expires_at=now + LOGIN_TOKEN_TTL,
                    created_at=now,
                )
            )
            request = PortalLoginEmailRequest(
                business_id=business.business_id,
                to=customer.email,
                business_display_name=business.display_name or business.name,
                login_url=portal_login_url(business.site_url, raw),
                idempotency_key=f"portal-login:{token_hash}",
                expires_at=now + LOGIN_TOKEN_TTL,
            )
            await unit_of_work.outbox.enqueue(
                portal_login_email_command(request, token_hash=token_hash)
            )
            await unit_of_work.commit()

    async def exchange_login_token(self, raw_token: str) -> tuple[str, PortalContext]:
        """Consume a magic link and mint a session; returns the raw session
        token (shown once) with the caller's context."""

        if not raw_token:
            raise PortalAuthenticationError("missing token")
        now = self._now()
        token_hash = hash_portal_token(raw_token)
        async with self._unit_of_work_factory() as unit_of_work:
            token = await unit_of_work.portal_login_tokens.find_by_hash(token_hash)
            if (
                token is None
                or not portal_token_matches(raw_token, token.token_hash)
                or not token.is_usable(now)
            ):
                await unit_of_work.commit()
                raise PortalAuthenticationError("invalid token")
            if not await unit_of_work.portal_login_tokens.mark_used(token_hash, now):
                await unit_of_work.commit()
                raise PortalAuthenticationError("token already used")
            customer = await unit_of_work.customers.get(token.business_id, token.customer_id)
            business = await unit_of_work.businesses.get(token.business_id)
            if customer is None or business is None:
                await unit_of_work.commit()
                raise PortalAuthenticationError("token has no account")
            raw_session = new_portal_token()
            session = PortalSession(
                token_hash=hash_portal_token(raw_session),
                business_id=business.business_id,
                customer_id=customer.customer_id,
                expires_at=now + PORTAL_SESSION_TTL,
                created_at=now,
            )
            await unit_of_work.portal_sessions.add(session)
            await unit_of_work.commit()
        return raw_session, PortalContext(session, customer, business)

    async def authenticate(self, raw_session_token: str) -> PortalContext:
        if not raw_session_token:
            raise PortalAuthenticationError("missing session")
        now = self._now()
        token_hash = hash_portal_token(raw_session_token)
        async with self._unit_of_work_factory() as unit_of_work:
            session = await unit_of_work.portal_sessions.find_by_hash(token_hash)
            if (
                session is None
                or not portal_token_matches(raw_session_token, session.token_hash)
                or not session.is_active(now)
            ):
                await unit_of_work.commit()
                raise PortalAuthenticationError("invalid session")
            customer = await unit_of_work.customers.get(session.business_id, session.customer_id)
            business = await unit_of_work.businesses.get(session.business_id)
            await unit_of_work.commit()
        if customer is None or business is None:
            raise PortalAuthenticationError("session has no account")
        return PortalContext(session, customer, business)

    async def revoke_session(self, raw_session_token: str) -> None:
        """Revoke a live session; a credential that would not authenticate
        is refused the same way every other portal route refuses it."""

        if not raw_session_token:
            raise PortalAuthenticationError("missing session")
        now = self._now()
        token_hash = hash_portal_token(raw_session_token)
        async with self._unit_of_work_factory() as unit_of_work:
            session = await unit_of_work.portal_sessions.find_by_hash(token_hash)
            if (
                session is None
                or not portal_token_matches(raw_session_token, session.token_hash)
                or not session.is_active(now)
            ):
                await unit_of_work.commit()
                raise PortalAuthenticationError("invalid session")
            revoked = await unit_of_work.portal_sessions.revoke(token_hash, now)
            await unit_of_work.commit()
        if not revoked:
            raise PortalAuthenticationError("session already revoked")

    # -- the customer's view ------------------------------------------------

    async def quotes(self, context: PortalContext) -> tuple[Quote, ...]:
        async with self._unit_of_work_factory() as unit_of_work:
            quotes = await unit_of_work.quotes.list_for_customer(
                context.business.business_id, context.customer.customer_id
            )
            await unit_of_work.commit()
        return tuple(sorted(quotes, key=lambda quote: quote.created_at, reverse=True))

    async def subscriptions(self, context: PortalContext) -> tuple[QuoteSubscriptionRecord, ...]:
        async with self._unit_of_work_factory() as unit_of_work:
            records = await unit_of_work.quote_subscriptions.list_for_customer(
                context.business.business_id, context.customer.customer_id
            )
            await unit_of_work.commit()
        return tuple(sorted(records, key=lambda record: record.created_at, reverse=True))

    async def billing_portal_url(self, context: PortalContext) -> str:
        customer_ref = context.customer.stripe_customer_id
        if customer_ref is None or self._billing_accounts is None:
            raise BillingPortalUnavailableError("no billing account")
        site_url = (context.business.site_url or "").rstrip("/")
        result = await self._billing_accounts.create_billing_portal_session(
            BillingPortalRequest(
                business_id=context.business.business_id,
                customer_ref=customer_ref,
                return_url=f"{site_url}/portal",
            )
        )
        return result.url

    async def submit_request(
        self, context: PortalContext, message: str, preferred_dates: str | None
    ) -> ServiceRequest:
        """Store the enquiry and tell the owner where they already talk to
        GVAS: the notice rides the conversation of the customer's latest quote,
        which is the channel that produced this customer in the first place."""

        now = self._now()
        request = ServiceRequest(
            request_id=ServiceRequestId(uuid4()),
            business_id=context.business.business_id,
            customer_id=context.customer.customer_id,
            message=message.strip(),
            preferred_dates=(preferred_dates or "").strip() or None,
            created_at=now,
        )
        customer = context.customer
        text = f"New service request from {customer.name} ({customer.email}): {request.message}"
        if request.preferred_dates:
            text += f"\nPreferred dates: {request.preferred_dates}"
        async with self._unit_of_work_factory() as unit_of_work:
            await unit_of_work.service_requests.add(request)
            quotes = await unit_of_work.quotes.list_for_customer(
                context.business.business_id, customer.customer_id
            )
            notified = False
            for quote in sorted(quotes, key=lambda quote: quote.created_at, reverse=True):
                notified = await enqueue_quote_owner_notice(
                    unit_of_work,
                    quote,
                    correlation_id=f"service_request:{request.request_id}",
                    text=text,
                )
                if notified:
                    break
            if not notified:
                logger.warning("service request stored without an owner conversation to notify")
            await unit_of_work.commit()
        return request


async def _link_quotes(
    unit_of_work: UnitOfWork,
    quotes: tuple[Quote, ...],
    customer: CustomerRecord,
    now: datetime,
) -> None:
    for quote in quotes:
        linked = quote.link_customer(customer.customer_id, now)
        if linked is quote:
            continue
        try:
            await unit_of_work.quotes.save(linked, expected_version=quote.version)
        except QuoteConcurrencyError:
            logger.info("quote %s changed while linking; it links on its next step", quote.quote_id)


__all__ = [
    "BillingPortalUnavailableError",
    "PortalAuthenticationError",
    "PortalContext",
    "PortalService",
]
