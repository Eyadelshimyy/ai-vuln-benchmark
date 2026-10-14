"""Customer portal routes: magic-link sign-in and a signed-in customer's
quotes, subscriptions, billing portal and service requests.

The login route always answers 202 so an address cannot be probed for an
account; it is throttled per client ip and per address. Every other route
carries ``Authorization: Bearer <sessionToken>`` and fails with the same
generic 401 whether the token is unknown, expired or revoked. Responses are
projected here from domain records: no internal customer ids, no provider
ids, and nothing about any other customer or business.
"""

import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from gvas.application.intake import IntakeLimitError, IntakeService
from gvas.application.owner import OwnerAuthenticationError, OwnerService
from gvas.application.portal import (
    BillingPortalUnavailableError,
    PortalAuthenticationError,
    PortalContext,
    PortalService,
)
from gvas.domain.customers import SERVICE_REQUEST_MAX_CHARS
from gvas.domain.enums import CustomerQuoteStatus
from gvas.domain.payments import PaymentCheckoutError, QuoteSubscriptionRecord
from gvas.domain.quotes import Quote, normalize_customer_email, public_quote_id
from gvas.interfaces.http.public import (
    IntakeStartBody,
    PerIpRateLimiter,
    client_ip,
    intake_start_payload,
)

logger = logging.getLogger(__name__)

GENERIC_UNAUTHORIZED = "unauthorized"
LOGIN_RATE_LIMIT_PER_HOUR = 5


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    email: str = Field(min_length=3, max_length=320)


class SessionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    token: str = Field(min_length=1, max_length=512)


class ServiceRequestBody(BaseModel):
    model_config = ConfigDict(extra="ignore")

    message: str = Field(min_length=1, max_length=SERVICE_REQUEST_MAX_CHARS)
    preferredDates: str | None = Field(default=None, max_length=500)  # noqa: N815


def _iso(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()


def customer_payload(context: PortalContext, *, with_phone: bool) -> dict[str, object]:
    payload: dict[str, object] = {
        "displayName": context.customer.name,
        "email": context.customer.email,
    }
    if with_phone:
        payload["phone"] = context.customer.phone
    return payload


def business_payload(context: PortalContext, *, with_calendly: bool) -> dict[str, object]:
    business = context.business
    payload: dict[str, object] = {
        "displayName": business.display_name or business.name,
        "siteUrl": business.site_url,
    }
    if with_calendly:
        payload["calendlyUrl"] = business.calendly_url
    return payload


def quote_payload(quote: Quote) -> dict[str, object]:
    draft = quote.draft
    # Same identifier and status the public /v1/quotes/{claimToken} route shows,
    # so a site can join the two without a second lookup.
    status = (
        quote.customer_status.value
        if quote.customer_status is not None
        else CustomerQuoteStatus.VIEWED.value
    )
    return {
        "id": public_quote_id(quote.quote_id),
        "status": status,
        "totalCents": draft.total_minor if draft is not None else 0,
        "currency": draft.currency if draft is not None else None,
        "createdAt": _iso(quote.created_at),
        "approvedAt": _iso(quote.approved_at),
        "claimToken": quote.claim_token,
        "billing": (draft.billing.value if draft is not None else "one_time"),
        "interval": (
            draft.interval.value if draft is not None and draft.interval is not None else None
        ),
    }


def subscription_payload(record: QuoteSubscriptionRecord) -> dict[str, object]:
    return {
        "id": str(record.subscription_id),
        "quoteId": public_quote_id(record.quote_id),
        "status": record.status,
        "interval": record.interval.value,
        "amountCents": record.amount_minor,
        "currency": record.currency,
        "currentPeriodEnd": _iso(record.current_period_end),
        "cancelAtPeriodEnd": record.cancel_at_period_end,
        "manual": record.is_manual,
        "paidThrough": None if record.paid_through is None else record.paid_through.isoformat(),
    }


def bearer_token(request: Request) -> str:
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(status_code=401, detail=GENERIC_UNAUTHORIZED)
    return token.strip()


def create_portal_router(
    service: PortalService,
    *,
    rate_limiter: PerIpRateLimiter | None = None,
    login_rate_limiter: PerIpRateLimiter | None = None,
    intake: IntakeService | None = None,
    owner: OwnerService | None = None,
) -> APIRouter:
    router = APIRouter()
    limiter = rate_limiter or PerIpRateLimiter(per_minute=120, burst=30)
    login_limiter = login_rate_limiter or PerIpRateLimiter(
        per_minute=LOGIN_RATE_LIMIT_PER_HOUR / 60, burst=LOGIN_RATE_LIMIT_PER_HOUR
    )

    async def rate_limit(request: Request) -> None:
        if not limiter.allow(client_ip(request)):
            raise HTTPException(status_code=429, detail="rate limited")

    async def authenticated(request: Request) -> PortalContext:
        token = bearer_token(request)
        try:
            return await service.authenticate(token)
        except PortalAuthenticationError as error:
            raise HTTPException(status_code=401, detail=GENERIC_UNAUTHORIZED) from error

    limited = [Depends(rate_limit)]

    @router.post("/v1/businesses/{public_key}/portal/login", dependencies=limited, status_code=202)
    async def request_login(public_key: str, body: LoginRequest, request: Request) -> JSONResponse:
        address = normalize_customer_email(body.email)
        ip_allowed = login_limiter.allow(f"ip:{client_ip(request)}")
        email_allowed = login_limiter.allow(f"email:{public_key}:{address}")
        if not (ip_allowed and email_allowed):
            return JSONResponse({"detail": "rate limited"}, status_code=429)
        await service.request_login(public_key, address)
        return JSONResponse({}, status_code=202)

    @router.post("/v1/portal/sessions", dependencies=limited)
    async def create_session(body: SessionRequest) -> JSONResponse:
        # One sign-in link format serves both roles; an owner link opens an
        # owner session, which only the /v1/owner routes accept.
        if owner is not None:
            try:
                owner_session = await owner.exchange_login_token(body.token)
            except OwnerAuthenticationError:
                return JSONResponse({"detail": GENERIC_UNAUTHORIZED}, status_code=401)
            if owner_session is not None:
                owner_token, owner_context = owner_session
                return JSONResponse(
                    {
                        "sessionToken": owner_token,
                        "role": "owner",
                        "owner": {"email": owner_context.session.email},
                        "business": {
                            "displayName": owner_context.business.display_name
                            or owner_context.business.name,
                            "siteUrl": owner_context.business.site_url,
                        },
                    },
                    status_code=200,
                )
        try:
            session_token, context = await service.exchange_login_token(body.token)
        except PortalAuthenticationError:
            return JSONResponse({"detail": GENERIC_UNAUTHORIZED}, status_code=401)
        return JSONResponse(
            {
                "sessionToken": session_token,
                "role": "customer",
                "customer": customer_payload(context, with_phone=False),
                "business": business_payload(context, with_calendly=False),
            },
            status_code=200,
        )

    @router.delete("/v1/portal/sessions", dependencies=limited, status_code=204)
    async def revoke_session(request: Request) -> Response:
        try:
            await service.revoke_session(bearer_token(request))
        except PortalAuthenticationError:
            return JSONResponse({"detail": GENERIC_UNAUTHORIZED}, status_code=401)
        return Response(status_code=204)

    @router.get("/v1/portal/me", dependencies=limited)
    async def me(context: PortalContext = Depends(authenticated)) -> JSONResponse:  # noqa: B008
        return JSONResponse(
            {
                "customer": customer_payload(context, with_phone=True),
                "business": business_payload(context, with_calendly=True),
            }
        )

    @router.get("/v1/portal/quotes", dependencies=limited)
    async def quotes(context: PortalContext = Depends(authenticated)) -> JSONResponse:  # noqa: B008
        records = await service.quotes(context)
        return JSONResponse({"quotes": [quote_payload(quote) for quote in records]})

    @router.get("/v1/portal/subscriptions", dependencies=limited)
    async def subscriptions(
        context: PortalContext = Depends(authenticated),  # noqa: B008
    ) -> JSONResponse:
        records = await service.subscriptions(context)
        return JSONResponse({"subscriptions": [subscription_payload(record) for record in records]})

    @router.post("/v1/portal/billing-portal", dependencies=limited)
    async def billing_portal(
        context: PortalContext = Depends(authenticated),  # noqa: B008
    ) -> JSONResponse:
        try:
            url = await service.billing_portal_url(context)
        except BillingPortalUnavailableError:
            return JSONResponse({"detail": "not found"}, status_code=404)
        except PaymentCheckoutError:
            logger.warning("billing portal provider failed")
            return JSONResponse({"detail": "payment provider unavailable"}, status_code=502)
        return JSONResponse({"url": url})

    @router.post("/v1/portal/requests", dependencies=limited, status_code=202)
    async def submit_request(
        body: ServiceRequestBody,
        context: PortalContext = Depends(authenticated),  # noqa: B008
    ) -> JSONResponse:
        if not body.message.strip():
            return JSONResponse({"detail": "message is required"}, status_code=422)
        await service.submit_request(context, body.message, body.preferredDates)
        return JSONResponse({}, status_code=202)

    if intake is not None:

        @router.post("/v1/portal/intake/conversations", dependencies=limited, status_code=201)
        async def create_portal_intake_conversation(
            body: IntakeStartBody | None = None,
            context: PortalContext = Depends(authenticated),  # noqa: B008
        ) -> JSONResponse:
            consent = None if body is None else body.sms_consent
            try:
                start = await intake.start_portal_conversation(
                    context.business, context.customer, sms_consent=consent
                )
            except IntakeLimitError:
                return JSONResponse({"detail": "rate limited"}, status_code=429)
            return JSONResponse(intake_start_payload(start), status_code=201)

    return router


__all__ = ["LOGIN_RATE_LIMIT_PER_HOUR", "create_portal_router"]
