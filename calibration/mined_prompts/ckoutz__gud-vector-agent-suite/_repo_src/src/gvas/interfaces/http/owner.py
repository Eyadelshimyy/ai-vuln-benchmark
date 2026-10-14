"""Owner dashboard routes.

Every route carries ``Authorization: Bearer <sessionToken>`` from an owner
session (issued by ``POST /v1/portal/sessions`` for the business's owner
e-mail) and fails with the same generic 401 for any other credential —
including a customer portal session. All data is the session's business only.
The owner's calendar link is write-only: responses carry its host, never the
link.
"""

from collections.abc import Awaitable, Callable
from datetime import UTC, date, datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from gvas.application.owner import (
    PAYMENT_KEY_PATTERN,
    PLAN_AMOUNT_MAX_MINOR,
    CustomerSummary,
    OwnerAuthenticationError,
    OwnerConflictError,
    OwnerContext,
    OwnerInputError,
    OwnerNotFoundError,
    OwnerService,
    ServiceRequestView,
    SettingsUpdate,
)
from gvas.domain.enums import QuoteStatus
from gvas.domain.intake import IntakeConversation
from gvas.domain.invoices import Invoice, invoice_status
from gvas.domain.owner import CalendarEvent, calendar_feed_host
from gvas.domain.payments import (
    PLAN_MONTHS_MAX,
    LedgerPayment,
    PaymentMethod,
    QuoteSubscriptionRecord,
    first_counted,
    month_totals,
)
from gvas.domain.quotes import Quote, public_quote_id
from gvas.domain.repositories import BusinessRecord
from gvas.domain.time_zones import business_zone
from gvas.interfaces.http.portal import GENERIC_UNAUTHORIZED, bearer_token
from gvas.interfaces.http.public import PerIpRateLimiter, client_ip


class DeclineBookingBody(BaseModel):
    model_config = ConfigDict(extra="ignore")

    reason: str | None = Field(default=None, max_length=500)


class SettingsBody(BaseModel):
    model_config = ConfigDict(extra="ignore")

    displayName: str | None = Field(default=None, max_length=500)  # noqa: N815
    calendlyUrl: str | None = Field(default=None, max_length=2048)  # noqa: N815
    intakeBrief: str | None = Field(default=None, max_length=5000)  # noqa: N815
    intakeQuestions: str | None = Field(default=None, max_length=5000)  # noqa: N815
    intakeOpening: str | None = Field(default=None, max_length=5000)  # noqa: N815
    intakeOfferLine: str | None = Field(default=None, max_length=5000)  # noqa: N815
    intakeVisitQuestion: str | None = Field(default=None, max_length=5000)  # noqa: N815
    notificationEmail: str | None = Field(default=None, max_length=320)  # noqa: N815
    calendarFeedUrl: str | None = Field(default=None, max_length=4096)  # noqa: N815
    timezone: str | None = Field(default=None, max_length=64)


class MarkPaidBody(BaseModel):
    model_config = ConfigDict(extra="ignore")

    paidOn: date  # noqa: N815
    method: PaymentMethod
    note: str | None = Field(default=None, max_length=500)


class PlanPaymentBody(BaseModel):
    model_config = ConfigDict(extra="ignore")

    key: str = Field(pattern=PAYMENT_KEY_PATTERN)
    paidOn: date  # noqa: N815
    method: PaymentMethod
    months: int = Field(ge=1, le=PLAN_MONTHS_MAX)
    amountCents: int = Field(ge=0, le=PLAN_AMOUNT_MAX_MINOR)  # noqa: N815
    note: str | None = Field(default=None, max_length=500)


def _iso(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()


def _sent_at(quote: Quote) -> datetime | None:
    """When the quote first reached the customer by e-mail or text. A portal
    handoff that e-mailed nobody only gave the owner a link to forward."""

    receipt = quote.delivery_receipt
    emailed_at = None if receipt is None or receipt.emailed is False else receipt.occurred_at
    reached = [at for at in (emailed_at, quote.texted_at) if at is not None]
    return min(reached) if reached else None


def owner_quote_payload(quote: Quote, *, payment: LedgerPayment | None = None) -> dict[str, object]:
    draft = quote.draft
    recipient = draft.recipient if draft is not None else None
    return {
        "id": public_quote_id(quote.quote_id),
        "status": quote.status.value,
        "customerStatus": quote.customer_status.value if quote.customer_status else None,
        "needsApproval": quote.status is QuoteStatus.AWAITING_APPROVAL,
        "customer": {
            "name": recipient.display_name if recipient is not None else None,
            "email": recipient.address if recipient is not None else None,
            "phone": recipient.phone if recipient is not None else None,
            "serviceAddress": recipient.service_address if recipient is not None else None,
        },
        "lineItems": [
            {
                "description": item.description,
                "quantity": item.quantity,
                "unitPriceCents": item.unit_price_minor,
            }
            for item in (draft.line_items if draft is not None else ())
        ],
        "totalCents": draft.total_minor if draft is not None else 0,
        "currency": draft.currency if draft is not None else None,
        "billing": draft.billing.value if draft is not None else "one_time",
        "interval": (
            draft.interval.value if draft is not None and draft.interval is not None else None
        ),
        "note": draft.owner_note if draft is not None else None,
        "createdAt": _iso(quote.created_at),
        "approvedAt": _iso(quote.approved_at),
        "sentAt": _iso(_sent_at(quote)),
        "paidOn": _iso(payment.paid_at if payment is not None else None),
        "paidBy": (
            None
            if payment is None
            else {
                "source": payment.source.value,
                "method": payment.method.value,
                "note": payment.note,
            }
        ),
        "updatedAt": _iso(quote.updated_at),
    }


def owner_invoice_payload(
    invoice: Invoice, payment: LedgerPayment | None, *, today: date
) -> dict[str, object]:
    return {
        "number": invoice.number,
        "quoteId": public_quote_id(invoice.quote_id),
        "status": invoice_status(invoice, payment, today=today).value,
        "customer": {
            "name": invoice.customer_name,
            "email": invoice.customer_email,
            "phone": invoice.customer_phone,
            "serviceAddress": invoice.service_address,
        },
        "lineItems": [
            {
                "description": item.description,
                "quantity": item.quantity,
                "unitPriceCents": item.unit_price_minor,
            }
            for item in invoice.line_items
        ],
        "totalCents": invoice.total_minor,
        "currency": invoice.currency,
        "sentAt": _iso(invoice.sent_at),
        "dueOn": invoice.due_on.isoformat(),
        "paidOn": _iso(payment.paid_at if payment is not None and payment.counts else None),
        "paidBy": (
            None
            if payment is None or not payment.counts
            else {
                "source": payment.source.value,
                "method": payment.method.value,
                "note": payment.note,
            }
        ),
    }


def owner_customer_payload(summary: CustomerSummary) -> dict[str, object]:
    customer = summary.customer
    quotes = summary.quotes
    paid = [
        quote
        for quote in quotes
        if quote.customer_status is not None and quote.customer_status.value == "paid"
    ]
    return {
        "email": customer.email,
        "name": customer.display_name,
        "phone": customer.phone,
        "smsConsent": customer.sms_consent,
        "createdAt": _iso(customer.created_at),
        "quoteCount": len(quotes),
        "paidCents": sum(quote.draft.total_minor for quote in paid if quote.draft is not None),
        "lastQuoteAt": _iso(max((quote.created_at for quote in quotes), default=None)),
        "quoteIds": [public_quote_id(quote.quote_id) for quote in quotes],
    }


def owner_payment_payload(payment: LedgerPayment) -> dict[str, object]:
    return {
        "id": str(payment.payment_id),
        "quoteId": public_quote_id(payment.quote_id),
        "kind": payment.kind.value,
        "source": payment.source.value,
        "method": payment.method.value,
        "amountCents": payment.amount_minor,
        "currency": payment.currency,
        "paidOn": _iso(payment.paid_at),
        "monthsCovered": payment.months_covered,
        "recordedBy": payment.recorded_by,
        "recordedAt": _iso(payment.recorded_at),
        "note": payment.note,
        "voidedAt": _iso(payment.voided_at),
        "voidedBy": payment.voided_by,
        "duplicate": payment.duplicate,
        "counts": payment.counts,
    }


def owner_subscription_payload(record: QuoteSubscriptionRecord) -> dict[str, object]:
    return {
        "id": str(record.subscription_id),
        "quoteId": public_quote_id(record.quote_id),
        "status": record.status,
        "interval": record.interval.value,
        "amountCents": record.amount_minor,
        "currency": record.currency,
        "currentPeriodEnd": _iso(record.current_period_end),
        "cancelAtPeriodEnd": record.cancel_at_period_end,
        "createdAt": _iso(record.created_at),
        "manual": record.is_manual,
        "paidFrom": None if record.paid_from is None else record.paid_from.isoformat(),
        "paidThrough": None if record.paid_through is None else record.paid_through.isoformat(),
    }


def owner_booking_payload(conversation: IntakeConversation) -> dict[str, object]:
    collected = conversation.collected
    return {
        "reference": conversation.reference,
        "state": conversation.state.value,
        "needsDecision": conversation.state.value == "awaiting_owner",
        "customer": {
            "name": collected.name,
            "email": collected.email,
            "phone": collected.phone,
            "address": collected.address,
        },
        "details": collected.details,
        "notes": collected.notes,
        "urgency": collected.urgency,
        "requestedStart": _iso(conversation.requested_slot_start),
        "requestedEnd": _iso(conversation.requested_slot_end),
        "booked": conversation.booked_event_uri is not None,
        "decisionAt": _iso(conversation.decision_at),
        "decisionReason": conversation.decision_reason,
        "createdAt": _iso(conversation.created_at),
        "updatedAt": _iso(conversation.updated_at),
    }


def owner_request_payload(view: ServiceRequestView) -> dict[str, object]:
    request = view.request
    customer = view.customer
    return {
        "id": str(request.request_id),
        "message": request.message,
        "preferredDates": request.preferred_dates,
        "status": request.status,
        "source": request.source,
        "createdAt": _iso(request.created_at),
        "customer": {
            "name": customer.display_name if customer is not None else None,
            "email": customer.email if customer is not None else None,
        },
    }


def calendar_event_payload(event: CalendarEvent) -> dict[str, object]:
    return {
        "source": event.source.value,
        "title": event.title,
        "start": _iso(event.start),
        "end": _iso(event.end),
        "allDay": event.all_day,
        "location": event.location,
        "inviteeName": event.invitee_name,
        "inviteeEmail": event.invitee_email,
        "reference": event.reference,
    }


def settings_payload(business: BusinessRecord) -> dict[str, object]:
    profile = business.intake_profile
    return {
        "displayName": business.display_name or business.name,
        "siteUrl": business.site_url,
        "calendlyUrl": business.calendly_url,
        "intakeBrief": profile.brief,
        "intakeQuestions": profile.questions,
        "intakeOpening": profile.opening,
        "intakeOfferLine": profile.offer_line,
        "intakeVisitQuestion": profile.visit_question,
        "notificationEmail": business.notification_email,
        "ownerEmail": business.owner_email,
        "timezone": business.timezone,
        "calendarFeed": {
            "connected": bool(business.calendar_feed_url),
            "host": calendar_feed_host(business.calendar_feed_url),
        },
        "connections": {"website": bool(business.site_url and business.public_key)},
    }


def create_owner_router(
    service: OwnerService,
    *,
    rate_limiter: PerIpRateLimiter | None = None,
    on_activity: Callable[[UUID], Awaitable[None]] | None = None,
) -> APIRouter:
    router = APIRouter()
    limiter = rate_limiter or PerIpRateLimiter(per_minute=120, burst=30)

    async def rate_limit(request: Request) -> None:
        if not limiter.allow(client_ip(request)):
            raise HTTPException(status_code=429, detail="rate limited")

    async def authenticated(request: Request) -> OwnerContext:
        token = bearer_token(request)
        try:
            context = await service.authenticate(token)
        except OwnerAuthenticationError as error:
            raise HTTPException(status_code=401, detail=GENERIC_UNAUTHORIZED) from error
        if on_activity is not None:
            await on_activity(context.business.business_id)
        return context

    limited = [Depends(rate_limit)]
    owner = Depends(authenticated)

    @router.get("/v1/owner/me", dependencies=limited)
    async def me(context: OwnerContext = owner) -> JSONResponse:
        return JSONResponse(
            {
                "role": "owner",
                "owner": {"email": context.session.email},
                "business": {
                    "displayName": context.business.display_name or context.business.name,
                    "siteUrl": context.business.site_url,
                    "calendlyUrl": context.business.calendly_url,
                    "timezone": context.business.timezone,
                    "invoices": bool(context.business.invoice_prefix),
                },
            }
        )

    @router.delete("/v1/owner/sessions", dependencies=limited, status_code=204)
    async def revoke_session(request: Request, _context: OwnerContext = owner) -> Response:
        await service.revoke_session(bearer_token(request))
        return Response(status_code=204)

    @router.get("/v1/owner/quotes", dependencies=limited)
    async def quotes(context: OwnerContext = owner) -> JSONResponse:
        records = await service.quotes(context)
        paid = first_counted(await service.payments(context))
        return JSONResponse(
            {
                "quotes": [
                    owner_quote_payload(quote, payment=paid.get(quote.quote_id))
                    for quote in records
                ]
            }
        )

    @router.get("/v1/owner/payments", dependencies=limited)
    async def payments(context: OwnerContext = owner) -> JSONResponse:
        records = await service.payments(context)
        zone = business_zone(context.business.timezone) or UTC
        now = datetime.now(UTC)
        return JSONResponse(
            {
                "payments": [owner_payment_payload(payment) for payment in records],
                "month": now.astimezone(zone).strftime("%Y-%m"),
                "paidThisMonth": month_totals(records, zone, now),
            }
        )

    @router.post("/v1/owner/quotes/{quote_id}/approve", dependencies=limited)
    async def approve_quote(quote_id: str, context: OwnerContext = owner) -> JSONResponse:
        return await _decide_quote(context, quote_id, approve=True)

    @router.post("/v1/owner/quotes/{quote_id}/reject", dependencies=limited)
    async def reject_quote(quote_id: str, context: OwnerContext = owner) -> JSONResponse:
        return await _decide_quote(context, quote_id, approve=False)

    async def _decide_quote(context: OwnerContext, quote_id: str, *, approve: bool) -> JSONResponse:
        try:
            quote = await service.decide_quote(context, quote_id, approve=approve)
        except OwnerNotFoundError:
            return JSONResponse({"detail": "not found"}, status_code=404)
        except OwnerConflictError as error:
            return JSONResponse({"detail": str(error)}, status_code=409)
        return JSONResponse({"quote": owner_quote_payload(quote)})

    @router.get("/v1/owner/invoices", dependencies=limited)
    async def invoices(context: OwnerContext = owner) -> JSONResponse:
        records = await service.invoices(context)
        paid = first_counted(await service.payments(context)) if records else {}
        zone = business_zone(context.business.timezone) or UTC
        today = datetime.now(UTC).astimezone(zone).date()
        return JSONResponse(
            {
                "enabled": bool(context.business.invoice_prefix),
                "invoices": [
                    owner_invoice_payload(invoice, paid.get(invoice.quote_id), today=today)
                    for invoice in records
                ],
            }
        )

    @router.post("/v1/owner/quotes/{quote_id}/send-invoice", dependencies=limited)
    async def send_invoice(quote_id: str, context: OwnerContext = owner) -> JSONResponse:
        try:
            invoice = await service.send_invoice(context, quote_id)
        except OwnerNotFoundError:
            return JSONResponse({"detail": "not found"}, status_code=404)
        except OwnerConflictError as error:
            return JSONResponse({"detail": str(error)}, status_code=409)
        paid = first_counted(await service.payments(context)).get(invoice.quote_id)
        zone = business_zone(context.business.timezone) or UTC
        today = datetime.now(UTC).astimezone(zone).date()
        return JSONResponse({"invoice": owner_invoice_payload(invoice, paid, today=today)})

    @router.post("/v1/owner/quotes/{quote_id}/mark-paid", dependencies=limited)
    async def mark_quote_paid(
        quote_id: str, body: MarkPaidBody, context: OwnerContext = owner
    ) -> JSONResponse:
        try:
            quote = await service.mark_paid(
                context, quote_id, paid_on=body.paidOn, method=body.method, note=body.note
            )
        except OwnerNotFoundError:
            return JSONResponse({"detail": "not found"}, status_code=404)
        except OwnerConflictError as error:
            return JSONResponse({"detail": str(error)}, status_code=409)
        except OwnerInputError as error:
            return JSONResponse({"detail": str(error)}, status_code=422)
        return await _paid_quote(context, quote)

    @router.post("/v1/owner/quotes/{quote_id}/mark-unpaid", dependencies=limited)
    async def mark_quote_unpaid(quote_id: str, context: OwnerContext = owner) -> JSONResponse:
        try:
            quote = await service.mark_unpaid(context, quote_id)
        except OwnerNotFoundError:
            return JSONResponse({"detail": "not found"}, status_code=404)
        except OwnerConflictError as error:
            return JSONResponse({"detail": str(error)}, status_code=409)
        return await _paid_quote(context, quote)

    @router.post("/v1/owner/quotes/{quote_id}/plan-payments", dependencies=limited)
    async def record_plan_payment(
        quote_id: str, body: PlanPaymentBody, context: OwnerContext = owner
    ) -> JSONResponse:
        try:
            plan = await service.record_plan_payment(
                context,
                quote_id,
                key=body.key,
                paid_on=body.paidOn,
                method=body.method,
                months=body.months,
                amount_minor=body.amountCents,
                note=body.note,
            )
        except OwnerNotFoundError:
            return JSONResponse({"detail": "not found"}, status_code=404)
        except OwnerConflictError as error:
            return JSONResponse({"detail": str(error)}, status_code=409)
        except OwnerInputError as error:
            return JSONResponse({"detail": str(error)}, status_code=422)
        return JSONResponse({"subscription": owner_subscription_payload(plan)})

    @router.post(
        "/v1/owner/quotes/{quote_id}/plan-payments/{payment_id}/undo", dependencies=limited
    )
    async def undo_plan_payment(
        quote_id: str, payment_id: str, context: OwnerContext = owner
    ) -> JSONResponse:
        try:
            plan = await service.void_plan_payment(context, quote_id, payment_id)
        except OwnerNotFoundError:
            return JSONResponse({"detail": "not found"}, status_code=404)
        except OwnerConflictError as error:
            return JSONResponse({"detail": str(error)}, status_code=409)
        return JSONResponse({"subscription": owner_subscription_payload(plan)})

    async def _paid_quote(context: OwnerContext, quote: Quote) -> JSONResponse:
        paid = first_counted(await service.payments(context))
        return JSONResponse({"quote": owner_quote_payload(quote, payment=paid.get(quote.quote_id))})

    @router.get("/v1/owner/customers", dependencies=limited)
    async def customers(context: OwnerContext = owner) -> JSONResponse:
        records = await service.customers(context)
        return JSONResponse({"customers": [owner_customer_payload(item) for item in records]})

    @router.get("/v1/owner/subscriptions", dependencies=limited)
    async def subscriptions(context: OwnerContext = owner) -> JSONResponse:
        records = await service.subscriptions(context)
        return JSONResponse(
            {"subscriptions": [owner_subscription_payload(record) for record in records]}
        )

    @router.get("/v1/owner/bookings", dependencies=limited)
    async def bookings(context: OwnerContext = owner) -> JSONResponse:
        records = await service.bookings(context)
        return JSONResponse({"bookings": [owner_booking_payload(item) for item in records]})

    @router.post("/v1/owner/bookings/{reference}/approve", dependencies=limited)
    async def approve_booking(reference: str, context: OwnerContext = owner) -> JSONResponse:
        outcome = await service.decide_booking(context, reference, approve=True)
        return JSONResponse({"applied": outcome.applied, "message": outcome.text})

    @router.post("/v1/owner/bookings/{reference}/decline", dependencies=limited)
    async def decline_booking(
        reference: str, body: DeclineBookingBody, context: OwnerContext = owner
    ) -> JSONResponse:
        outcome = await service.decide_booking(
            context, reference, approve=False, reason=body.reason
        )
        return JSONResponse({"applied": outcome.applied, "message": outcome.text})

    @router.get("/v1/owner/requests", dependencies=limited)
    async def service_requests(context: OwnerContext = owner) -> JSONResponse:
        records = await service.service_requests(context)
        return JSONResponse({"requests": [owner_request_payload(item) for item in records]})

    @router.get("/v1/owner/calendar", dependencies=limited)
    async def calendar(
        start: datetime = Query(...),  # noqa: B008
        end: datetime = Query(...),  # noqa: B008
        context: OwnerContext = owner,
    ) -> JSONResponse:
        try:
            view = await service.calendar(context, start, end)
        except OwnerInputError as error:
            return JSONResponse({"detail": str(error)}, status_code=422)
        return JSONResponse(
            {
                "events": [calendar_event_payload(event) for event in view.events],
                "problems": list(view.problems),
            }
        )

    @router.get("/v1/owner/settings", dependencies=limited)
    async def get_settings(context: OwnerContext = owner) -> JSONResponse:
        return JSONResponse({"settings": settings_payload(context.business)})

    @router.patch("/v1/owner/settings", dependencies=limited)
    async def update_settings(body: SettingsBody, context: OwnerContext = owner) -> JSONResponse:
        try:
            record = await service.update_settings(
                context,
                SettingsUpdate(
                    display_name=body.displayName,
                    calendly_url=body.calendlyUrl,
                    intake_brief=body.intakeBrief,
                    intake_questions=body.intakeQuestions,
                    intake_opening=body.intakeOpening,
                    intake_offer_line=body.intakeOfferLine,
                    intake_visit_question=body.intakeVisitQuestion,
                    notification_email=body.notificationEmail,
                    calendar_feed_url=body.calendarFeedUrl,
                    timezone=body.timezone,
                ),
            )
        except OwnerInputError as error:
            return JSONResponse({"detail": str(error)}, status_code=422)
        return JSONResponse({"settings": settings_payload(record)})

    return router


__all__ = ["create_owner_router"]
