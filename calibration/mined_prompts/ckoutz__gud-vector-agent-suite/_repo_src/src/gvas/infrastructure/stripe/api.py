"""Stripe adapter for accepted quotes and customer billing.

Only this module talks HTTP to ``https://api.stripe.com/v1``. Every call is a
form-encoded POST with the secret key as a bearer header:

* ``/checkout/sessions`` — mode ``payment`` for one-time quotes, mode
  ``subscription`` for recurring ones (prices inline via ``price_data`` with
  ``recurring[interval]``), keyed by the quote's idempotency key so a retried
  accept returns the session already opened;
* ``/customers`` — one Stripe Customer per (business, customer), created the
  first time a recurring quote is accepted, keyed by the customer id;
* ``/billing_portal/sessions`` — a Billing Portal link for that customer.

The key, request bodies and provider responses never leave this module; errors
carry the status code only.
"""

import logging
from datetime import UTC, datetime
from typing import Final
from urllib.parse import quote

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from gvas.domain.payments import (
    BillingCustomerRequest,
    BillingCustomerResult,
    BillingPortalRequest,
    BillingPortalResult,
    PaymentCheckoutClosedError,
    PaymentCheckoutError,
    PaymentCheckoutRequest,
    PaymentCheckoutResult,
)
from gvas.infrastructure.stripe.config import StripeSettings

logger = logging.getLogger(__name__)

CHECKOUT_SESSIONS_PATH: Final = "/checkout/sessions"
CUSTOMERS_PATH: Final = "/customers"
BILLING_PORTAL_SESSIONS_PATH: Final = "/billing_portal/sessions"


class StripeCheckoutError(PaymentCheckoutError):
    """Raised when a checkout call should be retried or surfaced as a failure."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class _CheckoutSessionResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    id: str
    url: str | None = None
    payment_intent: str | None = None
    expires_at: int | None = None


class _CustomerResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    id: str


class _BillingPortalSessionResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    url: str


def checkout_form(request: PaymentCheckoutRequest) -> dict[str, str]:
    """The form-encoded session body: one line per item, the return URLs, the
    public client reference and our tracking metadata.

    Recurring quotes switch to ``mode=subscription``: each price carries
    ``recurring[interval]``, the session is bound to the Stripe Customer, and
    the metadata is mirrored onto the subscription so its later lifecycle
    events can be traced back to the quote.
    """

    form: dict[str, str] = {
        "mode": "subscription" if request.is_subscription else "payment",
        "success_url": request.success_url,
        "cancel_url": request.cancel_url,
        "client_reference_id": request.client_reference,
    }
    if request.customer_ref is not None:
        form["customer"] = request.customer_ref
    for index, item in enumerate(request.line_items):
        prefix = f"line_items[{index}]"
        form[f"{prefix}[quantity]"] = str(item.quantity)
        form[f"{prefix}[price_data][currency]"] = request.currency
        form[f"{prefix}[price_data][unit_amount]"] = str(item.amount_minor)
        form[f"{prefix}[price_data][product_data][name]"] = item.description
        if request.recurring_interval is not None:
            form[f"{prefix}[price_data][recurring][interval]"] = request.recurring_interval.value
    for key, value in request.metadata.items():
        form[f"metadata[{key}]"] = value
        if request.is_subscription:
            form[f"subscription_data[metadata][{key}]"] = value
    return form


def customer_form(request: BillingCustomerRequest) -> dict[str, str]:
    form: dict[str, str] = {"email": request.email}
    if request.name:
        form["name"] = request.name
    if request.phone:
        form["phone"] = request.phone
    for key, value in request.metadata.items():
        form[f"metadata[{key}]"] = value
    return form


def billing_portal_form(request: BillingPortalRequest) -> dict[str, str]:
    return {"customer": request.customer_ref, "return_url": request.return_url}


class StripeCheckout:
    """Implements ``PaymentCheckoutPort`` and ``BillingAccountPort``."""

    def __init__(self, settings: StripeSettings, client: httpx.AsyncClient) -> None:
        if not settings.is_configured:
            raise StripeCheckoutError("stripe secret key is not configured")
        self._settings = settings
        self._client = client

    async def create_checkout(self, request: PaymentCheckoutRequest) -> PaymentCheckoutResult:
        payload = await self._post(
            CHECKOUT_SESSIONS_PATH,
            checkout_form(request),
            idempotency_key=request.idempotency_key,
            what="checkout session",
        )
        try:
            parsed = _CheckoutSessionResponse.model_validate(payload)
        except ValidationError as error:
            raise _unreadable("checkout session", error) from error
        if not parsed.url:
            raise StripeCheckoutError("payment provider returned no checkout url")
        return PaymentCheckoutResult(
            session_id=parsed.id,
            checkout_url=parsed.url,
            payment_intent_id=parsed.payment_intent,
            expires_at=(
                datetime.fromtimestamp(parsed.expires_at, UTC)
                if parsed.expires_at is not None
                else None
            ),
        )

    async def expire_checkout(self, session_id: str) -> None:
        try:
            await self._post(
                f"{CHECKOUT_SESSIONS_PATH}/{quote(session_id, safe='')}/expire",
                {},
                idempotency_key=None,
                what="checkout session expiry",
            )
        except StripeCheckoutError as error:
            # 404: Stripe no longer knows the session. 400 also covers other
            # request errors, so only a session that is no longer open counts.
            if error.status_code == 404 or (
                error.status_code == 400 and await self._session_status(session_id) != "open"
            ):
                raise PaymentCheckoutClosedError("checkout session is already closed") from error
            raise

    async def _session_status(self, session_id: str) -> object:
        try:
            response = await self._client.get(
                f"{self._settings.api_base_url.rstrip('/')}"
                f"{CHECKOUT_SESSIONS_PATH}/{quote(session_id, safe='')}",
                headers={"Authorization": f"Bearer {self._settings.secret_key}"},
                timeout=self._settings.timeout_seconds,
            )
        except httpx.HTTPError as error:
            raise StripeCheckoutError("payment provider was unreachable") from error
        if response.status_code >= 400:
            raise StripeCheckoutError(
                f"payment provider returned http {response.status_code}",
                status_code=response.status_code,
            )
        try:
            payload: object = response.json()
        except ValueError as error:
            raise _unreadable("checkout session lookup", error) from error
        return payload.get("status") if isinstance(payload, dict) else None

    async def create_customer(self, request: BillingCustomerRequest) -> BillingCustomerResult:
        payload = await self._post(
            CUSTOMERS_PATH,
            customer_form(request),
            idempotency_key=request.idempotency_key,
            what="customer",
        )
        try:
            parsed = _CustomerResponse.model_validate(payload)
        except ValidationError as error:
            raise _unreadable("customer", error) from error
        return BillingCustomerResult(customer_ref=parsed.id)

    async def create_billing_portal_session(
        self, request: BillingPortalRequest
    ) -> BillingPortalResult:
        payload = await self._post(
            BILLING_PORTAL_SESSIONS_PATH,
            billing_portal_form(request),
            idempotency_key=None,
            what="billing portal session",
        )
        try:
            parsed = _BillingPortalSessionResponse.model_validate(payload)
        except ValidationError as error:
            raise _unreadable("billing portal session", error) from error
        return BillingPortalResult(url=parsed.url)

    async def _post(
        self, path: str, form: dict[str, str], *, idempotency_key: str | None, what: str
    ) -> object:
        headers = {"Authorization": f"Bearer {self._settings.secret_key}"}
        if idempotency_key is not None:
            headers["Idempotency-Key"] = idempotency_key
        try:
            response = await self._client.post(
                f"{self._settings.api_base_url.rstrip('/')}{path}",
                data=form,
                headers=headers,
                timeout=self._settings.timeout_seconds,
            )
        except httpx.HTTPError as error:
            logger.warning("%s request failed: %s", what, type(error).__name__)
            raise StripeCheckoutError("payment provider was unreachable") from error
        if response.status_code >= 400:
            logger.warning("%s returned http %s", what, response.status_code)
            raise StripeCheckoutError(
                f"payment provider returned http {response.status_code}",
                status_code=response.status_code,
            )
        try:
            payload: object = response.json()
        except ValueError as error:
            raise _unreadable(what, error) from error
        return payload


def _unreadable(what: str, error: Exception) -> StripeCheckoutError:
    logger.warning("%s returned an unreadable response: %s", what, type(error).__name__)
    return StripeCheckoutError("payment provider returned an unreadable response")
