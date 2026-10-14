"""Local compatibility shim for emergentintegrations.payments.stripe.checkout.

The real `emergentintegrations` package has been removed from PyPI
entirely -- it is no longer installable anywhere. This reimplements the
same interface server.py already calls (StripeCheckout,
CheckoutSessionRequest, CheckoutSessionResponse, CheckoutStatusResponse)
directly against the standard, publicly-installable `stripe` Python SDK.
Unlike llm/chat.py, this is a real implementation, not a stub -- wallet
top-up (create_checkout_session/check_payment_status/stripe_webhook) is a
core feature, not something gated behind an optional key the way the AI
features are.
"""
import asyncio
import os
from dataclasses import dataclass, field
from typing import Optional

import stripe


@dataclass
class CheckoutSessionRequest:
    amount: float
    currency: str
    success_url: str
    cancel_url: str
    metadata: dict = field(default_factory=dict)
    payment_methods: list = field(default_factory=lambda: ["card"])


@dataclass
class CheckoutSessionResponse:
    session_id: str
    url: str


@dataclass
class CheckoutStatusResponse:
    status: str
    payment_status: str
    session_id: str


class StripeCheckout:
    """Thin async wrapper around stripe.checkout.Session, scoped to one API key."""

    def __init__(self, api_key: str, webhook_url: str = ""):
        self.api_key = api_key
        self.webhook_url = webhook_url

    async def create_checkout_session(self, req: CheckoutSessionRequest) -> CheckoutSessionResponse:
        session = await asyncio.to_thread(
            stripe.checkout.Session.create,
            api_key=self.api_key,
            mode="payment",
            payment_method_types=req.payment_methods,
            line_items=[{
                "price_data": {
                    "currency": req.currency,
                    "unit_amount": int(round(req.amount * 100)),
                    "product_data": {"name": "BillSure wallet top-up"},
                },
                "quantity": 1,
            }],
            success_url=req.success_url,
            cancel_url=req.cancel_url,
            metadata=req.metadata,
        )
        return CheckoutSessionResponse(session_id=session.id, url=session.url)

    async def get_checkout_status(self, session_id: str) -> CheckoutStatusResponse:
        session = await asyncio.to_thread(
            stripe.checkout.Session.retrieve, session_id, api_key=self.api_key
        )
        return CheckoutStatusResponse(
            status=session.status, payment_status=session.payment_status, session_id=session.id
        )

    async def handle_webhook(self, payload: bytes, signature: str) -> CheckoutStatusResponse:
        # Separate from STRIPE_PAYMENT_INTENT_WEBHOOK_SECRET (stripe_collections.py's
        # webhook) -- this is specifically the Checkout Session webhook's own secret,
        # from a second endpoint in the Stripe Dashboard subscribed to
        # checkout.session.completed / checkout.session.expired.
        webhook_secret = os.environ.get("STRIPE_CHECKOUT_WEBHOOK_SECRET", "")
        if not webhook_secret:
            raise RuntimeError(
                "STRIPE_CHECKOUT_WEBHOOK_SECRET is not set -- cannot verify the "
                "Stripe webhook signature, refusing to process an unverified event."
            )
        event = await asyncio.to_thread(
            stripe.Webhook.construct_event, payload, signature, webhook_secret
        )
        session = event["data"]["object"]
        return CheckoutStatusResponse(
            status=session.get("status", ""),
            payment_status=session.get("payment_status", ""),
            session_id=session.get("id", ""),
        )
