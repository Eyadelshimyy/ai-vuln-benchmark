"""
backend/stripe_collections.py
==============================
Real Stripe integration for saving a reusable, tokenized payment method and
charging it off-session on a schedule.

This replaces two things process_auto_deductions used to fake:
  1. It credited the ledger/wallet on a timer without ever charging anyone.
  2. The only "payment method" record that existed (payment_methods, via the
     existing POST /payment-methods) is raw customer-typed card/account
     numbers that never touch Stripe — so even before fixing (1), there was
     no actual token to charge.

This module only handles charging an ALREADY SAVED, ALREADY TOKENIZED
Stripe payment method off-session. Saving a NEW one must go through
Stripe's own tokenization in the browser (SetupIntent + Stripe Elements /
Payment Element) so raw card/account numbers never reach this backend —
create_setup_intent() / confirm_setup_intent_and_save() are the server-side
half of that; the browser half is a separate frontend change (Stripe.js),
not included here — see INTEGRATION_NOTES_STRIPE.md.

Existing payment_methods rows from the old raw-entry form have no
stripe_payment_method_id and are correctly treated as NOT chargeable —
get_chargeable_payment_method() skips them. A customer with only a legacy
row will need to re-add their payment method through the new SetupIntent
flow before scheduled collection can work for them.
"""

import logging
import os
from decimal import Decimal
from typing import Optional

import stripe

import supabase_db as sdb

logger = logging.getLogger(__name__)

STRIPE_API_KEY = os.environ.get("STRIPE_API_KEY", "")


def _client():
    if not STRIPE_API_KEY:
        raise RuntimeError("STRIPE_API_KEY is not set")
    stripe.api_key = STRIPE_API_KEY
    return stripe


def _to_cents(amount) -> int:
    return int((Decimal(str(amount)) * 100).to_integral_value())


# ---------------------------------------------------------------
# Saving a reusable payment method
# ---------------------------------------------------------------

async def get_or_create_stripe_customer(user: dict) -> str:
    """Reuses users.stripe_customer_id if already set; otherwise creates a
    Stripe Customer and persists the id. This should be the ONLY place a
    Stripe Customer is ever created for a user."""
    if user.get("stripe_customer_id"):
        return user["stripe_customer_id"]

    sc = _client()
    customer = sc.Customer.create(
        email=user["email"],
        name=user.get("full_name", ""),
        metadata={"billsure_user_id": user["id"]},
    )
    await sdb.update_one("users", {"id": user["id"]}, {"$set": {"stripe_customer_id": customer.id}})
    return customer.id


async def create_setup_intent(user: dict, payment_method_types: Optional[list] = None) -> dict:
    """Creates a SetupIntent for the user's Stripe customer and returns its
    client_secret. The frontend uses this with Stripe Elements / Payment
    Element to collect card or AU BECS details directly in the browser —
    raw details never transit this backend. Call
    confirm_setup_intent_and_save() once the browser reports the
    SetupIntent as succeeded."""
    sc = _client()
    customer_id = await get_or_create_stripe_customer(user)
    si = sc.SetupIntent.create(
        customer=customer_id,
        payment_method_types=payment_method_types or ["card", "au_becs_debit"],
        usage="off_session",
        metadata={"billsure_user_id": user["id"]},
    )
    return {"client_secret": si.client_secret, "setup_intent_id": si.id}


async def confirm_setup_intent_and_save(user: dict, setup_intent_id: str, label: str = "",
                                         is_primary: bool = False) -> dict:
    """Call after the frontend confirms the SetupIntent client-side. Verifies
    with Stripe directly that it actually succeeded and belongs to this
    user (never trust the client's word for it) and only then persists a
    payment_methods row carrying a real stripe_payment_method_id."""
    sc = _client()
    si = sc.SetupIntent.retrieve(setup_intent_id)

    if si.customer != user.get("stripe_customer_id"):
        raise ValueError("SetupIntent does not belong to this user's Stripe customer")
    if si.status != "succeeded":
        raise ValueError(f"SetupIntent status is '{si.status}', not 'succeeded' — cannot save")

    pm = sc.PaymentMethod.retrieve(si.payment_method)
    pm_type = pm.type  # 'card' | 'au_becs_debit'

    if is_primary:
        await sdb.update_many("payment_methods", {"user_id": user["id"]}, {"$set": {"is_primary": False}})

    display = {}
    if pm_type == "card" and pm.card:
        display = {"card_last4": pm.card.last4, "card_brand": pm.card.brand}
    elif pm_type == "au_becs_debit" and pm.au_becs_debit:
        # Real Stripe au_becs_debit PaymentMethod objects only carry
        # bsb_number, fingerprint, and last4 -- there is no bank_name field
        # (confirmed against a real sandbox response; the original code
        # assumed one existed and crashed with AttributeError on every
        # BECS confirm-setup call).
        display = {"account_number_masked": "****" + (pm.au_becs_debit.last4 or "")}

    row = await sdb.insert_one("payment_methods", {
        "user_id": user["id"],
        "type": "au_becs_debit" if pm_type == "au_becs_debit" else "card",
        "label": label or (display.get("card_brand") or ("Bank Account" if pm_type == "au_becs_debit" else pm_type)),
        "stripe_payment_method_id": pm.id,
        "is_primary": is_primary,
        **display,
    })
    return row


# ---------------------------------------------------------------
# Charging a saved payment method off-session
# ---------------------------------------------------------------

async def get_chargeable_payment_method(user_id: str) -> Optional[dict]:
    """Returns the user's primary payment_methods row IF it carries a real
    Stripe token. Returns None if the user has no eligible method —
    including if their only saved method is a legacy row from the old
    raw-entry form, which has no token to charge."""
    methods = await sdb.find_many("payment_methods", {"user_id": user_id, "is_primary": True})
    for m in methods:
        if m.get("stripe_payment_method_id"):
            return m
    all_methods = await sdb.find_many("payment_methods", {"user_id": user_id})
    for m in all_methods:
        if m.get("stripe_payment_method_id"):
            return m
    return None


async def collect_scheduled_contribution(user_id: str, amount, payment_plan_id: str,
                                          idempotency_key: str) -> dict:
    """
    Attempts a real off-session charge for a scheduled contribution.

    `idempotency_key` MUST be unique per (payment_plan_id, scheduled cycle)
    — e.g. f"{payment_plan_id}:{today_str}" — passed to Stripe so a
    scheduler retry after a crash can't double-charge (Stripe deduplicates
    any request replayed with the same key), AND stored with a UNIQUE
    constraint on collection_attempts.idempotency_key so a second attempt
    in the same cycle fails fast locally before even calling Stripe.

    Returns {"status": "succeeded"|"processing"|"failed"|"no_payment_method", ...}.
    "succeeded" here means Stripe confirmed synchronously (typical for
    cards) — the caller (server.py) is responsible for the atomic
    ledger-credit transition on this path. For "processing" (typical for
    au_becs_debit, which settles asynchronously), the ledger is credited
    later by the webhook only — see server.py's /webhook/stripe-payment-intents.
    """
    user = await sdb.find_one("users", {"id": user_id})
    if not user:
        return {"status": "failed", "reason": "user_not_found"}

    method = await get_chargeable_payment_method(user_id)
    if not method:
        logger.warning(f"No chargeable (tokenized) payment method for user {user_id} — skipping scheduled collection")
        return {"status": "no_payment_method"}

    existing = await sdb.find_one("collection_attempts", {"idempotency_key": idempotency_key})
    if existing:
        logger.info(f"Collection attempt {idempotency_key} already exists (status={existing['status']}) — not re-submitting")
        return {"status": existing["status"], "payment_intent_id": existing.get("stripe_payment_intent_id")}

    sc = _client()
    attempt = await sdb.insert_one("collection_attempts", {
        "payment_plan_id": payment_plan_id,
        "user_id": user_id,
        "amount": str(amount),
        "idempotency_key": idempotency_key,
        "status": "pending",
    })

    try:
        pi = sc.PaymentIntent.create(
            amount=_to_cents(amount),
            currency="aud",
            customer=user["stripe_customer_id"],
            payment_method=method["stripe_payment_method_id"],
            payment_method_types=[method["type"]] if method["type"] in ("card", "au_becs_debit") else None,
            off_session=True,
            confirm=True,
            metadata={
                "billsure_user_id": user_id,
                "payment_plan_id": payment_plan_id,
                "collection_attempt_id": attempt["id"],
                "kind": "scheduled_contribution",
                "payment_method_type": method["type"],
            },
            idempotency_key=idempotency_key,
        )
        # An off-session confirm can, in principle, come back with this
        # status directly rather than raising (see the CardError branch
        # below for the path Stripe's documented behaviour actually takes
        # for off_session=True — this is a defensive second path, not the
        # primary one).
        if pi.status == "requires_action":
            await sdb.update_one("collection_attempts", {"id": attempt["id"]}, {"$set": {
                "status": "requires_customer_action", "stripe_payment_intent_id": pi.id,
            }})
            return {
                "status": "requires_customer_action",
                "payment_intent_id": pi.id,
                "collection_attempt_id": attempt["id"],
                "reason": "authentication_required",
            }

        await sdb.update_one("collection_attempts", {"id": attempt["id"]}, {"$set": {
            "status": pi.status, "stripe_payment_intent_id": pi.id,
        }})
        return {"status": pi.status, "payment_intent_id": pi.id, "collection_attempt_id": attempt["id"]}

    except stripe.error.CardError as e:
        # This is the path Stripe's documented behaviour actually takes:
        # for off_session=True + confirm=True, a card that needs further
        # authentication (3D Secure etc.) does NOT come back as a normal
        # "requires_action" status -- Stripe raises a CardError with
        # code == "authentication_required" instead, specifically because
        # off-session confirmation isn't expected to complete an
        # interactive challenge. The PaymentIntent itself (now sitting in
        # requires_action) is exposed on the exception so the customer can
        # later come back on-session and confirm it themselves using its
        # client_secret -- that on-session confirmation UI is a separate,
        # not-yet-built piece of work; this only needs to record the state
        # accurately and make sure it's visible rather than silently
        # falling into "failed".
        if getattr(e, "code", None) == "authentication_required":
            pi_obj = getattr(e, "payment_intent", None) or (e.json_body or {}).get("error", {}).get("payment_intent")
            pi_id = (pi_obj or {}).get("id") if isinstance(pi_obj, dict) else getattr(pi_obj, "id", None)
            await sdb.update_one("collection_attempts", {"id": attempt["id"]}, {"$set": {
                "status": "requires_customer_action",
                "stripe_payment_intent_id": pi_id,
                "error_message": str(e.user_message or e),
            }})
            return {
                "status": "requires_customer_action",
                "payment_intent_id": pi_id,
                "collection_attempt_id": attempt["id"],
                "reason": "authentication_required",
            }

        await sdb.update_one("collection_attempts", {"id": attempt["id"]}, {"$set": {
            "status": "failed", "error_message": str(e.user_message or e),
        }})
        return {"status": "failed", "reason": e.user_message or "card_error"}

    except stripe.error.StripeError as e:
        await sdb.update_one("collection_attempts", {"id": attempt["id"]}, {"$set": {
            "status": "failed", "error_message": str(e),
        }})
        logger.error(f"Stripe error collecting scheduled contribution for user {user_id}: {e}")
        return {"status": "failed", "reason": "stripe_error"}
