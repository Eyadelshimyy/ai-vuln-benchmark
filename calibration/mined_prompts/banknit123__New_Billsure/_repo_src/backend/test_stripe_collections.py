"""
Standalone logic test for stripe_collections.py, against a mock of the
`stripe` SDK and an in-memory fake of supabase_db — proves the control flow
(save a tokenized method, charge it off-session, handle success/failure/
card errors, idempotency dedup, and the "no chargeable method" case) is
correct WITHOUT a real Stripe account or API key. You still need to test
against real Stripe test-mode keys before going live — this only proves the
code's logic, not Stripe's actual behaviour.

Run: python3 test_stripe_collections.py
"""
import asyncio
import sys
import types
import uuid
from datetime import datetime, timezone

# ---- in-memory fake of supabase_db (same as test_ledger_flow.py) ----
_tables = {}


def _matches(row, filters):
    for k, v in filters.items():
        if isinstance(v, dict):
            if "$in" in v and row.get(k) not in v["$in"]:
                return False
            if "$ne" in v and row.get(k) == v["$ne"]:
                return False
        elif row.get(k) != v:
            return False
    return True


async def find_one(table, filters, exclude_fields=None):
    for row in _tables.get(table, []):
        if _matches(row, filters):
            return dict(row)
    return None


async def find_many(table, filters=None, exclude_fields=None, order_by=None, order_desc=False, limit=10000):
    rows = [dict(r) for r in _tables.get(table, []) if not filters or _matches(r, filters)]
    return rows[:limit]


async def insert_one(table, data):
    row = dict(data)
    row.setdefault("id", str(uuid.uuid4()))
    row.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    _tables.setdefault(table, []).append(row)
    return dict(row)


async def update_one(table, filters, updates):
    set_data = updates.get("$set", updates)
    for row in _tables.get(table, []):
        if _matches(row, filters):
            row.update(set_data)
            return True
    return False


async def update_many(table, filters, updates):
    set_data = updates.get("$set", updates)
    n = 0
    for row in _tables.get(table, []):
        if _matches(row, filters):
            row.update(set_data)
            n += 1
    return n


fake_sdb = types.ModuleType("supabase_db")
fake_sdb.find_one = find_one
fake_sdb.find_many = find_many
fake_sdb.insert_one = insert_one
fake_sdb.update_one = update_one
fake_sdb.update_many = update_many
sys.modules["supabase_db"] = fake_sdb


# ---- mock of the `stripe` SDK surface stripe_collections.py touches ----
class FakeObj(dict):
    """Lets mock Stripe responses be accessed as obj.field, like real Stripe objects."""
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            return None


class FakeCardError(Exception):
    def __init__(self, user_message, code=None, payment_intent=None):
        self.user_message = user_message
        self.code = code
        self.payment_intent = payment_intent
        self.json_body = {"error": {"payment_intent": payment_intent}} if payment_intent else {}
        super().__init__(user_message)


class FakeStripeError(Exception):
    pass


class MockStripeState:
    """Controls what the mock Stripe returns, so tests can drive different scenarios."""
    def __init__(self):
        self.next_payment_intent_status = "succeeded"
        self.raise_card_error = False
        self.raise_generic_error = False
        self.raise_authentication_required = False
        self.payment_intent_calls = []


state = MockStripeState()

fake_stripe = types.ModuleType("stripe")
fake_stripe.api_key = None
fake_stripe.error = types.SimpleNamespace(CardError=FakeCardError, StripeError=FakeStripeError,
                                           SignatureVerificationError=Exception)


class _Customer:
    @staticmethod
    def create(**kwargs):
        return FakeObj(id="cus_" + str(uuid.uuid4())[:8], **kwargs)


class _SetupIntent:
    @staticmethod
    def create(**kwargs):
        return FakeObj(id="seti_" + str(uuid.uuid4())[:8], client_secret="seti_secret_x",
                        customer=kwargs["customer"], status="requires_payment_method")

    @staticmethod
    def retrieve(setup_intent_id):
        return FakeObj(id=setup_intent_id, customer="cus_test1", status="succeeded",
                        payment_method="pm_test1")


class _PaymentMethod:
    @staticmethod
    def retrieve(pm_id):
        return FakeObj(id=pm_id, type="card",
                        card=FakeObj(last4="4242", brand="visa"),
                        au_becs_debit=None)


class _PaymentIntent:
    @staticmethod
    def create(**kwargs):
        state.payment_intent_calls.append(kwargs)
        if state.raise_authentication_required:
            # Mirrors real Stripe: for off_session=True + confirm=True, a
            # card needing 3D Secure / SCA raises a CardError with
            # code=="authentication_required" rather than returning a
            # normal "requires_action" status -- the PaymentIntent itself
            # (now sitting in requires_action) is exposed on the exception.
            pi_id = "pi_" + str(uuid.uuid4())[:8]
            raise FakeCardError(
                "This payment requires additional authentication.",
                code="authentication_required",
                payment_intent={"id": pi_id, "status": "requires_action"},
            )
        if state.raise_card_error:
            raise FakeCardError("Your card was declined.")
        if state.raise_generic_error:
            raise FakeStripeError("network error")
        return FakeObj(id="pi_" + str(uuid.uuid4())[:8], status=state.next_payment_intent_status)


fake_stripe.Customer = _Customer
fake_stripe.SetupIntent = _SetupIntent
fake_stripe.PaymentMethod = _PaymentMethod
fake_stripe.PaymentIntent = _PaymentIntent
sys.modules["stripe"] = fake_stripe

import os
os.environ["STRIPE_API_KEY"] = "sk_test_fake_for_logic_testing_only"

import stripe_collections as sc  # noqa: E402


async def main():
    user = await insert_one("users", {"email": "alice@example.com", "full_name": "Alice"})

    # --- SetupIntent creation ---
    si = await sc.create_setup_intent(user)
    assert "client_secret" in si
    user = await find_one("users", {"id": user["id"]})
    assert user["stripe_customer_id"], "expected a Stripe customer to be created and persisted"
    print(f"[ok] SetupIntent created, Stripe customer persisted: {user['stripe_customer_id']}")

    # confirm_setup_intent_and_save requires the retrieved SetupIntent's
    # customer to match — patch the mock to match this test's actual customer id.
    real_customer_id = user["stripe_customer_id"]
    _SetupIntent.retrieve = staticmethod(lambda sid: FakeObj(id=sid, customer=real_customer_id,
                                                              status="succeeded", payment_method="pm_test1"))

    pm_row = await sc.confirm_setup_intent_and_save(user, "seti_abc", label="My Visa", is_primary=True)
    assert pm_row["stripe_payment_method_id"] == "pm_test1"
    assert pm_row["card_last4"] == "4242"
    print(f"[ok] payment method saved with real Stripe token: {pm_row['stripe_payment_method_id']}")

    # --- chargeable method lookup ---
    chargeable = await sc.get_chargeable_payment_method(user["id"])
    assert chargeable and chargeable["id"] == pm_row["id"]
    print("[ok] get_chargeable_payment_method finds the tokenized method")

    # A user with NO payment method at all must not be chargeable.
    bob = await insert_one("users", {"email": "bob@example.com", "full_name": "Bob"})
    result = await sc.collect_scheduled_contribution(bob["id"], "50.00", "plan-bob", "plan-bob:2026-07-25")
    assert result["status"] == "no_payment_method", f"expected no_payment_method, got {result}"
    print("[ok] user with no tokenized payment method correctly cannot be charged")

    # A user whose only saved method is a LEGACY (non-tokenized) row must
    # also not be chargeable — this is the specific bug this module fixes.
    carol = await insert_one("users", {"email": "carol@example.com", "full_name": "Carol",
                                        "stripe_customer_id": "cus_carol"})
    await insert_one("payment_methods", {"user_id": carol["id"], "type": "bank_account",
                                          "label": "Old raw-entry account", "is_primary": True})
    result = await sc.collect_scheduled_contribution(carol["id"], "50.00", "plan-carol", "plan-carol:2026-07-25")
    assert result["status"] == "no_payment_method", f"expected legacy row to be non-chargeable, got {result}"
    print("[ok] legacy (non-tokenized) payment_methods row correctly rejected as non-chargeable")

    # --- successful off-session charge ---
    state.next_payment_intent_status = "succeeded"
    result = await sc.collect_scheduled_contribution(user["id"], "42.50", "plan-alice", "plan-alice:2026-07-25")
    assert result["status"] == "succeeded"
    assert len(state.payment_intent_calls) == 1
    call = state.payment_intent_calls[0]
    assert call["amount"] == 4250, f"expected 4250 cents, got {call['amount']}"
    assert call["off_session"] is True and call["confirm"] is True
    assert call["idempotency_key"] == "plan-alice:2026-07-25"
    print(f"[ok] successful off-session charge: amount_cents={call['amount']}, idempotency_key={call['idempotency_key']}")

    # --- idempotency: retrying the SAME cycle must not hit Stripe again ---
    result2 = await sc.collect_scheduled_contribution(user["id"], "42.50", "plan-alice", "plan-alice:2026-07-25")
    assert result2["status"] == "succeeded"
    assert len(state.payment_intent_calls) == 1, "must NOT have called Stripe again for the same idempotency key"
    print("[ok] retrying the same scheduled cycle does not double-charge (no second Stripe call)")

    # --- card decline path ---
    state.raise_card_error = True
    result = await sc.collect_scheduled_contribution(user["id"], "10.00", "plan-alice", "plan-alice:2026-08-25")
    assert result["status"] == "failed" and "declined" in result["reason"].lower()
    print(f"[ok] card decline correctly surfaced as failed: {result['reason']}")
    state.raise_card_error = False

    # --- au_becs_debit goes async ("processing"), not immediate success ---
    state.next_payment_intent_status = "processing"
    result = await sc.collect_scheduled_contribution(user["id"], "10.00", "plan-alice", "plan-alice:2026-09-25")
    assert result["status"] == "processing"
    print("[ok] async payment method (e.g. BECS) correctly reported as 'processing', not 'succeeded'")

    # --- Phase 3: off-session charge requiring extra card authentication ---
    state.raise_authentication_required = True
    result = await sc.collect_scheduled_contribution(user["id"], "15.00", "plan-alice", "plan-alice:2026-10-25")
    assert result["status"] == "requires_customer_action", f"expected requires_customer_action, got {result}"
    assert result["payment_intent_id"] and result["payment_intent_id"].startswith("pi_")
    attempt_row = await find_one("collection_attempts", {"id": result["collection_attempt_id"]})
    assert attempt_row["status"] == "requires_customer_action", \
        f"collection_attempts row must be marked requires_customer_action, got {attempt_row['status']}"
    print(f"[ok] off-session charge needing extra authentication correctly surfaced as 'requires_customer_action' "
          f"(not silently dropped or marked 'failed'): {result['payment_intent_id']}")
    state.raise_authentication_required = False

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
