"""
HTTP-layer test for pilot_api.py, using FastAPI's TestClient (real
HTTP request/response cycle through Starlette's ASGI transport, not a
direct Python function call) — including the API key authentication
and per-endpoint permission layer added on top of the working API.

Same in-memory fake supabase_db pattern as every other test_*.py file
in this repo — no live database, no live network.

Run: python3 test_pilot_api.py
"""
import asyncio
import io
import sys
import types
import uuid
from decimal import Decimal
from datetime import datetime, timezone

from PIL import Image, ImageDraw, ImageFont

# ---- in-memory fake of supabase_db's public interface ----
_tables = {}


def _matches(row, filters):
    for k, v in filters.items():
        if row.get(k) != v:
            return False
    return True


async def find_one(table, filters, exclude_fields=None):
    for row in _tables.get(table, []):
        if _matches(row, filters):
            return dict(row)
    return None


async def find_many(table, filters=None, exclude_fields=None, limit=10000):
    filters = filters or {}
    return [dict(r) for r in _tables.get(table, []) if _matches(r, filters)][:limit]


async def insert_one(table, data):
    row = dict(data)
    row.setdefault("id", str(uuid.uuid4()))
    _tables.setdefault(table, []).append(row)
    return dict(row)


async def insert_many(table, rows):
    return [await insert_one(table, r) for r in rows]


async def update_one(table, filters, updates):
    for row in _tables.get(table, []):
        if _matches(row, filters):
            row.update(updates)
            return True
    return False


async def update_many(table, filters, updates):
    n = 0
    for row in _tables.get(table, []):
        if _matches(row, filters):
            row.update(updates)
            n += 1
    return n


fake_sdb = types.SimpleNamespace(
    find_one=find_one, find_many=find_many, insert_one=insert_one,
    insert_many=insert_many, update_one=update_one, update_many=update_many,
)
sys.modules["supabase_db"] = fake_sdb

import pilot_config as pc   # noqa: E402
import pilot_auth as pa    # noqa: E402

FAILURES = []

# Real UUIDs -- every customer_id/user_id column in the pilot schema is
# UUID-typed (migrations 013-021); this test now proves the fix that
# followed from finding that gap during live deployment testing.
JANE_UUID = str(uuid.uuid4())
OTHER_CUSTOMER_UUID = str(uuid.uuid4())


def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}")
    if not condition:
        FAILURES.append(label)


def make_bill_photo(biller_name, account_name, reference, amount, due_date) -> bytes:
    img = Image.new("RGB", (900, 500), color="white")
    draw = ImageDraw.Draw(img)
    try:
        font_big = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    except Exception:
        font_big = ImageFont.load_default()
        font_small = font_big
    draw.text((30, 30), biller_name, fill="black", font=font_big)
    draw.text((30, 100), f"Account Name: {account_name}", fill="black", font=font_small)
    draw.text((30, 150), f"Reference Number: {reference}", fill="black", font=font_small)
    draw.text((30, 200), f"Amount Due: ${amount}", fill="black", font=font_small)
    draw.text((30, 250), f"Due Date: {due_date}", fill="black", font=font_small)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def auth(raw_key: str) -> dict:
    return {"Authorization": f"Bearer {raw_key}"}


async def main():
    await pc.propose_config_version(pc.PilotConfig(), proposed_by="ops_lead", approved_by="compliance_lead", activate=True)

    import pilot_api  # noqa: E402  -- imported AFTER supabase_db is faked and config is active
    from fastapi.testclient import TestClient
    client = TestClient(pilot_api.app)

    # ---------------------------------------------------------------
    # Issue API keys for each role, exactly as an operator would
    # ---------------------------------------------------------------
    customer_key = await pa.issue_api_key(JANE_UUID, "customer", issued_by="ops_lead")
    case_worker_key = await pa.issue_api_key("case_worker_1", "case_worker", issued_by="ops_lead")
    compliance_key = await pa.issue_api_key("compliance_lead", "compliance_reviewer", issued_by="ops_lead", mfa_verified=True)
    admin_key_no_mfa = await pa.issue_api_key("admin_no_mfa", "admin", issued_by="ops_lead", mfa_verified=False)
    admin_key = await pa.issue_api_key("payments_admin_1", "admin", issued_by="ops_lead", mfa_verified=True)

    # ---------------------------------------------------------------
    # Public endpoints need no auth at all
    # ---------------------------------------------------------------
    resp = client.get("/health")
    check("GET /health requires no auth and returns 200", resp.status_code == 200)

    # ---------------------------------------------------------------
    # Sandbox console: served at both / and /console, no auth needed to
    # load the page itself (the API key is entered client-side and used
    # only for the actual API calls the page makes).
    # ---------------------------------------------------------------
    resp = client.get("/")
    check("GET / serves the sandbox console HTML with no auth required", resp.status_code == 200)
    check("the console page mentions the sandbox/no-real-money warning", "Sandbox" in resp.text and "No Real Money" in resp.text)

    resp = client.get("/console")
    check("GET /console serves the same page", resp.status_code == 200 and "BillSure" in resp.text)

    original_find_many = fake_sdb.find_many
    async def broken_find_many(*a, **kw):
        raise RuntimeError("simulated database outage")
    fake_sdb.find_many = broken_find_many
    resp = client.get("/health")
    check("GET /health returns a real 503 (not just a 200 with an unhealthy body) when the DB is unreachable", resp.status_code == 503)
    fake_sdb.find_many = original_find_many

    resp = client.get("/pilot/launch-gates/status")
    check("GET /pilot/launch-gates/status requires no auth", resp.status_code == 200)
    check("production_authorized is False with zero gates recorded (fail closed, over real HTTP)", resp.json()["production_authorized"] is False)

    # ---------------------------------------------------------------
    # Every other endpoint refuses an unauthenticated request
    # ---------------------------------------------------------------
    resp = client.post("/pilot/onboarding/apply", json={})
    check("POST /pilot/onboarding/apply with NO Authorization header returns 401, not 422/500", resp.status_code == 401)

    resp = client.post("/pilot/onboarding/apply", json={}, headers=auth("bsp_totally_made_up_invalid_key"))
    check("an invalid/garbage API key is rejected with 401", resp.status_code == 401)

    # ---------------------------------------------------------------
    # Wrong role for the endpoint -> 403, not a silent pass-through
    # ---------------------------------------------------------------
    apply_payload = {
        "user_id": JANE_UUID, "identity_verification_status": "verified",
        "age_confirmed": True, "residential_state": "VIC", "bank_account_verified": True,
        "income_amount": "5200", "income_frequency": "monthly", "employment_status": "full_time",
        "recurring_living_expenses": "2800", "existing_debts_and_bnpl": "0",
        "requested_credit_purpose": "electricity", "requirements_and_objectives": "Smooth out electricity bills",
        "utility_bill_ownership_verified": True,
        "consent_types_accepted": ["privacy", "identity_check", "affordability_check", "fraud_check"],
    }
    # A customer key CAN apply (has 'submit_application').
    resp = client.post("/pilot/onboarding/apply", json=apply_payload, headers=auth(customer_key.raw_key))
    check("a customer-role key CAN submit an application (has the permission)", resp.status_code == 200)
    application = resp.json()
    check("the application over authenticated HTTP is 'eligible'", application["eligibility_outcome"] == "eligible")

    # A key with a role that structurally cannot approve credit activation is refused with 403.
    resp = client.post(f"/pilot/onboarding/{application['id']}/activate-credit", json={
        "prepared_by": "credit_assessor_1", "approved_by": "compliance_lead",
        "contractual_limit": "2500.00", "active_customer_count": 0, "current_aggregate_contractual_exposure": "0",
    }, headers=auth(customer_key.raw_key))
    check("a customer-role key is REFUSED (403) from activating credit -- least privilege enforced over HTTP",
          resp.status_code == 403)

    # ---------------------------------------------------------------
    # MFA gating over HTTP: an admin key without MFA confirmation is refused
    # ---------------------------------------------------------------
    resp = client.post(f"/pilot/onboarding/{application['id']}/activate-credit", json={
        "prepared_by": "credit_assessor_1", "approved_by": "compliance_lead",
        "contractual_limit": "2500.00", "active_customer_count": 0, "current_aggregate_contractual_exposure": "0",
    }, headers=auth(admin_key_no_mfa.raw_key))
    check("an admin key issued WITHOUT mfa_verified=True is refused (403) from a privileged action, over real HTTP",
          resp.status_code == 403)

    # ---------------------------------------------------------------
    # The correctly-authorized, MFA-verified key succeeds
    # ---------------------------------------------------------------
    resp = client.post(f"/pilot/onboarding/{application['id']}/activate-credit", json={
        "prepared_by": "credit_assessor_1", "approved_by": "compliance_lead",
        "contractual_limit": "2500.00", "active_customer_count": 0, "current_aggregate_contractual_exposure": "0",
    }, headers=auth(compliance_key.raw_key))
    check("an MFA-verified compliance_reviewer key CAN activate credit", resp.status_code == 200)
    activation = resp.json()
    check("credit account is active over authenticated HTTP", activation["credit_account"]["status"] == "active")

    resp = client.get(f"/pilot/credit/accounts/{JANE_UUID}/balance", headers=auth(customer_key.raw_key))
    check("a customer CAN view their OWN balance", resp.status_code == 200)
    check("balance is correct", resp.json()["outstanding_principal"] == "0.00")

    other_customer_key = await pa.issue_api_key(OTHER_CUSTOMER_UUID, "customer", issued_by="ops_lead")
    resp = client.get(f"/pilot/credit/accounts/{JANE_UUID}/balance", headers=auth(other_customer_key.raw_key))
    check("a DIFFERENT customer CANNOT view Jane's balance (403, not leaked)", resp.status_code == 403)

    resp = client.get(f"/pilot/credit/accounts/{JANE_UUID}/balance", headers=auth(case_worker_key.raw_key))
    check("a case_worker CAN view any customer's balance (staff permission, not 'own data')", resp.status_code == 200)

    # ---------------------------------------------------------------
    # Bill upload requires auth + 'submit_bill'; real multipart + OCR
    # ---------------------------------------------------------------
    bill_bytes = make_bill_photo("Origin Energy", "Jane Dummy", "REF-HTTP-001", "120.00", "25/09/2026")

    resp = client.post(
        "/pilot/bills/upload",
        data={"customer_id": JANE_UUID, "customer_name_on_account": "Jane Dummy", "category": "electricity"},
        files={"file": ("bill.png", bill_bytes, "image/png")},
    )
    check("bill upload with no auth is refused (401)", resp.status_code == 401)

    resp = client.post(
        "/pilot/bills/upload",
        data={"customer_id": JANE_UUID, "customer_name_on_account": "Jane Dummy", "category": "electricity"},
        files={"file": ("bill.png", bill_bytes, "image/png")},
        headers=auth(customer_key.raw_key),
    )
    check("authenticated bill upload (real multipart + real OCR) succeeds", resp.status_code == 200)
    upload_result = resp.json()
    check("OCR ran the real tesseract_ocr method over the HTTP-uploaded file", upload_result["ocr"]["method"] == "tesseract_ocr")
    bill_id = upload_result["bill"]["id"]
    if upload_result["bill"]["verification_status"] == "manual_review":
        resp = client.post(f"/pilot/bills/{bill_id}/manual-review", json={"reviewer": "case_worker_1", "decision": "verified", "notes": "confirmed"},
                            headers=auth(case_worker_key.raw_key))
        check("manual review over authenticated HTTP moves the bill to verified", resp.json()["verification_status"] == "verified")

    # ---------------------------------------------------------------
    # Payment: requires 'process_payment' AND is still blocked by
    # launch_gates regardless of who's asking
    # ---------------------------------------------------------------
    resp = client.post(f"/pilot/bills/{bill_id}/pay", json={"customer_id": JANE_UUID, "requested_by": "payments_admin_1"},
                        headers=auth(customer_key.raw_key))
    check("a customer key is refused (403) from paying a bill -- least privilege, before even reaching launch_gates", resp.status_code == 403)

    resp = client.post(f"/pilot/bills/{bill_id}/pay", json={"customer_id": JANE_UUID, "requested_by": "payments_admin_1"},
                        headers=auth(admin_key.raw_key))
    check("an authorized, MFA-verified admin key STILL gets blocked (403) by launch_gates -- auth passing doesn't bypass the regulatory gate",
          resp.status_code == 403 and "launch gate" in resp.json()["detail"])

    balance_after_blocked_attempt = client.get(f"/pilot/credit/accounts/{JANE_UUID}/balance", headers=auth(customer_key.raw_key)).json()
    check("a blocked payment attempt over authenticated HTTP still moves zero money", balance_after_blocked_attempt["outstanding_principal"] == "0.00")

    # ---------------------------------------------------------------
    # Hardship and complaints: reachable by a customer key
    # ---------------------------------------------------------------
    resp = client.post("/pilot/hardship/requests", json={
        "customer_id": JANE_UUID, "reason": "reduced hours at work",
        "vulnerability_indicators": [], "requested_by": JANE_UUID,
    }, headers=auth(customer_key.raw_key))
    check("an authenticated customer can request hardship", resp.status_code == 200 and resp.json()["status"] == "open")

    resp = client.post("/pilot/complaints", json={
        "customer_id": JANE_UUID, "channel": "web_form", "description": "Test complaint over HTTP",
        "category": "standard", "severity": "low", "received_by": "agent1",
    }, headers=auth(customer_key.raw_key))
    check("an authenticated customer can submit a complaint", resp.status_code == 200)

    # ---------------------------------------------------------------
    # Documents: GET is public, POST accept requires auth
    # ---------------------------------------------------------------
    resp = client.get("/pilot/documents/credit_guide")
    check("GET a document is public (no auth needed) -- returns 404 since none is approved yet in this test, not 401",
          resp.status_code == 404)

    # ---------------------------------------------------------------
    # Reports: requires 'export_reports', not available to a customer
    # ---------------------------------------------------------------
    resp = client.get("/pilot/reports/credit-exposure", headers=auth(customer_key.raw_key))
    check("a customer key is refused (403) from the reports endpoint", resp.status_code == 403)

    resp = client.get("/pilot/reports/credit-exposure", headers=auth(admin_key.raw_key))
    check("an admin key can access the reports endpoint", resp.status_code == 200)
    check("credit exposure report shows the 1 activated customer over authenticated HTTP", resp.json()["active_customer_count"] == 1)

    # ---------------------------------------------------------------
    # UUID validation: a non-UUID customer/user id is a clear 422, never
    # a bare 500. Found live during initial deployment testing -- a
    # plain string id ("user-live-test-001") reached the real Postgres
    # UUID column, was rejected by the database with an opaque error,
    # and surfaced through this API's generic exception handler as an
    # unhelpful "internal error" (500). This is the regression test for
    # the fix: reject it at the API boundary instead, with a message
    # that names the exact field.
    # ---------------------------------------------------------------
    bad_apply_payload = {**{k: v for k, v in {
        "user_id": "not-a-real-uuid", "identity_verification_status": "verified",
        "age_confirmed": True, "residential_state": "VIC", "bank_account_verified": True,
        "requested_credit_purpose": "electricity", "requirements_and_objectives": "test",
        "utility_bill_ownership_verified": True, "consent_types_accepted": [],
    }.items()}}
    resp = client.post("/pilot/onboarding/apply", json=bad_apply_payload, headers=auth(customer_key.raw_key))
    check("a non-UUID user_id is rejected with 422, not a bare 500 internal error", resp.status_code == 422)
    check("the 422 error names the specific field that's invalid", "user_id" in str(resp.json()))

    resp = client.get("/pilot/credit/accounts/not-a-real-uuid/balance", headers=auth(admin_key.raw_key))
    check("a non-UUID customer_id in a path parameter is also rejected with 422, not 500", resp.status_code == 422)

    resp = client.post("/pilot/hardship/requests", json={
        "customer_id": "also-not-a-uuid", "reason": "test", "vulnerability_indicators": [], "requested_by": "test",
    }, headers=auth(customer_key.raw_key))
    check("a non-UUID customer_id in hardship requests is rejected with 422", resp.status_code == 422)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED:")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("ALL CHECKS PASSED -- pilot API auth + business logic verified over real HTTP")


if __name__ == "__main__":
    asyncio.run(main())
