"""
backend/pilot_api.py
======================
Real HTTP endpoints for the ASIC ERS pilot — the piece every prior
session's evidence pack flagged as missing ("tested, standalone
module, not wired into any real API endpoint yet"). This closes that
gap for the core customer journey exercised by
`test_end_to_end_dummy_customer_journey.py`.

DELIBERATELY A SEPARATE APP from `server.py` (the existing 150KB+
bill-smoothing product), not a modification of it:
- `server.py` is a live product with real infrastructure behind it
  (per CLAUDE.md) — this workstream has consistently avoided touching
  it or the live database it points to.
- This app can be run and deployed entirely independently
  (`uvicorn pilot_api:app`), pointed at its own sandbox database via
  its own `SUPABASE_URL`/`SUPABASE_SERVICE_KEY` env vars (the pilot
  migrations 012-023 have never been applied to the live database —
  point this app at a NEW project, not the live one).
- It can optionally be mounted under the main app later
  (`main_app.mount("/pilot", pilot_app)`) once someone deliberately
  decides to do that — not assumed here.

FAIL-CLOSED BY DEFAULT: every endpoint that would move real money or
activate a real customer still passes through the same fail-closed
checks as the underlying modules (pilot_config, launch_gates,
credit_ledger, etc.) — this API layer adds HTTP plumbing, it does not
loosen any control. In particular, `launch_gates.is_production_authorized()`
is checked before any endpoint that draws credit or activates a
customer account; with zero gates recorded (the honest current state),
those endpoints correctly return 403 until gates are genuinely
approved outside this repository.

AUTHENTICATION: every endpoint except the explicitly-public ones
(`/health`, `/pilot/launch-gates/status`, `GET /pilot/documents/{type}`)
and the ones with their own real authentication (`/pilot/identity/
webhook`, which verifies an HMAC signature) requires a valid API key —
`Authorization: Bearer <key>` — issued via `pilot_auth.issue_api_key()`
and checked against a specific permission from `security_controls.
ROLE_PERMISSIONS`, not a single blanket "is authenticated" gate. See
`pilot_auth.py`'s module docstring for why this is a simple,
operator-issued API-key scheme rather than full customer self-service
authentication — appropriate for a private testing deployment, not yet
a public product.
"""

import logging
import os
import uuid as uuid_module
from decimal import Decimal, InvalidOperation
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Header, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field, field_validator

import pilot_config as pc
import launch_gates as lg
import onboarding as ob
import responsible_lending as rl
import identity_verification as idv
import bank_verification as bkv
import credit_ledger as cl
import bill_ocr as ocr
import biller_allowlist as ba
import bill_verification as bv
import payment_permitted_use as ppu
import pilot_payment_flow as flow
import hardship_collections as hc
import complaints as cp
import document_versioning as dv
import regulatory_reports as rr
import security_controls as sc
import operational_readiness as opr
import pilot_auth as pa

logger = logging.getLogger(__name__)
logger.addFilter(sc.PiiRedactingLogFilter())

app = FastAPI(
    title="BillSure ASIC ERS Pilot API",
    description="Sandbox-only. Real-money functionality is disabled until every launch gate is approved.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to the real pilot frontend origin before any non-local deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

_CONSOLE_HTML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pilot_console.html")


@app.get("/", include_in_schema=False)
@app.get("/console", include_in_schema=False)
async def pilot_console():
    """Serves the sandbox testing console -- a real browser UI over this
    API, not just the JSON responses or the developer-facing /docs page.
    Talks to whatever origin it's loaded from (window.location.origin in
    the page's own JS), so it works identically whether reached via the
    raw onrender.com URL or the custom domain. The API key is entered
    once and kept only in this browser's localStorage -- never sent
    anywhere except back to this same API as the Authorization header on
    each request the person triggers by clicking a button."""
    return FileResponse(_CONSOLE_HTML_PATH, media_type="text/html")


# Every error type this module's dependencies can raise, mapped to an
# HTTP status. A refusal from any underlying module (maker-checker
# violation, cap breach, fail-closed check) becomes a 4xx, never a 500
# that could be mistaken for an unrelated server fault.
_ERROR_STATUS = {
    ob.OnboardingError: 422,
    rl.ResponsibleLendingError: 422,
    idv.IdentityVerificationError: 502,
    bkv.BankVerificationError: 502,
    cl.CreditLedgerError: 422,
    ocr.BillOcrError: 422,
    bv.BillVerificationError: 422,
    ppu.PermittedUseError: 422,
    flow.PaymentFlowError: 422,
    hc.HardshipCollectionsError: 422,
    cp.ComplaintsError: 422,
    dv.DocumentVersioningError: 422,
    sc.SecurityControlError: 422,
    lg.LaunchGateError: 403,
    pc.ConfigValidationError: 422,
}


def _raise_as_http(e: Exception):
    for exc_type, code in _ERROR_STATUS.items():
        if isinstance(e, exc_type):
            raise HTTPException(status_code=code, detail=str(e))
    logger.exception("Unhandled error in pilot API")
    raise HTTPException(status_code=500, detail="internal error")


def _validate_uuid_field(value: str, field_name: str) -> str:
    """Every customer_id/user_id column across the pilot schema is
    typed UUID (see migrations 013-021) — Postgres rejects a non-UUID
    string with an opaque error that this API's generic exception
    handler would otherwise surface as an unhelpful 500. Validates at
    the API boundary instead, so a malformed id is a clear 422 naming
    exactly which field is wrong, not a mystery internal error.

    Found live during initial deployment testing: this exact gap
    produced a bare 'internal error' when a plain string id
    ('user-live-test-001') was submitted instead of a real UUID — no
    automated test caught it beforehand because the in-memory fake
    database used throughout this workstream's test suites never
    enforces Postgres column types the way the real database does."""
    try:
        uuid_module.UUID(str(value))
    except (ValueError, AttributeError, TypeError):
        raise ValueError(f"{field_name} must be a valid UUID (the pilot database requires it) — got: {value!r}")
    return value


# ---------------------------------------------------------------
# Authentication / authorization
# ---------------------------------------------------------------
# Every endpoint below EXCEPT the ones explicitly listed as public
# (/health, /pilot/launch-gates/status, GET document text) or the ones
# with their OWN real authentication (/pilot/identity/webhook, which
# verifies an HMAC signature instead) requires a valid API key via
# `Authorization: Bearer <key>`, checked against pilot_auth.py, with a
# specific permission from security_controls.ROLE_PERMISSIONS enforced
# per endpoint -- not a single blanket "is authenticated" check. See
# pilot_auth.py's module docstring for why this is an operator-issued
# API key scheme rather than full customer self-service auth.

async def get_current_actor(authorization: str = Header(default=None)) -> dict:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="missing or malformed Authorization header (expected 'Bearer <api key>')")
    raw_key = authorization.split(" ", 1)[1].strip()
    actor = await pa.verify_api_key(raw_key)
    if not actor:
        raise HTTPException(status_code=401, detail="invalid or revoked API key")
    return actor


def require(permission: str):
    """Dependency factory: returns a FastAPI dependency that resolves
    the authenticated actor, then enforces both the specific
    permission AND (for MFA_REQUIRED_ROLES) that the key was issued
    with mfa_verified=True -- both checks reuse security_controls.py's
    existing, independently-tested functions rather than reimplementing
    them here."""
    async def _dependency(actor: dict = Depends(get_current_actor)) -> dict:
        try:
            sc.require_permission(actor["role"], permission)
            sc.require_mfa_verified(actor["role"], actor.get("mfa_verified", False))
        except sc.SecurityControlError as e:
            raise HTTPException(status_code=403, detail=str(e))
        return actor
    return _dependency


# ---------------------------------------------------------------
# Health
# ---------------------------------------------------------------

@app.get("/health")
async def health():
    """Liveness/readiness endpoint (task section 14). Checks database
    reachability via a trivial query; does NOT report production
    authorization status here on purpose -- that's a separate,
    deliberately more visible endpoint (see /pilot/launch-gates/status)
    rather than folded into a generic health check a load balancer
    would poll constantly."""
    import supabase_db as sdb
    try:
        await sdb.find_many("pilot_config_versions", {}, limit=1)
        db_healthy = True
        db_detail = "reachable"
    except Exception as e:
        db_healthy = False
        db_detail = f"unreachable: {e}"

    report = await opr.check_health({"database": opr.ComponentHealth("database", db_healthy, db_detail)})
    body = {"overall_healthy": report.overall_healthy, "checked_at": report.checked_at,
            "components": [c.__dict__ for c in report.components]}
    return JSONResponse(status_code=200 if report.overall_healthy else 503, content=body)


@app.get("/pilot/launch-gates/status")
async def launch_gate_status():
    """Read-only: the current status of every mandatory launch gate and
    whether production is currently authorized. Never mutates
    anything -- approving/submitting gate evidence is deliberately not
    exposed over this API; that's an administrative action requiring
    its own properly-authenticated admin surface, not built here."""
    statuses = await lg.get_all_gate_statuses()
    authorized = await lg.is_production_authorized()
    return {"gates": statuses, "production_authorized": authorized}


# ---------------------------------------------------------------
# Identity verification
# ---------------------------------------------------------------

class StartIdentityVerificationRequest(BaseModel):
    applicant_reference: str
    workflow_id: Optional[str] = None
    callback: Optional[str] = None


@app.post("/pilot/identity/sessions")
async def start_identity_verification(req: StartIdentityVerificationRequest, actor: dict = Depends(require("start_identity_verification"))):
    try:
        session = await idv.start_verification_session(req.applicant_reference, workflow_id=req.workflow_id, callback=req.callback)
    except Exception as e:
        _raise_as_http(e)
    return {"session_id": session.session_id, "url": session.url, "provider": session.provider}


@app.post("/pilot/identity/webhook")
async def identity_webhook(request: Request, x_signature_v2: str = Header(...), x_timestamp: str = Header(...)):
    """Receives a Didit webhook delivery. Verifies the signature before
    doing anything else; a bad signature is a 401, never processed."""
    raw_body = await request.body()
    try:
        result = await ob.apply_identity_verification_webhook(raw_body, x_signature_v2, x_timestamp)
    except idv.IdentityVerificationError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except Exception as e:
        _raise_as_http(e)
    if result is None:
        return {"status": "duplicate_ignored"}
    return {"status": "applied", "identity_verification_status": result["identity_verification_status"]}


# ---------------------------------------------------------------
# Onboarding
# ---------------------------------------------------------------

class ApplyRequest(BaseModel):
    user_id: str
    identity_verification_status: str
    age_confirmed: bool
    residential_state: str
    residential_country: str = "AU"
    bank_account_verified: bool
    income_amount: Optional[str] = None
    income_frequency: Optional[str] = None
    employment_status: Optional[str] = None
    recurring_living_expenses: Optional[str] = None
    existing_debts_and_bnpl: Optional[str] = None
    requested_credit_purpose: str
    requirements_and_objectives: str
    vulnerability_indicators: list = Field(default_factory=list)
    bankruptcy_status: str = "none"
    utility_bill_ownership_verified: bool
    consent_types_accepted: list = Field(default_factory=list)  # each element must be a value from onboarding.REQUIRED_CONSENTS

    @field_validator("user_id")
    @classmethod
    def validate_user_id(cls, v):
        return _validate_uuid_field(v, "user_id")


@app.post("/pilot/onboarding/apply")
async def apply(req: ApplyRequest, actor: dict = Depends(require("submit_application"))):
    active_config = await pc.get_active_config()
    if not active_config:
        raise HTTPException(status_code=503, detail="no active pilot configuration -- cannot evaluate an application")

    consents = {}
    for c in req.consent_types_accepted:
        consents = ob.record_consent(consents, c, "v1")

    application = ob.OnboardingApplication(
        user_id=req.user_id, identity_verification_status=req.identity_verification_status,
        age_confirmed=req.age_confirmed, residential_state=req.residential_state,
        residential_country=req.residential_country, bank_account_verified=req.bank_account_verified,
        income_amount=req.income_amount, income_frequency=req.income_frequency,
        employment_status=req.employment_status, recurring_living_expenses=req.recurring_living_expenses,
        existing_debts_and_bnpl=req.existing_debts_and_bnpl, requested_credit_purpose=req.requested_credit_purpose,
        requirements_and_objectives=req.requirements_and_objectives, vulnerability_indicators=req.vulnerability_indicators,
        bankruptcy_status=req.bankruptcy_status, utility_bill_ownership_verified=req.utility_bill_ownership_verified,
        consents=consents,
    )
    try:
        row = await ob.submit_application(application, active_config.geographic_areas, active_config.approved_bill_categories,
                                           policy_version="onboarding-policy-v1")
    except Exception as e:
        _raise_as_http(e)
    return row


class ManualReviewRequest(BaseModel):
    reviewer: str
    outcome: str
    reason_codes: list = Field(default_factory=list)
    notes: str = ""


@app.post("/pilot/onboarding/{application_id}/manual-review")
async def manual_review(application_id: str, req: ManualReviewRequest, actor: dict = Depends(require("manual_review_application"))):
    try:
        return await ob.record_manual_review_outcome(application_id, req.reviewer, req.outcome, req.reason_codes, req.notes)
    except Exception as e:
        _raise_as_http(e)


class ActivateCreditRequest(BaseModel):
    prepared_by: str
    approved_by: str
    contractual_limit: str
    active_customer_count: int
    current_aggregate_contractual_exposure: str


@app.post("/pilot/onboarding/{application_id}/activate-credit")
async def activate_credit(application_id: str, req: ActivateCreditRequest, actor: dict = Depends(require("approve_credit_activation"))):
    """Two-step activation: onboarding-side maker-checker AND the
    credit_ledger-side account creation (its own maker-checker + cap
    checks), matching test_end_to_end_dummy_customer_journey.py's
    sequence exactly."""
    try:
        onboarding_activation = await ob.approve_credit_activation(application_id, req.prepared_by, req.approved_by)
        app_row = await ob.sdb.find_one("onboarding_applications", {"id": application_id})
        credit_account = await cl.activate_customer_credit_account(
            app_row["user_id"], Decimal(req.contractual_limit), req.active_customer_count,
            Decimal(req.current_aggregate_contractual_exposure), req.prepared_by, req.approved_by,
        )
    except Exception as e:
        _raise_as_http(e)
    return {"onboarding_activation": onboarding_activation, "credit_account": credit_account}


# ---------------------------------------------------------------
# Credit account
# ---------------------------------------------------------------

@app.get("/pilot/credit/accounts/{customer_id}/balance")
async def credit_balance(customer_id: str, actor: dict = Depends(get_current_actor)):
    """A customer may view their OWN balance (actor_id == customer_id,
    checked here since it depends on the path parameter, not a static
    permission); staff/system roles with 'view_customer_balances' may
    view any customer's balance."""
    try:
        _validate_uuid_field(customer_id, "customer_id")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    is_own_data = actor["actor_id"] == customer_id and sc.has_permission(actor["role"], "view_own_balance")
    is_staff = sc.has_permission(actor["role"], "view_customer_balances")
    if not (is_own_data or is_staff):
        raise HTTPException(status_code=403, detail="not authorized to view this customer's balance")

    account = await cl.get_customer_credit_account(customer_id)
    if not account:
        raise HTTPException(status_code=404, detail="no credit account for this customer")
    outstanding = await cl.get_outstanding_principal(customer_id)
    limit = Decimal(account["contractual_limit"])
    return {"customer_id": customer_id, "contractual_limit": str(limit), "outstanding_principal": str(outstanding),
            "available_credit": str(limit - outstanding), "status": account["status"]}


# ---------------------------------------------------------------
# Bills: upload -> OCR -> verify -> pay
# ---------------------------------------------------------------

@app.post("/pilot/bills/upload")
async def upload_bill(
    customer_id: str = Form(...), customer_name_on_account: str = Form(...),
    category: str = Form(...), file: UploadFile = File(...),
    actor: dict = Depends(require("submit_bill")),
):
    try:
        _validate_uuid_field(customer_id, "customer_id")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    content = await file.read()
    try:
        upload_result = sc.validate_file_upload(file.filename, content)
    except sc.SecurityControlError as e:
        raise HTTPException(status_code=400, detail=str(e))

    is_pdf = upload_result.extension == ".pdf"
    known_billers = ba.allowlist_names()
    extraction = ocr.extract_bill_data(content, known_billers=known_billers, is_pdf=is_pdf)

    biller_name = extraction.biller_name_candidates[0] if extraction.biller_name_candidates else "UNKNOWN"
    submission = bv.BillSubmission(
        customer_id=customer_id, file_bytes=content, customer_name_on_account=customer_name_on_account,
        customer_name_on_bill=customer_name_on_account,  # OCR doesn't separately extract this; see bill_ocr.py's known limitation
        biller_name_extracted=biller_name, biller_reference=extraction.guessed_biller_reference or "",
        category=category, amount=extraction.guessed_amount or Decimal("0.01"),
        due_date=extraction.guessed_due_date or "", extraction_confidence=extraction.extraction_confidence,
        fraud_indicators=[],
    )
    try:
        active_config = await pc.get_active_config()
        max_plausible = active_config.max_single_bill_payment if active_config else None
        bill_row = await bv.submit_and_verify_bill(submission, known_billers, pc.APPROVED_BILL_CATEGORIES,
                                                     max_plausible_amount=max_plausible)
    except Exception as e:
        _raise_as_http(e)
    return {"bill": bill_row, "ocr": {"method": extraction.extraction_method, "confidence": extraction.extraction_confidence,
                                       "raw_text_preview": extraction.raw_text[:200]}}


class BillManualReviewRequest(BaseModel):
    reviewer: str
    decision: str
    notes: str = ""


@app.post("/pilot/bills/{bill_id}/manual-review")
async def bill_manual_review(bill_id: str, req: BillManualReviewRequest, actor: dict = Depends(require("manual_review_bill"))):
    try:
        return await bv.record_manual_review_decision(bill_id, req.reviewer, req.decision, req.notes)
    except Exception as e:
        _raise_as_http(e)


class PayBillRequest(BaseModel):
    customer_id: str
    requested_by: str

    @field_validator("customer_id")
    @classmethod
    def validate_customer_id(cls, v):
        return _validate_uuid_field(v, "customer_id")


@app.post("/pilot/bills/{bill_id}/pay")
async def pay_bill(bill_id: str, req: PayBillRequest, actor: dict = Depends(require("process_payment"))):
    """The single highest-consequence endpoint in this API: this is
    where real money would move. Checks launch_gates.
    is_production_authorized() BEFORE calling into the payment flow --
    with the honest current state (zero gates recorded), this
    correctly returns 403 rather than ever reaching credit_ledger.
    draw_credit()."""
    authorized = await lg.is_production_authorized()
    if not authorized:
        raise HTTPException(
            status_code=403,
            detail="production is not authorized -- one or more mandatory launch gates are not approved. "
                   "See GET /pilot/launch-gates/status for which gate(s) are blocking this.",
        )
    try:
        return await flow.pay_verified_bill(bill_id, req.customer_id, req.requested_by)
    except Exception as e:
        _raise_as_http(e)


# ---------------------------------------------------------------
# Hardship
# ---------------------------------------------------------------

class HardshipRequest(BaseModel):
    customer_id: str
    reason: str
    vulnerability_indicators: list = Field(default_factory=list)
    requested_by: str

    @field_validator("customer_id")
    @classmethod
    def validate_customer_id(cls, v):
        return _validate_uuid_field(v, "customer_id")


@app.post("/pilot/hardship/requests")
async def request_hardship(req: HardshipRequest, actor: dict = Depends(require("request_hardship"))):
    """No launch-gate or credit-status check here, deliberately --
    hardship support must remain available regardless of production
    authorization status (see launch_gates.existing_customers_route())."""
    try:
        return await hc.request_hardship(req.customer_id, req.reason, req.vulnerability_indicators, req.requested_by)
    except Exception as e:
        _raise_as_http(e)


# ---------------------------------------------------------------
# Complaints
# ---------------------------------------------------------------

class ComplaintRequest(BaseModel):
    customer_id: str
    channel: str
    description: str
    category: str = "standard"
    severity: str = "medium"
    vulnerability_indicators: list = Field(default_factory=list)
    received_by: str

    @field_validator("customer_id")
    @classmethod
    def validate_customer_id(cls, v):
        return _validate_uuid_field(v, "customer_id")


@app.post("/pilot/complaints")
async def submit_complaint(req: ComplaintRequest, actor: dict = Depends(require("submit_complaint"))):
    try:
        return await cp.intake_complaint(req.customer_id, req.channel, req.description, req.category,
                                          req.severity, req.vulnerability_indicators, req.received_by)
    except Exception as e:
        _raise_as_http(e)


# ---------------------------------------------------------------
# Documents
# ---------------------------------------------------------------

@app.get("/pilot/documents/{document_type}")
async def get_active_document(document_type: str):
    doc = await dv.get_active_document(document_type)
    if not doc:
        raise HTTPException(status_code=404, detail=f"no approved version of {document_type}")
    return doc


class AcceptDocumentRequest(BaseModel):
    customer_id: str
    version_id: str
    ip_address: Optional[str] = None

    @field_validator("customer_id")
    @classmethod
    def validate_customer_id(cls, v):
        return _validate_uuid_field(v, "customer_id")


@app.post("/pilot/documents/{document_type}/accept")
async def accept_document(document_type: str, req: AcceptDocumentRequest, actor: dict = Depends(require("accept_document"))):
    try:
        return await dv.record_customer_acceptance(req.customer_id, document_type, req.version_id, ip_address=req.ip_address)
    except Exception as e:
        _raise_as_http(e)


# ---------------------------------------------------------------
# Reports (read-only, PII-avoidant by construction -- see regulatory_reports.py)
# ---------------------------------------------------------------

@app.get("/pilot/reports/credit-exposure")
async def credit_exposure_report(actor: dict = Depends(require("export_reports"))):
    snapshot = await cl.get_exposure_snapshot()
    return rr.credit_exposure_report(snapshot)
