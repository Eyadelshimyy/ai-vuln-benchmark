from pathlib import Path
from dotenv import load_dotenv

# MUST run before any local module import below -- several of them
# (stripe_collections.py's STRIPE_API_KEY, utils/auth.py's JWT_SECRET/
# ENCRYPTION_KEY/RESEND_API_KEY) read os.environ at their own module
# top-level, not lazily inside a function. If load_dotenv() ran after
# those imports (as it used to, at the bottom of this import block), a
# .env file's values would silently never reach them -- the module-level
# variable is already bound to "" by the time .env loads, and nothing
# re-reads it later. Found this while testing a real STRIPE_API_KEY
# locally: it kept resolving to "not set" despite being in .env.
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

from fastapi import FastAPI, APIRouter, HTTPException, Depends, status, UploadFile, File, Form, Request, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse, JSONResponse
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import os
import logging
import asyncio
import csv
import secrets
from typing import List, Optional, Dict
from pydantic import BaseModel
import uuid
from decimal import Decimal
from datetime import datetime, timezone, timedelta
import base64
import re
import io
import httpx
import pdfplumber
from PIL import Image
from cryptography.fernet import Fernet
from emergentintegrations.llm.chat import LlmChat, UserMessage, ImageContent
from emergentintegrations.payments.stripe.checkout import StripeCheckout, CheckoutSessionResponse, CheckoutStatusResponse, CheckoutSessionRequest
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import supabase_db as sdb
import stripe
import ledger
import reconciliation
import payment_runs
import stripe_collections

# Models
from models.schemas import (
    UserRegister, UserLogin, User, Bill, BillCreate, BankDetails, BankDetailsCreate,
    PaymentStructure, PaymentStructureCreate, BillUpload, ParsedBillData,
    DirectDebitRequest, DirectDebitRequestCreate, ProviderConnection, ProviderConnectionCreate,
    Transaction, MockPayment, PaymentMethod, PaymentMethodCreate, AdminPayBillRequest,
)

# Utils
from utils.auth import (
    hash_password, verify_password, create_access_token, decode_token,
    get_current_user, get_admin_user, encrypt_field, decrypt_field,
    send_email, RESEND_API_KEY, SENDER_EMAIL, get_supabase_admin,
    security, pwd_context, SECRET_KEY, ALGORITHM, TOKEN_EXPIRE_HOURS,
)

# Create the main app
app = FastAPI()
api_router = APIRouter(prefix="/api")

# Rate limiter (brute force protection)
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Too many requests. Please try again later."}
    )


# Security headers middleware
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response


app.add_middleware(SecurityHeadersMiddleware)


# Routes
@api_router.get("/")
async def root():
    return {"message": "BillEasyPay API is running"}

# Auth routes

@api_router.post("/auth/register")
@limiter.limit("5/minute")
async def register(request: Request, user_data: UserRegister):
    # Check if user exists in our DB
    existing_user = await sdb.find_one("users", {"email": user_data.email})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    # Create Supabase Auth user
    sb = get_supabase_admin()
    supabase_uid = None
    access_token = None
    if sb:
        try:
            auth_res = sb.auth.admin.create_user({
                "email": user_data.email,
                "password": user_data.password,
                "email_confirm": True,
            })
            supabase_uid = auth_res.user.id
            # Ensure password is set correctly
            sb.auth.admin.update_user_by_id(supabase_uid, {"password": user_data.password})
            # Sign in to get access token
            sign_in = sb.auth.sign_in_with_password({
                "email": user_data.email,
                "password": user_data.password,
            })
            access_token = sign_in.session.access_token
        except Exception as e:
            logger.warning(f"Supabase Auth create failed: {e}")
            # Continue with custom JWT fallback

    # Create app user record
    hashed_password = hash_password(user_data.password)
    user = User(
        email=user_data.email,
        full_name=user_data.full_name,
        phone=user_data.phone
    )
    user_dict = user.model_dump()
    user_dict["password"] = hashed_password
    if supabase_uid:
        user_dict["supabase_uid"] = supabase_uid

    await sdb.insert_one("users", user_dict)

    # Use Supabase token if available, else custom JWT
    if not access_token:
        access_token = create_access_token({"user_id": user.id, "email": user.email})

    return {"token": access_token, "user": user}

@api_router.post("/auth/login")
@limiter.limit("10/minute")
async def login(request: Request, credentials: UserLogin):
    # Try Supabase Auth sign-in first
    sb = get_supabase_admin()
    access_token = None
    if sb:
        try:
            sign_in = sb.auth.sign_in_with_password({
                "email": credentials.email,
                "password": credentials.password,
            })
            access_token = sign_in.session.access_token
            supabase_uid = sign_in.user.id

            # Link supabase_uid if not yet linked
            user = await sdb.find_one("users", {"email": credentials.email})
            if user and not user.get("supabase_uid"):
                await sdb.update_one("users", {"id": user["id"]}, {"$set": {"supabase_uid": supabase_uid}})
        except Exception as e:
            logger.warning(f"Supabase sign-in failed for {credentials.email}: {e}")

    # Lookup user in our DB
    user = await sdb.find_one("users", {"email": credentials.email})
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    # If Supabase sign-in failed, verify password locally
    if not access_token:
        if not verify_password(credentials.password, user.get("password", "")):
            raise HTTPException(status_code=401, detail="Invalid email or password")

        # Backfill: create Supabase Auth user with the correct plaintext password
        if sb and not user.get("supabase_uid"):
            try:
                auth_res = sb.auth.admin.create_user({
                    "email": credentials.email,
                    "password": credentials.password,
                    "email_confirm": True,
                })
                supabase_uid = auth_res.user.id
                # Ensure password is set correctly
                sb.auth.admin.update_user_by_id(supabase_uid, {"password": credentials.password})
                await sdb.update_one("users", {"id": user["id"]}, {"$set": {"supabase_uid": supabase_uid}})
                # Now sign in to get Supabase token
                sign_in = sb.auth.sign_in_with_password({
                    "email": credentials.email,
                    "password": credentials.password,
                })
                access_token = sign_in.session.access_token
                logger.info(f"Backfilled Supabase Auth for {credentials.email}")
            except Exception as e:
                err_msg = str(e)
                if "already been registered" in err_msg or "already exists" in err_msg.lower():
                    # User exists in Supabase Auth but not linked — find and link
                    try:
                        sb_users = sb.auth.admin.list_users()
                        for sb_user in sb_users:
                            if sb_user.email == credentials.email:
                                sb.auth.admin.update_user_by_id(sb_user.id, {"password": credentials.password})
                                await sdb.update_one("users", {"id": user["id"]}, {"$set": {"supabase_uid": sb_user.id}})
                                sign_in = sb.auth.sign_in_with_password({
                                    "email": credentials.email,
                                    "password": credentials.password,
                                })
                                access_token = sign_in.session.access_token
                                logger.info(f"Linked and updated Supabase Auth for {credentials.email}")
                                break
                    except Exception:
                        pass
                else:
                    logger.debug(f"Supabase backfill on login: {e}")

        if not access_token:
            access_token = create_access_token({"user_id": user["id"], "email": user["email"]})

    user.pop("password", None)
    return {"token": access_token, "user": user}

@api_router.post("/auth/forgot-password")
@limiter.limit("3/minute")
async def forgot_password(request: Request, body: dict):
    """Send password reset email via Supabase Auth."""
    email = body.get("email", "")
    sb = get_supabase_admin()
    if not sb:
        raise HTTPException(status_code=503, detail="Password reset not available")

    user = await sdb.find_one("users", {"email": email})
    if not user:
        return {"message": "If that email is registered, a reset link has been sent."}

    try:
        sb.auth.admin.generate_link({
            "type": "recovery",
            "email": email,
        })
    except Exception as e:
        logger.warning(f"Password reset failed: {e}")

    return {"message": "If that email is registered, a reset link has been sent."}

@api_router.get("/auth/me")
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    current_user.pop("password", None)
    return current_user

# Bill routes
@api_router.post("/bills", response_model=Bill)
async def create_bill(bill_data: BillCreate, current_user: dict = Depends(get_current_user)):
    bill = Bill(
        user_id=current_user["id"],
        **bill_data.model_dump()
    )
    bill_dict = bill.model_dump()
    await sdb.insert_one("bills", bill_dict)
    return bill

@api_router.get("/bills", response_model=List[Bill])
async def get_bills(current_user: dict = Depends(get_current_user)):
    bills = await sdb.find_many("bills", {"user_id": current_user["id"]})
    return bills

@api_router.get("/bills/{bill_id}", response_model=Bill)
async def get_bill(bill_id: str, current_user: dict = Depends(get_current_user)):
    bill = await sdb.find_one("bills", {"id": bill_id, "user_id": current_user["id"]})
    if not bill:
        raise HTTPException(status_code=404, detail="Bill not found")
    return bill

@api_router.put("/bills/{bill_id}", response_model=Bill)
async def update_bill(bill_id: str, bill_data: BillCreate, current_user: dict = Depends(get_current_user)):
    result = await sdb.update_one("bills",
        {"id": bill_id, "user_id": current_user["id"]},
        {"$set": bill_data.model_dump()}
    )
    if not result:
        raise HTTPException(status_code=404, detail="Bill not found")

    updated_bill = await sdb.find_one("bills", {"id": bill_id, "user_id": current_user["id"]})
    return updated_bill

@api_router.delete("/bills/{bill_id}")
async def delete_bill(bill_id: str, current_user: dict = Depends(get_current_user)):
    result = await sdb.delete_one("bills", {"id": bill_id, "user_id": current_user["id"]})
    if not result:
        raise HTTPException(status_code=404, detail="Bill not found")
    return {"message": "Bill deleted successfully"}

# Payment structure routes
@api_router.post("/payment-structure", response_model=PaymentStructure)
async def create_payment_structure(data: PaymentStructureCreate, current_user: dict = Depends(get_current_user)):
    # Get all bills for the user
    bills = await sdb.find_many("bills", {"user_id": current_user["id"]})
    
    # Calculate total yearly bills based on frequency
    total_yearly = 0
    for bill in bills:
        if bill["frequency"] == "monthly":
            total_yearly += bill["amount"] * 12
        elif bill["frequency"] == "quarterly":
            total_yearly += bill["amount"] * 4
        elif bill["frequency"] == "yearly":
            total_yearly += bill["amount"]
    
    total_monthly = total_yearly / 12
    
    # Calculate contribution amount based on payment frequency
    if data.payment_frequency == "weekly":
        contribution = total_yearly / 52  # 52 weeks in a year
        days_until_next = 7
    elif data.payment_frequency == "fortnightly":
        contribution = total_yearly / 26  # 26 fortnights in a year
        days_until_next = 14
    else:  # monthly
        contribution = total_yearly / 12  # 12 months
        days_until_next = 30
    
    next_deduction = (datetime.now(timezone.utc) + timedelta(days=days_until_next)).isoformat()
    
    payment_structure = PaymentStructure(
        user_id=current_user["id"],
        payment_frequency=data.payment_frequency,
        total_yearly_bills=total_yearly,
        total_monthly_bills=total_monthly,
        contribution_amount=contribution,
        next_deduction_date=next_deduction,
        auto_deduct_enabled=data.auto_deduct_enabled
    )
    
    # Delete existing structure
    await sdb.delete_many("payment_structures", {"user_id": current_user["id"]})
    
    # Create new structure
    await sdb.insert_one("payment_structures", payment_structure.model_dump())
    
    return payment_structure

@api_router.get("/payment-structure", response_model=PaymentStructure)
async def get_payment_structure(current_user: dict = Depends(get_current_user)):
    structure = await sdb.find_one("payment_structures", {"user_id": current_user["id"]})
    if not structure:
        raise HTTPException(status_code=404, detail="Payment structure not set up")
    return structure

@api_router.put("/payment-structure/toggle-auto-deduct")
async def toggle_auto_deduct(current_user: dict = Depends(get_current_user)):
    structure = await sdb.find_one("payment_structures", {"user_id": current_user["id"]})
    if not structure:
        raise HTTPException(status_code=404, detail="Payment structure not set up")
    
    new_status = not structure.get("auto_deduct_enabled", True)
    await sdb.update_one("payment_structures", {"user_id": current_user["id"]}, {"$set": {"auto_deduct_enabled": new_status}})
    
    return {"auto_deduct_enabled": new_status, "message": f"Auto-deduction {'enabled' if new_status else 'disabled'}"}

# Bank Details routes
@api_router.post("/bank-details")
async def add_bank_details(bank_data: BankDetailsCreate, current_user: dict = Depends(get_current_user)):
    bank_details = BankDetails(
        user_id=current_user["id"],
        **bank_data.model_dump()
    )

    # Set other bank accounts as non-primary if this is primary
    if bank_details.is_primary:
        await sdb.update_many("bank_details", {"user_id": current_user["id"]}, {"$set": {"is_primary": False}})

    bd_dict = bank_details.model_dump()
    # Encrypt sensitive fields before storing
    bd_dict["account_number"] = encrypt_field(bd_dict["account_number"])
    bd_dict["routing_number"] = encrypt_field(bd_dict["routing_number"])
    await sdb.insert_one("bank_details", bd_dict)

    # Return masked version (never return encrypted or raw values)
    raw_acct = bank_data.account_number
    raw_rout = bank_data.routing_number
    bd_dict["account_number"] = "****" + raw_acct[-4:]
    bd_dict["routing_number"] = "****" + raw_rout[-4:]
    return bd_dict

@api_router.get("/bank-details")
async def get_bank_details(current_user: dict = Depends(get_current_user)):
    bank_accounts = await sdb.find_many("bank_details", {"user_id": current_user["id"]})
    # Decrypt then mask account numbers for security (show only last 4 digits)
    for account in bank_accounts:
        raw_acct = decrypt_field(account.get("account_number", ""))
        raw_rout = decrypt_field(account.get("routing_number", ""))
        account["account_number"] = "****" + raw_acct[-4:] if len(raw_acct) >= 4 else "****"
        account["routing_number"] = "****" + raw_rout[-4:] if len(raw_rout) >= 4 else "****"
    return bank_accounts

@api_router.delete("/bank-details/{bank_id}")
async def delete_bank_details(bank_id: str, current_user: dict = Depends(get_current_user)):
    result = await sdb.delete_one("bank_details", {"id": bank_id, "user_id": current_user["id"]})
    if not result:
        raise HTTPException(status_code=404, detail="Bank account not found")
    return {"message": "Bank account deleted successfully"}

# Bill Upload and Parsing route
@api_router.post("/bills/upload-and-parse")
async def upload_and_parse_bill(current_user: dict = Depends(get_current_user)):
    # This endpoint will receive the parsed data from frontend (Tesseract runs client-side)
    # Frontend will send extracted text and we'll parse it here
    return {"message": "Bill parsing handled on client-side with Tesseract.js"}

@api_router.post("/bills/save-parsed")
async def save_parsed_bill(bill_data: BillCreate, current_user: dict = Depends(get_current_user)):
    # Save bill from parsed data
    bill = Bill(
        user_id=current_user["id"],
        **bill_data.model_dump()
    )
    bill_dict = bill.model_dump()
    await sdb.insert_one("bills", bill_dict)
    return bill

# Automatic bill payment scheduler (simulated)
@api_router.post("/bills/process-auto-payments")
async def process_auto_payments(current_user: dict = Depends(get_current_user)):
    """
    Process automatic bill payments for bills due within next 3 days
    In production, this would be a scheduled job (cron/celery)
    """
    # Get pending bills due within next 3 days
    today = datetime.now(timezone.utc)
    three_days_later = today + timedelta(days=3)
    
    bills = await sdb.find_many("bills", {"user_id": current_user["id"], "status": "pending"})
    
    paid_bills = []
    failed_bills = []
    
    user = await sdb.find_one("users", {"id": current_user["id"]})
    
    for bill in bills:
        bill_due_date = datetime.fromisoformat(bill["due_date"].replace('Z', '+00:00'))
        
        if bill_due_date <= three_days_later:
            # Check wallet balance
            if user["wallet_balance"] >= bill["amount"]:
                # Create transaction
                transaction = Transaction(
                    user_id=current_user["id"],
                    type="bill_payment",
                    amount=bill["amount"],
                    description=f"Auto-payment for {bill['category']} - {bill['provider']}"
                )
                await sdb.insert_one("transactions", transaction.model_dump())
                
                # Update wallet balance (atomic — avoids lost updates under concurrent requests)
                await sdb.increment_wallet_balance(current_user["id"], -bill["amount"])

                # Update bill status
                await sdb.update_one("bills", {"id": bill["id"]}, {"$set": {"status": "paid"}})
                
                paid_bills.append(bill)
                user["wallet_balance"] -= bill["amount"]
            else:
                failed_bills.append(bill)
    
    return {
        "paid_count": len(paid_bills),
        "failed_count": len(failed_bills),
        "paid_bills": paid_bills,
        "failed_bills": failed_bills,
        "message": f"Processed {len(paid_bills)} bills successfully, {len(failed_bills)} failed due to insufficient balance"
    }

# Transaction routes
@api_router.post("/transactions/deposit")
async def deposit_to_wallet(payment: MockPayment, current_user: dict = Depends(get_current_user)):
    # SECURITY: this is a mock payment endpoint — it credits real wallet
    # balance for any amount the caller requests with no payment collected
    # at all. Any logged-in user could mint themselves free money by
    # calling it. Gated behind an explicit opt-in env var (same pattern as
    # SEED_DEMO_DATA) so it keeps working for local dev/testing without
    # Stripe configured, but is structurally impossible to hit unless
    # someone deliberately enables it — never set ALLOW_MOCK_PAYMENTS=true
    # in production. Use POST /payments/checkout (real Stripe) instead once
    # Stripe is configured.
    if os.environ.get("ALLOW_MOCK_PAYMENTS", "false").lower() != "true":
        raise HTTPException(status_code=403, detail="Mock payments are disabled. Set ALLOW_MOCK_PAYMENTS=true for local/dev testing only — never in production.")

    transaction = Transaction(
        user_id=current_user["id"],
        type="deposit",
        amount=payment.amount,
        description=f"Deposit via {payment.payment_method}"
    )
    await sdb.insert_one("transactions", transaction.model_dump())

    # Update wallet balance (atomic)
    await sdb.increment_wallet_balance(current_user["id"], payment.amount)

    return {"message": "Deposit successful", "transaction": transaction}

@api_router.post("/transactions/pay-bill/{bill_id}")
async def pay_bill(bill_id: str, current_user: dict = Depends(get_current_user)):
    # Get bill
    bill = await sdb.find_one("bills", {"id": bill_id, "user_id": current_user["id"]})
    if not bill:
        raise HTTPException(status_code=404, detail="Bill not found")
    
    if bill["status"] == "paid":
        raise HTTPException(status_code=400, detail="Bill already paid")
    
    # Check wallet balance
    user = await sdb.find_one("users", {"id": current_user["id"]})
    if user["wallet_balance"] < bill["amount"]:
        raise HTTPException(status_code=400, detail="Insufficient wallet balance")
    
    # Create transaction
    transaction = Transaction(
        user_id=current_user["id"],
        type="bill_payment",
        amount=bill["amount"],
        description=f"Payment for {bill['category']} - {bill['provider']}"
    )
    await sdb.insert_one("transactions", transaction.model_dump())

    # Update wallet balance (atomic)
    await sdb.increment_wallet_balance(current_user["id"], -bill["amount"])

    # Update bill status
    await sdb.update_one("bills", {"id": bill_id}, {"$set": {"status": "paid"}})

    return {"message": "Bill paid successfully", "transaction": transaction}

@api_router.get("/transactions", response_model=List[Transaction])
async def get_transactions(current_user: dict = Depends(get_current_user)):
    transactions = await sdb.find_many("transactions", {"user_id": current_user["id"]}, order_by="created_at", order_desc=True)
    return transactions

# Dashboard stats
@api_router.get("/dashboard/stats")
async def get_dashboard_stats(current_user: dict = Depends(get_current_user)):
    bills = await sdb.find_many("bills", {"user_id": current_user["id"]})
    transactions = await sdb.find_many("transactions", {"user_id": current_user["id"]})
    
    total_bills = len(bills)
    pending_bills = len([b for b in bills if b["status"] == "pending"])
    paid_bills = len([b for b in bills if b["status"] == "paid"])
    total_bill_amount = sum(b["amount"] for b in bills if b["status"] == "pending")
    
    # Calculate yearly prediction
    total_yearly_prediction = 0
    for bill in bills:
        if bill["frequency"] == "monthly":
            total_yearly_prediction += bill["amount"] * 12
        elif bill["frequency"] == "quarterly":
            total_yearly_prediction += bill["amount"] * 4
        elif bill["frequency"] == "yearly":
            total_yearly_prediction += bill["amount"]
    
    # Get bills due soon (next 7 days)
    today = datetime.now(timezone.utc)
    seven_days_later = today + timedelta(days=7)
    bills_due_soon = []
    
    for bill in bills:
        if bill["status"] == "pending":
            try:
                bill_due_date = datetime.fromisoformat(bill["due_date"].replace('Z', '+00:00'))
                if bill_due_date.tzinfo is None:
                    bill_due_date = bill_due_date.replace(tzinfo=timezone.utc)
                if bill_due_date <= seven_days_later:
                    bills_due_soon.append(bill)
            except Exception as e:
                print(f"Error parsing date for bill {bill.get('id')}: {e}")
    
    return {
        "wallet_balance": current_user["wallet_balance"],
        "total_bills": total_bills,
        "pending_bills": pending_bills,
        "paid_bills": paid_bills,
        "total_bill_amount": total_bill_amount,
        "total_yearly_prediction": total_yearly_prediction,
        "bills_due_soon": len(bills_due_soon),
        "bills_due_soon_list": bills_due_soon,
        "recent_transactions": transactions[:5]
    }

EMERGENT_LLM_KEY = os.environ.get('EMERGENT_LLM_KEY', '')


def build_bill_email(notification_type: str, user_name: str, bill_provider: str, bill_amount: float, due_date: str, message: str) -> str:
    """Build styled HTML email for bill notifications."""
    color = "#DC2626" if notification_type == "overdue" else "#D97706" if notification_type == "upcoming" else "#2563EB"
    label = "OVERDUE" if notification_type == "overdue" else "DUE SOON" if notification_type == "upcoming" else "NOTICE"
    return f"""
    <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; background: #f8fafc; padding: 24px;">
      <div style="background: white; border-radius: 12px; overflow: hidden; border: 1px solid #e2e8f0;">
        <div style="background: #0f172a; padding: 20px 24px;">
          <h1 style="color: white; font-size: 20px; margin: 0;">BillSure</h1>
        </div>
        <div style="padding: 24px;">
          <div style="display: inline-block; background: {color}15; color: {color}; font-size: 11px; font-weight: 700; padding: 4px 10px; border-radius: 4px; letter-spacing: 1px; margin-bottom: 16px;">{label}</div>
          <p style="color: #334155; font-size: 15px; margin: 0 0 8px;">Hi {user_name},</p>
          <p style="color: #64748b; font-size: 14px; line-height: 1.6; margin: 0 0 20px;">{message}</p>
          <table style="width: 100%; border-collapse: collapse; margin: 16px 0;">
            <tr style="border-bottom: 1px solid #e2e8f0;">
              <td style="padding: 10px 0; color: #94a3b8; font-size: 13px;">Provider</td>
              <td style="padding: 10px 0; color: #0f172a; font-size: 14px; font-weight: 600; text-align: right;">{bill_provider}</td>
            </tr>
            <tr style="border-bottom: 1px solid #e2e8f0;">
              <td style="padding: 10px 0; color: #94a3b8; font-size: 13px;">Amount</td>
              <td style="padding: 10px 0; color: {color}; font-size: 16px; font-weight: 700; text-align: right;">${bill_amount:.2f}</td>
            </tr>
            <tr>
              <td style="padding: 10px 0; color: #94a3b8; font-size: 13px;">Due Date</td>
              <td style="padding: 10px 0; color: #0f172a; font-size: 14px; text-align: right;">{due_date}</td>
            </tr>
          </table>
          <a href="https://www.billsure.com.au/dashboard" style="display: block; text-align: center; background: #0f172a; color: white; padding: 12px; border-radius: 8px; text-decoration: none; font-size: 14px; font-weight: 600; margin-top: 20px;">View Dashboard</a>
        </div>
        <div style="padding: 16px 24px; background: #f8fafc; border-top: 1px solid #e2e8f0;">
          <p style="color: #94a3b8; font-size: 11px; margin: 0; text-align: center;">BillSure &middot; www.billsure.com.au &middot; Australian Owned</p>
        </div>
      </div>
    </div>"""


def build_low_balance_email(user_name: str, wallet: float, pending_total: float) -> str:
    """Build styled HTML email for low wallet balance."""
    return f"""
    <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; background: #f8fafc; padding: 24px;">
      <div style="background: white; border-radius: 12px; overflow: hidden; border: 1px solid #e2e8f0;">
        <div style="background: #0f172a; padding: 20px 24px;">
          <h1 style="color: white; font-size: 20px; margin: 0;">BillSure</h1>
        </div>
        <div style="padding: 24px;">
          <div style="display: inline-block; background: #D9770615; color: #D97706; font-size: 11px; font-weight: 700; padding: 4px 10px; border-radius: 4px; letter-spacing: 1px; margin-bottom: 16px;">LOW BALANCE</div>
          <p style="color: #334155; font-size: 15px; margin: 0 0 8px;">Hi {user_name},</p>
          <p style="color: #64748b; font-size: 14px; line-height: 1.6; margin: 0 0 20px;">Your wallet balance may not cover your upcoming bills. Please consider topping up to avoid missed payments.</p>
          <table style="width: 100%; border-collapse: collapse; margin: 16px 0;">
            <tr style="border-bottom: 1px solid #e2e8f0;">
              <td style="padding: 10px 0; color: #94a3b8; font-size: 13px;">Wallet Balance</td>
              <td style="padding: 10px 0; color: #D97706; font-size: 16px; font-weight: 700; text-align: right;">${wallet:.2f}</td>
            </tr>
            <tr>
              <td style="padding: 10px 0; color: #94a3b8; font-size: 13px;">Pending Bills Total</td>
              <td style="padding: 10px 0; color: #0f172a; font-size: 14px; font-weight: 600; text-align: right;">${pending_total:.2f}</td>
            </tr>
          </table>
          <a href="https://www.billsure.com.au/dashboard/payment-plan" style="display: block; text-align: center; background: #2563EB; color: white; padding: 12px; border-radius: 8px; text-decoration: none; font-size: 14px; font-weight: 600; margin-top: 20px;">Top Up Wallet</a>
        </div>
        <div style="padding: 16px 24px; background: #f8fafc; border-top: 1px solid #e2e8f0;">
          <p style="color: #94a3b8; font-size: 11px; margin: 0; text-align: center;">BillSure &middot; www.billsure.com.au &middot; Australian Owned</p>
        </div>
      </div>
    </div>"""


async def extract_bill_with_vision(image_bytes: bytes, mime_type: str = "image/jpeg") -> dict:
    """
    Use GPT Vision to extract structured bill data from an image.
    Works for JPEG, PNG, and rendered PDF pages.
    """
    if not EMERGENT_LLM_KEY:
        return None

    b64_img = base64.b64encode(image_bytes).decode('utf-8')

    chat = LlmChat(
        api_key=EMERGENT_LLM_KEY,
        session_id=f"bill-extract-{uuid.uuid4().hex[:8]}",
        system_message="""You are an expert Australian bill data extractor. 
Extract the following fields from the bill image and return ONLY valid JSON (no markdown, no code fences):
{
  "provider": "Company/provider name",
  "category": "Electricity|Gas|Water|Internet|Mobile|Council|Insurance|Other",
  "account_number": "The account number (preserve all digits and spaces exactly as shown)",
  "biller_code": "The BPAY Biller Code (numeric, preserve spaces)",
  "reference_number": "The BPAY Reference Number (preserve all digits and spaces exactly as shown)",
  "amount": 0.00,
  "due_date": "YYYY-MM-DD",
  "frequency": "monthly|quarterly|yearly"
}
Rules:
- For account_number: Look near labels like "Account Number", "Account No", "Account #", "Acct". Copy the entire number including any spaces.
- For biller_code: Look near "Biller Code", "BPAY Biller Code", or in the BPAY section. It's usually 4-6 digits.
- For reference_number: Look near "BPAY Ref", "BPAY Reference", "Ref:", "Reference Number" in the BPAY section. Copy the entire number including any spaces.
- For amount: Look for "Total Amount Due", "Amount Due", "Total Due", "Pay This Amount", or the largest dollar amount.
- For due_date: Look for "Due Date", "Pay By", "Payment Due". Convert to YYYY-MM-DD format.
- If a field is not found, use null.
- Return ONLY the JSON object, no other text."""
    )
    chat.with_model("openai", "gpt-4o")

    image_content = ImageContent(image_base64=b64_img)
    user_msg = UserMessage(
        text="Extract all bill payment details from this image. Focus especially on the BPAY section for Biller Code and Reference Number. Preserve all digits and spaces in numbers exactly as they appear.",
        file_contents=[image_content]
    )

    try:
        response = await chat.send_message(user_msg)
        # Parse JSON from response
        import json as json_module
        # Clean up response - remove markdown code fences if present
        clean = response.strip()
        if clean.startswith("```"):
            clean = re.sub(r'^```(?:json)?\s*', '', clean)
            clean = re.sub(r'\s*```$', '', clean)
        data = json_module.loads(clean)
        return data
    except Exception as e:
        logger.warning(f"GPT Vision extraction failed: {e}")
        return None


def parse_bill_text_server(text: str) -> dict:
    """Advanced server-side bill text parsing with improved regex patterns"""
    parsed = {}
    text_lower = text.lower()
    lines = text.split('\n')

    # --- Amount extraction (priority order) ---
    amount_patterns = [
        r'(?:total\s*(?:amount\s*)?due|amount\s*due|balance\s*due|total\s*payable|amount\s*payable|pay\s*this\s*amount)[:\s]*\$?\s*([\d,]+\.?\d{0,2})',
        r'(?:total\s*charges?|total\s*new\s*charges?|new\s*charges?)[:\s]*\$?\s*([\d,]+\.?\d{0,2})',
        r'(?:please\s*pay)[:\s]*\$?\s*([\d,]+\.?\d{0,2})',
        r'(?:amount\s*owing|owing)[:\s]*\$?\s*([\d,]+\.?\d{0,2})',
    ]
    for pattern in amount_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            val = float(match.group(1).replace(',', ''))
            if 1 < val < 50000:
                parsed['amount'] = val
                break

    if 'amount' not in parsed:
        dollar_matches = re.findall(r'\$\s*([\d,]+\.\d{2})', text)
        amounts = [float(m.replace(',', '')) for m in dollar_matches if 5 < float(m.replace(',', '')) < 50000]
        if amounts:
            parsed['amount'] = max(amounts)

    # --- Due date extraction ---
    date_patterns = [
        r'(?:due\s*(?:date|by)?|pay\s*(?:by|before|on)|payment\s*due)[:\s]*(\d{1,2}[\s/\-\.]+\w{3,9}[\s/\-\.]+\d{2,4})',
        r'(?:due\s*(?:date|by)?|pay\s*(?:by|before|on)|payment\s*due)[:\s]*(\d{1,2}[\s/\-\.]+\d{1,2}[\s/\-\.]+\d{2,4})',
        r'(?:due\s*(?:date|by)?|pay\s*(?:by|before|on))[:\s]*(\w{3,9}\s+\d{1,2},?\s*\d{4})',
    ]
    for pattern in date_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            raw_date = match.group(1).strip()
            parsed_date = _try_parse_date(raw_date)
            if parsed_date:
                parsed['due_date'] = parsed_date
                break

    # --- Account number extraction (preserve spaces in numbers) ---
    account_patterns = [
        r'(?:account\s*(?:no\.?|number|#|num))[:\s]*([\d][\d\s\-]{3,25}[\d])',
        r'(?:acct\.?\s*(?:no\.?|#)?)[:\s]*([\d][\d\s\-]{3,25}[\d])',
        r'(?:customer\s*(?:no\.?|number|ref|reference|#))[:\s]*([\d][\d\s\-]{3,25}[\d])',
        r'(?:account\s*(?:no\.?|number|#|num))[:\s]*([A-Z0-9][\w\s\-]{3,25}[\w])',
    ]
    for pattern in account_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            val = match.group(1).strip()
            # Remove trailing non-alphanumeric but keep internal spaces
            val = re.sub(r'[\s\-]+$', '', val)
            if len(val) >= 4:
                parsed['account_number'] = val
                break

    # --- Provider detection ---
    known_providers = {
        'agl': 'AGL', 'origin': 'Origin Energy', 'energyaustralia': 'EnergyAustralia',
        'synergy': 'Synergy', 'alinta': 'Alinta Energy', 'simply energy': 'Simply Energy',
        'momentum': 'Momentum Energy', 'powershop': 'Powershop', 'red energy': 'Red Energy',
        'lumo': 'Lumo Energy', 'ergon': 'Ergon Energy', 'ausgrid': 'Ausgrid',
        'sydney water': 'Sydney Water', 'sa water': 'SA Water', 'yarra valley': 'Yarra Valley Water',
        'south east water': 'South East Water', 'telstra': 'Telstra', 'optus': 'Optus',
        'vodafone': 'Vodafone', 'tpg': 'TPG', 'nbn': 'NBN Co', 'dodo': 'Dodo',
        'iinet': 'iiNet', 'belong': 'Belong', 'aussie broadband': 'Aussie Broadband',
    }
    for key, name in known_providers.items():
        if key in text_lower:
            parsed['provider'] = name
            break

    if 'provider' not in parsed:
        for line in lines[:8]:
            clean = line.strip()
            if len(clean) > 3 and re.match(r'^[A-Za-z]', clean) and not re.match(r'^\d+$', clean):
                if not re.search(r'(tax\s*invoice|bill|statement|page|date|account)', clean, re.IGNORECASE):
                    parsed['provider'] = clean[:60]
                    break

    # --- Category detection ---
    category_keywords = {
        'Electricity': ['electric', 'power', 'kwh', 'kilowatt', 'energy charge', 'supply charge', 'tariff'],
        'Water': ['water', 'sewerage', 'drainage', 'water usage', 'water supply'],
        'Gas': ['gas', 'natural gas', 'gas usage', 'gas supply', 'mj ', 'megajoule'],
        'Internet': ['internet', 'broadband', 'nbn', 'wifi', 'data plan', 'download'],
        'Mobile': ['mobile', 'phone', 'call', 'sms', 'data', 'handset', 'sim'],
        'Council': ['council', 'rates', 'municipal', 'shire', 'city of'],
        'Insurance': ['insurance', 'premium', 'policy', 'cover', 'insurer'],
    }
    for category, keywords in category_keywords.items():
        if any(kw in text_lower for kw in keywords):
            parsed['category'] = category
            break

    # --- BPAY Biller Code extraction ---
    biller_code_patterns = [
        r'(?:biller\s*code|bpay\s*biller\s*code|bpay\s*code)[:\s]*(\d[\d\s]{2,10}\d)',
        r'(?:biller\s*code|bpay\s*biller\s*code|bpay\s*code)[:\s]*(\d{3,8})',
        r'biller[:\s]*(\d{4,8})',
    ]
    for pattern in biller_code_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            parsed['biller_code'] = match.group(1).strip()
            parsed['bpay_code'] = parsed['biller_code']
            break

    # --- BPAY Reference Number extraction (preserve spaces in numbers) ---
    # Priority 1: BPAY-specific reference
    bpay_ref_patterns = [
        r'(?:bpay\s*ref(?:erence)?\s*(?:no\.?|number|#)?)[:\s]*([\d][\d\s]{3,30}[\d])',
        r'(?:bpay\s*ref(?:erence)?)[:\s]*([A-Z0-9][\w\s\-]{3,30}[\w])',
        r'(?:^|\n)\s*ref(?:erence)?\s*(?:no\.?|number|#)?\s*[:\s]\s*([\d][\d\s]{5,30}[\d])',
    ]
    for pattern in bpay_ref_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            ref_val = match.group(1).strip()
            ref_val = re.sub(r'[\s\-]+$', '', ref_val)
            if len(ref_val) >= 4:
                parsed['reference_number'] = ref_val
                break

    # Priority 2: General reference number (fallback)
    if not parsed.get('reference_number'):
        general_ref_patterns = [
            r'(?:customer\s*ref(?:erence)?|payment\s*ref(?:erence)?|crn|your\s*ref(?:erence)?)[:\s]*([\d][\d\s\-]{3,30}[\d])',
            r'(?:ref(?:erence)?\s*(?:no\.?|number|#))[:\s]*([\d][\d\s\-]{3,30}[\d])',
        ]
        for pattern in general_ref_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                ref_val = match.group(1).strip()
                ref_val = re.sub(r'[\s\-]+$', '', ref_val)
                if len(ref_val) >= 4:
                    parsed['reference_number'] = ref_val
                    break

    # --- Frequency detection ---
    if any(w in text_lower for w in ['quarterly', 'quarter', '3 month', 'every 3']):
        parsed['frequency'] = 'quarterly'
    elif any(w in text_lower for w in ['annual', 'yearly', '12 month']):
        parsed['frequency'] = 'yearly'
    else:
        parsed['frequency'] = 'monthly'

    return parsed


def _try_parse_date(raw: str) -> Optional[str]:
    """Try to parse various date formats into YYYY-MM-DD"""
    from datetime import datetime as dt
    formats = [
        '%d/%m/%Y', '%d-%m-%Y', '%d.%m.%Y',
        '%d/%m/%y', '%d-%m-%y',
        '%d %B %Y', '%d %b %Y',
        '%B %d, %Y', '%b %d, %Y',
        '%d %B, %Y', '%d %b, %Y',
        '%m/%d/%Y', '%m-%d-%Y',
    ]
    cleaned = re.sub(r'\s+', ' ', raw.strip())
    for fmt in formats:
        try:
            parsed = dt.strptime(cleaned, fmt)
            return parsed.strftime('%Y-%m-%d')
        except ValueError:
            continue
    return None


MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB


@api_router.post("/bills/extract")
@limiter.limit("20/hour")
async def extract_bill_data(
    request: Request,
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """
    Extract bill data from any uploaded file (PDF, JPEG, PNG, etc.).
    Strategy:
    1. PDFs: Accurassi API → pdfplumber text → GPT Vision (for scanned PDFs)
    2. Images: GPT Vision (AI-powered extraction)
    3. All results go through regex refinement

    Rate-limited (20/hour per IP) and size-capped: this endpoint calls a paid
    GPT-4o Vision API per request, so without a limit it's a straightforward
    cost-exhaustion/DoS vector for an authenticated attacker looping uploads.
    """
    # Reject oversized uploads before doing any expensive parsing/AI-vision work.
    declared_size = request.headers.get("content-length")
    if declared_size and declared_size.isdigit() and int(declared_size) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large. Maximum upload size is 10MB.")

    file_content = await file.read()
    if len(file_content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large. Maximum upload size is 10MB.")

    file_type = (file.content_type or '').lower()
    filename = (file.filename or '').lower()

    # Detect file type from extension if content_type is unreliable
    is_pdf = 'pdf' in file_type or filename.endswith('.pdf')
    is_image = any(t in file_type for t in ['image', 'jpeg', 'jpg', 'png', 'webp']) or \
               any(filename.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'])

    extracted_text = ""
    extraction_method = "pdfplumber"

    try:
        # ===== PATH 1: PDF FILES =====
        if is_pdf:
            # Try pdfplumber text extraction
            try:
                with pdfplumber.open(io.BytesIO(file_content)) as pdf:
                    texts = []
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            texts.append(page_text)
                    extracted_text = '\n'.join(texts)
            except Exception as e:
                logger.warning(f"pdfplumber failed: {e}")

            # If pdfplumber got good text, use regex extraction
            if extracted_text and len(extracted_text.strip()) >= 20:
                parsed = parse_bill_text_server(extracted_text)
                parsed['extracted_text'] = extracted_text[:2000]
                parsed['extraction_method'] = 'pdfplumber'
                return parsed

            # Scanned/image-based PDF → render to image and use GPT Vision
            logger.info("PDF has no extractable text, trying GPT Vision...")
            try:
                import pypdfium2 as pdfium
                pdf_doc = pdfium.PdfDocument(file_content)
                if len(pdf_doc) > 0:
                    page = pdf_doc[0]
                    bitmap = page.render(scale=2)
                    pil_image = bitmap.to_pil()
                    img_buffer = io.BytesIO()
                    pil_image.save(img_buffer, format='JPEG', quality=85)
                    img_bytes = img_buffer.getvalue()
                    page.close()
                    pdf_doc.close()

                    vision_result = await extract_bill_with_vision(img_bytes, "image/jpeg")
                    if vision_result:
                        vision_result['extraction_method'] = 'ai_vision'
                        vision_result['extracted_text'] = 'Extracted via AI Vision (scanned PDF)'
                        # Ensure all expected fields exist
                        for field in ['category', 'provider', 'account_number', 'biller_code', 'reference_number', 'amount', 'due_date', 'frequency']:
                            if field not in vision_result:
                                vision_result[field] = None
                        return vision_result
            except Exception as e:
                logger.warning(f"pypdfium2/vision fallback failed: {e}")

            raise HTTPException(
                status_code=400,
                detail="Could not extract text from this PDF. Please try a clearer scan or enter details manually."
            )

        # ===== PATH 2: IMAGE FILES (JPEG, PNG, etc.) =====
        elif is_image:
            # Validate image first
            try:
                img = Image.open(io.BytesIO(file_content))
                img.verify()
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid image file. Please upload a valid JPEG, PNG, or PDF.")

            # Re-open image (verify() invalidates the object) and convert to JPEG for consistent processing
            img = Image.open(io.BytesIO(file_content))
            if img.mode in ('RGBA', 'P', 'LA'):
                img = img.convert('RGB')
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='JPEG', quality=85)
            img_bytes = img_buffer.getvalue()

            # Use GPT Vision to extract bill data from image
            vision_result = await extract_bill_with_vision(img_bytes, "image/jpeg")
            if vision_result:
                vision_result['extraction_method'] = 'ai_vision'
                vision_result['extracted_text'] = 'Extracted via AI Vision'
                for field in ['category', 'provider', 'account_number', 'biller_code', 'reference_number', 'amount', 'due_date', 'frequency']:
                    if field not in vision_result:
                        vision_result[field] = None
                return vision_result

            # GPT Vision unavailable — return manual entry prompt
            return {
                "category": None,
                "provider": None,
                "account_number": None,
                "biller_code": None,
                "reference_number": None,
                "amount": None,
                "due_date": None,
                "extracted_text": "Image uploaded but AI extraction unavailable. Please fill in the bill details manually.",
                "extraction_method": "manual",
                "requires_manual_entry": True
            }

        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a PDF, JPEG, or PNG file.")

    except HTTPException:
        raise
    except Exception as e:
        # Log full detail server-side only — don't echo internal exception
        # text (library internals, file paths, etc.) back to the client.
        logger.error(f"Bill extraction error: {e}")
        raise HTTPException(status_code=500, detail="Bill extraction failed. Please try again or enter details manually.")


# Direct Debit Request (DDR) Routes
@api_router.post("/direct-debit/create")
async def create_direct_debit_request(ddr_data: DirectDebitRequestCreate, current_user: dict = Depends(get_current_user)):
    """Create a new Direct Debit Request mandate"""
    
    # Validate BSB (Australian format: XXX-XXX)
    bsb_clean = ddr_data.bsb.replace("-", "").replace(" ", "")
    if len(bsb_clean) != 6 or not bsb_clean.isdigit():
        raise HTTPException(status_code=400, detail="Invalid BSB format. Must be 6 digits (XXX-XXX)")
    
    # Generate unique mandate reference
    mandate_ref = f"DDR-{uuid.uuid4().hex[:8].upper()}"
    
    # Create DDR data with formatted BSB
    ddr_dict = ddr_data.model_dump()
    formatted_bsb = bsb_clean[:3] + "-" + bsb_clean[3:]
    ddr_dict["bsb"] = formatted_bsb
    
    ddr = DirectDebitRequest(
        user_id=current_user["id"],
        mandate_reference=mandate_ref,
        **ddr_dict
    )
    
    ddr_store = ddr.model_dump()
    # Encrypt sensitive financial fields
    ddr_store["bsb"] = encrypt_field(formatted_bsb)
    ddr_store["account_number"] = encrypt_field(ddr_store["account_number"])
    ddr_store["provider_account_number"] = encrypt_field(ddr_store["provider_account_number"])
    await sdb.insert_one("direct_debit_requests", ddr_store)
    
    # Return masked version
    ddr_store["bsb"] = formatted_bsb[:3] + "-***"
    ddr_store["account_number"] = "****" + ddr_data.account_number[-4:]
    ddr_store["provider_account_number"] = "****" + ddr_data.provider_account_number[-4:]
    return ddr_store

@api_router.get("/direct-debit/mandates")
async def get_direct_debit_mandates(current_user: dict = Depends(get_current_user)):
    """Get all DDR mandates for the current user"""
    mandates = await sdb.find_many("direct_debit_requests", {"user_id": current_user["id"]})
    # Decrypt and mask sensitive fields
    for m in mandates:
        raw_bsb = decrypt_field(m.get("bsb", ""))
        raw_acct = decrypt_field(m.get("account_number", ""))
        raw_prov_acct = decrypt_field(m.get("provider_account_number", ""))
        m["bsb"] = raw_bsb[:3] + "-***" if len(raw_bsb) >= 3 else "***-***"
        m["account_number"] = "****" + raw_acct[-4:] if len(raw_acct) >= 4 else "****"
        m["provider_account_number"] = "****" + raw_prov_acct[-4:] if len(raw_prov_acct) >= 4 else "****"
    return mandates

@api_router.get("/direct-debit/mandate/{mandate_id}")
async def get_direct_debit_mandate(mandate_id: str, current_user: dict = Depends(get_current_user)):
    """Get a specific DDR mandate"""
    mandate = await sdb.find_one("direct_debit_requests", {"id": mandate_id, "user_id": current_user["id"]})
    if not mandate:
        raise HTTPException(status_code=404, detail="Mandate not found")
    # Decrypt and mask
    raw_bsb = decrypt_field(mandate.get("bsb", ""))
    raw_acct = decrypt_field(mandate.get("account_number", ""))
    raw_prov_acct = decrypt_field(mandate.get("provider_account_number", ""))
    mandate["bsb"] = raw_bsb[:3] + "-***" if len(raw_bsb) >= 3 else "***-***"
    mandate["account_number"] = "****" + raw_acct[-4:] if len(raw_acct) >= 4 else "****"
    mandate["provider_account_number"] = "****" + raw_prov_acct[-4:] if len(raw_prov_acct) >= 4 else "****"
    return mandate

@api_router.put("/direct-debit/mandate/{mandate_id}/cancel")
async def cancel_direct_debit_mandate(mandate_id: str, current_user: dict = Depends(get_current_user)):
    """Cancel a DDR mandate"""
    result = await sdb.update_one("direct_debit_requests", {"id": mandate_id, "user_id": current_user["id"]}, {"$set": {"status": "cancelled"}})
    
    if not result:
        raise HTTPException(status_code=404, detail="Mandate not found")
    
    return {"message": "Direct Debit mandate cancelled successfully", "mandate_id": mandate_id}

@api_router.post("/direct-debit/validate-bsb")
@limiter.limit("30/minute")
async def validate_bsb(request: Request, bsb: str, current_user: dict = Depends(get_current_user)):
    """Validate Australian BSB number"""
    bsb_clean = bsb.replace("-", "").replace(" ", "")

    if len(bsb_clean) != 6 or not bsb_clean.isdigit():
        return {"valid": False, "message": "BSB must be 6 digits"}

    # Basic BSB validation (in production, use a BSB lookup service)
    # First 2 digits indicate bank: 01-12 = major banks
    bank_names = {
        "01": "ANZ",
        "06": "Commonwealth Bank",
        "08": "NAB",
        "11": "Westpac",
        "03": "Bendigo Bank",
        "09": "Reserve Bank",
        "73": "Bank of Queensland"
    }
    
    bank_name = bank_names.get(bsb_clean[:2], "Other Bank")
    
    return {
        "valid": True,
        "formatted": bsb_clean[:3] + "-" + bsb_clean[3:],
        "bank_name": bank_name,
        "message": "Valid BSB"
    }

# Provider Connection Routes
@api_router.post("/provider/connect", response_model=ProviderConnection)
async def connect_provider(provider_data: ProviderConnectionCreate, current_user: dict = Depends(get_current_user)):
    """Connect to a utility provider"""
    
    provider = ProviderConnection(
        user_id=current_user["id"],
        **provider_data.model_dump()
    )
    
    # Check if connection already exists
    existing = await sdb.find_one("provider_connections", {
        "user_id": current_user["id"],
        "provider_name": provider_data.provider_name
    })
    
    if existing:
        raise HTTPException(status_code=400, detail="Provider already connected")
    
    await sdb.insert_one("provider_connections", provider.model_dump())
    
    return provider

@api_router.get("/provider/connections", response_model=List[ProviderConnection])
async def get_provider_connections(current_user: dict = Depends(get_current_user)):
    """Get all provider connections for the user"""
    connections = await sdb.find_many("provider_connections", {"user_id": current_user["id"]})
    return connections

@api_router.post("/provider/sync/{connection_id}")
async def sync_provider_bills(connection_id: str, current_user: dict = Depends(get_current_user)):
    """Sync bills from a connected provider"""
    
    connection = await sdb.find_one("provider_connections", {"id": connection_id, "user_id": current_user["id"]})
    
    if not connection:
        raise HTTPException(status_code=404, detail="Provider connection not found")
    
    # Fetch bills from provider API
    # This is a generic implementation - specific providers need specific API calls
    
    try:
        bills_fetched = []
        
        # Provider API sync is not yet implemented
        # Bills are currently added via manual upload + OCR/AI extraction
        
        # Update last sync time
        await sdb.update_one("provider_connections",
            {"id": connection_id},
            {"$set": {"last_sync": datetime.now(timezone.utc).isoformat()}}
        )
        
        return {
            "success": True,
            "message": f"Provider connection updated for {connection['provider_name']}. Use bill upload for new bills.",
            "bills_fetched": len(bills_fetched)
        }
        
    except Exception as e:
        logger.error(f"Provider sync failed for connection {connection_id}: {e}")
        raise HTTPException(status_code=500, detail="Provider sync failed. Please try again later.")

@api_router.delete("/provider/disconnect/{connection_id}")
async def disconnect_provider(connection_id: str, current_user: dict = Depends(get_current_user)):
    """Disconnect from a provider"""
    result = await sdb.delete_one("provider_connections", {"id": connection_id, "user_id": current_user["id"]})
    
    if not result:
        raise HTTPException(status_code=404, detail="Provider connection not found")
    
    return {"message": "Provider disconnected successfully"}

# Admin routes

@api_router.get("/admin/stats")
async def get_admin_stats(admin_user: dict = Depends(get_admin_user)):
    """Get overall platform statistics"""
    total_users = await sdb.count_documents("users", {})
    total_bills = await sdb.count_documents("bills", {})
    total_transactions = await sdb.count_documents("transactions", {})
    pending_bills = await sdb.count_documents("bills", {"status": "pending"})
    
    # Calculate total revenue (subscription fees)
    users = await sdb.find_many("users", {})
    total_monthly_revenue = sum(u.get("subscription_fee", 5.0) for u in users)
    
    return {
        "total_users": total_users,
        "total_bills": total_bills,
        "total_transactions": total_transactions,
        "pending_bills": pending_bills,
        "monthly_revenue": total_monthly_revenue
    }

@api_router.get("/admin/users")
async def get_all_users(admin_user: dict = Depends(get_admin_user)):
    """Get all users with their bill counts"""
    users = await sdb.find_many("users", exclude_fields=["password"])

    # Get bill counts per user (replaces MongoDB aggregation)
    all_bills = await sdb.find_many("bills")
    count_map = {}
    for b in all_bills:
        uid = b.get("user_id", "")
        count_map[uid] = count_map.get(uid, 0) + 1

    for user in users:
        user["bill_count"] = count_map.get(user["id"], 0)

    return users

@api_router.get("/admin/bulk-payment-report")
async def get_bulk_payment_report(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    provider: Optional[str] = None,
    report_type: str = "daily",  # daily, weekly, monthly, all
    admin_user: dict = Depends(get_admin_user)
):
    """
    Get bulk payment report for bills due within date range
    Groups by provider for bulk payment processing
    """
    # Calculate date range based on report type
    today = datetime.now(timezone.utc)
    
    if report_type == "all":
        start = datetime(2020, 1, 1, tzinfo=timezone.utc)
        end = datetime(2030, 12, 31, tzinfo=timezone.utc)
    elif report_type == "daily":
        start = today.replace(hour=0, minute=0, second=0, microsecond=0)
        end = today.replace(hour=23, minute=59, second=59, microsecond=999999)
    elif report_type == "weekly":
        start = today - timedelta(days=today.weekday())
        end = start + timedelta(days=6, hours=23, minutes=59, seconds=59)
    elif report_type == "monthly":
        start = today.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        next_month = start.replace(day=28) + timedelta(days=4)
        end = next_month - timedelta(days=next_month.day)
    
    # Use custom dates if provided
    if start_date:
        start = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
    if end_date:
        end = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
    
    # Query all pending bills and filter by date in Python
    # (due_date may be stored as plain date "YYYY-MM-DD" or ISO datetime)
    query = {"status": "pending"}
    if provider:
        query["provider"] = {"$regex": provider, "$options": "i"}

    all_pending = await sdb.find_many("bills", query)

    start_str = start.strftime('%Y-%m-%d')
    end_str = end.strftime('%Y-%m-%d')

    bills = []
    for bill in all_pending:
        due = bill.get("due_date", "")
        # Normalize: take only the date portion
        due_date_str = due[:10] if due else ""
        if due_date_str and start_str <= due_date_str <= end_str:
            bills.append(bill)
    
    # Get user details in batch
    user_ids = list(set(b["user_id"] for b in bills))
    users_list = await sdb.find_many("users", {"id": {"$in": user_ids}})
    user_map = {u["id"]: u for u in users_list}

    bill_reports = []
    for bill in bills:
        user = user_map.get(bill["user_id"])
        if user:
            bill_reports.append({
                "bill_id": bill["id"],
                "user_name": user["full_name"],
                "user_email": user["email"],
                "provider": bill["provider"],
                "category": bill["category"],
                "account_number": bill["account_number"],
                "biller_code": bill.get("biller_code") or bill.get("bpay_code"),
                "reference_number": bill.get("reference_number"),
                "bpay_code": bill.get("bpay_code"),
                "amount": bill["amount"],
                "due_date": bill["due_date"],
                "frequency": bill["frequency"]
            })
    
    # Group by provider for bulk payment
    providers_summary = {}
    for bill in bill_reports:
        provider_name = bill["provider"]
        if provider_name not in providers_summary:
            providers_summary[provider_name] = {
                "provider": provider_name,
                "total_amount": 0,
                "bill_count": 0,
                "bills": []
            }
        providers_summary[provider_name]["total_amount"] += bill["amount"]
        providers_summary[provider_name]["bill_count"] += 1
        providers_summary[provider_name]["bills"].append(bill)
    
    return {
        "report_type": report_type,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "total_bills": len(bill_reports),
        "total_amount": sum(b["amount"] for b in bill_reports),
        "providers_summary": list(providers_summary.values()),
        "detailed_bills": bill_reports
    }

@api_router.post("/admin/process-bulk-payment")
async def process_bulk_payment(
    provider: str,
    bill_ids: List[str],
    admin_user: dict = Depends(get_admin_user)
):
    """
    Mark bills as paid in bulk (after admin processes payment to provider)
    """
    now = datetime.now(timezone.utc).isoformat()
    result = await sdb.update_many("bills",
        {"id": {"$in": bill_ids}, "status": "pending"},
        {"$set": {"status": "paid", "paid_by": "admin", "paid_at": now}}
    )
    
    return {
        "message": f"Bulk payment processed for {provider}",
        "bills_updated": result
    }


@api_router.post("/admin/pay-bill")
async def admin_pay_single_bill(data: AdminPayBillRequest, admin_user: dict = Depends(get_admin_user)):
    """Admin marks a single bill as paid after making BPAY/bank payment on behalf of customer."""
    bill = await sdb.find_one("bills", {"id": data.bill_id, "status": "pending"})
    if not bill:
        raise HTTPException(status_code=404, detail="Bill not found or already paid")

    now = datetime.now(timezone.utc).isoformat()
    update = {
        "status": "paid",
        "paid_by": "admin",
        "paid_at": now,
    }
    if data.payment_reference:
        update["payment_reference"] = data.payment_reference

    await sdb.update_one("bills", {"id": data.bill_id}, {"$set": update})

    # Deduct from customer wallet (atomic)
    await sdb.increment_wallet_balance(bill["user_id"], -bill["amount"])

    # Record transaction
    tx = Transaction(
        user_id=bill["user_id"],
        type="bill_payment",
        amount=bill["amount"],
        description=f"Admin BPAY payment: {bill['provider']} - Ref: {data.payment_reference or 'N/A'}"
    )
    await sdb.insert_one("transactions", tx.model_dump())

    return {"message": f"Bill paid: {bill['provider']} ${bill['amount']:.2f}", "bill_id": data.bill_id}


class AdminBulkPayRequest(BaseModel):
    bill_ids: List[str]
    payment_reference: Optional[str] = None


@api_router.post("/admin/pay-bills-bulk")
async def admin_pay_bills_bulk(data: AdminBulkPayRequest, admin_user: dict = Depends(get_admin_user)):
    """Admin marks multiple bills as paid in bulk after making BPAY/bank payment."""
    now = datetime.now(timezone.utc).isoformat()
    paid_count = 0
    total_amount = 0

    for bill_id in data.bill_ids:
        bill = await sdb.find_one("bills", {"id": bill_id, "status": "pending"})
        if not bill:
            continue

        update = {"status": "paid", "paid_by": "admin", "paid_at": now}
        if data.payment_reference:
            update["payment_reference"] = data.payment_reference

        await sdb.update_one("bills", {"id": bill_id}, {"$set": update})
        await sdb.increment_wallet_balance(bill["user_id"], -bill["amount"])

        tx = Transaction(
            user_id=bill["user_id"],
            type="bill_payment",
            amount=bill["amount"],
            description=f"Admin bulk BPAY payment: {bill['provider']} - Ref: {data.payment_reference or 'N/A'}"
        )
        await sdb.insert_one("transactions", tx.model_dump())
        paid_count += 1
        total_amount += bill["amount"]

    return {"message": f"Bulk payment processed: {paid_count} bills, ${total_amount:.2f}", "paid_count": paid_count, "total_amount": total_amount}


@api_router.get("/admin/payment-queue")
async def admin_payment_queue(admin_user: dict = Depends(get_admin_user)):
    """
    Get all pending bills with full payment details for admin to process BPAY/bank payments.
    Grouped by provider with biller codes and reference numbers.
    """
    all_pending = await sdb.find_many("bills", {"status": "pending"})

    user_ids = list(set(b["user_id"] for b in all_pending))
    users_list = await sdb.find_many("users", {"id": {"$in": user_ids}})
    user_map = {u["id"]: u for u in users_list}

    queue = []
    for bill in all_pending:
        user = user_map.get(bill["user_id"])
        queue.append({
            "bill_id": bill["id"],
            "user_id": bill["user_id"],
            "user_name": user["full_name"] if user else "Unknown",
            "user_email": user["email"] if user else "",
            "provider": bill["provider"],
            "category": bill["category"],
            "account_number": bill["account_number"],
            "biller_code": bill.get("biller_code") or bill.get("bpay_code") or "",
            "reference_number": bill.get("reference_number") or "",
            "amount": bill["amount"],
            "due_date": bill["due_date"],
            "frequency": bill["frequency"],
            "created_at": bill.get("created_at", ""),
        })

    # Group by provider
    providers = {}
    for item in queue:
        prov = item["provider"]
        if prov not in providers:
            providers[prov] = {"provider": prov, "total_amount": 0, "bill_count": 0, "bills": []}
        providers[prov]["total_amount"] += item["amount"]
        providers[prov]["bill_count"] += 1
        providers[prov]["bills"].append(item)

    return {
        "total_pending": len(queue),
        "total_amount": sum(b["amount"] for b in queue),
        "providers": list(providers.values()),
        "bills": queue,
    }


# ===================== PAYMENT RUNS (prioritised, maker-checker bill payment) =====================
# admin_pay_bill / admin_pay_bills_bulk above remain in place for now (the
# frontend still calls them) but are superseded by this flow: it only pays a
# bill out of the OWNING customer's own cleared, available ledger balance
# (never wallet_balance / never another customer's money), requires a
# second admin to approve, and only debits the ledger once a payment is
# confirmed cleared rather than at the same moment as "mark paid". Plan to
# retire admin_pay_bill/admin_pay_bills_bulk once the admin UI moves over —
# see INTEGRATION_NOTES.md.

@api_router.post("/admin/payment-runs")
async def create_payment_run(horizon_days: int = 3, admin_user: dict = Depends(get_admin_user)):
    return await payment_runs.build_payment_run(horizon_days=horizon_days, created_by=admin_user["id"])


@api_router.get("/admin/payment-runs")
async def list_payment_runs(admin_user: dict = Depends(get_admin_user)):
    return await sdb.find_many("payment_runs", order_by="created_at", order_desc=True, limit=50)


@api_router.get("/admin/payment-runs/{run_id}/items")
async def list_payment_run_items(run_id: str, admin_user: dict = Depends(get_admin_user)):
    return await sdb.find_many("payment_run_items", {"payment_run_id": run_id}, order_by="priority_rank")


@api_router.post("/admin/payment-runs/{run_id}/approve")
async def approve_payment_run_endpoint(run_id: str, admin_user: dict = Depends(get_admin_user)):
    try:
        return await payment_runs.approve_payment_run(run_id, approver_user_id=admin_user["id"])
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


class ClearPaymentRunItemRequest(BaseModel):
    provider_payment_reference: str = ""


@api_router.post("/admin/payment-runs/items/{item_id}/submit")
async def submit_payment_run_item(item_id: str, data: ClearPaymentRunItemRequest, admin_user: dict = Depends(get_admin_user)):
    await payment_runs.mark_item_submitted(item_id, data.provider_payment_reference)
    return {"status": "submitted"}


@api_router.post("/admin/payment-runs/items/{item_id}/clear")
async def clear_payment_run_item(item_id: str, data: ClearPaymentRunItemRequest, admin_user: dict = Depends(get_admin_user)):
    """Call ONLY after independently confirming the BPAY/bank payment actually
    executed — a receipt, or a bank statement line. This is the single place
    the ledger is debited and bills.status flips to 'paid' for this item."""
    try:
        journal_id = await payment_runs.mark_item_cleared(
            item_id, data.provider_payment_reference, cleared_by=admin_user["id"]
        )
        return {"status": "cleared", "journal_id": journal_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


class FailPaymentRunItemRequest(BaseModel):
    reason: str


@api_router.post("/admin/payment-runs/items/{item_id}/fail")
async def fail_payment_run_item(item_id: str, data: FailPaymentRunItemRequest, admin_user: dict = Depends(get_admin_user)):
    await payment_runs.mark_item_failed(item_id, data.reason)
    return {"status": "failed"}


# ===================== RECONCILIATION =====================
@api_router.get("/admin/reconciliation/latest")
async def get_latest_reconciliation(admin_user: dict = Depends(get_admin_user)):
    runs = await sdb.find_many("reconciliation_runs", order_by="run_at", order_desc=True, limit=1)
    return runs[0] if runs else None


@api_router.get("/admin/reconciliation/exceptions")
async def get_open_reconciliation_exceptions(admin_user: dict = Depends(get_admin_user)):
    return await sdb.find_many("reconciliation_exceptions", {"status": "open"})


@api_router.post("/admin/reconciliation/run")
async def trigger_reconciliation(admin_user: dict = Depends(get_admin_user)):
    return await reconciliation.run_trust_reconciliation(triggered_by=admin_user["id"])


# ===================== PAYMENT METHODS =====================
@api_router.post("/payment-methods")
async def add_payment_method(data: PaymentMethodCreate, current_user: dict = Depends(get_current_user)):
    masked_account = None
    card_last4 = None
    encrypted_bsb = None

    # Only store masked/encrypted values — never raw card numbers
    if data.type == "bank_account" and data.account_number:
        masked_account = "****" + data.account_number[-4:]
        if data.bsb:
            encrypted_bsb = encrypt_field(data.bsb)
    if data.type in ("credit_card", "debit_card") and data.card_number:
        card_last4 = data.card_number[-4:]
        # Raw card number is NEVER stored — PCI DSS compliance

    if data.is_primary:
        await sdb.update_many("payment_methods", {"user_id": current_user["id"]}, {"$set": {"is_primary": False}})

    pm = PaymentMethod(
        user_id=current_user["id"],
        type=data.type,
        label=data.label,
        bank_name=data.bank_name,
        bsb=encrypted_bsb,
        account_number_masked=masked_account,
        card_last4=card_last4,
        card_brand=data.card_brand,
        is_primary=data.is_primary,
    )
    pm_dict = pm.model_dump()
    await sdb.insert_one("payment_methods", pm_dict)
    # Return safe version (decrypt bsb for masking)
    if pm_dict.get("bsb"):
        raw_bsb = decrypt_field(pm_dict["bsb"])
        pm_dict["bsb"] = raw_bsb[:3] + "-***" if len(raw_bsb) >= 3 else None
    return pm_dict

@api_router.get("/payment-methods")
async def get_payment_methods(current_user: dict = Depends(get_current_user)):
    methods = await sdb.find_many("payment_methods", {"user_id": current_user["id"]})
    for m in methods:
        if m.get("bsb"):
            raw_bsb = decrypt_field(m["bsb"])
            m["bsb"] = raw_bsb[:3] + "-***" if len(raw_bsb) >= 3 else None
    return methods

@api_router.delete("/payment-methods/{method_id}")
async def delete_payment_method(method_id: str, current_user: dict = Depends(get_current_user)):
    r = await sdb.delete_one("payment_methods", {"id": method_id, "user_id": current_user["id"]})
    if not r:
        raise HTTPException(status_code=404, detail="Payment method not found")
    return {"message": "Payment method removed"}

@api_router.put("/payment-methods/{method_id}/set-primary")
async def set_primary_payment_method(method_id: str, current_user: dict = Depends(get_current_user)):
    await sdb.update_many("payment_methods", {"user_id": current_user["id"]}, {"$set": {"is_primary": False}})
    r = await sdb.update_one("payment_methods", {"id": method_id, "user_id": current_user["id"]}, {"$set": {"is_primary": True}})
    if not r:
        raise HTTPException(status_code=404, detail="Payment method not found")
    return {"message": "Primary payment method updated"}


# ===================== TOKENIZED PAYMENT METHODS (real Stripe) =====================
# These two endpoints are the server-side half of saving a payment method
# that can actually be charged later without the customer present. The
# browser half (Stripe Elements / Payment Element collecting card or BECS
# details directly against the client_secret from create_setup_intent, so
# raw details never reach this backend) is a separate frontend change — see
# INTEGRATION_NOTES_STRIPE.md. Do NOT add raw card/account fields to any
# request model that calls these; the whole point is that this backend
# never sees them.

@api_router.post("/payment-methods/setup-intent")
async def start_payment_method_setup(current_user: dict = Depends(get_current_user)):
    if not STRIPE_API_KEY:
        raise HTTPException(status_code=500, detail="Stripe not configured")
    try:
        return await stripe_collections.create_setup_intent(current_user)
    except stripe.error.StripeError as e:
        logger.error(f"Stripe SetupIntent creation failed for user {current_user['id']}: {e}")
        raise HTTPException(status_code=502, detail="Could not start payment method setup")


class ConfirmSetupIntentRequest(BaseModel):
    setup_intent_id: str
    label: str = ""
    is_primary: bool = False


@api_router.post("/payment-methods/confirm-setup")
async def confirm_payment_method_setup(data: ConfirmSetupIntentRequest, current_user: dict = Depends(get_current_user)):
    if not STRIPE_API_KEY:
        raise HTTPException(status_code=500, detail="Stripe not configured")
    try:
        return await stripe_collections.confirm_setup_intent_and_save(
            current_user, data.setup_intent_id, label=data.label, is_primary=data.is_primary
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except stripe.error.StripeError as e:
        logger.error(f"Stripe SetupIntent confirmation failed for user {current_user['id']}: {e}")
        raise HTTPException(status_code=502, detail="Could not confirm payment method")


# ===================== SMART PAYMENT PLAN =====================
SAFETY_BUFFER = 0.08  # 8% buffer

def _calc_annual_total(bills: list) -> float:
    total = 0
    for b in bills:
        amt = b.get("amount", 0)
        freq = b.get("frequency", "monthly")
        if freq == "monthly":
            total += amt * 12
        elif freq == "quarterly":
            total += amt * 4
        elif freq == "yearly":
            total += amt
        elif freq == "fortnightly":
            total += amt * 26
        elif freq == "weekly":
            total += amt * 52
    return total


@api_router.get("/payment-plan/calculate")
async def calculate_payment_plan(current_user: dict = Depends(get_current_user)):
    """Calculate 3 deduction options based on all user bills with safety buffer."""
    bills = await sdb.find_many("bills", {"user_id": current_user["id"], "status": "pending"})
    annual_total = _calc_annual_total(bills)
    buffered_annual = annual_total * (1 + SAFETY_BUFFER)

    weekly = round(buffered_annual / 52, 2)
    fortnightly = round(buffered_annual / 26, 2)
    monthly = round(buffered_annual / 12, 2)

    return {
        "annual_bill_total": round(annual_total, 2),
        "safety_buffer_pct": SAFETY_BUFFER * 100,
        "buffered_annual": round(buffered_annual, 2),
        "total_pending_bills": len(bills),
        "options": [
            {"frequency": "weekly", "amount": weekly, "deductions_per_year": 52, "label": "Weekly"},
            {"frequency": "fortnightly", "amount": fortnightly, "deductions_per_year": 26, "label": "Fortnightly"},
            {"frequency": "monthly", "amount": monthly, "deductions_per_year": 12, "label": "Monthly"},
        ]
    }


@api_router.post("/payment-plan/select")
async def select_payment_plan(frequency: str, current_user: dict = Depends(get_current_user)):
    if frequency not in ("weekly", "fortnightly", "monthly"):
        raise HTTPException(status_code=400, detail="Invalid frequency")

    bills = await sdb.find_many("bills", {"user_id": current_user["id"], "status": "pending"})
    annual_total = _calc_annual_total(bills)
    buffered = annual_total * (1 + SAFETY_BUFFER)

    divisors = {"weekly": 52, "fortnightly": 26, "monthly": 12}
    days_map = {"weekly": 7, "fortnightly": 14, "monthly": 30}
    amount = round(buffered / divisors[frequency], 2)

    plan = {
        "id": str(uuid.uuid4()),
        "user_id": current_user["id"],
        "frequency": frequency,
        "deduction_amount": amount,
        "annual_total": round(annual_total, 2),
        "buffered_annual": round(buffered, 2),
        "safety_buffer_pct": SAFETY_BUFFER * 100,
        "next_deduction_date": (datetime.now(timezone.utc) + timedelta(days=days_map[frequency])).isoformat(),
        "status": "active",
        "total_collected": 0,
        "total_paid_out": 0,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    await sdb.delete_many("payment_plans", {"user_id": current_user["id"]})
    await sdb.insert_one("payment_plans", plan)
    return plan


@api_router.get("/payment-plan/current")
async def get_current_plan(current_user: dict = Depends(get_current_user)):
    plan = await sdb.find_one("payment_plans", {"user_id": current_user["id"]})
    if not plan:
        return {"status": "none", "message": "No payment plan selected"}
    return plan


@api_router.post("/payment-plan/simulate-deduction")
async def simulate_deduction(current_user: dict = Depends(get_current_user)):
    """Simulate a scheduled deduction from customer's payment method.

    SECURITY: same issue as deposit_to_wallet above — this credited real
    wallet balance with no payment collected, callable by any logged-in
    user, with no gating. Gated the same way. Use
    POST /scheduler/trigger-now instead once a real payment method is
    saved — that goes through actual Stripe collection.
    """
    if os.environ.get("ALLOW_MOCK_PAYMENTS", "false").lower() != "true":
        raise HTTPException(status_code=403, detail="Mock payments are disabled. Set ALLOW_MOCK_PAYMENTS=true for local/dev testing only — never in production.")

    plan = await sdb.find_one("payment_plans", {"user_id": current_user["id"]})
    if not plan or plan.get("status") != "active":
        raise HTTPException(status_code=400, detail="No active payment plan")

    amount = plan["deduction_amount"]
    # Record transaction
    tx = Transaction(
        user_id=current_user["id"],
        type="plan_deduction",
        amount=amount,
        description=f"Scheduled {plan['frequency']} deduction"
    )
    await sdb.insert_one("transactions", tx.model_dump())
    # Update wallet and plan (atomic)
    await sdb.increment_wallet_balance(current_user["id"], amount)
    await sdb.increment_active_plan_totals(current_user["id"], collected_delta=amount)
    return {"message": f"Deduction of ${amount:.2f} processed", "amount": amount}


# ===================== ENHANCED ADMIN ANALYTICS =====================
@api_router.get("/admin/financial-overview")
async def admin_financial_overview(admin_user: dict = Depends(get_admin_user)):
    """Company financial dashboard: collected vs owed, cash flow."""
    all_plans = await sdb.find_many("payment_plans", {})
    all_pending = await sdb.find_many("bills", {"status": "pending"})
    all_paid = await sdb.find_many("bills", {"status": "paid"})
    all_users_list = await sdb.find_many("users", {})
    active_plans = [p for p in all_plans if p.get("status") == "active"]

    total_collected = sum(p.get("total_collected", 0) for p in all_plans)
    total_paid_out = sum(p.get("total_paid_out", 0) for p in all_plans)
    total_wallet_balance = sum(u.get("wallet_balance", 0) for u in all_users_list)
    total_pending_amount = sum(b.get("amount", 0) for b in all_pending)
    total_paid_amount = sum(b.get("amount", 0) for b in all_paid)

    # Monthly collection forecast
    monthly_forecast = sum(p.get("deduction_amount", 0) * (12 if p.get("frequency") == "monthly" else 26 if p.get("frequency") == "fortnightly" else 52) / 12 for p in active_plans)

    return {
        "total_users": len(all_users_list),
        "active_plans": len(active_plans),
        "total_collected": round(total_collected, 2),
        "total_paid_out": round(total_paid_out, 2),
        "total_wallet_balance": round(total_wallet_balance, 2),
        "company_float": round(total_collected - total_paid_out, 2),
        "total_pending_bills": len(all_pending),
        "total_pending_amount": round(total_pending_amount, 2),
        "total_paid_bills": len(all_paid),
        "total_paid_amount": round(total_paid_amount, 2),
        "monthly_collection_forecast": round(monthly_forecast, 2),
    }


@api_router.get("/admin/outstanding-by-period")
async def admin_outstanding_by_period(admin_user: dict = Depends(get_admin_user)):
    """Outstanding bills grouped by time period for finance management."""
    all_pending = await sdb.find_many("bills", {"status": "pending"})
    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')

    overdue, this_month, next_30, next_60, next_90, beyond = [], [], [], [], [], []
    today_dt = datetime.now(timezone.utc)

    for b in all_pending:
        due_str = b.get("due_date", "")[:10]
        if not due_str:
            continue
        try:
            due_dt = datetime.strptime(due_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        days_away = (due_dt - today_dt).days
        entry = {
            "bill_id": b["id"], "user_id": b.get("user_id"), "provider": b.get("provider"),
            "category": b.get("category"), "amount": b.get("amount", 0),
            "due_date": due_str, "days_until_due": days_away
        }
        if days_away < 0:
            overdue.append(entry)
        elif days_away <= 30:
            this_month.append(entry)
        elif days_away <= 60:
            next_60.append(entry)
        elif days_away <= 90:
            next_90.append(entry)
        else:
            beyond.append(entry)

    def _summarize(lst):
        return {"count": len(lst), "total": round(sum(x["amount"] for x in lst), 2), "bills": lst}

    return {
        "overdue": _summarize(overdue),
        "next_30_days": _summarize(this_month),
        "30_to_60_days": _summarize(next_60),
        "60_to_90_days": _summarize(next_90),
        "beyond_90_days": _summarize(beyond),
    }


@api_router.get("/admin/customer-analytics")
async def admin_customer_analytics(admin_user: dict = Depends(get_admin_user)):
    """Customer-level analytics for risk and compliance."""
    users = await sdb.find_many("users", exclude_fields=["password"])

    # Batch fetch all bills and plans
    all_bills = await sdb.find_many("bills", {})
    all_plans = await sdb.find_many("payment_plans", {})

    # Build lookup maps
    bills_by_user = {}
    for b in all_bills:
        bills_by_user.setdefault(b.get("user_id"), []).append(b)
    plan_map = {p["user_id"]: p for p in all_plans}

    analytics = []
    for u in users:
        uid = u["id"]
        bills = bills_by_user.get(uid, [])
        plan = plan_map.get(uid)
        pending = [b for b in bills if b.get("status") == "pending"]
        paid = [b for b in bills if b.get("status") == "paid"]
        total_pending = sum(b.get("amount", 0) for b in pending)
        total_paid = sum(b.get("amount", 0) for b in paid)

        risk = "low"
        if total_pending > u.get("wallet_balance", 0) * 2:
            risk = "high"
        elif total_pending > u.get("wallet_balance", 0):
            risk = "medium"

        # Surfaces plans stuck needing the customer to complete extra card
        # authentication (Stripe returned "requires_action" on an
        # off-session charge -- see _collect_for_plan in this file). The
        # ledger is never credited for these, and the schedule doesn't
        # advance, so without this an admin has no way to notice a
        # customer's contribution is silently stuck.
        last_status = plan.get("last_collection_status") if plan else None

        analytics.append({
            "user_id": uid,
            "name": u.get("full_name", ""),
            "email": u.get("email", ""),
            "total_bills": len(bills),
            "pending_bills": len(pending),
            "paid_bills": len(paid),
            "total_pending_amount": round(total_pending, 2),
            "total_paid_amount": round(total_paid, 2),
            "wallet_balance": round(u.get("wallet_balance", 0), 2),
            "has_plan": plan is not None and plan.get("status") == "active",
            "plan_frequency": plan.get("frequency") if plan else None,
            "risk_level": risk,
            "last_collection_status": last_status,
            "needs_attention": last_status == "requires_customer_action",
        })
    return {"customers": analytics, "total": len(analytics)}


# ===================== STRIPE PAYMENT INTEGRATION =====================
STRIPE_API_KEY = os.environ.get('STRIPE_API_KEY', '')

# Wallet top-up packages (server-defined, never from frontend)
TOPUP_PACKAGES = {
    "small": 50.0,
    "medium": 100.0,
    "large": 250.0,
    "xlarge": 500.0,
    "custom_plan": None,  # Will be dynamically set from plan deduction amount
}


class TopUpRequest(BaseModel):
    package_id: str  # small, medium, large, xlarge, custom_plan
    origin_url: str
    payment_method_type: str = "card"  # "card" or "au_becs_debit" or "both"


@api_router.post("/payments/create-checkout")
async def create_checkout_session(data: TopUpRequest, request: Request, current_user: dict = Depends(get_current_user)):
    """Create Stripe checkout session for wallet top-up. Supports card and AU BECS Direct Debit."""
    if not STRIPE_API_KEY:
        raise HTTPException(status_code=500, detail="Stripe not configured")

    # Determine amount server-side
    if data.package_id == "custom_plan":
        plan = await sdb.find_one("payment_plans", {"user_id": current_user["id"], "status": "active"})
        if not plan:
            raise HTTPException(status_code=400, detail="No active plan to determine amount")
        amount = float(plan["deduction_amount"])
    elif data.package_id in TOPUP_PACKAGES:
        amount = TOPUP_PACKAGES[data.package_id]
    else:
        raise HTTPException(status_code=400, detail="Invalid package")

    host_url = str(request.base_url).rstrip('/')
    webhook_url = f"{host_url}api/webhook/stripe"
    stripe_checkout = StripeCheckout(api_key=STRIPE_API_KEY, webhook_url=webhook_url)

    success_url = f"{data.origin_url}/dashboard/payment-plan?session_id={{CHECKOUT_SESSION_ID}}"
    cancel_url = f"{data.origin_url}/dashboard/payment-plan"

    metadata = {
        "user_id": current_user["id"],
        "package_id": data.package_id,
        "type": "wallet_topup",
        "payment_method_type": data.payment_method_type
    }

    # Determine Stripe payment methods based on user's choice
    if data.payment_method_type == "au_becs_debit":
        stripe_methods = ["au_becs_debit"]
    elif data.payment_method_type == "both":
        stripe_methods = ["card", "au_becs_debit"]
    else:
        stripe_methods = ["card"]

    checkout_req = CheckoutSessionRequest(
        amount=amount,
        currency="aud",
        success_url=success_url,
        cancel_url=cancel_url,
        metadata=metadata,
        payment_methods=stripe_methods
    )

    try:
        session = await stripe_checkout.create_checkout_session(checkout_req)
    except Exception as e:
        error_msg = str(e)
        if "au_becs_debit" in error_msg and "invalid" in error_msg.lower():
            raise HTTPException(
                status_code=400,
                detail="BECS Direct Debit is not yet enabled on this Stripe account. Please use Card payment, or enable BECS in your Stripe Dashboard under Settings > Payment Methods."
            )
        logger.error(f"Stripe checkout creation failed: {error_msg}")
        raise HTTPException(status_code=500, detail="Checkout creation failed. Please try again later.")

    # Create payment transaction record (only columns that exist in Supabase table)
    tx = {
        "id": str(uuid.uuid4()),
        "session_id": session.session_id,
        "user_id": current_user["id"],
        "amount": amount,
        "package_id": data.package_id,
        "payment_status": "initiated",
        "status": "pending",
        "payment_method_type": data.payment_method_type,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    await sdb.insert_one("payment_transactions", tx)

    return {"url": session.url, "session_id": session.session_id, "amount": amount}


@api_router.get("/payments/status/{session_id}")
async def check_payment_status(session_id: str, current_user: dict = Depends(get_current_user)):
    """Poll Stripe for payment status and update records."""
    if not STRIPE_API_KEY:
        raise HTTPException(status_code=500, detail="Stripe not configured")

    tx = await sdb.find_one("payment_transactions", {"session_id": session_id})
    if not tx:
        raise HTTPException(status_code=404, detail="Transaction not found")

    # Prevent double crediting
    if tx.get("payment_status") == "paid":
        return {"status": "complete", "payment_status": "paid", "amount": tx["amount"], "already_processed": True}

    try:
        host_url = "https://placeholder.com/"
        webhook_url = f"{host_url}api/webhook/stripe"
        stripe_checkout = StripeCheckout(api_key=STRIPE_API_KEY, webhook_url=webhook_url)

        checkout_status = await stripe_checkout.get_checkout_status(session_id)

        if checkout_status.payment_status == "paid" and tx.get("payment_status") != "paid":
            # Atomically flip payment_status to "paid" only if it isn't already —
            # this is the idempotency guard. Filtering on {"$ne": "paid"} makes the
            # UPDATE itself the race-free check: if this webhook and a concurrent
            # status-poll request (or a retried webhook) both reach this line, only
            # one UPDATE will actually match a row (the other will affect zero rows),
            # so the wallet can never be credited twice for the same session.
            amount = tx["amount"]
            transitioned = await sdb.update_one(
                "payment_transactions",
                {"session_id": session_id, "payment_status": {"$ne": "paid"}},
                {"$set": {"payment_status": "paid", "status": "completed", "paid_at": datetime.now(timezone.utc).isoformat()}}
            )
            if transitioned:
                await ledger.record_contribution_cleared(
                    user_id=tx["user_id"], amount=amount,
                    reference_type="payment_transactions", reference_id=session_id,
                    payment_method_type=tx.get("payment_method_type", "card"),
                    description=f"Wallet top-up via Stripe (${amount:.2f})",
                )
                # Record in main transactions collection
                tx_record = Transaction(
                    user_id=tx["user_id"],
                    type="stripe_topup",
                    amount=amount,
                    description=f"Wallet top-up via Stripe (${amount:.2f})"
                )
                await sdb.insert_one("transactions", tx_record.model_dump())

                # Also update plan total_collected
                await sdb.increment_active_plan_totals(tx["user_id"], collected_delta=amount)

        elif checkout_status.status == "expired":
            await sdb.update_one("payment_transactions", {"session_id": session_id}, {"$set": {"payment_status": "expired", "status": "expired"}})

        return {
            "status": checkout_status.status,
            "payment_status": checkout_status.payment_status,
            "amount": tx["amount"],
        }
    except Exception as e:
        # If Stripe API call fails (e.g., test key limitations), return current DB status
        logger.warning(f"Stripe status check failed for {session_id}: {e}")
        return {
            "status": tx.get("status", "pending"),
            "payment_status": tx.get("payment_status", "pending"),
            "amount": tx["amount"],
            "note": "Status from database (Stripe API unavailable)"
        }


@api_router.post("/webhook/stripe")
async def stripe_webhook(request: Request):
    """Handle Stripe webhook events."""
    try:
        body = await request.body()
        sig = request.headers.get("Stripe-Signature", "")
        host_url = str(request.base_url).rstrip('/')
        webhook_url = f"{host_url}api/webhook/stripe"
        stripe_checkout = StripeCheckout(api_key=STRIPE_API_KEY, webhook_url=webhook_url)
        event = await stripe_checkout.handle_webhook(body, sig)

        if event.payment_status == "paid":
            tx = await sdb.find_one("payment_transactions", {"session_id": event.session_id})
            if tx and tx.get("payment_status") != "paid":
                amount = tx["amount"]
                # Atomic conditional transition — see matching comment in
                # check_payment_status() above for why this prevents double-crediting
                # when the webhook and a status-poll request race each other.
                transitioned = await sdb.update_one(
                    "payment_transactions",
                    {"session_id": event.session_id, "payment_status": {"$ne": "paid"}},
                    {"$set": {"payment_status": "paid", "status": "completed", "paid_at": datetime.now(timezone.utc).isoformat()}}
                )
                if transitioned:
                    await ledger.record_contribution_cleared(
                        user_id=tx["user_id"], amount=amount,
                        reference_type="payment_transactions", reference_id=event.session_id,
                        payment_method_type=tx.get("payment_method_type", "card"),
                        description=f"Wallet top-up via Stripe (${amount:.2f})",
                    )
                    await sdb.increment_active_plan_totals(tx["user_id"], collected_delta=amount)

        return {"status": "ok"}
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return {"status": "error"}


@api_router.post("/webhook/stripe-payment-intents")
async def stripe_payment_intent_webhook(request: Request):
    """
    Dedicated webhook for off-session PaymentIntents created by
    stripe_collections.collect_scheduled_contribution() — kept separate
    from stripe_webhook() above (which is built around the
    emergentintegrations StripeCheckout wrapper and its Checkout-Session
    event model) rather than assuming that wrapper also correctly surfaces
    raw payment_intent.* events for PaymentIntents it didn't create. This
    endpoint uses the official stripe-python SDK's own webhook
    verification directly, which is the well-documented path for this.

    Point a second webhook endpoint at this URL in the Stripe Dashboard,
    subscribed to payment_intent.succeeded and payment_intent.payment_failed,
    with its own signing secret in STRIPE_PAYMENT_INTENT_WEBHOOK_SECRET.
    """
    webhook_secret = os.environ.get("STRIPE_PAYMENT_INTENT_WEBHOOK_SECRET", "")
    if not webhook_secret:
        logger.error("STRIPE_PAYMENT_INTENT_WEBHOOK_SECRET not set — rejecting webhook")
        return JSONResponse(status_code=500, content={"status": "not_configured"})

    body = await request.body()
    sig = request.headers.get("Stripe-Signature", "")
    try:
        event = stripe.Webhook.construct_event(body, sig, webhook_secret)
    except (ValueError, stripe.error.SignatureVerificationError) as e:
        logger.warning(f"Rejected payment-intent webhook: invalid signature/payload ({e})")
        return JSONResponse(status_code=400, content={"status": "invalid_signature"})

    pi = event["data"]["object"]
    metadata = pi.get("metadata", {}) or {}
    if metadata.get("kind") != "scheduled_contribution":
        return {"status": "ignored"}  # not one of ours

    attempt_id = metadata.get("collection_attempt_id")
    user_id = metadata.get("billsure_user_id")
    plan_id = metadata.get("payment_plan_id")

    if event["type"] == "payment_intent.succeeded":
        # Same atomic idempotency pattern as the Checkout Session paths
        # above: only credit the ledger if this attempt hasn't already been
        # marked succeeded by another code path (the synchronous branch in
        # process_auto_deductions, or a duplicate webhook delivery).
        transitioned = await sdb.update_one(
            "collection_attempts",
            {"id": attempt_id, "status": {"$ne": "succeeded"}},
            {"$set": {"status": "succeeded", "stripe_payment_intent_id": pi["id"]}},
        )
        if transitioned:
            amount = Decimal(pi["amount"]) / 100
            method_type = metadata.get("payment_method_type", "card")
            await ledger.record_contribution_cleared(
                user_id=user_id, amount=amount,
                reference_type="collection_attempts", reference_id=attempt_id,
                payment_method_type=method_type,
                description="Scheduled contribution cleared",
            )
            await sdb.increment_active_plan_totals(user_id, collected_delta=float(amount))
            await sdb.update_one("payment_plans", {"id": plan_id}, {"$set": {
                "last_collection_status": "succeeded", "consecutive_failures": 0,
            }})

    elif event["type"] == "payment_intent.payment_failed":
        await sdb.update_one(
            "collection_attempts", {"id": attempt_id, "status": {"$ne": "succeeded"}},
            {"$set": {"status": "failed", "error_message": pi.get("last_payment_error", {}).get("message", "")}},
        )
        plan = await sdb.find_one("payment_plans", {"id": plan_id})
        if plan:
            await sdb.update_one("payment_plans", {"id": plan_id}, {"$set": {
                "last_collection_status": "failed",
                "consecutive_failures": (plan.get("consecutive_failures") or 0) + 1,
            }})

    return {"status": "ok"}


@api_router.get("/payments/history")
async def get_payment_history(current_user: dict = Depends(get_current_user)):
    """Get user's Stripe payment history."""
    txs = await sdb.find_many("payment_transactions", {"user_id": current_user["id"]}, order_by="created_at", order_desc=True, limit=100)
    return txs


# ===================== SCHEDULED CONTRIBUTION COLLECTION =====================
# This used to also auto-pay bills directly out of wallet_balance in the same
# loop. That's been removed: bill payment is now payment_runs' job
# (payment_run_scheduler_loop below), which only pays a bill out of a
# customer's own CLEARED, AVAILABLE ledger balance — this loop's only job is
# making that balance real by actually collecting the money.
#
# CONSECUTIVE_FAILURE_LIMIT: after this many failed collection attempts in a
# row for a plan, it's paused rather than retried indefinitely — a card that
# keeps failing needs a human/customer to fix it, not an infinite retry loop
# quietly doing nothing every 60 seconds forever.
CONSECUTIVE_FAILURE_LIMIT = 3


async def _collect_for_plan(plan: dict, now: datetime) -> dict:
    """Charges one due payment plan's saved payment method for real via
    Stripe, and advances the schedule / credits the ledger based on what
    Stripe actually reports. Shared by the background loop and the manual
    /scheduler/trigger-now endpoint so there is exactly one place this logic
    lives."""
    uid = plan["user_id"]
    amount = plan["deduction_amount"]
    freq = plan["frequency"]
    next_date_str = (plan.get("next_deduction_date") or "")[:10]
    idempotency_key = f"{plan['id']}:{next_date_str}"

    result = await stripe_collections.collect_scheduled_contribution(
        user_id=uid, amount=amount, payment_plan_id=plan["id"],
        idempotency_key=idempotency_key,
    )

    await sdb.update_one("payment_plans", {"id": plan["id"]}, {"$set": {
        "last_collection_attempt_at": now.isoformat(),
        "last_collection_status": result["status"],
    }})

    if result["status"] == "no_payment_method":
        logger.warning(f"Plan {plan['id']} (user {uid}) has no chargeable payment method — leaving next_deduction_date unchanged so it retries once one is added")
        return result  # do NOT advance the schedule — nothing was attempted

    if result["status"] == "requires_customer_action":
        # Off-session confirmation hit Stripe's authentication_required path
        # (see stripe_collections.collect_scheduled_contribution) — the card
        # needs an interactive 3D Secure / SCA challenge this backend can't
        # complete on the customer's behalf. Do NOT credit the ledger (no
        # branch below does), and do NOT advance next_deduction_date — the
        # collection_attempts row is idempotency-keyed to this cycle, so
        # leaving the schedule where it is means every subsequent tick keeps
        # reporting the same stuck state (visible via last_collection_status
        # below) rather than silently moving on and permanently skipping
        # this contribution. Surfaced two ways: a customer-facing
        # notification here, and admin visibility via needs_attention in
        # GET /admin/customer-analytics.
        existing_notif = await sdb.find_one("notifications", {
            "user_id": uid, "type": "payment_requires_action",
            "bill_id": result.get("collection_attempt_id"),
        })
        if not existing_notif:
            await sdb.insert_one("notifications", {
                "id": str(uuid.uuid4()),
                "user_id": uid,
                "bill_id": result.get("collection_attempt_id"),
                "type": "payment_requires_action",
                "title": "Action needed to complete your payment",
                "message": (
                    f"Your {freq} contribution of ${amount:.2f} couldn't be completed "
                    f"automatically — your card issuer needs you to verify this payment. "
                    f"Please contact support or update your payment method."
                ),
                "read": False,
                "email_sent": False,
                "created_at": now.isoformat(),
            })
        logger.warning(
            f"Scheduled collection for plan {plan['id']} (user {uid}) requires customer "
            f"action (Stripe authentication_required) — schedule not advanced, ledger not credited"
        )
        return result  # do NOT advance the schedule — see comment above

    if result["status"] == "succeeded":
        # Card payments confirm synchronously. Same atomic idempotency
        # pattern as the Checkout/webhook paths: only credit the ledger if
        # this attempt hasn't already been credited (e.g. the webhook won
        # the race, or this function ran twice for the same idempotency
        # key — which insert-if-absent in collect_scheduled_contribution
        # already prevents from reaching Stripe twice, but this is the
        # second, independent guard on the ledger-credit side specifically).
        transitioned = await sdb.update_one(
            "collection_attempts",
            {"id": result["collection_attempt_id"], "status": {"$ne": "credited"}},
            {"$set": {"status": "credited"}},
        )
        if transitioned:
            await ledger.record_contribution_cleared(
                user_id=uid, amount=amount,
                reference_type="collection_attempts", reference_id=result["collection_attempt_id"],
                payment_method_type="card",
                description=f"Scheduled {freq} contribution (${amount:.2f})",
            )
            await sdb.increment_active_plan_totals(uid, collected_delta=amount)
        logger.info(f"Scheduled collection succeeded: ${amount:.2f} for user {uid}")

    elif result["status"] == "processing":
        # au_becs_debit — settles asynchronously. Ledger credit (and the
        # BECS hold) happens later via /webhook/stripe-payment-intents.
        logger.info(f"Scheduled collection processing (BECS): ${amount:.2f} for user {uid}, awaiting webhook")

    elif result["status"] == "failed":
        failures = (plan.get("consecutive_failures") or 0) + 1
        updates = {"consecutive_failures": failures}
        if failures >= CONSECUTIVE_FAILURE_LIMIT:
            updates["status"] = "paused"
            logger.error(f"Plan {plan['id']} (user {uid}) paused after {failures} consecutive collection failures")
        await sdb.update_one("payment_plans", {"id": plan["id"]}, {"$set": updates})
        logger.warning(f"Scheduled collection failed for user {uid}: {result.get('reason')}")

    # Advance the schedule after any attempted charge (succeeded/processing/
    # failed) — a failure is retried next cycle via consecutive_failures
    # tracking, not by re-attempting immediately. no_payment_method already
    # returned above without reaching here.
    await sdb.update_one(
        "payment_plans", {"id": plan["id"], "status": "active"},
        {"$set": {"next_deduction_date": _next_deduction_date(freq).isoformat()}},
    )
    return result


async def _run_due_scheduled_collections() -> dict:
    """One pass: runs _collect_for_plan() for every due, active plan. Shared
    by the in-process loop below and the /internal/cron/scheduled-collections
    endpoint, so there's exactly one place this runs from regardless of which
    scheduling mode a deployment uses."""
    now = datetime.now(timezone.utc)
    today_str = now.strftime('%Y-%m-%d')
    active_plans = await sdb.find_many("payment_plans", {"status": "active"})
    attempted = 0
    for plan in active_plans:
        next_date_str = (plan.get("next_deduction_date") or "")[:10]
        if next_date_str and next_date_str <= today_str:
            await _collect_for_plan(plan, now)
            attempted += 1
    return {"active_plans": len(active_plans), "attempted": attempted}


async def process_scheduled_collections():
    """In-process background task (SCHEDULER_MODE=loop, the default): calls
    _run_due_scheduled_collections() every 60 seconds for as long as this
    process stays up. See SCHEDULER_MODE below for the cron-triggered
    alternative that doesn't depend on a long-lived process."""
    while True:
        try:
            await _run_due_scheduled_collections()
        except Exception as e:
            logger.error(f"Scheduled collection loop error: {e}")

        await asyncio.sleep(60)


# ===================== BILL PAYMENT: PRIORITISED RUNS =====================
PAYMENT_RUN_INTERVAL_SECONDS = 6 * 60 * 60  # every 6 hours — bills don't need per-minute polling


async def _run_payment_run_scheduler_once() -> dict:
    """One pass: builds a fresh payment run from whatever bills are due and
    can actually be afforded from cleared, available customer balances.
    Does NOT approve or execute anything — that stays a human decision (see
    POST /admin/payment-runs/{id}/approve). Shared by the in-process loop
    below and the /internal/cron/payment-run-queue endpoint."""
    run = await payment_runs.build_payment_run(horizon_days=3, created_by=None)
    if run["item_count"] > 0:
        logger.info(f"Payment run {run['id']} built: {run['item_count']} bills, ${run['total_amount']} total — awaiting admin approval")
    return run


async def payment_run_scheduler_loop():
    """In-process background task (SCHEDULER_MODE=loop, the default): calls
    _run_payment_run_scheduler_once() every PAYMENT_RUN_INTERVAL_SECONDS.
    This just keeps the admin queue populated automatically instead of
    requiring someone to remember to click 'create run'."""
    while True:
        try:
            await _run_payment_run_scheduler_once()
        except Exception as e:
            logger.error(f"Payment run scheduler error: {e}")
        await asyncio.sleep(PAYMENT_RUN_INTERVAL_SECONDS)


async def reconciliation_loop():
    """In-process background task (SCHEDULER_MODE=loop, the default): hourly
    trust-account reconciliation. approve_payment_run() refuses to approve
    while the most recent run here has an open exception, so this loop (or
    its cron-triggered equivalent, /internal/cron/reconciliation) being
    healthy is a precondition for bills actually getting paid, not just a
    reporting nicety."""
    while True:
        try:
            result = await reconciliation.run_trust_reconciliation()
            if result["status"] != "ok":
                logger.error(f"Reconciliation variance detected: {result}")
        except Exception as e:
            logger.error(f"Reconciliation loop error: {e}")
        await asyncio.sleep(3600)


# ===================== SCHEDULING MODES =====================
# Three ways to trigger process_scheduled_collections / payment_run_scheduler_loop /
# reconciliation_loop's underlying work, chosen via SCHEDULER_MODE:
#
#   loop (default)  — the in-process `while True: asyncio.sleep()` loops
#                      defined above. Zero extra config, but the schedule
#                      lives only in this process's memory (restart loses
#                      it) and assumes exactly one instance is ever running.
#   apscheduler      — backend/scheduler.py: APScheduler with a persistent
#                      Postgres job store (survives restarts) plus a
#                      Postgres advisory lock around each job (safe if
#                      multiple instances run at once). Requires
#                      DATABASE_URL — see scheduler.py's docstring.
#   cron             — the /internal/cron/* endpoints below: stateless,
#                      idempotent HTTP calls an external cron system hits
#                      on a schedule. Durability comes from the external
#                      scheduler rather than this process surviving.
#
# Whichever mode is NOT active for these three jobs, _start_background_loops
# (in startup_event) does not also start it — running two triggering
# mechanisms for the same jobs at once would double-run them.
#
# The /internal/cron/* endpoints stay registered regardless of
# SCHEDULER_MODE (harmless — CRON_SECRET still gates them), so they're
# always available for a manual/ops-triggered run even in loop or
# apscheduler mode.
#
# Auth on the cron endpoints is a shared secret rather than an admin JWT
# because external cron systems generally can't hold a short-lived,
# refreshable user session. Fails closed: if CRON_SECRET isn't set, these
# endpoints refuse every request rather than running unauthenticated.
SCHEDULER_MODE = os.environ.get('SCHEDULER_MODE', 'loop').lower()
CRON_SECRET = os.environ.get('CRON_SECRET', '')


def _verify_cron_secret(x_cron_secret: str = Header(default="", alias="X-Cron-Secret")):
    if not CRON_SECRET:
        raise HTTPException(status_code=503, detail="CRON_SECRET is not configured — cron endpoints are disabled")
    if not secrets.compare_digest(x_cron_secret, CRON_SECRET):
        raise HTTPException(status_code=401, detail="Invalid cron secret")


@api_router.post("/internal/cron/scheduled-collections")
async def cron_scheduled_collections(_: None = Depends(_verify_cron_secret)):
    return await _run_due_scheduled_collections()


@api_router.post("/internal/cron/payment-run-queue")
async def cron_payment_run_queue(_: None = Depends(_verify_cron_secret)):
    return await _run_payment_run_scheduler_once()


@api_router.post("/internal/cron/reconciliation")
async def cron_reconciliation(_: None = Depends(_verify_cron_secret)):
    return await reconciliation.run_trust_reconciliation()


def _next_deduction_date(freq: str) -> datetime:
    now = datetime.now(timezone.utc)
    if freq == "weekly":
        return now + timedelta(days=7)
    elif freq == "fortnightly":
        return now + timedelta(days=14)
    else:
        return now + timedelta(days=30)


@api_router.post("/scheduler/trigger-now")
async def trigger_scheduler_now(current_user: dict = Depends(get_current_user)):
    """
    Manually trigger today's scheduled collection for the calling user's own
    plan, for testing — without waiting for the 60-second background loop.

    SECURITY NOTE — this previously called increment_wallet_balance()
    directly, which meant any logged-in customer could call this repeatedly
    to fabricate wallet balance for free (and then auto-pay a bill with it,
    in the same request). It now goes through the exact same real Stripe
    off-session charge as the background loop — calling it twice in the
    same cycle is a no-op (idempotency_key is unique per plan+day), and no
    money appears without an actual successful charge. Bill payment is no
    longer triggered from here at all — see /admin/payment-runs/*.
    """
    now = datetime.now(timezone.utc)
    today_str = now.strftime('%Y-%m-%d')

    plan = await sdb.find_one("payment_plans", {"user_id": current_user["id"], "status": "active"})
    if not plan:
        return {"message": "No active payment plan for this user", "result": None}

    next_date_str = (plan.get("next_deduction_date") or "")[:10]
    if not (next_date_str and next_date_str <= today_str):
        return {"message": "Nothing due yet for this plan", "next_deduction_date": plan.get("next_deduction_date")}

    result = await _collect_for_plan(plan, now)
    return {"message": "Collection attempted", "result": result}


@api_router.get("/transactions/history")
async def get_transaction_history(current_user: dict = Depends(get_current_user)):
    """Get all user transactions (deductions, bill payments, top-ups)."""
    txs = await sdb.find_many("transactions", {"user_id": current_user["id"]}, order_by="created_at", order_desc=True, limit=200)
    return txs


# ===================== NOTIFICATION SYSTEM =====================
REMINDER_DAYS_BEFORE = 5  # Send reminder X days before due date

async def generate_notifications():
    """Background task: generate bill reminder notifications."""
    while True:
        try:
            now = datetime.now(timezone.utc)
            today_str = now.strftime('%Y-%m-%d')
            reminder_date = (now + timedelta(days=REMINDER_DAYS_BEFORE)).strftime('%Y-%m-%d')

            users = await sdb.find_many("users", exclude_fields=["password"])
            for user in users:
                uid = user["id"]
                pending = await sdb.find_many("bills", {"user_id": uid, "status": "pending"})
                wallet = user.get("wallet_balance", 0)

                for bill in pending:
                    due_str = bill.get("due_date", "")[:10]
                    if not due_str:
                        continue

                    # Overdue notification
                    if due_str < today_str:
                        existing = await sdb.find_one("notifications", {
                            "user_id": uid, "bill_id": bill["id"], "type": "overdue"
                        })
                        if not existing:
                            await sdb.insert_one("notifications", {
                                "id": str(uuid.uuid4()),
                                "user_id": uid,
                                "bill_id": bill["id"],
                                "type": "overdue",
                                "title": f"Overdue: {bill['provider']}",
                                "message": f"Your {bill['category']} bill of ${bill['amount']:.2f} from {bill['provider']} was due on {due_str}.",
                                "read": False,
                                "email_sent": False,
                                "created_at": now.isoformat(),
                            })

                    # Upcoming reminder
                    elif due_str <= reminder_date and due_str >= today_str:
                        existing = await sdb.find_one("notifications", {
                            "user_id": uid, "bill_id": bill["id"], "type": "upcoming"
                        })
                        if not existing:
                            days_left = (datetime.strptime(due_str, '%Y-%m-%d').replace(tzinfo=timezone.utc) - now).days
                            await sdb.insert_one("notifications", {
                                "id": str(uuid.uuid4()),
                                "user_id": uid,
                                "bill_id": bill["id"],
                                "type": "upcoming",
                                "title": f"Due Soon: {bill['provider']}",
                                "message": f"Your {bill['category']} bill of ${bill['amount']:.2f} is due in {max(days_left, 0)} day(s) ({due_str}).",
                                "read": False,
                                "email_sent": False,
                                "created_at": now.isoformat(),
                            })

                # Low wallet balance notification
                total_pending = sum(b.get("amount", 0) for b in pending)
                if total_pending > 0 and wallet < total_pending * 0.5:
                    existing = await sdb.find_one("notifications", {
                        "user_id": uid, "type": "low_balance"
                    })
                    if not existing:
                        await sdb.insert_one("notifications", {
                            "id": str(uuid.uuid4()),
                            "user_id": uid,
                            "bill_id": None,
                            "type": "low_balance",
                            "title": "Low Wallet Balance",
                            "message": f"Your wallet (${wallet:.2f}) may not cover your pending bills (${total_pending:.2f}). Consider topping up.",
                            "read": False,
                            "email_sent": False,
                            "created_at": now.isoformat(),
                        })

            # Send real emails for unsent notifications
            unsent = await sdb.find_many("notifications", {"email_sent": False})
            for n in unsent:
                user = await sdb.find_one("users", {"id": n["user_id"]})
                if user:
                    n_type = n.get("type", "")
                    if n_type == "low_balance":
                        total_pend = sum(b.get("amount", 0) for b in await sdb.find_many("bills", {"user_id": n["user_id"], "status": "pending"}))
                        html = build_low_balance_email(user.get("full_name", ""), float(n.get("message", "0").split("$")[1].split(")")[0]) if "$" in n.get("message", "") else 0, total_pend)
                    else:
                        bill = await sdb.find_one("bills", {"id": n.get("bill_id")}) if n.get("bill_id") else None
                        html = build_bill_email(
                            n_type, user.get("full_name", ""),
                            bill.get("provider", "") if bill else "",
                            bill.get("amount", 0) if bill else 0,
                            bill.get("due_date", "") if bill else "",
                            n.get("message", "")
                        )
                    await send_email(user["email"], n.get("title", "BillSure Notification"), html)
                await sdb.update_one("notifications", {"id": n["id"]}, {"$set": {"email_sent": True}})

        except Exception as e:
            logger.error(f"Notification generator error: {e}")

        await asyncio.sleep(120)  # Run every 2 minutes


@api_router.get("/notifications")
async def get_notifications(current_user: dict = Depends(get_current_user)):
    notifs = await sdb.find_many("notifications", {"user_id": current_user["id"]}, order_by="created_at", order_desc=True, limit=50)
    unread = sum(1 for n in notifs if not n.get("read"))
    return {"notifications": notifs, "unread_count": unread}


@api_router.put("/notifications/{notif_id}/read")
async def mark_notification_read(notif_id: str, current_user: dict = Depends(get_current_user)):
    await sdb.update_one("notifications", {"id": notif_id, "user_id": current_user["id"]}, {"$set": {"read": True}})
    return {"message": "Marked as read"}


@api_router.put("/notifications/read-all")
async def mark_all_read(current_user: dict = Depends(get_current_user)):
    await sdb.update_many("notifications", {"user_id": current_user["id"], "read": False}, {"$set": {"read": True}})
    return {"message": "All marked as read"}


@api_router.delete("/notifications/{notif_id}")
async def delete_notification(notif_id: str, current_user: dict = Depends(get_current_user)):
    await sdb.delete_one("notifications", {"id": notif_id, "user_id": current_user["id"]})
    return {"message": "Deleted"}


# ===================== EXPORT ENDPOINTS =====================
@api_router.get("/admin/export/outstanding-csv")
async def export_outstanding_csv(admin_user: dict = Depends(get_admin_user)):
    """Export outstanding bills as CSV."""
    all_pending = await sdb.find_many("bills", {"status": "pending"})

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Provider", "Category", "Amount", "Due Date", "Frequency", "User ID", "Account Number"])
    for b in all_pending:
        writer.writerow([
            b.get("provider", ""), b.get("category", ""), b.get("amount", 0),
            b.get("due_date", "")[:10], b.get("frequency", ""),
            b.get("user_id", ""), b.get("account_number", "")
        ])

    output.seek(0)
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=outstanding_bills_{datetime.now().strftime('%Y%m%d')}.csv"}
    )


@api_router.get("/admin/export/customers-csv")
async def export_customers_csv(admin_user: dict = Depends(get_admin_user)):
    """Export customer analytics as CSV."""
    users = await sdb.find_many("users", exclude_fields=["password"])

    # Batch fetch all bills and plans
    all_bills = await sdb.find_many("bills", {})
    all_plans = await sdb.find_many("payment_plans", {})
    bills_by_user = {}
    for b in all_bills:
        bills_by_user.setdefault(b.get("user_id"), []).append(b)
    plan_map = {p["user_id"]: p for p in all_plans}

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Name", "Email", "Total Bills", "Pending", "Paid", "Outstanding Amount", "Paid Amount", "Wallet Balance", "Plan", "Risk"])
    for u in users:
        uid = u["id"]
        bills = bills_by_user.get(uid, [])
        plan = plan_map.get(uid)
        pending = [b for b in bills if b.get("status") == "pending"]
        paid = [b for b in bills if b.get("status") == "paid"]
        total_pending = sum(b.get("amount", 0) for b in pending)
        total_paid = sum(b.get("amount", 0) for b in paid)
        wallet = u.get("wallet_balance", 0)
        risk = "high" if total_pending > wallet * 2 else "medium" if total_pending > wallet else "low"
        writer.writerow([
            u.get("full_name", ""), u.get("email", ""), len(bills), len(pending), len(paid),
            f"{total_pending:.2f}", f"{total_paid:.2f}", f"{wallet:.2f}",
            plan.get("frequency", "none") if plan else "none", risk
        ])

    output.seek(0)
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=customer_analytics_{datetime.now().strftime('%Y%m%d')}.csv"}
    )


@api_router.get("/admin/export/outstanding-pdf")
async def export_outstanding_pdf(admin_user: dict = Depends(get_admin_user)):
    """Export outstanding bills as PDF report."""
    all_pending = await sdb.find_many("bills", {"status": "pending"})

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=20*mm, bottomMargin=20*mm, leftMargin=15*mm, rightMargin=15*mm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('CustomTitle', parent=styles['Title'], fontSize=18, spaceAfter=12)
    subtitle_style = ParagraphStyle('Subtitle', parent=styles['Normal'], fontSize=10, textColor=colors.grey, spaceAfter=20)

    elements = []
    elements.append(Paragraph("BillSure - Outstanding Bills Report", title_style))
    elements.append(Paragraph(f"Generated: {datetime.now().strftime('%d %B %Y %H:%M')}", subtitle_style))

    total = sum(b.get("amount", 0) for b in all_pending)
    elements.append(Paragraph(f"Total Outstanding: ${total:,.2f} across {len(all_pending)} bills", styles['Normal']))
    elements.append(Spacer(1, 10*mm))

    # Table
    data = [["Provider", "Category", "Amount", "Due Date", "Frequency"]]
    for b in all_pending:
        data.append([
            b.get("provider", "")[:30], b.get("category", ""),
            f"${b.get('amount', 0):,.2f}", b.get("due_date", "")[:10],
            b.get("frequency", "")
        ])

    if len(data) > 1:
        t = Table(data, colWidths=[55*mm, 30*mm, 25*mm, 25*mm, 25*mm])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0F172A')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('ALIGN', (2, 0), (2, -1), 'RIGHT'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')]),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        elements.append(t)

    doc.build(elements)
    buf.seek(0)
    return StreamingResponse(
        buf, media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=outstanding_bills_{datetime.now().strftime('%Y%m%d')}.pdf"}
    )


@api_router.get("/admin/export/financial-pdf")
async def export_financial_pdf(admin_user: dict = Depends(get_admin_user)):
    """Export financial overview as PDF report."""
    all_plans = await sdb.find_many("payment_plans", {})
    all_pending = await sdb.find_many("bills", {"status": "pending"})
    all_paid = await sdb.find_many("bills", {"status": "paid"})
    all_users = await sdb.count_documents("users", {})
    active_plans = [p for p in all_plans if p.get("status") == "active"]

    total_collected = sum(p.get("total_collected", 0) for p in all_plans)
    total_paid_out = sum(p.get("total_paid_out", 0) for p in all_plans)
    total_pending_amount = sum(b.get("amount", 0) for b in all_pending)
    total_paid_amount = sum(b.get("amount", 0) for b in all_paid)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=20*mm, bottomMargin=20*mm, leftMargin=15*mm, rightMargin=15*mm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('CustomTitle', parent=styles['Title'], fontSize=18, spaceAfter=12)
    subtitle_style = ParagraphStyle('Subtitle', parent=styles['Normal'], fontSize=10, textColor=colors.grey, spaceAfter=20)

    elements = []
    elements.append(Paragraph("BillSure - Financial Overview Report", title_style))
    elements.append(Paragraph(f"Generated: {datetime.now().strftime('%d %B %Y %H:%M')}", subtitle_style))

    # KPI Table
    kpi_data = [
        ["Metric", "Value"],
        ["Total Users", str(all_users)],
        ["Active Plans", str(len(active_plans))],
        ["Total Collected", f"${total_collected:,.2f}"],
        ["Total Paid Out", f"${total_paid_out:,.2f}"],
        ["Company Float", f"${(total_collected - total_paid_out):,.2f}"],
        ["Pending Bills", f"{len(all_pending)} (${total_pending_amount:,.2f})"],
        ["Paid Bills", f"{len(all_paid)} (${total_paid_amount:,.2f})"],
    ]
    t = Table(kpi_data, colWidths=[70*mm, 70*mm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0F172A')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')]),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(t)

    doc.build(elements)
    buf.seek(0)
    return StreamingResponse(
        buf, media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=financial_overview_{datetime.now().strftime('%Y%m%d')}.pdf"}
    )


# ===================== DATA SECURITY STATUS =====================
@api_router.get("/security/compliance-status")
async def get_compliance_status(admin_user: dict = Depends(get_admin_user)):
    """Internal data-security status dashboard for admin.

    NOTE: this intentionally does NOT claim "PCI DSS compliant" for BillSure
    itself. PCI DSS is a formal, audited certification covering ~300 controls
    (network segmentation, vulnerability scanning, access control, logging,
    a QSA assessment, etc.) — encrypting two database columns does not confer
    that certification, and nothing here has been through a PCI DSS audit.
    Card PANs are out of scope for BillSure entirely (Stripe's hosted
    checkout handles them, and Stripe holds its own PCI DSS Level 1
    certification for that). This endpoint reports what BillSure itself
    actually does: encrypt bank/BSB data it stores, and never touch raw
    card numbers.
    """
    encryption_active = _fernet is not None
    return {
        "encryption_at_rest": encryption_active,
        "encryption_algorithm": "AES-128-CBC (Fernet)" if encryption_active else "NOT CONFIGURED — sensitive fields cannot be stored until ENCRYPTION_KEY is set",
        "card_storage_policy": "Raw card numbers are never received or stored by BillSure — card payments go through Stripe's hosted checkout. Only the last 4 digits (returned by Stripe) are retained for display.",
        "bank_details_encrypted": encryption_active,
        "ddr_data_encrypted": encryption_active,
        "card_data_pci_scope": "Out of scope for BillSure — Stripe (PCI DSS Level 1 certified) handles all card collection on its own hosted pages.",
        "becs_direct_debit": "Available via Stripe Checkout, and via BillSure's own Direct Debit Request mandate (bank details encrypted at rest for the latter)",
        "sensitive_fields_encrypted": [
            "bank_details.account_number",
            "bank_details.routing_number",
            "direct_debit_requests.bsb",
            "direct_debit_requests.account_number",
            "direct_debit_requests.provider_account_number",
            "payment_methods.bsb"
        ],
        "compliance_notes": [
            "BillSure is NOT independently PCI DSS certified — that claim should not be made in marketing copy",
            "Financial fields listed above are encrypted at rest using Fernet (AES-128-CBC) whenever ENCRYPTION_KEY is configured",
            "Raw card numbers are never received by BillSure — only last 4 digits (from Stripe) are retained",
            "Bank account numbers/BSBs collected via BillSure's own Direct Debit Request form ARE stored by BillSure (encrypted), not exclusively by Stripe",
            "Stripe independently holds PCI DSS Level 1 certification for the card/BECS collection it performs on its own hosted pages",
            "API responses always return masked values (****XXXX format), never raw stored values"
        ]
    }


@api_router.post("/admin/encrypt-existing-data")
async def encrypt_existing_data(admin_user: dict = Depends(get_admin_user)):
    """One-time migration: encrypt any plaintext bank/DDR data in the database."""
    if not _fernet:
        raise HTTPException(status_code=500, detail="Encryption key not configured")

    migrated = {"bank_details": 0, "ddr_mandates": 0, "payment_methods": 0}

    # Migrate bank_details
    bank_docs = await sdb.find_many("bank_details")
    for doc in bank_docs:
        acct = doc.get("account_number", "")
        rout = doc.get("routing_number", "")
        if acct and not acct.startswith("gAAAAA"):
            await sdb.update_one("bank_details",
                {"id": doc["id"]},
                {"$set": {
                    "account_number": encrypt_field(acct),
                    "routing_number": encrypt_field(rout)
                }}
            )
            migrated["bank_details"] += 1

    # Migrate DDR mandates
    ddr_docs = await sdb.find_many("direct_debit_requests")
    for doc in ddr_docs:
        bsb = doc.get("bsb", "")
        if bsb and not bsb.startswith("gAAAAA"):
            await sdb.update_one("direct_debit_requests",
                {"id": doc["id"]},
                {"$set": {
                    "bsb": encrypt_field(bsb),
                    "account_number": encrypt_field(doc.get("account_number", ""))
                }}
            )
            migrated["ddr_mandates"] += 1

    # Migrate payment_methods BSB
    pm_docs = await sdb.find_many("payment_methods")
    for doc in pm_docs:
        bsb = doc.get("bsb")
        if bsb and not bsb.startswith("gAAAAA"):
            await sdb.update_one("payment_methods",
                {"id": doc["id"]},
                {"$set": {"bsb": encrypt_field(bsb)}}
            )
            migrated["payment_methods"] += 1

    return {"message": "Encryption migration complete", "migrated_records": migrated}



# ========== AI BILL INTELLIGENCE ==========

UTILITY_BENCHMARKS = {
    "Electricity": {"avg_monthly": 150, "avg_quarterly": 450, "low": 80, "high": 300, "unit": "quarter"},
    "Gas": {"avg_monthly": 80, "avg_quarterly": 240, "low": 40, "high": 200, "unit": "quarter"},
    "Water": {"avg_monthly": 60, "avg_quarterly": 180, "low": 30, "high": 150, "unit": "quarter"},
    "Internet": {"avg_monthly": 75, "avg_quarterly": 225, "low": 50, "high": 120, "unit": "month"},
    "Mobile": {"avg_monthly": 50, "avg_quarterly": 150, "low": 20, "high": 80, "unit": "month"},
    "Council": {"avg_monthly": 150, "avg_quarterly": 450, "low": 100, "high": 500, "unit": "quarter"},
    "Insurance": {"avg_monthly": 120, "avg_quarterly": 360, "low": 60, "high": 300, "unit": "quarter"},
    "Other": {"avg_monthly": 100, "avg_quarterly": 300, "low": 30, "high": 250, "unit": "month"},
}


def compute_bill_analytics(bills: list) -> dict:
    """Compute spending analytics from user bills without AI."""
    now = datetime.now(timezone.utc)

    category_spend = {}
    provider_spend = {}
    monthly_spend = {}
    total_spend = 0
    bill_count = len(bills)

    for b in bills:
        amt = b.get("amount", 0) or 0
        cat = b.get("category", "Other")
        prov = b.get("provider", "Unknown")
        total_spend += amt

        # Category aggregation
        if cat not in category_spend:
            category_spend[cat] = {"total": 0, "count": 0, "bills": []}
        category_spend[cat]["total"] += amt
        category_spend[cat]["count"] += 1
        category_spend[cat]["bills"].append(b)

        # Provider aggregation
        key = f"{cat}::{prov}"
        if key not in provider_spend:
            provider_spend[key] = {"category": cat, "provider": prov, "total": 0, "count": 0}
        provider_spend[key]["total"] += amt
        provider_spend[key]["count"] += 1

        # Monthly trend
        created = b.get("created_at", "") or b.get("due_date", "")
        if created:
            month_key = created[:7]  # YYYY-MM
            if month_key not in monthly_spend:
                monthly_spend[month_key] = 0
            monthly_spend[month_key] += amt

    # Sort months
    sorted_months = sorted(monthly_spend.keys())
    monthly_trend = [{"month": m, "amount": round(monthly_spend[m], 2)} for m in sorted_months]

    # Detect category trends
    category_insights = []
    for cat, data in category_spend.items():
        avg_per_bill = data["total"] / data["count"] if data["count"] > 0 else 0
        benchmark = UTILITY_BENCHMARKS.get(cat, UTILITY_BENCHMARKS["Other"])

        # Compare to benchmark
        benchmark_avg = benchmark["avg_quarterly"] if benchmark["unit"] == "quarter" else benchmark["avg_monthly"]
        status = "normal"
        if avg_per_bill > benchmark["high"]:
            status = "high"
        elif avg_per_bill < benchmark["low"]:
            status = "low"

        category_insights.append({
            "category": cat,
            "total_spent": round(data["total"], 2),
            "bill_count": data["count"],
            "avg_per_bill": round(avg_per_bill, 2),
            "benchmark_avg": benchmark_avg,
            "benchmark_low": benchmark["low"],
            "benchmark_high": benchmark["high"],
            "status": status,
        })

    # Provider comparison within categories
    provider_comparison = []
    cats_with_providers = {}
    for key, data in provider_spend.items():
        cat = data["category"]
        if cat not in cats_with_providers:
            cats_with_providers[cat] = []
        cats_with_providers[cat].append(data)

    for cat, providers in cats_with_providers.items():
        if len(providers) >= 1:
            sorted_provs = sorted(providers, key=lambda x: x["total"] / x["count"])
            provider_comparison.append({
                "category": cat,
                "providers": [
                    {
                        "name": p["provider"],
                        "avg_cost": round(p["total"] / p["count"], 2),
                        "total": round(p["total"], 2),
                        "bill_count": p["count"],
                    }
                    for p in sorted_provs
                ],
            })

    # Overall trend direction
    trend_direction = "stable"
    if len(sorted_months) >= 2:
        recent = monthly_spend.get(sorted_months[-1], 0)
        prev = monthly_spend.get(sorted_months[-2], 0)
        if prev > 0:
            change_pct = ((recent - prev) / prev) * 100
            if change_pct > 10:
                trend_direction = "increasing"
            elif change_pct < -10:
                trend_direction = "decreasing"

    return {
        "total_spend": round(total_spend, 2),
        "bill_count": bill_count,
        "category_insights": category_insights,
        "provider_comparison": provider_comparison,
        "monthly_trend": monthly_trend,
        "trend_direction": trend_direction,
    }


# Simple in-memory cache for AI insights (15 min TTL)
_insights_cache: Dict[str, dict] = {}
_INSIGHTS_TTL = 900  # 15 minutes


@api_router.get("/insights/analyze")
async def get_bill_insights(user=Depends(get_current_user)):
    """AI-powered bill intelligence. Analytics for all, AI insights for Standard+."""
    bills = await sdb.find_many("bills", {"user_id": user["id"]})

    if not bills:
        return {
            "analytics": None,
            "ai_insights": None,
            "message": "No bills found. Upload some bills to get personalized insights."
        }

    analytics = compute_bill_analytics(bills)

    # Check cache
    import hashlib
    cache_key = hashlib.sha256(f"{user['id']}:{len(bills)}:{analytics['total_spend']}".encode()).hexdigest()
    cached = _insights_cache.get(cache_key)
    now_ts = datetime.now(timezone.utc).timestamp()

    ai_insights = None
    if cached and (now_ts - cached["ts"]) < _INSIGHTS_TTL:
        ai_insights = cached["data"]
    elif EMERGENT_LLM_KEY:
        try:
            # Build a concise summary for GPT
            summary_lines = [
                f"User has {analytics['bill_count']} bills totalling ${analytics['total_spend']:.2f}.",
                f"Overall spending trend: {analytics['trend_direction']}.",
                "Category breakdown:"
            ]
            for ci in analytics["category_insights"]:
                comparison = ""
                if ci["status"] == "high":
                    comparison = f" (ABOVE avg benchmark of ${ci['benchmark_avg']})"
                elif ci["status"] == "low":
                    comparison = f" (below avg benchmark of ${ci['benchmark_avg']})"
                summary_lines.append(
                    f"- {ci['category']}: ${ci['total_spent']} across {ci['bill_count']} bills, avg ${ci['avg_per_bill']}/bill{comparison}"
                )

            if analytics["provider_comparison"]:
                summary_lines.append("Providers by category:")
                for pc in analytics["provider_comparison"]:
                    provs = ", ".join([f"{p['name']} (avg ${p['avg_cost']})" for p in pc["providers"]])
                    summary_lines.append(f"- {pc['category']}: {provs}")

            if analytics["monthly_trend"]:
                recent_months = analytics["monthly_trend"][-6:]
                summary_lines.append("Monthly spend (last 6 months):")
                for mt in recent_months:
                    summary_lines.append(f"- {mt['month']}: ${mt['amount']}")

            data_summary = "\n".join(summary_lines)

            chat = LlmChat(
                api_key=EMERGENT_LLM_KEY,
                session_id=f"insights-{user['id']}-{uuid.uuid4().hex[:6]}",
                system_message="""You are a friendly Australian financial advisor specialising in household utility bills.
Analyse the user's bill data and provide practical, actionable insights.
Return ONLY valid JSON (no markdown, no code fences) with this structure:
{
  "summary": "A 2-3 sentence overview of their spending patterns",
  "highlights": [
    {"type": "increasing|decreasing|stable|warning|saving", "title": "Short title", "description": "1-2 sentence detail", "category": "category_name or null"}
  ],
  "savings_tips": [
    {"tip": "Actionable tip text", "potential_saving": "$X/month or $X/year", "category": "category_name or General", "priority": "high|medium|low"}
  ],
  "provider_insights": [
    {"category": "category_name", "insight": "Comparison insight or suggestion"}
  ],
  "seasonal_note": "Any seasonal pattern observation or null"
}
Rules:
- Give 3-5 highlights covering trends, anomalies, and positive patterns
- Give 3-5 savings tips ranked by potential impact, specific to Australian utility market
- Reference actual providers and amounts from the data where relevant
- Be encouraging and practical, not alarming
- Mention Australian-specific programs like energy rebates, concession cards, GreenPower
- If spending is below benchmarks, acknowledge that positively
- All dollar amounts in AUD"""
            )
            chat.with_model("openai", "gpt-4o")

            response = await chat.send_message(
                UserMessage(text=f"Analyse this Australian household's bill data and provide insights:\n\n{data_summary}")
            )

            clean = response.strip()
            if clean.startswith("```"):
                clean = re.sub(r'^```(?:json)?\s*', '', clean)
                clean = re.sub(r'\s*```$', '', clean)

            import json as json_mod
            ai_insights = json_mod.loads(clean)
            _insights_cache[cache_key] = {"data": ai_insights, "ts": now_ts}
        except Exception as e:
            logger.warning(f"AI insights generation failed: {e}")
            ai_insights = None

    return {
        "analytics": analytics,
        "ai_insights": ai_insights,
    }


# ========== V2 BILLING ENGINE ENDPOINTS ==========
from billing_engine import (
    predict_annual_bills,
    calculate_smoothed_payment,
    compute_plan_health,
    calculate_savings_comparison,
    SUBSCRIPTION_TIERS,
)


async def _get_user_tier(user_id: str) -> dict:
    """Get user's subscription tier with feature flags."""
    sub = await sdb.find_one("subscriptions", {"user_id": user_id})
    tier_id = sub.get("tier", "basic") if sub else "basic"
    return SUBSCRIPTION_TIERS.get(tier_id, SUBSCRIPTION_TIERS["basic"])


def _require_feature(tier: dict, feature: str):
    """Raise 403 if the user's tier doesn't include the feature."""
    if not tier.get(feature):
        raise HTTPException(
            status_code=403,
            detail=f"This feature requires a Standard or Premium plan. Please upgrade at /dashboard/subscription."
        )


@api_router.get("/v2/predict-bills")
async def v2_predict_bills(current_user: dict = Depends(get_current_user)):
    """12-month bill forecast with seasonal weighting. Requires Standard+."""
    tier = await _get_user_tier(current_user["id"])
    _require_feature(tier, "has_forecasting")
    bills = await sdb.find_many("bills", {"user_id": current_user["id"]})
    if not bills:
        return {"prediction": None, "message": "No bills to predict."}
    return {"prediction": predict_annual_bills(bills)}


@api_router.get("/v2/simulate-plan")
async def v2_simulate_plan(
    frequency: str = "monthly",
    current_user: dict = Depends(get_current_user),
):
    """Simulate smoothed payment plan with seasonal adjustments."""
    if frequency not in ("weekly", "fortnightly", "monthly"):
        raise HTTPException(status_code=400, detail="Invalid frequency")

    tier = await _get_user_tier(current_user["id"])
    _require_feature(tier, "has_forecasting")

    bills = await sdb.find_many("bills", {"user_id": current_user["id"], "status": "pending"})
    if not bills:
        return {"simulation": None, "message": "No pending bills."}

    buffer_pct = tier["buffer_pct"]
    simulation = calculate_smoothed_payment(bills, frequency, buffer_pct)
    return {"simulation": simulation}


@api_router.get("/v2/plan-health")
async def v2_plan_health(current_user: dict = Depends(get_current_user)):
    """Excess/deficit balancing — is the plan on track? Requires Standard+."""
    tier = await _get_user_tier(current_user["id"])
    _require_feature(tier, "has_forecasting")
    plan = await sdb.find_one("payment_plans", {"user_id": current_user["id"]})
    if not plan or plan.get("status") != "active":
        return {"health": None, "message": "No active plan."}

    bills = await sdb.find_many("bills", {"user_id": current_user["id"]})
    transactions = await sdb.find_many("transactions", {"user_id": current_user["id"]})
    user = await sdb.find_one("users", {"id": current_user["id"]})

    health = compute_plan_health(plan, bills, user.get("wallet_balance", 0), transactions)
    return {"health": health}


@api_router.get("/v2/savings-comparison")
async def v2_savings_comparison(current_user: dict = Depends(get_current_user)):
    """Compare smoothed vs traditional billing. Requires Standard+."""
    tier = await _get_user_tier(current_user["id"])
    _require_feature(tier, "has_forecasting")

    bills = await sdb.find_many("bills", {"user_id": current_user["id"]})
    if not bills:
        return {"comparison": None, "message": "No bills to compare."}

    comparison = calculate_savings_comparison(bills, tier["buffer_pct"])
    return {"comparison": comparison}


@api_router.get("/v2/subscription/tiers")
async def v2_get_subscription_tiers():
    """Get all available subscription tiers."""
    return {"tiers": list(SUBSCRIPTION_TIERS.values())}


@api_router.get("/v2/subscription/current")
async def v2_get_current_subscription(current_user: dict = Depends(get_current_user)):
    """Get user's current subscription."""
    sub = await sdb.find_one("subscriptions", {"user_id": current_user["id"]})
    if not sub:
        return {"subscription": SUBSCRIPTION_TIERS["basic"], "tier": "basic"}
    tier = SUBSCRIPTION_TIERS.get(sub.get("tier", "basic"), SUBSCRIPTION_TIERS["basic"])
    return {"subscription": tier, "tier": sub.get("tier", "basic")}


@api_router.post("/v2/subscription/select")
async def v2_select_subscription(tier: str, current_user: dict = Depends(get_current_user)):
    """Select a subscription tier."""
    if tier not in SUBSCRIPTION_TIERS:
        raise HTTPException(status_code=400, detail="Invalid tier")

    sub_doc = {
        "user_id": current_user["id"],
        "tier": tier,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "active",
    }
    await sdb.delete_many("subscriptions", {"user_id": current_user["id"]})
    await sdb.insert_one("subscriptions", sub_doc)
    return {
        "message": f"Subscribed to {SUBSCRIPTION_TIERS[tier]['name']} plan",
        "subscription": SUBSCRIPTION_TIERS[tier],
        "tier": tier,
    }


# Include router
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', 'http://localhost:3000').split(','),
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "Accept"],
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("startup")
async def startup_event():
    """Seed admin/test user (only if explicitly enabled) and start background schedulers."""
    sb = get_supabase_admin()

    # Ledger system accounts (TRUST_BANK, FEES_RECEIVABLE, OPERATING,
    # SUSPENSE) must exist before any of the loops below try to post a
    # journal against them. Idempotent — safe to call on every startup.
    await ledger.ensure_system_accounts()

    def _start_background_loops():
        if SCHEDULER_MODE == 'cron':
            logger.info(
                "SCHEDULER_MODE=cron — scheduled-collections, payment-run-queue and "
                "reconciliation are NOT started as in-process loops; an external cron "
                "must call POST /api/internal/cron/{scheduled-collections,payment-run-queue,reconciliation} "
                "with header X-Cron-Secret. generate_notifications still runs in-process."
            )
        elif SCHEDULER_MODE == 'apscheduler':
            import scheduler as persistent_scheduler
            persistent_scheduler.start()
            logger.info(
                "SCHEDULER_MODE=apscheduler — scheduled-collections, payment-run-queue and "
                "reconciliation are NOT started as in-process loops; backend/scheduler.py's "
                "persistent-job-store scheduler owns them instead. generate_notifications "
                "still runs in-process."
            )
        else:
            asyncio.create_task(process_scheduled_collections())
            asyncio.create_task(payment_run_scheduler_loop())
            asyncio.create_task(reconciliation_loop())
        asyncio.create_task(generate_notifications())

    # SECURITY FIX: this used to seed an admin account with a hardcoded
    # fallback password ('Admin123!') on every startup whenever
    # SEED_ADMIN_PASSWORD wasn't set — a well-known default that grants full
    # admin access (is_admin: True) to anyone who guesses/knows it. Seeding is
    # now opt-in via SEED_DEMO_DATA=true, and every credential must be
    # explicitly supplied via its own env var — there is no built-in
    # password. Skip and log clearly rather than falling back to a default.
    if os.environ.get('SEED_DEMO_DATA', 'false').lower() != 'true':
        _start_background_loops()
        logger.info(f"SEED_DEMO_DATA not enabled — skipping demo account seeding. Scheduler mode: {SCHEDULER_MODE}.")
        return

    SEED_ADMIN_EMAIL = os.environ.get('SEED_ADMIN_EMAIL', '')
    SEED_ADMIN_PASSWORD = os.environ.get('SEED_ADMIN_PASSWORD', '')
    SEED_TEST_EMAIL = os.environ.get('SEED_TEST_EMAIL', '')
    SEED_TEST_PASSWORD = os.environ.get('SEED_TEST_PASSWORD', '')

    if not SEED_ADMIN_EMAIL or not SEED_ADMIN_PASSWORD:
        logger.warning("SEED_DEMO_DATA is enabled but SEED_ADMIN_EMAIL/SEED_ADMIN_PASSWORD are not set — skipping admin seeding.")
    else:
        await _seed_admin_user(sb, SEED_ADMIN_EMAIL, SEED_ADMIN_PASSWORD)

    if not SEED_TEST_EMAIL or not SEED_TEST_PASSWORD:
        logger.warning("SEED_DEMO_DATA is enabled but SEED_TEST_EMAIL/SEED_TEST_PASSWORD are not set — skipping test user seeding.")
    else:
        await _seed_test_user(sb, SEED_TEST_EMAIL, SEED_TEST_PASSWORD)

    _start_background_loops()
    logger.info(f"Scheduler mode: {SCHEDULER_MODE}.")


async def _seed_admin_user(sb, SEED_ADMIN_EMAIL: str, SEED_ADMIN_PASSWORD: str):
    # Seed admin user if not exists
    admin = await sdb.find_one("users", {"email": SEED_ADMIN_EMAIL})
    if not admin:
        supabase_uid = None
        if sb:
            try:
                auth_res = sb.auth.admin.create_user({
                    "email": SEED_ADMIN_EMAIL,
                    "password": SEED_ADMIN_PASSWORD,
                    "email_confirm": True,
                })
                supabase_uid = auth_res.user.id
            except Exception as e:
                logger.warning(f"Admin Supabase auth seed: {e}")
        admin_dict = {
            "id": str(uuid.uuid4()),
            "full_name": "Admin User",
            "email": SEED_ADMIN_EMAIL,
            "password": hash_password(SEED_ADMIN_PASSWORD),
            "phone": "",
            "wallet_balance": 0,
            "is_admin": True,
            "role": "admin",
            "stripe_customer_id": "",
            "subscription_fee": 0,
            "supabase_uid": supabase_uid,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        await sdb.insert_one("users", admin_dict)
        logger.info(f"Admin user seeded: {SEED_ADMIN_EMAIL}")


async def _seed_test_user(sb, SEED_TEST_EMAIL: str, SEED_TEST_PASSWORD: str):
    # Seed test customer if not exists
    test_user = await sdb.find_one("users", {"email": SEED_TEST_EMAIL})
    if not test_user:
        supabase_uid = None
        if sb:
            try:
                auth_res = sb.auth.admin.create_user({
                    "email": SEED_TEST_EMAIL,
                    "password": SEED_TEST_PASSWORD,
                    "email_confirm": True,
                })
                supabase_uid = auth_res.user.id
            except Exception as e:
                logger.warning(f"Test user Supabase auth seed: {e}")
        test_dict = {
            "id": str(uuid.uuid4()),
            "full_name": "Test User",
            "email": SEED_TEST_EMAIL,
            "password": hash_password(SEED_TEST_PASSWORD),
            "phone": "",
            "wallet_balance": 500.0,
            "is_admin": False,
            "role": "customer",
            "stripe_customer_id": "",
            "subscription_fee": 0,
            "supabase_uid": supabase_uid,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        await sdb.insert_one("users", test_dict)
        logger.info(f"Test user seeded: {SEED_TEST_EMAIL}")

        # Seed sample bills for test user
        sample_bills = [
            {"category": "Electricity", "provider": "AGL Energy", "amount": 280.50, "frequency": "quarterly"},
            {"category": "Electricity", "provider": "AGL Energy", "amount": 281.95, "frequency": "quarterly"},
            {"category": "Gas", "provider": "Origin Energy", "amount": 145.00, "frequency": "quarterly"},
            {"category": "Gas", "provider": "Origin Gas", "amount": 89.50, "frequency": "monthly"},
            {"category": "Water", "provider": "Sydney Water", "amount": 84.50, "frequency": "quarterly"},
            {"category": "Internet", "provider": "Aussie Broadband", "amount": 87.00, "frequency": "monthly"},
        ]
        for sb_data in sample_bills:
            bill = {
                "id": str(uuid.uuid4()),
                "user_id": test_dict["id"],
                "category": sb_data["category"],
                "provider": sb_data["provider"],
                "account_number": "",
                "amount": sb_data["amount"],
                "due_date": (datetime.now(timezone.utc) + timedelta(days=30)).strftime("%Y-%m-%d"),
                "frequency": sb_data["frequency"],
                "status": "pending",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            await sdb.insert_one("bills", bill)
        logger.info(f"Seeded {len(sample_bills)} sample bills for test user")


@app.on_event("shutdown")
async def shutdown_db_client():
    if SCHEDULER_MODE == 'apscheduler':
        import scheduler as persistent_scheduler
        persistent_scheduler.shutdown()
    # Supabase client handles its own cleanup