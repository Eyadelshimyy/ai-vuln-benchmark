"""BillSure — Utility functions: auth, encryption, email
Now using Supabase Auth for authentication with custom JWT fallback.
"""
import os
import logging
import secrets
import asyncio
from datetime import datetime, timezone, timedelta
from passlib.context import CryptContext
from cryptography.fernet import Fernet
import jwt
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import supabase_db as sdb
from supabase import create_client

logger = logging.getLogger(__name__)

# Security
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# SECURITY FIX: this used to fall back to the hardcoded literal string
# 'your-secret-key-change-in-production' whenever JWT_SECRET was unset. That
# string is baked into the source (and therefore into every deployment that
# forgets to set the env var), so anyone could forge a JWT for any user_id —
# including an admin's — and pass get_current_user()/get_admin_user() below.
# There is now no usable hardcoded fallback: if JWT_SECRET isn't set we
# generate a random secret for this process's lifetime and log a loud
# warning. This keeps local/dev usage working without a config step, while
# guaranteeing the secret is never a known, guessable value. Tokens will not
# survive a process restart if JWT_SECRET is left unset (all users would need
# to log in again) — set JWT_SECRET explicitly in any real deployment so
# tokens remain valid across restarts/multiple instances.
_JWT_SECRET_ENV = os.environ.get('JWT_SECRET', '')
if _JWT_SECRET_ENV:
    SECRET_KEY = _JWT_SECRET_ENV
else:
    SECRET_KEY = secrets.token_hex(32)
    logger.warning(
        "JWT_SECRET is not set — generated a random, process-local signing key. "
        "All existing sessions will be invalidated on restart, and tokens won't "
        "be valid across multiple instances. Set the JWT_SECRET environment "
        "variable to a long random value before deploying to production."
    )
ALGORITHM = "HS256"
TOKEN_EXPIRE_HOURS = int(os.environ.get('TOKEN_EXPIRE_HOURS', '4'))

# Supabase Auth
_supabase_admin = None

def get_supabase_admin():
    global _supabase_admin
    if _supabase_admin is None:
        url = os.environ.get('SUPABASE_URL', '')
        key = os.environ.get('SUPABASE_SERVICE_KEY', '')
        if url and key:
            _supabase_admin = create_client(url, key)
    return _supabase_admin

# Encryption
_enc_key = os.environ.get('ENCRYPTION_KEY', '')
_fernet = Fernet(_enc_key.encode()) if _enc_key else None
if not _fernet:
    logger.warning(
        "ENCRYPTION_KEY is not set — bank account numbers, BSBs and DDR data "
        "CANNOT be encrypted at rest. encrypt_field()/decrypt_field() will now "
        "raise instead of silently storing plaintext financial data. Generate a "
        "key with `Fernet.generate_key()` and set it as ENCRYPTION_KEY."
    )


def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(hours=TOKEN_EXPIRE_HOURS))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str):
    """Decode a custom JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials

    # Try custom JWT first (backwards compatibility)
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("user_id")
        if user_id:
            user = await sdb.find_one("users", {"id": user_id})
            if user:
                return user
    except Exception:
        pass

    # Try Supabase token verification via admin API
    sb = get_supabase_admin()
    if sb:
        try:
            auth_response = sb.auth.get_user(token)
            if auth_response and auth_response.user:
                sb_user = auth_response.user
                # Find by supabase_uid or email
                user = await sdb.find_one("users", {"supabase_uid": sb_user.id})
                if not user and sb_user.email:
                    user = await sdb.find_one("users", {"email": sb_user.email})
                    # Link supabase_uid
                    if user and not user.get("supabase_uid"):
                        await sdb.update_one("users", {"id": user["id"]}, {"$set": {"supabase_uid": sb_user.id}})
                        user["supabase_uid"] = sb_user.id
                if user:
                    return user
        except Exception as e:
            logger.debug(f"Supabase token verification failed: {e}")

    raise HTTPException(status_code=401, detail="Invalid or expired token")


async def get_admin_user(current_user: dict = Depends(get_current_user)):
    if not current_user.get("is_admin"):
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user


def encrypt_field(value: str) -> str:
    if not value:
        return value
    if not _fernet:
        # SECURITY FIX: this used to silently return the plaintext value when
        # ENCRYPTION_KEY was unset, meaning bank/BSB/account numbers would be
        # written to the database completely unencrypted with no warning or
        # error anywhere. Fail closed instead — refuse to store sensitive
        # financial data unencrypted rather than doing so silently.
        raise HTTPException(
            status_code=500,
            detail="Server encryption is not configured; cannot store sensitive financial data. Contact support.",
        )
    return _fernet.encrypt(value.encode()).decode()

def decrypt_field(value: str) -> str:
    if not value:
        return value
    if not _fernet:
        raise HTTPException(
            status_code=500,
            detail="Server encryption is not configured; cannot read sensitive financial data. Contact support.",
        )
    try:
        return _fernet.decrypt(value.encode()).decode()
    except Exception:
        # Value doesn't decrypt under the current key — most likely legacy
        # plaintext written before ENCRYPTION_KEY was configured (see
        # /admin/encrypt-existing-data). Return as-is rather than failing the
        # whole request; this is a data-compatibility fallback, not a "key
        # missing" case (that's handled above).
        return value

# Email
RESEND_API_KEY = os.environ.get('RESEND_API_KEY', '')
SENDER_EMAIL = os.environ.get('SENDER_EMAIL', 'noreply@billsure.com.au')

if RESEND_API_KEY:
    import resend
    resend.api_key = RESEND_API_KEY

async def send_email(to_email: str, subject: str, html_body: str):
    if not RESEND_API_KEY:
        logger.info(f"[EMAIL SIM] To: {to_email} | Subject: {subject}")
        return False
    try:
        params = {"from": SENDER_EMAIL, "to": [to_email], "subject": subject, "html": html_body}
        await asyncio.to_thread(resend.Emails.send, params)
        logger.info(f"[EMAIL SENT] To: {to_email} | Subject: {subject}")
        return True
    except Exception as e:
        logger.error(f"[EMAIL FAIL] To: {to_email} | Error: {e}")
        return False
