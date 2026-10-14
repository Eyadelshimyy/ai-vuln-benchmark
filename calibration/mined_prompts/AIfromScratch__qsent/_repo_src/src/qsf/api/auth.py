import os
import re
import secrets
from datetime import datetime, timedelta, timezone

from authlib.integrations.httpx_client import AsyncOAuth2Client
from fastapi import APIRouter, Cookie, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse
from jose import JWTError, jwt

router = APIRouter(prefix="/auth", tags=["auth"])

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")
BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000")
SECURE_COOKIES = os.environ.get("SECURE_COOKIES", "false").lower() == "true"

REDIRECT_URI = f"{BASE_URL}/auth/callback"
ALGORITHM = "HS256"
SESSION_DURATION_HOURS = 8
COOKIE_NAME = "qsent_session"

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"

_ALLOWED_PICTURE_ORIGINS = re.compile(r"^https://lh[0-9]+\.googleusercontent\.com/")


def _sanitize_picture(url: str) -> str:
    if url and _ALLOWED_PICTURE_ORIGINS.match(url):
        return url
    return ""


def create_session_token(email: str, name: str, picture: str = "", google_sub: str = "") -> str:
    expire = datetime.now(timezone.utc) + timedelta(hours=SESSION_DURATION_HOURS)
    return jwt.encode(
        {"sub": email, "google_sub": google_sub, "name": name, "picture": picture, "exp": expire},
        SECRET_KEY,
        algorithm=ALGORITHM,
    )


def decode_session_token(token: str) -> dict:
    return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])


async def get_current_user(qsent_session: str | None = Cookie(default=None)) -> dict:
    if not qsent_session:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload = decode_session_token(qsent_session)
        return {
            "email": payload["sub"],
            "google_sub": payload.get("google_sub", ""),
            "name": payload["name"],
            "picture": payload.get("picture", ""),
        }
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired session")


@router.get("/login")
async def login():
    state = secrets.token_urlsafe(32)
    async with AsyncOAuth2Client(
        client_id=GOOGLE_CLIENT_ID,
        redirect_uri=REDIRECT_URI,
        scope="openid email profile",
    ) as client:
        uri, _ = client.create_authorization_url(GOOGLE_AUTH_URL, state=state)

    response = RedirectResponse(url=uri)
    response.set_cookie(
        key="oauth_state",
        value=state,
        httponly=True,
        max_age=300,
        samesite="lax",
    )
    return response


@router.get("/callback")
async def callback(request: Request):
    state_param = request.query_params.get("state")
    code = request.query_params.get("code")
    oauth_state = request.cookies.get("oauth_state")

    if not oauth_state or state_param != oauth_state:
        raise HTTPException(status_code=400, detail="Invalid state parameter")
    if not code:
        raise HTTPException(status_code=400, detail="Missing authorization code")

    async with AsyncOAuth2Client(
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        redirect_uri=REDIRECT_URI,
    ) as client:
        await client.fetch_token(GOOGLE_TOKEN_URL, code=code)
        userinfo_resp = await client.get(GOOGLE_USERINFO_URL)
        userinfo = userinfo_resp.json()

    session_token = create_session_token(
        email=userinfo["email"],
        name=userinfo.get("name", userinfo["email"]),
        picture=_sanitize_picture(userinfo.get("picture", "")),
        google_sub=userinfo.get("sub", ""),
    )

    response = RedirectResponse(url="/")
    response.delete_cookie("oauth_state")
    response.set_cookie(
        key=COOKIE_NAME,
        value=session_token,
        httponly=True,
        secure=SECURE_COOKIES,
        samesite="lax",
        max_age=SESSION_DURATION_HOURS * 3600,
    )
    return response


@router.get("/logout")
async def logout():
    response = RedirectResponse(url="/login.html")
    response.delete_cookie(COOKIE_NAME)
    return response


@router.get("/me")
async def me(user: dict = Depends(get_current_user)):
    return user


# Test-only bypass: only registered when TEST_MODE=true, never in production
if os.environ.get("TEST_MODE") == "true":
    @router.get("/test-login")
    async def test_login():
        token = create_session_token(email="test@example.com", name="Test User", google_sub="test-user-sub")
        response = RedirectResponse(url="/")
        response.set_cookie(
            key=COOKIE_NAME,
            value=token,
            httponly=True,
            samesite="lax",
            max_age=SESSION_DURATION_HOURS * 3600,
        )
        return response
