"""
Legacy payment middleware retained for historical reference only.

This app now uses x402 as the active payment flow. The middleware below was
used during an earlier Stripe MPP experiment and is no longer part of the
runtime request path.
"""

import base64
import hashlib
import hmac
import json
import time
import uuid
from typing import Awaitable, Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

ROUTE_PRICING: dict[str, int] = {
    "/paid/diagrams/generate/svg": 5,    # $0.05 in cents
    "/paid/diagrams/generate/png": 7,    # $0.07 in cents
}

_SCHEME = "Payment"
_CHALLENGE_TTL = 300  # 5 minutes


def _make_secret_key(secret_key: str) -> bytes:
    return hmac.new(
        secret_key.encode(),
        b"legacy-payment-signing",
        hashlib.sha256,
    ).digest()


def _build_challenge(
    *,
    realm: str,
    amount_cents: int,
    network_id: str,
    secret_key: bytes,
) -> str:
    challenge_id = str(uuid.uuid4())
    expires_at = int(time.time()) + _CHALLENGE_TTL

    payload = {
        "id": challenge_id,
        "realm": realm,
        "expiresAt": expires_at,
        "methods": [
            {
                "type": "legacy-payment",
                "amount": str(amount_cents),
                "currency": "usd",
                "networkId": network_id,
                "paymentMethodTypes": ["card", "link"],
            }
        ],
    }

    payload_bytes = json.dumps(payload, separators=(",", ":")).encode()
    sig = hmac.new(secret_key, payload_bytes, hashlib.sha256).digest()
    sig_b64 = base64.urlsafe_b64encode(sig).decode().rstrip("=")

    payload_b64 = base64.urlsafe_b64encode(payload_bytes).decode().rstrip("=")
    token = f"{payload_b64}.{sig_b64}"

    return f'{_SCHEME} realm="{realm}", token="{token}"'


def _is_valid_credential(authorization: str | None, secret_key: bytes) -> bool:
    """Verify a Payment credential token from the Authorization header."""
    if not authorization or not authorization.startswith(f"{_SCHEME} "):
        return False

    token_part = authorization[len(f"{_SCHEME} "):]
    # Extract token= value
    for part in token_part.split(","):
        part = part.strip()
        if part.startswith("token="):
            token = part[6:].strip('"')
            try:
                payload_b64, sig_b64 = token.rsplit(".", 1)
                # Re-pad base64
                payload_bytes = base64.urlsafe_b64decode(payload_b64 + "==")
                expected_sig = hmac.new(secret_key, payload_bytes, hashlib.sha256).digest()
                expected_b64 = base64.urlsafe_b64encode(expected_sig).decode().rstrip("=")
                if not hmac.compare_digest(sig_b64, expected_b64):
                    return False
                payload = json.loads(payload_bytes)
                if payload.get("expiresAt", 0) < int(time.time()):
                    return False
                return True
            except Exception:
                return False
    return False


class LegacyPaymentMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app: ASGIApp,
        secret_key: str,
        network_id: str,
        defer_to_x402: bool = False,
    ) -> None:
        super().__init__(app)
        self._profile_id = network_id
        self._secret_key = _make_secret_key(secret_key)
        self._defer_to_x402 = defer_to_x402

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        path = request.url.path
        method = request.method

        if method != "POST" or path not in ROUTE_PRICING:
            return await call_next(request)

        # If x402 is handling these routes, pass through entirely
        if self._defer_to_x402:
            return await call_next(request)

        # If request carries an x402 X-PAYMENT header, let x402 middleware handle it
        if request.headers.get("X-PAYMENT"):
            return await call_next(request)

        amount_cents = ROUTE_PRICING[path]
        realm = str(request.base_url).rstrip("/")
        authorization = request.headers.get("Authorization")

        if not _is_valid_credential(authorization, self._secret_key):
            challenge = _build_challenge(
                realm=realm,
                amount_cents=amount_cents,
                network_id=self._profile_id,
                secret_key=self._secret_key,
            )
            return JSONResponse(
                status_code=402,
                content={"error": "Payment required", "detail": "Valid legacy payment credential required."},
                headers={"WWW-Authenticate": challenge},
            )

        return await call_next(request)
