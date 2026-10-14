"""Stripe webhook signing (HMAC-SHA256 ``v1``).

Stripe sends ``Stripe-Signature: t=<unix seconds>,v1=<hex>[,v0=...]``; the
signed payload is ``"{t}.{raw body}"`` HMAC'd with the endpoint's signing
secret. The timestamp is bounded by ``tolerance`` so a captured request cannot
be replayed later. Implemented by hand so no SDK is pulled in for one check.
"""

import hashlib
import hmac
import time
from collections.abc import Callable

SIGNATURE_HEADER = "stripe-signature"
DEFAULT_TOLERANCE_SECONDS = 300


class StripeSignatureError(ValueError):
    pass


def _signed_payload(timestamp: str, body: bytes) -> bytes:
    return timestamp.encode() + b"." + body


class StripeWebhookVerifier:
    """Verifies one endpoint's ``v1`` signatures against its signing secret."""

    def __init__(
        self,
        webhook_secret: str,
        *,
        tolerance_seconds: int = DEFAULT_TOLERANCE_SECONDS,
        now: Callable[[], float] = time.time,
    ) -> None:
        if not webhook_secret:
            raise StripeSignatureError("stripe webhook secret is not configured")
        self._secret = webhook_secret
        self._tolerance = tolerance_seconds
        self._now = now

    def verify(self, body: bytes, signature_header: str | None) -> None:
        if not signature_header:
            raise StripeSignatureError("missing stripe signature header")
        timestamp: str | None = None
        signatures: list[str] = []
        for part in signature_header.split(","):
            key, separator, value = part.strip().partition("=")
            if not separator:
                continue
            if key == "t" and timestamp is None:
                timestamp = value
            elif key == "v1":
                signatures.append(value)
        if timestamp is None or not signatures:
            raise StripeSignatureError("malformed stripe signature header")
        try:
            signed_at = float(timestamp)
        except ValueError as error:
            raise StripeSignatureError("stripe signature timestamp is not a number") from error
        if abs(self._now() - signed_at) > self._tolerance:
            raise StripeSignatureError("stripe signature timestamp is outside tolerance")
        expected = hmac.new(
            self._secret.encode(), _signed_payload(timestamp, body), hashlib.sha256
        ).hexdigest()
        if not any(hmac.compare_digest(expected, signature) for signature in signatures):
            raise StripeSignatureError("stripe signature does not match")
