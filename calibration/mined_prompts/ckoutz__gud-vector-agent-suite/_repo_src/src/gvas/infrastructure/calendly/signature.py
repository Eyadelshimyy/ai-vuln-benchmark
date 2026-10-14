"""Calendly webhook signature verification.

Calendly signs every webhook request with the subscription's signing key:
``Calendly-Webhook-Signature: t=<unix seconds>,v1=<hex hmac>`` where the HMAC
is SHA-256 over ``"{t}.{raw body}"``. The timestamp also bounds replay: stale
timestamps are rejected.
"""

import hashlib
import hmac
from collections.abc import Callable
from datetime import datetime

SIGNATURE_HEADER = "Calendly-Webhook-Signature"
_SIGNATURE_TOLERANCE_SECONDS = 180.0


class CalendlySignatureError(ValueError):
    """The webhook signature is missing, malformed, stale, or wrong."""


def verify_calendly_signature(
    body: bytes,
    signature_header: str | None,
    signing_key: str,
    *,
    now: Callable[[], datetime],
    tolerance_seconds: float = _SIGNATURE_TOLERANCE_SECONDS,
) -> None:
    if not signature_header:
        raise CalendlySignatureError("missing signature header")
    fields = dict(part.split("=", 1) for part in signature_header.split(",") if "=" in part)
    timestamp_text = fields.get("t", "")
    provided = fields.get("v1", "")
    try:
        timestamp = float(timestamp_text)
    except ValueError as error:
        raise CalendlySignatureError("malformed signature timestamp") from error
    if not provided:
        raise CalendlySignatureError("missing v1 signature")
    if abs(now().timestamp() - timestamp) > tolerance_seconds:
        raise CalendlySignatureError("stale signature timestamp")
    expected = hmac.new(
        signing_key.encode(),
        timestamp_text.encode() + b"." + body,
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(expected, provided):
        raise CalendlySignatureError("invalid signature")
