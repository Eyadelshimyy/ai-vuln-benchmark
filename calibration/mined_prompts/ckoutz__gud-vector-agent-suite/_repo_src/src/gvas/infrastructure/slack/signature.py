import hmac
from datetime import datetime, timedelta
from hashlib import sha256

SIGNATURE_VERSION = "v0"


class SlackSignatureError(ValueError):
    pass


def compute_signature(signing_secret: str, timestamp: str, body: bytes) -> str:
    basestring = SIGNATURE_VERSION.encode() + b":" + timestamp.encode() + b":" + body
    digest = hmac.new(signing_secret.encode(), basestring, sha256).hexdigest()
    return f"{SIGNATURE_VERSION}={digest}"


def verify_signature(
    *,
    signing_secret: str,
    body: bytes,
    signature: str | None,
    timestamp: str | None,
    now: datetime,
    max_age: timedelta,
) -> None:
    if not signing_secret:
        raise SlackSignatureError("signing secret is not configured")
    if not signature or not timestamp:
        raise SlackSignatureError("missing signature headers")
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    try:
        sent_at = int(timestamp)
    except ValueError as error:
        raise SlackSignatureError("request timestamp is not an integer") from error
    age = abs(now.timestamp() - sent_at)
    if age > max_age.total_seconds():
        raise SlackSignatureError("request timestamp is outside the accepted window")
    expected = compute_signature(signing_secret, timestamp, body)
    if not hmac.compare_digest(expected, signature):
        raise SlackSignatureError("request signature does not match")
