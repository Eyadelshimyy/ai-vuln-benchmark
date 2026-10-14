"""Telnyx webhook signing (Ed25519).

Telnyx signs ``"{telnyx-timestamp}|{raw body}"`` with the account's Ed25519 key
and sends the base64 signature in ``telnyx-signature-ed25519``; the public key
shown in the Mission Control portal is the base64-encoded 32-byte verify key.
The timestamp is unix seconds and is bounded to ``max_age`` on either side of
now, so a captured request cannot be replayed later.
"""

import base64
import binascii
from datetime import datetime, timedelta

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

SIGNATURE_HEADER = "telnyx-signature-ed25519"
TIMESTAMP_HEADER = "telnyx-timestamp"
PUBLIC_KEY_LENGTH = 32
SIGNATURE_LENGTH = 64


class TelnyxSignatureError(ValueError):
    pass


def load_public_key(public_key: str) -> Ed25519PublicKey:
    try:
        raw = base64.b64decode(public_key, validate=True)
    except (binascii.Error, ValueError) as error:
        raise TelnyxSignatureError("telnyx public key is not valid base64") from error
    if len(raw) != PUBLIC_KEY_LENGTH:
        raise TelnyxSignatureError("telnyx public key must be a 32-byte Ed25519 key")
    return Ed25519PublicKey.from_public_bytes(raw)


def signed_message(timestamp: str, body: bytes) -> bytes:
    return timestamp.encode() + b"|" + body


def verify_signature(
    *,
    public_key: str,
    body: bytes,
    signature: str | None,
    timestamp: str | None,
    now: datetime,
    max_age: timedelta,
) -> None:
    if not signature or not timestamp:
        raise TelnyxSignatureError("missing telnyx signature headers")
    try:
        request_time = int(timestamp)
    except ValueError as error:
        raise TelnyxSignatureError("invalid telnyx timestamp") from error
    if abs(now.timestamp() - request_time) > max_age.total_seconds():
        raise TelnyxSignatureError("stale telnyx request")
    try:
        raw_signature = base64.b64decode(signature, validate=True)
    except (binascii.Error, ValueError) as error:
        raise TelnyxSignatureError("telnyx signature is not valid base64") from error
    if len(raw_signature) != SIGNATURE_LENGTH:
        raise TelnyxSignatureError("telnyx signature has the wrong length")
    try:
        load_public_key(public_key).verify(raw_signature, signed_message(timestamp, body))
    except InvalidSignature as error:
        raise TelnyxSignatureError("invalid telnyx signature") from error
