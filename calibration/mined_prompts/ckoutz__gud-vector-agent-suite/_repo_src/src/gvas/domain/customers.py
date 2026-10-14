"""Customer portal identities and credentials.

A customer is ``(business_id, lowercased e-mail)``: the same address at two
businesses is two customers, and nothing here ever crosses a business. Login
tokens and portal sessions follow the claim-token rule — ``token_urlsafe``
secrets, only the SHA-256 digest at rest, constant-time comparison.
"""

import hashlib
import hmac
import secrets
from collections.abc import Mapping
from datetime import datetime, timedelta
from typing import Protocol
from uuid import UUID, uuid5

from pydantic import BaseModel, ConfigDict, Field, field_validator

from gvas.domain.identifiers import (
    BusinessId,
    CustomerId,
    OutboxCommandId,
    ServiceRequestId,
)
from gvas.domain.outbox import OutboxCommand

PORTAL_TOKEN_BYTES = 32
LOGIN_TOKEN_TTL = timedelta(minutes=15)
PORTAL_SESSION_TTL = timedelta(days=30)
PORTAL_LOGIN_EMAIL_COMMAND_TYPE = "portal_login.email"
PORTAL_LOGIN_EMAIL_COMMAND_NAMESPACE = UUID("0d6a2f1e-7b3c-4e9a-9f21-5c8d4b3a2e10")
SERVICE_REQUEST_SOURCE_PORTAL = "portal"
SERVICE_REQUEST_STATUS_NEW = "new"
SERVICE_REQUEST_MAX_CHARS = 2000


class CustomerModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("customer timestamps must be timezone-aware")
    return value


class CustomerRecord(CustomerModel):
    customer_id: CustomerId
    business_id: BusinessId
    email: str = Field(min_length=3)
    display_name: str | None = None
    phone: str | None = None
    # The billing provider's customer handle, once a subscription was opened.
    stripe_customer_id: str | None = None
    # Whether the customer agreed to receive texts; only ``True`` lets GVAS
    # text them. ``None`` means they were never asked.
    sms_consent: bool | None = None
    sms_consent_at: datetime | None = None
    created_at: datetime

    @field_validator("email")
    @classmethod
    def email_is_normalized(cls, value: str) -> str:
        if value != value.strip().lower():
            raise ValueError("customer e-mail must be lowercased")
        return value

    @field_validator("created_at")
    @classmethod
    def timestamps_are_aware(cls, value: datetime) -> datetime:
        return _aware(value)

    @property
    def name(self) -> str:
        return self.display_name or self.email


class PortalLoginToken(CustomerModel):
    """One single-use magic link; only the digest is stored."""

    token_hash: str = Field(min_length=64, max_length=64)
    business_id: BusinessId
    customer_id: CustomerId
    expires_at: datetime
    used_at: datetime | None = None
    created_at: datetime

    def is_usable(self, now: datetime) -> bool:
        return self.used_at is None and now < self.expires_at


class PortalSession(CustomerModel):
    """A bearer credential for one customer; revocation sets ``revoked_at``."""

    token_hash: str = Field(min_length=64, max_length=64)
    business_id: BusinessId
    customer_id: CustomerId
    expires_at: datetime
    revoked_at: datetime | None = None
    created_at: datetime

    def is_active(self, now: datetime) -> bool:
        return self.revoked_at is None and now < self.expires_at


class ServiceRequest(CustomerModel):
    request_id: ServiceRequestId
    business_id: BusinessId
    customer_id: CustomerId
    message: str = Field(min_length=1, max_length=SERVICE_REQUEST_MAX_CHARS)
    preferred_dates: str | None = None
    status: str = SERVICE_REQUEST_STATUS_NEW
    source: str = SERVICE_REQUEST_SOURCE_PORTAL
    created_at: datetime


class PortalLoginEmailRequest(CustomerModel):
    """What the e-mail adapter needs to send one magic link."""

    business_id: BusinessId
    to: str = Field(min_length=3)
    business_display_name: str = Field(min_length=1)
    login_url: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)
    expires_at: datetime

    @property
    def subject(self) -> str:
        return f"Sign in to your {self.business_display_name} account"

    def is_expired(self, now: datetime) -> bool:
        """A link sent after its token expired can never sign anyone in."""

        return now >= self.expires_at


def new_portal_token() -> str:
    return secrets.token_urlsafe(PORTAL_TOKEN_BYTES)


def hash_portal_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def portal_token_matches(token: str, stored_hash: str) -> bool:
    return hmac.compare_digest(hash_portal_token(token), stored_hash)


def portal_login_url(site_url: str, raw_token: str) -> str:
    return f"{site_url.rstrip('/')}/portal/login?token={raw_token}"


def portal_login_email_command(
    request: PortalLoginEmailRequest, *, token_hash: str
) -> OutboxCommand:
    """One e-mail per issued token; the command is keyed by the token digest
    so a replayed enqueue cannot send the same link twice. The payload has
    to carry the link (the only copy of the raw token) until the worker
    sends it; the token itself expires in ``LOGIN_TOKEN_TTL``."""

    command_id = OutboxCommandId(uuid5(PORTAL_LOGIN_EMAIL_COMMAND_NAMESPACE, token_hash))
    return OutboxCommand(
        command_id=command_id,
        business_id=request.business_id,
        command_type=PORTAL_LOGIN_EMAIL_COMMAND_TYPE,
        payload={
            "to": request.to,
            "business_display_name": request.business_display_name,
            "login_url": request.login_url,
            "idempotency_key": request.idempotency_key,
            "expires_at": request.expires_at.isoformat(),
        },
        dedup_key=f"portal_login:{token_hash}",
    )


def portal_login_email_request(
    business_id: BusinessId, command_payload: Mapping[str, object]
) -> PortalLoginEmailRequest:
    """Rebuilds the request the worker sends from a claimed command."""

    fields = {
        key: command_payload.get(key)
        for key in ("to", "business_display_name", "login_url", "idempotency_key", "expires_at")
    }
    if not all(isinstance(value, str) for value in fields.values()):
        raise ValueError("portal login command payload is incomplete")
    try:
        expires_at = datetime.fromisoformat(str(fields["expires_at"]))
    except ValueError as error:
        raise ValueError("portal login command expiry is malformed") from error
    if expires_at.tzinfo is None:
        raise ValueError("portal login command expiry must be timezone-aware")
    return PortalLoginEmailRequest(
        business_id=business_id,
        to=str(fields["to"]),
        business_display_name=str(fields["business_display_name"]),
        login_url=str(fields["login_url"]),
        idempotency_key=str(fields["idempotency_key"]),
        expires_at=expires_at,
    )


class CustomerRepository(Protocol):
    async def get(self, business_id: BusinessId, customer_id: CustomerId) -> CustomerRecord | None:
        """Tenant-scoped: a customer id from another business is ``None``."""
        ...

    async def find_by_email(self, business_id: BusinessId, email: str) -> CustomerRecord | None: ...

    async def list_for_business(self, business_id: BusinessId) -> tuple[CustomerRecord, ...]:
        """Newest first."""
        ...

    async def upsert(
        self,
        business_id: BusinessId,
        email: str,
        *,
        display_name: str | None,
        phone: str | None,
        now: datetime,
    ) -> CustomerRecord:
        """The customer for ``(business_id, email)``, created when missing;
        blank stored details are filled from the arguments, never overwritten."""
        ...

    async def create(
        self,
        business_id: BusinessId,
        email: str,
        *,
        display_name: str | None,
        phone: str | None,
        now: datetime,
    ) -> CustomerRecord | None:
        """A new customer for ``(business_id, email)``, or ``None`` when one
        already exists — that record is left untouched."""
        ...

    async def set_stripe_customer_id(
        self, business_id: BusinessId, customer_id: CustomerId, stripe_customer_id: str
    ) -> None: ...

    async def set_sms_consent(
        self, business_id: BusinessId, customer_id: CustomerId, consent: bool, at: datetime
    ) -> None:
        """Records the answer unless the customer already has a newer one."""
        ...


class PortalLoginTokenRepository(Protocol):
    async def add(self, token: PortalLoginToken) -> None: ...

    async def find_by_hash(self, token_hash: str) -> PortalLoginToken | None: ...

    async def mark_used(self, token_hash: str, now: datetime) -> bool:
        """True when this call consumed the token; False when it was already
        used, so two racing exchanges cannot both succeed."""
        ...


class PortalSessionRepository(Protocol):
    async def add(self, session: PortalSession) -> None: ...

    async def find_by_hash(self, token_hash: str) -> PortalSession | None: ...

    async def revoke(self, token_hash: str, now: datetime) -> bool:
        """Conditionally revoke an active session; ``True`` only when this
        call is the one that revoked it (so concurrent callers disagree)."""
        ...


class ServiceRequestRepository(Protocol):
    async def add(self, request: ServiceRequest) -> None: ...

    async def list_for_business(
        self, business_id: BusinessId, *, limit: int
    ) -> tuple[ServiceRequest, ...]:
        """Newest first."""
        ...
