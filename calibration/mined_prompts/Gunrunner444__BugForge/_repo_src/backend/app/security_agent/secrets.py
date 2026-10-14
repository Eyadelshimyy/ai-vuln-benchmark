"""In-memory secret mechanism for research identities.

Raw cookies, Authorization headers, API tokens, session tokens, passwords,
and refresh tokens must never be written to research-session database rows.
This store holds execution-time credentials for the current process only.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from app.security_testing.errors import RestrictedActivityError

_SECRET_HEADER_NAMES = frozenset(
    {
        "authorization",
        "proxy-authorization",
        "cookie",
        "set-cookie",
        "x-api-key",
        "x-auth-token",
        "x-access-token",
        "x-csrf-token",
        "x-session-token",
        "api-key",
        "apikey",
    }
)

_SECRET_KEY_FRAGMENTS = (
    "password",
    "passwd",
    "authorization",
    "cookie",
    "set-cookie",
    "token",
    "access_token",
    "refresh_token",
    "api_key",
    "apikey",
    "secret",
    "session",
    "credential",
)


class SecretStore:
    """Process-local secret vault. Values are never serialized."""

    def __init__(self) -> None:
        self._values: dict[str, Any] = {}

    def put(self, value: Any) -> str:
        ident = uuid4().hex
        self._values[ident] = value
        return ident

    def get(self, ident: str) -> Any | None:
        if not ident:
            return None
        return self._values.get(ident)

    def available(self, ident: str) -> bool:
        return bool(ident) and ident in self._values

    def drop(self, ident: str) -> None:
        self._values.pop(ident, None)

    def require(self, ident: str) -> Any:
        value = self.get(ident)
        if value is None:
            raise RestrictedActivityError("identity_credentials_unavailable")
        return value


_SESSION_STORES: dict[str, SecretStore] = {}


def secrets_for(session_id: str) -> SecretStore:
    store = _SESSION_STORES.get(session_id)
    if store is None:
        store = SecretStore()
        _SESSION_STORES[session_id] = store
    return store


def drop_session_secrets(session_id: str) -> None:
    _SESSION_STORES.pop(session_id, None)


def is_secret_header(name: str) -> bool:
    lowered = name.lower()
    if lowered in _SECRET_HEADER_NAMES:
        return True
    return any(fragment in lowered for fragment in ("token", "secret", "password", "auth"))


def is_secret_key(name: str) -> bool:
    lowered = name.lower().replace("-", "_")
    return any(fragment in lowered for fragment in _SECRET_KEY_FRAGMENTS)
