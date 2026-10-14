"""Service-token auth for the `/v1` API.

A trusted-subsystem deployment: callers present a static bearer token, the
server holds only its SHA-256 in a client registry (``WOMBLEX_API_CLIENTS``, a
YAML file), and a match yields a :class:`Caller` carrying the client id and
scopes. This is private-network authentication, not a public identity system.

Registry shape::

    clients:
      - client_id: redline
        token_sha256: <64 hex>
        scopes: [submit, read]

``fastapi`` is imported lazily, so the registry and ``womblex api-token`` work
on an install without the ``ui`` extra.
"""

import hashlib
import hmac
import logging
import os
import re
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

REGISTRY_ENV = "WOMBLEX_API_CLIENTS"
TOKEN_PREFIX = "womblex_"

#: ``submit`` uploads and starts runs; ``read`` reads masked/non-text output;
#: ``read_raw`` also reads raw-PII files; ``admin`` sees every owner's runs
#: and implies every other scope.
SCOPES = ("submit", "read", "read_raw", "admin")

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class RegistryError(ValueError):
    """The client registry is unreadable or malformed."""


@dataclass(frozen=True)
class Client:
    """One registry entry: a caller's id, the hash of its token, its scopes."""

    client_id: str
    token_sha256: str
    scopes: frozenset[str]


@dataclass(frozen=True)
class Caller:
    """An authenticated caller."""

    client_id: str
    scopes: frozenset[str]

    def has(self, scope: str) -> bool:
        return scope in self.scopes or "admin" in self.scopes

    @property
    def is_admin(self) -> bool:
        return "admin" in self.scopes

    @property
    def owner(self) -> str | None:
        """The queue owner to scope reads to; ``None`` (unscoped) for an admin."""
        return None if self.is_admin else self.client_id


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def generate_token() -> tuple[str, str]:
    """A new random token and its SHA-256 — the registry stores only the hash."""
    token = TOKEN_PREFIX + secrets.token_urlsafe(32)
    return token, hash_token(token)


@dataclass(frozen=True)
class ClientRegistry:
    clients: tuple[Client, ...]

    @property
    def empty(self) -> bool:
        return not self.clients

    def authenticate(self, token: str) -> Caller | None:
        """The caller *token* belongs to, or ``None``.

        Compares against every entry with ``hmac.compare_digest`` and does not
        stop at the first match, so timing does not reveal which entry matched.
        """
        digest = hash_token(token)
        matched: Client | None = None
        for client in self.clients:
            if hmac.compare_digest(digest, client.token_sha256):
                matched = client
        return Caller(matched.client_id, matched.scopes) if matched else None


def parse_registry(data: Any) -> ClientRegistry:
    """Validate a loaded registry document; raise :class:`RegistryError`."""
    entries = data.get("clients") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        raise RegistryError("registry must be a mapping with a 'clients' list")
    clients: list[Client] = []
    seen: set[str] = set()
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise RegistryError(f"clients[{i}] must be a mapping")
        client_id = entry.get("client_id")
        digest = entry.get("token_sha256")
        scopes = entry.get("scopes")
        if not isinstance(client_id, str) or not client_id.strip():
            raise RegistryError(f"clients[{i}] needs a non-empty client_id")
        if client_id in seen:
            raise RegistryError(f"duplicate client_id {client_id!r}")
        if not isinstance(digest, str) or not _SHA256_RE.match(digest):
            raise RegistryError(f"client {client_id!r}: token_sha256 must be 64 lowercase hex")
        if not isinstance(scopes, list) or not scopes:
            raise RegistryError(f"client {client_id!r}: scopes must be a non-empty list")
        unknown = [s for s in scopes if s not in SCOPES]
        if unknown:
            raise RegistryError(
                f"client {client_id!r}: unknown scope(s) {unknown}; known: {list(SCOPES)}"
            )
        seen.add(client_id)
        clients.append(Client(client_id, digest, frozenset(scopes)))
    return ClientRegistry(tuple(clients))


def load_registry(path: str | Path) -> ClientRegistry:
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as e:
        raise RegistryError(f"cannot read client registry {str(path)!r}: {e}") from e
    return parse_registry(data)


def registry_from_env() -> ClientRegistry:
    """The registry named by ``$WOMBLEX_API_CLIENTS``, or an empty one if unset."""
    path = os.environ.get(REGISTRY_ENV)
    return load_registry(path) if path else ClientRegistry(())


def caller_dependency(registry: ClientRegistry | None):  # type: ignore[no-untyped-def]
    """A FastAPI dependency resolving the request's :class:`Caller`.

    A missing, malformed or unknown bearer token is a 401. ``registry=None``
    is the ``--insecure-no-auth`` development mode: every request is an
    all-scope caller named ``insecure``.
    """
    from fastapi import HTTPException, Request

    if registry is None:
        logger.warning("API auth disabled: every request is an admin caller")
        anonymous = Caller("insecure", frozenset(SCOPES))

        def open_caller() -> Caller:
            return anonymous

        return open_caller

    def get_caller(request: Request) -> Caller:
        scheme, _, token = request.headers.get("authorization", "").partition(" ")
        caller = registry.authenticate(token.strip()) if scheme.lower() == "bearer" else None
        if caller is None:
            raise HTTPException(
                status_code=401, detail="missing or invalid bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return caller

    return get_caller


def require_scope(get_caller, scope: str):  # type: ignore[no-untyped-def]
    """A dependency that yields the caller if it holds *scope*, else a 403."""
    from fastapi import Depends, HTTPException

    def checked(caller: Caller = Depends(get_caller)) -> Caller:  # noqa: B008
        if not caller.has(scope):
            raise HTTPException(status_code=403, detail=f"requires the {scope!r} scope")
        return caller

    return checked
