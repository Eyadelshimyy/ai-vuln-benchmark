"""Isolated research identities.

Isolation is a security-context property (identity id, credential reference,
browser context, HTTP session, storage namespace). Two identities remain
isolated even when cookie/header *values* happen to be identical.

Raw secrets live in :class:`SecretStore`. Database snapshots contain only
sanitized metadata and secret reference IDs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from app.security_agent.authorization_diff import AuthorizationHypothesis, compare_authorization
from app.security_agent.oracles import AuthorizationOracle
from app.security_agent.secrets import SecretStore, is_secret_header, secrets_for
from app.security_testing.errors import RestrictedActivityError

_AUTHENTICATED_STATES = frozenset({"authenticated", "expired", "unavailable"})


@dataclass
class ResearchIdentity:
    label: str
    id: str = field(default_factory=lambda: uuid4().hex)
    credential_ref: str = ""
    browser_context_id: str = ""
    http_session_id: str = ""
    storage_namespace: str = ""
    authentication_state: str = "unauthenticated"
    credential_provenance: str = "none"
    header_names: tuple[str, ...] = ()
    header_secret_refs: dict[str, str] = field(default_factory=dict)
    cookie_names: tuple[str, ...] = ()
    cookie_secret_ref: str = ""
    storage_keys: tuple[str, ...] = ()
    storage_secret_ref: str = ""
    # In-memory execution context only. Never written to the database.
    cookies: dict[str, str] = field(default_factory=dict, repr=False)
    storage: dict[str, str] = field(default_factory=dict, repr=False)
    headers: dict[str, str] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not self.browser_context_id:
            self.browser_context_id = f"browser:{self.id}"
        if not self.http_session_id:
            self.http_session_id = f"http:{self.id}"
        if not self.storage_namespace:
            self.storage_namespace = f"storage:{self.id}"
        if self.cookies and not self.cookie_names:
            self.cookie_names = tuple(sorted(self.cookies))
        if self.headers and not self.header_names:
            self.header_names = tuple(sorted(self.headers))
        if self.storage and not self.storage_keys:
            self.storage_keys = tuple(sorted(self.storage))
        if (self.cookies or self.headers) and self.authentication_state == "unauthenticated":
            self.authentication_state = "authenticated"
            if self.credential_provenance == "none":
                self.credential_provenance = "operator_provided"

    def isolation_tuple(self) -> tuple[str, str, str, str, str]:
        return (
            self.id,
            self.credential_ref or f"cred:{self.id}",
            self.browser_context_id,
            self.http_session_id,
            self.storage_namespace,
        )

    def ingest_secrets(self, store: SecretStore) -> None:
        """Move raw execution secrets into the store. Keep in-memory copies."""
        if self.cookies:
            self.cookie_secret_ref = store.put(dict(self.cookies))
            self.cookie_names = tuple(sorted(self.cookies))
            if not self.credential_ref:
                self.credential_ref = self.cookie_secret_ref
        if self.storage:
            self.storage_secret_ref = store.put(dict(self.storage))
            self.storage_keys = tuple(sorted(self.storage))
        refs = dict(self.header_secret_refs)
        names = set(self.header_names)
        for name, value in self.headers.items():
            names.add(name)
            if is_secret_header(name) or value:
                refs[name] = store.put(value)
        self.header_secret_refs = refs
        self.header_names = tuple(sorted(names))
        if self.headers or self.cookies:
            self.authentication_state = "authenticated"
            if self.credential_provenance == "none":
                self.credential_provenance = "operator_provided"

    def restore_headers(self, store: SecretStore | None) -> dict[str, str]:
        """Restore credentials for execution. Fail closed if a required secret is missing."""
        if self.headers:
            return dict(self.headers)
        if store is None:
            if any(is_secret_header(name) for name in self.header_names) or self.header_secret_refs:
                raise RestrictedActivityError("identity_credentials_unavailable")
            return {}
        restored: dict[str, str] = {}
        for name in self.header_names:
            ref = self.header_secret_refs.get(name)
            if not ref:
                if is_secret_header(name):
                    raise RestrictedActivityError("identity_credentials_unavailable")
                continue
            value = store.get(ref)
            if value is None:
                raise RestrictedActivityError("identity_credentials_unavailable")
            restored[name] = str(value)
        return restored

    def restore_cookies(self, store: SecretStore | None) -> dict[str, str]:
        if self.cookies:
            return dict(self.cookies)
        if not self.cookie_secret_ref and not self.cookie_names:
            return {}
        if store is None or not store.available(self.cookie_secret_ref):
            if self.cookie_names or self.cookie_secret_ref:
                raise RestrictedActivityError("identity_credentials_unavailable")
            return {}
        raw = store.get(self.cookie_secret_ref) or {}
        if isinstance(raw, dict):
            return {str(key): str(value) for key, value in raw.items()}
        return {}

    def credential_availability(self, store: SecretStore | None) -> str:
        if self.headers or self.cookies:
            return "in_memory"
        if store is None:
            return "unavailable"
        needed = [ref for ref in (self.cookie_secret_ref, self.storage_secret_ref) if ref]
        needed.extend(self.header_secret_refs.values())
        if not needed:
            return "none"
        if all(store.available(ref) for ref in needed):
            return "secret_store"
        return "unavailable"

    def snapshot(self) -> dict[str, Any]:
        """Sanitized metadata only. Never includes raw secret values."""
        return {
            "id": self.id,
            "label": self.label,
            "credential_ref": self.credential_ref,
            "browser_context_id": self.browser_context_id,
            "http_session_id": self.http_session_id,
            "storage_namespace": self.storage_namespace,
            "authentication_state": self.authentication_state,
            "credential_provenance": self.credential_provenance,
            "header_names": list(self.header_names),
            "header_secret_refs": dict(self.header_secret_refs),
            "cookie_names": list(self.cookie_names),
            "cookie_secret_ref": self.cookie_secret_ref,
            "storage_keys": list(self.storage_keys),
            "storage_secret_ref": self.storage_secret_ref,
            "credential_availability": self.credential_availability(None)
            if not (self.headers or self.cookies)
            else "in_memory",
        }


@dataclass
class IdentityPair:
    context_a: ResearchIdentity = field(default_factory=lambda: ResearchIdentity(label="A"))
    context_b: ResearchIdentity = field(default_factory=lambda: ResearchIdentity(label="B"))
    comparison_history: list[dict[str, Any]] = field(default_factory=list)

    def isolated(self) -> bool:
        """True when A and B are distinct security contexts.

        Cookie/header *values* are irrelevant: identical values still isolate
        if the context identifiers differ.
        """
        return self.context_a.isolation_tuple() != self.context_b.isolation_tuple()

    def snapshot(self) -> dict[str, Any]:
        return {
            "a": self.context_a.snapshot(),
            "b": self.context_b.snapshot(),
            "isolated": self.isolated(),
            "comparison_history": list(self.comparison_history),
        }


async def compare_identities(
    engine: Any,
    pair: IdentityPair,
    *,
    method: str,
    url: str,
    expectation: str,
    oracle: str = "",
    session_id: str = "",
) -> AuthorizationHypothesis:
    """Send the same request as A and B. Never share credentials. Never mutate."""

    if method.upper() in {"DELETE", "PUT", "PATCH"} and not engine.session.scope.lab_mode:
        raise RestrictedActivityError("destructive_identity_compare")
    store = secrets_for(session_id) if session_id else None
    headers_a = {
        **pair.context_a.restore_headers(store),
        **_cookie_header(pair.context_a.restore_cookies(store)),
    }
    headers_b = {
        **pair.context_b.restore_headers(store),
        **_cookie_header(pair.context_b.restore_cookies(store)),
    }
    exchange_a = await engine.http("identity_a").request(
        method,
        url,
        headers=headers_a,
        active=True,
        destructive=False,
    )
    exchange_b = await engine.http("identity_b").request(
        method,
        url,
        headers=headers_b,
        active=True,
        destructive=False,
    )
    parsed_oracle = AuthorizationOracle.CUSTOM_EXPECTATION
    if oracle:
        try:
            parsed_oracle = AuthorizationOracle(oracle)
        except ValueError:
            parsed_oracle = AuthorizationOracle.CUSTOM_EXPECTATION
    result = compare_authorization(
        _as_exchange(exchange_a),
        _as_exchange(exchange_b),
        expectation=expectation,
        oracle=parsed_oracle,
    )
    pair.comparison_history.append(result.snapshot())
    return result


def _cookie_header(cookies: dict[str, str]) -> dict[str, str]:
    if not cookies:
        return {}
    return {"Cookie": "; ".join(f"{key}={value}" for key, value in cookies.items())}


def _as_exchange(result: Any) -> dict[str, Any]:
    if hasattr(result, "response_status"):
        return {
            "url": result.url,
            "status": result.response_status,
            "response": {"status": result.response_status, "body": result.response_body},
        }
    return {"status": getattr(result, "state", None), "response": {"body": str(result)}}
