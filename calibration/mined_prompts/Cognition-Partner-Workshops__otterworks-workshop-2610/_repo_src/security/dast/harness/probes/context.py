"""Shared scan context: target, seeded identities, and HTTP helpers.

Every probe attacks the *running* application through the API gateway, using two
real accounts registered at scan time: an ``attacker`` and a ``victim``. Both are
namespaced by ``run_id`` so concurrent scans (CI, several sessions, several
tenants) never collide.
"""

from __future__ import annotations

import os
import secrets
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx


def scan_password() -> str:
    """A single-use password for an account the scan registers.

    The accounts are real and outlive the run: they persist in the target's
    database until the tenant is reaped. A constant here would be a committed,
    publicly known credential for every account any scan has ever created, so
    each one gets its own value that exists only in the running process.
    """
    return f"Dast-{secrets.token_urlsafe(24)}-1!"


@dataclass
class Identity:
    email: str
    password: str
    user_id: str = ""
    access_token: str = ""

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.access_token}"} if self.access_token else {}


class SeedError(RuntimeError):
    """Raised when the scan cannot establish the identities it needs."""


@dataclass
class ScanContext:
    base_url: str
    client: httpx.Client
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    #: Sized against RATE_LIMIT_RPS (default 100/s) plus the refill accrued over the
    #: burst's own duration, and large enough that the limiter throttles a clear
    #: majority of it: 200 concurrent requests draw no 429 from a deployed tenant,
    #: 600 leave ~94% served (too generous to tell a bypass from the allowance),
    #: 1500 leaves ~40%. Deployed tenants share one ingress controller and node group,
    #: so OTTERWORKS_DAST_RATE_LIMIT_BURST/_WORKERS turn the load down when other
    #: tenants are live — at the cost of the probe reporting `inconclusive` if the
    #: burst no longer separates a bypass from the limiter's allowance.
    rate_limit_burst: int = field(
        default_factory=lambda: int(os.getenv("OTTERWORKS_DAST_RATE_LIMIT_BURST", "1500"))
    )
    #: The burst is issued concurrently: a token bucket is never drained by a
    #: sequential client once a round trip costs more than the refill interval.
    rate_limit_workers: int = field(
        default_factory=lambda: int(os.getenv("OTTERWORKS_DAST_RATE_LIMIT_WORKERS", "64"))
    )
    brute_force_attempts: int = 12
    attacker: Identity = field(init=False)
    victim: Identity = field(init=False)
    #: A throwaway account for probes that abuse credentials. Keeping them off the
    #: victim means a target that correctly locks accounts does not strand the
    #: later probes that need the victim to be able to log in.
    burner: Identity = field(init=False)
    _victim_document: dict[str, Any] | None = field(default=None, init=False)
    _victim_document_attempted: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.attacker = Identity(
            email=f"dast-attacker-{self.run_id}@example.test", password=scan_password()
        )
        self.victim = Identity(
            email=f"dast-victim-{self.run_id}@example.test", password=scan_password()
        )
        self.burner = Identity(
            email=f"dast-burner-{self.run_id}@example.test", password=scan_password()
        )

    # ── lifecycle ────────────────────────────────────────────────────────────

    @property
    def far_future(self) -> int:
        return int(time.time()) + 3600

    @property
    def victim_marker(self) -> str:
        """A string that exists only inside the victim's own document.

        Any probe that plants content elsewhere must use `plant_marker`, or the
        search probe will read its own plant back as a cross-tenant leak.
        """
        return f"otterworks-dast-marker-{self.run_id}"

    @property
    def plant_marker(self) -> str:
        """A string for content probes write themselves, distinct from the victim's."""
        return f"otterworks-dast-plant-{self.run_id}"

    def wait_for_target(self, timeout: float = 60.0) -> None:
        deadline = time.monotonic() + timeout
        last: Exception | None = None
        while time.monotonic() < deadline:
            try:
                if self.client.get("/health").status_code < 500:
                    return
            except httpx.HTTPError as exc:
                last = exc
            time.sleep(1.0)
        raise SeedError(f"target {self.base_url} did not become reachable: {last}")

    def seed_identities(self, timeout: float = 120.0) -> None:
        """Register the scan's accounts, waiting for the auth backend to come up.

        The gateway's ``/health`` is a static handler, so it answers well before
        auth-service can serve a registration: a scan started right after
        ``docker compose up`` would otherwise fail to seed and skip most of the
        suite. Registration is the real readiness check, so it is the one retried.
        """
        deadline = time.monotonic() + timeout
        for identity in (self.attacker, self.victim, self.burner):
            while True:
                try:
                    self._register(identity)
                    break
                except SeedError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(2.0)

    @property
    def identities_ready(self) -> bool:
        """Whether every identity the authenticated probes assume actually exists.

        Seeding stops at the first failure, so a half-seeded run can leave the
        attacker usable and the victim blank. A cross-account attack aimed at an
        empty user id degrades into an ordinary self-owned request and would
        report `secure` without ever having been attempted.
        """
        return all(
            identity.access_token and identity.user_id
            for identity in (self.attacker, self.victim, self.burner)
        )

    def _register(self, identity: Identity) -> None:
        # Anything that goes wrong here is a setup problem, not a finding: a transport
        # error or a proxy's HTML error page must not escape as an unhandled exception,
        # whose exit status 1 the harness reserves for "the gate failed".
        try:
            response = self.client.post(
                "/api/v1/auth/register",
                json={
                    "email": identity.email,
                    "password": identity.password,
                    "displayName": f"DAST {identity.email.split('@')[0]}",
                },
            )
        except httpx.HTTPError as exc:
            raise SeedError(f"could not reach registration for {identity.email}: {exc}") from exc
        if response.status_code not in (200, 201):
            raise SeedError(
                f"could not register {identity.email}: {response.status_code} {response.text[:200]}"
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise SeedError(
                f"registration for {identity.email} returned a non-JSON body: {response.text[:200]}"
            ) from exc
        if not isinstance(body, dict):
            raise SeedError(f"registration for {identity.email} returned {type(body).__name__}")
        identity.access_token = body.get("accessToken", "")
        # A JSON null (or a string, or a list) under "user" would make .get() an
        # AttributeError, which is not a SeedError and would leave the harness exiting 1
        # — the status reserved for a failed gate.
        user = body.get("user")
        identity.user_id = str(user.get("id", "")) if isinstance(user, dict) else ""
        if not identity.access_token or not identity.user_id:
            raise SeedError(f"registration for {identity.email} returned no usable identity")

    def login_response(self, email: str, password: str) -> httpx.Response:
        """The raw login response, for probes that must tell a refusal from a 429/5xx."""
        return self.client.post("/api/v1/auth/login", json={"email": email, "password": password})

    def login(self, email: str, password: str) -> bool:
        return self.login_response(email, password).status_code == 200

    # ── seeded fixtures ──────────────────────────────────────────────────────

    def create_document_response(
        self,
        identity: Identity,
        title: str,
        content: str,
        *,
        owner_id: str | None = None,
        allow_owner_fallback: bool = True,
    ) -> httpx.Response:
        """Create a document, falling back to naming the owner explicitly.

        Some deployments reject a create whose owner cannot be derived from the
        token and ask the caller to supply owner_id instead; the fallback keeps
        the suite usable there. Pass ``allow_owner_fallback=False`` where the
        point of the request is that it carries no owner_id — a service hardened
        by dropping the field would reject the retry.

        Returns the raw response so callers can tell an explicit refusal apart
        from a backend that is simply broken.
        """
        body: dict[str, Any] = {"title": title, "content": content}
        if owner_id:
            body["owner_id"] = owner_id
        response = self.request("POST", "/api/v1/documents/", identity=identity, json=body)
        if response.status_code in (401, 403) and not owner_id and allow_owner_fallback:
            body["owner_id"] = identity.user_id
            response = self.request("POST", "/api/v1/documents/", identity=identity, json=body)
        return response

    def create_document(
        self,
        identity: Identity,
        title: str,
        content: str,
        *,
        owner_id: str | None = None,
    ) -> dict[str, Any] | None:
        """The created document, or None if the create did not succeed."""
        response = self.create_document_response(identity, title, content, owner_id=owner_id)
        if response.status_code not in (200, 201):
            return None
        try:
            return response.json()
        except ValueError:
            return None

    def victim_document(self) -> dict[str, Any] | None:
        """A document owned solely by the victim, created once per scan."""
        if not self._victim_document_attempted:
            self._victim_document_attempted = True
            self._victim_document = self.create_document(
                self.victim,
                title=f"victim-private-{self.run_id}",
                content=f"confidential {self.victim_marker}",
            )
        return self._victim_document

    # ── HTTP helpers ─────────────────────────────────────────────────────────

    def request(
        self,
        method: str,
        path: str,
        *,
        identity: Identity | None = None,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        json: Any = None,
    ) -> httpx.Response:
        merged = dict(identity.headers) if identity else {}
        merged.update(headers or {})
        return self.client.request(method, path, headers=merged, params=params, json=json)

    def get(self, path: str, **kwargs: Any) -> httpx.Response:
        return self.request("GET", path, **kwargs)

    def search_as(self, identity: Identity, query: str) -> list[Any] | None:
        """Control request: the result hits this identity sees, or None if unusable.

        Documents index asynchronously, so an empty result set for the attacker
        is only meaningful once the owner can find the document.
        """
        response = self.get("/api/v1/search/", params={"q": query}, identity=identity)
        if response.status_code != 200:
            return None
        try:
            payload = response.json()
        except ValueError:
            return None
        hits = payload.get("results") if isinstance(payload, dict) else None
        return hits if isinstance(hits, list) else None

    def owner_can_read(self, path: str, identity: Identity) -> bool:
        """Control request: can the legitimate owner read this object at all?

        Without this, a route that rejects *everyone* looks identical to a route
        that correctly rejects only the attacker.
        """
        return self.get(path, identity=identity).status_code == 200
