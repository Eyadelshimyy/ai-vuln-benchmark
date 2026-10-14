"""Authenticated local operator sessions.

A free-form ``operator="researcher"`` string is not authorization. Mutating
HackerOne and live-testing approval endpoints require a local operator token
presented in ``X-BugForge-Operator-Token``. The token is compared in constant
time against ``BUGFORGE_OPERATOR_TOKEN`` and is never persisted.
"""

from __future__ import annotations

import hmac
from dataclasses import dataclass
from datetime import UTC, datetime

from fastapi import Header, HTTPException, status

from app.security_testing.approvals import is_ai_operator
from app.security_testing.errors import RestrictedActivityError
from app.security_testing.secrets import environment_credential

OPERATOR_TOKEN_ENV = "BUGFORGE_OPERATOR_TOKEN"
OPERATOR_IDENTITY_ENV = "BUGFORGE_OPERATOR_IDENTITY"
OPERATOR_HEADER = "X-BugForge-Operator-Token"
DEFAULT_IDENTITY = "local-operator"


@dataclass(frozen=True)
class OperatorSession:
    """Proof that a local human operator authenticated.

    The AI cannot mint this object through the public API. Tests may construct
    sessions directly to exercise domain rules (for example, rejecting AI
    identities) without going through HTTP.
    """

    identity: str
    authenticated_at: datetime
    source: str = "operator_token"

    def assert_human(self) -> None:
        if not self.identity.strip():
            raise RestrictedActivityError("unauthenticated_operator")
        if is_ai_operator(self.identity):
            raise RestrictedActivityError("hackerone_submission")
        if self.source not in {"operator_token", "local_session"}:
            raise RestrictedActivityError("invalid_operator_session")


class OperatorAuthenticator:
    def __init__(self, token: str, identity: str) -> None:
        self._token = token
        self._identity = identity.strip() or DEFAULT_IDENTITY

    @classmethod
    def from_env(cls) -> OperatorAuthenticator:
        return cls(
            token=environment_credential(OPERATOR_TOKEN_ENV),
            identity=environment_credential(OPERATOR_IDENTITY_ENV) or DEFAULT_IDENTITY,
        )

    @property
    def configured(self) -> bool:
        return bool(self._token)

    def authenticate(self, presented: str) -> OperatorSession:
        if not self._token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=(
                    "Local operator authorization is not configured. Set "
                    f"{OPERATOR_TOKEN_ENV} and {OPERATOR_IDENTITY_ENV}."
                ),
            )
        if not presented or not hmac.compare_digest(presented, self._token):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing local operator token",
            )
        session = OperatorSession(
            identity=self._identity,
            authenticated_at=datetime.now(UTC),
            source="operator_token",
        )
        try:
            session.assert_human()
        except RestrictedActivityError as exc:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
        return session


def require_operator(
    x_bugforge_operator_token: str | None = Header(default=None, alias=OPERATOR_HEADER),
) -> OperatorSession:
    """FastAPI dependency: authenticated local operator, or 401."""
    return OperatorAuthenticator.from_env().authenticate(x_bugforge_operator_token or "")


def optional_operator(
    x_bugforge_operator_token: str | None = Header(default=None, alias=OPERATOR_HEADER),
) -> OperatorSession | None:
    presented = x_bugforge_operator_token or ""
    if not presented:
        return None
    return OperatorAuthenticator.from_env().authenticate(presented)
