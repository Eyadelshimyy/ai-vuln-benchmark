"""HackerOne API credentials. Never logged, never sent to the frontend."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

from app.adapters.hackerone.errors import HackerOneUrlRejectedError
from app.security_testing.secrets import environment_credential

USERNAME_ENV = "HACKERONE_API_USERNAME"
TOKEN_ENV = "HACKERONE_API_TOKEN"
BASE_URL_ENV = "HACKERONE_API_BASE_URL"
DEFAULT_BASE_URL = "https://api.hackerone.com/v1"


@dataclass
class HackerOneCredentials:
    username: str
    _token: str
    base_url: str = DEFAULT_BASE_URL

    def __repr__(self) -> str:
        return f"HackerOneCredentials(username={self.username!r}, token='***', base_url={self.base_url!r})"

    def __post_init__(self) -> None:
        parsed = urlparse(self.base_url)
        if (parsed.scheme or "").lower() != "https":
            raise HackerOneUrlRejectedError("HackerOne API origin must use HTTPS")
        if not parsed.hostname:
            raise HackerOneUrlRejectedError("HackerOne API origin is missing a hostname")

    @property
    def token(self) -> str:
        return self._token

    @property
    def configured(self) -> bool:
        return bool(self.username and self._token)

    @classmethod
    def from_env(cls) -> HackerOneCredentials:
        return cls(
            username=environment_credential(USERNAME_ENV),
            _token=environment_credential(TOKEN_ENV),
            base_url=environment_credential(BASE_URL_ENV) or DEFAULT_BASE_URL,
        )

    def auth_tuple(self) -> tuple[str, str]:
        return (self.username, self._token)
