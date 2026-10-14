"""HAR file ingestion. Passive evidence only — no live replay."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from app.adapters.proxies.base import ProxyAdapter
from app.domain.http import HttpExchange
from app.security_testing.api_import import import_har_exchanges
from app.security_testing.secrets import redact_exchange


class HarProxyAdapter(ProxyAdapter):
    def __init__(
        self, path: str | Path | None = None, *, exchanges: Sequence[HttpExchange] = ()
    ) -> None:
        self._path = Path(path) if path else None
        self._exchanges = tuple(exchanges)

    @property
    def adapter_id(self) -> str:
        return "har"

    def is_available(self) -> bool:
        return self._path is not None or bool(self._exchanges)

    def load(self, path: str | Path) -> None:
        self._path = Path(path)
        self._exchanges = tuple(redact_exchange(item) for item in import_har_exchanges(self._path))

    def _fetch_captured_exchanges(self) -> Sequence[HttpExchange]:
        if self._path is not None and not self._exchanges:
            self._exchanges = tuple(
                redact_exchange(item) for item in import_har_exchanges(self._path)
            )
        return self._exchanges
