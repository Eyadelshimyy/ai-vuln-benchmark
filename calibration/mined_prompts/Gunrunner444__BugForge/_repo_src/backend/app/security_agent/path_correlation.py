"""Correlate browser, HTTP, API spec, and source observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ExecutionPath:
    ui_action: str = ""
    request_id: str = ""
    endpoint: str = ""
    source_symbol: str = ""
    source_path: str = ""
    evidence_ids: tuple[str, ...] = ()

    def snapshot(self) -> dict[str, Any]:
        return {
            "ui_action": self.ui_action,
            "request_id": self.request_id,
            "endpoint": self.endpoint,
            "source_symbol": self.source_symbol,
            "source_path": self.source_path,
            "evidence_ids": list(self.evidence_ids),
        }


def correlate_client_server(
    *,
    ui_action: str,
    request: dict[str, Any] | None,
    spec_endpoint: str = "",
    source_symbol: str = "",
    source_path: str = "",
    evidence_ids: tuple[str, ...] = (),
) -> ExecutionPath:
    url = ""
    request_id = ""
    if request:
        url = str(request.get("url") or "")
        request_id = str(request.get("id") or request.get("exchange_id") or "")
    return ExecutionPath(
        ui_action=ui_action,
        request_id=request_id,
        endpoint=spec_endpoint or url,
        source_symbol=source_symbol,
        source_path=source_path,
        evidence_ids=evidence_ids,
    )


def correlate_source_runtime(
    *,
    source_symbol: str,
    source_path: str,
    endpoint: str,
    exchange: dict[str, Any] | None = None,
    evidence_ids: tuple[str, ...] = (),
) -> ExecutionPath:
    """Connect a static source location to a runtime HTTP exchange."""
    return correlate_client_server(
        ui_action="",
        request=exchange,
        spec_endpoint=endpoint,
        source_symbol=source_symbol,
        source_path=source_path,
        evidence_ids=evidence_ids,
    )
