"""Collect and attach security-testing evidence to the shared EvidenceBundle."""

from __future__ import annotations

from collections.abc import Sequence

from app.domain.evidence import Evidence, EvidenceBundle, EvidenceKind
from app.domain.http import HttpExchange
from app.security_testing.secrets import redact_exchange, redact_text


class TestingEvidenceCollector:
    def from_exchange(self, exchange: HttpExchange, *, summary: str | None = None) -> Evidence:
        safe = redact_exchange(exchange)
        return Evidence(
            kind=EvidenceKind.HTTP_RESPONSE if safe.response_status else EvidenceKind.HTTP_REQUEST,
            source=safe.source_tool or "http",
            summary=summary or f"{safe.method} {safe.url} -> {safe.response_status}",
            details=(safe.response_body or "")[:2000],
            metadata={"request_id": safe.request_id, "scope": safe.scope_decision or ""},
        )

    def from_browser(
        self,
        *,
        url: str,
        title: str | None = None,
        console: Sequence[str] = (),
        screenshot_path: str | None = None,
        failed_requests: Sequence[str] = (),
    ) -> EvidenceBundle:
        items = [
            Evidence(
                kind=EvidenceKind.BROWSER,
                source="browser",
                summary=redact_text(f"{title or 'page'} at {url}"),
                details="\n".join(redact_text(item) for item in console[:50]),
            )
        ]
        if screenshot_path:
            items.append(
                Evidence(
                    kind=EvidenceKind.SCREENSHOT,
                    source="browser",
                    summary="Browser screenshot",
                    artifact_path=screenshot_path,
                )
            )
        for failed in failed_requests:
            items.append(
                Evidence(
                    kind=EvidenceKind.HTTP_RESPONSE,
                    source="browser",
                    summary=f"Failed request {redact_text(failed)}",
                )
            )
        return EvidenceBundle.from_items(items)

    def informational_scanner_alert(self, tool: str, message: str) -> Evidence:
        return Evidence(
            kind=EvidenceKind.SCANNER,
            source=tool,
            summary=redact_text(message)[:300] or "scanner alert",
            details="Scanner alerts are untrusted observations, not verified findings.",
        )
