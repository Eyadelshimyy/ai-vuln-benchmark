"""Manual researcher evidence. AI does not pretend to perform this work."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from app.domain.evidence import Evidence, EvidenceBundle, EvidenceKind
from app.domain.http import HttpExchange
from app.security_testing.secrets import redact_exchange, redact_text


@dataclass(frozen=True)
class ManualObservation:
    observation: str
    expected_result: str = ""
    actual_result: str = ""
    comments: str = ""
    reproduction_notes: str = ""
    screenshot_path: str | None = None
    request: HttpExchange | None = None
    response_notes: str = ""
    recorded_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class ManualEvidenceCollector:
    def collect(
        self,
        notes: list[ManualObservation],
        *,
        source: str = "researcher",
        attribution: object | None = None,
    ) -> EvidenceBundle:
        from app.adapters.evidence.attribution import ServerAttribution, attribution_metadata

        server = (
            attribution_metadata(attribution) if isinstance(attribution, ServerAttribution) else {}
        )
        items: list[Evidence] = []
        for note in notes:
            items.append(
                Evidence(
                    kind=EvidenceKind.LOG,
                    source=source,
                    summary=redact_text(note.observation),
                    details=redact_text(
                        "\n".join(
                            part
                            for part in (
                                f"expected: {note.expected_result}" if note.expected_result else "",
                                f"actual: {note.actual_result}" if note.actual_result else "",
                                f"comments: {note.comments}" if note.comments else "",
                                f"repro: {note.reproduction_notes}"
                                if note.reproduction_notes
                                else "",
                            )
                            if part
                        )
                    ),
                    artifact_path=note.screenshot_path,
                    metadata=dict(server),
                )
            )
            if note.screenshot_path:
                items.append(
                    Evidence(
                        kind=EvidenceKind.SCREENSHOT,
                        source=source,
                        summary="Researcher screenshot",
                        artifact_path=note.screenshot_path,
                        metadata=dict(server),
                    )
                )
            if note.request is not None:
                redacted = redact_exchange(note.request)
                items.append(
                    Evidence(
                        kind=EvidenceKind.HTTP_REQUEST,
                        source=source,
                        summary=f"{redacted.method} {redacted.url}",
                        details=redacted.request_body or "",
                        metadata=dict(server),
                    )
                )
                if redacted.response_status is not None:
                    items.append(
                        Evidence(
                            kind=EvidenceKind.HTTP_RESPONSE,
                            source=source,
                            summary=f"status {redacted.response_status}",
                            details=(redacted.response_body or "")[:2000],
                            metadata=dict(server),
                        )
                    )
        return EvidenceBundle.from_items(items)
