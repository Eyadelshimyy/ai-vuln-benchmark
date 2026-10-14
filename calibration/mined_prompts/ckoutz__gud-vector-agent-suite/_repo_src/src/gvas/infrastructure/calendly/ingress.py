"""Calendly webhook ingress: verify, translate, hand to the intake service.

Every outcome other than a failed signature or a malformed body is
acknowledged with a 200 so Calendly stops redelivering — events for event
types we do not track, Calendly users not bound to a business, and invitees
with no matching conversation are all dropped silently.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from gvas.application.intake import IntakeBookingEventService
from gvas.domain.intake import IntakeBookingEvent
from gvas.infrastructure.calendly.config import CalendlyInstallation
from gvas.infrastructure.calendly.events import parse_calendly_event
from gvas.infrastructure.calendly.signature import verify_calendly_signature


class CalendlyIngressResult(StrEnum):
    CONFIRMED = "confirmed"
    REROUTED = "rerouted"
    CANCELED = "canceled"
    IGNORED = "ignored"
    UNKNOWN_INSTALLATION = "unknown_installation"


@dataclass(frozen=True)
class CalendlyIngressOutcome:
    result: CalendlyIngressResult
    detail: str | None = None


def _utc_now() -> datetime:
    return datetime.now(tz=UTC)


class CalendlyWebhookIngress:
    """Verifies Calendly webhook signatures and feeds booking events in."""

    def __init__(
        self,
        service: IntakeBookingEventService,
        installations: tuple[CalendlyInstallation, ...],
        *,
        signing_key: str,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._service = service
        self._signing_key = signing_key
        self._business_by_user = {
            installation.user_uri: installation.business_id for installation in installations
        }
        self._clock = clock

    async def handle(self, *, body: bytes, signature: str | None) -> CalendlyIngressOutcome:
        verify_calendly_signature(body, signature, self._signing_key, now=self._clock)
        event = parse_calendly_event(body)
        if event is None:
            return CalendlyIngressOutcome(
                CalendlyIngressResult.IGNORED, detail="unhandled event type"
            )
        business_id = self._business_by_user.get(event.user_uri)
        if business_id is None:
            return CalendlyIngressOutcome(
                CalendlyIngressResult.UNKNOWN_INSTALLATION,
                detail="calendly user is not bound to a business",
            )
        outcome = await self._service.handle(
            IntakeBookingEvent(
                kind=event.kind,
                business_id=business_id,
                invitee_email=event.invitee_email,
                event_uri=event.event_uri,
                start=event.start,
                end=event.end,
                event_type_uri=event.event_type_uri,
                reference=event.reference,
            )
        )
        return CalendlyIngressOutcome(CalendlyIngressResult(outcome.result.value))
