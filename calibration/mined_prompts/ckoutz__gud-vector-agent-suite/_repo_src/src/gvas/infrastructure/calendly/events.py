"""Calendly webhook payloads: invitee.created and invitee.canceled.

The payload is parsed defensively — Calendly wraps the invitee and its
scheduled event in a ``payload`` envelope, and only the fields the intake
booking flow needs are read (invitee email, the event URI, its start/end, and
the Calendly user that identifies the business).
"""

import json
from datetime import datetime

from gvas.domain.intake import BookingEventKind


class CalendlyPayloadError(ValueError):
    """The webhook body is not a well-formed Calendly event."""


class CalendlyInviteeEvent:
    """The translated webhook fact, before it is bound to a business."""

    def __init__(
        self,
        *,
        kind: BookingEventKind,
        user_uri: str,
        invitee_email: str,
        event_uri: str,
        start: datetime,
        end: datetime | None,
        event_type_uri: str | None = None,
        reference: str | None = None,
    ) -> None:
        self.kind = kind
        self.user_uri = user_uri
        self.invitee_email = invitee_email
        self.event_uri = event_uri
        self.start = start
        self.end = end
        self.event_type_uri = event_type_uri
        self.reference = reference


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CalendlyPayloadError(f"missing {field}")
    return value.strip()


def _when(value: object, field: str) -> datetime:
    text = _text(value, field)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise CalendlyPayloadError(f"invalid {field}") from error
    if parsed.tzinfo is None:
        raise CalendlyPayloadError(f"naive {field}")
    return parsed


def parse_calendly_event(body: bytes) -> CalendlyInviteeEvent | None:
    """Translate a webhook body into an invitee event.

    ``None`` means the body is a valid Calendly event of a kind the intake
    flow ignores (anything other than invitee.created / invitee.canceled).
    """

    try:
        envelope = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise CalendlyPayloadError("body is not json") from error
    if not isinstance(envelope, dict):
        raise CalendlyPayloadError("body is not an object")
    kind_by_event = {
        "invitee.created": BookingEventKind.CREATED,
        "invitee.canceled": BookingEventKind.CANCELED,
    }
    event_name = envelope.get("event")
    if not isinstance(event_name, str):
        raise CalendlyPayloadError("missing event")
    kind = kind_by_event.get(event_name)
    if kind is None:
        return None
    payload = envelope.get("payload")
    if not isinstance(payload, dict):
        raise CalendlyPayloadError("missing payload")
    scheduled = payload.get("scheduled_event")
    if not isinstance(scheduled, dict):
        raise CalendlyPayloadError("missing scheduled_event")
    memberships = scheduled.get("event_memberships")
    user_uri = ""
    if isinstance(memberships, list):
        for membership in memberships:
            if isinstance(membership, dict) and isinstance(membership.get("user"), str):
                user_uri = membership["user"]
                break
    if not user_uri:
        raise CalendlyPayloadError("missing event_memberships user")
    end: datetime | None = None
    end_value = scheduled.get("end_time")
    if end_value is not None:
        end = _when(end_value, "scheduled_event.end_time")
    event_type = scheduled.get("event_type")
    tracking = payload.get("tracking")
    utm_content = tracking.get("utm_content") if isinstance(tracking, dict) else None
    return CalendlyInviteeEvent(
        kind=kind,
        user_uri=user_uri,
        invitee_email=_text(payload.get("email"), "payload.email"),
        event_uri=_text(scheduled.get("uri"), "scheduled_event.uri"),
        start=_when(scheduled.get("start_time"), "scheduled_event.start_time"),
        end=end,
        event_type_uri=event_type.strip() if isinstance(event_type, str) else None,
        reference=utm_content.strip() if isinstance(utm_content, str) else None,
    )
