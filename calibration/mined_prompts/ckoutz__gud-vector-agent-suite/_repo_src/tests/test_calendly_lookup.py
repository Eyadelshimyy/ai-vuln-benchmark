"""Calendly API v2 adapter: request shapes, address extraction, sanitized failures."""

from datetime import UTC, datetime
from uuid import uuid4

import httpx
import pytest

from gvas.domain.appointments import AppointmentLookupError, AppointmentWindow
from gvas.domain.identifiers import BusinessId
from gvas.infrastructure.calendly.api import CalendlyAppointmentLookup
from gvas.infrastructure.calendly.config import (
    CalendlyInstallationError,
    CalendlySettings,
    parse_calendly_installations,
)

BUSINESS_ID = BusinessId(uuid4())
USER_URI = "https://api.calendly.com/users/AAAAAAAAAAAAAAAA"
TOKEN = "eyJ-not-a-real-token"  # noqa: S105
EVENT_A = "https://api.calendly.com/scheduled_events/EVENT_A"
EVENT_B = "https://api.calendly.com/scheduled_events/EVENT_B"
WINDOW = AppointmentWindow(
    business_id=BUSINESS_ID,
    start=datetime(2026, 3, 9, tzinfo=UTC),
    end=datetime(2026, 3, 12, tzinfo=UTC),
)


def settings() -> CalendlySettings:
    return CalendlySettings(token=TOKEN, installations=f"{BUSINESS_ID}={USER_URI}")


def event(uri: str, location: dict[str, str] | None, name: str = "Site visit") -> dict[str, object]:
    payload: dict[str, object] = {
        "uri": uri,
        "name": name,
        "status": "active",
        "start_time": "2026-03-10T19:00:00.000000Z",
        "end_time": "2026-03-10T20:00:00.000000Z",
        "event_type": "https://api.calendly.com/event_types/X",
    }
    if location is not None:
        payload["location"] = location
    return payload


def invitee(name: str, email: str, answers: list[dict[str, object]]) -> dict[str, object]:
    return {
        "uri": f"{EVENT_A}/invitees/{name}",
        "name": name,
        "email": email,
        "status": "active",
        "timezone": "America/Denver",
        "questions_and_answers": answers,
    }


class Recorder:
    def __init__(self, responses: dict[str, list[httpx.Response]]) -> None:
        self.responses = responses
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        queue = self.responses[request.url.path]
        return queue.pop(0) if len(queue) > 1 else queue[0]


def client(recorder: Recorder) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(recorder))


async def test_lists_active_events_then_invitees_with_bearer_auth() -> None:
    recorder = Recorder(
        {
            "/scheduled_events": [
                httpx.Response(
                    200,
                    json={
                        "collection": [
                            event(EVENT_A, {"type": "physical", "location": "234 Del Rd"}),
                            event(EVENT_B, {"type": "google_conference", "location": "meet"}),
                            {**event("https://x/scheduled_events/C", None), "status": "canceled"},
                        ],
                        "pagination": {"count": 3, "next_page_token": None},
                    },
                )
            ],
            "/scheduled_events/EVENT_A/invitees": [
                httpx.Response(
                    200,
                    json={
                        "collection": [
                            invitee("Jane Doe", "jane@example.test", []),
                            {**invitee("Gone", "gone@example.test", []), "status": "canceled"},
                        ],
                        "pagination": {"count": 2, "next_page_token": None},
                    },
                )
            ],
            "/scheduled_events/EVENT_B/invitees": [
                httpx.Response(
                    200,
                    json={
                        "collection": [
                            invitee(
                                "Bo Lee",
                                "bo@example.test",
                                [
                                    {"question": "Phone", "answer": "555", "position": 0},
                                    {
                                        "question": "What is the service address?",
                                        "answer": "343 Thing Ave",
                                        "position": 1,
                                    },
                                ],
                            )
                        ],
                        "pagination": {"count": 1, "next_page_token": None},
                    },
                )
            ],
        }
    )
    lookup = CalendlyAppointmentLookup(settings(), client(recorder))

    found = await lookup.find(WINDOW)

    events_request = recorder.requests[0]
    assert events_request.headers["Authorization"] == f"Bearer {TOKEN}"
    assert dict(events_request.url.params) == {
        "user": USER_URI,
        "status": "active",
        "min_start_time": "2026-03-09T00:00:00.000000Z",
        "max_start_time": "2026-03-12T00:00:00.000000Z",
        "count": "100",
    }
    assert [request.url.path for request in recorder.requests[1:]] == [
        "/scheduled_events/EVENT_A/invitees",
        "/scheduled_events/EVENT_B/invitees",
    ]
    assert [(a.invitee_name, a.invitee_email, a.address) for a in found] == [
        ("Jane Doe", "jane@example.test", "234 Del Rd"),
        ("Bo Lee", "bo@example.test", "343 Thing Ave"),
    ]
    assert found[0].display_time == "Tue 1:00pm"
    assert found[0].summary == "Calendly, Tue 1:00pm, 234 Del Rd"
    assert found[0].appointment_id == "calendly:EVENT_A:jane@example.test"
    assert found[1].event_name == "Site visit"


async def test_missing_address_falls_back_to_name_and_time() -> None:
    recorder = Recorder(
        {
            "/scheduled_events": [
                httpx.Response(
                    200,
                    json={"collection": [event(EVENT_A, None)], "pagination": {"count": 1}},
                )
            ],
            "/scheduled_events/EVENT_A/invitees": [
                httpx.Response(
                    200,
                    json={
                        "collection": [
                            invitee(
                                "Jane Doe",
                                "jane@example.test",
                                [{"question": "Notes", "answer": "gate code 1", "position": 0}],
                            )
                        ],
                        "pagination": {"count": 1},
                    },
                )
            ],
        }
    )
    lookup = CalendlyAppointmentLookup(settings(), client(recorder))

    (found,) = await lookup.find(WINDOW)

    assert found.address is None
    assert found.choice_label == "Jane Doe, Tue 1:00pm"


async def test_unconfigured_business_is_not_looked_up() -> None:
    recorder = Recorder({})
    lookup = CalendlyAppointmentLookup(settings(), client(recorder))

    found = await lookup.find(WINDOW.model_copy(update={"business_id": BusinessId(uuid4())}))

    assert found == ()
    assert recorder.requests == []


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(401, json={"title": "Unauthenticated", "message": TOKEN}),
        httpx.Response(200, content=b"<html>not json</html>"),
        httpx.Response(200, json={"unexpected": True}),
    ],
)
async def test_provider_failures_are_sanitized(response: httpx.Response) -> None:
    lookup = CalendlyAppointmentLookup(
        settings(), client(Recorder({"/scheduled_events": [response]}))
    )

    with pytest.raises(AppointmentLookupError) as error:
        await lookup.find(WINDOW)

    message = str(error.value)
    assert TOKEN not in message
    assert "html" not in message
    assert message.startswith("calendly ")


async def test_network_errors_are_sanitized() -> None:
    def explode(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"boom {TOKEN}", request=request)

    lookup = CalendlyAppointmentLookup(
        settings(), httpx.AsyncClient(transport=httpx.MockTransport(explode))
    )

    with pytest.raises(AppointmentLookupError, match="^calendly was unreachable$"):
        await lookup.find(WINDOW)


def test_installations_parse_and_reject_malformed_entries() -> None:
    (installation,) = parse_calendly_installations(f" {BUSINESS_ID} = {USER_URI} ,")
    assert installation.business_id == BUSINESS_ID
    assert installation.user_uri == USER_URI

    for value in (
        "",
        f"{BUSINESS_ID}",
        f"not-a-uuid={USER_URI}",
        f"{BUSINESS_ID}=https://calendly.com/someone",
        f"{BUSINESS_ID}={USER_URI},{BUSINESS_ID}={USER_URI}",
    ):
        with pytest.raises(CalendlyInstallationError):
            parse_calendly_installations(value)


def test_settings_are_optional_as_a_set() -> None:
    assert not CalendlySettings(token="", installations="").is_configured
    assert not CalendlySettings(token="", installations="").is_partially_configured
    assert CalendlySettings(token=TOKEN, installations="").is_partially_configured
    assert settings().is_configured
