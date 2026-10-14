from gvas.application.owner_assistant import _EVENT_STATUS
from gvas.domain.owner import CalendarEventSource


def test_every_calendar_event_has_a_plain_status_and_a_request_is_not_booked() -> None:
    assert set(_EVENT_STATUS) == set(CalendarEventSource)
    assert "waiting for the owner" in _EVENT_STATUS[CalendarEventSource.REQUEST]
    assert "booked" not in _EVENT_STATUS[CalendarEventSource.REQUEST]
