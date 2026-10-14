"""Owner assistant: plain-language answers to an owner's read-only questions.

The model never sees the database. It may call only the read tools the
application offers, the application runs them for the business the message
came from, and their results go back as data. Nothing here can change state.
"""

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from gvas.domain.identifiers import BusinessId
from gvas.domain.owner import CalendarEvent
from gvas.domain.repositories import BusinessRecord

OWNER_ASSISTANT_REPLY_MAX_CHARS = 640
OWNER_ASSISTANT_MAX_TOOL_ROUNDS = 3

ToolRunner = Callable[[str, Mapping[str, Any]], Awaitable[Any]]


@dataclass(frozen=True)
class OwnerAssistantTool:
    name: str
    description: str
    # JSON-schema ``properties`` of the tool's arguments; all are required.
    parameters: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OwnerAssistantRequest:
    business_id: BusinessId
    business_name: str
    # Local "now" for the business, e.g. "Mon Oct 12 2026, 11:05 AM PDT".
    local_now: str
    text: str
    tools: tuple[OwnerAssistantTool, ...]
    # Earlier turns of this owner thread, oldest first: ("owner"|"assistant", text).
    history: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class OwnerAssistantAnswer:
    text: str
    tools_called: tuple[str, ...] = ()


class OwnerAssistantError(RuntimeError):
    """The model could not be reached or answered something unusable."""


class OwnerAssistantSetupError(OwnerAssistantError):
    """The AI provider refused the request itself (key, model, request shape,
    credit): retrying won't help until someone fixes the setup."""


class OwnerAssistantPort(Protocol):
    async def answer(
        self, request: OwnerAssistantRequest, run_tool: ToolRunner
    ) -> OwnerAssistantAnswer: ...


class OwnerAnswering(Protocol):
    """Answers one owner message for the business it came from."""

    async def answer(
        self, business_id: BusinessId, text: str, history: tuple[tuple[str, str], ...] = ()
    ) -> str: ...


class BusinessCalendarView(Protocol):
    @property
    def events(self) -> Sequence[CalendarEvent]: ...

    @property
    def problems(self) -> Sequence[str]: ...


class BusinessCalendarPort(Protocol):
    async def business_calendar(
        self, business: BusinessRecord, start: datetime, end: datetime
    ) -> BusinessCalendarView: ...
