from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

SUPPORTED_MESSAGE_SUBTYPES = frozenset({"file_share"})


class SlackPayloadError(ValueError):
    pass


class SlackModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")


class SlackUrlVerification(SlackModel):
    type: Literal["url_verification"]
    challenge: str = Field(min_length=1)


class SlackFile(SlackModel):
    id: str = Field(min_length=1)
    name: str | None = None
    title: str | None = None
    mimetype: str | None = None
    filetype: str | None = None
    subtype: str | None = None
    size: int | None = Field(default=None, ge=0)


class SlackMessageEvent(SlackModel):
    type: str = Field(min_length=1)
    subtype: str | None = None
    channel: str = Field(min_length=1)
    channel_type: str | None = None
    user: str | None = None
    bot_id: str | None = None
    text: str | None = None
    ts: str = Field(min_length=1)
    thread_ts: str | None = None
    event_ts: str | None = None
    files: tuple[SlackFile, ...] = ()

    @property
    def thread_root_ts(self) -> str:
        return self.thread_ts or self.ts

    @property
    def is_thread_reply(self) -> bool:
        return self.thread_ts is not None and self.thread_ts != self.ts


class SlackEventCallback(SlackModel):
    type: Literal["event_callback"]
    team_id: str = Field(min_length=1)
    api_app_id: str | None = None
    enterprise_id: str | None = None
    event_id: str = Field(min_length=1)
    event_time: int | None = None
    event: SlackMessageEvent


SlackEnvelope = SlackUrlVerification | SlackEventCallback


def parse_envelope(payload: Mapping[str, object]) -> SlackEnvelope | None:
    """Parse a Slack Request URL body; unsupported envelopes return ``None``."""

    envelope_type = payload.get("type")
    if envelope_type == "url_verification":
        return _validate(SlackUrlVerification, payload)
    if envelope_type != "event_callback":
        return None
    event = payload.get("event")
    if not isinstance(event, Mapping) or event.get("type") != "message":
        return None
    return _validate(SlackEventCallback, payload)


def is_supported_owner_message(event: SlackMessageEvent) -> bool:
    if event.bot_id is not None or event.user is None:
        return False
    if event.subtype is not None and event.subtype not in SUPPORTED_MESSAGE_SUBTYPES:
        return False
    return bool((event.text or "").strip()) or bool(event.files)


def _validate[EnvelopeT: SlackModel](
    model: type[EnvelopeT], payload: Mapping[str, object]
) -> EnvelopeT:
    try:
        return model.model_validate(payload)
    except ValidationError as error:
        raise SlackPayloadError(str(error)) from error
