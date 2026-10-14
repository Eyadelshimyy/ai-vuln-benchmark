"""Channel-neutral vocabulary for understanding what an owner asked for.

Any owner channel normalizes a message to text plus whatever context it
knows (the booking the thread is about, …). Understanding maps that to one
of the workflow intents the router already knows — or to ``unclear``,
which never acts.
"""

from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from gvas.domain.identifiers import BusinessId, WorkflowIntent

UNCLEAR_OWNER_INTENT = "unclear"
OWNER_INTENT_REASON_MAX_CHARS = 200


class OwnerUnderstandingModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class OwnerMessageContext(OwnerUnderstandingModel):
    """What the message is known to be about, from the channel's thread."""

    booking_reference: str | None = None

    def summary(self) -> str:
        if self.booking_reference:
            return f"The owner is replying about booking request #{self.booking_reference}."
        return "No specific request is known for this message."


class OwnerIntentCandidate(OwnerUnderstandingModel):
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)


class OwnerIntentInterpretationRequest(OwnerUnderstandingModel):
    business_id: BusinessId
    text: str = Field(min_length=1)
    context: OwnerMessageContext
    candidates: tuple[OwnerIntentCandidate, ...] = Field(min_length=1)


class OwnerIntentInterpretation(OwnerUnderstandingModel):
    """``intent`` is a candidate name or ``unclear``; ``reason`` carries a
    decline reason in the owner's words when there is one."""

    intent: str = Field(min_length=1)
    reason: str | None = Field(default=None, max_length=OWNER_INTENT_REASON_MAX_CHARS)


class OwnerIntentInterpreterError(RuntimeError):
    """The model could not be asked; callers treat it as ``unclear``."""


class OwnerUnderstandingSource(StrEnum):
    GRAMMAR = "grammar"
    CONTEXT = "context"
    MODEL = "model"
    NONE = "none"


class OwnerUnderstanding(OwnerUnderstandingModel):
    """``intent`` + ``command_text`` when resolved; ``question`` (what to
    ask the owner) when unclear."""

    intent: WorkflowIntent | None = None
    command_text: str | None = None
    source: OwnerUnderstandingSource
    question: str | None = None

    @property
    def resolved(self) -> bool:
        return self.intent is not None and self.command_text is not None


class OwnerMessageUnderstandingPort(Protocol):
    """What any owner channel calls with the owner's text and thread context."""

    async def understand(
        self, business_id: BusinessId, text: str, context: OwnerMessageContext
    ) -> OwnerUnderstanding: ...


__all__ = [
    "OWNER_INTENT_REASON_MAX_CHARS",
    "UNCLEAR_OWNER_INTENT",
    "OwnerIntentCandidate",
    "OwnerIntentInterpretation",
    "OwnerIntentInterpretationRequest",
    "OwnerIntentInterpreterError",
    "OwnerMessageContext",
    "OwnerMessageUnderstandingPort",
    "OwnerUnderstanding",
    "OwnerUnderstandingSource",
]
