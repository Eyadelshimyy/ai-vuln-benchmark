from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from gvas.domain.enums import WorkflowRunStatus
from gvas.domain.identifiers import ConversationId, WorkflowIntent, WorkflowRunId
from gvas.domain.messages import NormalizedOwnerMessage, OutboundOwnerMessage
from gvas.domain.outbox import OutboxCommand


class WorkflowContext(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: WorkflowRunId
    intent: WorkflowIntent
    message: NormalizedOwnerMessage
    conversation_id: ConversationId | None = None


TERMINAL_WORKFLOW_STATUSES = frozenset({WorkflowRunStatus.SUCCEEDED, WorkflowRunStatus.FAILED})


class WorkflowResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    status: WorkflowRunStatus
    replies: tuple[OutboundOwnerMessage, ...] = Field(default_factory=tuple)
    commands: tuple[OutboxCommand, ...] = Field(default_factory=tuple)
    detail: str | None = None

    @model_validator(mode="after")
    def validate_terminal_status(self) -> "WorkflowResult":
        if self.status not in TERMINAL_WORKFLOW_STATUSES:
            raise ValueError("workflow results must have a terminal status")
        return self


class WorkflowHandler(Protocol):
    """Handlers must use deterministic reply correlation IDs for replay safety."""

    intent: WorkflowIntent

    async def handle(self, context: WorkflowContext) -> WorkflowResult: ...


class UnknownWorkflowIntentError(LookupError):
    pass


class WorkflowRouter:
    def __init__(self, handlers: list[WorkflowHandler]) -> None:
        self._handlers = {handler.intent: handler for handler in handlers}

    def route(self, intent: WorkflowIntent) -> WorkflowHandler:
        try:
            return self._handlers[intent]
        except KeyError as error:
            raise UnknownWorkflowIntentError(str(intent)) from error
