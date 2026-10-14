from pydantic import BaseModel, ConfigDict, Field

from gvas.domain.identifiers import WorkflowIntent

WORKFLOW_CONFLICT_INTENT = WorkflowIntent("workflow.conflict")
UNMATCHED_MESSAGE_INTENT = WorkflowIntent("message.unmatched")


class IntentResolution(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    intent: WorkflowIntent
    confidence: float | None = Field(default=None, ge=0, le=1)
    detail: str | None = None


class IntentUnresolvedError(LookupError):
    pass
