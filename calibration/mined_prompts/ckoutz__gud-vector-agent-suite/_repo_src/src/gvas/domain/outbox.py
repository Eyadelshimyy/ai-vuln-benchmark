from datetime import datetime
from uuid import UUID, uuid5

from pydantic import BaseModel, ConfigDict, Field, model_validator

from gvas.domain.enums import OutboxStatus
from gvas.domain.identifiers import BusinessId, JsonValue, MessageId, OutboxCommandId

DEFAULT_MAX_ATTEMPTS = 3
OWNER_REPLY_COMMAND_TYPE = "owner_reply.deliver"
OWNER_REPLY_COMMAND_NAMESPACE = UUID("4f54e5f4-6a71-4c68-8d37-7cb0e5e95a4a")
OWNER_MESSAGE_PROCESS_COMMAND_TYPE = "owner_message.process"
OWNER_MESSAGE_PROCESS_COMMAND_NAMESPACE = UUID("b2b1a0c4-5f3e-4a76-9c1d-2e8f4a6b7c50")
RESERVED_COMMAND_TYPES = frozenset({OWNER_REPLY_COMMAND_TYPE, OWNER_MESSAGE_PROCESS_COMMAND_TYPE})


class OutboxCommand(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    command_id: OutboxCommandId
    business_id: BusinessId
    command_type: str = Field(min_length=1)
    payload: dict[str, JsonValue]
    dedup_key: str | None = None
    outbound_message_id: MessageId | None = None
    inbound_message_id: MessageId | None = None
    #: Not claimable before this time; ``None`` means right away.
    not_before: datetime | None = None

    @model_validator(mode="after")
    def validate_framework_links(self) -> "OutboxCommand":
        if self.command_type == OWNER_MESSAGE_PROCESS_COMMAND_TYPE:
            if self.inbound_message_id is None or self.outbound_message_id is not None:
                raise ValueError("owner message process commands require an inbound-only linkage")
        elif self.command_type == OWNER_REPLY_COMMAND_TYPE:
            if self.outbound_message_id is None or self.inbound_message_id is not None:
                raise ValueError("owner reply commands require an outbound-only linkage")
        elif self.inbound_message_id is not None or self.outbound_message_id is not None:
            raise ValueError("custom commands cannot carry framework message linkages")
        return self


def owner_reply_command(business_id: BusinessId, outbound_message_id: MessageId) -> OutboxCommand:
    return OutboxCommand(
        command_id=OutboxCommandId(uuid5(OWNER_REPLY_COMMAND_NAMESPACE, str(outbound_message_id))),
        business_id=business_id,
        command_type=OWNER_REPLY_COMMAND_TYPE,
        payload={"outbound_message_id": str(outbound_message_id)},
        dedup_key=f"owner_reply:{outbound_message_id}",
        outbound_message_id=outbound_message_id,
    )


def owner_message_process_command(
    business_id: BusinessId, inbound_message_id: MessageId
) -> OutboxCommand:
    return OutboxCommand(
        command_id=OutboxCommandId(
            uuid5(OWNER_MESSAGE_PROCESS_COMMAND_NAMESPACE, str(inbound_message_id))
        ),
        business_id=business_id,
        command_type=OWNER_MESSAGE_PROCESS_COMMAND_TYPE,
        payload={"inbound_message_id": str(inbound_message_id)},
        dedup_key=f"owner_message:{inbound_message_id}",
        inbound_message_id=inbound_message_id,
    )


class ReservedOutboxCommandTypeError(ValueError):
    pass


class InvalidOutboxTransitionError(ValueError):
    pass


class LostOutboxLeaseError(ValueError):
    pass


class OutboxRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    command: OutboxCommand
    status: OutboxStatus = OutboxStatus.PENDING
    attempts: int = Field(default=0, ge=0)
    max_attempts: int = Field(default=DEFAULT_MAX_ATTEMPTS, ge=1)
    available_at: datetime
    last_error: str | None = None
    locked_by: str | None = None
    claim_attempts: int | None = None

    @model_validator(mode="after")
    def validate_state(self) -> "OutboxRecord":
        if self.status is OutboxStatus.DEAD and self.attempts < self.max_attempts:
            raise ValueError("dead outbox record must have exhausted attempts")
        return self

    def transition(self, status: OutboxStatus, *, error: str | None = None) -> "OutboxRecord":
        allowed = {
            OutboxStatus.PENDING: {OutboxStatus.IN_PROGRESS},
            OutboxStatus.IN_PROGRESS: {
                OutboxStatus.SUCCEEDED,
                OutboxStatus.FAILED,
                OutboxStatus.DEAD,
            },
            OutboxStatus.FAILED: {OutboxStatus.IN_PROGRESS, OutboxStatus.DEAD},
            OutboxStatus.SUCCEEDED: set(),
            OutboxStatus.DEAD: set(),
        }
        if status not in allowed[self.status]:
            raise InvalidOutboxTransitionError(f"{self.status} -> {status}")
        return self.model_copy(update={"status": status, "last_error": error})
