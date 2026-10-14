from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from gvas.domain.identifiers import MessageId, OutboxCommandId
from gvas.domain.messages import InboundOwnerMessage
from gvas.domain.outbox import (
    owner_message_process_command,
)
from gvas.domain.repositories import UnitOfWork


class IngestionStatus(StrEnum):
    ACCEPTED = "accepted"
    DUPLICATE = "duplicate"


@dataclass(frozen=True)
class IngestionOutcome:
    status: IngestionStatus
    message_id: MessageId | None = None
    process_command_id: OutboxCommandId | None = None
    detail: str | None = None


class UnitOfWorkFactory(Protocol):
    def __call__(self) -> UnitOfWork: ...


class IngestOwnerMessageService:
    def __init__(self, unit_of_work_factory: UnitOfWorkFactory) -> None:
        self._unit_of_work_factory = unit_of_work_factory

    async def ingest(self, message: InboundOwnerMessage) -> IngestionOutcome:
        async with self._unit_of_work_factory() as unit_of_work:
            endpoint_id = await unit_of_work.owner_channel_endpoints.get_or_create(
                message.endpoint, message.routing
            )
            conversation_id = await unit_of_work.conversations.get_or_create(
                message.message.conversation_ref, endpoint_id, message.routing
            )
            inbound_message_id = await unit_of_work.inbound_messages.create(
                message, conversation_id, endpoint_id
            )
            if inbound_message_id is None:
                await unit_of_work.rollback()
                return IngestionOutcome(IngestionStatus.DUPLICATE)
            process_command = owner_message_process_command(
                message.message.business_id, inbound_message_id
            )
            await unit_of_work.outbox.enqueue(process_command)
            await unit_of_work.commit()
            return IngestionOutcome(
                IngestionStatus.ACCEPTED,
                message_id=inbound_message_id,
                process_command_id=process_command.command_id,
            )
