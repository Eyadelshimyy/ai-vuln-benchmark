from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from gvas.domain.enums import DeliveryStatus
from gvas.domain.identifiers import MessageId
from gvas.domain.ports import OwnerReplyPort
from gvas.domain.repositories import UnitOfWork


class OwnerReplyDeliveryError(RuntimeError):
    pass


class OwnerReplyDeliveryStatus(StrEnum):
    DELIVERED = "delivered"
    ALREADY_DELIVERED = "already_delivered"
    MISSING = "missing"


@dataclass(frozen=True)
class OwnerReplyDeliveryOutcome:
    status: OwnerReplyDeliveryStatus
    outbound_message_id: MessageId
    detail: str | None = None


class OwnerReplyUnitOfWorkFactory(Protocol):
    def __call__(self) -> UnitOfWork: ...


class DeliverOwnerReplyService:
    """Delivers one persisted owner reply through a channel-neutral port.

    The port is invoked with no unit of work open, and an already delivered
    reply is never sent twice.
    """

    def __init__(
        self,
        unit_of_work_factory: OwnerReplyUnitOfWorkFactory,
        reply_port: OwnerReplyPort,
    ) -> None:
        self._unit_of_work_factory = unit_of_work_factory
        self._reply_port = reply_port

    async def deliver(self, outbound_message_id: MessageId) -> OwnerReplyDeliveryOutcome:
        async with self._unit_of_work_factory() as unit_of_work:
            record = await unit_of_work.outbound_messages.get_for_delivery(outbound_message_id)
            await unit_of_work.commit()
        if record is None:
            return OwnerReplyDeliveryOutcome(OwnerReplyDeliveryStatus.MISSING, outbound_message_id)
        if record.status is DeliveryStatus.DELIVERED:
            return OwnerReplyDeliveryOutcome(
                OwnerReplyDeliveryStatus.ALREADY_DELIVERED, outbound_message_id
            )

        receipt = await self._reply_port.send(record.message.conversation_ref, record.message)
        if receipt.status is DeliveryStatus.FAILED:
            raise OwnerReplyDeliveryError(receipt.detail or "owner reply delivery failed")
        async with self._unit_of_work_factory() as unit_of_work:
            await unit_of_work.outbound_messages.record_delivery(outbound_message_id, receipt)
            await unit_of_work.commit()
        return OwnerReplyDeliveryOutcome(
            OwnerReplyDeliveryStatus.DELIVERED, outbound_message_id, detail=receipt.detail
        )
