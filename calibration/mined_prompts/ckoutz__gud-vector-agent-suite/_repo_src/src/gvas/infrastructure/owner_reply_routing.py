"""Routes owner replies to the channel adapter that ingested the conversation."""

from collections.abc import Mapping

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from gvas.domain.messages import ConversationRef, DeliveryReceipt, OutboundOwnerMessage
from gvas.domain.ports import OwnerReplyPort
from gvas.infrastructure.models import Conversation, OwnerChannelEndpoint


class OwnerReplyRoutingError(RuntimeError):
    """Raised when no configured channel owns the conversation; the dispatcher retries."""


class ChannelOwnerReplyRouter:
    """One ``OwnerReplyPort`` over several channels, keyed by endpoint namespace.

    The conversation's persisted endpoint decides which adapter sends, so the
    application never learns which channels exist. A conversation whose channel
    is not configured is a delivery error rather than a silent drop.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        adapters: Mapping[str, OwnerReplyPort],
    ) -> None:
        if not adapters:
            raise ValueError("at least one owner reply channel is required")
        self._session_factory = session_factory
        self._adapters = dict(adapters)

    async def send(
        self, conversation_ref: ConversationRef, message: OutboundOwnerMessage
    ) -> DeliveryReceipt:
        namespace = await self._namespace_of(conversation_ref)
        adapter = None if namespace is None else self._adapters.get(namespace)
        if adapter is None:
            raise OwnerReplyRoutingError(
                f"no owner reply channel for conversation "
                f"{conversation_ref.external_conversation_id}"
            )
        return await adapter.send(conversation_ref, message)

    async def _namespace_of(self, conversation_ref: ConversationRef) -> str | None:
        async with self._session_factory() as session:
            endpoint = await session.scalar(
                select(OwnerChannelEndpoint)
                .join(Conversation, Conversation.endpoint_id == OwnerChannelEndpoint.id)
                .where(
                    Conversation.business_id == conversation_ref.business_id,
                    Conversation.external_conversation_id
                    == conversation_ref.external_conversation_id,
                )
                .order_by(Conversation.id)
                .limit(1)
            )
        return None if endpoint is None else endpoint.source_namespace
