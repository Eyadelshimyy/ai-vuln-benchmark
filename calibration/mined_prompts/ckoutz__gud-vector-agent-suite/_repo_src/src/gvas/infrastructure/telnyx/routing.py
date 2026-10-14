from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from gvas.domain.messages import ConversationRef
from gvas.infrastructure.models import Conversation, OwnerChannelEndpoint
from gvas.infrastructure.telnyx.delivery import TelnyxConversationRouting
from gvas.infrastructure.telnyx.installations import TELNYX_SOURCE_NAMESPACE


class SqlTelnyxRoutingResolver:
    """Reads persisted Telnyx routing in its own short, closed read transaction."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def resolve(self, conversation_ref: ConversationRef) -> TelnyxConversationRouting | None:
        async with self._session_factory() as session:
            row = await session.scalar(
                select(Conversation)
                .join(OwnerChannelEndpoint, OwnerChannelEndpoint.id == Conversation.endpoint_id)
                .where(
                    Conversation.business_id == conversation_ref.business_id,
                    Conversation.external_conversation_id
                    == conversation_ref.external_conversation_id,
                    OwnerChannelEndpoint.source_namespace == TELNYX_SOURCE_NAMESPACE,
                )
            )
            routing = None if row is None else dict(row.routing)
        if routing is None:
            return None
        return TelnyxConversationRouting.from_routing(routing)
