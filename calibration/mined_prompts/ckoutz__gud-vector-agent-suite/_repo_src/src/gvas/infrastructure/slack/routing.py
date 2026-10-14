from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from gvas.domain.messages import ConversationRef
from gvas.infrastructure.models import Conversation, OwnerChannelEndpoint
from gvas.infrastructure.slack.delivery import SlackConversationRouting
from gvas.infrastructure.slack.installations import SLACK_SOURCE_NAMESPACE


class SqlSlackRoutingResolver:
    """Reads persisted Slack routing in its own short, closed read transaction."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def resolve(self, conversation_ref: ConversationRef) -> SlackConversationRouting | None:
        async with self._session_factory() as session:
            row = await session.scalar(
                select(Conversation)
                .join(OwnerChannelEndpoint, OwnerChannelEndpoint.id == Conversation.endpoint_id)
                .where(
                    Conversation.business_id == conversation_ref.business_id,
                    Conversation.external_conversation_id
                    == conversation_ref.external_conversation_id,
                    OwnerChannelEndpoint.source_namespace == SLACK_SOURCE_NAMESPACE,
                )
            )
            routing = None if row is None else dict(row.routing)
        if routing is None:
            return None
        return SlackConversationRouting.from_routing(routing)
