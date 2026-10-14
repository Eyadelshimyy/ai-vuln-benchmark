from datetime import UTC, datetime
from uuid import UUID

import pytest

from gvas.domain.enums import WorkflowRunStatus
from gvas.domain.identifiers import BusinessId, MessageKey, WorkflowIntent, WorkflowRunId
from gvas.domain.intents import IntentResolution
from gvas.domain.messages import (
    ChannelEndpointRef,
    ConversationRef,
    InboundOwnerMessage,
    NormalizedOwnerMessage,
    OutboundOwnerMessage,
    TextPart,
)
from gvas.domain.workflows import WorkflowContext, WorkflowResult, WorkflowRouter


def fixture(
    source_namespace: str, endpoint_id: str, routing: dict[str, str]
) -> InboundOwnerMessage:
    business_id = BusinessId(UUID("00000000-0000-0000-0000-000000000001"))
    message = NormalizedOwnerMessage(
        message_key=MessageKey("stable"),
        business_id=business_id,
        conversation_ref=ConversationRef(business_id=business_id, external_conversation_id="same"),
        sender={"external_id": "owner", "role": "owner"},
        received_at=datetime(2025, 1, 1, tzinfo=UTC),
        parts=(TextPart(text="same content"),),
    )
    return InboundOwnerMessage(
        message=message,
        endpoint=ChannelEndpointRef(
            business_id=business_id,
            source_namespace=source_namespace,
            external_endpoint_id=endpoint_id,
        ),
        routing=routing,
    )


class FakeResolver:
    async def resolve(self, message: NormalizedOwnerMessage) -> IntentResolution:
        return IntentResolution(intent=WorkflowIntent("echo"))


class EchoHandler:
    intent = WorkflowIntent("echo")

    async def handle(self, context: WorkflowContext) -> WorkflowResult:
        return WorkflowResult(
            status=WorkflowRunStatus.SUCCEEDED,
            replies=(
                OutboundOwnerMessage(
                    business_id=context.message.business_id,
                    conversation_ref=context.message.conversation_ref,
                    parts=context.message.parts,
                    correlation_id="correlation",
                ),
            ),
        )


@pytest.mark.asyncio
async def test_adapters_do_not_assign_intent_and_workflow_ignores_routing() -> None:
    adapter_messages = [
        fixture("source-a", "endpoint-a", {"opaque": "a"}),
        fixture("source-b", "endpoint-b", {"opaque": "b"}),
    ]
    resolver = FakeResolver()
    resolutions = [await resolver.resolve(item.message) for item in adapter_messages]
    assert resolutions[0] == resolutions[1]

    handler = EchoHandler()
    router = WorkflowRouter([handler])
    selected_handlers = [router.route(resolution.intent) for resolution in resolutions]
    assert selected_handlers[0] is handler
    assert selected_handlers[1] is handler

    assert adapter_messages[0].message == adapter_messages[1].message
    run_id = WorkflowRunId(UUID("00000000-0000-0000-0000-000000000010"))
    results = [
        await selected.handle(
            WorkflowContext(run_id=run_id, intent=resolutions[index].intent, message=item.message)
        )
        for index, (selected, item) in enumerate(
            zip(selected_handlers, adapter_messages, strict=True)
        )
    ]
    assert results[0] == results[1]
    assert "routing" not in adapter_messages[0].message.model_dump()
    assert "routing" not in results[0].model_dump()
