from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from gvas.domain.enums import MediaKind, SenderRole, WorkflowRunStatus
from gvas.domain.identifiers import BusinessId, MessageId, MessageKey, OutboxCommandId
from gvas.domain.messages import (
    AttachmentReference,
    ChannelEndpointRef,
    ConversationRef,
    InboundOwnerMessage,
    NormalizedOwnerMessage,
    OutboundOwnerMessage,
    TextPart,
)
from gvas.domain.outbox import (
    OWNER_MESSAGE_PROCESS_COMMAND_TYPE,
    OWNER_REPLY_COMMAND_TYPE,
    OutboxCommand,
    owner_message_process_command,
    owner_reply_command,
)
from gvas.domain.repositories import WorkflowClaimResult, WorkflowRunClaim


def normalized(business_id: BusinessId | None = None) -> NormalizedOwnerMessage:
    business_id = business_id or BusinessId(uuid4())
    return NormalizedOwnerMessage(
        message_key=MessageKey("key-1"),
        business_id=business_id,
        conversation_ref=ConversationRef(
            business_id=business_id, external_conversation_id="conversation"
        ),
        sender={"external_id": "owner", "role": SenderRole.OWNER},
        received_at=datetime.now(UTC),
        parts=(TextPart(text="hello"),),
    )


def envelope() -> InboundOwnerMessage:
    message = normalized()
    return InboundOwnerMessage(
        message=message,
        endpoint=ChannelEndpointRef(
            business_id=message.business_id,
            source_namespace="fixture",
            external_endpoint_id="endpoint",
        ),
        routing={"opaque": "value"},
    )


def test_message_json_round_trip_and_discriminated_parts() -> None:
    original = envelope()
    restored = InboundOwnerMessage.model_validate_json(original.model_dump_json())
    assert restored == original
    assert isinstance(restored.message.parts, tuple)
    assert isinstance(restored.message.parts[0], TextPart)


def test_naive_timestamp_rejected() -> None:
    with pytest.raises(ValidationError):
        NormalizedOwnerMessage.model_validate(
            {**normalized().model_dump(), "received_at": datetime.now()}
        )


def test_parts_and_attachment_locator_validation() -> None:
    with pytest.raises(ValidationError):
        NormalizedOwnerMessage.model_validate({**normalized().model_dump(), "parts": []})
    with pytest.raises(ValidationError):
        TextPart(text="")
    with pytest.raises(ValidationError):
        AttachmentReference(
            attachment_id=uuid4(), media_kind=MediaKind.DOCUMENT, locator="https://example.test"
        )


def test_business_ids_and_routing_boundaries_are_validated() -> None:
    business_id = BusinessId(uuid4())
    other_business_id = BusinessId(uuid4())
    with pytest.raises(ValidationError):
        NormalizedOwnerMessage(
            message_key=MessageKey("mismatch"),
            business_id=business_id,
            conversation_ref=ConversationRef(
                business_id=other_business_id, external_conversation_id="conversation"
            ),
            sender={"external_id": "owner", "role": SenderRole.OWNER},
            received_at=datetime.now(UTC),
            parts=(TextPart(text="hello"),),
        )
    message = normalized(business_id)
    with pytest.raises(ValidationError):
        InboundOwnerMessage(
            message=message,
            endpoint=ChannelEndpointRef(
                business_id=other_business_id,
                source_namespace="fixture",
                external_endpoint_id="endpoint",
            ),
            routing={},
        )
    with pytest.raises(ValidationError):
        OutboundOwnerMessage.model_validate(
            {
                "business_id": business_id,
                "conversation_ref": {
                    "business_id": other_business_id,
                    "external_conversation_id": "conversation",
                },
                "parts": (TextPart(text="hello"),),
                "correlation_id": "correlation",
            }
        )


def test_intent_and_routing_are_only_on_envelope() -> None:
    from gvas.domain.workflows import WorkflowContext

    assert "intent" not in InboundOwnerMessage.model_fields
    assert "intent" not in NormalizedOwnerMessage.model_fields
    assert "routing" not in NormalizedOwnerMessage.model_fields
    assert "routing" not in WorkflowContext.model_fields
    assert "routing" not in OutboundOwnerMessage.model_fields
    with pytest.raises(ValidationError):
        OutboundOwnerMessage.model_validate(
            {
                "business_id": envelope().message.business_id,
                "conversation_ref": envelope().message.conversation_ref,
                "parts": (TextPart(text="reply"),),
                "correlation_id": "correlation",
                "routing": {},
            }
        )


def test_owner_reply_command_is_deterministic_and_linked() -> None:
    message_id = MessageId(uuid4())
    command = owner_reply_command(BusinessId(uuid4()), message_id)
    assert command.command_type == OWNER_REPLY_COMMAND_TYPE
    assert command.outbound_message_id == message_id
    assert command == owner_reply_command(command.business_id, message_id)
    with pytest.raises(ValidationError):
        OutboxCommand(
            command_id=command.command_id,
            business_id=command.business_id,
            command_type=OWNER_REPLY_COMMAND_TYPE,
            payload={},
        )


def test_owner_message_process_command_is_deterministic_and_linked() -> None:
    inbound_message_id = MessageId(uuid4())
    command = owner_message_process_command(BusinessId(uuid4()), inbound_message_id)
    assert command.command_type == OWNER_MESSAGE_PROCESS_COMMAND_TYPE
    assert command.inbound_message_id == inbound_message_id
    assert command.outbound_message_id is None
    assert command == owner_message_process_command(command.business_id, inbound_message_id)


def test_framework_command_links_are_exclusive() -> None:
    business_id = BusinessId(uuid4())
    inbound_id = MessageId(uuid4())
    outbound_id = MessageId(uuid4())
    with pytest.raises(ValidationError):
        OutboxCommand(
            command_id=OutboxCommandId(uuid4()),
            business_id=business_id,
            command_type=OWNER_MESSAGE_PROCESS_COMMAND_TYPE,
            payload={},
        )
    with pytest.raises(ValidationError):
        OutboxCommand(
            command_id=OutboxCommandId(uuid4()),
            business_id=business_id,
            command_type=OWNER_MESSAGE_PROCESS_COMMAND_TYPE,
            payload={},
            inbound_message_id=inbound_id,
            outbound_message_id=outbound_id,
        )
    with pytest.raises(ValidationError):
        OutboxCommand(
            command_id=OutboxCommandId(uuid4()),
            business_id=business_id,
            command_type=OWNER_REPLY_COMMAND_TYPE,
            payload={},
            outbound_message_id=outbound_id,
            inbound_message_id=inbound_id,
        )
    with pytest.raises(ValidationError):
        OutboxCommand(
            command_id=OutboxCommandId(uuid4()),
            business_id=business_id,
            command_type="custom",
            payload={},
            inbound_message_id=inbound_id,
        )
    with pytest.raises(ValidationError):
        OutboxCommand(
            command_id=OutboxCommandId(uuid4()),
            business_id=business_id,
            command_type="custom",
            payload={},
            outbound_message_id=outbound_id,
        )


def test_workflow_claim_token_only_exists_for_acquired_claims() -> None:
    common = {
        "run_id": uuid4(),
        "status": WorkflowRunStatus.RUNNING,
        "intent": None,
        "attempts": 1,
    }
    acquired = WorkflowRunClaim(
        **common,
        result=WorkflowClaimResult.ACQUIRED,
        lease_token=uuid4(),
    )
    assert acquired.lease_token is not None
    with pytest.raises(ValidationError):
        WorkflowRunClaim(**common, result=WorkflowClaimResult.ACQUIRED)
    with pytest.raises(ValidationError):
        WorkflowRunClaim(
            **common,
            result=WorkflowClaimResult.BUSY,
            lease_token=uuid4(),
        )
