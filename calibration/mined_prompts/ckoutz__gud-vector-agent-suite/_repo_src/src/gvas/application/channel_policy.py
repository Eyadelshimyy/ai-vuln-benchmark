"""Per-channel workflow scoping.

Some owner channels only carry a subset of the workflows. A policy names the
channel by its endpoint ``source_namespace`` and the intents it may run; the
resolver wraps the channel-neutral resolver and swaps any other intent for the
channel's unsupported intent, whose handler replies once with the channel's
own trigger list and succeeds so the message is not retried or dead-lettered.
"""

from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from gvas.domain.enums import WorkflowRunStatus
from gvas.domain.identifiers import WorkflowIntent
from gvas.domain.intents import IntentResolution
from gvas.domain.messages import NormalizedOwnerMessage, OutboundOwnerMessage, TextPart
from gvas.domain.ports import IntentResolutionPort
from gvas.domain.repositories import UnitOfWork
from gvas.domain.workflows import WorkflowContext, WorkflowResult

CHANNEL_UNSUPPORTED_INTENT_PREFIX = "message.channel_unsupported:"


class ChannelWorkflowPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_namespace: str = Field(min_length=1)
    allowed_intents: frozenset[WorkflowIntent]
    unsupported_reply: str = Field(min_length=1)

    @property
    def unsupported_intent(self) -> WorkflowIntent:
        return WorkflowIntent(f"{CHANNEL_UNSUPPORTED_INTENT_PREFIX}{self.source_namespace}")


class MessageUnitOfWorkFactory(Protocol):
    def __call__(self) -> UnitOfWork: ...


class ChannelScopedIntentResolver:
    def __init__(
        self,
        inner: IntentResolutionPort,
        policies: tuple[ChannelWorkflowPolicy, ...],
        unit_of_work_factory: MessageUnitOfWorkFactory,
    ) -> None:
        self._inner = inner
        self._policies = {policy.source_namespace: policy for policy in policies}
        self._unit_of_work_factory = unit_of_work_factory

    async def resolve(self, message: NormalizedOwnerMessage) -> IntentResolution:
        resolution = await self._inner.resolve(message)
        if not self._policies:
            return resolution
        async with self._unit_of_work_factory() as unit_of_work:
            endpoint = await unit_of_work.conversations.find_endpoint(message.conversation_ref)
            await unit_of_work.commit()
        if endpoint is None:
            return resolution
        policy = self._policies.get(endpoint.source_namespace)
        if policy is None or resolution.intent in policy.allowed_intents:
            return resolution
        return IntentResolution(
            intent=policy.unsupported_intent,
            confidence=1,
            detail=f"{resolution.intent} is not available over {policy.source_namespace}",
        )


class ChannelUnsupportedMessageHandler:
    def __init__(self, policy: ChannelWorkflowPolicy) -> None:
        self._policy = policy
        self.intent = policy.unsupported_intent

    async def handle(self, context: WorkflowContext) -> WorkflowResult:
        message = context.message
        reply = OutboundOwnerMessage(
            business_id=message.business_id,
            conversation_ref=message.conversation_ref,
            parts=(TextPart(text=self._policy.unsupported_reply),),
            correlation_id=f"{self.intent}:{message.message_key}",
        )
        return WorkflowResult(
            status=WorkflowRunStatus.SUCCEEDED,
            replies=(reply,),
            detail=f"workflow not available over {self._policy.source_namespace}: scope sent",
        )
