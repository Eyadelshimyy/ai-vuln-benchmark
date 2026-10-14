from datetime import timedelta

from fastapi import APIRouter
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from gvas.application.channel_policy import ChannelWorkflowPolicy
from gvas.application.ingestion import IngestOwnerMessageService
from gvas.domain.calendar_blocks import CALENDAR_BLOCK_INTENT
from gvas.domain.intake import BOOKING_INTENT
from gvas.domain.intents import UNMATCHED_MESSAGE_INTENT
from gvas.domain.quotes import QUOTE_INTENT, QUOTE_TRIGGER_PREFIX
from gvas.infrastructure.telnyx.config import TelnyxSettings
from gvas.infrastructure.telnyx.delivery import (
    TelnyxDeliveryLedger,
    TelnyxMessageSender,
    TelnyxOwnerReplyAdapter,
)
from gvas.infrastructure.telnyx.ingress import TelnyxMessagingIngress
from gvas.infrastructure.telnyx.installations import (
    TELNYX_SOURCE_NAMESPACE,
    StaticTelnyxInstallationDirectory,
)
from gvas.infrastructure.telnyx.routing import SqlTelnyxRoutingResolver
from gvas.interfaces.http.telnyx import create_telnyx_router

SMS_QUOTES_ONLY_REPLY_PREFIX = (
    "By text I can do quotes, booking replies and time blocks, so nothing was started. "
    f"Begin a quote with `{QUOTE_TRIGGER_PREFIX} ...` and reply here to approve or send it."
)


def sms_quotes_only_policy(
    field_notes_channel: str, *, owner_assistant: bool = False
) -> ChannelWorkflowPolicy:
    """Scope the SMS channel to the quote workflow.

    ``quote:`` triggers, quote follow-ups and approve/send replies run; every
    other message is answered once with this scope and where field notes go.
    """

    return ChannelWorkflowPolicy(
        source_namespace=TELNYX_SOURCE_NAMESPACE,
        # Booking decisions and time blocks ride SMS so the owner can answer
        # a booking request from a text.
        # With the owner assistant on, anything else that isn't a field note
        # gets a plain-language (read-only) answer instead of the scope reply.
        allowed_intents=frozenset(
            {QUOTE_INTENT, BOOKING_INTENT, CALENDAR_BLOCK_INTENT}
            | ({UNMATCHED_MESSAGE_INTENT} if owner_assistant else set())
        ),
        unsupported_reply=(
            f"{SMS_QUOTES_ONLY_REPLY_PREFIX} Field notes belong in {field_notes_channel}."
        ),
    )


def build_telnyx_ingress(
    ingest_service: IngestOwnerMessageService, settings: TelnyxSettings | None = None
) -> TelnyxMessagingIngress:
    resolved = settings or TelnyxSettings()
    return TelnyxMessagingIngress(
        ingest_service,
        StaticTelnyxInstallationDirectory.from_setting(resolved.installations),
        public_key=resolved.public_key,
        request_max_age=timedelta(seconds=resolved.request_max_age_seconds),
    )


def build_telnyx_webhook_router(
    ingest_service: IngestOwnerMessageService, settings: TelnyxSettings | None = None
) -> APIRouter:
    """HTTP route that verifies, normalizes, ingests and enqueues only."""

    resolved = settings or TelnyxSettings()
    return create_telnyx_router(
        build_telnyx_ingress(ingest_service, resolved), path=resolved.webhook_path
    )


def build_telnyx_owner_reply_adapter(
    sender: TelnyxMessageSender,
    session_factory: async_sessionmaker[AsyncSession],
    ledger: TelnyxDeliveryLedger,
    *,
    messaging_profile_id: str | None = None,
) -> TelnyxOwnerReplyAdapter:
    """Composes Telnyx owner-reply delivery over an explicitly supplied shared ledger."""

    return TelnyxOwnerReplyAdapter(
        sender,
        SqlTelnyxRoutingResolver(session_factory),
        ledger,
        messaging_profile_id=messaging_profile_id,
    )
