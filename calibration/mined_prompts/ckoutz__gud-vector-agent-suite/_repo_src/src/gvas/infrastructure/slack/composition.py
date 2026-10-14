from datetime import timedelta

from fastapi import APIRouter
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from gvas.application.ingestion import IngestOwnerMessageService
from gvas.domain.ports import AttachmentAccessPort
from gvas.infrastructure.slack.config import SlackSettings
from gvas.infrastructure.slack.delivery import (
    SlackChatPoster,
    SlackDeliveryLedger,
    SlackFileUploader,
    SlackOwnerReplyAdapter,
)
from gvas.infrastructure.slack.ingress import SlackEventIngress
from gvas.infrastructure.slack.installations import StaticSlackInstallationDirectory
from gvas.infrastructure.slack.routing import SqlSlackRoutingResolver
from gvas.interfaces.http.slack import create_slack_router


def build_slack_ingress(
    ingest_service: IngestOwnerMessageService, settings: SlackSettings | None = None
) -> SlackEventIngress:
    resolved = settings or SlackSettings()
    return SlackEventIngress(
        ingest_service,
        StaticSlackInstallationDirectory.from_setting(resolved.installations),
        signing_secret=resolved.signing_secret,
        request_max_age=timedelta(seconds=resolved.request_max_age_seconds),
    )


def build_slack_event_router(
    ingest_service: IngestOwnerMessageService, settings: SlackSettings | None = None
) -> APIRouter:
    """HTTP route that verifies, normalizes, ingests and enqueues only."""

    return create_slack_router(build_slack_ingress(ingest_service, settings))


def build_slack_owner_reply_adapter(
    poster: SlackChatPoster,
    session_factory: async_sessionmaker[AsyncSession],
    ledger: SlackDeliveryLedger,
    *,
    uploader: SlackFileUploader | None = None,
    attachments: AttachmentAccessPort | None = None,
) -> SlackOwnerReplyAdapter:
    """Composes Slack owner-reply delivery over an explicitly supplied ledger.

    The ledger is required rather than defaulted: retries are claimed by any
    worker process, so a delivery ledger that is not shared across processes
    cannot suppress a duplicate post. Deployments inject a durable shared
    ledger; tests may inject ``InMemorySlackDeliveryLedger``. An uploader plus
    an attachment source turn attachment parts into real shared files.
    """

    return SlackOwnerReplyAdapter(
        poster,
        SqlSlackRoutingResolver(session_factory),
        ledger,
        uploader=uploader,
        attachments=attachments,
    )
