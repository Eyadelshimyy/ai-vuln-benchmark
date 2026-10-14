"""Acceptance chains: signed Slack request in, provider call out, Slack reply back.

Nothing between the HTTP boundary and the provider is faked. The real ingress
verifies the signature, the real dispatcher drains the persisted outbox, the
real adapters build the provider requests, and the real Slack poster delivers
the owner's reply; only the network is replaced, by ``httpx.MockTransport``.
That is what isolated adapter tests cannot show: that the payload a workflow
builds is one the adapter accepts, and that the result reaches the thread the
owner wrote in.
"""

import hashlib
import hmac
import json
from datetime import UTC, datetime
from typing import TypedDict
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from gvas.application.checklist_evidence import MarkerChecklistEvidenceAttributor
from gvas.application.completeness_review import MarkerCompletenessReviewer
from gvas.application.deterministic_report import DeterministicReportGenerator
from gvas.application.templates import IndustryTemplateDefinition
from gvas.composition import Application, ApplicationPorts, build_application
from gvas.config import OpenAISettings, ResendSettings
from gvas.domain.completeness import ChecklistItem, ChecklistItemKey, ChecklistKey
from gvas.domain.identifiers import BusinessId
from gvas.domain.templates import IndustryKey, ReportTemplateSection, TemplateSetKey
from gvas.infrastructure.delivery_ledger import SqlChannelDeliveryLedger
from gvas.infrastructure.openai_transcription import OpenAITranscriber
from gvas.infrastructure.quote_drafting import DeterministicQuoteDrafter
from gvas.infrastructure.resend import ResendQuoteDeliveryAdapter
from gvas.infrastructure.slack.api import SlackFileAttachmentAccess, SlackWebApiChatPoster
from gvas.infrastructure.slack.composition import (
    build_slack_event_router,
    build_slack_owner_reply_adapter,
)
from gvas.infrastructure.slack.config import SlackSettings
from gvas.interfaces.http.app import create_app
from slack_fixtures import (
    APP_ID,
    CHANNEL,
    OWNER_USER,
    ROOT_TS,
    SIGNING_SECRET,
    TEAM_ID,
    seed_business,
)
from test_composition import Clock

QUOTE_REQUEST = "\n".join(
    (
        "quote:",
        "customer: person@example.com",
        "currency: USD",
        "item: 2 | Air sampling | 125.00",
        "item: 1 | Report | 200.00",
    )
)
DICTATED_NOTE = (
    "site: 123 Main Street. work: replaced the gutter run. "
    "sample: kitchen swab at 3.5 ppm. sample: crawlspace swab at 0.4 ppm."
)
CHECKLIST_KEY = ChecklistKey("field_notes")
CHECKLIST_ITEMS = (
    ChecklistItem(
        key=ChecklistItemKey("site"),
        prompt="Which site was visited?",
        evidence_markers=("site:",),
    ),
    ChecklistItem(
        key=ChecklistItemKey("work"),
        prompt="What work was performed?",
        evidence_markers=("work:",),
    ),
    ChecklistItem(
        key=ChecklistItemKey("samples"),
        prompt="Which samples were taken?",
        evidence_markers=("sample:",),
    ),
)
AUDIO = b"voice-note-bytes"

type JsonValue = str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]


class SlackFilePayload(TypedDict):
    id: str
    name: str
    mimetype: str
    size: int


class SlackEventFields(TypedDict):
    ts: str
    event_ts: str


class SlackEventPayload(SlackEventFields, total=False):
    """The parts of a Slack message event a chain test varies."""

    text: str
    thread_ts: str
    subtype: str
    files: list[SlackFilePayload]


def _json_object(raw: bytes) -> JsonObject:
    payload: object = json.loads(raw)
    assert isinstance(payload, dict)
    return payload


class SlackWorkspace:
    """Records the Slack posts and serves every provider this chain touches."""

    def __init__(self) -> None:
        self.posts: list[JsonObject] = []
        self.emails: list[JsonObject] = []
        self.transcribed: list[bytes] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/chat.postMessage"):
            assert request.headers["Authorization"] == "Bearer xoxb-chain"
            self.posts.append(_json_object(request.read()))
            return httpx.Response(200, json={"ok": True, "ts": f"17356{len(self.posts):05d}.0001"})
        if path.endswith("/files.info"):
            file_id = request.url.params["file"]
            return httpx.Response(
                200,
                json={
                    "ok": True,
                    "file": {
                        "id": file_id,
                        "name": "note.m4a",
                        "mimetype": "audio/mp4",
                        "size": len(AUDIO),
                        "url_private_download": "https://files.slack.com/note.m4a",
                    },
                },
            )
        if request.url.host == "files.slack.com":
            return httpx.Response(200, content=AUDIO)
        if path.endswith("/audio/transcriptions"):
            body = request.read()
            assert AUDIO in body
            self.transcribed.append(AUDIO)
            return httpx.Response(200, json={"text": DICTATED_NOTE})
        if path.endswith("/emails"):
            self.emails.append(_json_object(request.read()))
            return httpx.Response(200, json={"id": "email-1"})
        raise AssertionError(f"unexpected request to {request.url}")

    def texts(self) -> list[str]:
        return [str(post["text"]) for post in self.posts]

    def replies_in_thread(self) -> list[JsonObject]:
        return [
            post
            for post in self.posts
            if post["channel"] == CHANNEL and post["thread_ts"] == ROOT_TS
        ]


def slack_settings(business_id: BusinessId) -> SlackSettings:
    return SlackSettings(
        signing_secret=SIGNING_SECRET,
        bot_token="xoxb-chain",  # noqa: S106 - fake value for tests
        installations=f"{TEAM_ID}={business_id}:{OWNER_USER}",
    )


def build_chain(
    session_factory: async_sessionmaker[AsyncSession],
    client: httpx.AsyncClient,
    business_id: BusinessId,
) -> tuple[Application, TestClient]:
    """The production wiring, with the provider network mocked and nothing else."""

    settings = slack_settings(business_id)
    attachments = SlackFileAttachmentAccess(settings, client)
    ports = ApplicationPorts(
        owner_replies=build_slack_owner_reply_adapter(
            SlackWebApiChatPoster(settings, client),
            session_factory,
            SqlChannelDeliveryLedger(session_factory),
        ),
        quote_drafting=DeterministicQuoteDrafter(),
        quote_delivery=ResendQuoteDeliveryAdapter(
            ResendSettings(api_key="re-chain", from_address="quotes@gudvector.com"), client
        ),
        transcription=OpenAITranscriber(OpenAISettings(api_key="sk-chain"), client, attachments),
        completeness_review=MarkerCompletenessReviewer(),
        checklist_evidence=MarkerChecklistEvidenceAttributor(),
        report_generation=DeterministicReportGenerator(),
    )
    application = build_application(ports, session_factory=session_factory, now=Clock())
    app = create_app(routers=(build_slack_event_router(application.ingest_service, settings),))
    return application, TestClient(app)


def post_event(http: TestClient, event: SlackEventPayload) -> None:
    """Send a genuinely signed Slack event to the mounted Request URL."""

    payload = {
        "type": "event_callback",
        "team_id": TEAM_ID,
        "api_app_id": APP_ID,
        "event_id": f"Ev{event['ts']}",
        "event_time": int(float(event["ts"])),
        "event": {"type": "message", "channel": CHANNEL, "user": OWNER_USER, **event},
    }
    body = json.dumps(payload).encode()
    timestamp = str(int(datetime.now(tz=UTC).timestamp()))
    signature = hmac.new(
        SIGNING_SECRET.encode(), b"v0:" + timestamp.encode() + b":" + body, hashlib.sha256
    ).hexdigest()
    response = http.post(
        "/slack/events",
        content=body,
        headers={
            "X-Slack-Signature": f"v0={signature}",
            "X-Slack-Request-Timestamp": timestamp,
            "Content-Type": "application/json",
        },
    )
    assert response.status_code == 200
    assert response.json() == {"status": "accepted"}


async def configure_checklist(application: Application, business_id: BusinessId) -> None:
    await application.template_publisher.seed_industry(
        business_id,
        IndustryTemplateDefinition(
            industry_key=IndustryKey("environmental_testing"),
            template_set_key=TemplateSetKey(CHECKLIST_KEY),
            checklist_key=CHECKLIST_KEY,
            version=1,
            items=CHECKLIST_ITEMS,
            report_template_key="field_notes_report",
            report_title="Field Notes Report",
            report_sections=(
                ReportTemplateSection(
                    section_key="site_and_work",
                    heading="Site and Work",
                    checklist_item_keys=tuple(item.key for item in CHECKLIST_ITEMS),
                ),
            ),
        ),
    )


@pytest.mark.asyncio
async def test_signed_quote_request_reaches_resend_and_replies_in_the_slack_thread(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    workspace = SlackWorkspace()

    async with httpx.AsyncClient(transport=httpx.MockTransport(workspace.handle)) as client:
        application, http = build_chain(session_factory, client, business_id)
        post_event(http, {"text": QUOTE_REQUEST, "ts": ROOT_TS, "event_ts": ROOT_TS})
        await application.worker.drain()
        post_event(
            http,
            {
                "text": "approve",
                "ts": "1735689700.000200",
                "event_ts": "1735689700.000200",
                "thread_ts": ROOT_TS,
            },
        )
        await application.worker.drain()

    assert len(workspace.emails) == 1
    email = json.dumps(workspace.emails[0])
    assert "person@example.com" in email
    assert "250.00" in email
    assert "https://gudvector.com/portal/login" in email
    # Every owner reply went back to the thread the request arrived in.
    assert workspace.replies_in_thread() == workspace.posts
    assert any("Quote approved" in text for text in workspace.texts())


@pytest.mark.asyncio
async def test_signed_voice_note_is_transcribed_and_reported_into_the_slack_thread(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    workspace = SlackWorkspace()

    async with httpx.AsyncClient(transport=httpx.MockTransport(workspace.handle)) as client:
        application, http = build_chain(session_factory, client, business_id)
        await configure_checklist(application, business_id)
        post_event(
            http,
            {
                "text": "field notes:",
                "subtype": "file_share",
                "ts": ROOT_TS,
                "event_ts": ROOT_TS,
                "files": [
                    {"id": "F1", "name": "note.m4a", "mimetype": "audio/mp4", "size": len(AUDIO)}
                ],
            },
        )
        for _ in range(4):
            await application.worker.drain()

    assert workspace.transcribed == [AUDIO]
    reports = [text for text in workspace.texts() if text.startswith("Field Notes Report")]
    assert len(reports) == 1
    report = reports[0]
    assert report.splitlines()[:4] == [
        "Field Notes Report",
        "Report version 1",
        "",
        "Site and Work",
    ]
    # The dictated values, not the configured marker labels, and every
    # observation rather than only the first of each kind.
    assert "site: 123 Main Street." in report
    assert "work: replaced the gutter run." in report
    assert "sample: kitchen swab at 3.5 ppm." in report
    assert "sample: crawlspace swab at 0.4 ppm." in report
    assert workspace.replies_in_thread() == workspace.posts
