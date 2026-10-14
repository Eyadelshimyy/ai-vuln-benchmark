"""Phase 5 hardening: multi-program isolation, persistence, claims, pagination."""

from __future__ import annotations

import hashlib
import threading
from dataclasses import replace
from datetime import UTC, datetime

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.hackerone.errors import (
    HackerOneError,
    HackerOneSubmissionInProgressError,
)
from app.adapters.hackerone.hashes import sha256_bytes
from app.adapters.hackerone.intents import AttachmentReviewState, IntentLocalState
from app.adapters.hackerone.provider import HackerOneProvider
from app.adapters.hackerone.remote_state import HackerOneRemoteState, map_remote_state
from app.adapters.hackerone.reports import ReportSubmissionState
from app.domain.evidence import Evidence, EvidenceKind
from app.repositories.hackerone_repo import HackerOneRepository
from app.security_testing.operator_auth import OperatorSession
from tests.test_phase4.test_hackerone import (
    MockHackerOne,
    _provider,
    _session,
    _verified_finding,
)


def test_two_programs_cannot_share_scope() -> None:
    provider, _ = _provider()
    provider.sync_scope("demo")
    other = HackerOneProvider(credentials=provider.credentials, client=provider.client)
    other.lookup_program("openprog")
    provider.programs["openprog"] = other.programs["openprog"]
    demo = provider.programs["demo"]
    openprog = provider.programs["openprog"]
    demo_decision = provider.scope_provider.evaluate(demo, "https://demo.example/login")
    other_decision = provider.scope_provider.evaluate(openprog, "https://demo.example/login")
    assert demo_decision.in_scope is True
    assert other_decision.in_scope is False
    with pytest.raises(HackerOneError):
        provider.scope_provider.get_scope()


def test_attachment_hash_is_raw_bytes_and_create_is_received() -> None:
    provider, _ = _provider()
    provider.sync_scope("demo")
    finding = _verified_finding()
    draft = provider.draft_from_finding(finding, "demo", severity="high", weakness_id=639)
    intent = provider.intents.create_from_draft(draft, project_id="proj-a")
    payload = b"hello-bytes"
    record = provider.intents.add_attachment(
        intent["id"],
        filename="note.txt",
        content=payload,
        session=_session(),
        project_id="proj-a",
        reviewed=True,
    )
    assert record["review_state"] == AttachmentReviewState.RECEIVED.value
    assert record["reviewed"] is False
    assert record["sha256"] == hashlib.sha256(payload).hexdigest()
    assert record["sha256"] == sha256_bytes(payload)
    with pytest.raises(HackerOneError):
        provider.intents.authorize_upload(
            intent["id"], record["id"], session=_session(), project_id="proj-a"
        )
    reviewed = provider.intents.review_attachment(
        intent["id"], record["id"], session=_session(), project_id="proj-a"
    )
    assert reviewed["review_state"] == "reviewed"
    authorized = provider.intents.authorize_upload(
        intent["id"], record["id"], session=_session(), project_id="proj-a"
    )
    assert authorized["review_state"] == "upload_authorized"


def test_attachment_project_isolation_and_binary_secret_scan() -> None:
    provider, _ = _provider()
    provider.sync_scope("demo")
    finding = _verified_finding()
    draft = provider.draft_from_finding(finding, "demo", severity="high", weakness_id=639)
    intent = provider.intents.create_from_draft(draft, project_id="proj-a")
    with pytest.raises(HackerOneError):
        provider.intents.add_attachment(
            intent["id"],
            filename="note.txt",
            content=b"ok",
            session=_session(),
            project_id="proj-b",
        )
    with pytest.raises(HackerOneError):
        provider.intents.add_attachment(
            intent["id"],
            filename="key.bin",
            content=b"\x00\x01-----BEGIN RSA PRIVATE KEY-----\x00",
            session=_session(),
            project_id="proj-a",
        )


def test_intent_states_and_ai_cannot_approve() -> None:
    provider, _ = _provider()
    provider.sync_scope("demo")
    finding = _verified_finding()
    draft = provider.draft_from_finding(finding, "demo", severity="high", weakness_id=639)
    intent = provider.intents.create_from_draft(draft, project_id="proj-a")
    assert intent["local_status"] == IntentLocalState.LOCAL_DRAFT.value
    provider.intents.mark_ready(intent["id"], project_id="proj-a")
    ai = OperatorSession(identity="assistant", authenticated_at=datetime.now(UTC), source="ai")
    with pytest.raises(Exception):
        provider.intents.human_approve(intent["id"], session=ai, project_id="proj-a")
    approved = provider.intents.human_approve(intent["id"], session=_session(), project_id="proj-a")
    assert approved["human_review_state"] == "human_approved"
    remote = provider.intents.create_remote(
        intent["id"], provider.programs["demo"], session=_session(), project_id="proj-a"
    )
    assert remote["local_status"] == "remote_intent_created"
    assert remote["remote_intent_id"] == "ri-1"


def test_concurrent_local_submission_claims() -> None:
    provider, _ = _provider()
    provider.sync_scope("demo")
    finding = _verified_finding()
    draft = provider.draft_from_finding(finding, "demo", severity="high", weakness_id=639)
    provider.reports.mark_ready(draft.id, provider.programs["demo"], finding=finding)
    provider.reports.human_approve(
        draft.id, provider.programs["demo"], session=_session(), finding=finding
    )
    errors: list[str] = []

    def _claim() -> None:
        try:
            provider.reports.claim_submission(draft.id, provider.programs["demo"])
        except HackerOneSubmissionInProgressError:
            errors.append("in_progress")
        except HackerOneError as exc:
            errors.append(exc.code)

    first = threading.Thread(target=_claim)
    second = threading.Thread(target=_claim)
    first.start()
    second.start()
    first.join()
    second.join()
    assert "in_progress" in errors or "duplicate" in errors
    assert (
        provider.reports.drafts[draft.id].submission_state
        is ReportSubmissionState.SUBMISSION_ATTEMPTED
    )


def test_unknown_remote_state_preserved() -> None:
    mapped, raw = map_remote_state("awaiting-community-review")
    assert mapped is HackerOneRemoteState.UNKNOWN
    assert raw == "awaiting-community-review"
    triaged, triaged_raw = map_remote_state("triaged")
    assert triaged is HackerOneRemoteState.TRIAGED
    assert triaged_raw == "triaged"


def test_incomplete_sync_keeps_last_good_snapshot() -> None:
    import httpx

    good_provider, _ = _provider()
    good = good_provider.sync_scope("demo")
    assert good.scope_sync_complete is True
    last_good = tuple(item.asset_identifier for item in good.structured_scopes)
    version = good.scope_version

    class Incomplete(MockHackerOne):
        def __call__(self, request: httpx.Request) -> httpx.Response:
            if "structured_scopes" in request.url.path and request.method == "GET":
                return httpx.Response(
                    200,
                    json={
                        "data": [
                            {
                                "id": "1",
                                "attributes": {
                                    "asset_type": "Domain",
                                    "asset_identifier": "demo.example",
                                    "eligible_for_submission": True,
                                    "eligible_for_bounty": True,
                                },
                            }
                        ],
                        "links": {
                            "next": (
                                "https://api.hackerone.com/v1/hackers/programs/"
                                "demo/structured_scopes?page=2"
                            )
                        },
                    },
                )
            return super().__call__(request)

    incomplete = HackerOneProvider(
        credentials=good_provider.credentials, client=good_provider.client
    )
    incomplete.programs["demo"] = good
    incomplete.client._transport = httpx.MockTransport(Incomplete())
    original = incomplete.client.paginate_with_meta

    def capped(path: str, **kwargs: object) -> object:
        kwargs["page_cap"] = 1
        kwargs["continue_with_id_gt"] = False
        return original(path, **kwargs)  # type: ignore[arg-type]

    incomplete.client.paginate_with_meta = capped  # type: ignore[method-assign]
    result = incomplete.sync_scope("demo")
    assert result.scope_sync_complete is False
    assert tuple(item.asset_identifier for item in result.structured_scopes) == last_good
    assert result.scope_version == version


def test_unchanged_scope_does_not_increment_version() -> None:
    provider, _ = _provider()
    first = provider.sync_scope("demo")
    version = first.scope_version
    digest = first.scope_content_hash
    second = provider.sync_scope("demo")
    assert second.scope_content_hash == digest
    assert second.scope_version == version


def test_approval_invalid_after_evidence_change() -> None:
    provider, _ = _provider()
    provider.sync_scope("demo")
    finding = _verified_finding()
    draft = provider.draft_from_finding(finding, "demo", severity="high", weakness_id=639)
    provider.reports.mark_ready(draft.id, provider.programs["demo"], finding=finding)
    approved = provider.reports.human_approve(
        draft.id, provider.programs["demo"], session=_session(), finding=finding
    )
    extra = Evidence(
        kind=EvidenceKind.LOG,
        source="proxy",
        summary="new capture",
        details="GET /users/4",
    )
    changed = replace(finding, evidence=finding.evidence.extend((extra,)))
    with pytest.raises(HackerOneError) as exc:
        provider.reports.submit(
            approved.id, provider.programs["demo"], finding=changed, session=_session()
        )
    assert exc.value.code in {"approval_stale", "validation_failed"}


def test_approval_invalid_after_scope_change() -> None:
    provider, _ = _provider()
    provider.sync_scope("demo")
    finding = _verified_finding()
    draft = provider.draft_from_finding(finding, "demo", severity="high", weakness_id=639)
    provider.reports.mark_ready(draft.id, provider.programs["demo"], finding=finding)
    approved = provider.reports.human_approve(
        draft.id, provider.programs["demo"], session=_session(), finding=finding
    )
    provider.programs["demo"].scope_version += 1
    with pytest.raises(HackerOneError) as exc:
        provider.reports.submit(
            approved.id, provider.programs["demo"], finding=finding, session=_session()
        )
    assert exc.value.code in {"scope_changed", "approval_stale"}


def test_submission_persists_sanitized_result() -> None:
    provider, _ = _provider()
    provider.sync_scope("demo")
    finding = _verified_finding()
    draft = provider.draft_from_finding(finding, "demo", severity="high", weakness_id=639)
    provider.reports.mark_ready(draft.id, provider.programs["demo"], finding=finding)
    provider.reports.human_approve(
        draft.id, provider.programs["demo"], session=_session(), finding=finding
    )
    submitted = provider.reports.submit(
        draft.id, provider.programs["demo"], finding=finding, session=_session()
    )
    assert submitted.submission_result is not None
    assert submitted.submission_result.get("remote_report_id") == "r-100"
    assert submitted.submission_result.get("http_status") in {201, 200, None}
    blob = str(submitted.submission_result)
    assert "test-token-not-real" not in blob
    assert "Authorization" not in blob


def test_token_not_in_intent_or_attachment_records() -> None:
    provider, _ = _provider()
    provider.sync_scope("demo")
    finding = _verified_finding()
    draft = provider.draft_from_finding(finding, "demo", severity="high", weakness_id=639)
    intent = provider.intents.create_from_draft(draft, project_id="proj-a")
    blob = str(intent)
    assert "test-token-not-real" not in blob
    assert "HACKERONE_API_TOKEN" not in blob


@pytest.mark.asyncio
async def test_db_claim_rejects_second_submit(db_session: AsyncSession) -> None:
    provider, _ = _provider()
    provider.sync_scope("demo")
    finding = _verified_finding()
    draft = provider.draft_from_finding(finding, "demo", severity="high", weakness_id=639)
    provider.reports.mark_ready(draft.id, provider.programs["demo"], finding=finding)
    provider.reports.human_approve(
        draft.id, provider.programs["demo"], session=_session(), finding=finding
    )
    repo = HackerOneRepository(db_session)
    await repo.save_draft(provider.reports.drafts[draft.id])
    await db_session.commit()
    ok, _token = await repo.claim_submission(
        draft_id=draft.id, finding_id=str(finding.id), program_handle="demo"
    )
    assert ok is True
    ok2, reason = await repo.claim_submission(
        draft_id=draft.id, finding_id=str(finding.id), program_handle="demo"
    )
    assert ok2 is False
    assert reason in {"submission_in_progress", "duplicate"}


@pytest.mark.asyncio
async def test_intent_and_attachment_persist(db_session: AsyncSession) -> None:
    provider, _ = _provider()
    provider.sync_scope("demo")
    finding = _verified_finding()
    draft = provider.draft_from_finding(finding, "demo", severity="high", weakness_id=639)
    intent = provider.intents.create_from_draft(draft, project_id="proj-a")
    record = provider.intents.add_attachment(
        intent["id"],
        filename="note.txt",
        content=b"persist-me",
        session=_session(),
        project_id="proj-a",
        reviewed=True,
    )
    repo = HackerOneRepository(db_session)
    await repo.save_draft(draft)
    await repo.save_intent(provider.intents.record(intent["id"]))
    await repo.save_attachment(provider.intents.attachment(intent["id"], record["id"], "proj-a"))
    await db_session.commit()
    restored = HackerOneProvider(credentials=provider.credentials, client=provider.client)
    await repo.hydrate(restored)
    loaded = restored.intents.get(intent["id"])
    assert loaded["title"] == draft.title
    assert loaded["local_status"] == "local_draft"
    attachments = restored.intents.list_attachments(intent["id"])
    assert attachments[0]["sha256"] == record["sha256"]
    assert attachments[0]["review_state"] == "received"
