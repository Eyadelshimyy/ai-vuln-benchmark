import json
from uuid import uuid4

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from gvas.application.completeness import CompletenessStatus
from gvas.application.completeness_review import MarkerCompletenessReviewer
from gvas.application.contradiction_guard import GuardedCompletenessReviewer
from gvas.config import OpenAISettings
from gvas.domain.completeness import (
    ChecklistItemKey,
    CompletenessReviewOutcome,
    CompletenessReviewRequest,
    ContradictionGuardOutcome,
    CorrelatedAnswer,
    DetectedContradiction,
    MissingChecklistItem,
    MissingItemReason,
)
from gvas.domain.identifiers import BusinessId
from gvas.infrastructure.openai_contradiction_guard import (
    ContradictionGuardError,
    OpenAIContradictionGuard,
)
from test_completeness import (
    NOW,
    checklist,
    configure,
    outgoing_messages,
    owner_reply,
    seed_context,
    seed_reply,
    service,
)

OPENAI_KEY = "sk-test"
SITE = ChecklistItemKey("site")
WORK = ChecklistItemKey("work")


class RecordingGuard:
    def __init__(self, *outcomes: ContradictionGuardOutcome) -> None:
        self._outcomes = list(outcomes)
        self.requests: list[CompletenessReviewRequest] = []

    async def detect(self, request: CompletenessReviewRequest) -> ContradictionGuardOutcome:
        self.requests.append(request)
        return self._outcomes.pop(0) if self._outcomes else ContradictionGuardOutcome()


def conflict(item_key: ChecklistItemKey = WORK) -> DetectedContradiction:
    return DetectedContradiction(
        item_key=item_key,
        question="You said inspection but the findings read like abatement, which is it?",
        detail="'work: inspection' vs 'removed the pipe insulation'",
    )


def request(
    business_id: BusinessId,
    transcript: str,
    *answers: CorrelatedAnswer,
) -> CompletenessReviewRequest:
    return CompletenessReviewRequest(
        business_id=business_id,
        checklist=checklist(business_id),
        transcript_text=transcript,
        answers=answers,
        round_index=len(answers),
    )


@pytest.mark.asyncio
async def test_guard_does_not_run_while_required_items_are_missing() -> None:
    business_id = BusinessId(uuid4())
    guard = RecordingGuard(ContradictionGuardOutcome(contradictions=(conflict(),)))
    reviewer = GuardedCompletenessReviewer(MarkerCompletenessReviewer(), guard)

    outcome = await reviewer.review(request(business_id, "site: north"))

    assert [item.item_key for item in outcome.missing_items] == [WORK]
    assert outcome.missing_items[0].reason is MissingItemReason.MISSING
    assert guard.requests == []


@pytest.mark.asyncio
async def test_clear_guard_completes_the_review() -> None:
    business_id = BusinessId(uuid4())
    guard = RecordingGuard()
    reviewer = GuardedCompletenessReviewer(MarkerCompletenessReviewer(), guard)

    outcome = await reviewer.review(request(business_id, "site: north work: inspection"))

    assert outcome.is_complete
    assert len(guard.requests) == 1


@pytest.mark.asyncio
async def test_contradiction_becomes_exactly_one_question() -> None:
    business_id = BusinessId(uuid4())
    guard = RecordingGuard(
        ContradictionGuardOutcome(
            contradictions=(
                conflict(ChecklistItemKey("not-configured")),
                conflict(WORK),
                conflict(SITE),
            )
        )
    )
    reviewer = GuardedCompletenessReviewer(MarkerCompletenessReviewer(), guard)

    outcome = await reviewer.review(request(business_id, "site: north work: inspection"))

    assert outcome == CompletenessReviewOutcome(
        missing_items=(
            MissingChecklistItem(
                item_key=WORK,
                prompt=conflict().question,
                detail=conflict().detail,
                reason=MissingItemReason.CONTRADICTION,
            ),
        )
    )


@pytest.mark.asyncio
async def test_answered_contradiction_is_not_asked_again() -> None:
    business_id = BusinessId(uuid4())
    guard = RecordingGuard(ContradictionGuardOutcome(contradictions=(conflict(),)))
    reviewer = GuardedCompletenessReviewer(MarkerCompletenessReviewer(), guard)

    outcome = await reviewer.review(
        request(
            business_id,
            "site: north work: inspection",
            CorrelatedAnswer(item_key=WORK, text="Inspection only", received_at=NOW),
        )
    )

    assert outcome.is_complete


@pytest.mark.asyncio
async def test_contradiction_question_is_asked_and_its_answer_completes_the_review(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    conversation_id, inbound_id = await seed_context(session_factory, business_id)
    await configure(session_factory, checklist(business_id))
    guard = RecordingGuard(
        ContradictionGuardOutcome(contradictions=(conflict(),)),
        ContradictionGuardOutcome(contradictions=(conflict(),)),
    )
    completeness = service(
        session_factory, GuardedCompletenessReviewer(MarkerCompletenessReviewer(), guard)
    )

    started = await completeness.start_review(
        business_id,
        conversation_id,
        f"conversation-{business_id}",
        inbound_id,
        "site: north work: inspection, removed the pipe insulation",
    )
    assert started.status is CompletenessStatus.QUESTIONS_SENT
    assert started.missing_item_keys == (WORK,)
    (question,) = await outgoing_messages(session_factory)
    assert question.parts == [{"kind": "text", "text": conflict().question}]

    reply_id = await seed_reply(session_factory, business_id, conversation_id, "resolved")
    completed = await completeness.record_owner_reply(
        business_id,
        conversation_id,
        reply_id,
        owner_reply(business_id, question.correlation_id, "Inspection only, no removal"),
    )

    assert completed.status is CompletenessStatus.COMPLETE
    assert len(guard.requests) == 2
    assert [answer.item_key for answer in guard.requests[1].answers] == [WORK]
    assert len(await outgoing_messages(session_factory)) == 1


def openai_settings() -> OpenAISettings:
    return OpenAISettings(api_key=OPENAI_KEY, review_model="review-model")


def completion(content: object) -> dict[str, object]:
    return {"choices": [{"message": {"role": "assistant", "content": json.dumps(content)}}]}


@pytest.mark.asyncio
async def test_openai_guard_sends_checklist_transcript_and_answers_as_schema_request() -> None:
    business_id = BusinessId(uuid4())
    seen: list[httpx.Request] = []

    def handle(http_request: httpx.Request) -> httpx.Response:
        seen.append(http_request)
        return httpx.Response(
            200,
            json=completion(
                {
                    "contradictions": [
                        {
                            "item_key": " work ",
                            "question": "Inspection  or\nremoval?",
                            "detail": "conflicting spans",
                        },
                        {"item_key": "unknown", "question": "q", "detail": "d"},
                        {"item_key": "site", "question": "   ", "detail": "d"},
                    ]
                }
            ),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        outcome = await OpenAIContradictionGuard(openai_settings(), client).detect(
            request(
                business_id,
                "site: north work: inspection",
                CorrelatedAnswer(item_key=SITE, text="North depot", received_at=NOW),
            )
        )

    assert outcome == ContradictionGuardOutcome(
        contradictions=(
            DetectedContradiction(
                item_key=WORK, question="Inspection or removal?", detail="conflicting spans"
            ),
        )
    )
    http_request = seen[0]
    assert http_request.url.path.endswith("/chat/completions")
    assert http_request.headers["authorization"] == f"Bearer {OPENAI_KEY}"
    body = json.loads(http_request.read())
    assert body["model"] == "review-model"
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["strict"] is True
    user_content = json.loads(body["messages"][1]["content"])
    assert user_content["transcript"] == "site: north work: inspection"
    assert [item["item_key"] for item in user_content["checklist"]] == ["site", "work"]
    assert user_content["answers"] == [{"item_key": "site", "text": "North depot"}]


@pytest.mark.asyncio
async def test_openai_guard_returns_clear_for_an_empty_list() -> None:
    def handle(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=completion({"contradictions": []}))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        outcome = await OpenAIContradictionGuard(openai_settings(), client).detect(
            request(BusinessId(uuid4()), "site: north work: inspection")
        )

    assert outcome.is_clear


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(401, json={"error": {"message": f"bad key {OPENAI_KEY}"}}),
        httpx.Response(200, json={"choices": []}),
        httpx.Response(200, json=completion({"contradictions": "not a list"})),
        httpx.Response(200, json={"choices": [{"message": {"content": "not json"}}]}),
    ],
)
async def test_openai_guard_failures_raise_for_retry(response: httpx.Response) -> None:
    def handle(_: httpx.Request) -> httpx.Response:
        return response

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(ContradictionGuardError) as error:
            await OpenAIContradictionGuard(openai_settings(), client).detect(
                request(BusinessId(uuid4()), "site: north work: inspection")
            )

    assert OPENAI_KEY not in str(error.value)


@pytest.mark.asyncio
async def test_openai_guard_network_failure_raises_for_retry() -> None:
    def handle(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(ContradictionGuardError):
            await OpenAIContradictionGuard(openai_settings(), client).detect(
                request(BusinessId(uuid4()), "site: north work: inspection")
            )


def test_openai_guard_requires_a_configured_key() -> None:
    with pytest.raises(ContradictionGuardError):
        OpenAIContradictionGuard(OpenAISettings(api_key=""), httpx.AsyncClient())
