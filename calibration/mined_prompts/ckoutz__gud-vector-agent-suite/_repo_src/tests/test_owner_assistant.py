"""Owner assistant: plain-language, read-only answers to owner messages."""

import json
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from composition_fakes import (
    CustomerDeliveryFake,
    OwnerReplyFake,
    QuoteDraftingFake,
    ReportGenerationFake,
    TranscriptionFake,
    application_ports,
)
from gvas.application.owner import OwnerService
from gvas.application.owner_assistant import (
    SETUP_REPLY,
    UNAVAILABLE_REPLY,
    OwnerReadTools,
    totals_text,
)
from gvas.composition import Application, build_application
from gvas.config import OpenAISettings
from gvas.domain.identifiers import BusinessId
from gvas.domain.messages import TextPart
from gvas.domain.owner_assistant import (
    OwnerAssistantAnswer,
    OwnerAssistantError,
    OwnerAssistantRequest,
    OwnerAssistantSetupError,
    OwnerAssistantTool,
    ToolRunner,
)
from gvas.infrastructure.openai_owner_assistant import (
    SYSTEM_PROMPT,
    TOOL_RESULT_MAX_CHARS,
    OpenAIOwnerAssistant,
    _fit,
)
from gvas.infrastructure.telnyx.composition import (
    SMS_QUOTES_ONLY_REPLY_PREFIX,
    sms_quotes_only_policy,
)
from gvas.infrastructure.unit_of_work import SqlUnitOfWorkFactory
from telnyx_fixtures import message_payload, normalize, seed_business


class Clock:
    def __init__(self) -> None:
        self._now = datetime.now(UTC)

    def __call__(self) -> datetime:
        self._now += timedelta(seconds=1)
        return self._now


class ModelFake:
    """Calls the money tool, then answers with what it got."""

    def __init__(self, *, fail: bool = False, error: OwnerAssistantError | None = None) -> None:
        self.requests: list[OwnerAssistantRequest] = []
        self.results: list[Any] = []
        self._fail = fail
        self._error = error

    async def answer(
        self, request: OwnerAssistantRequest, run_tool: ToolRunner
    ) -> OwnerAssistantAnswer:
        self.requests.append(request)
        if self._error is not None:
            raise self._error
        if self._fail:
            raise OwnerAssistantError("down")
        result = await run_tool("money", {})
        self.results.append(result)
        return OwnerAssistantAnswer(
            text=f"Hi! Outstanding: {result['outstanding']}.", tools_called=("money",)
        )


def build(
    session_factory: async_sessionmaker[AsyncSession],
    owner_replies: OwnerReplyFake,
    model: ModelFake | None,
) -> Application:
    ports = replace(
        application_ports(
            owner_replies=owner_replies,
            quote_drafting=QuoteDraftingFake(),
            quote_delivery=CustomerDeliveryFake(),
            transcription=TranscriptionFake({}),
            report_generation=ReportGenerationFake(),
        ),
        channel_policies=(
            sms_quotes_only_policy("the team chat", owner_assistant=model is not None),
        ),
        owner_assistant=model,
    )
    return build_application(ports, session_factory=session_factory, now=Clock())


def reply_texts(owner_replies: OwnerReplyFake) -> list[str]:
    return [
        part.text
        for _, message in owner_replies.sent
        for part in message.parts
        if isinstance(part, TextPart)
    ]


async def send(application: Application, business_id: BusinessId, text: str) -> None:
    await application.ingest_service.ingest(normalize(message_payload(text=text), business_id))
    await application.worker.drain()
    await application.worker.drain()


@pytest.mark.asyncio
async def test_hello_by_text_gets_a_plain_language_answer_for_its_own_business(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    owner_replies = OwnerReplyFake()
    model = ModelFake()

    await send(build(session_factory, owner_replies, model), business_id, "hello")

    assert reply_texts(owner_replies) == ["Hi! Outstanding: $0."]
    assert [request.business_id for request in model.requests] == [business_id]
    assert model.requests[0].text == "hello"
    assert {tool.name for tool in model.requests[0].tools} == {
        "schedule",
        "needs_you",
        "money",
        "find_customer",
        "quotes",
    }


@pytest.mark.asyncio
async def test_field_notes_by_text_still_get_the_scope_reply_with_the_assistant_on(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    owner_replies = OwnerReplyFake()
    model = ModelFake()

    await send(build(session_factory, owner_replies, model), business_id, "field notes: anode")

    assert reply_texts(owner_replies)[0].startswith(SMS_QUOTES_ONLY_REPLY_PREFIX)
    assert model.requests == []


@pytest.mark.asyncio
async def test_without_the_assistant_free_text_by_sms_gets_the_scope_reply(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    owner_replies = OwnerReplyFake()

    await send(build(session_factory, owner_replies, None), business_id, "hello")

    assert reply_texts(owner_replies)[0].startswith(SMS_QUOTES_ONLY_REPLY_PREFIX)


@pytest.mark.asyncio
async def test_a_failing_model_gets_a_short_apology_not_an_error(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    owner_replies = OwnerReplyFake()

    await send(build(session_factory, owner_replies, ModelFake(fail=True)), business_id, "hi")

    assert reply_texts(owner_replies) == [UNAVAILABLE_REPLY]


@pytest.mark.asyncio
async def test_a_refused_ai_request_says_it_is_a_setup_problem_and_logs_an_alert(
    session_factory: async_sessionmaker[AsyncSession], caplog: pytest.LogCaptureFixture
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    owner_replies = OwnerReplyFake()
    model = ModelFake(error=OwnerAssistantSetupError("openai refused the request (400): no"))

    with caplog.at_level("ERROR", logger="gvas.application.owner_assistant"):
        await send(build(session_factory, owner_replies, model), business_id, "hi")

    assert reply_texts(owner_replies) == [SETUP_REPLY]
    alerts = [r for r in caplog.records if getattr(r, "alert", None) == "owner_assistant_setup"]
    assert alerts and alerts[0].levelname == "ERROR"
    assert "(400): no" in alerts[0].getMessage()


@pytest.mark.asyncio
async def test_read_tools_reject_short_lookups_and_unknown_tools(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    factory = SqlUnitOfWorkFactory(session_factory)
    tools = OwnerReadTools(factory, OwnerService(factory))
    async with factory() as unit_of_work:
        business = await unit_of_work.businesses.get(business_id)
    assert business is not None

    assert "error" in await tools.run(business, "find_customer", {"query": "a"})
    assert "error" in await tools.run(business, "drop_tables", {})
    assert (await tools.run(business, "schedule", {"start_day": "2026-10-12", "days": 30}))[
        "days"
    ] == 7
    needs = await tools.run(business, "needs_you", {})
    assert needs == {"quotes_to_approve": [], "booking_requests": []}


def _completion(message: Mapping[str, Any]) -> httpx.Response:
    return httpx.Response(
        200, json={"choices": [{"message": message}], "usage": {"total_tokens": 10}}
    )


@pytest.mark.asyncio
async def test_openai_adapter_runs_only_offered_tools_and_returns_the_text() -> None:
    bodies: list[dict[str, Any]] = []
    answers = iter(
        [
            _completion(
                {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "1",
                            "type": "function",
                            "function": {"name": "money", "arguments": "{}"},
                        },
                        {
                            "id": "2",
                            "type": "function",
                            "function": {"name": "mark_paid", "arguments": "{}"},
                        },
                    ],
                }
            ),
            _completion({"content": "You have $7,220 outstanding."}),
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return next(answers)

    ran: list[str] = []

    async def run_tool(name: str, args: Mapping[str, Any]) -> Any:
        ran.append(name)
        return {"outstanding": "$7,220"}

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = OpenAIOwnerAssistant(OpenAISettings(api_key="test-key"), client)
        answer = await adapter.answer(
            OwnerAssistantRequest(
                business_id=BusinessId(uuid4()),
                business_name="Larkspur",
                local_now="Monday 2026-10-12, 9:00 AM PDT",
                text="what's outstanding?",
                tools=(OwnerAssistantTool("money", "Money."),),
            ),
            run_tool,
        )

    assert answer.text == "You have $7,220 outstanding."
    assert answer.tools_called == ("money",)
    assert ran == ["money"]
    tool_messages = [m for m in bodies[1]["messages"] if m["role"] == "tool"]
    assert json.loads(tool_messages[1]["content"]) == {"error": "no such tool"}
    assert [tool["function"]["name"] for tool in bodies[0]["tools"]] == ["money"]
    # OpenAI rejects function tools with reasoning on chat completions.
    assert all(body["reasoning_effort"] == "none" for body in bodies)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "error", "setup"),
    [
        (400, {"message": "Function tools with reasoning_effort are not supported"}, True),
        (401, {"message": "Incorrect API key provided"}, True),
        (402, {"message": "Insufficient credits"}, True),
        (429, {"message": "You exceeded your current quota", "code": "insufficient_quota"}, True),
        (429, {"message": "Billing hard limit reached", "code": "insufficient_quota"}, True),
        (429, {"message": "Rate limit reached"}, False),
        (500, {"message": "The server had an error"}, False),
    ],
)
async def test_openai_adapter_tells_a_setup_problem_from_a_passing_one(
    status: int, error: dict[str, str], setup: bool
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": error})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = OpenAIOwnerAssistant(OpenAISettings(api_key="test-key"), client)
        with pytest.raises(OwnerAssistantError) as raised:
            await adapter.answer(
                OwnerAssistantRequest(
                    business_id=BusinessId(uuid4()),
                    business_name="Larkspur",
                    local_now="Monday 2026-10-12, 9:00 AM PDT",
                    text="hello",
                    tools=(OwnerAssistantTool("money", "Money."),),
                ),
                lambda name, args: None,  # type: ignore[arg-type,return-value]
            )

    assert isinstance(raised.value, OwnerAssistantSetupError) is setup
    assert error["message"] in str(raised.value)
    assert "test-key" not in str(raised.value)


def test_totals_never_add_across_currencies() -> None:
    assert totals_text([(1000, "usd"), (550, "USD"), (2000, "eur")]) == "$15.50 + $20 EUR"
    assert totals_text([]) == "$0"


def test_a_too_long_tool_result_is_still_valid_json() -> None:
    for row in ("x" * 100, 'é\\"' * 50, "\u2603" * 100):
        fitted = _fit(json.dumps({"rows": [row] * 500}))
        assert len(fitted) <= TOOL_RESULT_MAX_CHARS
        assert set(json.loads(fitted)) == {"cut_short", "partial"}


@pytest.mark.asyncio
async def test_a_schedule_at_the_end_of_time_is_an_error_not_a_crash(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    factory = SqlUnitOfWorkFactory(session_factory)
    async with factory() as unit_of_work:
        business = await unit_of_work.businesses.get(business_id)
    assert business is not None

    result = await OwnerReadTools(factory, OwnerService(factory)).run(
        business, "schedule", {"start_day": "9999-12-30", "days": 7}
    )

    assert "error" in result


async def _refusal(response: httpx.Response, api_key: str = "test-key") -> OwnerAssistantError:
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: response)) as client:
        adapter = OpenAIOwnerAssistant(OpenAISettings(api_key=api_key), client)
        with pytest.raises(OwnerAssistantError) as raised:
            await adapter.answer(
                OwnerAssistantRequest(
                    business_id=BusinessId(uuid4()),
                    business_name="Larkspur",
                    local_now="Monday 2026-10-12, 9:00 AM PDT",
                    text="hello",
                    tools=(),
                ),
                lambda name, args: None,  # type: ignore[arg-type,return-value]
            )
    return raised.value


@pytest.mark.asyncio
async def test_a_refusal_that_isnt_json_is_still_a_setup_problem() -> None:
    error = await _refusal(httpx.Response(401, text="<html>Unauthorized</html>"))
    assert isinstance(error, OwnerAssistantSetupError)
    passing = await _refusal(httpx.Response(502, text="<html>Bad gateway</html>"))
    assert not isinstance(passing, OwnerAssistantSetupError)


@pytest.mark.asyncio
async def test_a_refusal_never_carries_a_credential_into_the_logs() -> None:
    key = "sk-proj-AbCdEf0123456789secret"
    echoed = f"Incorrect API key provided: {key}. Also tried or-v1-abcdef123456."
    error = await _refusal(httpx.Response(401, json={"error": {"message": echoed}}), key)
    assert isinstance(error, OwnerAssistantSetupError)
    assert key not in str(error)
    assert "or-v1-abcdef123456" not in str(error)
    assert "Incorrect API key provided" in str(error)


@pytest.mark.asyncio
async def test_a_key_echoed_across_the_length_cut_is_still_redacted() -> None:
    key = "custom" + "Z" * 40
    echoed = "x" * 280 + key
    error = await _refusal(httpx.Response(401, json={"error": {"message": echoed}}), key)
    assert "ZZZZ" not in str(error)


def test_gus_is_told_to_state_every_visits_status() -> None:
    prompt = " ".join(SYSTEM_PROMPT.split())
    assert "Every time you mention a visit, say its status" in prompt
    assert '"requested, waiting for your approval"' in prompt
    assert "never call a requested visit booked" in prompt
