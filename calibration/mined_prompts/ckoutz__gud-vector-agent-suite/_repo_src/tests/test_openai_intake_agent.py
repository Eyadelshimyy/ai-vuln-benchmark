"""The OpenAI intake adapter: per-business prompt and the generic contract."""

import json
from typing import Any
from uuid import uuid4

import httpx
import pytest

from gvas.config import OpenAISettings
from gvas.domain.identifiers import BusinessId, IntakeConversationId
from gvas.domain.intake import IntakeTurnRequest
from gvas.infrastructure.openai_intake_agent import (
    DEFAULT_BRIEF,
    RESPONSE_SCHEMA,
    SYSTEM_PROMPT,
    OpenAIIntakeAgent,
    system_prompt,
)

BRIEF = "Güd Vector builds websites and automations; you book a discovery call."
QUESTIONS = "whether they want a website, automation, or both; their trade"


def turn_request(**overrides: Any) -> IntakeTurnRequest:
    return IntakeTurnRequest(
        business_id=BusinessId(uuid4()),
        conversation_id=IntakeConversationId(uuid4()),
        business_name="Güd Vector",
        **overrides,
    )


def model_reply(collected: dict[str, str]) -> dict[str, Any]:
    content = {
        "reply": "Got it — website, automation, or both?",
        "collected": collected,
        "ready_for_slots": False,
        "chosen_slot": None,
        "needs_human": False,
        "summary": "",
    }
    return {"choices": [{"message": {"content": json.dumps(content)}}], "usage": {}}


class Recorder:
    def __init__(self, collected: dict[str, str]) -> None:
        self.bodies: list[dict[str, Any]] = []
        self._collected = collected

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.bodies.append(json.loads(request.content))
        return httpx.Response(200, json=model_reply(self._collected))


EMPTY_COLLECTED = {key: "" for key in RESPONSE_SCHEMA["properties"]["collected"]["required"]}


@pytest.mark.asyncio
async def test_intake_agent_injects_the_business_profile_into_the_prompt() -> None:
    recorder = Recorder(
        {**EMPTY_COLLECTED, "details": " A new  website ", "notes": "Plumber, uses Jobber"}
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(recorder)) as client:
        agent = OpenAIIntakeAgent(OpenAISettings(api_key="sk-test"), client)
        turn = await agent.turn(turn_request(brief=BRIEF, questions=QUESTIONS))

    system = recorder.bodies[0]["messages"][0]["content"]
    assert BRIEF in system
    assert QUESTIONS in system
    assert DEFAULT_BRIEF not in system
    assert "property address" not in system
    user = json.loads(recorder.bodies[0]["messages"][1]["content"])
    assert user["business_brief"] == BRIEF
    assert user["intake_questions"] == QUESTIONS
    assert turn.collected.details == "A new website"
    assert turn.collected.notes == "Plumber, uses Jobber"
    assert turn.collected.address is None


@pytest.mark.asyncio
async def test_intake_agent_without_a_profile_uses_the_generic_default() -> None:
    recorder = Recorder(EMPTY_COLLECTED)
    async with httpx.AsyncClient(transport=httpx.MockTransport(recorder)) as client:
        agent = OpenAIIntakeAgent(OpenAISettings(api_key="sk-test"), client)
        await agent.turn(turn_request())

    system = recorder.bodies[0]["messages"][0]["content"]
    assert system == SYSTEM_PROMPT == system_prompt(None, None) == system_prompt("  ", "")
    assert DEFAULT_BRIEF in system
    assert "property address" in system


def test_intake_prompt_keeps_the_guardrails_for_every_profile() -> None:
    for prompt in (SYSTEM_PROMPT, system_prompt(BRIEF, QUESTIONS)):
        assert "Ask one question at a time" in prompt
        assert "follow-up questions the business says to\n  ask together" in prompt
        assert "these take precedence over the business description" in prompt
        assert "NEVER quote prices" in prompt
        assert "NEVER invent or estimate availability" in prompt
        assert "the owner will review and confirm —\n  never that anything" in prompt
        assert "never say there are no openings" in prompt
        assert "email and phone number, last" in prompt
        assert "`needs_human`" in prompt


def test_intake_schema_is_generic_and_strict() -> None:
    collected = RESPONSE_SCHEMA["properties"]["collected"]
    assert collected["required"] == [
        "name",
        "email",
        "phone",
        "details",
        "notes",
        "address",
        "propertyType",
        "urgency",
    ]
    assert set(collected["properties"]) == set(collected["required"])
    assert "problem" not in collected["properties"]
