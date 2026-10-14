"""Channel-agnostic owner understanding: grammar first, thread context,
then the review model — which can only answer a known intent or unclear."""

from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from gvas.application.owner_understanding import (
    APPROVE_BOOKING_CANDIDATE,
    DECLINE_BOOKING_CANDIDATE,
    OwnerMessageUnderstanding,
)
from gvas.domain.calendar_blocks import CALENDAR_BLOCK_INTENT
from gvas.domain.identifiers import BusinessId
from gvas.domain.intake import BOOKING_INTENT
from gvas.domain.owner_understanding import (
    OwnerIntentInterpretation,
    OwnerIntentInterpretationRequest,
    OwnerIntentInterpreterError,
    OwnerMessageContext,
    OwnerUnderstandingSource,
)
from gvas.infrastructure.unit_of_work import SqlUnitOfWorkFactory

REF = "65e7b537"
BOOKING = OwnerMessageContext(booking_reference=REF)


class InterpreterFake:
    def __init__(self, answer: OwnerIntentInterpretation | Exception) -> None:
        self.answer = answer
        self.requests: list[OwnerIntentInterpretationRequest] = []

    async def interpret(
        self, request: OwnerIntentInterpretationRequest
    ) -> OwnerIntentInterpretation:
        self.requests.append(request)
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


def understanding(
    session_factory: async_sessionmaker[AsyncSession],
    interpreter: InterpreterFake | None = None,
) -> OwnerMessageUnderstanding:
    return OwnerMessageUnderstanding(SqlUnitOfWorkFactory(session_factory), interpreter)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("text", "intent", "command"),
    [
        (f"approve booking {REF}", BOOKING_INTENT, f"approve booking {REF}"),
        (f"decline booking {REF} sick", BOOKING_INTENT, f"decline booking {REF} sick"),
        ("unavailable 8-12", CALENDAR_BLOCK_INTENT, f"unavailable 8-12 booking {REF}"),
        ("Okay, book it!", BOOKING_INTENT, f"approve booking {REF}"),
        ("Yes", BOOKING_INTENT, f"approve booking {REF}"),
        ("go ahead, thanks", BOOKING_INTENT, f"approve booking {REF}"),
        ("approve", BOOKING_INTENT, f"approve booking {REF}"),
        ("no", BOOKING_INTENT, f"decline booking {REF}"),
        (
            "Can't make it, out of town that week. Thanks",
            BOOKING_INTENT,
            f"decline booking {REF} out of town that week",
        ),
    ],
)
async def test_grammar_and_context_resolve_without_the_model(
    session_factory: async_sessionmaker[AsyncSession], text: str, intent: str, command: str
) -> None:
    model = InterpreterFake(OwnerIntentInterpretation(intent=APPROVE_BOOKING_CANDIDATE))
    result = await understanding(session_factory, model).understand(
        BusinessId(uuid4()), text, BOOKING
    )
    assert result.intent == intent
    assert result.command_text == command
    assert model.requests == []


@pytest.mark.asyncio
async def test_ambiguous_text_asks_the_model_with_only_known_intents(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    model = InterpreterFake(
        OwnerIntentInterpretation(intent=DECLINE_BOOKING_CANDIDATE, reason="Truck is in the shop")
    )
    result = await understanding(session_factory, model).understand(
        BusinessId(uuid4()), "ugh the truck is in the shop, sorry", BOOKING
    )
    assert result.source is OwnerUnderstandingSource.MODEL
    assert result.command_text == f"decline booking {REF} Truck is in the shop"
    names = {candidate.name for candidate in model.requests[0].candidates}
    assert names == {APPROVE_BOOKING_CANDIDATE, DECLINE_BOOKING_CANDIDATE}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answer",
    [
        OwnerIntentInterpretation(intent="unclear"),
        OwnerIntentInterpretation(intent="delete_everything"),
        OwnerIntentInterpreterError("down"),
    ],
)
async def test_unclear_unknown_or_failed_model_never_acts(
    session_factory: async_sessionmaker[AsyncSession],
    answer: OwnerIntentInterpretation | Exception,
) -> None:
    result = await understanding(session_factory, InterpreterFake(answer)).understand(
        BusinessId(uuid4()), "maybe, let me check with my wife", BOOKING
    )
    assert not result.resolved
    assert result.question is not None and '"approve"' in result.question


@pytest.mark.asyncio
async def test_context_free_replies_are_unclear(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    model = InterpreterFake(OwnerIntentInterpretation(intent=APPROVE_BOOKING_CANDIDATE))
    result = await understanding(session_factory, model).understand(
        BusinessId(uuid4()), "yes", OwnerMessageContext()
    )
    assert not result.resolved
    assert model.requests == []
