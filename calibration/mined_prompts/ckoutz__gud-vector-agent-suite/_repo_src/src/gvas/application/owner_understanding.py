"""Channel-agnostic owner message understanding.

Every owner channel — chat, SMS, e-mail — normalizes what the owner wrote
into text plus whatever it knows about the thread (``OwnerMessageContext``)
and asks this step what the owner wants. The answer is one of the workflow
intents the router already handles, with the command text that intent's
handler reads, or ``unclear``.

Order matters and mirrors the deterministic resolver:

1. the trigger grammar (``approve booking <ref>``, ``unavailable 8-12``,
   ``yes``/``no`` while a calendar block waits) — exact commands always win;
2. short replies that only make sense in context (``yes``, ``go ahead``,
   ``can't make it, out of town``) when the thread names a booking;
3. the review model, offered only the intents that fit the context, which
   must answer one of them or ``unclear``.

``unclear`` never acts: the channel asks the owner instead. Everything that
reaches outside GVAS still goes through its handler's own confirmation rule
(calendar blocks propose before touching the calendar; bookings move only
through ``decide_booking``).
"""

import logging
import re
from collections.abc import Callable
from datetime import UTC, datetime

from gvas.domain.calendar_blocks import (
    CALENDAR_BLOCK_INTENT,
    block_confirmation,
    unavailable_request,
    unblock_request,
)
from gvas.domain.identifiers import BusinessId
from gvas.domain.intake import BOOKING_INTENT, DECLINE_REASON_MAX_CHARS, booking_decision
from gvas.domain.owner_understanding import (
    UNCLEAR_OWNER_INTENT,
    OwnerIntentCandidate,
    OwnerIntentInterpretationRequest,
    OwnerIntentInterpreterError,
    OwnerMessageContext,
    OwnerUnderstanding,
    OwnerUnderstandingSource,
)
from gvas.domain.ports import OwnerIntentInterpreterPort
from gvas.domain.repositories import UnitOfWork
from gvas.domain.usage import UsageCeilingGuard, UsageKind

logger = logging.getLogger(__name__)

UnitOfWorkFactory = Callable[[], UnitOfWork]

APPROVE_BOOKING_CANDIDATE = "approve_booking"
DECLINE_BOOKING_CANDIDATE = "decline_booking"

_APPROVE_PHRASES = frozenset(
    {
        "approve",
        "approved",
        "approve it",
        "accept",
        "accepted",
        "book it",
        "ok book it",
        "okay book it",
        "yes book it",
        "please book it",
        "confirm",
        "confirmed",
        "do it",
        "go ahead",
        "go for it",
        "looks good",
        "ok",
        "okay",
        "sounds good",
        "sure",
        "that works",
        "works for me",
        "y",
        "yeah",
        "yep",
        "yes",
        "yes please",
        "yup",
    }
)
_DECLINE = re.compile(
    r"^(?:no|nope|decline|declined|reject|rejected|"
    r"can'?t make it|cannot make it|can not make it|"
    r"not available|i'?m not available|i am not available)"
    r"(?:\b|$)[\s,.:;!-]*(?P<reason>.*)$",
    re.IGNORECASE | re.DOTALL,
)
_SIGN_OFF = re.compile(
    r"[\s,.!-]*(?:thanks|thank you|thx|cheers)?[\s,.!]*$",
    re.IGNORECASE,
)


def _phrase(text: str) -> str:
    """Lowercased, apostrophes kept, trailing sign-off and punctuation gone."""

    words = " ".join(text.replace("’", "'").split()).casefold()
    words = _SIGN_OFF.sub("", words)
    return " ".join(re.sub(r"[^\w' ]+", " ", words).split())


def unclear_question(context: OwnerMessageContext) -> str:
    if context.booking_reference:
        ref = context.booking_reference
        return (
            f"I couldn't tell what you'd like for booking #{ref}, so nothing changed. "
            f'Reply "approve" to book it, or "decline" with a reason.'
        )
    return (
        "I couldn't tell what you'd like, so nothing changed. Reply with "
        "`approve booking <id>`, `decline booking <id> <reason>` or `unavailable 8-12`."
    )


class OwnerMessageUnderstanding:
    def __init__(
        self,
        unit_of_work_factory: UnitOfWorkFactory,
        interpreter: OwnerIntentInterpreterPort | None = None,
        *,
        ceilings: UsageCeilingGuard | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._unit_of_work_factory = unit_of_work_factory
        self._interpreter = interpreter
        self._ceilings = ceilings or UsageCeilingGuard()
        self._now = now

    async def understand(
        self, business_id: BusinessId, text: str, context: OwnerMessageContext
    ) -> OwnerUnderstanding:
        text = text.strip()
        if not text:
            return self._unclear(context)
        grammar = await self._grammar(business_id, text, context)
        if grammar is not None:
            return grammar
        reference = context.booking_reference
        if reference is None:
            return self._unclear(context)
        phrase = _phrase(text)
        if phrase in _APPROVE_PHRASES:
            return self._decision(reference, approve=True, source=OwnerUnderstandingSource.CONTEXT)
        decline = _DECLINE.match(phrase)
        if decline is not None:
            reason = _reason(text, decline.group("reason"))
            return self._decision(
                reference, approve=False, reason=reason, source=OwnerUnderstandingSource.CONTEXT
            )
        return await self._model(business_id, text, context, reference)

    async def _grammar(
        self, business_id: BusinessId, text: str, context: OwnerMessageContext
    ) -> OwnerUnderstanding | None:
        if booking_decision(text) is not None:
            return OwnerUnderstanding(
                intent=BOOKING_INTENT, command_text=text, source=OwnerUnderstandingSource.GRAMMAR
            )
        unavailable = unavailable_request(text)
        if unavailable is not None:
            command = text
            if unavailable.reference is None and context.booking_reference:
                # The thread names the booking the block is about, so the
                # customer gets the "pick another time" link.
                command = f"{text} booking {context.booking_reference}"
            return OwnerUnderstanding(
                intent=CALENDAR_BLOCK_INTENT,
                command_text=command,
                source=OwnerUnderstandingSource.GRAMMAR,
            )
        if unblock_request(text) is not None:
            return OwnerUnderstanding(
                intent=CALENDAR_BLOCK_INTENT,
                command_text=text,
                source=OwnerUnderstandingSource.GRAMMAR,
            )
        # A reply in a booking thread answers that booking, never a pending block.
        if context.booking_reference is None and block_confirmation(text) is not None:
            async with self._unit_of_work_factory() as unit_of_work:
                waiting = await unit_of_work.calendar_blocks.latest_proposed(business_id)
            if waiting is not None:
                return OwnerUnderstanding(
                    intent=CALENDAR_BLOCK_INTENT,
                    command_text=text,
                    source=OwnerUnderstandingSource.GRAMMAR,
                )
        return None

    async def _model(
        self,
        business_id: BusinessId,
        text: str,
        context: OwnerMessageContext,
        reference: str,
    ) -> OwnerUnderstanding:
        if self._interpreter is None:
            return self._unclear(context)
        if await self._ceilings.is_reached(business_id, UsageKind.REVIEW_TOKENS, now=self._now()):
            logger.info("owner understanding skipped the model: usage ceiling reached")
            return self._unclear(context)
        request = OwnerIntentInterpretationRequest(
            business_id=business_id,
            text=text,
            context=context,
            candidates=(
                OwnerIntentCandidate(
                    name=APPROVE_BOOKING_CANDIDATE,
                    description=(
                        f"The owner clearly agrees to booking request #{reference} "
                        "at the requested time."
                    ),
                ),
                OwnerIntentCandidate(
                    name=DECLINE_BOOKING_CANDIDATE,
                    description=(
                        f"The owner clearly refuses booking request #{reference}; "
                        "put the reason they give for the customer in `reason`."
                    ),
                ),
            ),
        )
        try:
            answer = await self._interpreter.interpret(request)
        except OwnerIntentInterpreterError:
            logger.warning("owner understanding model unavailable; asking the owner")
            return self._unclear(context)
        if answer.intent == APPROVE_BOOKING_CANDIDATE:
            return self._decision(reference, approve=True, source=OwnerUnderstandingSource.MODEL)
        if answer.intent == DECLINE_BOOKING_CANDIDATE:
            return self._decision(
                reference,
                approve=False,
                reason=_reason(answer.reason or "", answer.reason or ""),
                source=OwnerUnderstandingSource.MODEL,
            )
        if answer.intent != UNCLEAR_OWNER_INTENT:
            logger.warning("owner understanding model answered an unknown intent")
        return self._unclear(context)

    @staticmethod
    def _decision(
        reference: str,
        *,
        approve: bool,
        source: OwnerUnderstandingSource,
        reason: str | None = None,
    ) -> OwnerUnderstanding:
        command = f"approve booking {reference}"
        if not approve:
            command = f"decline booking {reference}"
            if reason:
                command = f"{command} {reason}"
        return OwnerUnderstanding(intent=BOOKING_INTENT, command_text=command, source=source)

    @staticmethod
    def _unclear(context: OwnerMessageContext) -> OwnerUnderstanding:
        return OwnerUnderstanding(
            source=OwnerUnderstandingSource.NONE, question=unclear_question(context)
        )


def _reason(original: str, matched: str) -> str | None:
    """The reason in the owner's own casing: the tail of ``original`` that
    the lowercased match captured."""

    matched = matched.strip()
    if not matched:
        return None
    flat = " ".join(original.replace("’", "'").split())
    index = flat.casefold().rfind(matched.split()[0])
    reason = flat[index:] if index >= 0 else matched
    reason = _SIGN_OFF.sub("", reason).strip(" ,.;:-")
    return reason[:DECLINE_REASON_MAX_CHARS] or None


__all__ = [
    "APPROVE_BOOKING_CANDIDATE",
    "DECLINE_BOOKING_CANDIDATE",
    "OwnerMessageUnderstanding",
    "unclear_question",
]
