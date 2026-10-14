from datetime import UTC, datetime

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from gvas.domain.identifiers import (
    BusinessId,
    IntakeConversationId,
    IntakeMessageId,
    JsonValue,
)
from gvas.domain.intake import (
    AvailableSlot,
    IntakeCollected,
    IntakeConversation,
    IntakeMessage,
    IntakeMessageRole,
    IntakeState,
    SupersededBooking,
)
from gvas.infrastructure.intake_models import IntakeConversation as IntakeRow
from gvas.infrastructure.intake_models import IntakeMessage as IntakeMessageRow


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _aware_or_none(value: datetime | None) -> datetime | None:
    return None if value is None else _aware(value)


def _superseded(stored: dict[str, JsonValue] | None) -> SupersededBooking | None:
    return None if not stored else SupersededBooking.model_validate(stored)


def _stored_superseded(conversation: IntakeConversation) -> dict[str, object] | None:
    booking = conversation.superseded_booking
    return None if booking is None else booking.model_dump(mode="json")


def _slots(rows: list[JsonValue] | None) -> tuple[AvailableSlot, ...]:
    return tuple(
        slot
        for entry in rows or ()
        if isinstance(entry, dict)
        for slot in [AvailableSlot.model_validate(entry)]
    )


class SqlIntakeConversationRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    @staticmethod
    def _record(row: IntakeRow) -> IntakeConversation:
        return IntakeConversation(
            conversation_id=IntakeConversationId(row.id),
            business_id=BusinessId(row.business_id),
            customer_id=row.customer_id,
            reference=row.reference,
            token_hash=row.token_hash,
            channel=row.channel,
            state=IntakeState(row.state),
            collected=IntakeCollected.model_validate(row.collected or {}),
            proposed_slots=_slots(row.proposed_slots),
            requested_slot_start=_aware_or_none(row.requested_slot_start),
            requested_slot_end=_aware_or_none(row.requested_slot_end),
            booking_kind=row.booking_kind,
            booking_link=row.booking_link,
            booking_attempted_at=_aware_or_none(row.booking_attempted_at),
            booked_event_uri=row.booked_event_uri,
            superseded_booking=_superseded(row.superseded_booking),
            reschedule_offered_at=_aware_or_none(row.reschedule_offered_at),
            booking_event_type_uri=row.booking_event_type_uri,
            decision_reason=row.decision_reason,
            decision_at=_aware_or_none(row.decision_at),
            owner_notified_at=_aware_or_none(row.owner_notified_at),
            escalation_notified_at=_aware_or_none(row.escalation_notified_at),
            sms_consent=row.sms_consent,
            sms_consent_at=_aware_or_none(row.sms_consent_at),
            expires_at=_aware(row.expires_at),
            created_at=_aware(row.created_at),
            updated_at=_aware(row.updated_at),
        )

    async def add(self, conversation: IntakeConversation) -> None:
        self.session.add(
            IntakeRow(
                id=conversation.conversation_id,
                business_id=conversation.business_id,
                customer_id=conversation.customer_id,
                reference=conversation.reference,
                token_hash=conversation.token_hash,
                channel=conversation.channel,
                state=conversation.state.value,
                collected=conversation.collected.as_stored(),
                proposed_slots=[
                    slot.model_dump(mode="json") for slot in conversation.proposed_slots
                ],
                requested_slot_start=conversation.requested_slot_start,
                requested_slot_end=conversation.requested_slot_end,
                booking_kind=conversation.booking_kind,
                booking_link=conversation.booking_link,
                booking_attempted_at=conversation.booking_attempted_at,
                booked_event_uri=conversation.booked_event_uri,
                superseded_booking=_stored_superseded(conversation),
                reschedule_offered_at=conversation.reschedule_offered_at,
                booking_event_type_uri=conversation.booking_event_type_uri,
                decision_reason=conversation.decision_reason,
                decision_at=conversation.decision_at,
                owner_notified_at=conversation.owner_notified_at,
                escalation_notified_at=conversation.escalation_notified_at,
                sms_consent=conversation.sms_consent,
                sms_consent_at=conversation.sms_consent_at,
                expires_at=conversation.expires_at,
                created_at=conversation.created_at,
                updated_at=conversation.updated_at,
            )
        )
        await self.session.flush()

    async def get(
        self, business_id: BusinessId, conversation_id: IntakeConversationId
    ) -> IntakeConversation | None:
        row = await self.session.scalar(
            select(IntakeRow).where(
                IntakeRow.business_id == business_id, IntakeRow.id == conversation_id
            )
        )
        return None if row is None else self._record(row)

    async def find_by_reference(
        self, business_id: BusinessId, reference: str
    ) -> IntakeConversation | None:
        row = await self.session.scalar(
            select(IntakeRow).where(
                IntakeRow.business_id == business_id, IntakeRow.reference == reference
            )
        )
        return None if row is None else self._record(row)

    async def lock(
        self, business_id: BusinessId, conversation_id: IntakeConversationId
    ) -> IntakeConversation | None:
        row = await self.session.scalar(
            select(IntakeRow)
            .where(IntakeRow.business_id == business_id, IntakeRow.id == conversation_id)
            .with_for_update()
        )
        return None if row is None else self._record(row)

    async def lock_by_reference(
        self, business_id: BusinessId, reference: str
    ) -> IntakeConversation | None:
        row = await self.session.scalar(
            select(IntakeRow)
            .where(IntakeRow.business_id == business_id, IntakeRow.reference == reference)
            .with_for_update()
        )
        return None if row is None else self._record(row)

    async def lock_latest_by_invitee_email(
        self, business_id: BusinessId, email: str
    ) -> IntakeConversation | None:
        row = await self.session.scalar(
            select(IntakeRow)
            .where(
                IntakeRow.business_id == business_id,
                func.lower(IntakeRow.collected["email"].as_string()) == email.lower(),
                IntakeRow.state.in_((IntakeState.AWAITING_OWNER.value, IntakeState.APPROVED.value)),
            )
            .order_by(IntakeRow.updated_at.desc())
            .limit(1)
            .with_for_update()
        )
        return None if row is None else self._record(row)

    async def find_by_token(
        self, conversation_id: IntakeConversationId, token_hash: str
    ) -> IntakeConversation | None:
        row = await self.session.scalar(
            select(IntakeRow).where(
                IntakeRow.id == conversation_id, IntakeRow.token_hash == token_hash
            )
        )
        return None if row is None else self._record(row)

    async def save(self, conversation: IntakeConversation) -> None:
        await self.session.execute(
            update(IntakeRow)
            .where(
                IntakeRow.business_id == conversation.business_id,
                IntakeRow.id == conversation.conversation_id,
            )
            .values(
                customer_id=conversation.customer_id,
                state=conversation.state.value,
                collected=conversation.collected.as_stored(),
                proposed_slots=[
                    slot.model_dump(mode="json") for slot in conversation.proposed_slots
                ],
                requested_slot_start=conversation.requested_slot_start,
                requested_slot_end=conversation.requested_slot_end,
                booking_kind=conversation.booking_kind,
                booking_link=conversation.booking_link,
                booking_attempted_at=conversation.booking_attempted_at,
                booked_event_uri=conversation.booked_event_uri,
                superseded_booking=_stored_superseded(conversation),
                reschedule_offered_at=conversation.reschedule_offered_at,
                booking_event_type_uri=conversation.booking_event_type_uri,
                decision_reason=conversation.decision_reason,
                decision_at=conversation.decision_at,
                owner_notified_at=conversation.owner_notified_at,
                escalation_notified_at=conversation.escalation_notified_at,
                sms_consent=conversation.sms_consent,
                sms_consent_at=conversation.sms_consent_at,
                expires_at=conversation.expires_at,
                updated_at=conversation.updated_at,
            )
        )

    async def list_booking_requests(
        self, business_id: BusinessId, *, limit: int
    ) -> tuple[IntakeConversation, ...]:
        rows = await self.session.scalars(
            select(IntakeRow)
            .where(
                IntakeRow.business_id == business_id,
                IntakeRow.state.in_(
                    (
                        IntakeState.AWAITING_OWNER.value,
                        IntakeState.APPROVED.value,
                        IntakeState.DECLINED.value,
                    )
                ),
            )
            .order_by(IntakeRow.updated_at.desc())
            .limit(limit)
        )
        return tuple(self._record(row) for row in rows)

    async def list_awaiting_owner(
        self, business_id: BusinessId, *, limit: int
    ) -> tuple[IntakeConversation, ...]:
        rows = await self.session.scalars(
            select(IntakeRow)
            .where(
                IntakeRow.business_id == business_id,
                IntakeRow.state == IntakeState.AWAITING_OWNER.value,
            )
            .order_by(IntakeRow.updated_at.desc())
            .limit(limit)
        )
        return tuple(self._record(row) for row in rows)

    async def count_created_since(self, business_id: BusinessId, since: datetime) -> int:
        return int(
            await self.session.scalar(
                select(func.count())
                .select_from(IntakeRow)
                .where(
                    IntakeRow.business_id == business_id,
                    IntakeRow.created_at >= since,
                )
            )
            or 0
        )


class SqlIntakeMessageRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    @staticmethod
    def _record(row: IntakeMessageRow) -> IntakeMessage:
        return IntakeMessage(
            message_id=IntakeMessageId(row.id),
            conversation_id=IntakeConversationId(row.conversation_id),
            business_id=BusinessId(row.business_id),
            role=IntakeMessageRole(row.role),
            content=row.content,
            created_at=_aware(row.created_at),
        )

    async def add(self, message: IntakeMessage) -> None:
        self.session.add(
            IntakeMessageRow(
                id=message.message_id,
                business_id=message.business_id,
                conversation_id=message.conversation_id,
                role=message.role.value,
                content=message.content,
                created_at=message.created_at,
            )
        )
        await self.session.flush()

    async def list_for(
        self, business_id: BusinessId, conversation_id: IntakeConversationId
    ) -> tuple[IntakeMessage, ...]:
        rows = await self.session.scalars(
            select(IntakeMessageRow)
            .where(
                IntakeMessageRow.business_id == business_id,
                IntakeMessageRow.conversation_id == conversation_id,
            )
            .order_by(IntakeMessageRow.created_at, IntakeMessageRow.id)
        )
        return tuple(self._record(row) for row in rows.all())

    async def count_user(
        self, business_id: BusinessId, conversation_id: IntakeConversationId
    ) -> int:
        return int(
            await self.session.scalar(
                select(func.count())
                .select_from(IntakeMessageRow)
                .where(
                    IntakeMessageRow.business_id == business_id,
                    IntakeMessageRow.conversation_id == conversation_id,
                    IntakeMessageRow.role == IntakeMessageRole.USER.value,
                )
            )
            or 0
        )
