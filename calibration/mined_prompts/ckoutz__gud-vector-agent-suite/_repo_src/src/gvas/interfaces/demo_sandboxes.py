"""Visitor sandboxes: each visitor of the demo gets their own copy of it.

A sandbox is a new business copied from the demo business (by slug) and
filled by the demo seed, so every visitor starts from the same believable
week and never sees anyone else's chats or clicks. The visitor holds a
sandbox token (only its digest is stored); it opens the chat with the
sandbox's public key and signs the visitor in to its owner dashboard with
no e-mail. Demo deployments only: nothing is sent from them, and the
composition only builds this when ``GVAS_DEMO_MODE`` and a template slug
are set.

A sandbox lives until it has been idle for ``sandbox_idle_minutes`` (no chat
message and no dashboard request); the worker's sweep then deletes the
business and everything under it.
"""

import logging
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from gvas.application.intake import IntakeVisitor
from gvas.config import DemoSettings
from gvas.domain.customers import hash_portal_token, portal_token_matches
from gvas.domain.identifiers import BusinessId
from gvas.domain.intake import IntakeCollected, IntakeMessageRole
from gvas.infrastructure.intake_models import IntakeMessage
from gvas.infrastructure.models import Base, Business, DemoSandbox
from gvas.infrastructure.usage_models import UsageLedgerMonth
from gvas.interfaces.seed_demo import mint_owner_sign_in, seed_demo

logger = logging.getLogger(__name__)

PUBLIC_KEY_PREFIX = "gvb_sbx_"
TOKEN_PREFIX = "gvs_"  # noqa: S105 - a prefix, not a secret
# Activity is written at most this often per sandbox.
TOUCH_EVERY = timedelta(minutes=1)
VISITOR_LIMIT_REPLY = (
    "That's the end of this demo chat. Open the owner view to see the request, "
    "or start a fresh demo from the home page."
)
DAILY_LIMIT_REPLY = (
    "The demo has had a lot of visitors today, so Gus is resting. Please come back tomorrow."
)
# Every visitor plays this customer, so nobody types (or makes up) their own
# name, e-mail, phone or address.
SAM = IntakeVisitor(
    collected=IntakeCollected(
        name="Sam Rivera",
        email="sam.rivera@example.com",
        phone="(510) 555-0199",
        address="12 Oak Ave, Oakland",
    ),
    opening=(
        "Hi Sam! I'm Gus, {business}'s booking assistant. I already have your contact "
        "details. What would you like done in your yard?"
    ),
)
COPIED_FIELDS = (
    "name",
    "site_url",
    "display_name",
    "intake_brief",
    "intake_questions",
    "intake_opening",
    "intake_offer_line",
    "intake_visit_question",
    "invoice_prefix",
    "owner_email",
    # A slot pick only lands when the owner can be told about it.
    "notification_email",
    "timezone",
)


class SandboxError(RuntimeError):
    """A sandbox cannot be created or used."""


class SandboxCapacityError(SandboxError):
    """The demo already holds as many live sandboxes as it may."""


class SandboxAuthenticationError(SandboxError):
    """The sandbox token is unknown, wrong or its sandbox is gone."""


@dataclass(frozen=True)
class SandboxCreated:
    business_id: BusinessId
    public_key: str
    sandbox_token: str


class DemoSandboxes:
    def __init__(
        self,
        settings: DemoSettings,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        if not settings.mode or not settings.sandbox_template_slug:
            raise SandboxError("sandboxes need demo mode and a template business")
        self._settings = settings
        self._session_factory = session_factory

    @property
    def idle(self) -> timedelta:
        return timedelta(minutes=self._settings.sandbox_idle_minutes)

    async def create(self, now: datetime | None = None) -> SandboxCreated:
        now = now or datetime.now(UTC)
        business_id = BusinessId(uuid4())
        public_key = PUBLIC_KEY_PREFIX + secrets.token_urlsafe(16)
        token = TOKEN_PREFIX + secrets.token_urlsafe(32)
        async with self._session_factory() as session:
            # Locking the template row serializes creates, so two visitors
            # can't both pass the count at the cap.
            template = await session.scalar(
                select(Business)
                .where(Business.slug == self._settings.sandbox_template_slug)
                .with_for_update()
            )
            if template is None:
                raise SandboxError("the sandbox template business does not exist")
            live = await session.scalar(
                select(func.count())
                .select_from(DemoSandbox)
                .where(DemoSandbox.last_active_at >= now - self.idle)
            )
            if (live or 0) >= self._settings.sandbox_max_live:
                raise SandboxCapacityError("too many live sandboxes")
            session.add(
                Business(
                    id=business_id,
                    slug=f"sandbox-{business_id.hex}",
                    public_key=public_key,
                    created_at=now,
                    updated_at=now,
                    **{field: getattr(template, field) for field in COPIED_FIELDS},
                )
            )
            await session.flush()
            session.add(
                DemoSandbox(
                    business_id=business_id,
                    token_hash=hash_portal_token(token),
                    created_at=now,
                    last_active_at=now,
                )
            )
            await session.commit()
        await seed_demo(self._session_factory, business_id, reset=False, now=now)
        logger.info("demo sandbox created (business %s)", business_id)
        return SandboxCreated(business_id, public_key, token)

    async def _find(self, session: AsyncSession, token: str, now: datetime) -> DemoSandbox:
        if not token.startswith(TOKEN_PREFIX):
            raise SandboxAuthenticationError("not a sandbox token")
        token_hash = hash_portal_token(token)
        sandbox = await session.scalar(
            select(DemoSandbox).where(DemoSandbox.token_hash == token_hash)
        )
        if (
            sandbox is None
            or not portal_token_matches(token, sandbox.token_hash)
            or _utc(sandbox.last_active_at) < now - self.idle
        ):
            raise SandboxAuthenticationError("unknown or expired sandbox")
        return sandbox

    async def sign_in(self, token: str, now: datetime | None = None) -> str:
        """A single-use owner sign-in token for the sandbox's dashboard."""

        now = now or datetime.now(UTC)
        async with self._session_factory() as session:
            sandbox = await self._find(session, token, now)
            business_id = BusinessId(sandbox.business_id)
        await self.touch(business_id, now)
        return await mint_owner_sign_in(self._session_factory, business_id, now=now)

    async def business_id(self, token: str, now: datetime | None = None) -> BusinessId:
        """The visitor's own sandbox business; also counts as activity."""

        now = now or datetime.now(UTC)
        async with self._session_factory() as session:
            sandbox = await self._find(session, token, now)
            business_id = BusinessId(sandbox.business_id)
        await self.touch(business_id, now)
        return business_id

    async def public_key(self, token: str, now: datetime | None = None) -> str:
        now = now or datetime.now(UTC)
        async with self._session_factory() as session:
            sandbox = await self._find(session, token, now)
            key = await session.scalar(
                select(Business.public_key).where(Business.id == sandbox.business_id)
            )
        if key is None:
            raise SandboxAuthenticationError("sandbox has no public key")
        return key

    async def touch(self, business_id: UUID, now: datetime | None = None) -> None:
        """Push back the sandbox's deletion; a no-op for any other business."""

        now = now or datetime.now(UTC)
        async with self._session_factory() as session:
            await session.execute(
                update(DemoSandbox)
                .where(
                    DemoSandbox.business_id == business_id,
                    DemoSandbox.last_active_at < now - TOUCH_EVERY,
                )
                .values(last_active_at=now)
            )
            await session.commit()

    async def is_sandbox(self, business_id: UUID) -> bool:
        async with self._session_factory() as session:
            found = await session.scalar(
                select(DemoSandbox.business_id).where(DemoSandbox.business_id == business_id)
            )
        return found is not None

    async def visitor(self, business_id: UUID) -> IntakeVisitor | None:
        """Sam Rivera, for a chat opened in a sandbox."""

        return SAM if await self.is_sandbox(business_id) else None

    async def refusal(self, business_id: UUID, now: datetime) -> str | None:
        """The reply Gus gives instead of calling the model, when a visitor or
        the whole demo is out of messages; ``None`` lets the turn run."""

        day_start = now.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        user = IntakeMessageRole.USER.value
        async with self._session_factory() as session:
            sandbox = await session.get(DemoSandbox, business_id)
            if sandbox is not None:
                # The seeded week's customer messages predate the sandbox.
                sent = await session.scalar(
                    select(func.count())
                    .select_from(IntakeMessage)
                    .where(
                        IntakeMessage.business_id == business_id,
                        IntakeMessage.role == user,
                        IntakeMessage.created_at >= sandbox.created_at,
                    )
                )
                # The message being answered is not committed yet: it is one more.
                if (sent or 0) >= self._settings.sandbox_messages_per_visitor:
                    return VISITOR_LIMIT_REPLY
            today = await session.scalar(
                select(func.count())
                .select_from(IntakeMessage)
                .join(DemoSandbox, DemoSandbox.business_id == IntakeMessage.business_id)
                .where(
                    IntakeMessage.role == user,
                    IntakeMessage.created_at >= day_start,
                    IntakeMessage.created_at >= DemoSandbox.created_at,
                )
            )
        if (today or 0) >= self._settings.sandbox_messages_per_day:
            return DAILY_LIMIT_REPLY
        return None

    async def sweep(self, now: datetime | None = None) -> int:
        """Delete every sandbox idle past the limit; returns how many."""

        now = now or datetime.now(UTC)
        async with self._session_factory() as session:
            expired = (
                await session.scalars(
                    select(DemoSandbox.business_id).where(
                        DemoSandbox.last_active_at < now - self.idle
                    )
                )
            ).all()
            swept = 0
            for business_id in expired:
                # Re-checked under the delete: a click since the select keeps it.
                claimed = await session.execute(
                    delete(DemoSandbox).where(
                        DemoSandbox.business_id == business_id,
                        DemoSandbox.last_active_at < now - self.idle,
                    )
                )
                if not claimed.rowcount:  # type: ignore[attr-defined]
                    continue
                swept += 1
                await session.execute(
                    delete(UsageLedgerMonth).where(UsageLedgerMonth.business_id == business_id)
                )
                for table in reversed(Base.metadata.sorted_tables):
                    if "business_id" in table.c:
                        await session.execute(
                            delete(table).where(table.c.business_id == business_id)
                        )
                await session.execute(delete(Business).where(Business.id == business_id))
            await session.commit()
        if swept:
            logger.info("demo sandboxes deleted: %d", swept)
        return swept


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
