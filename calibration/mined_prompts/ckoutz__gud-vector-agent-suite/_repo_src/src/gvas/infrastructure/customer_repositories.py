from datetime import UTC, datetime
from typing import Any, cast

from sqlalchemy import or_, select, update
from sqlalchemy.engine import CursorResult, Result
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from gvas.domain.customers import (
    CustomerRecord,
    PortalLoginToken,
    PortalSession,
    ServiceRequest,
)
from gvas.domain.identifiers import BusinessId, CustomerId, ServiceRequestId
from gvas.domain.owner import OwnerLoginToken, OwnerSession
from gvas.domain.quotes import normalize_customer_email
from gvas.infrastructure.models import (
    Customer,
    OwnerLoginTokenRecord,
    OwnerSessionRecord,
    PortalLoginTokenRecord,
    PortalSessionRecord,
    ServiceRequestRecord,
)


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _aware_or_none(value: datetime | None) -> datetime | None:
    return None if value is None else _aware(value)


def _rowcount(result: Result[Any]) -> int:
    return cast(CursorResult[Any], result).rowcount


class SqlCustomerRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    @staticmethod
    def _record(row: Customer) -> CustomerRecord:
        return CustomerRecord(
            customer_id=CustomerId(row.id),
            business_id=BusinessId(row.business_id),
            email=row.email,
            display_name=row.display_name,
            phone=row.phone,
            stripe_customer_id=row.stripe_customer_id,
            sms_consent=row.sms_consent,
            sms_consent_at=_aware_or_none(row.sms_consent_at),
            created_at=_aware(row.created_at),
        )

    async def get(self, business_id: BusinessId, customer_id: CustomerId) -> CustomerRecord | None:
        row = await self.session.scalar(
            select(Customer).where(Customer.business_id == business_id, Customer.id == customer_id)
        )
        return None if row is None else self._record(row)

    async def find_by_email(self, business_id: BusinessId, email: str) -> CustomerRecord | None:
        row = await self._row_by_email(business_id, normalize_customer_email(email))
        return None if row is None else self._record(row)

    async def list_for_business(self, business_id: BusinessId) -> tuple[CustomerRecord, ...]:
        rows = await self.session.scalars(
            select(Customer)
            .where(Customer.business_id == business_id)
            .order_by(Customer.created_at.desc())
        )
        return tuple(self._record(row) for row in rows)

    async def _row_by_email(self, business_id: BusinessId, email: str) -> Customer | None:
        row: Customer | None = await self.session.scalar(
            select(Customer).where(Customer.business_id == business_id, Customer.email == email)
        )
        return row

    async def upsert(
        self,
        business_id: BusinessId,
        email: str,
        *,
        display_name: str | None,
        phone: str | None,
        now: datetime,
    ) -> CustomerRecord:
        normalized = normalize_customer_email(email)
        row = await self._row_by_email(business_id, normalized)
        if row is None:
            row = Customer(
                business_id=business_id,
                email=normalized,
                display_name=display_name,
                phone=phone,
                created_at=now,
            )
            try:
                async with self.session.begin_nested():
                    self.session.add(row)
                    await self.session.flush()
            except IntegrityError:
                # A concurrent writer created the same (business, e-mail).
                row = await self._row_by_email(business_id, normalized)
                if row is None:
                    raise
        changed = False
        if row.display_name is None and display_name:
            row.display_name = display_name
            changed = True
        if row.phone is None and phone:
            row.phone = phone
            changed = True
        if changed:
            await self.session.flush()
        return self._record(row)

    async def create(
        self,
        business_id: BusinessId,
        email: str,
        *,
        display_name: str | None,
        phone: str | None,
        now: datetime,
    ) -> CustomerRecord | None:
        normalized = normalize_customer_email(email)
        if await self._row_by_email(business_id, normalized) is not None:
            return None
        row = Customer(
            business_id=business_id,
            email=normalized,
            display_name=display_name,
            phone=phone,
            created_at=now,
        )
        try:
            async with self.session.begin_nested():
                self.session.add(row)
                await self.session.flush()
        except IntegrityError:
            # A concurrent writer created it first; theirs stays untouched.
            return None
        return self._record(row)

    async def set_stripe_customer_id(
        self, business_id: BusinessId, customer_id: CustomerId, stripe_customer_id: str
    ) -> None:
        await self.session.execute(
            update(Customer)
            .where(
                Customer.business_id == business_id,
                Customer.id == customer_id,
                Customer.stripe_customer_id.is_(None),
            )
            .values(stripe_customer_id=stripe_customer_id)
        )

    async def set_sms_consent(
        self, business_id: BusinessId, customer_id: CustomerId, consent: bool, at: datetime
    ) -> None:
        await self.session.execute(
            update(Customer)
            .where(
                Customer.business_id == business_id,
                Customer.id == customer_id,
                or_(Customer.sms_consent_at.is_(None), Customer.sms_consent_at <= at),
            )
            .values(sms_consent=consent, sms_consent_at=at)
        )


class SqlPortalLoginTokenRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    @staticmethod
    def _record(row: PortalLoginTokenRecord) -> PortalLoginToken:
        return PortalLoginToken(
            token_hash=row.token_hash,
            business_id=BusinessId(row.business_id),
            customer_id=CustomerId(row.customer_id),
            expires_at=_aware(row.expires_at),
            used_at=_aware_or_none(row.used_at),
            created_at=_aware(row.created_at),
        )

    async def add(self, token: PortalLoginToken) -> None:
        self.session.add(
            PortalLoginTokenRecord(
                token_hash=token.token_hash,
                business_id=token.business_id,
                customer_id=token.customer_id,
                expires_at=token.expires_at,
                used_at=token.used_at,
                created_at=token.created_at,
            )
        )
        await self.session.flush()

    async def find_by_hash(self, token_hash: str) -> PortalLoginToken | None:
        row = await self.session.scalar(
            select(PortalLoginTokenRecord).where(PortalLoginTokenRecord.token_hash == token_hash)
        )
        return None if row is None else self._record(row)

    async def mark_used(self, token_hash: str, now: datetime) -> bool:
        result = await self.session.execute(
            update(PortalLoginTokenRecord)
            .where(
                PortalLoginTokenRecord.token_hash == token_hash,
                PortalLoginTokenRecord.used_at.is_(None),
            )
            .values(used_at=now)
        )
        return _rowcount(result) == 1


class SqlPortalSessionRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    @staticmethod
    def _record(row: PortalSessionRecord) -> PortalSession:
        return PortalSession(
            token_hash=row.token_hash,
            business_id=BusinessId(row.business_id),
            customer_id=CustomerId(row.customer_id),
            expires_at=_aware(row.expires_at),
            revoked_at=_aware_or_none(row.revoked_at),
            created_at=_aware(row.created_at),
        )

    async def add(self, session: PortalSession) -> None:
        self.session.add(
            PortalSessionRecord(
                token_hash=session.token_hash,
                business_id=session.business_id,
                customer_id=session.customer_id,
                expires_at=session.expires_at,
                revoked_at=session.revoked_at,
                created_at=session.created_at,
            )
        )
        await self.session.flush()

    async def find_by_hash(self, token_hash: str) -> PortalSession | None:
        row = await self.session.scalar(
            select(PortalSessionRecord).where(PortalSessionRecord.token_hash == token_hash)
        )
        return None if row is None else self._record(row)

    async def revoke(self, token_hash: str, now: datetime) -> bool:
        result = await self.session.execute(
            update(PortalSessionRecord)
            .where(
                PortalSessionRecord.token_hash == token_hash,
                PortalSessionRecord.revoked_at.is_(None),
                PortalSessionRecord.expires_at > now,
            )
            .values(revoked_at=now)
        )
        return _rowcount(result) == 1


class SqlServiceRequestRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(self, request: ServiceRequest) -> None:
        self.session.add(
            ServiceRequestRecord(
                id=ServiceRequestId(request.request_id),
                business_id=request.business_id,
                customer_id=request.customer_id,
                message=request.message,
                preferred_dates=request.preferred_dates,
                status=request.status,
                source=request.source,
                created_at=request.created_at,
            )
        )
        await self.session.flush()

    async def list_for_business(
        self, business_id: BusinessId, *, limit: int
    ) -> tuple[ServiceRequest, ...]:
        rows = await self.session.scalars(
            select(ServiceRequestRecord)
            .where(ServiceRequestRecord.business_id == business_id)
            .order_by(ServiceRequestRecord.created_at.desc())
            .limit(limit)
        )
        return tuple(
            ServiceRequest(
                request_id=ServiceRequestId(row.id),
                business_id=BusinessId(row.business_id),
                customer_id=CustomerId(row.customer_id),
                message=row.message,
                preferred_dates=row.preferred_dates,
                status=row.status,
                source=row.source,
                created_at=_aware(row.created_at),
            )
            for row in rows
        )


class SqlOwnerLoginTokenRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(self, token: OwnerLoginToken) -> None:
        self.session.add(
            OwnerLoginTokenRecord(
                token_hash=token.token_hash,
                business_id=token.business_id,
                email=token.email,
                expires_at=token.expires_at,
                used_at=token.used_at,
                created_at=token.created_at,
            )
        )
        await self.session.flush()

    async def find_by_hash(self, token_hash: str) -> OwnerLoginToken | None:
        row = await self.session.scalar(
            select(OwnerLoginTokenRecord).where(OwnerLoginTokenRecord.token_hash == token_hash)
        )
        if row is None:
            return None
        return OwnerLoginToken(
            token_hash=row.token_hash,
            business_id=BusinessId(row.business_id),
            email=row.email,
            expires_at=_aware(row.expires_at),
            used_at=_aware_or_none(row.used_at),
            created_at=_aware(row.created_at),
        )

    async def mark_used(self, token_hash: str, now: datetime) -> bool:
        result = await self.session.execute(
            update(OwnerLoginTokenRecord)
            .where(
                OwnerLoginTokenRecord.token_hash == token_hash,
                OwnerLoginTokenRecord.used_at.is_(None),
            )
            .values(used_at=now)
        )
        return _rowcount(result) == 1


class SqlOwnerSessionRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(self, session: OwnerSession) -> None:
        self.session.add(
            OwnerSessionRecord(
                token_hash=session.token_hash,
                business_id=session.business_id,
                email=session.email,
                expires_at=session.expires_at,
                revoked_at=session.revoked_at,
                created_at=session.created_at,
            )
        )
        await self.session.flush()

    async def find_by_hash(self, token_hash: str) -> OwnerSession | None:
        row = await self.session.scalar(
            select(OwnerSessionRecord).where(OwnerSessionRecord.token_hash == token_hash)
        )
        if row is None:
            return None
        return OwnerSession(
            token_hash=row.token_hash,
            business_id=BusinessId(row.business_id),
            email=row.email,
            expires_at=_aware(row.expires_at),
            revoked_at=_aware_or_none(row.revoked_at),
            created_at=_aware(row.created_at),
        )

    async def revoke(self, token_hash: str, now: datetime) -> bool:
        result = await self.session.execute(
            update(OwnerSessionRecord)
            .where(
                OwnerSessionRecord.token_hash == token_hash,
                OwnerSessionRecord.revoked_at.is_(None),
                OwnerSessionRecord.expires_at > now,
            )
            .values(revoked_at=now)
        )
        return _rowcount(result) == 1
