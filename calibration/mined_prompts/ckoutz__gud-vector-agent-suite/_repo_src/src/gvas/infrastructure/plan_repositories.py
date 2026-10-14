from datetime import UTC, datetime
from typing import cast
from uuid import UUID, uuid4

from sqlalchemy import func, select, update
from sqlalchemy.engine import CursorResult, Result
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.sql.elements import ColumnElement

from gvas.domain.enums import MediaKind
from gvas.domain.identifiers import BusinessId
from gvas.domain.messages import AttachmentReference
from gvas.domain.object_storage import StoredObject
from gvas.domain.plan_repositories import (
    CrossBusinessPlanError,
    LostPlanSetCopyLeaseError,
    PlanCustodyUnitOfWork,
    PlanSetCopyClaim,
    PlanSetCopyClaimResult,
    PlanSetUploadRegistration,
    PlanSetUploadRepository,
    SitePlanSetRepository,
    SitePlanSetVersionRepository,
    SiteRepository,
)
from gvas.domain.plans import (
    PlanSetUpload,
    PlanSetUploadId,
    PlanSetUploadStatus,
    Site,
    SiteId,
    SitePlanSet,
    SitePlanSetId,
    SitePlanSetVersion,
    SitePlanSetVersionId,
    plan_set_upload_id,
    site_plan_set_id,
    site_plan_set_version_id,
)
from gvas.domain.repositories import OutboxRepository
from gvas.infrastructure.models import Business as BusinessRow
from gvas.infrastructure.plan_models import (
    SitePlanSetRow,
    SitePlanSetUploadRow,
    SitePlanSetVersionRow,
    SiteRow,
)
from gvas.infrastructure.repositories import SqlOutboxRepository


def _rowcount(result: Result[tuple[()]]) -> int:
    return cast(CursorResult[tuple[()]], result).rowcount


def _aware(value: datetime, name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")


def _with_utc(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _site(row: SiteRow) -> Site:
    return Site(
        site_id=SiteId(row.id),
        business_id=BusinessId(row.business_id),
        label=row.label,
        external_ref=row.external_ref,
    )


def _plan_set(row: SitePlanSetRow) -> SitePlanSet:
    return SitePlanSet(
        plan_set_id=SitePlanSetId(row.id),
        business_id=BusinessId(row.business_id),
        site_id=SiteId(row.site_id),
        plan_set_key=row.plan_set_key,
    )


def _version(row: SitePlanSetVersionRow) -> SitePlanSetVersion:
    return SitePlanSetVersion(
        version_id=SitePlanSetVersionId(row.id),
        plan_set_id=SitePlanSetId(row.plan_set_id),
        business_id=BusinessId(row.business_id),
        site_id=SiteId(row.site_id),
        version=row.version,
        artifact=AttachmentReference(
            attachment_id=row.artifact_id,
            media_kind=MediaKind(row.media_kind),
            locator=row.artifact_locator,
            mime_type=row.mime_type,
            filename=row.filename,
            byte_size=row.byte_size,
        ),
        page_count=row.page_count,
        content_digest=row.content_digest,
        byte_size=row.byte_size,
        uploaded_at=_with_utc(row.uploaded_at),
    )


def _upload(row: SitePlanSetUploadRow) -> PlanSetUpload:
    return PlanSetUpload(
        upload_id=PlanSetUploadId(row.id),
        business_id=BusinessId(row.business_id),
        site_id=SiteId(row.site_id),
        plan_set_id=SitePlanSetId(row.plan_set_id),
        source=AttachmentReference(
            attachment_id=row.source_attachment_id,
            media_kind=MediaKind(row.source_media_kind),
            locator=row.source_locator,
            mime_type=row.source_mime_type,
            filename=row.source_filename,
            byte_size=row.source_byte_size,
        ),
        status=PlanSetUploadStatus(row.status),
        attempts=row.attempts,
        version_id=SitePlanSetVersionId(row.version_id) if row.version_id else None,
    )


class SqlSiteRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get(self, business_id: BusinessId, site_id: SiteId) -> Site | None:
        row = await self.session.scalar(
            select(SiteRow).where(SiteRow.business_id == business_id, SiteRow.id == site_id)
        )
        return _site(row) if row is not None else None

    async def get_or_create(
        self,
        business_id: BusinessId,
        *,
        label: str,
        external_ref: str | None = None,
        site_id: SiteId | None = None,
        now: datetime,
    ) -> Site:
        _aware(now, "now")
        business = await self.session.scalar(
            select(BusinessRow).where(BusinessRow.id == business_id)
        )
        if business is None:
            raise CrossBusinessPlanError("site references an unknown business")
        if site_id is not None:
            existing = await self.session.scalar(
                select(SiteRow).where(SiteRow.business_id == business_id, SiteRow.id == site_id)
            )
            if existing is not None:
                return _site(existing)
        if external_ref is not None:
            existing = await self.session.scalar(
                select(SiteRow).where(
                    SiteRow.business_id == business_id,
                    SiteRow.external_ref == external_ref,
                )
            )
            if existing is not None:
                return _site(existing)
        row = SiteRow(
            id=site_id or uuid4(),
            business_id=business_id,
            label=label,
            external_ref=external_ref,
            created_at=now,
            updated_at=now,
        )
        try:
            async with self.session.begin_nested():
                self.session.add(row)
                await self.session.flush()
        except IntegrityError:
            existing = await self.session.scalar(
                select(SiteRow).where(
                    SiteRow.business_id == business_id,
                    SiteRow.external_ref == external_ref,
                )
                if external_ref is not None
                else select(SiteRow).where(SiteRow.business_id == business_id, SiteRow.id == row.id)
            )
            if existing is None:
                raise
            return _site(existing)
        return _site(row)


class SqlSitePlanSetRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_or_create(
        self,
        business_id: BusinessId,
        site_id: SiteId,
        plan_set_key: str,
        *,
        now: datetime,
    ) -> SitePlanSet:
        _aware(now, "now")
        site = await self.session.scalar(
            select(SiteRow).where(SiteRow.business_id == business_id, SiteRow.id == site_id)
        )
        if site is None:
            raise CrossBusinessPlanError("plan set references a site outside this business")
        plan_set_id = site_plan_set_id(business_id, site_id, plan_set_key)
        existing = await self.session.scalar(
            select(SitePlanSetRow).where(
                SitePlanSetRow.business_id == business_id,
                SitePlanSetRow.id == plan_set_id,
            )
        )
        if existing is not None:
            return _plan_set(existing)
        row = SitePlanSetRow(
            id=plan_set_id,
            business_id=business_id,
            site_id=site_id,
            plan_set_key=plan_set_key,
            created_at=now,
        )
        try:
            async with self.session.begin_nested():
                self.session.add(row)
                await self.session.flush()
        except IntegrityError:
            existing = await self.session.scalar(
                select(SitePlanSetRow).where(
                    SitePlanSetRow.business_id == business_id,
                    SitePlanSetRow.id == plan_set_id,
                )
            )
            if existing is None:
                raise
            return _plan_set(existing)
        return _plan_set(row)


class SqlSitePlanSetVersionRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get(self, business_id: BusinessId, version_id: UUID) -> SitePlanSetVersion | None:
        row = await self.session.scalar(
            select(SitePlanSetVersionRow).where(
                SitePlanSetVersionRow.business_id == business_id,
                SitePlanSetVersionRow.id == version_id,
            )
        )
        return _version(row) if row is not None else None

    async def list_for_plan_set(
        self, business_id: BusinessId, plan_set_id: SitePlanSetId
    ) -> tuple[SitePlanSetVersion, ...]:
        rows = await self.session.scalars(
            select(SitePlanSetVersionRow)
            .where(
                SitePlanSetVersionRow.business_id == business_id,
                SitePlanSetVersionRow.plan_set_id == plan_set_id,
            )
            .order_by(SitePlanSetVersionRow.version)
        )
        return tuple(_version(row) for row in rows)


class SqlPlanSetUploadRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def register(
        self,
        business_id: BusinessId,
        site_id: SiteId,
        plan_set_id: SitePlanSetId,
        source: AttachmentReference,
        *,
        now: datetime,
    ) -> PlanSetUploadRegistration:
        _aware(now, "now")
        plan_set = await self.session.scalar(
            select(SitePlanSetRow).where(
                SitePlanSetRow.business_id == business_id,
                SitePlanSetRow.id == plan_set_id,
                SitePlanSetRow.site_id == site_id,
            )
        )
        if plan_set is None:
            raise CrossBusinessPlanError("plan-set upload references another business")
        upload_id = plan_set_upload_id(business_id, plan_set_id, source)
        existing = await self.session.scalar(
            select(SitePlanSetUploadRow).where(
                SitePlanSetUploadRow.business_id == business_id,
                SitePlanSetUploadRow.id == upload_id,
            )
        )
        if existing is not None:
            return PlanSetUploadRegistration(upload=_upload(existing), created=False)
        row = SitePlanSetUploadRow(
            id=upload_id,
            business_id=business_id,
            site_id=site_id,
            plan_set_id=plan_set_id,
            source_attachment_id=source.attachment_id,
            source_media_kind=source.media_kind.value,
            source_locator=source.locator,
            source_mime_type=source.mime_type,
            source_filename=source.filename,
            source_byte_size=source.byte_size,
            status=PlanSetUploadStatus.PENDING.value,
            attempts=0,
            lease_token=uuid4(),
            created_at=now,
            updated_at=now,
        )
        try:
            async with self.session.begin_nested():
                self.session.add(row)
                await self.session.flush()
        except IntegrityError:
            existing = await self.session.scalar(
                select(SitePlanSetUploadRow).where(
                    SitePlanSetUploadRow.business_id == business_id,
                    SitePlanSetUploadRow.id == upload_id,
                )
            )
            if existing is None:
                raise
            return PlanSetUploadRegistration(upload=_upload(existing), created=False)
        return PlanSetUploadRegistration(upload=_upload(row), created=True)

    async def claim(
        self,
        business_id: BusinessId,
        upload_id: PlanSetUploadId,
        *,
        now: datetime,
        stale_before: datetime,
    ) -> PlanSetCopyClaim:
        _aware(now, "now")
        _aware(stale_before, "stale_before")
        row = await self.session.scalar(
            select(SitePlanSetUploadRow)
            .where(
                SitePlanSetUploadRow.business_id == business_id,
                SitePlanSetUploadRow.id == upload_id,
            )
            .with_for_update()
        )
        if row is None:
            return PlanSetCopyClaim(result=PlanSetCopyClaimResult.MISSING)
        if row.status == PlanSetUploadStatus.STORED.value and row.version_id is not None:
            stored = await self.session.scalar(
                select(SitePlanSetVersionRow).where(
                    SitePlanSetVersionRow.business_id == business_id,
                    SitePlanSetVersionRow.id == row.version_id,
                )
            )
            if stored is not None:
                return PlanSetCopyClaim(
                    result=PlanSetCopyClaimResult.TERMINAL,
                    upload_id=PlanSetUploadId(row.id),
                    business_id=BusinessId(row.business_id),
                    site_id=SiteId(row.site_id),
                    plan_set_id=SitePlanSetId(row.plan_set_id),
                    attempts=row.attempts,
                    stored_version=_version(stored),
                )
        leased_at = _with_utc(row.leased_at) if row.leased_at is not None else None
        if (
            row.status == PlanSetUploadStatus.COPYING.value
            and leased_at is not None
            and leased_at >= stale_before
        ):
            return PlanSetCopyClaim(
                result=PlanSetCopyClaimResult.BUSY,
                upload_id=PlanSetUploadId(row.id),
                business_id=BusinessId(row.business_id),
                site_id=SiteId(row.site_id),
                plan_set_id=SitePlanSetId(row.plan_set_id),
                attempts=row.attempts,
            )
        row.status = PlanSetUploadStatus.COPYING.value
        row.attempts += 1
        row.leased_at = now
        row.lease_token = uuid4()
        row.last_error = None
        row.updated_at = now
        return PlanSetCopyClaim(
            result=PlanSetCopyClaimResult.ACQUIRED,
            upload_id=PlanSetUploadId(row.id),
            business_id=BusinessId(row.business_id),
            site_id=SiteId(row.site_id),
            plan_set_id=SitePlanSetId(row.plan_set_id),
            source=_upload(row).source,
            attempts=row.attempts,
            lease_token=row.lease_token,
        )

    @staticmethod
    def _lease_filter(claim: PlanSetCopyClaim) -> tuple[ColumnElement[bool], ...]:
        if (
            claim.result is not PlanSetCopyClaimResult.ACQUIRED
            or claim.upload_id is None
            or claim.business_id is None
            or claim.lease_token is None
        ):
            raise LostPlanSetCopyLeaseError("plan-set copy claim does not hold an active lease")
        return (
            SitePlanSetUploadRow.business_id == claim.business_id,
            SitePlanSetUploadRow.id == claim.upload_id,
            SitePlanSetUploadRow.lease_token == claim.lease_token,
            SitePlanSetUploadRow.status == PlanSetUploadStatus.COPYING.value,
        )

    async def record_stored(
        self,
        claim: PlanSetCopyClaim,
        stored: StoredObject,
        *,
        uploaded_at: datetime,
    ) -> SitePlanSetVersion:
        _aware(uploaded_at, "uploaded_at")
        lease_filter = self._lease_filter(claim)
        business_id = claim.business_id
        plan_set_id = claim.plan_set_id
        site_id = claim.site_id
        if business_id is None or plan_set_id is None or site_id is None:
            raise LostPlanSetCopyLeaseError("plan-set copy claim is incomplete")
        plan_set = await self.session.scalar(
            select(SitePlanSetRow)
            .where(
                SitePlanSetRow.business_id == business_id,
                SitePlanSetRow.id == plan_set_id,
            )
            .with_for_update()
        )
        if plan_set is None:
            raise CrossBusinessPlanError("plan set is outside this business")
        existing = await self.session.scalar(
            select(SitePlanSetVersionRow).where(
                SitePlanSetVersionRow.business_id == business_id,
                SitePlanSetVersionRow.plan_set_id == plan_set_id,
                SitePlanSetVersionRow.content_digest == stored.content_digest,
            )
        )
        if existing is None:
            highest = await self.session.scalar(
                select(func.max(SitePlanSetVersionRow.version)).where(
                    SitePlanSetVersionRow.business_id == business_id,
                    SitePlanSetVersionRow.plan_set_id == plan_set_id,
                )
            )
            version = (highest or 0) + 1
            existing = SitePlanSetVersionRow(
                id=site_plan_set_version_id(plan_set_id, version, stored.content_digest),
                business_id=business_id,
                site_id=site_id,
                plan_set_id=plan_set_id,
                version=version,
                artifact_id=stored.artifact.attachment_id,
                media_kind=stored.artifact.media_kind.value,
                artifact_locator=stored.artifact.locator,
                mime_type=stored.artifact.mime_type,
                filename=stored.artifact.filename,
                byte_size=stored.byte_size,
                content_digest=stored.content_digest,
                page_count=None,
                uploaded_at=uploaded_at,
            )
            self.session.add(existing)
            await self.session.flush()
        result = await self.session.execute(
            update(SitePlanSetUploadRow)
            .where(*lease_filter)
            .values(
                status=PlanSetUploadStatus.STORED.value,
                version_id=existing.id,
                leased_at=None,
                last_error=None,
                updated_at=uploaded_at,
            )
        )
        if _rowcount(result) != 1:
            raise LostPlanSetCopyLeaseError("plan-set copy claim is no longer active")
        return _version(existing)

    async def record_failure(self, claim: PlanSetCopyClaim, error: str) -> None:
        result = await self.session.execute(
            update(SitePlanSetUploadRow)
            .where(*self._lease_filter(claim))
            .values(
                status=PlanSetUploadStatus.FAILED.value,
                leased_at=None,
                last_error=error,
                updated_at=datetime.now(UTC),
            )
        )
        if _rowcount(result) != 1:
            raise LostPlanSetCopyLeaseError("plan-set copy claim is no longer active")


class SqlPlanCustodyUnitOfWork:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory
        self._session: AsyncSession | None = None

    async def __aenter__(self) -> "SqlPlanCustodyUnitOfWork":
        session: AsyncSession = self._session_factory()
        self._session = session
        self.sites: SiteRepository = SqlSiteRepository(session)
        self.plan_sets: SitePlanSetRepository = SqlSitePlanSetRepository(session)
        self.plan_set_versions: SitePlanSetVersionRepository = SqlSitePlanSetVersionRepository(
            session
        )
        self.plan_set_uploads: PlanSetUploadRepository = SqlPlanSetUploadRepository(session)
        self.outbox: OutboxRepository = SqlOutboxRepository(session)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None:
        if self._session is None:
            return
        if exc_type is not None:
            await self._session.rollback()
        await self._session.close()

    async def commit(self) -> None:
        if self._session is None:
            raise RuntimeError("unit of work is not active")
        await self._session.commit()

    async def rollback(self) -> None:
        if self._session is None:
            raise RuntimeError("unit of work is not active")
        await self._session.rollback()


class SqlPlanCustodyUnitOfWorkFactory:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    def __call__(self) -> PlanCustodyUnitOfWork:
        return SqlPlanCustodyUnitOfWork(self._session_factory)
