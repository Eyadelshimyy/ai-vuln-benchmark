from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from gvas.application.plan_custody import (
    CopyPlanSetIntoCustodyService,
    PlanSetCustodyOutcome,
    RegisterPlanSetUploadService,
)
from gvas.domain.enums import MediaKind, OutboxStatus
from gvas.domain.identifiers import BusinessId
from gvas.domain.messages import AttachmentPayload, AttachmentReference
from gvas.domain.object_storage import (
    ObjectCustodyError,
    ObjectCustodyRequest,
    StoredObject,
    content_digest,
)
from gvas.domain.plan_repositories import (
    CrossBusinessPlanError,
    LostPlanSetCopyLeaseError,
    PlanSetCopyClaimResult,
)
from gvas.domain.plans import (
    PLAN_SET_COPY_COMMAND_TYPE,
    PlanSetUploadStatus,
    SiteId,
    UnknownSiteError,
    plan_set_copy_command,
    plan_set_upload_id,
    site_plan_set_id,
)
from gvas.domain.ports import ObjectStoragePort
from gvas.infrastructure.models import Business, OutboxMessage
from gvas.infrastructure.object_storage import (
    InMemoryObjectStorage,
    decode_locator,
    tenant_object_key,
)
from gvas.infrastructure.plan_models import SitePlanSetUploadRow, SitePlanSetVersionRow
from gvas.infrastructure.plan_repositories import SqlPlanCustodyUnitOfWorkFactory

NOW = datetime(2025, 3, 1, 12, 0, tzinfo=UTC)
LEASE_TTL = timedelta(minutes=5)
PLAN_BYTES = b"%PDF-1.7 plan set bytes"


class StaticAttachments:
    """Stands in for the source channel's attachment access port."""

    def __init__(self, content: bytes = PLAN_BYTES) -> None:
        self.content = content
        self.calls = 0

    async def fetch(self, attachment: AttachmentReference) -> AttachmentPayload:
        self.calls += 1
        return AttachmentPayload(
            content=self.content, mime_type="application/pdf", filename="plans.pdf"
        )


class FailingAttachments:
    async def fetch(self, attachment: AttachmentReference) -> AttachmentPayload:
        raise RuntimeError("source channel is unavailable")


def source_reference(locator: str = "source-file-1") -> AttachmentReference:
    return AttachmentReference(
        attachment_id=uuid4(),
        media_kind=MediaKind.DOCUMENT,
        locator=locator,
        mime_type="application/pdf",
        filename="plans.pdf",
    )


def custody_request(business_id: BusinessId, name: str = "plans.pdf") -> ObjectCustodyRequest:
    return ObjectCustodyRequest(
        business_id=business_id,
        scope="plan-sets",
        name=name,
        content=PLAN_BYTES,
        media_kind=MediaKind.DOCUMENT,
        media_type="application/pdf",
        filename="plans.pdf",
    )


async def seed_business(
    session_factory: async_sessionmaker[AsyncSession], business_id: BusinessId
) -> None:
    async with session_factory() as session:
        session.add(
            Business(
                id=business_id,
                slug=f"business-{business_id}",
                name="Test Business",
                created_at=NOW,
                updated_at=NOW,
            )
        )
        await session.commit()


async def seed_site(
    session_factory: async_sessionmaker[AsyncSession], business_id: BusinessId
) -> SiteId:
    await seed_business(session_factory, business_id)
    factory = SqlPlanCustodyUnitOfWorkFactory(session_factory)
    async with factory() as unit_of_work:
        site = await unit_of_work.sites.get_or_create(
            business_id, label="12 Maple St", external_ref="job-1", now=NOW
        )
        await unit_of_work.commit()
    return site.site_id


def custody_service(
    session_factory: async_sessionmaker[AsyncSession],
    storage: ObjectStoragePort,
    attachments: StaticAttachments | FailingAttachments,
) -> CopyPlanSetIntoCustodyService:
    return CopyPlanSetIntoCustodyService(
        SqlPlanCustodyUnitOfWorkFactory(session_factory), attachments, storage
    )


@pytest.mark.asyncio
async def test_in_memory_adapter_satisfies_the_port_contract() -> None:
    storage: ObjectStoragePort = InMemoryObjectStorage()
    business_id = BusinessId(uuid4())
    stored = await storage.put(custody_request(business_id))
    assert isinstance(stored, StoredObject)
    assert stored.content_digest == content_digest(PLAN_BYTES)
    assert stored.byte_size == len(PLAN_BYTES)
    payload = await storage.fetch(business_id, stored.artifact)
    assert payload.content == PLAN_BYTES


@pytest.mark.asyncio
async def test_stored_reference_is_opaque_and_tenant_prefixed() -> None:
    storage = InMemoryObjectStorage()
    first = BusinessId(uuid4())
    second = BusinessId(uuid4())
    stored_first = await storage.put(custody_request(first))
    stored_second = await storage.put(custody_request(second))

    for stored in (stored_first, stored_second):
        assert "://" not in stored.artifact.locator
        assert "plan-sets" not in stored.artifact.locator
        assert "bucket" not in stored.artifact.locator.lower()

    assert storage.keys == tuple(
        sorted(
            (
                f"tenant/{first}/plan-sets/plans.pdf",
                f"tenant/{second}/plan-sets/plans.pdf",
            )
        )
    )
    assert decode_locator("mem", stored_first.artifact.locator).startswith(f"tenant/{first}/")
    assert tenant_object_key(custody_request(first)).startswith(f"tenant/{first}/")


@pytest.mark.asyncio
async def test_in_memory_adapter_rejects_conflicting_content_on_one_key() -> None:
    storage = InMemoryObjectStorage()
    business_id = BusinessId(uuid4())
    await storage.put(custody_request(business_id))
    with pytest.raises(ObjectCustodyError):
        await storage.put(
            ObjectCustodyRequest(
                business_id=business_id,
                scope="plan-sets",
                name="plans.pdf",
                content=b"other bytes",
                media_kind=MediaKind.DOCUMENT,
            )
        )


@pytest.mark.asyncio
async def test_registering_an_upload_twice_yields_one_upload_and_one_command(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    site_id = await seed_site(session_factory, business_id)
    service = RegisterPlanSetUploadService(SqlPlanCustodyUnitOfWorkFactory(session_factory))
    source = source_reference()

    first = await service.register(business_id, site_id, source, now=NOW)
    second = await service.register(business_id, site_id, source, now=NOW)

    assert first.created is True
    assert second.created is False
    assert first.upload.upload_id == second.upload.upload_id
    plan_set_id = site_plan_set_id(business_id, site_id, "default")
    assert first.upload.upload_id == plan_set_upload_id(business_id, plan_set_id, source)

    async with session_factory() as session:
        uploads = await session.scalar(select(func.count()).select_from(SitePlanSetUploadRow))
        commands = list(
            await session.scalars(
                select(OutboxMessage).where(
                    OutboxMessage.command_type == PLAN_SET_COPY_COMMAND_TYPE
                )
            )
        )
    assert uploads == 1
    assert len(commands) == 1
    assert commands[0].status == OutboxStatus.PENDING.value
    assert commands[0].id == plan_set_copy_command(business_id, first.upload.upload_id).command_id


@pytest.mark.asyncio
async def test_registering_an_upload_for_another_business_site_is_rejected(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    owner = BusinessId(uuid4())
    intruder = BusinessId(uuid4())
    site_id = await seed_site(session_factory, owner)
    await seed_business(session_factory, intruder)
    service = RegisterPlanSetUploadService(SqlPlanCustodyUnitOfWorkFactory(session_factory))

    with pytest.raises(UnknownSiteError):
        await service.register(intruder, site_id, source_reference(), now=NOW)


@pytest.mark.asyncio
async def test_copy_records_digest_byte_size_and_immutable_version(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    site_id = await seed_site(session_factory, business_id)
    registration = await RegisterPlanSetUploadService(
        SqlPlanCustodyUnitOfWorkFactory(session_factory)
    ).register(business_id, site_id, source_reference(), now=NOW)
    storage = InMemoryObjectStorage()
    attachments = StaticAttachments()

    report = await custody_service(session_factory, storage, attachments).copy(
        business_id, registration.upload.upload_id, now=NOW, stale_before=NOW - LEASE_TTL
    )

    assert report.outcome is PlanSetCustodyOutcome.STORED
    version = report.version
    assert version is not None
    assert version.version == 1
    assert version.content_digest == content_digest(PLAN_BYTES)
    assert version.byte_size == len(PLAN_BYTES)
    assert version.page_count is None
    assert "://" not in version.artifact.locator
    assert await storage.fetch(business_id, version.artifact) == AttachmentPayload(
        content=PLAN_BYTES, mime_type="application/pdf", filename="plans.pdf"
    )


@pytest.mark.asyncio
async def test_replaying_the_copy_creates_no_second_version_or_object(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    site_id = await seed_site(session_factory, business_id)
    uploads = RegisterPlanSetUploadService(SqlPlanCustodyUnitOfWorkFactory(session_factory))
    source = source_reference()
    registration = await uploads.register(business_id, site_id, source, now=NOW)
    storage = InMemoryObjectStorage()
    attachments = StaticAttachments()
    service = custody_service(session_factory, storage, attachments)

    first = await service.copy(
        business_id, registration.upload.upload_id, now=NOW, stale_before=NOW - LEASE_TTL
    )
    await uploads.register(business_id, site_id, source, now=NOW)
    second = await service.copy(
        business_id,
        registration.upload.upload_id,
        now=NOW + timedelta(hours=1),
        stale_before=NOW + timedelta(hours=1) - LEASE_TTL,
    )

    assert first.outcome is PlanSetCustodyOutcome.STORED
    assert second.outcome is PlanSetCustodyOutcome.ALREADY_STORED
    assert second.version == first.version
    assert len(storage.keys) == 1
    assert attachments.calls == 1
    async with session_factory() as session:
        versions = await session.scalar(select(func.count()).select_from(SitePlanSetVersionRow))
    assert versions == 1


@pytest.mark.asyncio
async def test_a_second_worker_cannot_double_copy_a_leased_upload(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    site_id = await seed_site(session_factory, business_id)
    registration = await RegisterPlanSetUploadService(
        SqlPlanCustodyUnitOfWorkFactory(session_factory)
    ).register(business_id, site_id, source_reference(), now=NOW)
    factory = SqlPlanCustodyUnitOfWorkFactory(session_factory)

    async with factory() as unit_of_work:
        first = await unit_of_work.plan_set_uploads.claim(
            business_id, registration.upload.upload_id, now=NOW, stale_before=NOW - LEASE_TTL
        )
        await unit_of_work.commit()
    async with factory() as unit_of_work:
        second = await unit_of_work.plan_set_uploads.claim(
            business_id,
            registration.upload.upload_id,
            now=NOW + timedelta(seconds=1),
            stale_before=NOW + timedelta(seconds=1) - LEASE_TTL,
        )
        await unit_of_work.commit()

    assert first.result is PlanSetCopyClaimResult.ACQUIRED
    assert second.result is PlanSetCopyClaimResult.BUSY
    assert second.lease_token is None


@pytest.mark.asyncio
async def test_a_stale_worker_cannot_complete_after_its_lease_was_taken(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    site_id = await seed_site(session_factory, business_id)
    registration = await RegisterPlanSetUploadService(
        SqlPlanCustodyUnitOfWorkFactory(session_factory)
    ).register(business_id, site_id, source_reference(), now=NOW)
    factory = SqlPlanCustodyUnitOfWorkFactory(session_factory)
    later = NOW + timedelta(hours=1)

    async with factory() as unit_of_work:
        stale = await unit_of_work.plan_set_uploads.claim(
            business_id, registration.upload.upload_id, now=NOW, stale_before=NOW - LEASE_TTL
        )
        await unit_of_work.commit()
    async with factory() as unit_of_work:
        fresh = await unit_of_work.plan_set_uploads.claim(
            business_id,
            registration.upload.upload_id,
            now=later,
            stale_before=later - LEASE_TTL,
        )
        await unit_of_work.commit()
    assert fresh.result is PlanSetCopyClaimResult.ACQUIRED

    stored = await InMemoryObjectStorage().put(custody_request(business_id))
    async with factory() as unit_of_work:
        await unit_of_work.plan_set_uploads.record_stored(fresh, stored, uploaded_at=later)
        await unit_of_work.commit()
    async with factory() as unit_of_work:
        with pytest.raises(LostPlanSetCopyLeaseError):
            await unit_of_work.plan_set_uploads.record_stored(stale, stored, uploaded_at=later)
        await unit_of_work.rollback()

    async with session_factory() as session:
        versions = await session.scalar(select(func.count()).select_from(SitePlanSetVersionRow))
    assert versions == 1


@pytest.mark.asyncio
async def test_a_failed_copy_is_recorded_and_stays_retryable(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    site_id = await seed_site(session_factory, business_id)
    registration = await RegisterPlanSetUploadService(
        SqlPlanCustodyUnitOfWorkFactory(session_factory)
    ).register(business_id, site_id, source_reference(), now=NOW)
    storage = InMemoryObjectStorage()

    report = await custody_service(session_factory, storage, FailingAttachments()).copy(
        business_id, registration.upload.upload_id, now=NOW, stale_before=NOW - LEASE_TTL
    )

    assert report.outcome is PlanSetCustodyOutcome.FAILED
    assert storage.keys == ()
    async with session_factory() as session:
        row = await session.scalar(select(SitePlanSetUploadRow))
    assert row is not None
    assert row.status == PlanSetUploadStatus.FAILED.value
    assert row.last_error is not None

    recovered = await custody_service(session_factory, storage, StaticAttachments()).copy(
        business_id, registration.upload.upload_id, now=NOW, stale_before=NOW - LEASE_TTL
    )
    assert recovered.outcome is PlanSetCustodyOutcome.STORED


@pytest.mark.asyncio
async def test_plan_reads_are_tenant_isolated(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    owner = BusinessId(uuid4())
    intruder = BusinessId(uuid4())
    site_id = await seed_site(session_factory, owner)
    await seed_business(session_factory, intruder)
    registration = await RegisterPlanSetUploadService(
        SqlPlanCustodyUnitOfWorkFactory(session_factory)
    ).register(owner, site_id, source_reference(), now=NOW)
    report = await custody_service(
        session_factory, InMemoryObjectStorage(), StaticAttachments()
    ).copy(owner, registration.upload.upload_id, now=NOW, stale_before=NOW - LEASE_TTL)
    version = report.version
    assert version is not None

    factory = SqlPlanCustodyUnitOfWorkFactory(session_factory)
    async with factory() as unit_of_work:
        assert await unit_of_work.sites.get(owner, site_id) is not None
        assert await unit_of_work.sites.get(intruder, site_id) is None
        assert (await unit_of_work.plan_set_versions.get(intruder, version.version_id)) is None
        assert await unit_of_work.plan_set_versions.get(owner, version.version_id) == version
        assert (
            await unit_of_work.plan_set_versions.list_for_plan_set(intruder, version.plan_set_id)
            == ()
        )
        assert await unit_of_work.plan_set_versions.list_for_plan_set(
            owner, version.plan_set_id
        ) == (version,)
        with pytest.raises(CrossBusinessPlanError):
            await unit_of_work.plan_sets.get_or_create(intruder, site_id, "default", now=NOW)


@pytest.mark.asyncio
async def test_copying_an_unknown_upload_is_missing(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    report = await custody_service(
        session_factory, InMemoryObjectStorage(), StaticAttachments()
    ).copy(
        business_id,
        plan_set_upload_id(
            business_id,
            site_plan_set_id(business_id, SiteId(uuid4()), "default"),
            source_reference(),
        ),
        now=NOW,
        stale_before=NOW - LEASE_TTL,
    )
    assert report.outcome is PlanSetCustodyOutcome.MISSING
