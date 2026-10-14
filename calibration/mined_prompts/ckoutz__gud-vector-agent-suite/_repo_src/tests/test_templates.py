from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from gvas.application.completeness import CompletenessStatus, FieldNoteCompletenessService
from gvas.application.completeness_review import MarkerCompletenessReviewer
from gvas.application.report_generation import GenerateFieldNotesReportService
from gvas.application.templates import (
    IndustryTemplateDefinition,
    PublishTemplateSetService,
    TemplateResolver,
)
from gvas.composition.snapshots import BuildFieldNoteCaseSnapshotService
from gvas.domain.completeness import (
    ChecklistItem,
    ChecklistItemKey,
    ChecklistKey,
    FieldNoteReviewId,
)
from gvas.domain.field_notes import FieldNoteCaseId
from gvas.domain.identifiers import BusinessId, ConversationId, JsonValue, MessageId
from gvas.domain.reporting import (
    ChecklistEvidence,
    ChecklistEvidenceRequest,
    ReportGenerationRequest,
)
from gvas.domain.templates import (
    BusinessTemplateProfile,
    IndustryKey,
    ReportTemplateSection,
    ReportTemplateVersionConflictError,
    TemplateSetKey,
    TemplateSetRef,
    TemplateSetStatus,
    UnknownChecklistBindingError,
    UnknownReportTemplateError,
    UnknownTemplateSetError,
)
from gvas.infrastructure.completeness_models import FieldNoteChecklist, FieldNoteReview
from gvas.infrastructure.industry_seeds import (
    UnknownIndustrySeedError,
    load_industry_definition,
    load_industry_definitions,
)
from gvas.infrastructure.models import Business, Conversation, InboundMessage, OwnerChannelEndpoint
from gvas.infrastructure.reporting_unit_of_work import SqlReportUnitOfWorkFactory
from gvas.infrastructure.template_models import FieldNoteTemplateSet
from gvas.infrastructure.unit_of_work import SqlCompletenessUnitOfWorkFactory
from test_architecture_boundaries import find_violations

NOW = datetime(2025, 1, 1, tzinfo=UTC)
ENVIRONMENTAL = IndustryKey("environmental_testing")
LANDSCAPE = IndustryKey("landscape_construction")


class NoEvidence:
    async def attribute(self, request: ChecklistEvidenceRequest) -> tuple[ChecklistEvidence, ...]:
        return ()


class CapturingGenerator:
    """Records the section schema the generation port is handed for each report."""

    def __init__(self) -> None:
        self.requests: list[ReportGenerationRequest] = []

    async def generate(self, request: ReportGenerationRequest) -> dict[str, JsonValue]:
        self.requests.append(request)
        return {
            "schema_version": "field-notes-report/v1",
            "title": request.report_template.title,
            "sections": [
                {
                    "section_key": section.section_key,
                    "heading": section.heading,
                    "blocks": [{"kind": "text", "text": "Recorded."}],
                }
                for section in request.report_template.sections
            ],
        }


def definition(
    industry_key: IndustryKey,
    *,
    version: int = 1,
    report_template_version: int = 1,
    item_key: str = "site",
) -> IndustryTemplateDefinition:
    return IndustryTemplateDefinition(
        industry_key=industry_key,
        template_set_key=TemplateSetKey(f"{industry_key}_notes"),
        checklist_key=ChecklistKey(f"{industry_key}_notes"),
        version=version,
        items=(
            ChecklistItem(
                key=ChecklistItemKey(item_key),
                prompt=f"What about {item_key}?",
                evidence_markers=(f"{item_key}:",),
            ),
        ),
        report_template_key=f"{industry_key}_report",
        report_template_version=report_template_version,
        report_title=f"{industry_key} report",
        report_sections=(
            ReportTemplateSection(
                section_key=f"{item_key}_section",
                heading=f"{item_key} findings",
                checklist_item_keys=(ChecklistItemKey(item_key),),
            ),
        ),
    )


async def seed_business(
    session_factory: async_sessionmaker[AsyncSession], business_id: BusinessId
) -> tuple[ConversationId, MessageId]:
    endpoint_id = uuid4()
    conversation_id = ConversationId(uuid4())
    inbound_id = MessageId(uuid4())
    async with session_factory() as session:
        session.add(
            Business(
                id=business_id,
                slug=f"business-{business_id}",
                name="Business",
                created_at=NOW,
                updated_at=NOW,
            )
        )
        session.add(
            OwnerChannelEndpoint(
                id=endpoint_id,
                business_id=business_id,
                source_namespace="test",
                external_endpoint_id=f"endpoint-{business_id}",
                routing={},
            )
        )
        session.add(
            Conversation(
                id=conversation_id,
                business_id=business_id,
                endpoint_id=endpoint_id,
                external_conversation_id=f"conversation-{business_id}",
                routing={},
            )
        )
        session.add(
            InboundMessage(
                id=inbound_id,
                business_id=business_id,
                endpoint_id=endpoint_id,
                conversation_id=conversation_id,
                message_key="transcript",
                sender_external_id="owner",
                sender_role="owner",
                received_at=NOW,
                parts=[{"kind": "text", "text": "field notes"}],
                reply_to=None,
                routing={},
            )
        )
        await session.commit()
    return conversation_id, inbound_id


def publisher(session_factory: async_sessionmaker[AsyncSession]) -> PublishTemplateSetService:
    return PublishTemplateSetService(SqlCompletenessUnitOfWorkFactory(session_factory))


def resolver(session_factory: async_sessionmaker[AsyncSession]) -> TemplateResolver:
    return TemplateResolver(SqlCompletenessUnitOfWorkFactory(session_factory))


def completeness(session_factory: async_sessionmaker[AsyncSession]) -> FieldNoteCompletenessService:
    return FieldNoteCompletenessService(
        SqlCompletenessUnitOfWorkFactory(session_factory),
        MarkerCompletenessReviewer(),
        resolver(session_factory),
    )


async def generate_report(
    session_factory: async_sessionmaker[AsyncSession],
    generator: CapturingGenerator,
    business_id: BusinessId,
    review_id: FieldNoteReviewId,
) -> None:
    snapshot = await BuildFieldNoteCaseSnapshotService(
        SqlCompletenessUnitOfWorkFactory(session_factory),
        NoEvidence(),
        resolver(session_factory),
    ).build(business_id, FieldNoteCaseId(uuid4()), review_id, completed_at=NOW)
    await GenerateFieldNotesReportService(
        SqlReportUnitOfWorkFactory(session_factory),
        generator,
        resolver(session_factory),
    ).generate(snapshot, now=NOW, stale_before=NOW)


async def review_row(session_factory: async_sessionmaker[AsyncSession]) -> FieldNoteReview:
    async with session_factory() as session:
        row = await session.scalar(select(FieldNoteReview))
    assert row is not None
    return row


@pytest.mark.asyncio
async def test_seeding_twice_is_a_no_op(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    first = await publisher(session_factory).seed_industry(business_id, definition(ENVIRONMENTAL))
    second = await publisher(session_factory).seed_industry(business_id, definition(ENVIRONMENTAL))

    assert first == second
    async with session_factory() as session:
        assert await session.scalar(select(func.count(FieldNoteTemplateSet.id))) == 1
        assert await session.scalar(select(func.count(FieldNoteChecklist.id))) == 1


@pytest.mark.asyncio
async def test_businesses_in_different_industries_resolve_different_checklists(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    first = BusinessId(uuid4())
    second = BusinessId(uuid4())
    first_conversation, _ = await seed_business(session_factory, first)
    second_conversation, _ = await seed_business(session_factory, second)
    await publisher(session_factory).seed_industry(first, definition(ENVIRONMENTAL))
    await publisher(session_factory).seed_industry(second, definition(LANDSCAPE, item_key="crew"))

    resolution = resolver(session_factory)
    first_set = await resolution.load(
        await resolution.resolve_for_new_case(first, first_conversation)
    )
    second_set = await resolution.load(
        await resolution.resolve_for_new_case(second, second_conversation)
    )

    assert first_set.industry_key == ENVIRONMENTAL
    assert second_set.industry_key == LANDSCAPE
    assert first_set.checklist_key != second_set.checklist_key


@pytest.mark.asyncio
async def test_business_without_custom_set_falls_back_to_industry_default(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    conversation_id, _ = await seed_business(session_factory, business_id)
    seeded = definition(ENVIRONMENTAL)
    await publisher(session_factory).seed_industry(business_id, seeded)
    unit_of_work_factory = SqlCompletenessUnitOfWorkFactory(session_factory)
    async with unit_of_work_factory() as unit_of_work:
        await unit_of_work.business_template_profiles.upsert(
            BusinessTemplateProfile(
                business_id=business_id,
                industry_key=ENVIRONMENTAL,
                template_set_key=TemplateSetKey("custom_unpublished"),
                default_template_set_key=seeded.template_set_key,
            )
        )
        await unit_of_work.commit()

    resolved = await resolver(session_factory).resolve_for_new_case(business_id, conversation_id)
    assert resolved.template_set_key == seeded.template_set_key


@pytest.mark.asyncio
async def test_resolution_is_tenant_scoped(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    owner = BusinessId(uuid4())
    intruder = BusinessId(uuid4())
    await seed_business(session_factory, owner)
    intruder_conversation, _ = await seed_business(session_factory, intruder)
    seeded = await publisher(session_factory).seed_industry(owner, definition(ENVIRONMENTAL))

    resolution = resolver(session_factory)
    with pytest.raises(UnknownTemplateSetError):
        await resolution.resolve_for_new_case(intruder, intruder_conversation)
    with pytest.raises(UnknownTemplateSetError):
        await resolution.load(
            TemplateSetRef(
                business_id=intruder,
                template_set_key=seeded.template_set_key,
                version=seeded.version,
            )
        )


@pytest.mark.asyncio
async def test_publishing_a_new_version_does_not_change_an_in_flight_case(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    conversation_id, inbound_id = await seed_business(session_factory, business_id)
    await publisher(session_factory).seed_industry(business_id, definition(ENVIRONMENTAL))
    service = completeness(session_factory)
    started = await service.start_review(
        business_id,
        conversation_id,
        f"conversation-{business_id}",
        inbound_id,
        "no markers here",
    )
    assert started.status is CompletenessStatus.QUESTIONS_SENT

    await publisher(session_factory).seed_industry(
        business_id, definition(ENVIRONMENTAL, version=2, item_key="site")
    )
    await service.start_review(
        business_id,
        conversation_id,
        f"conversation-{business_id}",
        inbound_id,
        "no markers here",
    )

    row = await review_row(session_factory)
    assert (row.template_set_version, row.checklist_version) == (1, 1)
    active = await resolver(session_factory).resolve_for_new_case(business_id, conversation_id)
    assert active.version == 2


@pytest.mark.asyncio
async def test_completed_case_snapshot_uses_its_original_template_set(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    conversation_id, inbound_id = await seed_business(session_factory, business_id)
    await publisher(session_factory).seed_industry(business_id, definition(ENVIRONMENTAL))
    completed = await completeness(session_factory).start_review(
        business_id,
        conversation_id,
        f"conversation-{business_id}",
        inbound_id,
        "site: north tower",
    )
    assert completed.status is CompletenessStatus.COMPLETE
    assert completed.review_id is not None
    await publisher(session_factory).seed_industry(
        business_id, definition(ENVIRONMENTAL, version=2, report_template_version=7)
    )

    snapshot = await BuildFieldNoteCaseSnapshotService(
        SqlCompletenessUnitOfWorkFactory(session_factory),
        NoEvidence(),
        resolver(session_factory),
    ).build(
        business_id,
        FieldNoteCaseId(uuid4()),
        completed.review_id,
        completed_at=NOW,
    )

    assert snapshot.report_template_key == f"{ENVIRONMENTAL}_report"
    assert snapshot.report_template_version == 1


@pytest.mark.asyncio
async def test_publishing_a_new_version_deprecates_the_previous_one(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    first = await publisher(session_factory).seed_industry(business_id, definition(ENVIRONMENTAL))
    await publisher(session_factory).seed_industry(
        business_id, definition(ENVIRONMENTAL, version=2)
    )

    previous = await resolver(session_factory).load(first)
    assert previous.status is TemplateSetStatus.DEPRECATED


@pytest.mark.asyncio
async def test_later_revision_of_an_open_case_keeps_the_first_reviews_pins(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    conversation_id, inbound_id = await seed_business(session_factory, business_id)
    await publisher(session_factory).seed_industry(business_id, definition(ENVIRONMENTAL))
    service = completeness(session_factory)
    completed = await service.start_review(
        business_id,
        conversation_id,
        f"conversation-{business_id}",
        inbound_id,
        "site: north tower",
    )
    assert completed.status is CompletenessStatus.COMPLETE

    await publisher(session_factory).seed_industry(
        business_id,
        definition(ENVIRONMENTAL, version=2, report_template_version=2),
    )
    revised = await service.start_review(
        business_id,
        conversation_id,
        f"conversation-{business_id}",
        inbound_id,
        "site: north tower, second visit",
    )
    assert revised.review_id is not None
    assert revised.review_id != completed.review_id

    async with session_factory() as session:
        rows = (
            await session.scalars(select(FieldNoteReview).order_by(FieldNoteReview.revision))
        ).all()
    assert [row.revision for row in rows] == [1, 2]
    assert {(row.template_set_version, row.checklist_version) for row in rows} == {(1, 1)}


@pytest.mark.asyncio
async def test_open_case_continues_when_no_active_template_remains(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    conversation_id, inbound_id = await seed_business(session_factory, business_id)
    seeded = await publisher(session_factory).seed_industry(business_id, definition(ENVIRONMENTAL))
    service = completeness(session_factory)
    await service.start_review(
        business_id,
        conversation_id,
        f"conversation-{business_id}",
        inbound_id,
        "site: north tower",
    )
    unit_of_work_factory = SqlCompletenessUnitOfWorkFactory(session_factory)
    async with unit_of_work_factory() as unit_of_work:
        await unit_of_work.template_sets.set_status(seeded, TemplateSetStatus.DEPRECATED)
        await unit_of_work.commit()

    with pytest.raises(UnknownTemplateSetError):
        await resolver(session_factory).resolve_for_new_case(business_id, conversation_id)
    revised = await service.start_review(
        business_id,
        conversation_id,
        f"conversation-{business_id}",
        inbound_id,
        "site: north tower, revisited",
    )

    assert revised.review_id is not None
    async with session_factory() as session:
        rows = (await session.scalars(select(FieldNoteReview))).all()
    assert {(row.template_set_version, row.checklist_version) for row in rows} == {(1, 1)}


@pytest.mark.asyncio
async def test_industries_deliver_their_own_report_sections_to_the_generator(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    generator = CapturingGenerator()
    sections: dict[IndustryKey, tuple[str, ...]] = {}
    for industry_key, item_key in ((ENVIRONMENTAL, "site"), (LANDSCAPE, "crew")):
        business_id = BusinessId(uuid4())
        conversation_id, inbound_id = await seed_business(session_factory, business_id)
        await publisher(session_factory).seed_industry(
            business_id, definition(industry_key, item_key=item_key)
        )
        completed = await completeness(session_factory).start_review(
            business_id,
            conversation_id,
            f"conversation-{business_id}",
            inbound_id,
            f"{item_key}: recorded",
        )
        assert completed.review_id is not None
        await generate_report(session_factory, generator, business_id, completed.review_id)
        request = generator.requests[-1]
        sections[industry_key] = tuple(
            section.heading for section in request.report_template.sections
        )

    assert sections[ENVIRONMENTAL] != sections[LANDSCAPE]
    assert sections[ENVIRONMENTAL] == ("site findings",)
    assert sections[LANDSCAPE] == ("crew findings",)


@pytest.mark.asyncio
async def test_report_generation_uses_the_pinned_schema_after_a_newer_one_is_published(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    conversation_id, inbound_id = await seed_business(session_factory, business_id)
    await publisher(session_factory).seed_industry(business_id, definition(ENVIRONMENTAL))
    completed = await completeness(session_factory).start_review(
        business_id,
        conversation_id,
        f"conversation-{business_id}",
        inbound_id,
        "site: north tower",
    )
    assert completed.review_id is not None
    republished = definition(ENVIRONMENTAL, version=2, report_template_version=2).model_copy(
        update={
            "report_sections": (
                ReportTemplateSection(
                    section_key="rewritten",
                    heading="Rewritten structure",
                    checklist_item_keys=(ChecklistItemKey("site"),),
                ),
            )
        }
    )
    await publisher(session_factory).seed_industry(business_id, republished)

    generator = CapturingGenerator()
    await generate_report(session_factory, generator, business_id, completed.review_id)

    request = generator.requests[-1]
    assert request.report_template.version == 1
    assert tuple(section.section_key for section in request.report_template.sections) == (
        "site_section",
    )


@pytest.mark.asyncio
async def test_report_template_definitions_are_immutable_and_tenant_scoped(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    owner = BusinessId(uuid4())
    intruder = BusinessId(uuid4())
    await seed_business(session_factory, owner)
    await seed_business(session_factory, intruder)
    seeded = definition(ENVIRONMENTAL)
    await publisher(session_factory).seed_industry(owner, seeded)
    pinned = seeded.report_template(owner).ref

    resolution = resolver(session_factory)
    assert (await resolution.load_report_template(pinned)).sections == seeded.report_sections
    with pytest.raises(UnknownReportTemplateError):
        await resolution.load_report_template(pinned.model_copy(update={"business_id": intruder}))

    unit_of_work_factory = SqlCompletenessUnitOfWorkFactory(session_factory)
    async with unit_of_work_factory() as unit_of_work:
        with pytest.raises(ReportTemplateVersionConflictError):
            await unit_of_work.report_templates.upsert(
                seeded.report_template(owner).model_copy(update={"title": "rewritten"})
            )
        await unit_of_work.rollback()


@pytest.mark.asyncio
async def test_publishing_validates_bindings_against_the_versions_it_pins(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    conversation_id, _ = await seed_business(session_factory, business_id)
    seeded = definition(ENVIRONMENTAL)
    await publisher(session_factory).seed_industry(business_id, seeded)
    renamed_items = definition(ENVIRONMENTAL, version=2, item_key="water")

    with pytest.raises(UnknownChecklistBindingError):
        await publisher(session_factory).publish(
            seeded.template_set(business_id).model_copy(
                update={"version": 2, "checklist_version": 2}
            ),
            renamed_items.checklist(business_id),
        )

    active = await resolver(session_factory).resolve_for_new_case(business_id, conversation_id)
    assert active == seeded.template_set(business_id).ref


def test_report_sections_must_bind_known_checklist_items() -> None:
    base = definition(ENVIRONMENTAL)
    unbound = (
        ReportTemplateSection(
            section_key="unbound",
            heading="Unbound",
            checklist_item_keys=(ChecklistItemKey("not_a_checklist_item"),),
        ),
    )
    with pytest.raises(ValidationError):
        IndustryTemplateDefinition(
            **base.model_dump(exclude={"report_sections"}),
            report_sections=unbound,
        )
    with pytest.raises(UnknownChecklistBindingError):
        base.report_template(BusinessId(uuid4())).model_copy(
            update={"sections": unbound}
        ).validate_against(base.checklist(BusinessId(uuid4())))


def test_checked_in_industry_seeds_cover_more_than_inspection_work() -> None:
    definitions = load_industry_definitions()
    assert {ENVIRONMENTAL, LANDSCAPE} <= set(definitions)
    assert definitions[ENVIRONMENTAL].checklist_key != definitions[LANDSCAPE].checklist_key
    assert load_industry_definition(ENVIRONMENTAL) == definitions[ENVIRONMENTAL]
    with pytest.raises(UnknownIndustrySeedError):
        load_industry_definition(IndustryKey("does_not_exist"))


def test_template_modules_respect_layer_boundaries() -> None:
    root = Path(__file__).parents[1] / "src" / "gvas"
    for layer, names in {
        "domain": ("templates.py", "template_repositories.py"),
        "application": ("templates.py",),
    }.items():
        for name in names:
            path = root / layer / name
            assert not find_violations(layer, path, path.read_text())
