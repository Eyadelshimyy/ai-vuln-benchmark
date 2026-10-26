"""Tests for DocumentService business logic."""

import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.document import (
    CommentCreate,
    DocumentCreate,
    DocumentPatch,
    DocumentUpdate,
    TemplateCreate,
)
from app.services.document_service import DocumentService


@pytest.mark.asyncio
async def test_create_and_get(db_session: AsyncSession, owner_id: uuid.UUID):
    service = DocumentService(db_session)
    data = DocumentCreate(
        title="Service Test", content="Body text", owner_id=owner_id
    )
    doc = await service.create(data)
    assert doc.title == "Service Test"
    assert doc.word_count == 2
    assert doc.version == 1

    fetched = await service.get(doc.id)
    assert fetched is not None
    assert fetched.id == doc.id


@pytest.mark.asyncio
async def test_list_documents_with_filters(
    db_session: AsyncSession, owner_id: uuid.UUID, folder_id: uuid.UUID
):
    service = DocumentService(db_session)
    other_owner = uuid.uuid4()

    await service.create(
        DocumentCreate(title="A", content="", owner_id=owner_id, folder_id=folder_id)
    )
    await service.create(
        DocumentCreate(title="B", content="", owner_id=owner_id)
    )
    await service.create(
        DocumentCreate(title="C", content="", owner_id=other_owner)
    )

    items, total = await service.list_documents(owner_id=owner_id)
    assert total == 2

    items, total = await service.list_documents(folder_id=folder_id)
    assert total == 1


@pytest.mark.asyncio
async def test_update_creates_version(db_session: AsyncSession, owner_id: uuid.UUID):
    service = DocumentService(db_session)
    doc = await service.create(
        DocumentCreate(title="V1", content="First", owner_id=owner_id)
    )

    updated = await service.update(
        doc.id, DocumentUpdate(title="V2", content="Second")
    )
    assert updated is not None
    assert updated.version == 2
    assert updated.title == "V2"

    versions = await service.list_versions(doc.id)
    assert len(versions) == 2


@pytest.mark.asyncio
async def test_patch_partial_update(db_session: AsyncSession, owner_id: uuid.UUID):
    service = DocumentService(db_session)
    doc = await service.create(
        DocumentCreate(
            title="Original", content="Body", content_type="text/plain", owner_id=owner_id
        )
    )

    patched = await service.patch(doc.id, DocumentPatch(title="New Title"))
    assert patched is not None
    assert patched.title == "New Title"
    assert patched.content == "Body"
    assert patched.version == 2


@pytest.mark.asyncio
async def test_soft_delete(db_session: AsyncSession, owner_id: uuid.UUID):
    service = DocumentService(db_session)
    doc = await service.create(
        DocumentCreate(title="Delete Me", content="", owner_id=owner_id)
    )

    assert await service.delete(doc.id) is True
    assert await service.get(doc.id) is None


@pytest.mark.asyncio
async def test_delete_nonexistent(db_session: AsyncSession):
    service = DocumentService(db_session)
    assert await service.delete(uuid.uuid4()) is False


@pytest.mark.asyncio
async def test_restore_version(db_session: AsyncSession, owner_id: uuid.UUID):
    service = DocumentService(db_session)
    doc = await service.create(
        DocumentCreate(title="Orig Title", content="Orig Content", owner_id=owner_id)
    )
    versions = await service.list_versions(doc.id)
    v1_id = versions[0].id

    await service.update(doc.id, DocumentUpdate(title="Changed", content="New"))
    restored = await service.restore_version(doc.id, v1_id)
    assert restored is not None
    assert restored.title == "Orig Title"
    assert restored.content == "Orig Content"
    assert restored.version == 3


@pytest.mark.asyncio
async def test_search(db_session: AsyncSession, owner_id: uuid.UUID):
    service = DocumentService(db_session)
    await service.create(
        DocumentCreate(title="Python Guide", content="Learn Python", owner_id=owner_id)
    )
    await service.create(
        DocumentCreate(title="Rust Guide", content="Learn Rust", owner_id=owner_id)
    )

    items, total = await service.search("Python")
    assert total == 1
    assert items[0].title == "Python Guide"


@pytest.mark.asyncio
async def test_export_formats(db_session: AsyncSession, owner_id: uuid.UUID):
    service = DocumentService(db_session)
    doc = await service.create(
        DocumentCreate(title="Export Test", content="Some text", owner_id=owner_id)
    )

    html_body, html_ct = service.export_document(doc, "html")
    assert "<h1>Export Test</h1>" in html_body
    assert html_ct == "text/html"

    md_body, md_ct = service.export_document(doc, "markdown")
    assert "# Export Test" in md_body
    assert md_ct == "text/markdown"

    pdf_body, pdf_ct = service.export_document(doc, "pdf")
    assert "TITLE: Export Test" in pdf_body
    assert pdf_ct == "application/pdf"


@pytest.mark.asyncio
async def test_comments_crud(db_session: AsyncSession, owner_id: uuid.UUID):
    service = DocumentService(db_session)
    doc = await service.create(
        DocumentCreate(title="Commented", content="", owner_id=owner_id)
    )
    author = uuid.uuid4()

    comment = await service.add_comment(
        doc.id, CommentCreate(author_id=author, content="Nice!")
    )
    assert comment is not None
    assert comment.content == "Nice!"

    comments = await service.list_comments(doc.id)
    assert len(comments) == 1

    assert await service.delete_comment(doc.id, comment.id) is True
    assert len(await service.list_comments(doc.id)) == 0


@pytest.mark.asyncio
async def test_add_comment_to_nonexistent_document(db_session: AsyncSession):
    service = DocumentService(db_session)
    result = await service.add_comment(
        uuid.uuid4(), CommentCreate(author_id=uuid.uuid4(), content="Orphan")
    )
    assert result is None


@pytest.mark.asyncio
async def test_template_crud_and_create_from(
    db_session: AsyncSession, owner_id: uuid.UUID
):
    service = DocumentService(db_session)
    template = await service.create_template(
        TemplateCreate(
            name="Report",
            description="Monthly report",
            content="## Report\n\nContent here",
            created_by=uuid.uuid4(),
        )
    )
    assert template.name == "Report"

    templates = await service.list_templates()
    assert len(templates) == 1

    from app.schemas.document import DocumentFromTemplate

    doc = await service.create_from_template(
        template.id,
        DocumentFromTemplate(title="Jan Report", owner_id=owner_id),
    )
    assert doc is not None
    assert doc.title == "Jan Report"
    assert doc.content == "## Report\n\nContent here"


@pytest.mark.asyncio
async def test_create_from_nonexistent_template(
    db_session: AsyncSession, owner_id: uuid.UUID
):
    service = DocumentService(db_session)
    from app.schemas.document import DocumentFromTemplate

    result = await service.create_from_template(
        uuid.uuid4(),
        DocumentFromTemplate(title="Orphan", owner_id=owner_id),
    )
    assert result is None


@pytest.mark.asyncio
async def test_paginate_helper():
    assert DocumentService.paginate(10, 1, 5) == 2
    assert DocumentService.paginate(11, 1, 5) == 3
    assert DocumentService.paginate(0, 1, 5) == 1
    assert DocumentService.paginate(10, 1, 0) == 1
