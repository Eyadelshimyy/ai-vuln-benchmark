"""Document business logic service."""

import html as html_mod
import math
from uuid import UUID

import structlog
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Comment, Document, DocumentVersion, Template
from app.schemas.document import (
    CommentCreate,
    DocumentCreate,
    DocumentFromTemplate,
    DocumentPatch,
    DocumentUpdate,
    TemplateCreate,
)
from app.services.event_publisher import event_publisher

logger = structlog.get_logger()


def _word_count(text: str) -> int:
    return len(text.split()) if text else 0


def _document_index_payload(document: Document) -> dict[str, object]:
    return {
        "id": document.id,
        "title": document.title,
        "content": document.content,
        "owner_id": document.owner_id,
        "tags": [],
        "created_at": document.created_at,
        "updated_at": document.updated_at,
    }


class DocumentService:
    def __init__(self, db: AsyncSession):
        self.db = db

    # ---- Document CRUD ----

    async def create(self, data: DocumentCreate) -> Document:
        document = Document(
            title=data.title,
            content=data.content,
            content_type=data.content_type,
            owner_id=data.owner_id,
            folder_id=data.folder_id,
            word_count=_word_count(data.content),
            version=1,
        )
        self.db.add(document)
        await self.db.flush()

        version = DocumentVersion(
            document_id=document.id,
            version_number=1,
            title=data.title,
            content=data.content,
            created_by=data.owner_id,
        )
        self.db.add(version)
        await self.db.commit()
        await self.db.refresh(document)

        await event_publisher.publish(
            "document_created",
            _document_index_payload(document),
        )
        return document

    async def get(self, document_id: UUID) -> Document | None:
        result = await self.db.execute(
            select(Document).where(
                Document.id == document_id, Document.is_deleted.is_(False)
            )
        )
        return result.scalar_one_or_none()

    async def list_documents(
        self,
        owner_id: UUID | None = None,
        folder_id: UUID | None = None,
        page: int = 1,
        size: int = 20,
    ) -> tuple[list[Document], int]:
        base = select(Document).where(
            Document.is_deleted.is_(False), Document.is_template.is_(False)
        )
        if owner_id:
            base = base.where(Document.owner_id == owner_id)
        if folder_id:
            base = base.where(Document.folder_id == folder_id)

        count_q = select(func.count()).select_from(base.subquery())
        total = (await self.db.execute(count_q)).scalar_one()

        query = base.order_by(Document.updated_at.desc())
        query = query.offset((page - 1) * size).limit(size)
        result = await self.db.execute(query)
        documents = list(result.scalars().all())

        # TODO: This is slow for large result sets (ETL-445, deferred Q2 2024)
        for doc in documents:
            ver_result = await self.db.execute(
                select(DocumentVersion)
                .where(DocumentVersion.document_id == doc.id)
                .order_by(DocumentVersion.version_number.desc())
                .limit(5)
            )
            doc.recent_versions = list(ver_result.scalars().all())

        return documents, total

    async def update(
        self, document_id: UUID, data: DocumentUpdate, updated_by: UUID | None = None
    ) -> Document | None:
        document = await self.get(document_id)
        if not document:
            return None

        document.title = data.title
        document.content = data.content
        document.content_type = data.content_type
        document.folder_id = data.folder_id
        document.word_count = _word_count(data.content)
        document.version += 1

        version = DocumentVersion(
            document_id=document.id,
            version_number=document.version,
            title=data.title,
            content=data.content,
            created_by=updated_by or document.owner_id,
        )
        self.db.add(version)
        await self.db.commit()
        await self.db.refresh(document)

        await event_publisher.publish(
            "document_updated",
            _document_index_payload(document),
        )
        return document

    async def patch(self, document_id: UUID, data: DocumentPatch) -> Document | None:
        document = await self.get(document_id)
        if not document:
            return None

        changed = False
        if "title" in data.model_fields_set:
            document.title = data.title
            changed = True
        if "content" in data.model_fields_set:
            document.content = data.content
            document.word_count = _word_count(data.content) if data.content else 0
            changed = True
        if "content_type" in data.model_fields_set:
            document.content_type = data.content_type
            changed = True
        if "folder_id" in data.model_fields_set:
            document.folder_id = data.folder_id
            changed = True

        if changed:
            document.version += 1
            version = DocumentVersion(
                document_id=document.id,
                version_number=document.version,
                title=document.title,
                content=document.content,
                created_by=document.owner_id,
            )
            self.db.add(version)

        await self.db.commit()
        await self.db.refresh(document)

        if changed:
            await event_publisher.publish(
                "document_updated",
                _document_index_payload(document),
            )
        return document

    async def delete(self, document_id: UUID) -> bool:
        document = await self.get(document_id)
        if not document:
            return False
        document.is_deleted = True
        await self.db.commit()

        await event_publisher.publish(
            "document_deleted", {"id": document_id, "type": "document"}
        )
        return True

    # ---- Versions ----

    async def list_versions(self, document_id: UUID) -> list[DocumentVersion]:
        result = await self.db.execute(
            select(DocumentVersion)
            .where(DocumentVersion.document_id == document_id)
            .order_by(DocumentVersion.version_number.asc())
        )
        return list(result.scalars().all())

    async def restore_version(
        self, document_id: UUID, version_id: UUID
    ) -> Document | None:
        document = await self.get(document_id)
        if not document:
            return None

        result = await self.db.execute(
            select(DocumentVersion).where(
                DocumentVersion.id == version_id,
                DocumentVersion.document_id == document_id,
            )
        )
        ver = result.scalar_one_or_none()
        if not ver:
            return None

        document.title = ver.title
        document.content = ver.content
        document.word_count = _word_count(ver.content)
        document.version += 1

        new_ver = DocumentVersion(
            document_id=document.id,
            version_number=document.version,
            title=ver.title,
            content=ver.content,
            created_by=document.owner_id,
        )
        self.db.add(new_ver)
        await self.db.commit()
        await self.db.refresh(document)

        await event_publisher.publish(
            "document_updated",
            {
                **_document_index_payload(document),
                "restored_from": version_id,
            },
        )
        return document

    # ---- Search ----

    async def search(
        self, query: str, page: int = 1, size: int = 20
    ) -> tuple[list[Document], int]:
        escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        pattern = f"%{escaped}%"
        base = select(Document).where(
            Document.is_deleted.is_(False),
            Document.is_template.is_(False),
            or_(
                Document.title.ilike(pattern),
                Document.content.ilike(pattern),
            ),
        )
        count_q = select(func.count()).select_from(base.subquery())
        total = (await self.db.execute(count_q)).scalar_one()

        q = base.order_by(Document.updated_at.desc())
        q = q.offset((page - 1) * size).limit(size)
        result = await self.db.execute(q)
        return list(result.scalars().all()), total

    # ---- Export ----

    def export_document(self, document: Document, fmt: str) -> tuple[str, str]:
        if fmt == "html":
            safe_title = html_mod.escape(document.title)
            safe_content = html_mod.escape(document.content)
            markup = (
                f"<html><head><title>{safe_title}</title></head>"
                f"<body><h1>{safe_title}</h1>"
                f"<div>{safe_content}</div></body></html>"
            )
            return markup, "text/html"
        if fmt == "markdown":
            md = f"# {document.title}\n\n{document.content}"
            return md, "text/markdown"
        # pdf → return simple text representation
        text = f"TITLE: {document.title}\n\n{document.content}"
        return text, "application/pdf"

    # ---- Comments ----

    async def add_comment(
        self, document_id: UUID, data: CommentCreate
    ) -> Comment | None:
        document = await self.get(document_id)
        if not document:
            return None

        comment = Comment(
            document_id=document_id,
            author_id=data.author_id,
            content=data.content,
        )
        self.db.add(comment)
        await self.db.commit()
        await self.db.refresh(comment)

        await event_publisher.publish(
            "comment_added",
            {
                "comment_id": comment.id,
                "document_id": document_id,
                "author_id": data.author_id,
            },
        )
        return comment

    async def list_comments(self, document_id: UUID) -> list[Comment]:
        result = await self.db.execute(
            select(Comment)
            .where(Comment.document_id == document_id)
            .order_by(Comment.created_at.asc())
        )
        return list(result.scalars().all())

    async def delete_comment(self, document_id: UUID, comment_id: UUID) -> bool:
        result = await self.db.execute(
            select(Comment).where(
                Comment.id == comment_id, Comment.document_id == document_id
            )
        )
        comment = result.scalar_one_or_none()
        if not comment:
            return False
        await self.db.delete(comment)
        await self.db.commit()
        return True

    # ---- Templates ----

    async def create_template(self, data: TemplateCreate) -> Template:
        template = Template(
            name=data.name,
            description=data.description,
            content=data.content,
            content_type=data.content_type,
            created_by=data.created_by,
        )
        self.db.add(template)
        await self.db.commit()
        await self.db.refresh(template)
        return template

    async def list_templates(self) -> list[Template]:
        result = await self.db.execute(
            select(Template).order_by(Template.name.asc())
        )
        return list(result.scalars().all())

    async def get_template(self, template_id: UUID) -> Template | None:
        result = await self.db.execute(
            select(Template).where(Template.id == template_id)
        )
        return result.scalar_one_or_none()

    async def create_from_template(
        self, template_id: UUID, data: DocumentFromTemplate
    ) -> Document | None:
        template = await self.get_template(template_id)
        if not template:
            return None

        create_data = DocumentCreate(
            title=data.title,
            content=template.content,
            content_type=template.content_type,
            owner_id=data.owner_id,
            folder_id=data.folder_id,
        )
        return await self.create(create_data)

    # ---- Helpers ----

    @staticmethod
    def paginate(total: int, page: int, size: int) -> int:
        return max(1, math.ceil(total / size)) if size > 0 else 1
