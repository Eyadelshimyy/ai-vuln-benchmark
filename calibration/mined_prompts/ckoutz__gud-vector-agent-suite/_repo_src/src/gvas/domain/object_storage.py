"""Neutral object-custody contracts.

Domain and application code sees an opaque ``AttachmentReference`` plus the
content facts it is allowed to reason about — digest, media type and byte size.
Buckets, keys, endpoints, credentials and presigned URLs belong to the
infrastructure adapter that implements :class:`ObjectStoragePort`.
"""

import hashlib
import re

from pydantic import BaseModel, ConfigDict, Field, field_validator

from gvas.domain.enums import MediaKind
from gvas.domain.identifiers import BusinessId
from gvas.domain.messages import AttachmentReference

CONTENT_DIGEST_PATTERN = r"^[0-9a-f]{64}$"
CUSTODY_SCOPE_PATTERN = r"^[a-z0-9][a-z0-9-]*$"
CUSTODY_NAME_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._/-]*$"

_DIGEST = re.compile(CONTENT_DIGEST_PATTERN)


def content_digest(content: bytes) -> str:
    """The sha256 hex digest used as the content identity of a stored object."""

    return hashlib.sha256(content).hexdigest()


class ObjectStorageModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ObjectCustodyTarget(ObjectStorageModel):
    """Where, logically, one business's bytes live.

    ``scope`` and ``name`` are logical, tenant-relative and deterministic: the
    same content of the same plan set always yields the same target, so a
    replay overwrites nothing and creates no second object. How they become a
    physical location is the adapter's business. Every adapter entry point that
    can reach bytes takes one of these, so no caller can name a location
    outside its own tenant.
    """

    business_id: BusinessId
    scope: str = Field(min_length=1, max_length=64, pattern=CUSTODY_SCOPE_PATTERN)
    name: str = Field(min_length=1, max_length=512, pattern=CUSTODY_NAME_PATTERN)


class ObjectCustodyRequest(ObjectCustodyTarget):
    """A request to take custody of bytes on behalf of one business."""

    content: bytes = Field(min_length=1)
    media_kind: MediaKind
    media_type: str | None = None
    filename: str | None = None


class StoredObject(ObjectStorageModel):
    """What custody yields: an opaque reference plus verifiable content facts."""

    artifact: AttachmentReference
    content_digest: str = Field(min_length=64, max_length=64)
    byte_size: int = Field(ge=0)
    media_type: str | None = None

    @field_validator("content_digest")
    @classmethod
    def digest_is_sha256_hex(cls, value: str) -> str:
        if _DIGEST.fullmatch(value) is None:
            raise ValueError("content digest must be a sha256 hex digest")
        return value


class ObjectCustodyError(RuntimeError):
    """Raised when managed storage cannot take or return custody of bytes."""
