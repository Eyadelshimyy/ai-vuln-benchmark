"""Managed object custody adapters (design §5, D7b).

Only this module knows buckets, keys, endpoints, credentials and presigned
URLs. Domain and application code receives opaque ``AttachmentReference``
locators, which are reversible here and nowhere else.
"""

import asyncio
from base64 import urlsafe_b64decode, urlsafe_b64encode
from typing import TYPE_CHECKING
from uuid import UUID, uuid5

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from gvas.config import ObjectStorageSettings
from gvas.domain.identifiers import BusinessId
from gvas.domain.messages import AttachmentPayload, AttachmentReference
from gvas.domain.object_storage import (
    ObjectCustodyError,
    ObjectCustodyRequest,
    ObjectCustodyTarget,
    StoredObject,
    content_digest,
)

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client

OBJECT_LOCATOR_NAMESPACE = UUID("6b9c2f74-1d38-5a41-9e07-5f8c3b2a4d16")
TENANT_KEY_PREFIX = "tenant"


def tenant_key_prefix(business_id: BusinessId) -> str:
    return f"{TENANT_KEY_PREFIX}/{business_id}/"


def tenant_object_key(target: ObjectCustodyTarget) -> str:
    """Every object lives under its tenant's prefix (accepted owner decision)."""

    return f"{tenant_key_prefix(target.business_id)}{target.scope}/{target.name}"


def encode_locator(scheme: str, key: str) -> str:
    token = urlsafe_b64encode(key.encode()).decode().rstrip("=")
    return f"{scheme}:{token}"


def decode_locator(scheme: str, locator: str) -> str:
    prefix = f"{scheme}:"
    if not locator.startswith(prefix):
        raise ObjectCustodyError("attachment locator was not issued by this object store")
    token = locator[len(prefix) :]
    padding = "=" * (-len(token) % 4)
    try:
        return urlsafe_b64decode(token + padding).decode()
    except (ValueError, UnicodeDecodeError) as error:
        raise ObjectCustodyError("attachment locator is malformed") from error


def tenant_key_of(scheme: str, business_id: BusinessId, artifact: AttachmentReference) -> str:
    """Reverse a locator, refusing any key outside the caller's tenant prefix.

    Locators are opaque but not secret, so tenant isolation cannot rest on the
    caller merely not knowing another business's reference.
    """

    key = decode_locator(scheme, artifact.locator)
    if not key.startswith(tenant_key_prefix(business_id)):
        raise ObjectCustodyError("object belongs to another business")
    return key


def _artifact(
    scheme: str, key: str, request: ObjectCustodyRequest, byte_size: int
) -> AttachmentReference:
    return AttachmentReference(
        attachment_id=uuid5(OBJECT_LOCATOR_NAMESPACE, key),
        media_kind=request.media_kind,
        locator=encode_locator(scheme, key),
        mime_type=request.media_type,
        filename=request.filename,
        byte_size=byte_size,
    )


class InMemoryObjectStorage:
    """Deterministic stand-in for R2 used by tests and local runs."""

    SCHEME = "mem"

    def __init__(self) -> None:
        self._objects: dict[str, bytes] = {}

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._objects))

    async def put(self, request: ObjectCustodyRequest) -> StoredObject:
        key = tenant_object_key(request)
        existing = self._objects.get(key)
        if existing is not None and existing != request.content:
            raise ObjectCustodyError("object key already holds different content")
        self._objects[key] = request.content
        return StoredObject(
            artifact=_artifact(self.SCHEME, key, request, len(request.content)),
            content_digest=content_digest(request.content),
            byte_size=len(request.content),
            media_type=request.media_type,
        )

    async def fetch(
        self, business_id: BusinessId, artifact: AttachmentReference
    ) -> AttachmentPayload:
        key = tenant_key_of(self.SCHEME, business_id, artifact)
        content = self._objects.get(key)
        if content is None:
            raise ObjectCustodyError("object is not in custody")
        return AttachmentPayload(
            content=content, mime_type=artifact.mime_type, filename=artifact.filename
        )

    def presigned_get_url(
        self, business_id: BusinessId, artifact: AttachmentReference, *, expires_in: int = 900
    ) -> str:
        key = tenant_key_of(self.SCHEME, business_id, artifact)
        return f"https://in-memory.invalid/{key}?expires_in={expires_in}"

    def presigned_put_url(self, target: ObjectCustodyTarget, *, expires_in: int = 900) -> str:
        return f"https://in-memory.invalid/{tenant_object_key(target)}?expires_in={expires_in}"


class R2ObjectStorage:
    """Cloudflare R2 over its S3-compatible API.

    boto3 is used rather than hand-rolled SigV4 because R2's S3 surface is the
    contract we depend on, including presigned URLs; blocking calls are pushed
    to a worker thread so the async call sites stay non-blocking.
    """

    SCHEME = "r2"

    def __init__(self, settings: ObjectStorageSettings) -> None:
        if not settings.is_configured:
            raise ObjectCustodyError("object storage is not configured")
        self._settings = settings
        self._client: S3Client = boto3.client(
            "s3",
            endpoint_url=settings.endpoint_url,
            aws_access_key_id=settings.access_key_id,
            aws_secret_access_key=settings.secret_access_key,
            region_name=settings.region,
            config=Config(
                signature_version="s3v4",
                retries={"max_attempts": 3, "mode": "standard"},
            ),
        )

    async def put(self, request: ObjectCustodyRequest) -> StoredObject:
        key = tenant_object_key(request)
        digest = content_digest(request.content)
        try:
            await asyncio.to_thread(self._put_object, key, request)
        except (BotoCoreError, ClientError) as error:
            raise ObjectCustodyError("object store rejected the upload") from error
        return StoredObject(
            artifact=_artifact(self.SCHEME, key, request, len(request.content)),
            content_digest=digest,
            byte_size=len(request.content),
            media_type=request.media_type,
        )

    def _put_object(self, key: str, request: ObjectCustodyRequest) -> None:
        if request.media_type is None:
            self._client.put_object(Bucket=self._settings.bucket, Key=key, Body=request.content)
            return
        self._client.put_object(
            Bucket=self._settings.bucket,
            Key=key,
            Body=request.content,
            ContentType=request.media_type,
        )

    async def fetch(
        self, business_id: BusinessId, artifact: AttachmentReference
    ) -> AttachmentPayload:
        key = tenant_key_of(self.SCHEME, business_id, artifact)
        try:
            content = await asyncio.to_thread(self._get_object, key)
        except (BotoCoreError, ClientError) as error:
            raise ObjectCustodyError("object is not retrievable from custody") from error
        return AttachmentPayload(
            content=content, mime_type=artifact.mime_type, filename=artifact.filename
        )

    def _get_object(self, key: str) -> bytes:
        response = self._client.get_object(Bucket=self._settings.bucket, Key=key)
        body: bytes = response["Body"].read()
        return body

    def presigned_get_url(
        self, business_id: BusinessId, artifact: AttachmentReference, *, expires_in: int = 900
    ) -> str:
        return self._presign(
            "get_object", tenant_key_of(self.SCHEME, business_id, artifact), expires_in
        )

    def presigned_put_url(self, target: ObjectCustodyTarget, *, expires_in: int = 900) -> str:
        return self._presign("put_object", tenant_object_key(target), expires_in)

    def _presign(self, operation: str, key: str, expires_in: int) -> str:
        url: str = self._client.generate_presigned_url(
            operation,
            Params={"Bucket": self._settings.bucket, "Key": key},
            ExpiresIn=expires_in,
        )
        return url
