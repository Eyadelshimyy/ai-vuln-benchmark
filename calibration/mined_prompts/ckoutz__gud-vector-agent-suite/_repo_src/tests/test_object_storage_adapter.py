"""Adapter-level tests for managed object custody.

No live credentials: the R2 client is driven through ``botocore``'s stubber
with obviously fake settings, so requests are validated and never sent.
"""

from io import BytesIO
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

import pytest
from botocore.response import StreamingBody
from botocore.stub import Stubber

from gvas.config import ObjectStorageSettings
from gvas.domain.enums import MediaKind
from gvas.domain.identifiers import BusinessId
from gvas.domain.messages import AttachmentReference
from gvas.domain.object_storage import (
    ObjectCustodyError,
    ObjectCustodyRequest,
    ObjectCustodyTarget,
    content_digest,
)
from gvas.infrastructure.object_storage import (
    InMemoryObjectStorage,
    R2ObjectStorage,
    encode_locator,
)

CONTENT = b"%PDF-1.7 plan set bytes"
ACCOUNT_ID = "0123456789abcdef0123456789abcdef"
BUCKET = "gvas-plan-sets"
FAKE_ACCESS_KEY = "TESTACCESSKEYID"
FAKE_SIGNING_MATERIAL = "TESTSIGNINGMATERIAL"


@pytest.fixture
def settings() -> ObjectStorageSettings:
    return ObjectStorageSettings(
        account_id=ACCOUNT_ID,
        bucket=BUCKET,
        access_key_id=FAKE_ACCESS_KEY,
        secret_access_key=FAKE_SIGNING_MATERIAL,
    )


@pytest.fixture
def business_id() -> BusinessId:
    return BusinessId(uuid4())


def target(business_id: BusinessId, name: str = "plan-set/abc") -> ObjectCustodyTarget:
    return ObjectCustodyTarget(business_id=business_id, scope="plan-sets", name=name)


def request_for(
    business_id: BusinessId, media_type: str | None = "application/pdf"
) -> ObjectCustodyRequest:
    return ObjectCustodyRequest(
        business_id=business_id,
        scope="plan-sets",
        name="plan-set/abc",
        content=CONTENT,
        media_kind=MediaKind.DOCUMENT,
        media_type=media_type,
        filename="plans.pdf",
    )


def test_client_is_configured_for_r2(settings: ObjectStorageSettings) -> None:
    storage = R2ObjectStorage(settings)
    client = storage._client
    assert client.meta.endpoint_url == f"https://{ACCOUNT_ID}.r2.cloudflarestorage.com"
    assert client.meta.region_name == "auto"


def test_unconfigured_settings_are_refused() -> None:
    with pytest.raises(ObjectCustodyError):
        R2ObjectStorage(ObjectStorageSettings())


@pytest.mark.asyncio
async def test_put_stores_private_tenant_prefixed_object(
    settings: ObjectStorageSettings, business_id: BusinessId
) -> None:
    storage = R2ObjectStorage(settings)
    key = f"tenant/{business_id}/plan-sets/plan-set/abc"
    with Stubber(storage._client) as stubber:
        stubber.add_response(
            "put_object",
            {},
            {
                "Bucket": BUCKET,
                "Key": key,
                "Body": CONTENT,
                "ContentType": "application/pdf",
            },
        )
        stored = await storage.put(request_for(business_id))
        stubber.assert_no_pending_responses()

    assert stored.content_digest == content_digest(CONTENT)
    assert stored.byte_size == len(CONTENT)
    assert stored.artifact.locator == encode_locator("r2", key)
    assert "://" not in stored.artifact.locator


@pytest.mark.asyncio
async def test_put_omits_content_type_when_unknown(
    settings: ObjectStorageSettings, business_id: BusinessId
) -> None:
    storage = R2ObjectStorage(settings)
    with Stubber(storage._client) as stubber:
        stubber.add_response(
            "put_object",
            {},
            {
                "Bucket": BUCKET,
                "Key": f"tenant/{business_id}/plan-sets/plan-set/abc",
                "Body": CONTENT,
            },
        )
        await storage.put(request_for(business_id, media_type=None))
        stubber.assert_no_pending_responses()


@pytest.mark.asyncio
async def test_put_failures_surface_as_custody_errors(
    settings: ObjectStorageSettings, business_id: BusinessId
) -> None:
    storage = R2ObjectStorage(settings)
    with Stubber(storage._client) as stubber:
        stubber.add_client_error("put_object", service_error_code="AccessDenied")
        with pytest.raises(ObjectCustodyError):
            await storage.put(request_for(business_id))


@pytest.mark.asyncio
async def test_fetch_rejects_a_reference_from_another_business(
    settings: ObjectStorageSettings, business_id: BusinessId
) -> None:
    storage = R2ObjectStorage(settings)
    other = BusinessId(uuid4())
    foreign = AttachmentReference(
        attachment_id=uuid4(),
        media_kind=MediaKind.DOCUMENT,
        locator=encode_locator("r2", f"tenant/{other}/plan-sets/plan-set/abc"),
    )
    with Stubber(storage._client) as stubber:
        with pytest.raises(ObjectCustodyError):
            await storage.fetch(business_id, foreign)
        with pytest.raises(ObjectCustodyError):
            storage.presigned_get_url(business_id, foreign)
        stubber.assert_no_pending_responses()


@pytest.mark.asyncio
async def test_in_memory_fetch_rejects_a_reference_from_another_business(
    business_id: BusinessId,
) -> None:
    storage = InMemoryObjectStorage()
    stored = await storage.put(request_for(business_id))
    other = BusinessId(uuid4())

    assert (await storage.fetch(business_id, stored.artifact)).content == CONTENT
    with pytest.raises(ObjectCustodyError):
        await storage.fetch(other, stored.artifact)
    with pytest.raises(ObjectCustodyError):
        storage.presigned_get_url(other, stored.artifact)


@pytest.mark.asyncio
async def test_fetch_returns_the_stored_bytes(
    settings: ObjectStorageSettings, business_id: BusinessId
) -> None:
    storage = R2ObjectStorage(settings)
    key = f"tenant/{business_id}/plan-sets/plan-set/abc"
    artifact = AttachmentReference(
        attachment_id=uuid4(),
        media_kind=MediaKind.DOCUMENT,
        locator=encode_locator("r2", key),
        mime_type="application/pdf",
        filename="plans.pdf",
    )
    with Stubber(storage._client) as stubber:
        stubber.add_response(
            "get_object",
            {"Body": StreamingBody(BytesIO(CONTENT), len(CONTENT))},
            {"Bucket": BUCKET, "Key": key},
        )
        payload = await storage.fetch(business_id, artifact)
        stubber.assert_no_pending_responses()

    assert payload.content == CONTENT
    assert payload.mime_type == "application/pdf"


def test_presigned_urls_address_only_the_caller_tenant_prefix(
    settings: ObjectStorageSettings, business_id: BusinessId
) -> None:
    storage = R2ObjectStorage(settings)
    key = f"tenant/{business_id}/plan-sets/plan-set/abc"
    artifact = AttachmentReference(
        attachment_id=uuid4(),
        media_kind=MediaKind.DOCUMENT,
        locator=encode_locator("r2", key),
    )

    get_url = storage.presigned_get_url(business_id, artifact, expires_in=300)
    put_url = storage.presigned_put_url(target(business_id), expires_in=300)

    for url in (get_url, put_url):
        parsed = urlparse(url)
        query = parse_qs(parsed.query)
        assert parsed.netloc == f"{ACCOUNT_ID}.r2.cloudflarestorage.com"
        assert parsed.path == f"/{BUCKET}/{key}"
        assert query["X-Amz-Algorithm"] == ["AWS4-HMAC-SHA256"]
        assert query["X-Amz-Expires"] == ["300"]
        assert query["X-Amz-Signature"]
        assert "x-amz-acl" not in url.lower()
    assert get_url != put_url


def test_presigned_put_cannot_name_a_malformed_or_absolute_location(
    business_id: BusinessId,
) -> None:
    for name in ("../escape", "https://evil.invalid/x", "/absolute"):
        with pytest.raises(ValueError):
            ObjectCustodyTarget(business_id=business_id, scope="plan-sets", name=name)
    with pytest.raises(ValueError):
        ObjectCustodyTarget(business_id=business_id, scope="Plan Sets", name="ok")


def test_the_adapter_never_requests_a_public_acl() -> None:
    source = (Path(__file__).parents[1] / "src/gvas/infrastructure/object_storage.py").read_text()
    assert "ACL" not in source
    assert "public-read" not in source
