"""S3 staging: round trip, byte uploads, missing objects, key layout and the runner's store selection (fake boto3)."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from ldm.config import load_manifest
from ldm.errors import ConfigError
from ldm.runner import make_blobs
from ldm.staging import AzureBlobStore, NoBlobStore, S3BlobStore, sha256_file

from .conftest import make_manifest_tree


class _ClientError(Exception):
    def __init__(self, code: str):
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


class _FakeS3:
    """Enough of boto3's S3 client for the store: one dict per bucket, keyed by object key."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.objects: dict[tuple[str, str], tuple[bytes, str | None]] = {}

    def upload_fileobj(self, f, bucket, key):
        self.objects[(bucket, key)] = (f.read(), None)

    def download_fileobj(self, bucket, key, f):
        if (bucket, key) not in self.objects:
            raise _ClientError("404")
        f.write(self.objects[(bucket, key)][0])

    def put_object(self, *, Bucket, Key, Body, ContentType):  # noqa: N803 - boto3 spelling
        self.objects[(Bucket, Key)] = (Body, ContentType)


@pytest.fixture
def fake_boto3(monkeypatch: pytest.MonkeyPatch) -> _FakeS3:
    holder: dict[str, _FakeS3] = {}

    def client(service: str, **kwargs):
        assert service == "s3"
        holder["c"] = _FakeS3(**kwargs)
        return holder["c"]

    boto3 = types.ModuleType("boto3")
    boto3.client = client  # type: ignore[attr-defined]
    botocore = types.ModuleType("botocore")
    exceptions = types.ModuleType("botocore.exceptions")
    exceptions.ClientError = _ClientError  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "boto3", boto3)
    monkeypatch.setitem(sys.modules, "botocore", botocore)
    monkeypatch.setitem(sys.modules, "botocore.exceptions", exceptions)
    return holder  # type: ignore[return-value]


def test_s3_round_trip_keeps_bytes_checksum_and_tenant_prefix(tmp_path: Path, fake_boto3) -> None:
    store = S3BlobStore("shared-staging", region="us-east-1", endpoint_url="http://localhost:9000")
    client = fake_boto3["c"]
    assert client.kwargs == {"region_name": "us-east-1", "endpoint_url": "http://localhost:9000"}
    src = tmp_path / "000001.dat"
    src.write_bytes(bytes(range(256)) * 10)
    key = "d24-after/run-1/unload/DOCARCH/000001.dat"
    store.upload(src, key)
    assert set(client.objects) == {("shared-staging", key)}
    dst = tmp_path / "back" / "000001.dat"
    assert store.download(key, dst) is True
    assert sha256_file(dst) == sha256_file(src)
    store.upload_bytes(b'{"ok":1}', "d24-after/run-1/manifest.json", "application/json")
    assert client.objects[("shared-staging", "d24-after/run-1/manifest.json")] == (b'{"ok":1}', "application/json")


def test_s3_missing_object_returns_false_and_leaves_no_partial_file(tmp_path: Path, fake_boto3) -> None:
    store = S3BlobStore("shared-staging")
    dst = tmp_path / "nested" / "missing.dat"
    assert store.download("d24-after/run-9/nothing.dat", dst) is False
    assert not dst.exists()
    fake_boto3["c"].download_fileobj = lambda *a: (_ for _ in ()).throw(_ClientError("AccessDenied"))
    with pytest.raises(_ClientError, match="AccessDenied"):
        store.download("d24-after/run-9/nothing.dat", dst)


def test_make_blobs_prefers_s3_then_no_blob_and_refuses_azure_when_disabled(tmp_path: Path, fake_boto3) -> None:
    loaded = load_manifest(make_manifest_tree(tmp_path, "s3a"), "s3a-after")
    ce = loaded.manifest.staging.connection_env
    assert ce.bucket and ce.region and ce.storage_account
    s3 = make_blobs(loaded, {ce.bucket: "shared-staging", ce.region: "us-east-1"})
    assert isinstance(s3, S3BlobStore) and s3.bucket == "shared-staging"
    assert isinstance(make_blobs(loaded, {}), NoBlobStore)
    with pytest.raises(ConfigError, match="azure: false"):
        make_blobs(loaded, {ce.storage_account: "acct"})


def test_make_blobs_azure_when_enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    loaded = load_manifest(make_manifest_tree(tmp_path, "s3b", azure_target=True), "s3b-after")
    ce = loaded.manifest.staging.connection_env
    assert loaded.manifest.azure and ce.storage_account and ce.container
    seen: dict[str, object] = {}
    monkeypatch.setattr(AzureBlobStore, "__init__", lambda self, *a: seen.update(args=a))
    make_blobs(loaded, {ce.storage_account: "acct", ce.container: "staging", "AZ_STORAGE_KEY": "k"})
    assert seen["args"] == ("acct", "staging", "k", None)
    with pytest.raises(ConfigError, match="is not"):
        make_blobs(loaded, {ce.storage_account: "acct"})
