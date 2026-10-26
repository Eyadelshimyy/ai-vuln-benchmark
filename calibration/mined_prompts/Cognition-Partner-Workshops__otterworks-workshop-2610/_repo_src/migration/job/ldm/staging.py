"""Staging: local fixed-width files plus an optional S3 bucket or Azure Blob container (CONTRACTS.md §6.4)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol

from .errors import ConfigError


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class BlobStore(Protocol):
    enabled: bool

    def upload(self, local: Path, blob_path: str) -> None: ...
    def download(self, blob_path: str, local: Path) -> bool:
        """Return False when the blob does not exist."""
        ...

    def upload_bytes(self, data: bytes, blob_path: str, content_type: str) -> None: ...


class NoBlobStore:
    """No container configured: files stay on the local staging volume."""

    enabled = False

    def upload(self, local: Path, blob_path: str) -> None:
        return None

    def download(self, blob_path: str, local: Path) -> bool:
        return False

    def upload_bytes(self, data: bytes, blob_path: str, content_type: str) -> None:
        return None


class DirectoryBlobStore:
    """A directory standing in for the container (tests / local dry runs)."""

    enabled = True

    def __init__(self, root: Path):
        self.root = root

    def upload(self, local: Path, blob_path: str) -> None:
        dest = self.root / blob_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(local.read_bytes())

    def download(self, blob_path: str, local: Path) -> bool:
        src = self.root / blob_path
        if not src.exists():
            return False
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(src.read_bytes())
        return True

    def upload_bytes(self, data: bytes, blob_path: str, content_type: str) -> None:
        dest = self.root / blob_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)


class S3BlobStore:
    """boto3 backed store on a shared bucket; tenant isolation is the `{namespace}/{run_id}/` key prefix.

    Credentials come from the default boto3 chain (IRSA on EKS, env/profile locally). `endpoint_url` points at a
    MinIO/localstack stand-in for tests.
    """

    enabled = True

    def __init__(self, bucket: str, region: str | None = None, endpoint_url: str | None = None):
        try:
            import boto3
            from botocore.exceptions import ClientError
        except ImportError as e:  # pragma: no cover - exercised only in the image
            raise ConfigError("boto3 is not installed; pip install 'ldm[s3]'") from e
        self.bucket = bucket
        self._not_found = ClientError
        self.client = boto3.client("s3", region_name=region or None, endpoint_url=endpoint_url or None)

    def upload(self, local: Path, blob_path: str) -> None:
        with local.open("rb") as f:
            self.client.upload_fileobj(f, self.bucket, blob_path)

    def download(self, blob_path: str, local: Path) -> bool:
        local.parent.mkdir(parents=True, exist_ok=True)
        try:
            with local.open("wb") as f:
                self.client.download_fileobj(self.bucket, blob_path, f)
        except self._not_found as e:
            code = e.response.get("Error", {}).get("Code")
            if code in ("404", "NoSuchKey", "NotFound"):
                local.unlink(missing_ok=True)
                return False
            raise
        return True

    def upload_bytes(self, data: bytes, blob_path: str, content_type: str) -> None:
        self.client.put_object(Bucket=self.bucket, Key=blob_path, Body=data, ContentType=content_type)


class AzureBlobStore:
    """azure-storage-blob backed store. Auth: account key, else DefaultAzureCredential (managed identity)."""

    enabled = True

    def __init__(self, account: str, container: str, account_key: str | None, client_id: str | None):
        try:
            from azure.storage.blob import ContainerClient
        except ImportError as e:  # pragma: no cover - exercised only in the image
            raise ConfigError("azure-storage-blob is not installed; pip install 'ldm[azuresql]'") from e
        url = f"https://{account}.blob.core.windows.net"
        if account_key:
            self.client = ContainerClient(url, container, credential=account_key)
        else:
            from azure.identity import DefaultAzureCredential

            cred = (
                DefaultAzureCredential(managed_identity_client_id=client_id) if client_id else DefaultAzureCredential()
            )
            self.client = ContainerClient(url, container, credential=cred)

    def upload(self, local: Path, blob_path: str) -> None:
        with local.open("rb") as f:
            self.client.upload_blob(blob_path, f, overwrite=True)

    def download(self, blob_path: str, local: Path) -> bool:
        from azure.core.exceptions import ResourceNotFoundError

        local.parent.mkdir(parents=True, exist_ok=True)
        try:
            stream = self.client.download_blob(blob_path)
        except ResourceNotFoundError:
            return False
        with local.open("wb") as f:
            stream.readinto(f)
        return True

    def upload_bytes(self, data: bytes, blob_path: str, content_type: str) -> None:
        from azure.storage.blob import ContentSettings

        self.client.upload_blob(
            blob_path, data, overwrite=True, content_settings=ContentSettings(content_type=content_type)
        )
