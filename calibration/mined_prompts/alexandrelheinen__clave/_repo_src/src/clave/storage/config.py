"""Configuration and credential management for Cloudflare R2 and D1.

Reads credentials from environment variables or a local .env file.
Credentials never land in git, adhering to AC-DATA-01.
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from pathlib import Path

from clave.errors import ClaveError


class StorageConfigError(ClaveError):
    """A required cloud storage or database configuration key is missing or invalid."""


@dataclass(frozen=True)
class R2Config:
    """Settings required to connect to Cloudflare R2 object storage."""

    bucket: str
    access_key_id: str
    secret_access_key: str
    endpoint_url: str
    region: str = "auto"


@dataclass(frozen=True)
class D1Config:
    """Settings required to connect to Cloudflare D1 SQL database."""

    account_id: str
    api_token: str
    database_id: str
    database_name: str | None = None


@dataclass(frozen=True)
class StorageConfig:
    """Combined remote storage and database configuration."""

    r2: R2Config | None = None
    d1: D1Config | None = None


def parse_env_file(path: Path) -> dict[str, str]:
    """Parse a simple .env file into key-value pairs without external libraries."""
    if not path.is_file():
        return {}
    env: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        env[key] = value
    return env


def _get_val(
    key: str,
    file_env: dict[str, str],
    environ: dict[str, str] | None = None,
) -> str | None:
    """Look up a setting in file environment first, then in process environment."""
    if key in file_env and file_env[key]:
        return file_env[key]
    source = os.environ if environ is None else environ
    return source.get(key)


def load_r2_config(
    env_path: Path | None = None,
    environ: dict[str, str] | None = None,
) -> R2Config:
    """Load and validate Cloudflare R2 credentials (AC-DATA-01).

    Args:
        env_path: Optional path to .env file (defaults to .env at cwd).
        environ: Optional environment dictionary override for testing.

    Returns:
        The validated R2Config.

    Raises:
        StorageConfigError: If any required R2 key is missing.
    """
    path = env_path if env_path is not None else Path(".env")
    file_env = parse_env_file(path)

    bucket = _get_val("R2_BUCKET_NAME", file_env, environ)
    access_key = _get_val("R2_ACCESS_KEY_ID", file_env, environ)
    secret_key = _get_val("R2_SECRET_ACCESS_KEY", file_env, environ)
    endpoint = _get_val("R2_ENDPOINT_URL", file_env, environ)
    region = _get_val("R2_REGION", file_env, environ) or "auto"

    missing: list[str] = []
    if not bucket:
        missing.append("R2_BUCKET_NAME")
    if not access_key:
        missing.append("R2_ACCESS_KEY_ID")
    if not secret_key:
        missing.append("R2_SECRET_ACCESS_KEY")
    if not endpoint:
        missing.append("R2_ENDPOINT_URL")

    if missing:
        raise StorageConfigError(
            f"missing required Cloudflare R2 configuration: {', '.join(missing)}"
        )

    assert bucket and access_key and secret_key and endpoint  # Type narrowing
    return R2Config(
        bucket=bucket,
        access_key_id=access_key,
        secret_access_key=secret_key,
        endpoint_url=endpoint.rstrip("/"),
        region=region,
    )


def load_d1_config(
    env_path: Path | None = None,
    environ: dict[str, str] | None = None,
) -> D1Config:
    """Load and validate Cloudflare D1 credentials (AC-DATA-01).

    Args:
        env_path: Optional path to .env file (defaults to .env at cwd).
        environ: Optional environment dictionary override for testing.

    Returns:
        The validated D1Config.

    Raises:
        StorageConfigError: If any required D1 key is missing.
    """
    path = env_path if env_path is not None else Path(".env")
    file_env = parse_env_file(path)

    account_id = _get_val("CLOUDFLARE_ACCOUNT_ID", file_env, environ)
    api_token = _get_val("CLOUDFLARE_API_TOKEN", file_env, environ)
    database_id = _get_val("D1_DATABASE_ID", file_env, environ)
    database_name = _get_val("D1_DATABASE_NAME", file_env, environ)

    missing: list[str] = []
    if not account_id:
        missing.append("CLOUDFLARE_ACCOUNT_ID")
    if not api_token:
        missing.append("CLOUDFLARE_API_TOKEN")
    if not database_id:
        missing.append("D1_DATABASE_ID")

    if missing:
        raise StorageConfigError(
            f"missing required Cloudflare D1 configuration: {', '.join(missing)}"
        )

    assert account_id and api_token and database_id  # Type narrowing
    return D1Config(
        account_id=account_id,
        api_token=api_token,
        database_id=database_id,
        database_name=database_name,
    )


def load_storage_config(
    env_path: Path | None = None,
    environ: dict[str, str] | None = None,
) -> StorageConfig:
    """Load available R2 and D1 configurations, ignoring individual missing sets.

    Returns StorageConfig with None for unconfigured services.
    """
    r2: R2Config | None = None
    d1: D1Config | None = None
    with contextlib.suppress(StorageConfigError):
        r2 = load_r2_config(env_path, environ)

    with contextlib.suppress(StorageConfigError):
        d1 = load_d1_config(env_path, environ)

    return StorageConfig(r2=r2, d1=d1)
