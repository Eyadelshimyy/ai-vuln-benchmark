"""Managed-provider database URL normalization."""

import pytest

from gvas.config import (
    DatabaseUrlError,
    Settings,
    normalize_async_database_url,
    require_managed_postgres_url,
)

MANAGED_URL = "postgresql://user:pw@host.railway.app:5432/railway"


@pytest.mark.parametrize(
    "mode", ["disable", "allow", "prefer", "require", "verify-ca", "verify-full"]
)
def test_libpq_sslmode_survives_as_the_asyncpg_ssl_argument(mode: str) -> None:
    """asyncpg spells the option ``ssl``; dropping it would downgrade TLS."""

    normalized = normalize_async_database_url(f"{MANAGED_URL}?sslmode={mode}")

    assert normalized.startswith("postgresql+asyncpg://")
    assert normalized.endswith(f"?ssl={mode}")


def test_options_asyncpg_cannot_accept_are_dropped_rather_than_passed_through() -> None:
    normalized = normalize_async_database_url(
        f"{MANAGED_URL}?sslmode=require&channel_binding=require"
    )

    assert normalized.endswith("?ssl=require")


def test_options_asyncpg_does_accept_are_preserved() -> None:
    """``target_session_attrs`` is a real asyncpg 0.30 keyword; routing must survive."""

    normalized = normalize_async_database_url(
        f"{MANAGED_URL}?sslmode=require&target_session_attrs=read-write"
    )

    assert normalized.endswith("?ssl=require&target_session_attrs=read-write")


def test_an_unknown_ssl_mode_is_refused_instead_of_silently_forwarded() -> None:
    with pytest.raises(DatabaseUrlError):
        normalize_async_database_url(f"{MANAGED_URL}?sslmode=required")


def test_non_postgres_urls_are_left_alone() -> None:
    assert normalize_async_database_url("sqlite+aiosqlite:///:memory:") == (
        "sqlite+aiosqlite:///:memory:"
    )


@pytest.mark.parametrize(
    "url",
    [
        "sqlite+aiosqlite:///./gvas.db",
        "mysql+aiomysql://user:pw@host/gvas",
        "postgresql+asyncpg://host.railway.app:5432/",
        "postgresql+asyncpg:///gvas",
    ],
)
def test_the_deployed_runtime_requires_a_managed_postgres_url(url: str) -> None:
    with pytest.raises(DatabaseUrlError):
        require_managed_postgres_url(url)


def test_a_normalized_managed_url_satisfies_the_production_requirement() -> None:
    require_managed_postgres_url(normalize_async_database_url(f"{MANAGED_URL}?sslmode=require"))


def test_settings_normalize_the_injected_provider_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", f"{MANAGED_URL}?sslmode=require")
    monkeypatch.delenv("GVAS_DATABASE_URL", raising=False)

    assert Settings().database_url == f"postgresql+asyncpg://{MANAGED_URL[13:]}?ssl=require"
