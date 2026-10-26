"""Env-driven settings. All knobs live here (CLAUDE.md rule 8)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve .env against the backend package, NOT the process CWD — so the app
# picks up backend/.env whether uvicorn is launched from backend/, the repo
# root, or a launcher. (On Vercel the file is absent and real env vars win.)
_BACKEND_DIR = Path(__file__).resolve().parent.parent  # backend/


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # mock = JSON facts only
    # sqlserver = SQL facts (requires registry); no mock fallback
    # hybrid = SQL when server registered, else mock fact files (demo default for multi-DB)
    data_provider: Literal["mock", "sqlserver", "hybrid", "databricks"] = "mock"
    mock_data_dir: str = "../mockdata"

    support_error_email: str = "support@example.invalid"
    email_transport: Literal["smtp", "graph", "noop"] = "noop"

    smtp_host: str | None = None
    smtp_port: int | None = None
    smtp_user: str | None = None
    smtp_pass: str | None = None

    export_max_rows: int = 100_000

    # Live wildfire overlay — NASA FIRMS satellite active-fire heat layer.
    # Free key: https://firms.modaps.eosdis.nasa.gov/api/map_key/
    # When unset, /api/wildfire/active still returns WFIGS burn-area
    # perimeters; only the satellite-heat layer is disabled (with a note).
    firms_map_key: str | None = None

    # Vite dev :5173 (proxied), Vite preview :4173 (direct, needs CORS).
    cors_allow_origins: str = "http://localhost:5173,http://localhost:4173"

    # ── Multi-EDM fact plane ──────────────────────────────────────
    # Cap on cached EDMs (each holds its full pre-aggregated fact list).
    fact_cache_max_datasets: int = 1024
    # 0 = no TTL expiry (only LRU eviction).
    fact_cache_ttl_seconds: float = 3600.0
    # Parallel fan-out when loading many EDMs (portfolio / cedent / office).
    fact_load_max_workers: int = 16

    # ── SQL Server multi-host registry ────────────────────────────
    # Path to JSON: { "BERMUDA-SQL01": { "host": "...", "port": 1433, ... }, ... }
    # Relative paths resolve from backend/.
    sqlserver_servers_file: str | None = "../mockdata/sql_servers.json"
    # Inline JSON alternative (useful for quick demos / Vercel env).
    sqlserver_servers_json: str | None = None
    # Defaults applied when a server entry omits credentials.
    sqlserver_default_user: str | None = None
    sqlserver_default_password: str | None = None
    sqlserver_default_driver: str = "ODBC Driver 18 for SQL Server"
    # Table name pattern inside each EDM database. Prefer a stable view
    # ``ee_exposure_facts`` when you control the schema (tried first).
    sqlserver_evolution_table_pattern: str = "{edm}__EVOLUTION"
    # Hybrid only: on SQL failure / unregistered server, fall back to mock JSON.
    # Firm builds should set false so failures surface as empty + WARN_EDM_LOAD_FAILED.
    hybrid_fallback_on_sql_error: bool = True
    # Legacy single-conn string kept for docs/back-compat; registry is preferred.
    sqlserver_conn: str | None = None

    databricks_host: str | None = None
    databricks_token: str | None = None
    databricks_warehouse_id: str | None = None

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_allow_origins.split(",") if o.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
