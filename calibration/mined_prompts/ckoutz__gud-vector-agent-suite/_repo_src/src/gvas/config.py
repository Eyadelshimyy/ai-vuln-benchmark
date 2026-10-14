from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ASYNC_DRIVER = "postgresql+asyncpg"
# SQLAlchemy hands the query string to asyncpg.connect() as keyword arguments.
# asyncpg spells libpq's sslmode as ``ssl`` and accepts the same values, so the
# TLS requirement survives the rewrite.
SSL_MODE_KEY = "sslmode"
ASYNCPG_SSL_KEY = "ssl"
SSL_MODES = frozenset({"disable", "allow", "prefer", "require", "verify-ca", "verify-full"})
# asyncpg 0.30 has no keyword for these libpq options, so they would raise
# TypeError at connect time. Options it does accept -- ``target_session_attrs``,
# ``passfile``, ``krbsrvname``, ``gsslib`` -- stay in the URL for the driver.
UNSUPPORTED_QUERY_KEYS = frozenset({"channel_binding"})
POSTGRES_SCHEMES = frozenset({"postgres", "postgresql", ASYNC_DRIVER})


class DatabaseUrlError(ValueError):
    """Raised when a configured database URL cannot be used safely."""


def _normalized_pair(key: str, value: str) -> tuple[str, str]:
    if key != SSL_MODE_KEY:
        return key, value
    if value not in SSL_MODES:
        raise DatabaseUrlError(f"unsupported sslmode '{value}' in database URL")
    return ASYNCPG_SSL_KEY, value


def normalize_async_database_url(url: str) -> str:
    """Accept a managed-provider libpq URL and return a SQLAlchemy asyncpg URL.

    Railway hands out ``postgresql://`` URLs that may carry libpq query
    parameters. Rewriting them here keeps the deployment from having to
    maintain a second, hand-edited copy of the same credential, and keeps a
    requested TLS mode in force instead of silently downgrading it.
    """

    parts = urlsplit(url)
    if parts.scheme not in POSTGRES_SCHEMES:
        return url
    query = [
        _normalized_pair(key, value)
        for key, value in parse_qsl(parts.query)
        if key not in UNSUPPORTED_QUERY_KEYS
    ]
    return urlunsplit((ASYNC_DRIVER, parts.netloc, parts.path, urlencode(query), parts.fragment))


def require_managed_postgres_url(url: str) -> None:
    """Reject anything the deployed runtime cannot actually run on.

    The deployment runs on managed PostgreSQL. A SQLite or otherwise
    non-PostgreSQL URL only fails once the first command touches the database,
    by which point the web service has already accepted inbound events.
    """

    parts = urlsplit(url)
    if parts.scheme != ASYNC_DRIVER:
        raise DatabaseUrlError(f"database URL must use {ASYNC_DRIVER}")
    if not parts.hostname or not parts.path.strip("/"):
        raise DatabaseUrlError("database URL must name a host and a database")


class PublicApiSettings(BaseSettings):
    """The public quote surface's own knobs, all under ``GVAS_PUBLIC_*``."""

    model_config = SettingsConfigDict(env_prefix="GVAS_PUBLIC_", env_file=".env", extra="ignore")

    # Extra browser origins allowed for CORS beyond the configured business
    # site_urls — previews and local frontends. Comma-separated.
    cors_extra_origins: str = ""
    # Per-client-IP token bucket for the public routes.
    rate_limit_per_minute: int = Field(default=120, ge=1)

    def extra_origins(self) -> frozenset[str]:
        return frozenset(
            origin.strip().rstrip("/")
            for origin in self.cors_extra_origins.split(",")
            if origin.strip()
        )


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GVAS_", env_file=".env", extra="ignore")

    app_env: str = "development"
    log_level: str = "INFO"
    # Managed Postgres providers inject an unprefixed DATABASE_URL.
    database_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/gvas",
        validation_alias=AliasChoices("GVAS_DATABASE_URL", "DATABASE_URL"),
    )

    @field_validator("database_url")
    @classmethod
    def database_url_uses_async_driver(cls, value: str) -> str:
        return normalize_async_database_url(value)


class OpenAISettings(BaseSettings):
    """OpenAI transcribes audio and runs the contradiction pass on completed reviews."""

    model_config = SettingsConfigDict(env_prefix="GVAS_OPENAI_", env_file=".env", extra="ignore")

    api_key: str = ""
    api_base_url: str = "https://api.openai.com/v1"
    transcription_model: str = "whisper-1"
    review_model: str = "gpt-5.6-luna"
    timeout_seconds: float = Field(default=60.0, gt=0)
    max_audio_bytes: int = Field(default=25 * 1024 * 1024, ge=1)

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)


class CostCeilingSettings(BaseSettings):
    """Monthly per-business limits on metered model calls, in provider units.

    Transcription is limited in audio seconds and review-model calls in input
    plus output tokens. ``0`` (the default) means unlimited.
    """

    model_config = SettingsConfigDict(
        env_prefix="GVAS_COST_CEILING_", env_file=".env", extra="ignore"
    )

    transcription_seconds: int = Field(default=0, ge=0)
    review_tokens: int = Field(default=0, ge=0)


class IntakeSettings(BaseSettings):
    """The website intake chat's own knobs, all under ``GVAS_INTAKE_*``."""

    model_config = SettingsConfigDict(env_prefix="GVAS_INTAKE_", env_file=".env", extra="ignore")

    # New conversations one business may open per UTC day; 0 = unlimited.
    max_conversations_per_day: int = Field(default=50, ge=0)
    # Customer messages one conversation accepts before the agent stops
    # answering; 0 = unlimited.
    max_messages_per_conversation: int = Field(default=30, ge=0)
    # One-click owner approve/decline links in the notification e-mail. Both
    # are needed: the secret signs the token and the base URL is this
    # service's public origin, which the links are built from. Leave unset to
    # e-mail notices without links (the owner channel command still works).
    decision_link_secret: str = ""
    decision_link_base_url: str = ""

    @property
    def decision_links_enabled(self) -> bool:
        if not (self.decision_link_secret and self.decision_link_base_url):
            return False
        # The links carry actionable tokens, so they are only minted over
        # https — a deployment typo of http:// must not expose them. Plain
        # http is still allowed against loopback for local dev and tests.
        parsed = urlsplit(self.decision_link_base_url)
        if parsed.scheme == "https":
            return True
        return parsed.scheme == "http" and parsed.hostname in {
            "localhost",
            "127.0.0.1",
            "::1",
        }

    def decision_link_origin(self) -> str:
        return self.decision_link_base_url.rstrip("/")


class ResendSettings(BaseSettings):
    """Resend delivers approved customer quotes by email."""

    model_config = SettingsConfigDict(env_prefix="GVAS_RESEND_", env_file=".env", extra="ignore")

    api_key: str = ""
    api_base_url: str = "https://api.resend.com"
    from_address: str = ""
    reply_to_address: str = ""
    portal_url: str = "https://gudvector.com/portal/login"
    timeout_seconds: float = Field(default=30.0, gt=0)

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.from_address)


class WorkerSettings(BaseSettings):
    """Runtime knobs for the continuously running outbox worker service."""

    model_config = SettingsConfigDict(env_prefix="GVAS_WORKER_", env_file=".env", extra="ignore")

    batch_size: int = Field(default=10, ge=1)
    poll_seconds: float = Field(default=1.0, gt=0)
    retry_seconds: float = Field(default=30.0, gt=0)
    lease_seconds: float = Field(default=300.0, gt=0)
    id_prefix: str = "outbox-worker"


class ObjectStorageSettings(BaseSettings):
    """Cloudflare R2 (S3-compatible) settings; values come from the environment only."""

    model_config = SettingsConfigDict(env_prefix="GVAS_R2_", env_file=".env", extra="ignore")

    account_id: str = ""
    bucket: str = ""
    access_key_id: str = ""
    secret_access_key: str = ""
    region: str = "auto"

    @property
    def endpoint_url(self) -> str:
        return f"https://{self.account_id}.r2.cloudflarestorage.com"

    @property
    def is_configured(self) -> bool:
        return bool(
            self.account_id and self.bucket and self.access_key_id and self.secret_access_key
        )


class DemoSettings(BaseSettings):
    """A demo deployment: a fictional business, and nothing leaves the process.

    With ``mode`` on, e-mail, texts and owner channel messages are written to the log instead of
    sent, the booking calendar is generated openings rather than Calendly, and
    startup refuses any provider credential that could reach a real person or a
    live account. The hours below shape the generated openings.
    """

    model_config = SettingsConfigDict(env_prefix="GVAS_DEMO_", env_file=".env", extra="ignore")

    mode: bool = False
    # Used when the business has no zone of its own.
    timezone: str = "America/Los_Angeles"
    slot_minutes: int = Field(default=60, ge=15, le=240)
    day_start_hour: int = Field(default=8, ge=0, le=23)
    day_end_hour: int = Field(default=17, ge=1, le=24)
    # Visitor sandboxes: each visitor gets a fresh copy of this business
    # (by slug); empty turns them off. Deleted after ``sandbox_idle_minutes``
    # without a chat message or a dashboard request.
    sandbox_template_slug: str = ""
    sandbox_idle_minutes: int = Field(default=120, ge=5)
    sandbox_max_live: int = Field(default=200, ge=1)
    sandbox_per_ip_per_hour: int = Field(default=3, ge=1)
    # Gus messages one sandbox gets, and the whole demo gets per UTC day.
    sandbox_messages_per_visitor: int = Field(default=30, ge=1)
    sandbox_messages_per_day: int = Field(default=500, ge=1)
    sandbox_sweep_minutes: int = Field(default=10, ge=1)

    @field_validator("timezone")
    @classmethod
    def timezone_is_known(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as error:
            raise ValueError("is not a known IANA time zone") from error
        return value

    @model_validator(mode="after")
    def hours_hold_a_slot(self) -> "DemoSettings":
        if (self.day_end_hour - self.day_start_hour) * 60 < self.slot_minutes:
            raise ValueError("the demo day must be long enough for one slot")
        return self
