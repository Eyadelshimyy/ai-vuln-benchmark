"""Production composition and the service entrypoints built on it."""

import asyncio
from collections.abc import Iterator
from uuid import uuid4

import pytest

from gvas.application.quotes import QuoteWorkflowHandler
from gvas.composition.production import (
    ProductionConfigurationError,
    ProductionSettings,
    build_production_runtime,
    load_production_settings,
    worker_identity,
)
from gvas.config import OpenAISettings, ResendSettings, Settings, WorkerSettings
from gvas.domain.identifiers import BusinessId
from gvas.domain.quotes import QUOTE_INTENT
from gvas.infrastructure.calendly.api import CalendlyAppointmentLookup
from gvas.infrastructure.object_storage import R2ObjectStorage
from gvas.infrastructure.slack.config import SlackSettings
from gvas.interfaces.worker import build_worker, run_worker

BUSINESS_ID = uuid4()
# Engine construction does not connect, so a syntactically valid managed URL is
# enough to build the runtime without a database.
MANAGED_DATABASE_URL = "postgresql+asyncpg://user:pw@db.railway.internal:5432/railway"
ENVIRONMENT = {
    "GVAS_DATABASE_URL": MANAGED_DATABASE_URL,
    "GVAS_SLACK_SIGNING_SECRET": "signing-secret",
    "GVAS_SLACK_BOT_TOKEN": "xoxb-not-a-real-token",
    "GVAS_SLACK_INSTALLATIONS": f"T0000000000={BUSINESS_ID}:U0000000000",
    "GVAS_OPENAI_API_KEY": "sk-not-a-real-key",
    "GVAS_RESEND_API_KEY": "re-not-a-real-key",
    "GVAS_RESEND_FROM_ADDRESS": "quotes@example.test",
}


@pytest.fixture
def production_environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for name, value in ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    yield


def settings() -> ProductionSettings:
    return ProductionSettings(
        app=Settings(),
        slack=SlackSettings(),
        openai=OpenAISettings(),
        resend=ResendSettings(),
        worker=WorkerSettings(),
    )


@pytest.mark.usefixtures("production_environment")
def test_startup_rejects_a_partially_configured_deployment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GVAS_RESEND_API_KEY")
    monkeypatch.delenv("GVAS_OPENAI_API_KEY")

    with pytest.raises(ProductionConfigurationError) as error:
        load_production_settings()

    message = str(error.value)
    assert "GVAS_RESEND_API_KEY" in message
    assert "GVAS_OPENAI_API_KEY" in message
    assert "signing-secret" not in message


@pytest.mark.usefixtures("production_environment")
def test_startup_rejects_a_deployment_that_inherited_the_development_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The localhost default is a development convenience, not a deployment."""

    monkeypatch.delenv("GVAS_DATABASE_URL")
    monkeypatch.delenv("DATABASE_URL", raising=False)

    with pytest.raises(ProductionConfigurationError) as error:
        load_production_settings()

    assert "DATABASE_URL" in str(error.value)


@pytest.mark.usefixtures("production_environment")
def test_startup_rejects_an_empty_database_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GVAS_DATABASE_URL", "")

    with pytest.raises(ProductionConfigurationError) as error:
        load_production_settings()

    assert "DATABASE_URL" in str(error.value)


@pytest.mark.usefixtures("production_environment")
def test_startup_rejects_a_database_the_deployment_is_not_specified_to_run_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Railway managed PostgreSQL is the specified store; SQLite is not."""

    monkeypatch.setenv("GVAS_DATABASE_URL", "sqlite+aiosqlite:///./gvas.db")

    with pytest.raises(ProductionConfigurationError) as error:
        load_production_settings()

    assert "postgresql+asyncpg" in str(error.value)


@pytest.mark.usefixtures("production_environment")
@pytest.mark.parametrize(
    "value",
    [
        f"T0000000000={BUSINESS_ID}:U0000000000|U0000000001",
        f"T0000000000={BUSINESS_ID}:U0000000000,T0000000001={BUSINESS_ID}:U0000000002",
    ],
)
def test_startup_rejects_more_than_the_one_owner_the_pilot_authorized(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("GVAS_SLACK_INSTALLATIONS", value)

    with pytest.raises(ProductionConfigurationError) as error:
        load_production_settings()

    assert "exactly one" in str(error.value)


@pytest.mark.usefixtures("production_environment")
def test_startup_rejects_a_malformed_installation_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GVAS_SLACK_INSTALLATIONS", "T0000000000=not-a-uuid:U0000000000")

    with pytest.raises(ProductionConfigurationError) as error:
        load_production_settings()

    assert "GVAS_SLACK_INSTALLATIONS" in str(error.value)


@pytest.mark.usefixtures("production_environment")
def test_production_app_serves_the_slack_request_url_and_health_check() -> None:
    runtime = build_production_runtime(load_production_settings())

    paths = {route.path for route in runtime.app.routes if hasattr(route, "path")}

    assert runtime.settings.slack.events_path in paths
    assert "/healthz" in paths
    assert not runtime.settings.telnyx.is_configured
    assert runtime.settings.telnyx.webhook_path not in paths


TELNYX_ENVIRONMENT = {
    "GVAS_TELNYX_PUBLIC_KEY": "cHVibGljLWtleQ==",
    "GVAS_TELNYX_API_KEY": "KEY-not-a-real-key",
    "GVAS_TELNYX_INSTALLATIONS": f"+15550100001={BUSINESS_ID}:+15550100000",
}


@pytest.mark.usefixtures("production_environment")
def test_telnyx_is_optional_as_a_set(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in TELNYX_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    runtime = build_production_runtime(load_production_settings())
    paths = {route.path for route in runtime.app.routes if hasattr(route, "path")}
    assert runtime.settings.telnyx.is_configured
    assert runtime.settings.telnyx.webhook_path in paths
    assert runtime.settings.slack.events_path in paths

    monkeypatch.delenv("GVAS_TELNYX_API_KEY")
    with pytest.raises(ProductionConfigurationError) as error:
        load_production_settings()
    message = str(error.value)
    assert "GVAS_TELNYX_API_KEY" in message
    assert "GVAS_TELNYX_PUBLIC_KEY" not in message
    assert "KEY-not-a-real-key" not in message
    assert "cHVibGljLWtleQ==" not in message


@pytest.mark.usefixtures("production_environment")
@pytest.mark.parametrize(
    "value",
    [
        f"+15550100001|+15550100002={BUSINESS_ID}:+15550100000",
        f"+15550100001={BUSINESS_ID}:+15550100000,+15550100003={BUSINESS_ID}:+15550100009",
        f"15550100001={BUSINESS_ID}:+15550100000",
    ],
)
def test_startup_rejects_telnyx_mappings_outside_the_pilot_boundary(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    for name, env_value in TELNYX_ENVIRONMENT.items():
        monkeypatch.setenv(name, env_value)
    monkeypatch.setenv("GVAS_TELNYX_INSTALLATIONS", value)

    with pytest.raises(ProductionConfigurationError) as error:
        load_production_settings()

    assert "GVAS_TELNYX_INSTALLATIONS" in str(error.value)


@pytest.mark.usefixtures("production_environment")
def test_without_r2_settings_reports_are_delivered_to_slack_only() -> None:
    runtime = build_production_runtime(load_production_settings())

    assert runtime.application.plan_custody_service is None
    assert runtime.application.report_publication_service._storage is None


@pytest.mark.usefixtures("production_environment")
def test_complete_r2_settings_wire_object_storage_into_the_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GVAS_R2_ACCOUNT_ID", "acct")
    monkeypatch.setenv("GVAS_R2_BUCKET", "gvas-artifacts")
    monkeypatch.setenv("GVAS_R2_ACCESS_KEY_ID", "r2-key-id")
    monkeypatch.setenv("GVAS_R2_SECRET_ACCESS_KEY", "r2-not-a-real-secret")

    runtime = build_production_runtime(load_production_settings())

    assert isinstance(runtime.application.report_publication_service._storage, R2ObjectStorage)
    assert runtime.application.plan_custody_service is not None


@pytest.mark.usefixtures("production_environment")
def test_startup_rejects_partial_r2_settings_without_leaking_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GVAS_R2_ACCOUNT_ID", "acct")
    monkeypatch.setenv("GVAS_R2_SECRET_ACCESS_KEY", "r2-not-a-real-secret")

    with pytest.raises(ProductionConfigurationError) as error:
        load_production_settings()

    message = str(error.value)
    assert "GVAS_R2_BUCKET" in message
    assert "GVAS_R2_ACCESS_KEY_ID" in message
    assert "GVAS_R2_ACCOUNT_ID" not in message
    assert "r2-not-a-real-secret" not in message


@pytest.mark.usefixtures("production_environment")
def test_worker_uses_configured_batching_and_a_replica_specific_identity() -> None:
    runtime = build_production_runtime(load_production_settings())

    build_worker(runtime)

    identity = worker_identity(runtime.settings.worker.id_prefix)
    assert identity.startswith(f"{runtime.settings.worker.id_prefix}-")
    assert identity != runtime.settings.worker.id_prefix


@pytest.mark.usefixtures("production_environment")
@pytest.mark.asyncio
async def test_worker_loop_stops_when_the_platform_asks_it_to() -> None:
    runtime = build_production_runtime(load_production_settings())
    stopping = asyncio.Event()
    stopping.set()

    batches = await run_worker(runtime, stopping)

    await runtime.aclose()
    assert batches == 0


def test_worker_identity_separates_replicas_on_one_host() -> None:
    assert worker_identity("outbox-worker") == worker_identity("outbox-worker")
    assert worker_identity("a") != worker_identity("b")


@pytest.mark.usefixtures("production_environment")
def test_production_settings_read_the_deployment_environment() -> None:
    resolved = settings()

    assert resolved.slack.installations
    assert resolved.app.database_url == MANAGED_DATABASE_URL


CALENDLY_ENVIRONMENT = {
    "GVAS_CALENDLY_TOKEN": "eyJ-not-a-real-calendly-token",
    "GVAS_CALENDLY_INSTALLATIONS": f"{BUSINESS_ID}=https://api.calendly.com/users/AAAAAAAA",
}


@pytest.mark.usefixtures("production_environment")
def test_calendly_lookup_is_optional_as_a_set(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = build_production_runtime(load_production_settings())
    handler = runtime.application.router.route(QUOTE_INTENT)
    assert isinstance(handler, QuoteWorkflowHandler)
    assert handler._appointment_lookup is None

    for name, value in CALENDLY_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    runtime = build_production_runtime(load_production_settings())
    handler = runtime.application.router.route(QUOTE_INTENT)
    assert isinstance(handler, QuoteWorkflowHandler)
    assert isinstance(handler._appointment_lookup, CalendlyAppointmentLookup)
    assert handler._appointment_lookup.serves(BusinessId(BUSINESS_ID))

    monkeypatch.delenv("GVAS_CALENDLY_INSTALLATIONS")
    with pytest.raises(ProductionConfigurationError) as error:
        load_production_settings()
    message = str(error.value)
    assert "GVAS_CALENDLY_INSTALLATIONS" in message
    assert "GVAS_CALENDLY_TOKEN" not in message
    assert "eyJ-not-a-real-calendly-token" not in message

    monkeypatch.setenv("GVAS_CALENDLY_INSTALLATIONS", "not-a-uuid=https://api.calendly.com/users/A")
    with pytest.raises(ProductionConfigurationError, match="GVAS_CALENDLY_INSTALLATIONS"):
        load_production_settings()


@pytest.mark.usefixtures("production_environment")
def test_calendly_webhook_key_requires_the_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GVAS_CALENDLY_WEBHOOK_SIGNING_KEY", "whsec-test")
    with pytest.raises(ProductionConfigurationError, match="GVAS_CALENDLY_WEBHOOK_SIGNING_KEY"):
        load_production_settings()
