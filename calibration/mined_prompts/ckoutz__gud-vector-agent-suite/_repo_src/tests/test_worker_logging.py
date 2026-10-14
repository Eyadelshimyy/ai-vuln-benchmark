"""The worker logs one line per command outcome and the entrypoints honor GVAS_LOG_LEVEL."""

import logging
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from composition_fakes import OwnerReplyFake, TranscriptionFake
from gvas.domain.identifiers import BusinessId
from gvas.domain.outbox import OWNER_MESSAGE_PROCESS_COMMAND_TYPE, OWNER_REPLY_COMMAND_TYPE
from gvas.interfaces.logging_setup import configure_logging, resolve_level
from test_composition import audio_part, configure_checklist, inbound, seed_business
from test_pilot_runtime import FailingTranscription, build, immediate_worker

LOGGER = "gvas.composition.dispatcher"


@pytest.mark.asyncio
async def test_successful_commands_log_at_info_with_identity(
    session_factory: async_sessionmaker[AsyncSession], caplog: pytest.LogCaptureFixture
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    application = build(session_factory, OwnerReplyFake(), TranscriptionFake({}))
    await configure_checklist(application, business_id)
    worker = immediate_worker(application)
    await application.ingest_service.ingest(
        inbound(business_id, "field notes: site visit", message_key="notes-1")
    )

    with caplog.at_level(logging.INFO, logger=LOGGER):
        await worker.drain()

    lines = [record.getMessage() for record in caplog.records if record.name == LOGGER]
    processed = [line for line in lines if f"command={OWNER_MESSAGE_PROCESS_COMMAND_TYPE}" in line]
    assert len(processed) == 1
    assert f"business={business_id}" in processed[0]
    assert "attempt=" in processed[0]
    assert any(f"command={OWNER_REPLY_COMMAND_TYPE}" in line for line in lines)
    assert any(line.startswith("batch done") for line in lines)
    assert all(record.levelno == logging.INFO for record in caplog.records if record.name == LOGGER)


@pytest.mark.asyncio
async def test_retries_warn_and_dead_letters_error(
    session_factory: async_sessionmaker[AsyncSession], caplog: pytest.LogCaptureFixture
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    application = build(session_factory, OwnerReplyFake(), FailingTranscription())
    worker = immediate_worker(application)
    await application.ingest_service.ingest(
        inbound(
            business_id,
            "field notes: site visit",
            message_key="notes-audio",
            attachments=(audio_part("channel-file:F1"),),
        )
    )

    with caplog.at_level(logging.INFO, logger=LOGGER):
        await worker.drain()
        await worker.drain()

    transcribe = [
        record
        for record in caplog.records
        if record.name == LOGGER and "command=field_note.transcribe" in record.getMessage()
    ]
    warnings = [record for record in transcribe if record.levelno == logging.WARNING]
    errors = [record for record in transcribe if record.levelno == logging.ERROR]
    assert len(errors) == 1
    assert "dead-lettered" in errors[0].getMessage()
    assert "attempt=3/3" in errors[0].getMessage()
    assert len(warnings) == 2
    assert all("failed, will retry" in record.getMessage() for record in warnings)
    # The logged error is whatever the adapter raised, exactly as stored in the outbox row.
    assert "error=RuntimeError" in errors[0].getMessage()


def test_log_level_names_resolve_with_info_fallback() -> None:
    assert resolve_level("debug") == logging.DEBUG
    assert resolve_level(" WARNING ") == logging.WARNING
    assert resolve_level("loud") == logging.INFO
    assert configure_logging("ERROR") == logging.ERROR
    assert logging.getLogger().level == logging.ERROR
    configure_logging("WARNING")
