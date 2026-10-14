"""Tests for JsonlPromptLogger and Markdown rendering."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from ellements.core import (
    JsonlPromptLogger,
    LLMClient,
    LLMError,
)
from ellements.core.observability import (
    LLMRequestEvent,
    LLMResponseEvent,
    format_log_markdown,
)


def _mock_completion(content: str = "hello") -> Any:
    message = MagicMock()
    message.content = content
    response = MagicMock()
    response.choices = [MagicMock(message=message)]
    response.usage = None
    return response


@pytest.mark.asyncio
async def test_logger_writes_one_jsonl_line_per_call(tmp_path):
    logger = JsonlPromptLogger(tmp_path)
    client = LLMClient(model="openai/gpt-4o-mini", observers=[logger])

    mock = AsyncMock(return_value=_mock_completion("hi"))
    with patch("ellements.core.llm.client.litellm.acompletion", mock):
        await client.complete("greet")
        await client.complete("again")

    path = logger.current_path
    lines = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert len(lines) == 2
    assert all(line["status"] == "ok" for line in lines)
    assert all(line["method"] == "complete" for line in lines)


@pytest.mark.asyncio
async def test_logger_serializes_concurrent_writes(tmp_path):
    logger = JsonlPromptLogger(tmp_path)
    client = LLMClient(model="openai/gpt-4o-mini", observers=[logger])

    mock = AsyncMock(return_value=_mock_completion("ok"))
    with patch("ellements.core.llm.client.litellm.acompletion", mock):
        await asyncio.gather(*(client.complete(f"p{i}") for i in range(25)))

    path = logger.current_path
    text = path.read_text(encoding="utf-8")
    raw_lines = [line for line in text.splitlines() if line.strip()]
    assert len(raw_lines) == 25
    for line in raw_lines:
        record = json.loads(line)
        assert record["status"] == "ok"


@pytest.mark.asyncio
async def test_logger_captures_error(tmp_path):
    import litellm

    logger = JsonlPromptLogger(tmp_path)
    client = LLMClient(
        model="openai/gpt-4o-mini",
        observers=[logger],
        max_retries=0,
    )

    err = litellm.AuthenticationError(
        message="boom", llm_provider="openai", model="gpt-4o-mini"
    )
    mock = AsyncMock(side_effect=err)
    with (
        patch("ellements.core.llm.client.litellm.acompletion", mock),
        pytest.raises(LLMError),
    ):
        await client.complete("hello")

    records = [
        json.loads(line)
        for line in logger.current_path.read_text().splitlines()
        if line.strip()
    ]
    assert len(records) == 1
    assert records[0]["status"] == "error"
    assert records[0]["error_type"]


@pytest.mark.asyncio
async def test_logger_handles_multimodal_messages(tmp_path):
    logger = JsonlPromptLogger(tmp_path)
    event = LLMRequestEvent(
        call_id="c1",
        method="complete",
        model="m",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {"type": "image_url", "image_url": {"url": "data:..."}},
                ],
            }
        ],
        temperature=0.7,
        max_tokens=None,
    )
    await logger.on_request(event)
    await logger.on_response(
        LLMResponseEvent(
            call_id="c1",
            method="complete",
            model="m",
            response="ok",
            duration_ms=10,
        )
    )

    records = [
        json.loads(line)
        for line in logger.current_path.read_text().splitlines()
        if line.strip()
    ]
    assert records[0]["messages"][0]["content"][0]["type"] == "text"
    assert records[0]["messages"][0]["content"][1]["type"] == "image_url"
    if os.name != "nt":
        assert tmp_path.stat().st_mode & 0o777 == 0o700
        assert logger.current_path.stat().st_mode & 0o777 == 0o600


def test_format_log_markdown_renders_each_record(tmp_path):
    log_path = tmp_path / "test.jsonl"
    log_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "call_id": "c1",
                        "timestamp": "2024-01-01T00:00:00+00:00",
                        "method": "complete",
                        "status": "ok",
                        "model": "openai/gpt-4o-mini",
                        "temperature": 0.7,
                        "messages": [{"role": "user", "content": "hi"}],
                        "response": "hello",
                        "duration_ms": 12,
                    }
                ),
                json.dumps(
                    {
                        "call_id": "c2",
                        "timestamp": "2024-01-01T00:01:00+00:00",
                        "method": "complete",
                        "status": "error",
                        "model": "openai/gpt-4o-mini",
                        "error_type": "AuthenticationError",
                        "error_message": "bad key",
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    rendered = format_log_markdown(log_path)
    assert "complete" in rendered
    assert "AuthenticationError" in rendered


def test_format_log_markdown_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        format_log_markdown(tmp_path / "nope.jsonl")
