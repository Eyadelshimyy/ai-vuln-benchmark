"""The per-request debug log never retains request credentials."""

import json
import uuid

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from starlette.responses import StreamingResponse

from app.chaos import FLAG_REQUEST_LOG
from app.middleware import request_log as request_log_module
from app.middleware.request_log import (
    REDACTED,
    RESPONSE_CAPTURE_BYTES,
    RequestLog,
    redact_body,
    redact_headers,
    redact_query,
)


def test_redact_headers_masks_credentials_only() -> None:
    redacted = redact_headers(
        {
            "Authorization": "Bearer secret",
            "Cookie": "session=abc",
            "Proxy-Authorization": "Basic xyz",
            "X-API-Key": "k",
            "Accept": "application/json",
        }
    )
    assert redacted["Authorization"] == REDACTED
    assert redacted["Cookie"] == REDACTED
    assert redacted["Proxy-Authorization"] == REDACTED
    assert redacted["X-API-Key"] == REDACTED
    assert redacted["Accept"] == "application/json"


def test_redact_query_masks_share_tokens_only() -> None:
    assert redact_query("token=s3cret&format=html") == f"token={REDACTED}&format=html"
    assert redact_query("owner_id=u1&size=100") == "owner_id=u1&size=100"
    assert redact_query("") == ""


def test_redact_body_masks_minted_tokens_only() -> None:
    body = json.dumps({"document_id": "d1", "token": 's3"cret', "title": "token"})
    redacted = json.loads(redact_body(body))
    assert redacted == {"document_id": "d1", "token": REDACTED, "title": "token"}
    assert redact_body('{"items": [], "total": 0}') == '{"items": [], "total": 0}'


@pytest.mark.asyncio
async def test_logged_request_keeps_body_but_not_bearer_token(
    client: AsyncClient,
    owner_id: uuid.UUID,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(request_log_module, "flag_active", lambda key: key == FLAG_REQUEST_LOG)
    monkeypatch.setattr(request_log_module, "request_log", RequestLog(str(tmp_path)))

    resp = await client.get(
        "/api/v1/documents/",
        params={"owner_id": str(owner_id)},
        headers={"Authorization": "Bearer top-secret", "Cookie": "sid=1"},
    )
    assert resp.status_code == 200

    files = list(tmp_path.iterdir())
    assert len(files) == 1
    raw = files[0].read_text()
    assert "top-secret" not in raw
    assert "sid=1" not in raw
    record = json.loads(raw.splitlines()[-1])
    assert record["headers"]["authorization"] == REDACTED
    assert record["headers"]["cookie"] == REDACTED
    assert json.loads(record["response"])["total"] == 0


@pytest.mark.asyncio
async def test_large_stream_is_delivered_whole_but_only_a_prefix_is_logged(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(request_log_module, "flag_active", lambda key: key == FLAG_REQUEST_LOG)
    monkeypatch.setattr(request_log_module, "request_log", RequestLog(str(tmp_path)))

    chunk = b"x" * 65536
    chunks = 3 * RESPONSE_CAPTURE_BYTES // len(chunk)
    app = FastAPI()
    request_log_module.install(app)

    @app.get("/export")
    async def export() -> StreamingResponse:
        async def gen():
            for _ in range(chunks):
                yield chunk

        return StreamingResponse(gen(), media_type="application/octet-stream")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.get("/export")
    assert resp.status_code == 200
    assert len(resp.content) == chunks * len(chunk)

    (log_file,) = tmp_path.iterdir()
    record = json.loads(log_file.read_text())
    assert len(record["response"]) == RESPONSE_CAPTURE_BYTES
    assert record["response_truncated"] is True
