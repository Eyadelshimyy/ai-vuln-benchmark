"""Per-request debug log.

When the ``request_log`` flag is on, every API request is appended to a JSONL
file under ``settings.request_log_dir`` together with the response body (up to
``RESPONSE_CAPTURE_BYTES``), so support can replay what a customer actually
received.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import socket
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from urllib.parse import parse_qsl, urlencode

import structlog
from fastapi import FastAPI, Request, Response
from starlette.responses import StreamingResponse

from app.chaos import FLAG_REQUEST_LOG, flag_active
from app.config import settings
from app.telemetry import REQUEST_LOG_BYTES, REQUEST_LOG_CAPACITY_BYTES, SERVICE

logger = structlog.get_logger()

SKIP_PATHS = ("/health", "/metrics", "/ready")
REDACTED_HEADERS = frozenset({"authorization", "cookie", "proxy-authorization", "x-api-key"})
REDACTED_QUERY_PARAMS = frozenset({"token", "access_token", "api_key", "signature"})
REDACTED = "[redacted]"
_SECRET_FIELD = re.compile(
    r'"(' + "|".join(sorted(REDACTED_QUERY_PARAMS)) + r')"(\s*:\s*)"(?:[^"\\]|\\.)*"',
    re.IGNORECASE,
)
# Responses that declare at most this many bytes are recorded whole before they
# are delivered. Larger or unknown-length responses (exports) stream straight
# through and only this much of the body is kept, so one request never holds
# more than this in memory on the log's account.
RESPONSE_CAPTURE_BYTES = 1 << 20


def redact_headers(headers: dict[str, str]) -> dict[str, str]:
    """Copy of the request headers with credential-bearing values masked."""
    return {k: (REDACTED if k.lower() in REDACTED_HEADERS else v) for k, v in headers.items()}


def redact_query(query: str) -> str:
    """The query string with bearer-like parameters (share-link tokens) masked."""
    pairs = parse_qsl(query, keep_blank_values=True)
    return urlencode(
        [(k, REDACTED if k.lower() in REDACTED_QUERY_PARAMS else v) for k, v in pairs],
        safe="[]",
    )


def redact_body(text: str) -> str:
    """The recorded response with credential-valued JSON fields (minted share
    tokens) masked; everything else is kept verbatim."""
    return _SECRET_FIELD.sub(rf'"\1"\2"{REDACTED}"', text)


class RequestLog:
    def __init__(self, directory: str) -> None:
        self.directory = directory
        self.path = os.path.join(directory, f"requests-{socket.gethostname()}.jsonl")
        self._bytes = self._scan()

    def _scan(self) -> int:
        try:
            with os.scandir(self.directory) as it:
                total = sum(e.stat().st_size for e in it if e.is_file())
        except OSError:
            total = 0
        self._publish(total)
        return total

    def _publish(self, total: int) -> None:
        REQUEST_LOG_BYTES.labels(SERVICE).set(total)
        with contextlib.suppress(OSError):
            REQUEST_LOG_CAPACITY_BYTES.labels(SERVICE).set(shutil.disk_usage(self.directory).total)

    def append(self, record: dict) -> None:
        os.makedirs(self.directory, exist_ok=True)
        line = json.dumps(record, default=str) + "\n"
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(line)
        self._bytes += len(line)
        self._publish(self._bytes)

    def purge(self) -> None:
        try:
            with os.scandir(self.directory) as it:
                for entry in it:
                    if entry.is_file():
                        os.unlink(entry.path)
        except OSError:
            pass
        self._bytes = self._scan()


request_log = RequestLog(settings.request_log_dir)


def declared_length(response: Response) -> int | None:
    raw = response.headers.get("content-length", "")
    return int(raw) if raw.isdigit() else None


async def _chunks(response: Response) -> AsyncIterator[bytes]:
    if isinstance(response, StreamingResponse):
        async for chunk in response.body_iterator:
            yield chunk if isinstance(chunk, bytes) else chunk.encode()
    else:
        yield response.body


def install(app: FastAPI) -> None:
    @app.middleware("http")
    async def _request_log(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if request.url.path in SKIP_PATHS or not flag_active(FLAG_REQUEST_LOG):
            return await call_next(request)

        started = time.perf_counter()
        response = await call_next(request)
        record = {
            "ts": time.time(),
            "method": request.method,
            "path": request.url.path,
            "query": redact_query(str(request.url.query)),
            "status": response.status_code,
            "headers": redact_headers(dict(request.headers)),
        }
        length = declared_length(response)

        if length is not None and length <= RESPONSE_CAPTURE_BYTES:
            body = b"".join([chunk async for chunk in _chunks(response)])
            record["duration_ms"] = round((time.perf_counter() - started) * 1000, 2)
            record["response"] = redact_body(body.decode("utf-8", errors="replace"))
            request_log.append(record)
            return Response(
                content=body,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type,
            )

        async def tee() -> AsyncIterator[bytes]:
            kept = bytearray()
            truncated = False
            async for chunk in _chunks(response):
                room = RESPONSE_CAPTURE_BYTES - len(kept)
                kept += chunk[:room]
                truncated = truncated or len(chunk) > room
                yield chunk
            record["duration_ms"] = round((time.perf_counter() - started) * 1000, 2)
            record["response"] = redact_body(kept.decode("utf-8", errors="replace"))
            record["response_truncated"] = truncated
            # The body is already with the client; a failed write cannot fail it.
            try:
                request_log.append(record)
            except OSError:
                logger.exception("request_log_append_failed", path=request.url.path)

        return StreamingResponse(
            tee(),
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type,
        )
