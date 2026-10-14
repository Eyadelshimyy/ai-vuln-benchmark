"""Burp Suite HTTP-history ingestion. Evidence by default; replay is gated."""

from __future__ import annotations

import base64
import json
import re
from collections.abc import Sequence
from pathlib import Path
from xml.etree import ElementTree

from app.adapters.proxies.base import ProxyAdapter
from app.domain.http import HttpExchange, HttpHeader
from app.security_testing.engine import SecurityTestEngine
from app.security_testing.errors import AuthorizationDeniedError
from app.security_testing.secrets import redact_exchange

_ITEM_TAG = re.compile(r"<item[\s>]", re.I)


class BurpHistoryAdapter(ProxyAdapter):
    """Import Burp XML or JSON HTTP history. Does not drive the Burp GUI."""

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path) if path else None
        self._exchanges: tuple[HttpExchange, ...] = ()

    @property
    def adapter_id(self) -> str:
        return "burp"

    def is_available(self) -> bool:
        return self._path is not None or bool(self._exchanges)

    def load(self, path: str | Path) -> tuple[HttpExchange, ...]:
        self._path = Path(path)
        text = self._path.read_text(encoding="utf-8", errors="replace")
        if text.lstrip().startswith("{") or text.lstrip().startswith("["):
            self._exchanges = _from_json(text)
        else:
            self._exchanges = _from_xml(text)
        self._exchanges = _dedupe(tuple(redact_exchange(item) for item in self._exchanges))
        return self._exchanges

    def ingest_text(self, text: str) -> tuple[HttpExchange, ...]:
        if text.lstrip().startswith("{") or text.lstrip().startswith("["):
            self._exchanges = _from_json(text)
        else:
            self._exchanges = _from_xml(text)
        self._exchanges = _dedupe(tuple(redact_exchange(item) for item in self._exchanges))
        return self._exchanges

    def _fetch_captured_exchanges(self) -> Sequence[HttpExchange]:
        if self._path is not None and not self._exchanges:
            self.load(self._path)
        return self._exchanges

    async def replay(self, exchange: HttpExchange, *, engine: SecurityTestEngine) -> object:
        """Replay is active testing and requires the full authorization chain."""
        decision = engine.authorize(
            exchange.url,
            method=exchange.method,
            tool="burp_replay",
            active=True,
            require_poc_approval=True,
        )
        if not decision.allowed:
            raise AuthorizationDeniedError(decision.reason, target=exchange.url, tool="burp_replay")
        return await engine.http("burp_replay").request(
            exchange.method,
            exchange.url,
            content=exchange.request_body,
        )


def _from_json(text: str) -> tuple[HttpExchange, ...]:
    data = json.loads(text)
    items = data if isinstance(data, list) else data.get("items") or data.get("messages") or []
    exchanges: list[HttpExchange] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or item.get("path") or "")
        method = str(item.get("method") or item.get("verb") or "GET")
        status = item.get("status") or item.get("status_code")
        exchanges.append(
            HttpExchange(
                method=method,
                url=url,
                request_body=str(item["request"]) if item.get("request") else None,
                response_body=str(item["response"]) if item.get("response") else None,
                response_status=int(status) if status else None,
                source_tool="burp",
            )
        )
    return tuple(exchanges)


def _from_xml(text: str) -> tuple[HttpExchange, ...]:
    try:
        root = ElementTree.fromstring(text)
    except ElementTree.ParseError:
        return ()
    exchanges: list[HttpExchange] = []
    for item in root.findall(".//item"):
        url = (item.findtext("url") or "").strip()
        method = (item.findtext("method") or "GET").strip()
        status_text = item.findtext("status")
        request = _maybe_b64(item.find("request"))
        response = _maybe_b64(item.find("response"))
        req_line, req_headers, req_body = _split_http(request)
        _resp_line, resp_headers, resp_body = _split_http(response)
        exchanges.append(
            HttpExchange(
                method=method or (req_line.split()[0] if req_line else "GET"),
                url=url,
                request_headers=req_headers,
                request_body=req_body,
                response_headers=resp_headers,
                response_body=resp_body,
                response_status=int(status_text) if status_text and status_text.isdigit() else None,
                source_tool="burp",
            )
        )
    return tuple(exchanges)


def _maybe_b64(node: ElementTree.Element | None) -> str:
    if node is None or node.text is None:
        return ""
    raw = node.text.strip()
    if node.attrib.get("base64") == "true":
        try:
            return base64.b64decode(raw).decode("utf-8", "replace")
        except (ValueError, OSError):
            return raw
    return raw


def _split_http(blob: str) -> tuple[str, tuple[HttpHeader, ...], str | None]:
    if not blob:
        return "", (), None
    head, _, body = blob.partition("\r\n\r\n")
    if "\r\n\r\n" not in blob:
        head, _, body = blob.partition("\n\n")
    lines = head.splitlines()
    start = lines[0] if lines else ""
    headers: list[HttpHeader] = []
    for line in lines[1:]:
        name, _, value = line.partition(":")
        if name:
            headers.append(HttpHeader(name=name.strip(), value=value.strip()))
    return start, tuple(headers), body or None


def _dedupe(items: tuple[HttpExchange, ...]) -> tuple[HttpExchange, ...]:
    seen: set[str] = set()
    unique: list[HttpExchange] = []
    for item in items:
        key = f"{item.method}:{item.url}:{item.response_status}"
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return tuple(unique)
