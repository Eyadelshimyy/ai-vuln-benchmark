"""Playwright browser adapter. Every navigation and network request goes through ScopeGuard."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from app.adapters.browsers.base import BrowserAdapter, BrowserSnapshot
from app.domain.http import HttpExchange, HttpHeader
from app.domain.scope import ScopeConstraint
from app.plugins.errors import AdapterNotImplementedError, OutOfScopeError
from app.security_testing.browser_network import BrowserNetworkPolicy
from app.security_testing.errors import AuthorizationDeniedError
from app.security_testing.screenshots import ScreenshotStore
from app.security_testing.secrets import redact_exchange, redact_mapping, redact_text

if TYPE_CHECKING:
    from app.security_testing.engine import SecurityTestEngine


@dataclass(frozen=True)
class BrowserActionPolicy:
    allow_navigation: bool = True
    allow_click: bool = False
    allow_form_submit: bool = False
    allow_destructive: bool = False
    capture_dom: bool = True
    capture_screenshots: bool = True
    capture_storage: bool = True

    def allows_click(self, label: str) -> bool:
        if self.allow_destructive:
            return self.allow_click
        lowered = label.lower()
        destructive_hints = ("delete", "destroy", "purchase", "pay", "transfer", "drop", "shutdown")
        if any(hint in lowered for hint in destructive_hints):
            return False
        return self.allow_click


@dataclass
class BrowserEvidence:
    url: str
    title: str | None = None
    dom: str | None = None
    screenshot_path: str | None = None
    console: tuple[str, ...] = ()
    cookies_meta: tuple[str, ...] = ()
    local_storage_meta: tuple[str, ...] = ()
    session_storage_meta: tuple[str, ...] = ()
    network: tuple[HttpExchange, ...] = ()
    history: tuple[str, ...] = ()
    captured_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    service_workers: str = "blocked"


class PlaywrightBrowserAdapter(BrowserAdapter):
    def __init__(
        self,
        *,
        engine: SecurityTestEngine | None = None,
        policy: BrowserActionPolicy | None = None,
        screenshot_store: ScreenshotStore | None = None,
        network_policy: BrowserNetworkPolicy | None = None,
    ) -> None:
        self._engine = engine
        self.policy = policy or BrowserActionPolicy()
        self._page: Any = None
        self._context: Any = None
        self._browser: Any = None
        self._playwright: Any = None
        self._console: list[str] = []
        self._network: list[HttpExchange] = []
        self._history: list[str] = []
        self._last: BrowserEvidence | None = None
        self._screenshot_store = screenshot_store
        self._network_policy = network_policy

    @property
    def adapter_id(self) -> str:
        return "playwright"

    def is_available(self) -> bool:
        try:
            import playwright.async_api  # noqa: F401
        except ImportError:
            return False
        return True

    def attach_engine(self, engine: SecurityTestEngine) -> None:
        self._engine = engine

    def network_policy(self) -> BrowserNetworkPolicy:
        if self._network_policy is not None:
            return self._network_policy
        if self._engine is None:
            raise OutOfScopeError(
                "*", detail="Browser network policy requires a SecurityTestEngine"
            )
        self._network_policy = BrowserNetworkPolicy(self._engine)
        return self._network_policy

    def screenshot_store(self) -> ScreenshotStore:
        if self._screenshot_store is not None:
            return self._screenshot_store
        project = self._engine.project_id if self._engine is not None else "default"
        self._screenshot_store = ScreenshotStore.for_project(project)
        return self._screenshot_store

    async def navigate(
        self,
        url: str,
        *,
        scope: ScopeConstraint | None = None,
        method: str = "GET",
    ) -> None:
        if self._engine is not None:
            decision = self._engine.authorize(url, method=method, tool="browser", active=True)
            if not decision.allowed:
                raise AuthorizationDeniedError(decision.reason, target=url, tool="browser")
            if decision.dry_run:
                self._history.append(url)
                return
        elif scope is not None:
            await super().navigate(url, scope=scope, method=method)
            return
        else:
            raise OutOfScopeError(
                url, detail="Browser navigation requires a ScopeGuard engine or ScopeConstraint"
            )
        if not self.policy.allow_navigation:
            raise AdapterNotImplementedError("Navigation is disabled by the action policy")
        await self._navigate(url)

    async def _navigate(self, url: str) -> None:
        if not self.is_available():
            # Headless-less recording mode used by unit tests.
            self.network_policy().require(url, resource_type="document")
            self._history.append(url)
            host = urlparse(url).hostname or ""
            self._last = sanitize_browser_evidence(
                BrowserEvidence(url=url, title=host, history=tuple(self._history))
            )
            return
        page = await self._ensure_page()
        page.on("console", lambda msg: self._console.append(f"{msg.type}: {msg.text}"))
        page.on(
            "response",
            lambda resp: self._network.append(
                redact_exchange(
                    HttpExchange(
                        method=resp.request.method,
                        url=resp.url,
                        response_status=resp.status,
                        source_tool="playwright",
                    )
                )
            ),
        )
        await page.goto(url, wait_until="domcontentloaded")
        self._history.append(url)

    async def snapshot(self) -> BrowserSnapshot:
        evidence = await self.capture()
        return BrowserSnapshot(
            url=evidence.url,
            title=evidence.title,
            dom=evidence.dom,
            screenshot_path=evidence.screenshot_path,
            console=evidence.console,
            cookies=evidence.cookies_meta,
            local_storage={k: "[meta]" for k in evidence.local_storage_meta},
            session_storage={k: "[meta]" for k in evidence.session_storage_meta},
            network=evidence.network,
        )

    async def capture(self) -> BrowserEvidence:
        if self._page is None:
            last_url = self._history[-1] if self._history else ""
            evidence = sanitize_browser_evidence(
                BrowserEvidence(
                    url=last_url,
                    title=self._last.title if self._last else None,
                    console=tuple(self._console),
                    network=tuple(self._network),
                    history=tuple(self._history),
                )
            )
            self._last = evidence
            return evidence
        page = self._page
        url = page.url
        title = await page.title()
        dom = await page.content() if self.policy.capture_dom else None
        screenshot_path = None
        if self.policy.capture_screenshots:
            screenshot_path = await self._capture_screenshot(page, url)
        cookies_meta: tuple[str, ...] = ()
        local_meta: tuple[str, ...] = ()
        session_meta: tuple[str, ...] = ()
        if self.policy.capture_storage:
            cookies = await page.context.cookies()
            cookies_meta = tuple(sorted({str(c.get("name", "")) for c in cookies}))
            local = await page.evaluate("() => Object.keys(localStorage)")
            session = await page.evaluate("() => Object.keys(sessionStorage)")
            local_redacted = redact_mapping({str(k): "x" for k in local})
            session_redacted = redact_mapping({str(k): "x" for k in session})
            local_meta = tuple(local_redacted.keys())
            session_meta = tuple(session_redacted.keys())
        evidence = sanitize_browser_evidence(
            BrowserEvidence(
                url=url,
                title=title,
                dom=dom,
                screenshot_path=screenshot_path,
                console=tuple(self._console),
                cookies_meta=cookies_meta,
                local_storage_meta=local_meta,
                session_storage_meta=session_meta,
                network=tuple(self._network),
                history=tuple(self._history),
            )
        )
        self._last = evidence
        return evidence

    async def _capture_screenshot(self, page: Any, url: str) -> str | None:
        try:
            data = await page.screenshot(type="png", full_page=False)
        except Exception:
            return None
        if not isinstance(data, (bytes, bytearray)):
            return None
        host = urlparse(url).hostname or "page"
        stored = self.screenshot_store().save(bytes(data), stem=f"{host}-{len(self._history)}")
        return str(stored)

    async def click(self, label: str) -> None:
        if not self.policy.allows_click(label):
            raise AdapterNotImplementedError(
                f"Click {label!r} is blocked by the browser action policy"
            )
        if self._page is not None:
            await self._page.get_by_text(label).click()

    async def _ensure_page(self) -> Any:
        if self._page is not None:
            return self._page
        from playwright.async_api import async_playwright

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=True)
        self._context = await self._browser.new_context(service_workers="block")
        policy = self.network_policy()
        await self._context.route("**/*", policy.handle_route)
        self._page = await self._context.new_page()
        return self._page


def sanitize_browser_evidence(evidence: BrowserEvidence) -> BrowserEvidence:
    """Redact secrets from DOM, console, URLs, storage metadata, and network."""
    network = tuple(redact_exchange(_strip_secret_headers(item)) for item in evidence.network)
    return BrowserEvidence(
        url=redact_text(evidence.url),
        title=redact_text(evidence.title) if evidence.title else None,
        dom=redact_text(evidence.dom) if evidence.dom else None,
        screenshot_path=evidence.screenshot_path,
        console=tuple(redact_text(item) for item in evidence.console),
        cookies_meta=tuple(redact_text(item) for item in evidence.cookies_meta),
        local_storage_meta=tuple(redact_text(item) for item in evidence.local_storage_meta),
        session_storage_meta=tuple(redact_text(item) for item in evidence.session_storage_meta),
        network=network,
        history=tuple(redact_text(item) for item in evidence.history),
        captured_at=evidence.captured_at,
        service_workers=evidence.service_workers,
    )


def _strip_secret_headers(exchange: HttpExchange) -> HttpExchange:
    headers = tuple(
        HttpHeader(name=header.name, value=header.value) for header in exchange.request_headers
    )
    return HttpExchange(
        method=exchange.method,
        url=exchange.url,
        request_id=exchange.request_id,
        timestamp=exchange.timestamp,
        scheme=exchange.scheme,
        host=exchange.host,
        port=exchange.port,
        path=exchange.path,
        request_headers=headers,
        request_body=exchange.request_body,
        request_body_meta=exchange.request_body_meta,
        query=exchange.query,
        cookies=(),
        response_status=exchange.response_status,
        response_headers=exchange.response_headers,
        response_body=exchange.response_body,
        response_body_meta=exchange.response_body_meta,
        elapsed_ms=exchange.elapsed_ms,
        source_tool=exchange.source_tool,
        scope_decision=exchange.scope_decision,
        metadata=exchange.metadata,
    )
