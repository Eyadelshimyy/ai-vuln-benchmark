"""Browser-context network authorization.

``page.goto`` authorization does not cover documents, fetch, XHR, scripts,
images, stylesheets, iframes, redirects, or WebSockets. Every browser
network request is checked before it reaches the network. Popup pages inherit
the BrowserContext-level policy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from app.security_testing.errors import AuthorizationDeniedError
from app.security_testing.scope_model import AuthorizationDecision

if TYPE_CHECKING:
    from app.security_testing.engine import SecurityTestEngine

_RESOURCE_TOOLS = {
    "document": "browser_document",
    "stylesheet": "browser_asset",
    "image": "browser_asset",
    "media": "browser_asset",
    "font": "browser_asset",
    "script": "browser_asset",
    "texttrack": "browser_asset",
    "xhr": "browser_xhr",
    "fetch": "browser_fetch",
    "eventsource": "browser_fetch",
    "websocket": "browser_websocket",
    "manifest": "browser_asset",
    "other": "browser_network",
}


class BrowserNetworkPolicy:
    """Context-wide interception policy. Independent of manual researcher policy."""

    def __init__(self, engine: SecurityTestEngine) -> None:
        self._engine = engine

    def check(
        self,
        url: str,
        *,
        resource_type: str = "document",
        method: str = "GET",
    ) -> AuthorizationDecision:
        tool = _RESOURCE_TOOLS.get(resource_type.lower(), "browser_network")
        if resource_type.lower() == "websocket" and url.startswith("http"):
            url = (
                "wss://" + url.split("://", 1)[-1]
                if url.startswith("https://")
                else "ws://" + url.split("://", 1)[-1]
            )
        return self._engine.authorize(url, method=method, tool=tool, active=True)

    def require(self, url: str, *, resource_type: str = "document", method: str = "GET") -> None:
        decision = self.check(url, resource_type=resource_type, method=method)
        if not decision.allowed:
            raise AuthorizationDeniedError(decision.reason, target=url, tool="browser_network")

    async def handle_route(self, route: Any) -> None:
        request = getattr(route, "request", route)
        url = str(getattr(request, "url", ""))
        resource_type = str(getattr(request, "resource_type", "other") or "other")
        method = str(getattr(request, "method", "GET") or "GET")
        decision = self.check(url, resource_type=resource_type, method=method)
        if not decision.allowed:
            abort = getattr(route, "abort", None)
            if abort is not None:
                result = abort()
                if hasattr(result, "__await__"):
                    await result
            return
        cont = getattr(route, "continue_", None) or getattr(route, "continue", None)
        if cont is not None:
            result = cont()
            if hasattr(result, "__await__"):
                await result
