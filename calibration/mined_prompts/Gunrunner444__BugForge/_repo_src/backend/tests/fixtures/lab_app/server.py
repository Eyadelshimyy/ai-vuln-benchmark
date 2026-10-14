"""Deliberately vulnerable local lab application. Bind to loopback only."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import Any
from urllib.parse import parse_qs, urlparse


class LabAppHandler(BaseHTTPRequestHandler):
    """Harmless fixtures: auth issues, IDOR, reflected output, API parameters."""

    users = {
        "1": {"id": "1", "name": "alice", "role": "user"},
        "2": {"id": "2", "name": "bob", "role": "user"},
        "3": {"id": "3", "name": "admin", "role": "admin"},
    }

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._json(200, {"ok": True, "lab": True})
            return
        if parsed.path == "/redirect/in-scope":
            self._redirect("/health")
            return
        if parsed.path == "/redirect/out":
            self._redirect("https://evil.example/")
            return
        if parsed.path == "/redirect/loop":
            self._redirect("/redirect/loop")
            return
        if parsed.path == "/redirect/scheme":
            host, port = self.server.server_address
            self._redirect(f"https://{host}:{port}/health")
            return
        if parsed.path == "/redirect/port":
            self._redirect("http://127.0.0.1:9/health")
            return
        if parsed.path == "/redirect/excluded":
            self._redirect("/admin")
            return
        if parsed.path == "/admin":
            self._json(200, {"admin": True})
            return
        if parsed.path == "/login":
            self._html(
                200,
                "<html><title>Lab Login</title><body>"
                "<form method='post' action='/login'>"
                "<input name='user'/><input name='password' type='password'/>"
                "<button>Login</button></form></body></html>",
            )
            return
        if parsed.path.startswith("/users/"):
            user_id = parsed.path.rsplit("/", 1)[-1]
            user = self.users.get(user_id, {"id": user_id, "name": "unknown", "role": "user"})
            self._json(200, user)
            return
        if parsed.path == "/search":
            query = parse_qs(parsed.query).get("q", [""])[0]
            self._html(200, f"<html><title>Search</title><body>Results for {query}</body></html>")
            return
        if parsed.path == "/api/items":
            self._json(200, {"items": [{"id": 1, "name": "alpha"}, {"id": 2, "name": "beta"}]})
            return
        if parsed.path == "/api/admin":
            # Authentication issue: no check.
            self._json(200, {"secret": "lab-only-not-a-real-secret", "users": list(self.users)})
            return
        if parsed.path == "/api/profile":
            cookie = self.headers.get("Cookie") or ""
            if "session=lab-session" not in cookie:
                self._json(401, {"error": "auth required"})
                return
            self._json(200, {"id": "1", "name": "alice", "role": "user"})
            return
        if parsed.path.startswith("/api/orders/"):
            order_id = parsed.path.rsplit("/", 1)[-1]
            self._json(200, {"id": order_id, "owner": "anyone", "total": 42})
            return
        if parsed.path.startswith("/api/safe/orders/"):
            cookie = self.headers.get("Cookie") or ""
            if "session=lab-session" not in cookie:
                self._json(401, {"error": "auth required"})
                return
            order_id = parsed.path.rsplit("/", 1)[-1]
            if order_id != "1":
                self._json(403, {"error": "forbidden"})
                return
            self._json(200, {"id": "1", "owner": "alice", "total": 10})
            return
        if parsed.path == "/xss/safe":
            query = parse_qs(parsed.query).get("q", [""])[0]
            escaped = (
                query.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
            )
            self._html(200, f"<html><body>Results for {escaped}</body></html>")
            return
        if parsed.path == "/api/echo":
            query = parse_qs(parsed.query).get("q", [""])[0]
            self._json(200, {"echo": query})
            return
        if parsed.path == "/api/safe/echo":
            query = parse_qs(parsed.query).get("q", [""])[0]
            if any(ch in query for ch in "<>\"';"):
                self._json(400, {"error": "invalid input"})
                return
            self._json(200, {"echo": query})
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length).decode("utf-8", "replace") if length else ""
        if parsed.path == "/login":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Set-Cookie", "session=lab-session")
            self.end_headers()
            self.wfile.write(b'{"token":"lab-token","ok":true}')
            return
        if parsed.path == "/api/items":
            self._json(201, {"created": True, "echo": raw[:200]})
            return
        self._json(404, {"error": "not found"})

    def do_PUT(self) -> None:  # noqa: N802
        self.do_POST()

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _html(self, status: int, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _redirect(self, location: str) -> None:
        self.send_response(302)
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self.end_headers()


class LabServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 0) -> None:
        if host not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("Lab server must bind to a loopback address")
        self._httpd = ThreadingHTTPServer((host, port), LabAppHandler)
        self._thread: Thread | None = None

    @property
    def port(self) -> int:
        return int(self._httpd.server_address[1])

    @property
    def origin(self) -> str:
        host, port = self._httpd.server_address
        return f"http://{host}:{port}"

    def start(self) -> LabServer:
        self._thread = Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
