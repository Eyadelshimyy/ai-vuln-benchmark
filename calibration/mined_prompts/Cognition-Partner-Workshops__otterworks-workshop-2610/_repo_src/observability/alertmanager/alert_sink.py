"""Local stand-in for the Devin Automation webhook and the Slack incoming webhook.

Alertmanager always mirrors the Devin page here (and the Slack message when
SLACK_WEBHOOK_URL is unset). Every delivery is appended to /data/<receiver>.jsonl and the latest one is
served back on GET /<receiver>/latest, so `make incident-simulate` can show the
exact message the on-call Devin would have received.

Standard library only: it runs from the stock python:3.12-alpine image.

Trust boundary: the sink has no authentication because it impersonates two
webhook endpoints that Alertmanager posts to without credentials. It must only
ever be reachable from the Compose network and the host's loopback interface
(docker-compose.incident.yml publishes it on 127.0.0.1). Anything that can
reach the port can read and erase the captured pages; never expose it beyond
the laptop -- a real deployment uses the Devin Automation webhook and Slack
directly and has no sink at all.
"""

import contextlib
import json
import os
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

DATA_DIR = os.environ.get("SINK_DATA_DIR", "/data")
RECEIVERS = ("devin", "slack", "sink")
# Alertmanager pages are a few KiB; the Devin receiver is capped at one alert.
MAX_BODY_BYTES = int(os.environ.get("SINK_MAX_BODY_BYTES", str(1 << 20)))
SECRET_HEADERS = frozenset({"authorization", "x-webhook-secret", "cookie"})


def _path(receiver: str) -> str:
    return os.path.join(DATA_DIR, f"{receiver}.jsonl")


class Handler(BaseHTTPRequestHandler):
    server_version = "otterworks-alert-sink/1.0"

    def _send(
        self, status: int, body: bytes, content_type: str = "application/json"
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        receiver = self.path.strip("/").split("/")[0]
        if receiver not in RECEIVERS:
            self._send(404, b'{"error":"unknown receiver"}')
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._send(400, b'{"error":"bad content-length"}')
            return
        if length < 0 or length > MAX_BODY_BYTES:
            self.close_connection = True
            self._send(413, b'{"error":"payload too large"}')
            return
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw or b"null")
        except json.JSONDecodeError:
            payload = {"raw": raw.decode("utf-8", errors="replace")}
        record = {
            "received_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "receiver": receiver,
            "headers": {
                k: ("<redacted>" if k.lower() in SECRET_HEADERS else v)
                for k, v in self.headers.items()
            },
            "payload": payload,
        }
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(_path(receiver), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        self._send(200, b"ok", "text/plain")

    def do_GET(self) -> None:  # noqa: N802
        parts = self.path.strip("/").split("/")
        if parts == [""] or parts == ["health"]:
            self._send(200, b'{"status":"ok"}')
            return
        receiver = parts[0]
        if receiver not in RECEIVERS:
            self._send(404, b'{"error":"unknown receiver"}')
            return
        try:
            with open(_path(receiver), encoding="utf-8") as fh:
                lines = [ln for ln in fh.read().splitlines() if ln.strip()]
        except FileNotFoundError:
            lines = []
        if len(parts) > 1 and parts[1] == "latest":
            body = lines[-1].encode() if lines else b"null"
        else:
            body = ("[" + ",".join(lines) + "]").encode()
        self._send(200, body)

    def do_DELETE(self) -> None:  # noqa: N802
        for receiver in RECEIVERS:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(_path(receiver))
        self._send(200, b"ok", "text/plain")

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"[alert-sink] {self.address_string()} {fmt % args}", flush=True)


if __name__ == "__main__":
    port = int(os.environ.get("SINK_PORT", "9095"))
    print(f"[alert-sink] listening on :{port}, writing to {DATA_DIR}", flush=True)
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()
