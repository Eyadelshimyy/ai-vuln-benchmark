"""Browser display helpers for local artifacts and remote URLs."""

from __future__ import annotations

import functools
import http.server
import threading
import time
import webbrowser
from pathlib import Path
from typing import Literal
from urllib.parse import quote, urlparse

from pydantic import BaseModel, Field

SourceType = Literal["remote-url", "local-file", "local-directory"]


class BrowserViewResult(BaseModel):
    """Structured result for a browser display target."""

    target: str = Field(description="Original resolved target URL/path.")
    display_url: str = Field(description="URL to navigate a browser to.")
    source_type: SourceType = Field(description="Kind of target being displayed.")
    opened: bool = Field(description="Whether the system browser was opened.")
    serve_root: str | None = Field(
        default=None,
        description="Local directory being served for local targets.",
    )
    host: str | None = Field(default=None, description="Local server host.")
    port: int | None = Field(default=None, description="Local server port.")
    duration: int | None = Field(
        default=None,
        description="Requested local-server lifetime in seconds.",
    )


def is_remote_url(target: str) -> bool:
    """Return whether a target is an HTTP(S) URL."""
    parsed = urlparse(target)
    return parsed.scheme in {"http", "https"}


def _url_path(path: Path) -> str:
    return "/".join(quote(part) for part in path.parts)


def _resolve_local_target(target: str | Path) -> tuple[Path, Path, str]:
    path = Path(target).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Local target not found: {path}")

    if path.is_dir():
        return path, path, ""
    return path, path.parent, _url_path(Path(path.name))


class LocalArtifactServer:
    """Small localhost HTTP server for browser-blocked local files/directories."""

    def __init__(
        self,
        target: str | Path,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        self.target = target
        self.host = host
        self.port = port
        self.resolved_target, self.serve_root, self._relative_url = (
            _resolve_local_target(target)
        )
        self._server: http.server.ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def source_type(self) -> SourceType:
        """Return whether the target is a file or directory."""
        return "local-directory" if self.resolved_target.is_dir() else "local-file"

    @property
    def display_url(self) -> str:
        """Return the browser URL for the served target."""
        if self._server is None:
            raise RuntimeError("Local artifact server has not been started.")
        port = int(self._server.server_address[1])
        display_url = f"http://{self.host}:{port}/"
        if self._relative_url:
            display_url += self._relative_url
        return display_url

    def start(self) -> LocalArtifactServer:
        """Start serving the target directory in a daemon thread."""
        if self._server is not None:
            return self

        handler = functools.partial(
            http.server.SimpleHTTPRequestHandler,
            directory=str(self.serve_root),
        )
        self._server = http.server.ThreadingHTTPServer((self.host, self.port), handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def shutdown(self) -> None:
        """Stop the local server if it is running."""
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        self._server = None
        self._thread = None

    def __enter__(self) -> LocalArtifactServer:
        return self.start()

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.shutdown()


class BrowserViewer:
    """Open remote URLs or serve local artifacts for browser display."""

    def open_remote(
        self, target: str, *, open_browser: bool = True
    ) -> BrowserViewResult:
        """Open a remote HTTP(S) URL or return the browser target."""
        if not is_remote_url(target):
            raise ValueError(f"Not an HTTP(S) URL: {target}")
        if open_browser:
            webbrowser.open(target)
        return BrowserViewResult(
            target=target,
            display_url=target,
            source_type="remote-url",
            opened=open_browser,
        )

    def serve_local(
        self,
        target: str | Path,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        open_browser: bool = True,
        duration: int | None = None,
    ) -> tuple[LocalArtifactServer, BrowserViewResult]:
        """Start serving a local file/directory and return the live server."""
        server = LocalArtifactServer(target, host=host, port=port).start()
        if open_browser:
            webbrowser.open(server.display_url)
        result = BrowserViewResult(
            target=str(server.resolved_target),
            display_url=server.display_url,
            source_type=server.source_type,
            opened=open_browser,
            serve_root=str(server.serve_root),
            host=host,
            port=int(server.display_url.split(":", 2)[2].split("/", 1)[0]),
            duration=duration,
        )
        return server, result

    def serve_for_duration(
        self,
        target: str | Path,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        duration: int = 300,
        open_browser: bool = True,
    ) -> BrowserViewResult:
        """Serve a local target for a bounded duration; 0 means until interrupted."""
        server, result = self.serve_local(
            target,
            host=host,
            port=port,
            open_browser=open_browser,
            duration=duration,
        )
        try:
            if duration == 0:
                while True:
                    time.sleep(3600)
            else:
                time.sleep(duration)
        except KeyboardInterrupt:
            pass
        finally:
            server.shutdown()
        return result

    def open(
        self,
        target: str | Path,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        duration: int = 300,
        open_browser: bool = True,
    ) -> BrowserViewResult:
        """Open a remote URL directly or serve a local target for browser display."""
        target_text = str(target)
        if is_remote_url(target_text):
            return self.open_remote(target_text, open_browser=open_browser)
        return self.serve_for_duration(
            target,
            host=host,
            port=port,
            duration=duration,
            open_browser=open_browser,
        )
