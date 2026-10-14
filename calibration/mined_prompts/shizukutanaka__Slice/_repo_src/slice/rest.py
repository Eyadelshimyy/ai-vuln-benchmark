"""REST layer on http.server — stdlib only, single process.

    GET  /                     viewer UI
    POST /analyze[?model=..&save=1]   raw image body -> Knowledge JSON
    POST /analyze?multi=1           -> {"people": [docs], "count": n}
    POST /audit[?model=..&robust=1]  raw image body -> self-audit JSON
    GET  /knowledge            list stored knowledge ids
    GET  /knowledge/<id>       one stored document
    GET  /overlay/<id>         PNG overlay (requires ?save=1 at analyze)
"""

from __future__ import annotations

import hmac
import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

from . import __version__, bitmap, knowledge, pipeline, render, selfcheck

_VIEWER = os.path.join(os.path.dirname(__file__), "viewer.html")


class Handler(BaseHTTPRequestHandler):
    store: "knowledge.KnowledgeStore" = None  # set by serve()
    # read-through cache only — the overlay PNG itself is persisted
    # beside the doc (see _save_overlay); the dict is bounded so a
    # long-running server doesn't grow it forever
    overlays: dict = {}
    OVERLAY_CACHE_MAX = 128
    token: str = None                         # bearer token; None = open
    server_version = "Slice/" + __version__

    def log_message(self, fmt, *args):
        pass  # quiet; CLI prints access itself if needed

    # -- helpers ----------------------------------------------------------

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _bytes(self, body, ctype, code=200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, code, msg):
        self._json({"error": msg}, code)

    def _authorized(self, path) -> bool:
        """When a token is configured, API routes require
        `Authorization: Bearer <token>`. `/` and `/health` stay open
        so the viewer page loads and probes work; the data it fetches
        is still gated."""
        if not self.token or path in ("/", "/health"):
            return True
        auth = self.headers.get("Authorization", "")
        expected = "Bearer " + self.token
        return hmac.compare_digest(auth, expected)

    def _require_auth(self) -> bool:
        if self._authorized(urlparse(self.path).path):
            return True
        self._error(401, "unauthorized")
        return False

    # -- routes -----------------------------------------------------------

    def do_GET(self):
        if not self._require_auth():
            return
        path = urlparse(self.path).path
        if path == "/":
            with open(_VIEWER, "rb") as f:
                self._bytes(f.read(), "text/html; charset=utf-8")
        elif path == "/health":
            self._json({"ok": True, "version": __version__})
        elif path == "/knowledge":
            self._json({"items": self.store.list()})
        elif path.startswith("/knowledge/"):
            kid = path.rsplit("/", 1)[-1]
            try:
                self._json(self.store.get(kid))
            except KeyError:
                self._error(404, "not found")
        elif path.startswith("/overlay/"):
            kid = path.rsplit("/", 1)[-1].removesuffix(".png")
            png = self._get_overlay(kid)
            if png:
                self._bytes(png, "image/png")
            else:
                self._error(404, "no overlay")
        else:
            self._error(404, "not found")

    def do_POST(self):
        if not self._require_auth():
            return
        url = urlparse(self.path)
        if url.path not in ("/analyze", "/audit"):
            return self._error(404, "not found")
        length = int(self.headers.get("Content-Length") or 0)
        if not 0 < length <= 25 * 1024 * 1024:
            return self._error(400, "empty or >25MB body")
        raw = self.rfile.read(length)
        qs = parse_qs(url.query)
        model = (qs.get("model") or [None])[0]
        if url.path == "/audit":
            return self._audit(raw, qs, model)
        if qs.get("multi"):
            return self._analyze_multi(raw, qs, model)
        try:
            doc = pipeline.analyze(raw, model=model,
                                   source_name=self.headers.get(
                                       "X-Image-Name", ""),
                                   robust=bool(qs.get("robust")))
        except bitmap.UnsupportedFormat as e:
            return self._error(415, str(e))
        except Exception as e:  # noqa: BLE001 - API must not 500-blank
            return self._error(422, f"analysis failed: {e}")
        out = pipeline.strip_runtime(doc)
        if qs.get("save"):
            kid = self.store.save(out)
            out["overlay_url"] = f"/overlay/{kid}.png"
            if doc["_skeleton"].joints:
                self._save_overlay(kid, render.overlay_png(
                    doc["_bitmap"], doc["_skeleton"]))
        self._json(out)

    def _audit(self, raw, qs, model):
        try:
            res = selfcheck.run(
                raw, model=model,
                source_name=self.headers.get("X-Image-Name", ""),
                robust=bool(qs.get("robust")))
        except bitmap.UnsupportedFormat as e:
            return self._error(415, str(e))
        except Exception as e:  # noqa: BLE001
            return self._error(422, f"audit failed: {e}")
        self._json(res)

    def _overlay_path(self, kid: str) -> str:
        return os.path.join(self.store.root, kid + ".overlay.png")

    def _cache_overlay(self, kid: str, png: bytes) -> None:
        if len(self.overlays) >= self.OVERLAY_CACHE_MAX:
            # insertion-ordered dict: drop the oldest entry
            self.overlays.pop(next(iter(self.overlays)))
        self.overlays[kid] = png

    def _save_overlay(self, kid: str, png: bytes) -> None:
        # persist beside the doc (tmp+replace, same as store.save) so
        # overlay_url survives a restart — memory is just a cache
        tmp = self._overlay_path(kid) + ".tmp"
        with open(tmp, "wb") as f:
            f.write(png)
        os.replace(tmp, self._overlay_path(kid))
        self._cache_overlay(kid, png)

    def _get_overlay(self, kid: str):
        png = self.overlays.get(kid)
        if png is None and re.fullmatch(r"k_[0-9a-f]{12}", kid):
            try:
                with open(self._overlay_path(kid), "rb") as f:
                    png = f.read()
                self._cache_overlay(kid, png)
            except OSError:
                png = None
        return png

    def _analyze_multi(self, raw, qs, model):
        try:
            docs = pipeline.analyze_multi(
                raw, model=model,
                source_name=self.headers.get("X-Image-Name", ""),
                robust=bool(qs.get("robust")))
        except bitmap.UnsupportedFormat as e:
            return self._error(415, str(e))
        except Exception as e:  # noqa: BLE001
            return self._error(422, f"analysis failed: {e}")
        people = []
        for d in docs:
            out = pipeline.strip_runtime(d)
            if qs.get("save"):
                kid = self.store.save(out)
                out["overlay_url"] = f"/overlay/{kid}.png"
                if d["_skeleton"].joints:
                    self._save_overlay(kid, render.overlay_png(
                        d["_bitmap"], d["_skeleton"]))
            people.append(out)
        self._json({"people": people, "count": len(people)})


def serve(port: int = 8000, store_dir: str = "knowledge",
          token: str = None):
    Handler.store = knowledge.KnowledgeStore(store_dir)
    Handler.token = token or os.environ.get("SLICE_TOKEN")
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Slice {__version__} listening on http://127.0.0.1:{port}")
    print(f"Knowledge store: {os.path.abspath(store_dir)}")
    print("auth: " + ("bearer token required for API routes"
                     if Handler.token else "open (set SLICE_TOKEN or "
                     "--token to require auth)"))
    httpd.serve_forever()
