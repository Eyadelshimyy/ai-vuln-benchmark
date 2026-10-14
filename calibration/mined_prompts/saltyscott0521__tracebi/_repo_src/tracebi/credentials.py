"""
A person's sign-in to a TraceBi server, kept on their machine.

``tracebi login --server <url>`` is an OAuth 2.1 authorization-code flow with
PKCE against the server's own ``/authorize`` and ``/token`` (the authorization
server the MCP connectors use), as a public client with a loopback redirect
(``http://127.0.0.1:<port>/callback``). The server hands the actual login to
the company's identity provider, so no password ever touches this code.

Tokens live in ``~/.config/tracebi/credentials.json`` (``$XDG_CONFIG_HOME``
respected), mode 0600, keyed by server URL:

    {"servers": {"https://bi.example.com": {
        "client_id": "...",            # from the server's /register, cached
        "access_token": "tbat_...", "refresh_token": "tbrt_...",
        "expires_at": 1760000000, "scope": "mcp offline_access"}}}

Anything that calls the server's API as this person asks for
:func:`access_token`, which refreshes an expired token first. Standard library
only, so it works from any install.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import http.server
import json
import os
import secrets
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from pathlib import Path
from typing import Optional

SCOPE = "mcp offline_access"
_CLIENT_NAME = "TraceBi CLI"
_LOOPBACK = ("localhost", "127.0.0.1", "::1")
_MAX_BODY = 256 * 1024
_EARLY = 60            # refresh this many seconds before a token expires


class CredentialsError(Exception):
    """Something the person can act on; the message says what."""


class NotSignedIn(CredentialsError):
    """No usable sign-in for this server: run ``tracebi login``."""


# ── The file ─────────────────────────────────────────────────────────────

def credentials_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME", "")
    root = Path(base) if base and os.path.isabs(base) else Path.home() / ".config"
    return root / "tracebi" / "credentials.json"


def server_key(url: str) -> str:
    """The canonical origin for *url*; refuses a URL that would send tokens
    over the network in clear."""
    parts = urllib.parse.urlsplit((url or "").strip())
    host = (parts.hostname or "").lower()
    if (parts.scheme not in ("http", "https") or not host or parts.username
            or parts.path.strip("/") or parts.query or parts.fragment):
        raise CredentialsError(
            f"{url!r} is not a server address; use its origin, such as "
            "https://bi.example.com")
    if parts.scheme == "http" and host not in _LOOPBACK:
        raise CredentialsError(
            f"{url} is plain http: tokens would cross the network in clear. "
            "Use the server's https address.")
    return f"{parts.scheme}://{parts.netloc.lower()}"


def _read() -> dict:
    path = credentials_path()
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        return {"servers": {}}
    except (OSError, ValueError) as exc:
        raise CredentialsError(f"{path} cannot be read ({exc}); fix or delete it") from exc
    if not isinstance(data, dict) or not isinstance(data.get("servers"), dict):
        raise CredentialsError(f"{path} is not a TraceBi credentials file; fix or delete it")
    return data


def _write(data: dict) -> None:
    path = credentials_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path.parent, 0o700)
    except OSError:
        pass
    # mkstemp creates the file 0600, so there is no moment it is readable.
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".credentials-")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def _save(server: str, record: Optional[dict]) -> None:
    data = _read()
    if record:
        data["servers"][server] = record
    else:
        data["servers"].pop(server, None)
    _write(data)


def _forget_client(server: str) -> None:
    """Drop the cached client registration (the server may have lost it);
    any tokens stay."""
    data = _read()
    data["servers"].get(server, {}).pop("client_id", None)
    _write(data)


@contextlib.contextmanager
def _lock():
    """One process refreshes at a time: a refresh token is single use, so two
    processes presenting the same one would end the sign-in."""
    try:
        import fcntl
    except ImportError:          # Windows
        yield
        return
    path = credentials_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path.with_suffix(".lock"), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


# ── The server ───────────────────────────────────────────────────────────

class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def _request(method: str, url: str, *, form: Optional[dict] = None,
             json_body: Optional[dict] = None,
             bearer: Optional[str] = None) -> tuple[int, dict]:
    """(status, JSON body). Never follows a redirect: a token must not be
    replayed somewhere the person did not name."""
    headers = {"Accept": "application/json"}
    data = None
    if form is not None:
        data = urllib.parse.urlencode(form).encode()
        headers["Content-Type"] = "application/x-www-form-urlencoded"
    elif json_body is not None:
        data = json.dumps(json_body).encode()
        headers["Content-Type"] = "application/json"
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.build_opener(_NoRedirect).open(req, timeout=15) as resp:
            status, body = resp.status, resp.read(_MAX_BODY)
    except urllib.error.HTTPError as exc:
        status, body = exc.code, exc.read(_MAX_BODY)
    except (urllib.error.URLError, OSError) as exc:
        raise CredentialsError(
            f"cannot reach {url}: {getattr(exc, 'reason', exc)}") from exc
    try:
        payload = json.loads(body)
    except ValueError:
        payload = {}
    return status, payload if isinstance(payload, dict) else {}


def _metadata(server: str) -> dict:
    status, meta = _request("GET", f"{server}/.well-known/oauth-authorization-server")
    if status != 200 or not meta:
        raise CredentialsError(
            f"{server} does not offer sign-in (it answered {status}). "
            "Is TRACEBI_OIDC_ISSUER set on it?")
    if str(meta.get("issuer", "")).rstrip("/") != server:
        raise CredentialsError(
            f"{server} says its address is {meta.get('issuer')!r}. Set "
            "TRACEBI_PUBLIC_URL on the server to the address you use.")
    for name in ("authorization_endpoint", "token_endpoint", "registration_endpoint"):
        if not str(meta.get(name, "")).startswith(server + "/"):
            raise CredentialsError(f"{server}'s {name} is missing or on another host")
    revoke = meta.get("revocation_endpoint")
    if revoke and not str(revoke).startswith(server + "/"):
        meta.pop("revocation_endpoint")
    return meta


def _register(meta: dict) -> str:
    status, reply = _request("POST", meta["registration_endpoint"], json_body={
        "client_name": _CLIENT_NAME, "redirect_uris": ["http://127.0.0.1/callback"],
        "token_endpoint_auth_method": "none", "response_types": ["code"],
        "grant_types": ["authorization_code", "refresh_token"], "scope": SCOPE})
    if status not in (200, 201) or not reply.get("client_id"):
        raise CredentialsError(
            f"the server refused to register this CLI: "
            f"{reply.get('error_description') or reply.get('error') or status}")
    return str(reply["client_id"])


def _record(client_id: str, reply: dict) -> dict:
    return {"client_id": client_id, "access_token": reply["access_token"],
            "refresh_token": reply.get("refresh_token"),
            "expires_at": int(time.time() + int(reply.get("expires_in") or 3600)),
            "scope": reply.get("scope") or SCOPE}


# ── login / logout ───────────────────────────────────────────────────────

def _callback_server(state: str, box: dict, done: threading.Event):
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):            # no request lines on the terminal
            pass

        def do_GET(self):
            parts = urllib.parse.urlsplit(self.path)
            q = dict(urllib.parse.parse_qsl(parts.query))
            if parts.path != "/callback" or done.is_set() or not secrets.compare_digest(
                    q.get("state", ""), state):
                self._reply(404, "Not found.")
                return
            box.update(code=q.get("code"), error=q.get("error"))
            done.set()
            self._reply(200, "Signed in to TraceBi. You can close this tab."
                        if q.get("code") else "Sign-in was not completed.")

        def _reply(self, status: int, message: str) -> None:
            body = message.encode()
            self.send_response(status)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

    return http.server.HTTPServer(("127.0.0.1", 0), Handler)


def login(server: str, *, timeout: float = 300, echo=print) -> dict:
    """Sign in to *server* through the browser; returns the stored record."""
    server = server_key(server)
    meta = _metadata(server)
    cached = _read()["servers"].get(server, {}).get("client_id")
    client_id = cached or _register(meta)

    verifier = secrets.token_urlsafe(48)
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
    state = secrets.token_urlsafe(24)
    box: dict = {}
    done = threading.Event()
    httpd = _callback_server(state, box, done)
    redirect = f"http://127.0.0.1:{httpd.server_address[1]}/callback"
    threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.1},
                     daemon=True).start()
    try:
        url = meta["authorization_endpoint"] + "?" + urllib.parse.urlencode({
            "response_type": "code", "client_id": client_id, "redirect_uri": redirect,
            "code_challenge": challenge, "code_challenge_method": "S256",
            "scope": SCOPE, "state": state})
        echo(f"Opening your browser to sign in. If it does not open, visit:\n  {url}")
        with contextlib.suppress(Exception):
            webbrowser.open(url)
        if not done.wait(timeout):
            if cached:     # the server may have forgotten this client: register afresh next time
                _forget_client(server)
            raise CredentialsError(
                "timed out waiting for the browser. If it showed an error about "
                "the client, run the command again.")
    finally:
        httpd.shutdown()
        httpd.server_close()
    if box.get("error") or not box.get("code"):
        raise CredentialsError(f"sign-in was not completed ({box.get('error') or 'no code'})")

    status, reply = _request("POST", meta["token_endpoint"], form={
        "grant_type": "authorization_code", "code": box["code"], "client_id": client_id,
        "redirect_uri": redirect, "code_verifier": verifier})
    if status != 200 or not reply.get("access_token"):
        if reply.get("error") == "invalid_client":
            _forget_client(server)
        raise CredentialsError(
            f"the server refused the sign-in: "
            f"{reply.get('error_description') or reply.get('error') or status}")
    record = _record(client_id, reply)
    with _lock():
        _save(server, record)
    status, me = _request("GET", f"{server}/api/me", bearer=record["access_token"])
    if status == 200 and me.get("actor"):
        echo(f"Signed in to {server} as {me['actor']} ({me.get('role')}).")
    else:
        echo(f"Signed in to {server}.")
    return record


def logout(server: str, *, echo=print) -> bool:
    """Revoke this machine's sign-in at *server* and forget it. True when
    there was one."""
    server = server_key(server)
    with _lock():
        record = _read()["servers"].get(server)
        if not record or not record.get("access_token"):
            echo(f"Not signed in to {server}.")
            return False
        try:
            meta = _metadata(server)
            if meta.get("revocation_endpoint"):
                _request("POST", meta["revocation_endpoint"], form={
                    "token": record.get("refresh_token") or record["access_token"],
                    "token_type_hint": "refresh_token" if record.get("refresh_token")
                    else "access_token", "client_id": record.get("client_id", "")})
        except CredentialsError as exc:
            echo(f"Could not reach {server} to revoke the sign-in ({exc}); "
                 "it will still expire there.")
        # Keep the client registration: signing in again needs no new one.
        _save(server, {"client_id": record.get("client_id", "")})
    echo(f"Signed out of {server}.")
    return True


# ── For anything that calls the API as this person ───────────────────────

def access_token(server: str) -> str:
    """A valid access token for *server*, refreshed first when it is about to
    expire. Raises :class:`NotSignedIn` when there is none to refresh."""
    server = server_key(server)
    record = _read()["servers"].get(server) or {}
    if not record.get("access_token"):
        raise NotSignedIn(f"not signed in to {server}: run `tracebi login --server {server}`")
    if record.get("expires_at", 0) - _EARLY > time.time():
        return record["access_token"]
    with _lock():
        # Another process may have refreshed while this one waited.
        record = _read()["servers"].get(server) or {}
        if record.get("expires_at", 0) - _EARLY > time.time() and record.get("access_token"):
            return record["access_token"]
        return _refresh(server, record)["access_token"]


def _refresh(server: str, record: dict) -> dict:
    again = f"run `tracebi login --server {server}`"
    if not record.get("refresh_token"):
        raise NotSignedIn(f"the sign-in to {server} has expired: {again}")
    meta = _metadata(server)
    status, reply = _request("POST", meta["token_endpoint"], form={
        "grant_type": "refresh_token", "refresh_token": record["refresh_token"],
        "client_id": record["client_id"]})
    if status == 200 and reply.get("access_token"):
        fresh = _record(record["client_id"], reply)
        _save(server, fresh)
        return fresh
    if status in (400, 401) and reply.get("error") in ("invalid_grant", "invalid_client"):
        _save(server, {"client_id": record["client_id"]})
        raise NotSignedIn(f"the sign-in to {server} has ended: {again}")
    raise CredentialsError(f"could not refresh the sign-in to {server} (it answered {status})")
