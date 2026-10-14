"""
Scheduled delivery — distribution with the receipt attached (roadmap #11, v1).

A report that leaves the project must carry its receipt: :func:`send_report`
emails the built ``.html`` **and** its ``.manifest.json`` side by side, with
a plain-text body stating what was rendered, when, from which code, and —
when a verify result is supplied — the receipt-level verdict. The CLI's
``tracebi report send`` refuses to call this at all while the receipt does
not verify (``--force`` overrides, pasting the failing verdict prominently
into the body): distribution never outruns verification, and a red flag
travels WITH the report, never silently.

Stdlib only — ``smtplib``/``email`` for mail, ``urllib`` for Slack.
Configuration is explicit environment variables (the framework never
reads connector URLs implicitly; a delivery endpoint is the same kind
of secret):

    TRACEBI_SMTP_URL         smtp://user:pass@host:port (STARTTLS when the
                             server offers it) or smtps://… (implicit TLS)
    TRACEBI_SMTP_FROM        the From: address
    TRACEBI_SLACK_WEBHOOK    optional text ping after a send. An incoming
                             webhook cannot upload a file.
    TRACEBI_SLACK_BOT_TOKEN  bot token (files:write) for file delivery
    TRACEBI_SLACK_CHANNEL    channel id or name the file is shared to.
                             Both this and the bot token are required.
"""

from __future__ import annotations

import json
import math
import os
import smtplib
import ssl
import urllib.error
import urllib.request
from email.message import EmailMessage
from html.parser import HTMLParser
from pathlib import Path
from typing import Optional, Sequence, Union
from urllib.parse import unquote, urlencode, urlsplit


def _smtp_settings():
    """``(parsed TRACEBI_SMTP_URL, TRACEBI_SMTP_FROM)``.

    Raises ``RuntimeError`` naming the missing variables, or a URL that is
    not ``smtp://`` or ``smtps://``.
    """
    url = os.environ.get("TRACEBI_SMTP_URL")
    sender = os.environ.get("TRACEBI_SMTP_FROM")
    missing = [name for name, value in
               (("TRACEBI_SMTP_URL", url), ("TRACEBI_SMTP_FROM", sender))
               if not value]
    if missing:
        raise RuntimeError(
            "report delivery is not configured: set " + " and ".join(missing)
            + ". TRACEBI_SMTP_URL is smtp://user:pass@host:port (or "
              "smtps:// for implicit TLS); TRACEBI_SMTP_FROM is the "
              "From: address."
        )
    parsed = urlsplit(url)
    if parsed.scheme not in ("smtp", "smtps"):
        raise RuntimeError(
            f"TRACEBI_SMTP_URL must start with smtp:// or smtps://, "
            f"got {parsed.scheme or url!r}"
        )
    return parsed, sender


def _transmit(msg: EmailMessage) -> None:
    """Send *msg* with the configured SMTP endpoint. Sets From when unset."""
    parsed, sender = _smtp_settings()
    if "From" not in msg:
        msg["From"] = sender
    host = parsed.hostname or "localhost"
    port = parsed.port or (465 if parsed.scheme == "smtps" else 25)
    # Verify the server certificate and hostname. Without an explicit context
    # smtplib uses an UNVERIFIED one, so a MITM could present any certificate,
    # terminate TLS, and capture the credentials sent in login().
    ctx = ssl.create_default_context()
    if parsed.scheme == "smtps":
        server = smtplib.SMTP_SSL(host, port, timeout=30, context=ctx)
    else:
        server = smtplib.SMTP(host, port, timeout=30)
    try:
        if parsed.scheme == "smtp":
            try:
                server.starttls(context=ctx)
            except smtplib.SMTPNotSupportedError:
                # No STARTTLS. Sending credentials over the cleartext link
                # would expose them, so refuse when any are set — a truly
                # local no-auth relay still works.
                if parsed.username:
                    raise RuntimeError(
                        "SMTP server does not support STARTTLS; refusing to "
                        "send credentials over an unencrypted connection — use "
                        "an smtps:// URL or a server that offers STARTTLS."
                    )
        if parsed.username:
            server.login(unquote(parsed.username),
                         unquote(parsed.password or ""))
        server.send_message(msg)
    finally:
        try:
            server.quit()
        except Exception:  # noqa: BLE001 — the send already happened or raised
            pass


def send_alert(to: str, subject: str, body: str) -> None:
    """Email one plain-text alert through the same SMTP settings as
    :func:`send_report`. *to* is one address."""
    if not isinstance(to, str) or "@" not in to:
        raise ValueError(f"send_alert: {to!r} is not an email address")
    msg = EmailMessage()
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body if body.endswith("\n") else body + "\n")
    _transmit(msg)


def _contracts_line(contracts: Optional[dict]) -> str:
    """The transform-contracts one-liner for the body — the phase-① join as
    recorded at build, never a claim this module checked anything."""
    if not contracts:
        return "Sink contracts: none recorded"
    counts: dict[str, int] = {}
    for rec in contracts.values():
        status = rec.get("status", "?") if isinstance(rec, dict) else "?"
        counts[status] = counts.get(status, 0) + 1
    joined = ", ".join(f"{n} {status}" for status, n in sorted(counts.items()))
    return f"Sink contracts: {joined}"


def _body(name: str, manifest: dict, verify_result: Optional[dict],
          manifest_name: str) -> str:
    lines: list[str] = []
    if verify_result is not None and verify_result.get("exit_code", 0) != 0:
        # The red flag travels with the report — first thing the reader sees.
        lines += [
            "*** RED FLAG: THIS RECEIPT DID NOT VERIFY ***",
            verify_result.get("verdict_detail", ""),
            "",
        ]
    lines += [
        f"Report: {name}",
        f"Rendered at: {manifest.get('rendered_at', 'unknown')}",
        f"Code (git SHA): {manifest.get('git_sha', 'unknown')}",
    ]
    if verify_result is not None:
        lines.append(f"Verify: {verify_result.get('verdict_detail', '')}")
    lines.append(_contracts_line(manifest.get("transform_contracts")))
    lines += [
        "",
        "The attached HTML is self-contained; its .manifest.json receipt is "
        "attached beside it.",
        f"Re-check it any time with: tracebi verify {manifest_name}",
    ]
    return "\n".join(lines) + "\n"


def send_report(html_path: Union[str, Path], manifest_path: Union[str, Path],
                to: Union[str, Sequence[str]], subject: Optional[str] = None,
                verify_result: Optional[dict] = None) -> list[str]:
    """
    Email the built report ``.html`` with its manifest receipt attached.

    *to* is a list of addresses or one comma-separated string. The body is
    plain text: report name, ``rendered_at``, ``git_sha``, the verify
    verdict line when *verify_result* (a :func:`tracebi.verify.verify_manifest`
    result dict) is passed, and the sink-contracts one-liner. When the
    passed result did not verify, its verdict is pasted prominently at the
    top — the caller decided the red flag travels with the report.

    Reads ``TRACEBI_SMTP_URL`` and ``TRACEBI_SMTP_FROM``; raises a
    ``RuntimeError`` naming exactly the missing variables when unset
    (invariant 4: fail loudly, name the fix). Returns the recipient list.
    """
    html_path = Path(html_path)
    manifest_path = Path(manifest_path)
    if isinstance(to, str):
        to = [a.strip() for a in to.split(",") if a.strip()]
    to = list(to)
    if not to:
        raise ValueError("send_report: no recipients given")

    _, sender = _smtp_settings()

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            manifest = {}
    except (OSError, json.JSONDecodeError):
        manifest = {}
    name = manifest.get("report_name") or html_path.stem

    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = ", ".join(to)
    msg["Subject"] = subject or f"[tracebi] {name}"
    msg.set_content(_body(name, manifest, verify_result, manifest_path.name))
    # BOTH artifacts, side by side: the page and the receipt that stands
    # behind it. The manifest bytes go verbatim — it is the receipt.
    msg.add_attachment(html_path.read_bytes(), maintype="text",
                       subtype="html", filename=html_path.name)
    msg.add_attachment(manifest_path.read_bytes(), maintype="application",
                       subtype="json", filename=manifest_path.name)
    _transmit(msg)
    return to


def slack_notify(webhook_url: str, text: str) -> int:
    """POST *text* as JSON to a Slack incoming webhook; returns the HTTP
    status. Network errors propagate — the caller decides whether a failed
    ping matters (the CLI reports it but keeps the send's exit code)."""
    req = urllib.request.Request(
        webhook_url,
        data=json.dumps({"text": text}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310 — operator-set URL
        return resp.status


_SLACK_API = "https://slack.com/api/"
_HEADLINE_LIMIT = 5
#: Placeholders a value figure shows before the build fills it, and the
#: empty string. None of these is a number the receipt recorded.
_NOT_A_VALUE = frozenset({"", "-", "–", "—"})
_VOID_TAGS = frozenset({
    "area", "base", "br", "col", "embed", "hr", "img", "input",
    "link", "meta", "source", "track", "wbr",
})


def _as_display(value) -> Optional[str]:
    """A value the manifest or the page already shows, or None.

    A number is rendered with ``str`` — no currency, percent, or grouping
    is applied here. Formatting a raw number would be a new presentation
    of it. An empty string and the build's em-dash placeholder are not
    values.
    """
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, str):
        text = " ".join(value.split())
        if text in _NOT_A_VALUE:
            return None
        return text
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return str(value)
    return None


def _recorded_display(fig: dict) -> Optional[str]:
    """The figure's own display text, when the manifest carries one."""
    for key in ("formatted", "formatted_value", "value"):
        if key not in fig:
            continue
        text = _as_display(fig.get(key))
        if text is not None:
            return text
    return None


class _ValueDisplays(HTMLParser):
    """Label and filled text of each ``data-tb-figure="value"`` element.

    The build writes the formatted number into the page (the KPI value
    span, or the figure's own text). This reads that text back. It does
    not format, query, or invent a number.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._depth = 0
        self._open: list[dict] = []
        self.by_id: dict[str, dict] = {}

    def handle_starttag(self, tag, attrs):
        if tag in _VOID_TAGS:
            return
        self._depth += 1
        ad = dict(attrs)
        classes = set((ad.get("class") or "").split())
        if ad.get("data-tb-figure") == "value" and ad.get("id"):
            self._open.append({
                "id": ad["id"], "depth": self._depth,
                "label": [], "value": [], "loose": [],
                "mode": None, "mode_depth": None, "saw_value": False,
            })
            return
        if not self._open:
            return
        cur = self._open[-1]
        if "tb-kpi-label" in classes:
            cur["mode"], cur["mode_depth"] = "label", self._depth
        elif "tb-kpi-value" in classes:
            cur["mode"], cur["mode_depth"] = "value", self._depth
            cur["saw_value"] = True
        elif "tb-kpi-context" in classes:
            cur["mode"], cur["mode_depth"] = "skip", self._depth

    def handle_endtag(self, tag):
        if tag in _VOID_TAGS:
            return
        if self._open:
            cur = self._open[-1]
            if cur["mode_depth"] == self._depth:
                cur["mode"] = None
                cur["mode_depth"] = None
            if cur["depth"] == self._depth:
                self._finish(cur)
                self._open.pop()
        self._depth = max(0, self._depth - 1)

    def handle_data(self, data):
        if not self._open:
            return
        cur = self._open[-1]
        mode = cur["mode"]
        if mode == "label":
            cur["label"].append(data)
        elif mode == "value":
            cur["value"].append(data)
        elif mode == "skip":
            return
        else:
            cur["loose"].append(data)

    def _finish(self, cur: dict) -> None:
        parts = cur["value"] if cur["saw_value"] else cur["loose"]
        self.by_id[cur["id"]] = {
            "label": "".join(cur["label"]).strip(),
            "value": "".join(parts).strip(),
        }


def _headline_label(fig: dict, parsed: dict) -> Optional[str]:
    for candidate in (
        fig.get("label"), parsed.get("label"), fig.get("cell"), fig.get("id"),
    ):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def _headline_lines(manifest: dict, html: str) -> list[str]:
    """Up to five value figures, in manifest order.

    A line is ``label: display``. The display is the formatted text on
    the figure record when the manifest has one, otherwise the text the
    build already wrote into that figure in the HTML. A figure with
    neither is skipped. Charts, tables, and a sixth value figure are
    not included.
    """
    displays = _ValueDisplays()
    if html:
        displays.feed(html)
        displays.close()
    lines: list[str] = []
    for fig in manifest.get("figures") or []:
        if len(lines) >= _HEADLINE_LIMIT:
            break
        if not isinstance(fig, dict) or fig.get("kind") != "value":
            continue
        parsed = displays.by_id.get(fig.get("id")) or {}
        text = _recorded_display(fig)
        if text is None:
            text = _as_display(parsed.get("value"))
        if text is None:
            continue
        label = _headline_label(fig, parsed)
        if not label:
            continue
        lines.append(f"{label}: {text}")
    return lines


def _slack_message(manifest: dict, html: str, verify_result: Optional[dict],
                   fallback_name: str) -> str:
    name = manifest.get("report_name") or fallback_name
    lines = [
        f"Report: {name}",
        f"Rendered at: {manifest.get('rendered_at') or 'unknown'}",
    ]
    if verify_result:
        if verify_result.get("exit_code", 0) != 0:
            lines.insert(0, "*** RED FLAG: THIS RECEIPT DID NOT VERIFY ***")
        detail = (verify_result.get("verdict_detail")
                  or verify_result.get("verdict") or "")
        lines.append(f"Verify: {detail}")
    lines.extend(_headline_lines(manifest, html))
    return "\n".join(lines)


def _slack_request(req: urllib.request.Request, timeout: int, label: str) -> bytes:
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return resp.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Slack {label} failed: HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Slack {label} failed: {exc.reason}") from exc


def _slack_api(token: str, method: str, fields: dict) -> dict:
    req = urllib.request.Request(
        _SLACK_API + method,
        data=urlencode(fields).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )
    raw = _slack_request(req, timeout=30, label=method)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Slack {method} returned a non-JSON body") from exc
    if not isinstance(payload, dict) or not payload.get("ok"):
        err = payload.get("error") if isinstance(payload, dict) else None
        raise RuntimeError(f"Slack {method} failed: {err or 'unknown'}")
    return payload


def _slack_post_bytes(upload_url: str, content: bytes) -> None:
    req = urllib.request.Request(
        upload_url,
        data=content,
        headers={"Content-Type": "application/octet-stream"},
        method="POST",
    )
    _slack_request(req, timeout=60, label="file upload")


def _slack_upload_one(token: str, filename: str, content: bytes) -> str:
    """One file through ``files.getUploadURLExternal``, then the byte POST."""
    got = _slack_api(token, "files.getUploadURLExternal", {
        "filename": filename,
        "length": str(len(content)),
    })
    upload_url = got.get("upload_url")
    file_id = got.get("file_id")
    if not isinstance(upload_url, str) or not upload_url or not file_id:
        raise RuntimeError(
            "Slack files.getUploadURLExternal returned no upload URL")
    _slack_post_bytes(upload_url, content)
    return str(file_id)


def slack_send_report(html_path: Union[str, Path],
                      manifest_path: Union[str, Path], *,
                      channel: str, token: str,
                      verify_result: Optional[dict] = None) -> str:
    """Upload the built HTML and its manifest to a Slack channel.

    Uses the current external-upload flow: ``files.getUploadURLExternal``
    for each file, a raw POST of the bytes to the returned URL, then one
    ``files.completeUploadExternal`` that shares both files with the
    summary as ``initial_comment``. Stdlib ``urllib`` only.

    *channel* is an id (``C…``) or a name; a leading ``#`` is removed.
    Slack's ``channel_id`` field receives that value. *token* is a bot
    token with ``files:write``.

    The summary is the report name, ``rendered_at``, the verify verdict
    line when *verify_result* is passed, and up to five headline value
    figures. A figure is included only when a displayable value is
    already on its manifest record or already written into the HTML.
    This function never formats or invents a number.

    Returns the message text. Network and Slack API errors raise
    ``RuntimeError``. The caller records them; nothing here retries.
    """
    token = (token or "").strip()
    channel_id = (channel or "").strip()
    if channel_id.startswith("#"):
        channel_id = channel_id[1:].strip()
    if not token:
        raise ValueError("slack_send_report: token is required")
    if not channel_id:
        raise ValueError("slack_send_report: channel is required")

    html_path = Path(html_path)
    manifest_path = Path(manifest_path)
    html_bytes = html_path.read_bytes()
    manifest_bytes = manifest_path.read_bytes()
    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
        if not isinstance(manifest, dict):
            manifest = {}
    except (UnicodeDecodeError, json.JSONDecodeError):
        manifest = {}
    text = _slack_message(
        manifest, html_bytes.decode("utf-8", "replace"),
        verify_result, html_path.stem)
    html_id = _slack_upload_one(token, html_path.name, html_bytes)
    manifest_id = _slack_upload_one(token, manifest_path.name, manifest_bytes)
    _slack_api(token, "files.completeUploadExternal", {
        "files": json.dumps([
            {"id": html_id, "title": html_path.name},
            {"id": manifest_id, "title": manifest_path.name},
        ]),
        "channel_id": channel_id,
        "initial_comment": text,
    })
    return text
