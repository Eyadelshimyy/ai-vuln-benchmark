"""Materialize user-attached images from conversation events into intake/attachments.

Scans the agent-canvas event store
(~/.openhands/agent-canvas/dev_conversations/<session_id>/events/event-*.json)
for `source=user` messages carrying image content, decodes each `data:` image
to `<workspace>/intake/attachments/<sha256[:12]>.<ext>`, and appends a
provenance record to `manifest.jsonl` next to the files. Runs on
session_start, user_prompt_submit, and stop; idempotent via the manifest and
a `.processed` marker of already-scanned event files. Always exits 0 — when
the events directory is unreachable (remote/docker runtimes) the fallback is
to drop files into `intake/` manually.
"""

from __future__ import annotations

import base64
import binascii
import contextlib
import hashlib
import json
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _provenance import project_dir

EVENTS_DIR_ENV = "FPGA_AGENT_EVENTS_DIR"
ATTACHMENTS_ENV = "FPGA_INTAKE_ATTACHMENTS_DIR"
DEFAULT_EVENTS_ROOT = Path(".openhands/agent-canvas/dev_conversations")
ATTACHMENTS_RELATIVE = Path("intake/attachments")
PROCESSED_MARKER = ".processed"

_MIME_EXT = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
}
_DATA_URL = re.compile(r"^data:(image/[a-zA-Z0-9.+-]+);base64,(.+)$", re.DOTALL)
_MAGIC_EXT = (
    (b"\x89PNG\r\n\x1a\n", ".png", "image/png"),
    (b"\xff\xd8\xff", ".jpg", "image/jpeg"),
    (b"GIF8", ".gif", "image/gif"),
    (b"RIFF", ".webp", "image/webp"),
)


def _events_dir(payload: dict[str, Any]) -> Path | None:
    override = os.environ.get(EVENTS_DIR_ENV)
    if override:
        candidate = Path(override).expanduser()
        return candidate if candidate.is_dir() else None
    session_id = payload.get("session_id")
    if not isinstance(session_id, str) or not session_id.strip():
        return None
    candidate = Path.home() / DEFAULT_EVENTS_ROOT / session_id.strip() / "events"
    return candidate if candidate.is_dir() else None


def _attachments_dir(payload: dict[str, Any]) -> Path:
    override = os.environ.get(ATTACHMENTS_ENV)
    if override:
        path = Path(override).expanduser()
        return path if path.is_absolute() else project_dir(payload) / path
    return project_dir(payload) / ATTACHMENTS_RELATIVE


def _image_blocks(value: Any) -> list[dict[str, Any]]:
    """Collect image content blocks from a serialized event (any nesting)."""
    blocks: list[dict[str, Any]] = []
    if isinstance(value, dict):
        record = cast(dict[str, Any], value)
        if record.get("type") == "image" and isinstance(record.get("image_urls"), list):
            blocks.append(record)
        else:
            for child in record.values():
                blocks.extend(_image_blocks(child))
    elif isinstance(value, list):
        for child in cast(list[Any], value):
            blocks.extend(_image_blocks(child))
    return blocks


def _decode_image(url: str) -> tuple[bytes, str, str] | None:
    """Decode a data: URL or bare base64 string to (bytes, ext, mime)."""
    match = _DATA_URL.match(url.strip())
    if match:
        mime, encoded = match.group(1).lower(), match.group(2)
    else:
        mime, encoded = "", url.strip()
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        return None
    for magic, ext, detected in _MAGIC_EXT:
        if data.startswith(magic):
            return data, ext, detected
    if mime in _MIME_EXT:
        return data, _MIME_EXT[mime], mime
    return None


def _scan_event_file(path: Path) -> list[dict[str, Any]]:
    try:
        value: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return []
    if not isinstance(value, dict) or cast(dict[str, Any], value).get("source") != "user":
        return []
    urls: list[str] = []
    for block in _image_blocks(cast(dict[str, Any], value)):
        for url in block["image_urls"]:
            if isinstance(url, str):
                urls.append(url)
    return [{"event_file": path.name, "image_urls": urls}] if urls else []


def _load_manifest(manifest: Path) -> set[str]:
    seen: set[str] = set()
    try:
        for line in manifest.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                record = cast(dict[str, Any], record)
                if isinstance(record.get("sha256"), str):
                    seen.add(record["sha256"])
    except OSError:
        pass
    return seen


def main() -> int:
    try:
        payload: Any = json.load(sys.stdin)
    except (json.JSONDecodeError, OSError):
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    payload = cast(dict[str, Any], payload)
    events_dir = _events_dir(payload)
    if events_dir is None:
        return 0
    out_dir = _attachments_dir(payload)
    marker = out_dir / PROCESSED_MARKER
    processed: set[str] = set()
    with contextlib.suppress(OSError):
        processed = set(marker.read_text(encoding="utf-8").split())
    manifest = out_dir / "manifest.jsonl"
    seen_sha = _load_manifest(manifest)
    pending = [
        entry
        for path in sorted(events_dir.glob("event-*.json"))
        if path.name not in processed
        for entry in _scan_event_file(path)
    ]
    scanned = [
        path.name for path in sorted(events_dir.glob("event-*.json")) if path.name not in processed
    ]
    if not pending and not scanned:
        return 0
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        records: list[dict[str, Any]] = []
        for entry in pending:
            for index, url in enumerate(entry["image_urls"]):
                decoded = _decode_image(url)
                record: dict[str, Any] = {
                    "event_file": entry["event_file"],
                    "image_index": index,
                    "recorded_at": datetime.now(UTC).isoformat(),
                }
                if decoded is None:
                    record["materialized"] = False
                    record["reason"] = (
                        "non-data-url" if not _DATA_URL.match(url.strip()) else "undecodable"
                    )
                    record["url_prefix"] = url[:80]
                else:
                    data, ext, mime = decoded
                    sha256 = hashlib.sha256(data).hexdigest()
                    record.update(
                        {
                            "materialized": True,
                            "sha256": sha256,
                            "mime": mime,
                            "bytes": len(data),
                        }
                    )
                    if sha256 in seen_sha:
                        record["duplicate"] = True
                    else:
                        image_path = out_dir / f"{sha256[:12]}{ext}"
                        image_path.write_bytes(data)
                        record["image_path"] = str(image_path)
                        seen_sha.add(sha256)
                records.append(record)
        with manifest.open("a", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
                stream.write("\n")
        with marker.open("a", encoding="utf-8") as stream:
            for name in scanned:
                stream.write(name + "\n")
    except OSError as exc:
        print(f"intake attachments unavailable: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
