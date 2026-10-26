"""Park hard job URLs for later ingest (phone-friendly; no CAPTCHA bypass).

From a phone you can save the share link without fetching Indeed. On a desktop
with Chrome CDP, drain the inbox and complete challenges by hand if shown.
"""

from __future__ import annotations

from pathlib import Path

from jobbot.config import JobbotConfig
from jobbot.jobs.indeed_url import IndeedUrlError, canonical_indeed_job_url
from jobbot.portals.detect import AtsKind, detect_ats

INBOX_FILENAME = "hard-link-inbox.txt"


def inbox_path(config: JobbotConfig) -> Path:
    """Plain-text queue under the project data dir (one URL per line)."""
    return config.root / "data" / INBOX_FILENAME


def normalize_park_url(url: str) -> str:
    """Strip tracking when we know how; otherwise keep a trimmed http(s) URL."""
    raw = url.strip()
    if not raw:
        raise ValueError("Empty URL")
    # Accidental double-share paste (iOS): same URL glued twice with no separator.
    second_https = raw.find("https://", 8)
    second_http = raw.find("http://", 7)
    cut = min(x for x in (second_https, second_http) if x > 0) if (
        second_https > 0 or second_http > 0
    ) else -1
    if cut > 0:
        raw = raw[:cut]
    if "://" not in raw:
        raise ValueError(f"Not an absolute URL: {raw}")
    if detect_ats(raw) == AtsKind.INDEED:
        try:
            return canonical_indeed_job_url(raw)
        except IndeedUrlError as exc:
            raise ValueError(str(exc)) from exc
    return raw


def list_parked(config: JobbotConfig) -> list[str]:
    path = inbox_path(config)
    if not path.is_file():
        return []
    urls: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text and not text.startswith("#"):
            urls.append(text)
    return urls


def park_url(config: JobbotConfig, url: str) -> tuple[str, bool]:
    """Append a normalized URL. Returns (stored_url, already_present)."""
    stored = normalize_park_url(url)
    path = inbox_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = list_parked(config)
    if stored in existing:
        return stored, True
    with path.open("a", encoding="utf-8") as fh:
        fh.write(stored + "\n")
    return stored, False


def remove_parked(config: JobbotConfig, url: str) -> bool:
    """Remove one URL from the inbox. Returns whether it was present."""
    path = inbox_path(config)
    if not path.is_file():
        return False
    try:
        target = normalize_park_url(url)
    except ValueError:
        target = url.strip()
    kept: list[str] = []
    removed = False
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        if text.startswith("#"):
            kept.append(text)
            continue
        if not removed and (text == url.strip() or text == target):
            removed = True
            continue
        kept.append(text)
    path.write_text(("\n".join(kept) + ("\n" if kept else "")), encoding="utf-8")
    return removed
