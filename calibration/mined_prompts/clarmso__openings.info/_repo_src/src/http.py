"""Shared HTTP utilities."""

import json
import time
import urllib.error
import urllib.request

USER_AGENT = "job-board-crawler/1.0"

# Retry config
_MAX_RETRIES = 4
_BACKOFF_BASE = 2  # seconds; wait = BACKOFF_BASE ** attempt → 2s, 4s, 8s, 16s


def _fetch(url, timeout=30):
    """Make an HTTP GET request with retry-with-backoff.

    Retries on:
    - HTTP 429 Too Many Requests: honours the Retry-After response header when
      present; otherwise falls back to exponential backoff.
    - Transient network errors (timeouts, connection resets, etc.): exponential
      backoff.

    Non-retryable HTTP errors (4xx except 429, 5xx) are re-raised immediately.
    Raises the last exception if all retries are exhausted.
    """
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_err = None
    for attempt in range(_MAX_RETRIES):
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                retry_after = e.headers.get("Retry-After", "")
                wait = (
                    int(retry_after)
                    if retry_after.strip().isdigit()
                    else _BACKOFF_BASE ** attempt
                )
                print(
                    f"    Rate limited (429). Waiting {wait}s "
                    f"(retry {attempt + 1}/{_MAX_RETRIES})..."
                )
                time.sleep(wait)
                last_err = e
            else:
                raise
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            wait = _BACKOFF_BASE ** attempt
            print(
                f"    Transient error: {e}. Waiting {wait}s "
                f"(retry {attempt + 1}/{_MAX_RETRIES})..."
            )
            time.sleep(wait)
            last_err = e
    raise last_err


def fetch_json(url):
    with _fetch(url) as resp:
        return json.loads(resp.read().decode())


def fetch_html(url):
    with _fetch(url) as resp:
        return resp.read().decode()


def fetch_xml(url):
    with _fetch(url) as resp:
        return resp.read().decode()
