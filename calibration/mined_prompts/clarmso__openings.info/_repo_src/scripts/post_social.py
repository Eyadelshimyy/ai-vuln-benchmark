#!/usr/bin/env python3
"""
Post a daily digest of newly-discovered jobs to Mastodon and Bluesky.

Reads docs/new_jobs.json (written by generate_new_jobs.py earlier in the
same crawl run) and compares every job URL against state/posted_new_jobs.json
— a persisted set of job URLs already announced on social media. Only jobs
not seen in a prior run are included in the digest, since new_jobs.json
itself is a 48h rolling window (LOOKBACK_HOURS) that re-lists jobs already
announced yesterday.

If there's nothing new to announce, this script exits quietly without
posting (no "0 new jobs today" spam).

Designed to fail soft per platform: missing/invalid credentials for either
Mastodon or Bluesky just skip that platform (logged, not fatal) so a
misconfigured or revoked token never breaks the crawl workflow. If BOTH
platforms fail (or neither is configured), newly-found jobs are NOT marked
as announced, so they're retried on the next run. If at least one platform
succeeds, all of today's new jobs are marked announced — a job already
announced on Mastodon won't be silently skipped if only Bluesky fails
today, but it also won't be re-announced there tomorrow.

Disable entirely by setting DISABLE_SOCIAL_POST=1 in the environment.
"""

import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

NEW_JOBS_JSON = "docs/new_jobs.json"
STATE_PATH = os.path.join("state", "posted_new_jobs.json")
SITE_URL = "https://openings.info/"
MAX_MESSAGE_LEN = 300  # Bluesky's limit; comfortably under Mastodon's 500 too.
PRUNE_AFTER_DAYS = 7  # Well above the 48h lookback window; bounds file growth.


def _load_state():
    try:
        with open(STATE_PATH) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _save_state(state):
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2, sort_keys=True)
        f.write("\n")


def _prune_state(state, now):
    cutoff = now - timedelta(days=PRUNE_AFTER_DAYS)
    pruned = {}
    for url, iso_date in state.items():
        try:
            seen_at = datetime.strptime(iso_date, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if seen_at >= cutoff:
            pruned[url] = iso_date
    return pruned


def _find_new_jobs(manifest, already_announced):
    """Return (new_job_urls, company_names_with_new_jobs), preserving the
    manifest's company order."""
    new_job_urls = []
    company_names = []
    for company in manifest.get("companies", []):
        has_new = False
        for job in company.get("jobs", []):
            url = job.get("url")
            if url and url not in already_announced:
                new_job_urls.append(url)
                has_new = True
        if has_new:
            company_names.append(company.get("name") or company.get("slug", "unknown"))
    return new_job_urls, company_names


def _build_message(job_count, company_names, url, max_len=MAX_MESSAGE_LEN):
    prefix = f"{job_count} new #job{'s' if job_count != 1 else ''} today:\n"
    suffix = f"\n{url}"
    budget = max_len - len(prefix) - len(suffix)

    included = []
    remaining = list(company_names)
    while remaining:
        candidate = ", ".join(included + [remaining[0]])
        more_after = len(remaining) - 1
        trailer = f" +{more_after} more" if more_after else ""
        if len(candidate) + len(trailer) <= budget:
            included.append(remaining.pop(0))
        else:
            break

    more_count = len(remaining)
    trailer = f" +{more_count} more" if more_count else ""
    body = ", ".join(included) + trailer
    return f"{prefix}{body}{suffix}"


def _post_mastodon(message):
    instance = os.environ.get("MASTODON_INSTANCE_URL")
    token = os.environ.get("MASTODON_ACCESS_TOKEN")
    if not instance or not token:
        print("  [post_social] Mastodon credentials not configured; skipping")
        return False

    url = instance.rstrip("/") + "/api/v1/statuses"
    data = json.dumps({"status": message}).encode()
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            resp.read()
        print("  [post_social] posted to Mastodon")
        return True
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        print(f"  [post_social] Mastodon post failed: {e}")
        return False


def _bluesky_facets(text, url):
    """Build an AT Protocol rich-text facet so the link is clickable."""
    idx = text.find(url)
    if idx == -1:
        return []
    byte_start = len(text[:idx].encode("utf-8"))
    byte_end = byte_start + len(url.encode("utf-8"))
    return [{
        "index": {"byteStart": byte_start, "byteEnd": byte_end},
        "features": [{"$type": "app.bsky.richtext.facet#link", "uri": url}],
    }]


def _bluesky_create_session(handle, app_password):
    body = json.dumps({"identifier": handle, "password": app_password}).encode()
    req = urllib.request.Request(
        "https://bsky.social/xrpc/com.atproto.server.createSession",
        data=body, method="POST", headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def _post_bluesky(message, url):
    handle = os.environ.get("BLUESKY_HANDLE")
    app_password = os.environ.get("BLUESKY_APP_PASSWORD")
    if not handle or not app_password:
        print("  [post_social] Bluesky credentials not configured; skipping")
        return False

    try:
        session = _bluesky_create_session(handle, app_password)
        record = {
            "$type": "app.bsky.feed.post",
            "text": message,
            "createdAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "facets": _bluesky_facets(message, url),
        }
        body = json.dumps({
            "repo": session["did"],
            "collection": "app.bsky.feed.post",
            "record": record,
        }).encode()
        req = urllib.request.Request(
            "https://bsky.social/xrpc/com.atproto.repo.createRecord",
            data=body, method="POST",
            headers={"Authorization": f"Bearer {session['accessJwt']}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            resp.read()
        print("  [post_social] posted to Bluesky")
        return True
    except (urllib.error.URLError, urllib.error.HTTPError, KeyError, json.JSONDecodeError) as e:
        print(f"  [post_social] Bluesky post failed: {e}")
        return False


def main():
    if os.environ.get("DISABLE_SOCIAL_POST"):
        print("  [post_social] DISABLE_SOCIAL_POST set; skipping")
        return

    try:
        with open(NEW_JOBS_JSON) as f:
            manifest = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"  [post_social] couldn't read {NEW_JOBS_JSON}: {e}")
        return

    state = _load_state()
    new_job_urls, company_names = _find_new_jobs(manifest, state)

    if not new_job_urls:
        print("  [post_social] no newly-discovered jobs since last post; skipping")
        return

    message = _build_message(len(new_job_urls), company_names, SITE_URL)
    print(f"  [post_social] posting ({len(message)} chars): {message}")

    posted_mastodon = _post_mastodon(message)
    posted_bluesky = _post_bluesky(message, SITE_URL)

    if not (posted_mastodon or posted_bluesky):
        print("  [post_social] both platforms failed or are unconfigured; "
              "not marking jobs as announced so they're retried next run")
        return

    now = datetime.now(timezone.utc)
    iso_now = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    for job_url in new_job_urls:
        state[job_url] = iso_now
    _save_state(_prune_state(state, now))


if __name__ == "__main__":
    main()
