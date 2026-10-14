#!/usr/bin/env python3
"""
Generate docs/index.html — a fully static page listing jobs actually posted
within the past LOOKBACK_HOURS (per the ATS's own posting timestamp, not
when we happened to crawl them), grouped by company, for the GitHub Pages
site. Also writes docs/new_jobs.json as a raw data snapshot.

Scans every job file currently under data/ and keeps only those whose
ATS-reported posting/publish timestamp falls within the lookback window, so
it doesn't matter how many times crawl.yml ran or when — "new" always
reflects the job's real posting date.
"""

import argparse
import csv
import html
import json
import os
import re
import sys
import urllib.parse
from datetime import datetime, timedelta, timezone

import ai_canada

DATA_DIR = "data"
LOOKBACK_HOURS = 48

_CA_REMOTE_RE = re.compile(r"\bca\s*[-\u2013\u2014]?\s*remote\b|\bremote\s*[-\u2013\u2014]?\s*ca\b")

# Canadian province/territory postal abbreviations, as they appear after a
# city name in free-text location strings (e.g. "Kitchener-Waterloo, ON").
# Deliberately case-sensitive and requires a preceding comma: matching case-
# insensitively would false-positive on the common English word "on".
_CA_PROVINCE_RE = re.compile(
    r",\s*(ON|BC|QC|AB|MB|SK|NS|NB|PE|NL|YT|NT|NU)\b"
)


def _mentions_canada_explicit(*texts):
    """Keyword match for locations that explicitly reference Canada.

    Matches "Canada" itself, a Canadian province/territory abbreviation
    following a city name (e.g. "Kitchener-Waterloo, ON; Toronto, ON"), and
    the "CA Remote" / "Remote - CA" shorthand some companies use.

    A bare "CA" token on its own is deliberately NOT treated as Canada:
    that's ambiguous with the US postal abbreviation for California, which
    shows up constantly in greenhouse/gem/rippling location strings (e.g.
    "San Mateo, CA"). Only "CA" directly adjacent to "Remote" is matched,
    since that combination is how some ATS postings abbreviate
    "Canada Remote".
    """
    raw = " ".join(str(t) for t in texts if t)
    if not raw:
        return False
    if _CA_PROVINCE_RE.search(raw):
        return True
    if "canada" in raw.lower():
        return True
    return bool(_CA_REMOTE_RE.search(raw.lower()))


def _mentions_canada_broad(*texts):
    """Keyword match for broader regions a Canada-based candidate could
    reasonably apply under ("Global", "North America", "Americas"), but
    which aren't themselves an explicit Canada reference. When a posting
    with one of these is *also* remote, it's remote worldwide/regionally —
    not Canada-specific — so callers should not tag it as Canada in that
    case (see _is_canada_job).
    """
    raw = " ".join(str(t) for t in texts if t)
    if not raw:
        return False
    combined = raw.lower()
    return any(keyword in combined for keyword in ("global", "north america", "americas"))


def _mentions_canada(*texts):
    """Combined explicit-or-broad match, for callers that don't need to
    distinguish the two tiers (e.g. company-level "could a Canada-based
    candidate apply here at all" eligibility)."""
    return _mentions_canada_explicit(*texts) or _mentions_canada_broad(*texts)


def _is_canada_job(platform, job, is_remote=False):
    """Best-effort: should this specific opening be tagged Canada?

    An explicit Canada reference (city, province, "Canada", "Canada Remote")
    always counts. A broader regional reference ("Global", "North America",
    "Americas") only counts when the posting is *not* remote — once it's
    also remote, it reads as remote-worldwide/regional rather than
    Canada-specific, so it should get the Remote tag instead, not Canada.

    Kept in sync with generate_companies_page.py's _is_canada_job.
    """
    if platform == "greenhouse":
        texts = [(job.get("location") or {}).get("name")]
        return _mentions_canada_explicit(*texts) or (
            _mentions_canada_broad(*texts) and not is_remote
        )

    if platform == "lever":
        if str(job.get("country") or "").strip().lower() == "ca":
            return True
        categories = job.get("categories") or {}
        texts = [categories.get("location"), *(categories.get("allLocations") or [])]
        return _mentions_canada_explicit(*texts) or (
            _mentions_canada_broad(*texts) and not is_remote
        )

    if platform == "ashby":
        address = ((job.get("address") or {}).get("postalAddress") or {})
        texts = [address.get("addressCountry"), job.get("location")]
        for secondary in job.get("secondaryLocations") or []:
            secondary_address = ((secondary.get("address") or {}).get("postalAddress") or {})
            texts.extend([secondary.get("location"), secondary_address.get("addressCountry")])
        return _mentions_canada_explicit(*texts) or (
            _mentions_canada_broad(*texts) and not is_remote
        )

    if platform == "workable":
        if str(job.get("country") or "").strip().lower() == "canada":
            return True
        for loc in job.get("locations") or []:
            if str(loc.get("country") or "").strip().lower() == "canada" or str(loc.get("countryCode") or "").strip().lower() == "ca":
                return True
        location_texts = [job.get("country"), job.get("city"), job.get("state")]
        for loc in job.get("locations") or []:
            location_texts.extend([loc.get("country"), loc.get("city"), loc.get("region")])
        return _mentions_canada_explicit(*location_texts) or (
            _mentions_canada_broad(*location_texts) and not is_remote
        )

    if platform == "smartrecruiters":
        loc = job.get("location") or {}
        if str(loc.get("country") or "").strip().lower() == "ca":
            return True
        texts = [loc.get("fullLocation"), loc.get("city")]
        return _mentions_canada_explicit(*texts) or (
            _mentions_canada_broad(*texts) and not is_remote
        )

    if platform == "gem":
        texts = [(job.get("location") or {}).get("name")]
        return _mentions_canada_explicit(*texts) or (
            _mentions_canada_broad(*texts) and not is_remote
        )

    if platform == "rippling":
        texts = list(job.get("workLocations") or [])
        return _mentions_canada_explicit(*texts) or (
            _mentions_canada_broad(*texts) and not is_remote
        )

    return False


CAREERS_URL_BY_PLATFORM = {
    "greenhouse": "https://job-boards.greenhouse.io/{slug}",
    "lever": "https://jobs.lever.co/{slug}",
    "ashby": "https://jobs.ashbyhq.com/{slug}",
    "workable": "https://apply.workable.com/{slug}/",
    "recruitee": "https://{slug}.recruitee.com/",
    "smartrecruiters": "https://careers.smartrecruiters.com/{slug}/",
    "gem": "https://jobs.gem.com/{slug}",
    "rippling": "https://ats.rippling.com/{slug}/jobs",
}


def _careers_url(slug, platform):
    template = CAREERS_URL_BY_PLATFORM.get(platform)
    return template.format(slug=slug) if template else None


def _load_ats_by_slug(companies_csv="companies.csv"):
    ats_by_slug = {}
    with open(companies_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("slug") and row.get("ats"):
                ats_by_slug[row["slug"]] = row["ats"]
    return ats_by_slug


def _parse_timestamp(raw):
    """Best-effort parse of an ATS-provided timestamp into an aware UTC datetime.

    Handles ISO-8601 strings (with or without timezone, with or without
    fractional seconds), epoch milliseconds (int/numeric string), and plain
    "YYYY-MM-DD" date strings.
    """
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        # Epoch milliseconds (lever, ashby-style numeric timestamps).
        try:
            return datetime.fromtimestamp(raw / 1000.0, tz=timezone.utc)
        except (OSError, OverflowError, ValueError):
            return None

    text = str(raw).strip()
    if not text:
        return None

    if text.isdigit():
        try:
            return datetime.fromtimestamp(int(text) / 1000.0, tz=timezone.utc)
        except (OSError, OverflowError, ValueError):
            return None

    # datetime.fromisoformat doesn't accept a trailing "Z" before 3.11.
    normalized = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        try:
            parsed = datetime.strptime(text, "%Y-%m-%d")
        except ValueError:
            return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _posted_at(platform, job):
    """Best-effort: when was this specific opening actually posted/published,
    according to the ATS itself (not when we happened to crawl it)?

    Returns an aware UTC datetime, or None if no usable timestamp field is
    present (in which case the job is treated as not-new, to avoid
    incorrectly surfacing stale postings).
    """
    if platform == "greenhouse":
        return _parse_timestamp(job.get("first_published") or job.get("updated_at"))
    if platform == "lever":
        return _parse_timestamp(job.get("createdAt"))
    if platform == "ashby":
        return _parse_timestamp(job.get("publishedAt"))
    if platform == "workable":
        return _parse_timestamp(job.get("published_on") or job.get("created_at"))
    if platform == "smartrecruiters":
        return _parse_timestamp(job.get("releasedDate"))
    if platform == "gem":
        return _parse_timestamp(job.get("first_published_at") or job.get("created_at"))
    if platform == "rippling":
        return _parse_timestamp(job.get("createdOn"))
    return None


def _all_data_files():
    """Return paths (relative to repo root) of every job file currently
    under data/. Filtering to "new" happens later, based on each job's own
    ATS-reported posting timestamp rather than when the file landed in
    data/."""
    matches = []
    for root, _dirs, files in os.walk(DATA_DIR):
        for name in files:
            if name.endswith(".json"):
                matches.append(os.path.join(root, name))
    return sorted(matches)


def _mode_from_text(text):
    """Best-effort guess of workplace mode from a free-text location string."""
    if not text:
        return ""
    lowered = text.lower()
    if "remote" in lowered:
        return "Remote"
    if "hybrid" in lowered:
        return "Hybrid"
    return "On-site"


def _normalize_mode(raw):
    """Normalize an ATS-provided workplace-type value to Remote/Hybrid/On-site."""
    if not raw:
        return ""
    lowered = str(raw).lower()
    if "remote" in lowered:
        return "Remote"
    if "hybrid" in lowered:
        return "Hybrid"
    if lowered in ("onsite", "on-site", "office"):
        return "On-site"
    return str(raw).title()


def _extract(platform, job):
    """Return (title, url, company_name, location, mode) normalized across ATS platforms.

    `location` is a human-readable string and `mode` is one of
    "Remote" / "Hybrid" / "On-site" (or "" when unknown).
    """
    if platform == "greenhouse":
        location = (job.get("location") or {}).get("name", "")
        return (
            job.get("title"), job.get("absolute_url"), job.get("company_name"),
            location, _mode_from_text(location),
        )
    if platform == "lever":
        categories = job.get("categories") or {}
        location = categories.get("location") or ", ".join(categories.get("allLocations") or [])
        mode = _normalize_mode(job.get("workplaceType")) or _mode_from_text(location)
        return job.get("text"), job.get("hostedUrl"), None, location, mode
    if platform == "ashby":
        location = job.get("location", "")
        mode = _normalize_mode(job.get("workplaceType")) or _mode_from_text(location)
        return job.get("title"), job.get("jobUrl"), None, location, mode
    if platform == "workable":
        location = ", ".join(part for part in (job.get("city"), job.get("state"), job.get("country")) if part)
        mode = "Remote" if job.get("telecommuting") else _mode_from_text(location) or "On-site"
        return job.get("title"), job.get("url"), None, location, mode
    if platform == "smartrecruiters":
        company = job.get("company", {})
        identifier = company.get("identifier")
        url = f"https://jobs.smartrecruiters.com/{identifier}/{job.get('id')}" if identifier else None
        loc = job.get("location") or {}
        location = loc.get("fullLocation") or loc.get("city", "")
        if loc.get("remote"):
            mode = "Remote"
        elif loc.get("hybrid"):
            mode = "Hybrid"
        else:
            mode = "On-site"
        return job.get("name"), url, company.get("name"), location, mode
    if platform == "gem":
        location = (job.get("location") or {}).get("name", "")
        mode = _normalize_mode(job.get("location_type")) or _mode_from_text(location)
        return job.get("title"), job.get("absolute_url"), None, location, mode
    if platform == "rippling":
        location = ", ".join(job.get("workLocations") or [])
        return job.get("name"), job.get("url"), job.get("companyName"), location, _mode_from_text(location)
    return None, None, None, "", ""


def _display_name(slug, extracted_name):
    if extracted_name:
        return extracted_name
    # Slugs can be URL-encoded (e.g. Ashby org names with spaces, stored as
    # "Superhuman%20Platform%20Inc" so the crawler builds a valid URL);
    # decode for display purposes until real job data supplies a sample_name.
    decoded = urllib.parse.unquote(slug)
    return decoded.replace("-", " ").replace("_", " ").title()


def _apply_ai_canada_classification(companies):
    """Override each job's regex-based "canada" flag with Copilot CLI's
    classification of its location string, wherever the AI has an opinion
    (see ai_canada.py). Jobs whose location wasn't classified — e.g. the
    CLI isn't available — keep their existing regex-based result.
    """
    all_locations = [
        job["location"] for data in companies.values() for job in data["jobs"]
    ]
    classifications = ai_canada.classify(all_locations)
    if not classifications:
        return
    for data in companies.values():
        for job in data["jobs"]:
            if job["location"] in classifications:
                job["canada"] = classifications[job["location"]]


def _build_manifest():
    ats_by_slug = _load_ats_by_slug()
    all_files = _all_data_files()
    cutoff = datetime.now(timezone.utc) - timedelta(hours=LOOKBACK_HOURS)

    companies = {}  # slug -> {"name": str, "jobs": [...]}

    for path in all_files:
        parts = path.split(os.sep)
        if len(parts) < 3 or parts[0] != DATA_DIR:
            continue
        slug = parts[1]
        platform = ats_by_slug.get(slug)
        if not platform:
            continue

        try:
            with open(path) as f:
                job = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue

        posted_at = _posted_at(platform, job)
        if not posted_at or posted_at < cutoff:
            continue

        title, url, name, location, mode = _extract(platform, job)
        if not title or not url:
            continue

        entry = companies.setdefault(
            slug, {"name": _display_name(slug, name), "careers_url": _careers_url(slug, platform), "jobs": []}
        )
        entry["jobs"].append({
            "title": title, "url": url, "location": location, "mode": mode,
            "canada": _is_canada_job(platform, job, is_remote=(mode == "Remote")),
        })

    _apply_ai_canada_classification(companies)

    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_new_jobs": sum(len(c["jobs"]) for c in companies.values()),
        "companies": [
            {
                "slug": slug, "name": data["name"], "careers_url": data["careers_url"],
                "jobs": sorted(data["jobs"], key=lambda j: j["title"]),
            }
            for slug, data in sorted(companies.items(), key=lambda kv: kv[1]["name"].lower())
        ],
    }


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>openings.info - New jobs in tech</title>
<link rel="icon" type="image/svg+xml" href="favicon.svg" />
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link href="https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:ital,wght@0,400;0,700;1,400;1,700&family=Inter:ital,wght@0,400;0,500;0,700;1,400&display=swap" rel="stylesheet" />
<style>
  :root {{
    --border: var(--accent-vermillion);
    --muted: #6b7280;
    --accent: #2563eb;
    --accent-vermillion: #d55e00;
    --bg: #ffffff;
    --brand: var(--accent-vermillion);
    --font-brand: "Atkinson Hyperlegible", "Inter", sans-serif;
    --font-body: "Inter", "Atkinson Hyperlegible", sans-serif;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: var(--font-body);
    max-width: 900px;
    margin: 0 auto;
    padding: 2rem 1.25rem 4rem;
    color: #111827;
    background: var(--bg);
  }}
  #brand {{
    display: inline-block;
    font-family: var(--font-brand);
    font-weight: 700;
    font-size: 1.1rem;
    color: var(--brand);
    text-decoration: none;
    margin-bottom: 0.75rem;
  }}
  header h1 {{
    margin: 0 0 0.25rem;
    font-size: 1.75rem;
  }}
  #meta {{
    color: var(--muted);
    font-size: 0.9rem;
    margin-bottom: 0.75rem;
  }}
  #site-footer {{
    margin-top: 3rem;
    padding-top: 1.25rem;
    border-top: 1px solid var(--border);
    color: var(--muted);
    font-size: 0.85rem;
  }}
  #footer-nav {{
    margin-bottom: 0.5rem;
    font-size: 0.9rem;
  }}
  #footer-nav a, #footer-credit a {{
    color: var(--accent);
    text-decoration: none;
  }}
  #footer-nav a:hover, #footer-credit a:hover {{ text-decoration: underline; }}
  #toc {{
    background: #f9fafb;
    border: 2px solid var(--accent-vermillion);
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin-bottom: 2rem;
  }}
  #toc h2 {{
    font-size: 0.95rem;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    color: #111827;
    margin: 0 0 0.5rem;
  }}
  #toc ul {{
    list-style: none;
    padding: 0;
    margin: 0;
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem 0.75rem;
  }}
  #toc a {{
    color: var(--accent);
    text-decoration: none;
    font-size: 0.95rem;
  }}
  #toc a:hover {{ text-decoration: underline; }}
  .company {{
    margin-bottom: 2rem;
    scroll-margin-top: 1rem;
  }}
  .company h2 {{
    font-size: 1.2rem;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.4rem;
    margin-bottom: 0.6rem;
  }}
  .company h2 .count {{
    color: var(--muted);
    font-weight: normal;
    font-size: 0.9rem;
  }}
  .company h2 a.company-link {{
    color: inherit;
    text-decoration: none;
  }}
  .company h2 a.company-link:hover {{ color: var(--accent); text-decoration: underline; }}
  .company ul {{
    list-style: none;
    padding: 0;
    margin: 0;
  }}
  .company li {{
    padding: 0.4rem 0;
  }}
  .company a {{
    color: #111827;
    text-decoration: none;
  }}
  .company a:hover {{ color: var(--accent); text-decoration: underline; }}
  .job-meta {{
    color: var(--muted);
    font-size: 0.85rem;
    margin-left: 0.5rem;
    white-space: nowrap;
  }}
  .job-canada {{
    display: inline-block;
    margin-left: 0.4rem;
    padding: 0.05rem 0.5rem;
    border-radius: 999px;
    border: 1px solid #fecaca;
    background: #fef2f2;
    color: #b91c1c;
    font-size: 0.75rem;
  }}
  #empty {{
    color: var(--muted);
    padding: 2rem 0;
  }}
  a.top {{
    position: fixed;
    bottom: 1.5rem;
    right: 1.5rem;
    background: var(--accent);
    color: #fff;
    padding: 0.5rem 0.9rem;
    border-radius: 999px;
    text-decoration: none;
    font-size: 0.85rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
  }}
</style>
</head>
<body>
  <header id="top">
    <a id="brand" href="index.html">openings.info</a>
    <div id="meta">{meta}</div>
  </header>

{toc}
  <main id="results">
{results}
  </main>

  <a class="top" href="#top">↑ Top</a>

  <footer id="site-footer">
    <nav id="footer-nav"><a href="companies.html">Companies crawled &rarr;</a></nav>
    <p id="footer-credit">
      <a href="https://github.com/clarmso/openings.info" target="_blank" rel="noopener noreferrer">GitHub repository</a>
      &middot;
      Created by <a href="https://clarmso.ca" target="_blank" rel="noopener noreferrer">@clarmso</a>
    </p>
  </footer>
</body>
</html>
"""


def _render_job_item(job):
    title = html.escape(job["title"])
    url = html.escape(job["url"])
    location = job.get("location") or ""
    is_canada = job.get("canada")

    badges = []
    if is_canada:
        badges.append('<span class="job-canada">🇨🇦 Canada</span>')
    badges_html = " ".join(badges)

    meta_parts = []
    if location:
        meta_parts.append(html.escape(location))
    if badges_html:
        meta_parts.append(badges_html)
    meta = f'<span class="job-meta">{" ".join(meta_parts)}</span>' if meta_parts else ""

    return (
        f'        <li><a href="{url}" target="_blank" rel="noopener noreferrer">{title}</a>{meta}</li>'
    )


def _render_html(manifest):
    generated = datetime.strptime(manifest["generated_at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    total = manifest["total_new_jobs"]
    companies = manifest["companies"]
    meta = (
        f"Jobs posted in the last {LOOKBACK_HOURS} hours (as of {generated.strftime('%Y-%m-%d %H:%M UTC')}) "
        f"&middot; {total} new job{'' if total == 1 else 's'} across "
        f"{len(companies)} compan{'y' if len(companies) == 1 else 'ies'}"
    )

    if not companies:
        toc = ""
        results = '    <p id="empty">No new jobs in the most recent crawl. Check back soon!</p>'
    else:
        toc_items = "\n".join(
            f'      <li><a href="#company-{html.escape(c["slug"])}">{html.escape(c["name"])} ({len(c["jobs"])})</a></li>'
            for c in companies
        )
        toc = f"""  <nav id="toc">
    <h2>Jump to company</h2>
    <ul id="toc-list">
{toc_items}
    </ul>
  </nav>
"""

        sections = []
        for c in companies:
            job_items = "\n".join(_render_job_item(j) for j in c["jobs"])
            name_html = html.escape(c["name"])
            if c.get("careers_url"):
                name_html = (
                    f'<a class="company-link" href="{html.escape(c["careers_url"])}" '
                    f'target="_blank" rel="noopener noreferrer">{name_html}</a>'
                )
            sections.append(
                f'    <section class="company" id="company-{html.escape(c["slug"])}">\n'
                f'      <h2>{name_html} <span class="count">({len(c["jobs"])})</span></h2>\n'
                f"      <ul>\n{job_items}\n      </ul>\n"
                f"    </section>"
            )
        results = "\n".join(sections)

    return HTML_TEMPLATE.format(meta=meta, toc=toc, results=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir", default="docs",
        help="Directory to write index.html/new_jobs.json into "
             "(default: docs). Use e.g. docs/staging for a preview build.",
    )
    args = parser.parse_args()

    json_output_path = os.path.join(args.out_dir, "new_jobs.json")
    html_output_path = os.path.join(args.out_dir, "index.html")

    manifest = _build_manifest()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(json_output_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")

    with open(html_output_path, "w") as f:
        f.write(_render_html(manifest))

    print(f"Wrote {html_output_path} and {json_output_path}: "
          f"{manifest['total_new_jobs']} new jobs across {len(manifest['companies'])} companies")


if __name__ == "__main__":
    sys.exit(main())
