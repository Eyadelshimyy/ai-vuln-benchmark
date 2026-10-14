#!/usr/bin/env python3
"""
Generate docs/companies.html — a static page listing every company tracked
in companies.csv, its ATS platform, HQ location, current open-role count,
how many of those openings can be based in Canada, and a link to its live
careers page. Also writes docs/companies.json as a raw data snapshot.

Unlike generate_new_jobs.py this isn't time-windowed: it reflects whatever
is currently in data/ (typically after `crawl.py` has just run), so it's
safe to run any time after (or independently of) a crawl.
"""

import argparse
import csv
import glob
import html
import json
import os
import re
import sys
import urllib.parse
from datetime import datetime, timezone

DATA_DIR = "data"
COMPANIES_CSV = "companies.csv"

# Per-company snapshot of job IDs currently live on the ATS, written by
# crawl.py on every run (see src/crawl.OPEN_JOBS_STATE_DIR). data/ itself is
# an append-only historical archive — job files are kept even after a
# posting closes — so this state is what lets the counts below reflect
# today's actual number of open roles instead of every posting ever seen.
OPEN_JOBS_STATE_DIR = os.path.join("state", "open_jobs")

# Platforms crawl.py knows how to fetch (kept in sync with src/crawl.PLATFORMS).
SUPPORTED_PLATFORMS = {
    "greenhouse", "lever", "ashby", "workable", "recruitee",
    "smartrecruiters", "gem", "rippling", "breezy", "bamboohr",
    "personio", "trakstar",
}

CAREERS_URL_BY_PLATFORM = {
    "greenhouse": "https://job-boards.greenhouse.io/{slug}",
    "lever": "https://jobs.lever.co/{slug}",
    "ashby": "https://jobs.ashbyhq.com/{slug}",
    "workable": "https://apply.workable.com/{slug}/",
    "recruitee": "https://{slug}.recruitee.com/",
    "smartrecruiters": "https://careers.smartrecruiters.com/{slug}/",
    "gem": "https://jobs.gem.com/{slug}",
    "rippling": "https://ats.rippling.com/{slug}/jobs",
    "breezy": "https://{slug}.breezy.hr/",
    "bamboohr": "https://{slug}.bamboohr.com/careers",
    "personio": "https://{slug}.jobs.personio.de/",
    "trakstar": "https://{slug}.hire.trakstar.com/",
}


def _careers_url(slug, platform):
    template = CAREERS_URL_BY_PLATFORM.get(platform)
    return template.format(slug=slug) if template else None


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
    distinguish the two tiers."""
    return _mentions_canada_explicit(*texts) or _mentions_canada_broad(*texts)


def _mode_from_text(text):
    """Best-effort guess of workplace mode from a free-text location string."""
    if not text:
        return ""
    lowered = str(text).lower()
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
    return ""


def _is_remote_job(platform, job):
    """Best-effort: is this specific opening fully remote?

    Mirrors the workplace-mode inference used by generate_new_jobs.py, using
    structured fields where the ATS provides them and falling back to
    keyword matching on free-text location fields.

    Note: this only signals "remote work mode" (vs. hybrid/on-site); it says
    nothing about geographic eligibility. Callers should treat a posting
    that's both remote *and* Canada-eligible (e.g. "Canada Remote") as
    Canada, not worldwide Remote — see _scan_company_jobs, which only counts
    a job as Remote when it is not already counted as Canada.
    """
    if platform == "greenhouse":
        location = (job.get("location") or {}).get("name", "")
        return _mode_from_text(location) == "Remote"

    if platform == "lever":
        categories = job.get("categories") or {}
        location = categories.get("location") or ", ".join(categories.get("allLocations") or [])
        mode = _normalize_mode(job.get("workplaceType")) or _mode_from_text(location)
        return mode == "Remote"

    if platform == "ashby":
        location = job.get("location", "")
        mode = _normalize_mode(job.get("workplaceType")) or _mode_from_text(location)
        return mode == "Remote"

    if platform == "workable":
        if job.get("telecommuting"):
            return True
        location = ", ".join(part for part in (job.get("city"), job.get("state"), job.get("country")) if part)
        return _mode_from_text(location) == "Remote"

    if platform == "smartrecruiters":
        loc = job.get("location") or {}
        if loc.get("remote"):
            return True
        location = loc.get("fullLocation") or loc.get("city", "")
        return _mode_from_text(location) == "Remote"

    if platform == "gem":
        location = (job.get("location") or {}).get("name", "")
        mode = _normalize_mode(job.get("location_type")) or _mode_from_text(location)
        return mode == "Remote"

    if platform == "rippling":
        location = ", ".join(job.get("workLocations") or [])
        return _mode_from_text(location) == "Remote"

    return False


def _is_canada_job(platform, job):
    """Best-effort: should this specific opening be tagged Canada?

    An explicit Canada reference (city, province, "Canada", "Canada Remote")
    always counts. A broader regional reference ("Global", "North America",
    "Americas") only counts when the posting is *not* remote — once it's
    also remote, it reads as remote-worldwide/regional rather than
    Canada-specific, so it should get the Remote tag instead, not Canada.
    """
    is_remote = _is_remote_job(platform, job)

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
        # Fall back to free text for regional postings ("Global", "Americas", ...)
        # that don't map to a single ISO country field.
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


def _load_open_ids(slug):
    """Return the set of job IDs currently live for `slug`, or None if no
    state has been recorded yet (e.g. before the first crawl.py run that
    includes this feature) — callers should fall back to counting every
    file in that case rather than reporting zero open jobs."""
    path = os.path.join(OPEN_JOBS_STATE_DIR, f"{slug}.json")
    try:
        with open(path) as f:
            return set(json.load(f).get("open_ids", []))
    except (OSError, json.JSONDecodeError):
        return None


def _scan_company_jobs(slug, platform):
    """Single pass over a company's crawled job files, counting only the
    ones still live on the ATS (see _load_open_ids) — data/ keeps every
    posting ever seen for historical analysis, but the counts here should
    reflect today's actual number of open roles.

    Returns (sample_company_name, job_count, canada_job_count, remote_job_count).
    """
    path = os.path.join(DATA_DIR, slug)
    if not os.path.isdir(path):
        return None, 0, 0, 0

    open_ids = _load_open_ids(slug)

    sample_name = None
    job_count = 0
    canada_job_count = 0
    remote_job_count = 0
    for match in glob.glob(os.path.join(path, "**", "*.json"), recursive=True):
        job_id = os.path.splitext(os.path.basename(match))[0]
        if open_ids is not None and job_id not in open_ids:
            continue

        try:
            with open(match) as f:
                job = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue

        job_count += 1
        is_canada = _is_canada_job(platform, job)
        if is_canada:
            canada_job_count += 1
        # "Remote" here specifically means remote *worldwide* (no location
        # restriction) — a posting explicitly scoped to Canada (e.g. "Canada
        # Remote", "Toronto, ON") is tagged Canada only, not double-counted
        # as Remote too.
        if not is_canada and _is_remote_job(platform, job):
            remote_job_count += 1

        if sample_name is None:
            sample_name = (
                job.get("company_name")
                or job.get("companyName")
                or (job.get("company") or {}).get("name")
            )

    return sample_name, job_count, canada_job_count, remote_job_count


def _display_name(slug, sample_name):
    if sample_name:
        return sample_name
    decoded = urllib.parse.unquote(slug)
    return decoded.replace("-", " ").replace("_", " ").title()


def _build_manifest():
    with open(COMPANIES_CSV, newline="") as f:
        rows = [row for row in csv.DictReader(f) if row.get("slug") and row.get("ats")]

    companies = []
    for row in rows:
        slug = row["slug"]
        platform = row["ats"]
        supported = platform in SUPPORTED_PLATFORMS
        sample_name, job_count, canada_job_count, remote_job_count = (
            _scan_company_jobs(slug, platform) if supported else (None, 0, 0, 0)
        )
        companies.append({
            "slug": slug,
            "name": _display_name(slug, sample_name),
            "ats": platform,
            "hq_country": row.get("hq_country") or "",
            "job_count": job_count,
            "canada_job_count": canada_job_count,
            "remote_job_count": remote_job_count,
            "careers_url": _careers_url(slug, platform),
            "supported": supported,
        })

    companies.sort(key=lambda c: c["name"].lower())

    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_companies": len(companies),
        "total_open_jobs": sum(c["job_count"] for c in companies),
        "companies": companies,
    }


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>openings.info - Companies</title>
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
  #search {{
    width: 100%;
    padding: 0.6rem 0.8rem;
    font-size: 0.95rem;
    border: 1px solid var(--border);
    border-radius: 8px;
    margin-bottom: 0.75rem;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.92rem;
  }}
  thead th {{
    text-align: left;
    color: var(--muted);
    font-weight: 600;
    text-transform: uppercase;
    font-size: 0.75rem;
    letter-spacing: 0.03em;
    border-bottom: 1px solid var(--border);
    padding: 0.5rem 0.6rem;
  }}
  tbody td {{
    padding: 0.55rem 0.6rem;
    vertical-align: middle;
  }}
  tbody tr:hover {{ background: #f9fafb; }}
  td.jobs, th.jobs {{ text-align: right; }}
  a.company-link {{
    color: #111827;
    text-decoration: none;
    font-weight: 500;
  }}
  a.company-link:hover {{ color: var(--accent); text-decoration: underline; }}
  .badge {{
    display: inline-block;
    padding: 0.05rem 0.5rem;
    border-radius: 999px;
    border: 1px solid var(--border);
    font-size: 0.75rem;
    color: var(--muted);
  }}
  .badge.unsupported {{ color: #b91c1c; border-color: #fecaca; background: #fef2f2; }}
  #empty-filter {{
    display: none;
    color: var(--muted);
    padding: 1.5rem 0;
  }}
</style>
</head>
<body>
  <header id="top">
    <a id="brand" href="index.html">openings.info</a>
    <h1>Companies</h1>
    <div id="meta">{meta}</div>
  </header>

  <input id="search" type="search" placeholder="Filter by company..." autocomplete="off" />

  <table id="companies-table">
    <thead>
      <tr>
        <th>Company</th>
        <th>HQ</th>
        <th class="jobs">Open jobs</th>
      </tr>
    </thead>
    <tbody>
{rows}
    </tbody>
  </table>
  <p id="empty-filter">No companies match your filter.</p>

  <script>
    const input = document.getElementById("search");
    const rows = Array.from(document.querySelectorAll("#companies-table tbody tr"));
    const emptyMsg = document.getElementById("empty-filter");

    function applyFilters() {{
      const q = input.value.trim().toLowerCase();
      let visible = 0;
      rows.forEach((row) => {{
        const match = row.dataset.search.includes(q);
        row.style.display = match ? "" : "none";
        if (match) visible++;
      }});
      emptyMsg.style.display = visible === 0 ? "block" : "none";
    }}

    input.addEventListener("input", applyFilters);
  </script>

  <footer id="site-footer">
    <nav id="footer-nav"><a href="index.html">&larr; New Jobs</a></nav>
    <p id="footer-credit">
      <a href="https://github.com/clarmso/openings.info" target="_blank" rel="noopener noreferrer">GitHub repository</a>
      &middot;
      Created by <a href="https://clarmso.ca" target="_blank" rel="noopener noreferrer">@clarmso</a>
    </p>
  </footer>
</body>
</html>
"""


def _render_row(c):
    name = html.escape(c["name"])
    hq = html.escape(c["hq_country"]) if c["hq_country"] else "&mdash;"
    search_key = html.escape(f"{c['name']} {c['hq_country']}".lower())

    if c["careers_url"]:
        name_cell = (
            f'<a class="company-link" href="{html.escape(c["careers_url"])}" '
            f'target="_blank" rel="noopener noreferrer">{name}</a>'
        )
    else:
        name_cell = name

    if c["supported"]:
        jobs_cell = str(c["job_count"])
    else:
        jobs_cell = '<span class="badge unsupported">not yet supported</span>'

    return (
        f'      <tr data-search="{search_key}">\n'
        f"        <td>{name_cell}</td>\n"
        f"        <td>{hq}</td>\n"
        f'        <td class="jobs">{jobs_cell}</td>\n'
        f"      </tr>"
    )


def _render_html(manifest):
    generated = datetime.strptime(manifest["generated_at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    meta = (
        f"{manifest['total_companies']} companies tracked &middot; "
        f"{manifest['total_open_jobs']} open jobs &middot; "
        f"as of {generated.strftime('%Y-%m-%d %H:%M UTC')}"
    )
    rows = "\n".join(_render_row(c) for c in manifest["companies"])
    return HTML_TEMPLATE.format(meta=meta, rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir", default="docs",
        help="Directory to write companies.html/companies.json into "
             "(default: docs). Use e.g. docs/staging for a preview build.",
    )
    args = parser.parse_args()

    json_output_path = os.path.join(args.out_dir, "companies.json")
    html_output_path = os.path.join(args.out_dir, "companies.html")

    manifest = _build_manifest()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(json_output_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")

    with open(html_output_path, "w") as f:
        f.write(_render_html(manifest))

    print(f"Wrote {html_output_path} and {json_output_path}: "
          f"{manifest['total_companies']} companies, {manifest['total_open_jobs']} open jobs")


if __name__ == "__main__":
    sys.exit(main())
