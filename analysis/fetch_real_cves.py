"""
Pulls REAL, CVE-confirmed vulnerable Python code from GitHub Security
Advisories for a set of target CWEs (default: CWE-89 SQLi, CWE-78 CmdI,
CWE-22 Path Traversal).

This is Job 1 of the thesis data plan: ground truth that nobody can argue
you invented yourself. Every sample this script saves comes from a real,
disclosed, fixed vulnerability in a real PyPI package, with the CVE/GHSA
id, the fix commit, and the pre-fix (vulnerable) file content attached.

WHY RUN THIS LOCALLY, NOT IN THE SANDBOX:
This cloud sandbox's outbound network is locked to package registries
only, so api.github.com is unreachable from there. Your own machine has
normal internet access, so run this script there.

SETUP:
    pip install requests
    export GITHUB_TOKEN=ghp_xxx   # optional but strongly recommended:
                                   # unauthenticated GitHub API calls are
                                   # rate-limited to 60/hour, authenticated
                                   # ones get 5000/hour. A token with no
                                   # scopes at all (just "public repo read")
                                   # is enough -- create one at
                                   # https://github.com/settings/tokens

USAGE:
    python3 analysis/fetch_real_cves.py --cwe CWE-89 --limit 30
    python3 analysis/fetch_real_cves.py --cwe CWE-78 --limit 30
    python3 analysis/fetch_real_cves.py --cwe CWE-22 --limit 30

OUTPUT:
    calibration/real_cve/<cwe>/<ghsa_id>.json   -- one file per candidate,
    containing: cve_id, ghsa_id, summary, package, vulnerable_range,
    fix_commit_url, and (when resolvable) the actual pre-fix source of
    every changed .py file, plus the diff/patch itself.

IMPORTANT -- these are candidates, not a finished dataset:
    Real fix commits are often large refactors, framework-internal, or
    touch many files. After running this, you (a human) still need to
    skim each JSON file and manually pick the ones that are:
      (a) small enough to isolate as a single vulnerable function,
      (b) actually the injection-class bug your thesis targets, not a
          different bug bundled into the same release.
    That manual triage step is expected and worth documenting in your
    methodology section -- it's exactly the kind of honest, defensible
    step examiners want to see.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import requests

API_ROOT = "https://api.github.com"
RAW_ROOT = "https://raw.githubusercontent.com"

ROOT = Path(__file__).parent.parent
OUT_DIR = ROOT / "calibration" / "real_cve"

# GitHub's advisory CWE filter wants the bare "CWE-89" form.
DEFAULT_CWES = ["CWE-89", "CWE-78", "CWE-22"]

COMMIT_URL_RE = re.compile(
    r"https://github\.com/([\w.-]+)/([\w.-]+)/commit/([0-9a-f]{7,40})"
)


def _headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "thesis-real-cve-fetcher",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _get(url: str, params: dict[str, Any] | None = None) -> requests.Response:
    resp = requests.get(url, headers=_headers(), params=params, timeout=30)
    if resp.status_code == 403 and "rate limit" in resp.text.lower():
        reset = resp.headers.get("X-RateLimit-Reset")
        wait = max(int(reset) - int(time.time()), 5) if reset else 60
        print(f"  rate limited, sleeping {wait}s...", file=sys.stderr)
        time.sleep(wait)
        return _get(url, params)
    return resp


def fetch_advisories(cwe: str, limit: int) -> list[dict[str, Any]]:
    """GitHub's advisory search supports filtering by ecosystem + CWE directly."""
    advisories: list[dict[str, Any]] = []
    page = 1
    while len(advisories) < limit:
        # GitHub's advisories API wants the bare CWE number ("89"), not the
        # "CWE-89" form -- passing the prefixed form silently matches nothing.
        cwe_number = cwe.split("-")[-1] if "-" in cwe else cwe
        resp = _get(
            f"{API_ROOT}/advisories",
            params={
                "ecosystem": "pip",
                "cwes": cwe_number,
                "per_page": min(100, limit - len(advisories)),
                "page": page,
                "sort": "published",
                "direction": "desc",
            },
        )
        if resp.status_code != 200:
            print(f"  advisories fetch failed: {resp.status_code} {resp.text[:200]}", file=sys.stderr)
            break
        batch = resp.json()
        if not batch:
            break
        advisories.extend(batch)
        page += 1
        if len(batch) < 100:
            break
    return advisories[:limit]


def extract_commit_refs(advisory: dict[str, Any]) -> list[tuple[str, str, str]]:
    """Return [(owner, repo, sha), ...] found in the advisory's references."""
    refs: list[tuple[str, str, str]] = []
    for ref in advisory.get("references", []) or []:
        url = ref.get("url", "") if isinstance(ref, dict) else str(ref)
        m = COMMIT_URL_RE.search(url)
        if m:
            refs.append(m.groups())
    return refs


def fetch_commit_detail(owner: str, repo: str, sha: str) -> dict[str, Any] | None:
    resp = _get(f"{API_ROOT}/repos/{owner}/{repo}/commits/{sha}")
    if resp.status_code != 200:
        return None
    return resp.json()


def fetch_pre_fix_file(owner: str, repo: str, parent_sha: str, path: str) -> str | None:
    """Raw file content as it was right BEFORE the fix -- i.e. still vulnerable."""
    resp = requests.get(
        f"{RAW_ROOT}/{owner}/{repo}/{parent_sha}/{path}",
        headers={"User-Agent": "thesis-real-cve-fetcher"},
        timeout=30,
    )
    if resp.status_code != 200:
        return None
    return resp.text


def process_advisory(cwe: str, advisory: dict[str, Any]) -> dict[str, Any] | None:
    ghsa_id = advisory.get("ghsa_id", "unknown")
    packages = [
        v.get("package", {}).get("name")
        for v in advisory.get("vulnerabilities", []) or []
        if v.get("package")
    ]
    record: dict[str, Any] = {
        "ghsa_id": ghsa_id,
        "cve_id": advisory.get("cve_id"),
        "cwe": cwe,
        "summary": advisory.get("summary"),
        "severity": advisory.get("severity"),
        "packages": packages,
        "vulnerable_ranges": [
            v.get("vulnerable_version_range")
            for v in advisory.get("vulnerabilities", []) or []
        ],
        "advisory_url": advisory.get("html_url"),
        "commits": [],
    }

    commit_refs = extract_commit_refs(advisory)
    for owner, repo, sha in commit_refs[:2]:  # cap: avoid huge multi-commit advisories
        detail = fetch_commit_detail(owner, repo, sha)
        if not detail:
            continue
        parents = detail.get("parents", [])
        parent_sha = parents[0]["sha"] if parents else None

        files_out = []
        for f in detail.get("files", []) or []:
            if not f.get("filename", "").endswith(".py"):
                continue
            pre_fix_source = None
            if parent_sha:
                pre_fix_source = fetch_pre_fix_file(owner, repo, parent_sha, f["filename"])
            files_out.append(
                {
                    "path": f["filename"],
                    "status": f.get("status"),
                    "patch": f.get("patch"),  # unified diff -- shows exactly what changed
                    "pre_fix_source": pre_fix_source,  # full file, still vulnerable
                }
            )

        record["commits"].append(
            {
                "url": f"https://github.com/{owner}/{repo}/commit/{sha}",
                "owner": owner,
                "repo": repo,
                "sha": sha,
                "parent_sha": parent_sha,
                "message": detail.get("commit", {}).get("message"),
                "files": files_out,
            }
        )

    return record if record["commits"] else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cwe", action="append", dest="cwes", help="e.g. CWE-89 (repeatable)")
    ap.add_argument("--limit", type=int, default=25, help="advisories to scan per CWE")
    args = ap.parse_args()
    cwes = args.cwes or DEFAULT_CWES

    for cwe in cwes:
        out_dir = OUT_DIR / cwe
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== {cwe} ===")
        advisories = fetch_advisories(cwe, args.limit)
        print(f"  found {len(advisories)} advisories with a pip-ecosystem package")

        saved = 0
        for adv in advisories:
            record = process_advisory(cwe, adv)
            if record is None:
                continue
            out_path = out_dir / f"{record['ghsa_id']}.json"
            out_path.write_text(json.dumps(record, indent=2))
            saved += 1
            print(f"  saved {out_path.relative_to(ROOT)}  ({record['cve_id']}, {record['packages']})")

        print(f"  -> {saved}/{len(advisories)} advisories had a resolvable commit+diff")

    print(f"\nDone. Now manually skim {OUT_DIR.relative_to(ROOT)}/<cwe>/*.json and pick the")
    print("candidates that are a single, isolatable vulnerable function -- those become")
    print("your held-out real-world ground truth set.")


if __name__ == "__main__":
    main()
