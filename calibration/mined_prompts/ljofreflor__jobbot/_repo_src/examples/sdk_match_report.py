#!/usr/bin/env python3
"""Match analysis report using JobBot SDK.

This script generates a detailed analysis of job matches:
- Search jobs by query
- Match against profile
- Generate formatted match report
- Export to JSON for further analysis
"""

from __future__ import annotations

import json
from pathlib import Path

from jobbot.sdk import JobBotClient


def format_match_summary(job, match) -> dict:
    """Format match data for export."""
    return {
        "job_id": job.id,
        "company": job.company,
        "title": job.title,
        "location": job.location,
        "remote": job.remote_type,
        "url": job.url,
        "match_score": match.score,
        "highlights": match.highlights or [],
        "gaps": match.gaps or [],
        "skills": job.skills or [],
    }


def main() -> None:
    """Generate match analysis report."""
    print("JobBot SDK - Match Analysis Report")
    print("=" * 60)

    # Configuration
    QUERY = "data scientist"
    REMOTE_ONLY = True
    LIMIT = 15
    OUTPUT_FILE = Path("match_report.json")

    with JobBotClient() as client:
        # Search jobs
        print(f"\n1. Searching: '{QUERY}' (remote={REMOTE_ONLY})")
        jobs = client.jobs.search(
            QUERY,
            source="torre",
            remote=REMOTE_ONLY,
            limit=LIMIT,
        )
        print(f"   Found {len(jobs)} jobs")

        if not jobs:
            print("   No jobs found.")
            return

        # Match jobs
        print("\n2. Analyzing matches...")
        matches = client.jobs.match(jobs)

        # Sort by match score
        sorted_results = sorted(
            zip(jobs, matches, strict=True),
            key=lambda x: x[1].score,
            reverse=True,
        )

        # Display summary
        print("\n3. Match Summary:")
        print("-" * 60)

        report_data = []
        for rank, (job, match) in enumerate(sorted_results, start=1):
            print(f"\n   [{rank}] {match.score}% - {job.company}")
            print(f"       Title: {job.title}")
            print(f"       Location: {job.location or 'Remote'}")

            if match.highlights:
                print(f"       ✓ Strengths: {', '.join(match.highlights[:3])}")
            if match.gaps:
                print(f"       ✗ Gaps: {', '.join(match.gaps[:3])}")

            # Add to report
            report_data.append(format_match_summary(job, match))

        # Export to JSON
        print(f"\n4. Exporting report to {OUTPUT_FILE}...")
        with OUTPUT_FILE.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "query": QUERY,
                    "total_jobs": len(jobs),
                    "matches": report_data,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        # Statistics
        high_matches = sum(1 for _, m in sorted_results if m.score >= 80)
        good_matches = sum(1 for _, m in sorted_results if 60 <= m.score < 80)
        low_matches = sum(1 for _, m in sorted_results if m.score < 60)

        print("\n" + "=" * 60)
        print("Match Statistics:")
        print(f"  High (≥80%): {high_matches} jobs")
        print(f"  Good (60-79%): {good_matches} jobs")
        print(f"  Low (<60%): {low_matches} jobs")
        print(f"\nReport saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
