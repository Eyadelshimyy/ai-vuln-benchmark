#!/usr/bin/env python3
"""Batch CV generation example using JobBot SDK.

This script demonstrates building CVs for multiple job matches:
- Search jobs matching criteria
- Filter by match score threshold
- Generate tailored CVs in bulk
"""

from __future__ import annotations

from pathlib import Path

from jobbot.sdk import JobBotClient


def main() -> None:
    """Build CVs for all strong job matches."""
    print("JobBot SDK - Batch CV Generation")
    print("=" * 50)

    # Configuration
    MIN_MATCH_SCORE = 70  # Only build CVs for 70%+ matches
    MAX_CVS = 5  # Limit to top 5 matches
    OUTPUT_DIR = Path("output/batch_cvs")

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with JobBotClient() as client:
        # Step 1: Search for jobs
        print("\n1. Searching for senior backend developer roles...")
        jobs = client.jobs.search(
            "senior backend developer",
            source="torre",
            remote=True,
            limit=20,
        )
        print(f"   Found {len(jobs)} jobs")

        if not jobs:
            print("   No jobs found.")
            return

        # Step 2: Match against profile
        print("\n2. Matching against profile...")
        matches = client.jobs.match(jobs)

        # Step 3: Filter strong matches
        strong_matches = [
            (job, match)
            for job, match in zip(jobs, matches, strict=True)
            if match.score >= MIN_MATCH_SCORE
        ]

        print(f"\n3. Found {len(strong_matches)} strong matches (≥{MIN_MATCH_SCORE}%)")

        if not strong_matches:
            print("   No strong matches found. Try lowering the threshold.")
            return

        # Sort by score (highest first)
        strong_matches.sort(key=lambda x: x[1].score, reverse=True)

        # Step 4: Build CVs for top matches
        print(f"\n4. Building CVs for top {min(MAX_CVS, len(strong_matches))} matches...")

        successful = 0
        for i, (job, match) in enumerate(strong_matches[:MAX_CVS], start=1):
            company_safe = job.company.replace(" ", "_").replace("/", "-")
            output_file = OUTPUT_DIR / f"{i:02d}_cv_{company_safe}.pdf"

            try:
                cv_path = client.cv.build(
                    job=job,
                    style="ats-optimized",
                    output_path=output_file,
                )

                print(f"\n   [{i}] {job.company} - {job.title}")
                print(f"       Match: {match.score}%")
                print(f"       CV: {cv_path.name}")
                successful += 1

            except Exception as e:
                print(f"\n   [{i}] {job.company} - FAILED")
                print(f"       Error: {e}")

        # Summary
        print("\n" + "=" * 50)
        print(f"Generated {successful}/{min(MAX_CVS, len(strong_matches))} CVs")
        print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
