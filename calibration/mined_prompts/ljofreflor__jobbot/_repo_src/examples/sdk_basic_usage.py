#!/usr/bin/env python3
"""Basic JobBot SDK usage example.

This script demonstrates core SDK functionality:
- Initializing the client
- Searching for jobs
- Matching against profile
- Building tailored CVs
"""

from __future__ import annotations

from jobbot.sdk import JobBotClient


def main() -> None:
    """Demonstrate basic SDK usage."""
    print("JobBot SDK - Basic Usage Example")
    print("=" * 50)

    # Initialize client (auto-discovers workspace)
    with JobBotClient() as client:
        print("\n1. Searching for Python jobs...")
        jobs = client.jobs.search(
            "python developer",
            source="torre",
            remote=True,
            limit=5,
        )
        print(f"   Found {len(jobs)} jobs")

        if not jobs:
            print("   No jobs found. Try a different query.")
            return

        # Display job details
        print("\n2. Job listings:")
        for job in jobs:
            print(f"\n   {job.id}: {job.company} - {job.title}")
            print(f"   Location: {job.location or 'Not specified'}")
            print(f"   Remote: {job.remote_type or 'Not specified'}")
            if job.skills:
                print(f"   Skills: {', '.join(job.skills[:5])}")

        # Match jobs against profile
        print("\n3. Matching jobs against profile...")
        matches = client.jobs.match(jobs)

        print("\n4. Match results:")
        for job, match in zip(jobs, matches, strict=True):
            print(f"\n   {job.id}: {job.company} - {match.score}%")
            if match.highlights:
                print(f"   ✓ Highlights: {', '.join(match.highlights[:3])}")
            if match.gaps:
                print(f"   ✗ Gaps: {', '.join(match.gaps[:3])}")

        # Build CV for best match
        best_job, best_match = max(
            zip(jobs, matches, strict=True),
            key=lambda x: x[1].score,
        )

        print(f"\n5. Building CV for best match ({best_match.score}%)...")
        print(f"   Company: {best_job.company}")
        print(f"   Role: {best_job.title}")

        cv_path = client.cv.build(
            job=best_job,
            style="modern",
            output_path=f"cv_{best_job.company.replace(' ', '_')}_example.pdf",
        )

        print(f"   ✓ CV saved: {cv_path}")

    print("\n" + "=" * 50)
    print("Example complete!")


if __name__ == "__main__":
    main()
