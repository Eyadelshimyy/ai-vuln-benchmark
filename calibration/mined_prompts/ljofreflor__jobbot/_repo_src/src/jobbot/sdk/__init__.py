"""JobBot SDK: Programmatic access to JobBot functionality.

The SDK provides a clean, stable API for integrating JobBot capabilities
into other applications without depending on CLI internals.

Example usage:

    from jobbot.sdk import JobBotClient

    # Initialize client
    client = JobBotClient()

    # Search for jobs
    jobs = client.jobs.search("data scientist", remote=True, limit=10)

    # Get job details
    job = client.jobs.get("J0001")

    # Match jobs against profile
    matches = client.jobs.match(jobs)

    # Build CV
    cv_path = client.cv.build(
        job=job,
        style="modern",
        output_path="cv_custom.pdf"
    )
"""

from __future__ import annotations

from jobbot.sdk.client import JobBotClient

__all__ = ["JobBotClient"]
