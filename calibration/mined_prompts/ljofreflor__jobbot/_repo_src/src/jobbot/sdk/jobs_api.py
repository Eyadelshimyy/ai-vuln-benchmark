"""Jobs API: search, retrieve, and match job opportunities."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from sqlalchemy.orm import Session

from jobbot.jobs.repository import JobRepository
from jobbot.matching.analyzer import RuleBasedJobAnalyzer
from jobbot.models.job import JobPosting
from jobbot.models.match import JobMatch

if TYPE_CHECKING:
    from collections.abc import Callable

    from jobbot.config import JobbotConfig
    from jobbot.models.candidate import Candidate


class JobsApi:
    """API for job search, retrieval, and matching operations.

    This API provides programmatic access to job-related functionality:
    - Search jobs from various sources (Torre, Indeed, etc.)
    - Retrieve stored jobs by ID
    - Match jobs against candidate profile
    - List all stored jobs

    Example:
        >>> api = JobsApi(config, session_factory, candidate)
        >>> jobs = api.search("python", source="torre", remote=True)
        >>> match = api.match(jobs[0])
        >>> print(f"Match score: {match.score}%")
    """

    def __init__(
        self,
        config: JobbotConfig,
        session_factory: Callable[[], Session],
        candidate: Candidate,
    ) -> None:
        """Initialize Jobs API.

        Args:
            config: JobBot configuration
            session_factory: Factory for creating database sessions
            candidate: Loaded candidate profile for matching
        """
        self.config = config
        self._session_factory = session_factory
        self.candidate = candidate

    def search(
        self,
        query: str,
        *,
        source: str = "torre",
        remote: bool = False,
        limit: int = 10,
    ) -> list[JobPosting]:
        """Search for jobs matching the query.

        Args:
            query: Search query (keywords, role name, etc.)
            source: Job source to search ("torre", "indeed", etc.)
            remote: Filter for remote positions only
            limit: Maximum number of results

        Returns:
            List of matching job postings

        Raises:
            ValueError: If source is not supported
        """
        if source == "torre":
            from jobbot.adapters.torre.jobs import TorreJobSource

            torre_source = TorreJobSource(self.config)
            from jobbot.jobs.sources import JobSearchQuery

            search_query = JobSearchQuery(
                query=query,
                remote=remote,
                limit=limit,
            )
            return torre_source.search_jobs(search_query)

        msg = f"Unsupported job source: {source}"
        raise ValueError(msg)

    def get(self, job_id: str) -> JobPosting | None:
        """Retrieve a job posting by ID.

        Args:
            job_id: Job ID (e.g. "J0001")

        Returns:
            Job posting if found, None otherwise
        """
        session = self._session_factory()
        try:
            repo = JobRepository(session)
            return repo.get(job_id)
        finally:
            session.close()

    def list_all(self, *, limit: int | None = None) -> list[JobPosting]:
        """List all stored jobs.

        Args:
            limit: Maximum number of jobs to return (None for all)

        Returns:
            List of stored job postings
        """
        session = self._session_factory()
        try:
            repo = JobRepository(session)
            jobs = repo.list_all()
            if limit is not None:
                return jobs[:limit]
            return jobs
        finally:
            session.close()

    def match(
        self,
        jobs: JobPosting | Sequence[JobPosting],
    ) -> JobMatch | list[JobMatch]:
        """Match job(s) against candidate profile.

        Args:
            jobs: Single job or list of jobs to match

        Returns:
            JobMatch or list of JobMatch objects with scores and analysis
        """
        analyzer = RuleBasedJobAnalyzer()

        if isinstance(jobs, JobPosting):
            return analyzer.analyze(self.candidate, jobs)

        return [analyzer.analyze(self.candidate, job) for job in jobs]

    def match_by_id(self, job_id: str) -> JobMatch | None:
        """Match a stored job by ID against candidate profile.

        Args:
            job_id: Job ID to match

        Returns:
            JobMatch if job found, None otherwise
        """
        job = self.get(job_id)
        if job is None:
            return None
        # match() with single JobPosting returns single JobMatch
        result = self.match(job)
        assert isinstance(result, JobMatch)
        return result
