"""CV API: build tailored CVs for job applications."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from jobbot.cv.build import BuildTarget, build_cv
from jobbot.cv.renderer import CvStyle
from jobbot.models.job import JobPosting

if TYPE_CHECKING:
    from jobbot.config import JobbotConfig
    from jobbot.models.candidate import Candidate


class CvApi:
    """API for building and managing CV documents.

    This API provides programmatic access to CV building functionality:
    - Build tailored CVs for specific jobs
    - Generate CVs in different styles (modern, classic, ats-optimized)
    - Export to PDF with customizable templates

    Example:
        >>> api = CvApi(config, candidate)
        >>> cv_path = api.build(
        ...     job=job_posting,
        ...     style="modern",
        ...     output_path="cv_acme_corp.pdf"
        ... )
    """

    def __init__(self, config: JobbotConfig, candidate: Candidate) -> None:
        """Initialize CV API.

        Args:
            config: JobBot configuration
            candidate: Loaded candidate profile
        """
        self.config = config
        self.candidate = candidate

    def build(
        self,
        *,
        job: JobPosting | None = None,
        style: str = "moderncv",
        output_path: Path | str | None = None,
        target: BuildTarget = BuildTarget.CV,
    ) -> Path:
        """Build a CV, optionally tailored for a specific job.

        Args:
            job: Job posting to tailor CV for (None for generic CV)
            style: CV style ("moderncv", "classic", etc.)
            output_path: Custom output path (None for default)
            target: Build target (CV or ATS)

        Returns:
            Path to generated PDF

        Raises:
            ValueError: If style is not supported
        """
        # Convert style string to CvStyle enum
        try:
            cv_style = CvStyle[style.upper()]
        except KeyError:
            valid_styles = [s.name.lower() for s in CvStyle]
            msg = f"Unsupported style: {style}. Valid styles: {', '.join(valid_styles)}"
            raise ValueError(msg) from None

        # Build CV
        paths = build_cv(
            candidate=self.candidate,
            templates_dir=self.config.paths.templates,
            output_dir=self.config.paths.output,
            target=target,
            job=job,
            style=cv_style,
            credit=self.config.cv.jobbot_signature,
        )

        # Find PDF in returned paths
        pdf_path = next((p for p in paths if p.suffix == ".pdf"), None)
        if pdf_path is None:
            msg = "PDF not found in build output"
            raise RuntimeError(msg)

        # Move to custom output path if specified
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            # Copy instead of rename to avoid cross-device link issues
            import shutil
            shutil.copy2(pdf_path, output_path)
            return output_path

        return pdf_path

    def build_for_job_id(
        self,
        job_id: str,
        *,
        style: str = "modern",
        output_path: Path | str | None = None,
    ) -> Path | None:
        """Build a CV tailored for a stored job by ID.

        Args:
            job_id: Job ID to build CV for
            style: CV style
            output_path: Custom output path

        Returns:
            Path to generated PDF, or None if job not found
        """
        from jobbot.db.engine import make_engine, make_session_factory
        from jobbot.jobs.repository import JobRepository

        engine = make_engine(self.config.database_path)
        session_factory = make_session_factory(engine)
        session = session_factory()

        try:
            repo = JobRepository(session)
            job = repo.get(job_id)

            if job is None:
                return None

            return self.build(job=job, style=style, output_path=output_path)
        finally:
            session.close()
            engine.dispose()

    def list_styles(self) -> list[str]:
        """List available CV styles.

        Returns:
            List of style names
        """
        return [style.name.lower().replace("_", "-") for style in CvStyle]
