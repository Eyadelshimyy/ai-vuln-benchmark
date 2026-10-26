"""Job source factory tests."""

from jobbot.config import JobbotConfig
from jobbot.jobs.sources import get_job_source


def test_get_job_source_linkedin_post(tmp_path: object) -> None:
    from jobbot.adapters.linkedin.posts_source import LinkedInPostJobSource

    config = JobbotConfig(root=tmp_path)  # type: ignore[arg-type]
    source = get_job_source("linkedin_post", config)
    assert isinstance(source, LinkedInPostJobSource)


def test_get_job_source_getonboard(tmp_path: object) -> None:
    from jobbot.adapters.getonboard.jobs import GetOnBoardJobSource

    config = JobbotConfig(root=tmp_path)  # type: ignore[arg-type]
    source = get_job_source("getonboard", config)
    assert isinstance(source, GetOnBoardJobSource)
