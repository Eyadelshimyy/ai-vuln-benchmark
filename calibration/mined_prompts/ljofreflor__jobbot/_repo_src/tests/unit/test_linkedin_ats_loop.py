"""End-to-end loop: LinkedIn fixture → portals → market suggest (no invention)."""

from pathlib import Path

from jobbot.adapters.linkedin.posts_source import LinkedInPostJobSource
from jobbot.config import JobbotConfig, PathsConfig
from jobbot.models.candidate import Candidate
from jobbot.portals.detect import AtsKind, detect_ats
from jobbot.portals.registry import PortalRegistry, domain_from_url
from jobbot.profile.market import suggest_from_market
from tests.fixtures.profile import sample_profile_dict


def test_fixture_sweep_learns_portals_and_market_gaps(project_root: Path, tmp_path: Path) -> None:
    config = JobbotConfig(paths=PathsConfig(output=tmp_path / "out"), root=project_root)
    source = LinkedInPostJobSource(config)
    jobs = source.search_from_fixture(
        project_root / "tests/fixtures/linkedin_posts.txt",
        query="data",
    )
    assert len(jobs) >= 3
    kinds = {j.ats_kind for j in jobs if j.ats_kind}
    assert "greenhouse" in kinds
    assert "getonboard" in kinds or any(
        j.ats_url and detect_ats(j.ats_url) == AtsKind.GETONBOARD for j in jobs
    )

    registry = PortalRegistry()
    for job in jobs:
        if not job.ats_url:
            continue
        domain = domain_from_url(job.ats_url)
        kind = AtsKind(job.ats_kind) if job.ats_kind else detect_ats(job.ats_url)
        registry.upsert(domain=domain, ats_kind=kind, example_url=job.ats_url, seen=True)
    domains = {p.domain for p in registry.portals}
    assert "boards.greenhouse.io" in domains

    candidate = Candidate.model_validate(sample_profile_dict())
    before = list(candidate.skills.all_skills())
    suggestion = suggest_from_market(candidate, jobs, min_count=1)
    # suggest_from_market must not mutate the candidate
    assert candidate.skills.all_skills() == before
    assert suggestion.market_terms or suggestion.missing_suspected or suggestion.present
