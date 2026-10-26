"""Torre integration: search, learn portal, detect ATS, categorize fields."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jobbot.cli import run_cli
from jobbot.exit_codes import SUCCESS


@pytest.fixture
def workspace(tmp_path: Path, project_root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Clean workspace with profile for integration test."""
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "data" / "profile.yaml").write_text(
        (project_root / "data" / "profile.example.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _mock_torre_api(monkeypatch: pytest.MonkeyPatch, project_root: Path) -> None:
    """Install fixture-based fetcher so tests never hit the network."""
    import jobbot.adapters.torre.jobs as torre

    payload = json.loads(
        (project_root / "tests" / "fixtures" / "torre_search.json").read_text(encoding="utf-8")
    )

    class FixtureFetcher:
        def post_json(self, url: str, body: dict) -> dict:
            return payload

    monkeypatch.setattr(torre, "UrllibFetcher", FixtureFetcher)


def test_torre_search_learns_portal_and_detects_ats(
    workspace: Path, project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Full flow: search Torre, learn portal, detect Torre ATS, detect external ATS."""
    _mock_torre_api(monkeypatch, project_root)

    # Search Torre
    result = run_cli(
        ["torre", "search", "analista de datos", "--limit", "3"], standalone_mode=False
    )

    assert result == SUCCESS

    # Check portal was learned
    from jobbot.portals.registry import load_registry

    registry = load_registry(workspace / "data" / "portals.yaml")
    torre_portals = [p for p in registry.portals if "torre" in p.domain]
    assert len(torre_portals) == 1
    assert torre_portals[0].ats_kind == "torre"

    # Check jobs were stored in database
    from jobbot.config import JobbotConfig
    from jobbot.db.engine import make_engine, make_session_factory
    from jobbot.jobs.repository import JobRepository

    config = JobbotConfig(root=workspace)
    session = make_session_factory(make_engine(config.database_path))()
    repo = JobRepository(session)

    jobs = repo.list_all()
    assert len(jobs) >= 4  # Fixture has 4 results

    # Check that external ATS was detected (Greenhouse job in fixture)
    from jobbot.portals.detect import AtsKind

    ats_kinds = [job.ats_kind for job in jobs]

    assert AtsKind.TORRE.value in ats_kinds
    assert AtsKind.GREENHOUSE.value in ats_kinds


def test_torre_application_adapter_detects_external_redirect(
    project_root: Path,
) -> None:
    """Torre adapter correctly identifies jobs that redirect to external ATS."""
    from jobbot.adapters.base import ApplyMethod
    from jobbot.adapters.torre.apply import TorreApplicationAdapter
    from jobbot.adapters.torre.jobs import job_from_api_item

    payload = json.loads(
        (project_root / "tests" / "fixtures" / "torre_external_redirect.json").read_text(
            encoding="utf-8"
        )
    )
    job = job_from_api_item(payload)

    adapter = TorreApplicationAdapter()
    method = adapter.detect_method(job)

    assert method == ApplyMethod.EXTERNAL_ATS
    assert "greenhouse.io" in job.url


def test_torre_portal_detector_recognizes_urls() -> None:
    """Portal detector correctly identifies Torre URLs."""
    from jobbot.portals.detectors.torre_detector import TorrePortalDetector

    detector = TorrePortalDetector()

    # Torre.ai domain
    result_ai = detector.detect("https://torre.ai/post/abc123")
    assert result_ai.is_job_portal is True
    assert result_ai.confidence == 1.0
    assert result_ai.ats_kind == "torre"

    # Torre.co domain
    result_co = detector.detect("https://torre.co/jobs/xyz789")
    assert result_co.is_job_portal is True

    # Non-Torre URL
    result_other = detector.detect("https://linkedin.com/jobs/view/123")
    assert result_other.is_job_portal is False


def test_torre_field_extraction_provides_learning_data(project_root: Path) -> None:
    """Torre field extraction converts API data to pseudo-form-fields for learning."""
    from jobbot.adapters.torre.field_extraction import extract_torre_fields

    payload = json.loads(
        (project_root / "tests" / "fixtures" / "torre_opportunity.json").read_text(
            encoding="utf-8"
        )
    )

    fields = extract_torre_fields(payload)

    assert len(fields) > 0

    # Check remote work field
    remote_fields = [f for f in fields if "remote" in f.name.lower()]
    assert len(remote_fields) > 0

    # Check skill experience fields
    skill_fields = [f for f in fields if "skill" in f.name.lower()]
    assert len(skill_fields) > 0

    # Verify field structure
    first_field = fields[0]
    assert hasattr(first_field, "name")
    assert hasattr(first_field, "label")
    assert hasattr(first_field, "kind")
