"""Get on Board API parsing tests."""

from __future__ import annotations

from typing import Any
from urllib.error import HTTPError

import pytest

from jobbot.adapters.getonboard import jobs as gob
from jobbot.adapters.getonboard.jobs import (
    job_from_api_item,
    remember_portal_from_url,
    search_jobs_api,
)
from jobbot.config import JobbotConfig
from jobbot.portals.detect import AtsKind
from jobbot.portals.registry import default_portals_path, load_registry


def test_job_from_api_item_maps_spanish_fields() -> None:
    item = {
        "id": "applied-scientist-neuralworks-santiago-e3c8",
        "type": "job",
        "attributes": {
            "title": "Applied Scientist",
            "description_headline": "Calificaciones clave",
            "description": "<ul><li>Dominio de Python</li><li>Español nativo</li></ul>",
            "projects": '<a href="https://www.getonbrd.com/companies/neuralworks">NeuralWorks</a>',
            "countries": ["Chile"],
            "lang": "es",
            "remote": False,
            "remote_modality": "hybrid",
        },
    }
    item["links"] = {
        "public_url": "https://www.getonbrd.com/jobs/applied-scientist-neuralworks-santiago-e3c8"
    }
    job = job_from_api_item(item)
    assert job.source == "getonboard"
    assert job.title == "Applied Scientist"
    assert job.company == "NeuralWorks"
    assert job.location and "Chile" in job.location
    assert job.ats_kind == AtsKind.GETONBOARD.value
    assert "Spanish" in job.language_requirements
    assert "Python" in job.description
    assert job.skills
    assert any("python" in s.lower() for s in job.skills)
    assert job.ats_url and "getonbrd.com" in job.ats_url


def test_company_name_from_company_api_id(monkeypatch: Any) -> None:
    gob._company_name_cache.clear()

    def fake_fetch(company_id: int) -> str | None:
        assert company_id == 10681
        return "NeuralWorks"

    monkeypatch.setattr(gob, "_fetch_company_name", fake_fetch)
    item = {
        "id": "applied-scientist-remote-us-abcd",
        "attributes": {
            "title": "Applied Scientist",
            "company": {"data": {"id": 10681, "type": "company"}},
            "description": "<p>Python</p>",
            "countries": ["Remote"],
            "lang": "es",
        },
        "links": {"public_url": "https://www.getonbrd.com/jobs/x"},
    }
    job = job_from_api_item(item)
    assert job.company == "NeuralWorks"


def test_remember_portal_from_url_learns_host(tmp_path: Any) -> None:
    root = tmp_path
    (root / "data").mkdir()
    config = JobbotConfig(root=root)
    domain = remember_portal_from_url(
        config,
        "https://boards.greenhouse.io/acme/jobs/1",
        notes="from linkedin post",
    )
    assert domain == "boards.greenhouse.io"
    registry = load_registry(default_portals_path(root))
    entry = registry.find("boards.greenhouse.io")
    assert entry is not None
    assert entry.ats_kind == AtsKind.GREENHOUSE
    assert entry.notes == "from linkedin post"


@pytest.mark.integration
def test_search_jobs_api_smoke() -> None:
    """Smoke test for GetOnBoard API.
    
    Skips when API returns 403 (rate limit, IP block, or access restriction).
    This is expected behavior when the portal's policy changes.
    """
    try:
        items = search_jobs_api("data scientist", per_page=2)
    except HTTPError as e:
        if e.code == 403:
            pytest.skip(f"GetOnBoard API returned 403 Forbidden: {e}")
        raise
    
    assert isinstance(items, list)
    assert len(items) >= 1
    assert "attributes" in items[0]
