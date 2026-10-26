"""Offline coverage for helpers that used to sit below the main gate."""

from __future__ import annotations

from pathlib import Path

import pytest

from jobbot.config import JobbotConfig, PathsConfig
from jobbot.jobs.normalization import normalize_many, normalize_skill
from jobbot.jobs.sources import get_job_source
from jobbot.ops.precommit import tests_needed as precommit_tests_needed
from jobbot.portals.detect import AtsKind
from jobbot.portals.registry import (
    PortalRegistry,
    domain_from_url,
    load_registry,
    save_registry,
)
from jobbot.publications.doi import DoiMetadata, _parse_work


def test_normalize_many_dedupes_aliases() -> None:
    assert normalize_many(["PostgreSQL", "postgres", "SQL"]) == ["postgresql", "sql"]
    assert normalize_skill("") == ""


def test_precommit_triggers_on_policy_and_code() -> None:
    assert precommit_tests_needed(["README.md"]) is False
    assert precommit_tests_needed(["src/jobbot/cli.py"]) is True
    assert precommit_tests_needed(["AGENTS.md"]) is True
    assert precommit_tests_needed(["templates/cv.tex.j2"]) is True


def test_get_job_source_factory(tmp_path: Path) -> None:
    config = JobbotConfig(root=tmp_path, paths=PathsConfig())
    assert get_job_source("torre", config).search_jobs  # type: ignore[attr-defined]
    assert get_job_source("getonboard", config).search_jobs  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="Unknown"):
        get_job_source("monster", config)  # type: ignore[arg-type]


def test_portal_registry_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "portals.yaml"
    registry = PortalRegistry()
    registry.upsert(
        domain="boards.greenhouse.io",
        ats_kind=AtsKind.GREENHOUSE,
        example_url="https://boards.greenhouse.io/x",
    )
    save_registry(registry, path)
    loaded = load_registry(path)
    assert loaded.find("boards.greenhouse.io") is not None
    assert domain_from_url("https://www.boards.greenhouse.io/acme") == "boards.greenhouse.io"


def test_parse_crossref_work_is_offline() -> None:
    meta = _parse_work(
        {
            "DOI": "10.1000/xyz",
            "title": ["A Study"],
            "container-title": ["Journal"],
            "author": [{"given": "Ada", "family": "Lovelace"}],
            "published-print": {"date-parts": [[2020, 1, 2]]},
        }
    )
    assert isinstance(meta, DoiMetadata)
    assert meta.doi == "10.1000/xyz"
    assert meta.title == "A Study"
    assert meta.journal == "Journal"
    assert meta.year == 2020
    assert meta.authors == ["Ada Lovelace"]
