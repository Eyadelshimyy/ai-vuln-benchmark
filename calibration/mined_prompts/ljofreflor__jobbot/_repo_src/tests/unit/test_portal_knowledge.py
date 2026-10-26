"""Portal knowledge: local registry + distributed seed + built-in host rules."""

from __future__ import annotations

from pathlib import Path

from jobbot.config import JobbotConfig
from jobbot.portals.detect import AtsKind
from jobbot.portals.knowledge import PortalKnowledgeSource, lookup_portal
from jobbot.portals.registry import load_registry, save_registry


def test_getonbrd_is_known_from_the_tracked_seed() -> None:
    config = JobbotConfig(root=Path("/tmp/jobbot-empty-no-portals"))
    hit = lookup_portal(
        config,
        "https://www.getonbrd.com/empleos/data-science-analytics/applied-scientist-x",
    )
    assert hit.known
    assert hit.domain == "getonbrd.com"
    assert hit.ats_kind == AtsKind.GETONBOARD
    assert hit.source in {PortalKnowledgeSource.SEED, PortalKnowledgeSource.BUILTIN}


def test_unknown_host_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    config = JobbotConfig(root=tmp_path)
    hit = lookup_portal(config, "https://careers.totally-unknown-corp.example/jobs/1")
    assert not hit.known
    assert hit.source == PortalKnowledgeSource.UNKNOWN


def test_local_registry_wins_over_seed(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    config = JobbotConfig(root=tmp_path)
    from jobbot.portals.registry import PortalRegistry, default_portals_path

    registry = PortalRegistry()
    registry.upsert(
        domain="getonbrd.com",
        ats_kind=AtsKind.GETONBOARD,
        notes="local copy",
        registered=True,
    )
    save_registry(registry, default_portals_path(tmp_path))

    hit = lookup_portal(config, "https://www.getonbrd.com/jobs/x")
    assert hit.source == PortalKnowledgeSource.LOCAL
    assert hit.entry is not None
    assert hit.entry.registered is True
    assert load_registry(default_portals_path(tmp_path)).find("getonbrd.com") is not None
