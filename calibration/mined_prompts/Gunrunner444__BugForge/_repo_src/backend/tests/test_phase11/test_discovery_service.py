"""Tests for discovery service filter logic (Phase 11)."""

from __future__ import annotations

from app.core.config import Settings
from app.services.discovery_service import GitHubDiscoveryService, build_criteria_snapshot


def make_settings(**overrides: object) -> Settings:
    defaults: dict[str, object] = {
        "discovery_min_stars": 1000,
        "discovery_max_stars": 0,
        "discovery_skip_forks": True,
        "discovery_skip_archived": True,
        "discovery_require_license": True,
        "discovery_languages": "Python",
        "discovery_max_staleness_days": 365,
        "discovery_max_size_kb": 50_000,
        "discovery_excluded_topics": "exploit,malware",
        "discovery_excluded_owners": "bad_org",
        "discovery_daily_repo_limit": 100,
        "discovery_mode": "disabled",
        "ai_provider": "mock",
    }
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]


def make_github_item(**overrides: object) -> dict[str, object]:
    """Construct a minimal GitHub API search result item."""
    base: dict[str, object] = {
        "id": 12345,
        "name": "testrepo",
        "full_name": "testorg/testrepo",
        "html_url": "https://github.com/testorg/testrepo",
        "clone_url": "https://github.com/testorg/testrepo.git",
        "stargazers_count": 2000,
        "fork": False,
        "archived": False,
        "default_branch": "main",
        "language": "Python",
        "license": {"key": "mit", "name": "MIT License"},
        "size": 5000,
        "open_issues_count": 10,
        "topics": ["python", "web"],
        "description": "A test repo",
        "updated_at": "2024-01-01T00:00:00Z",
        "pushed_at": "2024-06-01T00:00:00Z",
        "owner": {"login": "testorg", "type": "Organization"},
    }
    base.update(overrides)
    return base


class TestDiscoveryFilters:
    def test_passes_normal_item(self) -> None:
        svc = GitHubDiscoveryService(make_settings())
        item = make_github_item()
        assert svc._passes_filters(item, set(), set()) is True

    def test_excluded_owner_blocked(self) -> None:
        svc = GitHubDiscoveryService(make_settings())
        item = make_github_item(owner={"login": "bad_org"})
        assert svc._passes_filters(item, set(), {"bad_org"}) is False

    def test_excluded_topic_blocked(self) -> None:
        svc = GitHubDiscoveryService(make_settings())
        item = make_github_item(topics=["exploit", "python"])
        assert svc._passes_filters(item, {"exploit"}, set()) is False

    def test_size_limit_enforced(self) -> None:
        svc = GitHubDiscoveryService(make_settings(discovery_max_size_kb=1_000))
        item = make_github_item(size=50_000)
        assert svc._passes_filters(item, set(), set()) is False

    def test_no_license_blocked_when_required(self) -> None:
        svc = GitHubDiscoveryService(make_settings())
        item = make_github_item(license=None)
        assert svc._passes_filters(item, set(), set()) is False

    def test_non_osi_license_blocked(self) -> None:
        svc = GitHubDiscoveryService(make_settings())
        item = make_github_item(license={"key": "proprietary"})
        assert svc._passes_filters(item, set(), set()) is False

    def test_license_not_required_passes_without_license(self) -> None:
        svc = GitHubDiscoveryService(make_settings(discovery_require_license=False))
        item = make_github_item(license=None)
        assert svc._passes_filters(item, set(), set()) is True


class TestParseItem:
    def test_parse_basic_item(self) -> None:
        item = make_github_item()
        repo = GitHubDiscoveryService._parse_item(item)
        assert repo.github_repo_id == 12345
        assert repo.full_name == "testorg/testrepo"
        assert repo.stars == 2000
        assert repo.is_fork is False
        assert repo.license_key == "mit"
        assert repo.primary_language == "Python"
        assert repo.last_pushed_at is not None

    def test_parse_item_no_license(self) -> None:
        item = make_github_item(license=None)
        repo = GitHubDiscoveryService._parse_item(item)
        assert repo.license_key is None

    def test_parse_item_no_owner(self) -> None:
        item = make_github_item(owner=None)
        repo = GitHubDiscoveryService._parse_item(item)
        assert repo.owner == ""


class TestCriteriaSnapshot:
    def test_snapshot_contains_key_fields(self) -> None:
        s = make_settings()
        snap = build_criteria_snapshot(s)
        assert "min_stars" in snap
        assert "languages" in snap
        assert "require_license" in snap
        assert snap["min_stars"] == 1000
        assert "Python" in snap["languages"]

    def test_snapshot_is_json_serialisable(self) -> None:
        import json

        s = make_settings()
        snap = build_criteria_snapshot(s)
        # Should not raise
        json.dumps(snap)


class TestQueryBuilding:
    def test_query_includes_min_stars(self) -> None:
        svc = GitHubDiscoveryService(make_settings(discovery_min_stars=500))
        query = svc._build_search_query_for_language("Python")
        assert "stars:>=500" in query

    def test_query_includes_archived_false(self) -> None:
        svc = GitHubDiscoveryService(make_settings())
        query = svc._build_search_query_for_language(None)
        assert "archived:false" in query

    def test_query_includes_fork_false(self) -> None:
        svc = GitHubDiscoveryService(make_settings())
        query = svc._build_search_query_for_language(None)
        assert "fork:false" in query

    def test_query_includes_language(self) -> None:
        svc = GitHubDiscoveryService(make_settings())
        query = svc._build_search_query_for_language("Rust")
        assert "language:Rust" in query

    def test_query_max_stars_range(self) -> None:
        svc = GitHubDiscoveryService(make_settings(discovery_max_stars=5_000))
        query = svc._build_search_query_for_language(None)
        assert "stars:1000..5000" in query
