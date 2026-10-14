"""GitHub repository discovery service (v1.1.0).

Queries the GitHub Search API to find public repositories matching
configured criteria. Uses the existing GitHub token from settings.

Security:
  - Token comes from settings only — never logged, never stored.
  - All returned data is treated as untrusted until after eligibility screening.
  - Rate-limit headers are observed; we back off on 403/429.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import httpx

from app.core.config import Settings

logger = logging.getLogger(__name__)

_GITHUB_API_BASE = "https://api.github.com"
_SEARCH_PER_PAGE = 30  # GitHub max per page for search
_OSI_LICENSES = frozenset(
    {
        "mit",
        "apache-2.0",
        "gpl-2.0",
        "gpl-3.0",
        "lgpl-2.1",
        "lgpl-3.0",
        "bsd-2-clause",
        "bsd-3-clause",
        "mpl-2.0",
        "cddl-1.0",
        "epl-2.0",
        "agpl-3.0",
        "unlicense",
        "isc",
        "cc0-1.0",
        "eupl-1.1",
        "eupl-1.2",
        "ms-pl",
        "ms-rl",
        "0bsd",
        "ncsa",
    }
)


@dataclass
class DiscoveredRepo:
    """Lightweight container for one GitHub search result."""

    github_repo_id: int
    owner: str
    name: str
    full_name: str
    html_url: str
    clone_url: str
    stars: int
    is_fork: bool
    is_archived: bool
    default_branch: str
    primary_language: str | None
    license_key: str | None
    size_kb: int
    open_issues: int
    topics: list[str]
    description: str | None
    last_updated_at: datetime | None
    last_pushed_at: datetime | None


@dataclass
class DiscoveryResult:
    """Summary returned after one discovery sweep."""

    repos: list[DiscoveredRepo] = field(default_factory=list)
    api_requests: int = 0
    rate_limit_remaining: int | None = None
    rate_limit_reset: int | None = None  # unix timestamp
    error: str | None = None


class GitHubDiscoveryService:
    """Queries GitHub Search API for public repository candidates."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def _make_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self._settings.github_token:
            # Token is never logged — only sent in the Authorization header
            headers["Authorization"] = f"Bearer {self._settings.github_token}"
        return headers

    async def discover(self, max_pages: int = 5) -> DiscoveryResult:
        """Run one discovery sweep and return discovered repos.

        Respects GitHub rate limits. Never executes repository code.
        """
        result = DiscoveryResult()
        languages = self._settings.get_discovery_languages()
        excluded_topics = self._settings.get_excluded_topics()
        excluded_owners = self._settings.get_excluded_owners()
        daily_limit = self._settings.discovery_daily_repo_limit

        seen_ids: set[int] = set()

        # If multiple languages, issue one search per language to avoid
        # GitHub limiting language-specific queries to a single qualifier.
        language_queries: list[str | None] = list(languages) if languages else [None]

        async with httpx.AsyncClient(
            headers=self._make_headers(),
            timeout=30.0,
            follow_redirects=True,
        ) as client:
            for lang in language_queries:
                base_query = self._build_search_query_for_language(lang)
                logger.info("Discovery search query: %s", base_query)

                for page in range(1, max_pages + 1):
                    if daily_limit and len(result.repos) >= daily_limit:
                        logger.info("Discovery daily limit reached (%d).", daily_limit)
                        break

                    try:
                        response = await client.get(
                            f"{_GITHUB_API_BASE}/search/repositories",
                            params={
                                "q": base_query,
                                "sort": "stars",
                                "order": "desc",
                                "per_page": _SEARCH_PER_PAGE,
                                "page": page,
                            },
                        )
                        result.api_requests += 1
                        self._update_rate_limit(result, response)

                        if response.status_code in {403, 429}:
                            logger.warning(
                                "GitHub rate limit hit (%d); stopping discovery.",
                                response.status_code,
                            )
                            result.error = "github_rate_limit"
                            return result

                        if response.status_code == 422:
                            logger.warning("GitHub search query rejected: %s", base_query)
                            break

                        response.raise_for_status()
                        data = response.json()

                    except httpx.HTTPStatusError as exc:
                        logger.error("GitHub search error: %s", exc)
                        result.error = f"http_error:{exc.response.status_code}"
                        return result
                    except Exception as exc:
                        logger.error("GitHub discovery network error: %s", exc)
                        result.error = f"network_error:{exc}"
                        return result

                    items = data.get("items", [])
                    if not items:
                        break  # no more results for this language

                    for item in items:
                        if daily_limit and len(result.repos) >= daily_limit:
                            break

                        repo_id = item.get("id")
                        if repo_id in seen_ids:
                            continue
                        seen_ids.add(repo_id)

                        # Apply post-fetch filters
                        if not self._passes_filters(item, excluded_topics, excluded_owners):
                            continue

                        repo = self._parse_item(item)
                        result.repos.append(repo)

                    # If fewer than per_page items, we've exhausted results
                    if len(items) < _SEARCH_PER_PAGE:
                        break

                    # Polite async sleep to avoid secondary rate-limit hammering
                    await asyncio.sleep(1)

        return result

    def _build_search_query_for_language(self, language: str | None) -> str:
        parts: list[str] = []
        if self._settings.discovery_max_stars > 0:
            parts.append(
                f"stars:{self._settings.discovery_min_stars}..{self._settings.discovery_max_stars}"
            )
        else:
            parts.append(f"stars:>={self._settings.discovery_min_stars}")
        if self._settings.discovery_skip_archived:
            parts.append("archived:false")
        if self._settings.discovery_skip_forks:
            parts.append("fork:false")
        if language:
            parts.append(f"language:{language}")
        if self._settings.discovery_max_staleness_days > 0:
            cutoff = datetime.now(UTC) - timedelta(days=self._settings.discovery_max_staleness_days)
            parts.append(f"pushed:>{cutoff.strftime('%Y-%m-%d')}")
        return " ".join(parts)

    def _passes_filters(
        self,
        item: dict[str, object],
        excluded_topics: set[str],
        excluded_owners: set[str],
    ) -> bool:
        """Apply additional post-fetch filters not expressible in the search query."""
        # Excluded owner
        owner_raw = item.get("owner")
        if isinstance(owner_raw, dict):
            owner_login = str(owner_raw.get("login") or "")
            if owner_login.lower() in excluded_owners:
                return False

        # Excluded topics
        topics_raw = item.get("topics", [])
        topics: list[str] = [str(t) for t in topics_raw] if isinstance(topics_raw, list) else []
        if excluded_topics and topics:
            if any(t.lower() in excluded_topics for t in topics):
                return False

        # Size limit
        size_raw = item.get("size", 0)
        size_kb = int(size_raw) if isinstance(size_raw, (int, float)) else 0
        if (
            self._settings.discovery_max_size_kb > 0
            and size_kb > self._settings.discovery_max_size_kb
        ):
            return False

        # License (OSI check)
        if self._settings.discovery_require_license:
            license_raw = item.get("license")
            license_key = ""
            if isinstance(license_raw, dict):
                lk = license_raw.get("key")
                license_key = str(lk).lower() if lk else ""
            if not license_key or license_key not in _OSI_LICENSES:
                return False

        return True

    @staticmethod
    def _parse_item(item: dict[str, object]) -> DiscoveredRepo:
        """Parse one GitHub API search item into a DiscoveredRepo."""
        owner_login = ""
        owner_obj = item.get("owner")
        if isinstance(owner_obj, dict):
            owner_login = str(owner_obj.get("login", ""))

        license_key: str | None = None
        license_obj = item.get("license")
        if isinstance(license_obj, dict):
            lk = license_obj.get("key")
            license_key = str(lk) if lk is not None else None

        def _dt(value: object) -> datetime | None:
            if not value or not isinstance(value, str):
                return None
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return None

        def _int(value: object, default: int = 0) -> int:
            if isinstance(value, int):
                return value
            if isinstance(value, str):
                try:
                    return int(value)
                except ValueError:
                    return default
            return default

        topics_raw = item.get("topics", [])
        topics: list[str] = [str(t) for t in topics_raw] if isinstance(topics_raw, list) else []

        desc = item.get("description")
        lang = item.get("language")

        return DiscoveredRepo(
            github_repo_id=_int(item.get("id")),
            owner=owner_login,
            name=str(item.get("name", "")),
            full_name=str(item.get("full_name", "")),
            html_url=str(item.get("html_url", "")),
            clone_url=str(item.get("clone_url", "")),
            stars=_int(item.get("stargazers_count")),
            is_fork=bool(item.get("fork", False)),
            is_archived=bool(item.get("archived", False)),
            default_branch=str(item.get("default_branch", "main")),
            primary_language=str(lang) if lang is not None else None,
            license_key=license_key,
            size_kb=_int(item.get("size")),
            open_issues=_int(item.get("open_issues_count")),
            topics=topics,
            description=str(desc) if desc is not None else None,
            last_updated_at=_dt(item.get("updated_at")),
            last_pushed_at=_dt(item.get("pushed_at")),
        )

    @staticmethod
    def _update_rate_limit(result: DiscoveryResult, response: httpx.Response) -> None:
        try:
            result.rate_limit_remaining = int(response.headers.get("X-RateLimit-Remaining", -1))
            result.rate_limit_reset = int(response.headers.get("X-RateLimit-Reset", 0))
        except (ValueError, TypeError):
            pass


def build_criteria_snapshot(settings: Settings) -> dict[str, object]:
    """Return a JSON-serialisable snapshot of the current discovery criteria."""
    return {
        "min_stars": settings.discovery_min_stars,
        "max_stars": settings.discovery_max_stars,
        "max_size_kb": settings.discovery_max_size_kb,
        "max_files": settings.discovery_max_files,
        "languages": settings.get_discovery_languages(),
        "require_license": settings.discovery_require_license,
        "skip_forks": settings.discovery_skip_forks,
        "skip_archived": settings.discovery_skip_archived,
        "max_staleness_days": settings.discovery_max_staleness_days,
        "excluded_topics": list(settings.get_excluded_topics()),
        "excluded_owners": list(settings.get_excluded_owners()),
        "daily_repo_limit": settings.discovery_daily_repo_limit,
    }
