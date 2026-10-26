"""Post age: decoded from the LinkedIn activity id, filtered by max_age_days."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from jobbot.adapters.linkedin.sweep import posted_at_from_url
from jobbot.jobs.freshness import DEFAULT_MAX_AGE_DAYS, age_label, is_fresh

# Real post that JobBot stored as J0048; the user knew it was 10 months old.
J0048_URL = (
    "https://es.linkedin.com/posts/dcarrenom_nttdata-datascience-machinelearning-"
    "activity-7391867181325733888-y7P9"
)
NOW = datetime(2026, 9, 16, tzinfo=UTC)


def test_posted_at_from_posts_url() -> None:
    posted = posted_at_from_url(J0048_URL)
    assert posted is not None
    assert posted.astimezone(UTC).date() == datetime(2025, 11, 5, tzinfo=UTC).date()


def test_posted_at_from_feed_update_urn() -> None:
    url = "https://www.linkedin.com/feed/update/urn:li:activity:7391867181325733888/"
    posted = posted_at_from_url(url)
    assert posted is not None
    assert posted.astimezone(UTC).date() == datetime(2025, 11, 5, tzinfo=UTC).date()


def test_posted_at_from_share_id() -> None:
    """The link copied for J0044 used share-<id>, same encoding as activity-<id>."""
    url = (
        "https://www.linkedin.com/posts/gabriela-forton-714a86260_en-era-talent-"
        "share-7371583217478574080-YDc9/"
    )
    posted = posted_at_from_url(url)
    assert posted is not None
    assert posted.astimezone(UTC).date() == datetime(2025, 9, 10, tzinfo=UTC).date()


def test_posted_at_is_none_for_profile_fallback() -> None:
    assert posted_at_from_url("https://www.linkedin.com/in/rodrigo-fernandez-4272437/") is None
    assert posted_at_from_url(None) is None
    assert posted_at_from_url("") is None


def test_posted_at_ignores_implausible_ids() -> None:
    """A short numeric run is not an activity id; refuse to invent a date."""
    assert posted_at_from_url("https://www.linkedin.com/posts/x_activity-42-abc") is None


def test_is_fresh_drops_the_ten_month_old_post() -> None:
    assert DEFAULT_MAX_AGE_DAYS == 30
    posted = posted_at_from_url(J0048_URL)
    assert is_fresh(posted, max_age_days=30, now=NOW) is False
    assert is_fresh(posted, max_age_days=0, now=NOW) is True  # 0 = sin filtro


def test_is_fresh_keeps_recent_and_unknown() -> None:
    assert is_fresh(NOW - timedelta(days=5), max_age_days=30, now=NOW) is True
    # Unknown age is never dropped: JobBot does not hide work it cannot date.
    assert is_fresh(None, max_age_days=30, now=NOW) is True


def test_is_fresh_tolerates_naive_datetimes() -> None:
    naive = datetime(2026, 9, 10)  # noqa: DTZ001 — legacy rows have no tzinfo
    assert is_fresh(naive, max_age_days=30, now=NOW) is True


def test_age_label_reads_like_the_feed() -> None:
    assert age_label(None) == "?"
    assert age_label(NOW, now=NOW) == "hoy"
    assert age_label(NOW - timedelta(days=1), now=NOW) == "1 día"
    assert age_label(NOW - timedelta(days=5), now=NOW) == "5 días"
    assert age_label(NOW - timedelta(days=20), now=NOW) == "3 semanas"  # 2.9 semanas
    assert age_label(NOW - timedelta(days=60), now=NOW) == "2 meses"
    assert age_label(NOW - timedelta(days=31), now=NOW) == "1 mes"
    assert age_label(posted_at_from_url(J0048_URL), now=NOW) == "10 meses"
    assert age_label(NOW - timedelta(days=800), now=NOW) == "2 años"
    # 315 days is 10 months, not "1 año": babel's default threshold would round it up.
    assert age_label(NOW - timedelta(days=315), now=NOW) == "10 meses"


def test_config_defaults_and_reads_max_age_days(tmp_path: Path) -> None:
    from jobbot.config import load_config

    assert load_config(tmp_path).search.max_age_days == 30

    (tmp_path / ".jobbot.toml").write_text("[search]\nmax_age_days = 7\n", encoding="utf-8")
    assert load_config(tmp_path).search.max_age_days == 7


def test_sweep_drops_the_stale_post_and_keeps_the_fresh_one() -> None:
    """The 10-month-old NTT Data post must not reach the shortlist."""
    from jobbot.adapters.linkedin.posts_source import (
        MODERN_POST_CARD,
        collect_jobs_from_feed_page,
    )
    from jobbot.jobs.sources import JobSearchQuery
    from tests.unit.test_linkedin_sweep import _FakeSearchPage

    stale = (
        "Publicación en el feed\nDiego Carreño\n"
        "Buscamos Data Scientist Senior para Airport Operations en Santiago.\n"
        "Enviar CV a dnavarna@nttdata.com"
    )
    fresh = (
        "Publicación en el feed\nAna Recruiter\n"
        "Buscamos Data Scientist en Santiago de Chile.\nEnviar CV a ana@empresa.cl"
    )
    recent_id = (int(NOW.timestamp() * 1000) - 3 * 86_400_000) << 22
    cards = [
        (stale, [f"https://www.linkedin.com/feed/update/urn:li:activity:{7391867181325733888}/"]),
        (fresh, [f"https://www.linkedin.com/feed/update/urn:li:activity:{recent_id}/"]),
    ]
    page = _FakeSearchPage(cards, selector=MODERN_POST_CARD)
    query = JobSearchQuery(query="data scientist", limit=10, max_age_days=30)
    jobs = collect_jobs_from_feed_page(page, query, resolve_short_links=False, now=NOW)

    assert [j.ats_url for j in jobs] == ["mailto:ana@empresa.cl"]

    page = _FakeSearchPage(cards, selector=MODERN_POST_CARD)
    kept = collect_jobs_from_feed_page(
        page,
        JobSearchQuery(query="data scientist", limit=10, max_age_days=0),
        resolve_short_links=False,
        now=NOW,
    )
    assert len(kept) == 2
