"""ATS search-query templates (site: host dorks) — no search-engine calls."""

from jobbot.jobs.ats_queries import (
    ATS_SEARCH_HOSTS,
    AtsSearchQuery,
    build_ats_search_queries,
    format_queries_help,
)
from jobbot.portals.detect import AtsKind


def test_default_queries_cover_ashby_greenhouse_lever() -> None:
    qs = build_ats_search_queries(regions=("Spain",), remote=True)
    kinds = {q.kind for q in qs}
    assert kinds == {AtsKind.ASHBY, AtsKind.GREENHOUSE, AtsKind.LEVER}
    by_kind = {q.kind: q.query for q in qs}
    assert by_kind[AtsKind.ASHBY] == 'site:jobs.ashbyhq.com "remote" "Spain"'
    assert by_kind[AtsKind.GREENHOUSE] == 'site:greenhouse.io "remote" "Spain"'
    assert by_kind[AtsKind.LEVER] == 'site:lever.co "remote" "Spain"'


def test_emea_and_keywords() -> None:
    qs = build_ats_search_queries(
        kinds=(AtsKind.ASHBY,),
        regions=("EMEA",),
        remote=True,
        keywords=("data scientist",),
    )
    assert len(qs) == 1
    assert qs[0].query == 'site:jobs.ashbyhq.com "remote" "EMEA" "data scientist"'


def test_unknown_ats_kind_skipped() -> None:
    qs = build_ats_search_queries(kinds=(AtsKind.WORKDAY,), regions=("Chile",))
    assert qs == []


def test_hosts_match_detect_module() -> None:
    assert ATS_SEARCH_HOSTS[AtsKind.ASHBY] == "jobs.ashbyhq.com"
    assert "greenhouse" in ATS_SEARCH_HOSTS[AtsKind.GREENHOUSE]
    assert "lever" in ATS_SEARCH_HOSTS[AtsKind.LEVER]


def test_help_text_forbids_jobbot_searching_and_ats_rewrite() -> None:
    qs = [
        AtsSearchQuery(
            kind=AtsKind.ASHBY,
            host="jobs.ashbyhq.com",
            query='site:jobs.ashbyhq.com "remote"',
            regions=(),
            remote=True,
        )
    ]
    help_text = format_queries_help(qs)
    assert "does not query" in help_text.casefold()
    assert "search-results" in help_text
    assert "profile.yaml" in help_text
    assert "ATS rewrite" in help_text or "rewrite" in help_text.casefold()
