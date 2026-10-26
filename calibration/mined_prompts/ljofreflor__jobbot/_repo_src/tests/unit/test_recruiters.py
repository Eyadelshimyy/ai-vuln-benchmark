"""Recruiter knowledge: public sources, no identities, promoted by hand."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jobbot.companies.models import KnowledgeStatus
from jobbot.companies.oneshot import FetchResult
from jobbot.recruiters.discover import discover_sources
from jobbot.recruiters.playbook import (
    PracticeKind,
    advisor_notes,
    extract_practices,
)
from jobbot.recruiters.sources import (
    RecruiterSource,
    default_recruiters_path,
    load_sources,
    promote_source,
    reject_source,
    save_sources,
    upsert_source,
)

ARTICLE = """
<html><head><title>How recruiters screen CVs · Hiring Notes</title></head>
<body>
<h1>How we screen CVs</h1>
<p>Recruiters spend about seven seconds on the first pass, so put the role title
and the last employer where the eye lands first.</p>
<h2>What we look for</h2>
<ul>
  <li>Quantify the outcome: a number beats an adjective.</li>
  <li>Keep one page per ten years of experience.</li>
  <li>Write the acronym and its full name the first time.</li>
</ul>
<p>Written by Jane Doe, recruiter at SomeCorp. Contact: jane.doe@somecorp.example</p>
</body></html>
"""

LOGIN_WALL = """
<html><head><title>Sign in to continue</title></head>
<body><h1>Log in</h1><form action='/session'>
<input type='email' name='email'><input type='password' name='password'>
</form><p>Members only.</p></body></html>
"""

ROBOTS_CLOSED = "User-agent: *\nDisallow: /\n"


@dataclass
class FakeFetcher:
    pages: dict[str, FetchResult]
    requested: list[str] | None = None

    def fetch(self, url: str) -> FetchResult:
        if self.requested is None:
            self.requested = []
        self.requested.append(url)
        return self.pages.get(url, FetchResult(url=url, status=404))


def _page(url: str, html: str, status: int = 200) -> FetchResult:
    return FetchResult(url=url, status=status, html=html)


def test_a_public_article_becomes_a_candidate_source() -> None:
    url = "https://hiring.example.org/how-we-screen"
    fetcher = FakeFetcher(pages={url: _page(url, ARTICLE)})

    report = discover_sources([url], fetcher)

    assert len(report.sources) == 1
    source = report.sources[0]
    assert source.status is KnowledgeStatus.CANDIDATE
    assert source.title
    assert source.practices


def test_no_person_is_ever_recorded() -> None:
    """The practice is the knowledge; who wrote it is not ours to keep."""
    url = "https://hiring.example.org/how-we-screen"
    fetcher = FakeFetcher(pages={url: _page(url, ARTICLE)})

    dumped = discover_sources([url], fetcher).sources[0].model_dump_json()

    assert "Jane Doe" not in dumped
    assert "jane.doe@somecorp.example" not in dumped
    assert "SomeCorp" not in dumped


def test_content_behind_a_login_is_left_alone() -> None:
    url = "https://members.example.org/secret-playbook"
    fetcher = FakeFetcher(pages={url: _page(url, LOGIN_WALL)})

    report = discover_sources([url], fetcher)

    assert report.sources == []
    assert url in report.skipped_login


def test_robots_disallow_is_obeyed_and_reported_as_skipped() -> None:
    url = "https://hiring.example.org/how-we-screen"
    fetcher = FakeFetcher(
        pages={
            "https://hiring.example.org/robots.txt": _page(
                "https://hiring.example.org/robots.txt", ROBOTS_CLOSED
            ),
            url: _page(url, ARTICLE),
        }
    )

    report = discover_sources([url], fetcher)

    assert report.sources == []
    assert url in report.skipped_robots
    assert url not in (fetcher.requested or [])


def test_a_refusal_is_not_an_absence() -> None:
    url = "https://hiring.example.org/how-we-screen"
    fetcher = FakeFetcher(pages={url: _page(url, "", status=403)})

    report = discover_sources([url], fetcher)

    assert report.sources == []
    assert url in report.refused


def test_practices_are_structured_not_a_blob() -> None:
    practices = extract_practices(ARTICLE)

    kinds = {practice.kind for practice in practices}
    assert PracticeKind.LAYOUT in kinds or PracticeKind.LANGUAGE in kinds
    assert all(practice.text for practice in practices)
    assert all(len(practice.text) < 300 for practice in practices)


def test_only_promoted_knowledge_reaches_the_advisor(tmp_path: Path) -> None:
    url = "https://hiring.example.org/how-we-screen"
    fetcher = FakeFetcher(pages={url: _page(url, ARTICLE)})
    source = discover_sources([url], fetcher).sources[0]
    path = default_recruiters_path(tmp_path)
    save_sources(upsert_source([], source), path)

    assert advisor_notes(tmp_path) == [], "a candidate source advises nobody"

    sources = promote_source(load_sources(path), url)
    save_sources(sources, path)

    notes = advisor_notes(tmp_path)
    assert notes
    assert all(isinstance(note, str) and note for note in notes)


def test_rejecting_a_source_keeps_it_out_for_good(tmp_path: Path) -> None:
    url = "https://hiring.example.org/how-we-screen"
    fetcher = FakeFetcher(pages={url: _page(url, ARTICLE)})
    source = discover_sources([url], fetcher).sources[0]
    path = default_recruiters_path(tmp_path)
    save_sources(reject_source(upsert_source([], source), url), path)

    stored = load_sources(path)

    assert stored[0].status is KnowledgeStatus.REJECTED
    assert advisor_notes(tmp_path) == []


def test_seeing_the_same_source_again_does_not_duplicate_it_nor_undo_promotion() -> None:
    url = "https://hiring.example.org/how-we-screen"
    fetcher = FakeFetcher(pages={url: _page(url, ARTICLE)})
    source = discover_sources([url], fetcher).sources[0]

    sources = promote_source(upsert_source([], source), url)
    sources = upsert_source(sources, source)

    assert len(sources) == 1
    assert sources[0].status is KnowledgeStatus.ACTIVE


def test_round_trip_keeps_the_practices(tmp_path: Path) -> None:
    url = "https://hiring.example.org/how-we-screen"
    fetcher = FakeFetcher(pages={url: _page(url, ARTICLE)})
    source = discover_sources([url], fetcher).sources[0]
    path = default_recruiters_path(tmp_path)

    save_sources([source], path)
    reloaded = load_sources(path)

    assert len(reloaded) == 1
    assert [p.text for p in reloaded[0].practices] == [p.text for p in source.practices]


def test_the_file_is_local_and_named_like_the_other_registries() -> None:
    path = default_recruiters_path(Path("/tmp/root"))

    assert path == Path("/tmp/root/data/recruiters.yaml")


def test_a_source_with_no_practice_is_not_stored() -> None:
    """A page about hiring that teaches nothing is not knowledge."""
    url = "https://hiring.example.org/about"
    empty = "<html><head><title>About us</title></head><body><p>We hire.</p></body></html>"
    fetcher = FakeFetcher(pages={url: _page(url, empty)})

    report = discover_sources([url], fetcher)

    assert report.sources == []
    assert url in report.nothing_learned


def test_practices_carry_no_field_specific_vocabulary() -> None:
    """Advice for a nurse and for an engineer comes out of the same extraction."""
    html = ARTICLE.replace("CVs", "resumes")

    practices = extract_practices(html)

    blob = " ".join(practice.text for practice in practices).casefold()
    for word in ("python", "sql", "nurse", "journalist", "engineer"):
        assert word not in blob


def test_source_model_refuses_to_hold_a_person() -> None:
    """The shape itself has nowhere to put a name."""
    fields = set(RecruiterSource.model_fields)

    for forbidden in ("author", "person", "name", "email", "contact"):
        assert forbidden not in fields
