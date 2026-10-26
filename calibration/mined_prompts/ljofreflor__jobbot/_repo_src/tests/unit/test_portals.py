"""Portal detect + registry tests."""

from pathlib import Path

from jobbot.portals.detect import (
    JOB_BOARD_KINDS,
    AtsKind,
    detect_ats,
    detect_ats_in_html,
    extract_http_urls,
    first_external_ats_url,
)
from jobbot.portals.registry import PortalRegistry, domain_from_url, load_registry, save_registry


def test_detect_ats_kinds() -> None:
    assert detect_ats("https://boards.greenhouse.io/acme/jobs/1") == AtsKind.GREENHOUSE
    assert detect_ats("https://jobs.lever.co/x/y") == AtsKind.LEVER
    assert detect_ats("https://company.wd1.myworkdayjobs.com/en-US/careers") == AtsKind.WORKDAY
    assert detect_ats("https://jobs.ashbyhq.com/c/r") == AtsKind.ASHBY
    assert detect_ats("https://www.getonbrd.com/jobs/x") == AtsKind.GETONBOARD
    assert detect_ats("https://acme.breezy.hr/p/abc-role") == AtsKind.BREEZY
    assert detect_ats("https://www.linkedin.com/posts/x") == AtsKind.LINKEDIN
    assert detect_ats("https://example.com/jobs/1") == AtsKind.UNKNOWN
    assert detect_ats("https://careers.neuralworks.cl/jobs/1") == AtsKind.UNKNOWN


def test_teamtailor_cdn_on_a_custom_career_host_is_evidence(project_root: Path) -> None:
    """careers.neuralworks.cl is not teamtailor.com; the page still loads Teamtailor's CDN."""
    html = (project_root / "tests/fixtures/teamtailor_career_page.html").read_text(encoding="utf-8")
    kind, evidence = detect_ats_in_html(html)
    assert kind == AtsKind.TEAMTAILOR
    assert "teamtailor" in evidence.casefold()


def test_first_external_ats_prefers_greenhouse() -> None:
    urls = extract_http_urls(
        "see https://www.linkedin.com/feed/update/1 and https://boards.greenhouse.io/acme/jobs/9"
    )
    url, kind = first_external_ats_url(urls)
    assert kind == AtsKind.GREENHOUSE
    assert url is not None and "greenhouse" in url


def test_first_external_prefers_the_employer_page_over_an_aggregator() -> None:
    """An aggregator republishes someone else's posting; the employer's page is the source."""
    urls = [
        "https://cl.jobtome.com/empleo/genai-engineer/7ff122c7",
        "https://careers.acme.example/jobs/42",
    ]
    assert first_external_ats_url(urls) == ("https://careers.acme.example/jobs/42", AtsKind.UNKNOWN)
    # With nothing better, the aggregator is still a route to the vacancy.
    assert first_external_ats_url(urls[:1]) == (urls[0], AtsKind.JOBTOME)


def test_aggregators_never_identify_an_employer() -> None:
    assert AtsKind.JOBTOME in JOB_BOARD_KINDS
    assert AtsKind.REMOSHIFT in JOB_BOARD_KINDS
    assert AtsKind.VACANTES in JOB_BOARD_KINDS
    assert AtsKind.BREEZY not in JOB_BOARD_KINDS


def test_vacantes_com_is_a_job_board() -> None:
    """A vacantes.com opening never becomes a company career portal."""
    url = (
        "https://vacantes.com/es/vacantes/"
        "back-end-engineer-senior-llm-y-agentic-ai-witi-5d091ba5"
    )
    assert detect_ats(url) == AtsKind.VACANTES
    assert first_external_ats_url([url]) == (url, AtsKind.VACANTES)


def test_first_external_skips_lnkd_and_keeps_unknown_career() -> None:
    urls = [
        "https://lnkd.in/abc",
        "https://www.linkedin.com/jobs/view/1",
        "https://careers.acme.example/jobs/42",
    ]
    url, kind = first_external_ats_url(urls)
    assert url == "https://careers.acme.example/jobs/42"
    assert kind == AtsKind.UNKNOWN


def test_registry_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "portals.yaml"
    reg = PortalRegistry()
    reg.upsert(
        domain=domain_from_url("https://boards.greenhouse.io/acme/jobs/1"),
        ats_kind=AtsKind.GREENHOUSE,
        registered=True,
        example_url="https://boards.greenhouse.io/acme/jobs/1",
    )
    save_registry(reg, path)
    loaded = load_registry(path)
    assert len(loaded.portals) == 1
    assert loaded.portals[0].domain == "boards.greenhouse.io"
    assert loaded.portals[0].registered is True
