"""Parse a saved career-site job page (Phenom-style or generic) into a JobPosting.

Fixture-first: live download of unknown hosts stays refused. Structure and
markers only — no employer names in this module.
"""

from __future__ import annotations

from bs4 import BeautifulSoup, Tag

from jobbot.jobs.closure import closure_evidence, visible_soup
from jobbot.jobs.parsing import extract_skills_from_text
from jobbot.models.job import JobPosting
from jobbot.portals.detect import AtsKind


class CareerPageParseError(ValueError):
    """HTML is not a recognizable career job posting."""


class ClosedPostingError(ValueError):
    """Visible text says this vacancy is already filled."""

    def __init__(self, evidence: str) -> None:
        self.evidence = evidence
        super().__init__(evidence)


def job_from_career_html(html: str, *, url: str) -> JobPosting:
    """Extract title, company and description from saved career HTML.

    Hidden nodes (Phenom ``class="hide"`` expire templates) are stripped first.
    A *visible* filled banner raises ``ClosedPostingError`` so callers refuse
    to store the job.
    """
    evidence = closure_evidence(html)
    if evidence:
        raise ClosedPostingError(evidence)

    soup = visible_soup(html)
    title = _title(soup)
    company = _company(soup)
    if not title or not company:
        raise CareerPageParseError(
            "Career page HTML needs a visible job title and company name."
        )
    description = _description(soup)
    public = url.strip()
    if public and not public.lower().startswith(("http://", "https://")):
        public = "https://" + public
    return JobPosting(
        id="PENDING",
        source="career_page",
        url=public or None,
        title=title,
        company=company,
        description=description,
        raw_description=description,
        skills=extract_skills_from_text(description),
        ats_url=public or None,
        ats_kind=AtsKind.UNKNOWN.value,
        note="career page fixture",
    )


def looks_like_career_job_html(html: str) -> bool:
    """True when the HTML has enough structure to try a career parse."""
    if not html or not html.strip():
        return False
    try:
        job_from_career_html(html, url="https://example.invalid/job/1")
    except ClosedPostingError:
        return True
    except CareerPageParseError:
        return False
    return True


def _title(soup: BeautifulSoup) -> str:
    for selector in (
        '[data-ph-at-id="job-title"]',
        '[itemprop="title"]',
        "h1",
    ):
        node = soup.select_one(selector)
        text = _text(node)
        if text:
            return text
    title_tag = soup.find("title")
    if isinstance(title_tag, Tag):
        raw = title_tag.get_text(" ", strip=True)
        if raw:
            # "Role - Company | Careers" → Role
            head = raw.split("|", 1)[0].split(" - ", 1)[0].strip()
            if head:
                return head
    return ""


def _company(soup: BeautifulSoup) -> str:
    for selector in (
        '[data-ph-at-id="job-company"]',
        '[itemprop="hiringOrganization"] [itemprop="name"]',
        '[itemprop="hiringOrganization"]',
        ".job-company",
        '[data-ph-at-id="company-name"]',
    ):
        node = soup.select_one(selector)
        text = _text(node)
        if text:
            return text
    return ""


def _description(soup: BeautifulSoup) -> str:
    for selector in (
        '[data-ph-at-id="job-description"]',
        '[itemprop="description"]',
        ".job-description",
        "main",
        "article",
    ):
        node = soup.select_one(selector)
        text = _text(node)
        if text and len(text) > 40:
            return text
    body = soup.body
    return _text(body) if body is not None else ""


def _text(node: Tag | None) -> str:
    if node is None:
        return ""
    return " ".join(node.get_text(" ", strip=True).split())
