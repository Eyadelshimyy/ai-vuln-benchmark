"""LinkedIn post sweep parsing tests."""

from pathlib import Path

import pytest

from jobbot.adapters.linkedin.posts_source import LinkedInPostJobSource
from jobbot.adapters.linkedin.sweep import (
    dedupe_jobs_by_apply_target,
    looks_like_job_post,
    parse_posts_fixture,
    post_offers_wanted_country,
    post_to_job,
    post_to_jobs,
    split_vacancy_links,
)
from jobbot.portals.detect import AtsKind


def test_parse_posts_fixture_filters_and_detects_ats(project_root: Path) -> None:
    text = (project_root / "tests/fixtures/linkedin_posts.txt").read_text(encoding="utf-8")
    posts = parse_posts_fixture(text)
    assert len(posts) == 7
    relevant = [p for p in posts if looks_like_job_post(p.text)]
    assert len(relevant) == 6  # hiking post excluded
    greenhouse = next(p for p in relevant if p.ats_kind == AtsKind.GREENHOUSE)
    assert greenhouse.ats_url and "greenhouse" in greenhouse.ats_url
    gob = next(p for p in relevant if p.ats_kind == AtsKind.GETONBOARD)
    assert gob.ats_url and "getonbrd.com" in gob.ats_url
    email = next(p for p in relevant if p.ats_kind == AtsKind.EMAIL)
    assert email.ats_url == "mailto:seleccion@empresa.cl"
    job = post_to_job(greenhouse, job_id="J0001")
    assert job.source == "linkedin_post"
    assert job.ats_kind == "greenhouse"
    assert "Data Scientist" in job.title or "data scientist" in job.title.casefold()


def test_parse_post_expands_short_links(monkeypatch: object) -> None:
    from jobbot.adapters.linkedin.sweep import parse_post_blob

    def fake_expand(urls: list[str]) -> dict[str, str]:
        return {url: "https://boards.greenhouse.io/acme/jobs/99" for url in urls}

    import jobbot.adapters.linkedin.sweep as sweep_mod

    monkeypatch.setattr(sweep_mod, "expand_url_map", fake_expand)
    post = parse_post_blob(
        "Hiring DS! Apply: https://lnkd.in/short\n",
    )
    assert post.ats_kind.value == "greenhouse"
    assert post.ats_url and "greenhouse" in post.ats_url


def _multi_vacancy_post(project_root: Path) -> object:
    text = (project_root / "tests/fixtures/linkedin_multi_vacancy_post.txt").read_text(
        encoding="utf-8"
    )
    return parse_posts_fixture(text)[0]


def test_split_vacancy_links_reads_every_labelled_role(project_root: Path) -> None:
    post = _multi_vacancy_post(project_root)
    titles = [v.title for v in post.vacancies]  # type: ignore[attr-defined]
    assert titles == [
        "Lead Agentic AI Consultant",
        "Lead Data Scientist",
        "Senior Data Scientist – Generative AI",
        "Lead ML Engineer",
        "Senior Computer Vision Engineer",
    ]


def test_split_vacancy_links_ignores_links_that_name_no_role() -> None:
    vacancies = split_vacancy_links(
        "🔹 Lead Data Scientist: https://career.example.com/a-1\n"
        "🔹 Senior ML Engineer: https://career.example.com/b-2\n"
        "Learn more: https://www.example.com/about\n"
        "Our careers page: https://career.example.com/\n"
    )
    assert [v.title for v in vacancies] == ["Lead Data Scientist", "Senior ML Engineer"]


def test_split_vacancy_links_handles_a_collapsed_paragraph() -> None:
    """Some cards render the whole post as one line, bullets included."""
    vacancies = split_vacancy_links(
        "We're hiring! 🔹 Lead Data Scientist: https://career.example.com/a-1 "
        "🔹 Senior Computer Vision Engineer: https://career.example.com/b-2"
    )
    assert [v.title for v in vacancies] == [
        "Lead Data Scientist",
        "Senior Computer Vision Engineer",
    ]


def test_split_vacancy_links_drops_unresolved_short_links() -> None:
    """A lnkd.in URL is not appliable and must never reach the portal registry."""
    text = (
        "🔹 Lead Data Scientist: https://lnkd.in/e3KmPHyp\n🔹 Lead ML Engineer: https://lnkd.in/x2"
    )
    assert split_vacancy_links(text) == []
    resolved = split_vacancy_links(
        text, {"https://lnkd.in/e3KmPHyp": "https://career.example.com/lead-data-scientist-89203"}
    )
    assert [v.url for v in resolved] == ["https://career.example.com/lead-data-scientist-89203"]


def test_post_with_several_vacancies_becomes_one_job_each(project_root: Path) -> None:
    """Regression: five roles collapsed into one job left four of them unreachable."""
    jobs = post_to_jobs(_multi_vacancy_post(project_root))  # type: ignore[arg-type]

    assert len(jobs) == 5
    assert [j.title for j in jobs][:2] == ["Lead Agentic AI Consultant", "Lead Data Scientist"]
    # Each job is independently appliable and independently addressable in the database.
    assert len({j.ats_url for j in jobs}) == 5
    assert len({j.url for j in jobs}) == 5
    assert len({j.source_job_id for j in jobs}) == 5
    assert all(j.ats_url and "career.northwindlabs.example.com" in j.ats_url for j in jobs)
    # The advert stays the post the recruiter wrote, and the permalink is not lost.
    assert all("Northwind Labs" in j.description for j in jobs)
    assert all(
        j.note and "post=https://www.linkedin.com/posts/laura-recruiter" in j.note for j in jobs
    )
    assert all(j.posted_at is not None for j in jobs)


def test_zero_labelled_vacancies_keeps_whole_post_mapping(project_root: Path) -> None:
    """No "Role: url" lines → permalink is the job URL (guessed title, post-level ATS)."""
    text = (project_root / "tests/fixtures/linkedin_posts.txt").read_text(encoding="utf-8")
    greenhouse = next(p for p in parse_posts_fixture(text) if p.ats_kind == AtsKind.GREENHOUSE)
    assert greenhouse.vacancies == ()
    jobs = post_to_jobs(greenhouse)
    assert len(jobs) == 1
    assert jobs[0].ats_kind == "greenhouse"
    assert jobs[0].url == greenhouse.post_url
    assert jobs[0].source_job_id == greenhouse.post_id


def test_one_labelled_vacancy_uses_vacancy_title_and_url() -> None:
    """One labelled role follows the same mapper as N — not the whole-post permalink path."""
    from jobbot.adapters.linkedin.sweep import parse_post_blob

    post = parse_post_blob(
        "We're hiring!\n🔹 Lead Data Scientist – Chile: https://career.example.com/lead-ds-42\n",
        author="Ana Recruiter",
        post_url="https://www.linkedin.com/posts/ana-activity-1234567890123456789-abcd",
    )
    assert len(post.vacancies) == 1
    jobs = post_to_jobs(post, countries=["CL"])
    assert len(jobs) == 1
    job = jobs[0]
    assert job.title == "Lead Data Scientist – Chile"
    assert job.url == "https://career.example.com/lead-ds-42"
    assert job.ats_url == job.url
    assert job.source_job_id != post.post_id
    assert job.note and "post=https://www.linkedin.com/posts/ana-activity" in job.note
    # post_to_job is the same mapper's first (only) result.
    assert post_to_job(post).url == job.url


def test_one_labelled_vacancy_in_unwanted_country_is_filtered_out() -> None:
    """Country filter applies at len==1 the same way it does for roundups."""
    from jobbot.adapters.linkedin.sweep import parse_post_blob

    post = parse_post_blob(
        "🔹 Lead Data Scientist – Colombia: https://career.example.com/lead-ds-co\n",
        post_url="https://www.linkedin.com/posts/ana-activity-1234567890123456789-abcd",
    )
    assert post.vacancies[0].country == "CO"
    assert post_to_jobs(post, countries=["CL"]) == []
    assert len(post_to_jobs(post)) == 1


def _latam_roundup_post(project_root: Path) -> object:
    text = (project_root / "tests/fixtures/linkedin_latam_roundup_post.txt").read_text(
        encoding="utf-8"
    )
    return parse_posts_fixture(text)[0]


def test_roundup_post_survives_when_one_role_is_in_the_wanted_country(
    project_root: Path,
) -> None:
    """Regression: the post read as Colombian (first marker wins) and was dropped whole."""
    post = _latam_roundup_post(project_root)
    assert post_offers_wanted_country(post.text, wanted=["CL"]) is True  # type: ignore[attr-defined]


def test_roundup_post_is_dropped_when_no_role_is_in_the_wanted_country(
    project_root: Path,
) -> None:
    post = _latam_roundup_post(project_root)
    assert post_offers_wanted_country(post.text, wanted=["MX"]) is False  # type: ignore[attr-defined]


def test_roundup_keeps_only_the_roles_in_the_wanted_country(project_root: Path) -> None:
    """Each role carries its own country, so the choice is per role and not per post."""
    post = _latam_roundup_post(project_root)
    assert [v.country for v in post.vacancies] == ["CO", "PE", "AR", "CL"]  # type: ignore[attr-defined]

    jobs = post_to_jobs(post, countries=["CL"])  # type: ignore[arg-type]
    assert [j.title for j in jobs] == ["GenAI & Agentic AI Full-Stack Engineer – Chile"]

    every = post_to_jobs(post)  # type: ignore[arg-type]
    assert len(every) == 4


def _single_employer_post(project_root: Path) -> object:
    text = (project_root / "tests/fixtures/linkedin_single_employer_post.txt").read_text(
        encoding="utf-8"
    )
    return parse_posts_fixture(text)[0]


def test_employer_name_stops_where_the_recruiter_prose_starts() -> None:
    """Regression: the name ran on into the sentence ('NTT DATA buscamos 4 profesionales')."""
    from jobbot.adapters.linkedin.sweep import _guess_company

    assert _guess_company("En NORTHWIND DATA buscamos 4 profesionales para un proyecto") == (
        "NORTHWIND DATA"
    )


def test_a_preposition_inside_a_word_does_not_name_an_employer() -> None:
    """'when Chile' used to read as 'en Chile' and hand back a country as the employer."""
    from jobbot.adapters.linkedin.sweep import _guess_company

    assert _guess_company("Hiring remotely when Chile reopens") is None


def test_role_field_after_en_is_not_the_employer() -> None:
    """'BUSCAMOS COINVESTIGADOR EN INTELIGENCIA ARTIFICIAL' must not name the field."""
    from jobbot.adapters.linkedin.sweep import _guess_company, employer_from_post

    text = (
        "🔎 BUSCAMOS COINVESTIGADOR(A) EN INTELIGENCIA ARTIFICIAL\n\n"
        "En FST NEGOCIOS  – Centro de I+D+i estamos conformando un equipo en Prociencia Perú\n"
    )
    assert _guess_company(text) == "FST NEGOCIOS"
    assert employer_from_post(text, None, author="Freddy Silva Tuesta") == "FST NEGOCIOS"


def test_hashtag_confirms_the_employer_when_the_apply_link_is_in_the_comments(
    project_root: Path,
) -> None:
    """A post with no link in its body still names its employer: #NorthwindData."""
    post = _single_employer_post(project_root)
    job = post_to_job(post)  # type: ignore[arg-type]

    assert job.company == "NORTHWIND DATA"
    assert job.title == "Machine Learning & Customer Analytics"


def test_title_drops_the_word_naming_whom_the_recruiter_wants() -> None:
    """Regression: the role came out as 'talento | Machine Learning & Customer Analytics'."""
    from jobbot.adapters.linkedin.sweep import _guess_title

    assert _guess_title("🚀 Buscamos talento | Machine Learning & Customer Analytics") == (
        "Machine Learning & Customer Analytics"
    )
    assert _guess_title("Buscamos Data Scientist Senior para Airport Operations.") == (
        "Data Scientist Senior para Airport Operations"
    )
    assert (
        _guess_title("Seguimos buscando un(a) Editor de Contenidos | Chile | Híbrido")
        == "Editor de Contenidos"
    )


def test_title_reads_estoy_buscando_and_stops_before_para_empresa() -> None:
    """First-person hiring posts must still name the role, not the employer clause."""
    from jobbot.adapters.linkedin.sweep import _guess_title

    title = _guess_title(
        "Estoy buscando un Gerente de Tecnología , Datos e Inteligencia Artificial "
        "(CDDO Chief Digital & Data Officer) para empresa FIntech; tendrá a su cargo…"
    )
    assert title is not None
    assert "gerente de tecnología" in title.casefold()
    assert "cddo" in title.casefold()
    assert "fintech" not in title.casefold()


def test_mailto_company_domain_beats_recruiter_author() -> None:
    """postulaciones@peopletrust.cl names Peopletrust, not the LinkedIn author."""
    from jobbot.adapters.linkedin.sweep import employer_from_post

    assert (
        employer_from_post(
            "Estoy buscando un Gerente de Tecnología para empresa Fintech.",
            "mailto:postulaciones@peopletrust.cl",
            author="Rodolfo Amenabar",
        )
        == "Peopletrust"
    )
    assert (
        employer_from_post(
            "Send CV",
            "mailto:ana@gmail.com",
            author="Ana Recruiter",
        )
        == "Ana Recruiter"
    )


def test_unnamed_client_is_not_the_agency_hashtag_or_the_author() -> None:
    from jobbot.adapters.linkedin.sweep import employer_from_post

    text = (
        "Seguimos buscando un(a) Editor de Contenidos | Chile | Híbrido\n"
        "Es un rol en el equipo de una compañía líder de la región.\n"
        "Si te interesa, mandame un mensaje.\n"
        "#NewsroomAgency #Hiring"
    )
    permalink = "https://www.linkedin.com/posts/recruiter_activity-7511877806738411520-MJ5G"
    assert employer_from_post(text, permalink, author="Ana Recruiter") == "Unknown company"


def test_title_keeps_the_first_letter_of_applied_scientist() -> None:
    """Regression: '(?i)a' matched the A of Applied and the title became 'pplied Scientist'."""
    from jobbot.adapters.linkedin.sweep import _guess_title

    title = _guess_title("Buscamos Applied Scientist para sumarse a nuestro equipo.")
    assert title is not None
    assert title.startswith("Applied Scientist")
    assert not title.startswith("pplied")


def test_a_topic_hashtag_never_promotes_a_place_to_an_employer() -> None:
    """Only a hashtag spelling the name itself corroborates; #Empleo says nothing."""
    from jobbot.adapters.linkedin.sweep import parse_post_blob

    blob = (
        "Publicación en el feed\nAna Recruiter\n"
        "Buscamos Data Scientist en Santiago para un cliente del sector retail.\n"
        "#Empleo #DataScience #Hiring"
    )
    assert post_to_job(parse_post_blob(blob)).company == "Ana Recruiter"


def test_a_post_for_madrid_is_not_offered_for_chile(project_root: Path) -> None:
    post = _single_employer_post(project_root)
    assert post_offers_wanted_country(post.text, wanted=["CL"]) is False  # type: ignore[attr-defined]
    assert post_offers_wanted_country(post.text, wanted=["ES"]) is True  # type: ignore[attr-defined]


def test_roundup_middle_dot_bullets_survive_a_collapsed_paragraph() -> None:
    """Roundups bullet with '·', and some cards render the whole post on one line."""
    vacancies = split_vacancy_links(
        "Ofertas de la semana: · Lead Data Scientist – Chile: https://career.example.com/a-1 "
        "· ML Engineer – Perú: https://career.example.com/b-2"
    )
    assert [(v.title, v.country) for v in vacancies] == [
        ("Lead Data Scientist – Chile", "CL"),
        ("ML Engineer – Perú", "PE"),
    ]


def test_fixture_sweep_keeps_the_chile_role_out_of_a_latam_roundup(
    project_root: Path, tmp_path: Path
) -> None:
    from jobbot.config import JobbotConfig, PathsConfig

    config = JobbotConfig(paths=PathsConfig(output=tmp_path / "out"), root=project_root)
    jobs = LinkedInPostJobSource(config).search_from_fixture(
        project_root / "tests/fixtures/linkedin_latam_roundup_post.txt",
        query="inteligencia artificial",
        countries=["CL"],
    )
    assert [j.title for j in jobs] == ["GenAI & Agentic AI Full-Stack Engineer – Chile"]


def test_fixture_sweep_keeps_a_post_whose_roles_are_listed_one_by_one(
    project_root: Path, tmp_path: Path
) -> None:
    """The query need not appear verbatim: the listed vacancies are the apply route."""
    from jobbot.config import JobbotConfig, PathsConfig

    config = JobbotConfig(paths=PathsConfig(output=tmp_path / "out"), root=project_root)
    jobs = LinkedInPostJobSource(config).search_from_fixture(
        project_root / "tests/fixtures/linkedin_multi_vacancy_post.txt",
        query="hiring data scientist",
    )
    assert len(jobs) == 5


def test_source_from_fixture(project_root: Path, tmp_path: Path) -> None:
    from jobbot.config import JobbotConfig, PathsConfig

    config = JobbotConfig(paths=PathsConfig(output=tmp_path / "out"), root=project_root)
    source = LinkedInPostJobSource(config)
    jobs = source.search_from_fixture(
        project_root / "tests/fixtures/linkedin_posts.txt",
        query="data scientist",
    )
    assert len(jobs) >= 2
    kinds = {j.ats_kind for j in jobs}
    assert "greenhouse" in kinds
    assert "lever" in kinds


class _FakeButtons:
    def __init__(self, count: int, fail_on: set[int] | None = None) -> None:
        self._count = count
        self._fail_on = fail_on or set()
        self.clicked: list[int] = []

    def count(self) -> int:
        return self._count

    def nth(self, index: int) -> "_FakeButtons":
        self._index = index
        return self

    def click(self, **_: object) -> None:
        if self._index in self._fail_on:
            msg = "not clickable"
            raise TimeoutError(msg)
        self.clicked.append(self._index)


class _FakeFeedPage:
    def __init__(self, see_more: int = 0, fail_on: set[int] | None = None) -> None:
        self.scripts: list[str] = []
        self.waits: list[int] = []
        self.buttons = _FakeButtons(see_more, fail_on)

    def evaluate(self, script: str) -> None:
        self.scripts.append(script)

    def wait_for_timeout(self, ms: int) -> None:
        self.waits.append(ms)

    def locator(self, _selector: str) -> _FakeButtons:
        return self.buttons


def test_autoscroll_feed_scrolls_and_waits() -> None:
    """Agent runs have no human to scroll; lazy posts must load anyway."""
    from jobbot.adapters.linkedin.posts_source import autoscroll_feed

    page = _FakeFeedPage()
    autoscroll_feed(page, rounds=4, pause_ms=500)

    assert len(page.scripts) == 4
    assert all("scrollBy" in script for script in page.scripts)
    assert page.waits == [500, 500, 500, 500]


def test_expand_truncated_posts_clicks_see_more() -> None:
    """The apply email often lives behind LinkedIn's '…see more' toggle."""
    from jobbot.adapters.linkedin.posts_source import expand_truncated_posts

    page = _FakeFeedPage(see_more=3)
    assert expand_truncated_posts(page) == 3
    assert page.buttons.clicked == [0, 1, 2]


def test_expand_truncated_posts_survives_unclickable_toggle() -> None:
    from jobbot.adapters.linkedin.posts_source import expand_truncated_posts

    page = _FakeFeedPage(see_more=3, fail_on={1})
    assert expand_truncated_posts(page) == 2
    assert page.buttons.clicked == [0, 2]


class _FakeCards:
    """Cards for one selector: text plus anchor hrefs."""

    def __init__(self, cards: list[tuple[str, list[str]]]) -> None:
        self._cards = cards

    def count(self) -> int:
        return len(self._cards)

    def nth(self, index: int) -> "_FakeCard":
        text, hrefs = self._cards[index]
        return _FakeCard(text, hrefs)


class _FakeCard:
    def __init__(self, text: str, hrefs: list[str]) -> None:
        self._text = text
        self._hrefs = hrefs

    def inner_text(self, **_: object) -> str:
        return self._text

    def locator(self, _selector: str) -> "_FakeAnchors":
        return _FakeAnchors(self._hrefs)


class _FakeAnchors:
    def __init__(self, hrefs: list[str]) -> None:
        self._hrefs = hrefs

    def count(self) -> int:
        return len(self._hrefs)

    def nth(self, index: int) -> "_FakeAnchor":
        return _FakeAnchor(self._hrefs[index])


class _FakeAnchor:
    def __init__(self, href: str) -> None:
        self._href = href

    def get_attribute(self, _name: str) -> str:
        return self._href


class _FakeSearchPage:
    """Only the modern componentkey cards exist, like LinkedIn today."""

    def __init__(self, cards: list[tuple[str, list[str]]], *, selector: str) -> None:
        self._cards = cards
        self._selector = selector
        self.url = "https://www.linkedin.com/search/results/content/?keywords=enviar+CV"
        self.queried: list[str] = []

    def locator(self, selector: str) -> _FakeCards:
        self.queried.append(selector)
        return _FakeCards(self._cards if selector == self._selector else [])


_MODERN_POST = (
    "Publicación en el feed\nDiego Recruiter\nExpert Analyst @ NTT Data\n"
    "Buscamos Data Scientist con Python y SQL para proyecto de analytics.\n"
    "Interesados enviar CV a seleccion@nttdata.com"
)


def test_collect_jobs_reads_modern_componentkey_cards() -> None:
    """Regression: LinkedIn dropped div.feed-shared-update-v2 for componentkey cards."""
    from jobbot.adapters.linkedin.posts_source import (
        MODERN_POST_CARD,
        collect_jobs_from_feed_page,
    )
    from jobbot.jobs.sources import JobSearchQuery

    permalink = "https://www.linkedin.com/feed/update/urn:li:share:7438557958579417088/"
    page = _FakeSearchPage(
        [(_MODERN_POST, ["https://www.linkedin.com/in/diego/", permalink])],
        selector=MODERN_POST_CARD,
    )
    jobs = collect_jobs_from_feed_page(
        page,
        JobSearchQuery(query="enviar CV", limit=10),
        resolve_short_links=False,
    )

    assert len(jobs) == 1
    assert jobs[0].ats_kind == "email"
    assert jobs[0].ats_url == "mailto:seleccion@nttdata.com"
    assert jobs[0].url == permalink


def test_parse_post_blob_uses_mailto_anchor_when_text_has_no_email() -> None:
    """Recruiters often hide the address behind a 'escríbeme' mailto link."""
    from jobbot.adapters.linkedin.sweep import parse_post_blob

    post = parse_post_blob(
        "Buscamos Data Scientist para analytics. Postulaciones por correo.",
        mailto_urls=["mailto:irina@bcp.com.pe"],
    )
    assert post.ats_kind == AtsKind.EMAIL
    assert post.ats_url == "mailto:irina@bcp.com.pe"


def test_first_post_permalink_ignores_profiles_and_hashtags() -> None:
    from jobbot.adapters.linkedin.posts_source import first_post_permalink

    hrefs = [
        "https://www.linkedin.com/in/someone/",
        "https://www.linkedin.com/search/results/all/?keywords=%23hiring",
        "https://www.linkedin.com/feed/update/urn:li:activity:7438557958579417088/",
    ]
    assert first_post_permalink(hrefs) == hrefs[-1]
    assert first_post_permalink(hrefs[:2]) is None


_FEED_CARD_TEXT = """Publicación en el feed
Diego Carreño M.
Expert Analyst @ NTT Data
• 1er
10 meses • Editado •
Buscamos Data Scientist Senior para Airport Operations.
• Python y SQL
• Inferencia causal
Interesados enviar CV a seleccion@nttdata.com"""


def test_strip_feed_chrome_takes_author_and_drops_ui_lines() -> None:
    """Regression: company came out as 'el feed' from LinkedIn's own header."""
    from jobbot.adapters.linkedin.sweep import strip_feed_chrome

    author, body = strip_feed_chrome(_FEED_CARD_TEXT)

    assert author == "Diego Carreño M."
    assert "Publicación en el feed" not in body
    assert "• 1er" not in body
    assert "10 meses" not in body
    # Real bullet content from the post body must survive.
    assert "• Python y SQL" in body
    assert "• Inferencia causal" in body


def test_post_to_job_company_is_the_employer_not_the_feed() -> None:
    """Regression: company came out as 'el feed' from LinkedIn's own header.

    The post names NTT Data and the apply mailbox is @nttdata.com, so the employer is
    corroborated and wins over the recruiter who wrote the post.
    """
    from jobbot.adapters.linkedin.sweep import parse_post_blob

    job = post_to_job(parse_post_blob(_FEED_CARD_TEXT), job_id="J0042")

    assert job.company == "NTT Data"
    assert "el feed" not in job.company
    assert job.ats_url == "mailto:seleccion@nttdata.com"


def test_company_falls_back_to_the_author_when_nothing_corroborates() -> None:
    """Never promote a name the apply route does not confirm."""
    from jobbot.adapters.linkedin.sweep import parse_post_blob

    blob = (
        "Publicación en el feed\nAna Recruiter\n"
        "Buscamos Data Scientist en Santiago para un cliente del sector retail.\n"
        "Enviar CV a seleccion@empresa.cl"
    )
    assert post_to_job(parse_post_blob(blob)).company == "Ana Recruiter"


def test_multi_vacancy_company_is_the_employer_the_apply_host_confirms(
    project_root: Path,
) -> None:
    jobs = post_to_jobs(_multi_vacancy_post(project_root))  # type: ignore[arg-type]
    assert {j.company for j in jobs} == {"Northwind Labs"}


def test_strip_feed_chrome_leaves_fixture_posts_untouched() -> None:
    from jobbot.adapters.linkedin.sweep import strip_feed_chrome

    text = "We're hiring a Data Scientist.\nApply: https://boards.greenhouse.io/x/1"
    author, body = strip_feed_chrome(text)
    assert author is None
    assert body == text


def test_permalink_falls_back_to_urn_in_card_html() -> None:
    """Most search cards hide the permalink; the urn is still in their HTML."""
    from jobbot.adapters.linkedin.posts_source import permalink_from_html

    html = '<div data-x="urn:li:activity:7438557958579417088"><p>hiring</p></div>'
    assert (
        permalink_from_html(html)
        == "https://www.linkedin.com/feed/update/urn:li:activity:7438557958579417088/"
    )
    assert permalink_from_html("<div>no urn here</div>") is None


class _FakeWaitPage:
    def __init__(self, *, raises: bool = False) -> None:
        self._raises = raises
        self.waited: list[str] = []

    def wait_for_selector(self, selector: str, **_: object) -> None:
        if self._raises:
            msg = "timeout"
            raise TimeoutError(msg)
        self.waited.append(selector)


def test_wait_for_post_cards_waits_before_scrolling() -> None:
    """Regression: sweep scrolled an empty page and reported 'no relevant posts'."""
    from jobbot.adapters.linkedin.posts_source import (
        MODERN_POST_CARD,
        wait_for_post_cards,
    )

    page = _FakeWaitPage()
    assert wait_for_post_cards(page, timeout_ms=1000) is True
    assert MODERN_POST_CARD in page.waited[0]


def test_wait_for_post_cards_returns_false_on_timeout() -> None:
    from jobbot.adapters.linkedin.posts_source import wait_for_post_cards

    assert wait_for_post_cards(_FakeWaitPage(raises=True), timeout_ms=10) is False


def test_collect_jobs_drops_posts_outside_the_wanted_country() -> None:
    """Setting the country keeps CDMX adverts out of a Chile-only search."""
    from jobbot.adapters.linkedin.posts_source import (
        MODERN_POST_CARD,
        collect_jobs_from_feed_page,
    )
    from jobbot.jobs.sources import JobSearchQuery

    mexico = (
        "Publicación en el feed\nZayd Recruiter\n"
        "Vacantes exclusivas en TI para CDMX. Data Scientist con Python y SQL.\n"
        "Enviar CV a seleccion@empresa.mx"
    )
    chile = (
        "Publicación en el feed\nAna Recruiter\n"
        "Buscamos Data Scientist en Santiago, Chile. Python y SQL.\n"
        "Enviar CV a seleccion@empresa.cl"
    )
    page = _FakeSearchPage([(mexico, []), (chile, [])], selector=MODERN_POST_CARD)
    jobs = collect_jobs_from_feed_page(
        page,
        JobSearchQuery(query="enviar CV", limit=10, countries=("CL",)),
        resolve_short_links=False,
    )

    assert [j.ats_url for j in jobs] == ["mailto:seleccion@empresa.cl"]


def test_collect_jobs_without_country_preference_keeps_both() -> None:
    from jobbot.adapters.linkedin.posts_source import (
        MODERN_POST_CARD,
        collect_jobs_from_feed_page,
    )
    from jobbot.jobs.sources import JobSearchQuery

    mexico = (
        "Publicación en el feed\nZayd Recruiter\n"
        "Data Scientist para CDMX con Python.\nEnviar CV a seleccion@empresa.mx"
    )
    page = _FakeSearchPage([(mexico, [])], selector=MODERN_POST_CARD)
    jobs = collect_jobs_from_feed_page(
        page, JobSearchQuery(query="enviar CV", limit=10), resolve_short_links=False
    )
    assert len(jobs) == 1


def test_author_profile_url_is_the_fallback_when_no_permalink() -> None:
    """LinkedIn stopped exposing urn:li:activity; keep at least the author's profile."""
    from jobbot.adapters.linkedin.posts_source import author_profile_url

    # Real href order seen in a modern search card (author repeated, then company).
    hrefs = [
        "https://www.linkedin.com/in/gabriela-forton-714a86260/",
        "https://www.linkedin.com/in/gabriela-forton-714a86260/",
        "https://www.linkedin.com/company/eratalent/",
        "mailto:gforton@eratalent.one",
        "https://www.linkedin.com/",
    ]
    assert author_profile_url(hrefs) == "https://www.linkedin.com/in/gabriela-forton-714a86260/"
    no_profile = ["https://www.linkedin.com/", "https://boards.greenhouse.io/x"]
    assert author_profile_url(no_profile) is None


def test_collect_jobs_stores_author_profile_when_permalink_is_absent() -> None:
    from jobbot.adapters.linkedin.posts_source import (
        MODERN_POST_CARD,
        collect_jobs_from_feed_page,
    )
    from jobbot.jobs.sources import JobSearchQuery

    blob = (
        "Publicación en el feed\nGabriela Forton\n"
        "En ERA TALENT buscamos Data Scientist con Python y SQL.\n"
        "Enviar CV a gforton@eratalent.one"
    )
    hrefs = ["https://www.linkedin.com/in/gabriela-forton-714a86260/"]
    page = _FakeSearchPage([(blob, hrefs)], selector=MODERN_POST_CARD)
    jobs = collect_jobs_from_feed_page(
        page, JobSearchQuery(query="data scientist", limit=5), resolve_short_links=False
    )

    assert len(jobs) == 1
    assert str(jobs[0].url) == "https://www.linkedin.com/in/gabriela-forton-714a86260/"
    assert jobs[0].ats_url == "mailto:gforton@eratalent.one"


def test_permalink_wins_over_author_profile() -> None:
    from jobbot.adapters.linkedin.posts_source import (
        MODERN_POST_CARD,
        collect_jobs_from_feed_page,
    )
    from jobbot.jobs.sources import JobSearchQuery

    blob = (
        "Publicación en el feed\nAna Recruiter\n"
        "Buscamos Data Scientist con Python.\nEnviar CV a ana@empresa.cl"
    )
    hrefs = [
        "https://www.linkedin.com/in/ana-recruiter/",
        "https://www.linkedin.com/feed/update/urn:li:activity:7371583217478574080/",
    ]
    page = _FakeSearchPage([(blob, hrefs)], selector=MODERN_POST_CARD)
    jobs = collect_jobs_from_feed_page(
        page, JobSearchQuery(query="data scientist", limit=5), resolve_short_links=False
    )
    assert "urn:li:activity:7371583217478574080" in str(jobs[0].url)


class _FakeMenuItem:
    def __init__(self, label: str, *, on_click: object = None) -> None:
        self._label = label
        self._on_click = on_click
        self.clicks = 0

    def inner_text(self, **_kw: object) -> str:
        return self._label

    def click(self, **_kw: object) -> None:
        self.clicks += 1
        if callable(self._on_click):
            self._on_click()


class _FakeMenuLocator:
    """Playwright-ish locator over a fixed list of menu items."""

    def __init__(self, items: list[_FakeMenuItem]) -> None:
        self._items = items

    def count(self) -> int:
        return len(self._items)

    def nth(self, index: int) -> _FakeMenuItem:
        return self._items[index]

    @property
    def first(self) -> _FakeMenuItem:
        if not self._items:
            raise AssertionError("no menu item")
        return self._items[0]


class _MenuCard:
    """A card whose control menu copies the permalink to the clipboard."""

    def __init__(self, page: "_ClipboardPage", *, menu: bool = True) -> None:
        self.page = page
        self._menu = (
            _FakeMenuLocator([_FakeMenuItem("Abrir el menú de controles", on_click=page.open_menu)])
            if menu
            else _FakeMenuLocator([])
        )

    def locator(self, selector: str) -> _FakeMenuLocator:
        from jobbot.adapters.linkedin import selectors as sel

        assert selector == sel.POST_CONTROL_MENU
        return self._menu


class _ClipboardPage:
    def __init__(self, link: str = "https://lnkd.in/p/dRmZbTZG") -> None:
        self.link = link
        self.clipboard = ""
        self.menu_open = False
        self.keys: list[str] = []
        self.keyboard = self

    def open_menu(self) -> None:
        self.menu_open = True

    def _copy(self) -> None:
        self.clipboard = self.link

    def locator(self, selector: str) -> _FakeMenuLocator:
        if not self.menu_open:
            return _FakeMenuLocator([])
        return _FakeMenuLocator(
            [
                _FakeMenuItem("Guardar"),
                _FakeMenuItem("Copiar enlace a la publicación", on_click=self._copy),
                _FakeMenuItem("Denunciar publicación"),
            ]
        )

    def wait_for_timeout(self, _ms: int) -> None:
        return None

    def press(self, key: str) -> None:
        self.keys.append(key)
        self.menu_open = False

    def evaluate(self, _script: str, *_args: object) -> str:
        return self.clipboard


def test_copy_post_permalink_uses_the_control_menu() -> None:
    """LinkedIn hides the urn; the '…' menu still hands over the real post link."""
    from jobbot.adapters.linkedin.posts_source import copy_post_permalink

    page = _ClipboardPage()
    card = _MenuCard(page)
    assert copy_post_permalink(page, card) == "https://lnkd.in/p/dRmZbTZG"
    assert page.keys == ["Escape"]  # menu closed, feed left as we found it


def test_copy_post_permalink_returns_none_without_menu() -> None:
    from jobbot.adapters.linkedin.posts_source import copy_post_permalink

    page = _ClipboardPage()
    assert copy_post_permalink(page, _MenuCard(page, menu=False)) is None


def test_copy_post_permalink_ignores_non_link_clipboard() -> None:
    from jobbot.adapters.linkedin.posts_source import copy_post_permalink

    page = _ClipboardPage(link="algo que no es un link")
    assert copy_post_permalink(page, _MenuCard(page)) is None


def test_canonical_post_url_drops_tracking_and_member_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """The copied link carries rcm=<your member id>: never store that."""
    from jobbot.adapters.linkedin import posts_source

    resolved = (
        "https://www.linkedin.com/posts/gabriela-forton-714a86260_en-era-talent-"
        "share-7371583217478574080-YDc9/?utm_source=share&utm_medium=member_desktop"
        "&rcm=ACoAAAXJInQBLp3OTwoNIHue"
    )
    monkeypatch.setattr(posts_source, "follow_redirect_url", lambda _url: resolved)

    assert posts_source.canonical_post_url("https://lnkd.in/p/dRmZbTZG") == (
        "https://www.linkedin.com/posts/gabriela-forton-714a86260_en-era-talent-"
        "share-7371583217478574080-YDc9/"
    )


def test_canonical_post_url_keeps_unresolvable_link() -> None:
    from jobbot.adapters.linkedin.posts_source import canonical_post_url

    plain = "https://www.linkedin.com/feed/update/urn:li:activity:7371583217478574080/"
    assert canonical_post_url(plain) == plain


def test_collect_jobs_copies_permalink_from_menu(monkeypatch: pytest.MonkeyPatch) -> None:
    from jobbot.adapters.linkedin import posts_source
    from jobbot.jobs.sources import JobSearchQuery

    resolved = "https://www.linkedin.com/posts/ana-recruiter_share-7371583217478574080-YDc9/?utm_source=share"
    monkeypatch.setattr(posts_source, "follow_redirect_url", lambda _url: resolved)

    blob = (
        "Publicación en el feed\nAna Recruiter\n"
        "Buscamos Data Scientist con Python.\nEnviar CV a ana@empresa.cl"
    )
    hrefs = ["https://www.linkedin.com/in/ana-recruiter/"]
    page = _MenuSearchPage([(blob, hrefs)])
    jobs = posts_source.collect_jobs_from_feed_page(
        page,
        JobSearchQuery(query="data scientist", limit=5),
        resolve_short_links=False,
        copy_permalinks=True,
    )

    assert str(jobs[0].url) == (
        "https://www.linkedin.com/posts/ana-recruiter_share-7371583217478574080-YDc9/"
    )


class _MenuSearchPage(_ClipboardPage):
    """Modern cards whose only route to the post URL is the '…' menu."""

    def __init__(
        self,
        cards: list[tuple[str, list[str]]],
        link: str = "https://lnkd.in/p/dRmZbTZG",
    ) -> None:
        super().__init__(link)
        self._cards = cards
        self.url = "https://www.linkedin.com/search/results/content/?keywords=enviar+CV"

    def locator(self, selector: str) -> object:
        from jobbot.adapters.linkedin import posts_source as ps
        from jobbot.adapters.linkedin import selectors as sel

        if selector == ps.MODERN_POST_CARD:
            return _MenuCards(self, self._cards)
        if selector == sel.POST_MENU_ITEM:
            return super().locator(selector)
        return _FakeMenuLocator([])


class _MenuCards:
    def __init__(self, page: _MenuSearchPage, cards: list[tuple[str, list[str]]]) -> None:
        self._page = page
        self._cards = cards

    def count(self) -> int:
        return len(self._cards)

    def nth(self, index: int) -> "_MenuFakeCard":
        text, hrefs = self._cards[index]
        return _MenuFakeCard(self._page, text, hrefs)


class _MenuFakeCard(_FakeCard):
    def __init__(self, page: _MenuSearchPage, text: str, hrefs: list[str]) -> None:
        super().__init__(text, hrefs)
        self._page = page

    def locator(self, selector: str) -> object:
        from jobbot.adapters.linkedin import selectors as sel

        if selector == sel.POST_CONTROL_MENU:
            return _FakeMenuLocator(
                [_FakeMenuItem("Abrir el menú de controles", on_click=self._page.open_menu)]
            )
        return super().locator(selector)


# Real card text of the post stored as J0049: engagement chrome and the search URL leaked in.
J0049_CARD = """Publicación en el feed
Rodrigo Fernández
VP of Product Management, Mercado Envios
Buscamos Software Engineers y Data Scientists en el equipo de Mercado Envíos de Mercado Libre
Rodrigo Fernández en linkedin.com
68 reacciones
68
1 comentario
1 comentario
Recomendar
Comentar"""


def test_description_drops_engagement_chrome() -> None:
    """Regression: J0049 stored '68 reacciones' and 'Recomendar Comentar' as the JD."""
    from jobbot.adapters.linkedin.sweep import parse_post_blob

    post = parse_post_blob(J0049_CARD)
    assert post.author == "Rodrigo Fernández"
    assert "Mercado Envíos" in post.text
    for junk in ("68 reacciones", "Recomendar", "Comentar", "en linkedin.com", "1 comentario"):
        assert junk not in post.text


def test_extra_urls_feed_ats_detection_without_polluting_the_description() -> None:
    from jobbot.adapters.linkedin.sweep import parse_post_blob

    body = "Buscamos Data Scientist para el equipo de analytics. Postula en el link."
    post = parse_post_blob(
        body,
        extra_urls=[
            "https://boards.greenhouse.io/acme/jobs/123",
            "https://www.linkedin.com/search/results/content/?keywords=enviar+CV",
        ],
    )
    assert post.ats_url == "https://boards.greenhouse.io/acme/jobs/123"
    assert post.text == body
    assert "greenhouse" not in post.text
    assert "search/results" not in post.text


def test_collect_jobs_keeps_the_search_url_out_of_the_description() -> None:
    from jobbot.adapters.linkedin.posts_source import (
        MODERN_POST_CARD,
        collect_jobs_from_feed_page,
    )
    from jobbot.jobs.sources import JobSearchQuery

    blob = (
        "Publicación en el feed\nAna Recruiter\n"
        "Buscamos Data Scientist con Python.\nEnviar CV a ana@empresa.cl\n68 reacciones\nComentar"
    )
    hrefs = [
        "https://www.linkedin.com/in/ana-recruiter/",
        "https://www.linkedin.com/search/results/content/?keywords=enviar+CV",
    ]
    page = _FakeSearchPage([(blob, hrefs)], selector=MODERN_POST_CARD)
    jobs = collect_jobs_from_feed_page(
        page, JobSearchQuery(query="enviar CV", limit=5), resolve_short_links=False
    )

    assert jobs[0].ats_url == "mailto:ana@empresa.cl"
    assert "search/results" not in jobs[0].description
    assert "linkedin.com/in/" not in jobs[0].description
    assert "68 reacciones" not in jobs[0].description
    assert "Comentar" not in jobs[0].description


def test_reposts_of_the_same_vacancy_collapse_to_one_job() -> None:
    """Same mailto + title from different authors must not become several jobs."""
    from jobbot.adapters.linkedin.sweep import parse_post_blob

    first = parse_post_blob(
        "Buscamos Analista de Datos en Santiago. Enviar CV a seleccion@empresa.cl",
        author="Recruiter A",
        post_url="https://www.linkedin.com/feed/update/urn:li:activity:100/",
    )
    second = parse_post_blob(
        "Buscamos Analista de Datos en Santiago. Enviar CV a seleccion@empresa.cl",
        author="Recruiter B",
        post_url="https://www.linkedin.com/feed/update/urn:li:activity:200/",
    )
    other = parse_post_blob(
        "Hiring Enfermera Clínica. Enviar CV a rrhh@clinica.cl",
        author="Clínica",
        post_url="https://www.linkedin.com/feed/update/urn:li:activity:300/",
    )
    jobs = dedupe_jobs_by_apply_target(
        [post_to_job(first), post_to_job(second), post_to_job(other)]
    )
    assert len(jobs) == 2
    mailtos = sorted(j.ats_url or "" for j in jobs)
    assert mailtos == [
        "mailto:rrhh@clinica.cl",
        "mailto:seleccion@empresa.cl",
    ]
