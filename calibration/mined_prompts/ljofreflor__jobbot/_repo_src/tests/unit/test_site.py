"""The advisory page published on GitHub Pages: static, private, honest. Offline."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from bs4 import BeautifulSoup, Tag

from jobbot.ops import pii_guard

LINKEDIN = "https://www.linkedin.com/in/leonardojofre"
REPO = "https://github.com/ljofreflor/jobbot"

_TRACKING_MARKERS = (
    "googletagmanager",
    "gtag(",
    "google-analytics",
    "analytics",
    "connect.facebook.net",
    "fbq(",
    "hotjar",
    "plausible",
    "matomo",
    "clarity.ms",
)
_ANY_EMAIL = re.compile(r"[\w.%+\-]+@[\w\-]+\.[\w.\-]+")
_PHONE_LIKE = re.compile(r"\+?\d[\d\s().\-]{7,}\d")
_EXTERNAL = re.compile(r"^(https?:)?//", re.I)


@pytest.fixture
def site_dir(project_root: Path) -> Path:
    return project_root / "asesoria"


@pytest.fixture
def html(site_dir: Path) -> str:
    return (site_dir / "index.html").read_text(encoding="utf-8")


@pytest.fixture
def soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "html.parser")


@pytest.fixture
def css(site_dir: Path) -> str:
    return (site_dir / "styles.css").read_text(encoding="utf-8")


def test_page_is_spanish(soup: BeautifulSoup) -> None:
    root = soup.find("html")
    assert root is not None
    assert root.get("lang") == "es"


def test_headings_start_at_one_h1(soup: BeautifulSoup) -> None:
    assert len(soup.find_all("h1")) == 1
    assert soup.find_all("h2"), "sections need their own headings"


def test_no_scripts_at_all(soup: BeautifulSoup) -> None:
    """Plain HTML: nothing runs in the visitor's browser, so nothing can track."""
    assert soup.find_all("script") == []


@pytest.mark.parametrize("marker", _TRACKING_MARKERS)
def test_no_tracking_markers(html: str, css: str, marker: str) -> None:
    assert marker not in html.casefold()
    assert marker not in css.casefold()


def test_no_external_assets(soup: BeautifulSoup, css: str) -> None:
    """No fonts, CDNs or pixels: a visit reaches nobody but GitHub Pages."""
    for tag, attr in (("link", "href"), ("img", "src"), ("iframe", "src"), ("source", "src")):
        for node in soup.find_all(tag):
            if tag == "link" and "stylesheet" not in (node.get("rel") or []):
                continue
            assert not _EXTERNAL.match(str(node.get(attr, ""))), f"external <{tag}>"
    assert "@import" not in css
    for target in re.findall(r"url\(\s*['\"]?([^'\")]+)", css):
        assert not _EXTERNAL.match(target), f"external url() in CSS: {target}"


def test_no_contact_data_besides_linkedin(html: str, css: str) -> None:
    for name, text in (("asesoria/index.html", html), ("asesoria/styles.css", css)):
        assert pii_guard.scan_text(text, name) == []
        assert not _ANY_EMAIL.search(text), "contact is LinkedIn only"
        assert not _PHONE_LIKE.search(text), "contact is LinkedIn only"
    assert "mailto:" not in html
    assert "tel:" not in html


def test_links_to_linkedin_and_the_repo(soup: BeautifulSoup) -> None:
    hrefs = {str(a.get("href", "")).rstrip("/") for a in soup.find_all("a")}
    assert LINKEDIN in hrefs
    assert REPO in hrefs


def test_external_links_do_not_leak_the_referrer(soup: BeautifulSoup) -> None:
    for a in soup.find_all("a"):
        if _EXTERNAL.match(str(a.get("href", ""))):
            rel = a.get("rel") or []
            assert "noopener" in rel and "noreferrer" in rel, a.get("href")


def test_local_stylesheet_resolves(soup: BeautifulSoup, site_dir: Path) -> None:
    sheets = [
        str(link.get("href"))
        for link in soup.find_all("link")
        if "stylesheet" in (link.get("rel") or [])
    ]
    assert sheets, "the page must reference its stylesheet"
    for href in sheets:
        assert (site_dir / href).is_file(), href


def test_no_prices_and_no_promised_outcomes(soup: BeautifulSoup) -> None:
    text = soup.get_text(" ").casefold()
    assert not re.search(r"[$€]\s*\d|\d\s*(clp|usd|uf)\b", text), "fee is agreed, never listed"
    for promise in ("garantiza", "aseguramos", "consigue trabajo"):
        assert promise not in text


def test_no_currency_names_at_all(soup: BeautifulSoup) -> None:
    text = soup.get_text(" ").casefold()
    for currency in ("clp", "usd", "pesos", "dólares"):
        assert not re.search(rf"\b{currency}\b", text), currency


def test_hero_announces_the_free_pilot_with_twenty_spots(soup: BeautifulSoup) -> None:
    hero = soup.find("header", class_="hero")
    assert hero is not None
    text = " ".join(hero.get_text(" ").split()).casefold()
    assert "piloto gratuito" in text
    assert "20 cupos" in text
    cta = hero.find("a", class_="cta")
    assert cta is not None and "piloto" in cta.get_text().casefold()


def test_pilot_section_replaces_the_fee(soup: BeautifulSoup) -> None:
    assert soup.find(id="honorario") is None
    heading = soup.find(id="piloto")
    assert heading is not None and heading.name == "h2"
    section = heading.find_parent("section")
    assert section is not None
    text = " ".join(section.get_text(" ").split()).casefold()
    assert "20 personas" in text and "no pagan" in text
    assert "robustecer jobbot" in text
    assert "sin tus datos personales" in text, "public failure reports carry no PII"
    assert "sin cobro automático" in text
    assert "no prometo resultados" in text


def test_pilot_spots_are_stated_not_counted(soup: BeautifulSoup) -> None:
    """Static page: no live counter of remaining spots."""
    text = soup.get_text(" ").casefold()
    for counter in ("quedan", "restantes", "disponibles:"):
        assert counter not in text


def test_written_agreement_step_covers_the_pilot(soup: BeautifulSoup) -> None:
    steps = soup.find("ol", class_="steps")
    assert steps is not None
    second = steps.find_all("li")[1].get_text(" ").casefold()
    assert "piloto" in second
    assert "honorario" not in second
    for term in ("alcance", "datos", "duración"):
        assert term in second


def _map_section(soup: BeautifulSoup) -> Tag:
    heading = soup.find(id="mapa")
    assert heading is not None and heading.name == "h2"
    section = heading.find_parent("section")
    assert section is not None
    return section


def test_platform_map_section_comes_right_after_the_hero(soup: BeautifulSoup) -> None:
    main = soup.find("main")
    assert main is not None
    first = main.find("section")
    assert first is _map_section(soup)
    assert "plataformas" in _map_section(soup).find("h2").get_text().casefold()


def test_platform_map_section_sells_presence_without_promises(soup: BeautifulSoup) -> None:
    text = " ".join(_map_section(soup).get_text(" ").split()).casefold()
    for platform in ("workday", "greenhouse", "lever", "ashby", "smartrecruiters", "manatal"):
        assert platform in text, platform
    assert "mapa compartido" in text
    assert "conocimiento público" in text
    assert "se revisa antes de activarse" in text
    assert "tibia" in text, "presence kept warm, waiting for a match"
    assert "la contraseña la pones tú" in text
    assert "nada se envía sin ti" in text


def test_platform_map_section_holds_the_generated_diagram(html: str, soup: BeautifulSoup) -> None:
    section = _map_section(soup)
    figure = section.find("figure", class_="platform-map")
    assert figure is not None
    svgs = figure.find_all("svg")
    assert svgs and all(svg.get("role") == "img" for svg in svgs)
    assert all(svg.find("title") and svg.find("desc") for svg in svgs)
    assert html.count("platform-map:start") == 1
    assert html.count("platform-map:end") == 1


def test_diagram_layouts_swap_by_viewport(css: str) -> None:
    """Wide layout on desktop, narrow one on phones: text stays readable without JS."""
    assert ".pm-narrow" in css and ".pm-wide" in css
    assert re.search(r"@media\s*\(max-width:[^)]+\)\s*\{[^}]*\.pm-wide", css)


def test_diagram_flows_are_animated_with_css_only(css: str, soup: BeautifulSoup) -> None:
    assert soup.find_all("script") == []
    assert "@keyframes" in css
    assert "stroke-dashoffset" in css
    for selector in (".pm-flow", ".pm-feedback", ".pm-hub"):
        block = re.search(rf"{re.escape(selector)}[^{{]*\{{([^}}]*)\}}", css)
        assert block is not None, selector
    assert re.search(r"\.pm-flow[^{]*\{[^}]*animation:", css)
    assert re.search(r"\.pm-feedback[^{]*\{[^}]*animation:", css)


def test_diagram_motion_stops_for_reduced_motion(css: str) -> None:
    block = re.search(r"@media\s*\(prefers-reduced-motion:\s*reduce\)\s*\{(.*?)\n\}", css, re.S)
    assert block is not None
    body = block.group(1)
    for selector in (".pm-flow", ".pm-feedback", ".pm-hub"):
        assert selector in body, selector
    assert "animation: none" in body


def _stroke_width(css: str, selector: str) -> float:
    for block in re.findall(rf"^{re.escape(selector)}\s*\{{([^}}]*)\}}", css, re.M):
        width = re.search(r"stroke-width:\s*([\d.]+)", block)
        if width:
            return float(width.group(1))
    raise AssertionError(f"no stroke-width for {selector}")


def test_feedback_loop_stands_out_from_outbound_flows(css: str) -> None:
    assert _stroke_width(css, ".pm-feedback") > _stroke_width(css, ".pm-flow")
    feedback = re.findall(r"--feedback:\s*(#[0-9a-f]{6})", css)
    accent = re.findall(r"--accent:\s*(#[0-9a-f]{6})", css)
    assert len(feedback) == 2, "light and dark schemes"
    assert not set(feedback) & set(accent)


def test_reduced_motion_keeps_the_arrows_visible(css: str) -> None:
    block = re.search(r"@media\s*\(prefers-reduced-motion:\s*reduce\)\s*\{(.*?)\n\}", css, re.S)
    assert block is not None
    body = block.group(1)
    for hidden in ("display: none", "visibility: hidden", "opacity: 0", "stroke: none"):
        assert hidden not in body, hidden


def test_platform_map_section_explains_the_feedback_loop(soup: BeautifulSoup) -> None:
    text = " ".join(_map_section(soup).get_text(" ").split()).casefold()
    assert "vuelve" in text and "solo si tú las confirmas" in text


def test_pages_has_one_workflow_that_ships_the_page(project_root: Path) -> None:
    """A repo has one Pages site: two deploying workflows overwrite each other."""
    workflows = project_root / ".github" / "workflows"
    deployers = [p.name for p in workflows.glob("*.yml") if "deploy-pages" in p.read_text()]
    assert deployers == ["pages.yml"]

    pages = (workflows / "pages.yml").read_text(encoding="utf-8")
    assert "asesoria/**" in pages, "editing the page must redeploy it"
    assert "site/asesoria" in pages, "the page must land inside the published artefact"


def test_pages_pins_setup_uv_to_a_published_tag(project_root: Path) -> None:
    """setup-uv publishes no floating major tag (``@v10`` does not resolve)."""
    pages = (project_root / ".github" / "workflows" / "pages.yml").read_text(encoding="utf-8")
    pins = re.findall(r"astral-sh/setup-uv@(\S+)", pages)
    assert pins
    for pin in pins:
        assert re.fullmatch(r"v\d+\.\d+\.\d+", pin), pin
