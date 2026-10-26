"""Registering on a company portal: prepared for you, never done for you."""

from __future__ import annotations

from pathlib import Path

from jobbot.companies.models import CareerSite, CareerSiteType, KnowledgeStatus
from jobbot.companies.signup import (
    AccountNeed,
    signup_sheet,
    signup_target,
)
from jobbot.models.candidate import Candidate
from jobbot.portals.detect import AtsKind
from jobbot.portals.form_learn import learn_form_html
from tests.fixtures.profile import sample_profile_dict

FORM_FIXTURE = Path("tests/fixtures/forms/greenhouse_apply.html")


def _site(url: str, ats: AtsKind) -> CareerSite:
    return CareerSite(
        url=url,
        domain=url.split("//", 1)[-1].split("/", 1)[0],
        site_type=CareerSiteType.COMPANY_CAREER_PORTAL,
        ats=ats,
        status=KnowledgeStatus.ACTIVE,
    )


def test_the_sheet_lists_what_the_portal_will_ask_and_what_you_already_have() -> None:
    candidate = Candidate.model_validate(sample_profile_dict())

    sheet = signup_sheet(candidate)
    by_label = {item.label: item for item in sheet}

    assert by_label["Email"].value == str(candidate.personal.email)
    assert by_label["Full name"].value == candidate.personal.name
    assert all(item.source for item in sheet), "every value says where it came from"


def test_a_password_is_never_prepared() -> None:
    """JobBot has no business holding the credential to your own account."""
    sheet = signup_sheet(Candidate.model_validate(sample_profile_dict()))
    text = " ".join(f"{item.label} {item.value}" for item in sheet).casefold()

    assert "password" not in text
    assert "contraseña" not in text


def test_what_the_profile_does_not_answer_is_left_for_you() -> None:
    candidate = Candidate.model_validate(sample_profile_dict())
    candidate.personal.phone = None

    phone = next(item for item in signup_sheet(candidate) if item.label == "Phone")

    assert phone.value == ""
    assert phone.yours_to_decide
    assert "invent" not in phone.source.casefold()


def test_a_learned_form_makes_the_sheet_concrete(project_root: Path) -> None:
    """Once we have seen the form, the sheet is its questions, not our guesses."""
    html = (project_root / FORM_FIXTURE).read_text(encoding="utf-8")
    form = learn_form_html(html, url="https://boards.greenhouse.io/acme/jobs/4001")

    sheet = signup_sheet(Candidate.model_validate(sample_profile_dict()), form=form)
    labels = [item.label for item in sheet]

    assert "Resume/CV" in labels
    assert "When could you start?" in labels
    start = next(item for item in sheet if item.label == "When could you start?")
    assert start.yours_to_decide, "a screening question is not in your profile"


def test_an_ats_that_needs_no_account_says_so() -> None:
    target = signup_target(_site("https://boards.greenhouse.io/acme", AtsKind.GREENHOUSE))

    assert target.need is AccountNeed.NOT_NEEDED
    assert target.url == "https://boards.greenhouse.io/acme"
    assert "account" in target.note.casefold()


def test_an_ats_that_does_need_one_points_at_the_portal() -> None:
    target = signup_target(_site("https://acme.wd3.myworkdayjobs.com/careers", AtsKind.WORKDAY))

    assert target.need is AccountNeed.NEEDED
    assert target.url.startswith("https://acme.wd3.myworkdayjobs.com/")


def test_an_unknown_portal_is_unknown_not_assumed() -> None:
    target = signup_target(_site("https://empresa.cl/trabaja-con-nosotros", AtsKind.UNKNOWN))

    assert target.need is AccountNeed.UNKNOWN
    assert target.url == "https://empresa.cl/trabaja-con-nosotros"
    assert "crear cuenta" in target.note.casefold() or "sign up" in target.note.casefold()


def test_nothing_in_this_module_can_submit_anything() -> None:
    """The guarantee is structural: the module has no writer and no browser driver."""
    import ast
    import inspect

    from jobbot.companies import signup

    source = inspect.getsource(signup)
    forbidden_tokens = (
        "click(",
        "fill(",
        "submit(",
        "requests.",
        "urlopen",
        "httpx",
        "urllib.request",
        "http.client",
        "aiohttp",
        "playwright",
        "sync_playwright",
        "BrowserSession",
        "webbrowser",
    )
    for forbidden in forbidden_tokens:
        assert forbidden not in source, f"signup must not {forbidden}"

    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".", 1)[0])
    banned_imports = {
        "httpx",
        "requests",
        "urllib",
        "aiohttp",
        "playwright",
        "webbrowser",
        "http",
    }
    overlap = imported & banned_imports
    assert imported.isdisjoint(banned_imports), f"signup imports networking: {overlap}"
