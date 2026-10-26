"""Identity-provider buttons on a login page are knowledge, never a click."""

from __future__ import annotations

from pathlib import Path

from jobbot.companies.signup import signup_sheet
from jobbot.models.candidate import Candidate
from jobbot.portals.form_learn import learn_form_html
from jobbot.portals.sso import SsoProvider, detect_sso_providers
from tests.fixtures.profile import sample_profile_dict

FIXTURE = Path("tests/fixtures/forms/workday_login_sso.html")
LOGIN_URL = "https://example.wd5.myworkdayjobs.com/en-US/login"


def test_it_names_google_and_linkedin_sign_in_buttons(project_root: Path) -> None:
    html = (project_root / FIXTURE).read_text(encoding="utf-8")

    found = detect_sso_providers(html)

    assert SsoProvider.GOOGLE in found
    assert SsoProvider.LINKEDIN in found


def test_a_linkedin_profile_field_is_not_sso() -> None:
    html = """
    <form>
      <label for="li">LinkedIn Profile</label>
      <input id="li" name="linkedin" type="url">
      <label for="email">Email</label>
      <input id="email" name="email" type="email">
      <label for="cv">Resume</label>
      <input id="cv" name="resume" type="file">
    </form>
    """

    assert detect_sso_providers(html) == ()


def test_a_footer_follow_us_link_is_not_sso() -> None:
    html = """
    <footer>
      <a href="https://www.linkedin.com/company/acme">Follow us on LinkedIn</a>
      <a href="https://www.facebook.com/acme">Facebook</a>
    </footer>
    """

    assert detect_sso_providers(html) == ()


def test_learn_form_records_sso_on_a_login_wall(project_root: Path) -> None:
    html = (project_root / FIXTURE).read_text(encoding="utf-8")

    form = learn_form_html(html, url=LOGIN_URL, company="Acme")

    assert SsoProvider.GOOGLE.value in form.sso_providers
    assert SsoProvider.LINKEDIN.value in form.sso_providers


def test_the_signup_sheet_lists_sso_as_yours_to_click(project_root: Path) -> None:
    html = (project_root / FIXTURE).read_text(encoding="utf-8")
    form = learn_form_html(html, url=LOGIN_URL)
    candidate = Candidate.model_validate(sample_profile_dict())

    sheet = signup_sheet(candidate, form=form)
    labels = " ".join(item.label for item in sheet).casefold()

    assert "google" in labels
    assert "linkedin" in labels
    sso_rows = [item for item in sheet if "google" in item.label.casefold()]
    assert sso_rows
    assert sso_rows[0].yours_to_decide
    assert "oauth" in sso_rows[0].source.casefold() or "sso" in sso_rows[0].source.casefold()
