"""External ATS fill over CDP: known fields only, the CV attached, never submitted."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from jobbot.adapters.ats.apply_fill import (
    ApplyFillError,
    fill_open_page,
    looks_like_login,
    open_and_fill_over_cdp,
    plan_apply_fill,
    profile_key_for,
)
from jobbot.models.candidate import Candidate
from jobbot.portals.form_learn import FieldKind, FormField, learn_form_html
from tests.fixtures.profile import sample_profile_dict

GREENHOUSE = Path("tests/fixtures/forms/greenhouse_apply.html")
ATS_URL = "https://boards.greenhouse.io/acme/jobs/4001"

_SPANISH_FORM = """
<html><body><form>
  <label for="n">Nombre</label><input id="n" name="nombre" type="text">
  <label for="a">Apellido</label><input id="a" name="apellido" type="text">
  <label for="c">Correo electrónico</label><input id="c" name="correo" type="email">
  <label for="e">Nombre de la empresa actual</label><input id="e" name="empresa" type="text">
  <label for="p">¿Cuál es tu pretensión de renta?</label><input id="p" name="renta" type="text">
  <label for="cv">Currículum</label><input id="cv" name="cv" type="file" accept=".pdf">
  <button type="submit">Postular</button>
</form></body></html>
"""

_LOGIN_PAGE = """
<html><body><form>
  <label for="u">Email</label><input id="u" name="username" type="email">
  <label for="pw">Password</label><input id="pw" name="password" type="password">
  <label for="cv">Resume</label><input id="cv" name="resume" type="file">
  <button type="submit">Sign in</button>
</form></body></html>
"""

_NO_FORM = "<html><body><h1>Senior Analyst</h1><a href='/apply'>Apply</a></body></html>"


def _candidate() -> Candidate:
    data = sample_profile_dict()
    data["personal"].update(
        {
            "name": "Marta Soto Vera",
            "email": "marta.soto@example.com",
            "phone": "+1 555 0100",
            "linkedin": "https://www.linkedin.com/in/example",
        }
    )
    return Candidate.model_validate(data)


def _pdf(tmp_path: Path) -> Path:
    path = tmp_path / "output" / "jobs" / "J0001" / "application" / "cv.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"%PDF-1.4\n" + b"x" * 40)
    return path


class _Locator:
    def __init__(self, page: FakePage, selector: str) -> None:
        self._page = page
        self._selector = selector

    @property
    def first(self) -> _Locator:
        return self

    def input_value(self) -> str:
        return self._page.values.get(self._selector, "")

    def fill(self, value: str) -> None:
        self._page.filled[self._selector] = value

    def set_input_files(self, files: str) -> None:
        self._page.uploaded.append((self._selector, files))

    def click(self, *_a: Any, **_k: Any) -> None:
        self._page.clicked.append(self._selector)


class FakePage:
    """Records what the driver did. Any click would land in ``clicked``."""

    def __init__(self, html: str, url: str = ATS_URL, frames: list[Any] | None = None) -> None:
        self.html = html
        self.url = url
        self.frames = frames or []
        self.main_frame = None
        self.values: dict[str, str] = {}
        self.filled: dict[str, str] = {}
        self.uploaded: list[tuple[str, str]] = []
        self.clicked: list[str] = []
        self.pressed: list[str] = []
        self.visited: list[str] = []

    def content(self) -> str:
        return self.html

    def locator(self, selector: str) -> _Locator:
        return _Locator(self, selector)

    def click(self, selector: str, **_: Any) -> None:
        self.clicked.append(selector)

    def press(self, selector: str, key: str, **_: Any) -> None:
        self.pressed.append(f"{selector}:{key}")

    def goto(self, url: str, **_: Any) -> None:
        self.visited.append(url)

    def wait_for_load_state(self, *_a: Any, **_k: Any) -> None:
        return None

    def wait_for_selector(self, *_a: Any, **_k: Any) -> None:
        return None


def _greenhouse(project_root: Path) -> str:
    return (project_root / GREENHOUSE).read_text(encoding="utf-8")


def test_plan_fills_identity_and_attaches_resume_not_cover_letter(
    project_root: Path, tmp_path: Path
) -> None:
    form = learn_form_html(_greenhouse(project_root), url=ATS_URL)
    plan = plan_apply_fill(form, _candidate(), cv_path=_pdf(tmp_path))

    by_key = {fill.key: fill.value for fill in plan.fills}
    assert by_key == {
        "first_name": "Marta",
        "last_name": "Soto Vera",
        "email": "marta.soto@example.com",
        "phone": "+1 555 0100",
        "linkedin": "https://www.linkedin.com/in/example",
    }
    assert plan.attach_selector is not None and "resume" in plan.attach_selector
    assert "cover_letter" not in plan.attach_selector
    left = " | ".join(plan.left_for_human).casefold()
    assert "salary" in left
    assert "authorized" in left
    assert "submit" in left
    assert plan.will_submit is False


def test_unknown_questions_are_never_filled(tmp_path: Path) -> None:
    form = learn_form_html(_SPANISH_FORM, url="https://empleos.example.com/postular")
    plan = plan_apply_fill(form, _candidate(), cv_path=_pdf(tmp_path))

    by_label = {fill.label: fill.value for fill in plan.fills}
    assert by_label == {
        "Nombre": "Marta",
        "Apellido": "Soto Vera",
        "Correo electrónico": "marta.soto@example.com",
    }, "Nombre next to Apellido is the given name; company and salary stay empty"
    assert "Nombre de la empresa actual" in plan.left_for_human
    assert "¿Cuál es tu pretensión de renta?" in plan.left_for_human
    assert plan.attach_selector is not None


def test_autocomplete_names_a_field_without_a_useful_label() -> None:
    html = """<form>
      <label for="f1">Field 17</label>
      <input id="f1" name="field_17" type="text" autocomplete="given-name">
      <label for="f2">Field 18</label><input id="f2" name="field_18" type="email">
    </form>"""
    form = learn_form_html(html, url="https://ats.example.com/apply")
    assert form.fields[0].autocomplete == "given-name"
    assert profile_key_for(form.fields[0]) == "first_name"
    assert profile_key_for(form.fields[1]) is None


@pytest.mark.parametrize(
    "label",
    ["Password", "Confirm email", "Referrer email", "I agree to the terms"],
)
def test_secret_or_irreversible_fields_have_no_profile_key(label: str) -> None:
    kind = FieldKind.CHECKBOX if "agree" in label else FieldKind.EMAIL
    assert profile_key_for(FormField(name="x", label=label, kind=kind)) is None


def test_preferred_name_is_not_mistaken_for_a_referral() -> None:
    field = FormField(name="preferred", label="Preferred name", kind=FieldKind.TEXT)
    assert profile_key_for(field) is None


def test_fill_open_page_types_values_attaches_and_never_clicks(
    project_root: Path, tmp_path: Path
) -> None:
    pdf = _pdf(tmp_path)
    page = FakePage(_greenhouse(project_root))
    page.values['[name="job_application[first_name]"], [id="job_application[first_name]"]'] = (
        "Marta"
    )

    result = fill_open_page(page, _candidate(), cv_path=pdf)

    assert result.submitted is False
    assert page.clicked == [] and page.pressed == [], "never click or press submit"
    assert result.attached is True
    assert page.uploaded == [
        ('[name="job_application[resume]"], [id="job_application[resume]"]', str(pdf))
    ]
    assert "First Name" in result.kept, "a value already on the page is left alone"
    assert set(result.filled) == {"Last Name", "Email", "Phone", "LinkedIn Profile"}
    assert "marta.soto@example.com" in page.filled.values()


def test_login_wall_types_nothing(tmp_path: Path) -> None:
    page = FakePage(_LOGIN_PAGE, url="https://ats.example.com/candidate")
    result = fill_open_page(page, _candidate(), cv_path=_pdf(tmp_path))

    assert result.login_required is True
    assert page.filled == {} and page.uploaded == [] and page.clicked == []


def test_login_routes_are_recognised_without_a_password_field() -> None:
    assert looks_like_login("https://accounts.torre.ai/email/?next=/openid", "<html></html>")
    assert looks_like_login("https://ats.example.com/login?next=/apply", "")
    assert not looks_like_login("https://boards.greenhouse.io/acme/jobs/1", "<form></form>")


def test_unreadable_page_reports_instead_of_guessing() -> None:
    page = FakePage(_NO_FORM)
    result = fill_open_page(page, _candidate())

    assert result.readable is False
    assert result.filled == () and page.filled == {}
    assert "could not read" in result.note.casefold()


def test_form_inside_an_iframe_is_filled_in_that_frame(project_root: Path) -> None:
    frame = FakePage(_greenhouse(project_root), url="https://boards.greenhouse.io/embed/job_app")
    page = FakePage(_NO_FORM, url="https://careers.example.com/job/1", frames=[frame])

    result = fill_open_page(page, _candidate())

    assert result.filled
    assert page.filled == {}
    assert frame.filled and frame.clicked == []


class _PageInfo:
    def __init__(self, page: FakePage) -> None:
        self.value = page


class _FakeContext:
    def __init__(self, page: FakePage) -> None:
        self._page = page
        self.pages = [FakePage(_NO_FORM, url="https://mail.example.com/inbox")]
        self.new_pages = 0

    def new_page(self) -> FakePage:
        self.new_pages += 1
        return self._page

    @contextmanager
    def expect_page(self, **_kwargs: object) -> Iterator[_PageInfo]:
        yield _PageInfo(self._page)


class _FakeCdp:
    def __init__(self) -> None:
        self.sent: list[tuple[str, dict[str, object]]] = []

    def send(self, method: str, params: dict[str, object]) -> dict[str, str]:
        self.sent.append((method, params))
        return {"targetId": "T1"}

    def detach(self) -> None:
        return None


class _FakeBrowser:
    def __init__(self, context: _FakeContext) -> None:
        self.contexts = [context]
        self.closed = False
        self.cdp = _FakeCdp()

    def new_browser_cdp_session(self) -> _FakeCdp:
        return self.cdp

    def close(self) -> None:
        self.closed = True


class _FakeChromium:
    def __init__(self, browser: _FakeBrowser | None) -> None:
        self._browser = browser
        self.cdp: list[str] = []

    def connect_over_cdp(self, url: str) -> _FakeBrowser:
        self.cdp.append(url)
        if self._browser is None:
            raise RuntimeError("connect ECONNREFUSED")
        return self._browser


class _FakePlaywright:
    def __init__(self, browser: _FakeBrowser | None) -> None:
        self.chromium = _FakeChromium(browser)
        self.stopped = False

    def start(self) -> _FakePlaywright:
        return self

    def stop(self) -> None:
        self.stopped = True


def test_cdp_opens_a_new_tab_fills_and_only_disconnects(
    project_root: Path, tmp_path: Path
) -> None:
    page = FakePage(_greenhouse(project_root))
    context = _FakeContext(page)
    browser = _FakeBrowser(context)
    playwright = _FakePlaywright(browser)

    result = open_and_fill_over_cdp(
        "http://127.0.0.1:9222",
        ATS_URL,
        _candidate(),
        cv_path=_pdf(tmp_path),
        playwright_factory=lambda: playwright,
    )

    assert playwright.chromium.cdp == ["http://127.0.0.1:9222"]
    assert browser.cdp.sent == [
        ("Target.createTarget", {"url": "about:blank", "background": True})
    ], "a new background tab: your open tabs are never reused nor raised"
    assert context.new_pages == 0
    assert page.visited == [ATS_URL]
    assert result.filled and result.attached
    assert browser.closed is False, "the user's Chrome is never closed"
    assert playwright.stopped is True
    assert page.clicked == []


def test_cdp_connection_failure_raises_for_the_fallback() -> None:
    playwright = _FakePlaywright(None)
    with pytest.raises(ApplyFillError, match="Could not attach"):
        open_and_fill_over_cdp(
            "http://127.0.0.1:9",
            ATS_URL,
            _candidate(),
            playwright_factory=lambda: playwright,
        )
    assert playwright.stopped is True
