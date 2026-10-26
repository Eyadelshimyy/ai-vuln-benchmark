"""Regression: the Indeed resume is client-rendered, so pull must wait for the sections.

Reading page.content() right after domcontentloaded returned an empty shell, and the
snapshot stored experience=[] while the resume actually had entries — every later
`cv propagate` then re-planned the same experiences as missing.
"""

from __future__ import annotations

from pathlib import Path
from types import TracebackType

import pytest

from jobbot.adapters.indeed.client import IndeedAdapter
from jobbot.config import JobbotConfig, PathsConfig

SHELL_HTML = '<html><body><div id="app"></div></body></html>'


class FakeResumePage:
    """Serves the shell until the resume sections are awaited."""

    def __init__(self, rendered_html: str) -> None:
        self._rendered_html = rendered_html
        self.url = "https://profile.indeed.com/resume"
        self.rendered = False
        self.awaited_selectors: list[str] = []

    def goto(self, url: str, **_: object) -> None:
        self.url = url

    def wait_for_selector(self, selector: str, timeout: float | None = None) -> None:
        self.awaited_selectors.append(selector)
        self.rendered = True

    def content(self) -> str:
        return self._rendered_html if self.rendered else SHELL_HTML

    def inner_text(self, _selector: str) -> str:
        return "Experiencia laboral" if self.rendered else ""

    def wait_for_timeout(self, _ms: float) -> None:
        return None


class FakeSession:
    def __init__(self, page: FakeResumePage) -> None:
        self.page = page
        self.debug_dumps: list[str] = []

    def __enter__(self) -> FakeSession:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None

    def dump_debug(self, _component: str, reason: str) -> None:
        self.debug_dumps.append(reason)


@pytest.fixture
def adapter(
    tmp_path: Path, project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[IndeedAdapter, FakeResumePage]:
    profile_path = tmp_path / "data" / "profile.yaml"
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(
        (project_root / "data/profile.example.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    config = JobbotConfig(root=tmp_path, paths=PathsConfig(output=Path("output")))
    rendered = (project_root / "tests/fixtures/indeed_resume_sections.html").read_text(
        encoding="utf-8"
    )
    page = FakeResumePage(rendered)
    monkeypatch.setattr(IndeedAdapter, "_session", lambda _self: FakeSession(page))
    return IndeedAdapter(config), page


def test_pull_waits_for_the_resume_sections_before_reading_html(
    adapter: tuple[IndeedAdapter, FakeResumePage],
) -> None:
    client, page = adapter
    profile = client.pull_profile()
    assert page.awaited_selectors, "pull captured HTML without waiting for the resume to render"
    assert "work-experience-section" in page.awaited_selectors[0]
    assert [e.company for e in profile.experience] == [
        "Acme Analytics",
        "Nimbus Labs",
        "Instituto Ejemplo",
    ]
    assert [e.institution for e in profile.education] == ["Universidad Ejemplo de Chile"]
