"""Regression: the post-sync snapshot must not erase what the resume already has.

`_snapshot_after` stored `experience=[]` / `education=[]` and overwrote the canonical
snapshot, so the next `indeed diff` (and `cv propagate`) re-planned all 11 experiences
as missing right after a successful sync.
"""

from __future__ import annotations

from pathlib import Path
from types import TracebackType

from playwright.sync_api import TimeoutError as PlaywrightTimeout

from jobbot.adapters.diff_engine import load_snapshot, save_snapshot
from jobbot.adapters.indeed.client import IndeedAdapter
from jobbot.config import JobbotConfig, PathsConfig
from jobbot.models.external_profile import ExternalExperience, ExternalProfile
from jobbot.models.sync import SyncPlan
from jobbot.profile.loader import load_profile

SHELL_HTML = '<html><body><div id="app"></div></body></html>'


class FakeResumePage:
    """The saved resume, client-rendered like the real one."""

    def __init__(self, rendered_html: str, *, renders: bool = True) -> None:
        self._rendered_html = rendered_html
        self._renders = renders
        self.url = "https://profile.indeed.com/resume"
        self.rendered = False

    def goto(self, url: str, **_: object) -> None:
        self.url = url

    def wait_for_selector(self, _selector: str, timeout: float | None = None) -> None:
        if not self._renders:
            raise PlaywrightTimeout(_selector)
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

    def __enter__(self) -> FakeSession:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None

    def dump_debug(self, _component: str, _reason: str) -> None:
        return None


def _adapter(tmp_path: Path, project_root: Path) -> IndeedAdapter:
    profile_path = tmp_path / "data" / "profile.yaml"
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(
        (project_root / "data/profile.example.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return IndeedAdapter(JobbotConfig(root=tmp_path, paths=PathsConfig(output=Path("output"))))


def _snapshot(
    adapter: IndeedAdapter, page: FakeResumePage, tmp_path: Path, project_root: Path
) -> ExternalProfile | None:
    candidate = load_profile(adapter.config.profile_path)
    adapter._snapshot_after(  # noqa: SLF001 - the unit under test
        FakeSession(page),
        candidate,
        SyncPlan(target="indeed", actions=[]),
        "20260917T000000Z",
        edit_verified=True,
    )
    return load_snapshot(adapter.config.output_dir, "indeed")


def test_snapshot_after_sync_keeps_the_experience_the_page_shows(
    tmp_path: Path, project_root: Path
) -> None:
    adapter = _adapter(tmp_path, project_root)
    rendered = (project_root / "tests/fixtures/indeed_resume_sections.html").read_text(
        encoding="utf-8"
    )
    stored = _snapshot(adapter, FakeResumePage(rendered), tmp_path, project_root)

    assert stored is not None
    assert [e.company for e in stored.experience] == [
        "Acme Analytics",
        "Nimbus Labs",
        "Instituto Ejemplo",
    ]
    assert [e.institution for e in stored.education] == ["Universidad Ejemplo de Chile"]


def test_a_sparse_dom_never_deletes_the_previous_snapshot(
    tmp_path: Path, project_root: Path
) -> None:
    """If the page will not render, keep what we already knew instead of writing nothing."""
    adapter = _adapter(tmp_path, project_root)
    save_snapshot(
        ExternalProfile(
            source="indeed",
            experience=[ExternalExperience(company="Acme Analytics", title="Data Scientist")],
            skills=["python"],
        ),
        adapter.config.output_dir,
    )

    stored = _snapshot(adapter, FakeResumePage("", renders=False), tmp_path, project_root)

    assert stored is not None
    assert [e.company for e in stored.experience] == ["Acme Analytics"]
