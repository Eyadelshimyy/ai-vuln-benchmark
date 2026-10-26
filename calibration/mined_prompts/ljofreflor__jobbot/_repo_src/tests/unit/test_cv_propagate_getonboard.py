"""`cv propagate` must write the Get on Board profile, not only open the editor."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any

import pytest

from jobbot.config import JobbotConfig
from jobbot.exit_codes import SUCCESS

OLD_TEXT = "Me he dedicado a trabajos numéricos en matlab y C++."


@dataclass
class RecordingTrix:
    text: str = ""
    loaded_html: str | None = None

    def inner_text(self) -> str:
        return self.text

    def input_value(self) -> str:  # pragma: no cover
        raise AssertionError("trix is read with inner_text()")

    def evaluate(self, expression: str, arg: Any = None) -> Any:
        assert "loadHTML" in expression
        self.loaded_html = str(arg)
        return None

    def fill(self, value: str) -> None:  # pragma: no cover
        raise AssertionError("fill() never reaches trix")


@dataclass
class RecordingInput:
    value: str = ""

    def inner_text(self) -> str:  # pragma: no cover
        raise AssertionError("inputs are read with input_value()")

    def input_value(self) -> str:
        return self.value

    def evaluate(self, expression: str, arg: Any = None) -> Any:  # pragma: no cover
        raise AssertionError("inputs need no JS")

    def fill(self, value: str) -> None:
        self.value = value


class RecordingPage:
    def __init__(self) -> None:
        from jobbot.adapters.getonboard.profile_edit import FIELDS

        by_key: dict[str, Any] = {
            "description_es": RecordingInput(),
            "professional_es": RecordingTrix(OLD_TEXT),
            "academic_background_es": RecordingTrix(),
        }
        self.elements = {f.selector: by_key[f.key] for f in FIELDS}
        self.clicked: list[str] = []
        self.visited: list[str] = []

    def goto(self, url: str, **_: Any) -> None:
        self.visited.append(url)

    def query_selector(self, selector: str) -> Any:
        return self.elements.get(selector)

    def click(self, selector: str, **_: Any) -> None:
        self.clicked.append(selector)

    def wait_for_timeout(self, _timeout: float) -> None:
        return None


@dataclass
class FakeSession:
    page: RecordingPage = field(default_factory=RecordingPage)
    cdp_url: str | None = None

    def __enter__(self) -> FakeSession:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None


@pytest.fixture
def workspace(tmp_path: Path, project_root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "data" / "profile.yaml").write_text(
        (project_root / "data" / "profile.example.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    import jobbot.browser.sessions as sessions

    monkeypatch.setattr(sessions, "inspect_sessions", lambda *_a, **_k: [])
    return tmp_path


def _run(monkeypatch: pytest.MonkeyPatch, *args: str) -> FakeSession:
    """Run propagate against a fake edit page; return the session that was used."""
    import jobbot.cli as cli
    from jobbot.cli import run_cli

    session = FakeSession()
    monkeypatch.setattr(cli, "build_cv", lambda **_k: [Path("output/base/cv.tex")])
    monkeypatch.setattr(cli, "_getonboard_session", lambda *_a, **_k: session)
    # Profile-write tests stub the CV upload; that path has its own unit suite.
    monkeypatch.setattr(cli, "_propagate_getonboard_cv", lambda *_a, **_k: None)

    assert run_cli(list(args), standalone_mode=False) == SUCCESS
    return session


def test_dry_run_plans_the_portal_write_without_opening_a_browser(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    session = _run(monkeypatch, "cv", "propagate", "--targets", "getonboard")

    out = capsys.readouterr().out
    assert "getonbrd.com" in out or "portal" in out
    assert session.page.visited == [], "the plan must not touch the portal"
    assert session.page.clicked == []


def test_apply_writes_the_profile_fields_and_saves(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from jobbot.adapters.getonboard.profile_edit import FIELDS, GOB_EDIT_URL, SAVE_BUTTON

    session = _run(
        monkeypatch,
        "cv",
        "propagate",
        "--targets",
        "getonboard",
        "--apply",
        "--yes",
        "--cdp",
        "http://127.0.0.1:9224",
    )

    assert session.page.visited == [GOB_EDIT_URL]
    professional = session.page.elements[
        next(f.selector for f in FIELDS if f.key == "professional_es")
    ]
    assert professional.loaded_html, "the old portal text was left in place"
    assert "matlab" not in (professional.loaded_html or "").casefold()
    academic = session.page.elements[
        next(f.selector for f in FIELDS if f.key == "academic_background_es")
    ]
    assert academic.loaded_html, "the empty education block was not filled"
    assert session.page.clicked == [SAVE_BUTTON]


def test_config_is_the_one_the_command_loaded(workspace: Path) -> None:
    """Guard the fixture itself: the workspace profile is what propagate reads."""
    config = JobbotConfig(root=workspace)
    assert config.profile_path.is_file()
