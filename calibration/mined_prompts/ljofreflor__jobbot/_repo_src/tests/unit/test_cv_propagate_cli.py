"""`jobbot cv propagate` end to end: dry-run is inert, --apply delegates per destination."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from jobbot.adapters.diff_engine import save_snapshot
from jobbot.config import JobbotConfig, PathsConfig
from jobbot.exit_codes import SUCCESS, VALIDATION_FAILURE
from jobbot.models.external_profile import ExternalProfile
from jobbot.models.sync import SyncOperation, SyncOpType, SyncPlan, SyncResult


def _workspace(tmp_path: Path, project_root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A real CLI workspace: profile.yaml + templates, nothing else."""
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "data" / "profile.yaml").write_text(
        (project_root / "data" / "profile.example.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (tmp_path / ".jobbot.toml").write_text(
        f'[paths]\ntemplates = "{project_root / "templates"}"\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _config(tmp_path: Path, project_root: Path) -> JobbotConfig:
    return JobbotConfig(
        root=tmp_path,
        paths=PathsConfig(templates=project_root / "templates", output=Path("output")),
    )


@dataclass
class FakeIndeed:
    """Stands in for IndeedAdapter: records the plan/apply calls."""

    calls: list[str] = field(default_factory=list)
    cdp_url: str | None = None

    @classmethod
    def from_config(cls, _config: JobbotConfig, *, cdp_url: str | None = None) -> FakeIndeed:
        instance = cls(cdp_url=cdp_url)
        _REGISTRY["indeed"] = instance
        return instance

    def build_sync_plan(self, *, section: str | None = "headline") -> SyncPlan:
        self.calls.append(f"build_sync_plan:{section}")
        return SyncPlan(
            target="indeed",
            operations=[
                SyncOperation(
                    op=SyncOpType.CHANGE,
                    section="headline",
                    field="headline",
                    before="Old",
                    after="New",
                )
            ],
        )

    def apply_sync_plan(self, plan: SyncPlan) -> SyncResult:
        self.calls.append(f"apply_sync_plan:{len(plan.actionable)}")
        return SyncResult(target="indeed", verified=True, message="Indeed synced (fake).")


@dataclass
class FakeLinkedIn:
    calls: list[str] = field(default_factory=list)
    confirm_each: Any = None
    remote_titles: list[str] = field(default_factory=list)

    @classmethod
    def from_config(cls, _config: JobbotConfig, *, cdp_url: str | None = None) -> FakeLinkedIn:
        instance = cls()
        _REGISTRY["linkedin"] = instance
        return instance

    def fetch_remote_publication_titles(self) -> list[str]:
        self.calls.append("fetch_remote_publication_titles")
        return ["Already on LinkedIn"]

    def apply_publications(
        self,
        *,
        confirm_each: Any = None,
        remote_titles: list[str] | None = None,
        enrich_existing: bool = False,
    ) -> SyncResult:
        self.calls.append("apply_publications")
        self.confirm_each = confirm_each
        self.remote_titles = list(remote_titles or [])
        return SyncResult(target="linkedin", verified=True, message="Publications added (fake).")


@dataclass
class FakeGetOnBoard:
    calls: list[str] = field(default_factory=list)

    @classmethod
    def from_config(cls, config: JobbotConfig) -> FakeGetOnBoard:
        instance = cls()
        instance.output_dir = config.output_dir  # type: ignore[attr-defined]
        _REGISTRY["getonboard"] = instance
        return instance

    def prepare_package(self) -> tuple[Path, Any]:
        self.calls.append("prepare_package")
        from jobbot.adapters.getonboard.draft import (
            PermanentProfileFields,
            save_permanent_profile,
        )

        save_permanent_profile(
            PermanentProfileFields(
                experiencia_y_perfil="Perfil nuevo desde profile.yaml.",
                formacion_academica="Formación nueva.",
                headline="Senior Data Scientist",
                skills=["Python"],
            ),
            self.output_dir,  # type: ignore[attr-defined]
        )

        @dataclass
        class _Result:
            mode: str = "cumulative"

        return Path("output/getonboard/profile_permanent.md"), _Result()

    def open_profile_edit(self) -> str:
        self.calls.append("open_profile_edit")
        return "https://www.getonbrd.com/webpros/edit"

    def open_resumes(self) -> str:
        self.calls.append("open_resumes")
        return "https://www.getonbrd.com/webpros/resumes"


_REGISTRY: dict[str, Any] = {}


@pytest.fixture(autouse=True)
def _clear_registry() -> None:
    _REGISTRY.clear()


@pytest.fixture(autouse=True)
def _inert_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    """The preflight probes localhost ports: keep tests off the developer's browser."""
    import jobbot.browser.sessions as sessions

    monkeypatch.setattr(sessions, "inspect_sessions", lambda *_a, **_k: [])


def _install_fakes(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace every portal adapter and the CV builder; return recorded build targets."""
    import jobbot.adapters.getonboard.client as gob_client
    import jobbot.adapters.indeed.client as indeed_client
    import jobbot.adapters.linkedin.client as linkedin_client
    import jobbot.cli as cli

    monkeypatch.setattr(indeed_client, "IndeedAdapter", FakeIndeed)
    monkeypatch.setattr(linkedin_client, "LinkedInAdapter", FakeLinkedIn)
    monkeypatch.setattr(gob_client, "GetOnBoardProfileClient", FakeGetOnBoard)

    built: list[str] = []

    def fake_build_cv(**kwargs: Any) -> list[Path]:
        built.append(str(kwargs["target"].value))
        return [Path("output/base/cv.tex")]

    monkeypatch.setattr(cli, "build_cv", fake_build_cv)
    monkeypatch.setattr(cli, "_getonboard_session", lambda *_a, **_k: FakeGobSession())

    def fake_gob_cv(*_a: Any, **_k: Any) -> None:
        gob = _REGISTRY.get("getonboard")
        if gob is not None:
            gob.calls.append("upload_cv")

    monkeypatch.setattr(cli, "_propagate_getonboard_cv", fake_gob_cv)
    return built


class FakeGobEditPage:
    """The Get on Board edit form, empty: every field is a write."""

    def __init__(self) -> None:
        from jobbot.adapters.getonboard.profile_edit import FIELDS

        self.written: dict[str, str] = {}
        self.selectors = {f.selector: f for f in FIELDS}
        self.clicked: list[str] = []

    def goto(self, url: str, **_: Any) -> None:
        return None

    def query_selector(self, selector: str) -> Any:
        field_spec = self.selectors.get(selector)
        return _FakeGobField(self, field_spec) if field_spec else None

    def click(self, selector: str, **_: Any) -> None:
        self.clicked.append(selector)

    def wait_for_timeout(self, _timeout: float) -> None:
        return None


@dataclass
class _FakeGobField:
    page: FakeGobEditPage
    spec: Any

    def inner_text(self) -> str:
        return self.page.written.get(self.spec.key, "")

    def input_value(self) -> str:
        return self.page.written.get(self.spec.key, "")

    def evaluate(self, _expression: str, arg: Any = None) -> Any:
        self.page.written[self.spec.key] = str(arg)
        return None

    def fill(self, value: str) -> None:
        self.page.written[self.spec.key] = value


class FakeGobSession:
    def __init__(self) -> None:
        self.page = FakeGobEditPage()

    def __enter__(self) -> FakeGobSession:
        return self

    def __exit__(self, *_exc: Any) -> None:
        return None


def _answers(monkeypatch: pytest.MonkeyPatch, replies: dict[str, bool]) -> list[str]:
    """Answer typer.confirm by substring match; record every prompt shown."""
    import jobbot.cli as cli

    asked: list[str] = []

    def fake_confirm(text: str, default: bool = False, **_kwargs: Any) -> bool:
        asked.append(text)
        for needle, reply in replies.items():
            if needle in text:
                return reply
        return default

    monkeypatch.setattr(cli.typer, "confirm", fake_confirm)
    return asked


def _run(argv: list[str]) -> int:
    from jobbot.cli import run_cli

    return run_cli(argv, standalone_mode=False)


def test_dry_run_narrates_the_phase_and_writes_nothing(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _workspace(tmp_path, project_root, monkeypatch)
    built = _install_fakes(monkeypatch)
    asked = _answers(monkeypatch, {})

    assert _run(["cv", "propagate"]) == SUCCESS
    out = capsys.readouterr().out
    assert "propagando cv" in out
    assert "Dry-run" in out
    assert built == []
    assert asked == []
    assert _REGISTRY == {}
    assert not (tmp_path / "output" / "base").exists()


def test_planning_never_reaches_a_browser(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Indeed is reported blocked instead of silently opening a session to pull."""
    _workspace(tmp_path, project_root, monkeypatch)
    _install_fakes(monkeypatch)
    _answers(monkeypatch, {})

    import jobbot.browser.session as session_module

    def explode(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("planning must not start a browser session")

    monkeypatch.setattr(session_module.BrowserSession, "__enter__", explode, raising=False)

    assert _run(["cv", "propagate"]) == SUCCESS
    out = capsys.readouterr().out
    assert "blocked" in out
    assert "jobbot indeed pull" in out
    assert _REGISTRY == {}


def test_apply_delegates_to_every_destination_after_confirming(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _workspace(tmp_path, project_root, monkeypatch)
    save_snapshot(
        ExternalProfile(source="indeed", headline="Old headline"),
        _config(tmp_path, project_root).output_dir,
    )
    built = _install_fakes(monkeypatch)
    asked = _answers(
        monkeypatch,
        {
            "Propagate to": True,
            "Apply these": True,
            "Write these": True,
            "Upload this PDF": True,
        },
    )

    assert _run(["cv", "propagate", "--apply", "--section", "headline"]) == SUCCESS
    out = capsys.readouterr().out

    # base CV rebuilt as PDF + ATS text
    assert built == ["cv", "ats"]
    # one confirmation per destination, plus Indeed's own write confirmation
    destinations = [text for text in asked if "Propagate to" in text]
    assert len(destinations) == 4
    assert any("Apply these" in text for text in asked)

    # GoB writes profile fields, then uploads the validated PDF.
    assert _REGISTRY["getonboard"].calls == ["prepare_package", "upload_cv"]
    assert _REGISTRY["indeed"].calls == ["build_sync_plan:headline", "apply_sync_plan:1"]
    assert _REGISTRY["linkedin"].calls == [
        "fetch_remote_publication_titles",
        "apply_publications",
    ]
    # remote titles are fetched before deciding what to add
    assert _REGISTRY["linkedin"].remote_titles == ["Already on LinkedIn"]
    assert "Indeed synced (fake)." in out
    assert "Publications added (fake)." in out


def test_yes_skips_prompts_but_publications_still_go_through_a_callback(
    tmp_path: Path, project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _workspace(tmp_path, project_root, monkeypatch)
    _install_fakes(monkeypatch)
    asked = _answers(monkeypatch, {})

    assert _run(["cv", "propagate", "--apply", "--yes"]) == SUCCESS
    assert asked == []

    from jobbot.adapters.linkedin.package import LinkedInPublicationItem

    item = LinkedInPublicationItem(
        id="p1",
        title="Some paper",
        publisher=None,
        year=None,
        url=None,
        authors=(),
        coauthors=(),
    )
    confirm_each = _REGISTRY["linkedin"].confirm_each
    assert confirm_each is not None
    assert confirm_each(item) is True


def test_declining_one_destination_leaves_the_others_untouched(
    tmp_path: Path, project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _workspace(tmp_path, project_root, monkeypatch)
    built = _install_fakes(monkeypatch)
    _answers(
        monkeypatch,
        {"Propagate to getonboard": False, "Propagate to": True, "Apply these": True},
    )

    assert _run(["cv", "propagate", "--apply"]) == SUCCESS
    assert "getonboard" not in _REGISTRY
    assert built == ["cv", "ats"]
    assert "linkedin" in _REGISTRY


def test_targets_option_limits_propagation(
    tmp_path: Path, project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _workspace(tmp_path, project_root, monkeypatch)
    built = _install_fakes(monkeypatch)

    assert _run(["cv", "propagate", "--targets", "cv", "--apply", "--yes"]) == SUCCESS
    assert built == ["cv", "ats"]
    assert _REGISTRY == {}


def test_busy_profile_stops_that_destination_before_the_adapter_is_built(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Regression F0002: a leftover Chrome on the profile must be reported, not crashed into."""
    _workspace(tmp_path, project_root, monkeypatch)
    save_snapshot(
        ExternalProfile(source="indeed", headline="Old headline"),
        _config(tmp_path, project_root).output_dir,
    )
    _install_fakes(monkeypatch)
    _answers(monkeypatch, {"Propagate to": True, "Apply these": True})

    import jobbot.browser.sessions as sessions
    from jobbot.browser.sessions import SessionState, SessionStatus

    busy = SessionState(
        site="indeed",
        status=SessionStatus.PROFILE_BUSY,
        evidence="browser-data/indeed held by Chrome pid 30237",
        hint="close that Chrome window, or point --cdp at it",
        holders=(30237,),
    )
    monkeypatch.setattr(sessions, "inspect_sessions", lambda *_a, **_k: [busy])

    assert _run(["cv", "propagate", "--apply", "--section", "headline"]) == SUCCESS
    captured = capsys.readouterr()
    assert "30237" in captured.out + captured.err
    assert "indeed" not in _REGISTRY
    assert "linkedin" in _REGISTRY


def test_unknown_target_fails_validation(
    tmp_path: Path, project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _workspace(tmp_path, project_root, monkeypatch)
    _install_fakes(monkeypatch)

    assert _run(["cv", "propagate", "--targets", "twitter"]) == VALIDATION_FAILURE
