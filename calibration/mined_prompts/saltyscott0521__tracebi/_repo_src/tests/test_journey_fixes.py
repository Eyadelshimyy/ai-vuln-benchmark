"""
Analyst-journey fixes — ROADMAP "Now" items 4 (minimal slice) and 5.

Covers:
- `tracebi init`'s README uses the git-URL install form (TraceBi is not on
  PyPI; a bare `pip install "tracebi[...]"` is a dependency-confusion trap).
- `tracebi serve` fails loudly and actionably when the `web` package is not
  importable (the wheel does not ship the web app).
- `tracebi init` tells the user to `git init` (without running git for them),
  and renderers warn exactly once per render when a manifest is built with
  git_sha == "unknown" — code provenance missing from the audit trail.
- The repo .gitignore no longer ignores *.manifest.json — manifests are
  receipts and projects should retain them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_GIT_URL_FORM = "@ git+https://github.com/saltyscott0521/tracebi"


# ── tracebi init: README install lines ────────────────────────────────────

class TestInitReadmeInstallLines:
    def _init_project(self, tmp_path):
        from tracebi.cli import main
        project = tmp_path / "proj"
        assert main(["init", str(project)]) == 0
        return project

    def test_readme_uses_git_url_install_form(self, tmp_path):
        readme = (self._init_project(tmp_path) / "README.md").read_text()
        assert f'pip install "tracebi[analyst] {_GIT_URL_FORM}"' in readme
        # The web line composes with the analyst extras: [web] alone lacks
        # duckdb/openpyxl, so a user who picked only it would fail at the
        # README's own step ①.
        assert f'pip install "tracebi[analyst,web] {_GIT_URL_FORM}"' in readme

    def test_readme_has_no_bare_pypi_install_line(self, tmp_path):
        readme = (self._init_project(tmp_path) / "README.md").read_text()
        # A bare extras install (no git URL) would resolve against PyPI,
        # where tracebi is not published.
        assert 'pip install "tracebi[analyst]"' not in readme
        assert 'pip install "tracebi[web]"' not in readme
        assert "pip install 'tracebi[" not in readme

    def test_init_prints_git_init_next_step_without_running_git(
        self, tmp_path, capsys
    ):
        project = self._init_project(tmp_path)
        out = capsys.readouterr().out
        assert "git init" in out
        # init must only *suggest* git — never run it for the user.
        assert not (project / ".git").exists()

    def test_readme_leads_with_ask_build_schedule(self, tmp_path):
        readme = (self._init_project(tmp_path) / "README.md").read_text()
        assert "Ask it, build it, schedule it" in readme
        # Receipts stay as the supporting trust line, not the headline.
        lead = readme.split("## Install", 1)[0]
        assert "trust layer for AI-generated analytics" not in lead


class TestCliHelpLead:
    def test_help_leads_with_ask_build_schedule(self, capsys):
        from tracebi.cli import main

        with pytest.raises(SystemExit) as exc:
            main(["--help"])
        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert "ask it, build it, schedule it" in out.lower()
        assert "TraceBi — the trust layer" not in out


# ── tracebi serve: dead end without the web package ───────────────────────

class TestServeWithoutWebPackage:
    @pytest.fixture()
    def project_cwd(self, tmp_path, monkeypatch):
        """An init-shaped project as cwd, with sys.path restored afterwards
        (cmd_serve inserts cwd into sys.path)."""
        (tmp_path / "models").mkdir()
        (tmp_path / "requests").mkdir()
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "path", list(sys.path))
        return tmp_path

    def _serve(self):
        from tracebi.cli import main
        return main(["serve"])

    def test_missing_web_package_exits_nonzero_with_actionable_message(
        self, project_cwd, monkeypatch, capsys
    ):
        import tracebi.cli as cli
        monkeypatch.setattr(cli, "_web_app_importable", lambda: False)
        rc = self._serve()
        err = capsys.readouterr().err
        assert rc != 0
        # Actionable: name the problem, offer the clone and PYTHONPATH
        # escapes, and give the real (git-URL) install line.
        assert "tracebi.web.api" in err
        assert "git clone https://github.com/saltyscott0521/tracebi" in err
        assert "PYTHONPATH" in err
        assert _GIT_URL_FORM in err

    def test_missing_web_package_never_prints_serving_banner(
        self, project_cwd, monkeypatch, capsys
    ):
        import tracebi.cli as cli
        monkeypatch.setattr(cli, "_web_app_importable", lambda: False)
        assert self._serve() != 0
        assert "serving" not in capsys.readouterr().out

    def test_missing_uvicorn_message_uses_git_url_form(
        self, project_cwd, monkeypatch, capsys
    ):
        import tracebi.cli as cli
        monkeypatch.setattr(cli, "_web_app_importable", lambda: True)
        # A None entry in sys.modules makes `import uvicorn` raise
        # ImportError without touching the installed package.
        monkeypatch.setitem(sys.modules, "uvicorn", None)
        rc = self._serve()
        err = capsys.readouterr().err
        assert rc != 0
        assert _GIT_URL_FORM in err
        assert "pip install 'tracebi[web]'" not in err


# ── Renderers: warn when git_sha is unknown ───────────────────────────────

class TestUnknownGitShaWarning:
    _WARNING_MARKER = "code provenance"

    def _two_section_report(self):
        from tracebi.reports.report import Report, TextSection
        return (
            Report("Provenance Test")
            .add(TextSection(title="A", content="one"))
            .add(TextSection(title="B", content="two"))
        )

    def test_unknown_git_sha_warns_exactly_once_per_render(
        self, tmp_path, monkeypatch, capsys
    ):
        import tracebi.reports.base_renderer as br
        import tracebi.reports.report as report_mod
        from tracebi.reports.html_renderer import HTMLRenderer

        monkeypatch.setattr(report_mod, "_GIT_SHA", "unknown")
        monkeypatch.setattr(br, "_GIT_SHA_WARNED", False)   # no earlier build used it up
        HTMLRenderer().render(
            self._two_section_report(), str(tmp_path / "r.html")
        )
        err = capsys.readouterr().err
        # Once per render — not once per section, and not silenced.
        assert err.count(self._WARNING_MARKER) == 1

    def test_known_git_sha_does_not_warn(self, tmp_path, monkeypatch, capsys):
        import tracebi.reports.report as report_mod
        from tracebi.reports.html_renderer import HTMLRenderer

        monkeypatch.setattr(report_mod, "_GIT_SHA", "abc1234")
        HTMLRenderer().render(
            self._two_section_report(), str(tmp_path / "r.html")
        )
        assert self._WARNING_MARKER not in capsys.readouterr().err


# ── .gitignore: manifests are receipts, keep them ─────────────────────────

class TestGitignoreRetainsManifests:
    def _repo_gitignore(self) -> str:
        return (
            Path(__file__).resolve().parent.parent / ".gitignore"
        ).read_text()

    def test_init_gitignore_never_ignores_manifests(self, tmp_path):
        from tracebi.cli import main
        project = tmp_path / "proj"
        assert main(["init", str(project)]) == 0
        lines = [
            ln.strip()
            for ln in (project / ".gitignore").read_text().splitlines()
        ]
        assert "*.manifest.json" not in lines


# ─────────────────────────────────────────────
# Review-pass fixes (adversarial findings)
# ─────────────────────────────────────────────

def test_commitless_repo_records_unknown_not_the_literal_HEAD(tmp_path, monkeypatch):
    """`git rev-parse HEAD` in a repo with no commits exits 128 but still
    prints the literal string 'HEAD' — which was recorded as provenance and
    evaded the unknown-sha warning. Only an actual sha counts."""
    import subprocess

    import tracebi.reports.report as report_mod

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(report_mod, "_GIT_SHA", None)
    try:
        assert report_mod._current_git_sha() == "unknown"
    finally:
        report_mod._GIT_SHA = None   # do not leak the cwd's sha to other tests


def test_gitignore_negation_actually_retains_output_manifests(tmp_path):
    """git cannot re-include a file under an excluded DIRECTORY — the rule
    must be `output/*` + negation, not `output/`. Checked with git itself."""
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    gi = (Path(__file__).resolve().parents[1] / ".gitignore").read_text()
    (tmp_path / ".gitignore").write_text(gi)
    out = tmp_path / "output"
    out.mkdir()
    (out / "report.html").write_text("x")
    (out / "report.manifest.json").write_text("{}")

    def ignored(rel):
        return subprocess.run(
            ["git", "check-ignore", "-q", rel], cwd=tmp_path
        ).returncode == 0

    assert ignored("output/report.html")
    assert not ignored("output/report.manifest.json")
