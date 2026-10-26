"""`companies list` must show a career URL a reviewer can actually read."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from rich.console import Console

from jobbot.companies.models import CareerSiteType, DiscoverySource
from jobbot.companies.registry import CompanyRegistry, default_companies_path, save_companies
from jobbot.exit_codes import SUCCESS
from jobbot.portals.detect import AtsKind

LONG_URL = "https://careers.empresa-demo.cl/global/es/trabaja-con-nosotros/oportunidades"


def _workspace(tmp_path: Path, project_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "data" / "profile.yaml").write_text(
        (project_root / "data" / "profile.example.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    registry = CompanyRegistry()
    registry.observe(
        company="Empresa Demo",
        url=LONG_URL,
        source=DiscoverySource.USER_OBSERVATION,
        site_type=CareerSiteType.COMPANY_CAREER_PORTAL,
        country="CL",
    )
    registry.observe(
        company="Otra Demo",
        url="https://careers-page.com/otra-demo",
        source=DiscoverySource.USER_OBSERVATION,
        site_type=CareerSiteType.ATS_INSTANCE,
        ats=AtsKind.MANATAL,
    )
    save_companies(registry, default_companies_path(tmp_path))


def _site_column(out: str) -> str:
    """The Career site column with folded lines joined back together."""
    cells = [line.split("│")[3].strip() for line in out.splitlines() if line.count("│") > 3]
    return "".join(cells)


@pytest.mark.parametrize("width", [80, 200])
def test_companies_list_shows_host_and_path(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    width: int,
) -> None:
    """Regression: the column collapsed to 'https://…' and hid every portal."""
    _workspace(tmp_path, project_root, monkeypatch)
    import jobbot.cli as cli

    buffer = io.StringIO()
    monkeypatch.setattr(cli, "console", Console(file=buffer, width=width, color_system=None))

    assert cli.run_cli(["companies", "list"], standalone_mode=False) == SUCCESS
    out = buffer.getvalue()
    assert "https://…" not in out
    assert "…" not in out
    flat = _site_column(out)
    assert "careers.empresa-demo.cl/global/es/trabaja-con-nosotros/oportunidades" in flat
    assert "careers-page.com/otra-demo" in flat
