"""The advisory page diagram is read from the code, so it cannot drift from it. Offline."""

from __future__ import annotations

import re
from pathlib import Path
from typing import get_args

import pytest
import yaml
from bs4 import BeautifulSoup, Tag

from jobbot.adapters.ats.registry import supported_kinds
from jobbot.companies.signup import AccountNeed, account_need
from jobbot.jobs.sources import JobSourceName
from jobbot.ops import pii_guard
from jobbot.portals.detect import JOB_BOARD_KINDS, AtsKind, html_marker_kinds
from jobbot.portals.platform_map import (
    MAP_END,
    MAP_START,
    PAGE_RELPATH,
    featured_platforms,
    feedback_loop,
    inject_figure,
    main,
    platform_flavours,
    platform_label,
    render_figure,
    source_kinds,
    write_page,
)
from jobbot.workspace import repo_root

_PHONE_LIKE = re.compile(r"\+?\d[\d\s().\-]{7,}\d")


def _kinds_in(flavour_key: str) -> set[AtsKind]:
    flavour = next(f for f in platform_flavours() if f.key == flavour_key)
    return {p.kind for p in flavour.platforms if p.kind is not None}


def _all_kinds() -> list[AtsKind]:
    return [p.kind for f in platform_flavours() for p in f.platforms if p.kind is not None]


def test_committed_page_carries_the_current_diagram() -> None:
    committed = (repo_root() / PAGE_RELPATH).read_text(encoding="utf-8")
    assert committed == inject_figure(committed, render_figure()), (
        "asesoria/index.html diagram is stale — run `make site` and commit the result"
    )


def test_every_known_platform_lands_in_exactly_one_flavour() -> None:
    kinds = _all_kinds()
    assert len(kinds) == len(set(kinds))
    assert set(kinds) == set(AtsKind) - {AtsKind.UNKNOWN}


def test_every_kind_with_an_adapter_is_drawn_as_assisted() -> None:
    figure = render_figure()
    assisted = {p.kind for f in platform_flavours() for p in f.platforms if p.assisted}
    for kind in supported_kinds():
        assert kind in assisted
        assert platform_label(kind) in figure


def test_flavours_follow_the_account_table_and_the_board_set() -> None:
    assert AtsKind.GREENHOUSE in _kinds_in("ats_open")
    assert AtsKind.WORKDAY in _kinds_in("ats_account")
    assert AtsKind.SMARTRECRUITERS in _kinds_in("ats_unknown")
    assert _kinds_in("boards") == set(JOB_BOARD_KINDS)
    assert _kinds_in("email") == {AtsKind.EMAIL}
    for kind in _kinds_in("ats_open"):
        assert account_need(kind) is AccountNeed.NOT_NEEDED
    for kind in _kinds_in("ats_account"):
        assert account_need(kind) is AccountNeed.NEEDED


def test_every_job_source_maps_to_a_platform() -> None:
    """A new discovery source must not vanish from the diagram for lack of a mapping."""
    assert len(source_kinds()) == len(get_args(JobSourceName))
    assert AtsKind.LINKEDIN in source_kinds()


def test_job_sources_lead_the_board_flavour() -> None:
    boards = next(f for f in platform_flavours() if f.key == "boards")
    leading = tuple(p.kind for p in boards.platforms[: len(source_kinds())])
    assert leading == source_kinds()


def test_own_sites_note_counts_platforms_detectable_from_markup() -> None:
    own = next(f for f in platform_flavours() if f.key == "own_sites")
    assert str(len(html_marker_kinds())) in own.note
    assert AtsKind.MANATAL in html_marker_kinds()


def test_unmapped_kind_falls_back_to_its_value() -> None:
    assert platform_label(AtsKind.SMARTRECRUITERS) == "SmartRecruiters"
    assert platform_label(AtsKind.UNKNOWN) == "Unknown"


def test_figure_svgs_are_accessible_images() -> None:
    soup = BeautifulSoup(render_figure(), "html.parser")
    svgs = soup.find_all("svg")
    assert len(svgs) == 2, "one wide and one narrow layout"
    ids: set[str] = set()
    for svg in svgs:
        assert svg.get("role") == "img"
        assert svg.get("viewbox") or svg.get("viewBox")
        title, desc = svg.find("title"), svg.find("desc")
        assert title is not None and title.get_text(strip=True)
        assert desc is not None and "fuente única de verdad" in desc.get_text()
        labelled = str(svg.get("aria-labelledby", "")).split()
        assert labelled == [title.get("id"), desc.get("id")]
        for node in svg.find_all(id=True):
            assert node["id"] not in ids, f"duplicate id {node['id']}"
            ids.add(node["id"])


def test_figure_states_the_computed_counts() -> None:
    soup = BeautifulSoup(render_figure(), "html.parser")
    caption = soup.find("figcaption")
    assert caption is not None
    text = caption.get_text(" ")
    recognised = len(set(AtsKind) - {AtsKind.UNKNOWN})
    assisted = sum(p.assisted for f in platform_flavours() for p in f.platforms)
    assert f"{recognised} plataformas" in text
    assert f"en {assisted}" in text


def test_figure_is_self_contained_and_clean() -> None:
    figure = render_figure()
    assert "<script" not in figure.casefold()
    assert "http" not in figure.casefold(), "no external URL, not even an xmlns"
    assert "href" not in figure.casefold()
    assert "<image" not in figure.casefold()
    assert pii_guard.scan_text(figure, "asesoria/index.html") == []
    assert not _PHONE_LIKE.search(figure), "coordinate runs must not read as a phone number"


def test_figure_names_no_employer(project_root: Path) -> None:
    """The company base stays out: the diagram names platforms, never companies."""
    seed = yaml.safe_load(
        (project_root / "data" / "companies-cl.example.yaml").read_text(encoding="utf-8")
    )
    names = [str(row["name"]) for row in seed["companies"]]
    assert names
    text = BeautifulSoup(render_figure(), "html.parser").get_text(" ").casefold()
    for name in names:
        assert name.casefold() not in text, name


def _svgs() -> list[Tag]:
    return list(BeautifulSoup(render_figure(), "html.parser").find_all("svg"))


def _chip_labels(svg: Tag) -> set[str]:
    return {chip.get_text(" ", strip=True) for chip in svg.find_all("g", class_="pm-chip")}


def test_featured_platforms_still_exist_in_the_code() -> None:
    """The page names these in prose; renaming one in `AtsKind` must break the build."""
    kinds = featured_platforms()
    for value in (
        "workday",
        "greenhouse",
        "lever",
        "ashby",
        "smartrecruiters",
        "manatal",
        "workable",
        "getonboard",
        "indeed",
        "linkedin",
        "torre",
        "email",
    ):
        assert AtsKind(value) in kinds


def test_named_destinations_are_drawn_as_nodes_in_both_layouts() -> None:
    expected = {platform_label(kind) for kind in featured_platforms()}
    expected.add("Página de empleo propia")
    for svg in _svgs():
        assert expected <= _chip_labels(svg)
        titles = svg.get_text(" ")
        assert "Sitios propios de empresas" in titles
        assert "Postulación por email" in titles


def test_every_platform_is_its_own_node() -> None:
    labels = {p.label for f in platform_flavours() for p in f.platforms}
    for svg in _svgs():
        assert _chip_labels(svg) == labels


def test_centre_node_and_adapters_are_labelled() -> None:
    for svg in _svgs():
        source = svg.find("g", class_="pm-source")
        assert source is not None
        assert "Tu perfil · fuente única de verdad" in " ".join(source.get_text(" ").split())
        hub = svg.find("g", class_="pm-hub")
        assert hub is not None and "Adaptadores" in hub.get_text()


def test_flows_out_and_feedback_back_are_separate_animated_paths() -> None:
    for svg in _svgs():
        assert len(svg.find_all("path", class_="pm-flow")) >= len(platform_flavours())
        assert svg.find_all("path", class_="pm-feedback")


def test_feedback_loop_comes_from_existing_commands() -> None:
    """Every step of the loop is a command JobBot already has; nothing is invented."""
    from jobbot.cli import app
    from jobbot.ops.capabilities import collect_commands

    commands = {entry.path for entry in collect_commands(app)}
    steps = feedback_loop()
    assert [s.title for s in steps] == [
        "Ofertas leídas",
        "Preguntas de formularios",
        "Sugerencias del mercado",
        "Mejora diaria del CV",
    ]
    for step in steps:
        assert step.commands
        for command in step.commands:
            assert command in commands, command


def test_feedback_loop_is_drawn_and_gated_by_your_confirmation() -> None:
    for svg in _svgs():
        learned = [" ".join(g.get_text(" ").split()) for g in svg.find_all("g", class_="pm-learn")]
        for step in feedback_loop():
            assert any(step.title in text for text in learned), step.title
            for command in step.commands:
                assert any(command in text for text in learned), command
        gate = svg.find("g", class_="pm-gate")
        assert gate is not None
        text = " ".join(gate.get_text(" ").split())
        assert "Tú confirmas" in text
        assert "sin tu confirmación" in text
        lane = svg.find("text", class_="pm-lane")
        assert lane is not None
        label = " ".join(lane.get_text(" ").split())
        assert "vuelve a tu perfil" in label
        assert "tú confirmas" in label
        desc = svg.find("desc")
        assert desc is not None and "sin tu confirmación" in desc.get_text()


def test_render_is_deterministic() -> None:
    assert render_figure() == render_figure()


def test_inject_replaces_only_between_markers() -> None:
    page = f"<p>antes</p>\n{MAP_START}\nviejo\n{MAP_END}\n<p>después</p>\n"
    out = inject_figure(page, "<figure>nuevo</figure>")
    assert "viejo" not in out
    assert out.startswith("<p>antes</p>\n")
    assert out.endswith("<p>después</p>\n")
    assert f"{MAP_START}\n<figure>nuevo</figure>\n{MAP_END}" in out


def test_inject_keeps_marker_indentation() -> None:
    page = f"    {MAP_START}\n    {MAP_END}\n"
    out = inject_figure(page, "<figure>\n  <p>x</p>\n</figure>")
    assert out == f"    {MAP_START}\n    <figure>\n      <p>x</p>\n    </figure>\n    {MAP_END}\n"


def test_inject_refuses_a_page_without_markers() -> None:
    with pytest.raises(ValueError, match="markers"):
        inject_figure("<p>sin marcadores</p>", "<figure/>")


def test_write_page_updates_the_file(tmp_path: Path) -> None:
    page = tmp_path / PAGE_RELPATH
    page.parent.mkdir(parents=True)
    page.write_text(f"{MAP_START}\n{MAP_END}\n", encoding="utf-8")

    assert write_page(tmp_path) == page
    assert "<svg" in page.read_text(encoding="utf-8")


def test_main_writes_or_prints(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    page = tmp_path / PAGE_RELPATH
    page.parent.mkdir(parents=True)
    page.write_text(f"{MAP_START}\n{MAP_END}\n", encoding="utf-8")

    assert main([], root=tmp_path) == 0
    assert "<figure" in capsys.readouterr().out
    assert "<svg" not in page.read_text(encoding="utf-8")

    assert main(["--write"], root=tmp_path) == 0
    assert "<svg" in page.read_text(encoding="utf-8")
