from __future__ import annotations

import copy
import html
import re
from pathlib import Path

from rox_dox.links import source_url
from rox_dox.model import Page
from rox_dox.repo_tree import RepoEntry
from rox_dox.render import page_href, render_folder_page, render_page
from rox_dox.schema import extract_tables
from rox_dox.tree import build_tree

REPO_URL = "https://github.com/Rox-AI/rox-core"


def _empty_page(data: dict[str, object]) -> Page:
    payload = copy.deepcopy(data)
    payload["block"]["nodes"] = []
    payload["block"]["edges"] = []
    payload["data"]["sql_tables"] = []
    payload["data"]["nosql"] = []
    payload["sequences"] = []
    payload["states"] = []
    return Page.model_validate(payload)


def test_tldr_claim_and_table_citations_link_to_numbered_sources(
    page_data: dict[str, object],
    plantuml_jar: Path,
) -> None:
    payload = copy.deepcopy(page_data)
    payload["tldr"]["summary"][0]["sources"].append(
        {"path": "pkg/a.py", "lines": [4, 5]}
    )
    payload["tldr"]["additional_tables"] = [
        {
            "title": "Links to other domains",
            "columns": ["This table", "Other table", "Other domain", "Signal"],
            "rows": [["sessions", "users", "People", "ID column name"]],
            "links": [{"row": 0, "column": 2, "page": "domain-people"}],
            "sources": [{"path": "pkg/a.py", "lines": [6, 6]}],
        }
    ]
    page = _empty_page(payload)
    document = render_page(
        page,
        tree=build_tree([page]),
        repo_url=REPO_URL,
        tables={},
        jar=plantuml_jar,
    )

    first_source_url = source_url(
        page.tldr.summary[0].sources[0],
        repo_url=REPO_URL,
        commit=page.commit,
    )
    second_source_url = source_url(
        page.tldr.summary[0].sources[1],
        repo_url=REPO_URL,
        commit=page.commit,
    )
    table_source_url = source_url(
        page.tldr.table.sources[0],
        repo_url=REPO_URL,
        commit=page.commit,
    )
    assert f'<a href="{first_source_url}">[1]</a>' in document
    assert f'<a href="{second_source_url}">[2]</a>' in document
    assert f'<a href="{table_source_url}">[1]</a>' in document
    assert "<h3>Links to other domains</h3>" in document
    assert f'href="{page_href(page.id, "domain-people")}"' in document
    assert re.search(r"<h3>Summary</h3>.*?<h3>Key Points</h3>", document, re.DOTALL)
    assert "<h3>Table</h3>" in document
    assert "<h3>Interesting Notes</h3>" in document


def test_more_than_three_sources_collapse_in_claims_and_table(
    page_data: dict[str, object],
    plantuml_jar: Path,
) -> None:
    payload = copy.deepcopy(page_data)
    sources = [
        {"path": f"pkg/source-{index}.py", "lines": [index, index]}
        for index in range(1, 5)
    ]
    payload["tldr"]["summary"][0]["sources"] = sources
    payload["tldr"]["table"]["sources"] = sources
    page = _empty_page(payload)
    document = render_page(
        page,
        tree=build_tree([page]),
        repo_url=REPO_URL,
        tables={},
        jar=plantuml_jar,
    )

    assert (
        document.count('<details class="citations"><summary>Sources (4)</summary>') == 2
    )
    assert (
        '<div class="table-sources">Sources: '
        '<details class="citations"><summary>Sources (4)</summary>'
    ) in document
    assert ".citation-links {\n  overflow-wrap: anywhere;" in document


def test_three_or_fewer_sources_remain_inline(
    page_data: dict[str, object],
    plantuml_jar: Path,
) -> None:
    payload = copy.deepcopy(page_data)
    payload["tldr"]["summary"][0]["sources"] = [
        {"path": f"pkg/source-{index}.py", "lines": [index, index]}
        for index in range(1, 4)
    ]
    page = _empty_page(payload)
    document = render_page(
        page,
        tree=build_tree([page]),
        repo_url=REPO_URL,
        tables={},
        jar=plantuml_jar,
    )

    assert '<span class="citations"><sup><a href="' in document
    assert "<summary>Sources (3)</summary>" not in document


def test_plantuml_diagram_wrappers_support_intrinsic_scrolling(
    page_data: dict[str, object],
    plantuml_jar: Path,
) -> None:
    page = _empty_page(page_data)
    document = render_page(
        page,
        tree=build_tree([page]),
        repo_url=REPO_URL,
        tables={},
        jar=plantuml_jar,
    )

    expected_diagrams = len(page.sequences)
    assert document.count('<div class="diagram block-scroll">') == expected_diagrams
    assert ".block-scroll svg {\n  max-width: none;\n}" in document
    assert (
        ".diagram svg {\n"
        "  display: block;\n"
        "  max-width: 100%;\n"
        "  height: auto;\n"
        "  margin: 0 auto;\n}"
    ) in document


def test_diagram_cards_expand_to_their_own_ids_and_panel_controls_are_css_only(
    page_data: dict[str, object],
    git_repo: tuple[Path, str],
    plantuml_jar: Path,
) -> None:
    repo, commit = git_repo
    payload = copy.deepcopy(page_data)
    payload["sequences"].append(copy.deepcopy(payload["sequences"][0]))
    page = Page.model_validate(payload)
    document = render_page(
        page,
        tree=build_tree([page]),
        repo_url=REPO_URL,
        tables=extract_tables(repo, commit),
        jar=plantuml_jar,
    )

    diagram_ids = re.findall(
        r'<article class="diagram-card zoom-figure" id="([^"]+)">',
        document,
    )
    expand_targets = re.findall(
        r'<a class="expand" href="#([^"]+)">Expand full screen</a>',
        document,
    )
    close_targets = re.findall(
        r'<a class="collapse" href="#([^"]+)">Close</a>',
        document,
    )
    assert diagram_ids == [
        "block-figure",
        "fig-schema",
        "fig-sequences-1",
        "fig-sequences-2",
    ]
    assert expand_targets == diagram_ids
    assert close_targets == [
        "block",
        "schema",
        "sequences",
        "sequences",
    ]
    assert document.count('<div class="diagram block-scroll">') == (
        2 + len(page.sequences)
    )

    left_toggle = '<input class="panel-toggle" id="toggle-left" type="checkbox">'
    page_shell = '<div class="page-shell">'
    assert left_toggle in document
    assert 'id="toggle-right"' not in document
    assert 'id="context"' not in document
    assert document.index(left_toggle) < document.index(page_shell)
    assert '<main class="page-main"><div class="panel-controls">' in document
    assert (
        '</div><label class="panel-handle" for="toggle-left" title="Toggle side panel">'
        '<span class="panel-handle-grip"><svg'
    ) in document
    assert document.index('class="panel-handle"') < document.index(
        '<main class="page-main">'
    )
    assert 'for="toggle-right"' not in document
    assert '<div class="theme-switch" role="group" aria-label="Theme">' in document
    for choice in ("light", "dark", "system"):
        assert f'data-theme-choice="{choice}"' in document
    assert (
        ".page-shell {\n"
        "  display: grid;\n"
        "  grid-template-columns: 280px 1.25rem minmax(0, 1fr);"
    ) in document
    assert (
        "#toggle-left:checked ~ .page-shell {\n"
        "  grid-template-columns: 0 1.25rem minmax(0, 1fr);\n}"
    ) in document
    assert "@media (max-width: 1100px)" in document
    assert (
        "  #toggle-left:checked ~ .page-shell {\n"
        "    grid-template-columns: minmax(0, 1fr);\n  }"
    ) in document
    assert "toggle-right" not in document
    assert ".panel-toggle {\n  position: fixed;\n  top: 0;\n  left: 0;" in document
    assert "<script src" not in document
    assert document.count("<script>") == 1


def test_schema_sources_include_authored_relations(
    page_data: dict[str, object],
    git_repo: tuple[Path, str],
    plantuml_jar: Path,
) -> None:
    repo, commit = git_repo
    payload = copy.deepcopy(page_data)
    payload["data"]["relations"] = [
        {
            "src": "sessions.user_id",
            "dst": "users.id",
            "label": "implicit foreign key",
            "source": {"path": "pkg/a.py", "lines": [4, 5]},
        }
    ]
    page = Page.model_validate(payload)
    document = render_page(
        page,
        tree=build_tree([page]),
        repo_url=REPO_URL,
        tables=extract_tables(repo, commit),
        jar=plantuml_jar,
    )
    schema = re.search(r'<section id="schema">(.*?)</section>', document, re.DOTALL)
    assert schema is not None

    relation_url = source_url(
        page.data.relations[0].source,
        repo_url=REPO_URL,
        commit=page.commit,
    )
    assert "relation sessions.user_id -&gt; users.id" in schema.group(1)
    assert relation_url in schema.group(1)


def test_domain_schema_renders_legend_table_guide_and_key_table_sources(
    page_data: dict[str, object],
    git_repo: tuple[Path, str],
    plantuml_jar: Path,
) -> None:
    repo, commit = git_repo
    payload = copy.deepcopy(page_data)
    payload["data"]["sql_tables"] = []
    payload["data"]["columns"] = [["identity", "cache"]]
    payload["data"]["domains"] = [
        {
            "id": "identity",
            "title": "Identity",
            "tables": ["users", "sessions"],
            "key_tables": ["users"],
            "notes": [
                {
                    "text": "Tenant keys are summarized.",
                    "sources": [{"path": "pkg/a.py", "lines": [1, 2]}],
                }
            ],
        }
    ]
    page = Page.model_validate(payload)
    tables = extract_tables(repo, commit)
    document = render_page(
        page,
        tree=build_tree([page]),
        repo_url=REPO_URL,
        tables=tables,
        jar=plantuml_jar,
    )

    schema = re.search(r'<section id="schema">(.*?)</section>', document, re.DOTALL)
    assert schema is not None
    guide = schema.group(1)
    assert "Each card is a domain" in guide
    assert "orange = primary / referenced key" in guide
    assert "blue = referencing (FK-like) column" in guide
    assert "solid blue = DB-enforced FK" in guide
    assert "dashed grey = symbolic reference (no constraint)" in guide
    assert "dotted amber = blob pointer" in guide
    assert "keys the arrows use are listed on the right" in guide
    assert "width: 34px" in document
    assert "border-top-style: dashed" in document
    assert "border-top-style: dotted" in document
    assert "<h3>Table guide</h3>" in guide
    assert (
        '<details id="schema-guide-identity"><summary>Identity — 2 tables</summary>'
        in guide
    )
    assert (
        f'<a href="{source_url(tables["sessions"].source, repo_url=REPO_URL, commit=page.commit)}">'
        "<code>sessions</code></a>" in guide
    )
    assert "schema domain identity key table users" in guide
    assert "schema domain identity note 1" in guide


def test_related_page_links_are_relative_and_external_targets_link_out(
    page_data: dict[str, object],
    plantuml_jar: Path,
) -> None:
    payload = copy.deepcopy(page_data)
    payload["id"] = "a/b"
    payload["block"]["nodes"] = []
    payload["block"]["edges"] = []
    payload["data"]["sql_tables"] = []
    payload["data"]["nosql"] = []
    payload["sequences"] = []
    payload["states"] = []
    payload["related"] = [
        {
            "label": "Nested sibling",
            "page": "a/c/d",
            "source": {"path": "pkg/a.py", "lines": [1, 3]},
        },
        {
            "label": "External guide",
            "url": "https://docs.example.test/guide",
            "source": {"notion": "https://www.notion.so/rox/Foo-123"},
        },
    ]
    page = _empty_page(payload)
    document = render_page(
        page,
        tree=build_tree([page]),
        repo_url=REPO_URL,
        tables={},
        jar=plantuml_jar,
    )

    assert page_href("a/b", "a/c/d") == "c/d.html"
    assert '<a href="c/d.html">Nested sibling</a>' in document
    assert '<a href="https://docs.example.test/guide">External guide</a>' in document
    assert (
        '<a href="https://github.com/Rox-AI/rox-core/blob/'
        f'{page.commit}/pkg/a.py#L1-L3">[1]</a>'
    ) in document
    assert '<a href="https://www.notion.so/rox/Foo-123">[1]</a>' in document


def test_empty_key_points_show_module_placeholder(
    page_data: dict[str, object],
    plantuml_jar: Path,
) -> None:
    page = _empty_page(page_data)
    document = render_page(
        page,
        tree=build_tree([page]),
        repo_url=REPO_URL,
        tables={},
        jar=plantuml_jar,
    )

    assert (
        '<h3>Key Points</h3><p class="empty">Nothing to show for this module.</p>'
        in document
    )


def test_empty_notes_show_module_placeholder(
    page_data: dict[str, object],
    plantuml_jar: Path,
) -> None:
    payload = copy.deepcopy(page_data)
    payload["tldr"]["notes"] = []
    page = _empty_page(payload)
    document = render_page(
        page,
        tree=build_tree([page]),
        repo_url=REPO_URL,
        tables={},
        jar=plantuml_jar,
    )

    assert (
        '<h3>Interesting Notes</h3><p class="empty">Nothing to show for this module.</p>'
        in document
    )


def test_page_text_is_html_escaped(
    page_data: dict[str, object],
    plantuml_jar: Path,
) -> None:
    payload = copy.deepcopy(page_data)
    payload["title"] = 'Page <script>alert("unsafe")</script>'
    payload["tldr"]["summary"][0]["text"] = "Claim <img src=x>"
    page = _empty_page(payload)
    document = render_page(
        page,
        tree=build_tree([page]),
        repo_url=REPO_URL,
        tables={},
        jar=plantuml_jar,
    )

    assert "<script>alert" not in document
    assert "&lt;script&gt;alert" in document
    assert "Claim &lt;img src=x&gt;" in document


def test_focused_block_figure_renders_notes_and_its_own_sources(
    page_data: dict[str, object],
    git_repo: tuple[Path, str],
    plantuml_jar: Path,
) -> None:
    repo, commit = git_repo
    payload = copy.deepcopy(page_data)
    note_source = {"path": "pkg/a.py", "lines": [1, 2]}
    group_source = {"path": "pkg/a.py", "lines": [2, 3]}
    node_source = {"path": "pkg/a.py", "lines": [3, 4]}
    detail_source = {"path": "pkg/a.py", "lines": [4, 5]}
    edge_source = {"path": "pkg/a.py", "lines": [5, 6]}
    payload["block_figures"] = [
        {
            "id": "task-pipeline",
            "title": "Background task pipeline",
            "notes": [
                {
                    "text": "Based on the task pattern, corrected.",
                    "sources": [note_source],
                }
            ],
            "block": {
                "groups": [
                    {
                        "id": "worker",
                        "label": "Worker",
                        "source": group_source,
                    }
                ],
                "nodes": [
                    {
                        "id": "listener",
                        "label": "Listener",
                        "source": node_source,
                        "group": "worker",
                        "details": [
                            {"text": "Polls the queue.", "sources": [detail_source]}
                        ],
                    },
                    {"id": "handler", "label": "Handler", "source": node_source},
                ],
                "edges": [
                    {
                        "src": "listener",
                        "dst": "handler",
                        "label": "execute",
                        "source": edge_source,
                    }
                ],
            },
        }
    ]
    page = Page.model_validate(payload)
    document = render_page(
        page,
        tree=build_tree([page]),
        repo_url=REPO_URL,
        tables=extract_tables(repo, commit),
        jar=plantuml_jar,
    )

    note_url = source_url(
        page.block_figures[0].notes[0].sources[0],
        repo_url=REPO_URL,
        commit=commit,
    )
    figure_sources = {
        "block figure task-pipeline note 1": note_url,
        "block figure task-pipeline group worker": source_url(
            page.block_figures[0].block.groups[0].source,
            repo_url=REPO_URL,
            commit=commit,
        ),
        "block figure task-pipeline node listener": source_url(
            page.block_figures[0].block.nodes[0].source,
            repo_url=REPO_URL,
            commit=commit,
        ),
        "block figure task-pipeline node listener detail 1": source_url(
            page.block_figures[0].block.nodes[0].details[0].sources[0],
            repo_url=REPO_URL,
            commit=commit,
        ),
        "block figure task-pipeline edge listener->handler": source_url(
            page.block_figures[0].block.edges[0].source,
            repo_url=REPO_URL,
            commit=commit,
        ),
    }

    assert (
        '<article class="diagram-card zoom-figure" id="block-figure-task-pipeline">'
        in document
    )
    assert "<h3>Figure 2. Background task pipeline</h3>" in document
    assert "Based on the task pattern, corrected." in document
    assert f'<a href="{note_url}">[1]</a>' in document
    for label, url in figure_sources.items():
        assert f'<th scope="row">{html.escape(label)}</th>' in document
        assert f'<a href="{url}">Source</a>' in document
    for url, text in (
        (figure_sources["block figure task-pipeline node listener"], "Listener"),
        (
            figure_sources["block figure task-pipeline node listener detail 1"],
            "Polls the queue.",
        ),
        (
            figure_sources["block figure task-pipeline edge listener->handler"],
            "execute",
        ),
    ):
        assert re.search(
            rf'<a href="{re.escape(url)}" target="_top">'
            rf"(?:(?!</a>).)*{re.escape(text)}(?:(?!</a>).)*</a>",
            document,
            re.DOTALL,
        )


def test_unreached_block_files_render_in_a_collapsed_list(
    page_data: dict[str, object],
    git_repo: tuple[Path, str],
    plantuml_jar: Path,
) -> None:
    repo, commit = git_repo
    payload = copy.deepcopy(page_data)
    payload["block"]["unreached_files"] = ["backend/src/helpers/unreached.py"]
    page = Page.model_validate(payload)
    document = render_page(
        page,
        tree=build_tree([page]),
        repo_url=REPO_URL,
        tables=extract_tables(repo, commit),
        jar=plantuml_jar,
    )

    assert (
        '<details class="unreached-files">'
        "<summary>Files not reached by a component (1)</summary>"
        "<ul><li>backend/src/helpers/unreached.py</li></ul></details>"
    ) in document


def test_same_title_diagrams_only_list_their_own_sources(
    page_data: dict[str, object],
    git_repo: tuple[Path, str],
    plantuml_jar: Path,
) -> None:
    _, commit = git_repo
    payload = copy.deepcopy(page_data)
    payload["block"]["nodes"] = []
    payload["block"]["edges"] = []
    payload["data"]["sql_tables"] = []
    payload["data"]["nosql"] = []
    payload["related"] = []
    second_sequence = copy.deepcopy(payload["sequences"][0])
    second_sequence["participants"][0]["source"]["lines"] = [2, 3]
    second_sequence["participants"][1]["source"]["lines"] = [3, 4]
    second_sequence["steps"][0]["source"]["lines"] = [5, 6]
    payload["sequences"].append(second_sequence)
    page = Page.model_validate(payload)
    document = render_page(
        page,
        tree=build_tree([page]),
        repo_url=REPO_URL,
        tables={},
        jar=plantuml_jar,
    )

    sequence_cards = re.findall(
        r'<article class="diagram-card zoom-figure" id="[^"]+">(.*?)</article>',
        re.search(r'<section id="sequences">.*?</section>', document, re.DOTALL)[0],
        re.DOTALL,
    )
    first_sequence_url = source_url(
        page.sequences[0].participants[0].source,
        repo_url=REPO_URL,
        commit=commit,
    )
    second_sequence_url = source_url(
        page.sequences[1].participants[0].source,
        repo_url=REPO_URL,
        commit=commit,
    )
    assert len(sequence_cards) == 2
    assert first_sequence_url in sequence_cards[0]
    assert second_sequence_url not in sequence_cards[0]
    assert second_sequence_url in sequence_cards[1]
    assert first_sequence_url not in sequence_cards[1]


def test_block_sources_table_includes_group_and_detail_citations(
    page_data: dict[str, object],
    plantuml_jar: Path,
) -> None:
    payload = copy.deepcopy(page_data)
    block = payload["block"]
    group_source = {"path": "pkg/a.py", "lines": [4, 5]}
    detail_source = {"path": "pkg/a.py", "lines": [6, 7]}
    block["groups"] = [
        {
            "id": "backend",
            "label": "Backend",
            "source": group_source,
        }
    ]
    block["nodes"][0]["group"] = "backend"
    block["nodes"][0]["details"] = [
        {
            "text": "Handles requests",
            "sources": [detail_source],
        }
    ]
    payload["data"]["sql_tables"] = []
    payload["data"]["nosql"] = []
    payload["sequences"] = []
    payload["states"] = []
    page = Page.model_validate(payload)

    document = render_page(
        page,
        tree=build_tree([page]),
        repo_url=REPO_URL,
        tables={},
        jar=plantuml_jar,
    )
    block_section = re.search(r'<section id="block">.*?</section>', document, re.DOTALL)
    assert block_section is not None
    block_html = block_section.group()

    group_url = source_url(
        page.block.groups[0].source,
        repo_url=REPO_URL,
        commit=page.commit,
    )
    detail_url = source_url(
        page.block.nodes[0].details[0].sources[0],
        repo_url=REPO_URL,
        commit=page.commit,
    )
    assert '<th scope="row">block group backend</th>' in block_html
    assert '<th scope="row">block node api detail 1</th>' in block_html
    assert f'<a href="{group_url}">Source</a>' in block_html
    assert f'<a href="{detail_url}">Source</a>' in block_html


def test_feature_tree_and_explorer_are_separate(
    page_data: dict[str, object],
    plantuml_jar: Path,
) -> None:
    payload = copy.deepcopy(page_data)
    payload["data"]["sql_tables"] = []
    payload["data"]["nosql"] = []
    payload["sequences"] = []
    payload["states"] = []
    root = Page.model_validate(payload)
    child_payload = copy.deepcopy(payload)
    child_payload.update(
        {
            "id": "rox-core/pkg",
            "title": "pkg",
            "kind": "domain",
            "parent": "rox-core",
            "paths": ["pkg"],
        }
    )
    child = Page.model_validate(child_payload)
    tree = build_tree([root, child])
    entries = {
        root.id: [
            RepoEntry(path="pkg", is_dir=True),
            RepoEntry(path="models", is_dir=True),
            RepoEntry(path="README.md", is_dir=False),
            RepoEntry(path="pyproject.toml", is_dir=False),
        ],
        child.id: [RepoEntry(path="pkg/a.py", is_dir=False)],
    }

    document = render_page(
        root,
        tree=tree,
        repo_url=REPO_URL,
        tables={},
        jar=plantuml_jar,
        entries=entries,
    )
    feature_tree = re.search(
        r'<nav id="feature-tree" aria-label="Feature tree">(.*?)</nav>',
        document,
        re.DOTALL,
    )
    site_nav = re.search(
        r'<nav id="site-nav" aria-label="Site navigation">(.*?)</nav>',
        document,
        re.DOTALL,
    )
    explorer = re.search(
        r'<nav id="explorer" aria-label="Repository explorer">(.*?)</nav>',
        document,
        re.DOTALL,
    )
    assert feature_tree is not None
    assert site_nav is not None
    assert explorer is not None
    feature_html = feature_tree.group(1)
    explorer_html = explorer.group(1)

    def nav_rows(nav_html: str) -> list[tuple[str, str, str]]:
        rows = []
        for match in re.finditer(
            r'<a class="row ([^"]+)" href="([^"]+)"[^>]*>(.*?)</a>',
            nav_html,
            re.DOTALL,
        ):
            label = re.search(r"<span>(.*?)</span>", match.group(3), re.DOTALL)
            assert label is not None
            rows.append((match.group(1), match.group(2), label.group(1)))
        return rows

    explorer_html = explorer.group(1)
    explorer_rows = nav_rows(explorer_html)

    assert '<a class="row page current"' in feature_html
    assert 'aria-current="page"' in feature_html
    assert f'href="{page_href(root.id, child.id)}"' in feature_html
    assert "pkg" in feature_html
    assert 'class="row page' not in explorer_html
    assert 'aria-current="page"' not in explorer_html
    assert "a.py" not in explorer_html
    assert any(
        "folder" in classes and href == "folders/pkg.html" and label == "pkg"
        for classes, href, label in explorer_rows
    )
    assert any(
        "folder" in classes and href == "folders/models.html" and label == "models"
        for classes, href, label in explorer_rows
    )
    assert any(
        "file" in classes
        and href == f"{REPO_URL}/blob/{root.commit}/README.md"
        and label == "README.md"
        for classes, href, label in explorer_rows
    )
    folder_positions = [
        explorer_html.index(f"<span>{label}</span>") for label in ("pkg", "models")
    ]
    file_positions = [
        explorer_html.index(f"<span>{label}</span>")
        for label in ("README.md", "pyproject.toml")
    ]
    assert max(folder_positions) < min(file_positions)


def test_membership_section_follows_tldr_and_uses_numbered_citations(
    page_data: dict[str, object],
    plantuml_jar: Path,
) -> None:
    root = _empty_page(page_data)
    domain_payload = copy.deepcopy(page_data)
    domain_payload.update(
        {
            "id": "domain",
            "title": "Domain",
            "kind": "domain",
            "parent": root.id,
            "paths": ["pkg"],
        }
    )
    domain = _empty_page(domain_payload)
    feature_payload = copy.deepcopy(page_data)
    feature_payload.update(
        {
            "id": "feature",
            "title": "Feature",
            "kind": "feature",
            "parent": domain.id,
            "paths": ["pkg/a.py"],
            "membership": [
                {
                    "layer": "Web screens",
                    "rows": [
                        {
                            "path": "web/src/page.tsx",
                            "primary": True,
                            "evidence": "route tag campaigns",
                            "sources": [{"path": "pkg/a.py", "lines": [2, 2]}],
                        }
                    ],
                }
            ],
        }
    )
    feature = _empty_page(feature_payload)
    tree = build_tree([root, domain, feature])

    document = render_page(
        feature,
        tree=tree,
        repo_url=REPO_URL,
        tables={},
        jar=plantuml_jar,
    )

    assert document.index('id="tldr"') < document.index('id="membership"')
    assert document.index('id="membership"') < document.index('id="block"')
    assert 'href="#membership">Why these files are one feature' in document
    assert "<summary>Web screens — 1 files (0 shared)</summary>" in document
    assert '<th scope="col">File</th>' in document
    assert '<th scope="col">Primary/shared</th>' in document
    assert '<th scope="col">Evidence</th>' in document
    assert ">route tag campaigns" in document
    assert 'href="https://github.com/Rox-AI/rox-core/blob/' in document
    assert "overflow-wrap: anywhere" in document


def test_folder_page_toc_lists_folder_sections(
    page_data: dict[str, object],
) -> None:
    root = Page.model_validate(page_data)

    document = render_folder_page(
        folder_path=".",
        tree=build_tree([root]),
        entries={},
        repo_url=REPO_URL,
        feature_rows=[],
        uncovered_count=2,
        child_folders=[],
        tests_only=False,
    )

    assert 'href="#feature-coverage">Feature coverage' in document
    assert 'href="#uncovered-files">Uncovered files' in document
    assert 'href="#tldr"' not in document
    assert 'href="#block"' not in document


def test_feature_tree_leaf_row_has_no_chevron(
    page_data: dict[str, object],
    plantuml_jar: Path,
) -> None:
    payload = copy.deepcopy(page_data)
    payload["block"]["nodes"] = []
    payload["block"]["edges"] = []
    payload["data"]["sql_tables"] = []
    payload["data"]["nosql"] = []
    payload["sequences"] = []
    payload["states"] = []
    root = Page.model_validate(payload)
    tree = build_tree([root])
    document = render_page(
        root,
        tree=tree,
        repo_url=REPO_URL,
        tables={},
        jar=plantuml_jar,
    )
    feature_tree = re.search(
        r'<nav id="feature-tree" aria-label="Feature tree">(.*?)</nav>',
        document,
        re.DOTALL,
    )
    assert feature_tree is not None
    leaf_row = re.search(
        r'<li>(<a class="row page current".*?</a>)</li>',
        feature_tree.group(1),
        re.DOTALL,
    )
    assert leaf_row is not None
    assert 'class="chevron-spacer"' in leaf_row.group(1)
    assert 'class="chevron"' not in leaf_row.group(1)


def test_feature_tree_expands_current_path_and_labels_children(
    page_data: dict[str, object],
    plantuml_jar: Path,
) -> None:
    payload = copy.deepcopy(page_data)
    payload["block"]["nodes"] = []
    payload["block"]["edges"] = []
    payload["data"]["sql_tables"] = []
    payload["data"]["nosql"] = []
    payload["sequences"] = []
    payload["states"] = []
    root = Page.model_validate(payload)

    domain_payload = copy.deepcopy(payload)
    domain_payload.update(
        {
            "id": "rox-core/pkg",
            "title": "pkg",
            "kind": "domain",
            "parent": root.id,
            "paths": ["pkg"],
        }
    )
    domain = Page.model_validate(domain_payload)

    feature_payload = copy.deepcopy(payload)
    feature_payload.update(
        {
            "id": "rox-core/pkg/thing",
            "title": "thing",
            "kind": "feature",
            "parent": domain.id,
            "paths": ["other/thing"],
        }
    )
    feature = Page.model_validate(feature_payload)

    sibling_domain_payload = copy.deepcopy(payload)
    sibling_domain_payload.update(
        {
            "id": "rox-core/z-archive",
            "title": "z-archive",
            "kind": "domain",
            "parent": root.id,
            "paths": ["models"],
        }
    )
    sibling_domain = Page.model_validate(sibling_domain_payload)

    sibling_feature_payload = copy.deepcopy(payload)
    sibling_feature_payload.update(
        {
            "id": "rox-core/z-archive/hidden",
            "title": "hidden",
            "kind": "feature",
            "parent": sibling_domain.id,
            "paths": ["models"],
        }
    )
    sibling_feature = Page.model_validate(sibling_feature_payload)

    tree = build_tree([root, domain, feature, sibling_domain, sibling_feature])
    document = render_page(
        feature,
        tree=tree,
        repo_url=REPO_URL,
        tables={},
        jar=plantuml_jar,
    )
    feature_tree = re.search(
        r'<nav id="feature-tree" aria-label="Feature tree">(.*?)</nav>',
        document,
        re.DOTALL,
    )
    assert feature_tree is not None
    nav_html = feature_tree.group(1)
    rows = re.findall(
        r'<a class="row ([^"]+)" href="([^"]+)"[^>]*>(.*?)</a>',
        nav_html,
        re.DOTALL,
    )
    labels = [
        html.unescape(re.search(r"<span>(.*?)</span>", markup, re.DOTALL).group(1))
        for _classes, _href, markup in rows
    ]

    assert labels == ["Rox Core", "pkg", "thing", "z-archive", "hidden"]
    assert '<a class="row page current"' in nav_html
    assert 'aria-current="page"' in nav_html
    assert nav_html.count("<details open>") == 2
    sibling_start = nav_html.rfind(
        "<details",
        0,
        nav_html.index("<span>z-archive</span>"),
    )
    assert nav_html[sibling_start:].startswith("<details>")

    root_document = render_page(
        root,
        tree=tree,
        repo_url=REPO_URL,
        tables={},
        jar=plantuml_jar,
    )
    domain_document = render_page(
        domain,
        tree=tree,
        repo_url=REPO_URL,
        tables={},
        jar=plantuml_jar,
    )
    assert "<strong>Domains:</strong>" in root_document
    assert "<strong>Features:</strong>" in domain_document


def test_page_block_diagram_renders_notes_and_cited_sources(
    page_data: dict[str, object],
    plantuml_jar: Path,
) -> None:
    payload = copy.deepcopy(page_data)
    source = {"path": "pkg/a.py", "lines": [4, 5]}
    payload["block"]["notes"] = [{"text": "Runtime flow note.", "sources": [source]}]
    payload["data"]["sql_tables"] = []
    payload["data"]["nosql"] = []
    page = Page.model_validate(payload)
    document = render_page(
        page,
        tree=build_tree([page]),
        repo_url=REPO_URL,
        tables={},
        jar=plantuml_jar,
    )

    figure = document.split('id="block-figure"', maxsplit=1)[1].split(
        "</article>", maxsplit=1
    )[0]
    assert figure.index("block-scroll") < figure.index("Runtime flow note.")
    assert "block note 1" in figure
    assert (
        source_url(
            page.block.notes[0].sources[0],
            repo_url=REPO_URL,
            commit=page.commit,
        )
        in figure
    )


def test_svg_colours_are_theme_tokens(
    page_data: dict[str, object],
    git_repo: tuple[Path, str],
    plantuml_jar: Path,
) -> None:
    repo, commit = git_repo
    page = Page.model_validate(page_data)
    document = render_page(
        page,
        tree=build_tree([page]),
        repo_url=REPO_URL,
        tables=extract_tables(repo, commit),
        jar=plantuml_jar,
    )
    assert 'fill="var(--svg-ink, #1f2937)"' in document
    assert 'fill="var(--svg-card, #ffffff)"' in document
    assert 'fill="var(--svg-ink, #000)"' in document
    assert "stroke:var(--svg-store-stroke, #3B6FD8)" in document
    assert ':root[data-theme="dark"] {' in document
    assert "  --svg-ink: #e6ebf2;\n" in document
    assert "    --svg-ink: #e6ebf2;\n" in document
    assert "  --paper: #171d26;" in document
