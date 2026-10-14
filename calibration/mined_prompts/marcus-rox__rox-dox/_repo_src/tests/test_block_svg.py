from __future__ import annotations

import copy
import re
from xml.etree import ElementTree

from rox_dox.block_svg import (
    ARROW_COLOR,
    QUEUE_FILL,
    QUEUE_NOTCH,
    QUEUE_STROKE,
    _layout,
    block_layout_problems,
    block_svg,
)
from rox_dox.links import page_href, source_url
from rox_dox.model import Page
from rox_dox.render import PAGE_CSS

REPO_URL = "https://github.com/Rox-AI/rox-core"
SVG_NAMESPACE = "{http://www.w3.org/2000/svg}"


def _grouped_page_data(page_data: dict[str, object]) -> dict[str, object]:
    payload = copy.deepcopy(page_data)
    block = payload["block"]
    group_source = {"path": "pkg/a.py", "lines": [1, 2]}
    nested_group_source = {"path": "pkg/a.py", "lines": [2, 3]}
    api_source = {"path": "pkg/a.py", "lines": [3, 4]}
    worker_source = {"path": "pkg/a.py", "lines": [5, 6]}
    detail_source = {"path": "pkg/a.py", "lines": [7, 8]}
    edge_source = {"path": "pkg/a.py", "lines": [8, 9]}
    block["groups"] = [
        {"id": "backend", "label": "Backend", "source": group_source},
        {
            "id": "workers",
            "label": "Workers",
            "source": nested_group_source,
            "parent": "backend",
        },
    ]
    block["nodes"] = [
        {
            "id": "api",
            "label": "API",
            "source": api_source,
            "group": "backend",
        },
        {
            "id": "worker",
            "label": "Worker A",
            "source": worker_source,
            "group": "workers",
            "details": [
                {
                    "text": "Processes requests",
                    "sources": [detail_source],
                }
            ],
        },
    ]
    block["edges"] = [
        {
            "src": "api",
            "dst": "worker",
            "label": "calls worker",
            "source": edge_source,
        }
    ]
    return payload


def _anchors(svg: str) -> list[tuple[str, str]]:
    root = ElementTree.fromstring(svg)
    return [
        (anchor.attrib["href"], "".join(anchor.itertext()))
        for anchor in root.iter(f"{SVG_NAMESPACE}a")
    ]


def _page(data: dict[str, object]) -> Page:
    return Page.model_validate(copy.deepcopy(data))


def _block_svg(page: Page) -> str:
    return block_svg(page.block, page=page, repo_url=REPO_URL)


def test_block_elements_show_labels_and_link_to_their_sources(
    page_data: dict[str, object],
) -> None:
    page = _page(_grouped_page_data(page_data))
    svg = _block_svg(page)
    links = _anchors(svg)
    expected = [
        ("BACKEND", page.block.groups[0].source),
        ("Workers", page.block.groups[1].source),
        ("API", page.block.nodes[0].source),
        ("Worker A", page.block.nodes[1].source),
        ("Processes requests", page.block.nodes[1].details[0].sources[0]),
        ("calls worker", page.block.edges[0].source),
    ]

    for label, source in expected:
        expected_href = source_url(source, repo_url=REPO_URL, commit=page.commit)
        assert any(label in text and href == expected_href for href, text in links), (
            f"missing source link for {label!r}: {links!r}"
        )
        assert label in svg

    root = ElementTree.fromstring(svg)
    edge_label = next(
        anchor
        for anchor in root.iter(f"{SVG_NAMESPACE}a")
        if "calls worker" in "".join(anchor.itertext())
    )
    assert edge_label.find(f"{SVG_NAMESPACE}rect").attrib["class"] == (
        "edge-label-background"
    )


def test_node_links_use_relative_page_href(page_data: dict[str, object]) -> None:
    payload = copy.deepcopy(page_data)
    payload["block"]["nodes"][0]["link"] = "rox-core/pkg"
    page = _page(payload)

    assert any(
        href == page_href(page.id, "rox-core/pkg") and "API" in text
        for href, text in _anchors(_block_svg(page))
    )


def test_nested_groups_render_as_sections_inside_their_parent_column(
    page_data: dict[str, object],
) -> None:
    page = _page(_grouped_page_data(page_data))
    root = ElementTree.fromstring(_block_svg(page))
    section = next(
        element
        for element in root.iter(f"{SVG_NAMESPACE}rect")
        if element.attrib.get("class") == "group-boundary"
    )
    worker_title = next(
        element
        for element in root.iter(f"{SVG_NAMESPACE}text")
        if "".join(element.itertext()) == "Worker A"
    )
    section_x = float(section.attrib["x"])
    section_y = float(section.attrib["y"])
    section_right = section_x + float(section.attrib["width"])
    section_bottom = section_y + float(section.attrib["height"])
    worker_x = float(worker_title.attrib["x"])
    worker_y = float(worker_title.attrib["y"])

    assert section_x < worker_x < section_right
    assert section_y < worker_y < section_bottom


def test_group_boundaries_use_transparent_dashed_uml_style(
    page_data: dict[str, object],
) -> None:
    root = ElementTree.fromstring(_block_svg(_page(_grouped_page_data(page_data))))
    boundary = next(
        element
        for element in root.iter(f"{SVG_NAMESPACE}rect")
        if element.attrib.get("class") == "group-boundary"
    )

    assert boundary.attrib["fill"] == "none"
    assert boundary.attrib["stroke"] == "#9ca3af"
    assert boundary.attrib["stroke-width"] == "1.25"
    assert boundary.attrib["stroke-dasharray"] == "6 4"
    assert boundary.attrib["rx"] == "6"


def test_top_level_columns_use_unnumbered_uppercase_captions(
    page_data: dict[str, object],
) -> None:
    payload = copy.deepcopy(page_data)
    source = {"path": "pkg/a.py", "lines": [1, 2]}
    payload["block"]["groups"] = [
        {"id": "first", "label": "First", "source": source},
        {"id": "second", "label": "Second", "source": source},
    ]
    payload["block"]["nodes"][0]["group"] = "first"
    payload["block"]["nodes"][1]["group"] = "second"
    svg = _block_svg(_page(payload))

    assert "FIRST" in svg
    assert "SECOND" in svg
    assert "1 · First" not in svg
    assert "2 · Second" not in svg


def test_block_svg_has_explicit_dimensions_and_is_not_width_constrained(
    page_data: dict[str, object],
) -> None:
    svg = _block_svg(_page(page_data))
    root = ElementTree.fromstring(svg)

    assert int(root.attrib["width"]) > 0
    assert int(root.attrib["height"]) > 0
    assert re.fullmatch(
        rf"0 0 {root.attrib['width']} {root.attrib['height']}",
        root.attrib["viewBox"],
    )
    assert re.search(
        r"\.block-scroll\s+svg\s*{[^}]*max-width:\s*none\b",
        PAGE_CSS,
        re.DOTALL,
    )


def test_forward_skip_with_clear_corridor_uses_solid_styling(
    page_data: dict[str, object],
) -> None:
    payload = copy.deepcopy(page_data)
    source = {"path": "pkg/a.py", "lines": [1, 2]}
    payload["block"]["groups"] = [
        {"id": group_id, "label": label, "source": source}
        for group_id, label in (
            ("first", "First"),
            ("second", "Second"),
            ("third", "Third"),
        )
    ]
    payload["block"]["nodes"] = [
        {"id": node_id, "label": node_id.upper(), "source": source, "group": group_id}
        for node_id, group_id in (
            ("first_node", "first"),
            ("second_node", "second"),
            ("third_node", "third"),
        )
    ]
    payload["block"]["edges"] = [
        {
            "src": "first_node",
            "dst": "second_node",
            "label": "adjacent",
            "source": source,
        },
        {
            "src": "first_node",
            "dst": "third_node",
            "label": "skips a column",
            "source": source,
        },
    ]
    page = _page(payload)
    assert block_layout_problems(page.block) == []
    root = ElementTree.fromstring(_block_svg(page))
    edge_paths = [
        element
        for element in root.iter(f"{SVG_NAMESPACE}path")
        if "marker-end" in element.attrib
    ]

    assert [path.attrib["stroke"] for path in edge_paths] == [ARROW_COLOR] * 2
    assert "stroke-dasharray" not in edge_paths[0].attrib
    assert "stroke-dasharray" not in edge_paths[1].attrib
    assert all(path.attrib["stroke-width"] == "1.5" for path in edge_paths)
    assert [path.attrib["d"].count("L") for path in edge_paths] == [3, 5]


def test_forward_skip_uses_a_clear_corridor(
    page_data: dict[str, object],
) -> None:
    payload = copy.deepcopy(page_data)
    source = {"path": "pkg/a.py", "lines": [1, 2]}
    payload["block"]["groups"] = [
        {"id": group_id, "label": group_id.upper(), "source": source}
        for group_id in ("first", "second", "third")
    ]
    tall_details = [
        {"text": f"Detail line {index}", "sources": [source]} for index in range(6)
    ]
    payload["block"]["nodes"] = [
        {
            "id": "first_node",
            "label": "A",
            "source": source,
            "group": "first",
            "details": tall_details,
        },
        {
            "id": "blocker",
            "label": "B",
            "source": source,
            "group": "second",
        },
        {
            "id": "third_node",
            "label": "C",
            "source": source,
            "group": "third",
            "details": tall_details,
        },
    ]
    payload["block"]["edges"] = [
        {
            "src": "first_node",
            "dst": "third_node",
            "label": "skips the short middle card",
            "source": source,
        }
    ]
    page = _page(payload)
    _, cards, routes, _, _ = _layout(page.block)
    route = routes[0]
    blocker = cards["blocker"]
    corridor_y = route.points[2][1]

    assert not route.via_channel
    assert route.points[2][1] == route.points[3][1]
    assert not blocker.y <= corridor_y <= blocker.y + blocker.height
    assert block_layout_problems(page.block) == []


def test_backward_edge_keeps_the_shared_channel(
    page_data: dict[str, object],
) -> None:
    payload = copy.deepcopy(page_data)
    source = {"path": "pkg/a.py", "lines": [1, 2]}
    payload["block"]["groups"] = [
        {"id": group_id, "label": group_id.upper(), "source": source}
        for group_id in ("first", "second", "third")
    ]
    payload["block"]["nodes"] = [
        {
            "id": f"{group_id}_node",
            "label": group_id.upper(),
            "source": source,
            "group": group_id,
        }
        for group_id in ("first", "second", "third")
    ]
    payload["block"]["edges"] = [
        {
            "src": "third_node",
            "dst": "first_node",
            "label": "backward",
            "source": source,
        }
    ]
    page = _page(payload)
    _, _, routes, _, _ = _layout(page.block)

    assert routes[0].via_channel
    assert block_layout_problems(page.block) == [
        "third_node -> first_node (columns 2 -> 0; expected 2 -> 3)"
    ]


def test_backward_and_same_column_edges_are_layout_problems(
    page_data: dict[str, object],
) -> None:
    payload = copy.deepcopy(page_data)
    source = {"path": "pkg/a.py", "lines": [1, 2]}
    payload["block"]["groups"] = [
        {"id": group_id, "label": group_id.title(), "source": source}
        for group_id in ("first", "second", "third")
    ]
    payload["block"]["nodes"] = [
        {"id": node_id, "label": node_id, "source": source, "group": group_id}
        for node_id, group_id in (
            ("first_node", "first"),
            ("second_node", "second"),
            ("second_peer", "second"),
            ("third_node", "third"),
        )
    ]
    payload["block"]["edges"] = [
        {
            "src": "third_node",
            "dst": "second_node",
            "label": "backward",
            "source": source,
        },
        {
            "src": "second_node",
            "dst": "second_peer",
            "label": "same column",
            "source": source,
        },
    ]

    page = _page(payload)

    assert block_layout_problems(page.block) == [
        "third_node -> second_node (columns 2 -> 1; expected 2 -> 3)",
        "second_node -> second_peer (columns 1 -> 1; expected 1 -> 2)",
    ]


def test_component_uses_a_white_uml_class_box(page_data: dict[str, object]) -> None:
    payload = copy.deepcopy(page_data)
    nodes = payload["block"]["nodes"]
    nodes[0].update({"label": 'API "primary"\nroute', "kind": "component"})
    root = ElementTree.fromstring(_block_svg(_page(payload)))
    shape = next(
        element
        for element in root.iter(f"{SVG_NAMESPACE}rect")
        if element.attrib.get("class") == "component-shape"
    )
    rendered = "".join(root.itertext())

    assert shape.attrib["fill"] == "#ffffff"
    assert shape.attrib["stroke"] == "#1f2937"
    assert shape.attrib["rx"] == "4"
    assert any(
        element.attrib.get("class") == "component-divider"
        for element in root.iter(f"{SVG_NAMESPACE}line")
    )
    assert 'API "primary"' in rendered
    assert "route" in rendered


def test_long_component_text_wraps_within_card_bounds(
    page_data: dict[str, object],
) -> None:
    payload = copy.deepcopy(page_data)
    payload["block"]["nodes"][0].update(
        {
            "label": "A deliberately long service title that must wrap cleanly",
            "kind": "component",
            "details": [
                {
                    "text": (
                        "Operational details wrap inside the card without overlapping "
                        "its lower border."
                    ),
                    "sources": [{"path": "pkg/a.py", "lines": [1, 2]}],
                }
            ],
        }
    )
    root = ElementTree.fromstring(_block_svg(_page(payload)))
    node_group = next(
        element
        for element in root.iter(f"{SVG_NAMESPACE}g")
        if element.attrib.get("class") == "block-node block-kind-component"
    )
    shape = next(
        element
        for element in node_group.iter(f"{SVG_NAMESPACE}rect")
        if element.attrib.get("class") == "component-shape"
    )
    card_top = float(shape.attrib["y"])
    card_bottom = card_top + float(shape.attrib["height"])
    text_groups = [
        element
        for element in node_group.iter(f"{SVG_NAMESPACE}g")
        if element.attrib.get("class") in {"card-title", "card-detail"}
    ]

    assert len(text_groups) == 2
    for group in text_groups:
        for text in group.iter(f"{SVG_NAMESPACE}text"):
            line = text.text or ""
            width = len(line) * float(text.attrib["font-size"]) * 0.62
            assert width <= float(shape.attrib["width"]) - 2 * 14
            assert card_top < float(text.attrib["y"]) < card_bottom


def test_long_identifiers_wrap_at_semantic_boundaries(
    page_data: dict[str, object],
) -> None:
    payload = copy.deepcopy(page_data)
    node = payload["block"]["nodes"][0]
    node.update(
        {
            "label": "GET /message/stream/{conversation_id}",
            "kind": "component",
            "details": [
                {
                    "text": "GoogleWorkspaceAdminCreateTaskExecutor",
                    "sources": [node["source"]],
                }
            ],
        }
    )
    root = ElementTree.fromstring(_block_svg(_page(payload)))
    node_group = next(
        element
        for element in root.iter(f"{SVG_NAMESPACE}g")
        if element.attrib.get("class") == "block-node block-kind-component"
    )
    title_group = next(
        element
        for element in node_group.iter(f"{SVG_NAMESPACE}g")
        if element.attrib.get("class") == "card-title"
    )
    detail_group = next(
        element
        for element in node_group.iter(f"{SVG_NAMESPACE}g")
        if element.attrib.get("class") == "card-detail"
    )
    title_lines = [
        element.text or "" for element in title_group.iter(f"{SVG_NAMESPACE}text")
    ]
    detail_lines = [
        element.text or "" for element in detail_group.iter(f"{SVG_NAMESPACE}text")
    ]
    detail_text = detail_lines[0].removeprefix("•  ").strip() + "".join(
        line.strip() for line in detail_lines[1:]
    )

    assert title_lines == ["GET", "/message/stream", "/{conversation_id}"]
    assert detail_lines == [
        "•  GoogleWorkspaceAdminCreateTask",
        "    Executor",
    ]
    assert detail_text == "GoogleWorkspaceAdminCreateTaskExecutor"


def test_store_renders_as_a_vertical_cylinder_without_kind_tag(
    page_data: dict[str, object],
) -> None:
    payload = copy.deepcopy(page_data)
    payload["block"]["nodes"][0].update({"label": "Cache", "kind": "store"})
    root = ElementTree.fromstring(_block_svg(_page(payload)))
    node_group = next(
        element
        for element in root.iter(f"{SVG_NAMESPACE}g")
        if element.attrib.get("class") == "block-node block-kind-store"
    )
    shape = next(
        element
        for element in node_group.iter(f"{SVG_NAMESPACE}path")
        if element.attrib.get("class") == "store-shape"
    )

    assert "A" in shape.attrib["d"]
    assert shape.attrib["fill"] == "#eff6ff"
    assert shape.attrib["stroke"] == "#1e40af"
    assert any(
        element.attrib.get("class") == "store-lid-front"
        for element in node_group.iter(f"{SVG_NAMESPACE}path")
    )
    assert "STORE" not in "".join(node_group.itertext())


def test_queue_renders_as_a_chevron(page_data: dict[str, object]) -> None:
    payload = copy.deepcopy(page_data)
    source = payload["block"]["nodes"][0]["source"]
    payload["block"]["groups"] = [
        {"id": "producer", "label": "Producer", "source": source},
        {"id": "queue-group", "label": "Queue", "source": source},
        {"id": "consumer", "label": "Consumer", "source": source},
    ]
    payload["block"]["nodes"][0]["group"] = "producer"
    payload["block"]["nodes"][1].update({"label": "Queue", "kind": "queue"})
    payload["block"]["nodes"][1]["group"] = "queue-group"
    payload["block"]["nodes"].append(
        {"id": "sink", "label": "Sink", "source": source, "group": "consumer"}
    )
    payload["block"]["edges"].append(
        {"src": "store", "dst": "sink", "label": "writes", "source": source}
    )
    root = ElementTree.fromstring(_block_svg(_page(payload)))
    shape = next(
        element
        for element in root.iter(f"{SVG_NAMESPACE}polygon")
        if element.attrib.get("class") == "queue-shape"
    )
    points = [
        tuple(float(coordinate) for coordinate in point.split(","))
        for point in shape.attrib["points"].split()
    ]
    edges = [
        element
        for element in root.iter(f"{SVG_NAMESPACE}path")
        if element.attrib.get("class") == "block-edge"
    ]
    edge_points = [
        [
            (float(x), float(y))
            for x, y in re.findall(r"[ML]([0-9.]+)\s+([0-9.]+)", edge.attrib["d"])
        ]
        for edge in edges
    ]

    assert len(points) == 6
    assert shape.attrib["fill"] == QUEUE_FILL
    assert shape.attrib["stroke"] == QUEUE_STROKE
    assert edge_points[0][-1][0] == points[0][0] + QUEUE_NOTCH
    assert edge_points[0][-1][1] == points[-1][1]
    assert edge_points[1][0] == (points[2][0], points[2][1])
    assert not any(
        (element.attrib.get("class") or "").startswith("queue-cap-")
        for element in root.iter(f"{SVG_NAMESPACE}ellipse")
    )


def test_many_nodes_render_three_shapes_and_normal_nodes_one(
    page_data: dict[str, object],
) -> None:
    payload = copy.deepcopy(page_data)
    payload["block"]["nodes"][0]["many"] = True
    root = ElementTree.fromstring(_block_svg(_page(payload)))
    node_group = next(
        element
        for element in root.iter(f"{SVG_NAMESPACE}g")
        if element.attrib.get("class") == "block-node block-kind-component"
    )
    component_shapes = [
        element
        for element in node_group.iter(f"{SVG_NAMESPACE}rect")
        if element.attrib.get("class") == "component-shape"
    ]

    assert len(component_shapes) == 3
    assert [float(element.attrib["width"]) for element in component_shapes] == [
        300,
        294,
        288,
    ]
    front_shape = component_shapes[-1]
    edge = next(
        element
        for element in root.iter(f"{SVG_NAMESPACE}path")
        if element.attrib.get("class") == "block-edge"
    )
    edge_start = re.search(r"M([0-9.]+)\s+([0-9.]+)", edge.attrib["d"])
    assert edge_start is not None
    assert (
        float(edge_start.group(2))
        == float(front_shape.attrib["y"]) + float(front_shape.attrib["height"]) / 2
    )

    normal_root = ElementTree.fromstring(_block_svg(_page(page_data)))
    normal_node_group = next(
        element
        for element in normal_root.iter(f"{SVG_NAMESPACE}g")
        if element.attrib.get("class") == "block-node block-kind-component"
    )
    normal_shapes = [
        element
        for element in normal_node_group.iter(f"{SVG_NAMESPACE}rect")
        if element.attrib.get("class") == "component-shape"
    ]
    assert len(normal_shapes) == 1


def test_group_edges_anchor_to_column_and_section_boundaries(
    page_data: dict[str, object],
) -> None:
    payload = copy.deepcopy(page_data)
    source = {"path": "pkg/a.py", "lines": [1, 3]}
    payload["block"]["groups"] = [
        {"id": "producer", "label": "Producer", "source": source},
        {"id": "consumer", "label": "Consumer", "source": source},
        {
            "id": "reader",
            "label": "Reader",
            "source": source,
            "parent": "consumer",
        },
    ]
    payload["block"]["nodes"][0]["group"] = "producer"
    payload["block"]["nodes"][1]["group"] = "reader"
    payload["block"]["edges"] = [
        {
            "src": "producer",
            "dst": "reader",
            "label": "calls",
            "source": source,
        }
    ]
    root = ElementTree.fromstring(_block_svg(_page(payload)))
    source_shape = next(
        element
        for element in root.iter(f"{SVG_NAMESPACE}rect")
        if element.attrib.get("class") == "component-shape"
    )
    section = next(
        element
        for element in root.iter(f"{SVG_NAMESPACE}rect")
        if element.attrib.get("class") == "group-boundary"
    )
    edge = next(
        element
        for element in root.iter(f"{SVG_NAMESPACE}path")
        if element.attrib.get("class") == "block-edge"
    )
    points = [
        (float(x), float(y))
        for x, y in re.findall(r"[ML]([0-9.]+)\s+([0-9.]+)", edge.attrib["d"])
    ]

    assert points[0][0] == float(source_shape.attrib["x"]) + float(
        source_shape.attrib["width"]
    )
    assert points[-1][0] == float(section.attrib["x"])


def test_external_uses_a_dashed_stereotype_box(
    page_data: dict[str, object],
) -> None:
    payload = copy.deepcopy(page_data)
    payload["block"]["nodes"][0].update({"label": "External", "kind": "external"})
    root = ElementTree.fromstring(_block_svg(_page(payload)))
    shape = next(
        element
        for element in root.iter(f"{SVG_NAMESPACE}rect")
        if element.attrib.get("class") == "external-shape"
    )

    assert shape.attrib["fill"] == "#ffffff"
    assert shape.attrib["stroke"] == "#6b7280"
    assert shape.attrib["stroke-dasharray"] == "6 4"
    assert "«external»" in "".join(root.itertext())


def test_related_disclosure_markers_are_unicode_glyphs() -> None:
    assert 'content: "▸ ";' in PAGE_CSS
    assert 'content: "▾ ";' in PAGE_CSS
    assert "\u0015BE" not in PAGE_CSS
    assert "\u0000A0" not in PAGE_CSS
