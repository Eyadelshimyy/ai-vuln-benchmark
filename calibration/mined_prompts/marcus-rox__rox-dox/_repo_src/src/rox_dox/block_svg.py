"""Column-layout SVG renderer for block diagrams.

Top-level groups become columns, read left to right. Nested groups become
labelled sections inside their column. Edges leave a card on its right side
and enter the target on its left side, routed orthogonally through the gutters
between columns. Forward skips use a clear corridor where possible, with
same-column, backward, and blocked skips using a shared channel below.
"""

from __future__ import annotations

import html
import math
import textwrap
from collections.abc import Mapping
from dataclasses import dataclass, field

from rox_dox.links import page_href, source_url
from rox_dox.model import BlockDiagram, Edge, Group, Node, Page, Source


FONT_FAMILY = "Arial, Helvetica, sans-serif"
TEXT_COLOR = "#1f2937"
MUTED_COLOR = "#6b7280"
ARROW_COLOR = "#374151"
COMPONENT_STROKE = "#1f2937"
DIVIDER_COLOR = "#d1d5db"
STORE_FILL = "#eff6ff"
STORE_STROKE = "#1e40af"
QUEUE_FILL = "#f0fdf4"
QUEUE_STROKE = "#166534"
EXTERNAL_STROKE = "#6b7280"
GROUP_STROKE = "#9ca3af"
UNGROUPED_COLUMN_LABEL = "Other"

CARD_WIDTH = 300
CARD_PADDING = 14
MANY_OFFSET = 12
MANY_STEP = 6
TITLE_LINE_HEIGHT = 20
DETAIL_LINE_HEIGHT = 17
TITLE_FONT_SIZE = 13.5
DETAIL_FONT_SIZE = 11.5
STEREOTYPE_FONT_SIZE = 10
TEXT_WIDTH_EM = 0.62
DETAIL_PREFIX = "•  "
DETAIL_CONTINUATION = "    "
DETAIL_INDENT_CHARS = 4
DETAIL_TOP_GAP = 5
COMPONENT_DIVIDER_OFFSET = 1
COMPONENT_DIVIDER_GAP = 5
EXTERNAL_HEADER_HEIGHT = 14
STORE_ELLIPSE_RY = 10
QUEUE_NOTCH = 18
CARD_GAP = 16
SECTION_INSET = 12
SECTION_HEADER_HEIGHT = 28
SECTION_GAP = 18
COLUMN_CAPTION_HEIGHT = 26
TOP_MARGIN = 16
OUTER_MARGIN = 24
LANE_MARGIN = 14
LANE_STEP = 10
EDGE_LABEL_FONT_SIZE = 11
EDGE_LABEL_PADDING = 4
MIN_GUTTER_WIDTH = 90
CHANNEL_OFFSET = 34
CHANNEL_STEP = 14
LEGEND_FONT_SIZE = 11
LEGEND_HEIGHT = 56
LEGEND_SHAPE_WIDTH = 20
LEGEND_SHAPE_GAP = 7
LEGEND_ITEM_GAP = 20
LEGEND_QUEUE_WIDTH = 18
LEGEND_QUEUE_HEIGHT = 12
LEGEND_QUEUE_NOTCH = 4


@dataclass
class _Card:
    node: Node
    x: float
    y: float
    height: float
    outgoing: list[Edge] = field(default_factory=list)
    incoming: list[Edge] = field(default_factory=list)


@dataclass
class _Section:
    group: Group
    x: float
    y: float
    width: float
    height: float


@dataclass
class _Column:
    label: str
    source: Source | None
    width: float
    sections: list[_Section] = field(default_factory=list)
    cards: list[_Card] = field(default_factory=list)
    height: float = 0.0
    x: float = 0.0


def _escape(value: str) -> str:
    return html.escape(value, quote=True)


def _text_width(value: str, font_size: float) -> float:
    return len(value) * font_size * TEXT_WIDTH_EM


def _chevron_points(
    x: float,
    y: float,
    width: float,
    height: float,
    notch: float,
) -> str:
    midpoint = y + height / 2
    return " ".join(
        f"{point_x:.1f},{point_y:.1f}"
        for point_x, point_y in (
            (x, y),
            (x + width - notch, y),
            (x + width, midpoint),
            (x + width - notch, y + height),
            (x, y + height),
            (x + notch, midpoint),
        )
    )


def _wrap_long_word(word: str, max_width: float, font_size: float) -> list[str]:
    boundaries = [
        index
        for index in range(1, len(word))
        if word[index] in "/.-"
        or (word[index - 1].islower() and word[index].isupper())
        or (
            word[index - 1].isupper()
            and word[index].isupper()
            and index + 1 < len(word)
            and word[index + 1].islower()
        )
    ]
    pieces = []
    start = 0
    for boundary in boundaries:
        pieces.append(word[start:boundary])
        start = boundary
    pieces.append(word[start:])

    chars_per_line = max(1, math.floor(max_width / (font_size * TEXT_WIDTH_EM)))
    lines = []
    current = ""
    for piece in pieces:
        if _text_width(piece, font_size) > max_width:
            if current:
                lines.append(current)
                current = ""
            chunks = textwrap.wrap(
                piece,
                chars_per_line,
                break_long_words=True,
                break_on_hyphens=False,
            )
            lines.extend(chunks[:-1])
            current = chunks[-1]
        elif current and _text_width(current + piece, font_size) > max_width:
            lines.append(current)
            current = piece
        else:
            current += piece
    if current:
        lines.append(current)
    return lines or [word]


def _wrap(
    text: str,
    *,
    max_width: float,
    font_size: float,
    prefix_chars: int = 0,
) -> list[str]:
    available_width = max_width - prefix_chars * font_size * TEXT_WIDTH_EM
    if not text.strip():
        return [text]

    lines = []
    current = ""
    for word in text.split():
        if _text_width(word, font_size) <= available_width:
            candidate = f"{current} {word}".strip()
            if _text_width(candidate, font_size) <= available_width:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = word
            continue

        if current:
            lines.append(current)
            current = ""
        wrapped = _wrap_long_word(word, available_width, font_size)
        lines.extend(wrapped[:-1])
        current = wrapped[-1]

    if current:
        lines.append(current)
    return lines or [text]


def _title_lines(node: Node) -> list[str]:
    suffix = "  ›" if node.link is not None else ""
    max_width = CARD_WIDTH - (MANY_OFFSET if node.many else 0) - 2 * CARD_PADDING
    if node.kind == "queue":
        max_width -= 2 * QUEUE_NOTCH
    return _wrap(
        node.label + suffix,
        max_width=max_width,
        font_size=TITLE_FONT_SIZE,
    )


def _detail_lines(node: Node, detail_text: str) -> list[str]:
    max_width = CARD_WIDTH - (MANY_OFFSET if node.many else 0) - 2 * CARD_PADDING
    if node.kind == "queue":
        max_width -= 2 * QUEUE_NOTCH
    return _wrap(
        detail_text,
        max_width=max_width,
        font_size=DETAIL_FONT_SIZE,
        prefix_chars=DETAIL_INDENT_CHARS,
    )


def _card_height(node: Node) -> float:
    title_height = len(_title_lines(node)) * TITLE_LINE_HEIGHT
    detail_lines = sum(len(_detail_lines(node, detail.text)) for detail in node.details)
    gaps = max(len(node.details) - 1, 0) * 3
    height = 2 * CARD_PADDING + title_height
    if node.kind == "component":
        height += COMPONENT_DIVIDER_OFFSET
    if node.details:
        height += (
            DETAIL_TOP_GAP + DETAIL_FONT_SIZE + detail_lines * DETAIL_LINE_HEIGHT + gaps
        )
    if node.kind == "external":
        height += EXTERNAL_HEADER_HEIGHT
    elif node.kind == "store":
        height += 2 * STORE_ELLIPSE_RY
    if node.many:
        height += MANY_OFFSET
    return height


def _group_depth(group_id: str, children: Mapping[str, list[Group]]) -> int:
    child_depths = [_group_depth(child.id, children) for child in children[group_id]]
    return 1 + max(child_depths, default=0)


def _place_group_contents(
    column: _Column,
    group_id: str,
    *,
    x: float,
    y: float,
    width: float,
    children: Mapping[str, list[Group]],
    members: Mapping[str | None, list[Node]],
) -> float:
    for node in members[group_id]:
        height = _card_height(node)
        column.cards.append(
            _Card(node=node, x=x + (width - CARD_WIDTH) / 2, y=y, height=height)
        )
        y += height + CARD_GAP
    for child in children[group_id]:
        top = y
        bottom = _place_group_contents(
            column,
            child.id,
            x=x + SECTION_INSET,
            y=y + SECTION_HEADER_HEIGHT,
            width=width - 2 * SECTION_INSET,
            children=children,
            members=members,
        )
        section_height = bottom - top - CARD_GAP + SECTION_INSET
        column.sections.append(
            _Section(group=child, x=x, y=top, width=width, height=section_height)
        )
        y = top + section_height + SECTION_GAP
    return y


def _columns(block: BlockDiagram) -> list[_Column]:
    children: dict[str, list[Group]] = {group.id: [] for group in block.groups}
    members: dict[str | None, list[Node]] = {group.id: [] for group in block.groups}
    members[None] = []
    for group in block.groups:
        if group.parent is not None:
            children[group.parent].append(group)
    for node in block.nodes:
        members[node.group].append(node)

    columns = []
    top = TOP_MARGIN + COLUMN_CAPTION_HEIGHT
    for group in block.groups:
        if group.parent is not None:
            continue
        nested_depth = _group_depth(group.id, children) - 1
        width = CARD_WIDTH + 2 * SECTION_INSET * nested_depth
        column = _Column(label=group.label, source=group.source, width=width)
        bottom = _place_group_contents(
            column,
            group.id,
            x=0,
            y=top,
            width=width,
            children=children,
            members=members,
        )
        column.height = bottom - top
        columns.append(column)
    if members[None]:
        column = _Column(label=UNGROUPED_COLUMN_LABEL, source=None, width=CARD_WIDTH)
        y = top
        for node in members[None]:
            height = _card_height(node)
            column.cards.append(_Card(node=node, x=0, y=y, height=height))
            y += height + CARD_GAP
        column.height = y - top
        columns.append(column)
    return columns


def _shift_column(column: _Column, dx: float) -> None:
    column.x = dx
    for card in column.cards:
        card.x += dx
    for section in column.sections:
        section.x += dx


def _label_width(label: str) -> float:
    return _text_width(label, EDGE_LABEL_FONT_SIZE) + 2 * EDGE_LABEL_PADDING


@dataclass(frozen=True)
class _Route:
    edge: Edge
    points: list[tuple[float, float]]
    label_x: float
    label_y: float
    via_channel: bool


def _port_y(card: _Card, index: int, count: int) -> float:
    offset = MANY_OFFSET if card.node.many else 0
    content_height = card.height - offset
    return card.y + offset + content_height * (index + 1) / (count + 1)


def _front_geometry(card: _Card) -> tuple[float, float, float, float]:
    offset = MANY_OFFSET if card.node.many else 0
    return (
        card.x + offset,
        card.y + offset,
        CARD_WIDTH - offset,
        card.height - offset,
    )


def _endpoint_mid_y(
    endpoint_id: str,
    cards: dict[str, _Card],
    group_bounds: dict[str, tuple[float, float, float, float]],
) -> float:
    if endpoint_id in cards:
        card = cards[endpoint_id]
        _, y, _, height = _front_geometry(card)
        return y + height / 2
    _, y, _, height = group_bounds[endpoint_id]
    return y + height / 2


def _node_shape_svg(
    node: Node,
    x: float,
    y: float,
    width: float,
    height: float,
) -> str:
    if node.kind == "store":
        rx = width / 2
        ry = STORE_ELLIPSE_RY
        return (
            f'<path class="store-shape" d="M{x:.1f} {y + ry:.1f} '
            f"A{rx:.1f} {ry} 0 0 1 {x + width:.1f} {y + ry:.1f} "
            f"L{x + width:.1f} {y + height - ry:.1f} "
            f'A{rx:.1f} {ry} 0 0 1 {x:.1f} {y + height - ry:.1f} Z" '
            f'fill="{STORE_FILL}" stroke="{STORE_STROKE}" stroke-width="1.5"/>'
            f'<path class="store-lid-front" d="M{x:.1f} {y + ry:.1f} '
            f'A{rx:.1f} {ry} 0 0 0 {x + width:.1f} {y + ry:.1f}" '
            f'fill="none" stroke="{STORE_STROKE}" stroke-width="1.5"/>'
        )
    if node.kind == "queue":
        return (
            f'<polygon class="queue-shape" '
            f'points="{_chevron_points(x, y, width, height, QUEUE_NOTCH)}" '
            f'fill="{QUEUE_FILL}" stroke="{QUEUE_STROKE}" stroke-width="1.5"/>'
        )
    stroke = EXTERNAL_STROKE if node.kind == "external" else COMPONENT_STROKE
    dash = ' stroke-dasharray="6 4"' if node.kind == "external" else ""
    return (
        f'<rect class="{node.kind}-shape" x="{x:.1f}" y="{y:.1f}" '
        f'width="{width:.1f}" height="{height:.1f}" rx="4" fill="#ffffff" '
        f'stroke="{stroke}" stroke-width="1.5"{dash}/>'
    )


def _layout(
    diagram: BlockDiagram,
) -> tuple[list[_Column], dict[str, _Card], list[_Route], float, float]:
    columns = _columns(diagram)
    cards = {card.node.id: card for column in columns for card in column.cards}
    column_of = {
        card.node.id: index
        for index, column in enumerate(columns)
        for card in column.cards
    }
    groups_by_id = {group.id: group for group in diagram.groups}
    top_level_groups = [group for group in diagram.groups if group.parent is None]
    top_level_column = {group.id: index for index, group in enumerate(top_level_groups)}
    group_column_of = {}
    for group in diagram.groups:
        current = group
        while current.parent is not None:
            current = groups_by_id[current.parent]
        group_column_of[group.id] = top_level_column[current.id]
    endpoint_column = {**column_of, **group_column_of}

    edges = diagram.edges
    for edge in edges:
        if edge.src in cards:
            cards[edge.src].outgoing.append(edge)
        if edge.dst in cards:
            cards[edge.dst].incoming.append(edge)

    gutter_count = len(columns) + 1
    lanes: list[list[Edge]] = [[] for _ in range(gutter_count)]
    entry_labels: list[float] = [0.0] * gutter_count
    for edge in edges:
        exit_gutter = endpoint_column[edge.src] + 1
        entry_gutter = endpoint_column[edge.dst]
        lanes[exit_gutter].append(edge)
        if exit_gutter != entry_gutter:
            lanes[entry_gutter].append(edge)
        entry_labels[entry_gutter] = max(
            entry_labels[entry_gutter], _label_width(edge.label)
        )

    gutter_widths = []
    for index in range(gutter_count):
        lane_space = LANE_MARGIN + len(lanes[index]) * LANE_STEP if lanes[index] else 0
        width = lane_space + entry_labels[index]
        is_margin = index in (0, gutter_count - 1)
        if width == 0:
            gutter_widths.append(OUTER_MARGIN if is_margin else MIN_GUTTER_WIDTH)
        else:
            gutter_widths.append(max(width + EDGE_LABEL_PADDING, MIN_GUTTER_WIDTH))

    x = gutter_widths[0]
    gutter_lefts = [0.0]
    for index, column in enumerate(columns):
        _shift_column(column, x)
        x += column.width
        gutter_lefts.append(x)
        x += gutter_widths[index + 1]
    width = x

    columns_bottom = max(
        (TOP_MARGIN + COLUMN_CAPTION_HEIGHT + column.height for column in columns),
        default=0,
    )
    channel_top = columns_bottom + CHANNEL_OFFSET
    group_bounds = {
        group.id: (
            columns[index].x,
            TOP_MARGIN + COLUMN_CAPTION_HEIGHT,
            columns[index].width,
            columns[index].height,
        )
        for index, group in enumerate(top_level_groups)
    }
    group_bounds.update(
        {
            section.group.id: (
                section.x,
                section.y,
                section.width,
                section.height,
            )
            for column in columns
            for section in column.sections
        }
    )

    exit_y: dict[int, float] = {}
    entry_y: dict[int, float] = {}
    for card in cards.values():
        outgoing = sorted(
            card.outgoing,
            key=lambda edge: _endpoint_mid_y(edge.dst, cards, group_bounds),
        )
        for index, edge in enumerate(outgoing):
            exit_y[id(edge)] = _port_y(card, index, len(outgoing))
        incoming = sorted(
            card.incoming,
            key=lambda edge: _endpoint_mid_y(edge.src, cards, group_bounds),
        )
        for index, edge in enumerate(incoming):
            entry_y[id(edge)] = _port_y(card, index, len(incoming))

    starts: dict[int, tuple[float, float]] = {}
    ends: dict[int, tuple[float, float]] = {}
    for edge in edges:
        if edge.src in cards:
            source_card = cards[edge.src]
            source_x, source_y, source_width, source_height = _front_geometry(
                source_card
            )
            starts[id(edge)] = (
                source_x + source_width,
                source_y + source_height / 2
                if source_card.node.kind == "queue"
                else exit_y[id(edge)],
            )
        else:
            x, y, width, height = group_bounds[edge.src]
            starts[id(edge)] = (x + width, y + height / 2)

        if edge.dst in cards:
            target_card = cards[edge.dst]
            target_x, target_y, _, target_height = _front_geometry(target_card)
            ends[id(edge)] = (
                target_x + (QUEUE_NOTCH if target_card.node.kind == "queue" else 0),
                target_y + target_height / 2
                if target_card.node.kind == "queue"
                else entry_y[id(edge)],
            )
        else:
            x, y, _, height = group_bounds[edge.dst]
            ends[id(edge)] = (x, y + height / 2)

    lane_x: dict[tuple[int, int], float] = {}
    for gutter, gutter_edges in enumerate(lanes):
        ordered = sorted(gutter_edges, key=lambda edge: starts[id(edge)][1])
        for index, edge in enumerate(ordered):
            lane_x[(gutter, id(edge))] = (
                gutter_lefts[gutter] + LANE_MARGIN + index * LANE_STEP
            )

    corridor_y: dict[int, float] = {}
    chosen_corridors: list[tuple[int, int, float]] = []
    corridor_top = TOP_MARGIN + COLUMN_CAPTION_HEIGHT
    for edge in edges:
        exit_gutter = endpoint_column[edge.src] + 1
        entry_gutter = endpoint_column[edge.dst]
        if exit_gutter >= entry_gutter:
            continue

        crossed_columns = columns[exit_gutter:entry_gutter]
        blocked_bands = [
            (card.y - LANE_STEP, card.y + card.height + LANE_STEP)
            for column in crossed_columns
            for card in column.cards
        ]
        blocked_bands.extend(
            (section.y - LANE_STEP, section.y + section.height + LANE_STEP)
            for column in crossed_columns
            for section in column.sections
        )
        start_y = starts[id(edge)][1]
        end_y = ends[id(edge)][1]
        midpoint = (start_y + end_y) / 2
        band_edges = [
            candidate
            for band_top, band_bottom in blocked_bands
            for candidate in (band_top - 1, band_bottom + 1)
        ]
        candidates = [
            start_y,
            end_y,
            *sorted(band_edges, key=lambda candidate: abs(candidate - midpoint)),
        ]
        for candidate in candidates:
            if not corridor_top <= candidate <= columns_bottom:
                continue
            if any(
                band_top <= candidate <= band_bottom
                for band_top, band_bottom in blocked_bands
            ):
                continue
            if any(
                max(exit_gutter, other_exit) <= min(entry_gutter, other_entry)
                and abs(candidate - other_y) < LANE_STEP
                for other_exit, other_entry, other_y in chosen_corridors
            ):
                continue
            corridor_y[id(edge)] = candidate
            chosen_corridors.append((exit_gutter, entry_gutter, candidate))
            break

    channel_edges = [
        edge
        for edge in edges
        if endpoint_column[edge.src] + 1 != endpoint_column[edge.dst]
        and id(edge) not in corridor_y
    ]
    channel_y = {
        id(edge): channel_top + index * CHANNEL_STEP
        for index, edge in enumerate(channel_edges)
    }

    routes = []
    for edge in edges:
        exit_gutter = endpoint_column[edge.src] + 1
        entry_gutter = endpoint_column[edge.dst]
        start = starts[id(edge)]
        end = ends[id(edge)]
        first_lane = lane_x[(exit_gutter, id(edge))]
        via_channel = False
        if exit_gutter == entry_gutter:
            points = [start, (first_lane, start[1]), (first_lane, end[1]), end]
        elif exit_gutter < entry_gutter and id(edge) in corridor_y:
            last_lane = lane_x[(entry_gutter, id(edge))]
            lane_y = corridor_y[id(edge)]
            points = [
                start,
                (first_lane, start[1]),
                (first_lane, lane_y),
                (last_lane, lane_y),
                (last_lane, end[1]),
                end,
            ]
        else:
            last_lane = lane_x[(entry_gutter, id(edge))]
            bus = channel_y[id(edge)]
            via_channel = True
            points = [
                start,
                (first_lane, start[1]),
                (first_lane, bus),
                (last_lane, bus),
                (last_lane, end[1]),
                end,
            ]
        routes.append(
            _Route(
                edge=edge,
                points=points,
                label_x=end[0] - 8,
                label_y=end[1] - 5,
                via_channel=via_channel,
            )
        )

    channel_bottom = channel_top + max(len(channel_edges) - 1, 0) * CHANNEL_STEP
    height = (
        (channel_bottom if channel_edges else columns_bottom)
        + CHANNEL_OFFSET
        + LEGEND_HEIGHT
    )
    return columns, cards, routes, width, height


MAX_LAYOUT_PROBLEMS = 5


def block_layout_problems(diagram: BlockDiagram) -> list[str]:
    _, _, routes, _, _ = _layout(diagram)
    channel_edges = {id(route.edge) for route in routes if route.via_channel}
    top_level_groups = [group for group in diagram.groups if group.parent is None]
    groups_by_id = {group.id: group for group in diagram.groups}
    top_level_column = {group.id: index for index, group in enumerate(top_level_groups)}
    group_column = {}
    for group in diagram.groups:
        current = group
        while current.parent is not None:
            current = groups_by_id[current.parent]
        group_column[group.id] = top_level_column[current.id]

    node_column = {
        node.id: (
            group_column[node.group]
            if node.group is not None
            else len(top_level_groups)
        )
        for node in diagram.nodes
    }
    endpoint_column = {**node_column, **group_column}
    problems = []
    for edge in diagram.edges:
        if id(edge) not in channel_edges:
            continue
        source_column = endpoint_column[edge.src]
        target_column = endpoint_column[edge.dst]
        problems.append(
            f"{edge.src} -> {edge.dst} (columns {source_column} -> "
            f"{target_column}; expected {source_column} -> "
            f"{source_column + 1})"
        )
    return problems


def _link(href: str, content: str) -> str:
    return f'<a href="{_escape(href)}" target="_top">{content}</a>'


def _text(
    x: float,
    y: float,
    value: str,
    *,
    size: float,
    weight: int = 400,
    color: str = TEXT_COLOR,
    anchor: str = "start",
    extra: str = "",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT_FAMILY}" font-size="{size}" '
        f'font-weight="{weight}" fill="{color}" text-anchor="{anchor}"{extra}>'
        f"{_escape(value)}</text>"
    )


def _card_svg(card: _Card, *, page: Page, repo_url: str) -> str:
    node = card.node
    x, y, width, height = _front_geometry(card)
    parts = []
    if node.many:
        for offset in (0, MANY_STEP):
            parts.append(
                _node_shape_svg(
                    node,
                    card.x + offset,
                    card.y + offset,
                    CARD_WIDTH - offset,
                    height,
                )
            )
    parts.append(_node_shape_svg(node, x, y, width, height))
    if node.link is not None:
        title_href = page_href(page.id, node.link)
    else:
        title_href = source_url(node.source, repo_url=repo_url, commit=page.commit)
    title_lines = _title_lines(node)
    title_offset = (EXTERNAL_HEADER_HEIGHT if node.kind == "external" else 0) + (
        2 * STORE_ELLIPSE_RY if node.kind == "store" else 0
    )
    title_y = y + title_offset + CARD_PADDING + 13
    title = "".join(
        _text(
            x + width / 2,
            title_y + index * TITLE_LINE_HEIGHT,
            line,
            size=TITLE_FONT_SIZE,
            weight=700,
            color=TEXT_COLOR,
            anchor="middle",
        )
        for index, line in enumerate(title_lines)
    )
    parts.append(f'<g class="card-title">{_link(title_href, title)}</g>')
    if node.kind == "external":
        parts.append(
            _text(
                x + width / 2,
                y + CARD_PADDING + STEREOTYPE_FONT_SIZE,
                "«external»",
                size=STEREOTYPE_FONT_SIZE,
                color=MUTED_COLOR,
                anchor="middle",
                extra=' font-style="italic"',
            )
        )
    if node.kind == "component":
        divider_y = y + CARD_PADDING + len(title_lines) * TITLE_LINE_HEIGHT
        parts.append(
            f'<line class="component-divider" x1="{x + CARD_PADDING:.1f}" '
            f'y1="{divider_y:.1f}" x2="{x + width - CARD_PADDING:.1f}" '
            f'y2="{divider_y:.1f}" stroke="{DIVIDER_COLOR}" stroke-width="1"/>'
        )
        details_start_y = divider_y + COMPONENT_DIVIDER_OFFSET + COMPONENT_DIVIDER_GAP
    else:
        details_start_y = (
            y + title_offset + CARD_PADDING + len(title_lines) * TITLE_LINE_HEIGHT
        )
    details_start_y += DETAIL_FONT_SIZE
    if node.kind != "component":
        details_start_y += DETAIL_TOP_GAP
    left = x + CARD_PADDING + (QUEUE_NOTCH if node.kind == "queue" else 0)
    detail_y = details_start_y
    for detail in node.details:
        lines = _detail_lines(node, detail.text)
        texts = []
        for line_index, line in enumerate(lines):
            prefix = DETAIL_PREFIX if line_index == 0 else DETAIL_CONTINUATION
            texts.append(_text(left, detail_y, prefix + line, size=DETAIL_FONT_SIZE))
            detail_y += DETAIL_LINE_HEIGHT
        href = source_url(detail.sources[0], repo_url=repo_url, commit=page.commit)
        parts.append(f'<g class="card-detail">{_link(href, "".join(texts))}</g>')
        detail_y += 3
    return f'<g class="block-node block-kind-{_escape(node.kind)}">{"".join(parts)}</g>'


def _section_svg(section: _Section, *, page: Page, repo_url: str) -> str:
    href = source_url(section.group.source, repo_url=repo_url, commit=page.commit)
    tab_width = _text_width(section.group.label, 11.5) + 18
    tab_x = section.x + 10
    tab_y = section.y + 4
    label = _text(tab_x + 9, tab_y + 14, section.group.label, size=11.5, weight=700)
    return (
        f'<rect class="group-boundary" x="{section.x:.1f}" y="{section.y:.1f}" '
        f'width="{section.width:.1f}" height="{section.height:.1f}" rx="6" '
        f'fill="none" stroke="{GROUP_STROKE}" stroke-width="1.25" '
        'stroke-dasharray="6 4"/>'
        f'<rect class="group-tab" x="{tab_x:.1f}" y="{tab_y:.1f}" '
        f'width="{tab_width:.1f}" height="19" rx="2" fill="#ffffff" '
        f'stroke="{GROUP_STROKE}" stroke-width="1.25" stroke-dasharray="6 4"/>'
        f'<g class="section-label">{_link(href, label)}</g>'
    )


def _column_header_svg(column: _Column, *, page: Page, repo_url: str) -> str:
    label = _text(
        column.x + 8,
        TOP_MARGIN + 11,
        column.label.upper(),
        size=11,
        weight=600,
        color=MUTED_COLOR,
    )
    if column.source is None:
        return label
    href = source_url(column.source, repo_url=repo_url, commit=page.commit)
    return f'<g class="column-label">{_link(href, label)}</g>'


def _route_path_svg(route: _Route) -> str:
    path = " ".join(
        f"{'M' if index == 0 else 'L'}{x:.1f} {y:.1f}"
        for index, (x, y) in enumerate(route.points)
    )
    dash = ' stroke-dasharray="4 3"' if route.via_channel else ""
    return (
        f'<path class="block-edge" d="{path}" fill="none" stroke="{ARROW_COLOR}" '
        f'stroke-width="1.5" stroke-linejoin="round"{dash} '
        'marker-end="url(#block-arrow)"/>'
    )


def _route_label_svg(route: _Route, *, page: Page, repo_url: str) -> str:
    href = source_url(route.edge.source, repo_url=repo_url, commit=page.commit)
    text_width = _text_width(route.edge.label, EDGE_LABEL_FONT_SIZE)
    background_x = route.label_x - text_width - EDGE_LABEL_PADDING
    background_y = route.label_y - EDGE_LABEL_FONT_SIZE - 2
    background_width = text_width + 2 * EDGE_LABEL_PADDING
    background_height = EDGE_LABEL_FONT_SIZE + 5
    label = _text(
        route.label_x,
        route.label_y,
        route.edge.label,
        size=EDGE_LABEL_FONT_SIZE,
        color=TEXT_COLOR,
        anchor="end",
        extra=' class="edge-label"',
    )
    background = (
        f'<rect class="edge-label-background" x="{background_x:.1f}" '
        f'y="{background_y:.1f}" width="{background_width:.1f}" '
        f'height="{background_height:.1f}" rx="3" fill="#ffffff"/>'
    )
    return f'<g class="edge-link">{_link(href, background + label)}</g>'


def _legend_entries() -> list[tuple[str, str]]:
    return [
        ("component", "service/component"),
        ("store", "database/store"),
        ("queue", "queue/stream"),
        ("external", "external system"),
        ("group", "group"),
        ("many", "many instances"),
    ]


def _legend_width() -> float:
    return 2 * OUTER_MARGIN + sum(
        LEGEND_SHAPE_WIDTH
        + LEGEND_SHAPE_GAP
        + _text_width(label, LEGEND_FONT_SIZE)
        + LEGEND_ITEM_GAP
        for _, label in _legend_entries()
    )


def _legend_shape_svg(kind: str, x: float, y: float) -> str:
    if kind == "component":
        return (
            f'<rect x="{x:.1f}" y="{y - 7:.1f}" width="18" height="14" rx="4" '
            f'fill="#ffffff" stroke="{COMPONENT_STROKE}" stroke-width="1.5"/>'
        )
    if kind == "store":
        return (
            f'<path d="M{x:.1f} {y - 4:.1f} A9 3 0 0 0 {x + 18:.1f} {y - 4:.1f} '
            f'L{x + 18:.1f} {y + 4:.1f} A9 3 0 0 1 {x:.1f} {y + 4:.1f} Z" '
            f'fill="{STORE_FILL}" stroke="{STORE_STROKE}" stroke-width="1.2"/>'
        )
    if kind == "queue":
        points = _chevron_points(
            x,
            y - LEGEND_QUEUE_HEIGHT / 2,
            LEGEND_QUEUE_WIDTH,
            LEGEND_QUEUE_HEIGHT,
            LEGEND_QUEUE_NOTCH,
        )
        return (
            f'<polygon class="queue-legend-shape" '
            f'points="{points}" '
            f'fill="{QUEUE_FILL}" stroke="{QUEUE_STROKE}" stroke-width="1.2"/>'
        )
    if kind == "external":
        return (
            f'<rect x="{x:.1f}" y="{y - 7:.1f}" width="18" height="14" rx="2" '
            f'fill="#ffffff" stroke="{EXTERNAL_STROKE}" stroke-width="1.2" '
            'stroke-dasharray="4 3"/>'
        )
    if kind == "many":
        return "".join(
            f'<rect x="{x + offset:.1f}" y="{y - 6 + offset:.1f}" '
            'width="12" height="10" rx="2" fill="#ffffff" '
            f'stroke="{COMPONENT_STROKE}" stroke-width="1.2"/>'
            for offset in (0, 3, 6)
        )
    return (
        f'<rect x="{x:.1f}" y="{y - 7:.1f}" width="18" height="14" rx="4" '
        f'fill="none" stroke="{GROUP_STROKE}" stroke-width="1.2" '
        'stroke-dasharray="4 3"/>'
    )


def _legend_svg(height: float) -> str:
    y = height - LEGEND_HEIGHT / 2
    x = OUTER_MARGIN
    parts = []
    for kind, label in _legend_entries():
        parts.append(_legend_shape_svg(kind, x, y))
        parts.append(
            _text(
                x + LEGEND_SHAPE_WIDTH + LEGEND_SHAPE_GAP,
                y + 4,
                label,
                size=LEGEND_FONT_SIZE,
                color=MUTED_COLOR,
            )
        )
        x += (
            LEGEND_SHAPE_WIDTH
            + LEGEND_SHAPE_GAP
            + _text_width(label, LEGEND_FONT_SIZE)
            + LEGEND_ITEM_GAP
        )
    return "".join(parts)


def block_svg(
    diagram: BlockDiagram,
    *,
    page: Page,
    repo_url: str,
) -> str:
    columns, cards, routes, width, height = _layout(diagram)
    width = max(width, _legend_width())
    body = [
        f'<rect class="block-canvas" width="{width:.0f}" '
        f'height="{height:.0f}" fill="#ffffff"/>'
    ]
    for column in columns:
        for section in sorted(column.sections, key=lambda section: section.y):
            body.append(_section_svg(section, page=page, repo_url=repo_url))
    for column in columns:
        body.append(_column_header_svg(column, page=page, repo_url=repo_url))
    body.extend(_route_path_svg(route) for route in routes)
    for card in cards.values():
        body.append(_card_svg(card, page=page, repo_url=repo_url))
    body.extend(
        _route_label_svg(route, page=page, repo_url=repo_url) for route in routes
    )
    body.append(_legend_svg(height))
    return (
        f'<svg class="block-svg" xmlns="http://www.w3.org/2000/svg" '
        f'width="{width:.0f}" height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}" '
        'role="img">'
        '<defs><marker id="block-arrow" viewBox="0 0 8 8" refX="7" refY="4" '
        'markerWidth="8" markerHeight="8" markerUnits="userSpaceOnUse" orient="auto">'
        f'<path d="M0 0 L8 4 L0 8Z" fill="{ARROW_COLOR}"/></marker></defs>'
        f"{''.join(body)}</svg>"
    )
