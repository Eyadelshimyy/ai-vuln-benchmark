from __future__ import annotations

import html
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from rox_dox.links import page_href, source_url
from rox_dox.model import Page, Relation, SchemaDomain, Source
from rox_dox.plantuml import DiagramError
from rox_dox.schema import Column, Table

CARD_WIDTH = 300
CARD_WIDTH_CAP = 520
CARD_GAP = 36
COLUMN_GAP = 160
LEFT_MARGIN = 24
ROW_HEIGHT = 18
HEADER_VERTICAL_PADDING = 6
HEADER_LINE_HEIGHT = 16
HEADER_BASELINE_OFFSET = 13
HEADER_COUNT_GAP = 12
CARD_PADDING = 12
DOMAIN_KEY_GAP = 24
MONOSPACE_CHAR_WIDTH = 0.62
PK_COLOR = "#b45309"
FK_COLOR = "#2563eb"
BOILERPLATE_COLOR = "#9ca3af"
SYMBOLIC_COLOR = "#9ca3af"
BLOB_COLOR = "#d97706"
HEADER_COLOR = "#1f2937"
STORE_HEADER_COLOR = "#d97706"
STORE_BACKGROUND = "#fffbeb"
ROW_COLOR = "#f1f5f9"
CARD_BORDER = "#cbd5e1"
CARD_BACKGROUND = "#ffffff"
MONOSPACE = "Menlo, Consolas, monospace"
BOILERPLATE_COLUMNS = {
    "rox_org_id",
    "rox_user_id",
    "created_on",
    "created_at",
    "last_modified_on",
    "last_modified",
    "updated_at",
    "updated_on",
}


@dataclass
class _Item:
    name: str
    kind: str
    width: int
    height: int
    header_height: int
    header_lines: list[str]
    column: int = 0
    x: int = 0
    y: int = 0


@dataclass(frozen=True)
class _Endpoint:
    raw: str
    kind: str
    name: str
    field: str | None


@dataclass(frozen=True)
class _Edge:
    src: _Endpoint
    dst: _Endpoint
    label: str
    kind: str
    source: Source


def _escape(value: object, *, quote: bool = False) -> str:
    return html.escape(str(value), quote=quote)


def _text_width(value: str, font_size: float = 11.5) -> float:
    return len(value) * font_size * MONOSPACE_CHAR_WIDTH


def _wrap_text(value: str, max_width: float, font_size: float = 11.5) -> list[str]:
    max_chars = max(1, int(max_width / (font_size * MONOSPACE_CHAR_WIDTH)))
    words = value.split()
    if not words:
        return [""]
    lines: list[str] = []
    current = ""
    for word in words:
        if len(word) > max_chars:
            if current:
                lines.append(current)
                current = ""
            lines.extend(
                word[index : index + max_chars]
                for index in range(0, len(word), max_chars)
            )
            continue
        if not current:
            current = word
            continue
        candidate = f"{current} {word}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _truncate(value: str, max_width: float, font_size: float = 11.5) -> str:
    max_chars = max(1, int(max_width / (font_size * MONOSPACE_CHAR_WIDTH)))
    if len(value) <= max_chars:
        return value
    if max_chars <= 1:
        return "…"
    return f"{value[: max_chars - 1]}…"


def _source_href(
    source: Source,
    *,
    page: Page,
    repo_url: str,
) -> str:
    return _escape(
        source_url(source, repo_url=repo_url, commit=page.commit),
        quote=True,
    )


def _error(page: Page, relation: Relation, message: str) -> DiagramError:
    return DiagramError(
        f"page '{page.id}' relation '{relation.src} -> {relation.dst}': {message}"
    )


def _endpoint(
    value: str,
    *,
    relation: Relation,
    page: Page,
    tables: Mapping[str, Table],
    sql_names: set[str],
    store_names: set[str],
) -> _Endpoint:
    if value in store_names:
        return _Endpoint(raw=value, kind="nosql", name=value, field=None)

    if value in sql_names:
        if value not in tables:
            raise _error(
                page,
                relation,
                f"SQL table '{value}' is missing",
            )
        return _Endpoint(raw=value, kind="sql", name=value, field=None)

    store_name, separator, field = value.partition("::")
    if separator and store_name in store_names:
        return _Endpoint(raw=value, kind="nosql", name=store_name, field=field)

    if "." not in value:
        raise _error(
            page,
            relation,
            f"endpoint '{value}' is not a declared NoSQL store or SQL table.column",
        )
    table_name, column_name = value.rsplit(".", 1)
    if table_name not in sql_names:
        raise _error(
            page,
            relation,
            f"SQL table '{table_name}' for endpoint '{value}' is not listed",
        )
    table = tables.get(table_name)
    if table is None:
        raise _error(
            page,
            relation,
            f"SQL table '{table_name}' for endpoint '{value}' is missing",
        )
    if column_name not in {column.name for column in table.columns}:
        raise _error(page, relation, f"SQL column '{value}' is missing")
    return _Endpoint(
        raw=value,
        kind="sql",
        name=table_name,
        field=column_name,
    )


def _validate_relation(
    relation: Relation,
    *,
    page: Page,
    tables: Mapping[str, Table],
    sql_names: set[str],
    store_names: set[str],
    domain_by_table: Mapping[str, str],
) -> _Edge:
    src = _endpoint(
        relation.src,
        relation=relation,
        page=page,
        tables=tables,
        sql_names=sql_names,
        store_names=store_names,
    )
    dst = _endpoint(
        relation.dst,
        relation=relation,
        page=page,
        tables=tables,
        sql_names=sql_names,
        store_names=store_names,
    )
    if src.kind == "sql" and dst.kind == "sql":
        src_domain = domain_by_table.get(src.name)
        dst_domain = domain_by_table.get(dst.name)
        if src_domain is not None and src_domain == dst_domain:
            raise _error(
                page,
                relation,
                "intra-domain relation belongs on the domain page",
            )
    if relation.kind == "enforced":
        if src.kind != "sql" or dst.kind != "sql":
            raise _error(
                page,
                relation,
                "enforced relation must connect SQL tables or table columns",
            )
        if (src.field is None) != (dst.field is None):
            raise _error(
                page,
                relation,
                "enforced relation must connect two SQL tables or two columns",
            )
        if src.field is not None and dst.field is not None:
            source_column = next(
                column
                for column in tables[src.name].columns
                if column.name == src.field
            )
        else:
            source_column = None
        if source_column is not None and source_column.foreign_key != relation.dst:
            raise _error(
                page,
                relation,
                f"enforced source column '{relation.src}' declares foreign key "
                f"'{source_column.foreign_key}', not '{relation.dst}'",
            )
    return _Edge(
        src=src,
        dst=dst,
        label=relation.label,
        kind=relation.kind,
        source=relation.source,
    )


def _domain_key_tables(domains: Sequence[SchemaDomain]) -> set[str]:
    return {table_name for domain in domains for table_name in domain.key_tables}


def _declared_fk_edges(
    page: Page,
    *,
    tables: Mapping[str, Table],
    sql_names: set[str],
    authored_edges: Sequence[_Edge],
) -> list[_Edge]:
    authored_keys = {(edge.src.raw, edge.dst.raw, edge.kind) for edge in authored_edges}
    edges = list(authored_edges)
    for table_name in page.data.sql_tables:
        table = tables[table_name]
        for column in table.columns:
            if column.foreign_key is None:
                continue
            target_table = column.foreign_key.rsplit(".", 1)[0]
            if target_table not in sql_names:
                continue
            if target_table not in tables:
                raise DiagramError(
                    f"page '{page.id}' declared foreign key "
                    f"'{table_name}.{column.name} -> {column.foreign_key}' "
                    f"references missing table '{target_table}'"
                )
            target_column = column.foreign_key.rsplit(".", 1)[1]
            if target_column not in {
                target.name for target in tables[target_table].columns
            }:
                raise DiagramError(
                    f"page '{page.id}' declared foreign key "
                    f"'{table_name}.{column.name} -> {column.foreign_key}' "
                    f"references missing column '{target_column}'"
                )
            key = (f"{table_name}.{column.name}", column.foreign_key, "enforced")
            if key in authored_keys:
                continue
            edges.append(
                _Edge(
                    src=_Endpoint(
                        raw=key[0],
                        kind="sql",
                        name=table_name,
                        field=column.name,
                    ),
                    dst=_Endpoint(
                        raw=column.foreign_key,
                        kind="sql",
                        name=target_table,
                        field=target_column,
                    ),
                    label=column.name,
                    kind="enforced",
                    source=table.source,
                )
            )
    return edges


def _header_lines(
    value: str,
    *,
    width: int,
    reserved_width: float = 0,
) -> list[str]:
    return _wrap_text(
        value,
        max_width=width - 2 * CARD_PADDING - reserved_width,
        font_size=12.5,
    )


def _header_height(line_count: int) -> int:
    return 2 * HEADER_VERTICAL_PADDING + HEADER_LINE_HEIGHT * line_count


def _domain_item(
    domain: SchemaDomain,
    *,
    tables: Mapping[str, Table],
    roles: Mapping[tuple[str, str], set[str]],
) -> _Item:
    count = f"{len(domain.tables)} tables"
    header_width = (
        _text_width(domain.title, font_size=12.5)
        + HEADER_COUNT_GAP
        + _text_width(count, font_size=10.5)
        + 2 * CARD_PADDING
    )
    row_widths = []
    for table_name in domain.key_tables:
        table = tables[table_name]
        key_names = [
            column.name
            for column in table.columns
            if (table_name, column.name) in roles
        ]
        key_text = ", ".join(key_names)
        row_widths.append(
            _text_width(table_name)
            + DOMAIN_KEY_GAP
            + _text_width(key_text)
            + 2 * CARD_PADDING
        )
    width = min(
        CARD_WIDTH_CAP,
        max(CARD_WIDTH, math.ceil(max(header_width, *row_widths))),
    )
    reserved_width = HEADER_COUNT_GAP + _text_width(count, font_size=10.5)
    header_lines = _header_lines(
        domain.title,
        width=width,
        reserved_width=reserved_width,
    )
    header_height = _header_height(len(header_lines))
    note_lines = sum(
        len(_wrap_text(note.text, width - 2 * CARD_PADDING)) for note in domain.notes
    )
    height = (
        header_height
        + len(domain.key_tables) * ROW_HEIGHT
        + note_lines * ROW_HEIGHT
        + ROW_HEIGHT
    )
    return _Item(
        name=domain.id,
        kind="domain",
        width=width,
        height=height,
        header_height=header_height,
        header_lines=header_lines,
    )


def _table_item(table: Table) -> _Item:
    width = max(
        CARD_WIDTH,
        int(
            max(
                (
                    _text_width(f"{column.name} {column.type}") + 2 * CARD_PADDING
                    for column in table.columns
                ),
                default=0,
            )
            + 4
        ),
    )
    header_lines = _header_lines(table.name, width=width)
    header_height = _header_height(len(header_lines))
    height = header_height + len(table.columns) * ROW_HEIGHT
    return _Item(
        name=table.name,
        kind="table",
        width=width,
        height=height,
        header_height=header_height,
        header_lines=header_lines,
    )


def _store_item(store_name: str, kind: str, fields: Sequence[str]) -> _Item:
    row_width = (
        max(
            [_text_width(kind), *(_text_width(field) for field in fields)],
            default=0,
        )
        + 2 * CARD_PADDING
    )
    header_width = _text_width(store_name, font_size=12.5) + 2 * CARD_PADDING
    width = min(
        CARD_WIDTH_CAP, max(CARD_WIDTH, math.ceil(max(row_width, header_width)))
    )
    header_lines = _header_lines(store_name, width=width)
    header_height = _header_height(len(header_lines))
    height = header_height + (1 + len(fields)) * ROW_HEIGHT
    return _Item(
        name=store_name,
        kind="store",
        width=width,
        height=height,
        header_height=header_height,
        header_lines=header_lines,
    )


def _build_items(
    page: Page,
    tables: Mapping[str, Table],
    edges: Sequence[_Edge],
) -> list[_Item]:
    roles = _endpoint_roles(edges)
    if page.data.domains:
        items = [
            _domain_item(domain, tables=tables, roles=roles)
            for domain in page.data.domains
        ]
    else:
        items = [_table_item(tables[table_name]) for table_name in page.data.sql_tables]
    items.extend(
        _store_item(store.name, store.kind, store.fields) for store in page.data.nosql
    )
    return items


def _layout_columns(
    page: Page,
    items: Sequence[_Item],
) -> list[list[_Item]]:
    items_by_name = {item.name: item for item in items}
    if page.data.columns:
        columns = [
            [items_by_name[name] for name in column if name in items_by_name]
            for column in page.data.columns
        ]
        declared = {name for column in page.data.columns for name in column}
        extra = [item for item in items if item.name not in declared]
        if extra:
            columns.append(extra)
        return [column for column in columns if column]

    column_count = max(1, math.ceil(math.sqrt(len(items))))
    columns: list[list[_Item]] = [[] for _ in range(column_count)]
    heights = [0] * column_count
    for item in items:
        target = min(range(column_count), key=heights.__getitem__)
        columns[target].append(item)
        heights[target] += item.height + CARD_GAP
    return columns


def _place_items(columns: Sequence[Sequence[_Item]]) -> tuple[int, int]:
    x = LEFT_MARGIN
    max_bottom = 0
    for column_number, column in enumerate(columns):
        y = LEFT_MARGIN
        width = max((item.width for item in column), default=CARD_WIDTH)
        for item in column:
            item.column = column_number
            item.x = x
            item.y = y
            y += item.height + CARD_GAP
            max_bottom = max(max_bottom, item.y + item.height)
        x += width + COLUMN_GAP
    width = max(LEFT_MARGIN, x - COLUMN_GAP + LEFT_MARGIN)
    return width, max_bottom + LEFT_MARGIN


def _domain_by_table(page: Page) -> dict[str, str]:
    return {
        table_name: domain.id
        for domain in page.data.domains
        for table_name in domain.tables
    }


def _domain_row_index(domain: SchemaDomain, table_name: str) -> int:
    return domain.key_tables.index(table_name)


def _item_by_name(items: Sequence[_Item], name: str) -> _Item:
    return next(item for item in items if item.name == name)


def _anchor(
    endpoint: _Endpoint,
    *,
    page: Page,
    items: Sequence[_Item],
    tables: Mapping[str, Table],
) -> tuple[_Item, float]:
    if endpoint.kind == "nosql":
        item = _item_by_name(items, endpoint.name)
        store = next(store for store in page.data.nosql if store.name == endpoint.name)
        row_index = 0
        if endpoint.field in store.fields:
            row_index = store.fields.index(endpoint.field) + 1
        return (
            item,
            item.y + item.header_height + row_index * ROW_HEIGHT + ROW_HEIGHT / 2,
        )
    item_name = endpoint.name
    if page.data.domains:
        domain_id = _domain_by_table(page).get(endpoint.name)
        if domain_id is None:
            raise DiagramError(
                f"page '{page.id}' SQL table '{endpoint.name}' has no domain"
            )
        domain = next(domain for domain in page.data.domains if domain.id == domain_id)
        row_index = _domain_row_index(domain, endpoint.name)
        item = _item_by_name(items, domain.id)
        return (
            item,
            item.y + item.header_height + row_index * ROW_HEIGHT + ROW_HEIGHT / 2,
        )
    item = _item_by_name(items, item_name)
    if endpoint.field is None:
        return item, item.y + item.header_height / 2
    table = tables[item_name]
    row_index = next(
        index
        for index, column in enumerate(table.columns)
        if column.name == endpoint.field
    )
    return item, item.y + item.header_height + row_index * ROW_HEIGHT + ROW_HEIGHT / 2


def _endpoint_roles(edges: Sequence[_Edge]) -> dict[tuple[str, str], set[str]]:
    roles: dict[tuple[str, str], set[str]] = {}
    for edge in edges:
        if edge.src.kind == "sql" and edge.src.field is not None:
            roles.setdefault((edge.src.name, edge.src.field), set()).add("src")
        if edge.dst.kind == "sql" and edge.dst.field is not None:
            roles.setdefault((edge.dst.name, edge.dst.field), set()).add("dst")
    return roles


def _domain_key_markup(
    item: _Item,
    table_name: str,
    used: Sequence[tuple[str, set[str]]],
    *,
    y: float,
) -> str:
    available_width = max(
        0,
        item.width - 2 * CARD_PADDING - _text_width(table_name) - DOMAIN_KEY_GAP,
    )
    remaining = available_width
    segments: list[tuple[str, str]] = []
    truncated = False
    added_ellipsis = False
    for index, (column_name, column_roles) in enumerate(used):
        separator = ", " if index else ""
        separator_width = _text_width(separator)
        if separator_width > remaining:
            truncated = True
            break
        label_width = remaining - separator_width
        column_width = _text_width(column_name)
        if column_width > label_width:
            if separator:
                segments.append((separator, "#111827"))
            if label_width <= 0:
                truncated = True
                break
            visible = _truncate(column_name, label_width)
            role_color = (
                PK_COLOR
                if "dst" in column_roles and "src" not in column_roles
                else FK_COLOR
            )
            segments.append((visible, role_color))
            remaining = 0
            truncated = True
            added_ellipsis = True
            break
        role_color = (
            PK_COLOR
            if "dst" in column_roles and "src" not in column_roles
            else FK_COLOR
        )
        if separator:
            segments.append((separator, "#111827"))
        segments.append((column_name, role_color))
        remaining -= separator_width + column_width
    if truncated and not added_ellipsis and remaining >= _text_width("…"):
        segments.append(("…", "#111827"))
    rendered = []
    total_width = sum(_text_width(text) for text, _ in segments)
    x = item.width - CARD_PADDING - total_width
    for text, color in segments:
        rendered.append(_svg_text(x, y, text, fill=color, font_weight="600"))
        x += _text_width(text)
    return "".join(rendered)


def _svg_text(
    x: float,
    y: float,
    value: str,
    *,
    fill: str = "#111827",
    font_size: float = 11.5,
    font_weight: str = "400",
    font_style: str = "normal",
    anchor: str = "start",
) -> str:
    return (
        f'<text x="{x:g}" y="{y:g}" fill="{fill}" '
        f'font-family="{MONOSPACE}" font-size="{font_size:g}" '
        f'font-weight="{font_weight}" font-style="{font_style}" '
        f'text-anchor="{anchor}">{_escape(value)}</text>'
    )


def _header_text(item: _Item) -> str:
    lines: list[str] = []
    for line_number, line in enumerate(item.header_lines):
        y = (
            HEADER_VERTICAL_PADDING
            + HEADER_BASELINE_OFFSET
            + line_number * HEADER_LINE_HEIGHT
        )
        lines.append(
            _svg_text(
                CARD_PADDING,
                y,
                line,
                fill="white",
                font_size=12.5,
                font_weight="600",
            )
        )
    return "".join(lines)


def _card_header(
    item: _Item,
    *,
    title_href: str | None = None,
    count: str | None = None,
    store: bool = False,
) -> str:
    header_color = STORE_HEADER_COLOR if store else HEADER_COLOR
    background = STORE_BACKGROUND if store else CARD_BACKGROUND
    pieces = [
        f'<rect x="0" y="0" width="{item.width}" height="{item.header_height}" '
        f'fill="{header_color}" rx="6" ry="6"/>',
        f'<rect x="0" y="{item.header_height - 6}" width="{item.width}" height="6" '
        f'fill="{header_color}"/>',
    ]
    header = _header_text(item)
    if title_href is not None:
        header = f'<a href="{_escape(title_href, quote=True)}">{header}</a>'
    pieces.append(header)
    if count is not None:
        pieces.append(
            _svg_text(
                item.width - CARD_PADDING,
                HEADER_VERTICAL_PADDING + HEADER_BASELINE_OFFSET,
                count,
                fill="#cbd5e1",
                font_size=10.5,
                anchor="end",
            )
        )
    pieces.insert(
        0,
        f'<rect x="0" y="0" width="{item.width}" height="{item.height}" '
        f'fill="{background}" stroke="{STORE_HEADER_COLOR if store else CARD_BORDER}" '
        'stroke-width="1" rx="6" filter="url(#card-shadow)"/>',
    )
    return "".join(pieces)


def _row_separator(item: _Item, y: float) -> str:
    return (
        f'<line x1="0" y1="{y:g}" x2="{item.width}" y2="{y:g}" '
        f'stroke="{ROW_COLOR}" stroke-width="1"/>'
    )


def _domain_card(
    item: _Item,
    domain: SchemaDomain,
    *,
    page: Page,
    repo_url: str,
    tables: Mapping[str, Table],
    roles: Mapping[tuple[str, str], set[str]],
) -> str:
    header_href = page_href(page.id, domain.page) if domain.page is not None else None
    pieces = [
        f'<g class="schema-card schema-domain" transform="translate({item.x:g} {item.y:g})">',
        _card_header(
            item,
            title_href=header_href,
            count=f"{len(domain.tables)} tables",
        ),
    ]
    y = item.header_height
    for table_name in domain.key_tables:
        table = tables[table_name]
        source_href = _source_href(table.source, page=page, repo_url=repo_url)
        pieces.append(_row_separator(item, y))
        pieces.append(
            f'<a href="{source_href}">{_svg_text(CARD_PADDING, y + 13, table_name)}</a>'
        )
        used = [
            (column.name, roles.get((table_name, column.name), set()))
            for column in table.columns
            if (table_name, column.name) in roles
        ]
        if used:
            pieces.append(_domain_key_markup(item, table_name, used, y=y + 13))
        y += ROW_HEIGHT
    for note in domain.notes:
        pieces.append(_row_separator(item, y))
        lines = _wrap_text(note.text, item.width - 2 * CARD_PADDING)
        note_href = _source_href(note.sources[0], page=page, repo_url=repo_url)
        note_text = "".join(
            _svg_text(
                CARD_PADDING,
                y + 13 + index * ROW_HEIGHT,
                line,
                fill=BOILERPLATE_COLOR,
                font_style="italic",
            )
            for index, line in enumerate(lines)
        )
        pieces.append(f'<a href="{note_href}">{note_text}</a>')
        y += len(lines) * ROW_HEIGHT
    pieces.append(_row_separator(item, y))
    extra = len(domain.tables) - len(domain.key_tables)
    guide_href = f"#schema-guide-{_escape(domain.id, quote=True)}"
    pieces.append(
        f'<a href="{guide_href}">'
        f"{_svg_text(CARD_PADDING, y + 13, f'+ {extra} more tables', fill=BOILERPLATE_COLOR)}"
        "</a>"
    )
    pieces.append("</g>")
    return "".join(pieces)


def _column_fill(
    column: Column,
    *,
    table: Table,
    source_columns: set[tuple[str, str]],
) -> tuple[str, str]:
    if column.name == "id" and column.primary_key:
        has_public_key = any(
            candidate.name in {"public_id", "public_key", "rox_id"}
            for candidate in table.columns
        )
        if has_public_key:
            return BOILERPLATE_COLOR, "400"
    if column.name in BOILERPLATE_COLUMNS and column.name != "id":
        return BOILERPLATE_COLOR, "400"
    if (table.name, column.name) in source_columns or column.foreign_key is not None:
        return FK_COLOR, "600"
    if column.primary_key:
        return PK_COLOR, "700"
    return "#111827", "400"


def _table_card(
    item: _Item,
    table: Table,
    *,
    page: Page,
    repo_url: str,
    source_columns: set[tuple[str, str]],
) -> str:
    source_href = _source_href(table.source, page=page, repo_url=repo_url)
    pieces = [
        f'<g class="schema-card schema-table" transform="translate({item.x:g} {item.y:g})">',
        _card_header(
            item,
            title_href=source_href,
        ),
    ]
    y = item.header_height
    for column in table.columns:
        pieces.append(_row_separator(item, y))
        fill, weight = _column_fill(
            column,
            table=table,
            source_columns=source_columns,
        )
        pieces.append(
            _svg_text(
                CARD_PADDING,
                y + 13,
                column.name,
                fill=fill,
                font_weight=weight,
            )
        )
        pieces.append(
            _svg_text(
                item.width - CARD_PADDING,
                y + 13,
                column.type,
                fill="#6b7280",
                anchor="end",
            )
        )
        y += ROW_HEIGHT
    pieces.append("</g>")
    return "".join(pieces)


def _store_card(
    item: _Item,
    store_name: str,
    kind: str,
    fields: Sequence[str],
    *,
    page: Page,
    repo_url: str,
    source: Source,
) -> str:
    source_href = _source_href(source, page=page, repo_url=repo_url)
    pieces = [
        f'<g class="schema-card schema-store" transform="translate({item.x:g} {item.y:g})">',
        _card_header(
            item,
            title_href=source_href,
            store=True,
        ),
    ]
    y = item.header_height
    pieces.append(_row_separator(item, y))
    pieces.append(_svg_text(CARD_PADDING, y + 13, kind, fill="#92400e"))
    y += ROW_HEIGHT
    for field in fields:
        pieces.append(_row_separator(item, y))
        pieces.append(
            _svg_text(
                CARD_PADDING,
                y + 13,
                _truncate(field, item.width - 2 * CARD_PADDING),
            )
        )
        y += ROW_HEIGHT
    pieces.append("</g>")
    return "".join(pieces)


def _marker_defs() -> str:
    pieces = [
        "<defs>",
        '<filter id="card-shadow" x="-10%" y="-10%" width="120%" height="120%">'
        '<feDropShadow dx="0" dy="1" stdDeviation="2" flood-color="#0f172a" '
        'flood-opacity=".12"/></filter>',
        '<marker id="arrow-enforced" viewBox="0 0 10 10" refX="1" refY="5" '
        'markerWidth="8" markerHeight="8" orient="auto-start-reverse">'
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#2563eb"/></marker>',
        '<marker id="dot-enforced" viewBox="0 0 8 8" refX="4" refY="4" '
        'markerWidth="6" markerHeight="6" orient="auto">'
        '<circle cx="4" cy="4" r="3" fill="#2563eb"/></marker>',
        '<marker id="arrow-symbolic" viewBox="0 0 10 10" refX="1" refY="5" '
        'markerWidth="8" markerHeight="8" orient="auto-start-reverse">'
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#9ca3af"/></marker>',
        '<marker id="dot-symbolic" viewBox="0 0 8 8" refX="4" refY="4" '
        'markerWidth="6" markerHeight="6" orient="auto">'
        '<circle cx="4" cy="4" r="3" fill="#9ca3af"/></marker>',
        '<marker id="arrow-blob" viewBox="0 0 10 10" refX="1" refY="5" '
        'markerWidth="8" markerHeight="8" orient="auto-start-reverse">'
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#d97706"/></marker>',
        '<marker id="dot-blob" viewBox="0 0 8 8" refX="4" refY="4" '
        'markerWidth="6" markerHeight="6" orient="auto">'
        '<circle cx="4" cy="4" r="3" fill="#d97706"/></marker>',
        "</defs>",
    ]
    return "".join(pieces)


def _edge_style(kind: str) -> tuple[str, str, str]:
    if kind == "enforced":
        return FK_COLOR, "", "1.6"
    if kind == "blob":
        return BLOB_COLOR, "1 4", "1.6"
    return SYMBOLIC_COLOR, "6 4", "1.6"


def _edge_path(
    source_item: _Item,
    source_y: float,
    destination_item: _Item,
    destination_y: float,
    *,
    same_column_index: int,
) -> str:
    if source_item.column == destination_item.column:
        source_x = source_item.x + source_item.width
        destination_x = destination_item.x + destination_item.width
        bulge = 60 + 14 * same_column_index
        return (
            f"M {source_x:g} {source_y:g} "
            f"C {source_x + bulge:g} {source_y:g}, "
            f"{destination_x + bulge:g} {destination_y:g}, "
            f"{destination_x:g} {destination_y:g}"
        )
    if destination_item.column > source_item.column:
        source_x = source_item.x + source_item.width
        destination_x = destination_item.x
        control = max(60, abs(destination_x - source_x) / 2)
        return (
            f"M {source_x:g} {source_y:g} "
            f"C {source_x + control:g} {source_y:g}, "
            f"{destination_x - control:g} {destination_y:g}, "
            f"{destination_x:g} {destination_y:g}"
        )
    source_x = source_item.x
    destination_x = destination_item.x + destination_item.width
    control = max(60, abs(source_x - destination_x) / 2)
    return (
        f"M {source_x:g} {source_y:g} "
        f"C {source_x - control:g} {source_y:g}, "
        f"{destination_x + control:g} {destination_y:g}, "
        f"{destination_x:g} {destination_y:g}"
    )


def _edge_markup(
    edge: _Edge,
    *,
    page: Page,
    repo_url: str,
    items: Sequence[_Item],
    tables: Mapping[str, Table],
    source_offset: float,
    same_column_index: int,
) -> str:
    source_item, source_y = _anchor(edge.src, page=page, items=items, tables=tables)
    destination_item, destination_y = _anchor(
        edge.dst,
        page=page,
        items=items,
        tables=tables,
    )
    source_y += source_offset
    path = _edge_path(
        source_item,
        source_y,
        destination_item,
        destination_y,
        same_column_index=same_column_index,
    )
    color, dash, width = _edge_style(edge.kind)
    dash_attribute = f' stroke-dasharray="{dash}"' if dash else ""
    href = _source_href(edge.source, page=page, repo_url=repo_url)
    title = _escape(f"{edge.src.raw} → {edge.dst.raw}: {edge.label} ({edge.kind})")
    return (
        f'<a class="schema-edge" href="{href}"><title>{title}</title>'
        f'<path d="{path}" fill="none" stroke="{color}" stroke-width="{width}"'
        f'{dash_attribute} marker-start="url(#arrow-{edge.kind})" '
        f'marker-end="url(#dot-{edge.kind})"/></a>'
    )


def _render_edges(
    edges: Sequence[_Edge],
    *,
    page: Page,
    repo_url: str,
    items: Sequence[_Item],
    tables: Mapping[str, Table],
) -> str:
    source_counts: dict[str, int] = {}
    for edge in edges:
        source_counts[edge.src.raw] = source_counts.get(edge.src.raw, 0) + 1
    source_seen: dict[str, int] = {}
    same_column_seen: dict[int, int] = {}
    pieces = []
    for edge in edges:
        source_item, _ = _anchor(edge.src, page=page, items=items, tables=tables)
        destination_item, _ = _anchor(
            edge.dst,
            page=page,
            items=items,
            tables=tables,
        )
        same_column_index = 0
        if source_item.column == destination_item.column:
            same_column_index = same_column_seen.get(source_item.column, 0)
            same_column_seen[source_item.column] = same_column_index + 1
        count = source_counts[edge.src.raw]
        seen = source_seen.get(edge.src.raw, 0)
        source_seen[edge.src.raw] = seen + 1
        pieces.append(
            _edge_markup(
                edge,
                page=page,
                repo_url=repo_url,
                items=items,
                tables=tables,
                source_offset=(seen - (count - 1) / 2) * 6,
                same_column_index=same_column_index,
            )
        )
    return "".join(pieces)


def _render_cards(
    page: Page,
    *,
    repo_url: str,
    tables: Mapping[str, Table],
    items: Sequence[_Item],
    edges: Sequence[_Edge],
) -> str:
    roles = _endpoint_roles(edges)
    pieces = []
    if page.data.domains:
        domains = {domain.id: domain for domain in page.data.domains}
        for item in items:
            if item.kind == "domain":
                pieces.append(
                    _domain_card(
                        item,
                        domains[item.name],
                        page=page,
                        repo_url=repo_url,
                        tables=tables,
                        roles=roles,
                    )
                )
    else:
        table_map = {
            table_name: tables[table_name] for table_name in page.data.sql_tables
        }
        source_columns = {
            (edge.src.name, edge.src.field)
            for edge in edges
            if edge.src.kind == "sql" and edge.src.field is not None
        }
        for item in items:
            if item.kind == "table":
                pieces.append(
                    _table_card(
                        item,
                        table_map[item.name],
                        page=page,
                        repo_url=repo_url,
                        source_columns={
                            (table_name, column_name)
                            for table_name, column_name in source_columns
                            if column_name is not None
                        },
                    )
                )
    stores = {store.name: store for store in page.data.nosql}
    for item in items:
        if item.kind == "store":
            store = stores[item.name]
            pieces.append(
                _store_card(
                    item,
                    store.name,
                    store.kind,
                    store.fields,
                    page=page,
                    repo_url=repo_url,
                    source=store.source,
                )
            )
    return "".join(pieces)


def schema_svg(
    page: Page,
    *,
    repo_url: str,
    tables: Mapping[str, Table],
) -> str:
    domain_by_table = _domain_by_table(page)
    if page.data.domains:
        sql_names = _domain_key_tables(page.data.domains)
    else:
        sql_names = set(page.data.sql_tables)
    store_names = {store.name for store in page.data.nosql}
    edges = [
        _validate_relation(
            relation,
            page=page,
            tables=tables,
            sql_names=sql_names,
            store_names=store_names,
            domain_by_table=domain_by_table,
        )
        for relation in page.data.relations
    ]
    if not page.data.domains:
        edges = _declared_fk_edges(
            page,
            tables=tables,
            sql_names=sql_names,
            authored_edges=edges,
        )
    items = _build_items(page, tables, edges)
    columns = _layout_columns(page, items)
    width, height = _place_items(columns)
    same_column_counts: dict[int, int] = {}
    for edge in edges:
        source_item, _ = _anchor(edge.src, page=page, items=items, tables=tables)
        destination_item, _ = _anchor(
            edge.dst,
            page=page,
            items=items,
            tables=tables,
        )
        if source_item.column == destination_item.column:
            same_column_counts[source_item.column] = (
                same_column_counts.get(source_item.column, 0) + 1
            )
    if same_column_counts:
        width += max(60 + 14 * (count - 1) for count in same_column_counts.values())
    edges_markup = _render_edges(
        edges,
        page=page,
        repo_url=repo_url,
        items=items,
        tables=tables,
    )
    cards_markup = _render_cards(
        page,
        repo_url=repo_url,
        tables=tables,
        items=items,
        edges=edges,
    )
    return (
        f'<svg class="schema-svg" xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        'role="img" aria-label="Schema diagram">'
        f"{_marker_defs()}"
        "<style>"
        ".schema-edge path { transition: stroke-width .12s ease; }"
        ".schema-edge:hover path { stroke-width: 2.8; }"
        "</style>"
        f"{edges_markup}{cards_markup}</svg>"
    )
