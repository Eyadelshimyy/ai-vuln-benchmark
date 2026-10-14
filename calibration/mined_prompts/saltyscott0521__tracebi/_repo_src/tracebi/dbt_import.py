"""``tracebi import dbt`` — draft a model from a dbt ``manifest.json``.

Reads the manifest and writes ``models/<name>.py``. Does not run dbt, does
not connect to a warehouse, and does not load ``.env``. A ``--connection``
names a module ``tracebi connect`` already wrote; that module calls
``load_dotenv()`` itself.

One manifest shape: a JSON object with a ``nodes`` object. A node is drafted
when ``resource_type`` is ``model``, ``config.enabled`` is not false, and
``config.materialized`` is not ``ephemeral``. Columns are the ``columns``
object on the node (name and optional ``data_type``). Anything else is
ignored rather than interpreted as a second format.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import date
from pathlib import Path

from tracebi.connect import _find_connection, _is_key_column, _summable

_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_NUMERIC = {
    "int", "integer", "bigint", "smallint", "tinyint",
    "int32", "int64",
    "float", "float4", "float8", "float32", "float64",
    "double", "real", "decimal", "numeric", "number",
}


def import_dbt_command(args) -> int:
    """Draft ``models/<slug>.py`` from a dbt manifest. Writes nothing on refusal."""
    manifest_path = _resolve_manifest(Path(args.path))
    if manifest_path is None:
        given = args.path
        print(
            f"no manifest.json at {given} "
            "(pass a dbt project root containing target/manifest.json, "
            "or the file itself)",
            file=sys.stderr,
        )
        return 1
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        print(f"manifest unreadable: {exc}", file=sys.stderr)
        return 1
    if not isinstance(payload, dict) or not isinstance(payload.get("nodes"), dict):
        print(
            f"{manifest_path} has no nodes object — expected a dbt manifest.json",
            file=sys.stderr,
        )
        return 1

    models = _select_models(payload, (args.schema or "").strip())
    if not models:
        where = f" in schema {args.schema.strip()!r}" if (args.schema or "").strip() else ""
        print(
            "manifest has no dbt models to import"
            f"{where} (resource_type model, not ephemeral)",
            file=sys.stderr,
        )
        return 1

    label = (args.name or "").strip() or _default_label(payload, Path(args.path), manifest_path)
    slug = _slug(label)
    if not _NAME.fullmatch(slug):
        print(
            f"model name {label!r} must contain a letter or digit",
            file=sys.stderr,
        )
        return 2

    connection = (args.connection or "").strip()
    models_dir: Path = args.models_dir
    problems: list[str] = []
    if connection:
        if not _NAME.fullmatch(connection):
            print(
                f"connection name {connection!r} must be a Python identifier",
                file=sys.stderr,
            )
            return 2
        conn_path = _find_connection(models_dir, connection)
        if conn_path is None:
            problems.append(
                f"connection {connection!r} not found in connections/ — "
                f"run `tracebi connect {connection}` first"
            )
    out_path = models_dir / f"{slug}.py"
    if out_path.exists() and not args.force:
        problems.append(
            f"refusing to overwrite existing {out_path}; pass --force to replace"
        )
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        _model_source(slug, connection or None, models,
                      declared=bool(connection) and conn_path.suffix != ".py"),
        encoding="utf-8",
    )
    print(f"Created {out_path}")
    print("  Lines marked # DRAFT: review are not declarations — edit and approve them.")
    print("  Edit the file, then use it with:")
    print("    from tracebi.model_registry import get_model")
    print(f'    model = get_model("{slug}")')
    return 0


def _slug(label: str) -> str:
    """Same folding as ``tracebi.cli._slugify``, without the ``report`` fallback."""
    text = re.sub(r"[^a-z0-9]+", "_", label.strip().lower())
    return re.sub(r"_+", "_", text).strip("_")


def _resolve_manifest(path: Path) -> Path | None:
    """A project root's ``target/manifest.json``, or the file the user named."""
    if path.is_dir():
        candidate = path / "target" / "manifest.json"
        return candidate if candidate.is_file() else None
    if path.is_file():
        return path
    return None


def _default_label(payload: dict, given: Path, manifest_path: Path) -> str:
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        project = metadata.get("project_name")
        if isinstance(project, str) and project.strip():
            return project.strip()
    if given.is_dir():
        return given.name
    if manifest_path.parent.name == "target":
        return manifest_path.parent.parent.name
    return manifest_path.parent.name or manifest_path.stem


def _select_models(payload: dict, schema: str) -> list[dict]:
    project = ""
    metadata = payload.get("metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("project_name"), str):
        project = metadata["project_name"]
    found: list[dict] = []
    want = schema.casefold()
    for unique_id, node in payload["nodes"].items():
        if not isinstance(node, dict):
            continue
        if node.get("resource_type") != "model":
            continue
        config = node.get("config") if isinstance(node.get("config"), dict) else {}
        if config.get("enabled") is False:
            continue
        materialized = config.get("materialized")
        if isinstance(materialized, str) and materialized.lower() == "ephemeral":
            continue
        node_schema = node.get("schema") if isinstance(node.get("schema"), str) else ""
        if want and node_schema.casefold() != want:
            continue
        name = node.get("name") if isinstance(node.get("name"), str) else ""
        name = name.strip()
        if not name and isinstance(unique_id, str):
            name = unique_id.split(".")[-1]
        if not name:
            continue
        alias = node.get("alias") if isinstance(node.get("alias"), str) else ""
        source = alias.strip() or name
        package = node.get("package_name") if isinstance(node.get("package_name"), str) else ""
        relation = node.get("relation_name") if isinstance(node.get("relation_name"), str) else ""
        found.append({
            "unique_id": unique_id if isinstance(unique_id, str) else name,
            "name": name,
            "source": source,
            "schema": node_schema,
            "package": package.strip(),
            "relation": relation.strip(),
            "columns": _columns(node),
            "root": not package or package == project,
        })
    counts: dict[str, int] = {}
    root_counts: dict[str, int] = {}
    for item in found:
        counts[item["name"]] = counts.get(item["name"], 0) + 1
        if item["root"]:
            root_counts[item["name"]] = root_counts.get(item["name"], 0) + 1
    for item in found:
        collision = counts[item["name"]] > 1
        keep_root = item["root"] and root_counts.get(item["name"]) == 1
        if not collision or keep_root:
            item["logical"] = item["name"]
            item["disambiguated"] = False
        else:
            prefix = item["package"] or "model"
            item["logical"] = f"{prefix}__{item['name']}"
            item["disambiguated"] = True
    found.sort(key=lambda item: (item["schema"].casefold(), item["logical"], item["unique_id"]))
    return found


def _columns(node: dict) -> list[tuple[str, str]]:
    raw = node.get("columns")
    if not isinstance(raw, dict):
        return []
    columns: list[tuple[str, str]] = []
    for key, info in raw.items():
        dtype = ""
        if isinstance(info, dict):
            name = info.get("name") if isinstance(info.get("name"), str) else ""
            name = name.strip() or (key if isinstance(key, str) else "")
            raw_type = info.get("data_type")
            if isinstance(raw_type, str):
                dtype = raw_type.strip()
        elif isinstance(key, str):
            name = key
        else:
            continue
        name = " ".join(name.split())
        if name:
            columns.append((name, dtype))
    return columns


def _numeric(dtype: str) -> bool:
    head = dtype.split("(", 1)[0].strip().lower()
    return head in _NUMERIC


def _model_source(slug: str, connection: str | None, models: list[dict],
                  declared: bool = False) -> str:
    lines = [
        '"""',
        slug,
        "=" * len(slug),
        "",
        f"DataModel drafted by ``tracebi import dbt`` on {date.today().isoformat()}.",
        "",
        "Tables are dbt models from the manifest (``resource_type`` model,",
        "not ephemeral). dbt was not run and no rows were read. Lines marked",
        "``# DRAFT: review`` are not declarations: relationships are not",
        "inferred, and measures stay comments until you choose them.",
        "",
        "    from tracebi.model_registry import get_model",
        f'    model = get_model("{slug}")',
        '"""',
        "",
    ]
    if connection:
        if declared:
            wire = [
                "import os",
                "",
                "from tracebi import DataModel",
                "from tracebi.connections import connection_file, load_connection",
                "",
                "ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))",
                f"connector = load_connection(connection_file({connection!r}, ROOT), ROOT)",
            ]
        else:
            wire = [
                "import os",
                "import runpy",
                "",
                "from tracebi import DataModel",
                "",
                "_conn = runpy.run_path(os.path.join(",
                "    os.path.dirname(os.path.abspath(__file__)),",
                f"    {f'_connections/{connection}.py'!r}))",
                'connector = _conn["connector"]',
            ]
        lines.extend([
            *wire,
            "",
            "# Importing this file must not query the warehouse. A query connects.",
            "",
            f"model = DataModel({slug!r})",
            "model.add_connector(connector)",
        ])
    else:
        lines.extend([
            "from tracebi import DataModel",
            "",
            "# DRAFT: review — no connector wired. Re-run with --connection <name>",
            "# (models/_connections/<name>.py from `tracebi connect`) and pass --force,",
            "# or register a connector here before uncommenting the tables.",
            f"model = DataModel({slug!r})",
            "# model.add_connector(connector)",
        ])
    connector_expr = connection if connection else "<connection>"
    for item in models:
        if item["disambiguated"]:
            lines.append(
                f"# DRAFT: review — {item['name']!r} is declared by more than one "
                "package; logical name is disambiguated"
            )
        call = (
            f"model.add_table({item['logical']!r}, connector={connector_expr!r}, "
            f"source={item['source']!r})"
        )
        lines.append(call if connection else "# " + call)
        note = _table_note(item)
        if note:
            lines.append("# " + note)
    lines.extend([
        "",
        "# DRAFT: review — relationships are not read from the manifest.",
        "# Add one only after you have checked the join.",
        "# model.add_relationship(",
        '#     name="...",',
        '#     left_table="...", right_table="...",',
        '#     left_key="...",',
        "# )",
        "",
        "# DRAFT: review — measures are comments, not declarations.",
    ])
    suggested = _measure_comments(models)
    if suggested:
        lines.extend(suggested)
    else:
        lines.append('# model.add_measure("...", column="...", agg="sum")')
    lines.append("")
    return "\n".join(lines)


def _table_note(item: dict) -> str:
    parts: list[str] = []
    if item["package"] and not item["root"]:
        parts.append(f"package {item['package']}")
    if item["schema"]:
        parts.append(f"schema {item['schema']}")
    if item["relation"]:
        parts.append("relation " + " ".join(item["relation"].split()))
    if item["columns"]:
        rendered = []
        for name, dtype in item["columns"]:
            rendered.append(f"{name} ({dtype})" if dtype else name)
        parts.append("columns: " + ", ".join(rendered))
    return " — ".join(parts)


def _measure_comments(models: list[dict]) -> list[str]:
    lines: list[str] = []
    used: set[str] = set()
    for item in models:
        for name, dtype in item["columns"]:
            if not dtype or not _numeric(dtype) or _is_key_column(name) or not _summable(name):
                continue
            if not _NAME.fullmatch(name):
                continue
            measure = name if name not in used else f"{item['logical']}_{name}"
            if not _NAME.fullmatch(measure):
                continue
            used.add(measure)
            lines.append(
                f"# model.add_measure({measure!r}, column={name!r}, agg='sum')"
            )
    return lines
