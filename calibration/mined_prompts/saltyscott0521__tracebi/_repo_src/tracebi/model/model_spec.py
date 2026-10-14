"""
A model as data: ``models/<name>.yaml`` (also ``.yml``) or ``models/<name>.json``.

YAML is the human form of the same schema: comments, no quoting, one loader
(``yaml.safe_load`` semantics only, and a repeated key is an error rather than
last-wins). The document mirrors the :class:`DataModel` builder calls one to one, so a model
can be drafted without running anyone's Python. The schema is closed —
unknown keys are errors — and carries no credentials: a connector is either a
DuckDB file under the project, or a reference (``connection: <x>``) to
``connections/<x>.yaml`` or, failing that, the legacy
``models/_connections/<x>.py``; either way the secret stays in ``.env``.

Loading is as lazy as the Python form: building the :class:`DataModel` reads
no rows. A query is what opens the warehouse.
"""

from __future__ import annotations

import difflib
import json
import os
import re
import runpy
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from tracebi.model.data_model import DataModel

_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

SPEC_SUFFIXES = (".yaml", ".yml", ".json")

_TOP = {"name", "connectors", "tables", "relationships", "dimensions", "facts",
        "measures", "time_grains", "value_bins"}
_CONNECTOR = {"name", "type", "database", "connection"}
_TABLE = {"name", "connector", "source"}
_RELATIONSHIP = {"name", "left_table", "right_table", "left_key", "right_key",
                 "how"}
_DIMENSION = {"name", "table", "key", "attributes"}
_FACT = {"name", "table", "measures", "foreign_keys"}
# The keyword arguments of DataModel.add_time_grain / add_value_bins; `dim`
# is spelled out as `dimension`, like the dimension's own `table` and `key`.
_TIME_GRAIN = {"dimension", "name", "source", "grain"}
_VALUE_BINS = {"dimension", "name", "source", "edges", "labels"}
# Every keyword DataModel.add_measure takes, plus the measure's name.
_MEASURE = {"name", "column", "agg", "expr", "ratio", "share", "rank", "running",
            "partition_by", "period_end", "offset", "growth", "to_date",
            "description", "format", "allow_rate_agg", "allow_additive"}
_PAIR = {"ratio", "period_end", "to_date"}
_TRIPLE = {"offset", "growth"}
_STR = {"column", "agg", "expr", "share", "rank", "running", "description",
        "format"}
_HOWS = ("left", "inner", "right", "outer")


class ModelSpecError(ValueError):
    """A model file that does not fit the schema; ``errors`` lists each problem."""

    def __init__(self, filename: str, errors: list[str]) -> None:
        super().__init__(f"{filename}: " + "; ".join(errors))
        self.errors = errors


def _unique_key_loader():
    """A ``SafeLoader`` that refuses a repeated mapping key (PyYAML keeps the
    last one silently, which would let a second ``agg:`` quietly win)."""
    import yaml

    class _Loader(yaml.SafeLoader):
        def construct_mapping(self, node, deep=False):
            self.flatten_mapping(node)
            seen = set()
            for key_node, _ in node.value:
                key = self.construct_object(key_node, deep=True)
                try:
                    dup = key in seen
                    seen.add(key)
                except TypeError:
                    continue  # unhashable: the parent's construct_mapping says so
                if dup:
                    raise yaml.constructor.ConstructorError(
                        "while constructing a mapping", node.start_mark,
                        f"found duplicate key {key!r}", key_node.start_mark)
            return super().construct_mapping(node, deep)

    return _Loader


def read_model_doc(path: "str | os.PathLike[str]") -> Any:
    """The parsed document of a model file: ``.json`` as JSON, ``.yaml`` /
    ``.yml`` through a ``SafeLoader`` that refuses repeated keys (never an
    arbitrary-object loader). Raises :class:`ModelSpecError`."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ModelSpecError(path.name, [f"not valid JSON ({exc})"]) from exc
    try:
        import yaml
    except ImportError as exc:  # pyyaml is a base dependency; say so if it is gone
        raise ImportError("YAML models need PyYAML: pip install pyyaml") from exc
    try:
        return yaml.load(text, Loader=_unique_key_loader())  # noqa: S506 — a SafeLoader subclass
    except yaml.YAMLError as exc:
        why = " ".join(str(exc).split())
        raise ModelSpecError(path.name, [f"not valid YAML ({why})"]) from exc


def _unknown(keys: Any, allowed: set[str], path: str) -> list[str]:
    errors = []
    for key in keys:
        if key not in allowed:
            close = difflib.get_close_matches(str(key), sorted(allowed), n=1)
            hint = f" Did you mean '{close[0]}'?" if close else ""
            errors.append(
                f"{path}: unknown key '{key}'.{hint} "
                f"One of: {', '.join(sorted(allowed))}."
            )
    return errors


def _escapes(rel: str) -> bool:
    return (os.path.isabs(rel) or PureWindowsPath(rel).is_absolute()
            or PurePosixPath(rel.replace("\\", "/")).is_absolute()
            or ".." in PurePosixPath(rel.replace("\\", "/")).parts)


def _items(doc: dict, key: str, allowed: set[str], errors: list[str]) -> list:
    """The list under *key* when it is a list of objects; else record why not."""
    value = doc.get(key, [])
    if not isinstance(value, list):
        errors.append(f"{key}: must be a list")
        return []
    out = []
    for i, item in enumerate(value):
        path = f"{key}[{i}]"
        if not isinstance(item, dict):
            errors.append(f"{path}: must be an object")
            continue
        errors.extend(_unknown(item, allowed, path))
        out.append((path, item))
    return out


def _need(item: dict, path: str, fields: tuple[str, ...], errors: list[str]) -> None:
    for f in fields:
        if not isinstance(item.get(f), str) or not item.get(f):
            errors.append(f"{path}.{f}: required, a non-empty string")


def validate_model_spec(doc: Any) -> list[str]:
    """Structural errors in a model document, each led by its path
    (``measures[2].agg``). Empty means the shape is right; whether the builder
    accepts the declarations is checked when the model is compiled."""
    if not isinstance(doc, dict):
        return ["the file must hold one object (a mapping at the top level)"]
    errors = _unknown(doc, _TOP, "model")
    _need(doc, "model", ("name",), errors)

    for path, c in _items(doc, "connectors", _CONNECTOR, errors):
        _need(c, path, ("name",), errors)
        if "connection" in c:
            extra = [k for k in c if k not in ("name", "connection")]
            if extra:
                errors.append(f"{path}: a connection reference takes only name "
                              f"and connection, not {', '.join(sorted(extra))}")
            conn = c["connection"]
            if not isinstance(conn, str) or not _NAME.fullmatch(conn):
                errors.append(f"{path}.connection: must name a "
                              "connections/<name>.yaml (letters, digits, "
                              "underscore)")
        elif c.get("type") != "duckdb":
            errors.append(f"{path}.type: must be 'duckdb' (or give 'connection' "
                          f"to reuse connections/<name>.yaml), got "
                          f"{c.get('type')!r}")
        else:
            db = c.get("database")
            if not isinstance(db, str) or not db:
                errors.append(f"{path}.database: required, a project-relative "
                              "path to the .duckdb file")
            elif _escapes(db):
                errors.append(f"{path}.database: must stay inside the project "
                              f"(no absolute path, no '..'), got {db!r}")

    for path, t in _items(doc, "tables", _TABLE, errors):
        _need(t, path, ("name", "connector", "source"), errors)
    for path, r in _items(doc, "relationships", _RELATIONSHIP, errors):
        _need(r, path, ("name", "left_table", "right_table", "left_key"), errors)
        if "right_key" in r and not isinstance(r["right_key"], str):
            errors.append(f"{path}.right_key: must be a string")
        if r.get("how", "left") not in _HOWS:
            errors.append(f"{path}.how: must be one of {', '.join(_HOWS)}")
    for path, d in _items(doc, "dimensions", _DIMENSION, errors):
        _need(d, path, ("name", "table", "key"), errors)
        attrs = d.get("attributes", [])
        if not (isinstance(attrs, list) and all(isinstance(a, str) for a in attrs)):
            errors.append(f"{path}.attributes: must be a list of column names")
    for path, f in _items(doc, "facts", _FACT, errors):
        _need(f, path, ("name", "table"), errors)
        ms = f.get("measures")
        if not (isinstance(ms, list) and all(isinstance(m, str) for m in ms)):
            errors.append(f"{path}.measures: required, a list of column names")
        fks = f.get("foreign_keys", {})
        if not (isinstance(fks, dict)
                and all(isinstance(k, str) and isinstance(v, str)
                        for k, v in fks.items())):
            errors.append(f"{path}.foreign_keys: must map dimension name to "
                          "the fact's key column")
    for path, g in _items(doc, "time_grains", _TIME_GRAIN, errors):
        _need(g, path, ("dimension", "name", "source", "grain"), errors)
    for path, b in _items(doc, "value_bins", _VALUE_BINS, errors):
        _need(b, path, ("dimension", "name", "source"), errors)
        edges = b.get("edges")
        if not (isinstance(edges, list) and edges and all(
                isinstance(e, (int, float)) and not isinstance(e, bool)
                for e in edges)):
            errors.append(f"{path}.edges: required, a non-empty list of numbers")
        labels = b.get("labels")
        if labels is not None and not (
                isinstance(labels, list) and all(isinstance(v, str) for v in labels)):
            errors.append(f"{path}.labels: must be a list of strings")
    for path, m in _items(doc, "measures", _MEASURE, errors):
        _need(m, path, ("name",), errors)
        for k in _STR & m.keys():
            if not isinstance(m[k], str):
                errors.append(f"{path}.{k}: must be a string")
        for k in _PAIR & m.keys():
            if not (isinstance(m[k], list) and len(m[k]) == 2
                    and all(isinstance(v, str) for v in m[k])):
                errors.append(f"{path}.{k}: must be a list of two strings")
        for k in _TRIPLE & m.keys():
            v = m[k]
            if not (isinstance(v, list) and len(v) == 3
                    and isinstance(v[0], str) and isinstance(v[1], str)
                    and isinstance(v[2], int) and not isinstance(v[2], bool)):
                errors.append(f"{path}.{k}: must be [base_measure, unit, n]")
        pb = m.get("partition_by")
        if pb is not None and not (
                isinstance(pb, str)
                or (isinstance(pb, list) and all(isinstance(v, str) for v in pb))):
            errors.append(f"{path}.partition_by: must be a string or a list of strings")
        for k in ("allow_rate_agg", "allow_additive"):
            if k in m and not isinstance(m[k], bool):
                errors.append(f"{path}.{k}: must be true or false")
    return errors


def _connector(c: dict, root: Path):
    if "connection" in c:
        from tracebi.connections import connection_file, load_connection

        declared = connection_file(c["connection"], root)
        if declared is not None:
            connector = load_connection(declared, root)
            where = f"connections/{declared.name}"
        else:
            path = root / "models" / "_connections" / f"{c['connection']}.py"
            if not path.is_file():
                raise ValueError(f"connector '{c['name']}': connections/"
                                 f"{c['connection']}.yaml does not exist (run "
                                 f"`tracebi connect {c['connection']}`)")
            connector = runpy.run_path(str(path)).get("connector")
            if connector is None:
                raise ValueError(f"{path} does not define 'connector'")
            where = f"models/_connections/{c['connection']}.py"
        if connector.name != c["name"]:
            raise ValueError(f"connector '{c['name']}': {where} builds a "
                             f"connector named '{connector.name}'")
        return connector
    from tracebi.connectors.duckdb_connector import DuckDBConnector

    database = (root / c["database"]).resolve()
    if root.resolve() not in database.parents:
        raise ValueError(f"connector '{c['name']}': database must stay inside "
                         f"the project, got {c['database']!r}")
    return DuckDBConnector(c["name"], database=str(database))


def load_model_spec(path: "str | os.PathLike[str]",
                    root: "str | os.PathLike[str] | None" = None) -> DataModel:
    """Compile ``models/<name>.yaml`` (or ``.yml`` / ``.json``) into a
    :class:`DataModel`.

    *root* is the project root that ``database`` paths and
    ``models/_connections/`` resolve against (default: the folder above the
    file's own). The name must equal the file stem. Reads no rows.
    """
    path = Path(path)
    root = Path(root) if root is not None else path.resolve().parent.parent
    doc = read_model_doc(path)
    errors = validate_model_spec(doc)
    if not errors and doc["name"] != path.stem:
        errors.append(f"name: must equal the file name '{path.stem}', got "
                      f"{doc['name']!r}")
    if errors:
        raise ModelSpecError(path.name, errors)

    model = DataModel(doc["name"])
    for c in doc.get("connectors", []):
        model.add_connector(_connector(c, root))
    for t in doc.get("tables", []):
        model.add_table(t["name"], connector=t["connector"], source=t["source"])
    for r in doc.get("relationships", []):
        model.add_relationship(r["name"], left_table=r["left_table"],
                               right_table=r["right_table"],
                               left_key=r["left_key"],
                               right_key=r.get("right_key"),
                               how=r.get("how", "left"))
    for d in doc.get("dimensions", []):
        model.add_dimension(d["name"], table_name=d["table"], key_col=d["key"],
                            attributes=d.get("attributes"))
    for f in doc.get("facts", []):
        model.add_fact(f["name"], table_name=f["table"], measures=f["measures"],
                       foreign_keys=f.get("foreign_keys"))
    for g in doc.get("time_grains", []):
        model.add_time_grain(g["dimension"], g["name"], source=g["source"],
                             grain=g["grain"])
    for b in doc.get("value_bins", []):
        model.add_value_bins(b["dimension"], b["name"], source=b["source"],
                             edges=b["edges"], labels=b.get("labels"))
    for m in doc.get("measures", []):
        kwargs = {k: v for k, v in m.items() if k != "name"}
        for k in _PAIR | _TRIPLE:
            if k in kwargs:
                kwargs[k] = tuple(kwargs[k])
        model.add_measure(m["name"], **kwargs)
    return model


def validate_model_file(path: "str | os.PathLike[str]",
                        root: "str | os.PathLike[str] | None" = None) -> list[str]:
    """Everything wrong with a model file; empty means it compiles to a
    :class:`DataModel`. For a draft outside ``models/``, pass the project *root*."""
    try:
        load_model_spec(path, root)
    except ModelSpecError as exc:
        return exc.errors
    except Exception as exc:  # noqa: BLE001 — the answer is the message
        return [str(exc)]
    return []
