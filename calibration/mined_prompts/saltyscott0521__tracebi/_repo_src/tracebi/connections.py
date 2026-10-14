"""
Sources as data: ``connections/<name>.yaml``.

A connection is a type and its fields. Any value may be a ``${ENV_VAR}``
reference (the whole value, ``[A-Z_][A-Z0-9_]*``); a credential-bearing field
(a password, a URL that carries one) *must* be, so a secret is never in the
file. The schema is closed, and the file loads with ``yaml.safe_load``
semantics and a repeated key is an error (``tracebi.model.model_spec``'s
loader).

Loading is lazy twice over. Reading the file builds a connector object and
resolves nothing; the connector reads its environment variables when it
actually connects, so a missing variable is an error naming it at that
moment, not at discovery. ``describe()`` and ``storage()`` (what the Sources
page shows) never resolve a credential: a referenced secret appears as
``${NAME}``.

``${NAME}`` is looked up in the process environment, then in the project's
``.env`` (where ``tracebi connect`` put the secret). Only referenced names are
read, and ``.env`` is never copied into ``os.environ``.
"""

from __future__ import annotations

import os
import re
import threading
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qsl, urlsplit

from tracebi.model.model_spec import (
    ModelSpecError, _escapes, _NAME, _unknown, _unique_key_loader,
)

SUFFIXES = (".yaml", ".yml")
_REF = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")
_SECRET_QUERY = re.compile(r"pass|pwd|secret|token|key", re.I)

# type -> (connector class name, required fields, optional fields)
_TYPES: dict[str, tuple[str, tuple[str, ...], tuple[str, ...]]] = {
    "duckdb": ("DuckDBConnector", ("database",), ("directory",)),
    "csv": ("CSVConnector", ("directory",), ("encoding",)),
    "sql": ("SQLConnector", ("url",), ()),
    "postgres": ("SQLConnector", ("url",), ()),
    "snowflake": ("SnowflakeConnector",
                  ("account", "user", "password", "warehouse", "database"),
                  ("schema", "role")),
    "bigquery": ("BigQueryConnector", ("project", "dataset"), ("credentials",)),
}
# A literal value is never allowed: the field holds a credential.
_MUST_REFERENCE = {"password"}
# A literal is allowed only when it carries no credential (see _url_problem).
_URL_FIELDS = {"url"}
# Paths a literal may only name inside the project.
_PATH_FIELDS = {"database": ("duckdb",), "directory": ("duckdb", "csv")}


class ConnectionConfigError(ValueError):
    """A connection that cannot be built: bad file, or an unset variable."""


def connections_dir(root: "str | os.PathLike[str] | None" = None) -> Path:
    """``TRACEBI_CONNECTIONS_DIR``, else ``<root>/connections`` (root: cwd)."""
    env = os.environ.get("TRACEBI_CONNECTIONS_DIR")
    if env:
        return Path(env)
    return Path(root if root is not None else ".") / "connections"


def connection_file(name: str, root: "str | os.PathLike[str] | None" = None) -> Optional[Path]:
    """The declarative file for connection *name*, if there is one."""
    base = connections_dir(root)
    for suffix in SUFFIXES:
        path = base / f"{name}{suffix}"
        if path.is_file():
            return path
    return None


def read_connection_doc(path: "str | os.PathLike[str]") -> Any:
    path = Path(path)
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("YAML connections need PyYAML: pip install pyyaml") from exc
    try:
        return yaml.load(path.read_text(encoding="utf-8"),  # noqa: S506 — a SafeLoader subclass
                         Loader=_unique_key_loader())
    except yaml.YAMLError as exc:
        raise ModelSpecError(path.name, [f"not valid YAML ({' '.join(str(exc).split())})"]) from exc


def _url_problem(url: str) -> Optional[str]:
    try:
        parts = urlsplit(url)
        query = parse_qsl(parts.query)
        password = parts.password
    except ValueError:
        return "is not a parseable URL"
    if password or any(_SECRET_QUERY.search(k) for k, _ in query):
        return ("carries a credential; write the whole url as ${ENV_VAR} "
                "and put the value in .env")
    return None


def validate_connection_doc(doc: Any) -> list[str]:
    """Structural errors in a connection document, each led by its field."""
    if not isinstance(doc, dict):
        return ["the file must hold one object (a mapping at the top level)"]
    ctype = doc.get("type")
    if ctype not in _TYPES:
        return [f"type: required, one of {', '.join(_TYPES)}; got {ctype!r}"]
    _cls, required, optional = _TYPES[ctype]
    errors = _unknown(doc, {"name", "type", *required, *optional}, "connection")
    if not isinstance(doc.get("name"), str) or not _NAME.fullmatch(doc.get("name") or ""):
        errors.append("name: required, letters, digits and underscores")
    for field in required + optional:
        if field not in doc:
            if field in required:
                errors.append(f"{field}: required for type '{ctype}'")
            continue
        value = doc[field]
        if not isinstance(value, str) or not value:
            errors.append(f"{field}: must be a non-empty string")
            continue
        if _REF.fullmatch(value):
            continue
        if "${" in value:
            errors.append(f"{field}: ${{ENV_VAR}} must be the whole value "
                          f"([A-Z_][A-Z0-9_]*), got {value!r}")
        elif field in _MUST_REFERENCE:
            errors.append(f"{field}: a credential; write it as ${{ENV_VAR}} "
                          f"and put the value in .env, never in this file")
        elif field in _URL_FIELDS and (why := _url_problem(value)):
            errors.append(f"{field}: {why}")
        elif ctype in _PATH_FIELDS.get(field, ()) and value != ":memory:" and _escapes(value):
            errors.append(f"{field}: a literal path must stay inside the project "
                          f"(no absolute path, no '..'), got {value!r}; use "
                          f"${{ENV_VAR}} to point elsewhere")
    return errors


# ── Reading the environment ────────────────────────────────────────────────

def _read_dotenv(path: Path) -> dict[str, str]:
    """``KEY=value`` lines, as ``tracebi connect`` writes them. Not loaded
    into ``os.environ``; callers ask for the names they reference."""
    out: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return out
    for line in lines:
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        if raw.startswith("export "):
            raw = raw[len("export "):].strip()
        key, sep, value = raw.partition("=")
        if not sep:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] == '"':
            value = re.sub(r"\\(.)", lambda m: "\n" if m.group(1) == "n" else m.group(1),
                           value[1:-1])
        elif len(value) >= 2 and value[0] == value[-1] == "'":
            value = value[1:-1]
        out[key.strip()] = value
    return out


# ── The connector ──────────────────────────────────────────────────────────

class _Declared:
    """Mixed into the real connector class. The fields that hold ``${REF}``
    (or a path to resolve) read through ``_value`` on access, so nothing is
    resolved before the connector is used."""

    _decl: "Declaration"

    def __getattribute__(self, item: str):
        if not item.startswith("_"):
            decl = object.__getattribute__(self, "__dict__").get("_decl")
            if decl is not None and item in decl.fields:
                return decl.value(item)
        return object.__getattribute__(self, item)

    @property
    def declared_in(self) -> Optional[str]:
        return self._decl.source

    def describe(self) -> dict:
        return self._decl.describe(self)

    def storage(self) -> dict:
        try:
            return super().storage()
        except ConnectionConfigError as exc:
            return {"kind": "unknown", "where": f"unresolved: {exc}",
                    "exists": None, "size": None}

    def disconnect(self) -> None:
        release = getattr(super(), "disconnect", None)
        if release is None:
            return
        try:
            release()
        except ConnectionConfigError:
            pass  # the variables were never readable, so nothing was opened


class Declaration:
    """A parsed connection file: its fields, and how each one resolves."""

    def __init__(self, name: str, ctype: str, fields: dict[str, str],
                 root: Path, source: Optional[str]) -> None:
        self.name, self.ctype, self.fields = name, ctype, fields
        self.root, self.source = root, source
        self._lock = threading.Lock()

    def is_ref(self, field: str) -> bool:
        return bool(_REF.fullmatch(self.fields[field]))

    def shown(self, field: str) -> str:
        """The field as the file wrote it; a reference stays ``${NAME}``."""
        return self.fields[field]

    def _env(self, var: str, field: str) -> str:
        value = os.environ.get(var)
        if not value:
            value = _read_dotenv(self.root / ".env").get(var)
        if not value:
            raise ConnectionConfigError(
                f"connection '{self.name}': environment variable {var} "
                f"(field '{field}') is not set; set it in the environment or "
                f"in {self.root / '.env'}")
        return value

    def value(self, field: str) -> Any:
        raw = self.fields[field]
        ref = _REF.fullmatch(raw)
        value = self._env(ref.group(1), field) if ref else raw
        if field in _PATH_FIELDS and self.ctype in _PATH_FIELDS[field] \
                and value != ":memory:" and not os.path.isabs(value):
            return str((self.root / value).resolve())
        if field == "credentials" and self.ctype == "bigquery":
            from tracebi.connect import _bq_credentials
            return _bq_credentials(value)
        return value

    def describe(self, connector) -> dict:
        out = {"name": self.name, "type": type(connector).__name__}
        if self.source:
            out["declared_in"] = self.source
        for field in ("database", "directory"):
            if field in self.fields and self.ctype in _PATH_FIELDS[field]:
                try:
                    out[field] = self.value(field)
                except ConnectionConfigError:
                    out[field] = self.shown(field)
        if "url" in self.fields:
            if self.is_ref("url"):
                out["url"] = self.shown("url")
            else:
                out["url"] = connector._redacted_url()
        return out


def _connector_class(ctype: str):
    cls_name = _TYPES[ctype][0]
    import importlib
    mod = {"DuckDBConnector": "duckdb_connector", "CSVConnector": "csv_connector",
           "SQLConnector": "sql_connector", "SnowflakeConnector": "snowflake_connector",
           "BigQueryConnector": "bigquery_connector"}[cls_name]
    base = getattr(importlib.import_module(f"tracebi.connectors.{mod}"), cls_name)
    # Same __name__ as the real class: code that tells a DuckDB warehouse
    # apart by type(c).__name__ keeps working.
    return type(cls_name, (_Declared, base), {"__module__": base.__module__})


def build_connection(doc: dict, root: "str | os.PathLike[str]",
                     source: Optional[str] = None):
    """The connector for a validated document. Resolves nothing."""
    ctype = doc["type"]
    fields = {k: v for k, v in doc.items() if k not in ("name", "type")}
    decl = Declaration(doc["name"], ctype, fields, Path(root).resolve(), source)
    cls = _connector_class(ctype)
    # The real constructor stores its arguments and does no work; the
    # placeholders are never read, _Declared answers for them.
    args = {
        "bigquery": {"project": None, "dataset": None},
        "sql": {"url": None}, "postgres": {"url": None},
        "snowflake": {"account": None, "user": None, "password": None,
                      "warehouse": None, "database": None},
        "csv": {"directory": "."},
        "duckdb": {"database": ":memory:"},
    }[ctype]
    connector = cls(doc["name"], **args)
    # Installed after construction, so the constructor above does not resolve.
    connector.__dict__["_decl"] = decl
    return connector


def load_connection(path: "str | os.PathLike[str]",
                    root: "str | os.PathLike[str] | None" = None):
    """Compile ``connections/<name>.yaml`` into a connector. Reads no
    environment and opens nothing. *root* (default: the folder above
    ``connections/``) is what relative paths and ``.env`` resolve against."""
    path = Path(path)
    doc = read_connection_doc(path)
    errors = validate_connection_doc(doc)
    if not errors and doc["name"] != path.stem:
        errors.append(f"name: must equal the file name '{path.stem}', got {doc['name']!r}")
    if errors:
        raise ModelSpecError(path.name, errors)
    root = Path(root) if root is not None else path.resolve().parent.parent
    try:
        shown = os.path.relpath(path.resolve(), Path.cwd())
    except ValueError:
        shown = str(path)
    return build_connection(doc, root, shown)


def validate_connection_file(path: "str | os.PathLike[str]") -> list[str]:
    try:
        load_connection(path)
    except ModelSpecError as exc:
        return exc.errors
    except Exception as exc:  # noqa: BLE001 — the answer is the message
        return [str(exc)]
    return []


# ── Migrating a connection module ──────────────────────────────────────────

class ConnectionMigrationError(ValueError):
    """A connection module that cannot be written as YAML, with the reason."""


class _Marker(dict):
    """Stands in for ``os.environ`` while a legacy connection module runs:
    every name reads back as ``${NAME}``, so the connector it builds holds the
    references, not anyone's secrets."""

    def __getitem__(self, key):
        return "${" + str(key) + "}"

    def get(self, key, default=None):
        return "${" + str(key) + "}"

    def __contains__(self, key):
        return True

    def __setitem__(self, key, value):
        pass


class _PathCredentials:
    def __init__(self, path: str) -> None:
        self.path = path


def _run_legacy(path: Path):
    """Run a ``models/_connections/<x>.py`` module under :class:`_Marker`."""
    import runpy
    import sys
    import types

    service_account = types.ModuleType("google.oauth2.service_account")
    service_account.Credentials = types.SimpleNamespace(
        from_service_account_file=lambda p: _PathCredentials(p))
    stubs = {"google": types.ModuleType("google"),
             "google.oauth2": types.ModuleType("google.oauth2"),
             "google.oauth2.service_account": service_account}
    stubs["google"].oauth2 = stubs["google.oauth2"]
    stubs["google.oauth2"].service_account = service_account
    saved_modules = {k: sys.modules.get(k) for k in stubs}
    saved_environ = os.environ
    sys.modules.update(stubs)
    os.environ = _Marker()  # type: ignore[assignment]
    try:
        return runpy.run_path(str(path)).get("connector")
    finally:
        os.environ = saved_environ
        for key, module in saved_modules.items():
            if module is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = module


def _migrated_fields(connector) -> dict:
    from tracebi.connectors import (
        BigQueryConnector, CSVConnector, DuckDBConnector, SnowflakeConnector,
        SQLConnector,
    )

    if getattr(connector, "declared_in", None):
        raise ConnectionMigrationError("it is already declarative")
    if isinstance(connector, DuckDBConnector):
        out = {"type": "duckdb", "database": connector.database}
        if connector.directory:
            out["directory"] = connector.directory
        return out
    if isinstance(connector, CSVConnector):
        out = {"type": "csv", "directory": connector.directory}
        if connector.encoding != "utf-8":
            out["encoding"] = connector.encoding
        return out
    if isinstance(connector, SQLConnector):
        if connector._engine_kwargs:
            raise ConnectionMigrationError(
                "it passes extra engine arguments to SQLConnector, which a "
                "connection file cannot carry")
        return {"type": "sql", "url": connector.url}
    if isinstance(connector, SnowflakeConnector):
        out = {"type": "snowflake", "account": connector.account,
               "user": connector.user, "password": connector.password,
               "warehouse": connector.warehouse, "database": connector.database}
        if connector.schema != "PUBLIC":
            out["schema"] = connector.schema
        if connector.role:
            out["role"] = connector.role
        return out
    if isinstance(connector, BigQueryConnector):
        out = {"type": "bigquery", "project": connector.project,
               "dataset": connector.dataset}
        if isinstance(connector.credentials, _PathCredentials):
            out["credentials"] = connector.credentials.path
        elif connector.credentials is not None:
            raise ConnectionMigrationError(
                "its credentials are built in code, not read from a file path")
        return out
    raise ConnectionMigrationError(
        f"{type(connector).__name__} has no connection type (the types are "
        f"{', '.join(_TYPES)}); it stays Python")


def connection_to_yaml(path: "str | os.PathLike[str]") -> str:
    """The ``connections/<name>.yaml`` text for a legacy
    ``models/_connections/<name>.py``, or :class:`ConnectionMigrationError`
    naming what cannot be expressed. The module is run with every environment
    variable reading back as ``${NAME}``, so a secret is never read."""
    import yaml

    path = Path(path)
    try:
        connector = _run_legacy(path)
    except Exception as exc:  # noqa: BLE001 — a module that will not run is the answer
        raise ConnectionMigrationError(
            f"the module did not run ({type(exc).__name__}: {exc})") from exc
    if connector is None:
        raise ConnectionMigrationError("it does not define 'connector'")
    if connector.name != path.stem:
        raise ConnectionMigrationError(
            f"it builds a connector named {connector.name!r} but the file is "
            f"{path.stem!r}; a connection's name is its file name")
    fields = _migrated_fields(connector)
    root = Path.cwd().resolve()
    for key in ("database", "directory"):
        value = fields.get(key)
        if key in fields and fields["type"] in _PATH_FIELDS[key] \
                and isinstance(value, str) and not _REF.fullmatch(value) \
                and value != ":memory:" and os.path.isabs(value):
            try:
                fields[key] = Path(value).resolve().relative_to(root).as_posix()
            except ValueError:
                raise ConnectionMigrationError(
                    f"{key} {value!r} is outside the project; put it in an "
                    f"environment variable and read it as ${{ENV_VAR}}") from None
    doc = {"name": path.stem, **fields}
    errors = validate_connection_doc(doc)
    if errors:
        raise ConnectionMigrationError("; ".join(errors))
    body = "\n".join(yaml.safe_dump({k: v}, allow_unicode=True, width=10**6).rstrip()
                     for k, v in doc.items())
    return (f"# Connection {path.stem!r} ({doc['type']}), migrated from "
            f"models/_connections/{path.stem}.py.\n"
            "# ${NAME} is read from the environment, or from .env beside this\n"
            "# folder, when the connection is used. No secret belongs in this file.\n"
            + body + "\n")
