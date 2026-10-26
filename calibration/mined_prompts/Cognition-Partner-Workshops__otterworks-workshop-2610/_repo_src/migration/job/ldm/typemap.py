"""Type map + manifest overrides -> per-column conversion specs (CONTRACTS.md §4.3, §8)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

from .config import TableConfig, TypeOverride
from .copybook import Copybook, Field
from .errors import ConfigError

Kind = Literal["char", "int", "decimal", "timestamp12", "date8", "binary"]

_TYPE_RE = re.compile(r"^(?P<base>[A-Z0-9_ ]+?)(?:\((?P<args>[0-9A-Za-z{}_, ]+)\))?(?P<suffix> FOR BIT DATA)?$")


@dataclass(frozen=True)
class TypeRule:
    source_pattern: str
    target_template: str
    kind: Kind


@dataclass(frozen=True)
class TypeMap:
    rules: tuple[TypeRule, ...]
    inference: tuple[dict[str, object], ...]
    target: str = "azuresql"

    @property
    def char_type(self) -> str:
        """Fixed-width text type of the target (FOR BIT DATA columns decoded to text land here)."""
        return "CHAR" if self.target == "postgresql" else "NCHAR"

    def spell_target_type(self, parsed: ParsedType) -> str:
        """Render a manifest target_type override in this target's dialect.

        Overrides are written once, in the base manifest, in the T-SQL spelling the contract documents
        (NVARCHAR(40), DECIMAL(18,8), DATETIME2(7), VARBINARY(n)); a PostgreSQL target maps them to the
        equivalent native type so the same manifest drives either provider.
        """
        base = parsed.base
        if self.target == "postgresql":
            base = _PG_SPELLING.get(base, base)
            if base == "BYTEA":
                return base
        return f"{base}({','.join(parsed.args)})" if parsed.args else base

    def infer_source_type(self, field: Field) -> str:
        key = field.pic_key
        for rule in self.inference:
            m = re.match(str(rule["pic"]), key)
            if not m:
                continue
            g = {k: v for k, v in m.groupdict().items() if v is not None}
            if "source_by_digits" in rule:
                digits = int(g.get("n", "0"))
                table = rule["source_by_digits"]
                assert isinstance(table, dict)
                for limit in sorted(int(k) for k in table):
                    if digits <= limit:
                        return str(table[limit])
                raise ConfigError(f"no source type for {key!r} with {digits} digits")
            if "p" in g and "s" in g:
                g["ps"] = str(int(g["p"]) + int(g["s"]))
            return str(rule["source"]).format(**g)
        raise ConfigError(f"type map cannot infer a source type for PIC {key!r}")

    def map_source_type(self, source_type: str) -> tuple[str, Kind]:
        parsed = _parse_type(source_type)
        for rule in self.rules:
            pat = _parse_type(rule.source_pattern)
            if pat.base != parsed.base or pat.bit_data != parsed.bit_data or len(pat.args) != len(parsed.args):
                continue
            subs: dict[str, str] = {}
            ok = True
            for pa, a in zip(pat.args, parsed.args, strict=True):
                if pa.startswith("{") and pa.endswith("}"):
                    subs[pa[1:-1].lower()] = a
                elif pa != a:
                    ok = False
            if ok:
                return rule.target_template.format(**subs), rule.kind
        raise ConfigError(f"type map has no rule for source type {source_type!r}")


_PG_SPELLING = {
    "NCHAR": "CHAR",
    "NVARCHAR": "VARCHAR",
    "DECIMAL": "NUMERIC",
    "INT": "INTEGER",
    "TINYINT": "SMALLINT",
    "DATETIME2": "TIMESTAMP",
    "VARBINARY": "BYTEA",
    "BINARY": "BYTEA",
}


@dataclass(frozen=True)
class ParsedType:
    base: str
    args: tuple[str, ...]
    bit_data: bool

    @property
    def precision(self) -> int:
        return int(self.args[0]) if self.args else 0

    @property
    def scale(self) -> int:
        return int(self.args[1]) if len(self.args) > 1 else 0

    @property
    def length(self) -> int:
        return int(self.args[0]) if self.args else 0


def _parse_type(text: str) -> ParsedType:
    m = _TYPE_RE.match(text.strip().upper())
    if not m:
        raise ConfigError(f"cannot parse type {text!r}")
    args = tuple(a.strip() for a in (m.group("args") or "").split(",") if a.strip())
    return ParsedType(base=m.group("base").strip(), args=args, bit_data=m.group("suffix") is not None)


def load_typemap(path: Path) -> TypeMap:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise ConfigError(f"type map not found: {path}") from e
    except yaml.YAMLError as e:
        raise ConfigError(f"type map {path} is not valid YAML: {e}") from e
    if not isinstance(data, dict) or "types" not in data:
        raise ConfigError(f"type map {path} must have a 'types' list")
    rules = []
    for r in data["types"]:
        try:
            rules.append(TypeRule(source_pattern=r["source"], target_template=r["target"], kind=r["kind"]))
        except (KeyError, TypeError) as e:
            raise ConfigError(f"type map {path}: bad rule {r!r}") from e
        if r["kind"] not in ("char", "int", "decimal", "timestamp12", "date8", "binary"):
            raise ConfigError(f"type map {path}: unknown kind {r['kind']!r}")
    inference = tuple(data.get("copybook_inference") or ())
    for rule in inference:
        if "pic" not in rule or not ("source" in rule or "source_by_digits" in rule):
            raise ConfigError(f"type map {path}: bad copybook_inference rule {rule!r}")
    target = str(data.get("target") or "azuresql")
    if target not in ("azuresql", "postgresql"):
        raise ConfigError(f"type map {path}: unknown target {target!r}")
    return TypeMap(rules=tuple(rules), inference=inference, target=target)


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    field: Field
    source_type: str
    target_type: str
    kind: Kind
    is_key: bool
    encoding: str  # codec used to decode the bytes of char/date8 columns
    trim: bool  # target-side right trim (NVARCHAR)
    date_format: str | None
    value_map: dict[str, str]
    target_precision: int
    target_scale: int
    target_length: int

    @property
    def target_columns(self) -> tuple[str, ...]:
        if self.kind == "timestamp12":
            return (self.name, f"{self.name}_NANOS_TAIL")
        return (self.name,)


def _apply_override(
    spec_kind: Kind, target_type: str, ov: TypeOverride, name: str, typemap: TypeMap
) -> tuple[str, Kind]:
    if ov.target_type is None:
        return target_type, spec_kind
    new = _parse_type(ov.target_type)
    base = new.base
    if base in ("NCHAR", "NVARCHAR", "CHAR", "VARCHAR", "CHARACTER", "CHARACTER VARYING", "TEXT"):
        kind: Kind = "char"
    elif base in ("DECIMAL", "NUMERIC"):
        kind = "decimal"
    elif base in ("SMALLINT", "INT", "INTEGER", "BIGINT", "TINYINT"):
        kind = "int"
    elif base == "DATE":
        kind = "date8"
    elif base in ("DATETIME2", "TIMESTAMP"):
        kind = "timestamp12"
    elif base in ("VARBINARY", "BINARY", "BYTEA"):
        kind = "binary"
    else:
        raise ConfigError(f"column {name}: unsupported target_type override {ov.target_type!r}")
    return typemap.spell_target_type(new), kind


def build_column_specs(
    table: TableConfig, copybook: Copybook, typemap: TypeMap, text_encoding: str
) -> list[ColumnSpec]:
    specs: list[ColumnSpec] = []
    non_filler = [f for f in copybook.fields if not f.filler]
    unmapped = [f.name for f in non_filler if f.name not in table.field_map]
    if unmapped:
        raise ConfigError(f"table {table.name}: copybook fields without field_map entry: {unmapped}")
    for cob_name, col in table.field_map.items():
        field = copybook.field(cob_name)
        source_type = typemap.infer_source_type(field)
        ov = table.type_overrides.get(col, TypeOverride())
        if ov.encoding and _parse_type(source_type).base == "CHAR":
            source_type = f"{source_type} FOR BIT DATA"
        target_type, kind = typemap.map_source_type(source_type)
        if kind == "binary" and ov.encoding:
            # FOR BIT DATA decoded to text: mapped like CHAR(n) unless target_type says otherwise
            n = _parse_type(source_type).length
            target_type, kind = f"{typemap.char_type}({n})", "char"
        target_type, kind = _apply_override(kind, target_type, ov, col, typemap)
        if ov.format:
            if ov.format != "YYYYMMDD":
                raise ConfigError(f"column {col}: unsupported format {ov.format!r}")
            if kind != "date8":
                raise ConfigError(f"column {col}: format YYYYMMDD requires target_type DATE")
        elif kind == "date8" and field.length != 8:
            raise ConfigError(f"column {col}: DATE target requires an 8-byte CHAR field or format YYYYMMDD")
        parsed_target = _parse_type(target_type)
        if kind == "decimal":
            if parsed_target.precision > 38:
                raise ConfigError(f"column {col}: target {target_type} exceeds DECIMAL(38)")
            if field.scale != parsed_target.scale:
                raise ConfigError(f"column {col}: copybook scale {field.scale} != target scale {parsed_target.scale}")
        trim = ov.trim == "right" or (
            kind == "char" and parsed_target.base in ("NVARCHAR", "VARCHAR", "TEXT") and ov.trim is None
        )
        if ov.trim == "none":
            trim = False
        specs.append(
            ColumnSpec(
                name=col,
                field=field,
                source_type=source_type,
                target_type=target_type,
                kind=kind,
                is_key=col in table.key_columns,
                encoding=ov.encoding or text_encoding,
                trim=trim,
                date_format=ov.format,
                value_map=dict(ov.value_map or {}),
                target_precision=parsed_target.precision,
                target_scale=parsed_target.scale,
                target_length=parsed_target.length,
            )
        )
    return specs
