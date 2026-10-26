"""COBOL copybook parser: elementary PIC items -> byte slices (CONTRACTS.md §5.2, FIELD-DERIVATION.md)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .errors import ConfigError

Usage = Literal["display", "comp", "comp-3"]

_ITEM_RE = re.compile(
    r"^\s*(?P<level>\d{2})\s+(?P<name>[A-Z0-9-]+)"
    r"(?:\s+PIC(?:TURE)?\s+(?:IS\s+)?(?P<pic>[SX9V()0-9]+))?"
    r"(?:\s+(?:USAGE\s+(?:IS\s+)?)?(?P<usage>COMP(?:UTATIONAL)?(?:-[35])?|BINARY|PACKED-DECIMAL))?"
    r"\s*\.\s*$",
    re.IGNORECASE,
)
_PIC_X_RE = re.compile(r"^X(?:\((\d+)\))?$")
_PIC_9_RE = re.compile(r"^(S)?9(?:\((\d+)\))?(?:V9(?:\((\d+)\))?)?$")


@dataclass(frozen=True)
class Field:
    name: str
    pic: str
    usage: Usage
    offset: int
    length: int
    is_numeric: bool
    signed: bool
    digits: int  # total digits (numeric only)
    scale: int  # fraction digits (numeric only)

    @property
    def filler(self) -> bool:
        return self.name.upper() == "FILLER"

    @property
    def pic_key(self) -> str:
        """Canonical 'PIC + usage' key used by the type map's copybook_inference, e.g. 'S9(23)V9(8) COMP-3'."""
        usage = {"display": "", "comp": " COMP", "comp-3": " COMP-3"}[self.usage]
        return f"{self.pic}{usage}"

    def slice(self, record: bytes) -> bytes:
        return record[self.offset : self.offset + self.length]


@dataclass(frozen=True)
class Copybook:
    name: str
    fields: tuple[Field, ...]
    record_length: int

    def field(self, name: str) -> Field:
        for f in self.fields:
            if f.name == name:
                return f
        raise ConfigError(f"copybook {self.name}: no field named {name!r}")


def _normalize_usage(raw: str | None) -> Usage:
    if raw is None:
        return "display"
    u = raw.upper()
    if u in ("COMP", "COMPUTATIONAL", "COMP-5", "COMPUTATIONAL-5", "BINARY"):
        return "comp"
    if u in ("COMP-3", "COMPUTATIONAL-3", "PACKED-DECIMAL"):
        return "comp-3"
    raise ConfigError(f"unsupported USAGE {raw!r}")


def _pic_layout(pic: str, usage: Usage) -> tuple[int, bool, bool, int, int]:
    """Return (length, is_numeric, signed, digits, scale)."""
    m = _PIC_X_RE.match(pic)
    if m:
        if usage != "display":
            raise ConfigError(f"PIC {pic} cannot be {usage}")
        return int(m.group(1) or 1), False, False, 0, 0
    m = _PIC_9_RE.match(pic)
    if not m:
        raise ConfigError(f"unsupported PIC clause {pic!r} (supported: X(n), S9(n), S9(n)V9(m) with COMP/COMP-3)")
    signed = m.group(1) is not None
    int_digits = int(m.group(2) or 1)
    scale = int(m.group(3) or 1) if "V" in pic else 0
    digits = int_digits + scale
    if usage == "comp":
        if digits <= 4:
            length = 2
        elif digits <= 9:
            length = 4
        elif digits <= 18:
            length = 8
        else:
            raise ConfigError(f"PIC {pic} COMP exceeds 18 digits")
    elif usage == "comp-3":
        length = (digits + 2) // 2  # ceil((digits + 1) / 2)
    else:
        length = digits
    return length, True, signed, digits, scale


def _iter_source_lines(text: str):
    for raw in text.splitlines():
        line = raw.rstrip("\n")
        # Fixed-form comment: '*' in column 7. Also tolerate free-form '*' comments.
        if len(line) >= 7 and line[6] in "*/":
            continue
        if line.strip().startswith("*"):
            continue
        # Fixed-form: ignore sequence area (1-6) and identification area (73-80).
        if len(line) > 72 and line[:6].strip().isdigit():
            line = line[6:72]
        if not line.strip():
            continue
        yield line


def parse_copybook(path: Path, expected_record_length: int | None = None) -> Copybook:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as e:
        raise ConfigError(f"copybook not found: {path}") from e
    fields: list[Field] = []
    offset = 0
    # Join continuation lines: a statement ends with '.'
    buf = ""
    for line in _iter_source_lines(text):
        buf = (buf + " " + line.strip()).strip()
        if not buf.endswith("."):
            continue
        stmt, buf = buf, ""
        m = _ITEM_RE.match(stmt)
        if not m:
            raise ConfigError(f"copybook {path.name}: cannot parse {stmt!r}")
        level = int(m.group("level"))
        pic = m.group("pic")
        if pic is None:
            if level in (1, 66, 77, 88) or m.group("usage") is None:
                continue  # group item
            raise ConfigError(f"copybook {path.name}: item {m.group('name')} has USAGE without PIC")
        if "OCCURS" in stmt.upper():
            raise ConfigError(f"copybook {path.name}: OCCURS is not supported")
        usage = _normalize_usage(m.group("usage"))
        pic_u = pic.upper()
        length, is_numeric, signed, digits, scale = _pic_layout(pic_u, usage)
        fields.append(
            Field(
                name=m.group("name").upper(),
                pic=pic_u,
                usage=usage,
                offset=offset,
                length=length,
                is_numeric=is_numeric,
                signed=signed,
                digits=digits,
                scale=scale,
            )
        )
        offset += length
    if buf:
        raise ConfigError(f"copybook {path.name}: unterminated statement {buf!r}")
    if not fields:
        raise ConfigError(f"copybook {path.name}: no elementary items")
    if expected_record_length is not None and offset != expected_record_length:
        raise ConfigError(
            f"copybook {path.name}: elementary items total {offset} bytes but manifest record_length "
            f"is {expected_record_length}"
        )
    return Copybook(name=path.name, fields=tuple(fields), record_length=offset)
