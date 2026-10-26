"""Per-record conversion of fixed-width copybook bytes to target values (CONTRACTS.md §5.2, §9.4)."""

from __future__ import annotations

import codecs
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal, InvalidOperation

from .copybook import Field
from .errors import ConfigError
from .typemap import ColumnSpec

TS12_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})-(\d{2})\.(\d{2})\.(\d{2})\.(\d{12})$")
DATE8_RE = re.compile(r"^(\d{4})(\d{2})(\d{2})$")

Value = str | int | Decimal | bytes | None


@dataclass(frozen=True)
class Timestamp12:
    """A Db2 TIMESTAMP(12): canonical 32-char text, DATETIME2(7) part and the 5-digit nanos tail."""

    text: str
    dt: datetime
    nanos_tail: int

    @classmethod
    def parse(cls, text: str) -> Timestamp12:
        m = TS12_RE.match(text)
        if not m:
            raise ValueError(f"not a TIMESTAMP(12) literal: {text!r}")
        y, mo, d, h, mi, s, frac = m.groups()
        dt = datetime(int(y), int(mo), int(d), int(h), int(mi), int(s), microsecond=int(frac[:6]))
        # DATETIME2(7) keeps 7 fraction digits; python datetime keeps 6. The 7th digit is stored with the tail
        # by the target driver: we carry the full canonical text and a 7-digit fraction separately.
        return cls(text=text, dt=dt, nanos_tail=int(frac[7:]))

    @property
    def fraction7(self) -> str:
        return self.text[20:27]

    @property
    def datetime2_literal(self) -> str:
        """'YYYY-MM-DD HH:MM:SS.fffffff' accepted by DATETIME2(7)."""
        t = self.text
        return f"{t[0:10]} {t[11:13]}:{t[14:16]}:{t[17:19]}.{self.fraction7}"

    @classmethod
    def from_target(cls, dt2_text: str, nanos_tail: int) -> Timestamp12:
        """Rebuild the canonical text from 'YYYY-MM-DD HH:MM:SS.fffffff' and the tail."""
        d, t = dt2_text[:10], dt2_text[11:]
        hh, mm, rest = t.split(":", 2)
        ss, frac = rest.split(".", 1) if "." in rest else (rest, "")
        frac = (frac + "0000000")[:7]
        text = f"{d}-{hh}.{mm}.{ss}.{frac}{nanos_tail:05d}"
        return cls.parse(text)


@dataclass(frozen=True)
class FieldError:
    column: str
    rule: str
    error: str
    field_bytes: bytes
    sqlstate: str | None = None
    native_error: int | None = None


@dataclass
class ConvertedRecord:
    source_key: str
    raw: bytes
    values: dict[str, Value | Timestamp12] = field(default_factory=dict)
    source_text: dict[str, str | int | Decimal | bytes] = field(default_factory=dict)
    error: FieldError | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def target_row(self) -> dict[str, Value | Timestamp12 | date]:
        """Flatten values into target column -> value. Timestamp12 stays whole (drivers bind its 7-digit
        DATETIME2 literal, python datetime would lose the 7th fraction digit) plus `<col>_NANOS_TAIL`."""
        out: dict[str, Value | Timestamp12 | date] = {}
        for col, v in self.values.items():
            out[col] = v
            if isinstance(v, Timestamp12):
                out[f"{col}_NANOS_TAIL"] = v.nanos_tail
        return out


def _decode(raw: bytes, encoding: str) -> str:
    return codecs.decode(raw, encoding, "strict")


def _control_byte(text: str, raw: bytes) -> tuple[int, int] | None:
    for i, ch in enumerate(text):
        o = ord(ch)
        if o < 0x20 or 0x7F <= o <= 0x9F:
            return i, raw[i] if i < len(raw) else o
    return None


def unpack_comp3(raw: bytes, scale: int) -> Decimal:
    digits = []
    for b in raw:
        digits.append(b >> 4)
        digits.append(b & 0x0F)
    sign_nibble = digits.pop()
    if any(d > 9 for d in digits):
        raise ValueError(f"invalid packed decimal digit in X'{raw.hex().upper()}'")
    if sign_nibble in (0xC, 0xF, 0xA, 0xE):
        sign = ""
    elif sign_nibble in (0xD, 0xB):
        sign = "-"
    else:
        raise ValueError(f"invalid packed decimal sign nibble X'{sign_nibble:X}' in X'{raw.hex().upper()}'")
    s = "".join(str(d) for d in digits)
    if scale:
        s = s[:-scale] + "." + s[-scale:]
    return Decimal(sign + s)


def pack_comp3(value: Decimal, length: int, scale: int) -> bytes:
    q = value.quantize(Decimal(1).scaleb(-scale)) if scale else value.quantize(Decimal(1))
    digits = f"{abs(q):f}".replace(".", "")
    total = length * 2 - 1
    if len(digits) > total:
        raise ValueError(f"{value} does not fit in {length}-byte packed decimal")
    digits = digits.rjust(total, "0")
    nibbles = [int(c) for c in digits] + [0xD if q < 0 else 0xC]
    return bytes((nibbles[i] << 4) | nibbles[i + 1] for i in range(0, len(nibbles), 2))


def _decimal_fits(value: Decimal, precision: int, scale: int) -> bool:
    if not value.is_finite():
        return False
    _, digits, exponent = value.as_tuple()
    assert isinstance(exponent, int)
    integer_digits = len(digits) + exponent
    return integer_digits <= precision - scale


def source_key_of(record: bytes, specs: list[ColumnSpec]) -> str:
    parts = []
    for s in specs:
        if s.is_key:
            parts.append(_decode(s.field.slice(record), s.encoding))
    return "|".join(parts)


def convert_record(record: bytes, specs: list[ColumnSpec]) -> ConvertedRecord:
    """Convert one record. The first failing field stops conversion (rule priority §9.4 at the field level)."""
    rec = ConvertedRecord(source_key=source_key_of(record, specs), raw=record)
    for spec in specs:
        raw = spec.field.slice(record)
        try:
            value, text = _convert_field(raw, spec)
        except _RejectError as r:
            rec.error = FieldError(
                column=spec.name,
                rule=r.rule,
                error=r.message,
                field_bytes=raw,
                sqlstate=r.sqlstate,
            )
            return rec
        rec.values[spec.name] = value
        rec.source_text[spec.name] = text
    return rec


class _RejectError(Exception):
    def __init__(self, rule: str, message: str, sqlstate: str | None = None):
        super().__init__(message)
        self.rule = rule
        self.message = message
        self.sqlstate = sqlstate


def _convert_field(raw: bytes, spec: ColumnSpec) -> tuple[Value | Timestamp12, str | int | Decimal | bytes]:
    f: Field = spec.field
    kind = spec.kind
    if kind == "binary":
        return raw, raw
    if kind in ("char", "date8", "timestamp12"):
        try:
            text = _decode(raw, spec.encoding)
        except UnicodeDecodeError as e:
            raise _RejectError(
                "CCSID_UNMAPPABLE",
                f"{spec.encoding} byte X'{raw[e.start]:02X}' at offset {e.start} has no UTF-8 mapping",
            ) from e
        if kind == "char":
            ctl = _control_byte(text, raw)
            if ctl is not None:
                off, byte = ctl
                label = "CCSID037" if spec.encoding.lower() == "cp037" else spec.encoding
                raise _RejectError(
                    "CCSID_UNMAPPABLE", f"{label} byte X'{byte:02X}' at offset {off} has no UTF-8 mapping"
                )
            source_text = text
            value = text.rstrip(" ") if spec.trim else text
            if spec.value_map:
                key = value.rstrip(" ")
                if key in spec.value_map:
                    mapped = spec.value_map[key]
                    value = mapped if spec.trim else mapped.ljust(len(value))
            if spec.target_length and len(value) > spec.target_length:
                raise _RejectError(
                    "STRING_TRUNCATION",
                    f"value of length {len(value)} exceeds {spec.target_type}",
                    sqlstate="22001",
                )
            return value, source_text
        if kind == "date8":
            m = DATE8_RE.match(text)
            if not m:
                raise _RejectError(
                    "DATE_INVALID",
                    f"DATE text X'{raw.hex().upper()}' rejected by target: Conversion failed when converting "
                    f"date and/or time from character string.",
                    sqlstate="22007",
                )
            try:
                return date(int(m.group(1)), int(m.group(2)), int(m.group(3))), text
            except ValueError as e:
                raise _RejectError("DATE_INVALID", f"DATE text {text!r} is not a calendar date: {e}", "22007") from e
        try:
            return Timestamp12.parse(text), text
        except ValueError as e:
            raise _RejectError(
                "TIMESTAMP_INVALID", f"TIMESTAMP(12) text X'{raw.hex().upper()}' is invalid", "22007"
            ) from e
    if kind == "int":
        if f.usage == "comp":
            value = int.from_bytes(raw, "big", signed=True)
        elif f.usage == "display":
            try:
                value = int(_decode(raw, spec.encoding))
            except (UnicodeDecodeError, ValueError) as e:
                raise _RejectError(
                    "NUMERIC_INVALID", f"display numeric X'{raw.hex().upper()}' is invalid", "22018"
                ) from e
        else:
            try:
                value = int(unpack_comp3(raw, 0))
            except ValueError as e:
                raise _RejectError("PACKED_INVALID", str(e), "22018") from e
        limits = {"SMALLINT": 15, "INT": 31, "INTEGER": 31, "BIGINT": 63, "TINYINT": 8}
        base = spec.target_type.split("(")[0]
        bits = limits.get(base, 63)
        if not (-(1 << bits) <= value < (1 << bits)):
            raise _RejectError("INTEGER_OVERFLOW", f"value {value} exceeds {spec.target_type}", "22003")
        return value, value
    if kind == "decimal":
        if f.usage == "comp-3":
            try:
                value_d = unpack_comp3(raw, f.scale)
            except ValueError as e:
                raise _RejectError("PACKED_INVALID", str(e), "22018") from e
        elif f.usage == "display":
            try:
                s = _decode(raw, spec.encoding)
                value_d = Decimal(s[: len(s) - f.scale] + "." + s[len(s) - f.scale :]) if f.scale else Decimal(s)
            except (UnicodeDecodeError, InvalidOperation) as e:
                raise _RejectError(
                    "NUMERIC_INVALID", f"display numeric X'{raw.hex().upper()}' is invalid", "22018"
                ) from e
        else:
            value_d = Decimal(int.from_bytes(raw, "big", signed=True)).scaleb(-f.scale)
        if not _decimal_fits(value_d, spec.target_precision, spec.target_scale):
            raise _RejectError(
                "DECIMAL_OVERFLOW",
                f"value {value_d} exceeds {spec.target_type}",
                sqlstate="22003",
            )
        return value_d, value_d
    raise ConfigError(f"column {spec.name}: unsupported kind {kind}")


def encode_record(values: dict[str, Value | Timestamp12], specs: list[ColumnSpec], record_length: int) -> bytes:
    """Inverse of convert_record for the built-in fixed-width writer and test fixtures.

    `values` holds *source* values keyed by column name: str for text (padding added), bytes for verbatim
    field bytes, int/Decimal for numerics, str/Timestamp12 for timestamps.
    """
    out = bytearray(b" " * record_length)
    for spec in specs:
        f = spec.field
        v = values.get(spec.name)
        if isinstance(v, bytes):
            b = v.ljust(f.length, b"\x40" if spec.encoding.lower() == "cp037" else b" ")
        elif f.is_numeric:
            if f.usage == "comp":
                b = int(v or 0).to_bytes(f.length, "big", signed=True)
            elif f.usage == "comp-3":
                b = pack_comp3(Decimal(v or 0), f.length, f.scale)
            else:
                q = Decimal(v or 0).quantize(Decimal(1).scaleb(-f.scale)) if f.scale else Decimal(v or 0)
                b = f"{abs(q):f}".replace(".", "").rjust(f.length, "0").encode(spec.encoding)
        else:
            text = v.text if isinstance(v, Timestamp12) else ("" if v is None else str(v))
            b = codecs.encode(text.ljust(f.length), spec.encoding)
        if len(b) != f.length:
            raise ValueError(f"column {spec.name}: value {v!r} does not fit {f.length} bytes")
        out[f.offset : f.offset + f.length] = b
    return bytes(out)
