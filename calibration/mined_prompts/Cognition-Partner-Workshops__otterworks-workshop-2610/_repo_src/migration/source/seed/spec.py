"""Constants, PRNG and encoders shared by every table generator (SEED-SPEC.md §2, §4, §5)."""

from __future__ import annotations

import calendar
import hashlib
import uuid
from datetime import date, timedelta
from math import gcd

M64 = (1 << 64) - 1
SEED = 0x4F54544552574B53  # "OTTERWKS"
STREAM = {"RETNPLCY": 1, "DOCARCH": 2, "FILEAUD": 3, "PLANTED": 4}

LRECL = {"RETNPLCY": 128, "DOCARCH": 256, "FILEAUD": 160}

CUTOFF_TEXT = "2019-01-01-00.00.00.000000000000"
CUTOFF_SECS = calendar.timegm((2019, 1, 1, 0, 0, 0))
T2012 = calendar.timegm((2012, 1, 1, 0, 0, 0))
T2019 = CUTOFF_SECS
T2026H2 = calendar.timegm((2026, 7, 1, 0, 0, 0))
YEAR3_SECS = 3 * 365 * 86400

# Full-scale row counts (§3, §5, §6).
DOCARCH_GENERATED = 1_199_958
DOCARCH_S = 179_963
DOCARCH_R = 300_000
DOCARCH_MULT = 1_000_003
DOCARCH_ADD = 424_242

FILEAUD_GENERATED = 4_099_995
FILEAUD_SELECTED = 619_995
FILEAUD_MULT = 7_777_777
FILEAUD_ADD = 13

# RETNPLCY fixed list (§4): (code, description, years, successor, active)
RETNPLCY_ROWS: list[tuple[str, str, int, str, str]] = [
    ("FIN7", "Financial records", 7, "", "Y"),
    ("LGL7", "Legal correspondence", 7, "", "Y"),
    ("HRS7", "Personnel files", 7, "", "Y"),
    ("TAX7", "Tax filings", 7, "", "Y"),
    ("AUD7", "Internal audit workpapers", 7, "", "Y"),
    ("F07R", "Financial records (retired code)", 7, "FIN7", "N"),
    ("L07R", "Legal correspondence (retired code)", 7, "LGL7", "N"),
    ("H07R", "Personnel files (retired code)", 7, "HRS7", "N"),
    ("OPS1", "Operations records", 1, "", "Y"),
    ("OPS3", "Operations records", 3, "", "Y"),
    ("OPS5", "Operations records", 5, "", "Y"),
    ("MKT1", "Marketing material", 1, "", "Y"),
    ("MKT2", "Marketing material", 2, "", "Y"),
    ("MKT3", "Marketing material", 3, "", "Y"),
    ("ENG3", "Engineering documents", 3, "", "Y"),
    ("ENG5", "Engineering documents", 5, "", "Y"),
    ("ENG9", "Engineering documents", 9, "", "Y"),
    ("SLS3", "Sales records", 3, "", "Y"),
    ("SLS5", "Sales records", 5, "", "Y"),
    ("SUP1", "Support tickets", 1, "", "Y"),
    ("SUP2", "Support tickets", 2, "", "Y"),
    ("SUP3", "Support tickets", 3, "", "Y"),
    ("PRJ3", "Project files", 3, "", "Y"),
    ("PRJ5", "Project files", 5, "", "Y"),
    ("PRJ9", "Project files", 9, "", "Y"),
    ("CMP5", "Compliance evidence", 5, "", "Y"),
    ("CMP9", "Compliance evidence", 9, "", "Y"),
    ("SEC3", "Security logs", 3, "", "Y"),
    ("SEC7", "Security logs", 7, "", "Y"),
    ("INS9", "Insurance policies", 9, "", "Y"),
    ("PHI9", "Health information", 9, "", "Y"),
    ("PERM", "Permanent records", 99, "", "Y"),
    ("TMP0", "Temporary working files", 0, "", "Y"),
    ("TMP1", "Temporary working files", 1, "", "Y"),
    ("EML2", "Email archives", 2, "", "Y"),
    ("EML5", "Email archives", 5, "", "Y"),
    ("CHT1", "Chat transcripts", 1, "", "Y"),
    ("CHT2", "Chat transcripts", 2, "", "Y"),
    ("BRD9", "Board minutes", 9, "", "Y"),
    ("GOV9", "Governance records", 9, "", "Y"),
]
assert len(RETNPLCY_ROWS) == 40

RETENTION_YEARS = {code: years for code, _, years, _, _ in RETNPLCY_ROWS}
SUCCESSOR = {code: succ for code, _, _, succ, _ in RETNPLCY_ROWS}
SET_ACTIVE = ["FIN7", "LGL7", "HRS7", "TAX7", "AUD7"]
CLOSED_7Y = SET_ACTIVE + ["F07R", "L07R"]
OTHER_ACTIVE = [code for code, _, _, _, active in RETNPLCY_ROWS if active == "Y" and code not in SET_ACTIVE]
assert len(OTHER_ACTIVE) == 32

SURNAMES = ["ADAMS", "BROOKS", "CHEN", "DIAZ", "EVANS", "FOSTER", "GARCIA", "HUGHES",
            "IBRAHIM", "JONES", "KOWALSKI", "LOPEZ", "MURPHY", "NGUYEN", "OKAFOR", "PATEL"]
SOURCE_SYS = ["OWD", "OWF", "IMP"]
EVENT_TYPES = ["VIEW", "DNLD", "HOLD", "RLSE", "DISP", "XPRT"]


def system_of_record_class(code: str) -> str:
    """Retired code -> active successor per RETNPLCY (the MIG-07 system of record)."""
    return SUCCESSOR.get(code) or code


# --- PRNG (§2) -------------------------------------------------------------------------------

def splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & M64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & M64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & M64
    return x ^ (x >> 31)


_STREAM_BASE = {name: splitmix64(SEED ^ sid) for name, sid in STREAM.items()}


def h(stream: str, index: int, field: int) -> int:
    """Uniform 64-bit value for (stream, 0-based row index, copybook field number)."""
    return splitmix64(_STREAM_BASE[stream] ^ ((index << 6) | field))


def pick(stream: str, index: int, field: int, n: int) -> int:
    return h(stream, index, field) % n


# --- Encoders (FIELD-DERIVATION.md §5) --------------------------------------------------------

def ascii_field(text: str, width: int) -> bytes:
    raw = text.encode("ascii")
    if len(raw) > width:
        raise ValueError(f"{text!r} exceeds {width} bytes")
    return raw.ljust(width, b" ")


def cp037_field(text: str, width: int) -> bytes:
    raw = text.encode("cp037")
    if len(raw) > width:
        raise ValueError(f"{text!r} exceeds {width} bytes")
    return raw.ljust(width, b"\x40")


def comp(value: int, size: int) -> bytes:
    return value.to_bytes(size, "big", signed=True)


def comp3_31_8(scaled: int) -> bytes:
    """DECIMAL(31,8) packed decimal from an integer scaled by 10**8 (16 bytes, sign C/D)."""
    digits = f"{abs(scaled):031d}"
    if len(digits) != 31:
        raise ValueError(f"{scaled} does not fit DECIMAL(31,8)")
    return bytes.fromhex(digits + ("D" if scaled < 0 else "C"))


def decimal8(scaled: int) -> str:
    sign = "-" if scaled < 0 else ""
    scaled = abs(scaled)
    return f"{sign}{scaled // 10**8}.{scaled % 10**8:08d}"


def ts12(secs: int, frac: int) -> str:
    """Db2 TIMESTAMP(12) text for whole seconds since the epoch (UTC) and a 12-digit fraction."""
    y, mo, d, hh, mm, ss = _gmtime(secs)
    return f"{y:04d}-{mo:02d}-{d:02d}-{hh:02d}.{mm:02d}.{ss:02d}.{frac:012d}"


def _gmtime(secs: int) -> tuple[int, int, int, int, int, int]:
    days, rem = divmod(secs, 86400)
    d = date(1970, 1, 1) + timedelta(days=days)
    hh, rem = divmod(rem, 3600)
    mm, ss = divmod(rem, 60)
    return d.year, d.month, d.day, hh, mm, ss


def add_years_yyyymmdd(secs: int, years: int) -> str:
    y, mo, d, *_ = _gmtime(secs)
    y += years
    if mo == 2 and d == 29 and not calendar.isleap(y):
        d = 28
    return f"{y:04d}{mo:02d}{d:02d}"


def uuid_text(hi: int, lo: int) -> str:
    return str(uuid.UUID(int=(hi << 64) | lo, version=4))


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --- Scaling ---------------------------------------------------------------------------------

class Sizes:
    """Row cohort sizes; `scale=1.0` is the binding full seed, smaller scales keep the same shape."""

    def __init__(self, scale: float = 1.0) -> None:
        if scale <= 0 or scale > 1:
            raise ValueError("--scale must be in (0, 1]")
        self.scale = scale
        s = (lambda n: n) if scale == 1.0 else (lambda n: max(1, round(n * scale)))
        self.docarch_generated = s(DOCARCH_GENERATED)
        self.docarch_s = s(DOCARCH_S)
        self.docarch_r = s(DOCARCH_R)
        if self.docarch_s + self.docarch_r >= self.docarch_generated:
            raise ValueError("scale too small for the DOCARCH cohorts")
        self.docarch_ns = self.docarch_generated - self.docarch_s
        self.fileaud_generated = s(FILEAUD_GENERATED)
        self.fileaud_selected = s(FILEAUD_SELECTED)
        if self.fileaud_selected >= self.fileaud_generated:
            raise ValueError("scale too small for the FILEAUD cohorts")
        for mult, n in ((DOCARCH_MULT, self.docarch_generated), (FILEAUD_MULT, self.fileaud_generated)):
            if gcd(mult, n) != 1:
                raise ValueError(f"scale {scale} gives row count {n} not coprime with {mult}; choose another scale")

    def docarch_cohort(self, g: int) -> str:
        p = (DOCARCH_MULT * g + DOCARCH_ADD) % self.docarch_generated
        if p < self.docarch_s:
            return "S"
        if p < self.docarch_s + self.docarch_r:
            return "R"
        return "O"

    def docarch_p(self, g: int) -> int:
        return (DOCARCH_MULT * g + DOCARCH_ADD) % self.docarch_generated

    def docarch_g_for_p(self, p: int) -> int:
        inv = pow(DOCARCH_MULT, -1, self.docarch_generated)
        return ((p - DOCARCH_ADD) * inv) % self.docarch_generated

    def fileaud_q(self, m: int) -> int:
        return (FILEAUD_MULT * m + FILEAUD_ADD) % self.fileaud_generated
