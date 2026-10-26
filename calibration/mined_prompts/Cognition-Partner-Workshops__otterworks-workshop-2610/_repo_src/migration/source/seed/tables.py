"""Row generators for RETNPLCY, DOCARCH, FILEAUD and the planted MIG-01..07 rows (SEED-SPEC.md §4-§7)."""

from __future__ import annotations

from dataclasses import dataclass, replace

from . import spec
from .spec import (
    EVENT_TYPES, OTHER_ACTIVE, RETENTION_YEARS, RETNPLCY_ROWS, SET_ACTIVE, SOURCE_SYS, SURNAMES,
    T2012, T2019, T2026H2, YEAR3_SECS, Sizes, add_years_yyyymmdd, ascii_field, comp, comp3_31_8,
    cp037_field, pick, h, sha256_hex, ts12, uuid_text,
)

COHORT_RANGE = {"S": (T2012, T2019 - T2012), "R": (T2019, T2026H2 - T2019), "O": (T2012, T2026H2 - T2012)}


@dataclass(frozen=True)
class DocarchRow:
    arch_key: bytes            # 16 bytes as stored (MIG-04 keys carry their padding)
    doc_id: str
    version_no: int
    retention_class: str
    last_access_secs: int
    last_access_frac: int
    storage_charge: int        # scaled by 10**8
    unit_rate: int             # scaled by 10**8
    owner_name: bytes          # 40 raw bytes (cp037)
    disposition_dt: bytes      # 8 raw bytes (cp037 digits or low-values)
    legal_hold_flag: str
    checksum_alg: str
    content_sha256: str
    byte_size: int
    source_sys: str
    cohort: str                # S / R / O / P (planted)

    @property
    def last_access_ts(self) -> str:
        return ts12(self.last_access_secs, self.last_access_frac)

    def record(self) -> bytes:
        rec = b"".join((
            self.arch_key,
            ascii_field(self.doc_id, 36),
            comp(self.version_no, 2),
            ascii_field(self.retention_class, 4),
            ascii_field(self.last_access_ts, 32),
            comp3_31_8(self.storage_charge),
            comp3_31_8(self.unit_rate),
            self.owner_name,
            self.disposition_dt,
            ascii_field(self.legal_hold_flag, 1),
            ascii_field(self.checksum_alg, 8),
            ascii_field(self.content_sha256, 64),
            comp(self.byte_size, 8),
            ascii_field(self.source_sys, 3),
            b"  ",
        ))
        assert len(rec) == 256, len(rec)
        return rec


def docarch_base(stream: str, index: int, arch_key: str, cohort: str) -> DocarchRow:
    """§5 field rules for one row. Planted rows use cohort S ranges before their overrides."""
    lo, span = COHORT_RANGE[cohort]
    if cohort == "O":
        cls = OTHER_ACTIVE[pick(stream, index, 4, 32)]
    else:
        cls = SET_ACTIVE[pick(stream, index, 4, 5)]
    secs = lo + pick(stream, index, 5, span)
    frac = pick(stream, index, 42, 10**12)
    byte_size = 1024 + pick(stream, index, 13, 50_000_000)
    unit_rate = 1 + pick(stream, index, 7, 99_999)
    owner = SURNAMES[pick(stream, index, 8, 16)] + ", " + chr(65 + pick(stream, index, 43, 26)) + "."
    hold = "Y" if (cohort == "O" and pick(stream, index, 10, 50) == 0) else "N"
    key_bytes = ascii_field(arch_key, 16)
    return DocarchRow(
        arch_key=key_bytes,
        doc_id=uuid_text(h(stream, index, 2), h(stream, index, 41)),
        version_no=1 + pick(stream, index, 3, 9),
        retention_class=cls,
        last_access_secs=secs,
        last_access_frac=frac,
        storage_charge=byte_size * unit_rate,
        unit_rate=unit_rate,
        owner_name=cp037_field(owner, 40),
        disposition_dt=cp037_field(add_years_yyyymmdd(secs, RETENTION_YEARS[cls]), 8),
        legal_hold_flag=hold,
        checksum_alg="SHA256",
        content_sha256=sha256_hex(key_bytes),
        byte_size=byte_size,
        source_sys=SOURCE_SYS[pick(stream, index, 14, 3)],
        cohort=cohort,
    )


def docarch_key(g: int) -> str:
    return f"DA{g + 1:014d}"


def docarch_generated(sizes: Sizes, g: int) -> DocarchRow:
    return docarch_base("DOCARCH", g, docarch_key(g), sizes.docarch_cohort(g))


# --- RETNPLCY (§4) ---------------------------------------------------------------------------

RETNPLCY_EFFECTIVE_TS = "2010-01-01-00.00.00.000000000000"


@dataclass(frozen=True)
class RetnplcyRow:
    policy_code: str
    policy_desc: str
    retention_years: int
    successor_code: str
    active_flag: str
    effective_ts: str = RETNPLCY_EFFECTIVE_TS

    @property
    def disposition_action(self) -> str:
        if self.policy_code == "PERM":
            return "PERM"
        return "REVW" if self.retention_years == 9 else "DEST"

    def record(self) -> bytes:
        rec = b"".join((
            ascii_field(self.policy_code, 4),
            ascii_field(self.policy_desc, 60),
            comp(self.retention_years, 2),
            ascii_field(self.successor_code, 4),
            ascii_field(self.active_flag, 1),
            ascii_field(self.disposition_action, 4),
            ascii_field(self.effective_ts, 32),
            b" " * 21,
        ))
        assert len(rec) == 128
        return rec


def retnplcy_record(code: str, desc: str, years: int, successor: str, active: str) -> bytes:
    return RetnplcyRow(code, desc, years, successor, active).record()


def retnplcy_rows() -> list[RetnplcyRow]:
    return [RetnplcyRow(*row) for row in RETNPLCY_ROWS]


def retnplcy_records() -> list[bytes]:
    return [row.record() for row in retnplcy_rows()]


# --- FILEAUD (§6) ----------------------------------------------------------------------------

@dataclass(frozen=True)
class FileaudRow:
    audit_key: str
    arch_key: bytes            # 16 bytes exactly as the parent stores them
    event_type: str
    event_ts: str              # TIMESTAMP(12) text
    actor_id: str
    retention_class: str
    client_ip: str
    detail_text: str
    disposition_code: str = "00"

    def record(self) -> bytes:
        rec = b"".join((
            ascii_field(self.audit_key, 20),
            self.arch_key,
            ascii_field(self.event_type, 4),
            ascii_field(self.event_ts, 32),
            ascii_field(self.actor_id, 12),
            ascii_field(self.retention_class, 4),
            ascii_field(self.disposition_code, 2),
            ascii_field(self.client_ip, 15),
            ascii_field(self.detail_text, 40),
            b" " * 15,
        ))
        assert len(rec) == 160
        return rec


def fileaud_record(audit_key: str, arch_key: bytes, event_type: str, event_ts: str, actor: str,
                   ret_class: str, client_ip: str, detail: str) -> bytes:
    return FileaudRow(audit_key, arch_key, event_type, event_ts, actor, ret_class, client_ip, detail).record()


def fileaud_key(m: int) -> str:
    return f"FA{m + 1:018d}"


def fileaud_generated(sizes: Sizes, m: int, s_keys: list[int], ns_keys: list[int]) -> tuple[bytes, DocarchRow, int]:
    """Returns (record, parent row, event seconds). Parent is regenerated from its index."""
    row, parent, ev_secs = fileaud_generated_row(sizes, m, s_keys, ns_keys)
    return row.record(), parent, ev_secs


def fileaud_generated_row(
    sizes: Sizes, m: int, s_keys: list[int], ns_keys: list[int]
) -> tuple[FileaudRow, DocarchRow, int]:
    q = sizes.fileaud_q(m)
    if q < sizes.fileaud_selected:
        parent = docarch_generated(sizes, s_keys[q % len(s_keys)])
    else:
        parent = docarch_generated(sizes, ns_keys[(q - sizes.fileaud_selected) % len(ns_keys)])
    if parent.cohort == "R":
        ev_secs = T2019 + pick("FILEAUD", m, 44, (parent.last_access_secs - T2019) + 1)
    else:
        ev_secs = parent.last_access_secs - pick("FILEAUD", m, 44, YEAR3_SECS)
    event_type = EVENT_TYPES[pick("FILEAUD", m, 3, 6)]
    row = FileaudRow(
        fileaud_key(m), parent.arch_key, event_type, ts12(ev_secs, parent.last_access_frac),
        f"U{pick('FILEAUD', m, 5, 10**11):011d}", parent.retention_class,
        f"10.{pick('FILEAUD', m, 45, 256)}.{pick('FILEAUD', m, 46, 256)}.{pick('FILEAUD', m, 47, 254) + 1}",
        f"{event_type} v{parent.version_no}",
    )
    return row, parent, ev_secs


# --- Planted rows (§7) -----------------------------------------------------------------------

def _ts(text: str) -> tuple[int, int]:
    """Parse 'YYYY-MM-DD-HH.MM.SS.NNNNNNNNNNNN' into (epoch seconds, 12-digit fraction)."""
    import calendar
    y, mo, d, rest = text.split("-")
    hh, mm, ss, frac = rest.split(".")
    return calendar.timegm((int(y), int(mo), int(d), int(hh), int(mm), int(ss))), int(frac)


MIG07_F07R = [100_00000000, 200_00000000, 300_00000000, 400_00000000, 500_00000000, 1100_12345678]
MIG07_L07R = [433_00000000, 433_00000000, 433_00000000, 433_00000000, 433_00000000, 435_12345679]


def planted_docarch() -> list[tuple[str, DocarchRow]]:
    """The 42 planted DOCARCH rows as (failure class, row), in planted-ordinal order."""
    rows: list[tuple[str, DocarchRow]] = []

    def base(ordinal: int, key: str, cls: str, ts: str) -> DocarchRow:
        row = docarch_base("PLANTED", ordinal, key, "S")
        secs, frac = _ts(ts)
        return replace(row, retention_class=cls, last_access_secs=secs, last_access_frac=frac, cohort="P")

    for k in range(1, 6):
        row = base(k - 1, f"MIG01-{k:010d}", "FIN7", f"2016-03-0{k}-10.15.30.123456789012")
        owner = ("LOPEZ".encode("cp037") + b"\x3f" + ", M.".encode("cp037")).ljust(40, b"\x40")
        rows.append(("MIG-01", replace(row, owner_name=owner)))
    for k in range(1, 6):
        row = base(5 + k - 1, f"MIG02-{k:010d}", "TAX7", f"2015-06-1{k}-08.00.00.000000000001")
        rows.append(("MIG-02", replace(row, unit_rate=12345678901234_56789000 + k)))
    for k in range(1, 6):
        row = base(10 + k - 1, f"MIG03-{k:010d}", "LGL7", f"2014-11-2{k}-17.45.00.500000000000")
        rows.append(("MIG-03", replace(row, disposition_dt=b"\x00" * 8)))
    for k in range(1, 6):
        rows.append(("MIG-04", base(15 + k - 1, f"MIG04-{k:02d}", "HRS7", f"2013-02-0{k}-09.30.00.000000000000")))
    for k in range(1, 6):
        rows.append(("MIG-05-parent", base(20 + k - 1, f"MIG05-{k:010d}", "FIN7", f"2025-02-0{k}-12.00.00.000000000000")))
    for k in range(1, 6):
        rows.append(("MIG-06", base(25 + k - 1, f"MIG06-{k:010d}", "AUD7", f"2016-09-0{k}-11.11.11.111111111111")))
    for k in range(1, 7):
        row = base(30 + k - 1, f"MIG07-{k:010d}", "F07R", f"2017-01-0{k}-00.00.00.000000000000")
        rows.append(("MIG-07", replace(row, storage_charge=MIG07_F07R[k - 1])))
    for k in range(1, 7):
        row = base(36 + k - 1, f"MIG07-{k + 6:010d}", "L07R", f"2017-02-0{k}-00.00.00.000000000000")
        rows.append(("MIG-07", replace(row, storage_charge=MIG07_L07R[k - 1])))
    assert len(rows) == 42
    return rows


def planted_fileaud_rows(mig05_parents: list[DocarchRow]) -> list[FileaudRow]:
    """The 5 MIG-05 orphan FILEAUD rows (selected; their parents are not)."""
    out = []
    for k in range(1, 6):
        m = k - 1
        parent = mig05_parents[m]
        out.append(FileaudRow(
            f"MIG05-{k:014d}", parent.arch_key, "VIEW", f"2017-05-0{k}-07.00.00.000000000000",
            f"U{pick('PLANTED', m, 5, 10**11):011d}", "FIN7",
            f"10.{pick('PLANTED', m, 45, 256)}.{pick('PLANTED', m, 46, 256)}.{pick('PLANTED', m, 47, 254) + 1}",
            f"VIEW v{parent.version_no}",
        ))
    return out


def planted_fileaud(mig05_parents: list[DocarchRow]) -> list[bytes]:
    return [row.record() for row in planted_fileaud_rows(mig05_parents)]


def is_selected_docarch(row: DocarchRow) -> bool:
    return row.retention_class in spec.CLOSED_7Y and row.last_access_ts < spec.CUTOFF_TEXT
