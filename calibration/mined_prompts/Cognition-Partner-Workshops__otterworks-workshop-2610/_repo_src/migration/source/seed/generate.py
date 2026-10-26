"""Parallel, deterministic writer for the three fixed-width seed files plus seed-summary.json."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from . import spec, tables
from .spec import CUTOFF_SECS, CUTOFF_TEXT, LRECL, Sizes, decimal8, system_of_record_class

CHUNK = 20_000

_SIZES: Sizes | None = None
_S_KEYS: list[int] = []
_NS_KEYS: list[int] = []


def _init_worker(scale: float, s_keys: list[int], ns_keys: list[int]) -> None:
    global _SIZES, _S_KEYS, _NS_KEYS
    _SIZES = Sizes(scale)
    _S_KEYS, _NS_KEYS = s_keys, ns_keys


def cohort_keys(sizes: Sizes) -> tuple[list[int], list[int]]:
    """S_KEYS / NS_KEYS (§5): DOCARCH indexes of cohort S, resp. R+O, sorted by ARCH_KEY (= by index)."""
    s_keys = sorted(sizes.docarch_g_for_p(p) for p in range(sizes.docarch_s))
    s_set = set(s_keys)
    ns_keys = [g for g in range(sizes.docarch_generated) if g not in s_set]
    assert len(s_keys) == sizes.docarch_s and len(ns_keys) == sizes.docarch_ns
    return s_keys, ns_keys


def _docarch_chunk(bounds: tuple[int, int]) -> tuple[bytes, dict]:
    lo, hi = bounds
    out = bytearray()
    sel_class: Counter[str] = Counter()
    sel_charge: Counter[str] = Counter()
    for g in range(lo, hi):
        row = tables.docarch_generated(_SIZES, g)
        if row.cohort == "S":
            assert row.last_access_secs < CUTOFF_SECS and row.retention_class in spec.SET_ACTIVE
            sel_class[row.retention_class] += 1
            sel_charge[row.retention_class] += row.storage_charge
        else:
            assert not tables.is_selected_docarch(row), row.arch_key
        out += row.record()
    return bytes(out), {"class": sel_class, "charge": sel_charge}


def _fileaud_chunk(bounds: tuple[int, int]) -> tuple[bytes, dict]:
    lo, hi = bounds
    out = bytearray()
    sel_class: Counter[str] = Counter()
    for m in range(lo, hi):
        rec, parent, ev_secs = tables.fileaud_generated(_SIZES, m, _S_KEYS, _NS_KEYS)
        if parent.cohort == "S":
            if ev_secs >= CUTOFF_SECS:
                raise AssertionError(f"child {m} of selected parent has EVENT_TS >= cutoff")
            sel_class[parent.retention_class] += 1
        elif parent.cohort == "R":
            if ev_secs < CUTOFF_SECS:
                raise AssertionError(f"child {m} of recent parent has EVENT_TS < cutoff")
        out += rec
    return bytes(out), {"class": sel_class}


def _chunks(n: int) -> list[tuple[int, int]]:
    return [(lo, min(lo + CHUNK, n)) for lo in range(0, n, CHUNK)]


def _bounded_map(pool: ProcessPoolExecutor, fn, items, window: int):
    """Ordered map that keeps at most `window` results in flight (bounds memory on small pods)."""
    pending: deque = deque()
    for item in items:
        pending.append(pool.submit(fn, item))
        if len(pending) >= window:
            yield pending.popleft().result()
    while pending:
        yield pending.popleft().result()


class Summary:
    def __init__(self) -> None:
        self.files: dict[str, dict] = {}
        self.selected: dict[str, int] = {}
        self.selected_by_class: dict[str, Counter[str]] = {}
        self.selected_charge_by_class: dict[str, Counter[str]] = {}

    def to_json(self) -> dict:
        return {
            "seed": f"0x{spec.SEED:016X}",
            "cutoff": CUTOFF_TEXT,
            "files": self.files,
            "selected": self.selected,
            "selected_by_class": {t: dict(sorted(c.items())) for t, c in self.selected_by_class.items()},
            "selected_charge_by_class": {
                t: {k: decimal8(v) for k, v in sorted(c.items())} for t, c in self.selected_charge_by_class.items()
            },
        }


def _write(path: Path, lrecl: int, parts, expected_rows: int) -> dict:
    sha = hashlib.sha256()
    rows = 0
    with open(path, "wb") as fh:
        for blob in parts:
            fh.write(blob)
            sha.update(blob)
            rows += len(blob) // lrecl
    if rows != expected_rows:
        raise AssertionError(f"{path.name}: wrote {rows} rows, expected {expected_rows}")
    return {"rows": rows, "bytes": rows * lrecl, "lrecl": lrecl, "sha256": sha.hexdigest()}


def generate(out_dir: Path, which: list[str], scale: float = 1.0, workers: int | None = None) -> Summary:
    sizes = Sizes(scale)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = Summary()
    workers = workers or max(1, os.cpu_count() or 1)
    s_keys, ns_keys = cohort_keys(sizes)
    planted = tables.planted_docarch()

    if "RETNPLCY" in which:
        recs = tables.retnplcy_records()
        summary.files["RETNPLCY.asc"] = _write(out_dir / "RETNPLCY.asc", LRECL["RETNPLCY"], recs, 40)
        summary.selected["RETNPLCY"] = 40
        summary.selected_by_class["RETNPLCY"] = Counter({code: 1 for code, *_ in spec.RETNPLCY_ROWS})

    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                             initargs=(scale, s_keys, ns_keys)) as pool:
        if "DOCARCH" in which:
            cls: Counter[str] = Counter()
            chg: Counter[str] = Counter()

            def docarch_parts():
                for blob, stats in _bounded_map(pool, _docarch_chunk, _chunks(sizes.docarch_generated), workers * 3):
                    cls.update(stats["class"])
                    chg.update(stats["charge"])
                    yield blob
                planted_blob = bytearray()
                for _, row in planted:
                    planted_blob += row.record()
                    if tables.is_selected_docarch(row):
                        sor = system_of_record_class(row.retention_class)
                        cls[sor] += 1
                        chg[sor] += row.storage_charge
                yield bytes(planted_blob)

            n = sizes.docarch_generated + len(planted)
            summary.files["DOCARCH.asc"] = _write(out_dir / "DOCARCH.asc", LRECL["DOCARCH"], docarch_parts(), n)
            summary.selected["DOCARCH"] = sum(cls.values())
            summary.selected_by_class["DOCARCH"] = cls
            summary.selected_charge_by_class["DOCARCH"] = chg
            if sum(cls.values()) != sizes.docarch_s + 37:
                raise AssertionError(f"DOCARCH selected {sum(cls.values())} != {sizes.docarch_s + 37}")

        if "FILEAUD" in which:
            fcls: Counter[str] = Counter()

            def fileaud_parts():
                for blob, stats in _bounded_map(pool, _fileaud_chunk, _chunks(sizes.fileaud_generated), workers * 3):
                    fcls.update(stats["class"])
                    yield blob
                parents = [row for tag, row in planted if tag == "MIG-05-parent"]
                orphans = tables.planted_fileaud(parents)
                fcls["FIN7"] += len(orphans)
                yield b"".join(orphans)

            n = sizes.fileaud_generated + 5
            summary.files["FILEAUD.asc"] = _write(out_dir / "FILEAUD.asc", LRECL["FILEAUD"], fileaud_parts(), n)
            summary.selected["FILEAUD"] = sum(fcls.values())
            summary.selected_by_class["FILEAUD"] = fcls
            if sum(fcls.values()) != sizes.fileaud_selected + 5:
                raise AssertionError(f"FILEAUD selected {sum(fcls.values())} != {sizes.fileaud_selected + 5}")

    (out_dir / "seed-summary.json").write_text(json.dumps(summary.to_json(), indent=2) + "\n")
    return summary
