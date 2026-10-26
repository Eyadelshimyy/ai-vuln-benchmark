"""Merge per-peril Farmers BDA fact files into multi-peril per-year EDMs.

The cedent fixture used to model the BDA office as three single-peril chains
(WS / EQ / CS). Real-world reinsurance practice is closer to one bundled
multi-peril EDM per office per year, with the peril multi-select at the top
of the page handling drill-in. This script collapses the per-peril fact files
into per-year files:

  ds-farmers-bda-2027 = ds-farmers-27-ws + ds-farmers-27-eq + ds-farmers-27-cs
  ds-farmers-bda-2026 = ds-farmers-26-ws + ds-farmers-26-eq + ds-farmers-26-cs
  ds-farmers-bda-2025 = ds-farmers-25-ws + synthesized 25-eq + synthesized 25-cs

Idempotent. Run anytime:
    cd backend && .venv/Scripts/python scripts/merge_farmers_bda.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

OUT_DIR = (Path(__file__).resolve().parents[2] / "mockdata" / "exposure_facts").resolve()


def _read(name: str) -> list[dict]:
    fp = OUT_DIR / f"{name}.json"
    if not fp.exists():
        return []
    with fp.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write(name: str, rows: list[dict]) -> None:
    fp = OUT_DIR / f"{name}.json"
    with fp.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
        f.write("\n")
    print(f"wrote {fp.name} — {len(rows)} rows")


def _rebrand(rows: list[dict], dataset_id: str, edm_name: str, portname: str) -> list[dict]:
    out: list[dict] = []
    for r in rows:
        clone = dict(r)
        clone["datasetId"] = dataset_id
        clone["sourceDatabaseName"] = edm_name
        clone["sourceTableName"] = f"{edm_name}__EVOLUTION"
        clone["portname"] = portname
        out.append(clone)
    return out


def _synthesize(template_rows: list[dict], peril: str, scale: float) -> list[dict]:
    """Build a fake per-peril fact set from a different-year template.
    Scales numeric measures so the result reads as plausibly different from the
    template year (e.g. 2025 EQ ≈ 90% of 2026 EQ)."""
    out: list[dict] = []
    for r in template_rows:
        clone = dict(r)
        clone["peril"] = peril
        for k in ("building", "contents", "bi", "tiv", "explimGross", "explimNet", "invalidTiv"):
            v = clone.get(k)
            if isinstance(v, (int, float)):
                clone[k] = round(v * scale, 2)
        for k in ("locationCount", "accountCount", "invalidCount"):
            v = clone.get(k)
            if isinstance(v, int):
                clone[k] = max(1, int(v * scale))
        out.append(clone)
    return out


def main() -> int:
    if not OUT_DIR.exists():
        print(f"facts dir not found: {OUT_DIR}", file=sys.stderr)
        return 1

    # 2027 — all three perils already exist
    merged_2027 = (
        _rebrand(_read("ds-farmers-27-ws"), "ds-farmers-bda-2027",
                 "Re_BER_27_Farmers_BDA_EDM_01", "09302025")
        + _rebrand(_read("ds-farmers-27-eq"), "ds-farmers-bda-2027",
                   "Re_BER_27_Farmers_BDA_EDM_01", "09302025")
        + _rebrand(_read("ds-farmers-27-cs"), "ds-farmers-bda-2027",
                   "Re_BER_27_Farmers_BDA_EDM_01", "09302025")
    )
    _write("ds-farmers-bda-2027", merged_2027)

    # 2026 — all three perils already exist
    merged_2026 = (
        _rebrand(_read("ds-farmers-26-ws"), "ds-farmers-bda-2026",
                 "Re_BER_26_Farmers_BDA_EDM_01", "09302024")
        + _rebrand(_read("ds-farmers-26-eq"), "ds-farmers-bda-2026",
                   "Re_BER_26_Farmers_BDA_EDM_01", "09302024")
        + _rebrand(_read("ds-farmers-26-cs"), "ds-farmers-bda-2026",
                   "Re_BER_26_Farmers_BDA_EDM_01", "09302024")
    )
    _write("ds-farmers-bda-2026", merged_2026)

    # 2025 — only WS exists; synthesize EQ and CS from the 2026 templates.
    base_ws_2025 = _read("ds-farmers-25-ws")
    synthesized_eq_2025 = _synthesize(_read("ds-farmers-26-eq"), peril="EQ", scale=0.92)
    synthesized_cs_2025 = _synthesize(_read("ds-farmers-26-cs"), peril="CS", scale=0.88)
    merged_2025 = (
        _rebrand(base_ws_2025, "ds-farmers-bda-2025",
                 "Re_BER_25_Farmers_BDA_EDM_01", "09302023")
        + _rebrand(synthesized_eq_2025, "ds-farmers-bda-2025",
                   "Re_BER_25_Farmers_BDA_EDM_01", "09302023")
        + _rebrand(synthesized_cs_2025, "ds-farmers-bda-2025",
                   "Re_BER_25_Farmers_BDA_EDM_01", "09302023")
    )
    _write("ds-farmers-bda-2025", merged_2025)

    return 0


if __name__ == "__main__":
    sys.exit(main())
