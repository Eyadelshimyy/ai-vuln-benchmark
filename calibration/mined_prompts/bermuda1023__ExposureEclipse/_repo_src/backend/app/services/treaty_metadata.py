"""Treaty metadata (inbound from the placement system) + EDM linkage.

Production flow:
  Inwards business system → exports treaty extract (CSV / XLSX) →
  Admin imports into this service → analyst maps each programme to a
  SQL server + EDM database → Peril Vista pulls facts via the
  mapped EDM going forward.

For v1 we ship a JSON-backed mock — both the treaty list and the
mapping live as files under ``mockdata/``. Swapping for a real
upstream feed only touches the loaders here.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from pathlib import Path

from ..config import get_settings


# ─────────────────────────── data shapes ───────────────────────────


@dataclass(slots=True)
class TreatyRow:
    """One signed line on one layer of a treaty (matches a single row in
    the broker / placement-system extract)."""

    fs_display_id: str
    reinsured_name: str
    broker_name: str
    broker_office: str | None
    layer_number: int
    inception_date: str
    layer_status: str          # Signed | Quoted | NTU | Declined | Expired
    risk_id: str
    currency: str
    weighted_share_pct: float
    signed_line_pct: float
    risk_location: str
    tji: str | None
    cob1: str | None
    cob2: str | None
    cob3: str | None
    event_limit_usd: float
    deductible_usd: float
    rol_pct: float
    gul_pct: float


@dataclass(slots=True)
class EDMLink:
    fs_display_id: str
    server_name: str | None
    edm_database_name: str | None


@dataclass(slots=True)
class TreatyView:
    """Joined view returned to the admin UI."""

    treaty: TreatyRow
    link: EDMLink
    status: str                  # "mapped" | "unmapped"
    suggested_server: str | None
    suggested_edm: str | None


# ─────────────────────────── file paths ───────────────────────────


def _mockdata_dir() -> Path:
    settings = get_settings()
    p = Path(settings.mock_data_dir)
    if not p.is_absolute():
        p = (Path(__file__).resolve().parents[2] / p).resolve()
    return p


def _treaty_path() -> Path:
    return _mockdata_dir() / "treaty_metadata.json"


def _linkage_path() -> Path:
    return _mockdata_dir() / "edm_linkage.json"


# ─────────────────────────── load / save ───────────────────────────


def load_treaty_rows() -> list[TreatyRow]:
    path = _treaty_path()
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    out: list[TreatyRow] = []
    for r in data:
        out.append(
            TreatyRow(
                fs_display_id=r["fsDisplayId"],
                reinsured_name=r.get("reinsuredName", ""),
                broker_name=r.get("brokerName", ""),
                broker_office=r.get("brokerOffice"),
                layer_number=int(r.get("layerNumber", 1)),
                inception_date=r.get("inceptionDate", ""),
                layer_status=r.get("layerStatus", "Signed"),
                risk_id=r.get("riskId", ""),
                currency=r.get("currency", "USD"),
                weighted_share_pct=float(r.get("weightedSharePct", 0.0)),
                signed_line_pct=float(r.get("signedLinePct", 0.0)),
                risk_location=r.get("riskLocation", ""),
                tji=r.get("tji"),
                cob1=r.get("cob1"),
                cob2=r.get("cob2"),
                cob3=r.get("cob3"),
                event_limit_usd=float(r.get("eventLimitUsd", 0.0)),
                deductible_usd=float(r.get("deductibleUsd", 0.0)),
                rol_pct=float(r.get("rolPct", 0.0)),
                gul_pct=float(r.get("gulPct", 0.0)),
            )
        )
    return out


def load_linkage() -> dict[str, EDMLink]:
    path = _linkage_path()
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, EDMLink] = {}
    for fs_id, v in data.items():
        out[fs_id] = EDMLink(
            fs_display_id=fs_id,
            server_name=v.get("serverName"),
            edm_database_name=v.get("edmDatabaseName"),
        )
    return out


def save_linkage(links: dict[str, EDMLink]) -> None:
    """Persist the full linkage map to disk. Replaces the file."""
    payload = {
        fs_id: {
            "serverName": l.server_name,
            "edmDatabaseName": l.edm_database_name,
        }
        for fs_id, l in links.items()
        if l.server_name or l.edm_database_name
    }
    _linkage_path().write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )


def save_treaty_rows(rows: list[TreatyRow]) -> None:
    """Persist the treaty list (e.g. after a CSV import). Replaces the file."""
    payload = [
        {
            "fsDisplayId": r.fs_display_id,
            "reinsuredName": r.reinsured_name,
            "brokerName": r.broker_name,
            "brokerOffice": r.broker_office,
            "layerNumber": r.layer_number,
            "inceptionDate": r.inception_date,
            "layerStatus": r.layer_status,
            "riskId": r.risk_id,
            "currency": r.currency,
            "weightedSharePct": r.weighted_share_pct,
            "signedLinePct": r.signed_line_pct,
            "riskLocation": r.risk_location,
            "tji": r.tji,
            "cob1": r.cob1,
            "cob2": r.cob2,
            "cob3": r.cob3,
            "eventLimitUsd": r.event_limit_usd,
            "deductibleUsd": r.deductible_usd,
            "rolPct": r.rol_pct,
            "gulPct": r.gul_pct,
        }
        for r in rows
    ]
    _treaty_path().write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ─────────────────────────── auto-suggest ───────────────────────────


def _norm(name: str) -> str:
    """Cheap normalisation for fuzzy matching: lowercase + alphanum only."""
    return "".join(c.lower() for c in (name or "") if c.isalnum())


def suggest_edm(
    treaty: TreatyRow,
    cedents_index: dict[str, dict],
) -> tuple[str | None, str | None]:
    """Given a TreatyRow + the cedents.json index, return (server, edm_name)
    if any existing programme has a name that contains the reinsured name
    (or vice versa). Coarse first-pass — admin still confirms."""
    target = _norm(treaty.reinsured_name)
    if not target:
        return None, None
    best: tuple[int, str | None, str | None] = (0, None, None)
    for cedent in cedents_index.get("cedents", []):
        c_name = _norm(cedent.get("cedentName", ""))
        score = 0
        if c_name and (c_name in target or target in c_name):
            score = max(len(c_name), len(target))
        if score == 0:
            # Try last-name-only match (e.g. "Munich Re America" vs "Munich Re")
            c_first = (cedent.get("cedentName") or "").split()[0].lower()
            t_first = (treaty.reinsured_name or "").split()[0].lower()
            if c_first and c_first == t_first:
                score = len(c_first)
        if score > best[0]:
            # Prefer a programme whose year matches the treaty year
            treaty_year = treaty.inception_date[:4]
            picked = None
            for ch in cedent.get("chains", []):
                for prog in ch.get("programmes", []):
                    if str(prog.get("treatyYear", "")) == treaty_year:
                        picked = prog
                        break
                if picked:
                    break
            if picked is None:
                # fallback to the latest programme on the first chain
                for ch in cedent.get("chains", []):
                    progs = sorted(
                        ch.get("programmes", []),
                        key=lambda p: -int(p.get("treatyYear", 0)),
                    )
                    if progs:
                        picked = progs[0]
                        break
            if picked:
                edm = picked.get("edm") or {}
                best = (score, edm.get("serverName"), edm.get("edmDatabaseName"))
    return best[1], best[2]


# ─────────────────────────── joined view ───────────────────────────


def joined_view(
    *,
    cedents_index: dict | None = None,
) -> list[TreatyView]:
    """Return treaty rows joined with their EDM linkage + auto-suggest."""
    rows = load_treaty_rows()
    links = load_linkage()
    cedents = cedents_index or {}
    out: list[TreatyView] = []
    for r in rows:
        link = links.get(r.fs_display_id) or EDMLink(
            fs_display_id=r.fs_display_id,
            server_name=None,
            edm_database_name=None,
        )
        sug_server, sug_edm = suggest_edm(r, cedents) if cedents else (None, None)
        status = "mapped" if (link.server_name and link.edm_database_name) else "unmapped"
        out.append(
            TreatyView(
                treaty=r,
                link=link,
                status=status,
                suggested_server=sug_server,
                suggested_edm=sug_edm,
            )
        )
    return out


# ─────────────────────────── CSV import ───────────────────────────


# Treaty extracts from the placement system come as wide CSVs with the same
# header set as the screenshot. We accept that shape directly (camelCase or
# the original Excel labels) and tolerate missing columns.
_CSV_FIELD_ALIASES: dict[str, str] = {
    "fs_display": "fsDisplayId",
    "fs display": "fsDisplayId",
    "fsdisplay": "fsDisplayId",
    "reinsured": "reinsuredName",
    "reinsured (layer)": "reinsuredName",
    "broker": "brokerName",
    "broker (layer)": "brokerName",
    "broker office": "brokerOffice",
    "layer #": "layerNumber",
    "layer": "layerNumber",
    "inception": "inceptionDate",
    "inception dt": "inceptionDate",
    "layer status": "layerStatus",
    "status": "layerStatus",
    "risket": "riskId",
    "riskid": "riskId",
    "risk id": "riskId",
    "curr.": "currency",
    "curr": "currency",
    "wt %": "weightedSharePct",
    "wt%": "weightedSharePct",
    "signed line": "signedLinePct",
    "risk location": "riskLocation",
    "risk location name": "riskLocation",
    "tji": "tji",
    "cob1 name": "cob1",
    "cob1": "cob1",
    "cob2 name": "cob2",
    "cob2": "cob2",
    "cob3": "cob3",
    "event limit usd": "eventLimitUsd",
    "event limit": "eventLimitUsd",
    "deductible usd": "deductibleUsd",
    "deductible": "deductibleUsd",
    "rol": "rolPct",
    "rol %": "rolPct",
    "gul": "gulPct",
    "gul %": "gulPct",
}


def _parse_pct(s: str) -> float:
    s = (s or "").strip().rstrip("%").replace(",", "")
    try:
        return float(s)
    except ValueError:
        return 0.0


def _parse_money(s: str) -> float:
    s = (s or "").strip().replace(",", "").replace("$", "")
    if not s:
        return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def parse_csv(body: str) -> list[TreatyRow]:
    """Parse an uploaded CSV body into TreatyRows. Robust to header naming
    variants from the placement system."""
    sio = io.StringIO(body)
    reader = csv.DictReader(sio)
    if reader.fieldnames is None:
        return []
    norm_map: dict[str, str] = {}
    for raw in reader.fieldnames:
        canon = _CSV_FIELD_ALIASES.get(raw.strip().lower())
        if canon:
            norm_map[raw] = canon
        else:
            norm_map[raw] = raw  # keep as-is; ignored if unrecognised

    out: list[TreatyRow] = []
    for raw_row in reader:
        row = {norm_map.get(k, k): v for k, v in raw_row.items()}
        if not row.get("fsDisplayId"):
            continue
        try:
            out.append(
                TreatyRow(
                    fs_display_id=row.get("fsDisplayId", "").strip(),
                    reinsured_name=row.get("reinsuredName", "").strip(),
                    broker_name=row.get("brokerName", "").strip(),
                    broker_office=(row.get("brokerOffice") or "").strip() or None,
                    layer_number=int(float(row.get("layerNumber", "1") or "1")),
                    inception_date=row.get("inceptionDate", "").strip(),
                    layer_status=row.get("layerStatus", "Signed").strip() or "Signed",
                    risk_id=row.get("riskId", "").strip(),
                    currency=row.get("currency", "USD").strip() or "USD",
                    weighted_share_pct=_parse_pct(row.get("weightedSharePct", "")),
                    signed_line_pct=_parse_pct(row.get("signedLinePct", "")),
                    risk_location=row.get("riskLocation", "").strip(),
                    tji=(row.get("tji") or "").strip() or None,
                    cob1=(row.get("cob1") or "").strip() or None,
                    cob2=(row.get("cob2") or "").strip() or None,
                    cob3=(row.get("cob3") or "").strip() or None,
                    event_limit_usd=_parse_money(row.get("eventLimitUsd", "")),
                    deductible_usd=_parse_money(row.get("deductibleUsd", "")),
                    rol_pct=_parse_pct(row.get("rolPct", "")),
                    gul_pct=_parse_pct(row.get("gulPct", "")),
                )
            )
        except (ValueError, KeyError):
            continue
    return out


# Suppress the lint about the `field` import — kept for future dataclass needs.
_ = field
