from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict


class LeagueConfig(TypedDict):
    id: str
    label: str
    season: str
    xlsx: Path
    defensive_csv: Path
    heatmaps: bool
    heatmaps_dir: Path | None


ROOT = Path(__file__).resolve().parents[2]

LEAGUES: dict[str, LeagueConfig] = {
    "serie-a": {
        "id": "serie-a",
        "label": "Série A",
        "season": "2025/26",
        "xlsx": ROOT / "Serie A 26.xlsx",
        "defensive_csv": ROOT / "BR26_defensive.csv",
        "heatmaps": True,
        "heatmaps_dir": ROOT / "HeatmapsBR26",
    },
    "serie-c": {
        "id": "serie-c",
        "label": "Série C",
        "season": "2026 · 1ª fase",
        "xlsx": ROOT / "serie c primeira fase.xlsx",
        "defensive_csv": ROOT / "BRC26_defensive.csv",
        "heatmaps": False,
        "heatmaps_dir": None,
    },
}
