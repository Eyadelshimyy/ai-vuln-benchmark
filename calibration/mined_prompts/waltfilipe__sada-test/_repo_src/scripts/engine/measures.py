from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _num(value: Any, default: float = 0.0) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def compute_base_measures(row: pd.Series, max_minutes: float) -> dict[str, float]:
    minutes = _num(row.get("Minutos jogados:"))
    pct_minutes = minutes / max_minutes if max_minutes > 0 else 0.0

    passes = _num(row.get("Passes/90"))
    passes_prog = _num(row.get("Passes progressivos/90"))
    passes_long = _num(row.get("Passes longos/90"))
    ptf = _num(row.get("Passes para terço final/90"))
    accel = _num(row.get("Acelerações/90"))
    corridas = _num(row.get("Corridas progressivas/90"))
    duelos_def = _num(row.get("Duelos defensivos/90"))
    pct_duelos_def = _num(row.get("Duelos defensivos ganhos, %")) / 100
    duelos_ar = _num(row.get("Duelos aérios/90"))
    pct_duelos_ar = _num(row.get("Duelos aéreos ganhos, %")) / 100
    duelos_of = _num(row.get("Duelos ofensivos/90"))
    dribles = _num(row.get("Dribles/90"))
    pct_duelos_of = _num(row.get("Duelos ofensivos ganhos, %")) / 100
    pct_eff_prog = _num(row.get("Passes progressivos certos, %")) / 100
    pct_eff_long = _num(row.get("Passes longos certos, %")) / 100
    pct_eff_ptf = _num(row.get("Passes certos para terço final, %")) / 100
    interceptions = _num(row.get("Interseções/90"))
    carrinhos = _num(row.get("Cortes/90"))
    ad = _num(row.get("Ações defensivas com êxito/90"))

    duelos_def_l = (100 - _num(row.get("Duelos defensivos ganhos, %"))) * -1
    duelos_def_l = (duelos_def * duelos_def_l) if duelos_def_l else 0.0
    eff_duelos_def = ((duelos_def * pct_duelos_def) + carrinhos) / duelos_def_l if duelos_def_l else 0.0
    leitura_def = ad / duelos_def_l if duelos_def_l else 0.0

    duelos_of_w = ((duelos_of * pct_duelos_of) - dribles) / duelos_of if duelos_of else 0.0

    return {
        "%Minutos": pct_minutes,
        "PassesProg": passes_prog,
        "PassesLongos": passes_long,
        "PTF": ptf,
        "Cond.Prog": corridas - accel,
        "DuelosDef": duelos_def,
        "%DuelosDefW": pct_duelos_def,
        "DuelosAr": duelos_ar,
        "%DuelosAr": pct_duelos_ar,
        "DuelosOf": duelos_of - dribles,
        "%DuelosOfW": duelos_of_w,
        "%EffPassProg": pct_eff_prog,
        "%EffPassesLng": pct_eff_long,
        "%EffPassTF": pct_eff_ptf,
        "Interseções": interceptions,
        "Carrinhos": carrinhos,
        "EffDuelosDef": eff_duelos_def,
        "LeituraDef.": leitura_def,
        "CompPassesProg": passes_prog * pct_eff_prog,
        "CompBL": passes_long * pct_eff_long,
        "CompPTF": ptf * pct_eff_ptf,
        "Passe": passes,
        "%PProg": passes_prog / passes if passes else 0.0,
        "%PassesLng": passes_long / passes if passes else 0.0,
        "%PPTF": ptf / passes if passes else 0.0,
        "Dribles": dribles,
        "Acelerações": accel,
        "AçõesDef": ad,
        "Gols": _num(row.get("Golos")),
        "Assist": _num(row.get("Assistências")),
        "Assist2": _num(row.get("Segundas assistências/90")),
        "Assist3": _num(row.get("Terceiras assistências/90")),
        "PreAssists": _num(row.get("Segundas assistências/90")) + _num(row.get("Terceiras assistências/90")),
        "PreAssistsTotal": (
            (_num(row.get("Segundas assistências/90")) + _num(row.get("Terceiras assistências/90")))
            * minutes
            / 90
            if minutes
            else 0.0
        ),
        "xA": _num(row.get("Assistências esperadas")),
        "xGTotal": _num(row.get("Golos esperados")),
        "Finalizações": _num(row.get("Remates/90")),
        "ToquesArea": _num(row.get("Toques na área/90")),
        "Cruz.": _num(row.get("Cruzamentos/90")),
        "%EffCruz.": _num(row.get("Cruzamentos certos, %")) / 100,
        "PassesChave": _num(row.get("Passes chave/90")),
        "PassesProf": _num(row.get("Passes em profundidade/90")),
        "%EffPassesProf": _num(row.get("Passes em profundidade certos, %")) / 100,
        "PassesFrente": _num(row.get("Passes para a frente/90")),
        "Rec.PassesPrf.": _num(row.get("Receção de passes em profundidade/90")),
        "PasseAreaW": _num(row.get("Passes para a área de penálti/90"))
        * (_num(row.get("Passes precisos para a área de penálti, %")) / 100),
        "PassesCriativos": _num(row.get("Passes inteligentes/90")),
        "PassesPerigosos": _num(row.get("Passes chave/90")),
        "%EffPasse": _num(row.get("Passes certos, %")) / 100,
        "%EffDribles": _num(row.get("Dribles com sucesso, %")) / 100,
        "AçõesAtW": _num(row.get("Toques na área/90")),
        "RecPasse": _num(row.get("Passes recebidos/90")),
        "RecPassesLngs": _num(row.get("Passes longos recebidos/90")),
        "FaltasSfd": _num(row.get("Faltas sofridas/90")),
        "npxG": _num(row.get("Golos esperados/90")),
        "Golsxg": _num(row.get("Golos/90")) - _num(row.get("Golos esperados/90")),
        "GolspTq": _num(row.get("Golos/90")) / max(_num(row.get("Toques na área/90")), 0.1),
        "%EffFin": _num(row.get("Remates à baliza, %")) / 100,
        "CompPasse": passes * (_num(row.get("Passes certos, %")) / 100),
    }


def attach_base_measures(df: pd.DataFrame) -> pd.DataFrame:
    max_minutes = float(df["Minutos jogados:"].max() or 1)
    measures = [compute_base_measures(row, max_minutes) for _, row in df.iterrows()]
    measures_df = pd.DataFrame(measures)
    return pd.concat([df.reset_index(drop=True), measures_df], axis=1)
