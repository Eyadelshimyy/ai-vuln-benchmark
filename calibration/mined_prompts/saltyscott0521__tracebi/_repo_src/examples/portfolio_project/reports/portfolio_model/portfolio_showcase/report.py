"""Escape hatch demo: a python-derived figure — honestly never green."""


def build(frames):
    df = frames["by_sector"].copy()
    total = df["fair_value"].sum()
    df["cum_share"] = (df["fair_value"].cumsum() / total).round(4)
    return {"concentration": df[["dim_issuer.sector", "cum_share"]]}
