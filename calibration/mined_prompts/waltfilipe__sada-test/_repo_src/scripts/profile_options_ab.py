#!/usr/bin/env python3
"""Compare alternative zagueiro profile classifiers (Option A/B) vs current DAX labels."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, confusion_matrix, normalized_mutual_info_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from engine.load_data import load_players_dataframe
from engine.measures import attach_base_measures
from engine.positions import compute_family_metrics

ROOT = Path(__file__).resolve().parents[1]
COMPOSITE_FEATURES = [
    "CompPassesProg",
    "CompBL",
    "CompPTF",
    "EffDuelosDef",
    "LeituraDef.",
    "Cond.Prog",
]
CONSTRUCTION_FEATURES = ["CompPassesProg", "CompBL", "CompPTF"]
DEFENSE_FEATURES = ["EffDuelosDef", "LeituraDef."]
PROFILE_ORDER = ["Combativo", "Construtor", "Posicional", "Híbrido"]

# Tactical names for Option A (post-cluster mapping)
OPTION_A_LABELS = {
    "construtor": "Construtor",
    "leitura": "Leitura",
    "duelista": "Duelista",
    "hibrido": "Híbrido",
}

# Map new labels to current taxonomy for comparison matrices
COMPARE_LABELS = ["Construtor", "Posicional", "Combativo", "Híbrido"]
OPTION_A_COMPARE = {
    "Construtor": "Construtor",
    "Leitura": "Posicional",
    "Duelista": "Combativo",
    "Híbrido": "Híbrido",
}
OPTION_B_COMPARE = {
    "Construtor": "Construtor",
    "Posicional": "Posicional",
    "Combativo": "Combativo",
    "Híbrido": "Híbrido",
}


def load_merged() -> pd.DataFrame:
    raw = attach_base_measures(load_players_dataframe())
    zags = compute_family_metrics(raw, "zagueiros")
    pool = raw[(raw["Posição"] == "Zagueiro") & (raw["player_id"].isin(zags["player_id"]))].copy()
    pool = pool.set_index("player_id").loc[zags["player_id"].values].reset_index()
    return zags.merge(pool, on="player_id", suffixes=("", "_raw"))


def confusion_df(y_true: pd.Series, y_pred: pd.Series, labels: list[str]) -> pd.DataFrame:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(matrix, index=[f"atual:{x}" for x in labels], columns=[f"pred:{x}" for x in labels])


def metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    return {
        "ari": float(adjusted_rand_score(y_true, y_pred)),
        "nmi": float(normalized_mutual_info_score(y_true, y_pred)),
        "agree_pct": float((y_true.values == y_pred.values).mean() * 100),
    }


def assign_component_labels(centroids_raw: np.ndarray, features: list[str], k: int) -> list[str]:
    """Map each GMM component to a distinct tactical label using standardized centroids."""
    scaler = StandardScaler()
    centroids_z = scaler.fit_transform(centroids_raw)

    construction_idx = [features.index(f) for f in CONSTRUCTION_FEATURES]
    defense_idx = [features.index(f) for f in DEFENSE_FEATURES]
    duel_idx = features.index("EffDuelosDef")
    leitura_idx = features.index("LeituraDef.")

    profiles = ["construtor", "leitura", "duelista"]
    if k > 3:
        profiles.append("hibrido")

    scores = []
    for i in range(k):
        c = centroids_z[i]
        scores.append(
            {
                "i": i,
                "construtor": float(np.mean(c[construction_idx])),
                "leitura": float(c[leitura_idx] + c[features.index("CompPTF")] * 0.3),
                "duelista": float(c[duel_idx] * 0.65 + c[features.index("Cond.Prog")] * 0.35),
                "defesa": float(np.mean(c[defense_idx])),
            }
        )

    # Greedy: each tactical label picks its best remaining component
    assigned: dict[int, str] = {}
    for profile in profiles[: min(k, 3)]:
        best = max(
            (s for s in scores if s["i"] not in assigned),
            key=lambda s: s[profile],
        )
        assigned[best["i"]] = profile

    for s in scores:
        if s["i"] not in assigned:
            assigned[s["i"]] = "hibrido"

    return [OPTION_A_LABELS[assigned[i]] for i in range(k)]


def option_a_gmm(df: pd.DataFrame, k: int = 3, hybrid_threshold: float = 0.45) -> pd.DataFrame:
    Xdf = df[COMPOSITE_FEATURES].fillna(0)
    scaler = StandardScaler()
    X = scaler.fit_transform(Xdf)

    gmm = GaussianMixture(n_components=k, random_state=42, covariance_type="full", n_init=15)
    gmm.fit(X)
    centroids_raw = scaler.inverse_transform(gmm.means_)
    component_names = assign_component_labels(centroids_raw, COMPOSITE_FEATURES, k)

    proba = gmm.predict_proba(X)
    hard_idx = proba.argmax(axis=1)
    labels = []
    for idx, probs in zip(hard_idx, proba):
        if probs.max() < hybrid_threshold:
            labels.append("Híbrido")
        else:
            labels.append(component_names[idx])

    out = df[["Jogador", "Equipe", "perfil"]].copy()
    out["option_a"] = labels
    out["option_a_conf"] = proba.max(axis=1).round(3)
    out["option_a_entropy"] = (-(proba * np.log(proba + 1e-12)).sum(axis=1)).round(3)
    out["option_a_k"] = k
    out["option_a_components"] = [component_names[i] for i in hard_idx]
    return out


def option_b_quadrants(df: pd.DataFrame, hybrid_margin: float = 0.35) -> pd.DataFrame:
    Xdf = df[COMPOSITE_FEATURES].fillna(0)
    scaler = StandardScaler()
    X = scaler.fit_transform(Xdf)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    pc1, pc2 = coords[:, 0], coords[:, 1]

    # Signed axes: construction (+PC1), defense (+PC2)
    construction_score = pc1
    defense_score = pc2

    labels = []
    for c_score, d_score in zip(construction_score, defense_score):
        high_c = c_score > hybrid_margin
        low_c = c_score < -hybrid_margin
        high_d = d_score > hybrid_margin
        low_d = d_score < -hybrid_margin

        if high_c and low_d:
            labels.append("Construtor")
        elif low_c and high_d:
            labels.append("Combativo")
        elif high_c and high_d:
            labels.append("Posicional")
        elif low_c and low_d:
            labels.append("Híbrido")
        else:
            labels.append("Híbrido")

    out = df[["Jogador", "Equipe", "perfil"]].copy()
    out["option_b"] = labels
    out["pc1_construcao"] = construction_score.round(3)
    out["pc2_defesa"] = defense_score.round(3)
    out["pc1_var"] = float(pca.explained_variance_ratio_[0])
    out["pc2_var"] = float(pca.explained_variance_ratio_[1])
    return out, pca


def compare_to_current(results: pd.DataFrame, pred_col: str, mapping: dict[str, str]) -> pd.DataFrame:
    mapped = results[pred_col].map(mapping)
    return mapped


def main() -> None:
    df = load_merged()
    print(f"Zagueiros analisados: {len(df)}\n")

    current = df["perfil"]
    print("=== Distribuição — modelo atual (DAX) ===")
    print(current.value_counts().to_string())
    print()

    # KMeans baseline on same 6 composites (hard labels)
    sc = StandardScaler().fit(df[COMPOSITE_FEATURES].fillna(0))
    X = sc.transform(df[COMPOSITE_FEATURES].fillna(0))
    km3 = KMeans(3, random_state=42, n_init=30).fit(X)
    km_names = assign_component_labels(sc.inverse_transform(km3.cluster_centers_), COMPOSITE_FEATURES, 3)
    km_tactical = pd.Series([OPTION_A_COMPARE[km_names[l]] for l in km3.labels_])

    print("=== K-Means K=3 (6 composites, baseline) ===")
    print(km_tactical.value_counts().to_string())
    mk = metrics(current, km_tactical)
    print(f"Concordância vs atual: {mk['agree_pct']:.1f}% | ARI={mk['ari']:.3f} | NMI={mk['nmi']:.3f}")
    print()

    a3 = option_a_gmm(df, k=3, hybrid_threshold=0.45)
    a4 = option_a_gmm(df, k=4, hybrid_threshold=0.45)

    for label, res in [("Option A (GMM K=3)", a3), ("Option A (GMM K=4)", a4)]:
        mapped = compare_to_current(res, "option_a", OPTION_A_COMPARE)
        print(f"=== {label} ===")
        print(res["option_a"].value_counts().to_string())
        m = metrics(current, mapped)
        print(f"Concordância vs atual: {m['agree_pct']:.1f}% | ARI={m['ari']:.3f} | NMI={m['nmi']:.3f}")
        print("Híbridos por confiança (<45%):", int((res["option_a"] == "Híbrido").sum()))
        print()

    # Option B — test margins
    for margin in [0.25, 0.35, 0.45]:
        b, pca = option_b_quadrants(df, hybrid_margin=margin)
        b_mapped = compare_to_current(b, "option_b", OPTION_B_COMPARE)
        mb = metrics(current, b_mapped)
        print(f"--- Option B (margem ±{margin}) ---")
        print(b["option_b"].value_counts().to_string())
        print(f"Concordância vs atual: {mb['agree_pct']:.1f}% | ARI={mb['ari']:.3f} | NMI={mb['nmi']:.3f}")
        print()

    b, pca = option_b_quadrants(df, hybrid_margin=0.25)
    b_mapped = compare_to_current(b, "option_b", OPTION_B_COMPARE)
    print("=== Option B (PCA quadrantes) ===")
    print(f"PC1 variância: {pca.explained_variance_ratio_[0]*100:.1f}%")
    print(f"PC2 variância: {pca.explained_variance_ratio_[1]*100:.1f}%")
    print(f"Acumulado: {pca.explained_variance_ratio_.sum()*100:.1f}%")
    loadings = pd.DataFrame(pca.components_.T, index=COMPOSITE_FEATURES, columns=["PC1", "PC2"])
    print("\nLoadings:")
    print(loadings.round(3).to_string())
    print()
    print(b["option_b"].value_counts().to_string())
    mb = metrics(current, b_mapped)
    print(f"Concordância vs atual: {mb['agree_pct']:.1f}% | ARI={mb['ari']:.3f} | NMI={mb['nmi']:.3f}")
    print()

    # Headline comparison table
    rows = []
    for name, pred in [
        ("Atual (DAX / 5 dim ZG)", current),
        ("K-Means 6 comp", km_tactical),
        ("Option A GMM K=3", compare_to_current(a3, "option_a", OPTION_A_COMPARE)),
        ("Option A GMM K=4", compare_to_current(a4, "option_a", OPTION_A_COMPARE)),
        ("Option B PCA (±0.25)", b_mapped),
    ]:
        m = metrics(current, pred) if name != "Atual (DAX / 5 dim ZG)" else {"ari": 1.0, "nmi": 1.0, "agree_pct": 100.0}
        counts = pred.value_counts().reindex(COMPARE_LABELS, fill_value=0)
        rows.append(
            {
                "modelo": name,
                "concordancia_%": round(m["agree_pct"], 1),
                "ARI": round(m["ari"], 3),
                "NMI": round(m["nmi"], 3),
                "Construtor": int(counts["Construtor"]),
                "Posicional": int(counts["Posicional"]),
                "Combativo": int(counts["Combativo"]),
                "Híbrido": int(counts["Híbrido"]),
            }
        )
    summary = pd.DataFrame(rows)
    print("=== TABELA COMPARATIVA ===")
    print(summary.to_string(index=False))
    print()

    # Confusion matrices (mapped to current taxonomy)
    for title, pred in [
        ("K-Means 6 comp", km_tactical),
        ("Option A GMM K=3", compare_to_current(a3, "option_a", OPTION_A_COMPARE)),
        ("Option A GMM K=4", compare_to_current(a4, "option_a", OPTION_A_COMPARE)),
        ("Option B PCA", b_mapped),
    ]:
        print(f"=== Matriz de confusão — {title} ===")
        print(confusion_df(current, pred, COMPARE_LABELS).to_string())
        print()

    # Agreement / disagreement highlights
    merged = df[["Jogador", "Equipe", "perfil"]].copy()
    merged["option_a3"] = a3["option_a"].values
    merged["option_a4"] = a4["option_a"].values
    merged["option_b"] = b["option_b"].values
    merged["a3_mapped"] = compare_to_current(a3, "option_a", OPTION_A_COMPARE).values
    merged["a4_mapped"] = compare_to_current(a4, "option_a", OPTION_A_COMPARE).values
    merged["b_mapped"] = b_mapped.values

    merged["all_new_agree"] = (merged["a3_mapped"] == merged["b_mapped"]) & (merged["a4_mapped"] == merged["b_mapped"])
    merged["any_diff_from_current"] = (
        (merged["a3_mapped"] != merged["perfil"])
        | (merged["a4_mapped"] != merged["perfil"])
        | (merged["b_mapped"] != merged["perfil"])
    )

    print("=== Novos modelos concordam entre si ===")
    print(f"{merged['all_new_agree'].sum()} / {len(merged)} atletas")
    print()

    changed = merged[merged["any_diff_from_current"]].sort_values("Jogador")
    print(f"=== Atletas que mudam vs atual ({len(changed)}) — amostra ===")
    for _, row in changed.head(20).iterrows():
        print(
            f"  {row['Jogador']:22s} atual={row['perfil']:10s} | "
            f"A3={row['option_a3']:10s} A4={row['option_a4']:10s} B={row['option_b']}"
        )
    if len(changed) > 20:
        print(f"  ... +{len(changed)-20} atletas")

    out = ROOT / "reference" / "profile_options_ab.json"
    payload = {
        "n_players": len(df),
        "summary": summary.to_dict(orient="records"),
        "pca_loadings": loadings.round(4).to_dict(),
        "players": merged[
            ["Jogador", "Equipe", "perfil", "option_a3", "option_a4", "option_b", "a3_mapped", "b_mapped"]
        ].to_dict(orient="records"),
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nArtefato: {out}")


if __name__ == "__main__":
    main()
