#!/usr/bin/env python3
"""Complete clustering study for 71 zagueiros using 11 raw per-90 stats."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    confusion_matrix,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from engine.load_data import load_players_dataframe
from engine.measures import attach_base_measures
from engine.positions import compute_family_metrics

ROOT = Path(__file__).resolve().parents[1]

RAW_FEATURES = [
    "Duelos defensivos/90",
    "Duelos aérios/90",
    "Remates intercetados/90",
    "Interseções/90",
    "Cortes/90",
    "Faltas/90",
    "Duelos ofensivos/90",
    "Corridas progressivas/90",
    "Passes longos/90",
    "Passes para terço final/90",
    "Passes progressivos/90",
]

DEFENSE = [
    "Duelos defensivos/90",
    "Duelos aérios/90",
    "Remates intercetados/90",
    "Interseções/90",
    "Cortes/90",
    "Faltas/90",
]
ATTACK = [
    "Duelos ofensivos/90",
    "Corridas progressivas/90",
    "Passes longos/90",
    "Passes para terço final/90",
    "Passes progressivos/90",
]

PROFILE_ORDER = ["Combativo", "Construtor", "Posicional", "Híbrido"]


def load_data() -> pd.DataFrame:
    raw = attach_base_measures(load_players_dataframe())
    zags = compute_family_metrics(raw, "zagueiros")
    pool = raw[(raw["Posição"] == "Zagueiro") & (raw["player_id"].isin(zags["player_id"]))].copy()
    pool = pool.set_index("player_id").loc[zags["player_id"].values].reset_index()
    merged = zags.merge(pool, on="player_id", suffixes=("", "_raw"))
    return merged


def label_cluster_from_centroid(centroid: pd.Series) -> str:
    def_mean = float(centroid[DEFENSE].mean())
    att_mean = float(centroid[ATTACK].mean())
    duel_def = float(centroid["Duelos defensivos/90"])
    passes = float(centroid[["Passes progressivos/90", "Passes longos/90", "Passes para terço final/90"]].mean())
    if passes > def_mean * 1.15 and passes > att_mean:
        return "Construtor"
    if duel_def > passes * 1.1 and duel_def >= float(centroid["Interseções/90"]):
        return "Combativo"
    if float(centroid["Interseções/90"]) + float(centroid["Duelos aérios/90"]) > passes:
        return "Posicional"
    return "Misto"


def assign_unique_labels(centroids: pd.DataFrame) -> list[str]:
    k = len(centroids)
    profiles = ["Construtor", "Combativo", "Posicional", "Misto"]
    scores = []
    for i, row in centroids.iterrows():
        scores.append(
            {
                "i": i,
                "Construtor": float(row[["Passes progressivos/90", "Passes longos/90", "Passes para terço final/90"]].mean()),
                "Combativo": float(row["Duelos defensivos/90"] + row["Cortes/90"] * 0.5),
                "Posicional": float(row["Interseções/90"] + row["Duelos aérios/90"] + row["Remates intercetados/90"]),
            }
        )
    assigned: dict[int, str] = {}
    for profile in profiles[: min(k, 3)]:
        best = max((s for s in scores if s["i"] not in assigned), key=lambda s: s[profile])
        assigned[best["i"]] = profile
    for s in scores:
        if s["i"] not in assigned:
            assigned[s["i"]] = "Misto"
    return [assigned[i] for i in range(k)]


def compare_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    return {
        "agree_pct": float((y_true.values == y_pred.values).mean() * 100),
        "ari": float(adjusted_rand_score(y_true, y_pred)),
        "nmi": float(normalized_mutual_info_score(y_true, y_pred)),
    }


def main() -> None:
    df = load_data()
    Xdf = df[RAW_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0)
    y = df["perfil"]

    print(f"{'='*70}")
    print(f"ESTUDO COMPLETO — 11 STATS BRUTOS — {len(df)} ZAGUEIROS")
    print(f"{'='*70}\n")

    print("Features:")
    for f in RAW_FEATURES:
        print(f"  • {f}")
    print()

    # 1. Descriptive stats
    print("=== 1. Estatísticas descritivas (média / dp / mediana) ===")
    desc = Xdf.agg(["mean", "std", "median"]).T.round(2)
    print(desc.to_string())
    print()

    # 2. Correlation
    print("=== 2. Correlação entre features (Pearson) ===")
    corr = Xdf.corr().round(2)
    # highlight blocks
    print(corr.to_string())
    print("\nBlocos:")
    print(f"  Defesa interna (6 stats): corr média = {corr.loc[DEFENSE, DEFENSE].where(~np.eye(len(DEFENSE), dtype=bool)).stack().mean():.2f}")
    print(f"  Construção/prog (5 stats): corr média = {corr.loc[ATTACK, ATTACK].where(~np.eye(len(ATTACK), dtype=bool)).stack().mean():.2f}")
    print(f"  Defesa × Construção: corr média = {corr.loc[DEFENSE, ATTACK].stack().mean():.2f}")
    print()

    scaler = StandardScaler()
    X = scaler.fit_transform(Xdf)

    # 3. PCA
    pca = PCA(n_components=min(5, len(RAW_FEATURES)))
    X_pca = pca.fit_transform(X)
    print("=== 3. PCA ===")
    for i, v in enumerate(pca.explained_variance_ratio_, 1):
        print(f"  PC{i}: {v*100:.1f}%")
    print(f"  Acumulado PC1-3: {pca.explained_variance_ratio_[:3].sum()*100:.1f}%")
    loadings = pd.DataFrame(pca.components_.T, index=RAW_FEATURES, columns=[f"PC{i}" for i in range(1, pca.n_components_ + 1)])
    print("\nLoadings PC1-PC3:")
    print(loadings[["PC1", "PC2", "PC3"]].round(3).to_string())
    print()

    # 4. K selection
    print("=== 4. Seleção de K (2–6) ===")
    k_rows = []
    for k in range(2, 7):
        km = KMeans(k, random_state=42, n_init=30).fit(X)
        gmm = GaussianMixture(k, random_state=42, n_init=10).fit(X)
        k_rows.append(
            {
                "K": k,
                "silhouette": silhouette_score(X, km.labels_),
                "calinski": calinski_harabasz_score(X, km.labels_),
                "davies_bouldin": davies_bouldin_score(X, km.labels_),
                "gmm_bic": gmm.bic(X),
                "ari_vs_atual": adjusted_rand_score(y, km.labels_),
            }
        )
    kdf = pd.DataFrame(k_rows)
    print(kdf.round(3).to_string(index=False))
    best_k_sil = int(kdf.loc[kdf["silhouette"].idxmax(), "K"])
    best_k_bic = int(kdf.loc[kdf["gmm_bic"].idxmin(), "K"])
    print(f"\n  Melhor K por silhouette: {best_k_sil}")
    print(f"  Melhor K por GMM-BIC: {best_k_bic}")
    print()

    # 5. Clustering K=3 (tactical) and K=4
    results = {}
    for k in [3, 4, best_k_bic]:
        km = KMeans(k, random_state=42, n_init=30).fit(X)
        hc = AgglomerativeClustering(k).fit(X)
        gmm = GaussianMixture(k, random_state=42, n_init=10).fit(X)

        centroids = pd.DataFrame(scaler.inverse_transform(km.cluster_centers_), columns=RAW_FEATURES)
        labels = assign_unique_labels(centroids)

        km_mapped = pd.Series([labels[l] for l in km.labels_])
        # map Misto -> Híbrido for comparison
        km_compare = km_mapped.replace({"Misto": "Híbrido", "Combativo": "Combativo", "Construtor": "Construtor", "Posicional": "Posicional"})

        results[k] = {
            "km": km,
            "hc": hc,
            "gmm": gmm,
            "centroids": centroids,
            "labels": labels,
            "km_mapped": km_compare,
        }

        print(f"=== 5. Clustering K={k} ===")
        print(f"K-Means labels: {dict(km_compare.value_counts())}")
        m = compare_metrics(y, km_compare)
        print(f"K-Means vs DAX: concordância={m['agree_pct']:.1f}% ARI={m['ari']:.3f} NMI={m['nmi']:.3f}")

        gmm_hard = gmm.predict(X)
        gmm_centroids = pd.DataFrame(scaler.inverse_transform(gmm.means_), columns=RAW_FEATURES)
        gmm_labels = assign_unique_labels(gmm_centroids)
        gmm_mapped = pd.Series([gmm_labels[i].replace("Misto", "Híbrido") for i in gmm_hard])
        mg = compare_metrics(y, gmm_mapped)
        print(f"GMM vs DAX:     concordância={mg['agree_pct']:.1f}% ARI={mg['ari']:.3f} NMI={mg['nmi']:.3f}")

        hc_centroids_idx = []
        for cid in range(k):
            hc_centroids_idx.append(X[hc.labels_ == cid].mean(axis=0))
        hc_centroids = pd.DataFrame(scaler.inverse_transform(np.array(hc_centroids_idx)), columns=RAW_FEATURES)
        hc_labels = assign_unique_labels(hc_centroids)
        hc_mapped = pd.Series([hc_labels[l].replace("Misto", "Híbrido") for l in hc.labels_])
        mh = compare_metrics(y, hc_mapped)
        print(f"Hier vs DAX:    concordância={mh['agree_pct']:.1f}% ARI={mh['ari']:.3f} NMI={mh['nmi']:.3f}")
        print()

    # 6. Centroids K=3 detailed
    k3 = results[3]
    print("=== 6. Centróides K=3 (valores reais /90) ===")
    for i, row in k3["centroids"].iterrows():
        label = k3["labels"][i]
        n = int((k3["km"].labels_ == i).sum())
        print(f"\nCluster {i} → {label} (n={n})")
        for block_name, block in [("DEFESA", DEFENSE), ("CONSTRUÇÃO", ATTACK)]:
            print(f"  {block_name}:")
            for col in block:
                print(f"    {col:35s} {row[col]:6.2f}")
    print()

    # 7. Confusion matrix K=3 KMeans vs DAX
    km3 = results[3]["km_mapped"]
    print("=== 7. Matriz de confusão — K-Means K=3 vs DAX ===")
    matrix = confusion_matrix(y, km3, labels=PROFILE_ORDER)
    print(pd.DataFrame(matrix, index=[f"atual:{p}" for p in PROFILE_ORDER], columns=[f"pred:{p}" for p in PROFILE_ORDER]).to_string())
    print()

    # 8. PCA quadrants on raw 11
    pc1, pc2 = X_pca[:, 0], X_pca[:, 1]
    quad_labels = []
    for a, b in zip(pc1, pc2):
        if a > 0.4 and b < -0.2:
            quad_labels.append("Construtor")
        elif a < -0.2 and b > 0.4:
            quad_labels.append("Combativo")
        elif a > 0.3 and b > 0.3:
            quad_labels.append("Posicional")
        else:
            quad_labels.append("Híbrido")
    quad = pd.Series(quad_labels)
    mq = compare_metrics(y, quad)
    print("=== 8. PCA quadrantes (PC1 vs PC2) ===")
    print(quad.value_counts().to_string())
    print(f"vs DAX: concordância={mq['agree_pct']:.1f}% ARI={mq['ari']:.3f}")
    print()

    # 9. Bootstrap stability
    rng = np.random.default_rng(42)
    def bootstrap(k: int, reps: int = 80) -> float:
        base = KMeans(k, random_state=42, n_init=20).fit(X).labels_
        scores = []
        for _ in range(reps):
            idx = rng.choice(len(X), size=int(len(X) * 0.8), replace=True)
            lab = KMeans(k, random_state=42, n_init=10).fit(X[idx]).labels_
            scores.append(adjusted_rand_score(base[idx], lab))
        return float(np.mean(scores))

    print("=== 9. Estabilidade bootstrap (K-Means) ===")
    for k in [2, 3, 4]:
        print(f"  K={k}: ARI bootstrap médio = {bootstrap(k):.3f}")
    print()

    # 10. Player-level output for K=3
    out_df = df[["Jogador", "Equipe", "perfil"]].copy()
    out_df["cluster_k3"] = results[3]["km"].labels_
    out_df["cluster_label"] = km3.values
    out_df["pc1"] = X_pca[:, 0].round(3)
    out_df["pc2"] = X_pca[:, 1].round(3)
    out_df["changed"] = out_df["perfil"] != out_df["cluster_label"]

    unstable = out_df[out_df["changed"]].sort_values("Jogador")
    print(f"=== 10. Atletas que mudam vs DAX ({len(unstable)}/{len(out_df)}) ===")
    for _, r in unstable.head(25).iterrows():
        print(f"  {r['Jogador']:22s} DAX={r['perfil']:10s} → cluster={r['cluster_label']}")
    if len(unstable) > 25:
        print(f"  ... +{len(unstable)-25} atletas")
    print()

    # 11. Summary vs other approaches
    print("=== 11. RESUMO vs abordagens anteriores ===")
    summary = [
        ("DAX atual (5 dim ZG)", 100.0, 1.0),
        ("11 stats brutos K=3", compare_metrics(y, km3)["agree_pct"], compare_metrics(y, km3)["ari"]),
        ("6 composites K=3 (ref.)", 29.6, 0.440),
        ("GMM 6 comp K=3 (ref.)", 54.9, 0.306),
        ("PCA quadrantes 11 raw", mq["agree_pct"], mq["ari"]),
    ]
    print(f"{'Modelo':30s} {'Concord%':>10s} {'ARI':>8s}")
    for name, agree, ari in summary:
        print(f"{name:30s} {agree:10.1f} {ari:8.3f}")

    artifact = ROOT / "reference" / "profile_raw11_study.json"
    artifact.write_text(
        json.dumps(
            {
                "features": RAW_FEATURES,
                "n_players": len(df),
                "descriptive": desc.reset_index().to_dict(orient="records"),
                "pca_variance": pca.explained_variance_ratio_.tolist(),
                "k_selection": kdf.to_dict(orient="records"),
                "players": out_df.to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nArtefato: {artifact}")


if __name__ == "__main__":
    main()
