#!/usr/bin/env python3
"""Phase 1 — exploratory profile diagnostics for zagueiros (CB/LCB/RCB)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, confusion_matrix, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from engine.load_data import load_players_dataframe
from engine.measures import attach_base_measures
from engine.positions import compute_family_metrics

ROOT = Path(__file__).resolve().parents[1]
FEATURES = [
    "ZG_Construção",
    "ZG_Condução2",
    "ZG_DuelosDefensivo",
    "ZG_DuelosAr",
    "ZG_LeituraDefensiva2",
]
PROFILE_ORDER = ["Combativo", "Construtor", "Posicional", "Híbrido"]
CLUSTER_TO_PROFILE = {
    "combativo": "Combativo",
    "construtor": "Construtor",
    "posicional": "Posicional",
}


def load_zagueiros() -> pd.DataFrame:
    df = attach_base_measures(load_players_dataframe())
    return compute_family_metrics(df, "zagueiros")


def name_cluster_by_centroid(centroid: np.ndarray, features: list[str]) -> str:
    mapping = {
        "ZG_DuelosDefensivo": "combativo",
        "ZG_Construção": "construtor",
        "ZG_Condução2": "construtor",
        "ZG_LeituraDefensiva2": "posicional",
        "ZG_DuelosAr": "posicional",
    }
    scores = {label: 0.0 for label in CLUSTER_TO_PROFILE}
    for feature, value in zip(features, centroid):
        scores[mapping[feature]] += value
    return CLUSTER_TO_PROFILE[max(scores, key=scores.get)]


def map_cluster_labels(model_labels: np.ndarray, centers: np.ndarray, features: list[str]) -> np.ndarray:
    cluster_names = [name_cluster_by_centroid(centers[i], features) for i in range(len(centers))]
    return np.array([cluster_names[label] for label in model_labels])


def assign_gmm_profiles(
    gmm: GaussianMixture,
    X_scaled: np.ndarray,
    scaler: StandardScaler,
    features: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    component_names = [
        name_cluster_by_centroid(scaler.inverse_transform(gmm.means_)[i], features)
        for i in range(gmm.n_components)
    ]
    proba = gmm.predict_proba(X_scaled)
    hard_idx = proba.argmax(axis=1)
    hard_labels = np.array([component_names[i] for i in hard_idx])
    max_prob = proba.max(axis=1)
    # Mark uncertain assignments as Híbrido
    hybrid_mask = max_prob < 0.45
    final = hard_labels.copy()
    final[hybrid_mask] = "Híbrido"
    return final, max_prob


def assign_kmeans_profiles(km: KMeans, X_scaled: np.ndarray, scaler: StandardScaler, features: list[str]) -> np.ndarray:
    centers = scaler.inverse_transform(km.cluster_centers_)
    cluster_names = [name_cluster_by_centroid(centers[i], features) for i in range(km.n_clusters)]
    return np.array([cluster_names[label] for label in km.labels_])


def assign_hierarchical_profiles(model: AgglomerativeClustering, X_scaled: np.ndarray, scaler: StandardScaler, features: list[str]) -> np.ndarray:
    # Recompute centroids from labels
    labels = model.labels_
    centers = []
    for cluster_id in sorted(set(labels)):
        mask = labels == cluster_id
        centers.append(X_scaled[mask].mean(axis=0))
    centers = np.array(centers)
    centers_raw = scaler.inverse_transform(centers)
    cluster_names = [name_cluster_by_centroid(centers_raw[i], features) for i in range(len(centers))]
    return np.array([cluster_names[label] for label in labels])


def confusion_table(y_true: pd.Series, y_pred: pd.Series) -> pd.DataFrame:
    labels = PROFILE_ORDER
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(matrix, index=[f"atual:{p}" for p in labels], columns=[f"pred:{p}" for p in labels])


def main() -> None:
    zags = load_zagueiros()
    print(f"Zagueiros elegíveis: {len(zags)}")
    print(zags["perfil"].value_counts().to_string())
    print()

    X = zags[FEATURES].to_numpy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=min(3, len(FEATURES)))
    X_pca = pca.fit_transform(X_scaled)
    print("=== PCA ===")
    for i, ratio in enumerate(pca.explained_variance_ratio_, start=1):
        print(f"  PC{i}: {ratio*100:.1f}% da variância")
    print(f"  Acumulado 2 PCs: {pca.explained_variance_ratio_[:2].sum()*100:.1f}%")
    loadings = pd.DataFrame(pca.components_.T, index=FEATURES, columns=[f"PC{i+1}" for i in range(pca.n_components_)])
    print("\nLoadings (top por PC):")
    for col in loadings.columns:
        top = loadings[col].abs().sort_values(ascending=False).head(3)
        print(f"  {col}: " + ", ".join(f"{idx} ({loadings.loc[idx, col]:+.2f})" for idx in top.index))
    print()

    results = zags[["Jogador", "Equipe", "perfil"]].copy()
    results = results.rename(columns={"perfil": "atual"})

    # KMeans K=3
    km = KMeans(n_clusters=3, random_state=42, n_init=30)
    km.fit(X_scaled)
    results["kmeans"] = assign_kmeans_profiles(km, X_scaled, scaler, FEATURES)
    print(f"KMeans silhouette (K=3): {silhouette_score(X_scaled, km.labels_):.3f}")
    print(f"KMeans ARI vs atual: {adjusted_rand_score(results['atual'], results['kmeans']):.3f}")

    # Hierarchical K=3
    hc = AgglomerativeClustering(n_clusters=3)
    hc.fit(X_scaled)
    results["hier"] = assign_hierarchical_profiles(hc, X_scaled, scaler, FEATURES)
    print(f"Hierarchical ARI vs atual: {adjusted_rand_score(results['atual'], results['hier']):.3f}")

    # GMM K=3
    gmm = GaussianMixture(n_components=3, random_state=42, covariance_type="full", n_init=10)
    gmm.fit(X_scaled)
    gmm_labels, gmm_conf = assign_gmm_profiles(gmm, X_scaled, scaler, FEATURES)
    results["gmm"] = gmm_labels
    results["gmm_conf"] = gmm_conf.round(3)
    print(f"GMM BIC (K=3): {gmm.bic(X_scaled):.1f}")
    print(f"GMM ARI vs atual: {adjusted_rand_score(results['atual'], results['gmm']):.3f}")
    print()

    methods = ["kmeans", "hier", "gmm"]
    for method in methods:
        print(f"=== Matriz de confusão — {method} vs atual ===")
        print(confusion_table(results["atual"], results[method]).to_string())
        print()

    # Stability: agreement across methods
    def mode_label(row: pd.Series) -> str:
        votes = [row["kmeans"], row["hier"], row["gmm"]]
        if len(set(votes)) == 1:
            return votes[0]
        counts = pd.Series(votes).value_counts()
        if counts.iloc[0] == counts.iloc[1] and len(counts) > 1:
            return "Instável"
        return counts.index[0]

    results["consenso"] = results.apply(mode_label, axis=1)
    results["instavel"] = results.apply(
        lambda r: len({r["kmeans"], r["hier"], r["gmm"]}) > 1,
        axis=1,
    )
    results["muda_vs_atual"] = results.apply(
        lambda r: r["atual"] != r["kmeans"] or r["atual"] != r["hier"] or r["atual"] != r["gmm"],
        axis=1,
    )

    print("=== Estabilidade entre métodos ===")
    print(f"Consenso total (3 métodos iguais): {(~results['instavel']).sum()} / {len(results)}")
    print(f"Instáveis (discordância entre métodos): {results['instavel'].sum()} / {len(results)}")
    print(f"Divergem do label atual em ≥1 método: {results['muda_vs_atual'].sum()} / {len(results)}")
    print()

    unstable = results[results["instavel"]].sort_values(["atual", "Jogador"])
    print("=== Atletas instáveis (kmeans ≠ hier ≠ gmm ou empate) ===")
    for _, row in unstable.iterrows():
        print(
            f"  {row['Jogador']:22s} ({row['Equipe']:18s}) "
            f"atual={row['atual']:10s} | km={row['kmeans']:10s} hi={row['hier']:10s} gmm={row['gmm']:10s}"
        )
    print()

    changed = results[results["muda_vs_atual"] & ~results["instavel"]].sort_values("Jogador")
    print("=== Concordam entre si, mas diferem do atual ===")
    for _, row in changed.iterrows():
        print(
            f"  {row['Jogador']:22s} atual={row['atual']:10s} -> "
            f"km={row['kmeans']} hi={row['hier']} gmm={row['gmm']}"
        )
    print()

    # Save artifact
    out = ROOT / "reference" / "profile_analysis_zag.json"
    payload = {
        "n_players": len(results),
        "pca_variance": pca.explained_variance_ratio_.tolist(),
        "confusion": {m: confusion_table(results["atual"], results[m]).to_dict() for m in methods},
        "unstable_players": unstable[
            ["Jogador", "Equipe", "atual", "kmeans", "hier", "gmm", "gmm_conf"]
        ].to_dict(orient="records"),
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Artefato salvo em {out}")


if __name__ == "__main__":
    main()
