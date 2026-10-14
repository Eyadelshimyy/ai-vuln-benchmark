from __future__ import annotations

from fast_mlsirm import FitConfig, MLS2PLMConfig, fit, simulate


def main() -> None:
    data = simulate(MLS2PLMConfig(n_persons=12, n_dims=1, items_per_dim=2, latent_dim=1, seed=20260702))
    result = fit(
        data.Y,
        data.factor_id,
        config=FitConfig(model="MLS2PLM", optimizer="adam", max_iter=1, n_restarts=1, backend="rust"),
    )
    print({"backend": result.backend, "model": result.model, "objective": float(result.objective)})


if __name__ == "__main__":
    main()
