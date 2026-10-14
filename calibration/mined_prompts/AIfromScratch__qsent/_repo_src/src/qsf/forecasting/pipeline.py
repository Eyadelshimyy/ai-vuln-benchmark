"""
End-to-end forecasting pipeline.

Combines the QSent sentiment pipeline output with the Phase 2 feature
engineering and classical ML models to produce price movement forecasts.
"""
import logging
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from qsf.features.engineering import engineer_features
from qsf.features.targets import create_targets
from qsf.features.preparation import prepare_data_bundle
from qsf.forecasting.models import (
    LogisticRegressionModel,
    RandomForestModel,
    RidgeRegressionModel,
    ModelResult,
)

logger = logging.getLogger(__name__)

ANNUALIZATION_FACTOR = 252.0


def _compute_metrics(
    y_true: np.ndarray, predictions: np.ndarray, model_type: str
) -> dict[str, float]:
    """Compute evaluation metrics for a model's predictions."""
    metrics: dict[str, float] = {}

    if model_type == "classifier":
        metrics["accuracy"] = float(np.mean(y_true == predictions))
        direction_true = (y_true > 0).astype(int) if y_true.dtype != int else y_true
        metrics["directional_accuracy"] = float(np.mean(direction_true == predictions))
    else:
        residuals = y_true - predictions
        metrics["rmse"] = float(np.sqrt(np.mean(residuals**2)))
        metrics["mae"] = float(np.mean(np.abs(residuals)))
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        metrics["r2"] = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        dir_correct = np.sign(predictions) == np.sign(y_true)
        metrics["directional_accuracy"] = float(np.mean(dir_correct))

    return metrics


def _compute_trading_metrics(
    returns: np.ndarray,
    predictions: np.ndarray,
    cost_bps: float = 20.0,
) -> dict[str, float]:
    """Simulate a sign-based strategy and compute trading metrics."""
    position = (predictions > 0).astype(float)
    turnover = np.abs(np.diff(position, prepend=position[0]))
    cost_per_side = (cost_bps / 2.0) / 10_000.0
    lagged_pos = np.roll(position, 1)
    lagged_pos[0] = 0.0

    strategy_returns = lagged_pos * returns - turnover * cost_per_side
    equity = np.cumprod(1.0 + strategy_returns)

    mean_daily = np.mean(strategy_returns)
    std_daily = np.std(strategy_returns, ddof=0)
    sharpe = 0.0
    if std_daily > 0:
        sharpe = float(
            (mean_daily * ANNUALIZATION_FACTOR)
            / (std_daily * np.sqrt(ANNUALIZATION_FACTOR))
        )

    cummax = np.maximum.accumulate(equity)
    drawdowns = (equity - cummax) / cummax
    max_drawdown = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

    return {
        "total_return_pct": float((equity[-1] - 1.0) * 100),
        "sharpe": sharpe,
        "max_drawdown_pct": float(max_drawdown * 100),
        "trade_fraction_pct": float(np.mean(position) * 100),
    }


class ForecastingPipeline:
    """Orchestrates feature engineering, model training, and evaluation."""

    def __init__(
        self,
        ticker: str = "IONQ",
        period: str = "2y",
        cost_bps: float = 20.0,
    ):
        self.ticker = ticker
        self.period = period
        self.cost_bps = cost_bps

    def run(
        self,
        sentiment_daily: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Execute the full forecasting pipeline.

        Parameters
        ----------
        sentiment_daily : pd.DataFrame, optional
            Daily sentiment scores from the QSent sentiment pipeline.
            Expected columns: news_sentiment, social_sentiment.

        Returns
        -------
        dict
            Pipeline results including model comparisons and best forecast.
        """
        # 1. Fetch market data
        logger.info("[%s] Fetching market data (%s)", self.ticker, self.period)
        hist = yf.Ticker(self.ticker).history(period=self.period)
        if hist.empty:
            return {"error": f"No market data for {self.ticker}"}
        hist.index = hist.index.tz_localize(None)

        # 2. Feature engineering
        logger.info("[%s] Engineering features", self.ticker)
        df_features, feature_cols = engineer_features(
            hist, sentiment_daily=sentiment_daily
        )

        # 3. Create targets
        df_targets, tau = create_targets(df_features)

        # 4. Prepare data bundle
        bundle = prepare_data_bundle(
            df_targets, feature_cols=feature_cols, primary_target="target_return"
        )

        X_train = bundle["X_train"]
        y_train = bundle["y_train"]
        X_test = bundle["X_test"]
        y_test = bundle["y_test"]
        X_val = bundle["X_val"]
        y_val = bundle["y_val"]

        # Binary direction targets for classifiers
        y_train_dir = (y_train > 0).astype(int)
        y_test_dir = (y_test > 0).astype(int)
        y_val_dir = (y_val > 0).astype(int)

        # 5. Train models
        logger.info("[%s] Training models", self.ticker)
        classifiers = [
            LogisticRegressionModel(),
            RandomForestModel(),
        ]
        regressors = [
            RidgeRegressionModel(),
        ]

        # Try to add XGBoost if available
        try:
            from qsf.forecasting.models import (
                XGBoostClassifierModel,
                XGBoostRegressorModel,
            )
            classifiers.append(XGBoostClassifierModel())
            regressors.append(XGBoostRegressorModel())
        except ImportError:
            logger.info("xgboost not available, skipping XGBoost models")

        model_results: list[dict] = []

        for model in classifiers:
            result = model.run(X_train, y_train_dir, X_test)
            metrics = _compute_metrics(y_test_dir, result.predictions, "classifier")

            # Trading metrics: convert classifier predictions to return-signed
            pred_returns = np.where(result.predictions == 1, 1.0, -1.0)
            trading = _compute_trading_metrics(y_test, pred_returns, self.cost_bps)

            model_results.append({
                "name": result.name,
                "type": "classifier",
                "metrics": metrics,
                "trading": trading,
            })

        for model in regressors:
            result = model.run(X_train, y_train, X_test)
            metrics = _compute_metrics(y_test, result.predictions, "regressor")
            trading = _compute_trading_metrics(
                y_test, result.predictions, self.cost_bps
            )

            model_results.append({
                "name": result.name,
                "type": "regressor",
                "metrics": metrics,
                "trading": trading,
            })

        # 6. Select best model by Sharpe ratio
        best = max(model_results, key=lambda r: r["trading"]["sharpe"])

        return {
            "ticker": self.ticker,
            "last_updated": datetime.now().isoformat(),
            "data_points": len(hist),
            "train_samples": len(bundle["train_df"]),
            "val_samples": len(bundle["val_df"]),
            "test_samples": len(bundle["test_df"]),
            "n_features": len(feature_cols),
            "feature_names": feature_cols,
            "dead_zone_tau": float(tau),
            "models": model_results,
            "best_model": {
                "name": best["name"],
                "sharpe": best["trading"]["sharpe"],
                "directional_accuracy": best["metrics"]["directional_accuracy"],
                "total_return_pct": best["trading"]["total_return_pct"],
                "max_drawdown_pct": best["trading"]["max_drawdown_pct"],
            },
        }
