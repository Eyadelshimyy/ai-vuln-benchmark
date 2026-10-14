"""
Walk-forward validation framework from the Phase 2 notebook.

Uses rolling windows to simulate live deployment: train on a window of
historical data, predict the next period, roll forward, repeat.
"""
import logging
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from qsf.backtesting.metrics import compute_risk_metrics

logger = logging.getLogger(__name__)

DEFAULT_TRAIN_WINDOW = 252  # ~1 year of trading days
DEFAULT_TEST_WINDOW = 63    # ~1 quarter
DEFAULT_STEP_SIZE = 21      # ~1 month


def walk_forward_validate(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    model_factory: Callable[[], Any],
    train_window: int = DEFAULT_TRAIN_WINDOW,
    test_window: int = DEFAULT_TEST_WINDOW,
    step_size: int = DEFAULT_STEP_SIZE,
    is_classifier: bool = False,
) -> dict[str, Any]:
    """Run walk-forward validation on a time-series dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with features and target.
    feature_cols : list[str]
        Feature column names.
    target_col : str
        Target column name.
    model_factory : callable
        Returns a fresh model instance with .fit() and .predict() methods.
    train_window : int
        Number of rows in each training window.
    test_window : int
        Number of rows in each test window.
    step_size : int
        Number of rows to step forward between folds.
    is_classifier : bool
        If True, binarize target for training.

    Returns
    -------
    dict with keys: fold_results (list of per-fold dicts), all_predictions,
    all_actuals, all_dates, aggregate_metrics.
    """
    n = len(df)
    fold_results = []
    all_predictions = []
    all_actuals = []
    all_dates = []

    fold_idx = 0
    start = 0

    while start + train_window + test_window <= n:
        train_end = start + train_window
        test_end = min(train_end + test_window, n)

        train_slice = df.iloc[start:train_end]
        test_slice = df.iloc[train_end:test_end]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_slice[feature_cols].values)
        X_test = scaler.transform(test_slice[feature_cols].values)

        y_train = train_slice[target_col].values
        y_test = test_slice[target_col].values

        if is_classifier:
            y_train_fit = (y_train > 0).astype(int)
        else:
            y_train_fit = y_train

        model = model_factory()
        model.fit(X_train, y_train_fit)
        preds = model.predict(X_test)

        if is_classifier:
            pred_returns = np.where(preds == 1, 1.0, -1.0)
        else:
            pred_returns = preds

        dir_correct = np.sign(pred_returns) == np.sign(y_test)
        dir_acc = float(np.mean(dir_correct))

        fold_results.append({
            "fold": fold_idx,
            "train_start": str(train_slice.index[0].date()),
            "train_end": str(train_slice.index[-1].date()),
            "test_start": str(test_slice.index[0].date()),
            "test_end": str(test_slice.index[-1].date()),
            "test_size": len(test_slice),
            "directional_accuracy": dir_acc,
        })

        all_predictions.extend(pred_returns.tolist())
        all_actuals.extend(y_test.tolist())
        all_dates.extend([str(d.date()) for d in test_slice.index])

        fold_idx += 1
        start += step_size

    all_preds_arr = np.array(all_predictions)
    all_actuals_arr = np.array(all_actuals)

    aggregate = {}
    if len(all_preds_arr) > 0:
        dir_correct = np.sign(all_preds_arr) == np.sign(all_actuals_arr)
        aggregate["directional_accuracy"] = float(np.mean(dir_correct))
        aggregate["n_folds"] = fold_idx
        aggregate["total_predictions"] = len(all_preds_arr)

        risk = compute_risk_metrics(
            np.where(all_preds_arr > 0, 1.0, 0.0) * all_actuals_arr
        )
        aggregate.update(risk)

    return {
        "fold_results": fold_results,
        "all_predictions": all_predictions,
        "all_actuals": all_actuals,
        "all_dates": all_dates,
        "aggregate_metrics": aggregate,
    }
