"""
Classical ML forecasting models ported from the Phase 2 notebook.

Two-stage prediction strategy:
  Stage 1 — Direction classification (up/down)
  Stage 2 — Magnitude regression (predicted return size)

Models are kept simple and scikit-learn-based for this first integration pass.
Deep learning models (LSTM, GRU, NBEATS, TFT) are deferred to a future phase.
"""
import logging
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

logger = logging.getLogger(__name__)


class Classifier(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> None: ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


class Regressor(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> None: ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...


@dataclass
class ModelResult:
    name: str
    model_type: str  # "classifier" or "regressor"
    predictions: np.ndarray
    train_predictions: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Classifiers (Stage 1: direction prediction)
# ---------------------------------------------------------------------------

class LogisticRegressionModel:
    name = "LogisticRegression"

    def __init__(self, max_iter: int = 1000, C: float = 1.0):
        self._model = LogisticRegression(max_iter=max_iter, C=C, solver="lbfgs")

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)

    def run(
        self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray
    ) -> ModelResult:
        self.fit(X_train, y_train)
        return ModelResult(
            name=self.name,
            model_type="classifier",
            predictions=self.predict(X_test),
            train_predictions=self.predict(X_train),
        )


class RandomForestModel:
    name = "RandomForest"

    def __init__(self, n_estimators: int = 200, max_depth: int | None = 10, random_state: int = 42):
        self._model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)

    def feature_importances(self) -> np.ndarray:
        return self._model.feature_importances_

    def run(
        self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray
    ) -> ModelResult:
        self.fit(X_train, y_train)
        return ModelResult(
            name=self.name,
            model_type="classifier",
            predictions=self.predict(X_test),
            train_predictions=self.predict(X_train),
        )


class XGBoostClassifierModel:
    name = "XGBoost_Classifier"

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        random_state: int = 42,
    ):
        try:
            from xgboost import XGBClassifier
        except ImportError as e:
            raise ImportError("xgboost is required for XGBoostClassifierModel") from e
        self._model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            use_label_encoder=False,
            eval_metric="logloss",
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)

    def run(
        self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray
    ) -> ModelResult:
        self.fit(X_train, y_train)
        return ModelResult(
            name=self.name,
            model_type="classifier",
            predictions=self.predict(X_test),
            train_predictions=self.predict(X_train),
        )


# ---------------------------------------------------------------------------
# Regressors (Stage 2: magnitude prediction)
# ---------------------------------------------------------------------------

class RidgeRegressionModel:
    name = "Ridge"

    def __init__(self, alpha: float = 1.0):
        self._model = Ridge(alpha=alpha)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def run(
        self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray
    ) -> ModelResult:
        self.fit(X_train, y_train)
        return ModelResult(
            name=self.name,
            model_type="regressor",
            predictions=self.predict(X_test),
            train_predictions=self.predict(X_train),
        )


class XGBoostRegressorModel:
    name = "XGBoost_Regressor"

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        random_state: int = 42,
    ):
        try:
            from xgboost import XGBRegressor
        except ImportError as e:
            raise ImportError("xgboost is required for XGBoostRegressorModel") from e
        self._model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def run(
        self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray
    ) -> ModelResult:
        self.fit(X_train, y_train)
        return ModelResult(
            name=self.name,
            model_type="regressor",
            predictions=self.predict(X_test),
            train_predictions=self.predict(X_train),
        )
