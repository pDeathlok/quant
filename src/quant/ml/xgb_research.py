"""Reusable XGBoost research model wrapper."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class XGBResearchModel:
    """Persisted model wrapper used by research scripts.

    It keeps the raw project feature order, fitted preprocessing objects, and
    the fitted XGBoost classifier together so downstream strategy exploration can
    call `predict_proba` without knowing training internals.
    """

    feature_names_in_: list[str]
    selected_features_: list[str]
    imputer: object
    selector: object | None
    classifier: object
    best_iteration: int | None = None
    factor_schema_version_: str | None = None

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        raw = X[self.feature_names_in_].replace([np.inf, -np.inf], np.nan)
        imputed = self.imputer.transform(raw)
        if self.selector is None:
            return imputed
        return self.selector.transform(imputed)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        transformed = self.transform(X)
        if self.best_iteration is None:
            return self.classifier.predict_proba(transformed)
        return self.classifier.predict_proba(transformed, iteration_range=(0, self.best_iteration + 1))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)
