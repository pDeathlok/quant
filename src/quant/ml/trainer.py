import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass


@dataclass
class TrainingResult:
    model: Any
    train_score: float
    test_score: float
    feature_importance: Optional[pd.Series] = None


class ModelTrainer:
    def __init__(
        self,
        model_class,
        model_params: Optional[Dict] = None
    ):
        self.model_class = model_class
        self.model_params = model_params or {}

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> TrainingResult:
        if len(X_train.shape) > 2:
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            X_test_flat = X_test.reshape(X_test.shape[0], -1)
        else:
            X_train_flat = X_train
            X_test_flat = X_test

        model = self.model_class(**self.model_params)
        model.fit(X_train_flat, y_train)

        train_score = model.score(X_train_flat, y_train)
        test_score = model.score(X_test_flat, y_test)

        feature_importance = None
        if hasattr(model, "feature_importances_") and feature_names is not None:
            feature_importance = pd.Series(
                model.feature_importances_,
                index=feature_names
            ).sort_values(ascending=False)

        return TrainingResult(
            model=model,
            train_score=train_score,
            test_score=test_score,
            feature_importance=feature_importance
        )


class WalkForwardTrainer:
    def __init__(
        self,
        model_class,
        model_params: Optional[Dict] = None,
        train_window: int = 100,
        test_window: int = 20
    ):
        self.model_class = model_class
        self.model_params = model_params or {}
        self.train_window = train_window
        self.test_window = test_window

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> pd.DataFrame:
        results = []

        for i in range(self.train_window, len(X) - self.test_window, self.test_window):
            X_train = X[i - self.train_window:i]
            y_train = y[i - self.train_window:i]
            X_test = X[i:i + self.test_window]
            y_test = y[i:i + self.test_window]

            if len(X_train.shape) > 2:
                X_train_flat = X_train.reshape(X_train.shape[0], -1)
                X_test_flat = X_test.reshape(X_test.shape[0], -1)
            else:
                X_train_flat = X_train
                X_test_flat = X_test

            model = self.model_class(**self.model_params)
            model.fit(X_train_flat, y_train)

            train_pred = model.predict(X_train_flat)
            test_pred = model.predict(X_test_flat)

            train_error = np.mean((y_train - train_pred) ** 2)
            test_error = np.mean((y_test - test_pred) ** 2)

            results.append({
                "train_start": i - self.train_window,
                "train_end": i,
                "test_start": i,
                "test_end": i + self.test_window,
                "train_error": train_error,
                "test_error": test_error
            })

        return pd.DataFrame(results)
