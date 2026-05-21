import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from sklearn.preprocessing import StandardScaler


class MLDataSet:
    def __init__(
        self,
        data: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        lookback: int = 20,
        train_ratio: float = 0.8
    ):
        self.data = data
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.lookback = lookback
        self.train_ratio = train_ratio
        self.scaler = StandardScaler()

    def build_sequences(self) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []

        for i in range(self.lookback, len(self.data)):
            X.append(self.data[self.feature_cols].iloc[i-self.lookback:i].values)
            y.append(self.data[self.target_col].iloc[i])

        X = np.array(X)
        y = np.array(y)

        split_idx = int(len(X) * self.train_ratio)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        X_train_scaled = self.scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_test_scaled = self.scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

        return X_train_scaled, X_test_scaled, y_train, y_test

    def build_latest(self) -> np.ndarray:
        X = self.data[self.feature_cols].iloc[-self.lookback:].values
        X_scaled = self.scaler.transform(X).reshape(1, self.lookback, -1)
        return X_scaled
