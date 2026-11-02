"""Return-aligned regression ranker for left-side cross-sectional research."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBRegressor

from quant.ml.xgb_research import XGBResearchModel
from quant.research.right_side_unified import Float32NaNTransformer


RETURN_RANKER_SCHEMA_VERSION = "left-side-return-ranker-v1"
SUPPORTED_RETURN_TARGETS: tuple[str, ...] = (
    "terminal_return_daily_rank",
    "risk_adjusted_path_daily_rank",
)


def materialize_return_rank_targets(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach causal-outcome research targets ranked within each signal date."""

    required = {"date", "terminal_return", "mfe", "mae"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"left-side return targets miss columns: {missing}")
    out = frame.copy(deep=False)
    terminal = pd.to_numeric(out["terminal_return"], errors="coerce")
    mfe = pd.to_numeric(out["mfe"], errors="coerce")
    mae = pd.to_numeric(out["mae"], errors="coerce")
    # Reward realized terminal return, retain a small upside-path reward, and
    # penalize adverse excursion.  MAE is non-positive under the label contract.
    path_utility = terminal + 0.20 * mfe + 0.50 * mae
    out["terminal_return_daily_rank"] = terminal.groupby(out["date"]).rank(
        method="average", pct=True
    )
    out["risk_adjusted_path_daily_rank"] = path_utility.groupby(out["date"]).rank(
        method="average", pct=True
    )
    return out


@dataclass
class ReturnRankedProbabilityModel:
    """Persist a continuous ranker with a binary probability calibration head."""

    base_model: XGBResearchModel
    calibrator: LogisticRegression
    target_name: str
    probability_target: str
    schema_version: str = RETURN_RANKER_SCHEMA_VERSION

    @property
    def feature_names_in_(self) -> list[str]:
        return list(self.base_model.feature_names_in_)

    def predict_score(self, values: pd.DataFrame) -> np.ndarray:
        transformed = self.base_model.imputer.transform(
            values[self.feature_names_in_]
        )
        return np.asarray(
            self.base_model.classifier.predict(transformed), dtype=float
        )

    def predict_proba(self, values: pd.DataFrame) -> np.ndarray:
        score = self.predict_score(values).reshape(-1, 1)
        return np.asarray(self.calibrator.predict_proba(score), dtype=float)


def fit_return_ranker(
    train: pd.DataFrame,
    early_stop: pd.DataFrame,
    calibration: pd.DataFrame,
    *,
    features: Sequence[str],
    target: str,
    probability_target: str,
    n_jobs: int,
) -> ReturnRankedProbabilityModel:
    """Fit a deterministic XGBoost regressor and validation-only calibration."""

    if target not in SUPPORTED_RETURN_TARGETS:
        raise ValueError(f"unsupported left-side return target: {target}")
    feature_names = list(dict.fromkeys(str(value) for value in features))
    if not feature_names:
        raise ValueError("left-side return ranker requires at least one feature")
    for name, frame in (
        ("train", train),
        ("early_stop", early_stop),
        ("calibration", calibration),
    ):
        missing = sorted(
            set((*feature_names, target, probability_target)) - set(frame.columns)
        )
        if missing:
            raise ValueError(f"{name} frame misses return-ranker columns: {missing}")
        if frame[target].isna().any():
            raise ValueError(f"{name} frame contains missing return targets")

    transformer = Float32NaNTransformer()
    train_x = transformer.fit_transform(train[feature_names])
    early_x = transformer.transform(early_stop[feature_names])
    regressor = XGBRegressor(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.035,
        min_child_weight=8.0,
        subsample=0.85,
        colsample_bytree=0.80,
        reg_alpha=0.15,
        reg_lambda=4.0,
        max_bin=128,
        early_stopping_rounds=50,
        objective="reg:pseudohubererror",
        eval_metric="rmse",
        tree_method="hist",
        n_jobs=int(n_jobs),
        random_state=42,
    )
    regressor.fit(
        train_x,
        pd.to_numeric(train[target], errors="raise"),
        eval_set=[
            (
                early_x,
                pd.to_numeric(early_stop[target], errors="raise"),
            )
        ],
        verbose=False,
    )
    best_iteration = (
        int(regressor.best_iteration)
        if regressor.best_iteration is not None
        else None
    )
    base = XGBResearchModel(
        feature_names_in_=feature_names,
        selected_features_=feature_names,
        imputer=transformer,
        selector=None,
        classifier=regressor,
        best_iteration=best_iteration,
        factor_schema_version_=RETURN_RANKER_SCHEMA_VERSION,
    )
    calibration_x = transformer.transform(calibration[feature_names])
    calibration_score = np.asarray(
        regressor.predict(calibration_x), dtype=float
    ).reshape(-1, 1)
    calibration_target = calibration[probability_target].astype(int)
    if calibration_target.nunique() < 2:
        raise ValueError("probability calibration frame lacks both classes")
    calibrator = LogisticRegression(max_iter=1000, random_state=42)
    calibrator.fit(calibration_score, calibration_target)
    return ReturnRankedProbabilityModel(
        base_model=base,
        calibrator=calibrator,
        target_name=target,
        probability_target=probability_target,
    )


__all__ = [
    "RETURN_RANKER_SCHEMA_VERSION",
    "SUPPORTED_RETURN_TARGETS",
    "ReturnRankedProbabilityModel",
    "fit_return_ranker",
    "materialize_return_rank_targets",
]
