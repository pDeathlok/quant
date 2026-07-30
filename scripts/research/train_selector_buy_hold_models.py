#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train time-split selector buy/hold return models with fixed-history scores.

The two model targets deliberately remain separate:

* buy: future five-session maximum high return;
* hold: future five-session close-to-close return.

Displayed scores are fixed-reference percentiles of predicted returns.  This
makes scores comparable across dates and strictly ordered by model-predicted
return, without using the current candidate pool as a normalization input.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quant.features.daily_factor_layer import BASE_FACTOR_COLUMNS


DEFAULT_HISTORY = PROJECT_ROOT / "data/research/selector_history_full/selector_stock_history_samples.parquet"
DEFAULT_FACTOR_DATA = PROJECT_ROOT / "data/features/z_skill_model_dataset.parquet"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports/selector_buy_hold_model"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models/candidates/selector_buy_hold"
TARGETS = {
    "buy": {
        "column": "future_max_high_t5_pct",
        "training_target": "clipped_return",
        "clip": (-5.0, 30.0),
        "hit_threshold": 5.0,
        "min_test_spearman": 0.05,
        "min_test_decile_spread": 1.0,
        "normalization_width": 6.0,
    },
    "hold": {
        "column": "future_return_t5_pct",
        "training_target": "daily_return_rank",
        "clip": (-15.0, 15.0),
        "hit_threshold": 0.0,
        "min_test_spearman": 0.02,
        "min_test_decile_spread": 0.25,
        "normalization_width": 2.0,
    },
}

# Keep only scale-free or bounded factors that are available at signal close.
# Absolute price/volume levels are excluded to reduce regime and stock-price bias.
MODEL_FACTOR_COLUMNS = [
    col
    for col in BASE_FACTOR_COLUMNS
    if col.startswith("alpha")
    or col.startswith("amplitude")
    or col.startswith("bias")
    or col.startswith("kdj_")
    or col.startswith("volume_relative")
    or col.startswith("volume_change")
    or col.startswith("volume_zscore")
    or col.startswith("volume_breakout")
    or col.startswith("volume_price_strength")
    or col.startswith("weekly_ma")
    or col.startswith("weekly_bull")
    or col.startswith("z_")
    or col
    in {
        "cci",
        "bbi_ma60_diff",
        "bbi_ma60_ratio",
        "macd_hist",
        "momentum_5d",
        "momentum_20d",
        "momentum_60d",
        "pct_chg",
        "psy_24",
        "return_1d",
        "return_5d",
        "return_10d",
        "return_60d",
        "return_120d",
        "reversal_5d",
        "rsi_12",
        "yidong_20d",
        "strong_yidong_20d",
        "days_since_yidong",
        "post_yidong_shrink",
        "ground_volume_60d",
        "b2_confirm_3d",
        "s1_distribution",
        "sell_score_simple",
    }
]


@dataclass(frozen=True)
class SplitMetrics:
    rows: int
    days: int
    spearman: float
    decile_spread: float
    decile_trend: float
    top20_avg_return: float
    top20_hit_rate: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train selector buy/hold historical-score models.")
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--factor-data", type=Path, default=DEFAULT_FACTOR_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--train-start", default=None)
    parser.add_argument("--train-end", default="2025-12-31")
    parser.add_argument("--valid-end", default="2026-03-31")
    parser.add_argument("--test-end", default="2026-06-30")
    parser.add_argument("--promote", action="store_true", help="Write passing models to models/production.")
    parser.add_argument("--n-jobs", type=int, default=6)
    return parser.parse_args()


def _group_values(value: Any) -> list[str]:
    if isinstance(value, np.ndarray):
        return [str(item) for item in value.tolist()]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return [item.strip() for item in str(value).strip("[]").replace("'", "").split(",") if item.strip()]


def load_dataset(history_path: Path, factor_path: Path) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    import pyarrow.parquet as pq

    history_schema = set(pq.read_schema(history_path).names)
    history_factor_columns = sorted(column for column in history_schema if column.startswith("selector_"))
    stock_columns = [
        "symbol",
        "date",
        "matched_count",
        "matched_groups",
        "best_profit_factor",
        "best_avg_return_pct",
        *(item["column"] for item in TARGETS.values()),
        *history_factor_columns,
    ]
    stock = pd.read_parquet(history_path, columns=stock_columns)
    stock["date"] = pd.to_datetime(stock["date"], errors="coerce")

    available = set(pd.read_parquet(factor_path, columns=[]).columns)
    # PyArrow returns no schema columns for columns=[] on some versions.
    if not available:
        available = set(pq.read_schema(factor_path).names)
    factor_columns = [column for column in MODEL_FACTOR_COLUMNS if column in available]
    factors = pd.read_parquet(factor_path, columns=["symbol", "date", *factor_columns])
    factors["date"] = pd.to_datetime(factors["date"], errors="coerce")
    factors = factors.sort_values("date").drop_duplicates(["symbol", "date"], keep="last")

    data = stock.merge(factors, on=["symbol", "date"], how="left", indicator="_factor_merge")
    factor_coverage = float(data["_factor_merge"].eq("both").mean())
    data = data.drop(columns="_factor_merge")

    all_groups = sorted({group for value in data["matched_groups"] for group in _group_values(value)})
    group_features = []
    for group in all_groups:
        column = f"group__{group}"
        data[column] = data["matched_groups"].map(lambda value, target=group: float(target in _group_values(value)))
        group_features.append(column)

    numeric_context = ["matched_count", "best_profit_factor", "best_avg_return_pct"]
    numeric_context = [column for column in numeric_context if data[column].notna().any()]
    features = [*history_factor_columns, *factor_columns, *numeric_context, *group_features]
    for column in features:
        data[column] = pd.to_numeric(data[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    features = [column for column in features if data[column].notna().any()]
    metadata = {
        "history_rows": int(len(stock)),
        "merged_rows": int(len(data)),
        "factor_coverage": factor_coverage,
        "factor_columns": len(factor_columns),
        "history_factor_columns": len(history_factor_columns),
        "group_columns": len(group_features),
        "date_min": data["date"].min().date().isoformat(),
        "date_max": data["date"].max().date().isoformat(),
    }
    return data, features, metadata


def _split_masks(data: pd.DataFrame, args: argparse.Namespace) -> dict[str, pd.Series]:
    train_end = pd.Timestamp(args.train_end)
    valid_end = pd.Timestamp(args.valid_end)
    test_end = pd.Timestamp(args.test_end)
    train = data["date"] <= train_end
    if args.train_start:
        train &= data["date"] >= pd.Timestamp(args.train_start)
    return {
        "train": train,
        "valid": (data["date"] > train_end) & (data["date"] <= valid_end),
        "test": (data["date"] > valid_end) & (data["date"] <= test_end),
    }


def _model_candidates(n_jobs: int) -> list[tuple[str, dict[str, Any]]]:
    common = {
        "objective": "reg:pseudohubererror",
        "n_estimators": 450,
        "subsample": 0.8,
        "colsample_bytree": 0.75,
        "min_child_weight": 20,
        "reg_lambda": 8.0,
        "random_state": 42,
        "n_jobs": n_jobs,
        "tree_method": "hist",
    }
    return [
        ("shallow", {**common, "max_depth": 3, "learning_rate": 0.035}),
        ("balanced", {**common, "max_depth": 5, "learning_rate": 0.025}),
    ]


def historical_percentile_scores(
    predictions: np.ndarray,
    reference: np.ndarray,
    normalization_width: float = 2.0,
) -> np.ndarray:
    """Map predictions monotonically using a fixed robust historical scale.

    An arctangent tail keeps out-of-range future regimes ordered instead of
    collapsing every prediction above the historical maximum to 100.
    """
    values = np.asarray(reference, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        raise ValueError("Historical score reference cannot be empty")
    median = float(np.median(values))
    q25, q75 = np.quantile(values, [0.25, 0.75])
    scale = max(float((q75 - q25) / 1.349), 1e-6)
    z_score = (np.asarray(predictions, dtype=float) - median) / scale
    width = max(float(normalization_width), 0.1)
    return np.clip(50.0 + 100.0 / np.pi * np.arctan(z_score / width), 0.0, 100.0)


def evaluate_predictions(
    dates: pd.Series,
    target: pd.Series,
    predictions: np.ndarray,
    hit_threshold: float,
) -> tuple[SplitMetrics, pd.DataFrame]:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(dates).to_numpy(),
            "target": pd.to_numeric(target, errors="coerce").to_numpy(),
            "prediction": np.asarray(predictions, dtype=float),
        }
    ).dropna()
    if frame.empty:
        raise ValueError("Evaluation split is empty")
    correlation = spearmanr(frame["prediction"], frame["target"]).statistic
    frame["rank"] = frame["prediction"].rank(method="first", pct=True)
    frame["decile"] = np.ceil(frame["rank"] * 10).clip(1, 10).astype(int)
    frame["hit"] = frame["target"] > hit_threshold
    deciles = frame.groupby("decile", as_index=False).agg(
        rows=("target", "size"),
        avg_return=("target", "mean"),
        hit_rate=("hit", "mean"),
        avg_prediction=("prediction", "mean"),
    )
    decile_spread = float(deciles.loc[deciles["decile"] == 10, "avg_return"].iloc[0]) - float(
        deciles.loc[deciles["decile"] == 1, "avg_return"].iloc[0]
    )
    decile_trend = spearmanr(deciles["decile"], deciles["avg_return"]).statistic
    top20 = frame.sort_values(["date", "prediction"], ascending=[True, False]).groupby("date").head(20)
    metrics = SplitMetrics(
        rows=int(len(frame)),
        days=int(frame["date"].nunique()),
        spearman=float(correlation) if np.isfinite(correlation) else 0.0,
        decile_spread=decile_spread,
        decile_trend=float(decile_trend) if np.isfinite(decile_trend) else 0.0,
        top20_avg_return=float(top20["target"].mean()),
        top20_hit_rate=float(top20["hit"].mean()),
    )
    return metrics, deciles


def evaluate_stability(
    dates: pd.Series,
    target: pd.Series,
    predictions: np.ndarray,
    hit_threshold: float,
    block_sessions: int = 20,
) -> dict[str, Any]:
    """Measure ordering stability in consecutive out-of-sample session blocks."""
    unique_dates = pd.Series(pd.to_datetime(dates).dropna().unique()).sort_values().tolist()
    blocks = []
    date_values = pd.to_datetime(dates)
    for offset in range(0, len(unique_dates), block_sessions):
        block_dates = unique_dates[offset : offset + block_sessions]
        if len(block_dates) < max(5, block_sessions // 2):
            continue
        mask = date_values.isin(block_dates)
        metrics, _ = evaluate_predictions(
            dates.loc[mask], target.loc[mask], np.asarray(predictions)[mask.to_numpy()], hit_threshold
        )
        blocks.append(
            {
                "date_start": pd.Timestamp(block_dates[0]).date().isoformat(),
                "date_end": pd.Timestamp(block_dates[-1]).date().isoformat(),
                **asdict(metrics),
            }
        )
    spreads = np.asarray([block["decile_spread"] for block in blocks], dtype=float)
    correlations = np.asarray([block["spearman"] for block in blocks], dtype=float)
    return {
        "block_sessions": block_sessions,
        "blocks": blocks,
        "positive_spread_ratio": float(np.mean(spreads > 0)) if len(spreads) else 0.0,
        "median_decile_spread": float(np.median(spreads)) if len(spreads) else 0.0,
        "positive_spearman_ratio": float(np.mean(correlations > 0)) if len(correlations) else 0.0,
        "median_spearman": float(np.median(correlations)) if len(correlations) else 0.0,
    }


def _selection_objective(metrics: SplitMetrics) -> float:
    return metrics.spearman * 8.0 + metrics.decile_spread * 0.25 + metrics.decile_trend * 0.15


def _training_target(data: pd.DataFrame, mask: pd.Series, config: dict[str, Any]) -> pd.Series:
    target = data.loc[mask, config["column"]]
    if config["training_target"] == "daily_return_rank":
        return target.groupby(data.loc[mask, "date"]).rank(method="average", pct=True) - 0.5
    return target.clip(*config["clip"])


def _equal_date_weights(dates: pd.Series) -> np.ndarray:
    counts = dates.groupby(dates).transform("size").astype(float)
    weights = 1.0 / counts
    return (weights / weights.mean()).to_numpy()


def train_mode(
    data: pd.DataFrame,
    features: list[str],
    masks: dict[str, pd.Series],
    mode: str,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = TARGETS[mode]
    target_column = config["column"]
    valid_target = data[target_column].notna()
    train_mask = masks["train"] & valid_target
    valid_mask = masks["valid"] & valid_target
    test_mask = masks["test"] & valid_target

    imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    x_train = imputer.fit_transform(data.loc[train_mask, features])
    x_valid = imputer.transform(data.loc[valid_mask, features])
    y_train = _training_target(data, train_mask, config)
    train_weights = _equal_date_weights(data.loc[train_mask, "date"])

    candidates = []
    fitted: dict[str, XGBRegressor] = {}
    for name, params in _model_candidates(args.n_jobs):
        model = XGBRegressor(**params)
        model.fit(x_train, y_train, sample_weight=train_weights, verbose=False)
        valid_prediction = model.predict(x_valid)
        valid_metrics, _ = evaluate_predictions(
            data.loc[valid_mask, "date"],
            data.loc[valid_mask, target_column],
            valid_prediction,
            config["hit_threshold"],
        )
        fitted[name] = model
        candidates.append({"name": name, "params": params, "valid": asdict(valid_metrics)})
    best = max(candidates, key=lambda item: _selection_objective(SplitMetrics(**item["valid"])))

    # Refit the selected structure on train + validation; the test remains untouched.
    fit_mask = (masks["train"] | masks["valid"]) & valid_target
    final_imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    x_fit = final_imputer.fit_transform(data.loc[fit_mask, features])
    x_test = final_imputer.transform(data.loc[test_mask, features])
    y_fit = _training_target(data, fit_mask, config)
    fit_weights = _equal_date_weights(data.loc[fit_mask, "date"])
    final_model = XGBRegressor(**best["params"])
    final_model.fit(x_fit, y_fit, sample_weight=fit_weights, verbose=False)
    fit_prediction = final_model.predict(x_fit)
    test_prediction = final_model.predict(x_test)
    test_metrics, test_deciles = evaluate_predictions(
        data.loc[test_mask, "date"],
        data.loc[test_mask, target_column],
        test_prediction,
        config["hit_threshold"],
    )
    test_stability = evaluate_stability(
        data.loc[test_mask, "date"],
        data.loc[test_mask, target_column],
        test_prediction,
        config["hit_threshold"],
    )
    passing = (
        test_metrics.spearman >= config["min_test_spearman"]
        and test_metrics.decile_spread >= config["min_test_decile_spread"]
        and test_metrics.decile_trend > 0
    )
    artifact = {
        "schema_version": "selector_buy_hold_return_model_v1",
        "mode": mode,
        "target": target_column,
        "training_target": config["training_target"],
        "features": features,
        "imputer": final_imputer,
        "model": final_model,
        "score_reference": np.sort(fit_prediction.astype(float)),
        "normalization_width": config["normalization_width"],
        "score_definition": "fixed robust historical transform of predicted target return",
        "trained_through": args.valid_end,
    }
    report = {
        "mode": mode,
        "target": target_column,
        "selected_candidate": best["name"],
        "candidates": candidates,
        "test": asdict(test_metrics),
        "test_stability": test_stability,
        "test_deciles": test_deciles.to_dict(orient="records"),
        "gate": {
            "passing": passing,
            "min_test_spearman": config["min_test_spearman"],
            "min_test_decile_spread": config["min_test_decile_spread"],
            "requires_positive_decile_trend": True,
        },
    }
    return artifact, report


def train_multitask_hold(
    data: pd.DataFrame,
    features: list[str],
    masks: dict[str, pd.Series],
    reports: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Blend hold-return rank with the more stable max-high representation."""

    train_mask = masks["train"] & data["future_return_t5_pct"].notna() & data["future_max_high_t5_pct"].notna()
    valid_mask = masks["valid"] & data["future_return_t5_pct"].notna()
    test_mask = masks["test"] & data["future_return_t5_pct"].notna()
    imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    x_train = imputer.fit_transform(data.loc[train_mask, features])
    component_models: dict[str, XGBRegressor] = {}
    references: dict[str, np.ndarray] = {}
    split_scores: dict[str, dict[str, np.ndarray]] = {"valid": {}, "test": {}}
    for mode in ("buy", "hold"):
        selected_name = reports[mode]["selected_candidate"]
        params = next(item["params"] for item in reports[mode]["candidates"] if item["name"] == selected_name)
        config = TARGETS[mode]
        target = _training_target(data, train_mask, config)
        model = XGBRegressor(**params)
        model.fit(
            x_train,
            target,
            sample_weight=_equal_date_weights(data.loc[train_mask, "date"]),
            verbose=False,
        )
        reference = np.sort(model.predict(x_train).astype(float))
        component_models[mode] = model
        references[mode] = reference
        for split_name, split_mask in (("valid", valid_mask), ("test", test_mask)):
            prediction = model.predict(imputer.transform(data.loc[split_mask, features]))
            split_scores[split_name][mode] = historical_percentile_scores(
                prediction,
                reference,
                config["normalization_width"],
            )

    blend_candidates = []
    for buy_weight in (0.0, 0.25, 0.5, 0.75, 1.0):
        prediction = (
            buy_weight * split_scores["valid"]["buy"]
            + (1.0 - buy_weight) * split_scores["valid"]["hold"]
        )
        metrics, _ = evaluate_predictions(
            data.loc[valid_mask, "date"],
            data.loc[valid_mask, "future_return_t5_pct"],
            prediction,
            TARGETS["hold"]["hit_threshold"],
        )
        blend_candidates.append({"buy_weight": buy_weight, "valid": asdict(metrics)})
    selected = max(
        blend_candidates,
        key=lambda item: _selection_objective(SplitMetrics(**item["valid"])),
    )
    buy_weight = float(selected["buy_weight"])

    # Once validation has selected the blend, refit both components on train
    # plus validation.  The test window remains untouched and is evaluated
    # exactly once, matching the single-target training path above.
    fit_mask = (masks["train"] | masks["valid"]) & data["future_return_t5_pct"].notna() & data[
        "future_max_high_t5_pct"
    ].notna()
    final_imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    x_fit = final_imputer.fit_transform(data.loc[fit_mask, features])
    x_test = final_imputer.transform(data.loc[test_mask, features])
    final_models: dict[str, XGBRegressor] = {}
    final_references: dict[str, np.ndarray] = {}
    final_test_scores: dict[str, np.ndarray] = {}
    for mode in ("buy", "hold"):
        selected_name = reports[mode]["selected_candidate"]
        params = next(item["params"] for item in reports[mode]["candidates"] if item["name"] == selected_name)
        config = TARGETS[mode]
        model = XGBRegressor(**params)
        model.fit(
            x_fit,
            _training_target(data, fit_mask, config),
            sample_weight=_equal_date_weights(data.loc[fit_mask, "date"]),
            verbose=False,
        )
        reference = np.sort(model.predict(x_fit).astype(float))
        final_models[mode] = model
        final_references[mode] = reference
        final_test_scores[mode] = historical_percentile_scores(
            model.predict(x_test), reference, config["normalization_width"]
        )
    test_prediction = (
        buy_weight * final_test_scores["buy"]
        + (1.0 - buy_weight) * final_test_scores["hold"]
    )
    test_metrics, test_deciles = evaluate_predictions(
        data.loc[test_mask, "date"],
        data.loc[test_mask, "future_return_t5_pct"],
        test_prediction,
        TARGETS["hold"]["hit_threshold"],
    )
    test_stability = evaluate_stability(
        data.loc[test_mask, "date"],
        data.loc[test_mask, "future_return_t5_pct"],
        test_prediction,
        TARGETS["hold"]["hit_threshold"],
    )
    config = TARGETS["hold"]
    passing = (
        test_metrics.spearman >= config["min_test_spearman"]
        and test_metrics.decile_spread >= config["min_test_decile_spread"]
        and test_metrics.decile_trend > 0
    )
    artifact = {
        "schema_version": "selector_buy_hold_return_model_v1",
        "mode": "hold",
        "target": config["column"],
        "training_target": "multitask_fixed_percentile_blend",
        "features": features,
        "imputer": final_imputer,
        "models": final_models,
        "score_references": final_references,
        "normalization_widths": {mode: TARGETS[mode]["normalization_width"] for mode in ("buy", "hold")},
        "buy_weight": buy_weight,
        "score_definition": "fixed blend of robust historical scores for hold return rank and max-high return",
        "trained_through": args.valid_end,
    }
    report = {
        "mode": "hold",
        "target": config["column"],
        "selected_candidate": "multitask_blend",
        "blend_candidates": blend_candidates,
        "selected_buy_weight": buy_weight,
        "test": asdict(test_metrics),
        "test_stability": test_stability,
        "test_deciles": test_deciles.to_dict(orient="records"),
        "gate": {
            "passing": passing,
            "min_test_spearman": config["min_test_spearman"],
            "min_test_decile_spread": config["min_test_decile_spread"],
            "requires_positive_decile_trend": True,
        },
    }
    return artifact, report


def write_report(payload: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "selector_buy_hold_model_report.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    return path


def predict_artifact(artifact: dict[str, Any], data: pd.DataFrame, mask: pd.Series) -> np.ndarray:
    features = list(artifact["features"])
    feature_frame = data.reindex(columns=features)
    transformed = artifact["imputer"].transform(feature_frame.loc[mask])
    if "models" in artifact:
        component_scores = {}
        for mode, model in artifact["models"].items():
            component_scores[mode] = historical_percentile_scores(
                model.predict(transformed),
                np.asarray(artifact["score_references"][mode]),
                float(artifact.get("normalization_widths", {}).get(mode, 2.0)),
            )
        buy_weight = float(artifact.get("buy_weight", 0.0))
        return buy_weight * component_scores["buy"] + (1.0 - buy_weight) * component_scores["hold"]
    prediction = artifact["model"].predict(transformed)
    return historical_percentile_scores(
        prediction,
        np.asarray(artifact["score_reference"]),
        float(artifact.get("normalization_width", 2.0)),
    )


def evaluate_incumbents(
    production_dir: Path,
    data: pd.DataFrame,
    masks: dict[str, pd.Series],
) -> tuple[dict[str, Any], list[str]]:
    comparisons: dict[str, Any] = {}
    errors: list[str] = []
    for mode, config in TARGETS.items():
        path = production_dir / f"{mode}.joblib"
        if not path.exists():
            continue
        try:
            artifact = joblib.load(path)
            mask = masks["test"] & data[config["column"]].notna()
            prediction = predict_artifact(artifact, data, mask)
            metrics, _ = evaluate_predictions(
                data.loc[mask, "date"], data.loc[mask, config["column"]], prediction, config["hit_threshold"]
            )
            comparisons[mode] = {
                "artifact": str(path),
                "test": asdict(metrics),
                "objective": _selection_objective(metrics),
            }
        except Exception as exc:
            errors.append(f"{mode}: {type(exc).__name__}: {exc}")
    return comparisons, errors


def main() -> None:
    args = parse_args()
    production_dir = PROJECT_ROOT / "models/production/selector_buy_hold"
    production_manifest = production_dir / "manifest.json"
    data, features, metadata = load_dataset(args.history, args.factor_data)
    masks = _split_masks(data, args)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    reports: dict[str, Any] = {}
    artifacts: dict[str, dict[str, Any]] = {}
    for mode in TARGETS:
        artifact, report = train_mode(data, features, masks, mode, args)
        artifacts[mode] = artifact
        reports[mode] = report
        joblib.dump(artifact, args.model_dir / f"{mode}.joblib")

    hold_artifact, hold_report = train_multitask_hold(data, features, masks, reports, args)
    artifacts["hold"] = hold_artifact
    reports["hold"] = hold_report
    joblib.dump(hold_artifact, args.model_dir / "hold.joblib")

    all_passing = all(report["gate"]["passing"] for report in reports.values())
    incumbent_comparison, incumbent_errors = evaluate_incumbents(production_dir, data, masks)
    no_regression = not incumbent_errors
    for mode, report in reports.items():
        previous = incumbent_comparison.get(mode)
        if previous is None:
            continue
        challenger_objective = _selection_objective(SplitMetrics(**report["test"]))
        report["same_window_incumbent"] = previous
        report["same_window_challenger_objective"] = challenger_objective
        report["same_window_improvement"] = challenger_objective - float(previous["objective"])
        if challenger_objective + 1e-9 < float(previous["objective"]):
            no_regression = False
    production_paths: dict[str, str] = {}
    if args.promote and all_passing and no_regression:
        production_dir.mkdir(parents=True, exist_ok=True)
        for mode, artifact in artifacts.items():
            path = production_dir / f"{mode}.joblib"
            joblib.dump(artifact, path)
            production_paths[mode] = str(path)
        manifest = {
            "schema_version": "selector_buy_hold_production_manifest_v1",
            "promoted_at": datetime.now().isoformat(timespec="seconds"),
            "splits": {
                "train_start": args.train_start,
                "train_end": args.train_end,
                "valid_end": args.valid_end,
                "test_end": args.test_end,
            },
            "models": {
                mode: {
                    "target": report["target"],
                    "selected_candidate": report["selected_candidate"],
                    "test": report["test"],
                }
                for mode, report in reports.items()
            },
        }
        production_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    payload = {
        "status": "success",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "splits": {
            "train_start": args.train_start,
            "train_end": args.train_end,
            "valid_end": args.valid_end,
            "test_end": args.test_end,
        },
        "dataset": metadata,
        "feature_count": len(features),
        "models": reports,
        "all_passing": all_passing,
        "champion_no_regression": no_regression,
        "incumbent_comparison_errors": incumbent_errors,
        "promoted": bool(production_paths),
        "promotion_block_reason": (
            None
            if production_paths or not args.promote
            else "quality_gate_failed" if not all_passing else "worse_than_production_champion"
        ),
        "production_paths": production_paths,
    }
    report_path = write_report(payload, args.output_dir)
    print(json.dumps({**payload, "report": str(report_path)}, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
