#!/usr/bin/env python
"""Train point-in-time weekly models for good-stock entry prices.

The first experiment deliberately performs no univariate, correlation, or
importance-based feature selection. Factors enter only when the training fold
has enough non-null observations, sufficient coverage, and more than one
value. Model feedback and case analysis drive the next iteration.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, mean_absolute_error, roc_auc_score
from xgboost import XGBClassifier, XGBRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "research"))

from backtest_long_dividend_quality import load_analyst_forecast_asof
from backtest_long_good_price_weekly_absolute import prepare_weekly_good_stocks
from backtest_long_good_stock_price import attach_forward_returns
from quant.features.long_weekly_factors import (
    add_long_entry_interactions,
    build_long_weekly_factor_frame,
    long_model_candidate_columns,
)
from quant.features.project_factor_layer import (
    PROJECT_FACTOR_SCHEMA_VERSION,
    admit_factors_by_sample,
)


REPORT_DIR = PROJECT_ROOT / "reports/long_entry_model_v1"
MODEL_DIR = PROJECT_ROOT / "models/research/long_entry_model_v1"
DATASET_CACHE = PROJECT_ROOT / "data/features/long_entry/weekly_training_v1.parquet"
DATASET_MANIFEST = PROJECT_ROOT / "data/features/long_entry/weekly_training_v1.manifest.json"
DATASET_CONTRACT_VERSION = "weekly-v2-industry-context"
HORIZONS = {"13w": 63, "26w": 126, "52w": 252}


@dataclass(frozen=True)
class Fold:
    name: str
    validation_start: str
    validation_end: str
    reused_test: bool = False


FOLDS = (
    Fold("wf_2020", "2020-01-01", "2020-12-31"),
    Fold("wf_2021", "2021-01-01", "2021-12-31"),
    Fold("wf_2022", "2022-01-01", "2022-12-31"),
    Fold("wf_2023", "2023-01-01", "2023-12-31"),
    Fold("reused_2024_plus", "2024-01-01", "2026-12-31", True),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train long-entry weekly multi-head models")
    parser.add_argument("--start", default="20130101")
    parser.add_argument("--end", default=None)
    parser.add_argument("--force-dataset", action="store_true")
    parser.add_argument("--n-estimators", type=int, default=220)
    parser.add_argument("--n-jobs", type=int, default=min(8, os.cpu_count() or 4))
    parser.add_argument("--minimum-non-null", type=int, default=1000)
    parser.add_argument("--minimum-coverage", type=float, default=0.05)
    return parser.parse_args()


def _atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp.json")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp.parquet")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp.csv")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_joblib(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp.joblib")
    joblib.dump(payload, temporary)
    os.replace(temporary, path)


def add_market_context(frame: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    market = benchmark[["date", "benchmark_equity"]].copy()
    market["date"] = pd.to_datetime(market["date"])
    market = market.sort_values("date").drop_duplicates("date", keep="last")
    equity = pd.to_numeric(market["benchmark_equity"], errors="coerce")
    market["market_return_13w"] = equity.pct_change(63)
    market["market_return_26w"] = equity.pct_change(126)
    market["market_return_52w"] = equity.pct_change(252)
    market["market_drawdown_52w"] = equity / equity.rolling(252, min_periods=63).max() - 1.0
    market["market_volatility_13w"] = equity.pct_change().rolling(63, min_periods=20).std() * np.sqrt(252)
    left = frame.sort_values("date")
    merged = pd.merge_asof(left, market, on="date", direction="backward")
    merged["return_120d_minus_market"] = (
        pd.to_numeric(merged["return_120d"], errors="coerce")
        - pd.to_numeric(merged["market_return_26w"], errors="coerce")
    )
    if "industry_return_120d_mean" in merged.columns:
        merged["industry_return_120d_minus_market"] = (
            pd.to_numeric(merged["industry_return_120d_mean"], errors="coerce")
            - pd.to_numeric(merged["market_return_26w"], errors="coerce")
        )
    return merged.sort_values(["date", "ts_code"]).reset_index(drop=True)


def add_labels(frame: pd.DataFrame, daily: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    labelled = attach_forward_returns(frame, daily, benchmark, horizons=HORIZONS)
    valid_52w = pd.to_numeric(labelled["excess_return_52w"], errors="coerce").notna()
    labelled["label_value_rank_52w"] = np.nan
    labelled.loc[valid_52w, "label_value_rank_52w"] = (
        labelled.loc[valid_52w]
        .groupby("date")["excess_return_52w"]
        .rank(method="average", pct=True)
    )
    mae_13w = pd.to_numeric(labelled["mae_13w"], errors="coerce")
    mae_26w = pd.to_numeric(labelled["mae_26w"], errors="coerce")
    labelled["label_wait_risk_13w"] = np.where(mae_13w.notna(), mae_13w.le(-0.10).astype(float), np.nan)
    labelled["label_drawdown_risk_26w"] = np.where(mae_26w.notna(), mae_26w.le(-0.20).astype(float), np.nan)
    return labelled


def upgrade_cached_dataset_contract(cached: pd.DataFrame) -> pd.DataFrame:
    """Add point-in-time derived columns without rebuilding source/as-of joins."""

    upgraded = add_long_entry_interactions(cached)
    upgraded["return_120d_minus_market"] = (
        pd.to_numeric(upgraded["return_120d"], errors="coerce")
        - pd.to_numeric(upgraded["market_return_26w"], errors="coerce")
    )
    upgraded["industry_return_120d_minus_market"] = (
        pd.to_numeric(upgraded["industry_return_120d_mean"], errors="coerce")
        - pd.to_numeric(upgraded["market_return_26w"], errors="coerce")
    )
    return upgraded.sort_values(["date", "ts_code"]).reset_index(drop=True)


def build_dataset(start: str, end: str | None, force: bool) -> tuple[pd.DataFrame, dict]:
    if DATASET_CACHE.exists() and DATASET_MANIFEST.exists() and not force:
        manifest = json.loads(DATASET_MANIFEST.read_text(encoding="utf-8"))
        cached_schema = str(manifest.get("factor_schema_version") or "")
        schema_is_current = cached_schema == PROJECT_FACTOR_SCHEMA_VERSION
        weekly_only_upgrade = cached_schema in {
            "project-v2",
            "project-v3-causal-alpha",
        }
        if schema_is_current or weekly_only_upgrade:
            cached = pd.read_parquet(DATASET_CACHE)
            cached["date"] = pd.to_datetime(cached["date"])
            if weekly_only_upgrade and not any(column.startswith("alpha") for column in cached.columns):
                manifest["factor_schema_version"] = PROJECT_FACTOR_SCHEMA_VERSION
                manifest["schema_upgraded_at"] = datetime.now().isoformat(timespec="seconds")
                _atomic_json(manifest, DATASET_MANIFEST)
            elif weekly_only_upgrade:
                cached = pd.DataFrame()
            if cached.empty:
                pass
            elif manifest.get("dataset_contract_version") != DATASET_CONTRACT_VERSION:
                cached = upgrade_cached_dataset_contract(cached)
                _atomic_parquet(cached, DATASET_CACHE)
                manifest["dataset_contract_version"] = DATASET_CONTRACT_VERSION
                manifest["contract_upgraded_at"] = datetime.now().isoformat(timespec="seconds")
                manifest["columns"] = int(len(cached.columns))
                _atomic_json(manifest, DATASET_MANIFEST)
            if not cached.empty:
                return cached, {**manifest, "cache_hit": True}

    started = perf_counter()
    weekly, daily, benchmark, source_coverage = prepare_weekly_good_stocks(start, end)
    factors = build_long_weekly_factor_frame(weekly, history_windows=((2, 5), (2, 7)))
    factors = load_analyst_forecast_asof(factors)
    factors = add_long_entry_interactions(factors)
    factors = add_market_context(factors, benchmark)
    dataset = add_labels(factors, daily, benchmark)
    dataset = dataset.sort_values(["date", "ts_code"]).reset_index(drop=True)
    _atomic_parquet(dataset, DATASET_CACHE)
    manifest = {
        "status": "success",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "factor_schema_version": PROJECT_FACTOR_SCHEMA_VERSION,
        "dataset_contract_version": DATASET_CONTRACT_VERSION,
        "sampling": "weekly_last_trading_day",
        "financial_point_in_time": "ann_date <= signal_date",
        "analyst_point_in_time": "report_date <= signal_date; is_predict=true for forward aggregates",
        "history_windows": [{"minimum_years": 2, "maximum_years": 5}, {"minimum_years": 2, "maximum_years": 7}],
        "rows": int(len(dataset)),
        "symbols": int(dataset["ts_code"].nunique()),
        "date_min": dataset["date"].min().date().isoformat(),
        "date_max": dataset["date"].max().date().isoformat(),
        "source_coverage": source_coverage,
        "elapsed_seconds": perf_counter() - started,
        "path": str(DATASET_CACHE),
        "cache_hit": False,
    }
    _atomic_json(manifest, DATASET_MANIFEST)
    return dataset, manifest


def date_equal_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("date")["ts_code"].transform("count").astype(float)
    weights = 1.0 / counts.clip(lower=1.0)
    return (weights / weights.mean()).to_numpy(dtype=float)


def make_regressor(n_estimators: int, n_jobs: int) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=n_estimators,
        max_depth=4,
        learning_rate=0.04,
        min_child_weight=10,
        subsample=0.80,
        colsample_bytree=0.80,
        reg_alpha=0.10,
        reg_lambda=2.0,
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=n_jobs,
        random_state=42,
    )


def make_classifier(n_estimators: int, n_jobs: int) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=4,
        learning_rate=0.04,
        min_child_weight=10,
        subsample=0.80,
        colsample_bytree=0.80,
        reg_alpha=0.10,
        reg_lambda=2.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=n_jobs,
        random_state=42,
    )


def safe_auc(actual: pd.Series, prediction: np.ndarray) -> float:
    return float(roc_auc_score(actual, prediction)) if actual.nunique() > 1 else np.nan


def fit_fold(
    dataset: pd.DataFrame,
    candidates: list[str],
    fold: Fold,
    *,
    n_estimators: int,
    n_jobs: int,
    minimum_non_null: int,
    minimum_coverage: float,
) -> tuple[pd.DataFrame, list[dict], pd.DataFrame, pd.DataFrame, dict[str, dict]]:
    validation_start = pd.Timestamp(fold.validation_start)
    validation_end = pd.Timestamp(fold.validation_end)
    purge_cutoff = validation_start - pd.Timedelta(days=366)
    train = dataset[dataset["date"] < purge_cutoff].copy()
    validation = dataset[dataset["date"].between(validation_start, validation_end)].copy()
    if len(train) < 5000 or len(validation) < 500:
        raise RuntimeError(f"{fold.name}: insufficient train/validation rows {len(train)}/{len(validation)}")

    fold_minimum = max(minimum_non_null, int(len(train) * minimum_coverage))
    features, admission = admit_factors_by_sample(
        train,
        candidates,
        minimum_non_null_rows=fold_minimum,
        minimum_coverage=minimum_coverage,
    )
    if not features:
        raise RuntimeError(f"{fold.name}: no factors passed the sample gate")
    admission.insert(0, "fold", fold.name)
    admission["train_rows"] = len(train)

    predictions = validation[
        [
            "date",
            "ts_code",
            "name",
            "industry",
            "close",
            "pe_ttm",
            "pb",
            "roe",
            "pr_pe",
            "pr_pb",
            "pr",
            "return_52w",
            "excess_return_52w",
            "mae_13w",
            "mae_26w",
            "mae_52w",
            "label_value_rank_52w",
            "label_wait_risk_13w",
            "label_drawdown_risk_26w",
        ]
    ].copy()
    metrics: list[dict] = []
    importances: list[pd.DataFrame] = []
    artifacts: dict[str, dict] = {}
    heads = (
        ("value_rank_52w", "label_value_rank_52w", "regression"),
        ("wait_risk_13w", "label_wait_risk_13w", "classification"),
        ("drawdown_risk_26w", "label_drawdown_risk_26w", "classification"),
    )
    for head, target, kind in heads:
        train_head = train.dropna(subset=[target]).copy()
        validation_head = validation.dropna(subset=[target]).copy()
        train_matrix = train_head[features].apply(pd.to_numeric, errors="coerce")
        validation_matrix = validation_head[features].apply(pd.to_numeric, errors="coerce")
        model = make_regressor(n_estimators, n_jobs) if kind == "regression" else make_classifier(n_estimators, n_jobs)
        model.fit(
            train_matrix,
            train_head[target],
            sample_weight=date_equal_weights(train_head),
        )
        prediction = (
            model.predict(validation_matrix)
            if kind == "regression"
            else model.predict_proba(validation_matrix)[:, 1]
        )
        prediction_column = f"pred_{head}"
        predictions[prediction_column] = np.nan
        predictions.loc[validation_head.index, prediction_column] = prediction
        if kind == "regression":
            metric = {
                "spearman": float(pd.Series(prediction).corr(validation_head[target].reset_index(drop=True), method="spearman")),
                "mae": float(mean_absolute_error(validation_head[target], prediction)),
            }
        else:
            metric = {
                "roc_auc": safe_auc(validation_head[target], prediction),
                "average_precision": float(average_precision_score(validation_head[target], prediction)),
                "brier": float(brier_score_loss(validation_head[target], prediction)),
                "positive_rate": float(validation_head[target].mean()),
            }
        metrics.append(
            {
                "fold": fold.name,
                "reused_test": fold.reused_test,
                "head": head,
                "train_rows": int(len(train_head)),
                "validation_rows": int(len(validation_head)),
                "factor_count": len(features),
                "purge_cutoff": purge_cutoff.date().isoformat(),
                **metric,
            }
        )
        importances.append(
            pd.DataFrame(
                {
                    "fold": fold.name,
                    "head": head,
                    "factor": features,
                    "importance": model.feature_importances_,
                }
            )
        )
        artifacts[head] = {
            "model": model,
            "features": features,
            "target": target,
            "kind": kind,
            "fold": asdict(fold),
        }

    predictions["fold"] = fold.name
    predictions["reused_test"] = fold.reused_test
    predictions["pred_value_rank_pct"] = predictions.groupby("date")["pred_value_rank_52w"].rank(pct=True)
    predictions["entry_score"] = 100.0 * (
        predictions["pred_value_rank_pct"] * 0.60
        + (1.0 - predictions["pred_wait_risk_13w"]) * 0.25
        + (1.0 - predictions["pred_drawdown_risk_26w"]) * 0.15
    )
    predictions["entry_score_pct"] = predictions.groupby("date")["entry_score"].rank(pct=True)
    complete = predictions.dropna(subset=["return_52w", "entry_score_pct"])
    selected = complete[complete["entry_score_pct"] >= 0.80]

    def weekly_mean(frame: pd.DataFrame, column: str) -> float:
        return float(frame.groupby("date")[column].mean().mean()) if not frame.empty else np.nan

    metrics.append(
        {
            "fold": fold.name,
            "reused_test": fold.reused_test,
            "head": "composite_top20",
            "train_rows": int(len(train)),
            "validation_rows": int(len(selected)),
            "factor_count": len(features),
            "purge_cutoff": purge_cutoff.date().isoformat(),
            "mean_return_52w": weekly_mean(selected, "return_52w"),
            "mean_excess_52w": weekly_mean(selected, "excess_return_52w"),
            "mean_mae_26w": weekly_mean(selected, "mae_26w"),
            "baseline_return_52w": weekly_mean(complete, "return_52w"),
            "baseline_excess_52w": weekly_mean(complete, "excess_return_52w"),
            "baseline_mae_26w": weekly_mean(complete, "mae_26w"),
            "return_delta_vs_good_stock": weekly_mean(selected, "return_52w") - weekly_mean(complete, "return_52w"),
            "excess_delta_vs_good_stock": weekly_mean(selected, "excess_return_52w") - weekly_mean(complete, "excess_return_52w"),
        }
    )
    importance = pd.concat(importances, ignore_index=True)
    return predictions, metrics, admission, importance, artifacts


def build_cases(predictions: pd.DataFrame, dataset: pd.DataFrame, top_factors: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_rows = dataset[["date", "ts_code", *[factor for factor in top_factors if factor in dataset.columns]]]
    merged = predictions.merge(feature_rows, on=["date", "ts_code"], how="left")
    score_percentile = (
        merged["calibrated_score_pct"]
        if "calibrated_score_pct" in merged.columns
        else merged["entry_score_pct"]
    )
    false_positive = merged[
        (score_percentile >= 0.80)
        & ((merged["excess_return_52w"] <= 0) | (merged["mae_26w"] <= -0.20))
    ].sort_values(
        ["calibrated_score" if "calibrated_score" in merged.columns else "entry_score", "date"],
        ascending=[False, False],
    )
    future_rank = merged.groupby("date")["excess_return_52w"].rank(pct=True)
    false_negative = merged[
        (score_percentile <= 0.30) & (future_rank >= 0.80)
    ].sort_values(["excess_return_52w", "date"], ascending=[False, False])
    return false_positive.head(300), false_negative.head(300)


def calibrate_composite(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float | str]]:
    """Choose score weights on 2020-2023 only, then diagnose 2024+ once."""

    complete = predictions.dropna(
        subset=[
            "return_52w",
            "pred_value_rank_pct",
            "pred_wait_risk_13w",
            "pred_drawdown_risk_26w",
        ]
    ).copy()
    grid_rows: list[dict] = []
    weight_values = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    for value_weight in weight_values:
        for wait_weight in weight_values:
            drawdown_weight = 1.0 - value_weight - wait_weight
            if drawdown_weight < -1e-9:
                continue
            score = (
                complete["pred_value_rank_pct"] * value_weight
                + (1.0 - complete["pred_wait_risk_13w"]) * wait_weight
                + (1.0 - complete["pred_drawdown_risk_26w"]) * drawdown_weight
            )
            for neutralization in ("market", "industry"):
                if neutralization == "market":
                    score_pct = score.groupby(complete["date"]).rank(pct=True)
                else:
                    score_pct = score.groupby(
                        [complete["date"], complete["industry"].fillna("未知")]
                    ).rank(pct=True)
                for threshold in (0.70, 0.80, 0.90):
                    fold_deltas: list[float] = []
                    fold_mae: list[float] = []
                    for _, baseline in complete[~complete["reused_test"]].groupby("fold"):
                        selected = baseline[score_pct.loc[baseline.index] >= threshold]
                        baseline_return = baseline.groupby("date")["return_52w"].mean().mean()
                        selected_return = selected.groupby("date")["return_52w"].mean().mean()
                        baseline_mae = baseline.groupby("date")["mae_26w"].mean().mean()
                        selected_mae = selected.groupby("date")["mae_26w"].mean().mean()
                        fold_deltas.append(float(selected_return - baseline_return))
                        fold_mae.append(float(selected_mae - baseline_mae))
                    grid_rows.append(
                        {
                            "value_weight": value_weight,
                            "wait_weight": wait_weight,
                            "drawdown_weight": drawdown_weight,
                            "neutralization": neutralization,
                            "threshold": threshold,
                            "positive_validation_folds": int(sum(delta > 0 for delta in fold_deltas)),
                            "worst_validation_delta": min(fold_deltas),
                            "mean_validation_delta": float(np.mean(fold_deltas)),
                            "mean_validation_mae_improvement": float(np.mean(fold_mae)),
                        }
                    )
    grid = pd.DataFrame(grid_rows).sort_values(
        ["positive_validation_folds", "worst_validation_delta", "mean_validation_delta"],
        ascending=False,
    ).reset_index(drop=True)
    chosen = grid.iloc[0].to_dict()
    calibrated_score = (
        predictions["pred_value_rank_pct"] * float(chosen["value_weight"])
        + (1.0 - predictions["pred_wait_risk_13w"]) * float(chosen["wait_weight"])
        + (1.0 - predictions["pred_drawdown_risk_26w"]) * float(chosen["drawdown_weight"])
    )
    predictions = predictions.copy()
    predictions["calibrated_score"] = calibrated_score
    if chosen["neutralization"] == "market":
        predictions["calibrated_score_pct"] = calibrated_score.groupby(predictions["date"]).rank(pct=True)
    else:
        predictions["calibrated_score_pct"] = calibrated_score.groupby(
            [predictions["date"], predictions["industry"].fillna("未知")]
        ).rank(pct=True)

    metric_rows: list[dict] = []
    threshold = float(chosen["threshold"])
    for fold, baseline in predictions.dropna(subset=["return_52w", "calibrated_score_pct"]).groupby("fold"):
        selected = baseline[baseline["calibrated_score_pct"] >= threshold]

        def weekly_mean(frame: pd.DataFrame, column: str) -> float:
            return float(frame.groupby("date")[column].mean().mean()) if not frame.empty else np.nan

        metric_rows.append(
            {
                "fold": fold,
                "reused_test": bool(baseline["reused_test"].iloc[0]),
                "head": "calibrated_composite",
                "train_rows": np.nan,
                "validation_rows": int(len(selected)),
                "factor_count": np.nan,
                "purge_cutoff": None,
                "mean_return_52w": weekly_mean(selected, "return_52w"),
                "mean_excess_52w": weekly_mean(selected, "excess_return_52w"),
                "mean_mae_26w": weekly_mean(selected, "mae_26w"),
                "baseline_return_52w": weekly_mean(baseline, "return_52w"),
                "baseline_excess_52w": weekly_mean(baseline, "excess_return_52w"),
                "baseline_mae_26w": weekly_mean(baseline, "mae_26w"),
                "return_delta_vs_good_stock": weekly_mean(selected, "return_52w") - weekly_mean(baseline, "return_52w"),
                "excess_delta_vs_good_stock": weekly_mean(selected, "excess_return_52w") - weekly_mean(baseline, "excess_return_52w"),
            }
        )
    selected_config = {
        key: chosen[key]
        for key in ("value_weight", "wait_weight", "drawdown_weight", "neutralization", "threshold")
    }
    return predictions, grid, pd.DataFrame(metric_rows), selected_config


def build_report(
    manifest: dict,
    metrics: pd.DataFrame,
    admission: pd.DataFrame,
    importance: pd.DataFrame,
    false_positive: pd.DataFrame,
    false_negative: pd.DataFrame,
    calibrated_composite: dict[str, float | str],
) -> str:
    validation = metrics[(metrics["head"] == "calibrated_composite") & ~metrics["reused_test"]]
    reused = metrics[(metrics["head"] == "calibrated_composite") & metrics["reused_test"]]
    admitted = admission[admission["admitted"]]
    rejected = admission[~admission["admitted"]]
    top = (
        importance.groupby(["head", "factor"], as_index=False)["importance"].mean()
        .sort_values(["head", "importance"], ascending=[True, False])
        .groupby("head")
        .head(8)
    )
    lines = [
        "# 长线好价格周频模型：首轮实验",
        "",
        "## 结论",
        "",
        f"- 样本：{manifest['rows']:,} 条好股票周样本、{manifest['symbols']} 只股票，{manifest['date_min']} 至 {manifest['date_max']}。",
        "- 首轮不做单因子收益、相关性、重要性或人工偏好筛选；每个走步折只按非空样本数、覆盖率和是否退化准入。",
        "- 模型为三个头：52周相对价值排序、13周仍可能便宜10%以上的等待风险、26周回撤20%以上风险；综合分只用于好股票池内判断建仓价格。",
        (
            "- 仅在2020–2023走步验证上校准后的组合为："
            f"价值{float(calibrated_composite['value_weight']):.0%}、"
            f"避免等待风险{float(calibrated_composite['wait_weight']):.0%}、"
            f"避免深回撤{float(calibrated_composite['drawdown_weight']):.0%}，"
            f"取分位前{1-float(calibrated_composite['threshold']):.0%}；"
            f"中性化={calibrated_composite['neutralization']}。"
        ),
        "- 训练与验证之间清洗366天，特征使用周末时点数据，财务按公告日、研报按报告日向后合并。",
        "- 2024+ 已在此前规则探索中观察过，本报告只列复用诊断，不把它称为全新独立样本外。",
        "- 校准组合虽在四个走步验证折均跑赢好股基线，但2024+复用诊断明显落后；本轮结论为不升级生产推荐，只保留风险头与case队列继续研究。",
        "",
        "## 组合效果（每周等权）",
        "",
        "| 折 | 推荐52周收益 | 好股基线 | 收益增量 | 推荐超额 | 超额增量 | 推荐26周MAE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in pd.concat([validation, reused]).itertuples(index=False):
        suffix = "（复用诊断）" if row.reused_test else ""
        lines.append(
            f"| {row.fold}{suffix} | {row.mean_return_52w:.2%} | {row.baseline_return_52w:.2%} | "
            f"{row.return_delta_vs_good_stock:.2%} | {row.mean_excess_52w:.2%} | "
            f"{row.excess_delta_vs_good_stock:.2%} | {row.mean_mae_26w:.2%} |"
        )
    lines.extend(
        [
            "",
            "## 因子准入",
            "",
            f"- 各折累计准入记录 {len(admitted)} 条、拒绝 {len(rejected)} 条；拒绝仅因样本/覆盖/常量，不代表因子无效。",
            "- 下一轮将先从高分失败、低分成功案例判断缺失机制，再决定删除、补充或重构因子，避免用同一验证集反复追重要性。",
            "",
            "## 各模型平均重要性前列（诊断，不作为本轮筛选）",
            "",
        ]
    )
    for head, group in top.groupby("head", sort=False):
        rendered = "、".join(f"{row.factor}({row.importance:.3f})" for row in group.itertuples(index=False))
        lines.append(f"- {head}: {rendered}")
    lines.extend(
        [
            "",
            "## Case 分析队列",
            "",
            f"- 高分失败样本：{len(false_positive)} 条；低分但后续高超额样本：{len(false_negative)} 条（均截取最多300条供逐案复盘）。",
            "- 优先检查：行业估值范式错配、ROE周期高点、盈利预测离散度、估值低但基本面正转弱、价格尚未稳定五类机制。",
            "",
            "## 当前数据缺口",
            "",
            "- 两融余额只有不完整字段、质押历史为空、股东增减持与龙虎榜/资金流只有很短历史，不能进入正式长周期模型；若要使用，需要补齐可点时的多年历史。",
            "- 退市历史基础表仍可能不完整，结论存在幸存者偏差；周频重叠标签也不是独立样本，指标应按周组合解读。",
            "- 这是研究模型，不是生产推荐或投资建议；在新的封存时间段或前向纸面运行通过前不升级页面规则。",
        ]
    )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict:
    started = perf_counter()
    dataset, dataset_manifest = build_dataset(args.start, args.end, args.force_dataset)
    candidates = long_model_candidate_columns(dataset)
    prediction_frames: list[pd.DataFrame] = []
    metric_rows: list[dict] = []
    admission_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []
    last_artifacts: dict[str, dict] = {}
    for fold in FOLDS:
        print(f"training {fold.name}", flush=True)
        predictions, metrics, admission, importance, artifacts = fit_fold(
            dataset,
            candidates,
            fold,
            n_estimators=args.n_estimators,
            n_jobs=args.n_jobs,
            minimum_non_null=args.minimum_non_null,
            minimum_coverage=args.minimum_coverage,
        )
        prediction_frames.append(predictions)
        metric_rows.extend(metrics)
        admission_frames.append(admission)
        importance_frames.append(importance)
        last_artifacts = artifacts
    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    admission = pd.concat(admission_frames, ignore_index=True)
    importance = pd.concat(importance_frames, ignore_index=True)
    predictions, calibration_grid, calibrated_metrics, calibrated_composite = calibrate_composite(
        predictions
    )
    metrics = pd.concat([metrics, calibrated_metrics], ignore_index=True, sort=False)
    top_factors = (
        importance.groupby("factor")["importance"].mean().sort_values(ascending=False).head(20).index.tolist()
    )
    false_positive, false_negative = build_cases(predictions, dataset, top_factors)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    _atomic_parquet(predictions, REPORT_DIR / "walk_forward_predictions.parquet")
    _atomic_csv(metrics, REPORT_DIR / "walk_forward_metrics.csv")
    _atomic_csv(admission, REPORT_DIR / "factor_admission.csv")
    _atomic_csv(importance, REPORT_DIR / "feature_importance.csv")
    _atomic_csv(calibration_grid, REPORT_DIR / "composite_calibration_grid.csv")
    _atomic_csv(false_positive, REPORT_DIR / "false_positive_cases.csv")
    _atomic_csv(false_negative, REPORT_DIR / "false_negative_cases.csv")
    for head, artifact in last_artifacts.items():
        _atomic_joblib(
            {
                **artifact,
                "factor_schema_version": PROJECT_FACTOR_SCHEMA_VERSION,
                "research_only": True,
                "created_at": datetime.now().isoformat(timespec="seconds"),
            },
            MODEL_DIR / f"{head}.joblib",
        )
    experiment_manifest = {
        "status": "success",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "factor_schema_version": PROJECT_FACTOR_SCHEMA_VERSION,
        "dataset": dataset_manifest,
        "candidate_factors": len(candidates),
        "folds": [asdict(fold) for fold in FOLDS],
        "sample_gate": {
            "minimum_non_null": args.minimum_non_null,
            "minimum_coverage": args.minimum_coverage,
            "performance_prefilter": False,
        },
        "heads": {
            "value_rank_52w": "cross-sectional rank of 52-week CSI300 excess return",
            "wait_risk_13w": "future 13-week minimum <= -10%",
            "drawdown_risk_26w": "future 26-week minimum <= -20%",
        },
        "initial_composite_weights": {"value_rank": 0.60, "no_wait_risk": 0.25, "no_drawdown_risk": 0.15},
        "calibrated_composite": calibrated_composite,
        "elapsed_seconds": perf_counter() - started,
    }
    _atomic_json(experiment_manifest, REPORT_DIR / "experiment_manifest.json")
    report = build_report(
        dataset_manifest,
        metrics,
        admission,
        importance,
        false_positive,
        false_negative,
        calibrated_composite,
    )
    report_path = REPORT_DIR / "report.md"
    temporary = report_path.with_suffix(f".{os.getpid()}.tmp.md")
    temporary.write_text(report, encoding="utf-8")
    os.replace(temporary, report_path)
    return {**experiment_manifest, "report": str(report_path)}


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
