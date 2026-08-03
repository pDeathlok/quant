#!/usr/bin/env python
"""Case-driven V2 experiments for long-horizon good-price entry models.

This script preserves V1 artifacts and writes a separate research package.  It
compares labels and model families on 2020-2023 walk-forward folds, while the
already-observed 2024+ period remains diagnostic only.
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
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "research"))

from train_long_entry_price_models import Fold, date_equal_weights
from quant.features.long_weekly_factors import long_model_candidate_columns
from quant.features.project_factor_layer import admit_factors_by_sample
from quant.research.long_entry_v2 import (
    add_entry_labels,
    add_external_factor_transforms,
    classify_case_causes,
    cooldown_cases,
    month_end_week_mask,
    select_industry_capped,
    summarize_selection,
)


REPORT_DIR = PROJECT_ROOT / "reports/long_entry_model_v2"
MODEL_DIR = PROJECT_ROOT / "models/research/long_entry_model_v2"
BASE_DATASET = PROJECT_ROOT / "data/features/long_entry/weekly_training_v1.parquet"
EXTERNAL_DATASET = PROJECT_ROOT / "data/features/long_entry/weekly_external_v1.parquet"
V2_DATASET = PROJECT_ROOT / "data/features/long_entry/weekly_training_v2.parquet"
V2_MANIFEST = V2_DATASET.with_suffix(".manifest.json")
V2_DATASET_VERSION = "weekly-v5-multi-horizon-industry-utility-pit"


FOLDS = (
    Fold("wf_2020", "2020-01-01", "2020-12-31"),
    Fold("wf_2021", "2021-01-01", "2021-12-31"),
    Fold("wf_2022", "2022-01-01", "2022-12-31"),
    Fold("wf_2023", "2023-01-01", "2023-12-31"),
    Fold("reused_2024", "2024-01-01", "2024-12-31", True),
    Fold("reused_2025", "2025-01-01", "2025-12-31", True),
    Fold("reused_2026", "2026-01-01", "2026-12-31", True),
)


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    target: str
    use_external: bool
    purge_days: int


CANDIDATES = (
    Candidate("xgb_v1_endpoint_all", "xgb", "label_value_rank_52w", True, 366),
    Candidate("xgb_utility_market_all", "xgb", "label_entry_utility_market", True, 366),
    Candidate("xgb_utility_industry_core", "xgb", "label_entry_utility", False, 366),
    Candidate("xgb_utility_industry_all", "xgb", "label_entry_utility", True, 366),
    Candidate("xgb_utility_26w_core", "xgb", "label_entry_utility_26w", False, 190),
    Candidate("xgb_utility_26w_all", "xgb", "label_entry_utility_26w", True, 190),
    Candidate("xgb_utility_13w_core", "xgb", "label_entry_utility_13w", False, 100),
    Candidate("xgb_utility_13w_all", "xgb", "label_entry_utility_13w", True, 100),
    Candidate("extra_trees_utility_industry_all", "extra_trees", "label_entry_utility", True, 366),
    Candidate("hist_gbdt_utility_industry_all", "hist_gbdt", "label_entry_utility", True, 366),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-dataset", action="store_true")
    parser.add_argument("--n-estimators", type=int, default=220)
    parser.add_argument("--n-jobs", type=int, default=min(8, os.cpu_count() or 4))
    parser.add_argument("--minimum-non-null", type=int, default=1000)
    parser.add_argument("--minimum-coverage", type=float, default=0.05)
    parser.add_argument("--skip-slow-models", action="store_true")
    parser.add_argument("--recalibrate-only", action="store_true")
    return parser.parse_args()


def _atomic_frame(frame: pd.DataFrame, path: Path, *, parquet: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = "parquet" if parquet else "csv"
    temporary = path.with_suffix(f".{os.getpid()}.tmp.{suffix}")
    if parquet:
        frame.to_parquet(temporary, index=False)
    else:
        frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp.json")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def load_dataset(force: bool) -> tuple[pd.DataFrame, dict]:
    if V2_DATASET.exists() and V2_MANIFEST.exists() and not force:
        manifest = json.loads(V2_MANIFEST.read_text(encoding="utf-8"))
        if manifest.get("version") == V2_DATASET_VERSION:
            frame = pd.read_parquet(V2_DATASET)
            frame["date"] = pd.to_datetime(frame["date"])
            return frame, {**manifest, "cache_hit": True}
    base = pd.read_parquet(BASE_DATASET)
    external = pd.read_parquet(EXTERNAL_DATASET)
    base["date"] = pd.to_datetime(base["date"])
    external["date"] = pd.to_datetime(external["date"])
    if external.duplicated(["date", "ts_code"]).any():
        raise ValueError("external weekly factor cache contains duplicate signal keys")
    frame = base.merge(external, on=["date", "ts_code"], how="left", validate="one_to_one")
    frame = add_external_factor_transforms(frame)
    frame = add_entry_labels(frame)
    frame = frame.sort_values(["date", "ts_code"]).reset_index(drop=True)
    _atomic_frame(frame, V2_DATASET, parquet=True)
    external_columns = [column for column in external.columns if column not in {"date", "ts_code"}]
    external_columns += [
        "margin_balance_to_mv", "short_balance_to_mv", "log_margin_balance",
        "log_short_balance", "has_holder_trade_180d", "has_top_list_20d",
        "pledge_ratio_high", "large_flow_price_divergence_5d",
    ]
    manifest = {
        "status": "success",
        "version": V2_DATASET_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "rows": int(len(frame)),
        "symbols": int(frame["ts_code"].nunique()),
        "date_min": frame["date"].min().date().isoformat(),
        "date_max": frame["date"].max().date().isoformat(),
        "external_columns": sorted(set(column for column in external_columns if column in frame.columns)),
        "label_contract": {
            "label_entry_utility_market": "10% rank(excess13w)+25% rank(excess26w)+45% rank(excess52w)+20% rank(MAE26w)",
            "label_entry_utility": "55% market utility + 45% same-industry utility; groups below five fall back to market",
            "label_entry_utility_26w": "25% excess13w + 45% excess26w + 15% MAE13w + 15% MAE26w ranks; 55% market + 45% industry",
            "label_entry_utility_13w": "70% excess13w + 30% MAE13w ranks; 55% market + 45% industry",
            "label_downside_rank_26w": "cross-sectional rank of 26-week MAE; higher is safer",
        },
        "cache_hit": False,
    }
    _atomic_json(manifest, V2_MANIFEST)
    return frame, manifest


def make_model(candidate: Candidate, n_estimators: int, n_jobs: int):
    if candidate.family == "xgb":
        return XGBRegressor(
            n_estimators=n_estimators,
            max_depth=4,
            learning_rate=0.035,
            min_child_weight=12,
            subsample=0.80,
            colsample_bytree=0.80,
            reg_alpha=0.15,
            reg_lambda=2.5,
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=n_jobs,
            random_state=42,
        )
    if candidate.family == "extra_trees":
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=max(140, int(n_estimators * 0.75)),
                        max_depth=16,
                        min_samples_leaf=12,
                        max_features=0.75,
                        n_jobs=n_jobs,
                        random_state=42,
                    ),
                ),
            ]
        )
    if candidate.family == "hist_gbdt":
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_iter=max(120, int(n_estimators * 0.65)),
                        learning_rate=0.05,
                        max_leaf_nodes=31,
                        min_samples_leaf=20,
                        l2_regularization=2.0,
                        random_state=42,
                    ),
                ),
            ]
        )
    raise ValueError(candidate.family)


def fit_with_weights(model, matrix: pd.DataFrame, target: pd.Series, weights: np.ndarray) -> None:
    if isinstance(model, Pipeline):
        model.fit(matrix, target, model__sample_weight=weights)
    else:
        model.fit(matrix, target, sample_weight=weights)


def feature_importance(model, features: list[str]) -> pd.DataFrame:
    values = None
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif isinstance(model, Pipeline) and hasattr(model.named_steps["model"], "feature_importances_"):
        values = model.named_steps["model"].feature_importances_
    if values is None:
        return pd.DataFrame(columns=["factor", "importance"])
    return pd.DataFrame({"factor": features, "importance": values})


def fold_data(
    dataset: pd.DataFrame,
    fold: Fold,
    purge_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    validation_start = pd.Timestamp(fold.validation_start)
    purge_cutoff = validation_start - pd.Timedelta(days=purge_days)
    train = dataset[dataset["date"] < purge_cutoff].copy()
    validation = dataset[dataset["date"].between(fold.validation_start, fold.validation_end)].copy()
    return train, validation, purge_cutoff


def run_fold(
    dataset: pd.DataFrame,
    fold: Fold,
    candidates: tuple[Candidate, ...],
    *,
    n_estimators: int,
    n_jobs: int,
    minimum_non_null: int,
    minimum_coverage: float,
    external_columns: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, dict]]:
    _, validation, _ = fold_data(dataset, fold, 366)
    all_candidates = long_model_candidate_columns(dataset)
    prediction_columns = [
        "date", "ts_code", "name", "industry", "close", "good_stock_score",
        "or_yoy", "basic_eps_yoy", "roe", "debt_to_assets", "return_120d",
        "analyst_eps_revision_180d", "pr", "pr_pe", "pr_pb",
        "return_13w", "return_26w", "return_52w", "excess_return_13w",
        "excess_return_26w", "excess_return_52w", "mae_13w", "mae_26w",
        "mae_52w", "label_entry_utility_market", "label_entry_utility_industry",
        "label_entry_utility", "label_entry_utility_26w", "label_entry_utility_13w", "label_downside_rank_26w", "label_value_rank_52w",
        "market_return_26w", "market_drawdown_52w",
        "historical_value_score_5y", "close_to_ma120", "return_120d_cross_section_pct",
    ]
    base_predictions = validation[[column for column in prediction_columns if column in validation.columns]].copy()
    metric_rows: list[dict] = []
    admission_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []
    artifacts: dict[str, dict] = {}

    feature_sets: dict[tuple[bool, int], list[str]] = {}
    train_sets: dict[int, tuple[pd.DataFrame, pd.Timestamp]] = {}
    requested_sets = {(candidate.use_external, candidate.purge_days) for candidate in candidates}
    requested_sets.add((True, 190))
    for use_external, purge_days in requested_sets:
        train, _, purge_cutoff = fold_data(dataset, fold, purge_days)
        train_sets[purge_days] = (train, purge_cutoff)
        requested = [
            column for column in all_candidates
            if use_external or column not in external_columns
        ]
        fold_minimum = max(minimum_non_null, int(len(train) * minimum_coverage))
        admitted, admission = admit_factors_by_sample(
            train,
            requested,
            minimum_non_null_rows=fold_minimum,
            minimum_coverage=minimum_coverage,
        )
        admission.insert(0, "fold", fold.name)
        admission.insert(1, "feature_set", f"{'all' if use_external else 'core'}_{purge_days}d")
        admission_frames.append(admission)
        feature_sets[(use_external, purge_days)] = admitted

    risk_train_source, risk_purge_cutoff = train_sets[190]
    risk_features = feature_sets[(True, 190)]
    risk_train = risk_train_source.dropna(subset=["label_downside_rank_26w"])
    risk_model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=4,
        learning_rate=0.035,
        min_child_weight=12,
        subsample=0.80,
        colsample_bytree=0.80,
        reg_alpha=0.15,
        reg_lambda=2.5,
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=n_jobs,
        random_state=137,
    )
    fit_with_weights(
        risk_model,
        risk_train[risk_features].apply(pd.to_numeric, errors="coerce"),
        risk_train["label_downside_rank_26w"],
        date_equal_weights(risk_train),
    )
    risk_prediction = risk_model.predict(validation[risk_features].apply(pd.to_numeric, errors="coerce"))
    risk_valid = validation["label_downside_rank_26w"].notna()
    risk_spearman = (
        float(pd.Series(risk_prediction[risk_valid.to_numpy()]).corr(
            validation.loc[risk_valid, "label_downside_rank_26w"].reset_index(drop=True), method="spearman"
        ))
        if risk_valid.any()
        else np.nan
    )
    risk_mae = (
        float(mean_absolute_error(
            validation.loc[risk_valid, "label_downside_rank_26w"], risk_prediction[risk_valid.to_numpy()]
        ))
        if risk_valid.any()
        else np.nan
    )
    metric_rows.append(
        {
            "fold": fold.name,
            "reused_test": fold.reused_test,
            "candidate": "xgb_downside_rank_all",
            "target": "label_downside_rank_26w",
            "factor_count": len(risk_features),
            "train_rows": len(risk_train),
            "validation_rows": int(risk_valid.sum()),
            "spearman": risk_spearman,
            "mae": risk_mae,
            "purge_cutoff": risk_purge_cutoff.date().isoformat(),
        }
    )

    prediction_frames: list[pd.DataFrame] = []
    for candidate in candidates:
        print(f"  {candidate.name}", flush=True)
        train, purge_cutoff = train_sets[candidate.purge_days]
        features = feature_sets[(candidate.use_external, candidate.purge_days)]
        train_head = train.dropna(subset=[candidate.target])
        model = make_model(candidate, n_estimators, n_jobs)
        fit_with_weights(
            model,
            train_head[features].apply(pd.to_numeric, errors="coerce"),
            train_head[candidate.target],
            date_equal_weights(train_head),
        )
        prediction = model.predict(validation[features].apply(pd.to_numeric, errors="coerce"))
        valid = validation[candidate.target].notna()
        metrics = {
            "spearman": (
                float(pd.Series(prediction[valid.to_numpy()]).corr(
                    validation.loc[valid, candidate.target].reset_index(drop=True), method="spearman"
                ))
                if valid.any()
                else np.nan
            ),
            "mae": (
                float(mean_absolute_error(validation.loc[valid, candidate.target], prediction[valid.to_numpy()]))
                if valid.any()
                else np.nan
            ),
        }
        metric_rows.append(
            {
                "fold": fold.name,
                "reused_test": fold.reused_test,
                "candidate": candidate.name,
                "target": candidate.target,
                "factor_count": len(features),
                "train_rows": len(train_head),
                "validation_rows": int(valid.sum()),
                **metrics,
                "purge_cutoff": purge_cutoff.date().isoformat(),
            }
        )
        current = base_predictions.copy()
        current["fold"] = fold.name
        current["reused_test"] = fold.reused_test
        current["candidate"] = candidate.name
        current["pred_opportunity"] = prediction
        current["pred_downside_safe"] = risk_prediction
        prediction_frames.append(current)
        importance = feature_importance(model, features)
        if not importance.empty:
            importance.insert(0, "fold", fold.name)
            importance.insert(1, "candidate", candidate.name)
            importance_frames.append(importance)
        artifacts[candidate.name] = {"model": model, "features": features, "candidate": asdict(candidate)}
    artifacts["xgb_downside_rank_all"] = {
        "model": risk_model,
        "features": risk_features,
        "target": "label_downside_rank_26w",
    }
    return (
        pd.concat(prediction_frames, ignore_index=True),
        pd.DataFrame(metric_rows),
        pd.concat(admission_frames, ignore_index=True),
        pd.concat(importance_frames, ignore_index=True) if importance_frames else pd.DataFrame(),
        artifacts,
    )


def guardrail_mask(frame: pd.DataFrame, name: str) -> pd.Series:
    if name == "none":
        return pd.Series(True, index=frame.index)
    historical_value = pd.to_numeric(frame["historical_value_score_5y"], errors="coerce")
    pr = pd.to_numeric(frame["pr"], errors="coerce")
    trend = pd.to_numeric(frame["close_to_ma120"], errors="coerce")
    momentum = pd.to_numeric(frame["return_120d_cross_section_pct"], errors="coerce")
    if name == "historical_value_50":
        return historical_value.ge(50)
    if name == "value_or_absolute_trend":
        return (historical_value.ge(55) | pr.between(0, 2, inclusive="neither")) & trend.ge(-0.10)
    if name == "momentum_and_value":
        return momentum.ge(0.40) & (historical_value.ge(50) | pr.between(0, 2, inclusive="neither"))
    raise ValueError(name)


def calibrate(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    frame = predictions.copy()
    frame["opportunity_pct"] = frame.groupby(["candidate", "date"])["pred_opportunity"].rank(pct=True)
    frame["downside_safe_pct"] = frame.groupby(["candidate", "date"])["pred_downside_safe"].rank(pct=True)
    rows: list[dict] = []
    for candidate, candidate_frame in frame.groupby("candidate", sort=False):
        for risk_weight in (0.0, 0.20, 0.40):
            score_column = f"_score_{risk_weight:.1f}"
            candidate_frame = candidate_frame.copy()
            candidate_frame[score_column] = (
                candidate_frame["opportunity_pct"] * (1.0 - risk_weight)
                + candidate_frame["downside_safe_pct"] * risk_weight
            )
            for guardrail in ("none", "historical_value_50", "value_or_absolute_trend", "momentum_and_value"):
                eligible_candidate = candidate_frame[guardrail_mask(candidate_frame, guardrail)]
                for industry_cap in (2, 3):
                    fold_results: list[dict] = []
                    for fold, fold_frame in candidate_frame[~candidate_frame["reused_test"]].groupby("fold"):
                        monthly = fold_frame[month_end_week_mask(fold_frame)]
                        labelled = monthly.dropna(subset=["return_52w"])
                        eligible = eligible_candidate[eligible_candidate["fold"].eq(fold)]
                        eligible = eligible[month_end_week_mask(eligible)].dropna(subset=["return_52w"])
                        selected = select_industry_capped(
                            eligible,
                            score_column=score_column,
                            top_n=20,
                            max_per_industry=industry_cap,
                        )
                        summary = summarize_selection(selected, labelled)
                        fold_results.append({"fold": fold, **summary})
                    rows.append(
                        {
                            "candidate": candidate,
                            "risk_weight": risk_weight,
                            "guardrail": guardrail,
                            "industry_cap": industry_cap,
                            "positive_validation_folds": int(sum(row["return_delta"] > 0 for row in fold_results)),
                            "worst_validation_delta": float(min(row["return_delta"] for row in fold_results)),
                            "mean_validation_delta": float(np.mean([row["return_delta"] for row in fold_results])),
                            "mean_validation_excess": float(np.mean([row["mean_excess_52w"] for row in fold_results])),
                            "mean_validation_mae": float(np.mean([row["mean_mae_26w"] for row in fold_results])),
                        }
                    )
    grid = pd.DataFrame(rows).sort_values(
        ["positive_validation_folds", "worst_validation_delta", "mean_validation_delta", "mean_validation_mae"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    chosen = grid.iloc[0].to_dict()
    frame["entry_score"] = (
        frame["opportunity_pct"] * (1.0 - float(chosen["risk_weight"]))
        + frame["downside_safe_pct"] * float(chosen["risk_weight"])
    )
    frame["entry_score_pct"] = frame.groupby(["candidate", "date"])["entry_score"].rank(pct=True)
    return frame, grid, {
        "candidate": str(chosen["candidate"]),
        "risk_weight": float(chosen["risk_weight"]),
        "guardrail": str(chosen["guardrail"]),
        "industry_cap": int(chosen["industry_cap"]),
    }


def evaluate_chosen(predictions: pd.DataFrame, chosen: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    chosen_frame = predictions[predictions["candidate"].eq(chosen["candidate"])].copy()
    monthly_baseline = chosen_frame[month_end_week_mask(chosen_frame)].copy()
    eligible = chosen_frame[guardrail_mask(chosen_frame, str(chosen["guardrail"]))]
    monthly = eligible[month_end_week_mask(eligible)].copy()
    selections: list[pd.DataFrame] = []
    metrics: list[dict] = []
    for fold, fold_frame in monthly.groupby("fold", sort=False):
        labelled = monthly_baseline[
            monthly_baseline["fold"].eq(fold)
        ].dropna(subset=["return_52w"])
        selected = select_industry_capped(
            fold_frame,
            score_column="entry_score",
            top_n=20,
            max_per_industry=int(chosen["industry_cap"]),
        )
        selected["fold"] = fold
        selections.append(selected)
        evaluated = selected.dropna(subset=["return_52w"])
        summary = summarize_selection(evaluated, labelled)
        metrics.append(
            {
                "fold": fold,
                "reused_test": bool(fold_frame["reused_test"].iloc[0]),
                **summary,
            }
        )
    return pd.concat(selections, ignore_index=True), pd.DataFrame(metrics)


def build_cases(predictions: pd.DataFrame, chosen: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = predictions[predictions["candidate"].eq(chosen["candidate"])].dropna(
        subset=["entry_score_pct", "label_entry_utility", "excess_return_52w", "mae_26w"]
    ).copy()
    frame = frame[guardrail_mask(frame, str(chosen["guardrail"]))]
    failure = frame[
        frame["entry_score_pct"].ge(0.80)
        & (frame["excess_return_52w"].le(0) | frame["mae_26w"].le(-0.20))
    ].copy()
    failure["severity"] = (
        failure["entry_score_pct"] - failure["label_entry_utility"]
        + (-failure["mae_26w"]).clip(lower=0)
    )
    missed = frame[
        frame["entry_score_pct"].le(0.30) & frame["label_entry_utility"].ge(0.80)
    ].copy()
    missed["severity"] = missed["label_entry_utility"] - missed["entry_score_pct"]
    failure = classify_case_causes(
        cooldown_cases(failure, severity_column="severity"), kind="false_positive"
    )
    missed = classify_case_causes(
        cooldown_cases(missed, severity_column="severity"), kind="false_negative"
    )
    return failure, missed


def build_report(
    manifest: dict,
    chosen: dict,
    metrics: pd.DataFrame,
    grid: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    false_positive: pd.DataFrame,
    false_negative: pd.DataFrame,
) -> str:
    candidate_summary = (
        metrics[~metrics["reused_test"]]
        .groupby(["candidate", "target"], as_index=False)
        .agg(mean_spearman=("spearman", "mean"), worst_spearman=("spearman", "min"), folds=("fold", "nunique"))
        .sort_values("mean_spearman", ascending=False)
    )
    chosen_grid = grid.iloc[0]
    lines = [
        "# 长线好价格模型 V2：case 驱动实验",
        "",
        "## 当前结论",
        "",
        f"- 入选候选：`{chosen['candidate']}`；机会分权重 {1-float(chosen['risk_weight']):.0%}、回撤安全分权重 {float(chosen['risk_weight']):.0%}，价格护栏=`{chosen['guardrail']}`，每月20只、单行业最多{chosen['industry_cap']}只。",
        f"- 选择期（2020–2023）四折中正增量 {int(chosen_grid.positive_validation_folds)}/4，最差折相对好股基线 {float(chosen_grid.worst_validation_delta):.2%}，平均增量 {float(chosen_grid.mean_validation_delta):.2%}。",
        "- 2024+ 已被此前研究反复观察，只作复用诊断；不能因为 V2 改善就重新称为独立样本外。",
        "- 新标签同时纳入13/26/52周超额和26周最大不利波动，并在行业内重排，目的仍是相对低位建仓而非猜最低点。",
        "",
        "## 模型/标签走步相关性（2020–2023）",
        "",
        "| 候选 | 标签 | 平均 Spearman | 最差折 | 折数 |",
        "|---|---|---:|---:|---:|",
    ]
    for row in candidate_summary.itertuples(index=False):
        lines.append(f"| {row.candidate} | {row.target} | {row.mean_spearman:.3f} | {row.worst_spearman:.3f} | {row.folds} |")
    lines.extend(["", "## 月末推荐组合事件回测", "", "| 折 | 52周收益 | 好股基线 | 增量 | 52周超额 | 26周MAE | 超额胜率 |", "|---|---:|---:|---:|---:|---:|---:|"])
    for row in fold_metrics.itertuples(index=False):
        suffix = "（复用诊断）" if row.reused_test else ""
        lines.append(
            f"| {row.fold}{suffix} | {row.mean_return_52w:.2%} | {row.baseline_return_52w:.2%} | {row.return_delta:.2%} | {row.mean_excess_52w:.2%} | {row.mean_mae_26w:.2%} | {row.hit_rate_excess_52w:.2%} |"
        )
    lines.extend(
        [
            "",
            "## Case 反馈",
            "",
            f"- 去除同一股票91天内的重复周样本后，高分失败 {len(false_positive)} 个独立 episode，低分成功 {len(false_negative)} 个。",
            f"- 高分失败分类：{false_positive['case_cause'].value_counts().to_dict() if not false_positive.empty else {}}。",
            f"- 低分成功分类：{false_negative['case_cause'].value_counts().to_dict() if not false_negative.empty else {}}。",
            "- `new_information` 不能仅凭价格结果推断；没有接入信号后公告/事件连接的 case 保持 unresolved，避免事后编故事。",
            "",
            "## 数据与边界",
            "",
            f"- 数据集 {manifest['rows']:,} 行、{manifest['symbols']} 只股票，外部数据严格按交易日/公告日/统计截止日向后合并。",
            "- 当前行业字段来自现行 stock_basic 映射，不是历史行业快照，行业中性结果仍有重分类偏差。",
            "- 本阶段是模型选择与事件回测；完整的月度梯形资金组合、交易成本、分批建仓、退市敏感性在同目录的 full_backtest 报告中单独给出。",
            "- 研究用途，不构成投资建议；未通过新的封存期或前向纸面运行前不升级生产推荐。",
        ]
    )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict:
    started = perf_counter()
    dataset, manifest = load_dataset(args.force_dataset)
    external_columns = set(manifest["external_columns"])
    candidates = tuple(
        candidate for candidate in CANDIDATES
        if not args.skip_slow_models or candidate.family == "xgb"
    )
    prediction_frames: list[pd.DataFrame] = []
    metric_frames: list[pd.DataFrame] = []
    admission_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []
    last_artifacts: dict[str, dict] = {}
    if args.recalibrate_only:
        predictions = pd.read_parquet(REPORT_DIR / "walk_forward_predictions.parquet")
        guardrail_columns = [
            "date", "ts_code", "historical_value_score_5y", "close_to_ma120",
            "return_120d_cross_section_pct",
        ]
        missing = [column for column in guardrail_columns if column not in predictions.columns]
        if missing:
            right = dataset[["date", "ts_code", *missing]].drop_duplicates(["date", "ts_code"])
            predictions = predictions.merge(right, on=["date", "ts_code"], how="left", validate="many_to_one")
        metrics = pd.read_csv(REPORT_DIR / "model_metrics.csv")
        admission = pd.read_csv(REPORT_DIR / "factor_admission.csv")
        importance = pd.read_csv(REPORT_DIR / "feature_importance.csv")
    else:
        for fold in FOLDS:
            print(f"training {fold.name}", flush=True)
            predictions, metrics, admission, importance, artifacts = run_fold(
                dataset,
                fold,
                candidates,
                n_estimators=args.n_estimators,
                n_jobs=args.n_jobs,
                minimum_non_null=args.minimum_non_null,
                minimum_coverage=args.minimum_coverage,
                external_columns=external_columns,
            )
            prediction_frames.append(predictions)
            metric_frames.append(metrics)
            admission_frames.append(admission)
            if not importance.empty:
                importance_frames.append(importance)
            last_artifacts = artifacts
        predictions = pd.concat(prediction_frames, ignore_index=True)
        metrics = pd.concat(metric_frames, ignore_index=True)
        admission = pd.concat(admission_frames, ignore_index=True)
        importance = pd.concat(importance_frames, ignore_index=True) if importance_frames else pd.DataFrame()
    predictions, grid, chosen = calibrate(predictions)
    selections, fold_metrics = evaluate_chosen(predictions, chosen)
    false_positive, false_negative = build_cases(predictions, chosen)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    _atomic_frame(predictions, REPORT_DIR / "walk_forward_predictions.parquet", parquet=True)
    _atomic_frame(metrics, REPORT_DIR / "model_metrics.csv", parquet=False)
    _atomic_frame(admission, REPORT_DIR / "factor_admission.csv", parquet=False)
    _atomic_frame(importance, REPORT_DIR / "feature_importance.csv", parquet=False)
    _atomic_frame(grid, REPORT_DIR / "model_calibration_grid.csv", parquet=False)
    _atomic_frame(selections, REPORT_DIR / "monthly_selections.parquet", parquet=True)
    _atomic_frame(fold_metrics, REPORT_DIR / "fold_event_metrics.csv", parquet=False)
    _atomic_frame(false_positive, REPORT_DIR / "false_positive_episodes.csv", parquet=False)
    _atomic_frame(false_negative, REPORT_DIR / "false_negative_episodes.csv", parquet=False)

    if last_artifacts:
        for artifact_name in (chosen["candidate"], "xgb_downside_rank_all"):
            artifact = last_artifacts[artifact_name]
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            target = MODEL_DIR / f"{artifact_name}.joblib"
            temporary = target.with_suffix(f".{os.getpid()}.tmp.joblib")
            joblib.dump(
                {
                    **artifact,
                    "research_only": True,
                    "dataset_version": V2_DATASET_VERSION,
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                },
                temporary,
            )
            os.replace(temporary, target)
    experiment = {
        "status": "success",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": manifest,
        "folds": [asdict(fold) for fold in FOLDS],
        "candidates": [asdict(candidate) for candidate in candidates],
        "chosen": chosen,
        "selection_period": "2020-2023 walk-forward only",
        "reused_diagnostic_period": "2024-2026",
        "recalibrate_only": bool(args.recalibrate_only),
        "elapsed_seconds": perf_counter() - started,
    }
    _atomic_json(experiment, REPORT_DIR / "experiment_manifest.json")
    report = build_report(manifest, chosen, metrics, grid, fold_metrics, false_positive, false_negative)
    report_path = REPORT_DIR / "report.md"
    temporary_report = report_path.with_suffix(f".{os.getpid()}.tmp.md")
    temporary_report.write_text(report, encoding="utf-8")
    os.replace(temporary_report, report_path)
    return {**experiment, "report": str(report_path)}


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2), flush=True)
