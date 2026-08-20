#!/usr/bin/env python
"""Validate the third long-strategy quality iteration.

The experiment is deliberately incremental.  It keeps the selected V2 risk
head fixed and tests whether newly available point-in-time annual evidence
improves selection when used as:

1. a hard cash-conversion/durability gate;
2. a soft enhanced-quality score;
3. a quality plus industry-relative value overlay.
4. raw factors or interpretable sub-scores in a retrained opportunity head.

2020-2023 walk-forward folds select the recommendation.  The already-observed
2024+ period is reported only as reused historical diagnosis.
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

import numpy as np
import pandas as pd
from xgboost import XGBRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "research"))

from backtest_long_entry_model_v2 import (  # noqa: E402
    build_price_lookup,
    find_full_daily_cache,
    load_delist_dates,
    load_prices,
    run_ladder,
)
from quant.features.long_quality_factors import (  # noqa: E402
    add_enhanced_long_scores,
    build_annual_quality_events,
    merge_annual_quality_asof,
)
from quant.features.factor_registry import (  # noqa: E402
    LONG_ANNUAL_QUALITY_RAW_FACTOR_COLUMNS,
    LONG_ANNUAL_QUALITY_SCORE_FACTOR_COLUMNS,
)
from quant.research.long_entry_v2 import (  # noqa: E402
    equity_metrics,
    month_end_week_mask,
    select_industry_capped,
    summarize_selection,
)
from quant.features.long_weekly_factors import long_model_candidate_columns  # noqa: E402
from quant.features.project_factor_layer import admit_factors_by_sample  # noqa: E402


REPORT_DIR = PROJECT_ROOT / "reports/long_quality_iteration_v3"
BASE_DATASET = PROJECT_ROOT / "data/features/long_entry/weekly_training_v2.parquet"
PREDICTIONS = PROJECT_ROOT / "reports/long_entry_model_v2/walk_forward_predictions.parquet"
V2_MANIFEST = PROJECT_ROOT / "reports/long_entry_model_v2/experiment_manifest.json"
QUALITY_FACTOR_CACHE = PROJECT_ROOT / "data/features/long_entry/weekly_quality_factors_v1.parquet"
QUALITY_FACTOR_MANIFEST = QUALITY_FACTOR_CACHE.with_suffix(".manifest.json")
QUALITY_FACTOR_VERSION = "annual-quality-v1-first-published-pit"


@dataclass(frozen=True)
class Candidate:
    name: str
    score_column: str
    gate: str = "none"
    description: str = ""


CANDIDATES = (
    Candidate("current_v2", "score_current_v2", description="当前 V2 建仓模型分"),
    Candidate(
        "current_v2_cashflow_gate",
        "score_current_v2",
        gate="cashflow_gate_08",
        description="当前 V2 + 三年现金转换硬门槛",
    ),
    Candidate(
        "v3_quality20",
        "score_v3_quality20",
        description="80% V2 + 20% 增强质量分",
    ),
    Candidate(
        "v3_quality20_cashflow_gate",
        "score_v3_quality20",
        gate="cashflow_gate_08",
        description="质量增强分 + 现金流硬门槛",
    ),
    Candidate(
        "v3_quality_value",
        "score_v3_quality_value",
        description="70% V2 + 20% 增强质量 + 10% 混合价值",
    ),
    Candidate(
        "v3_quality_value_durable",
        "score_v3_quality_value",
        gate="durability_gate",
        description="质量价值模型 + 现金流/盈利持续性/商誉硬门槛",
    ),
    Candidate(
        "rule_quality_value",
        "score_rule_quality_value",
        description="纯可解释质量/价值/趋势/风险分",
    ),
    Candidate(
        "enhanced_quality_only",
        "score_enhanced_quality",
        description="只按增强好股票分排序",
    ),
    Candidate(
        "old_quality_only",
        "score_old_quality",
        description="旧好股票分排序对照",
    ),
)

QUALITY_RAW_FEATURES = LONG_ANNUAL_QUALITY_RAW_FACTOR_COLUMNS
QUALITY_SCORE_FEATURES = LONG_ANNUAL_QUALITY_SCORE_FACTOR_COLUMNS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate long quality iteration V3")
    parser.add_argument("--force-factors", action="store_true")
    parser.add_argument("--skip-full-backtest", action="store_true")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--industry-cap", type=int, default=3)
    return parser.parse_args()


def _atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp.json")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp.csv")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp.parquet")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _read_raw(name: str, columns: list[str]) -> pd.DataFrame:
    path = PROJECT_ROOT / f"data/raw/{name}.parquet"
    available = set(pd.read_parquet(path, engine="pyarrow").columns)
    return pd.read_parquet(path, columns=[column for column in columns if column in available])


def build_quality_factor_cache(base: pd.DataFrame, *, force: bool) -> tuple[pd.DataFrame, dict]:
    if QUALITY_FACTOR_CACHE.exists() and QUALITY_FACTOR_MANIFEST.exists() and not force:
        manifest = json.loads(QUALITY_FACTOR_MANIFEST.read_text(encoding="utf-8"))
        if manifest.get("version") == QUALITY_FACTOR_VERSION:
            cached = pd.read_parquet(QUALITY_FACTOR_CACHE)
            cached["date"] = pd.to_datetime(cached["date"])
            return cached, {**manifest, "cache_hit": True}

    started = perf_counter()
    fina = _read_raw(
        "fina_indicator",
        [
            "ts_code",
            "ann_date",
            "end_date",
            "roe",
            "roa",
            "netprofit_margin",
            "grossprofit_margin",
            "debt_to_assets",
            "current_ratio",
            "quick_ratio",
            "ar_turn",
            "inv_turn",
            "assets_turn",
            "or_yoy",
            "basic_eps_yoy",
        ],
    )
    income = _read_raw(
        "income",
        ["ts_code", "ann_date", "end_date", "report_type", "revenue", "n_income_attr_p"],
    )
    cashflow = _read_raw(
        "cashflow",
        [
            "ts_code",
            "ann_date",
            "end_date",
            "report_type",
            "n_cashflow_act",
            "c_pay_acq_const_fiolta",
        ],
    )
    balance = _read_raw(
        "balancesheet",
        [
            "ts_code",
            "ann_date",
            "end_date",
            "report_type",
            "total_assets",
            "total_liab",
            "total_hldr_eqy_exc_min_int",
            "money_cap",
            "inventories",
            "intan_assets",
            "goodwill",
        ],
    )
    events = build_annual_quality_events(fina, income, cashflow, balance)
    enriched = merge_annual_quality_asof(base, events)
    enriched = add_enhanced_long_scores(enriched)
    new_columns = [
        column
        for column in enriched.columns
        if column not in base.columns or column in {"date", "ts_code"}
    ]
    factors = enriched[new_columns].copy()
    _atomic_parquet(factors, QUALITY_FACTOR_CACHE)
    manifest = {
        "version": QUALITY_FACTOR_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "point_in_time": "first local annual announcement; available_at <= signal date",
        "rows": int(len(factors)),
        "symbols": int(factors["ts_code"].nunique()),
        "events": int(len(events)),
        "columns": list(factors.columns),
        "elapsed_seconds": perf_counter() - started,
        "cache_hit": False,
    }
    _atomic_json(manifest, QUALITY_FACTOR_MANIFEST)
    return factors, manifest


def build_experiment_frame(force_factors: bool) -> tuple[pd.DataFrame, dict]:
    base = pd.read_parquet(BASE_DATASET)
    base["date"] = pd.to_datetime(base["date"])
    factors, factor_manifest = build_quality_factor_cache(base, force=force_factors)
    factor_columns = [column for column in factors.columns if column not in {"date", "ts_code"}]
    base = base.merge(factors, on=["date", "ts_code"], how="left", validate="one_to_one")

    model_manifest = json.loads(V2_MANIFEST.read_text(encoding="utf-8"))
    chosen_name = str(model_manifest["chosen"]["candidate"])
    predictions = pd.read_parquet(PREDICTIONS)
    predictions["date"] = pd.to_datetime(predictions["date"])
    predictions = predictions[predictions["candidate"].eq(chosen_name)][
        ["date", "ts_code", "fold", "reused_test", "entry_score", "pred_downside_safe"]
    ].drop_duplicates(["date", "ts_code"])
    frame = base.merge(predictions, on=["date", "ts_code"], how="left", validate="one_to_one")
    frame["score_current_v2"] = pd.to_numeric(frame["entry_score"], errors="coerce")
    enhanced = pd.to_numeric(frame["enhanced_good_stock_score"], errors="coerce") / 100.0
    blended_value = pd.to_numeric(frame["blended_value_score"], errors="coerce") / 100.0
    frame["score_v3_quality20"] = frame["score_current_v2"] * 0.80 + enhanced * 0.20
    frame["score_v3_quality_value"] = (
        frame["score_current_v2"] * 0.70 + enhanced * 0.20 + blended_value * 0.10
    )
    frame["score_rule_quality_value"] = pd.to_numeric(
        frame["rule_long_model_score"], errors="coerce"
    ) / 100.0
    frame["score_enhanced_quality"] = enhanced
    frame["score_old_quality"] = pd.to_numeric(frame["good_stock_score"], errors="coerce") / 100.0
    return frame, {
        "factors": factor_manifest,
        "factor_columns": factor_columns,
        "external_columns": model_manifest.get("dataset", {}).get("external_columns", []),
        "v2_candidate": chosen_name,
        "v2_chosen": model_manifest["chosen"],
    }


def _date_equal_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("date")["ts_code"].transform("count").astype(float)
    weights = 1.0 / counts.clip(lower=1.0)
    return (weights / weights.mean()).to_numpy(dtype=float)


def train_quality_factor_models(
    frame: pd.DataFrame,
    data_manifest: dict,
    *,
    n_estimators: int = 220,
) -> tuple[pd.DataFrame, tuple[Candidate, ...], pd.DataFrame, pd.DataFrame]:
    """Train factor ablations while keeping the current V2 risk head fixed."""

    fold_specs = (
        ("wf_2020", "2020-01-01", "2020-12-31", False),
        ("wf_2021", "2021-01-01", "2021-12-31", False),
        ("wf_2022", "2022-01-01", "2022-12-31", False),
        ("wf_2023", "2023-01-01", "2023-12-31", False),
        ("reused_2024", "2024-01-01", "2024-12-31", True),
        ("reused_2025", "2025-01-01", "2025-12-31", True),
        ("reused_2026", "2026-01-01", "2026-12-31", True),
    )
    all_candidates = long_model_candidate_columns(frame)
    existing_external = set(data_manifest["factors"].get("external_columns", ()))
    # The V2 manifest stores external columns at the dataset-manifest level.
    existing_external.update(data_manifest.get("external_columns", ()))
    new_factor_columns = set(data_manifest["factor_columns"])
    forbidden = {
        "entry_score",
        "pred_downside_safe",
        "fold",
        "reused_test",
        "score_current_v2",
        "score_v3_quality20",
        "score_v3_quality_value",
        "score_rule_quality_value",
        "score_enhanced_quality",
        "score_old_quality",
    }
    base_core = [
        column
        for column in all_candidates
        if column not in existing_external
        and column not in new_factor_columns
        and column not in forbidden
        and not column.startswith("score_")
    ]
    feature_sets = {
        "xgb_quality_raw": [*base_core, *[column for column in QUALITY_RAW_FEATURES if column in frame]],
        "xgb_quality_scores": [
            *base_core,
            *[column for column in QUALITY_SCORE_FEATURES if column in frame],
        ],
    }
    predictions: list[pd.DataFrame] = []
    metric_rows: list[dict] = []
    importance_rows: list[pd.DataFrame] = []
    for fold, validation_start, validation_end, reused in fold_specs:
        purge_cutoff = pd.Timestamp(validation_start) - pd.Timedelta(days=190)
        train = frame[frame["date"] < purge_cutoff].dropna(subset=["label_entry_utility_26w"])
        validation = frame[frame["date"].between(validation_start, validation_end)].copy()
        if train.empty or validation.empty:
            continue
        for candidate, requested in feature_sets.items():
            admitted, _ = admit_factors_by_sample(
                train,
                requested,
                minimum_non_null_rows=max(1000, int(len(train) * 0.05)),
                minimum_coverage=0.05,
            )
            model = XGBRegressor(
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
                n_jobs=min(8, os.cpu_count() or 4),
                random_state=42,
            )
            model.fit(
                train[admitted].apply(pd.to_numeric, errors="coerce"),
                train["label_entry_utility_26w"],
                sample_weight=_date_equal_weights(train),
            )
            opportunity = model.predict(
                validation[admitted].apply(pd.to_numeric, errors="coerce")
            )
            current = validation[["date", "ts_code"]].copy()
            current["candidate"] = candidate
            current["fold"] = fold
            current["reused_test"] = reused
            current["predicted_opportunity"] = opportunity
            current["opportunity_pct"] = current.groupby("date")["predicted_opportunity"].rank(
                pct=True
            )
            risk = pd.to_numeric(validation["pred_downside_safe"], errors="coerce")
            current["risk_pct"] = risk.groupby(validation["date"]).rank(pct=True).to_numpy()
            current["model_score"] = current["opportunity_pct"] * 0.60 + current["risk_pct"] * 0.40
            predictions.append(current)
            valid = validation["label_entry_utility_26w"].notna()
            spearman = pd.Series(opportunity[valid.to_numpy()]).corr(
                validation.loc[valid, "label_entry_utility_26w"].reset_index(drop=True),
                method="spearman",
            )
            metric_rows.append(
                {
                    "candidate": candidate,
                    "fold": fold,
                    "reused_test": reused,
                    "train_rows": int(len(train)),
                    "validation_rows": int(valid.sum()),
                    "factor_count": int(len(admitted)),
                    "spearman": float(spearman) if pd.notna(spearman) else np.nan,
                }
            )
            importance = pd.DataFrame(
                {
                    "candidate": candidate,
                    "fold": fold,
                    "factor": admitted,
                    "importance": model.feature_importances_,
                }
            )
            importance_rows.append(importance)
    prediction_frame = pd.concat(predictions, ignore_index=True)
    wide = prediction_frame.pivot(index=["date", "ts_code"], columns="candidate", values="model_score")
    wide = wide.rename(columns=lambda name: f"score_{name}").reset_index()
    merged = frame.merge(wide, on=["date", "ts_code"], how="left", validate="one_to_one")
    candidates = (
        *CANDIDATES,
        Candidate(
            "xgb_quality_raw",
            "score_xgb_quality_raw",
            description="V2 核心因子 + 原始年度现金流/持续性因子重训",
        ),
        Candidate(
            "xgb_quality_scores",
            "score_xgb_quality_scores",
            description="V2 核心因子 + 可解释质量子分重训",
        ),
    )
    return (
        merged,
        candidates,
        pd.DataFrame(metric_rows),
        pd.concat(importance_rows, ignore_index=True),
    )


def gate_mask(frame: pd.DataFrame, gate: str) -> pd.Series:
    if gate == "none":
        return pd.Series(True, index=frame.index)
    return frame[gate].fillna(False).astype(bool)


def select_and_evaluate(
    frame: pd.DataFrame,
    *,
    candidates: tuple[Candidate, ...],
    top_n: int,
    industry_cap: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = frame[frame["fold"].notna()].copy()
    monthly = frame[month_end_week_mask(frame)].copy()
    baseline = monthly.copy()
    historical_guard = pd.to_numeric(monthly["historical_value_score_5y"], errors="coerce").ge(50)
    selection_parts: list[pd.DataFrame] = []
    metric_rows: list[dict] = []

    for candidate in candidates:
        eligible = monthly[historical_guard & gate_mask(monthly, candidate.gate)].copy()
        selected = select_industry_capped(
            eligible,
            score_column=candidate.score_column,
            top_n=top_n,
            max_per_industry=industry_cap,
        )
        selected["candidate"] = candidate.name
        selected["candidate_description"] = candidate.description
        selection_parts.append(selected)
        for fold, fold_frame in monthly.groupby("fold", sort=False):
            labelled_baseline = baseline[baseline["fold"].eq(fold)].dropna(subset=["return_52w"])
            evaluated = selected[selected["fold"].eq(fold)].dropna(subset=["return_52w"])
            summary = summarize_selection(evaluated, labelled_baseline)
            fold_eligible = eligible[eligible["fold"].eq(fold)]
            metric_rows.append(
                {
                    "candidate": candidate.name,
                    "description": candidate.description,
                    "fold": fold,
                    "reused_test": bool(fold_frame["reused_test"].iloc[0]),
                    "eligible_rows": int(len(fold_eligible)),
                    "eligible_symbols": int(fold_eligible["ts_code"].nunique()),
                    "average_picks": float(evaluated.groupby("date")["ts_code"].nunique().mean())
                    if not evaluated.empty
                    else 0.0,
                    **summary,
                }
            )
    selections = pd.concat(selection_parts, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    current = metrics[metrics["candidate"].eq("current_v2")][
        ["fold", "return_delta", "mean_excess_52w", "mean_mae_26w"]
    ].rename(
        columns={
            "return_delta": "current_return_delta",
            "mean_excess_52w": "current_mean_excess_52w",
            "mean_mae_26w": "current_mean_mae_26w",
        }
    )
    metrics = metrics.merge(current, on="fold", how="left", validate="many_to_one")
    metrics["return_delta_vs_current"] = metrics["return_delta"] - metrics["current_return_delta"]
    metrics["excess_delta_vs_current"] = (
        metrics["mean_excess_52w"] - metrics["current_mean_excess_52w"]
    )
    metrics["mae_delta_vs_current"] = metrics["mean_mae_26w"] - metrics["current_mean_mae_26w"]

    validation = metrics[~metrics["reused_test"]].copy()
    summary_rows: list[dict] = []
    for candidate, group in validation.groupby("candidate", sort=False):
        summary_rows.append(
            {
                "candidate": candidate,
                "description": group["description"].iloc[0],
                "validation_folds": int(len(group)),
                "positive_vs_current_folds": int(group["return_delta_vs_current"].gt(0).sum()),
                "worst_return_delta_vs_current": float(group["return_delta_vs_current"].min()),
                "mean_return_delta_vs_current": float(group["return_delta_vs_current"].mean()),
                "mean_excess_delta_vs_current": float(group["excess_delta_vs_current"].mean()),
                "mean_mae_delta_vs_current": float(group["mae_delta_vs_current"].mean()),
                "mean_return_delta_vs_good_pool": float(group["return_delta"].mean()),
                "mean_excess_52w": float(group["mean_excess_52w"].mean()),
                "mean_mae_26w": float(group["mean_mae_26w"].mean()),
                "average_picks": float(group["average_picks"].mean()),
            }
        )
    candidate_summary = pd.DataFrame(summary_rows).sort_values(
        [
            "positive_vs_current_folds",
            "worst_return_delta_vs_current",
            "mean_return_delta_vs_current",
            "mean_mae_delta_vs_current",
        ],
        ascending=[False, False, False, False],
    )
    return selections, metrics, candidate_summary.reset_index(drop=True)


def choose_candidate(candidate_summary: pd.DataFrame) -> tuple[str, str]:
    eligible = candidate_summary[
        (candidate_summary["positive_vs_current_folds"] >= 3)
        & (candidate_summary["worst_return_delta_vs_current"] >= -0.02)
        & (candidate_summary["average_picks"] >= 15)
    ]
    if eligible.empty:
        return "current_v2", "没有候选满足至少3/4折改善、最差折不低于-2个百分点和月均15只的升级门槛"
    chosen = str(eligible.iloc[0]["candidate"])
    if chosen == "current_v2":
        return chosen, "当前 V2 仍是验证期最稳健候选"
    return chosen, "满足预注册的多数折改善、最差折和覆盖率门槛"


def run_full_backtests(selections: pd.DataFrame, candidates: list[str]) -> pd.DataFrame:
    chosen = selections[selections["candidate"].isin(candidates)].copy()
    symbols = set(chosen["ts_code"].astype(str))
    daily_path = find_full_daily_cache()
    prices = load_prices(daily_path, symbols)
    lookup = build_price_lookup(prices)
    index = pd.read_parquet(PROJECT_ROOT / "data/raw/index_000300.SH.parquet")
    index["date"] = pd.to_datetime(index["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
    calendar = pd.DatetimeIndex(index["date"].dropna().sort_values().unique())
    delist_dates = load_delist_dates()
    rows: list[dict] = []
    for candidate in candidates:
        picks = chosen[chosen["candidate"].eq(candidate)]
        equity, diagnostics = run_ladder(
            picks,
            lookup=lookup,
            calendar=calendar,
            delist_dates=delist_dates,
            slots=12,
            buy_cost=0.001,
            sell_cost=0.002,
            staged=False,
            conservative_delist=True,
        )
        if equity.empty:
            continue
        _atomic_csv(equity, REPORT_DIR / f"equity_{candidate}.csv")
        rows.append({"candidate": candidate, **equity_metrics(equity), **diagnostics})
    return pd.DataFrame(rows)


def factor_coverage(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "cashflow_quality_3y",
        "free_cashflow_margin_3y",
        "accruals_to_assets_3y",
        "profit_positive_share_5y",
        "cfo_positive_share_5y",
        "revenue_growth_positive_share_5y",
        "roe_mean_5y",
        "roe_std_5y",
        "annual_goodwill_to_assets",
        "enhanced_good_stock_score",
        "blended_value_score",
    ]
    rows = []
    for column in columns:
        usable = pd.to_numeric(frame[column], errors="coerce").notna()
        rows.append(
            {
                "factor": column,
                "coverage": float(usable.mean()),
                "rows": int(usable.sum()),
                "symbols": int(frame.loc[usable, "ts_code"].nunique()),
                "first_date": frame.loc[usable, "date"].min().date().isoformat() if usable.any() else None,
            }
        )
    return pd.DataFrame(rows)


def build_report(
    chosen: str,
    reason: str,
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    coverage: pd.DataFrame,
    full: pd.DataFrame,
) -> str:
    validation = metrics[~metrics["reused_test"]]
    diagnostic = metrics[metrics["reused_test"]]
    cash_gate = summary[summary["candidate"].eq("current_v2_cashflow_gate")]
    quality = summary[summary["candidate"].eq("v3_quality20")]
    quality_value = summary[summary["candidate"].eq("v3_quality_value")]

    def finding(frame: pd.DataFrame, label: str) -> str:
        if frame.empty:
            return f"- {label}：无有效结果。"
        row = frame.iloc[0]
        return (
            f"- {label}：相对当前 V2 的验证期平均 52 周增量 "
            f"{row['mean_return_delta_vs_current']:.2%}，改善 {int(row['positive_vs_current_folds'])}/4 折，"
            f"最差折 {row['worst_return_delta_vs_current']:.2%}，26 周 MAE 变化 "
            f"{row['mean_mae_delta_vs_current']:.2%}。"
        )

    lines = [
        "# 长线策略 V3：质量、现金流与行业相对价值验证",
        "",
        "## 结论",
        "",
        f"- 研究选择：`{chosen}`。{reason}。",
        "- 2020—2023 为本轮选择期；2024+ 已被旧研究反复观察，只作为复用诊断，不重新声称独立样本外。",
        "- 所有年度财务因子使用本地首次披露值，并要求 `available_at <= signal_date`；后续更正不回填历史信号。",
        "",
        "## 建议逐项验证",
        "",
        finding(cash_gate, "现金流 0.8 硬门槛"),
        finding(quality, "现金流/盈利持续性作为20%软质量分"),
        finding(quality_value, "软质量分并加入10%行业相对价值"),
        "",
        "## 验证期候选摘要",
        "",
        "| 候选 | 改善折数 | 最差折相对V2 | 平均相对V2 | 相对好股池 | 26周MAE | 月均只数 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.candidate} | {row.positive_vs_current_folds}/4 | "
            f"{row.worst_return_delta_vs_current:.2%} | {row.mean_return_delta_vs_current:.2%} | "
            f"{row.mean_return_delta_vs_good_pool:.2%} | {row.mean_mae_26w:.2%} | {row.average_picks:.1f} |"
        )
    lines.extend(
        [
            "",
            "## 逐折事件结果",
            "",
            "| 候选 | 折 | 复用诊断 | 52周收益 | 相对好股池 | 相对V2 | 52周超额 | 26周MAE |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in pd.concat([validation, diagnostic]).itertuples(index=False):
        lines.append(
            f"| {row.candidate} | {row.fold} | {'是' if row.reused_test else '否'} | "
            f"{row.mean_return_52w:.2%} | {row.return_delta:.2%} | {row.return_delta_vs_current:.2%} | "
            f"{row.mean_excess_52w:.2%} | {row.mean_mae_26w:.2%} |"
        )
    if not full.empty:
        lines.extend(
            [
                "",
                "## 完整资金梯形回测（2020+，含复用诊断）",
                "",
                "| 候选 | 年化 | 总收益 | 最大回撤 | 年化波动 | 平均投入 |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in full.itertuples(index=False):
            lines.append(
                f"| {row.candidate} | {row.annual_return:.2%} | {row.total_return:.2%} | "
                f"{row.max_drawdown:.2%} | {row.annual_volatility:.2%} | "
                f"{getattr(row, 'average_invested_ratio', np.nan):.2%} |"
            )
    lines.extend(
        [
            "",
            "## 因子覆盖",
            "",
            "| 因子 | 覆盖率 | 股票数 | 首次可用 |",
            "|---|---:|---:|---|",
        ]
    )
    for row in coverage.itertuples(index=False):
        lines.append(f"| {row.factor} | {row.coverage:.2%} | {row.symbols} | {row.first_date or '-'} |")
    lines.extend(
        [
            "",
            "## 证据边界",
            "",
            "- 本轮只在旧好股票门控后的股票池内验证新增质量证据，不能证明被旧门控排除的股票是否应重新纳入。",
            "- 行业字段仍是当前 stock_basic 映射，不是历史行业成分快照；行业相对分存在重分类偏差。",
            "- 首次披露值口径避免未来更正回填，但不会在后续更正日更新同一年度数据，是保守近似。",
            "- 真正升级生产仍需要模型冻结后的新增周数据或前向纸面运行。仅供研究与教育用途。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    started = perf_counter()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    frame, data_manifest = build_experiment_frame(args.force_factors)
    frame, candidates, model_metrics, feature_importance = train_quality_factor_models(
        frame, data_manifest
    )
    coverage = factor_coverage(frame)
    selections, metrics, summary = select_and_evaluate(
        frame,
        candidates=candidates,
        top_n=args.top_n,
        industry_cap=args.industry_cap,
    )
    chosen, reason = choose_candidate(summary)
    full = pd.DataFrame()
    if not args.skip_full_backtest:
        full = run_full_backtests(selections, ["current_v2", chosen])

    _atomic_csv(coverage, REPORT_DIR / "factor_coverage.csv")
    _atomic_csv(metrics, REPORT_DIR / "fold_metrics.csv")
    _atomic_csv(summary, REPORT_DIR / "candidate_summary.csv")
    _atomic_csv(model_metrics, REPORT_DIR / "model_metrics.csv")
    _atomic_csv(feature_importance, REPORT_DIR / "feature_importance.csv")
    _atomic_parquet(selections, REPORT_DIR / "monthly_selections.parquet")
    if not full.empty:
        _atomic_csv(full, REPORT_DIR / "full_backtest_summary.csv")
    report = build_report(chosen, reason, metrics, summary, coverage, full)
    (REPORT_DIR / "report.md").write_text(report, encoding="utf-8")

    chosen_candidate = next(candidate for candidate in candidates if candidate.name == chosen)
    manifest = {
        "status": "success",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "selection_period": "2020-2023 walk-forward folds",
        "reused_diagnostic_period": "2024-2026",
        "top_n": args.top_n,
        "industry_cap": args.industry_cap,
        "candidates": [asdict(candidate) for candidate in candidates],
        "chosen": asdict(chosen_candidate),
        "chosen_reason": reason,
        "upgrade_gate": {
            "minimum_positive_folds": 3,
            "minimum_worst_fold_delta": -0.02,
            "minimum_average_picks": 15,
        },
        "data": data_manifest,
        "elapsed_seconds": perf_counter() - started,
    }
    _atomic_json(manifest, REPORT_DIR / "experiment_manifest.json")
    print(report)


if __name__ == "__main__":
    main()
