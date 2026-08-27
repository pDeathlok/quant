#!/usr/bin/env python3
"""Backtest right-side confirmation after completed monthly low-9 anchors."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant.features.long_quality_factors import build_annual_quality_events
from quant.research.blood_chip_deep_base import build_deep_base_features
from quant.research.blood_chip_multidimensional import (
    MultidimensionalGateConfig,
    merge_financial_survival_asof,
)
from quant.research.monthly_low_zone import (
    EVENT_PERIODS,
    MonthlyLowZoneConfig,
    evaluate_monthly_low_zone_events,
)
from quant.research.monthly_low_zone_confirmation import (
    CONFIRMATION_RULES,
    MonthlyConfirmationConfig,
    build_benchmark_confirmation_features,
    build_market_breadth_features,
    generate_monthly_confirmation_signals,
    summarize_confirmation_events,
)


SELECTION_PERIODS = ("development_2013_2016", "validation_2017_2020")
SELECTION_HORIZON = 252
DAILY_COLUMNS = [
    "ts_code",
    "date",
    "open",
    "high",
    "low",
    "adjusted_open",
    "adjusted_high",
    "adjusted_low",
    "adjusted_close",
    "prior_amount_median_20d",
    "sessions_since_new_low",
    "return_20d",
    "base_position",
    "down_amount_share_ratio",
    "volatility_contraction_ratio",
]
FINANCIAL_SOURCE_COLUMNS = {
    "fina_indicator": [
        "ts_code",
        "end_date",
        "ann_date",
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
    "income": [
        "ts_code",
        "ann_date",
        "end_date",
        "report_type",
        "revenue",
        "n_income_attr_p",
    ],
    "cashflow": [
        "ts_code",
        "ann_date",
        "end_date",
        "report_type",
        "n_cashflow_act",
        "c_pay_acq_const_fiolta",
    ],
    "balancesheet": [
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
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-cache",
        type=Path,
        default=Path("data/research/blood_chip_deep_base/features.parquet"),
    )
    parser.add_argument(
        "--supplemental-root",
        type=Path,
        default=Path("data/research/low9_kdj_rebound/supplemental_daily"),
    )
    parser.add_argument(
        "--monthly-signals",
        type=Path,
        default=Path("reports/research/monthly_low_zone/signals.parquet"),
    )
    parser.add_argument(
        "--weekly-features",
        type=Path,
        default=Path("data/research/monthly_low_zone/weekly_features.parquet"),
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path(
            "data/research/low9_kdj_rebound/index_000001.SH_20100101_20260731.parquet"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/research/monthly_low_zone_confirmation"),
    )
    parser.add_argument("--minimum-sessions-since-new-low", type=int, default=20)
    parser.add_argument("--enable-breadth-rules", action="store_true")
    parser.add_argument("--enable-survival-ablation", action="store_true")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    return parser.parse_args()


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(_json_value(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _load_primary_features(path: Path, cutoff: pd.Timestamp) -> pd.DataFrame:
    import pyarrow.parquet as pq

    available = set(pq.ParquetFile(path).schema.names)
    missing = sorted(set(DAILY_COLUMNS) - available)
    if missing:
        raise ValueError(f"feature cache missing columns: {missing}")
    frame = pd.read_parquet(path, columns=DAILY_COLUMNS)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    return frame.loc[frame["date"].le(cutoff)].copy()


def _load_supplemental_features(root: Path, cutoff: pd.Timestamp) -> pd.DataFrame:
    raw_columns = [
        "ts_code",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "pct_chg",
        "vol",
        "amount",
    ]
    frames: list[pd.DataFrame] = []
    paths = sorted(root.glob("*.parquet"))
    for index, path in enumerate(paths, start=1):
        frames.append(pd.read_parquet(path, columns=raw_columns))
        if index % 100 == 0:
            print(f"loaded supplemental histories: {index}/{len(paths)}", flush=True)
    if not frames:
        return pd.DataFrame(columns=DAILY_COLUMNS)
    raw = pd.concat(frames, ignore_index=True, sort=False)
    print(f"building supplemental deep-base features: {len(raw):,} rows", flush=True)
    features = build_deep_base_features(raw)
    features = features.loc[features["date"].le(cutoff)].copy()
    return features[DAILY_COLUMNS]


def _load_daily(
    feature_cache: Path,
    supplemental_root: Path,
    cutoff: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    print(f"loading primary features: {feature_cache}", flush=True)
    primary = _load_primary_features(feature_cache, cutoff)
    supplemental = _load_supplemental_features(supplemental_root, cutoff)
    primary_symbols = set(primary["ts_code"].astype(str).unique())
    overlap_rows = int(supplemental["ts_code"].astype(str).isin(primary_symbols).sum())
    combined = pd.concat([primary, supplemental], ignore_index=True, sort=False)
    combined = (
        combined.sort_values(["ts_code", "date"])
        .drop_duplicates(["ts_code", "date"], keep="last")
        .reset_index(drop=True)
    )
    return combined, {
        "primary_rows": int(len(primary)),
        "primary_symbols": int(primary["ts_code"].nunique()),
        "supplemental_rows": int(len(supplemental)),
        "supplemental_symbols": int(supplemental["ts_code"].nunique()),
        "supplemental_overlap_rows": overlap_rows,
        "combined_rows": int(len(combined)),
        "combined_symbols": int(combined["ts_code"].nunique()),
    }


def _load_benchmark(path: Path, cutoff: pd.Timestamp) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    values = frame["trade_date"] if "trade_date" in frame else frame["date"]
    parsed = pd.to_datetime(values, errors="coerce")
    compact = pd.to_datetime(
        values.astype("string").str.replace(r"\.0$", "", regex=True).str[:8],
        format="%Y%m%d",
        errors="coerce",
    )
    frame["date"] = compact.fillna(parsed).dt.normalize()
    return frame.loc[frame["date"].le(cutoff)].sort_values("date").copy()


def _read_available_columns(path: Path, columns: list[str]) -> pd.DataFrame:
    import pyarrow.parquet as pq

    available = set(pq.ParquetFile(path).schema.names)
    return pd.read_parquet(path, columns=[column for column in columns if column in available])


def _add_survival_ablation(
    signals: pd.DataFrame,
    diagnostics: pd.DataFrame,
    raw_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    sources = {
        name: _read_available_columns(raw_dir / f"{name}.parquet", columns)
        for name, columns in FINANCIAL_SOURCE_COLUMNS.items()
    }
    annual_events = build_annual_quality_events(
        sources["fina_indicator"],
        sources["income"],
        sources["cashflow"],
        sources["balancesheet"],
    )
    source_to_target = {
        "breadth_relative_weekly": "breadth_relative_weekly_survival",
        "breadth_relative_weekly_exhaustion": (
            "breadth_relative_weekly_exhaustion_survival"
        ),
    }
    gate_config = MultidimensionalGateConfig()
    source_signals = signals.loc[signals["rule"].isin(source_to_target)].copy()
    enriched = merge_financial_survival_asof(source_signals, annual_events)
    signal_date = pd.to_datetime(enriched["signal_date"], errors="coerce")
    available_at = pd.to_datetime(
        enriched["annual_quality_available_at"], errors="coerce"
    )
    enriched["survival_gate"] = (
        available_at.le(signal_date).fillna(False)
        & enriched["financial_age_days"].between(
            0, gate_config.maximum_financial_age_days, inclusive="both"
        )
        & enriched["annual_history_years"].ge(
            gate_config.minimum_annual_history_years
        )
        & enriched["profit_positive_share_5y"].ge(
            gate_config.minimum_profit_positive_share
        )
        & enriched["cfo_positive_share_5y"].ge(
            gate_config.minimum_cfo_positive_share
        )
        & enriched["income_n_income_attr_p"].gt(0.0)
        & enriched["cashflow_n_cashflow_act"].gt(0.0)
    ).fillna(False)
    next_signal_id = int(pd.to_numeric(signals["signal_id"], errors="coerce").max()) + 1
    signal_parts = [signals]
    diagnostic_parts = [diagnostics]
    manifest: dict[str, Any] = {
        "annual_events": int(len(annual_events)),
        "gate_config": gate_config.to_dict(),
        "rules": {},
    }
    for source_rule, target_rule in source_to_target.items():
        source_enriched = enriched.loc[enriched["rule"].eq(source_rule)].copy()
        passed = source_enriched.loc[source_enriched["survival_gate"]].copy()
        if not passed.empty:
            passed["rule"] = target_rule
            passed["signal_id"] = np.arange(
                next_signal_id,
                next_signal_id + len(passed),
                dtype=np.int64,
            )
            next_signal_id += len(passed)
            signal_parts.append(passed[signals.columns])
        new_ids_by_anchor = (
            passed.set_index("anchor_id")["signal_id"].to_dict()
            if not passed.empty
            else {}
        )
        source_diagnostics = diagnostics.loc[diagnostics["rule"].eq(source_rule)].copy()
        source_diagnostics["rule"] = target_rule
        originally_confirmed = source_diagnostics["confirmation_status"].eq("confirmed")
        survival_passed = source_diagnostics["anchor_id"].isin(new_ids_by_anchor)
        source_diagnostics.loc[
            originally_confirmed & ~survival_passed, "confirmation_status"
        ] = "rejected_survival"
        source_diagnostics["signal_id"] = source_diagnostics["anchor_id"].map(
            new_ids_by_anchor
        ).astype("Int64")
        diagnostic_parts.append(source_diagnostics)
        manifest["rules"][target_rule] = {
            "source_confirmed": int(len(source_enriched)),
            "financial_coverage": int(source_enriched["financial_coverage"].sum()),
            "survival_passed": int(len(passed)),
        }
    return (
        pd.concat(signal_parts, ignore_index=True, sort=False),
        pd.concat(diagnostic_parts, ignore_index=True, sort=False),
        manifest,
    )


def _period_for_dates(values: pd.Series) -> pd.Series:
    dates = pd.to_datetime(values, errors="coerce")
    result = pd.Series(pd.NA, index=values.index, dtype="string")
    for period, (start, end) in EVENT_PERIODS.items():
        result.loc[dates.between(start, end, inclusive="both")] = period
    return result


def _decision(metrics: pd.DataFrame) -> dict[str, Any]:
    primary = metrics.loc[metrics["horizon"].eq(SELECTION_HORIZON)].copy()
    checks: dict[str, dict[str, bool]] = {}
    scores: list[tuple[tuple[float, float, float, int], str]] = []
    for rule in CONFIRMATION_RULES:
        selected = primary.loc[primary["rule"].eq(rule)].set_index("period").reindex(
            SELECTION_PERIODS
        )
        complete = not selected.isna().all(axis=1).any()
        selection_checks = {
            "periods_complete": bool(complete),
            "minimum_development_60_validation_100_events": bool(
                complete
                and selected.loc[
                    "development_2013_2016", "completed_events"
                ]
                >= 60
                and selected.loc[
                    "validation_2017_2020", "completed_events"
                ]
                >= 100
            ),
            "minimum_24_signal_dates_each": bool(
                complete and selected["signal_dates"].ge(24).all()
            ),
            "win_rate_at_least_55pct_each": bool(
                complete and selected["win_rate"].ge(0.55).all()
            ),
            "positive_median_each": bool(
                complete and selected["median_net_return"].gt(0.0).all()
            ),
            "profit_factor_at_least_1_50_each": bool(
                complete and selected["profit_factor"].ge(1.50).all()
            ),
            "excess_win_rate_at_least_50pct_each": bool(
                complete and selected["excess_win_rate"].ge(0.50).all()
            ),
            "positive_date_cluster_ci95_lower_each": bool(
                complete and selected["date_cluster_ci95_low"].gt(0.0).all()
            ),
        }
        diagnostic = primary.loc[
            primary["rule"].eq(rule)
            & primary["period"].eq("seen_diagnostic_2021_2024")
        ]
        diagnostic_complete = len(diagnostic) == 1
        diagnostic_checks = {
            "seen_win_rate_at_least_50pct": bool(
                diagnostic_complete and diagnostic.iloc[0]["win_rate"] >= 0.50
            ),
            "seen_positive_median": bool(
                diagnostic_complete and diagnostic.iloc[0]["median_net_return"] > 0.0
            ),
            "seen_profit_factor_at_least_1_20": bool(
                diagnostic_complete and diagnostic.iloc[0]["profit_factor"] >= 1.20
            ),
        }
        rule_checks = {**selection_checks, **diagnostic_checks}
        rule_checks["selection_passed"] = all(selection_checks.values())
        rule_checks["seen_diagnostic_passed"] = all(diagnostic_checks.values())
        rule_checks["all_passed_before_neighborhood"] = bool(
            rule_checks["selection_passed"] and rule_checks["seen_diagnostic_passed"]
        )
        checks[rule] = rule_checks
        if complete:
            score = (
                float(selected["win_rate"].min()),
                float(selected["profit_factor"].min()),
                float(selected["median_net_return"].min()),
                int(selected["completed_events"].sum()),
            )
            scores.append((score, rule))
    sample_usable = [
        (score, rule)
        for score, rule in scores
        if checks[rule]["minimum_development_60_validation_100_events"]
        and checks[rule]["minimum_24_signal_dates_each"]
    ]
    pre_neighborhood = [
        (score, rule)
        for score, rule in scores
        if checks[rule]["all_passed_before_neighborhood"]
    ]
    candidate = max(pre_neighborhood)[1] if pre_neighborhood else None
    return {
        "selection_periods": list(SELECTION_PERIODS),
        "selection_horizon_sessions": SELECTION_HORIZON,
        "seen_diagnostic_excluded_from_selection": True,
        "checks": checks,
        "best_sample_usable_rule": max(sample_usable)[1] if sample_usable else None,
        "pre_neighborhood_candidate": candidate,
        "selected_rule": None,
        "selection_status": (
            "candidate_requires_15_20_30_session_neighborhood_validation"
            if candidate
            else "no_reasonable_signal_structure"
        ),
        "staged_portfolio_decision": "do_not_build_until_neighborhood_passes",
    }


def _yearly_metrics(
    events: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    lookup = diagnostics.loc[
        diagnostics["confirmation_status"].eq("confirmed")
        & diagnostics["signal_id"].notna(),
        ["signal_id", "anchor_date"],
    ].drop_duplicates("signal_id")
    frame = events.merge(lookup, on="signal_id", how="left", validate="many_to_one")
    frame = frame.loc[
        frame["entry_status"].eq("accepted")
        & frame["outcome_completed"].fillna(False)
    ].copy()
    frame["anchor_year"] = pd.to_datetime(frame["anchor_date"]).dt.year
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(
        ["rule", "horizon", "anchor_year"], observed=True, sort=True
    ):
        returns = pd.to_numeric(group["net_return"], errors="coerce").dropna()
        losses = float(-returns.loc[returns <= 0.0].sum())
        rows.append(
            {
                "rule": keys[0],
                "horizon": int(keys[1]),
                "anchor_year": int(keys[2]),
                "events": int(len(returns)),
                "win_rate": float(returns.gt(0.0).mean()),
                "median_net_return": float(returns.median()),
                "profit_factor": (
                    float(returns.loc[returns > 0.0].sum() / losses)
                    if losses > 0.0
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _case_catalog(
    signals: pd.DataFrame,
    diagnostics: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    signal_columns = [
        "signal_id",
        "anchor_id",
        "anchor_close",
        "confirmation_close",
        "confirmation_drawdown_from_anchor_peak",
    ]
    signal_lookup = signals[signal_columns].drop_duplicates("signal_id")
    diagnostic_columns = [
        "signal_id",
        "anchor_id",
        "anchor_date",
        "confirmation_date",
        "confirmation_wait_sessions",
        "waiting_path_drawdown",
        "confirmation_status",
    ]
    main_events = events.loc[
        events["horizon"].eq(SELECTION_HORIZON)
        & events["entry_status"].eq("accepted")
        & events["outcome_completed"].fillna(False)
    ].copy()
    confirmed_diagnostics = diagnostics.loc[
        diagnostics["signal_id"].notna(), diagnostic_columns
    ].drop_duplicates("signal_id")
    main_events = main_events.merge(
        confirmed_diagnostics,
        on="signal_id",
        how="left",
        validate="many_to_one",
    ).merge(signal_lookup, on=["signal_id", "anchor_id"], how="left")
    rows: list[dict[str, Any]] = []

    def append_case(row: pd.Series, category: str) -> None:
        rows.append(
            {
                "category": category,
                "rule": row["rule"],
                "ts_code": row["ts_code"],
                "anchor_id": row["anchor_id"],
                "anchor_date": row["anchor_date"],
                "confirmation_date": row["signal_date"],
                "confirmation_wait_sessions": row["confirmation_wait_sessions"],
                "waiting_path_drawdown": row["waiting_path_drawdown"],
                "anchor_close": row["anchor_close"],
                "confirmation_close": row["confirmation_close"],
                "confirmation_drawdown_from_anchor_peak": row[
                    "confirmation_drawdown_from_anchor_peak"
                ],
                "net_return": row["net_return"],
                "excess_net_return": row["excess_net_return"],
                "mae": row["mae"],
                "mfe": row["mfe"],
                "exit_reason": row["exit_reason"],
            }
        )

    for _, row in main_events.iterrows():
        append_case(row, "confirmed_winner" if row["net_return"] > 0.0 else "confirmed_loser")
        if row["mae"] <= -0.20:
            append_case(row, "confirmed_severe_adverse")
        if row["confirmation_wait_sessions"] >= 75:
            append_case(row, "late_confirmation")

    direct = main_events.loc[main_events["rule"].eq("anchor_direct")].set_index(
        "anchor_id"
    )
    expired = diagnostics.loc[
        diagnostics["confirmation_status"].isin(["expired", "rejected_survival"])
        & diagnostics["rule"].ne("anchor_direct")
    ]
    direct_signals = signals.loc[signals["rule"].eq("anchor_direct")].set_index(
        "anchor_id"
    )
    for _, expired_row in expired.iterrows():
        anchor_id = int(expired_row["anchor_id"])
        if anchor_id not in direct.index:
            continue
        outcome = direct.loc[anchor_id]
        if isinstance(outcome, pd.DataFrame):
            outcome = outcome.iloc[0]
        if outcome["net_return"] >= 0.20:
            category = "expired_then_rebounded"
        elif outcome["net_return"] <= -0.20:
            category = "expired_avoided_loss"
        else:
            continue
        direct_signal = direct_signals.loc[anchor_id]
        if isinstance(direct_signal, pd.DataFrame):
            direct_signal = direct_signal.iloc[0]
        rows.append(
            {
                "category": category,
                "rule": expired_row["rule"],
                "ts_code": expired_row["ts_code"],
                "anchor_id": anchor_id,
                "anchor_date": expired_row["anchor_date"],
                "confirmation_date": pd.NaT,
                "confirmation_wait_sessions": np.nan,
                "waiting_path_drawdown": np.nan,
                "anchor_close": direct_signal["anchor_close"],
                "confirmation_close": np.nan,
                "confirmation_drawdown_from_anchor_peak": np.nan,
                "net_return": outcome["net_return"],
                "excess_net_return": outcome["excess_net_return"],
                "mae": outcome["mae"],
                "mfe": outcome["mfe"],
                "exit_reason": outcome["exit_reason"],
            }
        )
    return pd.DataFrame(rows)


def _case_summary(cases: pd.DataFrame) -> pd.DataFrame:
    if cases.empty:
        return pd.DataFrame(
            columns=["rule", "category", "cases", "median_net_return", "median_mae"]
        )
    return (
        cases.groupby(["rule", "category"], as_index=False, observed=True)
        .agg(
            cases=("anchor_id", "count"),
            median_net_return=("net_return", "median"),
            median_mae=("mae", "median"),
            median_wait_sessions=("confirmation_wait_sessions", "median"),
        )
        .sort_values(["rule", "category"])
    )


def _percent(value: object) -> str:
    number = pd.to_numeric(value, errors="coerce")
    return "—" if pd.isna(number) else f"{float(number):.2%}"


def _number(value: object) -> str:
    number = pd.to_numeric(value, errors="coerce")
    return "—" if pd.isna(number) else f"{float(number):.2f}"


def _write_report(
    path: Path,
    metrics: pd.DataFrame,
    yearly: pd.DataFrame,
    cases: pd.DataFrame,
    decision: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    primary = metrics.loc[metrics["horizon"].eq(SELECTION_HORIZON)].copy()
    display_columns = [
        "period",
        "rule",
        "anchors",
        "confirmed_anchors",
        "confirmation_rate",
        "median_confirmation_wait_sessions",
        "median_waiting_path_drawdown",
        "completed_events",
        "signal_dates",
        "win_rate",
        "median_net_return",
        "profit_factor",
        "excess_win_rate",
        "mean_mae",
        "date_equal_mean_net_return",
        "date_cluster_ci95_low",
        "date_cluster_ci95_high",
    ]
    display = primary[display_columns].copy()
    for column in (
        "confirmation_rate",
        "median_waiting_path_drawdown",
        "win_rate",
        "median_net_return",
        "excess_win_rate",
        "mean_mae",
        "date_equal_mean_net_return",
        "date_cluster_ci95_low",
        "date_cluster_ci95_high",
    ):
        display[column] = display[column].map(_percent)
    display["profit_factor"] = display["profit_factor"].map(_number)
    best = decision.get("best_sample_usable_rule")
    stability = yearly.loc[
        yearly["rule"].eq(best)
        & yearly["horizon"].eq(SELECTION_HORIZON)
        & yearly["anchor_year"].between(2013, 2024)
    ].copy()
    if not stability.empty:
        stability["win_rate"] = stability["win_rate"].map(_percent)
        stability["median_net_return"] = stability["median_net_return"].map(_percent)
        stability["profit_factor"] = stability["profit_factor"].map(_number)
    case_display = cases.copy()
    for column in ("median_net_return", "median_mae"):
        if column in case_display:
            case_display[column] = case_display[column].map(_percent)
    conclusion = (
        f"`{decision['pre_neighborhood_candidate']}` 通过主阶段和已见诊断，等待15/20/30日邻域验证。"
        if decision.get("pre_neighborhood_candidate")
        else "没有确认结构通过冻结门槛，必须依据成败案例提出新的可证伪机制后再迭代。"
    )
    path.write_text(
        f"""# 月线低位到右侧确认状态机回测

生成时间：{datetime.now().isoformat(timespec='seconds')}

## 结论

{conclusion}

- 月线低9只定义低位锚；确认规则最多等待126个市场交易日。
- 开发/验证中达到最低样本量的综合第一：`{best}`。
- 当前状态：`{decision['selection_status']}`；不修改线上策略。

## 12个月主检验

{display.to_markdown(index=False)}

## 综合第一的年度稳定性

{stability.to_markdown(index=False) if not stability.empty else '_无已完成事件_'}

## 案例类型摘要

{case_display.to_markdown(index=False) if not case_display.empty else '_无案例_'}

案例目录只用于解释失败机制和冻结下一轮假设，不用于回改本轮阈值。

## 数据审计

```json
{json.dumps(_json_value(metadata), ensure_ascii=False, indent=2)}
```

本研究仅供研究与教育用途，不构成投资建议、收益承诺或交易指令。
""",
        encoding="utf-8",
    )


def main() -> None:
    args = _parse_args()
    cutoff = pd.Timestamp("2026-07-31")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    benchmark = _load_benchmark(args.benchmark, cutoff)
    calendar = pd.DatetimeIndex(sorted(benchmark["date"].dropna().unique()))
    config = MonthlyConfirmationConfig(
        minimum_sessions_since_new_low=args.minimum_sessions_since_new_low
    )
    benchmark_features = build_benchmark_confirmation_features(benchmark)
    daily, daily_metadata = _load_daily(
        args.feature_cache,
        args.supplemental_root,
        cutoff,
    )
    breadth_metadata: dict[str, Any] = {"enabled": bool(args.enable_breadth_rules)}
    if args.enable_breadth_rules:
        print("building same-date liquid-universe breadth", flush=True)
        breadth = build_market_breadth_features(daily, config)
        benchmark_features = benchmark_features.merge(
            breadth,
            on="date",
            how="left",
            validate="one_to_one",
        )
        breadth_metadata.update(
            {
                "rows": int(len(breadth)),
                "valid_rows": int(
                    breadth["breadth_median_return_20d"].notna().sum()
                ),
            }
        )
    weekly = pd.read_parquet(
        args.weekly_features,
        columns=["ts_code", "weekly_available_date", "weekly_j", "weekly_prev_j"],
    )
    monthly_signals = pd.read_parquet(args.monthly_signals)
    anchors = monthly_signals.loc[monthly_signals["rule"].eq("monthly_low9")].copy()
    print(
        f"generating confirmation state machine for {len(anchors):,} monthly anchors",
        flush=True,
    )
    signals, diagnostics = generate_monthly_confirmation_signals(
        daily,
        weekly,
        anchors,
        benchmark_features,
        calendar,
        config,
    )
    survival_metadata: dict[str, Any] = {
        "enabled": bool(args.enable_survival_ablation)
    }
    if args.enable_survival_ablation:
        if not args.enable_breadth_rules:
            raise ValueError("--enable-survival-ablation requires --enable-breadth-rules")
        print("attaching point-in-time annual survival gate", flush=True)
        signals, diagnostics, survival_details = _add_survival_ablation(
            signals,
            diagnostics,
            args.raw_dir,
        )
        survival_metadata.update(survival_details)
    print(f"confirmed signals including direct baseline: {len(signals):,}", flush=True)
    execution_config = MonthlyLowZoneConfig(
        round_trip_cost_bps=config.round_trip_cost_bps,
        horizons=config.horizons,
    )
    events = evaluate_monthly_low_zone_events(
        daily,
        signals,
        benchmark,
        calendar,
        execution_config,
    )
    metrics = summarize_confirmation_events(events, diagnostics)
    yearly = _yearly_metrics(events, diagnostics)
    signals.to_parquet(args.output_dir / "signals.parquet", index=False, compression="zstd")
    diagnostics.to_parquet(
        args.output_dir / "anchor_diagnostics.parquet", index=False, compression="zstd"
    )
    events.to_parquet(args.output_dir / "events.parquet", index=False, compression="zstd")
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    yearly.to_csv(args.output_dir / "yearly_metrics.csv", index=False)
    decision = _decision(metrics)
    cases = _case_catalog(signals, diagnostics, events)
    case_summary = _case_summary(cases)
    metadata = {
        "analysis_cutoff": cutoff,
        "config": config.to_dict(),
        "anchor_rule": "monthly_low9",
        "anchors": int(len(anchors)),
        "confirmed_signals_including_direct": int(len(signals)),
        "anchor_diagnostic_rows": int(len(diagnostics)),
        "event_rows": int(len(events)),
        "case_rows": int(len(cases)),
        "daily": daily_metadata,
        "breadth": breadth_metadata,
        "survival_ablation": survival_metadata,
        "benchmark": str(args.benchmark),
        "monthly_signals": str(args.monthly_signals),
        "weekly_features": str(args.weekly_features),
    }
    cases.to_parquet(args.output_dir / "case_catalog.parquet", index=False, compression="zstd")
    case_summary.to_csv(args.output_dir / "case_summary.csv", index=False)
    _write_json(args.output_dir / "decision.json", decision)
    _write_json(args.output_dir / "metadata.json", metadata)
    _write_report(
        args.output_dir / "report.md",
        metrics,
        yearly,
        case_summary,
        decision,
        metadata,
    )
    print(f"candidate: {decision['pre_neighborhood_candidate']}", flush=True)
    print(f"report: {args.output_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
