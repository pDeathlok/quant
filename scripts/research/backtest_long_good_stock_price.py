"""Point-in-time validation for the long-horizon good-stock/good-price screen.

The research question is deliberately narrower than portfolio management:

1. Identify good companies without using valuation or price trend.
2. Compare pre-registered PE/PB/PR historical-percentile price rules.
3. Measure forward 6/12/24-month returns from the next trading day's close.

All historical percentiles are trailing per-symbol percentiles. Financial data
is merged by ``ann_date <= signal_date`` by the shared long-strategy loader.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_tea_master_long import (
    PROJECT_ROOT,
    build_tea_scores,
    load_benchmark,
    prepare_data,
)


REPORT_DIR = PROJECT_ROOT / "reports/long_good_stock_price"
PRICE_SCORE_BAND_CONFIG_PATH = PROJECT_ROOT / "config/long_price_score_bands.json"
HORIZONS = {"6m": 126, "12m": 252, "24m": 504}
HISTORY_WINDOWS = [36, 60, 84]
MINIMUM_HISTORY_MONTHS = 24
PRICE_SCORE_BANDS = [
    {
        "key": "80_100",
        "label": "80–100",
        "name": "深度相对低估",
        "minimum": 80.0,
        "maximum": 100.0,
        "meaning": "估值处于自身历史低位，但需警惕预期下修和价值陷阱",
        "decision": "结构通过后才可推荐",
    },
    {
        "key": "60_80",
        "label": "60–<80",
        "name": "好价候选",
        "minimum": 60.0,
        "maximum": 80.0,
        "meaning": "相对历史偏便宜，达到当前价格分推荐线",
        "decision": "结构通过后可推荐",
    },
    {
        "key": "40_60",
        "label": "40–<60",
        "name": "中性价格",
        "minimum": 40.0,
        "maximum": 60.0,
        "meaning": "相对自身历史不便宜也不极端昂贵",
        "decision": "继续观察价格",
    },
    {
        "key": "20_40",
        "label": "20–<40",
        "name": "相对偏贵",
        "minimum": 20.0,
        "maximum": 40.0,
        "meaning": "估值处于自身历史偏高区域",
        "decision": "不建议新建仓",
    },
    {
        "key": "0_20",
        "label": "0–<20",
        "name": "显著相对高估",
        "minimum": 0.0,
        "maximum": 20.0,
        "meaning": "估值接近自身历史高位",
        "decision": "等待估值消化",
    },
]


@dataclass(frozen=True)
class PriceRule:
    name: str
    description: str


RULES = [
    PriceRule("all_good_stocks", "好股票基线，不增加价格过滤"),
    PriceRule("pe_pb_p40", "PE和PB均处于各自历史40%分位以下"),
    PriceRule("pr_p40", "类型自适应双PR综合分位处于历史40%以下"),
    PriceRule("triple_p50", "PE、PB、PR均处于自身历史50%分位以下"),
    PriceRule("triple_p40", "PE、PB、PR均处于自身历史40%分位以下"),
    PriceRule("composite_60", "PE/PB/PR历史价值综合分不低于60"),
    PriceRule("composite_65", "PE/PB/PR历史价值综合分不低于65"),
    PriceRule("composite_60_guard", "综合分不低于60，且长期价格结构未明显破坏"),
]


def rule_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    has_history = frame["valuation_history_points"].fillna(0) >= 24
    pe = frame["pe_hist_percentile"]
    pb = frame["pb_hist_percentile"]
    pr = frame["pr_hist_percentile"]
    composite = frame["historical_value_score"]
    trend_guard = (
        (frame["close"] >= frame["ma_120"] * 0.90)
        & (frame["ma_120_slope_20d"] >= -0.06)
    )
    return {
        "all_good_stocks": pd.Series(True, index=frame.index),
        "pe_pb_p40": has_history & pe.le(40) & pb.le(40),
        "pr_p40": has_history & pr.le(40),
        "triple_p50": has_history & pe.le(50) & pb.le(50) & pr.le(50),
        "triple_p40": has_history & pe.le(40) & pb.le(40) & pr.le(40),
        "composite_60": has_history & composite.ge(60),
        "composite_65": has_history & composite.ge(65),
        "composite_60_guard": has_history & composite.ge(60) & trend_guard,
    }


def attach_forward_returns(
    signals: pd.DataFrame,
    daily: pd.DataFrame,
    benchmark: pd.DataFrame,
    *,
    horizons: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Attach next-day returns on a shared market-session calendar.

    Stock suspensions must not silently extend a 6/12/24-month horizon. Entry
    uses the first stock close available on or after the next market session;
    exit uses the last stock close on or before the target market session.
    """
    selected_horizons = horizons or HORIZONS
    daily = daily[["date", "ts_code", "close"]].copy()
    daily["date"] = pd.to_datetime(daily["date"])
    daily["close"] = pd.to_numeric(daily["close"], errors="coerce")
    daily = daily.dropna(subset=["date", "ts_code", "close"]).sort_values(["ts_code", "date"])
    lookup = {
        str(code): (
            group["date"].to_numpy(dtype="datetime64[ns]"),
            group["close"].to_numpy(dtype=float),
        )
        for code, group in daily.groupby("ts_code", sort=False)
    }
    benchmark = benchmark[["date", "benchmark_equity"]].copy()
    benchmark["date"] = pd.to_datetime(benchmark["date"])
    benchmark["benchmark_equity"] = pd.to_numeric(
        benchmark["benchmark_equity"], errors="coerce"
    )
    benchmark = benchmark.dropna().drop_duplicates("date", keep="last").sort_values("date")
    market_dates = benchmark["date"].to_numpy(dtype="datetime64[ns]")
    market_values = benchmark["benchmark_equity"].to_numpy(dtype=float)
    if not len(market_dates):
        raise ValueError("CSI 300 benchmark calendar is required for strict horizon alignment")

    rows: list[dict[str, float | str | pd.Timestamp]] = []
    signal_keys = signals[["date", "ts_code"]].drop_duplicates()
    for item in signal_keys.itertuples(index=False):
        code = str(item.ts_code)
        series = lookup.get(code)
        if series is None:
            continue
        dates, closes = series
        signal_date = np.datetime64(pd.Timestamp(item.date), "ns")
        next_market_index = int(np.searchsorted(market_dates, signal_date, side="right"))
        if next_market_index >= len(market_dates):
            continue
        next_market_date = market_dates[next_market_index]
        entry_index = int(np.searchsorted(dates, next_market_date, side="left"))
        if entry_index >= len(dates) or not np.isfinite(closes[entry_index]) or closes[entry_index] <= 0:
            continue
        entry_price = float(closes[entry_index])
        entry_date = dates[entry_index]
        market_entry_index = int(np.searchsorted(market_dates, entry_date, side="left"))
        if market_entry_index >= len(market_dates):
            continue
        result: dict[str, float | str | pd.Timestamp] = {
            "date": pd.Timestamp(item.date),
            "ts_code": code,
            "entry_date": pd.Timestamp(entry_date),
            "entry_price": entry_price,
        }
        for label, sessions in selected_horizons.items():
            market_exit_index = market_entry_index + sessions
            if market_exit_index >= len(market_dates):
                result[f"return_{label}"] = np.nan
                result[f"mae_{label}"] = np.nan
                result[f"benchmark_return_{label}"] = np.nan
                result[f"excess_return_{label}"] = np.nan
                continue
            target_date = market_dates[market_exit_index]
            exit_index = int(np.searchsorted(dates, target_date, side="right")) - 1
            if exit_index <= entry_index:
                result[f"return_{label}"] = np.nan
                result[f"mae_{label}"] = np.nan
                result[f"benchmark_return_{label}"] = np.nan
                result[f"excess_return_{label}"] = np.nan
                continue
            path = closes[entry_index : exit_index + 1]
            stock_return = float(closes[exit_index] / entry_price - 1.0)
            benchmark_return = float(
                market_values[market_exit_index] / market_values[market_entry_index] - 1.0
            )
            result[f"return_{label}"] = stock_return
            result[f"mae_{label}"] = float(np.nanmin(path) / entry_price - 1.0)
            result[f"benchmark_return_{label}"] = benchmark_return
            result[f"excess_return_{label}"] = stock_return - benchmark_return
        rows.append(result)
    return signals.merge(pd.DataFrame(rows), on=["date", "ts_code"], how="left")


def period_label(date: pd.Timestamp) -> str:
    if date.year <= 2019:
        return "development_2013_2019"
    if date.year <= 2023:
        return "validation_2020_2023"
    return "test_2024_plus"


def summarize(signals: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for (window_months, rule), rule_frame in signals.groupby(
        ["history_window_months", "rule"], sort=False
    ):
        for period, period_frame in rule_frame.groupby("period", sort=False):
            monthly_counts = period_frame.groupby("date")["ts_code"].nunique()
            for horizon in HORIZONS:
                columns = [
                    "date",
                    f"return_{horizon}",
                    f"excess_return_{horizon}",
                    f"mae_{horizon}",
                ]
                observations = period_frame[columns].copy()
                for column in columns[1:]:
                    observations[column] = pd.to_numeric(observations[column], errors="coerce")
                observations = observations.dropna(subset=[f"return_{horizon}"])
                monthly = observations.groupby("date").agg(
                    portfolio_return=(f"return_{horizon}", "mean"),
                    portfolio_excess=(f"excess_return_{horizon}", "mean"),
                    portfolio_mae=(f"mae_{horizon}", "mean"),
                )
                if monthly.empty:
                    continue
                rows.append(
                    {
                        "rule": rule,
                        "history_window_months": int(window_months),
                        "period": period,
                        "horizon": horizon,
                        "signals": int(len(observations)),
                        "months": int(len(monthly)),
                        "avg_names_per_month": float(monthly_counts.mean()),
                        "mean_return": float(monthly["portfolio_return"].mean()),
                        "median_return": float(monthly["portfolio_return"].median()),
                        "positive_rate": float((monthly["portfolio_return"] > 0).mean()),
                        "p10_return": float(monthly["portfolio_return"].quantile(0.10)),
                        "mean_excess": float(monthly["portfolio_excess"].mean()),
                        "median_excess": float(monthly["portfolio_excess"].median()),
                        "mean_mae": float(monthly["portfolio_mae"].mean()),
                        "p10_mae": float(monthly["portfolio_mae"].quantile(0.10)),
                    }
                )
    summary = pd.DataFrame(rows)
    baseline = summary[summary["rule"] == "all_good_stocks"][
        ["history_window_months", "period", "horizon", "mean_return"]
    ].rename(columns={"mean_return": "baseline_mean_return"})
    summary = summary.merge(
        baseline,
        on=["history_window_months", "period", "horizon"],
        how="left",
    )
    summary["baseline_delta"] = summary["mean_return"] - summary["baseline_mean_return"]
    return summary


def summarize_price_score_bands(signals: pd.DataFrame) -> pd.DataFrame:
    """Summarize disjoint production-score bands without selecting on outcomes."""

    frame = signals[
        (signals["history_window_months"] == 84)
        & (signals["rule"] == "all_good_stocks")
    ].drop_duplicates(["date", "ts_code"]).copy()
    frame["historical_value_score"] = pd.to_numeric(
        frame["historical_value_score"], errors="coerce"
    )
    frame = frame.dropna(subset=["historical_value_score"])
    rows: list[dict[str, float | int | str]] = []
    periods = {
        "validation": frame["date"].dt.year.between(2020, 2023),
        "test": frame["date"].dt.year.ge(2024),
    }
    for band in PRICE_SCORE_BANDS:
        score_mask = frame["historical_value_score"].ge(band["minimum"])
        if band["maximum"] >= 100:
            score_mask &= frame["historical_value_score"].le(band["maximum"])
        else:
            score_mask &= frame["historical_value_score"].lt(band["maximum"])
        for period, period_mask in periods.items():
            observations = frame.loc[
                score_mask & period_mask,
                ["date", "ts_code", "return_12m", "excess_return_12m", "mae_12m"],
            ].copy()
            for column in ["return_12m", "excess_return_12m", "mae_12m"]:
                observations[column] = pd.to_numeric(observations[column], errors="coerce")
            observations = observations.dropna(subset=["return_12m"])
            monthly = observations.groupby("date").agg(
                portfolio_return=("return_12m", "mean"),
                portfolio_excess=("excess_return_12m", "mean"),
                portfolio_mae=("mae_12m", "mean"),
            )
            if monthly.empty:
                continue
            rows.append(
                {
                    "key": str(band["key"]),
                    "period": period,
                    "signals": int(len(observations)),
                    "periods": int(len(monthly)),
                    "mean_return": float(monthly["portfolio_return"].mean()),
                    "positive_rate": float((monthly["portfolio_return"] > 0).mean()),
                    "mean_excess": float(monthly["portfolio_excess"].mean()),
                    "mean_mae": float(monthly["portfolio_mae"].mean()),
                }
            )
    return pd.DataFrame(rows)


def price_score_band_payload(summary: pd.DataFrame) -> dict[str, object]:
    results = {
        (str(row.key), str(row.period)): {
            "signals": int(row.signals),
            "periods": int(row.periods),
            "mean_return": round(float(row.mean_return), 10),
            "positive_rate": round(float(row.positive_rate), 10),
            "mean_excess": round(float(row.mean_excess), 10),
            "mean_mae": round(float(row.mean_mae), 10),
        }
        for row in summary.itertuples(index=False)
    }
    bands = []
    for definition in PRICE_SCORE_BANDS:
        item = {
            key: value
            for key, value in definition.items()
            if key not in {"minimum", "maximum"}
        }
        item["validation"] = results.get((str(definition["key"]), "validation"), {})
        item["test"] = results.get((str(definition["key"]), "test"), {})
        bands.append(item)
    return {
        "schema_version": "long_price_score_bands_v1",
        "score_definition": "100 - (PE历史分位×30% + PB历史分位×25% + 类型自适应PR历史分位×45%)",
        "sampling": "monthly_last_trading_day",
        "history_window_months": 84,
        "minimum_history_months": 24,
        "execution": "next_trading_day_close",
        "horizon": "12m",
        "portfolio_aggregation": "每个信号月内好股票等权，再跨月统计",
        "validation_period": "2020-01-01/2023-12-31",
        "test_period": "2024-01-01/2025-06-30",
        "conclusion": "价格分只衡量相对自身历史的便宜程度，回测未呈现分数越高、未来收益越高的单调关系；60分仅是好价候选线，推荐仍需长期价格结构通过。",
        "bands": bands,
    }


def choose_validation_candidate(summary: pd.DataFrame) -> tuple[int, str]:
    # PE/PB-only and PR-only rules are diagnostic baselines. Production
    # candidates must jointly consider all three historical percentiles.
    joint_rules = {
        "triple_p50",
        "triple_p40",
        "composite_60",
        "composite_65",
        "composite_60_guard",
    }
    validation = summary[
        (summary["period"] == "validation_2020_2023")
        & (summary["horizon"] == "12m")
        & (summary["rule"].isin(joint_rules))
        & (summary["avg_names_per_month"] >= 3)
    ].copy()
    if validation.empty:
        return 60, "composite_60_guard"
    validation["selection_score"] = (
        validation["median_excess"] * 0.25
        + validation["mean_excess"] * 0.25
        + validation["baseline_delta"] * 0.25
        + validation["positive_rate"] * 0.10
        + validation["p10_return"] * 0.15
    )
    selected = validation.sort_values("selection_score", ascending=False).iloc[0]
    return int(selected["history_window_months"]), str(selected["rule"])


def percent(value: float) -> str:
    return "-" if pd.isna(value) else f"{value:.2%}"


def markdown_report(
    summary: pd.DataFrame,
    selected_window_months: int,
    selected_rule: str,
    scored: pd.DataFrame,
    coverage: dict,
) -> str:
    rule_description = next(item.description for item in RULES if item.name == selected_rule)
    lines = [
        "# 长线好股票 / 好价格历史验证",
        "",
        "## 结论",
        "",
        f"- 验证期预选：最近 {selected_window_months // 12} 年月末样本 + `{selected_rule}`（{rule_description}）。",
        "- 所有信号按月末收盘后生成，从下一交易日收盘价开始计算，避免同日信号收益前视。",
        "- 收益周期使用统一的沪深300交易日历；停牌不会被误算为延长持有期。",
        "- 结果先按每个信号月等权聚合，再跨月统计，并同时比较沪深300和好股票基线。",
        "- 比较最近3年、5年、7年月末采样窗口；至少24个月数据才启用。月采样比周采样更省计算且更稳健。",
        "- 规则选择只看2020–2023验证期；2024年至今作为样本外测试，不参与选规则。",
        "",
        "## PR口径",
        "",
        "```text",
        "roe_decimal = ROE / 100",
        "PR = PE / roe_decimal / 100 = PE / ROE",
        "PR_from_PB = PB / roe_decimal² / 100 = 100 × PB / ROE²",
        "```",
        "",
        "- 轻资产/利润驱动行业：PR-PE 70%，PR-PB 30%。",
        "- 重资产/资本驱动行业：PR-PE 30%，PR-PB 70%。",
        "- 其他行业：两种PR各50%；两个原始值和历史分位都完整保留。",
        "",
        "## 分阶段结果",
        "",
        "| 窗口 | 规则 | 阶段 | 周期 | 样本 | 月均标的 | 月度等权平均 | 中位收益 | 胜率 | 对沪深300超额 | 较好股基线 | P10收益 |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{row.history_window_months // 12}年",
                    str(row.rule),
                    str(row.period),
                    str(row.horizon),
                    str(row.signals),
                    f"{row.avg_names_per_month:.1f}",
                    percent(row.mean_return),
                    percent(row.median_return),
                    percent(row.positive_rate),
                    percent(row.mean_excess),
                    percent(row.baseline_delta),
                    percent(row.p10_return),
                ]
            )
            + " |"
        )
    formula_gap = pd.to_numeric(scored["pr_formula_gap"], errors="coerce").dropna()
    lines.extend(
        [
            "",
            "## 数据审计",
            "",
            f"- daily_basic起止：{coverage.get('first_trade_date')} 至 {coverage.get('last_trade_date')}。",
            f"- 月度截面数：{coverage.get('monthly_rebalance_dates')}。",
            f"- PR双公式相对差异中位数：{percent(float(formula_gap.median()) if not formula_gap.empty else np.nan)}。",
            "- 双公式差异来自PE/PB的TTM与时点ROE口径不完全同频；因此按行业估值类型加权，同时保留两者供审计。",
            "",
            "## 限制",
            "",
            "- 当前是历史量化筛选验证，不等同单只公司的完整基本面估值。",
            "- 6/12/24个月信号仍存在跨月重叠，因此规则选择基于月度等权组合，而不是把个股信号视为独立交易次数。",
            "- 股票基础表可能存在退市样本覆盖不足；这会造成幸存者偏差，结果不能被解释为可直接复刻的组合收益。",
            "- 金融行业暂不以普通企业资产负债率做硬过滤，后续应补资本充足率、净息差和资产质量专用模型。",
        ]
    )
    return "\n".join(lines) + "\n"


def run(start: str, end: str | None) -> dict[str, object]:
    merged, daily_returns, _, coverage = prepare_data(start, end)
    benchmark = load_benchmark(pd.to_datetime(start), pd.to_datetime(end) if end else None)
    signal_frames: list[pd.DataFrame] = []
    keep_columns = [
        "date",
        "ts_code",
        "name",
        "industry",
        "good_stock_score",
        "historical_value_score",
        "pe_hist_percentile",
        "pb_hist_percentile",
        "pr_hist_percentile",
        "pr_pe_hist_percentile",
        "pr_pb_hist_percentile",
        "pr",
        "pr_pe",
        "pr_pb",
        "pr_formula_gap",
        "valuation_profile",
        "pr_pe_weight",
        "pr_pb_weight",
        "valuation_history_points",
    ]
    scored_by_window: dict[int, pd.DataFrame] = {}
    for window_months in HISTORY_WINDOWS:
        scored = build_tea_scores(
            merged,
            valuation_window_months=window_months,
            valuation_minimum_months=MINIMUM_HISTORY_MONTHS,
        )
        scored_by_window[window_months] = scored
        good = scored[scored["is_good_stock"].fillna(False)].copy()
        masks = rule_masks(good)
        for rule in RULES:
            selected = good.loc[masks[rule.name], keep_columns].copy()
            selected["rule"] = rule.name
            selected["history_window_months"] = window_months
            signal_frames.append(selected)
    signals = pd.concat(signal_frames, ignore_index=True)
    signals = attach_forward_returns(signals, daily_returns, benchmark)
    signals["period"] = signals["date"].map(period_label)
    summary = summarize(signals)
    price_score_bands = summarize_price_score_bands(signals)
    selected_window_months, selected_rule = choose_validation_candidate(summary)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(REPORT_DIR / "summary.csv", index=False)
    price_score_bands.to_csv(REPORT_DIR / "price_score_bands.csv", index=False)
    signals.to_parquet(REPORT_DIR / "signals.parquet", index=False)
    PRICE_SCORE_BAND_CONFIG_PATH.write_text(
        json.dumps(price_score_band_payload(price_score_bands), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    selected_scored = scored_by_window[selected_window_months]
    report = markdown_report(
        summary,
        selected_window_months,
        selected_rule,
        selected_scored,
        coverage,
    )
    (REPORT_DIR / "report.md").write_text(report, encoding="utf-8")
    selection = {
        "selected_rule": selected_rule,
        "selected_history_window_months": selected_window_months,
        "selected_rule_description": next(item.description for item in RULES if item.name == selected_rule),
        "selection_period": "2020-2023",
        "test_period": "2024+",
        "history_window_months": selected_window_months,
        "tested_history_windows_months": HISTORY_WINDOWS,
        "sampling": "monthly_last_trading_day",
        "minimum_history_months": MINIMUM_HISTORY_MONTHS,
        "execution": "next_trading_day_close",
        "formula": {
            "pr_from_pe": "PE / ROE_percent",
            "pr_from_pb": "100 * PB / ROE_percent^2",
        },
        "type_adaptive_weights": {
            "earnings_based": {"pr_from_pe": 0.70, "pr_from_pb": 0.30},
            "asset_based": {"pr_from_pe": 0.30, "pr_from_pb": 0.70},
            "balanced": {"pr_from_pe": 0.50, "pr_from_pb": 0.50},
        },
    }
    (REPORT_DIR / "selected_rule.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "selected_rule": selected_rule,
        "selected_history_window_months": selected_window_months,
        "summary_rows": len(summary),
        "signals": len(signals),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="20130101")
    parser.add_argument("--end")
    args = parser.parse_args()
    print(json.dumps(run(args.start, args.end), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
