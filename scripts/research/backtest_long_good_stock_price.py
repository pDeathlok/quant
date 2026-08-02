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

from backtest_tea_master_long import PROJECT_ROOT, build_tea_scores, prepare_data


REPORT_DIR = PROJECT_ROOT / "reports/long_good_stock_price"
HORIZONS = {"6m": 126, "12m": 252, "24m": 504}


@dataclass(frozen=True)
class PriceRule:
    name: str
    description: str


RULES = [
    PriceRule("all_good_stocks", "好股票基线，不增加价格过滤"),
    PriceRule("pe_pb_p40", "PE和PB均处于各自历史40%分位以下"),
    PriceRule("pr_p40", "PR处于自身历史40%分位以下"),
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


def attach_forward_returns(signals: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
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

    rows: list[dict[str, float | str | pd.Timestamp]] = []
    for item in signals.itertuples(index=False):
        code = str(item.ts_code)
        series = lookup.get(code)
        if series is None:
            continue
        dates, closes = series
        signal_date = np.datetime64(pd.Timestamp(item.date), "ns")
        entry_index = int(np.searchsorted(dates, signal_date, side="right"))
        if entry_index >= len(dates) or not np.isfinite(closes[entry_index]) or closes[entry_index] <= 0:
            continue
        entry_price = float(closes[entry_index])
        result: dict[str, float | str | pd.Timestamp] = {
            "date": pd.Timestamp(item.date),
            "ts_code": code,
            "entry_date": pd.Timestamp(dates[entry_index]),
            "entry_price": entry_price,
        }
        for label, sessions in HORIZONS.items():
            exit_index = entry_index + sessions
            if exit_index >= len(dates):
                result[f"return_{label}"] = np.nan
                result[f"mae_{label}"] = np.nan
                continue
            path = closes[entry_index : exit_index + 1]
            result[f"return_{label}"] = float(closes[exit_index] / entry_price - 1.0)
            result[f"mae_{label}"] = float(np.nanmin(path) / entry_price - 1.0)
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
    for rule, rule_frame in signals.groupby("rule", sort=False):
        for period, period_frame in rule_frame.groupby("period", sort=False):
            monthly_counts = period_frame.groupby("date")["ts_code"].nunique()
            for horizon in HORIZONS:
                returns = pd.to_numeric(period_frame[f"return_{horizon}"], errors="coerce").dropna()
                adverse = pd.to_numeric(period_frame[f"mae_{horizon}"], errors="coerce").dropna()
                if returns.empty:
                    continue
                rows.append(
                    {
                        "rule": rule,
                        "period": period,
                        "horizon": horizon,
                        "signals": int(len(returns)),
                        "months": int(period_frame.loc[returns.index, "date"].nunique()),
                        "avg_names_per_month": float(monthly_counts.mean()),
                        "mean_return": float(returns.mean()),
                        "median_return": float(returns.median()),
                        "positive_rate": float((returns > 0).mean()),
                        "p10_return": float(returns.quantile(0.10)),
                        "mean_mae": float(adverse.mean()) if not adverse.empty else np.nan,
                        "p10_mae": float(adverse.quantile(0.10)) if not adverse.empty else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def choose_validation_rule(summary: pd.DataFrame) -> str:
    validation = summary[
        (summary["period"] == "validation_2020_2023")
        & (summary["horizon"] == "12m")
        & (summary["rule"] != "all_good_stocks")
        & (summary["avg_names_per_month"] >= 3)
    ].copy()
    if validation.empty:
        return "composite_60_guard"
    validation["selection_score"] = (
        validation["median_return"] * 0.45
        + validation["mean_return"] * 0.25
        + validation["positive_rate"] * 0.10
        + validation["p10_return"] * 0.20
    )
    return str(validation.sort_values("selection_score", ascending=False).iloc[0]["rule"])


def percent(value: float) -> str:
    return "-" if pd.isna(value) else f"{value:.2%}"


def markdown_report(
    summary: pd.DataFrame,
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
        f"- 验证期预选规则：`{selected_rule}`（{rule_description}）。",
        "- 所有信号按月末收盘后生成，从下一交易日收盘价开始计算，避免同日信号收益前视。",
        "- PE、PB、PR分位均为单只股票自身最多60个月滚动历史分位，至少24个月数据才启用。",
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
        "## 分阶段结果",
        "",
        "| 规则 | 阶段 | 周期 | 样本 | 月均标的 | 平均收益 | 中位收益 | 胜率 | P10收益 | 平均最大不利波动 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.rule),
                    str(row.period),
                    str(row.horizon),
                    str(row.signals),
                    f"{row.avg_names_per_month:.1f}",
                    percent(row.mean_return),
                    percent(row.median_return),
                    percent(row.positive_rate),
                    percent(row.p10_return),
                    percent(row.mean_mae),
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
            "- 双公式差异来自PE/PB的TTM与时点ROE口径不完全同频；主指标使用PE公式，PB公式用于审计。",
            "",
            "## 限制",
            "",
            "- 当前是历史量化筛选验证，不等同单只公司的完整基本面估值。",
            "- 6/12/24个月信号存在跨月重叠，结果同时报告样本数和月份数，不把样本量当独立交易次数。",
            "- 金融行业暂不以普通企业资产负债率做硬过滤，后续应补资本充足率、净息差和资产质量专用模型。",
        ]
    )
    return "\n".join(lines) + "\n"


def run(start: str, end: str | None) -> dict[str, object]:
    merged, daily_returns, _, coverage = prepare_data(start, end)
    scored = build_tea_scores(merged)
    good = scored[scored["is_good_stock"].fillna(False)].copy()
    masks = rule_masks(good)
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
        "pr",
        "pr_pb",
        "pr_formula_gap",
        "valuation_history_points",
    ]
    for rule in RULES:
        selected = good.loc[masks[rule.name], keep_columns].copy()
        selected["rule"] = rule.name
        signal_frames.append(selected)
    signals = pd.concat(signal_frames, ignore_index=True)
    signals = attach_forward_returns(signals, daily_returns)
    signals["period"] = signals["date"].map(period_label)
    summary = summarize(signals)
    selected_rule = choose_validation_rule(summary)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(REPORT_DIR / "summary.csv", index=False)
    signals.to_parquet(REPORT_DIR / "signals.parquet", index=False)
    report = markdown_report(summary, selected_rule, scored, coverage)
    (REPORT_DIR / "report.md").write_text(report, encoding="utf-8")
    selection = {
        "selected_rule": selected_rule,
        "selected_rule_description": next(item.description for item in RULES if item.name == selected_rule),
        "selection_period": "2020-2023",
        "test_period": "2024+",
        "history_window_months": 60,
        "minimum_history_months": 24,
        "execution": "next_trading_day_close",
        "formula": {
            "pr": "PE / ROE_percent",
            "pr_from_pb": "100 * PB / ROE_percent^2",
        },
    }
    (REPORT_DIR / "selected_rule.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"selected_rule": selected_rule, "summary_rows": len(summary), "signals": len(signals)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="20130101")
    parser.add_argument("--end")
    args = parser.parse_args()
    print(json.dumps(run(args.start, args.end), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
