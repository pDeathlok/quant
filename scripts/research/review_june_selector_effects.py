#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Review June 2026 selector outcomes by score, strategy, entry, and exit."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data/research/june_2026_selector_review"
DEFAULT_DAILY_DIR = PROJECT_ROOT / "data/raw/daily"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports/selector_review"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review June selector effects.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--daily-dir", type=Path, default=DEFAULT_DAILY_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--start-date", default="2026-06-01")
    parser.add_argument("--end-date", default="2026-06-30")
    return parser.parse_args()


def daily_file(daily_dir: Path, symbol: str) -> Path | None:
    plain = symbol.replace(".SH", "").replace(".SZ", "").replace(".BJ", "")
    for candidate in (daily_dir / f"{symbol}.parquet", daily_dir / f"{plain}.parquet"):
        if candidate.exists():
            return candidate
    return None


def add_t7_labels(stock: pd.DataFrame, daily_dir: Path) -> pd.DataFrame:
    frames = []
    for symbol, part in stock.groupby("symbol", sort=False):
        path = daily_file(daily_dir, str(symbol))
        if path is None:
            frame = part[["date", "symbol"]].copy()
            frame["future_return_t7_pct"] = np.nan
            frame["future_max_high_t7_pct"] = np.nan
            frame["future_max_drawdown_t7_pct"] = np.nan
            frames.append(frame)
            continue
        daily = pd.read_parquet(path)
        if "trade_date" in daily.columns:
            daily["date"] = pd.to_datetime(daily["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
        else:
            daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
        daily = daily.dropna(subset=["date", "close", "high", "low"]).sort_values("date").reset_index(drop=True)
        base = daily["close"].replace(0, np.nan)
        label = daily[["date"]].copy()
        label["symbol"] = symbol
        label["future_return_t7_pct"] = (daily["close"].shift(-7) / base - 1) * 100
        future_high = pd.concat([daily["high"].shift(-day) for day in range(1, 8)], axis=1).max(axis=1)
        future_low = pd.concat([daily["low"].shift(-day) for day in range(1, 8)], axis=1).min(axis=1)
        label["future_max_high_t7_pct"] = (future_high / base - 1) * 100
        label["future_max_drawdown_t7_pct"] = (future_low / base - 1) * 100
        wanted = part[["date", "symbol"]].copy()
        frames.append(wanted.merge(label, on=["date", "symbol"], how="left"))
    labels = pd.concat(frames, ignore_index=True)
    return stock.merge(labels, on=["date", "symbol"], how="left")


def entry_bucket(buy_plan: str) -> str:
    text = str(buy_plan or "")
    if "0% <= T+1 开盘涨幅 <= 1%" in text or "T+1 开盘 0%-1%" in text:
        return "T+1 开盘 0%-1%"
    if "0% <= T+1 开盘涨幅 <= 2%" in text or "T+1 开盘 0%-2%" in text:
        return "T+1 开盘 0%-2%"
    if "T+1 开盘涨幅 >= 0%" in text:
        return "T+1 开盘 >=0%"
    if "T+1 开盘 -2%-1%" in text:
        return "T+1 开盘 -2%-1%"
    if "T+1 开盘涨幅 <=1%" in text or "T+1 开盘涨幅<=1%" in text:
        return "T+1 开盘 <=1%"
    if "T+1 开盘涨幅<=2%" in text or "T+1 开盘涨幅 <=2%" in text:
        return "T+1 开盘 <=2%"
    if "T+1 开盘不超过 1%" in text:
        return "T+1 开盘 <=1%"
    if "T+1 不低开" in text:
        return "T+1 开盘 >=0%"
    if "不限制 T+1 开盘涨跌幅" in text:
        return "T+1 开盘不限"
    return "其他/未归类"


def exit_bucket(sell_plan: str) -> str:
    text = str(sell_plan or "")
    if "盈利达到 10.0%" in text and "亏损达到 1.5%" in text and "T+7" in text:
        return "TP10/SL1.5/T7"
    if "止盈 10%" in text and "最长持有到 T+7" in text:
        return "TP10/SL1.5/T7"
    if "盈利达到 4.0%" in text and "亏损达到 1.5%" in text and "T+5" in text:
        return "TP4/SL1.5/T5"
    if "盈利达到 8.0%" in text and "亏损达到 1.5%" in text:
        return "TP8/SL1.5/T5"
    if "盈利达到 8.0%" in text and "亏损达到 1.0%" in text:
        return "TP8/SL1/T5"
    if "最多持有到 T+3" in text and "不设固定止盈止损" in text:
        return "无止盈止损/T3"
    if "最多持有到 T+5" in text and "不设固定止盈止损" in text:
        return "无止盈止损/T5"
    if "最多持有到 T+7" in text and "不设固定止盈止损" in text:
        return "无止盈止损/T7"
    return "其他/未归类"


def realized_proxy(row: pd.Series) -> float:
    bucket = row["exit_bucket"]
    if bucket == "无止盈止损/T3":
        return row["future_return_t3_pct"]
    if bucket in {"TP10/SL1.5/T7", "无止盈止损/T7"}:
        return row["future_return_t7_pct"]
    return row["future_return_t5_pct"]


def win_rate(series: pd.Series) -> float:
    valid = series.dropna()
    return float((valid > 0).mean()) if len(valid) else np.nan


def agg_table(df: pd.DataFrame, keys: list[str], min_n: int = 1) -> pd.DataFrame:
    work = df.copy()
    if "metrics_avg_return_pct" not in work.columns:
        work["metrics_avg_return_pct"] = np.nan
    grouped = (
        work.groupby(keys, dropna=False)
        .agg(
            n=("symbol", "size"),
            valid=("realized_proxy_pct", "count"),
            expected_avg_pct=("metrics_avg_return_pct", "mean"),
            actual_avg_pct=("realized_proxy_pct", "mean"),
            win_rate=("realized_proxy_pct", win_rate),
            t1_avg_pct=("future_return_t1_pct", "mean"),
            t3_avg_pct=("future_return_t3_pct", "mean"),
            t5_avg_pct=("future_return_t5_pct", "mean"),
            t7_avg_pct=("future_return_t7_pct", "mean"),
            high5_avg_pct=("future_max_high_t5_pct", "mean"),
            drawdown5_avg_pct=("future_max_drawdown_t5_pct", "mean"),
        )
        .reset_index()
    )
    return grouped[grouped["valid"] >= min_n].sort_values("actual_avg_pct", ascending=False)


def fmt_pct(value: float) -> str:
    if pd.isna(value):
        return "-"
    return f"{value:.2f}%"


def markdown_table(df: pd.DataFrame, columns: list[str], limit: int | None = None) -> str:
    view = df[columns].copy()
    if limit is not None:
        view = view.head(limit)
    for col in view.columns:
        if col in {"expected_avg_pct", "actual_avg_pct", "t1_avg_pct", "t3_avg_pct", "t5_avg_pct", "t7_avg_pct", "high5_avg_pct", "drawdown5_avg_pct"}:
            view[col] = view[col].map(fmt_pct)
        elif col == "win_rate":
            view[col] = view[col].map(lambda x: "-" if pd.isna(x) else f"{x:.1%}")
        elif col in {"n", "valid"}:
            view[col] = view[col].map(lambda x: f"{int(x)}")
    return view.to_markdown(index=False)


def score_bucket(score: float) -> str:
    if score >= 95:
        return "95-100"
    if score >= 85:
        return "85-95"
    if score >= 70:
        return "70-85"
    if score >= 50:
        return "50-70"
    return "<50"


def ordered_score_table(table: pd.DataFrame) -> pd.DataFrame:
    order = {"<50": 0, "50-70": 1, "70-85": 2, "85-95": 3, "95-100": 4}
    return table.assign(_order=table["stock_score_bucket"].map(order)).sort_values("_order").drop(columns="_order")


def main() -> None:
    args = parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)

    stock = pd.read_parquet(args.input_dir / "selector_stock_history_samples.parquet")
    signal = pd.read_parquet(args.input_dir / "selector_signal_history_samples.parquet")
    stock["date"] = pd.to_datetime(stock["date"])
    signal["date"] = pd.to_datetime(signal["date"])
    stock = stock[(stock["date"] >= args.start_date) & (stock["date"] <= args.end_date)].copy()
    signal = signal[(signal["date"] >= args.start_date) & (signal["date"] <= args.end_date)].copy()
    stock = add_t7_labels(stock, args.daily_dir)

    join_cols = [
        "date",
        "symbol",
        "selector_score",
        "rank",
        "future_return_t1_pct",
        "future_return_t3_pct",
        "future_return_t5_pct",
        "future_max_high_t5_pct",
        "future_max_drawdown_t5_pct",
        "future_return_t7_pct",
        "future_max_high_t7_pct",
        "future_max_drawdown_t7_pct",
    ]
    signal = signal.merge(stock[join_cols], on=["date", "symbol"], how="left")
    signal["entry_bucket"] = signal["buy_plan"].map(entry_bucket)
    signal["exit_bucket"] = signal["sell_plan"].map(exit_bucket)
    signal["realized_proxy_pct"] = signal.apply(realized_proxy, axis=1)
    signal["stock_score_bucket"] = signal["selector_score"].map(score_bucket)

    stock["stock_score_bucket"] = stock["selector_score"].map(score_bucket)
    stock["realized_proxy_pct"] = stock["future_return_t5_pct"]
    score_table = ordered_score_table(agg_table(stock, ["stock_score_bucket"]))
    rank_table = agg_table(stock.assign(rank_bucket=pd.cut(stock["rank"], [0, 10, 30, 100, 10_000], labels=["Top10", "Top11-30", "Top31-100", "100+"])), ["rank_bucket"])
    group_table = agg_table(signal, ["strategy_group"], min_n=10)
    strategy_table = agg_table(signal, ["strategy_group", "strategy_name"], min_n=50)
    entry_table = agg_table(signal, ["entry_bucket"], min_n=30)
    exit_table = agg_table(signal, ["exit_bucket"], min_n=30)
    entry_exit_table = agg_table(signal, ["entry_bucket", "exit_bucket"], min_n=30)

    for name, table in {
        "score_buckets": score_table,
        "rank_buckets": rank_table,
        "strategy_groups": group_table,
        "strategies": strategy_table,
        "entry_buckets": entry_table,
        "exit_buckets": exit_table,
        "entry_exit_buckets": entry_exit_table,
    }.items():
        table.to_csv(args.report_dir / f"june_2026_{name}.csv", index=False)

    valid_t5_by_date = stock.groupby(stock["date"].dt.strftime("%Y-%m-%d")).agg(n=("symbol", "size"), t5_valid=("future_return_t5_pct", "count"))
    summary = {
        "start_date": args.start_date,
        "end_date": args.end_date,
        "stock_rows": int(len(stock)),
        "signal_rows": int(len(signal)),
        "dates": int(stock["date"].nunique()),
        "symbols": int(stock["symbol"].nunique()),
        "t5_valid_rows": int(stock["future_return_t5_pct"].count()),
        "t7_valid_rows": int(stock["future_return_t7_pct"].count()),
    }
    (args.report_dir / "june_2026_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 2026 年 6 月策略统一复盘",
        "",
        "## 口径",
        "",
        f"- 样本范围：{args.start_date} 至 {args.end_date}，共 {summary['dates']} 个交易日、{summary['stock_rows']} 条股票级候选、{summary['signal_rows']} 条信号级候选。",
        f"- 本次使用仓库内 `{args.input_dir.relative_to(PROJECT_ROOT)}` 重建样本，并用 `{args.daily_dir.relative_to(PROJECT_ROOT)}` 本地日线补前瞻收益。",
        "- 买卖点复盘采用后验收益代理：T3 卖点看 T+3 收益，T5 卖点看 T+5 收益，B1 的 T7 卖点看 T+7 收益；未逐笔模拟 T+1 开盘区间是否真实成交，也未逐日重放止盈/止损触发顺序。",
        "- 因本地行情到 2026-07-03，6 月 29-30 日没有 T+5/T+7 完整标签；这些样本保留在 T+1/T+3 和冲高/回撤统计中，但不会计入对应卖点代理收益。",
        "",
        "## 总结判断",
        "",
        f"- 6 月股票级候选 T+5 平均收益为 {fmt_pct(stock['future_return_t5_pct'].mean())}，胜率 {win_rate(stock['future_return_t5_pct']):.1%}；T+5 平均最高冲高 {fmt_pct(stock['future_max_high_t5_pct'].mean())}，平均最大回撤 {fmt_pct(stock['future_max_drawdown_t5_pct'].mean())}。",
        "- 结论偏不符合预期：多数策略历史 OOT 均值为正，但 6 月实盘后验收益普遍转负，说明月内环境或候选拥挤度与历史样本不一致。",
        "- 分数效果不完全符合预期：85-95 分桶表现最好，但 95-100 分桶回落到低分区附近；分数仍有过滤价值，但 6 月不是单调越高越好。",
        "- 买点上，低吸型 `T+1 开盘 -2%-1%` 和 `T+1 开盘 <=1%` 好于追涨型 `0%-2%`；6 月追涨确认类更容易冲高回落。",
        "- 卖点上，短持有/不硬扛的组合相对最好，T5/T7 持有普遍被后半段回撤吞掉；这和“平均冲高很高但 T+5 收益为负”的现象一致。",
        "",
        "## 分数效果",
        "",
        markdown_table(score_table, ["stock_score_bucket", "n", "valid", "actual_avg_pct", "win_rate", "high5_avg_pct", "drawdown5_avg_pct"]),
        "",
        "## 排名效果",
        "",
        markdown_table(rank_table, ["rank_bucket", "n", "valid", "actual_avg_pct", "win_rate", "high5_avg_pct", "drawdown5_avg_pct"]),
        "",
        "## 策略组效果",
        "",
        markdown_table(group_table, ["strategy_group", "n", "valid", "expected_avg_pct", "actual_avg_pct", "win_rate", "high5_avg_pct", "drawdown5_avg_pct"]),
        "",
        "## 单策略效果（样本 >= 50）",
        "",
        markdown_table(strategy_table, ["strategy_group", "strategy_name", "n", "valid", "expected_avg_pct", "actual_avg_pct", "win_rate", "high5_avg_pct", "drawdown5_avg_pct"]),
        "",
        "## 买点效果",
        "",
        markdown_table(entry_table, ["entry_bucket", "n", "valid", "expected_avg_pct", "actual_avg_pct", "win_rate", "high5_avg_pct", "drawdown5_avg_pct"]),
        "",
        "## 卖点效果",
        "",
        markdown_table(exit_table, ["exit_bucket", "n", "valid", "expected_avg_pct", "actual_avg_pct", "win_rate", "high5_avg_pct", "drawdown5_avg_pct"]),
        "",
        "## 买卖点组合",
        "",
        markdown_table(entry_exit_table, ["entry_bucket", "exit_bucket", "n", "valid", "expected_avg_pct", "actual_avg_pct", "win_rate", "high5_avg_pct", "drawdown5_avg_pct"]),
        "",
        "## 标签完整性",
        "",
        markdown_table(valid_t5_by_date.reset_index().tail(8), ["date", "n", "t5_valid"]),
        "",
        "## 建议",
        "",
        "1. 7 月实盘前降低追涨确认类权重，尤其是 `T+1 开盘 0%-2%` 且持有到 T5 的组合。",
        "2. 高分候选仍可保留，但建议把 85 分以下作为观察池，不直接进入交易池。",
        "3. 对黄金碗、支撑回踩类低吸信号保留跟踪；它们是 6 月少数 T5 后验不差的策略。",
        "4. 下一步应做逐笔成交版复盘：按 T+1 开盘涨跌幅过滤真实成交，再按日内高低价重放 TP/SL，确认当前代理结论是否成立。",
        "",
    ]
    report_path = args.report_dir / "june_2026_strategy_review.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
