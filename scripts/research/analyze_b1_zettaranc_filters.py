#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Backtest zettaranc-inspired daily B1 filters.

This first pass intentionally uses only daily data available in this project.
Minute rules such as opening volume ratio, 9:33/9:37, and 14:55 execution are
supported by the data layer but not mixed into this daily-only backtest.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from analyze_b1_entry_exit_grid import ExitRule, add_future_prices, summarize_returns
from analyze_b1_xgb_entry_exit_grid import (
    DEFAULT_DAILY_DIR,
    DEFAULT_DATASET,
    DEFAULT_MODEL_DIR,
    DEFAULT_OUTPUT_DIR,
    drop_overlapping_trades,
    predict_xgb_models,
)
from quant.strategies.custom.b1_family import add_b1_family_signals


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ResearchCombo:
    name: str
    buy_filter: str
    exit_mode: str
    description: str


def build_combos() -> list[ResearchCombo]:
    return [
        ResearchCombo("M1_base", "base", "fixed", "原新稳健观察版 + 当前固定卖出"),
        ResearchCombo("M2_gap_up", "gap_up", "fixed", "T+1 gap>=0 + 当前固定卖出"),
        ResearchCombo("M3_gap_weekly", "gap_weekly", "fixed", "T+1 gap>=0 + 周线趋势 + 当前固定卖出"),
        ResearchCombo("M4_gap_yidong", "gap_yidong", "fixed", "T+1 gap>=0 + 异动缩量 + 当前固定卖出"),
        ResearchCombo("M5_gap_j10", "gap_j10", "fixed", "T+1 gap>=0 + J<=-10 + 当前固定卖出"),
        ResearchCombo("M6_structure_stop", "gap_up", "structure_stop", "T+1 gap>=0 + 结构止损"),
        ResearchCombo("M7_time_stop", "gap_up", "structure_time", "结构止损 + 3天不涨退出"),
        ResearchCombo("M8_bbi_break", "gap_up", "structure_bbi", "结构止损 + BBI两日破位"),
        ResearchCombo("M9_b2_extend", "gap_up", "b2_extend", "B2确认延长，否则T7"),
        ResearchCombo("M10_sell_score", "gap_up", "sell_score", "防卖飞评分退出"),
        ResearchCombo("M11_s1_veto", "gap_up", "s1_veto", "S1一票否决"),
        ResearchCombo("M12_weekly_structure", "gap_weekly", "structure_bbi_s1", "周线趋势 + 结构止损 + BBI破位 + S1"),
    ]


def read_daily_with_signals(symbol: str) -> pd.DataFrame | None:
    path = DEFAULT_DAILY_DIR / f"{symbol}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if "date" not in df.columns and "trade_date" in df.columns:
        df["date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    else:
        df["date"] = pd.to_datetime(df["date"])
    if "vol" in df.columns and "volume" not in df.columns:
        df = df.rename(columns={"vol": "volume"})
    if "ts_code" not in df.columns:
        df["ts_code"] = symbol
    if "symbol" not in df.columns:
        df["symbol"] = symbol
    df = df.sort_values("date").reset_index(drop=True)
    return add_b1_family_signals(df)


def build_signal_feature_frame(symbols: list[str]) -> pd.DataFrame:
    frames = []
    keep = [
        "symbol",
        "date",
        "weekly_bull_ma55_144",
        "weekly_bull_ma55_144_233",
        "post_yidong_shrink",
        "ground_volume_60d",
        "kdj_d_j",
        "bbi",
        "signal_b2",
        "s1_distribution",
        "sell_score_simple",
    ]
    for n, symbol in enumerate(symbols, start=1):
        df = read_daily_with_signals(symbol)
        if df is not None:
            present = [col for col in keep if col in df.columns]
            frames.append(df[present].copy())
        if n % 500 == 0 or n == len(symbols):
            print(f"  signal features: {n}/{len(symbols)} symbols", flush=True)
    if not frames:
        raise RuntimeError("no signal feature frames built")
    return pd.concat(frames, ignore_index=True)


def add_future_signal_context(candidates: pd.DataFrame, feature_frame: pd.DataFrame, max_hold_days: int) -> pd.DataFrame:
    frames = []
    for symbol, daily in feature_frame.groupby("symbol"):
        daily = daily.sort_values("date").reset_index(drop=True)
        future = daily[["symbol", "date"]].copy()
        for day in range(1, max_hold_days + 2):
            for col in ["bbi", "signal_b2", "s1_distribution", "sell_score_simple"]:
                if col in daily.columns:
                    future[f"{col}_t{day}"] = daily[col].shift(-day)
        frames.append(future)
    future_all = pd.concat(frames, ignore_index=True)
    return candidates.merge(future_all, on=["symbol", "date"], how="left")


def base_entry_mask(df: pd.DataFrame) -> pd.Series:
    return (df["pred_up10_es"] >= 0.25) & (df["pred_down3_es"] <= 0.40)


def apply_buy_filter(df: pd.DataFrame, name: str) -> pd.Series:
    mask = base_entry_mask(df)
    gap = (df["entry_open"] / df["close"] - 1) * 100
    if name in {"gap_up", "gap_weekly", "gap_yidong", "gap_j10"}:
        mask &= gap >= 0
    if name == "gap_weekly":
        mask &= df.get("weekly_bull_ma55_144", 0).fillna(0) > 0
    if name == "gap_yidong":
        mask &= (df.get("post_yidong_shrink", 0).fillna(0) > 0) | (df.get("ground_volume_60d", 0).fillna(0) > 0)
    if name == "gap_j10":
        mask &= df["kdj_d_j"] <= -10
    return mask


def fixed_exit(entry_df: pd.DataFrame) -> pd.DataFrame:
    rule = ExitRule("fixed_tp10.0%_sl1.5%_intraday_T7", "fixed", 6, 0.10, 0.015, stop_trigger="intraday")
    from analyze_b1_entry_exit_grid import simulate_exit

    return simulate_exit(entry_df, rule)


def structure_exit(entry_df: pd.DataFrame, mode: str) -> pd.DataFrame:
    n = len(entry_df)
    entry = entry_df["entry_open"].to_numpy(dtype=float)
    signal_low = entry_df["low"].to_numpy(dtype=float)
    t1_low = entry_df["low_t1"].to_numpy(dtype=float)
    stop_price = np.nanmin(np.vstack([signal_low, t1_low]), axis=0)
    ret = np.full(n, np.nan)
    exit_day = np.full(n, -1, dtype=int)
    exit_type = np.full(n, "unknown", dtype=object)
    exit_date = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
    hold_days = 11 if mode == "b2_extend" else 6

    for day in range(2, hold_days + 2):
        unresolved = np.isnan(ret)
        if not unresolved.any():
            break
        open_ = entry_df[f"open_t{day}"].to_numpy(dtype=float)
        high = entry_df[f"high_t{day}"].to_numpy(dtype=float)
        low = entry_df[f"low_t{day}"].to_numpy(dtype=float)
        close = entry_df[f"close_t{day}"].to_numpy(dtype=float)
        date_t = pd.to_datetime(entry_df[f"date_t{day}"]).to_numpy(dtype="datetime64[ns]")
        valid = unresolved & ~np.isnan(open_) & ~np.isnan(high) & ~np.isnan(low) & ~np.isnan(close)
        if not valid.any():
            continue

        s1 = entry_df.get(f"s1_distribution_t{day}", pd.Series(0, index=entry_df.index)).to_numpy(dtype=float)
        if mode in {"s1_veto", "structure_bbi_s1"}:
            hit = valid & (s1 > 0)
            if hit.any():
                ret[hit] = close[hit] / entry[hit] - 1
                exit_day[hit] = day
                exit_date[hit] = date_t[hit]
                exit_type[hit] = "s1_veto"
                valid &= np.isnan(ret)

        stop_hit = valid & (close <= stop_price)
        if stop_hit.any():
            ret[stop_hit] = close[stop_hit] / entry[stop_hit] - 1
            exit_day[stop_hit] = day
            exit_date[stop_hit] = date_t[stop_hit]
            exit_type[stop_hit] = "structure_stop"
            valid &= np.isnan(ret)

        if mode == "structure_time" and day == 4:
            time_hit = valid & (close <= entry * 1.02)
            if time_hit.any():
                ret[time_hit] = close[time_hit] / entry[time_hit] - 1
                exit_day[time_hit] = day
                exit_date[time_hit] = date_t[time_hit]
                exit_type[time_hit] = "time_stop"
                valid &= np.isnan(ret)

        if mode in {"structure_bbi", "structure_bbi_s1"} and day >= 3:
            bbi_today = entry_df.get(f"bbi_t{day}", pd.Series(np.nan, index=entry_df.index)).to_numpy(dtype=float)
            bbi_prev = entry_df.get(f"bbi_t{day - 1}", pd.Series(np.nan, index=entry_df.index)).to_numpy(dtype=float)
            close_prev = entry_df[f"close_t{day - 1}"].to_numpy(dtype=float)
            bbi_hit = valid & (close < bbi_today) & (close_prev < bbi_prev)
            if bbi_hit.any():
                ret[bbi_hit] = close[bbi_hit] / entry[bbi_hit] - 1
                exit_day[bbi_hit] = day
                exit_date[bbi_hit] = date_t[bbi_hit]
                exit_type[bbi_hit] = "bbi_break"
                valid &= np.isnan(ret)

        if mode == "sell_score":
            score = entry_df.get(f"sell_score_simple_t{day}", pd.Series(5, index=entry_df.index)).to_numpy(dtype=float)
            score_hit = valid & (score < 3)
            if score_hit.any():
                ret[score_hit] = close[score_hit] / entry[score_hit] - 1
                exit_day[score_hit] = day
                exit_date[score_hit] = date_t[score_hit]
                exit_type[score_hit] = "sell_score"
                valid &= np.isnan(ret)

        tp_hit = valid & (high >= entry * 1.10)
        if tp_hit.any():
            ret[tp_hit] = 0.10
            exit_day[tp_hit] = day
            exit_date[tp_hit] = date_t[tp_hit]
            exit_type[tp_hit] = "take_profit"
            valid &= np.isnan(ret)

        if mode == "b2_extend" and day == 7:
            b2_confirm = np.zeros(n, dtype=bool)
            for b2_day in [1, 2, 3, 4]:
                b2 = entry_df.get(f"signal_b2_t{b2_day}", pd.Series(0, index=entry_df.index)).to_numpy(dtype=float)
                b2_confirm |= b2 > 0
            no_b2_expiry = valid & ~b2_confirm
            if no_b2_expiry.any():
                ret[no_b2_expiry] = close[no_b2_expiry] / entry[no_b2_expiry] - 1
                exit_day[no_b2_expiry] = day
                exit_date[no_b2_expiry] = date_t[no_b2_expiry]
                exit_type[no_b2_expiry] = "no_b2_expiry"
                valid &= np.isnan(ret)

        expiry = valid & (day == hold_days + 1)
        if expiry.any():
            ret[expiry] = close[expiry] / entry[expiry] - 1
            exit_day[expiry] = day
            exit_date[expiry] = date_t[expiry]
            exit_type[expiry] = "expiry"

    result = entry_df[["date", "symbol"]].copy()
    result["return_pct"] = ret * 100
    result["exit_day"] = exit_day
    result["exit_date"] = exit_date
    result["exit_type"] = exit_type
    return result.dropna(subset=["return_pct"])


def evaluate_combo(candidates: pd.DataFrame, combo: ResearchCombo) -> list[dict]:
    entry_df = candidates[apply_buy_filter(candidates, combo.buy_filter)].copy()
    if entry_df.empty:
        return []
    trades = fixed_exit(entry_df) if combo.exit_mode == "fixed" else structure_exit(entry_df, combo.exit_mode)
    trades = trades.merge(entry_df[["date", "symbol", "split"]], on=["date", "symbol"], how="left")
    raw_trades = len(trades)
    trades = drop_overlapping_trades(trades)
    skipped = raw_trades - len(trades)
    rows = []
    for split in ["train", "test", "oot"]:
        part = trades[trades["split"] == split]
        metrics = summarize_returns(part)
        if metrics:
            rows.append({
                "combo": combo.name,
                "description": combo.description,
                "buy_filter": combo.buy_filter,
                "exit_mode": combo.exit_mode,
                "split": split,
                "raw_trades": raw_trades,
                "skipped_overlaps": skipped,
                "overlap_skip_rate": skipped / raw_trades if raw_trades else np.nan,
                "min_return_pct": float(part["return_pct"].min()) if not part.empty else np.nan,
                "max_return_pct": float(part["return_pct"].max()) if not part.empty else np.nan,
                **metrics,
            })
    return rows


def fmt_pct(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.2f}%"


def fmt_rate(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value) * 100:.2f}%"


def fmt_num(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{int(round(float(value))):,}"


def markdown_table(rows: list[dict], headers: list[str]) -> str:
    lines = ["|" + "|".join(headers) + "|", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("|" + "|".join(str(row.get(header, "")) for header in headers) + "|")
    return "\n".join(lines)


def format_rows(df: pd.DataFrame) -> list[dict]:
    rows = []
    for _, row in df.iterrows():
        rows.append({
            "组合": row["combo"],
            "说明": row["description"],
            "交易数": fmt_num(row["trades"]),
            "均值": fmt_pct(row["avg_return_pct"]),
            "胜率": fmt_rate(row["win_rate"]),
            "最大回撤": fmt_pct(row["max_drawdown_pct"]),
            "PF": f"{row['profit_factor']:.2f}" if pd.notna(row["profit_factor"]) else "",
            "最差单笔": fmt_pct(row["min_return_pct"]),
            "止损率": fmt_rate(row["stop_rate"]),
            "到期率": fmt_rate(row["expiry_rate"]),
        })
    return rows


def write_report(summary: pd.DataFrame, output_dir: Path, timestamp: str) -> Path:
    oot = summary[summary["split"] == "oot"].copy()
    ranked = oot.sort_values(["profit_factor", "avg_return_pct", "max_drawdown_pct"], ascending=[False, False, False])
    path = output_dir / f"b1_zettaranc_filters_backtest_{timestamp}.md"
    with path.open("w", encoding="utf-8") as f:
        f.write("# B1 zettaranc 规则优化回测\n\n")
        f.write("本轮只使用日线可得数据。Tushare 分钟接口已接入数据层，但开盘量比、9:33/9:37、14:55 规则未混入本轮结果。\n\n")
        f.write("## OOT 结果\n\n")
        f.write(markdown_table(format_rows(ranked), ["组合", "说明", "交易数", "均值", "胜率", "最大回撤", "PF", "最差单笔", "止损率", "到期率"]))
        f.write("\n\n")
        f.write("## 初步结论\n\n")
        if not ranked.empty:
            best = ranked.iloc[0]
            f.write(
                f"- 当前 OOT PF 最好的组合是 `{best['combo']}`：{best['description']}，"
                f"交易数 {fmt_num(best['trades'])}，均值 {fmt_pct(best['avg_return_pct'])}，"
                f"最大回撤 {fmt_pct(best['max_drawdown_pct'])}，PF {best['profit_factor']:.2f}。\n"
            )
        f.write("- 若某些组合交易数过少，应只作为观察，不应直接替代当前策略。\n")
        f.write("- 下一步应使用 Tushare `stk_mins` 补真实开盘量比，再验证 zettaranc 的 B1+1 开盘执行规则。\n")
    return path


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"loading dataset: {DEFAULT_DATASET}", flush=True)
    candidates = pd.read_parquet(DEFAULT_DATASET)
    candidates["date"] = pd.to_datetime(candidates["date"])
    candidates = predict_xgb_models(candidates, DEFAULT_MODEL_DIR)

    symbols = candidates["symbol"].dropna().astype(str).drop_duplicates().tolist()
    feature_frame = build_signal_feature_frame(symbols)
    feature_cols = [col for col in feature_frame.columns if col not in {"symbol", "date"}]
    new_feature_cols = [col for col in feature_cols if col not in candidates.columns]
    if new_feature_cols:
        candidates = candidates.merge(
            feature_frame[["symbol", "date", *new_feature_cols]],
            on=["symbol", "date"],
            how="left",
        )

    max_hold_days = 11
    print("adding future prices", flush=True)
    candidates = add_future_prices(candidates, DEFAULT_DAILY_DIR, max_hold_days=max_hold_days)
    candidates = add_future_signal_context(candidates, feature_frame, max_hold_days=max_hold_days)
    candidates = candidates.dropna(subset=["entry_open"]).copy()

    rows = []
    for combo in build_combos():
        print(f"evaluating {combo.name}", flush=True)
        rows.extend(evaluate_combo(candidates, combo))

    summary = pd.DataFrame(rows)
    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"b1_zettaranc_filters_backtest_{timestamp}.csv"
    summary.to_csv(summary_path, index=False)
    latest_summary = output_dir / "latest_b1_zettaranc_filters_backtest.csv"
    summary.to_csv(latest_summary, index=False)
    report_path = write_report(summary, output_dir, timestamp)
    latest_report = output_dir / "latest_b1_zettaranc_filters_backtest.md"
    latest_report.write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"summary: {summary_path}", flush=True)
    print(f"report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
