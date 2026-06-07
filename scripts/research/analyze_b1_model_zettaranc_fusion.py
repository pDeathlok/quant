#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fuse XGBoost probability thresholds with zettaranc-style B1 filters.

This research path keeps model scores explicit. It does not use the older
manual weighted entry_score because that score did not have a defensible
calibration logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_b1_entry_exit_grid import ExitRule, add_future_prices, simulate_exit, summarize_returns
from analyze_b1_xgb_entry_exit_grid import (
    DEFAULT_DAILY_DIR,
    DEFAULT_DATASET,
    DEFAULT_MODEL_DIR,
    DEFAULT_OUTPUT_DIR,
    ThresholdEntryRule,
    apply_entry_rule,
    drop_overlapping_trades,
    predict_xgb_models,
)
from analyze_b1_zettaranc_filters import (
    add_future_signal_context,
    build_signal_feature_frame,
    fixed_exit,
    structure_exit,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SIGNAL_CACHE = PROJECT_ROOT / "data/features/b1/b1_family_signal_features.parquet"


@dataclass(frozen=True)
class BuyFilter:
    name: str
    description: str


@dataclass(frozen=True)
class FusionCombo:
    name: str
    entry_rule: ThresholdEntryRule
    buy_filter: BuyFilter
    exit_rule: ExitRule | None
    exit_mode: str
    description: str


def build_entry_rules() -> list[ThresholdEntryRule]:
    """Model threshold rules selected from prior XGB explorations plus B1 robust rules."""
    return [
        ThresholdEntryRule("up10_ge_0.25_down3_le_0.40", min_up10=0.25, max_down3=0.40),
        ThresholdEntryRule("up10_ge_0.30_down3_le_0.40", min_up10=0.30, max_down3=0.40),
        ThresholdEntryRule("up10_ge_0.40_down3_le_0.40", min_up10=0.40, max_down3=0.40),
        ThresholdEntryRule("up8_ge_0.65_down3_le_0.40", min_up8=0.65, max_down3=0.40),
        ThresholdEntryRule("up8_ge_0.70_down3_le_0.40", min_up8=0.70, max_down3=0.40),
        ThresholdEntryRule("up8_ge_0.70_down3_le_0.45", min_up8=0.70, max_down3=0.45),
        ThresholdEntryRule("up8_ge_0.65_up10_ge_0.25_down3_le_0.40", min_up8=0.65, min_up10=0.25, max_down3=0.40),
        ThresholdEntryRule("up8_ge_0.60_up10_ge_0.25_down3_le_0.40", min_up8=0.60, min_up10=0.25, max_down3=0.40),
        ThresholdEntryRule("up5_ge_0.60_up8_ge_0.60_down2_le_0.50", min_up5=0.60, min_up8=0.60, max_down2=0.50),
    ]


def build_buy_filters() -> list[BuyFilter]:
    return [
        BuyFilter("model_only", "只使用模型阈值"),
        BuyFilter("gap_up", "T+1 开盘价 >= 信号日收盘价"),
        BuyFilter("gap_j0", "T+1 不低开 + 信号日 J<=0"),
        BuyFilter("gap_j10", "T+1 不低开 + 信号日 J<=-10"),
        BuyFilter("gap_j10_weekly", "T+1 不低开 + J<=-10 + 周线55/144多头"),
        BuyFilter("gap_j10_volume_calm", "T+1 不低开 + J<=-10 + 地量或异动后缩量"),
        BuyFilter("gap_0_to_2_j10", "T+1 开盘涨幅 0%-2% + J<=-10"),
        BuyFilter("gap_le_1_j10", "T+1 开盘涨幅 <=1% + J<=-10"),
    ]


def build_exit_rules() -> list[tuple[str, ExitRule | None]]:
    return [
        ("fixed_tp10_sl15_T7", ExitRule("fixed_tp10.0%_sl1.5%_intraday_T7", "fixed", 6, 0.10, 0.015, stop_trigger="intraday")),
        ("fixed_tp8_sl15_T7", ExitRule("fixed_tp8.0%_sl1.5%_intraday_T7", "fixed", 6, 0.08, 0.015, stop_trigger="intraday")),
        ("fixed_tp6_sl15_T7", ExitRule("fixed_tp6.0%_sl1.5%_intraday_T7", "fixed", 6, 0.06, 0.015, stop_trigger="intraday")),
        ("fixed_tp4_sl15_T5", ExitRule("fixed_tp4.0%_sl1.5%_intraday_T5", "fixed", 4, 0.04, 0.015, stop_trigger="intraday")),
        ("trail_target3_dd15_sl15_T5", ExitRule("trail_target3.0%_dd1.5%_sl1.5%_intraday_T5", "trailing", 4, 0.03, 0.015, 0.015, "intraday")),
        ("trail_target4_dd2_sl15_T7", ExitRule("trail_target4.0%_dd2.0%_sl1.5%_intraday_T7", "trailing", 6, 0.04, 0.015, 0.02, "intraday")),
        ("trail_target5_dd2_sl15_T9", ExitRule("trail_target5.0%_dd2.0%_sl1.5%_intraday_T9", "trailing", 8, 0.05, 0.015, 0.02, "intraday")),
        ("structure_stop", None),
        ("structure_time", None),
        ("sell_score", None),
    ]


def get_signal_features(symbols: list[str], force_refresh: bool = False) -> pd.DataFrame:
    if SIGNAL_CACHE.exists() and not force_refresh:
        cached = pd.read_parquet(SIGNAL_CACHE)
        cached["date"] = pd.to_datetime(cached["date"])
        cached_symbols = set(cached["symbol"].astype(str).unique())
        if set(symbols) <= cached_symbols:
            return cached[cached["symbol"].astype(str).isin(symbols)].copy()
    feature_frame = build_signal_feature_frame(symbols)
    SIGNAL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    feature_frame.to_parquet(SIGNAL_CACHE, index=False)
    return feature_frame


def apply_buy_filter(df: pd.DataFrame, buy_filter: BuyFilter) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    gap = (df["entry_open"] / df["close"] - 1) * 100
    if buy_filter.name != "model_only":
        mask &= gap >= 0
    if buy_filter.name in {"gap_j0", "gap_j10", "gap_j10_weekly", "gap_j10_volume_calm", "gap_0_to_2_j10", "gap_le_1_j10"}:
        threshold = 0 if buy_filter.name == "gap_j0" else -10
        mask &= df["kdj_d_j"] <= threshold
    if buy_filter.name == "gap_j10_weekly":
        mask &= df.get("weekly_bull_ma55_144", 0).fillna(0) > 0
    if buy_filter.name == "gap_j10_volume_calm":
        mask &= (df.get("ground_volume_60d", 0).fillna(0) > 0) | (df.get("post_yidong_shrink", 0).fillna(0) > 0)
    if buy_filter.name == "gap_0_to_2_j10":
        mask &= gap <= 2
    if buy_filter.name == "gap_le_1_j10":
        mask &= gap <= 1
    return mask


def build_combos() -> list[FusionCombo]:
    combos = []
    for entry_rule in build_entry_rules():
        for buy_filter in build_buy_filters():
            for exit_name, exit_rule in build_exit_rules():
                combos.append(
                    FusionCombo(
                        name=f"{entry_rule.name}__{buy_filter.name}__{exit_name}",
                        entry_rule=entry_rule,
                        buy_filter=buy_filter,
                        exit_rule=exit_rule,
                        exit_mode=exit_name,
                        description=f"{entry_rule.name} + {buy_filter.description} + {exit_name}",
                    )
                )
    return combos


def evaluate_combo(candidates: pd.DataFrame, combo: FusionCombo) -> list[dict]:
    mask = apply_entry_rule(candidates, combo.entry_rule) & apply_buy_filter(candidates, combo.buy_filter)
    entry_df = candidates[mask].copy()
    if len(entry_df) < 20:
        return []
    if combo.exit_rule is not None:
        trades = simulate_exit(entry_df, combo.exit_rule)
    elif combo.exit_mode == "fixed_tp10_sl15_T7":
        trades = fixed_exit(entry_df)
    else:
        trades = structure_exit(entry_df, combo.exit_mode)
    if trades.empty:
        return []

    trades = trades.merge(entry_df[["date", "symbol", "split"]], on=["date", "symbol"], how="left")
    raw_trades = len(trades)
    trades = drop_overlapping_trades(trades)
    skipped = raw_trades - len(trades)
    rows = []
    for split in ["train", "test", "oot"]:
        part = trades[trades["split"] == split].copy()
        metrics = summarize_returns(part)
        if not metrics:
            continue
        rows.append(
            {
                "combo": combo.name,
                "entry_rule": combo.entry_rule.name,
                "buy_filter": combo.buy_filter.name,
                "buy_filter_desc": combo.buy_filter.description,
                "exit_mode": combo.exit_mode,
                "description": combo.description,
                "split": split,
                "raw_trades": raw_trades,
                "skipped_overlaps": skipped,
                "overlap_skip_rate": skipped / raw_trades if raw_trades else np.nan,
                "min_return_pct": float(part["return_pct"].min()) if not part.empty else np.nan,
                "max_return_pct": float(part["return_pct"].max()) if not part.empty else np.nan,
                **metrics,
            }
        )
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
        rows.append(
            {
                "组合": row["combo"],
                "买入过滤": row["buy_filter"],
                "卖出": row["exit_mode"],
                "交易数": fmt_num(row["trades"]),
                "均值": fmt_pct(row["avg_return_pct"]),
                "胜率": fmt_rate(row["win_rate"]),
                "最大回撤": fmt_pct(row["max_drawdown_pct"]),
                "PF": f"{row['profit_factor']:.2f}" if pd.notna(row["profit_factor"]) else "",
                "最差单笔": fmt_pct(row["min_return_pct"]),
                "止损率": fmt_rate(row["stop_rate"]),
                "到期率": fmt_rate(row["expiry_rate"]),
            }
        )
    return rows


def stable_score(df: pd.DataFrame) -> pd.Series:
    """Rank by OOT quality while penalizing train/test instability."""
    pivot = df.pivot_table(index="combo", columns="split", values=["avg_return_pct", "profit_factor", "max_drawdown_pct", "trades"], aggfunc="first")
    rows = []
    for combo in pivot.index:
        try:
            oot_pf = pivot.loc[combo, ("profit_factor", "oot")]
            oot_avg = pivot.loc[combo, ("avg_return_pct", "oot")]
            oot_dd = pivot.loc[combo, ("max_drawdown_pct", "oot")]
            test_avg = pivot.loc[combo, ("avg_return_pct", "test")]
            train_avg = pivot.loc[combo, ("avg_return_pct", "train")]
            oot_trades = pivot.loc[combo, ("trades", "oot")]
        except KeyError:
            continue
        if pd.isna(oot_pf) or pd.isna(oot_avg) or pd.isna(oot_dd) or pd.isna(oot_trades):
            continue
        gap_penalty = abs(float(train_avg or 0) - float(test_avg or 0)) if pd.notna(test_avg) and pd.notna(train_avg) else 0
        trade_penalty = 0 if oot_trades >= 100 else (100 - oot_trades) / 100
        rows.append(
            {
                "combo": combo,
                "selection_score": float(oot_pf) + 0.25 * float(oot_avg) + 0.02 * float(oot_dd) - gap_penalty - trade_penalty,
            }
        )
    if not rows:
        return pd.Series(dtype=float)
    return pd.DataFrame(rows).set_index("combo")["selection_score"]


def write_report(summary: pd.DataFrame, output_dir: Path, timestamp: str) -> Path:
    path = output_dir / f"b1_model_zettaranc_fusion_{timestamp}.md"
    oot = summary[(summary["split"] == "oot") & (summary["trades"] >= 50)].copy()
    scores = stable_score(summary)
    oot["selection_score"] = oot["combo"].map(scores)
    by_score = oot.sort_values(["selection_score", "profit_factor", "avg_return_pct"], ascending=[False, False, False]).head(20)
    by_pf = oot.sort_values(["profit_factor", "avg_return_pct", "max_drawdown_pct"], ascending=[False, False, False]).head(20)
    by_dd = oot.sort_values(["max_drawdown_pct", "profit_factor", "avg_return_pct"], ascending=[False, False, False]).head(20)
    by_win = oot[oot["trades"] >= 100].sort_values(["win_rate", "profit_factor", "avg_return_pct"], ascending=[False, False, False]).head(20)

    with path.open("w", encoding="utf-8") as f:
        f.write("# B1 模型分 + zettaranc 规则融合回测\n\n")
        f.write("本轮买入侧只使用 XGBoost 模型概率阈值，不使用旧版手工加权 `entry_score`。\n\n")
        f.write("执行口径：T+0 出信号，T+1 开盘买入，T+2 起检查卖出；盘中止损/回撤止盈均采用 gap-aware 口径，跳空按实际开盘价成交；同一股票未卖出前不重复买入。\n\n")
        f.write("## OOT 综合排序 Top 20\n\n")
        f.write(markdown_table(format_rows(by_score), ["组合", "买入过滤", "卖出", "交易数", "均值", "胜率", "最大回撤", "PF", "最差单笔", "止损率", "到期率"]))
        f.write("\n\n")
        f.write("## OOT PF Top 20\n\n")
        f.write(markdown_table(format_rows(by_pf), ["组合", "买入过滤", "卖出", "交易数", "均值", "胜率", "最大回撤", "PF", "最差单笔", "止损率", "到期率"]))
        f.write("\n\n")
        f.write("## OOT 低回撤 Top 20\n\n")
        f.write(markdown_table(format_rows(by_dd), ["组合", "买入过滤", "卖出", "交易数", "均值", "胜率", "最大回撤", "PF", "最差单笔", "止损率", "到期率"]))
        f.write("\n\n")
        f.write("## OOT 高胜率 Top 20\n\n")
        f.write(markdown_table(format_rows(by_win), ["组合", "买入过滤", "卖出", "交易数", "均值", "胜率", "最大回撤", "PF", "最差单笔", "止损率", "到期率"]))
        f.write("\n\n")
        f.write("## 初步结论\n\n")
        if not by_score.empty:
            best = by_score.iloc[0]
            f.write(
                f"- 综合排序第一是 `{best['combo']}`，OOT 交易数 {fmt_num(best['trades'])}，"
                f"均值 {fmt_pct(best['avg_return_pct'])}，胜率 {fmt_rate(best['win_rate'])}，"
                f"最大回撤 {fmt_pct(best['max_drawdown_pct'])}，PF {best['profit_factor']:.2f}。\n"
            )
        f.write("- 这份结果用于筛选候选组合；最终上线前仍应使用分钟数据验证 T+1 开盘量比和实际成交点。\n")
    return path


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"loading dataset: {DEFAULT_DATASET}", flush=True)
    candidates = pd.read_parquet(DEFAULT_DATASET)
    candidates["date"] = pd.to_datetime(candidates["date"])
    candidates = predict_xgb_models(candidates, DEFAULT_MODEL_DIR)

    symbols = candidates["symbol"].dropna().astype(str).drop_duplicates().tolist()
    feature_frame = get_signal_features(symbols)
    feature_cols = [col for col in feature_frame.columns if col not in {"symbol", "date"} and col not in candidates.columns]
    if feature_cols:
        candidates = candidates.merge(feature_frame[["symbol", "date", *feature_cols]], on=["symbol", "date"], how="left")

    max_hold_days = 11
    print("adding future prices", flush=True)
    candidates = add_future_prices(candidates, DEFAULT_DAILY_DIR, max_hold_days=max_hold_days)
    candidates = add_future_signal_context(candidates, feature_frame, max_hold_days=max_hold_days)
    candidates = candidates.dropna(subset=["entry_open"]).copy()

    rows = []
    combos = build_combos()
    print(f"evaluating {len(combos)} fusion combos", flush=True)
    for idx, combo in enumerate(combos, start=1):
        rows.extend(evaluate_combo(candidates, combo))
        if idx % 100 == 0 or idx == len(combos):
            print(f"  combos: {idx}/{len(combos)}", flush=True)

    summary = pd.DataFrame(rows)
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = DEFAULT_OUTPUT_DIR / f"b1_model_zettaranc_fusion_{timestamp}.csv"
    summary.to_csv(summary_path, index=False)
    latest_summary = DEFAULT_OUTPUT_DIR / "latest_b1_model_zettaranc_fusion.csv"
    summary.to_csv(latest_summary, index=False)
    report_path = write_report(summary, DEFAULT_OUTPUT_DIR, timestamp)
    latest_report = DEFAULT_OUTPUT_DIR / "latest_b1_model_zettaranc_fusion.md"
    latest_report.write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"summary: {summary_path}", flush=True)
    print(f"report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
