#!/usr/bin/env python
"""Backtest daily Chan-theory structure signals on local stock history."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quant.data import MarketDataStore, MarketDataStoreConfig, read_partitioned_symbol_file
from quant.strategies.custom.chan_daily import add_chan_daily_signals


DEFAULT_DAILY_DIR = PROJECT_ROOT / "data/stocks_daily"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports/chan_daily"


@dataclass(frozen=True)
class ExitRule:
    name: str
    kind: str
    hold_days: int
    take_profit: float | None = None
    stop_loss: float | None = None
    trail_drawdown: float | None = None
    ma_exit: str | None = None


def read_daily_file(
    path: Path,
    start_date: str | pd.Timestamp | None = None,
    source_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    df = source_frame.copy() if source_frame is not None else read_partitioned_symbol_file(path, start_date=start_date)
    if "trade_date" in df.columns:
        df["date"] = pd.to_datetime(df["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        raise ValueError("daily file missing date/trade_date")
    if "volume" not in df.columns and "vol" in df.columns:
        df = df.rename(columns={"vol": "volume"})
    elif "vol" in df.columns and ("volume" not in df.columns or df["volume"].isna().all()):
        df["volume"] = df["vol"]
    if "symbol" not in df.columns or df["symbol"].isna().all():
        df["symbol"] = df["ts_code"].astype(str) if "ts_code" in df.columns else path.stem
    return df.sort_values("date").dropna(subset=["date"]).reset_index(drop=True)


def build_candidates(
    daily_dir: Path,
    start_date: str,
    max_workers: int = 1,
    min_rows: int = 140,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    start_ts = pd.to_datetime(start_date)
    history_start = start_ts - pd.Timedelta(days=800)
    store = MarketDataStore(MarketDataStoreConfig(backend="parquet", root=daily_dir.parent))
    market = store.read_market_range(daily_dir.name, start_date=history_start.strftime("%Y%m%d"))
    tasks = [
        (daily_dir / f"{symbol}.parquet", group.reset_index(drop=True))
        for symbol, group in market.groupby("ts_code", sort=False)
    ]

    def process(task: tuple[Path, pd.DataFrame]) -> pd.DataFrame | None:
        path, source_frame = task
        try:
            daily = read_daily_file(path, source_frame=source_frame)
            if len(daily) < min_rows:
                return None
            signal_frame = add_chan_daily_signals(daily)
            signals = signal_frame[
                (signal_frame["date"] >= start_ts)
                & (
                    (signal_frame["signal_chan_daily_long"] == 1)
                    | (signal_frame["chan_buy1_watch"] == 1)
                )
            ].copy()
            if signals.empty:
                return None
            keep = [
                "date",
                "symbol",
                "name",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "chan_buy1_watch",
                "chan_buy2_confirm",
                "chan_buy3_confirm",
                "signal_chan_daily_long",
                "signal_chan_daily_exit",
                "chan_center_low",
                "chan_center_high",
                "chan_center_width",
                "chan_stroke_amplitude",
                "chan_score",
                "chan_signal_name",
                "chan_structure_note",
            ]
            return signals[[col for col in keep if col in signals.columns]]
        except Exception as exc:
            print(f"  skip {path.name}: {exc}")
            return None

    if max_workers <= 1:
        for n, task in enumerate(tasks, start=1):
            result = process(task)
            if result is not None:
                frames.append(result)
            if n % 500 == 0 or n == len(tasks):
                print(f"  candidates: {n}/{len(tasks)} symbols")
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process, task) for task in tasks]
            for n, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                if result is not None:
                    frames.append(result)
                if n % 500 == 0 or n == len(tasks):
                    print(f"  candidates: {n}/{len(tasks)} symbols")

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)


def add_future_prices(candidates: pd.DataFrame, daily_dir: Path, max_hold_days: int) -> pd.DataFrame:
    frames = []
    for symbol in candidates["symbol"].dropna().astype(str).unique():
        path = daily_dir / f"{symbol}.parquet"
        if not path.exists():
            path = daily_dir / f"{symbol.replace('.SH', '').replace('.SZ', '').replace('.BJ', '')}.parquet"
        if not path.exists():
            continue
        daily = read_daily_file(path)
        daily["ma5"] = daily["close"].rolling(5).mean()
        daily["ma10"] = daily["close"].rolling(10).mean()
        daily["ma20"] = daily["close"].rolling(20).mean()
        future_cols = {
            "symbol": daily["symbol"].astype(str),
            "date": daily["date"],
            "entry_open": daily["open"].shift(-1),
        }
        for day in range(1, max_hold_days + 2):
            future_cols[f"date_t{day}"] = daily["date"].shift(-day)
            for col in ["open", "high", "low", "close", "ma5", "ma10", "ma20"]:
                future_cols[f"{col}_t{day}"] = daily[col].shift(-day)
        frames.append(pd.DataFrame(future_cols))

    if not frames:
        raise RuntimeError(f"No future prices found under {daily_dir}")
    merged = candidates.merge(pd.concat(frames, ignore_index=True), on=["symbol", "date"], how="left")
    return merged.dropna(subset=["entry_open"]).copy()


def build_exit_rules() -> list[ExitRule]:
    rules: list[ExitRule] = []
    for hold in [5, 10, 15, 20]:
        rules.append(ExitRule(f"time_T{hold + 1}_close", "time", hold))
    for hold in [10, 15, 20]:
        for tp in [0.08, 0.12, 0.16, 0.20]:
            for sl in [0.04, 0.06, 0.08]:
                rules.append(ExitRule(f"fixed_tp{tp:.0%}_sl{sl:.0%}_T{hold + 1}", "fixed", hold, tp, sl))
    for hold in [15, 20]:
        for tp in [0.10, 0.14, 0.18]:
            for trail in [0.04, 0.06, 0.08]:
                rules.append(ExitRule(f"trail_target{tp:.0%}_dd{trail:.0%}_T{hold + 1}", "trailing", hold, tp, 0.06, trail))
    for ma in ["ma5", "ma10", "ma20"]:
        rules.append(ExitRule(f"lose_{ma}_or_tp16_T21", "technical", 20, 0.16, 0.06, None, ma))
    return rules


def simulate_exit(df: pd.DataFrame, rule: ExitRule) -> pd.DataFrame:
    entry = df["entry_open"].to_numpy(dtype=float)
    n = len(df)
    ret = np.full(n, np.nan)
    exit_day = np.full(n, -1, dtype=int)
    exit_date = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
    exit_type = np.full(n, "unknown", dtype=object)
    peak = np.zeros(n, dtype=float)
    active = np.zeros(n, dtype=bool)

    for day in range(2, rule.hold_days + 2):
        unresolved = np.isnan(ret) & ~np.isnan(entry)
        if not unresolved.any():
            break
        open_ = df[f"open_t{day}"].to_numpy(dtype=float)
        high = df[f"high_t{day}"].to_numpy(dtype=float)
        low = df[f"low_t{day}"].to_numpy(dtype=float)
        close = df[f"close_t{day}"].to_numpy(dtype=float)
        date_t = pd.to_datetime(df[f"date_t{day}"]).to_numpy(dtype="datetime64[ns]")
        valid = unresolved & ~np.isnan(close) & ~np.isnan(high) & ~np.isnan(low)
        if not valid.any():
            continue

        stop_price = entry * (1 - (rule.stop_loss or 0))
        stop_hit = valid & (rule.stop_loss is not None) & (low <= stop_price)
        if stop_hit.any():
            gap = stop_hit & (open_ <= stop_price)
            normal = stop_hit & ~gap
            ret[gap] = open_[gap] / entry[gap] - 1
            ret[normal] = stop_price[normal] / entry[normal] - 1
            exit_day[stop_hit] = day
            exit_date[stop_hit] = date_t[stop_hit]
            exit_type[stop_hit] = "stop_loss"

        still = valid & np.isnan(ret)
        if not still.any():
            continue

        if rule.kind == "fixed":
            tp_hit = still & (high >= entry * (1 + (rule.take_profit or 0)))
            ret[tp_hit] = rule.take_profit or 0
            exit_day[tp_hit] = day
            exit_date[tp_hit] = date_t[tp_hit]
            exit_type[tp_hit] = "take_profit"
        elif rule.kind == "trailing":
            # Daily OHLC does not reveal whether the high or low occurred first.
            # Only peaks confirmed before this bar may tighten today's stop.
            trail_price = peak * (1 - (rule.trail_drawdown or 0))
            trail_hit = still & active & (low <= trail_price)
            gap = trail_hit & (open_ <= trail_price)
            normal = trail_hit & ~gap
            ret[gap] = open_[gap] / entry[gap] - 1
            ret[normal] = trail_price[normal] / entry[normal] - 1
            exit_day[trail_hit] = day
            exit_date[trail_hit] = date_t[trail_hit]
            exit_type[trail_hit] = "trailing_stop"
            survivors = valid & np.isnan(ret)
            peak[survivors] = np.maximum(peak[survivors], high[survivors])
            active |= survivors & (peak >= entry * (1 + (rule.take_profit or 0)))
        elif rule.kind == "technical":
            tp_hit = still & (high >= entry * (1 + (rule.take_profit or 0)))
            ret[tp_hit] = rule.take_profit or 0
            exit_day[tp_hit] = day
            exit_date[tp_hit] = date_t[tp_hit]
            exit_type[tp_hit] = "take_profit"
            still = valid & np.isnan(ret)
            ma = df[f"{rule.ma_exit}_t{day}"].to_numpy(dtype=float)
            lost = still & (close < ma)
            ret[lost] = close[lost] / entry[lost] - 1
            exit_day[lost] = day
            exit_date[lost] = date_t[lost]
            exit_type[lost] = "technical_exit"

        expiry = valid & np.isnan(ret) & (day == rule.hold_days + 1)
        ret[expiry] = close[expiry] / entry[expiry] - 1
        exit_day[expiry] = day
        exit_date[expiry] = date_t[expiry]
        exit_type[expiry] = "expiry"

    result_cols = [
        "date",
        "symbol",
        "name",
        "chan_signal_name",
        "chan_buy1_watch",
        "chan_buy2_confirm",
        "chan_buy3_confirm",
        "chan_score",
        "chan_center_width",
        "chan_stroke_amplitude",
        "close",
        "entry_open",
    ]
    result = df[[col for col in result_cols if col in df.columns]].copy()
    result["exit_rule"] = rule.name
    result["return_pct"] = ret * 100
    result["entry_date"] = pd.to_datetime(df["date_t1"], errors="coerce")
    result["exit_day"] = exit_day
    result["exit_date"] = exit_date
    result["exit_type"] = exit_type
    result["entry_gap_pct"] = (df["entry_open"] / df["close"] - 1) * 100
    return result.dropna(subset=["return_pct"])


def summarize_returns(trades: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for key, group in trades.groupby(group_cols, dropna=False):
        returns = group["return_pct"].astype(float)
        wins = returns > 0
        gross_profit = returns[returns > 0].sum()
        gross_loss = -returns[returns < 0].sum()
        if not isinstance(key, tuple):
            key = (key,)
        rows.append(
            {
                **dict(zip(group_cols, key)),
                "trades": int(len(group)),
                "avg_return_pct": float(returns.mean()),
                "median_return_pct": float(returns.median()),
                "win_rate": float(wins.mean()),
                "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0 else np.inf,
                "max_loss_pct": float(returns.min()),
                "max_gain_pct": float(returns.max()),
                "avg_hold_days": float(group["exit_day"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["avg_return_pct", "profit_factor"], ascending=False)


def add_buckets(candidates: pd.DataFrame) -> pd.DataFrame:
    out = candidates.copy()
    out["signal_bucket"] = np.select(
        [
            out["chan_buy3_confirm"].eq(1),
            out["chan_buy2_confirm"].eq(1),
            out["chan_buy1_watch"].eq(1),
        ],
        ["buy3", "buy2", "buy1_watch"],
        default="other",
    )
    out["score_bucket"] = pd.cut(
        out["chan_score"].fillna(0),
        bins=[-np.inf, 80, 90, 95, np.inf],
        labels=["score_lt80", "score_80_90", "score_90_95", "score_ge95"],
    )
    out["center_width_bucket"] = pd.cut(
        out["chan_center_width"],
        bins=[-np.inf, 0.04, 0.08, 0.12, np.inf],
        labels=["center_lt4", "center_4_8", "center_8_12", "center_ge12"],
    )
    out["entry_gap_bucket"] = pd.cut(
        out["entry_gap_pct"],
        bins=[-np.inf, 0, 3, 6, np.inf],
        labels=["gap_le0", "gap_0_3", "gap_3_6", "gap_gt6"],
    )
    return out


def build_topn_portfolio(
    trades: pd.DataFrame,
    top_n: int = 10,
    round_trip_cost_pct: float = 0.2,
) -> pd.DataFrame:
    """Build cash-constrained realized equity for overlapping trades."""

    if top_n <= 0:
        raise ValueError("top_n must be positive")
    required = {"date", "entry_date", "exit_date", "return_pct", "chan_signal_name"}
    missing = required - set(trades.columns)
    if missing:
        raise ValueError(f"trades missing portfolio columns: {sorted(missing)}")

    eligible = trades[trades["chan_signal_name"].ne("")].copy()
    eligible["entry_date"] = pd.to_datetime(eligible["entry_date"], errors="coerce")
    eligible["exit_date"] = pd.to_datetime(eligible["exit_date"], errors="coerce")
    eligible = eligible.dropna(subset=["entry_date", "exit_date", "return_pct"])
    eligible = eligible.sort_values(["date", "chan_score"], ascending=[True, False])
    eligible = eligible.groupby("date", group_keys=False).head(top_n)
    if eligible.empty:
        return pd.DataFrame(
            columns=["date", "opened_positions", "closed_positions", "active_positions", "cash", "equity"]
        )

    entries = {date: group for date, group in eligible.groupby("entry_date")}
    event_dates = sorted(set(eligible["entry_date"]) | set(eligible["exit_date"]))
    cash = 1.0
    positions: list[dict[str, float | pd.Timestamp]] = []
    rows: list[dict[str, float | int | pd.Timestamp]] = []

    for event_date in event_dates:
        opened = 0
        for _, trade in entries.get(event_date, pd.DataFrame()).iterrows():
            if len(positions) >= top_n or cash <= 0:
                continue
            book_value = sum(float(position["allocation"]) for position in positions)
            target_allocation = (cash + book_value) / top_n
            allocation = min(cash, target_allocation)
            if allocation <= 0:
                continue
            cash -= allocation
            positions.append(
                {
                    "allocation": allocation,
                    "exit_date": pd.Timestamp(trade["exit_date"]),
                    "return_pct": float(trade["return_pct"]) - round_trip_cost_pct,
                }
            )
            opened += 1

        closed = 0
        remaining = []
        for position in positions:
            if position["exit_date"] == event_date:
                cash += float(position["allocation"]) * (1 + float(position["return_pct"]) / 100)
                closed += 1
            else:
                remaining.append(position)
        positions = remaining
        book_value = sum(float(position["allocation"]) for position in positions)
        rows.append(
            {
                "date": event_date,
                "opened_positions": opened,
                "closed_positions": closed,
                "active_positions": len(positions),
                "cash": cash,
                "equity": cash + book_value,
            }
        )
    return pd.DataFrame(rows)


def run_backtest(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Build candidates from {args.daily_dir}")
    candidates = build_candidates(args.daily_dir, args.start_date, args.max_workers)
    if candidates.empty:
        raise RuntimeError("No Chan candidates found")
    candidates.to_parquet(output_dir / "chan_daily_candidates.parquet", index=False)
    candidates.to_csv(output_dir / "chan_daily_candidates.csv", index=False)

    print("[2/4] Add future prices")
    max_hold_days = max(rule.hold_days for rule in build_exit_rules())
    priced = add_future_prices(candidates, args.daily_dir, max_hold_days)

    print("[3/4] Simulate exit rules")
    trade_frames = [simulate_exit(priced, rule) for rule in build_exit_rules()]
    trades = pd.concat(trade_frames, ignore_index=True)
    trades = add_buckets(trades)
    trades.to_csv(output_dir / "chan_daily_trades.csv", index=False)
    trades.to_parquet(output_dir / "chan_daily_trades.parquet", index=False)

    print("[4/4] Summarize")
    summary = summarize_returns(trades[trades["chan_signal_name"].ne("")], ["exit_rule"])
    by_signal = summarize_returns(trades, ["exit_rule", "signal_bucket"])
    score_buckets = summarize_returns(trades[trades["chan_signal_name"].ne("")], ["exit_rule", "score_bucket"])
    center_buckets = summarize_returns(trades[trades["chan_signal_name"].ne("")], ["exit_rule", "center_width_bucket"])
    gap_buckets = summarize_returns(trades[trades["chan_signal_name"].ne("")], ["exit_rule", "entry_gap_bucket"])

    summary.to_csv(output_dir / "chan_daily_summary.csv", index=False)
    by_signal.to_csv(output_dir / "chan_daily_by_signal.csv", index=False)
    score_buckets.to_csv(output_dir / "chan_daily_score_buckets.csv", index=False)
    center_buckets.to_csv(output_dir / "chan_daily_center_buckets.csv", index=False)
    gap_buckets.to_csv(output_dir / "chan_daily_entry_gap_buckets.csv", index=False)
    portfolio_frames = []
    for exit_rule, rule_trades in trades.groupby("exit_rule", sort=False):
        equity = build_topn_portfolio(rule_trades, top_n=args.top_n)
        equity["exit_rule"] = exit_rule
        portfolio_frames.append(equity)
    pd.concat(portfolio_frames, ignore_index=True).to_csv(
        output_dir / "chan_daily_topn_equity.csv",
        index=False,
    )

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "daily_dir": str(args.daily_dir),
        "start_date": args.start_date,
        "candidate_rows": int(len(candidates)),
        "priced_rows": int(len(priced)),
        "trade_rows": int(len(trades)),
        "top_summary": summary.head(10).to_dict("records"),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.head(12).to_string(index=False))
    print(f"Saved reports to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily-dir", type=Path, default=DEFAULT_DAILY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-date", default="2015-01-01")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--top-n", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    run_backtest(parse_args())
