#!/usr/bin/env python3
"""Run a market-wide event study of completed low-9 plus negative KDJ-J.

This script deliberately separates signal discovery at the close from an
executable next-open entry.  It reads the project's canonical raw Tushare
daily partitions, applies causal corporate-action continuity, and clusters
inference by signal date.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from quant.research.low9_kdj_rebound import (  # noqa: E402
    DEFAULT_HORIZONS,
    DEFAULT_J_THRESHOLDS,
    SymbolSignalState,
    benjamini_hochberg,
    paired_incremental_j_test,
    summarize_event_subset,
)


ORDINARY_A_CODE = re.compile(r"^\d{6}\.(?:SH|SZ|BJ)$")
DAILY_COLUMNS = [
    "ts_code",
    "trade_date",
    "date",
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "pct_chg",
    "vol",
    "amount",
    "name",
]


def parse_number_list(raw: str, cast) -> tuple:
    values = tuple(cast(item.strip()) for item in raw.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("list must not be empty")
    return values


def normalize_daily(frame: pd.DataFrame, name_lookup: dict[str, str]) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    if "date" not in out.columns and "trade_date" in out.columns:
        out["date"] = pd.to_datetime(
            out["trade_date"].astype(str), format="%Y%m%d", errors="coerce"
        )
    else:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if "trade_date" not in out.columns:
        out["trade_date"] = out["date"].dt.strftime("%Y%m%d")
    if "ts_code" not in out.columns and "symbol" in out.columns:
        out["ts_code"] = out["symbol"].astype(str)
    if "amount" not in out.columns:
        out["amount"] = np.nan
    if "name" not in out.columns:
        out["name"] = out["ts_code"].map(name_lookup)
    else:
        out["name"] = out["name"].where(out["name"].notna(), out["ts_code"].map(name_lookup))
    keep = [column for column in DAILY_COLUMNS if column in out.columns]
    out = out[keep].dropna(subset=["ts_code", "date", "open", "high", "low", "close"])
    out["ts_code"] = out["ts_code"].astype(str)
    return out


def load_universe(stock_basic_history: Path, start_date: str, end_date: str) -> pd.DataFrame:
    history = pd.read_parquet(stock_basic_history)
    history["ts_code"] = history["ts_code"].astype(str)
    history = history[history["ts_code"].str.match(ORDINARY_A_CODE)]
    history = history[history["exchange"].isin(["SSE", "SZSE", "BSE"])]
    history = history[history["market"].isin(["主板", "创业板", "科创板", "北交所"])]
    history["list_date"] = history["list_date"].fillna("00000000").astype(str)
    history["delist_date"] = history["delist_date"].fillna("99999999").astype(str)
    history = history[
        (history["list_date"] <= end_date) & (history["delist_date"] >= start_date)
    ]
    return history.drop_duplicates("ts_code", keep="last").reset_index(drop=True)


def local_symbol_coverage(daily_root: Path) -> set[str]:
    symbols: set[str] = set()
    paths = sorted(daily_root.glob("year_month=*/data.parquet"))
    for index, path in enumerate(paths, start=1):
        values = pd.read_parquet(path, columns=["ts_code"])["ts_code"]
        symbols.update(values.dropna().astype(str).unique().tolist())
        if index % 50 == 0:
            print(f"coverage scan: {index}/{len(paths)} partitions", flush=True)
    return symbols


def fetch_missing_delisted(
    *,
    universe: pd.DataFrame,
    local_symbols: set[str],
    supplemental_root: Path,
    start_date: str,
    end_date: str,
    sleep_seconds: float,
) -> dict[str, object]:
    """Fetch delisted histories absent from the canonical local store."""

    try:
        import tushare as ts
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("tushare is required for --fetch-missing-delist") from exc
    token = os.getenv("TUSHARE_TOKEN")
    if not token:
        raise RuntimeError("TUSHARE_TOKEN is not configured")
    pro = ts.pro_api(token)
    supplemental_root.mkdir(parents=True, exist_ok=True)
    candidates = universe[
        universe["list_status"].eq("D") & ~universe["ts_code"].isin(local_symbols)
    ].copy()
    manifest_path = supplemental_root / "fetch_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"status": {}}
    status: dict[str, dict[str, object]] = manifest.setdefault("status", {})

    fetched = 0
    empty = 0
    failed = 0
    skipped = 0
    for position, row in enumerate(candidates.itertuples(index=False), start=1):
        code = str(row.ts_code)
        output_path = supplemental_root / f"{code}.parquet"
        prior = status.get(code, {})
        if output_path.exists() or prior.get("state") == "empty":
            skipped += 1
            continue
        request_start = max(start_date, str(row.list_date))
        request_end = min(end_date, str(row.delist_date))
        last_error = ""
        daily = pd.DataFrame()
        for attempt in range(1, 4):
            try:
                daily = pro.daily(
                    ts_code=code,
                    start_date=request_start,
                    end_date=request_end,
                )
                last_error = ""
                break
            except Exception as exc:  # pragma: no cover - network-specific
                last_error = f"{type(exc).__name__}: {exc}"
                time.sleep(max(sleep_seconds, 0.5) * attempt)
        if last_error:
            failed += 1
            status[code] = {"state": "failed", "error": last_error}
        elif daily.empty:
            empty += 1
            status[code] = {
                "state": "empty",
                "start_date": request_start,
                "end_date": request_end,
            }
        else:
            daily = daily.sort_values("trade_date").reset_index(drop=True)
            daily["date"] = pd.to_datetime(
                daily["trade_date"].astype(str), format="%Y%m%d", errors="coerce"
            )
            daily["name"] = str(row.name)
            temporary = output_path.with_suffix(".parquet.tmp")
            daily.to_parquet(temporary, index=False)
            os.replace(temporary, output_path)
            fetched += 1
            status[code] = {
                "state": "fetched",
                "rows": int(len(daily)),
                "start_date": str(daily["trade_date"].min()),
                "end_date": str(daily["trade_date"].max()),
            }
        manifest.update(
            {
                "source": "Tushare pro.daily",
                "requested_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
                "requested_candidates": int(len(candidates)),
            }
        )
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if position % 20 == 0 or position == len(candidates):
            print(
                "supplemental fetch: "
                f"{position}/{len(candidates)}, fetched={fetched}, empty={empty}, "
                f"failed={failed}, skipped={skipped}",
                flush=True,
            )
        time.sleep(max(sleep_seconds, 0.0))
    return {
        "requested_candidates": int(len(candidates)),
        "fetched_this_run": fetched,
        "empty_this_run": empty,
        "failed_this_run": failed,
        "skipped_existing": skipped,
    }


def load_supplemental_by_month(
    supplemental_root: Path,
    name_lookup: dict[str, str],
) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    frames: list[pd.DataFrame] = []
    for path in sorted(supplemental_root.glob("*.parquet")):
        try:
            frame = normalize_daily(pd.read_parquet(path), name_lookup)
        except Exception as exc:
            print(f"skip unreadable supplemental file {path}: {exc}", flush=True)
            continue
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return {}, {"files": 0, "rows": 0, "symbols": 0}
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["month"] = combined["date"].dt.strftime("%Y%m")
    by_month = {
        month: group.drop(columns="month").reset_index(drop=True)
        for month, group in combined.groupby("month", sort=True)
    }
    return by_month, {
        "files": len(frames),
        "rows": int(len(combined)),
        "symbols": int(combined["ts_code"].nunique()),
    }


def load_market_index(path: Path) -> tuple[dict[pd.Timestamp, dict[str, float]], dict[str, object]]:
    frame = pd.read_parquet(path)
    if "date" not in frame.columns and "trade_date" in frame.columns:
        frame["date"] = pd.to_datetime(
            frame["trade_date"].astype(str), format="%Y%m%d", errors="coerce"
        )
    else:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date", "open", "close"]).sort_values("date")
    mapping = {
        pd.Timestamp(row.date): {"open": float(row.open), "close": float(row.close)}
        for row in frame[["date", "open", "close"]].itertuples(index=False)
    }
    metadata = {
        "path": str(path),
        "rows": int(len(frame)),
        "start": frame["date"].min().date().isoformat(),
        "end": frame["date"].max().date().isoformat(),
    }
    return mapping, metadata


def recompute_event_market_returns(
    events: pd.DataFrame,
    market_index: dict[pd.Timestamp, dict[str, float]],
) -> pd.DataFrame:
    """Refresh benchmark and abnormal returns without rebuilding stock signals."""

    out = events.copy()
    for column in ["signal_date", "entry_date", "exit_date"]:
        out[column] = pd.to_datetime(out[column], errors="coerce")
    market_open = {date: values["open"] for date, values in market_index.items()}
    market_close = {date: values["close"] for date, values in market_index.items()}
    signal_market_close = out["signal_date"].map(market_close)
    entry_market_open = out["entry_date"].map(market_open)
    exit_market_close = out["exit_date"].map(market_close)
    out["market_close_return"] = exit_market_close / signal_market_close - 1.0
    out["market_executable_return"] = exit_market_close / entry_market_open - 1.0
    out["abnormal_close_return"] = (
        out["close_return"] - out["market_close_return"]
    )
    out["abnormal_executable_return"] = (
        out["executable_return"] - out["market_executable_return"]
    )
    return out


def iter_daily_months(
    *,
    daily_root: Path,
    supplemental_by_month: dict[str, pd.DataFrame],
    name_lookup: dict[str, str],
    universe_symbols: set[str],
    start_date: str,
    end_date: str,
) -> Iterable[tuple[str, pd.DataFrame]]:
    start_month = start_date[:6]
    end_month = end_date[:6]
    local_paths = {
        path.parent.name.partition("=")[2]: path
        for path in daily_root.glob("year_month=*/data.parquet")
    }
    months = sorted(
        month
        for month in set(local_paths) | set(supplemental_by_month)
        if start_month <= month <= end_month
    )
    for month in months:
        pieces: list[pd.DataFrame] = []
        if month in local_paths:
            pieces.append(normalize_daily(pd.read_parquet(local_paths[month]), name_lookup))
        if month in supplemental_by_month:
            pieces.append(supplemental_by_month[month])
        if not pieces:
            continue
        frame = pd.concat(pieces, ignore_index=True, sort=False)
        frame = frame[frame["ts_code"].isin(universe_symbols)]
        ymd = frame["date"].dt.strftime("%Y%m%d")
        frame = frame[(ymd >= start_date) & (ymd <= end_date)]
        frame = (
            frame.drop_duplicates(["ts_code", "date"], keep="last")
            .sort_values(["ts_code", "date"])
            .reset_index(drop=True)
        )
        yield month, frame


def build_events(
    *,
    daily_root: Path,
    supplemental_by_month: dict[str, pd.DataFrame],
    name_lookup: dict[str, str],
    universe_symbols: set[str],
    market_index: dict[pd.Timestamp, dict[str, float]],
    start_date: str,
    end_date: str,
    horizons: tuple[int, ...],
    min_history_bars: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    states: dict[str, SymbolSignalState] = {}
    results: list[dict[str, object]] = []
    rows_processed = 0
    started = time.monotonic()
    for month_number, (month, frame) in enumerate(
        iter_daily_months(
            daily_root=daily_root,
            supplemental_by_month=supplemental_by_month,
            name_lookup=name_lookup,
            universe_symbols=universe_symbols,
            start_date=start_date,
            end_date=end_date,
        ),
        start=1,
    ):
        for row in frame.itertuples(index=False):
            symbol = str(row.ts_code)
            state = states.get(symbol)
            if state is None:
                state = SymbolSignalState(
                    symbol=symbol,
                    horizons=horizons,
                    min_history_bars=min_history_bars,
                )
                states[symbol] = state
            payload = row._asdict()
            trade_date = pd.Timestamp(payload["date"])
            results.extend(state.process_bar(payload, market_index.get(trade_date)))
        rows_processed += len(frame)
        if month_number % 12 == 0 or month == end_date[:6]:
            elapsed = time.monotonic() - started
            print(
                f"event scan: through {month}, rows={rows_processed:,}, "
                f"resolved_rows={len(results):,}, elapsed={elapsed:.1f}s",
                flush=True,
            )
    events = pd.DataFrame(results)
    if not events.empty:
        events = events.sort_values(["signal_date", "symbol", "horizon"]).reset_index(drop=True)
    metadata = {
        "rows_processed": rows_processed,
        "symbols_processed": len(states),
        "resolved_event_horizon_rows": int(len(events)),
        "unique_low9_events": int(
            events[["symbol", "signal_date"]].drop_duplicates().shape[0]
            if not events.empty
            else 0
        ),
    }
    return events, metadata


def apply_primary_filters(frame: pd.DataFrame, minimum_amount_thousand: float) -> pd.DataFrame:
    return frame[
        frame["signal_amount"].ge(minimum_amount_thousand)
        & frame["entry_amount"].gt(0)
        & ~frame["entry_one_price"].fillna(True)
        & frame["abnormal_executable_return"].notna()
    ].copy()


def build_summary(
    events: pd.DataFrame,
    *,
    thresholds: tuple[float, ...],
    horizons: tuple[int, ...],
    cost_bps: float,
    minimum_amount_thousand: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = apply_primary_filters(events, minimum_amount_thousand)
    scopes = {
        "unfiltered": events[events["abnormal_executable_return"].notna()],
        "primary_liquid_tradable": primary,
        "primary_nonoverlap20": primary[
            primary["previous_signal_gap_bars"].isna()
            | primary["previous_signal_gap_bars"].gt(20)
        ],
    }
    summary_rows: list[dict[str, object]] = []
    threshold_specs: list[tuple[str, float | None]] = [("all_low9", None)] + [
        (f"J<={threshold:g}", threshold) for threshold in thresholds
    ]
    for scope_name, scope in scopes.items():
        for horizon in horizons:
            low9_horizon = scope[scope["horizon"].eq(horizon)]
            for label, threshold in threshold_specs:
                subset = (
                    low9_horizon
                    if threshold is None
                    else low9_horizon[low9_horizon["j_value"].le(threshold)]
                )
                row: dict[str, object] = {
                    "scope": scope_name,
                    "threshold_label": label,
                    "j_threshold": threshold,
                    "horizon": horizon,
                }
                row.update(
                    summarize_event_subset(
                        subset, horizon=horizon, round_trip_cost_bps=cost_bps
                    )
                )
                if threshold is not None:
                    row.update(
                        paired_incremental_j_test(
                            low9_horizon,
                            threshold=threshold,
                            horizon=horizon,
                        )
                    )
                summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    primary_tests = summary[
        summary["scope"].eq("primary_liquid_tradable")
        & summary["j_threshold"].notna()
    ].copy()
    summary["hac_q_bh"] = np.nan
    summary.loc[primary_tests.index, "hac_q_bh"] = benjamini_hochberg(
        primary_tests["hac_p"].to_numpy()
    )
    summary["incremental_hac_q_bh"] = np.nan
    summary.loc[primary_tests.index, "incremental_hac_q_bh"] = benjamini_hochberg(
        primary_tests["incremental_hac_p"].to_numpy()
    )

    periods = {
        "2010-2018": (pd.Timestamp("2010-01-01"), pd.Timestamp("2018-12-31")),
        "2019-2022": (pd.Timestamp("2019-01-01"), pd.Timestamp("2022-12-31")),
        "2023-2026": (pd.Timestamp("2023-01-01"), pd.Timestamp("2026-12-31")),
        "validation_2019-2026": (pd.Timestamp("2019-01-01"), pd.Timestamp("2026-12-31")),
    }
    period_rows: list[dict[str, object]] = []
    for period_name, (period_start, period_end) in periods.items():
        period = primary[
            primary["signal_date"].between(period_start, period_end, inclusive="both")
        ]
        for horizon in horizons:
            low9_horizon = period[period["horizon"].eq(horizon)]
            for threshold in thresholds:
                subset = low9_horizon[low9_horizon["j_value"].le(threshold)]
                row = {
                    "period": period_name,
                    "j_threshold": threshold,
                    "horizon": horizon,
                }
                row.update(
                    summarize_event_subset(
                        subset, horizon=horizon, round_trip_cost_bps=cost_bps
                    )
                )
                row.update(
                    paired_incremental_j_test(
                        low9_horizon,
                        threshold=threshold,
                        horizon=horizon,
                    )
                )
                period_rows.append(row)
    period_summary = pd.DataFrame(period_rows)
    period_summary["hac_q_bh_within_period"] = np.nan
    period_summary["incremental_hac_q_bh_within_period"] = np.nan
    for _, indices in period_summary.groupby("period").groups.items():
        index_values = list(indices)
        period_summary.loc[index_values, "hac_q_bh_within_period"] = benjamini_hochberg(
            period_summary.loc[index_values, "hac_p"].to_numpy()
        )
        period_summary.loc[
            index_values, "incremental_hac_q_bh_within_period"
        ] = benjamini_hochberg(
            period_summary.loc[index_values, "incremental_hac_p"].to_numpy()
        )
    return summary, period_summary


def percent(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return "—" if not np.isfinite(number) else f"{number * 100:.2f}%"


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without the optional tabulate package."""

    if frame.empty:
        return "_无数据_"
    display = frame.fillna("—").astype(str)
    headers = [str(column).replace("|", "\\|") for column in display.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in display.itertuples(index=False, name=None):
        values = [str(value).replace("|", "\\|") for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def render_report(
    *,
    summary: pd.DataFrame,
    period_summary: pd.DataFrame,
    metadata: dict[str, object],
    output_dir: Path,
) -> None:
    primary = summary[
        summary["scope"].eq("primary_liquid_tradable")
        & summary["j_threshold"].notna()
    ].copy()
    primary = primary.sort_values(["j_threshold", "horizon"], ascending=[False, True])
    display_columns = [
        "j_threshold",
        "horizon",
        "events",
        "signal_dates",
        "mean_net_return",
        "win_rate",
        "mean_abnormal_net_return",
        "cluster_mean_abnormal_net_return",
        "cluster_ci95_low",
        "cluster_ci95_high",
        "hac_q_bh",
        "incremental_mean",
        "incremental_ci95_low",
        "incremental_ci95_high",
        "incremental_hac_q_bh",
        "prob_mfe_3pct",
    ]
    table = primary[display_columns].copy()
    for column in [
        "mean_net_return",
        "win_rate",
        "mean_abnormal_net_return",
        "cluster_mean_abnormal_net_return",
        "cluster_ci95_low",
        "cluster_ci95_high",
        "incremental_mean",
        "incremental_ci95_low",
        "incremental_ci95_high",
        "prob_mfe_3pct",
    ]:
        table[column] = table[column].map(percent)
    for column in ["hac_q_bh", "incremental_hac_q_bh"]:
        table[column] = table[column].map(
            lambda value: "—" if not np.isfinite(value) else f"{value:.4f}"
        )
    table = table.rename(
        columns={
            "j_threshold": "J阈值",
            "horizon": "持有日",
            "events": "事件数",
            "signal_dates": "信号日数",
            "mean_net_return": "平均净收益",
            "win_rate": "胜率",
            "mean_abnormal_net_return": "平均超额净收益",
            "cluster_mean_abnormal_net_return": "日期等权超额净收益",
            "cluster_ci95_low": "超额95%CI下限",
            "cluster_ci95_high": "超额95%CI上限",
            "hac_q_bh": "超额FDR-q",
            "incremental_mean": "相对其余低9增量",
            "incremental_ci95_low": "增量95%CI下限",
            "incremental_ci95_high": "增量95%CI上限",
            "incremental_hac_q_bh": "增量FDR-q",
            "prob_mfe_3pct": "窗口内触及+3%",
        }
    )
    validation = period_summary[
        period_summary["period"].eq("validation_2019-2026")
        & period_summary["j_threshold"].isin([-10.0, -20.0])
        & period_summary["horizon"].isin([3, 5, 10])
    ].copy()
    validation_table = validation[
        [
            "j_threshold",
            "horizon",
            "events",
            "mean_net_return",
            "mean_abnormal_net_return",
            "cluster_mean_abnormal_net_return",
            "cluster_ci95_low",
            "cluster_ci95_high",
            "hac_q_bh_within_period",
            "incremental_mean",
            "incremental_ci95_low",
            "incremental_ci95_high",
            "incremental_hac_q_bh_within_period",
        ]
    ].copy()
    for column in [
        "mean_net_return",
        "mean_abnormal_net_return",
        "cluster_mean_abnormal_net_return",
        "cluster_ci95_low",
        "cluster_ci95_high",
        "incremental_mean",
        "incremental_ci95_low",
        "incremental_ci95_high",
    ]:
        validation_table[column] = validation_table[column].map(percent)
    markdown = [
        "# 低9 + 日线J负值：A股事件研究",
        "",
        f"生成时间：{pd.Timestamp.now(tz='Asia/Shanghai').isoformat()}",
        "",
        "## 口径",
        "",
        "- 低9：连续9根日线收盘价低于各自4根K线前收盘价，仅第9根完成日触发一次。",
        "- KDJ：9日RSV，K/D按1/3递推平滑，J=3K-2D；价格使用只向后调整的因果连续OHLC。",
        "- 信号在收盘后确认，下一交易日开盘进入，持有1/3/5/10/20根个股K线。",
        "- 主样本要求信号日成交额至少2000万元、次日有成交且非一字价格；收益扣双边合计20bp。",
        "- 超额收益相对上证综指同日开盘至退出日收盘；显著性按信号日期等权聚类，Newey-West滞后等于持有期。",
        "- q值为4个J阈值×5个持有期共20次检验的Benjamini-Hochberg FDR修正。",
        "",
        "## 全样本主结果",
        "",
        dataframe_to_markdown(table),
        "",
        "## 2019年以来验证窗口摘录",
        "",
        dataframe_to_markdown(validation_table),
        "",
        "## 数据审计",
        "",
        "```json",
        json.dumps(metadata, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 边界",
        "",
        "事件研究只能说明历史条件均值，不能证明未来必然反弹。当前本地正式日线对更早退市股票存在缺口；补充抓取状态见元数据。MFE是窗口内事后最大有利波动，只用于描述路径，不是可提前实现的收益。",
    ]
    (output_dir / "report.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--daily-root",
        type=Path,
        default=PROJECT_ROOT / "data/raw/daily_partitioned",
    )
    parser.add_argument(
        "--stock-basic-history",
        type=Path,
        default=PROJECT_ROOT / "data/raw/stock_basic_history.parquet",
    )
    parser.add_argument(
        "--market-index",
        type=Path,
        default=PROJECT_ROOT / "data/raw/index_000001.SH.parquet",
    )
    parser.add_argument(
        "--supplemental-root",
        type=Path,
        default=PROJECT_ROOT / "data/research/low9_kdj_rebound/supplemental_daily",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "reports/research/low9_kdj_rebound",
    )
    parser.add_argument("--start-date", default="20100101")
    parser.add_argument("--end-date", default="20260731")
    parser.add_argument(
        "--horizons",
        type=lambda value: parse_number_list(value, int),
        default=DEFAULT_HORIZONS,
    )
    parser.add_argument(
        "--j-thresholds",
        type=lambda value: parse_number_list(value, float),
        default=DEFAULT_J_THRESHOLDS,
    )
    parser.add_argument("--min-history-bars", type=int, default=60)
    parser.add_argument("--minimum-amount-thousand", type=float, default=20_000.0)
    parser.add_argument("--round-trip-cost-bps", type=float, default=20.0)
    parser.add_argument("--fetch-missing-delist", action="store_true")
    parser.add_argument("--fetch-sleep-seconds", type=float, default=0.15)
    parser.add_argument(
        "--reuse-events",
        action="store_true",
        help="Reuse output-dir/events.parquet and only recompute benchmark returns/statistics",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    universe = load_universe(args.stock_basic_history, args.start_date, args.end_date)
    universe_symbols = set(universe["ts_code"].astype(str))
    name_lookup = dict(zip(universe["ts_code"].astype(str), universe["name"].astype(str)))
    local_symbols = local_symbol_coverage(args.daily_root)
    fetch_metadata: dict[str, object] = {"not_requested": True}
    if args.fetch_missing_delist:
        fetch_metadata = fetch_missing_delisted(
            universe=universe,
            local_symbols=local_symbols,
            supplemental_root=args.supplemental_root,
            start_date=args.start_date,
            end_date=args.end_date,
            sleep_seconds=args.fetch_sleep_seconds,
        )
    market_index, index_metadata = load_market_index(args.market_index)
    existing_metadata_path = args.output_dir / "metadata.json"
    existing_metadata = (
        json.loads(existing_metadata_path.read_text(encoding="utf-8"))
        if existing_metadata_path.exists()
        else {}
    )
    if args.reuse_events:
        events_path = args.output_dir / "events.parquet"
        if not events_path.exists():
            raise FileNotFoundError(f"cannot reuse missing events file: {events_path}")
        events = recompute_event_market_returns(pd.read_parquet(events_path), market_index)
        scan_metadata = existing_metadata.get(
            "event_scan",
            {
                "resolved_event_horizon_rows": int(len(events)),
                "unique_low9_events": int(
                    events[["symbol", "signal_date"]].drop_duplicates().shape[0]
                ),
            },
        )
        supplemental_metadata = existing_metadata.get("supplemental_loaded", {})
        if not args.fetch_missing_delist:
            fetch_metadata = existing_metadata.get("supplemental_fetch", fetch_metadata)
    else:
        supplemental_by_month, supplemental_metadata = load_supplemental_by_month(
            args.supplemental_root, name_lookup
        )
        events, scan_metadata = build_events(
            daily_root=args.daily_root,
            supplemental_by_month=supplemental_by_month,
            name_lookup=name_lookup,
            universe_symbols=universe_symbols,
            market_index=market_index,
            start_date=args.start_date,
            end_date=args.end_date,
            horizons=tuple(sorted(set(args.horizons))),
            min_history_bars=args.min_history_bars,
        )
    if events.empty:
        raise RuntimeError("no completed low-9 events were resolved")
    summary, period_summary = build_summary(
        events,
        thresholds=tuple(args.j_thresholds),
        horizons=tuple(sorted(set(args.horizons))),
        cost_bps=args.round_trip_cost_bps,
        minimum_amount_thousand=args.minimum_amount_thousand,
    )
    events.to_parquet(args.output_dir / "events.parquet", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    period_summary.to_csv(args.output_dir / "period_summary.csv", index=False)
    metadata = {
        "analysis_cutoff": f"{args.end_date[:4]}-{args.end_date[4:6]}-{args.end_date[6:]}T18:00:00+08:00",
        "source": "project canonical Tushare raw daily partitions plus Tushare supplemental delisted histories",
        "universe": {
            "ordinary_a_history_rows": int(len(universe)),
            "listed": int(universe["list_status"].eq("L").sum()),
            "delisted": int(universe["list_status"].eq("D").sum()),
            "local_symbols": int(len(local_symbols & universe_symbols)),
            "delisted_missing_before_supplement": int(
                (
                    universe["list_status"].eq("D")
                    & ~universe["ts_code"].isin(local_symbols)
                ).sum()
            ),
        },
        "supplemental_fetch": fetch_metadata,
        "supplemental_loaded": supplemental_metadata,
        "market_index": index_metadata,
        "event_scan": scan_metadata,
        "parameters": {
            "start_date": args.start_date,
            "end_date": args.end_date,
            "horizons": list(args.horizons),
            "j_thresholds": list(args.j_thresholds),
            "min_history_bars": args.min_history_bars,
            "minimum_amount_thousand_yuan": args.minimum_amount_thousand,
            "round_trip_cost_bps": args.round_trip_cost_bps,
            "low9_rule": "nine consecutive close[t] < close[t-4], trigger only when count == 9",
            "entry_rule": "next available stock trading day open",
            "price_continuity": "causal forward-only factor *= previous_raw_close / current_pre_close",
            "inference": "equal-weight signal-date cohorts; Newey-West lag=horizon; BH-FDR over 20 primary tests",
        },
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    render_report(
        summary=summary,
        period_summary=period_summary,
        metadata=metadata,
        output_dir=args.output_dir,
    )
    print(f"wrote analysis to {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
