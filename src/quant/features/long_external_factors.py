"""Point-in-time external factors for weekly long-entry research.

The builders in this module intentionally accept the weekly signal keys as
their left-hand universe.  This keeps the research dataset compact while each
source still observes a strict ``source_date <= signal_date`` contract.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


EXTERNAL_FACTOR_VERSION = "long-external-v1-pit"


def _numeric(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def _rolling_sum(frame: pd.DataFrame, column: str, window: int, minimum: int) -> pd.Series:
    return frame.groupby("ts_code", sort=False)[column].transform(
        lambda values: values.rolling(window, min_periods=minimum).sum()
    )


def _read_partitioned_source(
    directory: Path,
    *,
    columns: list[str],
    symbols: set[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Read a date-partitioned parquet source with Arrow pushdown when possible."""

    paths = sorted(directory.glob("*.parquet"))
    if not paths:
        return pd.DataFrame(columns=columns)
    start_text = start.strftime("%Y%m%d")
    end_text = end.strftime("%Y%m%d")
    try:
        import pyarrow.dataset as ds

        dataset = ds.dataset([str(path) for path in paths], format="parquet")
        available = set(dataset.schema.names)
        selected = [column for column in columns if column in available]
        predicate = (
            (ds.field("trade_date") >= start_text)
            & (ds.field("trade_date") <= end_text)
            & ds.field("ts_code").isin(sorted(symbols))
        )
        return dataset.to_table(columns=selected, filter=predicate).to_pandas()
    except Exception:
        frames: list[pd.DataFrame] = []
        for path in paths:
            date_text = path.stem.rsplit("_", 1)[-1]
            date = pd.to_datetime(date_text, format="%Y%m%d", errors="coerce")
            if pd.isna(date) or date < start or date > end:
                continue
            available = pd.read_parquet(path).columns
            selected = [column for column in columns if column in available]
            part = pd.read_parquet(path, columns=selected)
            if part.empty or "ts_code" not in part.columns:
                continue
            part = part[part["ts_code"].astype(str).isin(symbols)]
            if not part.empty:
                frames.append(part)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)


def build_moneyflow_weekly(signal_keys: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "large_net_amount_ratio",
        "large_net_3d_ratio",
        "large_net_5d_ratio",
        "moneyflow_net_ratio",
        "small_net_amount_ratio",
        "medium_net_amount_ratio",
        "large_flow_persistence_5d",
    ]
    if source.empty:
        return signal_keys.assign(**{column: np.nan for column in columns})
    frame = source.copy()
    frame["date"] = pd.to_datetime(frame["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
    amount_columns = [
        "buy_sm_amount", "sell_sm_amount", "buy_md_amount", "sell_md_amount",
        "buy_lg_amount", "sell_lg_amount", "buy_elg_amount", "sell_elg_amount",
        "net_mf_amount",
    ]
    _numeric(frame, amount_columns)
    frame = frame.dropna(subset=["date", "ts_code"]).sort_values(["ts_code", "date"])
    zero = pd.Series(0.0, index=frame.index)
    get = lambda name: frame[name].fillna(0.0) if name in frame.columns else zero
    frame["small_net"] = get("buy_sm_amount") - get("sell_sm_amount")
    frame["medium_net"] = get("buy_md_amount") - get("sell_md_amount")
    frame["large_net"] = (
        get("buy_lg_amount") + get("buy_elg_amount")
        - get("sell_lg_amount") - get("sell_elg_amount")
    )
    frame["flow_gross"] = sum((get(column) for column in amount_columns[:-1]), start=zero.copy())
    frame["large_net_amount_ratio"] = _safe_ratio(frame["large_net"], frame["flow_gross"])
    frame["moneyflow_net_ratio"] = _safe_ratio(get("net_mf_amount"), frame["flow_gross"])
    frame["small_net_amount_ratio"] = _safe_ratio(frame["small_net"], frame["flow_gross"])
    frame["medium_net_amount_ratio"] = _safe_ratio(frame["medium_net"], frame["flow_gross"])
    for window, minimum in ((3, 2), (5, 3)):
        numerator = _rolling_sum(frame, "large_net", window, minimum)
        denominator = _rolling_sum(frame, "flow_gross", window, minimum)
        frame[f"large_net_{window}d_ratio"] = _safe_ratio(numerator, denominator)
    positive = frame["large_net"].gt(0).astype(float)
    frame["large_flow_persistence_5d"] = positive.groupby(frame["ts_code"], sort=False).transform(
        lambda values: values.rolling(5, min_periods=3).mean()
    )
    daily = frame[["date", "ts_code", *columns]].drop_duplicates(["date", "ts_code"], keep="last")
    return signal_keys.merge(daily, on=["date", "ts_code"], how="left")


def build_margin_weekly(signal_keys: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "margin_balance",
        "margin_balance_change",
        "margin_buy_ratio_5d",
        "short_balance",
        "short_pressure_change_5d",
    ]
    if source.empty:
        return signal_keys.assign(**{column: np.nan for column in columns})
    frame = source.copy()
    frame["date"] = pd.to_datetime(frame["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
    _numeric(frame, ["rzye", "rqye", "rzmre", "rzche"])
    frame = frame.dropna(subset=["date", "ts_code"]).sort_values(["ts_code", "date"])
    frame["margin_balance"] = frame.get("rzye")
    frame["short_balance"] = frame.get("rqye")
    frame["margin_balance_change"] = frame.groupby("ts_code", sort=False)["margin_balance"].pct_change(20)
    margin_buy = _rolling_sum(frame, "rzmre", 5, 3)
    margin_sell = _rolling_sum(frame, "rzche", 5, 3)
    frame["margin_buy_ratio_5d"] = _safe_ratio(margin_buy - margin_sell, margin_buy + margin_sell)
    frame["short_pressure_change_5d"] = frame.groupby("ts_code", sort=False)["short_balance"].pct_change(5)
    daily = frame[["date", "ts_code", *columns]].drop_duplicates(["date", "ts_code"], keep="last")
    return signal_keys.merge(daily, on=["date", "ts_code"], how="left")


def _window_bounds(event_dates: np.ndarray, signal_date: pd.Timestamp, lookback_date: pd.Timestamp) -> tuple[int, int]:
    right = int(np.searchsorted(event_dates, np.datetime64(signal_date), side="right"))
    left = int(np.searchsorted(event_dates, np.datetime64(lookback_date), side="left"))
    return left, right


def build_top_list_weekly(
    signal_keys: pd.DataFrame,
    source: pd.DataFrame,
    market_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    columns = [
        "top_list_count_20d",
        "top_list_net_ratio_20d",
        "top_list_positive_days_20d",
        "top_list_reason_concentration_60d",
    ]
    if source.empty:
        return signal_keys.assign(**{column: 0.0 for column in columns})
    frame = source.copy()
    frame["date"] = pd.to_datetime(frame["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
    _numeric(frame, ["net_amount", "amount"])
    frame = frame.dropna(subset=["date", "ts_code"])
    frame["reason"] = frame.get("reason", "未知").fillna("未知").astype(str)
    by_symbol = {str(code): group.sort_values("date") for code, group in frame.groupby("ts_code", sort=False)}
    calendar = np.asarray(pd.DatetimeIndex(market_dates).sort_values().unique(), dtype="datetime64[ns]")
    rows: list[dict[str, object]] = []
    for code, signals in signal_keys.groupby("ts_code", sort=False):
        events = by_symbol.get(str(code))
        for signal in signals.itertuples(index=False):
            signal_date = pd.Timestamp(signal.date)
            calendar_index = int(np.searchsorted(calendar, np.datetime64(signal_date), side="right")) - 1
            if calendar_index < 0 or events is None:
                rows.append({"date": signal_date, "ts_code": code, **{column: 0.0 for column in columns}})
                continue
            start20 = pd.Timestamp(calendar[max(0, calendar_index - 19)])
            start60 = pd.Timestamp(calendar[max(0, calendar_index - 59)])
            event_dates = events["date"].to_numpy(dtype="datetime64[ns]")
            left20, right = _window_bounds(event_dates, signal_date, start20)
            left60, _ = _window_bounds(event_dates, signal_date, start60)
            recent20 = events.iloc[left20:right]
            recent60 = events.iloc[left60:right]
            net = pd.to_numeric(recent20.get("net_amount"), errors="coerce").sum(min_count=1)
            amount = pd.to_numeric(recent20.get("amount"), errors="coerce").sum(min_count=1)
            positive_days = recent20.loc[pd.to_numeric(recent20.get("net_amount"), errors="coerce") > 0, "date"].nunique()
            reason_concentration = recent60["reason"].value_counts(normalize=True).max() if not recent60.empty else 0.0
            rows.append(
                {
                    "date": signal_date,
                    "ts_code": code,
                    "top_list_count_20d": float(len(recent20)),
                    "top_list_net_ratio_20d": float(net / amount) if pd.notna(net) and pd.notna(amount) and amount else 0.0,
                    "top_list_positive_days_20d": float(positive_days),
                    "top_list_reason_concentration_60d": float(reason_concentration),
                }
            )
    return pd.DataFrame(rows)


def build_holder_weekly(signal_keys: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "holder_net_change_ratio_30d",
        "holder_net_change_ratio_90d",
        "holder_buy_event_count_180d",
        "holder_avg_price_gap",
        "holder_after_ratio_change_180d",
    ]
    if source.empty:
        return signal_keys.assign(**{column: np.nan for column in columns})
    frame = source.copy()
    frame["date"] = pd.to_datetime(frame["ann_date"].astype(str), format="%Y%m%d", errors="coerce")
    _numeric(frame, ["change_ratio", "change_vol", "avg_price"])
    frame = frame.dropna(subset=["date", "ts_code"])
    direction = frame["in_de"].astype(str).map({"IN": 1.0, "DE": -1.0})
    frame["signed_change_ratio"] = pd.to_numeric(frame["change_ratio"], errors="coerce") * direction
    by_symbol = {str(code): group.sort_values("date") for code, group in frame.groupby("ts_code", sort=False)}
    rows: list[dict[str, object]] = []
    for code, signals in signal_keys.groupby("ts_code", sort=False):
        events = by_symbol.get(str(code))
        for signal in signals.itertuples(index=False):
            signal_date = pd.Timestamp(signal.date)
            empty = {
                "holder_net_change_ratio_30d": 0.0,
                "holder_net_change_ratio_90d": 0.0,
                "holder_buy_event_count_180d": 0.0,
                "holder_avg_price_gap": np.nan,
                "holder_after_ratio_change_180d": 0.0,
            }
            if events is None:
                rows.append({"date": signal_date, "ts_code": code, **empty})
                continue
            dates = events["date"].to_numpy(dtype="datetime64[ns]")
            windows: dict[int, pd.DataFrame] = {}
            for days in (30, 90, 180):
                left, right = _window_bounds(dates, signal_date, signal_date - pd.Timedelta(days=days))
                windows[days] = events.iloc[left:right]
            recent180 = windows[180]
            average_price = pd.to_numeric(recent180.get("avg_price"), errors="coerce").median()
            close = float(getattr(signal, "close", np.nan))
            rows.append(
                {
                    "date": signal_date,
                    "ts_code": code,
                    "holder_net_change_ratio_30d": windows[30]["signed_change_ratio"].sum(),
                    "holder_net_change_ratio_90d": windows[90]["signed_change_ratio"].sum(),
                    "holder_buy_event_count_180d": float(windows[180]["in_de"].astype(str).eq("IN").sum()),
                    "holder_avg_price_gap": close / average_price - 1.0 if np.isfinite(close) and pd.notna(average_price) and average_price > 0 else np.nan,
                    "holder_after_ratio_change_180d": recent180["signed_change_ratio"].sum(),
                }
            )
    return pd.DataFrame(rows)


def build_pledge_weekly(signal_keys: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "pledge_ratio",
        "pledge_ratio_change_13w",
        "pledge_ratio_change_52w",
        "pledge_event_count_26w",
        "pledge_release_ratio_26w",
    ]
    if source.empty:
        return signal_keys.assign(**{column: np.nan for column in columns})
    frame = source.copy()
    frame["date"] = pd.to_datetime(frame["end_date"].astype(str), format="%Y%m%d", errors="coerce")
    _numeric(frame, ["pledge_ratio", "pledge_count"])
    frame = frame.dropna(subset=["date", "ts_code", "pledge_ratio"]).sort_values(["ts_code", "date"])
    frame = frame.drop_duplicates(["ts_code", "date"], keep="last")
    by_symbol = {str(code): group for code, group in frame.groupby("ts_code", sort=False)}
    rows: list[dict[str, object]] = []
    for code, signals in signal_keys.groupby("ts_code", sort=False):
        history = by_symbol.get(str(code))
        for signal in signals.itertuples(index=False):
            signal_date = pd.Timestamp(signal.date)
            if history is None:
                rows.append({"date": signal_date, "ts_code": code, **{column: np.nan for column in columns}})
                continue
            dates = history["date"].to_numpy(dtype="datetime64[ns]")
            ratios = history["pledge_ratio"].to_numpy(dtype=float)
            current_index = int(np.searchsorted(dates, np.datetime64(signal_date), side="right")) - 1
            if current_index < 0:
                rows.append({"date": signal_date, "ts_code": code, **{column: np.nan for column in columns}})
                continue
            def at_or_before(days: int) -> float:
                cutoff = np.datetime64(signal_date - pd.Timedelta(days=days))
                index = int(np.searchsorted(dates, cutoff, side="right")) - 1
                return float(ratios[index]) if index >= 0 else np.nan
            current = float(ratios[current_index])
            old13 = at_or_before(91)
            old52 = at_or_before(365)
            start26 = np.datetime64(signal_date - pd.Timedelta(days=182))
            left = int(np.searchsorted(dates, start26, side="left"))
            recent = ratios[left : current_index + 1]
            changes = np.diff(recent) if len(recent) > 1 else np.array([], dtype=float)
            released = float(np.maximum(-changes, 0).sum()) if len(changes) else 0.0
            rows.append(
                {
                    "date": signal_date,
                    "ts_code": code,
                    "pledge_ratio": current,
                    "pledge_ratio_change_13w": current - old13 if np.isfinite(old13) else np.nan,
                    "pledge_ratio_change_52w": current - old52 if np.isfinite(old52) else np.nan,
                    "pledge_event_count_26w": float(np.count_nonzero(changes)),
                    "pledge_release_ratio_26w": released / old13 if np.isfinite(old13) and old13 > 0 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def build_weekly_external_factor_cache(
    signals: pd.DataFrame,
    *,
    raw_dir: Path,
    cache_path: Path,
    manifest_path: Path,
    force: bool = False,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build or read the unified weekly external-factor cache."""

    if cache_path.exists() and manifest_path.exists() and not force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("version") == EXTERNAL_FACTOR_VERSION:
            cached = pd.read_parquet(cache_path)
            cached["date"] = pd.to_datetime(cached["date"])
            return cached, {**manifest, "cache_hit": True}

    keys = signals[["date", "ts_code", "close"]].copy()
    keys["date"] = pd.to_datetime(keys["date"])
    keys = keys.drop_duplicates(["date", "ts_code"]).sort_values(["date", "ts_code"])
    symbols = set(keys["ts_code"].astype(str))
    start = keys["date"].min() - pd.Timedelta(days=60)
    end = keys["date"].max()
    market_dates = pd.DatetimeIndex(keys["date"].unique()).sort_values()

    moneyflow_raw = _read_partitioned_source(
        raw_dir / "moneyflow",
        columns=[
            "ts_code", "trade_date", "buy_sm_amount", "sell_sm_amount", "buy_md_amount",
            "sell_md_amount", "buy_lg_amount", "sell_lg_amount", "buy_elg_amount",
            "sell_elg_amount", "net_mf_amount",
        ],
        symbols=symbols, start=start, end=end,
    )
    if not moneyflow_raw.empty and "trade_date" in moneyflow_raw.columns:
        source_market_dates = pd.to_datetime(
            moneyflow_raw["trade_date"].astype(str), format="%Y%m%d", errors="coerce"
        ).dropna().unique()
        if len(source_market_dates):
            market_dates = pd.DatetimeIndex(source_market_dates).sort_values()
    result = build_moneyflow_weekly(keys[["date", "ts_code"]], moneyflow_raw)
    del moneyflow_raw

    margin_raw = _read_partitioned_source(
        raw_dir / "margin_detail",
        columns=["ts_code", "trade_date", "rzye", "rqye", "rzmre", "rzche"],
        symbols=symbols, start=start, end=end,
    )
    margin = build_margin_weekly(keys[["date", "ts_code"]], margin_raw)
    result = result.merge(margin, on=["date", "ts_code"], how="left")
    del margin_raw, margin

    top_raw = _read_partitioned_source(
        raw_dir / "top_list",
        columns=["ts_code", "trade_date", "net_amount", "amount", "reason"],
        symbols=symbols, start=start, end=end,
    )
    top = build_top_list_weekly(keys[["date", "ts_code"]], top_raw, market_dates)
    result = result.merge(top, on=["date", "ts_code"], how="left")
    del top_raw, top

    holder_path = raw_dir / "holder_trade.parquet"
    holder_raw = pd.read_parquet(holder_path) if holder_path.exists() else pd.DataFrame()
    if not holder_raw.empty:
        holder_raw = holder_raw[holder_raw["ts_code"].astype(str).isin(symbols)]
    holder = build_holder_weekly(keys, holder_raw)
    result = result.merge(holder, on=["date", "ts_code"], how="left")
    del holder_raw, holder

    pledge_path = raw_dir / "pledge_stat.parquet"
    pledge_raw = pd.read_parquet(pledge_path) if pledge_path.exists() else pd.DataFrame()
    if not pledge_raw.empty:
        pledge_raw = pledge_raw[pledge_raw["ts_code"].astype(str).isin(symbols)]
    pledge = build_pledge_weekly(keys[["date", "ts_code"]], pledge_raw)
    result = result.merge(pledge, on=["date", "ts_code"], how="left")
    del pledge_raw, pledge

    result = result.sort_values(["date", "ts_code"]).replace([np.inf, -np.inf], np.nan)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(f".{os.getpid()}.tmp.parquet")
    result.to_parquet(temporary, index=False)
    os.replace(temporary, cache_path)
    factor_columns = [column for column in result.columns if column not in {"date", "ts_code"}]
    coverage = {column: float(result[column].notna().mean()) for column in factor_columns}
    manifest: dict[str, object] = {
        "status": "success",
        "version": EXTERNAL_FACTOR_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "rows": int(len(result)),
        "symbols": int(result["ts_code"].nunique()),
        "date_min": result["date"].min().date().isoformat(),
        "date_max": result["date"].max().date().isoformat(),
        "factor_columns": factor_columns,
        "coverage": coverage,
        "point_in_time": {
            "daily": "trade_date <= signal_date; exact weekly close join",
            "holder": "ann_date <= signal_date",
            "pledge": "end_date <= signal_date",
        },
        "cache_hit": False,
    }
    temporary_manifest = manifest_path.with_suffix(f".{os.getpid()}.tmp.json")
    temporary_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary_manifest, manifest_path)
    return result, manifest
