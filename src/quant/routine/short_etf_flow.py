"""Refresh and publish market-wide ETF flow context for the short workspace."""
from __future__ import annotations

import argparse
from datetime import datetime
import math
import fcntl
import json
import logging
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from quant.data.atomic_io import atomic_write_json, atomic_write_parquet
from quant.data.long_factor_backfill import RequestPolicy, load_open_trade_dates, request_frame
from quant.features.ownership_flow import ETF_BASKETS, ETF_GROUPS, ETF_MEASURES, dates, etf_flow_features
from quant.infrastructure.publication import publication_path
from quant.routine.paths import PROJECT_ROOT

SNAPSHOT_DIRECTORY = "data/features/short_etf_flow"
CACHE_DIRECTORY = "data/raw/short_etf_flow"
SCHEMA_VERSION = "short-etf-flow-v1"
SOURCES = {
    "fund_share": ("trade_date", ("fd_share",)),
    "fund_nav": ("nav_date", ("ann_date", "unit_nav")),
    "fund_daily": ("trade_date", ("amount",)),
    "fund_adj": ("trade_date", ("adj_factor",)),
}
LABELS = {
    "all": "重点宽基合计", "csi300": "沪深300", "sse50": "上证50", "csi500": "中证500",
    "csi1000": "中证1000", "star50": "科创50", "chinext": "创业板指",
    "chinext50": "创业板50", "csi_a500": "中证A500",
}
REQUEST_POLICY = RequestPolicy(retries=2, max_retry_wait_seconds=3)
logger = logging.getLogger(__name__)


def _path(project_root: Path, date: pd.Timestamp) -> Path:
    return project_root / SNAPSHOT_DIRECTORY / f"{date:%Y%m%d}.json"


def read_short_etf_flow(signal_date: str | None, *, project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Read only a matching committed signal date, including on historical replay."""
    try:
        target = pd.Timestamp(signal_date)
        if pd.isna(target):
            raise ValueError("missing signal date")
        payload = json.loads(publication_path(_path(project_root, target)).read_text(encoding="utf-8"))
        if (isinstance(payload, dict) and payload.get("schema_version") == SCHEMA_VERSION
                and payload.get("signal_date") == target.date().isoformat()
                and payload.get("status") == "success"):
            return payload
    except (OSError, ValueError, TypeError):
        pass
    return {"status": "unavailable"}


def _read_cache(path: Path, columns: list[str]) -> pd.DataFrame:
    try:
        return pd.read_parquet(path, columns=columns)
    except FileNotFoundError:
        return pd.DataFrame(columns=columns)


def _refresh_series(
    pro: Any, cache: Path, api: str, code: str, start: pd.Timestamp, target: pd.Timestamp,
) -> pd.DataFrame:
    date_col, values = SOURCES[api]
    columns = ["ts_code", date_col, *values]
    path = cache / api / f"{code}.parquet"
    old = _read_cache(path, columns)
    old_dates = dates(old[date_col])
    # Cached coverage on both ends is necessary before restricting the request.
    if old_dates.le(start + pd.Timedelta(days=7)).any() and old_dates.le(target).any():
        last = old_dates[old_dates.le(target)].max()
        start_request = max(start, min(last, target - pd.Timedelta(days=14)))
    else:
        start_request = start
    params = {"ts_code": code, "start_date": start_request.strftime("%Y%m%d"),
              "end_date": target.strftime("%Y%m%d"), "fields": ",".join(columns)}
    if api == "fund_nav":
        params["market"] = "E"
    fresh = request_frame(lambda: getattr(pro, api)(**params), REQUEST_POLICY)
    if not fresh.empty:
        if not set(columns).issubset(fresh):
            raise ValueError(f"{api} response schema changed")
        fresh_dates = dates(fresh[date_col])
        if not (fresh.ts_code.eq(code).all() and fresh_dates.between(start_request, target).all()):
            raise ValueError(f"{api} returned rows outside the request for {code}")
        key = [date_col, "ann_date"] if api == "fund_nav" else [date_col]
        fresh = fresh[columns].drop_duplicates()
        if fresh.duplicated(key).any():
            raise ValueError(f"{api} returned conflicting records for {code}")
        combined = pd.concat([old, fresh], ignore_index=True) if not old.empty else fresh
        old = combined.drop_duplicates(key, keep="last").sort_values(date_col)
        atomic_write_parquet(old, path, index=False)
    return old.loc[dates(old[date_col]).between(start, target), columns]


def _build(pro: Any, target: pd.Timestamp, cache: Path) -> dict[str, Any]:
    target_text = target.strftime("%Y%m%d")
    start = target - pd.Timedelta(days=185)
    raw_calendar = request_frame(
        lambda: pd.DataFrame({"date": load_open_trade_dates(pro, start.strftime("%Y%m%d"), target_text)}),
        REQUEST_POLICY,
    )
    calendar = pd.DatetimeIndex(dates(raw_calendar.date))
    if len(calendar) < 65 or calendar[-1] != target:
        raise ValueError("ETF flow requires an exchange session and at least 65 calendar observations")
    listings_path = cache / "listings.parquet"
    listing_cols = ["ts_code", "index_code", "list_date"]
    listings = _read_cache(listings_path, listing_cols)
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    if listings.empty or now.timestamp() - listings_path.stat().st_mtime > 7 * 86400:
        fetched = request_frame(lambda: pro.etf_basic(fields=",".join(listing_cols)), REQUEST_POLICY)
        if not set(listing_cols).issubset(fetched) or fetched.ts_code.duplicated().any():
            raise ValueError("Invalid ETF listings response")
        listings = fetched[listing_cols]
        atomic_write_parquet(listings, listings_path, index=False)
    metadata = listings.set_index("ts_code")
    for index_code, symbols in ETF_BASKETS.values():
        if not metadata.index_code.reindex(symbols).eq(index_code).all():
            raise ValueError(f"ETF index assignment changed for {symbols}")
    codes = [code for code in sorted(metadata.index.intersection(
        {code for group in ETF_GROUPS.values() for code in group}
    )) if pd.Timestamp(metadata.loc[code, "list_date"]) <= target]
    if not codes:
        raise ValueError("No listed ETF in the selected baskets")
    data = {
        api: pd.concat([_refresh_series(pro, cache, api, code, start, target) for code in codes], ignore_index=True)
        for api in SOURCES
    }
    data["fund_nav"] = data["fund_nav"].loc[dates(data["fund_nav"].ann_date).le(target)]
    features, details = etf_flow_features(
        data["fund_share"], data["fund_nav"], data["fund_daily"], data["fund_adj"],
        listings, ETF_GROUPS, calendar, include_observation_dates=True,
    )
    source_dates = {}
    for column in ("fd_share", "unit_nav", "amount", "adj_factor"):
        latest = details.loc[details[column].notna()].groupby("code").trade_date.max().reindex(codes)
        expected = target if column == "amount" else calendar[-3]
        if latest.isna().any() or latest.lt(expected).any():
            raise ValueError(f"Stale {column} for {latest.index[latest.isna() | latest.lt(expected)].tolist()}")
        source_dates[column] = latest.min().date().isoformat()
    latest = features.iloc[-1].drop("date")
    observed_through = {name: value.date().isoformat() if pd.notna(value) else None
                        for name, value in latest.items() if name.endswith("_through")}
    values = {name: float(value) if pd.notna(value) and math.isfinite(float(value)) else None
              for name, value in latest.items() if not name.endswith("_through")}
    return {
        "status": "success", "schema_version": SCHEMA_VERSION, "source": "Tushare",
        "signal_date": target.date().isoformat(), "observed_at": now.isoformat(),
        "source_dates": source_dates, "features": values,
        "baskets": [
            {"name": name, "label": label, "coverage_d2": values[f"etf_{name}_coverage"],
             "through_5": observed_through[f"etf_{name}_flow_yuan_5_through"],
             "through_20": observed_through[f"etf_{name}_flow_yuan_20_through"],
             "ratio_through_5": observed_through[f"etf_{name}_flow_pct_5_through"],
             "ratio_through_20": observed_through[f"etf_{name}_flow_pct_20_through"],
             "days_5": values[f"etf_{name}_flow_yuan_5_days"],
             "days_20": values[f"etf_{name}_flow_yuan_20_days"],
             **{metric: values[f"etf_{name}_{metric}"] for metric in ETF_MEASURES}}
            for name, label in LABELS.items()
        ],
    }


def refresh_short_etf_flow(
    trade_date: str, *, project_root: Path = PROJECT_ROOT, pro: Any = None,
) -> dict[str, Any]:
    """Keep existing committed context readable when an independent feed fails."""
    target = pd.Timestamp(trade_date).normalize()
    cache = project_root / CACHE_DIRECTORY
    try:
        cache.mkdir(parents=True, exist_ok=True)
        with (cache / ".refresh.lock").open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if pro is None:
                from quant.data.tushare_fetcher import TushareDataFetcher

                pro = TushareDataFetcher(cache_dir=cache).pro
            payload = _build(pro, target, cache)
            atomic_write_json(payload, _path(project_root, target))
        return {key: payload[key] for key in ("status", "signal_date", "source_dates")}
    except Exception as exc:
        logger.warning("ETF flow context unavailable for %s (%s)", target.date(), type(exc).__name__)
        return {"status": "degraded", "signal_date": target.date().isoformat(),
                "error_type": type(exc).__name__,
                "same_date_snapshot_retained": read_short_etf_flow(trade_date, project_root=project_root)["status"] == "success"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trade_date")
    args = parser.parse_args()
    result = refresh_short_etf_flow(args.trade_date)
    print(json.dumps(result, ensure_ascii=False))
    if result["status"] != "success":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
