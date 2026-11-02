"""Materialize all166 registered market/active-value/candle research factors."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from audit_overnight_right_provenance import sha256

from quant.core.paths import PROJECT_ROOT
from quant.features.active_market_value import (
    KEY_INDEX_SPECS,
    compute_key_index_active_market_value_features,
    compute_market_active_market_value_features,
    compute_stock_active_market_value_features,
)
from quant.features.candlestick_context import (
    CANDLE_CONTEXT_RESEARCH_FEATURE_COLUMNS,
    compute_candlestick_context_features,
)
from quant.features.factor_registry import registry_frame
from quant.features.market_breadth import compute_market_breadth_features


def main() -> None:
    root = PROJECT_ROOT / "reports/research"
    output = root / "right_t012_research_factors_v3"
    output.mkdir(exist_ok=False)
    bars_path = root / "overnight_unfiltered_universe_v1/daily_bars.parquet"
    basic_path = root / "overnight_unfiltered_full_factors_v1/dataset.parquet"
    index_root = root / "right_t012_index_sources_v1"
    paths = [bars_path, basic_path, *sorted(index_root.glob("*.parquet")),
             PROJECT_ROOT / "src/quant/features/active_market_value.py",
             PROJECT_ROOT / "src/quant/features/market_breadth.py",
             PROJECT_ROOT / "src/quant/features/candlestick_context.py"]
    hashes = {str(p): sha256(p) for p in paths}
    bars = pd.read_parquet(bars_path).sort_values(["symbol", "date"]).reset_index(drop=True)
    date_text = {d: pd.Timestamp(d).strftime("%Y%m%d") for d in bars.date.unique()}
    bars["trade_date"] = bars.date.map(date_text)
    basic = pd.read_parquet(basic_path, columns=["symbol", "date", "turnover_rate", "circ_mv"])
    basic["trade_date"] = basic.date.map(date_text)
    stock = compute_stock_active_market_value_features(basic)
    stock = stock.rename(columns={"ts_code": "symbol"})
    print("Stock active-value factors complete", flush=True)
    market = compute_market_active_market_value_features(basic, market_daily=bars,
                    stock_features=stock.rename(columns={"symbol": "ts_code"}))
    breadth = compute_market_breadth_features(bars)
    market = market.merge(breadth.drop(columns="trade_date"), on="date", validate="one_to_one")
    print("Market active-value/breadth factors complete", flush=True)
    expected_dates = pd.DatetimeIndex(bars.date.unique()).sort_values()
    index_bars, index_basic, coverage = [], [], []
    for spec in KEY_INDEX_SPECS:
        daily = pd.read_parquet(index_root / f"{spec.ts_code}_index_daily.parquet")
        dates = pd.DatetimeIndex(pd.to_datetime(daily.trade_date))
        missing = expected_dates.difference(dates)
        if len(missing):
            raise ValueError(f"Index daily history incomplete: {spec.ts_code}, missing={len(missing)}")
        index_bars.append(daily)
        basic_file = index_root / f"{spec.ts_code}_index_dailybasic.parquet"
        if basic_file.exists():
            ib = pd.read_parquet(basic_file)
            basic_missing = expected_dates.difference(pd.DatetimeIndex(pd.to_datetime(ib.trade_date)))
            index_basic.append(ib)
            coverage.append(dict(symbol=spec.ts_code, daily_missing=0, basic_missing=len(basic_missing)))
        else:
            coverage.append(dict(symbol=spec.ts_code, daily_missing=0, basic_missing=len(expected_dates)))
    indices = compute_key_index_active_market_value_features(pd.concat(index_bars), pd.concat(index_basic))
    market = market.merge(indices.drop(columns="trade_date"), on="date", how="left", validate="one_to_one")
    candle_parts = []
    for _, b in bars.groupby("symbol", sort=False):
        b = b.reset_index(drop=True).rename(columns={"vol": "volume"})
        c = compute_candlestick_context_features(b)
        candle_parts.append(pd.concat([b[["symbol", "date"]], c[list(CANDLE_CONTEXT_RESEARCH_FEATURE_COLUMNS)]], axis=1))
    candle = pd.concat(candle_parts, ignore_index=True)
    stock = stock.merge(candle, on=["symbol", "date"], validate="one_to_one")
    numeric = registry_frame().query("role == 'research_feature'").name.tolist()
    observed = set(stock) | set(market)
    if set(numeric) - observed:
        raise ValueError(f"Unmaterialized registered columns: {set(numeric) - observed}")
    for frame in [stock, market]:
        cols = sorted(set(numeric) & set(frame))
        frame[cols] = frame[cols].replace([np.inf, -np.inf], np.nan).astype("float32")
    stock.to_parquet(output / "stock.parquet", index=False)
    market.to_parquet(output / "market.parquet", index=False)
    rows = []
    for name in numeric:
        f = stock if name in stock else market
        rows.append(dict(name=name, rows=len(f), known=int(f[name].notna().sum()),
                         unique=int(f[name].nunique()), source="stock" if name in stock else "market"))
    pd.DataFrame(rows).to_csv(output / "coverage.csv", index=False)
    for path, digest in hashes.items():
        if sha256(Path(path)) != digest:
            raise ValueError(f"Source changed: {path}")
    (output / "complete.json").write_text(json.dumps(dict(sources=hashes, registry_count=len(numeric),
        informative=sum(r["unique"] > 1 for r in rows), index_coverage=coverage,
        stock_sha256=sha256(output / "stock.parquet"), market_sha256=sha256(output / "market.parquet"),
        timing="Stock/market active MV and index dailybasic-dependent variables must be lagged one global session before close entry",
        warmup="Stock/market input begins2024; early rolling values may be missing; indices use2018warmup",
        units_reference="https://tushare.pro/document/2?doc_id=128; index float_mv=CNY, stock circ_mv=10000CNY",
        limitations="Provider revisions and historical constituent fallback not reconstructed; unavailable indices retained missing",
        script_sha256=sha256(Path(__file__))), indent=2))
    print(f"Materialized{len(numeric)} registered research factors; informative={sum(r['unique'] > 1 for r in rows)}", flush=True)


if __name__ == "__main__":
    main()
