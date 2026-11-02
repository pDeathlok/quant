"""Redesign right samples and join the complete historical numeric registry."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from audit_overnight_right_provenance import sha256
from train_overnight_unfiltered_comparison import direct_exit_flags
from train_right_t012 import KEYS, map_entries

from quant.core.paths import PROJECT_ROOT
from quant.features.factor_registry import registry_frame
from quant.features.long_weekly_factors import (
    MONTHLY_VALUATION_HISTORY_COLUMNS,
    add_monthly_valuation_history,
)
from quant.features.market_sentiment import read_top_list_features
from quant.features.variable_library import PROJECT_FACTOR_COLUMNS
from quant.research.short_side_groups import GROUP_MEMBERS


def asof_history(keys: pd.DataFrame, source: pd.DataFrame, tolerance: int = 14) -> pd.DataFrame:
    """Join strictly before the entry session; preserve a source-date audit."""
    if source.duplicated(KEYS).any():
        raise ValueError("Duplicate historical factor keys")
    right = source.rename(columns={"date": "source_date"}).sort_values(["source_date", "symbol"])
    left = keys.assign(row_id=np.arange(len(keys))).sort_values(["date", "symbol"])
    result = pd.merge_asof(left, right, left_on="date", right_on="source_date", by="symbol",
                           direction="backward", allow_exact_matches=False, tolerance=pd.Timedelta(days=tolerance))
    result = result.sort_values("row_id").drop(columns="row_id").reset_index(drop=True)
    if not result.loc[result.source_date.notna(), "source_date"].lt(result.loc[result.source_date.notna(), "date"]).all():
        raise ValueError("Historical source is not strictly prior to entry")
    return result


def episode_age(events: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.Series:
    """Causal length of consecutive signal sessions, not future episode size."""
    if events.duplicated(["symbol", "screen_date"]).any():
        raise ValueError("Duplicate event keys")
    positions = pd.Series(np.arange(len(calendar)), index=calendar)
    work = events[["symbol", "screen_date"]].copy()
    work["position"] = work.screen_date.map(positions)
    if work.position.isna().any():
        raise ValueError("Signal date outside calendar")
    work = work.sort_values(["symbol", "screen_date"])
    new = work.groupby("symbol").position.diff().ne(1)
    work["episode"] = new.cumsum()
    age = work.groupby("episode").cumcount() + 1
    return age.reindex(events.index)


def historical_group_flag(frame: pd.DataFrame, group: str) -> pd.Series:
    """Accept a canonical aggregate or nullable OR of available raw members."""
    if group in frame:
        return frame[group].astype("boolean").astype("Float32")
    flags = frame.reindex(columns=GROUP_MEMBERS[group]).astype("boolean")
    positive = flags.eq(True).any(axis=1)
    return positive.astype("Float32").mask(~positive & flags.isna().any(axis=1))


def derive_source_factors(keys: pd.DataFrame, valuation_path: Path) -> tuple[pd.DataFrame, dict]:
    """Source-level materialization independent of the live SQL backend."""
    root = PROJECT_ROOT
    bar_path = root / "reports/research/overnight_unfiltered_universe_v1/daily_bars.parquet"
    warmup_path = root / "data/raw/daily_partitioned/year_month=202312/data.parquet"
    top_dir = root / "data/raw/top_list"
    top_paths = [p for p in sorted(top_dir.glob("*top_list_*.parquet")) if "20240101" <= p.stem.rsplit("_", 1)[-1] <= "20260904"]
    hashes = {str(p): sha256(p) for p in [bar_path, warmup_path, valuation_path, *top_paths]}
    daily = pd.read_parquet(bar_path, columns=["symbol", "date", "close"])
    warmup = pd.read_parquet(warmup_path, columns=["ts_code", "trade_date", "close"])
    warmup = warmup.rename(columns={"ts_code": "symbol"})
    warmup["date"] = pd.to_datetime(warmup.trade_date)
    daily = pd.concat([warmup[["symbol", "date", "close"]], daily], ignore_index=True).sort_values(["symbol", "date"])
    if daily.duplicated(KEYS).any():
        raise ValueError("Duplicate market sentiment source keys")
    daily["ret"] = daily.groupby("symbol").close.pct_change(fill_method=None) * 100
    daily = daily.assign(up=daily.ret.gt(0), up_limit=daily.ret.ge(9.5), down_limit=daily.ret.le(-9.5), strong=daily.ret.ge(5))
    market = daily.groupby("date").agg(n=("ret", "count"), limit_up_count_proxy=("up_limit", "sum"),
        down_count=("down_limit", "sum"), strong_count=("strong", "sum"), market_up_ratio=("up", "mean"), market_median_ret_1d=("ret", "median"))
    market["limit_up_ratio_proxy"] = market.limit_up_count_proxy / market.n.replace(0, np.nan)
    market["limit_down_ratio_proxy"] = market.down_count / market.n.replace(0, np.nan)
    market["strong_up_ratio_proxy"] = market.strong_count / market.n.replace(0, np.nan)
    market["market_sentiment_5d"] = market.limit_up_ratio_proxy.rolling(5, min_periods=2).mean()
    market["market_panic_5d"] = market.limit_down_ratio_proxy.rolling(5, min_periods=2).mean()
    market = market.drop(columns=["n", "down_count", "strong_count"]).reset_index()
    result = keys.merge(market, on="date", how="left", validate="many_to_one")
    top_columns = ["top_list_count", "top_net_amount_ratio", "top_net_rate"]
    top = read_top_list_features(top_dir, start=keys.date.min())
    top = top.rename(columns={"ts_code": "symbol"})
    result = result.merge(top[KEYS + top_columns], on=KEYS, how="left", validate="one_to_one")
    available = pd.to_datetime([p.stem.rsplit("_", 1)[-1] for p in top_paths])
    covered = result.date.isin(available)
    result.loc[covered, top_columns] = result.loc[covered, top_columns].fillna(0)
    valuation = pd.read_parquet(valuation_path, columns=["date", "ts_code", "industry", "roe", "pe_ttm", "pb"])
    valuation["date"] = pd.to_datetime(valuation.date)
    valuation = valuation.loc[valuation.date.le(keys.date.max())].copy()
    valuation["month"] = valuation.date.dt.to_period("M")
    valuation = valuation.sort_values(["ts_code", "date"]).drop_duplicates(["ts_code", "month"], keep="last").drop(columns="month")
    valuation = add_monthly_valuation_history(valuation)
    values = valuation[["ts_code", "date", *MONTHLY_VALUATION_HISTORY_COLUMNS]].rename(columns={"ts_code": "symbol"})
    joined = asof_history(keys, values, tolerance=45)
    result = pd.concat([result, joined[list(MONTHLY_VALUATION_HISTORY_COLUMNS)]], axis=1)
    for path, digest in hashes.items():
        if sha256(Path(path)) != digest:
            raise ValueError(f"Derived source changed: {path}")
    return result, dict(sources=hashes, market_warmup="2023-12", top_list_covered_rows=int(covered.sum()),
                        monthly_history="Prior monthly snapshots,84months,min24; revisions not proven PIT", live_sql_used=False)


def main() -> None:
    root = PROJECT_ROOT / "reports/research"
    output = root / "right_t012_full_dataset_v4"
    history_root = root / "right_t012_history_v1"
    research_root = root / "right_t012_research_factors_v3"
    history_check = json.loads((history_root / "complete.json").read_text())
    signal_path = history_root / "all_daily_signals.parquet"
    if sha256(signal_path) != history_check["dataset_sha256"]:
        raise ValueError("Unverified new signal history")
    history_audit = json.loads((root / "right_t012_history_audit_v1/complete.json").read_text())
    if history_audit["signal_sha256"] != history_check["dataset_sha256"] or history_audit["overlap_different_cells"]:
        raise ValueError("New signal history did not pass the completed parity audit")
    extra_check = json.loads((research_root / "complete.json").read_text())
    for filename, key in [("stock.parquet", "stock_sha256"), ("market.parquet", "market_sha256")]:
        if sha256(research_root / filename) != extra_check[key]:
            raise ValueError("Research factor source changed")
    registry = registry_frame()
    registry = registry.loc[registry.role.isin(["feature", "research_feature"])].copy()
    numeric = registry.name.tolist()
    labels_path = root / "overnight_unfiltered_labels_v1/labels.parquet"
    factors_path = root / "overnight_unfiltered_full_factors_v1/dataset.parquet"
    gate_path = root / "overnight_unfiltered_entry_gate_v1/entry_gate.parquet"
    audit_path = root / "overnight_unfiltered_exit_audit_v1/exit_audit.parquet"
    sources = [signal_path, labels_path, factors_path, gate_path, audit_path,
               research_root / "stock.parquet", research_root / "market.parquet"]
    sources += [PROJECT_ROOT / p for p in [
        "data/research/right_side_unified_current_v6_20260824/unified_right_side_dataset.parquet",
        "data/research/left_side_unified_v3_group4_input_parity/events.parquet",
        "data/research/right_side_registry_607/daily_basic_features.parquet",
        "data/research/right_side_registry_607/daily_extras.parquet",
        "data/research/selector_model_history_2020.parquet",
        "data/research/right_side_registry_607/analyst_weekly.parquet",
        "data/features/long_entry/weekly_external_v1.parquet",
        "data/features/long_entry/weekly_quality_factors_v1.parquet",
        "data/features/long_entry/weekly_training_v2.parquet"]]
    raw_entry_path = root / "overnight_unfiltered_universe_v1/daily_bars.parquet"
    sources.append(raw_entry_path)
    hashes = {str(p): sha256(p) for p in sources}
    output.mkdir(exist_ok=False)
    (output / "protocol.json").write_text(json.dumps(dict(sources=hashes, registry_candidates=numeric,
        timing="T pure-right pool; T+1 close proxy entry; T+2 sell; postclose inputs lagged toT or strict prior source date",
        sample="All right events; no5day maturity/tradability filters or daily cap; causal episode_age; per-date training weights",
        source_policy="Full historical research arm. Current vendor revisions/legacy feature versions not proven PIT; separate strict price-derived control required",
        no_lookahead_claim="Lagging dates does not repair unknown source revisions; this is explicitly a historical research arm",
        script_sha256=sha256(Path(__file__))), indent=2))
    labels = pd.read_parquet(labels_path)
    calendar = pd.DatetimeIndex(labels.date.unique()).sort_values()
    signal_frame = pd.read_parquet(signal_path)
    frame = map_entries(signal_frame, calendar)
    frame["episode_age"] = episode_age(frame, calendar)
    frame = frame.loc[frame.date.notna()].reset_index(drop=True)
    gate = pd.read_parquet(gate_path)
    frame = frame.merge(gate[KEYS + ["eligible", "unknown_entry", "at_upper"]], on=KEYS, how="left", validate="one_to_one")
    frame.to_parquet(output / "all_candidates.parquet", index=False)
    # Outcome availability never participates in the entry gate.
    frame = frame.loc[frame.eligible.eq(True) & frame.planned_exit_date.notna()].reset_index(drop=True)
    frame = frame.merge(labels, on=KEYS, how="left", validate="one_to_one")
    raw_entry = pd.read_parquet(raw_entry_path, columns=KEYS + ["pre_close"])
    raw_entry = raw_entry.rename(columns={"pre_close": "entry_raw_pre_close"})
    frame = frame.merge(raw_entry, on=KEYS, how="left", validate="one_to_one")
    if not frame.exit_date.eq(frame.planned_exit_date).all():
        raise ValueError("T+2 label mismatch")
    screen_close = labels[KEYS + ["entry_close"]].rename(columns={"date": "screen_date", "entry_close": "screen_close"})
    frame = frame.merge(screen_close, on=["symbol", "screen_date"], how="left", validate="one_to_one")
    frame["screen_to_entry_return"] = frame.entry_close / frame.screen_close - 1
    frame["screen_to_entry_nominal_return"] = frame.screen_to_entry_return
    del labels, gate, screen_close
    audit = pd.read_parquet(audit_path, columns=KEYS + ["down_limit", "price_limit_inconsistent", "limit1_exit_uncertain", "limit1_net_after_exit_audit"])
    frame = frame.merge(audit, on=KEYS, how="left", validate="one_to_one")
    frame = pd.concat([frame, direct_exit_flags(frame, "net_close")], axis=1)
    del audit
    # Use current market-derived factors but prior-session provider daily_basic values.
    market_names = list(pd.read_parquet(root / "overnight_unfiltered_market_factors_v1/by_symbol/000001.SZ.parquet").columns)
    market_names = [c for c in PROJECT_FACTOR_COLUMNS if c in market_names]
    basic_names = [c for c in PROJECT_FACTOR_COLUMNS if c not in market_names]
    factors = pd.read_parquet(factors_path)
    frame = frame.merge(factors[KEYS + market_names], on=KEYS, how="left", validate="one_to_one")
    lagged = factors[KEYS + basic_names].rename(columns={"date": "screen_date"})
    frame = frame.merge(lagged, on=["symbol", "screen_date"], how="left", validate="one_to_one")
    # T+1 vendor pre_close is known on entry day and accounts for ex-right resets.
    # Preserve the literal T/T+1 ratio separately for corporate-action badcases.
    frame["screen_to_entry_return"] = frame.entry_close / frame.entry_raw_pre_close - 1
    del factors, lagged
    ownership = {c: "T+1 market-derived" for c in market_names}
    ownership.update({c: "T daily_basic (published after close)" for c in basic_names})
    filters = [("date", ">=", pd.Timestamp("2023-01-01")), ("date", "<=", pd.Timestamp("2026-09-04"))]
    exact_sources = [
        (sources[7], {"right_side_rule"}), (sources[8], {"left_side_rule"}),
        (sources[9], {"chan_live"}), (sources[10], {"project_daily_candidate", "chan_live", "long_external_candidate"}),
        (sources[11], {"selector_live"}),
    ]
    for path, layers in exact_sources:
        names = set(pq.ParquetFile(path).schema.names)
        columns = [c for c in registry.loc[registry.layer.isin(layers), "name"] if c in names and c not in ownership]
        if not columns:
            continue
        values = pd.read_parquet(path, columns=KEYS + columns, filters=filters)
        if values.duplicated(KEYS).any():
            raise ValueError(f"Duplicate exact factor source: {path}")
        values = values.rename(columns={"date": "screen_date"})
        frame = frame.merge(values, on=["symbol", "screen_date"], how="left", validate="one_to_one")
        ownership.update({c: f"T historical cache: {path}" for c in columns})
        print(f"Exact {path.name}: {len(columns)} columns", flush=True)
    weekly_sources = [
        (sources[12], {"long_snapshot", "long_research"}),
        (sources[13], {"long_external_candidate"}),
        (sources[14], {"long_research"}),
        (sources[15], {"long_snapshot", "long_research"}),
    ]
    for number, (path, layers) in enumerate(weekly_sources):
        names = set(pq.ParquetFile(path).schema.names)
        columns = [c for c in registry.loc[registry.layer.isin(layers), "name"] if c in names and c not in ownership]
        values = pd.read_parquet(path, columns=["ts_code", "date", *columns]).rename(columns={"ts_code": "symbol"})
        values["date"] = pd.to_datetime(values.date)
        joined = asof_history(frame[KEYS], values)
        joined[KEYS + ["source_date"]].to_parquet(output / f"weekly_source_dates_{number}.parquet", index=False)
        frame = pd.concat([frame, joined[columns]], axis=1)
        ownership.update({c: f"Strict prior<=14days: {path}; revisions unverified" for c in columns})
        print(f"Weekly {path.name}: {len(columns)} columns", flush=True)
    # Reuse only source materializers, never their label-filtered training table.
    screen_keys = frame[["symbol", "screen_date"]].rename(columns={"screen_date": "date"})
    derived, derived_audit = derive_source_factors(screen_keys, sources[15])
    derived_columns = [c for c in derived if c in numeric and c not in ownership]
    frame = frame.merge(derived[KEYS + derived_columns].rename(columns={"date": "screen_date"}),
                        on=["symbol", "screen_date"], how="left", validate="one_to_one")
    ownership.update({c: "T source-level market/toplist/monthly valuation materializer" for c in derived_columns})
    (output / "derived_source_audit.json").write_text(json.dumps(derived_audit, indent=2))
    # Group membership is descriptive T information, never future episode length.
    left_path = sources[8]
    raw_members = {s for members in GROUP_MEMBERS.values() for s in members} | set(GROUP_MEMBERS)
    left_columns = sorted(raw_members & (set(pq.ParquetFile(left_path).schema.names) - set(frame)))
    left_flags = pd.read_parquet(left_path, columns=KEYS + left_columns, filters=filters)
    frame = frame.merge(left_flags.rename(columns={"date": "screen_date"}), on=["symbol", "screen_date"], how="left", validate="one_to_one")
    for group, members in GROUP_MEMBERS.items():
        name = f"group__{group}"
        if name in numeric:
            frame[name] = historical_group_flag(frame, group)
            ownership[name] = "T actual signal flags; unavailable left flags stay missing"
    # Postclose research factors lag toT. Price-only breadth/candles useT+1.
    stock = pd.read_parquet(research_root / "stock.parquet").drop(columns="trade_date")
    stock_names = [c for c in stock if c in numeric]
    candle_names = [c for c in stock_names if c.startswith("rs_")]
    active_names = [c for c in stock_names if c not in candle_names]
    frame = frame.merge(stock[KEYS + candle_names], on=KEYS, how="left", validate="one_to_one")
    frame = frame.merge(stock[KEYS + active_names].rename(columns={"date": "screen_date"}), on=["symbol", "screen_date"], how="left", validate="one_to_one")
    market = pd.read_parquet(research_root / "market.parquet").drop(columns="trade_date")
    breadth = [c for c in market if c in numeric and c.startswith("market_breadth_")]
    # Use registry layer rather than relying on feature prefixes.
    breadth = [c for c in market if c in set(registry.loc[registry.layer.eq("market_breadth_research"), "name"])]
    delayed_market = [c for c in market if c in numeric and c not in breadth]
    frame = frame.merge(market[["date", *breadth]], on="date", how="left", validate="many_to_one")
    frame = frame.merge(market[["date", *delayed_market]].rename(columns={"date": "screen_date"}), on="screen_date", how="left", validate="many_to_one")
    ownership.update({c: "T+1 price-only research calculator" for c in [*candle_names, *breadth]})
    ownership.update({c: "T lagged active-value/index calculator" for c in [*active_names, *delayed_market]})
    absent = [c for c in numeric if c not in frame]
    if absent:
        frame = pd.concat([frame, pd.DataFrame(np.nan, index=frame.index, columns=absent)], axis=1)
    frame[numeric] = frame[numeric].astype("float32").replace([np.inf, -np.inf], np.nan)
    coverage = registry[["name", "layer", "role"]].copy()
    coverage["source"] = coverage.name.map(ownership).fillna("no_admissible_historical_source")
    coverage["known"] = [int(frame[c].notna().sum()) for c in numeric]
    coverage["unique"] = [int(frame.loc[frame.date.lt("2025-01-01"), c].nunique()) for c in numeric]
    coverage["coverage"] = coverage.known / len(frame)
    coverage.to_csv(output / "factor_coverage.csv", index=False)
    frame.to_parquet(output / "dataset.parquet", index=False)
    for path, digest in hashes.items():
        if sha256(Path(path)) != digest:
            raise ValueError(f"Source changed: {path}")
    (output / "complete.json").write_text(json.dumps(dict(rows=len(frame), stocks=frame.symbol.nunique(),
        dates=frame.date.nunique(), registry_candidates=len(numeric), materialized=len(numeric) - len(absent),
        absent=absent, observed_features=int(coverage.known.gt(0).sum()), informative2024=int(coverage.unique.gt(1).sum()),
        strict_market_columns=market_names, dataset_sha256=sha256(output / "dataset.parquet"),
        status="Historical full-registry research dataset; no independent/PIT-revision confirmation"), indent=2))
    print(f"Rows={len(frame)} registered={len(numeric)} observed={coverage.known.gt(0).sum()}", flush=True)


if __name__ == "__main__":
    main()
