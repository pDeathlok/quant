#!/usr/bin/env python
"""Offline historical tail-entry experiments using existing project screens.

No quote fetching, orders, or production model publication. See strategy record
for close-proxy and historical monthly-pool limitations.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from quant.features.variable_library import (  # noqa: E402
    DAILY_BASIC_PROJECT_FACTOR_COLUMNS,
    PROJECT_FACTOR_COLUMNS,
    load_daily_basic_features,
)
from quant.research.tail_overnight import (  # noqa: E402
    POOL_NAMES,
    add_interactions,
    attach_lagged_features,
    build_labels,
    build_snapshot_features,
    calendar_links,
    daily_exit_scenarios,
    pool_union,
    purged_split,
)

LEFT = "data/research/left_side_unified_v3_group4_input_parity/events.parquet"
RIGHT = "data/research/right_side_unified_current_v6_20260824/unified_right_side_dataset.parquet"
LONG = "reports/long_good_stock_price/signals.parquet"


def read_features(path, start, end):
    schema = pq.ParquetFile(path).schema_arrow.names
    columns = ["symbol", "date", *[c for c in PROJECT_FACTOR_COLUMNS if c in schema]]
    f = pd.read_parquet(
        path,
        columns=columns,
        filters=[("date", ">=", pd.Timestamp(start)), ("date", "<=", pd.Timestamp(end))],
    )
    f["date"] = pd.to_datetime(f.date).dt.normalize()
    return f.drop_duplicates(["symbol", "date"])


def read_daily(start, end):
    parts = []
    warmup = pd.Timestamp(start) - pd.Timedelta(days=120)
    for path in sorted((ROOT / "data/raw/daily_partitioned").glob("year_month=*/data.parquet")):
        month = path.parent.name.split("=")[1]
        if warmup.strftime("%Y%m") <= month <= pd.Timestamp(end).strftime("%Y%m"):
            f = pd.read_parquet(
                path,
                columns=[
                    "symbol",
                    "date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "pre_close",
                    "volume",
                    "amount",
                    "name",
                ],
            )
            parts.append(f)
    if not parts:
        raise ValueError("no local daily partitions")
    daily = pd.concat(parts, ignore_index=True)
    daily["date"] = pd.to_datetime(daily.date).dt.normalize()
    daily = daily[daily.date.between(warmup, pd.Timestamp(end))].copy()
    daily["volume_shares"] = daily.volume * 100
    daily["amount_yuan"] = daily.amount * 1000
    daily = daily.drop(columns=["volume", "amount"]).sort_values(["symbol", "date"])
    if daily.duplicated(["symbol", "date"]).any():
        raise ValueError("duplicate daily bars")
    limits = []
    for path in (ROOT / "data/raw/tradability").glob("*.parquet"):
        if warmup.strftime("%Y%m%d") <= path.stem <= pd.Timestamp(end).strftime("%Y%m%d"):
            f = pd.read_parquet(path)
            f["date"] = pd.to_datetime(f.trade_date.astype(str))
            f["symbol"] = f.ts_code
            limits.append(f[["symbol", "date", "up_limit", "down_limit", "is_suspended"]])
    if limits:
        daily = daily.merge(
            pd.concat(limits).drop_duplicates(["symbol", "date"]),
            on=["symbol", "date"],
            how="left",
            validate="one_to_one",
        )
    return daily.reset_index(drop=True)


def historical_pools(short_keys, calendar, start, end, daily):
    """Carry monthly selection only until next scheduled monthly rebalance.

    No use of forward-return columns in long-screen reports. Daily archived
    full long factor screens replace monthly membership on their exact dates.
    """
    links = calendar_links(calendar)
    monthly = pd.read_parquet(
        ROOT / LONG, columns=["date", "ts_code", "rule", "history_window_months"]
    )
    monthly["date"] = pd.to_datetime(monthly.date)
    monthly = monthly[monthly.history_window_months.eq(84)]
    pools = {"short": short_keys}
    for pool, rule in [("good", "all_good_stocks"), ("long", "composite_60_guard")]:
        signals = monthly[monthly.rule.eq(rule)].rename(columns={"ts_code": "symbol"})
        parts = []
        for date, group in signals.groupby("date"):
            next_month_end = date.to_period("M").to_timestamp("M") + pd.offsets.MonthEnd(1)
            if date > pd.Timestamp(end) or next_month_end < pd.Timestamp(start):
                continue
            # The rebalance date's close is first available for next-session entry.
            next_month_sessions = links[
                links.date.dt.to_period("M") == next_month_end.to_period("M")
            ]
            if len(next_month_sessions) and next_month_end <= links.date.max():
                next_month_end = next_month_sessions.date.max()
            dates = links.loc[(links.date >= date) & (links.date < next_month_end), "date"]
            if len(dates):
                parts.append(
                    group[["symbol"]]
                    .drop_duplicates()
                    .merge(pd.DataFrame({"date": dates}), how="cross")
                )
        pools[pool] = (
            pd.concat(parts, ignore_index=True)
            if parts
            else pd.DataFrame(columns=["symbol", "date"])
        )
    # Newest complete daily good-stock screens are not truncated web rows.
    # Reuse the canonical page's actual good-stock predicate on archived factors.
    from quant.webapp.services import _long_good_price_assessment, _long_good_stock_assessment

    # Causal adjustment on each date's own raw-price scale, including splits.
    scale = (daily.close / daily.pre_close).groupby(daily.symbol).cumprod()
    mean = scale.groupby(daily.symbol).transform(lambda s: s.rolling(120, min_periods=120).mean())
    ma = daily[["symbol", "date"]].copy()
    ma["ma_120"] = mean * daily.close / scale
    for path in sorted((ROOT / "data/features/long").glob("20*.parquet")):
        date = pd.to_datetime(path.stem)
        if not pd.Timestamp(start) <= date <= pd.Timestamp(end):
            continue
        f = pd.read_parquet(path)
        f["date"] = pd.to_datetime(f.date)
        f["symbol"] = f.ts_code
        f = f.merge(ma, on=["symbol", "date"], how="left", validate="one_to_one")
        good = f.apply(lambda row: _long_good_stock_assessment(row)["is_good_stock"], axis=1)
        keys = f.loc[good, ["ts_code", "date"]].rename(columns={"ts_code": "symbol"})
        keys["date"] = pd.to_datetime(keys.date)
        pools["good"] = pd.concat([pools["good"][pools["good"].date.ne(date)], keys])
        price = f.apply(lambda row: _long_good_price_assessment(row)["is_good_price"], axis=1)
        long_keys = f.loc[good & price, ["symbol", "date"]]
        pools["long"] = pd.concat([pools["long"][pools["long"].date.ne(date)], long_keys])
    return pool_union(pools)


def raw_lag_features(daily):
    """Additional causal T-1 overnight statistics, separate from canonical names."""
    f = daily[["symbol", "date"]].copy()
    # pre_close is exchange reference: these are price-change stats, not a
    # dividend-inclusive return series.
    ret = daily.close / daily.pre_close - 1
    gap = daily.open / daily.pre_close - 1
    group = daily.symbol
    f["daily_gap"] = gap
    f["daily_return"] = ret
    f["daily_range"] = (daily.high - daily.low) / daily.pre_close
    for n in [5, 20, 60]:
        f[f"daily_return_mean_{n}"] = ret.groupby(group).transform(
            lambda s: s.rolling(n, min_periods=n).mean()
        )
        f[f"daily_gap_mean_{n}"] = gap.groupby(group).transform(
            lambda s: s.rolling(n, min_periods=n).mean()
        )
        f[f"daily_gap_std_{n}"] = gap.groupby(group).transform(
            lambda s: s.rolling(n, min_periods=n).std()
        )
    f["daily_amount_mean20"] = daily.amount_yuan.groupby(group).transform(
        lambda s: s.rolling(20, min_periods=20).mean()
    )
    return f


def current_watchlist(out_dir):
    source = ROOT / "data/research/similar_patterns/watchlist.json"
    payload = json.loads(source.read_text())
    pd.DataFrame({"symbol": payload["symbols"], "observed_at": payload["updated_at"]}).to_csv(
        out_dir / "current_watchlist.csv", index=False
    )
    return {
        "count": len(payload["symbols"]),
        "observed_at": payload["updated_at"],
        "historical_membership_available": False,
    }


def export_current_pool(pools, out_dir):
    """Current screening union, with watchlist observed time kept separately."""
    source = json.loads((ROOT / "data/research/similar_patterns/watchlist.json").read_text())
    latest = pools.date.max()
    parts = {
        name: pools.loc[pools.date.eq(latest) & pools[f"pool_{name}"], ["symbol", "date"]]
        for name in ("good", "long")
    }
    active = pd.read_parquet(
        ROOT / "data/features/b1/active_candidate_project_features.parquet",
        columns=["symbol", "date"],
    )
    active["date"] = pd.to_datetime(active.date)
    if not active.date.eq(latest).all():
        return {
            "status": "not_exported",
            "reason": "requested historical range does not match current factor cache",
        }
    parts["short"] = active
    parts["watchlist"] = pd.DataFrame({"symbol": source["symbols"], "date": latest})
    out = pool_union(parts).rename(columns={"date": "daily_screen_asof"})
    out["watchlist_observed_at"] = source["updated_at"]
    out.to_csv(out_dir / "current_pool.csv", index=False)
    return {name: int(out[f"pool_{name}"].sum()) for name in POOL_NAMES} | {"union": len(out)}


def materialize(args):
    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Loading local daily bars and point-in-time short-screen factors", flush=True)
    daily = read_daily(args.start, args.end)
    print(f"Loaded {len(daily):,} daily bars", flush=True)
    # Explicit full-market date inventory (not stock-local shift). Small or
    # partial market partitions are rejected rather than assumed to be holidays.
    counts = daily.groupby("date").symbol.nunique()
    if (counts < 1000).any():
        raise ValueError("incomplete market date inventory (<1000 symbols)")
    calendar = sorted(daily.date.unique())
    calendar_links(calendar).to_csv(out_dir / "market_calendar.csv", index=False)
    sources = [ROOT / LEFT, ROOT / RIGHT]
    for folder in ("data/features/left_side_unified", "data/features/right_side_unified"):
        sources.extend(sorted((ROOT / folder).glob("20*_features.parquet")))
    factors = pd.concat(
        [read_features(p, args.start, args.end) for p in sources], ignore_index=True
    )
    print(f"Loaded {len(factors):,} cached factor rows", flush=True)
    # The two caches overlap; deterministic canonical left-cache precedence.
    overlap = int(factors.duplicated(["symbol", "date"]).sum())
    factors = factors.drop_duplicates(["symbol", "date"], keep="first")
    pools = historical_pools(factors[["symbol", "date"]], calendar, args.start, args.end, daily)
    current_pool_counts = export_current_pool(pools, out_dir)
    # Date here is historical screening date; map to next market day's purchase.
    pools = pools.merge(calendar_links(calendar)[["date", "exit_date"]], on="date", how="left")
    pools = pools.drop(columns="date").rename(columns={"exit_date": "date"}).dropna(subset=["date"])
    pools = pools[pools.date.between(pd.Timestamp(args.start), pd.Timestamp(args.end))]
    pools.to_parquet(out_dir / "historical_pool.parquet", index=False)
    print(f"Historical union: {len(pools):,} symbol/session samples", flush=True)
    frame = attach_lagged_features(pools, factors, calendar, PROJECT_FACTOR_COLUMNS)
    del factors
    extra = raw_lag_features(daily)
    extra_cols = [c for c in extra if c.startswith("daily_")]
    extra = attach_lagged_features(pools[["symbol", "date"]], extra, calendar, extra_cols)
    frame = frame.merge(
        extra.drop(columns="feature_date"), on=["symbol", "date"], validate="one_to_one"
    )
    # Labels are computed without carrying model features through multiple joins.
    labels = build_labels(pools[["symbol", "date"]], daily, calendar, cost_bps=args.cost_bps)
    label_cols = [
        "symbol",
        "date",
        "exit_date",
        "entry_executable",
        "exit_open_executable",
        "entry_limit_source",
        "exit_limit_source",
        "corporate_action_gap",
        "outcome_observed",
        *[c for c in labels if c.startswith("y_")],
    ]
    labels[label_cols].to_parquet(out_dir / "labels.parquet", index=False)
    frame = frame.merge(labels[label_cols], on=["symbol", "date"], validate="one_to_one")
    del labels
    quotes = pools[["symbol", "date"]].merge(
        daily, on=["symbol", "date"], how="left", validate="one_to_one"
    )
    quotes["price"] = quotes.close
    quotes["turnover_pct"] = np.nan
    basics = []
    for p in sorted((ROOT / "data/raw/daily_basic").glob("20*.parquet")):
        if (
            pd.Timestamp(args.start).strftime("%Y%m%d")
            <= p.stem
            <= pd.Timestamp(args.end).strftime("%Y%m%d")
        ):
            f = pd.read_parquet(p, columns=["ts_code", "trade_date", "turnover_rate"])
            f["symbol"] = f.ts_code
            f["date"] = pd.to_datetime(f.trade_date.astype(str))
            basics.append(f[["symbol", "date", "turnover_rate"]])
    if basics:
        quotes = quotes.merge(
            pd.concat(basics).drop_duplicates(["symbol", "date"]),
            on=["symbol", "date"],
            how="left",
            validate="one_to_one",
        )
        quotes["turnover_pct"] = quotes.turnover_rate
    tail = build_snapshot_features(quotes, mode="close_proxy")
    frame = add_interactions(frame.merge(tail, on=["symbol", "date"], validate="one_to_one"))
    frame["tail_amount_vs_lag1_mean20"] = (
        frame.tail_amount_est_yuan / frame.lag1_daily_amount_mean20.replace(0, np.nan)
    )
    frame["tail_gap_surprise20"] = (
        frame.tail_gap - frame.lag1_daily_gap_mean_20
    ) / frame.lag1_daily_gap_std_20.replace(0, np.nan)
    # Exclude at entry only, never based on future exit availability.
    name_bad = quotes.name.astype(str).str.contains("ST|退", regex=True, case=False).to_numpy()
    frame["entry_policy_eligible"] = frame.entry_executable & ~name_bad
    coverage = []
    for pool in POOL_NAMES:
        part = frame[frame[f"pool_{pool}"]]
        coverage.append(
            {
                "pool": pool,
                "samples": len(part),
                "full_project_factor_rows": int(
                    part[[f"lag1_{c}" for c in PROJECT_FACTOR_COLUMNS]].notna().all(axis=1).sum()
                ),
                "project_factor_mean_coverage": float(
                    part[[f"lag1_{c}" for c in PROJECT_FACTOR_COLUMNS]].notna().mean().mean()
                )
                if len(part)
                else None,
            }
        )
    pd.DataFrame(coverage).to_csv(out_dir / "pool_coverage.csv", index=False)
    frame.to_parquet(out_dir / "dataset.parquet", index=False)
    manifest = {
        "status": "research_only",
        "input_mode": "close_proxy",
        "samples": len(frame),
        "start": str(frame.date.min()),
        "end": str(frame.date.max()),
        "screen_sources": [str(p.relative_to(ROOT)) for p in sources] + [LONG],
        "short_screen": "historical left and unified right rule-event union",
        "long_screen": "84-month composite_60_guard monthly selection; expires next month-end",
        "good_screen": "84-month all_good_stocks monthly selection plus exact-date daily archived factors",
        "calendar_source": "full-market daily partition date inventory, >=1000 symbols/session",
        "overlap_rows_left_precedence": overlap,
        "watchlist": current_watchlist(out_dir),
        "pool_coverage": coverage,
        "current_pool_counts": current_pool_counts,
        "cost_bps": args.cost_bps,
        "limitations": [
            "monthly long screen is not daily recomputed long selection",
            "sparse canonical factor cache outside short events; coverage disclosed",
            "historical exact limit prices largely unavailable; conservative proxies",
            "close_proxy includes after-14:30 information",
            "historical screen reports are reconstructed, not proof of production availability",
            "current watchlist is not backfilled into historical training",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps({"samples": len(frame), "coverage": coverage}, ensure_ascii=False), flush=True)


def enrich_daily_basic(args):
    """Recover 35 existing factors deliberately nulled in older model caches."""
    path = args.output / "dataset.parquet"
    keys = pd.read_parquet(path, columns=["symbol", "date", "feature_date"])
    targets = keys[["symbol", "feature_date"]].drop_duplicates().dropna()
    targets["ts_code"] = targets.symbol
    date_strings = {
        date: pd.Timestamp(date).strftime("%Y%m%d") for date in targets.feature_date.unique()
    }
    targets["trade_date"] = targets.feature_date.map(date_strings)
    print("Restoring T-1 daily_basic factors with the canonical rolling loader", flush=True)
    basic = load_daily_basic_features(ROOT / "data/raw/daily_basic", target_keys=targets)
    basic["symbol"] = basic.ts_code
    basic["feature_date"] = pd.to_datetime(basic.trade_date.astype(str))
    cols = list(DAILY_BASIC_PROJECT_FACTOR_COLUMNS)
    basic = basic[["symbol", "feature_date", *cols]]
    aligned = keys.merge(basic, on=["symbol", "feature_date"], how="left", validate="many_to_one")
    del basic, targets, keys
    frame = pd.read_parquet(path)
    if not frame[["symbol", "date"]].equals(aligned[["symbol", "date"]]):
        raise ValueError("daily_basic alignment changed row order")
    for col in cols:
        frame[f"lag1_{col}"] = aligned[col].to_numpy()
    del aligned
    frame = add_interactions(frame)
    frame.to_parquet(path, index=False)
    coverage = []
    for pool in POOL_NAMES:
        part = frame[frame[f"pool_{pool}"]]
        values = part[[f"lag1_{c}" for c in PROJECT_FACTOR_COLUMNS]].notna()
        coverage.append(
            {
                "pool": pool,
                "samples": len(part),
                "full_project_factor_rows": int(values.all(axis=1).sum()),
                "project_factor_mean_coverage": float(values.mean().mean()) if len(part) else None,
            }
        )
    pd.DataFrame(coverage).to_csv(args.output / "pool_coverage.csv", index=False)
    manifest_path = args.output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["pool_coverage"] = coverage
    manifest["daily_basic_enrichment"] = {
        "columns": cols,
        "source": "canonical load_daily_basic_features",
        "timing": "exact T-1",
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("Restored all 35 registered daily_basic factor definitions", flush=True)


def metrics(predictions, top_k, cost_bps):
    rows = []
    for (experiment, fold), group in predictions.groupby(["experiment", "fold"]):
        for pool in ["union", *POOL_NAMES]:
            eligible = group if pool == "union" else group[group[f"pool_{pool}"]]
            if eligible.empty:
                continue
            picks = (
                eligible.sort_values(["date", "score", "symbol"], ascending=[True, False, True])
                .groupby("date")
                .head(top_k)
            )
            if experiment == "all_candidates":
                picks = eligible
            for cost in sorted(set([10.0, 20.0, 30.0, cost_bps])):
                marked = picks.y_open - cost / 10000
                unresolved = ~picks.exit_open_executable | marked.isna()
                daily_mark = marked.groupby(picks.date).mean().dropna()
                rows.append(
                    {
                        "experiment": experiment,
                        "fold": fold,
                        "pool": pool,
                        "cost_bps": cost,
                        "selected_rows": len(picks),
                        "sessions": picks.date.nunique(),
                        "unresolved_exit_rows": int(unresolved.sum()),
                        "observed_open_mark_mean_bps": float(marked.mean() * 10000),
                        "daily_equal_weight_open_mark_bps": float(daily_mark.mean() * 10000),
                        "daily_open_mark_p05_bps": float(daily_mark.quantile(0.05) * 10000),
                        "observed_open_mark_win_rate": float(marked.dropna().gt(0).mean()),
                        "observed_high_mean_bps": float(picks.y_high.mean() * 10000),
                        "realized_portfolio_return": None,
                    }
                )
    return pd.DataFrame(rows)


def train(args):
    frame = pd.read_parquet(args.output / "dataset.parquet")
    frame = frame[frame.entry_policy_eligible].copy()
    train_frame, validation, test = purged_split(frame, args.validation_start, args.test_start)
    lag = [f"lag1_{c}" for c in PROJECT_FACTOR_COLUMNS] + [
        c for c in frame if c.startswith("lag1_daily_")
    ]
    pool_cols = [f"pool_{p}" for p in POOL_NAMES]
    experiments = {
        "lag1": lag + pool_cols,
        "lag1_tail_proxy": lag + pool_cols + [c for c in frame if c.startswith("tail_")],
    }
    predictions = []
    admissions = []
    model_records = []
    targets = {
        "open": ("y_open", "squared_error", None),
        "high": ("y_high", "squared_error", None),
        "low_q10": ("y_low", "quantile", 0.1),
    }
    for experiment, requested in experiments.items():
        # Admission uses training data only. Missing native histogram values are
        # learned only during fit; no validation-based imputation or selection.
        for target in args.targets:
            column, loss, quantile = targets[target]
            known = train_frame[train_frame[column].notna()]
            cols = [
                c for c in requested if known[c].notna().mean() >= 0.05 and known[c].nunique() > 1
            ]
            admissions.extend(
                {
                    "experiment": experiment,
                    "target": target,
                    "feature": c,
                    "train_coverage": float(known[c].notna().mean()),
                    "admitted": c in cols,
                }
                for c in requested
            )
            if len(known) < 1000 or not cols:
                raise ValueError("insufficient training samples/features")
            model = HistGradientBoostingRegressor(
                loss=loss,
                quantile=quantile,
                max_iter=args.iterations,
                max_leaf_nodes=15,
                min_samples_leaf=100,
                l2_regularization=10.0,
                early_stopping=False,
                random_state=42,
            )
            print(
                f"Fitting {experiment}/{target}: {len(known):,} rows, {len(cols)} features",
                flush=True,
            )
            with threadpool_limits(limits=args.threads):
                model.fit(
                    known[cols].astype("float32").replace([np.inf, -np.inf], np.nan), known[column]
                )
                for fold, part in [("validation", validation), ("test", test)]:
                    if part.empty:
                        continue
                    result = part[
                        [
                            "symbol",
                            "date",
                            "exit_date",
                            "exit_open_executable",
                            "y_open",
                            "y_high",
                            "y_low",
                            *pool_cols,
                        ]
                    ].copy()
                    result["score"] = model.predict(
                        part[cols].astype("float32").replace([np.inf, -np.inf], np.nan)
                    )
                    result["fold"] = fold
                    result["experiment"] = f"{experiment}/{target}"
                    predictions.append(result)
            joblib.dump(
                {
                    "model": model,
                    "features": cols,
                    "input_mode": "close_proxy",
                    "train_last_date": known.date.max(),
                    "target": column,
                },
                args.output / f"{experiment}_{target}.joblib",
            )
            model_records.append(
                {
                    "experiment": experiment,
                    "target": column,
                    "train_rows": len(known),
                    "train_last_date": str(known.date.max()),
                    "features": len(cols),
                }
            )
    for fold, part in [("validation", validation), ("test", test)]:
        for name in ("all_candidates", "lag1_gap_mean20_rule", "tail_strength_rule"):
            result = part[
                [
                    "symbol",
                    "date",
                    "exit_date",
                    "exit_open_executable",
                    "y_open",
                    "y_high",
                    "y_low",
                    *pool_cols,
                ]
            ].copy()
            result["fold"] = fold
            result["experiment"] = name
            result["score"] = 0.0
            if name == "lag1_gap_mean20_rule":
                result["score"] = part.lag1_daily_gap_mean_20
            elif name == "tail_strength_rule":
                result["score"] = (
                    part.tail_close_position
                    - 0.5 * part.tail_upper_shadow_share
                    + 10 * part.tail_return.clip(-0.05, 0.05)
                )
            predictions.append(result)
    predictions = pd.concat(predictions, ignore_index=True)
    predictions.to_parquet(args.output / "predictions.parquet", index=False)
    metrics(predictions, args.top_k, args.cost_bps).to_csv(args.output / "metrics.csv", index=False)
    pd.DataFrame(admissions).to_csv(args.output / "feature_admission.csv", index=False)
    (args.output / "models.json").write_text(
        json.dumps(
            {
                "models": model_records,
                "validation_start": args.validation_start,
                "test_start": args.test_start,
                "top_k": args.top_k,
                "status": "exploratory_holdout_not_production",
                "selection": "No automatic winner or publication",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print("Saved models, held-out predictions, coverage and cost scenarios", flush=True)


def evaluate_exits(args):
    """Post-fit daily scenarios, exploratory after seeing open results."""
    predictions = pd.read_parquet(args.output / "predictions.parquet")
    terminal = pd.read_parquet(
        args.output / "labels.parquet", columns=["symbol", "date", "y_close"]
    )
    results, diagnostics = [], []
    for (experiment, fold), group in predictions.groupby(["experiment", "fold"]):
        if "/" in experiment:
            target = experiment.split("/")[-1]
            column = {"open": "y_open", "high": "y_high", "low_q10": "y_low"}[target]
            known = group[group[column].notna()]
            residual = known[column] - known.score
            diagnostics.append(
                {
                    "experiment": experiment,
                    "fold": fold,
                    "rows": len(known),
                    "mae_bps": float(residual.abs().mean() * 10000),
                    "actual_below_prediction_fraction": float((residual <= 0).mean()),
                    "q10_pinball_bps": float(
                        np.maximum(0.1 * residual, -0.9 * residual).mean() * 10000
                    )
                    if target == "low_q10"
                    else None,
                }
            )
        for pool in ["union", "good", "short", "long"]:
            candidates = group if pool == "union" else group[group[f"pool_{pool}"]]
            if candidates.empty:
                continue
            picks = (
                candidates
                if experiment == "all_candidates"
                else candidates.sort_values(
                    ["date", "score", "symbol"], ascending=[True, False, True]
                )
                .groupby("date")
                .head(args.top_k)
            )
            picks = picks.merge(terminal, on=["symbol", "date"], how="left", validate="many_to_one")
            for tp, sl in [(0.01, 0.01), (0.02, 0.01)]:
                scenario = daily_exit_scenarios(
                    picks, take_profit=tp, stop_loss=sl, cost_bps=args.cost_bps
                )
                for exit_name in [
                    "next_close_net",
                    "bracket_stop_first_net",
                    "bracket_take_first_net",
                ]:
                    if exit_name == "next_close_net" and tp != 0.01:
                        continue
                    daily_mark = scenario[exit_name].groupby(picks.date).mean().dropna()
                    results.append(
                        {
                            "experiment": experiment,
                            "fold": fold,
                            "pool": pool,
                            "exit_scenario": exit_name,
                            "take_profit": tp if exit_name != "next_close_net" else None,
                            "stop_loss": sl if exit_name != "next_close_net" else None,
                            "cost_bps": args.cost_bps,
                            "selected_rows": len(picks),
                            "ambiguous_order_rows": int(scenario.ambiguous_order.sum())
                            if exit_name != "next_close_net"
                            else 0,
                            "missing_outcome_rows": int(scenario[exit_name].isna().sum()),
                            "open_execution_problem_rows": int((~picks.exit_open_executable).sum()),
                            "daily_equal_weight_scenario_bps": float(daily_mark.mean() * 10000),
                            "realized_portfolio_return": None,
                        }
                    )
    pd.DataFrame(results).to_csv(args.output / "exit_scenarios.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(args.output / "label_model_metrics.csv", index=False)
    print("Saved conditional exit scenarios and target diagnostics", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", default="2024-01-01")
    p.add_argument("--end", default="2026-09-04")
    p.add_argument("--validation-start", default="2025-01-01")
    p.add_argument("--test-start", default="2026-01-01")
    p.add_argument("--output", type=Path, default=ROOT / "reports/research/tail_overnight")
    p.add_argument(
        "--stage", choices=["all", "materialize", "enrich", "train", "evaluate"], default="all"
    )
    p.add_argument(
        "--targets",
        nargs="+",
        choices=["open", "high", "low_q10"],
        default=["open", "high", "low_q10"],
    )
    p.add_argument("--cost-bps", type=float, default=20.0)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--iterations", type=int, default=60)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    if args.stage in ("all", "materialize"):
        materialize(args)
    if args.stage in ("all", "materialize", "enrich"):
        enrich_daily_basic(args)
    if args.stage in ("all", "train"):
        train(args)
    if args.stage in ("all", "train", "evaluate"):
        evaluate_exits(args)


if __name__ == "__main__":
    main()
