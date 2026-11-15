#!/usr/bin/env python3
"""Evaluate ownership/ETF feature increments on frozen left and right short-side cohorts."""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import gc
import json
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "scripts/research")]

from quant.data.atomic_io import atomic_write_csv, atomic_write_json, atomic_write_parquet
from quant.features.left_side_factor_contract import LEFT_SIDE_FACTOR_COLUMNS
from quant.features.ownership_flow import (
    CONCENTRATION_COLUMNS, ETF_BASKETS, ETF_GROUPS, ETF_MEASURES,
    HOLDER_COUNT_COLUMNS, KEYS, TRANSACTION_COLUMNS,
    concentration_features, etf_flow_features, holder_count_features, transaction_features,
)
from quant.features.variable_library import PROJECT_FACTOR_COLUMNS
from quant.research.right_side_unified import DEFAULT_YEAR_FOLDS, split_by_year_fold
from scripts.research.refresh_ownership_inputs import DATA
from scripts.research.train_right_t012_full_registry import FOLDS, SIGNALS, sample_weights, split_masks
from scripts.research.validate_left_side_disclosure_gain import (
    DATA as LEFT_DATA, REPORT as LEFT_REPORT, block_interval, load_frame, sha256,
)
from scripts.research.validate_unified_left_side_models import _fit_unified_model, _split_validation_stages

OUTPUT = ROOT / "reports/research/short_side_ownership_v2"
RIGHT_DATA = ROOT / "reports/research/right_t012_full_dataset_v4"
RIGHT_REPORT = ROOT / "reports/research/right_t012_full_registry_models_v1"
RIGHT_TARGET = "net_close_net_after_exit_audit"
RIGHT_EXTRAS = (*SIGNALS, "episode_age", "screen_to_entry_return")
ETF_COLUMNS = tuple(f"etf_{group}_{metric}"
                    for group in ("all", *ETF_GROUPS)
                    for metric in ETF_MEASURES)
STOCK_COLUMNS = HOLDER_COUNT_COLUMNS + CONCENTRATION_COLUMNS + TRANSACTION_COLUMNS
ARMS = {
    "stock_ownership": STOCK_COLUMNS,
    "etf_flow": ETF_COLUMNS,
    "combined": STOCK_COLUMNS + ETF_COLUMNS,
}


def calendar(as_of: pd.Timestamp | None = None) -> pd.DatetimeIndex:
    dates = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(LEFT_DATA / "calendar.csv").date))
    dates = dates.union(pd.DatetimeIndex(pd.to_datetime(pd.read_parquet(
        RIGHT_DATA / "dataset.parquet", columns=["date"],
    ).date.unique())))
    # Official local daily partitions supply the newest dates, not business-day guesses.
    end_month = (as_of if as_of is not None else pd.Timestamp(dates.max())).to_period("M")
    first_month = pd.Timestamp(dates.max()).to_period("M")
    if as_of is None:
        partitions = sorted((ROOT / "data/raw/daily_partitioned").glob("year_month=*"))
        if partitions:
            end_month = pd.Period(partitions[-1].name.split("=")[-1], freq="M")
    for month in pd.period_range(first_month, end_month, freq="M"):
        directory = ROOT / "data/raw/daily_partitioned" / f"year_month={month.strftime('%Y%m')}"
        for path in directory.glob("*.parquet"):
            recent = pd.read_parquet(path, columns=["trade_date"]).trade_date
            dates = dates.union(pd.DatetimeIndex(pd.to_datetime(recent.astype("string").unique())))
    dates = dates.normalize().unique().sort_values()
    return dates if as_of is None else dates[dates <= as_of]


def raw(api: str, columns: list[str] | None = None, *, root: Path = DATA) -> pd.DataFrame:
    paths = sorted(p for p in (root / api).glob("*.parquet") if p.with_suffix(".json").exists()
                   and not p.name.startswith("."))
    if not paths:
        raise ValueError(f"No completed raw partitions for {api}")
    return pd.concat([pd.read_parquet(p, columns=columns) for p in paths], ignore_index=True)


def require_history_coverage() -> None:
    for api, periods in (
        ("stk_holdernumber", pd.period_range("2019-01", "2026-11", freq="M")),
        ("stk_holdertrade", pd.period_range("2019-01", "2026-11", freq="M")),
        ("top10_holders", pd.period_range("2019Q1", "2026Q3", freq="Q")),
        ("top10_floatholders", pd.period_range("2019Q1", "2026Q3", freq="Q")),
    ):
        missing = [str(period) for period in periods if not (DATA / api / f"{period}.json").exists()]
        if missing:
            raise ValueError(f"{api}: incomplete raw history: {missing}")


def build_stock() -> None:
    require_history_coverage()
    keys = pd.concat([
        pd.read_parquet(LEFT_DATA / "events.parquet", columns=KEYS),
        pd.read_parquet(RIGHT_DATA / "dataset.parquet", columns=KEYS),
    ], ignore_index=True).drop_duplicates()
    keys = keys.astype({"symbol": "string", "date": "datetime64[ns]"})
    cal = calendar().astype("datetime64[ns]")
    out = make_stock_features(keys, cal)
    atomic_write_parquet(out, DATA / "stock_features.parquet", index=False)
    print(f"stock: {len(out):,} symbol-date observations", flush=True)


def make_stock_features(keys: pd.DataFrame, cal: pd.DatetimeIndex) -> pd.DataFrame:
    if not keys.date.isin(cal).all():
        raise ValueError("Missing signal dates in calendar")
    frames = [
        holder_count_features(keys, raw("stk_holdernumber")),
        transaction_features(keys, raw("stk_holdertrade"), cal),
    ]
    columns = ["ts_code", "ann_date", "end_date", "holder_name",
               "hold_amount", "hold_ratio", "hold_float_ratio"]
    for api, kind in [("top10_holders", "top"), ("top10_floatholders", "float")]:
        frames.append(concentration_features(keys, raw(api, columns), kind))
    out = keys
    for frame in frames:
        out = out.merge(frame, on=KEYS, validate="one_to_one")
    out[list(STOCK_COLUMNS)] = out[list(STOCK_COLUMNS)].astype("float32")
    return out


def build_etf() -> None:
    audit = json.loads((DATA / "refresh_audit.json").read_text())
    if audit["status"] != "ok":
        raise ValueError("Tushare refresh had failed partitions")
    start = pd.Timestamp("2019-01-01")
    cal = calendar(pd.Timestamp(audit["as_of"]))
    out, detail = make_etf_features(cal[cal >= start])
    atomic_write_parquet(out, DATA / "etf_features_v2.parquet", index=False)
    summary = audit_etf(detail)
    atomic_write_csv(summary, OUTPUT / "etf_source_coverage.csv", index=False)
    print(f"ETF: {len(out):,} sessions across {len(summary)} ETFs", flush=True)


def make_etf_features(cal: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    listings = raw("etf_basic")
    index = listings.set_index("ts_code").index_code
    for index_code, symbols in ETF_BASKETS.values():
        if not index.reindex(symbols).eq(index_code).all():
            raise ValueError(f"ETF index assignment changed for {symbols}")
    out, detail = etf_flow_features(
        raw("fund_share"), raw("fund_nav"), raw("fund_daily"), raw("fund_adj"),
        listings, ETF_GROUPS, cal,
    )
    out[list(ETF_COLUMNS)] = out[list(ETF_COLUMNS)].astype("float32")
    return out, detail


def audit_etf(detail: pd.DataFrame) -> pd.DataFrame:
    active = detail.loc[detail.active]
    summary = active.groupby("code").agg(
        expected=("trade_date", "size"), shares=("fd_share", "count"), nav=("unit_nav", "count"),
        bars=("amount", "count"), adjustment=("adj_factor", "count"), flows=("flow", "count"),
        adjustments_excluded=("adjustment_day", "sum"),
    )
    for field in ("fd_share", "unit_nav", "amount", "adj_factor"):
        summary[f"last_{field}"] = active.loc[active[field].notna()].groupby("code").trade_date.max()
    return summary.reset_index()


def export_etf_periods() -> None:
    audit = json.loads((DATA / "refresh_audit.json").read_text())
    if audit["status"] != "ok":
        raise ValueError("Latest input refresh is incomplete")
    cal = calendar(pd.Timestamp(audit["as_of"]))
    _, detail = make_etf_features(cal[cal >= "2019-01-01"])
    baskets = {"all": tuple(sorted(detail.code.unique())), **ETF_GROUPS}
    outputs = []
    for name, symbols in baskets.items():
        rows = detail.loc[detail.active & detail.code.isin(symbols)]
        day = rows.groupby("trade_date").agg(
            expected=("code", "size"), usable=("flow", "count"),
            flow=("flow", lambda x: x.sum(min_count=1)), opening_asset=("prev_asset", "sum"),
        ).sort_index()
        for frequency in ("W-FRI", "M"):
            period = day.index.to_period(frequency)
            groups = day.groupby(period)
            results = groups.agg(
                expected=("expected", "sum"), usable=("usable", "sum"),
                observed_net_cny=("flow", lambda x: x.sum(min_count=1)),
                opening_asset=("opening_asset", "first"),
            )
            results["coverage"] = results.usable / results.expected
            results["complete_calendar_period"] = results.index.end_time.normalize() <= cal[-1]
            results["net_cny"] = results.observed_net_cny.where(results.usable.eq(results.expected))
            results["net_to_opening_asset"] = results.net_cny / results.opening_asset.where(results.opening_asset.gt(0))
            results["period_end"] = results.index.end_time.normalize()
            results["basket"] = name
            results["frequency"] = "week" if frequency == "W-FRI" else "month"
            outputs.append(results.reset_index(drop=True))
    atomic_write_csv(pd.concat(outputs, ignore_index=True), OUTPUT / "etf_period_flows.csv", index=False)


def publish_snapshot() -> None:
    with (DATA / ".refresh.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        _publish_snapshot()


def _publish_snapshot() -> None:
    audit = json.loads((DATA / "refresh_audit.json").read_text())
    if audit["status"] != "ok":
        raise ValueError("Latest input refresh is incomplete")
    as_of = pd.Timestamp(audit["as_of"])
    cal = calendar(as_of).astype("datetime64[ns]")
    if as_of not in cal:
        raise ValueError(f"No official trading session on {as_of.date()}")
    monthly = ROOT / "data/raw/daily_partitioned" / f"year_month={as_of:%Y%m}"
    keys = pd.concat([pd.read_parquet(path, columns=["trade_date", "ts_code"])
                      for path in sorted(monthly.glob("*.parquet"))], ignore_index=True)
    keys["date"] = pd.to_datetime(keys.trade_date.astype("string"), format="mixed").astype("datetime64[ns]")
    keys = keys.loc[keys.date.eq(as_of), ["ts_code", "date"]].rename(columns={"ts_code": "symbol"})
    keys["symbol"] = keys.symbol.astype("string")
    keys = keys.drop_duplicates()
    if keys.empty:
        raise ValueError("Latest official universe is empty")
    stock = make_stock_features(keys, cal)
    market, detail = make_etf_features(cal[-100:])
    coverage = audit_etf(detail)
    cutoff = cal[-3]
    lagged = coverage.loc[
        coverage.last_amount.ne(as_of) | coverage.last_fd_share.lt(cutoff)
        | coverage.last_unit_nav.lt(cutoff) | coverage.last_adj_factor.lt(cutoff)
        | coverage[["last_amount", "last_fd_share", "last_unit_nav", "last_adj_factor"]].isna().any(axis=1),
        "code",
    ].tolist()
    if lagged:
        raise ValueError(f"ETF source stale for {lagged}; existing snapshot retained")
    result = stock.merge(market.loc[market.date.eq(as_of)], on="date", validate="many_to_one")
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S%f")
    path = DATA / "snapshots" / f"{as_of:%Y%m%d}_{stamp}.parquet"
    atomic_write_parquet(result, path, index=False)
    manifest = {"as_of": audit["as_of"], "source_observed_at": audit["observed_at"],
                "rows": len(result), "path": str(path), "sha256": sha256(path),
                "coverage": result.drop(columns=KEYS).notna().mean().to_dict(),
                "etf_freshness": coverage.to_dict("records")}
    atomic_write_json(manifest, DATA / "snapshots" / "latest.json")
    print(f"Published {len(result):,} shadow feature rows for {as_of.date()}", flush=True)


def evaluate(frame: pd.DataFrame, probability: np.ndarray, side: str, fold: str, arm: str) -> None:
    sample = frame[KEYS + ["target", "net"]].assign(probability=probability)
    known = sample.dropna(subset=["target", "net"])
    record = {
        "side": side, "fold": fold, "arm": arm, "rows": len(sample),
        "evaluated_rows": len(known), "ap": average_precision_score(known.target, known.probability),
        "auc": roc_auc_score(known.target, known.probability),
    }
    ranked = sample.sort_values(["date", "probability", "symbol"],
                                ascending=[True, False, True], kind="stable")
    panels = []
    for k in ((10, 20) if side == "left" else (1, 5, 20)):
        picks = ranked.groupby("date", sort=False).head(k)
        daily = picks.groupby("date").agg(net=("net", "mean"), known=("net", "count"), picks=("net", "size"))
        # Keep missing fills visible. A day's result is unknown if any selected exit is unknown.
        daily["net"] = daily.net.where(daily.known.eq(daily.picks))
        record.update({f"top{k}_mean_net": daily.net.mean(), f"top{k}_picks": len(picks),
                       f"top{k}_unknown_exits": int(picks.net.isna().sum())})
        panels.append(daily.reset_index().assign(k=k))
    atomic_write_parquet(pd.concat(panels, ignore_index=True),
                         OUTPUT / f"daily_{side}_{fold}_{arm}.parquet", index=False)
    atomic_write_json(record, OUTPUT / f"metrics_{side}_{fold}_{arm}.json")
    print(json.dumps(record), flush=True)


def add_inputs(frame: pd.DataFrame, additions: tuple[str, ...]) -> pd.DataFrame:
    if set(additions) & set(STOCK_COLUMNS):
        stock = pd.read_parquet(DATA / "stock_features.parquet", columns=KEYS + list(STOCK_COLUMNS))
        stock["symbol"] = stock.symbol.astype(frame.symbol.dtype)
        stock["date"] = stock.date.astype(frame.date.dtype)
        frame = frame.merge(stock, on=KEYS, how="left", validate="one_to_one")
    if set(additions) & set(ETF_COLUMNS):
        etf = pd.read_parquet(DATA / "etf_features_v2.parquet", columns=["date", *ETF_COLUMNS])
        etf["date"] = etf.date.astype(frame.date.dtype)
        frame = frame.merge(etf, on="date", how="left", validate="many_to_one")
    return frame


def coverage(frame: pd.DataFrame, additions: tuple[str, ...], side: str, arm: str) -> None:
    output = frame[list(additions)].notna().groupby(frame.date.dt.year).mean().T
    output.index.name = "feature"
    atomic_write_csv(output.reset_index(), OUTPUT / f"coverage_{side}_{arm}.csv", index=False)


def train_left(arm: str, jobs: int) -> None:
    protocol = json.loads((LEFT_REPORT / "experiment_contract.json").read_text())
    for filename in ("events.parquet", "labels.parquet", "timing_features.parquet"):
        if sha256(LEFT_DATA / filename) != protocol["input_hashes"][filename]:
            raise ValueError("Frozen left training inputs changed")
    additions = ARMS[arm]
    frame = add_inputs(load_frame(), additions)
    frame["target"] = frame.good_path5
    frame["net"] = frame.terminal_return - .002
    coverage(frame, additions, "left", arm)
    for fold in DEFAULT_YEAR_FOLDS:
        split = split_by_year_fold(frame, fold)
        frozen = pd.read_parquet(LEFT_REPORT / f"predictions_{fold.name}_baseline.parquet")
        pd.testing.assert_frame_equal(split.test[KEYS].reset_index(drop=True), frozen[KEYS])
        evaluate(split.test, frozen.probability.to_numpy(), "left", fold.name, "baseline")
        early, calibration, _ = _split_validation_stages(split.validation.reset_index(drop=True), "good_path5")
        print(f"left/{fold.name}/{arm}: fitting {len(split.train):,} rows", flush=True)
        model = _fit_unified_model(split.train.reset_index(drop=True), early, calibration,
                                   [*LEFT_SIDE_FACTOR_COLUMNS, *additions], "good_path5", n_jobs=jobs)
        evaluate(split.test, model.predict_proba(split.test)[:, 1], "left", fold.name, arm)
        del split, model, early, calibration
        gc.collect()


def train_right(arm: str, jobs: int) -> None:
    digest = json.loads((RIGHT_REPORT / "protocol.json").read_text())["dataset_sha256"]
    if sha256(RIGHT_DATA / "dataset.parquet") != digest:
        raise ValueError("Frozen right training inputs changed")
    baseline_columns = tuple(dict.fromkeys([*PROJECT_FACTOR_COLUMNS, *RIGHT_EXTRAS]))
    inputs = list(dict.fromkeys([*KEYS, "planned_exit_date", RIGHT_TARGET, *baseline_columns]))
    additions = ARMS[arm]
    frame = add_inputs(pd.read_parquet(RIGHT_DATA / "dataset.parquet", columns=inputs), additions)
    frame["net"] = frame[RIGHT_TARGET]
    frame["target"] = frame.net.gt(0).astype(float).where(frame.net.notna())
    coverage(frame, additions, "right", arm)
    params = json.loads((RIGHT_REPORT / "protocol.json").read_text())["params"]
    for fold, boundaries in FOLDS.items():
        masks = split_masks(frame, boundaries)
        fit = masks["fit"] & frame.net.notna()
        calibration = masks["calibration"] & frame.net.notna()
        test = frame.loc[masks["diagnostic"]].copy()
        frozen = pd.read_parquet(RIGHT_REPORT / f"{fold}_core145_net_close_predictions.parquet")
        frozen = frozen.loc[frozen.stage.eq("diagnostic"), KEYS + ["probability"]]
        paired = test[KEYS].merge(frozen, on=KEYS, validate="one_to_one", how="left")
        if len(frozen) != len(test) or paired.probability.isna().any():
            raise ValueError("Right frozen baseline does not match diagnostic rows")
        evaluate(test, paired.probability.to_numpy(), "right", fold, "baseline")
        usable = [c for c in [*baseline_columns, *additions] if frame.loc[fit, c].nunique(dropna=True) > 1]
        print(f"right/{fold}/{arm}: fitting {int(fit.sum()):,} rows, {len(usable)} features", flush=True)
        with threadpool_limits(limits=jobs):
            x = frame[usable].astype("float32").replace([np.inf, -np.inf], np.nan)
            model = HistGradientBoostingClassifier(**params).fit(
                x.loc[fit], frame.loc[fit, "target"].astype(int), sample_weight=sample_weights(frame.loc[fit]),
            )
            iso = IsotonicRegression(out_of_bounds="clip").fit(
                model.predict_proba(x.loc[calibration])[:, 1], frame.loc[calibration, "target"].astype(int),
                sample_weight=sample_weights(frame.loc[calibration]),
            )
            p = iso.predict(model.predict_proba(x.loc[masks["diagnostic"]])[:, 1])
        evaluate(test, p, "right", fold, arm)


def summarize() -> None:
    scores = [json.loads(path.read_text()) for path in sorted(OUTPUT.glob("metrics_*.json"))]
    metrics = pd.DataFrame(scores)
    expected = {(side, fold, arm)
                for side, folds in (("left", [fold.name for fold in DEFAULT_YEAR_FOLDS]), ("right", FOLDS))
                for fold in folds for arm in ("baseline", *ARMS)}
    if set(metrics[["side", "fold", "arm"]].itertuples(index=False, name=None)) != expected:
        raise ValueError("Missing or unexpected paired metrics")
    atomic_write_json({
        "left_label": "good_path5, next close entry, five trading days, realized return minus 20bps",
        "right_label": "T012 close-entry net_close_net_after_exit_audit > 0, one-session exit",
        "left_folds": {fold.name: fold.test_year for fold in DEFAULT_YEAR_FOLDS},
        "right_folds": FOLDS,
        "baseline": "frozen predictions on identical held-out keys",
        "interval": "2000 circular moving-block resamples of paired daily portfolio means, block length five",
        "right_exit_uncertainty": "require fully known exit outcomes in both arms for paired return",
        "right_top1_stress": "substitute -5% or -10% for every unknown exit in either arm",
        "stock_visibility": "first session strictly after announcement, provider retrospective revisions unverified",
        "etf_visibility": "D+2 sessions minimum and strictly after current/prior NAV announcements; final turnover D+1",
        "historical_basket": "fixed researcher-selected basket with listing gates, retrospective membership",
        "arms": ARMS,
        "stock_features_sha256": sha256(DATA / "stock_features.parquet"),
        "etf_features_sha256": sha256(DATA / "etf_features_v2.parquet"),
        "production_promoted": False,
    }, OUTPUT / "experiment_contract.json")
    atomic_write_csv(metrics, OUTPUT / "metrics.csv", index=False)
    comparisons = []
    for (side, fold), group in metrics.groupby(["side", "fold"]):
        for arm in group.arm.unique():
            if arm == "baseline":
                continue
            base = pd.read_parquet(OUTPUT / f"daily_{side}_{fold}_baseline.parquet")
            candidate = pd.read_parquet(OUTPUT / f"daily_{side}_{fold}_{arm}.parquet")
            paired = base.merge(candidate, on=["date", "k"], suffixes=("_base", "_new"), validate="one_to_one")
            by_arm = group.set_index("arm")
            for k, picks in paired.groupby("k"):
                delta = (picks.net_new - picks.net_base).dropna().to_numpy()
                row = {"side": side, "fold": fold, "arm": arm, "top_k": int(k),
                       "paired_dates": len(delta), "total_dates": len(picks),
                       "daily_mean_delta": float(delta.mean()) if len(delta) else None,
                       "daily_delta_ci95": block_interval(delta) if len(delta) else None,
                       "ap_delta": float(by_arm.loc[arm, "ap"] - by_arm.loc["baseline", "ap"])}
                if side == "right" and k == 1:
                    for loss in (.05, .10):
                        stressed = picks.net_new.fillna(-loss) - picks.net_base.fillna(-loss)
                        row[f"unknown_exit_minus_{int(loss * 100)}pct_delta"] = float(stressed.mean())
                comparisons.append(row)
    atomic_write_json(comparisons, OUTPUT / "paired_comparison.json")
    atomic_write_csv(pd.DataFrame(comparisons), OUTPUT / "paired_comparison.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("stock", "etf", "etf-periods", "snapshot", "left", "right", "summarize"))
    parser.add_argument("--arm", choices=tuple(ARMS), default="combined")
    parser.add_argument("--jobs", type=int, default=5)
    args = parser.parse_args()
    started = time.monotonic()
    if args.command == "stock":
        build_stock()
    elif args.command == "etf":
        build_etf()
    elif args.command == "etf-periods":
        export_etf_periods()
    elif args.command == "snapshot":
        publish_snapshot()
    elif args.command == "left":
        train_left(args.arm, args.jobs)
    elif args.command == "right":
        train_right(args.arm, args.jobs)
    else:
        summarize()
    print(f"{args.command} complete in {time.monotonic() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
