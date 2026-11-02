"""Broad legacy-pool lag-only diagnostic, not an executable or independent test."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from threadpoolctl import threadpool_limits
from train_overnight_direct import metrics, select

ROOT = Path(__file__).resolve().parents[2]


def same_day_features(frame: pd.DataFrame, output: Path, *, rebuilt: bool = False) -> tuple[pd.DataFrame, list[str]]:
    """User-authorized EOD inputs; deployment substitutes a 14:30 daily snapshot."""
    from quant.features.variable_library import PROJECT_FACTOR_COLUMNS

    paths = [ROOT / "data/research/left_side_unified_v3_group4_input_parity/events.parquet",
             ROOT / "data/research/right_side_unified_current_v6_20260824/unified_right_side_dataset.parquet"]
    parts, provenance = [], []
    if rebuilt:
        root = ROOT / "reports/research/overnight_current_full_factors_v1"
        complete = json.loads((root / "complete.json").read_text())
        path = root / "dataset.parquet"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != complete["dataset_sha256"]:
            raise ValueError("Rebuilt daily factors changed")
        factors = pd.read_parquet(path, columns=["symbol", "date", *PROJECT_FACTOR_COLUMNS])
        pd.testing.assert_frame_equal(frame[["symbol", "date"]].sort_values(["symbol", "date"]).reset_index(drop=True),
                                      factors[["symbol", "date"]].sort_values(["symbol", "date"]).reset_index(drop=True))
        provenance.append({"path": str(path), "sha256": digest, "source": "all candidate dates, not event cache"})
    else:
        for path in paths:
            factors = pd.read_parquet(path, columns=["symbol", "date", "factor_schema_version", *PROJECT_FACTOR_COLUMNS])
            if not factors.factor_schema_version.eq("project-v5-canonical-alias-free").all():
                raise ValueError("Unexpected daily factor schema")
            factors = factors[factors.date.between(frame.date.min(), frame.date.max())]
            parts.append(factors.drop(columns="factor_schema_version"))
            provenance.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        factors = pd.concat(parts, ignore_index=True).drop_duplicates(["symbol", "date"], keep="first")
    factors = factors.rename(columns={c: f"today_factor_{c}" for c in PROJECT_FACTOR_COLUMNS})
    result = frame.merge(factors, on=["symbol", "date"], how="left", validate="one_to_one")
    pd.testing.assert_frame_equal(frame, result[frame.columns])
    del factors, parts
    raw = ds.dataset(ROOT / "data/raw/daily_partitioned", format="parquet", partitioning="hive")
    bars = raw.to_table(columns=["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "vol", "volume", "amount"],
        filter=(ds.field("trade_date") >= "20240101") & (ds.field("trade_date") <= "20260904")).to_pandas()
    bars["vol"] = bars["vol"].fillna(bars["volume"])
    bars = bars.drop(columns="volume")
    bars["date"] = pd.to_datetime(bars.trade_date)
    bars = bars.rename(columns={"ts_code": "symbol"}).drop(columns="trade_date")
    prefix = "today_raw_"
    bars[prefix + "return"] = bars.close / bars.pre_close - 1
    bars[prefix + "gap"] = bars.open / bars.pre_close - 1
    bars[prefix + "range"] = (bars.high - bars.low) / bars.pre_close
    bars[prefix + "close_position"] = (bars.close - bars.low) / (bars.high - bars.low).replace(0, np.nan)
    bars = bars.rename(columns={c: prefix + c for c in ("open", "high", "low", "close", "pre_close", "vol", "amount")})
    result = result.merge(bars, on=["symbol", "date"], how="left", validate="one_to_one")
    pd.testing.assert_frame_equal(frame, result[frame.columns])
    columns = [c for c in result if c.startswith(("lag1_daily_", "today_"))]
    (output / "same_day_sources.json").write_text(json.dumps(provenance, indent=2))
    result[["symbol", "date", *[c for c in columns if c.startswith("today_raw_")]]].to_parquet(
        output / "same_day_raw_snapshot.parquet", index=False)
    return result, columns


def limit_targets(frame: pd.DataFrame, output: Path) -> pd.DataFrame:
    """Rounded sell-limit marks; verify the parent return denominator first."""
    raw = ds.dataset(ROOT / "data/raw/daily_partitioned", format="parquet", partitioning="hive")
    bars = raw.to_table(columns=["ts_code", "trade_date", "close", "high"],
        filter=(ds.field("trade_date") >= "20240101") & (ds.field("trade_date") <= "20260904")).to_pandas()
    bars["date"] = pd.to_datetime(bars.trade_date)
    bars = bars.rename(columns={"ts_code": "symbol"}).drop(columns="trade_date")
    keys = frame[["symbol", "date", "exit_date", "y_high", "y_close"]]
    joined = keys.merge(bars[["symbol", "date", "close"]].rename(columns={"close": "entry_close"}),
                        on=["symbol", "date"], how="left", validate="many_to_one")
    joined = joined.merge(bars.rename(columns={"date": "exit_date", "close": "next_close", "high": "next_high"}),
                          on=["symbol", "exit_date"], how="left", validate="many_to_one")
    pd.testing.assert_frame_equal(keys.reset_index(drop=True), joined[keys.columns])
    known = joined[["y_high", "y_close"]].notna().all(axis=1)
    for name in ("high", "close"):
        if not np.allclose(joined.loc[known, f"next_{name}"] / joined.loc[known, "entry_close"] - 1,
                           joined.loc[known, f"y_{name}"], rtol=1e-7, atol=1e-9):
            raise ValueError(f"Parent/current {name} label mismatch; audit source before training")
    targets = pd.DataFrame(index=frame.index)
    for name, threshold in (("limit05", .005), ("limit1", .01)):
        price = np.ceil(joined.entry_close * (1 + threshold) * 100 - 1e-9) / 100
        reached = joined.next_high.ge(price + joined.entry_close * .001)
        targets[name] = (joined.y_close.where(~reached, price / joined.entry_close - 1) - .002).where(known)
    pd.concat([joined, targets], axis=1).to_parquet(output / "limit_label_source.parquet", index=False)
    return targets


def exact_entry_gate(frame: pd.DataFrame, output: Path) -> pd.Series:
    """Current EOD eligibility proxy, never a future-outcome filter or fill proof."""
    root = ROOT / "reports/research/overnight_limit_history"
    if not (root / "complete.json").exists():
        raise ValueError("Incomplete limit cache")
    parts, hashes = [], []
    for path in sorted(root.glob("*.parquet")):
        parts.append(pd.read_parquet(path, columns=["ts_code", "trade_date", "up_limit"]))
        hashes.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    limits = pd.concat(parts, ignore_index=True)
    limits["date"] = pd.to_datetime(limits.trade_date)
    limits = limits.rename(columns={"ts_code": "symbol"})
    audit = frame[["symbol", "date", "today_raw_close", "today_raw_vol"]].merge(
        limits[["symbol", "date", "up_limit"]], on=["symbol", "date"], how="left", validate="one_to_one")
    audit["unknown_entry"] = ~(audit.today_raw_close.gt(0) & audit.today_raw_vol.gt(0) & audit.up_limit.gt(0))
    audit["at_upper"] = audit.today_raw_close.ge(audit.up_limit - .0051) & ~audit.unknown_entry
    audit["eligible"] = ~audit.unknown_entry & ~audit.at_upper
    audit.to_parquet(output / "exact_entry_gate.parquet", index=False)
    (output / "entry_limit_hashes.json").write_text(json.dumps(hashes, indent=2))
    summary = {"before": len(audit), "after": int(audit.eligible.sum()), "unknown": int(audit.unknown_entry.sum()),
               "at_upper": int(audit.at_upper.sum()), "timing": "EOD proxy, not actual14:30 fill evidence"}
    (output / "exact_entry_gate.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
    return audit.eligible


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expanded-features", action="store_true")
    parser.add_argument("--limit-targets", action="store_true")
    parser.add_argument("--same-day-features", action="store_true")
    parser.add_argument("--exact-entry-gate", action="store_true")
    parser.add_argument("--rebuilt-current-factors", action="store_true")
    parser.add_argument("--relative-price-features", action="store_true")
    parser.add_argument("--market-context", action="store_true")
    args = parser.parse_args()
    source = ROOT / "reports/research/tail_overnight/dataset.parquet"
    output = ROOT / ("reports/research/overnight_lag_expanded_audit_v1" if args.expanded_features
                     else "reports/research/overnight_lag_only_audit_v1")
    if args.limit_targets:
        output = ROOT / ("reports/research/overnight_lag_expanded_limit_v1" if args.expanded_features
                         else "reports/research/overnight_lag_only_limit_v1")
    if args.same_day_features:
        if not args.limit_targets:
            raise ValueError("Same-day experiment requires fixed limit targets")
        output = ROOT / "reports/research/overnight_same_day_limit_v1"
    if args.exact_entry_gate:
        if not args.same_day_features:
            raise ValueError("Exact entry gate requires same-day EOD inputs")
        output = ROOT / "reports/research/overnight_same_day_limit_gated_v1"
    if args.rebuilt_current_factors:
        if not args.exact_entry_gate:
            raise ValueError("Rebuilt-factor comparison requires the same exact entry gate")
        output = ROOT / "reports/research/overnight_same_day_rebuilt_limit_gated_v1"
    if args.relative_price_features:
        if not args.rebuilt_current_factors:
            raise ValueError("Relative-price experiment requires rebuilt current factors")
        output = ROOT / "reports/research/overnight_same_day_relative_limit_gated_v1"
    if args.market_context:
        if not args.rebuilt_current_factors or args.relative_price_features:
            raise ValueError("Market context comparison uses the unnormalized rebuilt parent")
        output = ROOT / "reports/research/overnight_same_day_market_limit_gated_v2"
    output.mkdir(exist_ok=False)
    prefix = "lag1_" if args.expanded_features else "lag1_daily_"
    columns = [c for c in pq.read_schema(source).names if c.startswith(prefix)]
    params = dict(max_iter=120, max_leaf_nodes=15, min_samples_leaf=200,
                  l2_regularization=30, learning_rate=.06, early_stopping=False,
                  random_state=20260912)
    protocol = {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "features": columns, "params": params, "expanded_features": args.expanded_features,
        "fit": "2024 through 2025H1, target date <2025-07-01",
        "calibration": "2025Q3, target date <2025-10-01; date-weighted isotonic",
        "diagnostic": "2025Q4, target date <2026-01-01; reused validation, no independent confirmation",
        "entry_gate": "NONE; final-day entry eligibility is postdecision diagnostic only",
        "targets": ["limit05", "limit1"] if args.limit_targets else ["open", "close"], "cost_bps": [20, 40],
        "limit_policy": "upward cent-rounded limit,10bps penetration,no open improvement,close fallback;"
                        " reachability proxy only" if args.limit_targets else None,
        "probability_thresholds": [0., .5, .6, .7, .8, .9],
        "minimum_net": [None, 0.], "top_k": [1, 3, 5],
        "selection": "rank probability, expected net, symbol; no future labels used",
        "limitations": ["Legacy signal pool provenance incompletely audited",
                        "Daily close/open marks are not fills; limit and suspension uncertainty",
                        "No 2026 evaluation or promotion; diagnostic only",
                        "Unknown outcomes retained; -10% is a scoring penalty, not a loss bound"],
    }
    if args.rebuilt_current_factors:
        protocol["limitations"].append(
            "Recomputed factors use2018 history anchors; cached absolute-price scales and cumulative OBV differ. "
            "This compares rebuilt sources as a whole, not a pure missing-coverage ablation.")
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    frame = pd.read_parquet(source, columns=["symbol", "date", "feature_date", "exit_date",
        "entry_policy_eligible", "y_open_net", "y_close", *(["y_high"] if args.limit_targets else []), *columns])
    if frame.duplicated(["symbol", "date"]).any() or not frame.feature_date.lt(frame.date).all():
        raise ValueError("Duplicate keys or non-lagged input dates")
    target_frame = limit_targets(frame, output) if args.limit_targets else pd.DataFrame(
        {"open": frame.y_open_net, "close": frame.y_close - .002})
    if args.same_day_features:
        frame, columns = same_day_features(frame, output, rebuilt=args.rebuilt_current_factors)
        if args.market_context:
            from overnight_market_context import attach_market_context

            frame, market_columns = attach_market_context(frame, ROOT, output)
            columns.extend(market_columns)
            protocol.update(market_context_features=market_columns,
                            market_context_timing="current full daily A-share context, not minute verified")
        if args.relative_price_features:
            from overnight_relative_prices import DROPPED, PRICE_COLUMNS, relative_prices

            frame, columns = relative_prices(frame, columns)
            protocol.update(relative_price_features=list(PRICE_COLUMNS), dropped_price_features=list(DROPPED),
                relative_price_rule="factor / positive current adjusted close; raw prices unchanged; missing preserved",
                relative_price_script_sha256=hashlib.sha256(
                    (ROOT / "scripts/research/overnight_relative_prices.py").read_bytes()).hexdigest())
        protocol.update(features=columns, input_timing="Current EOD daily factors plus13 prior-day statistics;"
                        " user-authorized14:30 daily-snapshot approximation at deployment, NOT minute-verified",
                        factor_join=("Exact current date; rebuilt all candidate dates; missing raw data retained"
                                     if args.rebuilt_current_factors else
                                     "Exact current date; left-cache precedence, missing factors retained"),
                        same_day_features=True)
        (output / "protocol.json").write_text(json.dumps(protocol, indent=2))
        print(f"same-day inputs: {len(columns)} features, {len(frame)} samples", flush=True)
    market_calendar = frame[["date", "exit_date"]].drop_duplicates()
    if args.exact_entry_gate:
        gate = exact_entry_gate(frame, output).to_numpy()
        frame = frame.loc[gate].reset_index(drop=True)
        target_frame = target_frame.loc[gate].reset_index(drop=True)
        protocol.update(entry_gate="Positive current close/volume,known actual upper limit,close below upper by0.0051;"
                        " applied before fit/calibration/selection, never on future outcomes",
                        original_entry_flag="retained only as legacy diagnostic; may use conservative proxy")
        (output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    fit = frame.date.lt("2025-07-01") & frame.exit_date.lt("2025-07-01")
    cal = frame.date.ge("2025-07-01") & frame.exit_date.lt("2025-10-01")
    val = frame.date.ge("2025-10-01") & frame.date.lt("2026-01-01") & frame.exit_date.lt("2026-01-01")
    days = pd.DatetimeIndex(market_calendar.loc[market_calendar.date.ge("2025-10-01")
        & market_calendar.date.lt("2026-01-01") & market_calendar.exit_date.lt("2026-01-01"), "date"].unique()).sort_values()
    x = frame[columns].astype("float32").replace([np.inf, -np.inf], np.nan)
    summary = []
    for target in protocol["targets"]:
        net = target_frame[target]
        fm, cm = fit & net.notna(), cal & net.notna()
        weights = 1 / frame.loc[fm].groupby("date").date.transform("size")
        weights /= weights.mean()
        with threadpool_limits(limits=4):
            clf = HistGradientBoostingClassifier(**params).fit(x.loc[fm], net[fm].gt(0), sample_weight=weights)
            reg = HistGradientBoostingRegressor(**params).fit(x.loc[fm], net[fm], sample_weight=weights)
            iso = IsotonicRegression(out_of_bounds="clip").fit(
                clf.predict_proba(x.loc[cm])[:, 1], net[cm].gt(0),
                sample_weight=1 / frame.loc[cm].groupby("date").date.transform("size"))
            pred = frame.loc[val, ["symbol", "date", "entry_policy_eligible"]].copy()
            pred["net"] = net[val]
            pred["probability"] = iso.predict(clf.predict_proba(x.loc[val])[:, 1])
            pred["expected_net"] = reg.predict(x.loc[val])
        pred.to_parquet(output / f"{target}_predictions.parquet", index=False)
        joblib.dump({"classifier": clf, "regressor": reg, "calibrator": iso, "columns": columns},
                    output / f"{target}.joblib")
        rows = []
        for probability in protocol["probability_thresholds"]:
            for minimum_net in protocol["minimum_net"]:
                for top_k in protocol["top_k"]:
                    picks = select(pred, probability, -np.inf if minimum_net is None else minimum_net, top_k)
                    record = {"probability": probability, "minimum_net": minimum_net, "top_k": top_k,
                              **metrics(picks, days, top_k),
                              "postdecision_old_entry_failures": int((~picks.entry_policy_eligible).sum())}
                    stress = picks.copy()
                    stress["net"] -= .002
                    record.update({f"cost40_{k}": v for k, v in metrics(stress, days, top_k).items()})
                    rows.append(record)
        table = pd.DataFrame(rows)
        table.to_csv(output / f"{target}_diagnostics.csv", index=False)
        supported = table[table.signals.ge(50) & table.dates.ge(20)]
        result = {"target": target, "fit_known": int(fm.sum()), "cal_known": int(cm.sum()),
                  "validation_rows": int(val.sum()), "validation_dates": len(days),
                  "maximum_probability": float(pred.probability.max()),
                  "supported_rules": len(supported),
                  "best_supported_diagnostic": None if supported.empty else
                      supported.sort_values("win_missing_as_failure", ascending=False).iloc[0].to_dict()}
        summary.append(result)
        print(json.dumps(result), flush=True)
    (output / "complete.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
