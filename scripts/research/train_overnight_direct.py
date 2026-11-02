"""Direct overnight net-profit classification, with validation-locked selection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from quant.research.holding_event_probability import add_t0_interactions  # noqa: E402
from quant.research.tail_overnight_selection import (  # noqa: E402
    add_selection_features,
    take_profit_return,
)


def select(frame: pd.DataFrame, probability: float, minimum_net: float, top_k: int,
           maximum_risk: float = 1.) -> pd.DataFrame:
    mask = frame.probability.ge(probability) & frame.expected_net.ge(minimum_net)
    if maximum_risk < 1.:
        mask &= frame.risk.le(maximum_risk)
    return (frame[mask]
            .sort_values(["date", "probability", "expected_net", "symbol"], ascending=[True, False, False, True])
            .groupby("date").head(top_k))


def metrics(picks: pd.DataFrame, days: pd.DatetimeIndex, top_k: int) -> dict:
    known = picks.net.notna()
    daily = picks.net.fillna(-.10).groupby(picks.date).sum().reindex(days, fill_value=0) / top_k
    return {"signals": len(picks), "dates": int(picks.date.nunique()),
            "unknown": int((~known).sum()), "known_win": float(picks.loc[known, "net"].gt(0).mean()),
            "win_missing_as_failure": float(picks.net.gt(0).mean()),
            "known_mean_net": float(picks.net.mean()),
            "daily_penalized_mean": float(daily.mean()),
            "daily_score": float(daily.mean() - 2 * daily.std() / np.sqrt(len(days))),
            "allocation": len(picks) / (len(days) * top_k)}


def below_upper_limit(close: pd.Series, upper: pd.Series) -> pd.Series:
    """Conservative close-price gate, not a guarantee of an auction fill."""
    return close.gt(0) & upper.gt(0) & np.isfinite(close) & np.isfinite(upper) & close.lt(upper - .0051)


def barrier_returns(frame: pd.DataFrame, take_profit: float, stop_loss: float) -> tuple[pd.Series, pd.Series]:
    """Daily-bar execution proxy: open first, ambiguous later paths stop first.

    Stop-price fills and the maximum loss are NOT guaranteed. Opening gaps
    retain their full loss. Intraday execution requires minute/tick auditing.
    """
    known = np.isfinite(frame[["y_open", "y_high", "y_low", "y_close"]]).all(axis=1)
    gap_exit = frame.y_open.ge(take_profit) | frame.y_open.le(-stop_loss)
    high_touch, low_touch = frame.y_high.ge(take_profit), frame.y_low.le(-stop_loss)
    ambiguous = known & ~gap_exit & high_touch & low_touch
    gross = frame.y_close.mask(high_touch, take_profit).mask(low_touch, -stop_loss)
    gross = gross.mask(gap_exit, frame.y_open)
    return (gross - .002).where(known), ambiguous


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/research/overnight_direct_v1")
    parser.add_argument("--exact-limits", action="store_true")
    parser.add_argument("--recent-window", action="store_true")
    parser.add_argument("--small-target", action="store_true")
    parser.add_argument("--risk-head", action="store_true")
    parser.add_argument("--intraday-stop", action="store_true")
    args = parser.parse_args()
    output = args.output
    limit_root = ROOT / "reports/research/overnight_limit_history"
    if args.exact_limits and not (limit_root / "complete.json").exists():
        raise ValueError("Exact-limit history not complete; do not silently train on partial dates")
    output.mkdir(exist_ok=False)
    protocol = {"fit": "2024", "calibration": "2025H1", "selection": "2025H2",
                "confirmation": "2026 previously examined; exploratory only",
                "input": "authorized close proxy; not executable backtest",
                "entry": "abs(tail_return)<.095; not exact limit screening",
                "targets": ["tp1", "close"], "cost_bps": 20,
                "probability_thresholds": [.5, .6, .7, .8, .85, .9, .95],
                "minimum_net": [0., .002, .005], "top_k": [1, 3, 5],
                "selection_rule": "at least50 trades,20 dates,win>=80%,unknown=0,positive daily_score; maximize allocation",
                "unknown": "retained; -10% scoring penalty is not realized return or loss bound"}
    protocol["exact_limit_gate"] = args.exact_limits
    if args.recent_window:
        protocol.update(fit="2024 through 2025H1", calibration="2025Q3", selection="2025Q4")
    if args.small_target:
        protocol["targets"] = ["tp05", "tp1"]
    protocol["risk_thresholds"] = [1., .2, .1, .05] if args.risk_head else [1.]
    protocol["risk_event"] = "net return under the fixed exit rule < -2%" if args.risk_head else None
    protocol["intraday_stop"] = "1%; ambiguous daily paths stop first; price-only proxy" if args.intraday_stop else None
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    print("loading historical short-strategy dataset", flush=True)
    frame = add_t0_interactions(add_selection_features(pd.read_parquet(
        ROOT / "reports/research/tail_overnight/dataset.parquet")))
    if args.exact_limits:
        limits = pd.concat([pd.read_parquet(path) for path in sorted(limit_root.glob("*.parquet"))], ignore_index=True)
        limits["date"] = pd.to_datetime(limits.trade_date)
        limits = limits.rename(columns={"ts_code": "symbol"})
        raw = ds.dataset(ROOT / "data/raw/daily_partitioned", format="parquet", partitioning="hive")
        bars = raw.to_table(columns=["ts_code", "trade_date", "close"],
                            filter=(ds.field("trade_date") >= "20240101")
                            & (ds.field("trade_date") <= "20260904")).to_pandas()
        bars["date"] = pd.to_datetime(bars.trade_date)
        bars = bars.rename(columns={"ts_code": "symbol", "close": "entry_close"})
        frame = frame.merge(limits[["symbol", "date", "up_limit"]], on=["symbol", "date"],
                            how="left", validate="one_to_one")
        frame = frame.merge(bars[["symbol", "date", "entry_close"]], on=["symbol", "date"],
                            how="left", validate="one_to_one")
        gate = below_upper_limit(frame.entry_close, frame.up_limit)
        missing = ~frame.entry_close.gt(0) | ~frame.up_limit.gt(0)
        (output / "entry_gate.json").write_text(json.dumps({"before": len(frame), "after": int(gate.sum()),
            "unknown": int(missing.sum()), "at_or_above_limit": int((~gate & ~missing).sum()),
            "note": "filter uses only entry-day values; exit unknowns not removed"}, indent=2))
        frame = frame[gate].copy()
        del bars, limits
    frame = frame[frame.tail_return.abs().lt(.095)].reset_index(drop=True)
    columns = [c for c in frame if c.startswith(("lag1_", "tail_", "context_", "t0x_"))]
    # Do not learn the current watchlist identity. Other pool identities omitted for comparability.
    fit_end = "2025-07-01" if args.recent_window else "2025-01-01"
    cal_end = "2025-10-01" if args.recent_window else "2025-07-01"
    train = frame.date.ge("2024-01-01") & frame.exit_date.lt(fit_end)
    cal = frame.date.ge(fit_end) & frame.exit_date.lt(cal_end)
    forecast = frame.date.ge(cal_end)
    admitted = [c for c in columns if frame.loc[train, c].notna().mean() >= .05
                and frame.loc[train, c].nunique() > 1]
    x = frame[admitted].astype("float32").replace([np.inf, -np.inf], np.nan)
    meta = {"features": admitted, "rows": len(frame), "train_rows": int(train.sum()),
            "calibration_rows": int(cal.sum()), "prediction_rows": int(forecast.sum())}
    (output / "data.json").write_text(json.dumps(meta, indent=2))
    summaries = []
    for target in protocol["targets"]:
        ambiguous = pd.Series(False, index=frame.index)
        if target in ("tp1", "tp05"):
            level = .01 if target == "tp1" else .005
            if args.intraday_stop:
                net, ambiguous = barrier_returns(frame, level, .01)
            else:
                net = take_profit_return(frame, level, 20)
        else:
            net = frame.y_close - .002
        fit_mask, cal_mask = train & net.notna(), cal & net.notna()
        weights = 1 / frame.loc[fit_mask].groupby("date").date.transform("size")
        weights /= weights.mean()
        params = dict(max_iter=120, max_leaf_nodes=15, min_samples_leaf=200,
                      l2_regularization=30, learning_rate=.06, early_stopping=False, random_state=20260912)
        classifier = HistGradientBoostingClassifier(**params)
        regressor = HistGradientBoostingRegressor(**params)
        risk_model, risk_calibrator = None, None
        print(f"{target}: fitting {fit_mask.sum()} rows, {len(admitted)} features", flush=True)
        with threadpool_limits(limits=4):
            classifier.fit(x.loc[fit_mask], net[fit_mask].gt(0), sample_weight=weights)
            print(f"{target}: classifier complete", flush=True)
            regressor.fit(x.loc[fit_mask], net[fit_mask], sample_weight=weights)
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(classifier.predict_proba(x.loc[cal_mask])[:, 1], net[cal_mask].gt(0),
                           sample_weight=1 / frame.loc[cal_mask].groupby("date").date.transform("size"))
            pred = frame.loc[forecast, ["symbol", "date", "exit_date", "pool_short", "pool_good"]].copy()
            pred["probability"] = calibrator.predict(classifier.predict_proba(x.loc[forecast])[:, 1])
            pred["expected_net"] = regressor.predict(x.loc[forecast])
            pred["net"] = net[forecast]
            pred["ambiguous_path"] = ambiguous[forecast]
            if args.risk_head:
                risk_model = HistGradientBoostingClassifier(**params)
                risk_model.fit(x.loc[fit_mask], net[fit_mask].lt(-.02), sample_weight=weights)
                risk_calibrator = IsotonicRegression(out_of_bounds="clip")
                risk_calibrator.fit(risk_model.predict_proba(x.loc[cal_mask])[:, 1], net[cal_mask].lt(-.02),
                    sample_weight=1 / frame.loc[cal_mask].groupby("date").date.transform("size"))
                pred["risk"] = risk_calibrator.predict(risk_model.predict_proba(x.loc[forecast])[:, 1])
                print(f"{target}: downside-risk classifier complete", flush=True)
        joblib.dump({"classifier": classifier, "regressor": regressor, "calibrator": calibrator,
                     "risk_model": risk_model, "risk_calibrator": risk_calibrator,
                     "features": admitted, "protocol": protocol}, output / f"{target}.joblib")
        validation = pred[pred.date.lt("2026-01-01") & pred.exit_date.lt("2026-01-01")]
        days = pd.DatetimeIndex(sorted(validation.date.unique()))
        rows = []
        for prob in protocol["probability_thresholds"]:
            for minimum in protocol["minimum_net"]:
                for k in protocol["top_k"]:
                    for maximum_risk in protocol["risk_thresholds"]:
                        picks = select(validation, prob, minimum, k, maximum_risk)
                        rows.append({"probability": prob, "minimum_net": minimum, "top_k": k,
                                     "maximum_risk": maximum_risk, **metrics(picks, days, k)})
        grid = pd.DataFrame(rows)
        grid.to_csv(output / f"{target}_validation.csv", index=False)
        eligible = grid[(grid.signals >= 50) & (grid.dates >= 20) & grid.unknown.eq(0)
                        & grid.win_missing_as_failure.ge(.8) & grid.daily_score.gt(0)]
        locked = None if eligible.empty else eligible.sort_values("allocation", ascending=False).iloc[0].to_dict()
        (output / f"{target}_locked.json").write_text(json.dumps(locked, indent=2))
        # Freeze policy on disk BEFORE consulting any confirmation outcomes.
        confirmation = pred[pred.date.ge("2026-01-01")]
        summary = {"target": target, "validation_qualified": len(eligible), "locked": locked}
        if locked is not None:
            picks = select(confirmation, locked["probability"], locked["minimum_net"], int(locked["top_k"]),
                           locked["maximum_risk"])
            summary["confirmation"] = metrics(picks, pd.DatetimeIndex(sorted(confirmation.date.unique())),
                                              int(locked["top_k"]))
            picks.to_parquet(output / f"{target}_locked_picks.parquet", index=False)
        pred.to_parquet(output / f"{target}_predictions.parquet", index=False)
        summaries.append(summary)
        (output / "summary.json").write_text(json.dumps(summaries, indent=2))
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
