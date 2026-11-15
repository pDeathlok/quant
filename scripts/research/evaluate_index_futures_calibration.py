"""Past-OOS calibration, empirical tail bounds, and many-start funded replays."""
import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np
import pandas as pd

from backtest_index_futures import ROOT, metrics
from compare_index_futures_adaptation import OUT as SOURCE, POLICIES, groups_for, history_before, stability, trade_episodes
from evaluate_index_futures_directions import CAPITAL, load_bars
from evaluate_index_futures_exit_value import OUT as PREVIOUS, exit_callback
from evaluate_index_futures_risk_paths import risk_replay, first_year, diagnose_trades
from evaluate_index_futures_trade_wins import annual_stats, trade_stats
from refine_index_futures_adaptation import capital_quality, decide

OUT = ROOT / "exit_calibration_starts"
VARIANTS = ("control", "loss_exit", "mean5", "calibrated5", "tail5")
PROFILES = ((.0003, .15), (.0006, .25))


def weighted_quantile(values, weights, probability):
    values, weights = np.asarray(values), np.asarray(weights)
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    index = np.searchsorted(cumulative, probability*cumulative[-1], side="left")
    return float(values[order[min(index, len(values)-1)]])


def affine_calibration(predicted, actual, weights):
    """Weighted least squares with slope constrained to [0,1], no sign reversal."""
    x, y = np.asarray(predicted), np.asarray(actual)
    mx, my = np.average(x, weights=weights), np.average(y, weights=weights)
    variance = np.average((x-mx)**2, weights=weights)
    slope = np.clip(np.average((x-mx)*(y-my), weights=weights)/variance, 0, 1) if variance > 1e-12 else 0.
    return float(my-slope*mx), float(slope)


def calibration_history(frame, cutoff, direction, bps):
    start = (pd.Timestamp(cutoff)-pd.DateOffset(months=24)).strftime("%Y%m%d")
    columns = [f"net_5_{bps}", f"risk_5_{bps}", f"pred_net_5_{bps}", f"pred_risk_5_{bps}"]
    return frame[frame.direction.eq(direction)&frame.signal_date.ge(start)&frame.maturity.notna()
                 &frame.maturity.lt(cutoff)].dropna(subset=columns)


def episode_weights(frame, cutoff=None):
    weights = 1/frame.groupby("episode").episode.transform("size").to_numpy()
    if cutoff is not None:
        age = (pd.Timestamp(cutoff)-pd.to_datetime(frame.maturity)).dt.days.to_numpy()
        weights *= np.exp2(-age/365.25)
    return weights/weights.mean()


def psi(reference, recent):
    """Descriptive distribution drift; never used to tune this run's rules."""
    reference, recent = np.asarray(reference), np.asarray(recent)
    reference, recent = reference[np.isfinite(reference)], recent[np.isfinite(recent)]
    if len(reference) < 20 or len(recent) < 10:
        return np.nan
    edges = np.unique(np.r_[-np.inf, np.quantile(reference, np.arange(.1, 1, .1)), np.inf])
    a, b = np.histogram(reference, edges)[0]+.5, np.histogram(recent, edges)[0]+.5
    a, b = a/a.sum(), b/b.sum()
    return float(np.sum((b-a)*np.log(b/a)))


def calibrate(frame, panel):
    result = frame.copy()
    for bps in (3, 6):
        result[f"cal_net_{bps}"] = result[f"pred_net_5_{bps}"]
        result[f"cal_risk_{bps}"] = result[f"pred_risk_5_{bps}"]
        result[f"risk90_{bps}"] = np.nan
        result[f"net10_{bps}"] = np.nan
        result[f"cal_ready_{bps}"] = False
    audits, drift = [], []
    for quarter, sample in panel.groupby("quarter", sort=True):
        cutoff = sample.trade_date.min()
        for direction in (1, -1):
            test = frame[frame.quarter.eq(quarter)&frame.direction.eq(direction)]
            for bps in (3, 6):
                past = calibration_history(frame, cutoff, direction, bps)
                ready = past.signal_date.nunique() >= 122 and past.episode.nunique() >= 60
                detail = dict(quarter=quarter, cutoff=cutoff, direction=direction, bps=bps, rows=len(past),
                              episodes=past.episode.nunique(), signal_days=past.signal_date.nunique(),
                              last_maturity=past.maturity.max() if len(past) else None, ready=ready)
                if ready:
                    weights = episode_weights(past, cutoff)
                    a, b = affine_calibration(past[f"pred_net_5_{bps}"], past[f"net_5_{bps}"], weights)
                    ra, rb = affine_calibration(past[f"pred_risk_5_{bps}"], past[f"risk_5_{bps}"], weights)
                    # Residuals are from raw genuinely past-quarter predictions,
                    # not from refitting a calibrator and scoring its own rows.
                    rq = weighted_quantile(past[f"risk_5_{bps}"]-past[f"pred_risk_5_{bps}"], weights, .9)
                    nq = weighted_quantile(past[f"net_5_{bps}"]-past[f"pred_net_5_{bps}"], weights, .1)
                    result.loc[test.index, f"cal_net_{bps}"] = a+b*test[f"pred_net_5_{bps}"]
                    mean_risk = np.maximum(0, ra+rb*test[f"pred_risk_5_{bps}"])
                    result.loc[test.index, f"cal_risk_{bps}"] = mean_risk
                    result.loc[test.index, f"risk90_{bps}"] = np.maximum(mean_risk, test[f"pred_risk_5_{bps}"]+rq)
                    result.loc[test.index, f"net10_{bps}"] = np.minimum(a+b*test[f"pred_net_5_{bps}"], test[f"pred_net_5_{bps}"]+nq)
                    result.loc[test.index, f"cal_ready_{bps}"] = True
                    detail.update(net_intercept=a, net_slope=b, risk_intercept=ra, risk_slope=rb,
                                  risk90_residual=rq, net10_residual=nq)
                    assert past.maturity.max() < cutoff
                audits.append(detail)
            # At the cutoff, only completed historical outcomes are available.
            matured = result[result.direction.eq(direction)&result.maturity.notna()&result.maturity.lt(cutoff)]
            split = (pd.Timestamp(cutoff)-pd.DateOffset(months=6)).strftime("%Y%m%d")
            beginning = (pd.Timestamp(cutoff)-pd.DateOffset(months=24)).strftime("%Y%m%d")
            for label, portion in (("recent6m", matured[matured.signal_date.ge(split)]),
                                   ("previous18m", matured[matured.signal_date.ge(beginning)&matured.signal_date.lt(split)])):
                portion = portion.dropna(subset=["cal_net_3", "net_5_3"])
                if len(portion):
                    w = episode_weights(portion)
                    drift.append(dict(quarter=quarter, cutoff=cutoff, direction=direction, window=label,
                                      rows=len(portion), episodes=portion.episode.nunique(), last_maturity=portion.maturity.max(),
                                      raw_bias=np.average(portion.pred_net_5_3-portion.net_5_3, weights=w),
                                      calibrated_bias=np.average(portion.cal_net_3-portion.net_5_3, weights=w),
                                      calibrated_mae=np.average(abs(portion.cal_net_3-portion.net_5_3), weights=w)))
            # Age five avoids counting changing mixes of holding age as drift.
            known = frame[frame.direction.eq(direction)&frame.signal_date.lt(cutoff)&frame.age.eq(5)]
            dates = sorted(known.signal_date.unique())
            if len(dates) >= 80:
                recent = known[known.signal_date.ge(dates[-60])]
                reference = known[known.signal_date.lt(dates[-60])&known.signal_date.ge(dates[max(0, len(dates)-304)])]
                drift.append(dict(quarter=quarter, cutoff=cutoff, direction=direction, window="known_prediction_shift",
                                  rows=len(recent), prediction_psi=psi(reference.pred_net_5_3, recent.pred_net_5_3),
                                  volatility_psi=psi(reference.f_volatility_20, recent.f_volatility_20)))
    result.to_parquet(OUT/"calibrated_forecasts.parquet", index=False)
    pd.DataFrame(audits).to_csv(OUT/"calibration_audit.csv", index=False)
    pd.DataFrame(drift).to_csv(OUT/"drift_at_cutoff.csv", index=False)
    return result


def calibrated_callback(lookup, variant, fee):
    bps = round(fee*10000)
    previous_key, mean_streak, tail_streak = None, 0, 0
    def choose(day, entry, row, lots, close_equity):
        nonlocal previous_key, mean_streak, tail_streak
        state = lookup[entry, day, row.symbol]
        assert state.direction == lots
        key = (entry, row.symbol)
        if key != previous_key:
            mean_streak, tail_streak = 0, 0
        net, bound = getattr(state, f"cal_net_{bps}"), getattr(state, f"risk90_{bps}")
        tail_fraction = bound*state.notional/close_equity if close_equity > 0 else np.inf
        mean_streak = mean_streak+1 if np.isfinite(net) and net <= 0 else 0
        tail_streak = tail_streak+1 if np.isfinite(bound) and tail_fraction > .05 else 0
        previous_key = key
        exit_tail = variant == "tail5" and tail_streak >= 2
        return mean_streak >= 2 or exit_tail, dict(exit_action="tail_budget_exit" if exit_tail else "calibrated_mean_exit",
                                                 exit_entry=entry, calibrated_net=net, risk90=bound,
                                                 risk90_equity_fraction=tail_fraction, close_equity=close_equity,
                                                 calibration_ready=getattr(state, f"cal_ready_{bps}"))
    return choose


def liquidated_prefix(ledger, count, opens, fee):
    prefix = ledger.iloc[:count].copy().reset_index(drop=True)
    if len(prefix) < len(ledger) and prefix.iloc[-1].positions:
        last = prefix.iloc[-1]
        code, sign = last.positions.split(":")
        multiplier = 300 if code[:2] in ("IF", "IH") else 200
        cost = abs(int(sign))*opens[last.exit_date, code]*multiplier*fee
        prefix.loc[prefix.index[-1], ["pnl", "equity"]] -= cost
        prefix.loc[prefix.index[-1], "fees"] += cost
    return prefix


def start_specs(panel, year):
    specs = []
    for month in range(1, 11 if year == 2026 else 13):
        start = f"{year}{month:02}01"
        specs.append(dict(start=start, start_kind="monthly", phase=0, annual=month == 1))
    days = sorted(panel.loc[panel.entry_date.str.startswith(f"{year}01"), "entry_date"].unique())
    specs.extend(dict(start=day, start_kind="phase", phase=i, annual=False) for i, day in enumerate(days[:5]) if i)
    return specs


def run_year(year):
    out = OUT/str(year)
    out.mkdir(exist_ok=True)
    (out/"ledgers").mkdir(exist_ok=True)
    panel = pd.read_parquet(SOURCE/"quarterly_predictions.parquet")
    forecasts = pd.read_parquet(OUT/"calibrated_forecasts.parquet")
    keep = ["entry", "date", "symbol", "direction", "notional"]
    keep += [f"{prefix}_{bps}" for bps in (3, 6) for prefix in ("pred_net_5", "pred_risk_5", "cal_net", "risk90", "cal_ready")]
    lookup = {(r.entry, r.date, r.symbol): r for r in forecasts[keep].itertuples(index=False)}
    bars = load_bars()
    opens, closes = bars.open.to_dict(), bars.close.to_dict()
    audits, choices, summaries, annual_rows, trades_all, traces = [], [], [], [], [], []
    matches = 0

    @lru_cache(maxsize=None)
    def history(cutoff):
        return groups_for(history_before(panel, cutoff))

    def replay(groups, fee, margin, variant, capital=CAPITAL, policy="base", decision=None, trace=False):
        simple = exit_callback(lookup, "mean5", fee) if variant == "mean5" else None
        stateful = calibrated_callback(lookup, variant, fee) if variant in ("calibrated5", "tail5") else None
        return risk_replay(groups, opens, closes, fee, margin, capital=capital, policy=policy,
                           decision=decision, trace=trace, variant=variant, exit_decider=simple, state_exit_decider=stateful)

    @lru_cache(maxsize=4096)
    def rankings(day, cutoff, equity, fee, margin, variant):
        groups = history(cutoff)
        scores = {}
        for policy in POLICIES:
            ledger, _, _ = replay(groups, fee, margin, variant, capital=equity, policy=policy)
            episodes = trade_episodes(ledger, opens, fee)
            scores[policy] = capital_quality(ledger, day, equity, len(episodes))
            audits.append(dict(date=day, cutoff=cutoff, validation_initial_equity=equity, fee=fee, margin=margin,
                               variant=variant, policy=policy, last_validation_exit=ledger.exit_date.iloc[-1], **scores[policy]))
        assert groups[-1][0].exit_date <= cutoff < day
        return scores

    for spec in start_specs(panel, year):
        start = spec["start"]
        groups = groups_for(panel[panel.entry_date.ge(start)])
        profiles = (*PROFILES, (.0003, .25), (.0006, .15)) if spec["annual"] else PROFILES
        variants = VARIANTS[:3] if spec["start_kind"] == "phase" else VARIANTS
        for fee, margin in profiles:
            for variant in variants:
                fields = dict(**spec, fee=fee, margin=margin, variant=variant, first_entry_date=groups[0][0].entry_date)
                def decision(day, cutoff, incumbent, equity, positions, rows):
                    scores = rankings(day, cutoff, equity, fee, margin, variant)
                    selected, reason, detail = decide(scores, incumbent, rows, positions, equity, fee, margin, True)
                    choices.append(dict(**fields, date=day, cutoff=cutoff, equity=equity,
                                        incumbent=incumbent, selected=selected, reason=reason, **detail))
                    return selected
                ledger, events, trace = replay(groups, fee, margin, variant, decision=decision, trace=True)
                if spec["annual"] and variant in VARIANTS[:3]:
                    old = pd.read_csv(PREVIOUS/f"ledger_{start}_{variant}_{fee}_{margin}.csv", dtype={"entry_date": str, "exit_date": str}).fillna({"positions": ""})
                    pd.testing.assert_frame_equal(old, ledger, check_dtype=False, atol=1e-6, rtol=1e-12)
                    matches += 1
                trades = trade_episodes(ledger, opens, fee)
                trades_all.append(trades.assign(**fields))
                if spec["annual"]:
                    annual_rows.extend(dict(**fields, **r) for r in annual_stats(ledger, trades))
                    traces.append(trace.assign(**fields))
                windows = [("to_latest", ledger)]
                if spec["annual"]:
                    windows.append(("single_year", first_year(ledger, start, opens, fee)))
                for count in (122, 244):
                    if len(ledger) >= count:
                        windows.append((f"sessions_{count}", liquidated_prefix(ledger, count, opens, fee)))
                for window, sample in windows:
                    episodes = trade_episodes(sample, opens, fee)
                    daily = trace[trace.date.le(sample.entry_date.iloc[-1])]
                    summaries.append(dict(**fields, window=window, end=sample.exit_date.iloc[-1],
                                          **metrics(sample, CAPITAL), **trade_stats(episodes), **stability(sample),
                                          capital_blocks=int(daily.action.eq("capital_block").sum()),
                                          tail_exits=int(daily.action.eq("tail_budget_exit").sum()),
                                          model_exits=int(daily.action.isin(["learned_exit", "calibrated_mean_exit"]).sum()),
                                          changes=sum(e["date"] <= sample.entry_date.iloc[-1] for e in events)))
                ledger.to_parquet(out/"ledgers"/f"{start}_{variant}_{fee}_{margin}.parquet", index=False)
        pd.DataFrame(summaries).to_csv(out/"summary.csv", index=False)
        print("evaluated", year, start, spec["start_kind"], flush=True)
    pd.DataFrame(audits).to_csv(out/"validation_audit.csv", index=False)
    pd.DataFrame(choices).to_csv(out/"selection.csv", index=False)
    pd.DataFrame(annual_rows).to_csv(out/"annual_marked.csv", index=False)
    pd.concat(trades_all, ignore_index=True).to_csv(out/"trades.csv", index=False)
    pd.concat(traces, ignore_index=True).to_csv(out/"annual_decisions.csv", index=False)
    (out/"checks.json").write_text(json.dumps(dict(control_matches=matches, accounts=sum(r["window"] == "to_latest" for r in summaries),
                                                  windows=len(summaries), validation_records=len(audits)), indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int)
    parser.add_argument("--reuse-calibration", action="store_true")
    args = parser.parse_args()
    if args.year:
        run_year(args.year)
        return
    OUT.mkdir(exist_ok=True)
    protocol = dict(capital=CAPITAL, max_lots=1, variants=VARIANTS, monthly_profiles=PROFILES,
                    annual_profiles=[*PROFILES, [.0003, .25], [.0006, .15]],
                    monthly_starts="2022-01 through 2026-10, first available session, 58 starts",
                    phase_starts="first five January sessions in each 2022..2026 year; original 3 controls, two primary profiles",
                    common_windows=[122, 244], calibration_window_months=24, calibration_half_life_months=12,
                    mean_calibration="past-OOS weighted affine correction; slope clipped[0,1]; intercept free; per-episode normalization",
                    risk_calibration="same affine mean correction plus raw OOS risk residual weighted q90; bound >= corrected mean",
                    net_lower_bound="raw OOS net residual weighted q10; diagnostic only, <= calibrated mean",
                    minimum_history="122 distinct mature signal days + 60 episodes per direction; otherwise raw mean5, no tail bound",
                    tail_rule="risk90 * last-close one-lot notional > 5% last-close current equity for 2 observations; or calibrated net<=0 for2",
                    native_constraints="original quarterly guarded selection, reserve, max20 holding; same-overlay historical validation",
                    startup="each start resets equity500k, flat holdings, base policy and 5-session entry clock; no yearly reset thereafter",
                    limits="empirical bounds not guaranteed coverage; serial and overlapping samples; all prior history inspected; no new OOT",
                    no_parameter_sweep=True)
    (OUT/"protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    if not args.reuse_calibration:
        calibrate(pd.read_parquet(PREVIOUS/"forecasts.parquet"), pd.read_parquet(SOURCE/"quarterly_predictions.parquet"))
    def worker(year):
        subprocess.run([sys.executable, __file__, "--year", str(year)], check=True)
    with ThreadPoolExecutor(max_workers=3) as executor:
        list(executor.map(worker, range(2022, 2027)))
    for name in ("summary", "validation_audit", "selection", "annual_marked", "trades", "annual_decisions"):
        pd.concat([pd.read_csv(OUT/str(year)/f"{name}.csv") for year in range(2022, 2027)], ignore_index=True).to_csv(OUT/f"{name}.csv", index=False)
    checks = [json.loads((OUT/str(year)/"checks.json").read_text()) for year in range(2022, 2027)]
    (OUT/"checks.json").write_text(json.dumps({key: sum(c[key] for c in checks) for key in checks[0]}, indent=2))


if __name__ == "__main__":
    main()
