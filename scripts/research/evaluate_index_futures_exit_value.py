"""Quarterly causal hold-versus-close return and adverse-path regression."""
import argparse
import json
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from backtest_index_futures import ROOT, metrics
from compare_index_futures_adaptation import OUT as SOURCE, POLICIES, groups_for, history_before, stability, trade_episodes
from evaluate_index_futures_directions import CAPITAL, load_bars
from evaluate_index_futures_risk_paths import OUT as CONTROLS, risk_replay, first_year, diagnose_trades
from evaluate_index_futures_trade_wins import annual_stats, trade_stats
from refine_index_futures_adaptation import capital_quality, decide

OUT = ROOT / "exit_value"
VARIANTS = ("control", "loss_exit", "mean5", "mean_remaining", "utility_remaining")
FEES = (.0003, .0006)
TARGETS = [f"{kind}_{horizon}_{bps}" for bps in (3, 6) for horizon in ("5", "remaining") for kind in ("net", "risk")]
STATE_FEATURES = ["age_fraction", "marked_return", "pullback", "own20", "opposite20", "own60", "opposite60"]
PRODUCT_FEATURES = [f"product_{product}" for product in ("IF", "IH", "IC", "IM")]


def hold_path(rows, index, direction, opens, closes, fee, horizon):
    """Net incremental CNY vs closing the old contract at the decision open.

Only labels inspect these future quotes. Paid entry/previous roll costs cancel.
Risk uses executable-open and close marks, not unknown intraday extrema.
    """
    old = rows[index-1].ts_code
    first = rows[index]
    multiplier = first.multiplier
    incremental = opens[first.entry_date, old]*multiplier*fee  # avoided immediate close fee
    gross, worst = 0., 0.
    path = []
    for row in rows[index:index+horizon]:
        if row.ts_code != old:
            incremental -= (opens[row.entry_date, old]+row.entry_open)*multiplier*fee
        open_value = gross+incremental-row.entry_open*multiplier*fee
        close = closes[row.entry_date, row.ts_code]
        close_value = gross+incremental+direction*(close-row.entry_open)*multiplier-close*multiplier*fee
        gross += direction*(row.exit_open-row.entry_open)*multiplier
        end_value = gross+incremental-row.exit_open*multiplier*fee
        worst = min(worst, open_value, close_value, end_value)
        path.append((row.exit_date, end_value, -worst))
        old = row.ts_code
    return path


def build_states(panel, bars, market_features):
    opens, closes = bars.open.to_dict(), bars.close.to_dict()
    records = []
    for symbol, frame in panel.groupby("symbol"):
        rows = list(frame.sort_values("entry_date").itertuples(index=False))
        assert all(a.exit_date == b.entry_date for a, b in zip(rows, rows[1:]))
        paths = {}
        for entry, first in enumerate(rows):
            # Generate all eligible candidate entries, even if never selected in
            # a particular funded portfolio; do not select on realized winners.
            direction = 1 if first.long_60 > first.short_60 else -1
            if (pd.isna(first.etf_ret_1) or max(first.long_60, first.short_60) <= .45
                    or abs(first.long_60-first.short_60) < .10):
                continue
            entry_notional = first.entry_open*first.multiplier
            gross, peak = 0., 0.
            for age in range(1, min(20, len(rows)-entry)):
                i = entry+age
                row, previous = rows[i], rows[i-1]
                gross += direction*(previous.exit_open-previous.entry_open)*row.multiplier
                marked = gross+direction*(closes[row.trade_date, previous.ts_code]-
                                          opens[row.entry_date, previous.ts_code])*row.multiplier
                peak = max(peak, marked)
                notional = closes[row.trade_date, previous.ts_code]*row.multiplier
                record = dict(entry=first.entry_date, date=row.entry_date, signal_date=row.trade_date,
                              symbol=symbol, direction=direction, quarter=row.quarter, age=age,
                              episode=f"{symbol}:{first.entry_date}:{direction}", notional=notional,
                              maturity=None, age_fraction=age/20, marked_return=marked/entry_notional,
                              pullback=(peak-marked)/entry_notional,
                              own20=row.long_20 if direction > 0 else row.short_20,
                              opposite20=row.short_20 if direction > 0 else row.long_20,
                              own60=row.long_60 if direction > 0 else row.short_60,
                              opposite60=row.short_60 if direction > 0 else row.long_60)
                record.update({feature: getattr(row, feature) for feature in market_features})
                record.update({f"product_{product}": float(product == symbol) for product in ("IF", "IH", "IC", "IM")})
                for fee in FEES:
                    key = (i, direction, fee)
                    if key not in paths:
                        paths[key] = hold_path(rows, i, direction, opens, closes, fee, 19)
                    path = paths[key]
                    for name, horizon in (("5", min(5, 20-age)), ("remaining", 20-age)):
                        if len(path) >= horizon:
                            maturity, net, risk = path[horizon-1]
                            record[f"net_{name}_{round(fee*10000)}"] = net/notional
                            record[f"risk_{name}_{round(fee*10000)}"] = risk/notional
                            if name == "remaining":
                                record["maturity"] = maturity
                records.append(record)
        print("states", symbol, len(records), flush=True)
    states = pd.DataFrame(records).replace([np.inf, -np.inf], np.nan)
    assert not states.duplicated(["entry", "date", "symbol"]).any()
    assert states.signal_date.lt(states.date).all()
    return states


def training_before(states, cutoff, direction):
    start = (pd.Timestamp(cutoff)-pd.DateOffset(months=36)).strftime("%Y%m%d")
    return states[states.direction.eq(direction) & states.signal_date.ge(start)
                  & states.maturity.notna() & states.maturity.lt(cutoff)].dropna(subset=TARGETS)


def state_weights(train, cutoff):
    age = (pd.Timestamp(cutoff)-pd.to_datetime(train.maturity)).dt.days.to_numpy()
    # Counts use ONLY the mature training subset, so no future episode length leaks.
    weight = np.exp2(-age/365.25)/train.groupby("episode").episode.transform("size").to_numpy()
    return weight/weight.mean()


def fit_forecasts(states, features, panel):
    output = states.copy()
    predictions = [f"pred_{target}" for target in TARGETS]
    output[predictions] = np.nan
    audit, coefficients = [], []
    for quarter, part in panel.groupby("quarter", sort=True):
        cutoff = part.trade_date.min()
        for direction in (1, -1):
            train = training_before(states, cutoff, direction)
            test = states[states.quarter.eq(quarter)&states.direction.eq(direction)]
            episodes, days = train.episode.nunique(), train.signal_date.nunique()
            detail = dict(quarter=quarter, direction=direction, cutoff=cutoff, rows=len(train),
                          episodes=episodes, signal_days=days, last_maturity=train.maturity.max() if len(train) else None)
            if days < 244 or episodes < 60 or test.empty:
                audit.append(dict(**detail, trained=False))
                continue
            assert train.maturity.max() < cutoff <= test.signal_date.min()
            weights = state_weights(train, cutoff)
            # Fixed normalized L2 penalty; no search on historical evaluation years.
            alpha = .1*len(train)
            model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                                  StandardScaler(), Ridge(alpha=alpha))
            model.fit(train[features], train[TARGETS], standardscaler__sample_weight=weights,
                      ridge__sample_weight=weights)
            predicted = model.predict(test[features])
            for j, name in enumerate(TARGETS):
                if name.startswith("risk"):
                    predicted[:, j] = np.maximum(0, predicted[:, j])
            output.loc[test.index, predictions] = predicted
            audit.append(dict(**detail, trained=True, alpha=alpha))
            for name, coef in zip(TARGETS, model[-1].coef_):
                coefficients.extend(dict(quarter=quarter, direction=direction, target=name, feature=f, coefficient=c)
                                    for f, c in zip(features, coef))
        print("trained exits", quarter, flush=True)
    pd.DataFrame(audit).to_csv(OUT/"training_audit.csv", index=False)
    pd.DataFrame(coefficients).to_csv(OUT/"coefficients.csv", index=False)
    output.to_parquet(OUT/"forecasts.parquet", index=False)
    return output


def exit_callback(lookup, variant, fee):
    streak, previous_key = 0, None
    bps = round(fee*10000)
    horizon = "5" if variant == "mean5" else "remaining"

    def choose(day, entry, row, lots):
        nonlocal streak, previous_key
        key = (entry, row.symbol)
        state = lookup.get((entry, day, row.symbol))
        if state is None:
            raise AssertionError(f"Missing causal candidate state: {entry} {day} {row.symbol}")
        assert state.direction == lots
        if variant == "fixed5":
            return state.age >= 5, dict(exit_action="fixed5_exit", exit_entry=entry)
        net, risk = getattr(state, f"pred_net_{horizon}_{bps}"), getattr(state, f"pred_risk_{horizon}_{bps}")
        utility = net-(.5*risk if variant == "utility_remaining" else 0.)
        if key != previous_key:
            streak = 0
        streak = streak+1 if np.isfinite(utility) and utility <= 0 else 0
        previous_key = key
        return streak >= 2, dict(exit_model_available=np.isfinite(net), predicted_incremental=net,
                                predicted_adverse=risk, predicted_utility=utility, exit_confirmations=streak,
                                predicted_incremental_cny=net*state.notional,
                                predicted_adverse_cny=risk*state.notional, exit_entry=entry)
    return choose


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-forecasts", action="store_true")
    parser.add_argument("--supplement-fixed5", action="store_true")
    args = parser.parse_args()
    if args.supplement_fixed5 and not args.reuse_forecasts:
        parser.error("The supplementary control must reuse frozen forecasts")
    OUT.mkdir(exist_ok=True)
    variants = (*VARIANTS, "fixed5") if args.supplement_fixed5 else VARIANTS
    market_features = json.loads((ROOT/"relative_indices/feature_sets.json").read_text())["combined"]
    features = market_features+STATE_FEATURES+PRODUCT_FEATURES
    protocol = dict(capital=CAPITAL, max_lots=1, variants=VARIANTS, fees=FEES, margins=[.15, .25],
                    start_years=list(range(2022, 2027)), features=features,
                    labels="net hold-minus-close over min(5,remaining) or remaining sessions; maximum adverse open/close incremental path",
                    terminal="fixed remaining horizon, actual rolls, no future discretionary model exit or margin-dependent stopping in label",
                    risk_rule="predicted net - 0.5*predicted adverse; negative for two consecutive observed closes => next-open exit",
                    mean_rule="predicted net <= 0 for two consecutive observed closes => next-open exit",
                    models="separate long/short Ridge multi-output, normalized L2=.1, no hyperparameter search",
                    train="quarterly; past36months; all targets mature strictly before first signal cutoff; 12m half-life; per-episode state normalization",
                    minimum_history="244 distinct signal days and 60 candidate episodes per direction; otherwise keep native exits",
                    sampling="every base-signal-eligible entry date/product/direction, ages1..19 regardless of realized outcome or funded selection",
                    controls="unchanged original quarterly guarded reselection and 5% close-loss exit",
                    evaluation="same-overlay historical candidate replay; each year independent single-year and start-to-latest windows",
                    limitations="overlapping candidate states not independent; mean adverse prediction not VaR; previously inspected history not fresh OOT; daily execution")
    if args.supplement_fixed5:
        supplement = dict(timing="added after primary results, purely diagnostic; not a preregistered primary arm",
                          rule="exit at age5 sessions, with native early exits and original guarded quarterly selection",
                          reason="distinguish learned value from mechanically shorter holdings; no parameter search",
                          reuse_frozen_forecasts=True)
        (OUT/"supplement_protocol.json").write_text(json.dumps(supplement, indent=2))
        if not (OUT/"primary_summary.csv").exists():
            (OUT/"primary_summary.csv").write_bytes((OUT/"summary.csv").read_bytes())
    else:
        (OUT/"protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    panel, bars = pd.read_parquet(SOURCE/"quarterly_predictions.parquet"), load_bars()
    opens, closes = bars.open.to_dict(), bars.close.to_dict()
    if args.reuse_forecasts:
        forecasts = pd.read_parquet(OUT/"forecasts.parquet")
    else:
        states = build_states(panel, bars, market_features)
        states.to_parquet(OUT/"states.parquet", index=False)
        forecasts = fit_forecasts(states, features, panel)
    lookup = {(r.entry, r.date, r.symbol): r for r in forecasts.itertuples(index=False)}
    audits, choices, summaries, annual, trades_all, traces, switches = [], [], [], [], [], [], []
    matches = 0

    @lru_cache(maxsize=None)
    def history(cutoff):
        past = history_before(panel, cutoff)
        assert past.exit_date.max() <= cutoff
        return groups_for(past)

    def replay(groups, fee, margin, variant, capital=CAPITAL, policy="base", decision=None, trace=False):
        callback = exit_callback(lookup, variant, fee) if variant not in ("control", "loss_exit") else None
        return risk_replay(groups, opens, closes, fee, margin, capital=capital, policy=policy,
                           decision=decision, trace=trace, variant=variant, exit_decider=callback)

    @lru_cache(maxsize=512)
    def rankings(day, cutoff, equity, fee, margin, variant):
        groups = history(cutoff)
        scores = {}
        for policy in POLICIES:
            ledger, _, _ = replay(groups, fee, margin, variant, capital=equity, policy=policy)
            trades = trade_episodes(ledger, opens, fee)
            scores[policy] = capital_quality(ledger, day, equity, len(trades))
            audits.append(dict(date=day, cutoff=cutoff, validation_initial_equity=equity, fee=fee, margin=margin,
                               variant=variant, policy=policy, last_validation_exit=ledger.exit_date.iloc[-1], **scores[policy]))
        assert groups[-1][0].exit_date <= cutoff < day
        return scores

    for year in protocol["start_years"]:
        start = f"{year}0101"
        groups = groups_for(panel[panel.entry_date.ge(start)])
        for fee in FEES:
            for margin in protocol["margins"]:
                for variant in variants:
                    fields = dict(start=start, fee=fee, margin=margin, variant=variant)
                    def decision(day, cutoff, incumbent, equity, positions, rows):
                        scores = rankings(day, cutoff, equity, fee, margin, variant)
                        selected, reason, detail = decide(scores, incumbent, rows, positions, equity, fee, margin, True)
                        choices.append(dict(**fields, date=day, cutoff=cutoff, equity=equity,
                                            incumbent=incumbent, selected=selected, reason=reason, **detail))
                        return selected
                    ledger, events, trace = replay(groups, fee, margin, variant, decision=decision, trace=True)
                    if variant in ("control", "loss_exit"):
                        old = pd.read_csv(CONTROLS/f"ledger_{start}_{variant}_{fee}_{margin}.csv", dtype={"entry_date": str, "exit_date": str}).fillna({"positions": ""})
                        pd.testing.assert_frame_equal(ledger, old, check_dtype=False, atol=1e-6, rtol=1e-12)
                        matches += 1
                    trades = trade_episodes(ledger, opens, fee)
                    trades_all.append(diagnose_trades(ledger, trades, trace, closes, opens, fee).assign(**fields))
                    annual.extend(dict(**fields, **r) for r in annual_stats(ledger, trades))
                    traces.append(trace.assign(**fields))
                    switches.extend(dict(**fields, **e) for e in events)
                    for window, sample in (("to_latest", ledger), ("single_year", first_year(ledger, start, opens, fee))):
                        episodes = trade_episodes(sample, opens, fee)
                        summaries.append(dict(**fields, window=window, end=sample.exit_date.iloc[-1],
                                              **metrics(sample, CAPITAL), **trade_stats(episodes), **stability(sample)))
                    ledger.to_csv(OUT/f"ledger_{start}_{variant}_{fee}_{margin}.csv", index=False)
                print("evaluated", start, fee, margin, flush=True)
        pd.DataFrame(summaries).to_csv(OUT/"summary.csv", index=False)
        pd.DataFrame(audits).to_csv(OUT/"validation_audit.csv", index=False)
        pd.DataFrame(choices).to_csv(OUT/"selection.csv", index=False)
    pd.DataFrame(annual).to_csv(OUT/"annual_marked.csv", index=False)
    pd.concat(trades_all, ignore_index=True).to_csv(OUT/"trades.csv", index=False)
    pd.concat(traces, ignore_index=True).to_csv(OUT/"daily_decisions.csv", index=False)
    pd.DataFrame(switches).to_csv(OUT/"switches.csv", index=False)
    accounts = len(protocol["start_years"])*len(FEES)*len(protocol["margins"])*len(variants)
    if args.supplement_fixed5:
        prior = pd.read_csv(OUT/"primary_summary.csv", dtype={"start": str, "end": str})
        current = pd.DataFrame(summaries).query("variant != 'fixed5'")
        # A diagnostic extra arm must not change any of the original results.
        pd.testing.assert_frame_equal(prior.reset_index(drop=True), current.reset_index(drop=True),
                                      check_dtype=False, atol=1e-8, rtol=1e-10)
    (OUT/"checks.json").write_text(json.dumps(dict(control_matches=matches, accounts=accounts, evaluated_windows=accounts*2,
                                                  candidate_states=len(forecasts), causal_validation=True,
                                                  trade_reconciliation=True), indent=2))


if __name__ == "__main__":
    main()
