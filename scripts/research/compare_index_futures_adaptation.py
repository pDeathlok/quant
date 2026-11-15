"""Compare frozen, quarterly refit, and nested quarterly policy selection."""
import argparse
import json
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from backtest_index_futures import ROOT, build_panel, metrics, simulate
from evaluate_index_futures_directions import CAPITAL, load_bars, margin_reserve
from evaluate_index_futures_labels import build_labels
from evaluate_index_futures_lr import ETFS, etf_features
from evaluate_index_futures_relative_indices import TARGETS, EXTERNAL, relative_features
from evaluate_index_futures_reversals import futures_features
from evaluate_index_futures_trade_wins import profit_labels, annual_stats, trade_stats

OUT = ROOT / "adaptation_comparison"
POLICIES = ("base", "aux55", "confidence60", "opposing_exit")
METHODS = ("fixed", "refit", "reselect")
SIGNALS = ("long_20", "short_20", "long_60", "short_60", "win_long", "win_short")


def mature_training(panel, cutoff, maturity):
    beginning = (pd.Timestamp(cutoff) - pd.DateOffset(months=36)).strftime("%Y%m%d")
    return panel[panel[maturity].lt(cutoff) & panel.trade_date.ge(beginning)
                 & panel.etf_ret_1.notna()].copy()


def data_panel():
    source, _ = build_panel()
    source = pd.concat(profit_labels(build_labels(g)) for _, g in source.groupby("symbol"))
    etf = pd.concat(etf_features(symbol, code) for symbol, code in ETFS.items())
    source = source.merge(etf, on=["symbol", "trade_date"], how="left", validate="one_to_one")
    source = source.merge(futures_features(load_bars()), on=["ts_code", "trade_date"], validate="one_to_one")
    prices = []
    for code in dict.fromkeys([*TARGETS.values(), *EXTERNAL.values()]):
        frame = pd.read_parquet(f"data/raw/index_{code}.parquet")
        prices.append(frame.set_index("trade_date").close.rename(code))
    relatives = relative_features(pd.concat(prices, axis=1).sort_index())
    source = source.merge(relatives, on=["symbol", "trade_date"], how="left", validate="one_to_one")
    return source.replace([np.inf, -np.inf], np.nan)


def predictions():
    panel = data_panel()
    sets = json.loads((ROOT / "relative_indices/feature_sets.json").read_text())
    end = pd.read_parquet(ROOT / "relative_indices/predictions.parquet").exit_date.max()
    output = panel[panel.entry_date.ge("20200101") & panel.exit_date.le(end)].copy()
    output["quarter"] = pd.to_datetime(output.entry_date).dt.to_period("Q").astype(str)
    audit, frozen_models = [], {}
    for quarter, test in output.groupby("quarter", sort=True):
        cutoff = test.trade_date.min()
        models = {}
        for horizon in (20, 60, "win"):
            maturity = "profit_end" if horizon == "win" else f"direction_{horizon}_end"
            features = sets["combined"] if horizon == "win" else sets["base"]
            train = mature_training(panel, cutoff, maturity)
            assert len(train) >= 100
            age = (pd.Timestamp(cutoff) - pd.to_datetime(train[maturity])).dt.days.to_numpy()
            weight = np.exp2(-age / 365.25)
            weight /= weight.mean()
            for side, direction in (("long", 1), ("short", -1)):
                label = (train[f"profit_{side}"].astype(int) if horizon == "win"
                         else train[f"direction_{horizon}"].eq(direction).astype(int))
                assert label.nunique() == 2
                model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                      LogisticRegression(C=.1, max_iter=1000))
                model.fit(train[features], label, standardscaler__sample_weight=weight,
                          logisticregression__sample_weight=weight)
                key = f"win_{side}" if horizon == "win" else f"{side}_{horizon}"
                output.loc[test.index, key] = model.predict_proba(test[features])[:, 1]
                models[key] = (model, features)
                audit.append(dict(quarter=quarter, signal=key, cutoff=cutoff, rows=len(train),
                                  last_maturity=train[maturity].max(), first_signal=train.trade_date.min()))
        if quarter.endswith("Q1") and quarter[:4] >= "2022":
            frozen_models[quarter[:4]] = models
        print("trained", quarter, flush=True)
    output["flat"] = 0.
    for symbol in ETFS:
        output[f"long_{symbol}"] = output.symbol.eq(symbol).astype(float)
        output[f"short_{symbol}"] = -output.symbol.eq(symbol).astype(float)
    for year, models in frozen_models.items():
        test = output[output.entry_date.ge(year+"0101")].copy()
        for key, (model, features) in models.items():
            test[key] = model.predict_proba(test[features])[:, 1]
        # Frozen and refitted arms must have exactly the same starting models.
        first = test.quarter.eq(year+"Q1")
        np.testing.assert_allclose(test.loc[first, list(SIGNALS)], output.loc[test.index[first], list(SIGNALS)], atol=1e-12)
        test.to_parquet(OUT / f"frozen_{year}.parquet", index=False)
    pd.DataFrame(audit).to_csv(OUT / "training_audit.csv", index=False)
    output.to_parquet(OUT / "quarterly_predictions.parquet", index=False)
    return output


def groups_for(frame):
    columns = ["entry_date", "exit_date", "trade_date", "symbol", "ts_code", "entry_open",
               "exit_open", "entry_settle", "multiplier", "f_volatility_20", "etf_ret_1",
               "quarter", "flat", *SIGNALS]
    columns += [f"{side}_{symbol}" for symbol in ETFS for side in ("long", "short")]
    return [list(g.itertuples(index=False)) for _, g in frame[columns].groupby("entry_date", sort=True)]


def replay(groups, fee, margin, policy="base", schedule=None):
    by_day = {rows[0].entry_date: rows for rows in groups}
    step, opened = -1, 0
    current = policy
    events = []

    def select(day, positions):
        nonlocal step, opened, current
        step += 1
        rows = by_day[day]
        wanted = schedule.get(day, current) if schedule else current
        if wanted != current:
            events.append(dict(date=day, old=current, new=wanted, closed_for_switch=bool(positions)))
            current = wanted
            # Exit the old rule's position; never silently transfer a holding
            # across strategies or reverse without charging closing/opening fees.
            if positions:
                return ("flat", .5, 1, "both", margin_reserve)
        if positions:
            code, (lots, _) = next(iter(positions.items()))
            row = next(r for r in rows if r.symbol == code[:2])
            opposing = row.short_20 if lots > 0 else row.long_20
            if step - opened >= 20 or (current == "opposing_exit" and opposing >= .35):
                return ("flat", .5, 1, "both", margin_reserve)
            return None
        candidates = []
        if step % 5 == 0:
            for row in rows:
                side = "long" if row.long_60 > row.short_60 else "short"
                confidence = max(row.long_60, row.short_60)
                if pd.isna(row.etf_ret_1) or confidence <= .45 or abs(row.long_60-row.short_60) < .10:
                    continue
                if current == "confidence60" and confidence < .60:
                    continue
                if current == "aux55" and getattr(row, f"win_{side}") < .55:
                    continue
                candidates.append((confidence, row.symbol, side))
        if not candidates:
            return ("flat", .5, 1, "both", margin_reserve)
        _, symbol, side = sorted(candidates, key=lambda x: (-x[0], x[1]))[0]
        opened = step
        return (f"{side}_{symbol}", .5, 10000, "both", margin_reserve)

    ledger = simulate(groups, "flat", .5, 1, "both", 1, fee, margin_rate=margin,
                      capital=CAPITAL, enforce_margin=True, lot_cap=1, policy_selector=select)
    assert ledger.gross_lots.le(1).all()
    np.testing.assert_allclose(ledger.equity, CAPITAL + ledger.pnl.cumsum(), atol=1e-6, rtol=0)
    return ledger, events


def trade_episodes(ledger, opens, fee):
    """Fast one-lot accounting, independently reconciled to the daily ledger."""
    active, result = None, []
    last = None
    for row in ledger.itertuples():
        last = row
        code, _, sign = row.positions.partition(":")
        exposure = (code[:2], int(sign)) if code else None
        if active and (active["product"], active["direction"]) != exposure:
            active["pnl"] -= opens[row.entry_date, active["code"]] * active["multiplier"] * fee
            result.append(dict(active, exit=row.entry_date))
            active = None
        if not code:
            continue
        product, direction = exposure
        multiplier = 300 if product in ("IF", "IH") else 200
        if active is None:
            active = dict(entry=row.entry_date, code="", product=product, direction=direction,
                          side="long" if direction > 0 else "short", multiplier=multiplier, pnl=0., days=0)
        opening, ending = opens[row.entry_date, code], opens[row.exit_date, code]
        if active["code"] != code:
            active["pnl"] -= opening * multiplier * fee
            if active["code"]:
                active["pnl"] -= opens[row.entry_date, active["code"]] * multiplier * fee
        active["pnl"] += direction * (ending-opening) * multiplier
        active["days"] += 1
        active["code"] = code
    if active:
        active["pnl"] -= opens[last.exit_date, active["code"]] * active["multiplier"] * fee
        result.append(dict(active, exit=last.exit_date))
    trades = pd.DataFrame(result) if result else pd.DataFrame(columns=["entry", "exit", "pnl", "side"])
    assert np.isclose(trades.pnl.sum(), ledger.pnl.sum(), atol=1e-6)
    return trades


def quality(ledger, day, trade_count):
    path = np.r_[CAPITAL, ledger.equity]
    returns = np.diff(path) / path[:-1]
    age = (pd.Timestamp(day) - pd.to_datetime(ledger.exit_date)).dt.days.to_numpy()
    weight = np.exp2(-age/365.25)
    mean = np.average(returns, weights=weight)
    downside = np.sqrt(np.average(np.minimum(returns, 0)**2, weights=weight))
    return dict(weighted_mean=mean, weighted_sortino=np.sqrt(244)*mean/max(downside, 1e-6),
                max_drawdown=(path/np.maximum.accumulate(path)-1).min(), trades=trade_count,
                active_days=int(ledger.gross_lots.gt(0).sum()), forced=int(ledger.forced_exit.sum()),
                total_return=path[-1]/CAPITAL-1, days=len(ledger))


def choose_policy(scores, current):
    base = scores["base"]
    eligible = {name: row for name, row in scores.items()
                if row["days"] >= 244 and row["trades"] >= max(6, .5*base["trades"])
                and row["active_days"] >= max(40, .5*base["active_days"])
                and row["forced"] == 0 and row["total_return"] > 0 and row["weighted_mean"] > 0
                and row["max_drawdown"] >= base["max_drawdown"]-1e-12}
    if not eligible:
        return current, "insufficient_evidence_keep_current"
    winner = max(eligible, key=lambda name: (eligible[name]["weighted_sortino"], name == current, name))
    return winner, "historical_risk_adjusted_selection"


def history_before(frame, cutoff):
    beginning = (pd.Timestamp(cutoff) - pd.DateOffset(months=36)).strftime("%Y%m%d")
    return frame[frame.entry_date.ge(beginning) & frame.exit_date.le(cutoff)]


def stability(ledger, initial=CAPITAL):
    path = np.r_[initial, ledger.equity]
    underwater = path[1:] < np.maximum.accumulate(path)[1:] - 1e-6
    run = maximum = 0
    for flag in underwater:
        run = run+1 if flag else 0
        maximum = max(maximum, run)
    monthly = ledger.assign(month=ledger.exit_date.str[:6]).groupby("month").pnl.sum()
    return dict(positive_month_ratio=float(monthly.gt(0).mean()),
                max_underwater_sessions=maximum, forced=int(ledger.forced_exit.sum()),
                settlement_breaches=int(ledger.settlement_breach.sum()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-predictions", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    protocol = dict(capital=CAPITAL, max_lots=1, methods=METHODS, policies=POLICIES,
                    training_months=36, training_half_life=12, validation_months=36,
                    validation_half_life=12, update="calendar quarter", initial_policy="base",
                    selection="positive recent-weighted mean and cumulative net return, >=6 trades and >=half baseline coverage, no forced exit, drawdown no worse than base; then weighted Sortino",
                    fallback="retain current policy; not a guarantee of safety",
                    policy_switch="close at next open, resume entries on next eligible 5-session grid; pay all costs",
                    starts=["20220101", "20250101"], costs=[.0003, .0006], margins=[.15, .25],
                    note="Existing 2020-2026 history is exploratory, not untouched OOT; label models unchanged to isolate adaptation")
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    panel = pd.read_parquet(OUT / "quarterly_predictions.parquet") if args.reuse_predictions else predictions()
    opens = load_bars().open.to_dict()
    rank_audit = []

    @lru_cache(maxsize=None)
    def rankings(day, cutoff, fee, margin):
        past = history_before(panel, cutoff)
        groups = groups_for(past)
        scores = {}
        for policy in POLICIES:
            ledger, _ = replay(groups, fee, margin, policy)
            trades = trade_episodes(ledger, opens, fee)
            scores[policy] = quality(ledger, day, len(trades))
            rank_audit.append(dict(date=day, cutoff=cutoff, fee=fee, margin=margin, policy=policy,
                                   first_validation_entry=past.entry_date.min(), last_validation_exit=past.exit_date.max(),
                                   **scores[policy]))
        assert past.exit_date.max() <= cutoff < day
        return scores

    annual, summaries, episodes, decisions, switches = [], [], [], [], []
    windows = [("continuous", start, panel[panel.entry_date.ge(start)]) for start in protocol["starts"]]
    for year in ("2022", "2023", "2024", "2025", "2026"):
        windows.append(("annual_reset", year+"0101", panel[panel.entry_date.str.startswith(year) & panel.exit_date.str.startswith(year)]))
    for evaluation, start, sample in windows:
        frozen = pd.read_parquet(OUT / f"frozen_{start[:4]}.parquet")
        frozen = frozen[frozen.entry_date.isin(sample.entry_date.unique())]
        changing_groups, fixed_groups = groups_for(sample), groups_for(frozen)
        for fee in (.0003, .0006):
            for margin in (.15, .25):
                current, schedule = "base", {}
                periods = list(sample.groupby("quarter", sort=True))
                for i, (quarter, part) in enumerate(periods):
                    day, cutoff = part.entry_date.min(), part.trade_date.min()
                    previous = current
                    if i:
                        current, reason = choose_policy(rankings(day, cutoff, fee, margin), current)
                    else:
                        reason = "identical_initial_policy"
                    schedule[day] = current
                    decisions.append(dict(evaluation=evaluation, start=start, fee=fee, margin=margin,
                                          quarter=quarter, date=day, cutoff=cutoff, previous=previous,
                                          selected=current, changed=current != previous, reason=reason))
                first_quarter_ledgers = []
                for method in METHODS:
                    fields = dict(evaluation=evaluation, start=start, fee=fee, margin=margin, method=method)
                    groups = fixed_groups if method == "fixed" else changing_groups
                    ledger, events = replay(groups, fee, margin, schedule=schedule if method == "reselect" else None)
                    trades = trade_episodes(ledger, opens, fee)
                    first_dates = periods[0][1].entry_date.unique()
                    first_quarter_ledgers.append(ledger[ledger.entry_date.isin(first_dates)].reset_index(drop=True))
                    stat = dict(**fields, **metrics(ledger, CAPITAL), **trade_stats(trades), **stability(ledger),
                                policy_changes=len(events), switch_closures=sum(e["closed_for_switch"] for e in events))
                    summaries.append(stat)
                    equity = CAPITAL
                    extras = {}
                    for year, part in ledger.groupby(ledger.exit_date.str[:4]):
                        extras[year] = stability(part, equity)
                        equity = part.equity.iloc[-1]
                    for row in annual_stats(ledger, trades):
                        if row["side"] == "all":
                            row.update(extras[row["year"]])
                        annual.append(dict(**fields, **row))
                    episodes.append(trades.assign(**fields))
                    switches.extend(dict(**fields, **event) for event in events)
                    ledger.to_csv(OUT / f"ledger_{evaluation}_{start}_{method}_{fee}_{margin}.csv", index=False)
                for other in first_quarter_ledgers[1:]:
                    pd.testing.assert_frame_equal(first_quarter_ledgers[0], other)
        pd.DataFrame(rank_audit).to_csv(OUT / "validation_audit.csv", index=False)
        pd.DataFrame(decisions).to_csv(OUT / "selection.csv", index=False)
        print("evaluated", evaluation, start, flush=True)
    pd.DataFrame(annual).to_csv(OUT / "annual.csv", index=False)
    pd.DataFrame(summaries).to_csv(OUT / "summary.csv", index=False)
    pd.concat(episodes, ignore_index=True).to_csv(OUT / "trades.csv", index=False)
    pd.DataFrame(switches).to_csv(OUT / "switches.csv", index=False)


if __name__ == "__main__":
    main()
