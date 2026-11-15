"""Walk-forward close-versus-hold models using actual one-lot position states."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from backtest_index_futures import MULTIPLIERS, ROOT, metrics, simulate
from evaluate_index_futures_directions import CAPITAL, load_bars, margin_reserve
from evaluate_index_futures_side_exits import ENTRY, forecasts

HOLD = 20
HORIZON = 5
FEATURES = ["age", "marked_return", "pullback", "opposing_5", "opposing_20",
            "support_20", "basis", "volatility"]


def replay(groups, quotes, fee, margin, models=None, threshold=.65, reserve=False):
    index = -1
    opened = 0
    first_open = opening = peak = gross = 0.
    held_code = ""
    samples = []
    groups_by_date = {rows[0].entry_date: rows for rows in groups}
    candidates = {r.symbol: {} for rows in groups for r in rows}
    for rows in groups:
        for row in rows:
            candidates[row.symbol][row.entry_date] = row

    position_filter = margin_reserve if reserve else None

    def select(day, positions):
        nonlocal index, opened, first_open, opening, peak, gross, held_code
        index += 1
        rows = groups_by_date[day]
        if positions:
            code, (side, _) = next(iter(positions.items()))
            row = candidates[code[:2]][day]
            assert code == held_code
            age = index - opened
            marked = gross + side * (quotes[row.trade_date, code][1] - opening)
            peak = max(peak, marked)
            opposing, support = ("short", "long") if side > 0 else ("long", "short")
            state = dict(date=day, signal_date=row.trade_date, side=side, age=age,
                         marked_return=marked / first_open, pullback=(peak - marked) / first_open,
                         opposing_5=getattr(row, f"{opposing}_5"),
                         opposing_20=getattr(row, f"{opposing}_20"),
                         support_20=getattr(row, f"{support}_20"),
                         basis=row.f_basis_annual, volatility=row.f_volatility_20)
            early = False
            if age <= HOLD - HORIZON:
                if models is None:
                    # Close immediately, or keep the same product for five
                    # sessions with actual rolls and close at the sixth open.
                    path_day, prior = day, code
                    incremental = quotes[day, code][0] * fee
                    for _ in range(HORIZON):
                        leg = candidates[row.symbol].get(path_day)
                        if leg is None:
                            break
                        if leg.ts_code != prior:
                            incremental -= (quotes[path_day, prior][0] + leg.entry_open) * fee
                        incremental += side * (leg.exit_open - leg.entry_open)
                        path_day, prior = leg.exit_date, leg.ts_code
                    else:
                        incremental -= leg.exit_open * fee
                        samples.append(dict(**state, maturity=path_day, pnl_hold_minus_close=incremental * MULTIPLIERS[row.symbol]))
                else:
                    model = models.get((day[:4], side))
                    if model is not None:
                        probability = model.predict_proba([[state[key] for key in FEATURES]])[0, 1]
                        early = probability >= threshold
                        if early:
                            samples.append(dict(**state, probability=probability))
            if age >= HOLD or early:
                return ("flat", .5, 1, "both", position_filter)
            if row.ts_code != code:
                gross += side * (quotes[day, code][0] - opening)
                held_code, opening = row.ts_code, row.entry_open
            return None
        if index % 5:
            return ("flat", .5, 1, "both", position_filter)
        row = max(rows, key=lambda r: (abs(r.entry_60), tuple(-ord(c) for c in r.symbol)))
        if abs(row.entry_60) <= ENTRY:
            return ("flat", .5, 1, "both", position_filter)
        opened, held_code, opening, peak, gross = index, row.ts_code, row.entry_open, 0., 0.
        first_open = opening
        return ("entry_60", ENTRY, 10_000, "both", position_filter)

    ledger = simulate(groups, "flat", .5, 1, "both", 1, fee,
                      margin_rate=margin, capital=CAPITAL, enforce_margin=True,
                      lot_cap=1, policy_selector=select,
                      position_filter=margin_reserve if reserve else None)
    assert ledger.gross_lots.le(1).all()
    assert np.allclose(ledger.equity, CAPITAL + ledger.pnl.cumsum(), atol=1e-6, rtol=0)
    return ledger, samples


def main():
    out = ROOT / "position_exits"
    out.mkdir(exist_ok=True)
    panel = pd.concat(forecasts(), ignore_index=True)
    panel["flat"] = 0.
    long, short = panel.long_60, panel.short_60
    panel["entry_60"] = np.where((long - short).abs().ge(.10) & panel.etf_ret_1.notna(),
                                 np.where(long > short, long, -short), 0.)
    quotes = load_bars()[["open", "close"]].apply(tuple, axis=1).to_dict()
    groups = [list(g.itertuples(index=False)) for _, g in panel.groupby("entry_date")]
    _, raw = replay(groups, quotes, .0003, .15)
    states = pd.DataFrame(raw)
    states.to_csv(out / "training_states.csv", index=False)
    summaries, annual, audit = [], [], []
    for half_life, reserve in ((0, False), (12, False), (0, True), (12, True)):
        models = {}
        for year in range(2021, 2027):
            cutoff = panel.loc[panel.entry_date.str.startswith(str(year)), "trade_date"].min()
            for side in (1, -1):
                train = states[states.maturity.lt(cutoff) & states.side.eq(side)]
                labels = train.pnl_hold_minus_close.lt(0)
                if len(train) < 30 or labels.nunique() != 2:
                    continue
                age = (pd.Timestamp(cutoff) - pd.to_datetime(train.maturity)).dt.days
                weights = np.exp2(-age.to_numpy() / 365.25) if half_life else np.ones(len(train))
                model = make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=1000))
                model.fit(train[FEATURES].to_numpy(), labels,
                          logisticregression__sample_weight=weights / weights.mean())
                models[str(year), side] = model
                audit.append(dict(year=year, side=side, half_life=half_life, rows=len(train),
                                  last_maturity=train.maturity.max(), cutoff=cutoff))
        for start in ("20220101", "20250101"):
            subset = [g for g in groups if g[0].entry_date >= start]
            for fee, margin in ((.0003, .15), (.0006, .25)):
                for threshold in ((0, .60, .70) if not half_life else (.60, .70)):
                    ledger, exits = replay(subset, quotes, fee, margin,
                                           models if threshold else {}, threshold or 1., reserve)
                    fields = dict(start=start, half_life=half_life, threshold=threshold, margin=margin, reserve=reserve)
                    summaries.append(dict(**fields, **metrics(ledger, CAPITAL),
                                          forced=int(ledger.forced_exit.sum()),
                                          close_long=sum(e["side"] > 0 for e in exits),
                                          close_short=sum(e["side"] < 0 for e in exits)))
                    ledger.to_csv(out / f"{start}_{half_life}_{threshold}_{margin}_{int(reserve)}.csv", index=False)
                    if exits:
                        pd.DataFrame(exits).to_csv(out / f"signals_{start}_{half_life}_{threshold}_{margin}_{int(reserve)}.csv", index=False)
                    equity = CAPITAL
                    for year, part in ledger.groupby(ledger.exit_date.str[:4]):
                        path = np.r_[equity, part.equity]
                        annual.append(dict(**fields, year=year, total_return=path[-1] / equity - 1,
                                           max_drawdown=(path / np.maximum.accumulate(path) - 1).min()))
                        equity = path[-1]
    pd.DataFrame(summaries).to_csv(out / "summary.csv", index=False)
    pd.DataFrame(annual).to_csv(out / "annual.csv", index=False)
    pd.DataFrame(audit).to_csv(out / "training_audit.csv", index=False)


if __name__ == "__main__":
    main()
