"""Walk-forward research on actual CFFEX contracts with a two-lot gross cap."""
from itertools import product

import numpy as np
import pandas as pd
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fetch_index_futures import INDICES, ROOT
from quant.features.joinquant_candidates import price_factors

MULTIPLIERS = {"IF": 300, "IH": 300, "IC": 200, "IM": 200}
CAPITAL = 4_000_000
VALIDATION = "20230101"
HOLDOUT = "20250101"


def build_panel(*, source_root=None, include_latest=False):
    source_root = ROOT if source_root is None else source_root
    panels = []
    coverage = []
    for symbol in INDICES:
        bars = pd.read_parquet(source_root / f"{symbol}_daily.parquet")
        spot = pd.read_parquet(source_root / f"{symbol}_spot.parquet").set_index("trade_date").sort_index()
        contracts = pd.read_parquet(source_root / f"{symbol}_contracts.parquet").set_index("ts_code")
        if bars.duplicated(["trade_date", "ts_code"]).any() or not spot.index.is_unique:
            raise ValueError(f"Duplicate source rows for {symbol}")
        first, last = bars.trade_date.min(), bars.trade_date.max()
        if last != spot.index.max():
            raise ValueError(f"{symbol} futures and index end on different dates")
        expected = spot.loc[first:last].index
        missing = expected.difference(bars.trade_date)
        index_dates = pd.Index(spot.index)
        spans = contracts.loc[bars.ts_code.unique(), ["list_date", "delist_date"]]
        expected_counts = index_dates.searchsorted(spans.delist_date.clip(upper=last), side="right") - index_dates.searchsorted(spans.list_date)
        actual_counts = bars.groupby("ts_code").size().reindex(spans.index)
        if not np.array_equal(actual_counts, expected_counts):
            raise ValueError(f"{symbol} has missing or extra individual contract sessions")
        if len(missing):
            raise ValueError(f"{symbol} missing all contracts on {missing.tolist()}")
        coverage.append(dict(symbol=symbol, contracts=bars.ts_code.nunique(), bars=len(bars),
                             first=first, last=last, spot_first=spot.index.min(),
                             spot_last=spot.index.max(), trading_days=len(expected)))
        bars = bars[bars.open.gt(0) & bars.close.gt(0) & bars.settle.gt(0)].copy()
        bars["expiry"] = bars.ts_code.map(contracts.delist_date)
        bars["days"] = (pd.to_datetime(bars.expiry) - pd.to_datetime(bars.trade_date)).dt.days
        dates = pd.Index(sorted(bars.trade_date.unique()))
        liquid = bars[bars.days.ge(10)].sort_values(
            ["trade_date", "oi", "ts_code"], ascending=[True, False, True]
        ).drop_duplicates("trade_date").set_index("trade_date")
        frame = liquid.join(spot.add_prefix("spot_"), how="inner")
        close = spot.close
        ret = close.pct_change()
        for window in (5, 20, 60, 120):
            frame[f"f_momentum_{window}"] = close.pct_change(window)
            frame[f"f_volatility_{window}"] = ret.rolling(window).std()
        factor_input = spot.reset_index().rename(columns={"trade_date": "date", "vol": "volume"})
        factor_input["symbol"] = symbol
        factor_input["ret"] = ret.to_numpy()
        # Exchange index bars lack a meaningful turnover rate. Only reuse factors independent of it.
        factor_input["turnover_rate"] = np.nan
        factors = price_factors(pl.from_pandas(factor_input)).to_pandas().set_index("date")
        existing = ["jq_VOSC", "jq_WVAD", "jq_MFI14", "jq_arron_up_25", "jq_arron_down_25",
                    "jq_Skewness20", "jq_Kurtosis20"]
        frame = frame.join(factors[existing].add_prefix("f_"))
        frame["f_jq_WVAD"] /= spot.vol.rolling(20).mean()
        frame["f_basis"] = frame.close / frame.spot_close - 1
        frame["f_basis_annual"] = frame.f_basis * 365 / frame.days
        frame["f_basis_change"] = frame.f_basis.diff(5)
        frame["f_oi_change"] = frame.oi.pct_change(5)
        frame["f_intraday"] = frame.close / frame.open - 1
        frame["f_days"] = frame.days
        next_date = pd.Series(dates[1:], index=dates[:-1])
        frame["entry_date"] = next_date
        frame["exit_date"] = frame.entry_date.map(next_date)
        lookup = bars.set_index(["trade_date", "ts_code"])
        for key, field in (("entry", "open"), ("entry", "settle"), ("exit", "open")):
            frame[f"{key}_{field}"] = lookup[field].reindex(
                pd.MultiIndex.from_arrays([frame[f"{key}_date"], frame.ts_code])
            ).to_numpy()
        five_day_exit = next_date
        for _ in range(4):
            five_day_exit = five_day_exit.map(next_date)
        frame["exit_5d_date"] = frame.entry_date.map(five_day_exit)
        exit_5d_open = lookup.open.reindex(
            pd.MultiIndex.from_arrays([frame.exit_5d_date, frame.ts_code])
        ).to_numpy()
        frame["target_5d"] = (exit_5d_open / frame.entry_open - 1) / 5
        frame["f_symbol"] = list(INDICES).index(symbol)
        frame["symbol"] = symbol
        frame["multiplier"] = MULTIPLIERS[symbol]
        frame["target"] = frame.exit_open / frame.entry_open - 1
        panels.append(frame.reset_index())
    pd.DataFrame(coverage).to_csv(source_root / "coverage.csv", index=False)
    panel = pd.concat(panels, ignore_index=True).replace([np.inf, -np.inf], np.nan)
    features = panel.filter(regex="^f_").columns.tolist()
    return panel.dropna(subset=features + ([] if include_latest else ["target", "entry_settle"])).sort_values(
        ["entry_date", "symbol"]
    ), features


def predict(panel, features):
    outputs = []
    for year in range(2020, int(panel.entry_date.max()[:4]) + 1):
        cutoff = f"{year}0101"
        train = panel[panel.exit_date.lt(cutoff)]
        test = panel[panel.entry_date.str.startswith(str(year))].copy()
        if test.empty:
            continue
        for name, model in {
            "ridge": make_pipeline(StandardScaler(), Ridge(alpha=100)),
            "tree": HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=7,
                                                  min_samples_leaf=80, l2_regularization=1),
        }.items():
            model.fit(train[features], train.target.clip(-0.08, 0.08))
            test[name] = model.predict(test[features])
        train_5d = train[train.exit_5d_date.lt(cutoff)].dropna(subset=["target_5d"])
        slow = make_pipeline(StandardScaler(), Ridge(alpha=100))
        slow.fit(train_5d[features], train_5d.target_5d.clip(-0.04, 0.04))
        test["ridge_5d"] = slow.predict(test[features])
        test["ensemble"] = (test.ridge + test.tree) / 2
        test["trend"] = test.f_momentum_20 * 0.04
        test["long_baseline"] = 1.0
        test["carry"] = -test.f_basis_annual / 244
        outputs.append(test)
    return pd.concat(outputs, ignore_index=True)


def simulate(groups, score, threshold, allocation, side, cadence, friction, margin_rate=0.15, capital=CAPITAL, enforce_margin=False, lot_cap=2, risk_filter=None, atomic=False, position_filter=None, policy_selector=None, state_policy_selector=None):
    if policy_selector is not None and state_policy_selector is not None:
        raise ValueError("Use either policy_selector or state_policy_selector")
    positions = {}
    previous_exit = None
    equity = capital
    force_flat = False
    records = []
    policy_start = 0
    for day, rows in enumerate(groups):
        first = rows[0]
        if previous_exit is not None and previous_exit != first.entry_date:
            raise ValueError(f"Missing executable portfolio session after {previous_exit}")
        if any(row.exit_date != first.exit_date for row in rows):
            raise ValueError(f"Contract calendars disagree on {first.entry_date}")
        previous_exit = first.exit_date
        if len({row.symbol for row in rows}) != len(rows):
            raise ValueError(f"Duplicate products on {first.entry_date}")
        quotes = {row.ts_code: row.entry_open * row.multiplier for row in rows}
        if policy_selector is not None or state_policy_selector is not None:
            # Equity is marked to THIS opening from the previous record; no
            # current session's exit/settlement outcome has been applied yet.
            selection = (state_policy_selector(first.entry_date, positions.copy(), equity)
                         if state_policy_selector is not None
                         else policy_selector(first.entry_date, positions))
            if selection is not None:
                score, threshold, cadence, side, position_filter = selection
                policy_start = day
        ranked = sorted(rows, key=lambda row: (-abs(getattr(row, score)), row.symbol))
        desired = {}
        remaining = lot_cap
        for row in ranked:
            value = getattr(row, score)
            if abs(value) <= threshold or (side == "long" and value < 0):
                continue
            lots = min(remaining, allocation)
            desired[row.ts_code] = (lots if value > 0 else -lots, row)
            remaining -= lots
            if remaining == 0:
                break
        if (day - policy_start) % cadence:
            previous_lots = {code[:2]: lots
                             for code, (lots, _) in positions.items()}
            desired = {row.ts_code: (previous_lots[row.symbol], row)
                       for row in rows if row.symbol in previous_lots}
        if risk_filter is not None:
            desired = {code: (lots, row) for code, (lots, row) in desired.items()
                       if lots < 0 or not getattr(row, risk_filter)}
        if position_filter is not None:
            desired = {code: (lots, row) for code, (lots, row) in desired.items()
                       if position_filter(row, lots, equity, margin_rate)}
        # Positions entered at this open have the next open pre-priced. That
        # quote also prices exits if the next signal switches contracts.
        def transaction_cost(target):
            return sum(
                abs(target.get(code, (0, None))[0] - positions.get(code, (0, 0))[0])
                * (quotes[code] if code in quotes else positions[code][1]) * friction
                for code in positions.keys() | target.keys()
            )

        requested = sum(abs(lots) for lots, _ in desired.values())
        prior_margin = sum(abs(lots) * price * margin_rate for lots, price in positions.values())
        forced = enforce_margin and (force_flat or (bool(positions) and equity < prior_margin))
        if forced:
            desired = {}
        if enforce_margin:
            while desired:
                margin = sum(abs(lots) * quotes[code] * margin_rate
                             for code, (lots, _) in desired.items())
                if equity >= margin + transaction_cost(desired):
                    break
                if atomic:
                    desired = {}
                    break
                code = next(reversed(desired))
                lots, row = desired[code]
                if abs(lots) == 1:
                    del desired[code]
                else:
                    desired[code] = (lots - int(np.sign(lots)), row)
        costs = transaction_cost(desired)
        gross = sum(lots * (row.exit_open - row.entry_open) * row.multiplier
                    for lots, row in desired.values())
        open_margin = sum(abs(lots) * row.entry_open * row.multiplier * margin_rate
                          for lots, row in desired.values())
        settlement_pnl = sum(lots * (row.entry_settle - row.entry_open) * row.multiplier
                             for lots, row in desired.values())
        settlement_margin = sum(abs(lots) * row.entry_settle * row.multiplier * margin_rate
                                for lots, row in desired.values())
        if sum(abs(lots) for lots, _ in desired.values()) > 2:
            raise ValueError(f"Position cap exceeded on {first.entry_date}")
        equity_before = equity
        force_flat = bool(desired) and equity_before - costs + settlement_pnl < settlement_margin
        equity += gross - costs
        records.append((first.entry_date, first.exit_date, gross - costs, equity,
                        sum(abs(lots) for lots, _ in desired.values()),
                        min(equity_before - costs - open_margin,
                            equity_before - costs + settlement_pnl - settlement_margin),
                        costs, ";".join(f"{key}:{value[0]:+d}" for key, value in desired.items()),
                        requested, forced, force_flat))
        positions = {code: (lots, row.exit_open * row.multiplier)
                     for code, (lots, row) in desired.items()}
    if positions:
        closing = sum(abs(lots) * price * friction for lots, price in positions.values())
        last = list(records[-1])
        last[2] -= closing
        last[3] -= closing
        last[6] += closing
        records[-1] = tuple(last)
    return pd.DataFrame(records, columns=["entry_date", "exit_date", "pnl", "equity",
                                          "gross_lots", "margin_buffer", "fees", "positions",
                                          "requested_lots", "forced_exit", "settlement_breach"])


def metrics(ledger, capital=CAPITAL):
    equity = np.r_[capital, ledger.equity.to_numpy()]
    returns = np.diff(equity) / equity[:-1]
    deviation = returns.std()
    yearly_returns = pd.Series(returns).groupby(ledger.entry_date.str[:4]).apply(
        lambda values: (1 + values).prod() - 1
    )
    return {
        "worst_calendar_return": yearly_returns.min(),
        "total_return": equity[-1] / capital - 1,
        "annual_return": (equity[-1] / capital) ** (244 / len(returns)) - 1,
        "sharpe": np.sqrt(244) * returns.mean() / deviation if deviation else 0,
        "max_drawdown": min(equity / np.maximum.accumulate(equity) - 1),
        "long_days": int(ledger.positions.str.contains(r":\+").sum()),
        "short_days": int(ledger.positions.str.contains(":-").sum()),
        "flat_days": int(ledger.gross_lots.eq(0).sum()),
        "active_days": int(ledger.gross_lots.gt(0).sum()),
        "min_margin_buffer": ledger.margin_buffer.min(),
        "days": len(ledger),
        "fees": ledger.fees.sum(),
    }


def main():
    panel, features = build_panel()
    predictions = predict(panel, features)
    predictions.to_parquet(ROOT / "predictions.parquet", index=False)
    groups = {}
    for name, sample in {
        "validation": predictions[predictions.entry_date.ge(VALIDATION)
                                  & predictions.exit_date.lt(HOLDOUT)],
        "holdout": predictions[predictions.entry_date.ge(HOLDOUT)],
    }.items():
        groups[name] = [list(group.itertuples(index=False))
                        for _, group in sample.groupby("entry_date", sort=True)]
    configurations = list(product(("ridge", "ridge_5d", "tree", "ensemble", "trend", "carry"),
                                  (0, 0.0002, 0.0005, 0.001, 0.002), (1, 2), ("both", "long"), (1, 5, 10)))
    results = []
    for score, threshold, lots, side, cadence in configurations:
        params = score, threshold, lots, side, cadence
        stats = metrics(simulate(groups["validation"], *params, friction=0.0003))
        results.append(dict(zip(("score", "threshold", "lots_per_product", "side", "cadence"), params),
                            **stats))
    candidates = pd.DataFrame(results)
    candidates.to_csv(ROOT / "validation.csv", index=False)
    eligible = candidates[candidates.active_days.ge(60) & candidates.min_margin_buffer.gt(0)]
    selections = []
    parameter_names = ["score", "threshold", "lots_per_product", "side", "cadence"]
    for name, ordering in {
        "return": ["total_return", "sharpe"],
        "stable": ["sharpe", "max_drawdown"],
        "drawdown": ["max_drawdown", "sharpe"],
        "two_way": ["sharpe", "max_drawdown"],
        "balanced": ["worst_calendar_return", "sharpe"],
    }.items():
        pool = eligible if name != "drawdown" else eligible[eligible.total_return.gt(0)]
        if name == "two_way":
            pool = pool[pool.long_days.ge(20) & pool.short_days.ge(20) & pool.flat_days.ge(20)]
        winner = pool.sort_values(ordering, ascending=False).iloc[0]
        params = tuple(winner[k] for k in parameter_names)
        for period, grouped in groups.items():
            for friction in (0.0003, 0.0006):
                ledger = simulate(grouped, *params, friction=friction,
                                  margin_rate=0.15 if friction == 0.0003 else 0.25)
                assert ledger.gross_lots.le(2).all()
                assert ledger.margin_buffer.gt(0).all(), "Margin breach requires liquidation simulation"
                ledger.to_csv(ROOT / f"{name}_{period}_{friction:.4f}.csv", index=False)
                selections.append(dict(objective=name, period=period, friction=friction,
                                       margin_rate=0.15 if friction == 0.0003 else 0.25,
                                       score=params[0], threshold=params[1], lots=params[2],
                                       side=params[3], cadence=params[4], **metrics(ledger)))
    for symbol in INDICES:
        for period, grouped in groups.items():
            baseline = [[row for row in rows if row.symbol == symbol] for rows in grouped]
            baseline = [rows for rows in baseline if rows]
            ledger = simulate(baseline, "long_baseline", 0, 1, "long", 1, friction=0.0003)
            selections.append(dict(objective=f"long_{symbol}", period=period, friction=0.0003,
                                   margin_rate=0.15, score="long_baseline", threshold=0, lots=1,
                                   side="long", cadence=1, **metrics(ledger)))
    summary = pd.DataFrame(selections)
    summary.to_csv(ROOT / "selected.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
