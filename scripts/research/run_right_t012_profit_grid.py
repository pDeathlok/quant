"""Frozen-model, profit-first policy grid; no diagnostic-period rule search."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from audit_overnight_right_provenance import sha256
from train_right_t012 import SIGNALS
from train_right_t012_full_registry import FOLDS

from quant.core.paths import PROJECT_ROOT
from quant.research.short_side_groups import RIGHT_GROUP_MEMBERS

EXITS = ["open", "close", "tp1", "tp2", "tp3", "tp5", "half_tp2_close", "gap_minus2_open_else_close"]
ENTRIES = ["any", "positive_not_hot", "calm", "right_reconfirmed"]
RANKS = ["expected_net", "risk_adjusted", "probability"]
ARMS = {"core145": "right_t012_full_registry_models_v1", "full_registry": "right_t012_full_registry_models_v1",
        "full_registry_sequence": "right_t012_full_registry_fusion_v1"}


def exit_returns(frame: pd.DataFrame) -> np.ndarray:
    """Gross daily execution proxies. Unknown exits stay NaN, never filtered."""
    entry = frame.entry_close.to_numpy(float)
    op, hi, lo, cl = (frame[f"next_{p}"].to_numpy(float) for p in ["open", "high", "low", "close"])
    down = frame.down_limit.to_numpy(float)
    valid = np.isfinite(np.column_stack([entry, op, hi, lo, cl, down])).all(axis=1)
    valid &= np.column_stack([entry, op, hi, lo, cl, down]).min(axis=1) > 0
    valid &= frame.next_valid_bar.fillna(False).to_numpy(bool)
    valid &= ~frame.price_limit_inconsistent.fillna(True).to_numpy(bool)
    valid &= (hi + .0051 >= np.maximum(op, cl)) & (lo - .0051 <= np.minimum(op, cl))
    opening = np.where(valid & (op > down + .0051), op / entry - 1, np.nan)
    closing = np.where(valid & (cl > down + .0051), cl / entry - 1, np.nan)
    parts = [opening, closing]
    for threshold in [.01, .02, .03, .05]:
        limit = np.ceil(entry * (1 + threshold) * 100 - 1e-9) / 100
        hit = valid & (hi + 1e-9 >= limit + entry * .001)
        parts.append(np.where(hit, limit / entry - 1, closing))
    parts += [.5 * parts[3] + .5 * closing, np.where(op / entry - 1 <= -.02, opening, closing)]
    return np.column_stack(parts)


def pick_indices(order: np.ndarray, accepted: np.ndarray, dates: np.ndarray, top_k: int) -> np.ndarray:
    """Order is already sorted by date, decreasing score, then symbol."""
    candidates = order[accepted[order]]
    if not len(candidates):
        return candidates
    day = dates[candidates]
    starts = np.maximum.accumulate(np.where(np.r_[True, day[1:] != day[:-1]], np.arange(len(day)), 0))
    return candidates[np.arange(len(day)) - starts < top_k]


def portfolio_metrics(gross: np.ndarray, exit_day: np.ndarray, n_days: int, top_k: int,
                      cost_bps: int, unknown_net: float) -> tuple[dict, np.ndarray]:
    """Fixed initial-capital slots; unknown_net is a scenario, not a loss bound."""
    net = gross - cost_bps / 10000
    known = net[np.isfinite(net)]
    daily = np.bincount(exit_day, weights=np.nan_to_num(net, nan=unknown_net) * .5 / top_k, minlength=n_days)
    wealth = np.r_[1., 1. + np.cumsum(daily)]
    peak = np.maximum.accumulate(wealth)
    wins, losses = known[known > 0], known[known < 0]
    half = n_days // 2
    result = dict(trades=len(net), dates=len(np.unique(exit_day)), unknown=int(np.isnan(net).sum()),
        known_mean_net=float(known.mean()) if len(known) else np.nan,
        known_win=float((known > 0).mean()) if len(known) else np.nan,
        profit_factor=float(wins.sum() / -losses.sum()) if len(losses) else np.nan,
        without_best3_mean=float(np.sort(known)[:-3].mean()) if len(known) > 3 else np.nan,
        net_return_scenario=float(daily.sum()), max_drawdown_scenario=float(np.max(1 - wealth / peak)),
        score=float(daily.mean() - 2 * daily.std(ddof=1) / np.sqrt(n_days)),
        first_half_pnl=float(daily[:half].sum()), second_half_pnl=float(daily[half:].sum()),
        worst_known=float(known.min()) if len(known) else np.nan)
    return result, daily


def entry_masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    move = frame.screen_to_entry_return
    return dict(any=np.ones(len(frame), dtype=bool), positive_not_hot=move.between(0, .07).to_numpy(),
                calm=move.between(-.03, .03).to_numpy(), right_reconfirmed=frame.entry_right.fillna(False).to_numpy(bool))


def cashbook(rows: pd.DataFrame, days: pd.DatetimeIndex, top_k: int, cost_bps: int,
             unknown_net: float) -> tuple[dict, np.ndarray]:
    """Cash-constrained daily book; unresolved positions never release cash."""
    entries = {date: block for date, block in rows.groupby("date")}
    cash, equity, locked_value = 1., 1., 0.
    pending: list[tuple[pd.Timestamp, float, float]] = []
    nav, skipped = [], 0
    for date in days:
        # Reserve new purchases before crediting any same-session sales.
        if date in entries:
            slot = max(0., min(.5 * equity, cash)) / top_k
            for row in entries[date].itertuples():
                if slot <= 1e-12:
                    skipped += 1
                    continue
                pending.append((row.planned_exit_date, slot, row.gross))
                cash -= slot
        remaining = []
        for exit_date, capital, gross in pending:
            if exit_date == date:
                if np.isfinite(gross):
                    cash += capital * (1 + gross - cost_bps / 10000)
                else:
                    locked_value += capital * (1 + unknown_net)
            else:
                remaining.append((exit_date, capital, gross))
        pending = remaining
        if cash < -1e-9:
            raise ValueError("Cashbook borrowed money")
        equity = cash + locked_value + sum(capital for _, capital, _ in pending)
        nav.append(equity)
    wealth = np.r_[1., nav]
    return dict(cashbook_return=float(equity - 1), cashbook_max_drawdown=float(np.max(1 - wealth / np.maximum.accumulate(wealth))),
                cashbook_locked_value=float(locked_value), cashbook_cash=float(cash), cashbook_skipped=skipped), np.asarray(nav)


def pool_masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {"all_right": np.ones(len(frame), dtype=bool), **{
        group: frame[list(members)].eq(True).any(axis=1).to_numpy()
        for group, members in RIGHT_GROUP_MEMBERS.items()}}


def ranked_order(frame: pd.DataFrame, rank: str) -> np.ndarray:
    score = frame.expected_net - .02 * frame.risk if rank == "risk_adjusted" else frame[rank]
    return frame.assign(score=score, row=np.arange(len(frame))).sort_values(
        ["date", "score", "symbol"], ascending=[True, False, True]).row.to_numpy()


def selected_rows(frame: pd.DataFrame, rule: dict) -> np.ndarray:
    accepted = pool_masks(frame)[rule["pool"]] & entry_masks(frame)[rule["entry"]]
    accepted &= frame.expected_net.ge(rule["min_expected"]).to_numpy() & frame.risk.le(rule["max_risk"]).to_numpy()
    accepted &= np.isfinite(frame[["expected_net", "risk", "probability"]]).all(axis=1).to_numpy()
    dates = pd.factorize(frame.date, sort=True)[0]
    return pick_indices(ranked_order(frame, rule["rank"]), accepted, dates, rule["top_k"])


def search_selection(frame: pd.DataFrame, days: pd.DatetimeIndex, fold: str, arm: str) -> list[dict]:
    returns = exit_returns(frame)
    exit_day = days.get_indexer(frame.planned_exit_date)
    if (exit_day < 0).any():
        raise ValueError("Exit outside selection calendar")
    dates = pd.factorize(frame.date, sort=True)[0]
    pools, entries = pool_masks(frame), entry_masks(frame)
    orders = {rank: ranked_order(frame, rank) for rank in RANKS}
    finite = np.isfinite(frame[["expected_net", "risk", "probability"]]).all(axis=1).to_numpy()
    records = []
    combinations = itertools.product(pools, entries, RANKS, [0., .005], [.1, 1.], [1, 3, 5])
    for pool, entry, rank, min_expected, max_risk, top_k in combinations:
        mask = pools[pool] & entries[entry] & finite
        mask &= frame.expected_net.ge(min_expected).to_numpy() & frame.risk.le(max_risk).to_numpy()
        chosen = pick_indices(orders[rank], mask, dates, top_k)
        rule = dict(fold=fold, arm=arm, pool=pool, entry=entry, rank=rank, min_expected=min_expected,
                    max_risk=max_risk, top_k=top_k)
        for j, name in enumerate(EXITS):
            metrics, _ = portfolio_metrics(returns[chosen, j], exit_day[chosen], len(days), top_k, 40, -.1)
            records.append(dict(rule, exit=name, **metrics))
    return records


def main() -> None:
    root = PROJECT_ROOT / "reports/research"
    output = root / "right_t012_profit_grid_v1"
    dataset = root / "right_t012_full_dataset_v4/dataset.parquet"
    audit = json.loads((root / "right_t012_full_dataset_audit_v1/complete.json").read_text())
    source_hashes = {str(dataset): sha256(dataset)}
    if source_hashes[str(dataset)] != audit["dataset_sha256"]:
        raise ValueError("Dataset admission hash mismatch")
    paths = []
    for fold in FOLDS:
        for arm, directory in ARMS.items():
            if not (root / directory / "audit_complete.json").exists():
                raise ValueError("Parent models not audited")
            paths.append(root / directory / f"{fold}_{arm}_net_close_predictions.parquet")
    signal_path = root / "right_t012_history_v1/all_daily_signals.parquet"
    calendar_path = root / "overnight_unfiltered_labels_v1/labels.parquet"
    paths += [signal_path, calendar_path, Path(__file__), root / "right_t012_profit_grid_plan.md"]
    source_hashes.update({str(path): sha256(path) for path in paths})
    output.mkdir(exist_ok=False)
    protocol = dict(sources=source_hashes, folds=FOLDS, arms=ARMS, pools=["all_right", *RIGHT_GROUP_MEMBERS],
        entry=ENTRIES, ranks=RANKS, min_expected=[0., .005], max_risk=[.1, 1.], top_k=[1, 3, 5], exits=EXITS,
        selection_cost=40, report_costs=[20, 40, 60], unknown_scenarios=[-.1, -.3],
        capital="Fixed initial capital50% per entry date,0.5/K per slot, no reinvestment, cash unfilled; scenario PnL curve, not verified self-financing portfolio",
        selection="Per arm/fold best score with>=20dates and unknown0; qualified iff score>0 and both chronological halves profitable; no win-rate gate",
        warnings=["Repeated historical diagnostics; grid selection score is not a multiple-testing-corrected confidence bound",
                  "Daily penetration does not prove fills; unresolved exits are scenarios not forced fills",
                  "Cross-arm common keys; learners had different fit populations; no retraining in this grid"])
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    columns = ["symbol", "date", "screen_date", "planned_exit_date", "entry_close", "next_open", "next_high", "next_low", "next_close",
               "next_valid_bar", "down_limit", "price_limit_inconsistent", "screen_to_entry_return", *SIGNALS]
    base = pd.read_parquet(dataset, columns=columns)
    signals = pd.read_parquet(signal_path, columns=["symbol", "date", *SIGNALS])
    signals["entry_right"] = signals[list(SIGNALS)].eq(True).any(axis=1)
    base = base.merge(signals[["symbol", "date", "entry_right"]], on=["symbol", "date"], how="left", validate="one_to_one")
    calendar = pd.DatetimeIndex(pd.read_parquet(calendar_path, columns=["date"]).date.unique()).sort_values()
    grids, summaries, picks, curves, locked, coverage = [], [], [], [], [], []
    for fold, boundaries in FOLDS.items():
        predictions = {arm: pd.read_parquet(root / directory / f"{fold}_{arm}_net_close_predictions.parquet",
            columns=["symbol", "date", "stage", "expected_net", "probability", "risk"]) for arm, directory in ARMS.items()}
        common = predictions["core145"][["symbol", "date"]]
        for prediction in predictions.values():
            common = common.merge(prediction[["symbol", "date"]], on=["symbol", "date"], validate="one_to_one")
        population = base.merge(common, on=["symbol", "date"], validate="one_to_one")
        for arm, prediction in predictions.items():
            frame = population.merge(prediction, on=["symbol", "date"], validate="one_to_one")
            coverage.append(dict(fold=fold, arm=arm, original=len(prediction), common=len(frame)))
            selection = frame.loc[frame.stage.eq("selection")].reset_index(drop=True)
            days = calendar[(calendar >= boundaries[1]) & (calendar < boundaries[2])]
            print(f"{fold}/{arm}: searching selection rows={len(selection)}", flush=True)
            records = search_selection(selection, days, fold, arm)
            grid = pd.DataFrame(records)
            grids.append(grid)
            eligible = grid.loc[grid.dates.ge(20) & grid.unknown.eq(0)]
            winner = None if eligible.empty else eligible.sort_values(["score", "net_return_scenario", "top_k", "exit", "pool", "entry", "rank", "min_expected", "max_risk"],
                ascending=[False, False, True, True, True, True, True, True, True], kind="stable").iloc[0]
            rule = None if winner is None else {key: winner[key].item() if isinstance(winner[key], np.generic) else winner[key]
                for key in ["fold", "arm", "pool", "entry", "rank", "min_expected", "max_risk", "top_k", "exit"]}
            qualified = bool(winner is not None and winner.score > 0 and winner.first_half_pnl > 0 and winner.second_half_pnl > 0)
            locked.append(dict(fold=fold, arm=arm, rule=rule, selection_qualified=qualified,
                               selection_score=None if winner is None else float(winner.score)))
            # Only this frozen rule is evaluated on diagnostics, never the full grid.
            for stage, start, end in [("selection", boundaries[1], boundaries[2]), ("diagnostic", boundaries[2], boundaries[3])]:
                subset = frame.loc[frame.stage.eq(stage)].reset_index(drop=True)
                stage_days = calendar[(calendar >= start) & (calendar < end)]
                chosen = np.array([], dtype=int) if rule is None else selected_rows(subset, rule)
                rows = subset.iloc[chosen].copy()
                top_k = 1 if rule is None else rule["top_k"]
                gross = np.array([]) if rule is None else exit_returns(rows)[:, EXITS.index(rule["exit"])]
                day_ids = stage_days.get_indexer(rows.planned_exit_date)
                if (day_ids < 0).any():
                    raise ValueError("Exit outside locked stage")
                rows = rows.assign(fold=fold, arm=arm, stage=stage, exit="abstain" if rule is None else rule["exit"],
                                   top_k=top_k, gross=gross, weight=.5 / top_k, selection_qualified=qualified)
                picks.append(rows)
                for cost, stress in itertools.product([20, 40, 60], [-.1, -.3]):
                    metrics, daily = portfolio_metrics(gross, day_ids, len(stage_days), top_k, cost, stress)
                    book, nav = cashbook(rows, stage_days, top_k, cost, stress)
                    summaries.append(dict(fold=fold, arm=arm, stage=stage, cost_bps=cost, unknown_scenario=stress,
                                          selection_qualified=qualified, **metrics, **book))
                    curves.append(pd.DataFrame(dict(date=stage_days, pnl=daily, nav_scenario=1+np.cumsum(daily),
                        cashbook_nav=nav, fold=fold, arm=arm, stage=stage, cost_bps=cost, unknown_scenario=stress)))
            pd.concat(grids).to_csv(output / "selection_grid.csv", index=False)
            (output / "locked_rules.json").write_text(json.dumps(locked, indent=2))
            pd.DataFrame(summaries).to_csv(output / "summary.csv", index=False)
            print(f"Locked {fold}/{arm}: qualified={qualified} rule={rule}", flush=True)
    trades = pd.concat(picks, ignore_index=True)
    trades.to_parquet(output / "locked_picks.parquet", index=False)
    trades.loc[trades.gross.isna() | trades.gross.le(.004)].to_csv(output / "badcases.csv", index=False)
    daily = pd.concat(curves, ignore_index=True)
    daily.to_csv(output / "daily_curves.csv", index=False)
    daily["month"] = daily.date.dt.to_period("M").astype(str)
    daily.groupby(["fold", "arm", "stage", "cost_bps", "unknown_scenario", "month"]).pnl.sum().to_csv(output / "monthly.csv")
    pd.DataFrame(coverage).to_csv(output / "matching_coverage.csv", index=False)
    for path, digest in source_hashes.items():
        if sha256(Path(path)) != digest:
            raise ValueError(f"Source changed: {path}")
    (output / "complete.json").write_text(json.dumps(dict(grid_rows=sum(len(g) for g in grids), locked_rules=len(locked),
        picks_sha256=sha256(output / "locked_picks.parquet"), status="Frozen policy grid complete; no deployment or independently confirmed profitability"), indent=2))
    print("Grid completed", flush=True)


if __name__ == "__main__":
    main()
