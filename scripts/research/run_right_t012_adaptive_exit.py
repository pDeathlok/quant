"""Choose next-session exit using predicted utility, never realized payoffs."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from audit_overnight_right_provenance import sha256
from run_right_t012_profit_grid import cashbook, entry_masks, pick_indices, portfolio_metrics
from train_right_t012_payoff_v3 import (
    META,
    OUTPUT,
    ROOT,
    TARGETS,
    load_protocol,
    payoff_selection,
    pools,
)


def with_action(frame: pd.DataFrame, penalty: float) -> pd.DataFrame:
    gain = frame[[f"gain_{t}" for t in TARGETS]].to_numpy()
    loss = frame[[f"loss_{t}" for t in TARGETS]].to_numpy()
    scores = gain - (1 + penalty) * loss
    action = np.where(np.isfinite(scores), scores, -np.inf).argmax(axis=1)
    positions = np.arange(len(frame))
    result = frame.copy()
    result["gain"], result["loss"] = gain[positions, action], loss[positions, action]
    result["selected_exit"] = np.asarray(TARGETS)[action]
    # Realized outcome lookup occurs only after the action is irrevocably chosen.
    result["gross"] = frame[[f"gross_{t}" for t in TARGETS]].to_numpy()[positions, action]
    return result


def load_group(protocol: dict, fold: str, arm: str, window: int) -> pd.DataFrame:
    base = pd.read_parquet(OUTPUT / f"{fold}_prepared.parquet", columns=[*META, "stage", *[f"gross_{t}" for t in TARGETS]])
    base = base.loc[base.stage.isin(["selection", "diagnostic"])]
    for target in TARGETS:
        config = next(c for c in protocol["configs"] if c["fold"] == fold and c["arm"] == arm and c["window"] == window and c["exit"] == target)
        prediction = pd.read_parquet(OUTPUT / f"{config['model_id']}_predictions.parquet")
        prediction = prediction.rename(columns={"gain": f"gain_{target}", "loss": f"loss_{target}"})
        count = len(base)
        base = base.merge(prediction.drop(columns="stage"), on=["symbol", "date"], validate="one_to_one")
        if len(base) != count:
            raise ValueError("Actions have different candidate populations")
    return base.reset_index(drop=True)


def search(frame: pd.DataFrame, days: pd.DatetimeIndex, config: dict) -> pd.DataFrame:
    records = []
    day_ids = days.get_indexer(frame.planned_exit_date)
    if (day_ids < 0).any():
        raise ValueError("Stage boundary crossed")
    date_ids = pd.factorize(frame.date, sort=True)[0]
    pool_masks, entries = pools(frame), entry_masks(frame)
    for penalty in [0., .5, 1.]:
        action = with_action(frame, penalty)
        mu = action.gain-action.loss
        score = mu-penalty*action.loss
        order = action.assign(score=score, row=np.arange(len(action))).sort_values(["date", "score", "symbol"], ascending=[True, False, True]).row.to_numpy()
        for pool, entry, minimum, k in itertools.product(pool_masks, entries, [0., .005], [1, 3, 5]):
            mask = pool_masks[pool] & entries[entry] & mu.ge(minimum).to_numpy() & np.isfinite(score).to_numpy()
            chosen = pick_indices(order, mask, date_ids, k)
            metrics, _ = portfolio_metrics(action.gross.to_numpy()[chosen], day_ids[chosen], len(days), k, 40, -.1)
            records.append(dict(config, pool=pool, entry=entry, penalty=penalty, min_mu=minimum, top_k=k, **metrics))
    return pd.DataFrame(records)


def main() -> None:
    protocol = load_protocol()
    if not (OUTPUT / "audit_complete.json").exists():
        raise ValueError("Corrected fixed-policy experiment must finish its audit first")
    training = json.loads((OUTPUT / "training_complete.json").read_text())
    sources = {str(Path(__file__)): sha256(Path(__file__)),
        str(ROOT / "right_t012_adaptive_exit_plan.md"): sha256(ROOT / "right_t012_adaptive_exit_plan.md"),
        str(OUTPUT / "protocol.json"): sha256(OUTPUT / "protocol.json"),
        str(OUTPUT / "audit_complete.json"): sha256(OUTPUT / "audit_complete.json")}
    for model in training["models"]:
        path = OUTPUT / f"{model['id']}_predictions.parquet"
        if sha256(path) != model["predictions_sha256"]:
            raise ValueError("Prediction checkpoint changed")
        sources[str(path)] = model["predictions_sha256"]
    output = ROOT / "right_t012_adaptive_exit_v1"
    output.mkdir(exist_ok=False)
    (output / "protocol.json").write_text(json.dumps(dict(sources=sources, action="Buy-time argmax gain-(1+lambda)*loss; tie order=open,close,tp2,tp5",
        selection="Same eligibility and ranking protocol as corrected payoff v3; diagnostics only frozen rules", scope="Exploratory reused dates; no new model training"), indent=2))
    calendar = pd.DatetimeIndex(pd.read_parquet(OUTPUT / "calendar.parquet").date)
    grids, locks, picks, summary, curves = [], [], [], [], []
    replays = 0
    for fold, boundaries in protocol["folds"].items():
        for arm in ["core145", "full_registry", "full_registry_sequence"]:
            frames = {window: load_group(protocol, fold, arm, window) for window in [126, 252]}
            days = calendar[(calendar >= boundaries[1]) & (calendar < boundaries[2])]
            pieces = [search(frame.loc[frame.stage.eq("selection")].reset_index(drop=True), days,
                dict(fold=fold, arm=arm, window=window)) for window, frame in frames.items()]
            grid = pd.concat(pieces, ignore_index=True)
            grids.append(grid)
            eligible = grid.loc[grid.dates.ge(20) & grid.unknown.eq(0)]
            winner = None if eligible.empty else eligible.sort_values(["score", "net_return_scenario", "window", "pool", "entry", "top_k", "penalty", "min_mu"],
                ascending=[False, False, True, True, True, True, True, True], kind="stable").iloc[0]
            keys = ["fold", "arm", "window", "pool", "entry", "top_k", "penalty", "min_mu"]
            rule = None if winner is None else {k: winner[k].item() if isinstance(winner[k], np.generic) else winner[k] for k in keys}
            qualified = bool(winner is not None and winner.score > 0 and winner.first_half_pnl > 0 and winner.second_half_pnl > 0)
            locks.append(dict(fold=fold, arm=arm, rule=rule, qualified=qualified))
            for stage, start, end in [("selection", boundaries[1], boundaries[2]), ("diagnostic", boundaries[2], boundaries[3])]:
                frame = frames[126 if rule is None else rule["window"]]
                stage_frame = frame.loc[frame.stage.eq(stage)].reset_index(drop=True)
                action = with_action(stage_frame, 0. if rule is None else rule["penalty"])
                indices = np.array([], dtype=int) if rule is None else payoff_selection(action, rule)
                # Adversarial replay: fictitious future returns must not change actions or picks.
                perturbed = stage_frame.copy()
                perturbed[[f"gross_{t}" for t in TARGETS]] = 999.
                changed = with_action(perturbed, 0. if rule is None else rule["penalty"])
                assert action.selected_exit.equals(changed.selected_exit)
                if rule is not None:
                    assert np.array_equal(indices, payoff_selection(changed, rule))
                replays += 1
                chosen = action.iloc[indices].copy()
                k = 1 if rule is None else rule["top_k"]
                chosen = chosen.assign(fold=fold, arm=arm, top_k=k, qualified=qualified)
                picks.append(chosen)
                days = calendar[(calendar >= start) & (calendar < end)]
                for cost, stress in itertools.product([20, 40, 60], [-.1, -.3]):
                    metrics, daily = portfolio_metrics(chosen.gross.to_numpy(), days.get_indexer(chosen.planned_exit_date), len(days), k, cost, stress)
                    book, nav = cashbook(chosen, days, k, cost, stress)
                    summary.append(dict(fold=fold, arm=arm, stage=stage, qualified=qualified, cost_bps=cost, unknown_scenario=stress, **metrics, **book))
                    curves.append(pd.DataFrame(dict(date=days, pnl=daily, cashbook_nav=nav, fold=fold, arm=arm, stage=stage, cost_bps=cost, unknown_scenario=stress)))
            print(f"Adaptive locked {fold}/{arm}: qualified={qualified} rule={rule}", flush=True)
    trades = pd.concat(picks, ignore_index=True)
    trades.to_parquet(output / "picks.parquet", index=False)
    trades.loc[trades.gross.isna() | trades.gross.le(.004)].to_csv(output / "badcases.csv", index=False)
    pd.concat(grids).to_csv(output / "selection_grid.csv", index=False)
    (output / "locked_rules.json").write_text(json.dumps(locks, indent=2))
    pd.DataFrame(summary).to_csv(output / "summary.csv", index=False)
    pd.concat(curves).to_csv(output / "daily_curves.csv", index=False)
    trades.groupby(["fold", "arm", "stage", "selected_exit"]).size().to_csv(output / "action_distribution.csv")
    for path, digest in sources.items():
        if sha256(Path(path)) != digest:
            raise ValueError("Frozen adaptive source changed")
    (output / "complete.json").write_text(json.dumps(dict(grid_rows=sum(len(g) for g in grids), frozen_rules=len(locks),
        adversarial_stage_replays=replays, picks_sha256=sha256(output / "picks.parquet"),
        status="Adaptive buy-time exits evaluated, no independent confirmation"), indent=2))


if __name__ == "__main__":
    main()
