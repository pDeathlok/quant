"""Train exit-specific positive/negative payoff heads on matched historical inputs."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from audit_overnight_right_provenance import sha256
from run_right_t012_profit_grid import (
    EXITS,
    cashbook,
    entry_masks,
    pick_indices,
    portfolio_metrics,
)
from run_right_t012_profit_grid import (
    exit_returns as unchecked_exit_returns,
)
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from train_right_t012 import SIGNALS
from train_right_t012_full_registry import FOLDS, sample_weights, split_masks

from quant.core.paths import PROJECT_ROOT
from quant.features.variable_library import PROJECT_FACTOR_COLUMNS
from quant.research.short_side_groups import RIGHT_GROUP_MEMBERS

ROOT = PROJECT_ROOT / "reports/research"
OUTPUT = ROOT / "right_t012_payoff_v2"
TARGETS = ["open", "close", "tp2", "tp5"]
PARAMS = dict(max_iter=120, max_leaf_nodes=15, min_samples_leaf=200, l2_regularization=30,
              learning_rate=.06, early_stopping=False, random_state=20260913)
META = ["symbol", "date", "screen_date", "planned_exit_date", "entry_close", "next_open", "next_high", "next_low", "next_close",
        "next_valid_bar", "label_status", "next_pre_close", "down_limit", "price_limit_inconsistent", "screen_to_entry_return", "entry_right", "episode_age", *SIGNALS]


def exit_returns(frame: pd.DataFrame) -> np.ndarray:
    """Preserve source label eligibility; raw prices alone are insufficient."""
    if not {"label_status", "next_pre_close"}.issubset(frame.columns):
        raise ValueError("Missing exit label admission columns")
    result = unchecked_exit_returns(frame)
    known = frame.label_status.eq("ok").fillna(False).to_numpy(bool)
    known &= np.isfinite(frame.next_pre_close).to_numpy() & frame.next_pre_close.gt(0).to_numpy()
    known &= frame.next_pre_close.sub(frame.entry_close).abs().le(.011).to_numpy()
    result[~known] = np.nan
    return result


def payoff_targets(gross: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    net = gross - .004
    return np.maximum(net, 0), np.maximum(-net, 0)


def recent_mask(frame: pd.DataFrame, calendar: pd.DatetimeIndex, cutoff: str, window: int) -> np.ndarray:
    dates = calendar[calendar < cutoff][-window:]
    return (frame.date.ge(dates.min()) & frame.date.lt(cutoff) & frame.planned_exit_date.lt(cutoff)).to_numpy()


def calibrate_scale(actual: np.ndarray, prediction: np.ndarray, weights: np.ndarray, n_dates: int) -> float:
    ratio = np.average(actual, weights=weights) / max(np.average(prediction, weights=weights), 1e-8)
    return float(np.clip((n_dates * ratio + 20) / (n_dates + 20), .5, 2.))


def pools(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    result = {"all_right": np.ones(len(frame), dtype=bool)}
    for name, members in RIGHT_GROUP_MEMBERS.items():
        if all(member in frame for member in members):
            result[name] = frame[list(members)].eq(True).any(axis=1).to_numpy()
    return result


def payoff_selection(frame: pd.DataFrame, rule: dict) -> np.ndarray:
    mask = pools(frame)[rule["pool"]] & entry_masks(frame)[rule["entry"]]
    if rule.get("arm") == "no_model":
        score = frame.screen_to_entry_return
        mask &= np.isfinite(score).to_numpy()
    else:
        mu = frame.gain - frame.loss
        score = mu - rule["penalty"] * frame.loss
        mask &= mu.ge(rule["min_mu"]).to_numpy() & np.isfinite(score).to_numpy()
    order = frame.assign(score=score, row=np.arange(len(frame))).sort_values(
        ["date", "score", "symbol"], ascending=[True, False, True]).row.to_numpy()
    return pick_indices(order, mask, pd.factorize(frame.date, sort=True)[0], int(rule["top_k"]))


def verify_sources(protocol: dict) -> None:
    for path, digest in protocol["sources"].items():
        if sha256(Path(path)) != digest:
            raise ValueError(f"Changed frozen source: {path}")


def prepare() -> None:
    dataset = ROOT / "right_t012_full_dataset_v4/dataset.parquet"
    digest = sha256(dataset)
    audit = json.loads((ROOT / "right_t012_full_dataset_audit_v1/complete.json").read_text())
    if digest != audit["dataset_sha256"]:
        raise ValueError("Dataset admission mismatch")
    registry_path = dataset.with_name("protocol.json")
    candidates = json.loads(registry_path.read_text())["registry_candidates"]
    signal_path = ROOT / "right_t012_history_v1/all_daily_signals.parquet"
    calendar_path = ROOT / "overnight_unfiltered_labels_v1/labels.parquet"
    source_paths = [dataset, registry_path, signal_path, calendar_path, Path(__file__),
        ROOT / "right_t012_payoff_v2_plan.md", PROJECT_ROOT / "scripts/research/run_right_t012_profit_grid.py",
        PROJECT_ROOT / "scripts/research/train_right_t012_full_registry.py", PROJECT_ROOT / "scripts/research/train_right_t012.py",
        PROJECT_ROOT / "src/quant/features/variable_library.py", PROJECT_ROOT / "src/quant/research/short_side_groups.py"]
    sequence_roots = {"2025H2": ROOT / "right_t012_full_sequence_fold_a_v1", "2026": ROOT / "right_t012_full_sequence_v1"}
    for directory in sequence_roots.values():
        source_paths += [directory / "complete.json", directory / "sequence_variables.parquet"]
    protocol = dict(sources={str(p): sha256(p) for p in source_paths}, params=PARAMS, folds=FOLDS,
        windows=[126, 252], targets=TARGETS, labels="Unclipped gross minus40bps, gain=max(net,0), loss=max(-net,0)",
        calibration="Per-head scale shrunk20dates to1, clipped[0.5,2], fit only separate calibration segment",
        selection="Per arm best among>=20dates, unknown0; qualify if score>0 and both halves>0; no win-rate gate",
        limits="Reused historical diagnostics; uncertain daily fills and financial vintages; identical sequence-ready populations")
    OUTPUT.mkdir(exist_ok=False)
    (OUTPUT / "protocol.json").write_text(json.dumps(protocol, indent=2))
    base = pd.read_parquet(dataset)
    signals = pd.read_parquet(signal_path, columns=["symbol", "date", *SIGNALS])
    signals["entry_right"] = signals[list(SIGNALS)].eq(True).any(axis=1)
    base = base.merge(signals[["symbol", "date", "entry_right"]], on=["symbol", "date"], how="left", validate="one_to_one")
    calendar = pd.DatetimeIndex(pd.read_parquet(calendar_path, columns=["date"]).date.unique()).sort_values()
    calendar.to_series(index=range(len(calendar)), name="date").to_frame().to_parquet(OUTPUT / "calendar.parquet", index=False)
    groups, configs, prepared = [], [], {}
    label_admission = []
    extras = [*SIGNALS, "episode_age", "screen_to_entry_return"]
    for fold, boundaries in FOLDS.items():
        seq_root = sequence_roots[fold]
        check = json.loads((seq_root / "complete.json").read_text())
        if check["dataset_sha256"] != digest or sha256(seq_root / "sequence_variables.parquet") != check["variables_sha256"]:
            raise ValueError("Sequence output admission mismatch")
        seq_names = check["features"]
        seq = pd.read_parquet(seq_root / "sequence_variables.parquet", columns=["symbol", "date", *seq_names])
        frame = base.merge(seq, on=["symbol", "date"], how="left", validate="one_to_one")
        frame = frame.loc[frame[seq_names].notna().all(axis=1)].reset_index(drop=True)
        masks = split_masks(frame, boundaries)
        frame["stage"] = "purged"
        for stage, mask in masks.items():
            frame.loc[mask, "stage"] = stage
        for window in [126, 252]:
            frame[f"fit_{window}"] = recent_mask(frame, calendar, boundaries[0], window)
        gross = exit_returns(frame)
        if np.isfinite(gross[frame.label_status.ne("ok").to_numpy()]).any():
            raise ValueError("Unknown source labels were resurrected")
        for price, j in [("open", 0), ("close", 1)]:
            known = np.isfinite(gross[:, j])
            original = frame[f"net_{price}"].to_numpy() + .002
            if not np.allclose(gross[known, j], original[known], rtol=1e-10, atol=1e-12):
                raise ValueError("Direct exits do not replay source labels")
        raw = unchecked_exit_returns(frame)
        for j, target in enumerate(EXITS):
            label_admission.append(dict(fold=fold, exit=target,
                previously_resurrected=int((np.isfinite(raw[:, j]) & ~np.isfinite(gross[:, j])).sum()),
                known=int(np.isfinite(gross[:, j]).sum()), unknown=int(np.isnan(gross[:, j]).sum())))
        for target in TARGETS:
            frame[f"gross_{target}"] = gross[:, EXITS.index(target)]
        path = OUTPUT / f"{fold}_prepared.parquet"
        frame.to_parquet(path, index=False)
        prepared[str(path)] = sha256(path)
        seen = {}
        for arm, columns in [("core145", list(PROJECT_FACTOR_COLUMNS)), ("full_registry", candidates),
                             ("full_registry_sequence", candidates + seq_names)]:
            for window in [126, 252]:
                fit = frame[f"fit_{window}"].to_numpy()
                usable = [c for c in dict.fromkeys([*columns, *extras]) if frame.loc[fit, c].nunique(dropna=True) > 1]
                key_hash = hashlib.sha256(np.flatnonzero(fit).tobytes()).hexdigest()
                key = (arm, key_hash, tuple(usable))
                group_id = f"{fold}_{arm}_{window}"
                if key not in seen:
                    seen[key] = group_id
                    groups.append(dict(id=group_id, fold=fold, arm=arm, window=window, path=str(path),
                                       columns=usable, fit_keys_sha256=key_hash, fit_rows=int(fit.sum())))
                for target in TARGETS:
                    configs.append(dict(fold=fold, arm=arm, window=window, exit=target, model_id=f"{seen[key]}_{target}",
                                        reused=seen[key] != group_id))
    pd.DataFrame(label_admission).to_csv(OUTPUT / "exit_label_admission.csv", index=False)
    protocol.update(groups=groups, configs=configs, prepared=prepared, calendar_sha256=sha256(OUTPUT / "calendar.parquet"))
    verify_sources(protocol)
    (OUTPUT / "protocol.json").write_text(json.dumps(protocol, indent=2))
    (OUTPUT / "prepared_complete.json").write_text(json.dumps(dict(groups=len(groups), nominal_dual_models=len(configs),
        actual_dual_models=4*len(groups), regressors=8*len(groups)), indent=2))
    print(f"Prepared {len(configs)} nominal dual models; {4*len(groups)} unique dual models", flush=True)


def train_group(group: dict) -> list[dict]:
    path = Path(group["path"])
    columns = list(dict.fromkeys([*group["columns"], *META, "stage", f"fit_{group['window']}", *[f"gross_{t}" for t in TARGETS]]))
    frame = pd.read_parquet(path, columns=columns)
    x = frame[group["columns"]].astype("float32").replace([np.inf, -np.inf], np.nan)
    fit_base = frame[f"fit_{group['window']}"]
    evaluation = frame.stage.isin(["selection", "diagnostic"])
    records = []
    for target in TARGETS:
        name = f"{group['id']}_{target}"
        complete_path = OUTPUT / f"{name}_complete.json"
        if complete_path.exists():
            check = json.loads(complete_path.read_text())
            if sha256(OUTPUT / f"{name}.joblib") != check["model_sha256"] or sha256(OUTPUT / f"{name}_predictions.parquet") != check["predictions_sha256"]:
                raise ValueError("Checkpoint changed")
            records.append(check)
            continue
        gain, loss = payoff_targets(frame[f"gross_{target}"].to_numpy())
        known = np.isfinite(gain)
        fit = fit_base & known
        cal = frame.stage.eq("calibration") & known
        weights = sample_weights(frame.loc[fit])
        cal_weights = sample_weights(frame.loc[cal])
        n_dates = frame.loc[cal, "date"].nunique()
        models, scales, predicted = {}, {}, {}
        print(f"Training {name}: fit={fit.sum()} cal={cal.sum()} features={len(group['columns'])}", flush=True)
        with threadpool_limits(limits=2):
            for head, labels in [("gain", gain), ("loss", loss)]:
                model = HistGradientBoostingRegressor(**PARAMS).fit(x.loc[fit], labels[fit], sample_weight=weights)
                cal_pred = np.maximum(model.predict(x.loc[cal]), 0)
                scale = calibrate_scale(labels[cal], cal_pred, cal_weights, n_dates)
                predicted[head] = np.maximum(model.predict(x.loc[evaluation]), 0) * scale
                models[head], scales[head] = model, scale
        prediction = frame.loc[evaluation, ["symbol", "date", "stage"]].copy()
        prediction["gain"], prediction["loss"] = predicted["gain"], predicted["loss"]
        prediction.to_parquet(OUTPUT / f"{name}_predictions.parquet", index=False)
        joblib.dump(dict(models=models, scales=scales, columns=group["columns"], target=target), OUTPUT / f"{name}.joblib")
        record = dict(id=name, fit=int(fit.sum()), fit_dates=int(frame.loc[fit, "date"].nunique()), cal=int(cal.sum()),
            max_fit_exit=str(frame.loc[fit, "planned_exit_date"].max()), max_cal_exit=str(frame.loc[cal, "planned_exit_date"].max()),
            prediction_min_date=str(prediction.date.min()), scales=scales,
            model_sha256=sha256(OUTPUT / f"{name}.joblib"), predictions_sha256=sha256(OUTPUT / f"{name}_predictions.parquet"))
        complete_path.write_text(json.dumps(record, indent=2))
        records.append(record)
        print(f"Completed {name}", flush=True)
    return records


def load_protocol() -> dict:
    if not (OUTPUT / "prepared_complete.json").exists():
        raise ValueError("Prepare must complete first")
    protocol = json.loads((OUTPUT / "protocol.json").read_text())
    verify_sources(protocol)
    for path, digest in protocol["prepared"].items():
        if sha256(Path(path)) != digest:
            raise ValueError("Prepared data changed")
    if sha256(OUTPUT / "calendar.parquet") != protocol["calendar_sha256"]:
        raise ValueError("Prepared calendar changed")
    return protocol


def train() -> None:
    protocol = load_protocol()
    records = []
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(train_group, group) for group in protocol["groups"]]
        for future in as_completed(futures):
            records.extend(future.result())
    verify_sources(protocol)
    (OUTPUT / "training_complete.json").write_text(json.dumps(dict(dual_models=len(records), regressors=2*len(records), models=records), indent=2))
    print(f"Training complete: {len(records)} dual models", flush=True)


def search(frame: pd.DataFrame, days: pd.DatetimeIndex, config: dict) -> pd.DataFrame:
    exit_day = days.get_indexer(frame.planned_exit_date)
    if (exit_day < 0).any():
        raise ValueError("Selection exit outside calendar")
    pool_choices, entries = pools(frame), entry_masks(frame)
    no_model = config["arm"] == "no_model"
    dates = pd.factorize(frame.date, sort=True)[0]
    penalties = [0.] if no_model else [0., .5, 1.]
    thresholds = [0.] if no_model else [0., .005]
    records = []
    for penalty in penalties:
        mu = frame.screen_to_entry_return if no_model else frame.gain-frame.loss
        score = mu if no_model else mu-penalty*frame.loss
        order = frame.assign(score=score, row=np.arange(len(frame))).sort_values(["date", "score", "symbol"], ascending=[True, False, True]).row.to_numpy()
        for pool, entry, min_mu, top_k in itertools.product(pool_choices, entries, thresholds, [1, 3, 5]):
            mask = pool_choices[pool] & entries[entry] & np.isfinite(score).to_numpy()
            if not no_model:
                mask &= mu.ge(min_mu).to_numpy()
            chosen = pick_indices(order, mask, dates, top_k)
            metrics, _ = portfolio_metrics(frame.gross.to_numpy()[chosen], exit_day[chosen], len(days), top_k, 40, -.1)
            records.append(dict(config, pool=pool, entry=entry, min_mu=min_mu, top_k=top_k, penalty=penalty, **metrics))
    return pd.DataFrame(records)


def evaluate() -> None:
    if (OUTPUT / "complete.json").exists():
        raise FileExistsError("Completed evaluation already exists")
    protocol = load_protocol()
    trained = json.loads((OUTPUT / "training_complete.json").read_text())
    for item in trained["models"]:
        if sha256(OUTPUT / f"{item['id']}_predictions.parquet") != item["predictions_sha256"]:
            raise ValueError("Changed predictions")
    calendar = pd.DatetimeIndex(pd.read_parquet(OUTPUT / "calendar.parquet").date)
    grids, locked, picks, summaries, curves = [], [], [], [], []
    for fold, boundaries in FOLDS.items():
        base = pd.read_parquet(OUTPUT / f"{fold}_prepared.parquet", columns=[*META, "stage", *[f"gross_{t}" for t in TARGETS]])
        days = calendar[(calendar >= boundaries[1]) & (calendar < boundaries[2])]
        selection_base = base.loc[base.stage.eq("selection")]
        for arm in ["core145", "full_registry", "full_registry_sequence", "no_model"]:
            configs = [c for c in protocol["configs"] if c["fold"] == fold and c["arm"] == arm]
            if arm == "no_model":
                configs = [dict(fold=fold, arm=arm, window=0, exit=t, model_id="no_model", reused=False) for t in TARGETS]
            arm_grids = []
            for config in configs:
                frame = selection_base.copy()
                if arm != "no_model":
                    pred = pd.read_parquet(OUTPUT / f"{config['model_id']}_predictions.parquet")
                    frame = frame.merge(pred.drop(columns="stage"), on=["symbol", "date"], validate="one_to_one")
                    if len(frame) != len(selection_base):
                        raise ValueError("Unmatched selection population")
                frame["gross"] = frame[f"gross_{config['exit']}"]
                arm_grids.append(search(frame.reset_index(drop=True), days, config))
            grid = pd.concat(arm_grids, ignore_index=True)
            grids.append(grid)
            acceptable = grid.loc[grid.dates.ge(20) & grid.unknown.eq(0)]
            winner = None if acceptable.empty else acceptable.sort_values(["score", "net_return_scenario", "window", "exit", "pool", "entry", "top_k", "penalty", "min_mu"],
                ascending=[False, False, True, True, True, True, True, True, True], kind="stable").iloc[0]
            keys = ["fold", "arm", "window", "exit", "model_id", "pool", "entry", "top_k", "penalty", "min_mu"]
            rule = None if winner is None else {k: winner[k].item() if isinstance(winner[k], np.generic) else winner[k] for k in keys}
            qualified = bool(winner is not None and winner.score > 0 and winner.first_half_pnl > 0 and winner.second_half_pnl > 0)
            locked.append(dict(fold=fold, arm=arm, rule=rule, qualified=qualified, score=None if winner is None else float(winner.score)))
            for stage, start, end in [("selection", boundaries[1], boundaries[2]), ("diagnostic", boundaries[2], boundaries[3])]:
                frame = base.loc[base.stage.eq(stage)].copy()
                if rule is not None and arm != "no_model":
                    pred = pd.read_parquet(OUTPUT / f"{rule['model_id']}_predictions.parquet")
                    frame = frame.merge(pred.drop(columns="stage"), on=["symbol", "date"], validate="one_to_one")
                frame = frame.reset_index(drop=True)
                ids = np.array([], dtype=int) if rule is None else payoff_selection(frame, rule)
                chosen = frame.iloc[ids].copy()
                chosen["gross"] = np.array([]) if rule is None else chosen[f"gross_{rule['exit']}"]
                k = 1 if rule is None else rule["top_k"]
                chosen = chosen.assign(fold=fold, arm=arm, qualified=qualified, top_k=k, exit="abstain" if rule is None else rule["exit"])
                picks.append(chosen)
                stage_days = calendar[(calendar >= start) & (calendar < end)]
                for cost, stress in itertools.product([20, 40, 60], [-.1, -.3]):
                    stats, daily = portfolio_metrics(chosen.gross.to_numpy(), stage_days.get_indexer(chosen.planned_exit_date), len(stage_days), k, cost, stress)
                    book, nav = cashbook(chosen, stage_days, k, cost, stress)
                    summaries.append(dict(fold=fold, arm=arm, stage=stage, qualified=qualified, cost_bps=cost, unknown_scenario=stress, **stats, **book))
                    curves.append(pd.DataFrame(dict(date=stage_days, pnl=daily, cashbook_nav=nav, fold=fold, arm=arm,
                        stage=stage, cost_bps=cost, unknown_scenario=stress)))
            pd.concat(grids).to_csv(OUTPUT / "selection_grid.csv", index=False)
            (OUTPUT / "locked_rules.json").write_text(json.dumps(locked, indent=2))
            pd.DataFrame(summaries).to_csv(OUTPUT / "summary.csv", index=False)
            print(f"Locked {fold}/{arm}: qualified={qualified} rule={rule}", flush=True)
    trades = pd.concat(picks, ignore_index=True)
    trades.to_parquet(OUTPUT / "picks.parquet", index=False)
    trades.loc[trades.gross.isna() | trades.gross.le(.004)].to_csv(OUTPUT / "badcases.csv", index=False)
    pd.concat(curves).to_csv(OUTPUT / "daily_curves.csv", index=False)
    verify_sources(protocol)
    (OUTPUT / "complete.json").write_text(json.dumps(dict(grid_rows=sum(len(g) for g in grids), frozen_rules=len(locked),
        trained_dual_models=trained["dual_models"], picks_sha256=sha256(OUTPUT / "picks.parquet"),
        status="Payoff retraining and locked policy evaluation complete; not independent confirmation"), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--prepare", action="store_true")
    actions.add_argument("--train", action="store_true")
    actions.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.train:
        train()
    else:
        evaluate()


if __name__ == "__main__":
    main()
