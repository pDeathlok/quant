"""Retrain equal-sample core/full registry models with separated time stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from audit_overnight_right_provenance import sha256
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from threadpoolctl import threadpool_limits
from train_right_t012 import SIGNALS, choose, summarize

from quant.core.paths import PROJECT_ROOT
from quant.features.variable_library import PROJECT_FACTOR_COLUMNS

FOLDS = {
    "2025H2": ("2025-01-01", "2025-04-01", "2025-07-01", "2026-01-01"),
    "2026": ("2025-07-01", "2025-10-01", "2026-01-01", "2026-09-04"),
}


def split_masks(frame: pd.DataFrame, boundaries: tuple[str, str, str, str]) -> dict[str, pd.Series]:
    start = "2024-01-01"
    out = {}
    for stage, end in zip(["fit", "calibration", "selection", "diagnostic"], boundaries, strict=True):
        out[stage] = frame.date.ge(start) & frame.date.lt(end) & frame.planned_exit_date.lt(end)
        start = end
    if pd.concat(out, axis=1).sum(axis=1).gt(1).any():
        raise ValueError("Overlapping chronological stages")
    return out


def sample_weights(frame: pd.DataFrame) -> np.ndarray:
    w = 1 / np.sqrt(frame.episode_age.astype(float))
    w = w / w.groupby(frame.date).transform("sum")
    return (w / w.mean()).to_numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-fusion", action="store_true")
    args = parser.parse_args()
    root = PROJECT_ROOT / "reports/research"
    source = root / "right_t012_full_dataset_v4"
    output = root / "right_t012_full_registry_models_v1"
    if args.sequence_fusion:
        output = root / "right_t012_full_registry_fusion_v1"
    check = json.loads((source / "complete.json").read_text())
    dataset_path = source / "dataset.parquet"
    digest = sha256(dataset_path)
    if digest != check["dataset_sha256"]:
        raise ValueError("Dataset differs from completed build")
    audit = json.loads((root / "right_t012_full_dataset_audit_v1/complete.json").read_text())
    if audit["dataset_sha256"] != digest or (source / "DO_NOT_TRAIN.md").exists():
        raise ValueError("Dataset has not passed admission")
    dataset_protocol = json.loads((source / "protocol.json").read_text())
    candidates = dataset_protocol["registry_candidates"]
    extras = [*SIGNALS, "episode_age", "screen_to_entry_return"]
    params = dict(max_iter=120, max_leaf_nodes=15, min_samples_leaf=200, l2_regularization=30,
                  learning_rate=.06, early_stopping=False, random_state=20260913)
    output.mkdir(exist_ok=False)
    protocol = dict(dataset_sha256=digest, folds=FOLDS, candidate_columns=candidates, extras=extras, params=params,
                    arms=["core145", "full_registry"], targets=["net_close", "limit1"],
                    weighting="Within-date normalize1/sqrt(causal episode age); equal total date weight",
                    temporal="Separate fit/calibration/selection/diagnostic stages, purge each stage by exit date",
                    grid={"probability": [0., .5, .6, .7, .8, .9], "risk": [1., .2, .1], "rank": ["probability", "expected_net"]},
                    high_return="At least20selection dates; maximize40bps daily score; even negative best is diagnostic only",
                    high_precision="At least20selection dates, eventprecision>=80%, unknown0, positive40bps daily_score; then maximize dates",
                    status="Reused historical diagnostics; historical-variable revisions not proven PIT; no production promotion",
                    script_sha256=sha256(Path(__file__)))
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    frame = pd.read_parquet(dataset_path)
    arms = [("core145", list(PROJECT_FACTOR_COLUMNS)), ("full_registry", candidates)]
    base_frame = frame
    sequence_sources = {}
    if args.sequence_fusion:
        for fold, directory in [("2025H2", "right_t012_full_sequence_fold_a_v1"), ("2026", "right_t012_full_sequence_v1")]:
            seq_root = root / directory
            seq_check = json.loads((seq_root / "complete.json").read_text())
            seq_path = seq_root / "sequence_variables.parquet"
            if seq_check["dataset_sha256"] != digest or sha256(seq_path) != seq_check["variables_sha256"]:
                raise ValueError("Sequence outputs do not match the redesigned dataset")
            sequence_sources[fold] = dict(path=str(seq_path), sha256=seq_check["variables_sha256"], features=seq_check["features"])
        arms = [("full_registry_matched", candidates), ("full_registry_sequence", candidates + seq_check["features"])]
        protocol.update(arms=[a for a, _ in arms], sequence_sources=sequence_sources,
                        matching="Both arms only use forward sequence-ready samples in every stage; earliest fit starts2024H2",
                        sequence_timing="2025H2 outer fold uses learner frozen before2025 for all2025;2026 outer fold uses learner frozen before2025H2 for all later dates")
        (output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    # Keep the complete market calendar, including dates on which the strategy abstains.
    calendar = pd.DatetimeIndex(pd.read_parquet(root / "overnight_unfiltered_labels_v1/labels.parquet", columns=["date"]).date.unique()).sort_values()
    all_summary, all_picks, all_grid, column_records = [], [], [], []
    for fold, boundaries in FOLDS.items():
        frame = base_frame
        sequence_ready = pd.Series(True, index=frame.index)
        if args.sequence_fusion:
            seq_source = sequence_sources[fold]
            seq = pd.read_parquet(seq_source["path"], columns=["symbol", "date", *seq_source["features"]])
            frame = base_frame.merge(seq, on=["symbol", "date"], how="left", validate="one_to_one")
            sequence_ready = frame[seq_source["features"]].notna().all(axis=1)
        masks = split_masks(frame, boundaries)
        masks = {name: mask & sequence_ready for name, mask in masks.items()}
        for arm, universe in arms:
            usable = [c for c in [*universe, *extras] if frame.loc[masks["fit"], c].nunique(dropna=True) > 1]
            for name in [*universe, *extras]:
                column_records.append(dict(fold=fold, arm=arm, name=name, used=name in usable,
                                           reason="used" if name in usable else "all_missing_or_constant_in_fit_only"))
            x = frame[usable].astype("float32").replace([np.inf, -np.inf], np.nan)
            for target in ["net_close", "limit1"]:
                net = frame[f"{target}_net_after_exit_audit"]
                event = net.gt(0).astype("boolean").mask(net.isna()) if target == "net_close" else frame.limit1_penetrated.astype("boolean").mask(net.isna())
                fm = masks["fit"] & net.notna()
                cm = masks["calibration"] & net.notna()
                risk_y = net.lt(-.02)
                print(f"{fold}/{arm}/{target}: fit={fm.sum()} cal={cm.sum()} inputs={len(usable)}", flush=True)
                with threadpool_limits(limits=4):
                    clf = HistGradientBoostingClassifier(**params).fit(x[fm], event[fm].astype(int), sample_weight=sample_weights(frame[fm]))
                    reg = HistGradientBoostingRegressor(**params).fit(x[fm], net[fm], sample_weight=sample_weights(frame[fm]))
                    risk_model = HistGradientBoostingClassifier(**params).fit(x[fm], risk_y[fm], sample_weight=sample_weights(frame[fm]))
                    iso = IsotonicRegression(out_of_bounds="clip").fit(clf.predict_proba(x[cm])[:, 1], event[cm].astype(int), sample_weight=sample_weights(frame[cm]))
                    risk_iso = IsotonicRegression(out_of_bounds="clip").fit(risk_model.predict_proba(x[cm])[:, 1], risk_y[cm], sample_weight=sample_weights(frame[cm]))
                    evaluation = masks["selection"] | masks["diagnostic"]
                    p = iso.predict(clf.predict_proba(x[evaluation])[:, 1])
                    expected = reg.predict(x[evaluation])
                    risk = risk_iso.predict(risk_model.predict_proba(x[evaluation])[:, 1])
                bundle = dict(classifier=clf, regressor=reg, risk=risk_model, calibrator=iso, risk_calibrator=risk_iso, features=usable)
                joblib.dump(bundle, output / f"{fold}_{arm}_{target}.joblib")
                columns = ["symbol", "date", "screen_date", "planned_exit_date", "entry_close", "next_open", "next_high", "next_low", "next_close", "episode_age", "screen_to_entry_return", *SIGNALS]
                pred = frame.loc[evaluation, columns].copy()
                pred = pred.assign(fold=fold, arm=arm, target=target, probability=p, expected_net=expected, risk=risk,
                                   net=net[evaluation], event_hit=event[evaluation], stage=np.where(masks["selection"][evaluation], "selection", "diagnostic"))
                pred.to_parquet(output / f"{fold}_{arm}_{target}_predictions.parquet", index=False)
                select_days = calendar[(calendar >= boundaries[1]) & (calendar < boundaries[2])]
                # Exclude the final market session whose exit would cross the stage boundary.
                select_days = select_days[:-1]
                selection = pred.loc[pred.stage.eq("selection")]
                grid = []
                for threshold in protocol["grid"]["probability"]:
                    for max_risk in protocol["grid"]["risk"]:
                        for rank in protocol["grid"]["rank"]:
                            picked = choose(selection.loc[selection.risk.le(max_risk)], threshold, rank)
                            grid.append(dict(fold=fold, arm=arm, target=target, threshold=threshold, maximum_risk=max_risk, rank=rank,
                                             **summarize(picked.assign(net=picked.net - .002), select_days)))
                grid_frame = pd.DataFrame(grid)
                all_grid.append(grid_frame)
                acceptable = grid_frame.loc[grid_frame.dates.ge(20)]
                precision = acceptable.loc[acceptable.event_hit_rate.ge(.8) & acceptable.unknown.eq(0) & acceptable.daily_score.gt(0)]
                for policy, eligible in [("high_return", acceptable), ("high_precision", precision)]:
                    sort = ["daily_score", "threshold", "maximum_risk", "rank"] if policy == "high_return" else ["dates", "daily_score", "threshold", "maximum_risk", "rank"]
                    ascending = [False, True, True, True] if policy == "high_return" else [False, False, True, True, True]
                    winner = None if eligible.empty else eligible.sort_values(sort, ascending=ascending).iloc[0]
                    for stage, start, end in [("selection", boundaries[1], boundaries[2]), ("diagnostic", boundaries[2], boundaries[3])]:
                        subset = pred.loc[pred.stage.eq(stage)]
                        picked = subset.iloc[:0].copy() if winner is None else choose(subset.loc[subset.risk.le(float(winner.maximum_risk))], float(winner.threshold), str(winner["rank"]))
                        picked["policy"] = policy
                        all_picks.append(picked)
                        days = calendar[(calendar >= start) & (calendar < end)]
                        # dataset labels are only available before the last local market day.
                        if len(days):
                            days = days[:-1]
                        for cost in [20, 40]:
                            all_summary.append(dict(fold=fold, arm=arm, target=target, policy=policy, stage=stage, cost_bps=cost,
                                selected_rule=winner is not None, selection_score=None if winner is None else float(winner.daily_score),
                                **summarize(picked.assign(net=picked.net - (cost - 20) / 10000), days)))
                pd.DataFrame(all_summary).to_csv(output / "summary.csv", index=False)
                pd.DataFrame(column_records).to_csv(output / "model_column_admission.csv", index=False)
                print(f"Completed {fold}/{arm}/{target}", flush=True)
    picks = pd.concat(all_picks, ignore_index=True)
    picks.to_parquet(output / "picks.parquet", index=False)
    picks.loc[picks.net.isna() | picks.net.le(0) | ~picks.event_hit.fillna(False)].to_parquet(output / "badcases.parquet", index=False)
    pd.concat(all_grid).to_csv(output / "selection_grid.csv", index=False)
    if sha256(dataset_path) != digest:
        raise ValueError("Dataset changed during training")
    for seq_source in sequence_sources.values():
        if sha256(Path(seq_source["path"])) != seq_source["sha256"]:
            raise ValueError("Sequence source changed during training")
    (output / "complete.json").write_text(json.dumps(dict(rows=len(frame), candidate_features=len(candidates),
        models=8, picks_sha256=sha256(output / "picks.parquet"), status="Matched full-registry fusion retraining finished" if args.sequence_fusion else "Core/full retraining finished, sequence retraining not part of this script"), indent=2))


if __name__ == "__main__":
    main()
