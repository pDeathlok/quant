#!/usr/bin/env python
"""Frozen left-model index-MA ablations and market admission experiments."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from quant.features.active_market_value import KEY_INDEX_SPECS
from quant.features.index_ma_state import (
    ALL_MA_MODEL_COLUMNS, ANNUAL_MODEL_COLUMNS, CORE_INDEX_SLUGS,
    INDEX_MA_STATE_FEATURE_COLUMNS, INDEX_MA_STATE_SCHEMA_VERSION,
    PUBLIC_START_DATES, compute_index_ma_state_features,
)
from quant.features.left_side_factor_contract import LEFT_SIDE_FACTOR_COLUMNS
from quant.research.right_side_long_task import LONG_TASK_DEEP_XGB_CLASSIFIER_SPEC
from quant.research.right_side_unified import DEFAULT_YEAR_FOLDS, split_by_year_fold
from scripts.research.validate_unified_left_side_models import _fit_unified_model, _split_validation_stages
from scripts.research.validate_left_side_disclosure_gain import load_frame, sha256, save_json, block_interval

DATA = ROOT / "data/research/left_side_index_ma_20261101"
REPORT = ROOT / "reports/research/left_side_index_ma_20261101"
BASE_DATA = ROOT / "data/research/left_side_disclosure_gain_20261029"
BASE_REPORT = ROOT / "reports/research/left_side_disclosure_gain_20261029"
SOURCE = ROOT / "reports/research/right_t012_index_sources_v1"
ARMS = {"baseline": (), "annual": ANNUAL_MODEL_COLUMNS, "all_ma": ALL_MA_MODEL_COLUMNS}


def build() -> None:
    sources, frames = [], []
    for spec in KEY_INDEX_SPECS:
        path = SOURCE / f"{spec.ts_code}_index_daily.parquet"
        f = pd.read_parquet(path, columns=["ts_code", "trade_date", "close"])
        frames.append(f)
        sources.append({"code":spec.ts_code, "slug":spec.slug, "path":str(path),
                        "rows":len(f), "start":str(f.trade_date.min()),
                        "end":str(f.trade_date.max()), "sha256":sha256(path)})
    calendar = pd.to_datetime(pd.read_csv(BASE_DATA / "calendar.csv").date)
    end = pd.to_datetime(max(s["end"] for s in sources), format="%Y%m%d")
    features = compute_index_ma_state_features(pd.concat(frames, ignore_index=True), market_calendar=calendar, as_of=end)
    features.to_parquet(DATA / "index_ma_features.parquet", index=False)
    # Audit exact-date availability, never silently merge stale snapshots.
    keys = pd.read_parquet(BASE_DATA / "events.parquet", columns=["symbol","date"])
    selected = keys.merge(features, on="date", how="left", validate="many_to_one")
    coverage = pd.DataFrame({"feature":INDEX_MA_STATE_FEATURE_COLUMNS,
                              "coverage":selected[list(INDEX_MA_STATE_FEATURE_COLUMNS)].notna().mean().to_numpy()})
    coverage.to_csv(REPORT / "feature_coverage.csv", index=False)
    for c in ALL_MA_MODEL_COLUMNS:
        if selected[c].isna().any():
            raise ValueError(f"Core index history incomplete on event date: {c}")
    del selected
    save_json(REPORT / "feature_manifest.json", {
        "schema_version":INDEX_MA_STATE_SCHEMA_VERSION, "sources":sources,
        "public_start_dates":PUBLIC_START_DATES, "columns":list(INDEX_MA_STATE_FEATURE_COLUMNS),
        "annual_model_columns":list(ANNUAL_MODEL_COLUMNS), "all_ma_model_columns":list(ALL_MA_MODEL_COLUMNS),
        "first_date":str(features.date.min()), "last_date":str(features.date.max()),
        "rows":len(features), "feature_sha256":sha256(DATA/"index_ma_features.parquet"),
        "warmup_policy":"full window, no pre-launch backcast, missing exchange day resets window",
        "join_policy":"exact signal date; no forward/backward fill", "production_modified":False,
    })
    print(f"Built {len(features)} market dates, {len(INDEX_MA_STATE_FEATURE_COLUMNS)} variables; annual={len(ANNUAL_MODEL_COLUMNS)}, all={len(ALL_MA_MODEL_COLUMNS)}", flush=True)


def metrics(frame: pd.DataFrame, probability: np.ndarray, arm: str, fold: str):
    work = frame[["symbol","date","terminal_return","good_path5"]].copy()
    work["probability"] = probability
    row = {"arm":arm,"fold":fold,"rows":len(work), "ap":float(average_precision_score(work.good_path5,probability)),
           "auc":float(roc_auc_score(work.good_path5,probability))}
    daily = []
    ranked = work.sort_values(["date","probability","symbol"], ascending=[True,False,True], kind="stable")
    for k in (10,20):
        top = ranked.groupby("date",sort=False).head(k).copy()
        top["net"] = top.terminal_return-.002
        net = top.net
        gp,gl = net[net>0].sum(), -net[net<0].sum()
        row.update({f"top{k}_net_return":float(net.mean()), f"top{k}_win_rate":float((net>0).mean()),
                    f"top{k}_profit_factor":float(gp/gl) if gl else None, f"top{k}_trades":len(top)})
        d = top.groupby("date").agg(net=("net","mean"),count=("net","size")).reset_index()
        d["top_k"],d["arm"],d["fold"] = k,arm,fold
        daily.append(d)
    return row,pd.concat(daily,ignore_index=True)


def gate_flags(features: pd.DataFrame) -> pd.DataFrame:
    out = features[["date"]].copy()
    annual = features.market_index_ma250_above_ratio
    medium = features.market_index_ma60_above_ratio
    confirmed = features[[f"index_{s}_ma250_below_days" for s in CORE_INDEX_SLUGS]].ge(5).sum(axis=1)
    valid = annual.notna() & medium.notna()
    out["annual_majority"] = valid & annual.ge(.5)
    out["trend_majority"] = valid & annual.ge(.5) & medium.ge(.5)
    out["avoid_confirmed_bear"] = valid & ~(confirmed.ge(4) & medium.lt(.5))
    out["market_state"] = np.select([annual.ge(.5)&medium.ge(.5),annual.lt(.5)&medium.lt(.5)], ["bull","bear"], default="mixed")
    out.loc[~valid,"market_state"] = "unavailable"
    out["csi300_annual_state"] = np.select([
        features.index_csi300_ma250_below_days.between(1,5),
        features.index_csi300_ma250_below_days.between(6,20),
        features.index_csi300_ma250_below_days.gt(20),
        features.index_csi300_ma250_above.eq(1)],
        ["below_1_5","below_6_20","below_over20","above"], default="equal_or_unavailable")
    return out


def analyze_gates(frame: pd.DataFrame, predictions: pd.DataFrame, fold: str) -> None:
    features = pd.read_parquet(DATA / "index_ma_features.parquet")
    flags = gate_flags(features)
    selected = frame[["symbol","date","terminal_return"]].merge(predictions,on=["symbol","date"],validate="one_to_one")
    selected = selected.sort_values(["date","probability","symbol"],ascending=[True,False,True],kind="stable").groupby("date",sort=False).head(20)
    selected["net"] = selected.terminal_return-.002
    daily = selected.groupby("date").agg(net=("net","mean"), trades=("net","size")).reset_index()
    daily = daily.merge(flags,on="date",validate="one_to_one",how="left")
    if daily.market_state.isna().any() or daily.market_state.eq("unavailable").any():
        raise ValueError("Market state missing for test date")
    rows=[]
    for gate in ["annual_majority","trend_majority","avoid_confirmed_bear"]:
        opened=daily[gate].astype(bool)
        gated=daily.net.where(opened,0.0)
        delta=gated-daily.net
        admitted=selected[selected.date.isin(daily.loc[opened,"date"])]
        win=admitted.net
        gp,gl=win[win>0].sum(),-win[win<0].sum()
        rows.append({"fold":fold,"gate":gate,"test_dates":len(daily),"admitted_dates":int(opened.sum()),
            "date_coverage":float(opened.mean()),"trade_coverage":len(admitted)/len(selected),
            "admitted_trades":len(admitted),"conditional_net_return":float(win.mean()) if len(win) else None,
            "baseline_cash_inclusive_daily_return":float(daily.net.mean()),
            "gated_cash_inclusive_daily_return":float(gated.mean()),
            "delta_cash_inclusive_daily_return":float(delta.mean()),
            "delta_daily_ci95":block_interval(delta.to_numpy()),
            "conditional_profit_factor":float(gp/gl) if gl else None,
            "conditional_win_rate":float((win>0).mean()) if len(win) else None})
    save_json(REPORT/f"gates_{fold}.json",rows)
    daily["fold"]=fold
    daily.to_parquet(REPORT/f"gate_daily_{fold}.parquet",index=False)
    strata=[]
    for category in ["market_state","csi300_annual_state"]:
        f=selected.merge(flags[["date",category]],on="date",validate="many_to_one")
        for value,g in f.groupby(category):
            strata.append({"fold":fold,"category":category,"state":value,"dates":g.date.nunique(),
                           "trades":len(g),"net_return":float(g.net.mean()),"win_rate":float(g.net.gt(0).mean())})
    pd.DataFrame(strata).to_csv(REPORT/f"state_strata_{fold}.csv",index=False)


def train(jobs: int) -> None:
    prior=json.loads((BASE_REPORT/"experiment_contract.json").read_text())
    for name in ["events.parquet","labels.parquet"]:
        if sha256(BASE_DATA/name)!=prior["input_hashes"][name]:
            raise ValueError("Frozen baseline training inputs changed")
    contract={"feature_schema":INDEX_MA_STATE_SCHEMA_VERSION,"model_spec":asdict(LONG_TASK_DEEP_XGB_CLASSIFIER_SPEC),
        "model_seed":42,"target":"good_path5","entry":"next_close","horizon":5,"cost_bps":20,
        "arms":ARMS,"folds":[asdict(f) for f in DEFAULT_YEAR_FOLDS],
        "feature_hash":sha256(DATA/"index_ma_features.parquet"),"base_hashes":prior["input_hashes"],
        "core_indices":CORE_INDEX_SLUGS,"gate_definitions":{
            "annual_majority":"at least 4 of 7 core indices above MA250",
            "trend_majority":"at least 4 of 7 above MA60 and at least 4 above MA250",
            "avoid_confirmed_bear":"exclude >=4 of 7 below MA250 for >=5 sessions while MA60 majority below"},
        "gate_execution":"entry admission only, fixed 5-day exit, off days earn zero",
        "no_tuning_or_test_selection":True,"previously_observed_test_years":True,"production_modified":False}
    path=REPORT/"experiment_contract.json"
    if path.exists() and json.loads(path.read_text())!=json.loads(json.dumps(contract)):
        raise ValueError("Experiment contract changed; use a new report path")
    save_json(path,contract)
    frame=load_frame()
    features=pd.read_parquet(DATA/"index_ma_features.parquet", columns=["date",*ALL_MA_MODEL_COLUMNS])
    features[list(ALL_MA_MODEL_COLUMNS)] = features[list(ALL_MA_MODEL_COLUMNS)].astype("float32")
    frame=frame.merge(features,on="date",how="left",validate="many_to_one")
    print(f"Training events {len(frame):,}, new features {len(ALL_MA_MODEL_COLUMNS)}",flush=True)
    for fold in DEFAULT_YEAR_FOLDS:
        split=split_by_year_fold(frame,fold)
        test=split.test
        baseline=pd.read_parquet(BASE_REPORT/f"predictions_{fold.name}_baseline.parquet")
        pd.testing.assert_frame_equal(test[["symbol","date"]].reset_index(drop=True),baseline[["symbol","date"]].reset_index(drop=True))
        base,daily=metrics(test,baseline.probability.to_numpy(),"baseline",fold.name)
        save_json(REPORT/f"metrics_{fold.name}_baseline.json",base)
        daily.to_parquet(REPORT/f"daily_{fold.name}_baseline.parquet",index=False)
        analyze_gates(test,baseline,fold.name)
        early,calibration,_=_split_validation_stages(split.validation.reset_index(drop=True),"good_path5")
        for arm in ("annual","all_ma"):
            path=REPORT/f"metrics_{fold.name}_{arm}.json"
            if path.exists():continue
            print(f"Fitting {fold.name} {arm}: train={len(split.train):,}, test={len(test):,}",flush=True)
            started=time.monotonic()
            model=_fit_unified_model(split.train.reset_index(drop=True),early,calibration,[*LEFT_SIDE_FACTOR_COLUMNS,*ARMS[arm]],"good_path5",n_jobs=jobs)
            probability=model.predict_proba(test)[:,1]
            row,daily=metrics(test,probability,arm,fold.name)
            row.update(seconds=round(time.monotonic()-started,2),best_iteration=model.base_model.best_iteration,
                       train_rows=len(split.train),new_feature_count=len(ARMS[arm]))
            importance=dict(zip(model.base_model.feature_names_in_,model.base_model.classifier.feature_importances_.tolist()))
            row["new_feature_importance_total"]=sum(importance[c] for c in ARMS[arm])
            pred=test[["symbol","date"]].copy();pred["probability"]=probability
            pred.to_parquet(REPORT/f"predictions_{fold.name}_{arm}.parquet",index=False)
            daily.to_parquet(REPORT/f"daily_{fold.name}_{arm}.parquet",index=False)
            save_json(path,row)
            print(json.dumps(row),flush=True)
            del model,probability,pred;gc.collect()
        del split,test,early,calibration;gc.collect()
    summarize()


def summarize() -> None:
    records=[json.loads(p.read_text()) for p in sorted(REPORT.glob("metrics_*.json"))]
    pd.DataFrame(records).to_csv(REPORT/"metrics.csv",index=False)
    daily=pd.concat([pd.read_parquet(p) for p in sorted(REPORT.glob("daily_*.parquet"))],ignore_index=True)
    comparisons=[]
    for fold in ("A","B","C"):
        sub=daily[(daily.fold==fold)&(daily.top_k==20)]
        paired=sub.pivot(index="date",columns="arm",values="net")
        for arm in ("annual","all_ma"):
            delta=(paired[arm]-paired.baseline).to_numpy()
            comparisons.append({"fold":fold,"arm":arm,"daily_top20_delta":float(delta.mean()),"ci95":block_interval(delta)})
    save_json(REPORT/"increment_comparison.json",comparisons)
    gates=[row for p in sorted(REPORT.glob("gates_*.json")) for row in json.loads(p.read_text())]
    save_json(REPORT/"gate_comparison.json",gates)
    pd.DataFrame(gates).to_csv(REPORT/"gate_comparison.csv",index=False)
    pd.concat([pd.read_csv(p) for p in sorted(REPORT.glob("state_strata_*.csv"))],ignore_index=True).to_csv(REPORT/"state_strata.csv",index=False)
    print(json.dumps(comparisons,ensure_ascii=False,indent=2),flush=True)


def baseline_gates() -> None:
    labels = pd.read_parquet(BASE_DATA/"labels.parquet", columns=["symbol","date","terminal_return"])
    for fold in DEFAULT_YEAR_FOLDS:
        prediction = pd.read_parquet(BASE_REPORT/f"predictions_{fold.name}_baseline.parquet")
        selected = prediction[["symbol","date"]].merge(labels,on=["symbol","date"],validate="one_to_one")
        if selected.terminal_return.isna().any():
            raise ValueError("Frozen baseline has missing realized return")
        analyze_gates(selected,prediction,fold.name)
    print("Baseline gate analysis complete for A/B/C",flush=True)


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=["build","train","summarize","gates"])
    p.add_argument("--jobs",type=int,default=6)
    args=p.parse_args();DATA.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    if args.command=="build":build()
    elif args.command=="gates":baseline_gates()
    elif args.command=="train":train(args.jobs)
    else:summarize()


if __name__=="__main__":main()
