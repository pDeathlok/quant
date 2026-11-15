"""Frozen model/exit hypotheses; no retrospective bull/bear turning points."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from backtest_index_futures import ROOT
from evaluate_index_futures_exit_value import STATE_FEATURES, PRODUCT_FEATURES, state_weights
from evaluate_index_futures_regime import OUT as PREVIOUS
from index_futures_state_rules import evidence_admits, holding_limit, profit_giveback

OUT=ROOT/"state_model_policies"
MODEL_NAMES=("base","factor","interaction")
EXIT_NAMES=("none","model_base","model_state","fixed10","state_horizon","trail_uniform","state_trail","combined")
ARMS={
    "original":(0,0,"none"), "hard_entry":(0,0,"entry"), "hard_both":(0,0,"both"),
    "model_base":(1,0,"none"), "state_factor":(2,0,"none"), "state_interaction":(3,0,"none"),
    "exit_base":(3,1,"none"), "exit_state":(3,2,"none"),
    "fixed10":(3,3,"none"), "state_horizon":(3,4,"none"),
    "trail_uniform":(3,5,"none"), "state_trail":(3,6,"none"), "combined":(3,7,"none"),
}
INTERACTIONS=("f_momentum_5","f_momentum_20","f_basis_annual","f_oi_change",
              "etf_ma_gap_20","etf_vol_ratio","future_close_location","size_mid_large20")
REGIME_FEATURES=["regime_bull","regime_bear","regime_distance","regime_slope","regime_atr"]


def register() -> None:
    OUT.mkdir(exist_ok=True)
    protocol=dict(end="20261106",capital=500000,max_lots=1,arms=ARMS,
        states="reuse frozen per-spot MA200/ATR20/MA20slope from bull_bear_regime; no MA search in this study",
        features="same60 futures/ETF/relative-index factors plus product indicators; add bull/bear,distanceATR clipped+-10,slopeATR clipped+-5,ATR/close; interactions bull/bear x eight predefined factors; continuation adds state x age/marked_return/pullback",
        entries="retain original candidate directions, ranking, calendar, margin/reserve; filter with predicted5d net profit probability>=.55 AND expected net-.5*expected adverse>0; same threshold for trend and countertrend, no hard direction ban",
        entry_labels="all products/dates,both sides,5session executable open/close mark paths including entry,roll,exit fees, each3/6bps; training not selected by future profitability or realized portfolio entries",
        models="quarterly separate long/short LR C=.1 and Ridge alpha=.1*n; no hyperparameter search; past36months mature targets,12m half-life; training-only imputation/scaling",
        warmup="entry needs244 distinct mature signal dates and both binary classes; otherwise explicitly original entry fallback. Continuation needs244 dates and60 mature episodes, otherwise no supplemental model exit; fallback statistics disclosed",
        continuation="same past candidate episodes ages1..19 as frozen exit research; train on full20d episode maturity strictly before quarter cutoff; raw5 hold-vs-close net and adverse; base versus state-interaction models; two consecutive utility<=0 closes add exit, all old exits retained",
        expiry="fixed10 control vs entry-state-frozen aligned20/transition10/counter5 maximum sessions; state changes cannot extend the deadline; older protective/model exits may occur earlier",
        trailing="previous-close net episode profit including paid fees/rolls and hypothetical close fee; arm when peak>=5% entry equity; uniform50%giveback control vs current-state aligned50%,otherwise25%; no reset on rolls/quarters; close next open, gaps may overshoot",
        combined="interaction entry + state continuation + state deadline + state trailing; no added stop or cash candidate",
        evaluation="fixed base vs original quarterly policy reselection within each frozen arm; all13arms; new grid control/loss_exit/mean5, same58monthly starts+Jan5day+annual4profiles; other old calibrated/tail studies retained, no new claims for their unrun combinations",
        selector="same past36m completed122/244d starts,12m weights,exactcurrentequity,same inner/outer arm; same4original candidate policies and cost/funding gates; no retrospective best-arm ensemble",
        metrics="single-year,each-start-to-common-end,122/244sessions,continuousannual,monthly/phase,worststart,flat/opportunity/side/regime; no topups, cashinterest0",
        safeguards="existing predictions read via feature whitelist; all labels truncated to frozen end; no random crossvalidation; old/hard controls parity; causal-prefix and Python/C++ daily parity tests",
        limits="history repeatedly inspected not freshOOT; overlapping samples not independent; utility is expected risk,not a calibrated tail bound; paper fills and margin scenarios do not prove intraday solvency",
        reference="https://scikit-learn.org/stable/common_pitfalls")
    path=OUT/"protocol.json"; content=json.dumps(protocol,ensure_ascii=False,indent=2)
    if path.exists() and path.read_text()!=content:
        raise RuntimeError("Refuse modifying preregistered hypotheses")
    path.write_text(content)
    (OUT/"global_calendar.csv").write_bytes((PREVIOUS/"global_calendar.csv").read_bytes())
    sources=[]
    for path in (PREVIOUS/"states_200.parquet", ROOT/"adaptation_comparison/quarterly_predictions.parquet",
                 ROOT/"exit_value/states.parquet", ROOT/"exit_calibration_starts/calibrated_forecasts.parquet"):
        sources.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (OUT/"sources.json").write_text(json.dumps(sources,indent=2))


def add_regime_features(frame: pd.DataFrame, date_column: str, *, states=None) -> pd.DataFrame:
    if states is None:
        states=pd.read_parquet(PREVIOUS/"states_200.parquet")
    state=states[["trade_date","symbol","state","distance_atr","slope20","atr20","close"]].copy()
    state["regime_bull"]=state.state.eq(1).astype(float)
    state["regime_bear"]=state.state.eq(-1).astype(float)
    state["regime_distance"]=state.distance_atr.clip(-10,10)
    state["regime_slope"]=(state.slope20/state.atr20).clip(-5,5)
    state["regime_atr"]=state.atr20/state.close
    result=frame.merge(state[["trade_date","symbol",*REGIME_FEATURES]].rename(columns={"trade_date":date_column}),
                       on=[date_column,"symbol"],how="left",validate="many_to_one")
    assert result[REGIME_FEATURES].notna().all().all()
    for product in ("IF","IH","IC","IM"):
        result[f"product_{product}"]=result.symbol.eq(product).astype(float)
    variables=[*INTERACTIONS,*(["age_fraction","marked_return","pullback"] if "age_fraction" in frame else [])]
    for state_name in ("regime_bull","regime_bear"):
        for feature in variables:
            result[f"interaction_{state_name}_{feature}"]=result[state_name]*result[feature]
    return result


def entry_path(rows, start: int, side: int, fee: float, closes: dict, horizon: int = 5):
    path=rows[start:start+horizon]
    if len(path)<horizon or any(a.exit_date!=b.entry_date for a,b in zip(path,path[1:])):
        return None
    first=path[0]; multiplier=first.multiplier
    costs=first.entry_open*multiplier*fee; gross=0.; worst=-2*costs
    for j,r in enumerate(path):
        if j and r.ts_code!=path[j-1].ts_code:
            costs+=(path[j-1].exit_open+r.entry_open)*multiplier*fee
        opened=gross-costs-r.entry_open*multiplier*fee
        close=closes[r.entry_date,r.ts_code]
        marked=gross+side*(close-r.entry_open)*multiplier-costs-close*multiplier*fee
        gross+=side*(r.exit_open-r.entry_open)*multiplier
        end=gross-costs-r.exit_open*multiplier*fee
        worst=min(worst,opened,marked,end)
    notional=first.entry_open*multiplier
    return dict(maturity=path[-1].exit_date,net=end/notional,risk=-worst/notional)


def build_entry_labels(panel: pd.DataFrame, closes: dict) -> pd.DataFrame:
    records=[]
    for _,part in panel.groupby("symbol"):
        rows=list(part.sort_values("entry_date").itertuples(index=False))
        for i,r in enumerate(rows):
            for direction in (1,-1):
                record=dict(entry_date=r.entry_date,symbol=r.symbol,direction=direction)
                for bps in (3,6):
                    target=entry_path(rows,i,direction,bps/10000,closes)
                    if target:
                        record.update(maturity=target["maturity"],**{f"{k}_{bps}":target[k] for k in ("net","risk")})
                records.append(record)
    return panel.merge(pd.DataFrame(records),on=["entry_date","symbol"],validate="one_to_many")


def pipeline(estimator):
    return make_pipeline(SimpleImputer(strategy="median",keep_empty_features=True),StandardScaler(),estimator)


def train_models() -> None:
    from evaluate_index_futures_directions import load_bars
    panel=pd.read_parquet(ROOT/"adaptation_comparison/quarterly_predictions.parquet")
    assert panel.exit_date.max()=="20261106"
    bars=load_bars();closes=bars.close.to_dict()
    base=json.loads((ROOT/"relative_indices/feature_sets.json").read_text())["combined"]+PRODUCT_FEATURES
    entries=add_regime_features(build_entry_labels(panel,closes),"trade_date")
    sets={"base":base,"factor":base+REGIME_FEATURES,
          "interaction":base+REGIME_FEATURES+entries.filter(regex="^interaction_").columns.tolist()}
    entries.to_parquet(OUT/"entry_dataset.parquet",index=False)
    audits=[]; predictions=[]; coeff=[]
    for quarter,part in panel.groupby("quarter",sort=True):
        cutoff=part.trade_date.min();begin=(pd.Timestamp(cutoff)-pd.DateOffset(months=36)).strftime("%Y%m%d")
        for direction in (1,-1):
            train=entries[entries.direction.eq(direction)&entries.trade_date.ge(begin)&entries.maturity.lt(cutoff)]
            test=entries[entries.direction.eq(direction)&entries.quarter.eq(quarter)]
            age=(pd.Timestamp(cutoff)-pd.to_datetime(train.maturity)).dt.days.to_numpy()
            weights=np.exp2(-age/365.25); weights=weights/weights.mean() if len(weights) else weights
            for model,features in sets.items():
                for bps in (3,6):
                    y=train[f"net_{bps}"].gt(0).astype(int)
                    ready=train.trade_date.nunique()>=244 and y.nunique()==2
                    output=test[["entry_date","trade_date","symbol","direction","quarter","maturity",f"net_{bps}",f"risk_{bps}",*REGIME_FEATURES]].copy()
                    output["model"]=model;output["bps"]=bps;output["ready"]=ready
                    output[["probability","pred_net","pred_risk"]]=np.nan
                    if ready:
                        assert train.maturity.max()<cutoff<=test.trade_date.min()
                        lr=pipeline(LogisticRegression(C=.1,max_iter=1000))
                        ridge=pipeline(Ridge(alpha=.1*len(train)))
                        lr.fit(train[features],y,standardscaler__sample_weight=weights,logisticregression__sample_weight=weights)
                        ridge.fit(train[features],train[[f"net_{bps}",f"risk_{bps}"]],
                            standardscaler__sample_weight=weights,ridge__sample_weight=weights)
                        output["probability"]=lr.predict_proba(test[features])[:,1]
                        raw=ridge.predict(test[features]);output["pred_net"]=raw[:,0];output["pred_risk"]=np.maximum(0,raw[:,1])
                        coeff.extend(dict(quarter=quarter,direction=direction,bps=bps,model=model,feature=f,coefficient=c)
                                     for f,c in zip(features,lr[-1].coef_[0]))
                    audits.append(dict(kind="entry",quarter=quarter,direction=direction,bps=bps,model=model,cutoff=cutoff,
                        rows=len(train),days=train.trade_date.nunique(),last_maturity=train.maturity.max(),ready=ready,features=len(features)))
                    predictions.append(output)
        print("entry models",quarter,flush=True)
    pd.concat(predictions,ignore_index=True).to_parquet(OUT/"entry_forecasts.parquet",index=False)
    pd.DataFrame(coeff).to_csv(OUT/"entry_coefficients.csv",index=False)
    states=add_regime_features(pd.read_parquet(ROOT/"exit_value/states.parquet"),"signal_date")
    exitsets={"base":base+STATE_FEATURES,"interaction":base+STATE_FEATURES+REGIME_FEATURES+
              states.filter(regex="^interaction_").columns.tolist()}
    exit_predictions=[]
    for quarter,part in panel.groupby("quarter",sort=True):
        cutoff=part.trade_date.min();begin=(pd.Timestamp(cutoff)-pd.DateOffset(months=36)).strftime("%Y%m%d")
        for direction in (1,-1):
            train=states[states.direction.eq(direction)&states.signal_date.ge(begin)&states.maturity.lt(cutoff)].dropna(subset=["net_5_3","risk_5_3","net_5_6","risk_5_6"])
            test=states[states.direction.eq(direction)&states.quarter.eq(quarter)]
            ready=train.signal_date.nunique()>=244 and train.episode.nunique()>=60
            for model,features in exitsets.items():
                output=test[["entry","date","signal_date","symbol","direction","quarter","maturity","episode","notional","net_5_3","risk_5_3","net_5_6","risk_5_6"]].copy()
                output["model"]=model;output["ready"]=ready
                targets=["net_5_3","risk_5_3","net_5_6","risk_5_6"]
                output[[f"pred_{k}" for k in targets]]=np.nan
                if ready and len(test):
                    assert train.maturity.max()<cutoff<=test.signal_date.min()
                    weights=state_weights(train,cutoff)
                    estimator=pipeline(Ridge(alpha=.1*len(train)))
                    estimator.fit(train[features],train[targets],standardscaler__sample_weight=weights,ridge__sample_weight=weights)
                    values=estimator.predict(test[features]); values[:,[1,3]]=np.maximum(0,values[:,[1,3]])
                    output[[f"pred_{k}" for k in targets]]=values
                audits.append(dict(kind="exit",quarter=quarter,direction=direction,model=model,cutoff=cutoff,
                    rows=len(train),days=train.signal_date.nunique(),episodes=train.episode.nunique(),
                    last_maturity=train.maturity.max(),ready=ready,features=len(features)))
                exit_predictions.append(output)
        print("exit models",quarter,flush=True)
    pd.concat(exit_predictions,ignore_index=True).to_parquet(OUT/"exit_forecasts.parquet",index=False)
    pd.DataFrame(audits).to_csv(OUT/"training_audit.csv",index=False)
    (OUT/"feature_sets.json").write_text(json.dumps(dict(entry=sets,exit=exitsets),indent=2))


if __name__=="__main__":
    register()
    train_models()
