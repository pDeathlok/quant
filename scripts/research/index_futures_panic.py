"""Frozen panic-feature ablation on the current state-horizon research candidate."""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"src"))
from quant.features.index_panic import QVIX_FEATURES, STRESS_FEATURES, QVIX_MAP, price_stress, option_features
from backtest_index_futures import ROOT
from index_futures_state_models import OUT as PREVIOUS, pipeline

OUT = ROOT/"panic_factors"
EVIDENCE_NAMES = ("base", "factor", "interaction", "qvix", "stress", "both")
ARMS = {"baseline": (3, 4, "none"), "qvix": (4, 4, "none"),
        "stress": (5, 4, "none"), "both": (6, 4, "none")}
EXTRA = dict(qvix=QVIX_FEATURES, stress=STRESS_FEATURES, both=QVIX_FEATURES+STRESS_FEATURES)
END = "20261106"


def register():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = dict(end=END, capital=500000, max_lots=1, arms=ARMS,
        qvix="AKShare/optbbs third-party historical snapshot; IH50ETF,IF300ETF,IC500ETF,IM1000index; common50ETF; source conflicts quarantined; no fill or extrapolation. Rank vs past252/min60,5session relative change,IV minus underlying20session realized vol,missing flag. One extra session lag relative to signal close; forecasts executed next open.",
        stress="per-index past252/min60 mid-percentiles of20session downside RMS,60session drawdown,5session negative log-return times volume/prior20volume. Equal3-component0-100 price-stress score; three component ranks+5day score change in model. Not option-implied VIX; no stock-universe survivorship dependency.",
        models="retain frozen85 features,5session fee/risk labels,244day warmup,quarterly long/short LR C=.1 and Ridge alpha=.1*n,past36months matured strictly before cutoff,12month half-life,training-only imputation/scaling. No tuning, no new direction/threshold/exit labels.",
        missing="own QVIX missing explicitly flagged; imputation learned on train only. If common QVIX missing, qvix arm falls back original85-feature forecast, both falls back stress forecast; stress warmup falls back original. Do not manufacture a missing IV history.",
        execution="original candidate directions/calendar/ranking,probability>=.55 and net-.5*risk>0; mean5 protective exits and entry-state locked20/10/5 horizon; no topups; max1lot; no new exit veto",
        selection="fixed base policy first,quarterly rolling4existing policies second; exact current equity historical122/244session validation,previous36months,recency weights and unchanged switch/cost gates; no hindsight selection among four research arms",
        evaluation="same2022-2026 monthly/January4phase/annual starts,15% and25% margin,3/6bps; single-year,to-common-end,122/244sessions,continuousannual; side,flat,missedpositive/avoidednegative baseline pnl; original state-horizon parity and Python/C++ parity",
        caveats="reused history is exploratory not freshOOT; vendor revisions/publication timestamps not archived,extra lag is assumption not proof; source discontinuities and overlapping windows disclosed; fills are daily not intraday solvency guarantee")
    path = OUT/"protocol.json"
    content = json.dumps(protocol, ensure_ascii=False, indent=2)
    if path.exists() and path.read_text()!=content:
        raise RuntimeError("Refuse changing frozen panic hypotheses")
    path.write_text(content)
    sources = [PREVIOUS/"entry_dataset.parquet", PREVIOUS/"entry_forecasts.parquet", PREVIOUS/"feature_sets.json", OUT/"sources/optbbs_k.csv"]
    sources += [ROOT/f"{symbol}_spot.parquet" for symbol in QVIX_MAP]
    sources += [ROOT/"adaptation_comparison/quarterly_predictions.parquet",
                ROOT/"exit_calibration_starts/calibrated_forecasts.parquet",
                PREVIOUS/"global_calendar.csv"]
    (OUT/"input_hashes.json").write_text(json.dumps([dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sources],indent=2))
    (OUT/"global_calendar.csv").write_bytes((PREVIOUS/"global_calendar.csv").read_bytes())


def build_features():
    spots, stress = {}, {}
    for symbol in QVIX_MAP:
        frame = pd.read_parquet(ROOT/f"{symbol}_spot.parquet")
        frame = frame[frame.trade_date.le(END)].set_index("trade_date").sort_index()
        spots[symbol] = frame
        stress[symbol] = price_stress(frame)
    qseries = {}
    for name in set(QVIX_MAP.values()):
        frame = pd.read_parquet(OUT/f"sources/qvix_{name}.parquet")
        qseries[name] = frame.loc[frame.trade_date.le(END)].set_index("trade_date").close
    common = option_features(qseries["50etf"], stress["IH"].realized_vol20, spots["IH"].index)
    parts = []
    for symbol, spot in spots.items():
        own = option_features(qseries[QVIX_MAP[symbol]], stress[symbol].realized_vol20, spot.index)
        part = stress[symbol].drop(columns="realized_vol20").join(common.add_prefix("panic_common_")).join(own.add_prefix("panic_own_"))
        part["symbol"] = symbol
        parts.append(part.reset_index())
    result = pd.concat(parts, ignore_index=True)
    result.to_parquet(OUT/"features.parquet", index=False)
    for scope in ("common", "own"):
        date = result[f"panic_{scope}_source_date"]
        assert date.dropna().lt(result.loc[date.notna(), "trade_date"]).all()
    result["year"] = result.trade_date.str[:4]
    coverage = result[result.year.ge("2020")].groupby(["year", "symbol"]).agg(
        days=("trade_date", "size"), common_missing=("panic_common_missing", "mean"),
        own_missing=("panic_own_missing", "mean"))
    coverage.to_csv(OUT/"coverage.csv")
    return result.drop(columns="year")


def train():
    register()
    features = build_features()
    entries = pd.read_parquet(PREVIOUS/"entry_dataset.parquet").merge(features, on=["trade_date", "symbol"], how="left", validate="many_to_one")
    base = json.loads((PREVIOUS/"feature_sets.json").read_text())["entry"]["interaction"]
    sets = {k:base+v for k,v in EXTRA.items()}
    (OUT/"feature_sets.json").write_text(json.dumps(sets,indent=2))
    old = pd.read_parquet(PREVIOUS/"entry_forecasts.parquet")
    old = old[old.model.eq("interaction")].set_index(["entry_date", "symbol", "direction", "bps"])
    predictions, audits, coefficients = [], [], []
    with threadpool_limits(limits=1):
        for quarter, part in entries.groupby("quarter", sort=True):
            cutoff = part.trade_date.min()
            begin = (pd.Timestamp(cutoff)-pd.DateOffset(months=36)).strftime("%Y%m%d")
            for direction in (1,-1):
                train = entries[entries.direction.eq(direction)&entries.trade_date.ge(begin)&entries.maturity.lt(cutoff)]
                test = part[part.direction.eq(direction)]
                age = (pd.Timestamp(cutoff)-pd.to_datetime(train.maturity)).dt.days.to_numpy()
                weights = np.exp2(-age/365.25)
                if len(weights): weights /= weights.mean()
                for bps in (3,6):
                    previous = old.reindex(pd.MultiIndex.from_arrays([test.entry_date,test.symbol,test.direction,np.full(len(test),bps)],names=old.index.names))
                    stress_prediction = previous[["probability", "pred_net", "pred_risk"]].to_numpy()
                    for name in ("stress","qvix","both"):
                        y = train[f"net_{bps}"].gt(0).astype(int)
                        ready = train.trade_date.nunique()>=244 and y.nunique()==2
                        output = test[["entry_date","trade_date","symbol","direction","quarter","maturity",f"net_{bps}",f"risk_{bps}","panic_common_missing","panic_own_missing"]].copy()
                        output["bps"], output["model"], output["ready"] = bps,name,ready
                        output[["probability","pred_net","pred_risk"]] = previous[["probability","pred_net","pred_risk"]].to_numpy()
                        output["fallback"] = "baseline_warmup"
                        if ready:
                            assert train.maturity.max()<cutoff<=test.trade_date.min()
                            lr = pipeline(LogisticRegression(C=.1,max_iter=1000))
                            ridge = pipeline(Ridge(alpha=.1*len(train)))
                            lr.fit(train[sets[name]], y, standardscaler__sample_weight=weights, logisticregression__sample_weight=weights)
                            ridge.fit(train[sets[name]],train[[f"net_{bps}",f"risk_{bps}"]], standardscaler__sample_weight=weights,ridge__sample_weight=weights)
                            output["probability"] = lr.predict_proba(test[sets[name]])[:,1]
                            raw = ridge.predict(test[sets[name]])
                            output["pred_net"], output["pred_risk"] = raw[:,0],np.maximum(0,raw[:,1])
                            output["fallback"] = "none"
                            columns = ["probability","pred_net","pred_risk"]
                            missing = test[STRESS_FEATURES].isna().any(axis=1) if name=="stress" else test.panic_common_missing.ne(0)
                            fallback = stress_prediction if name=="both" else previous[columns].to_numpy()
                            output.loc[missing,columns] = fallback[missing.to_numpy()]
                            output.loc[missing,"fallback"] = "stress_missing_common" if name=="both" else "baseline_missing_features"
                            coefficients.extend(dict(quarter=quarter,direction=direction,bps=bps,model=name,feature=f,coefficient=float(c)) for f,c in zip(sets[name],lr[-1].coef_[0]))
                        if name=="stress": stress_prediction = output[["probability","pred_net","pred_risk"]].to_numpy()
                        audits.append(dict(quarter=quarter,direction=direction,bps=bps,model=name,cutoff=cutoff,begin=begin,
                            rows=len(train),days=train.trade_date.nunique(),last_maturity=train.maturity.max(),ready=ready,features=len(sets[name]),fallback_rows=output.fallback.ne("none").sum()))
                        predictions.append(output)
            print("panic models",quarter,flush=True)
    pd.concat(predictions,ignore_index=True).to_parquet(OUT/"forecasts.parquet",index=False)
    pd.DataFrame(audits).to_csv(OUT/"training_audit.csv",index=False)
    pd.DataFrame(coefficients).to_csv(OUT/"coefficients.csv",index=False)


if __name__=="__main__": train()
