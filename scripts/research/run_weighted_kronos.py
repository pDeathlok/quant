#!/usr/bin/env python
"""Six/twelve-month recency-weighted monthly heads and weekly OOT calibration."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from run_kronos_rolling import capital_inputs, prediction_metrics, save_json, sha256
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from quant.research.kronos_multiscale import predict_subscores
from quant.research.kronos_rolling import enough, refit_segments
from quant.research.sequence_strategy_validation import (
    capital_backtest,
    choose_confidence_gate,
    paired_block_gain,
    top_picks,
)
from quant.research.trend_research import fit_weighted_subscores, recency_weights, recent_window

ROOT = Path(__file__).resolve().parents[2]
FEATURES = ["baseline_percentile", "sequence_return", "sequence_profit", "sequence_downside"]


def train_predictions(frame, embeddings, calendar, months, half_life, directory):
    result = frame[frame.date.ge("2025-01-01")].copy()
    result["baseline_percentile"] = result.groupby("date").baseline_score.rank(pct=True)
    result[["sequence_return", "sequence_profit", "sequence_downside"]] = np.nan
    result["head_refit_date"] = pd.NaT
    audit = []
    directory.mkdir(parents=True)
    dates = calendar[(calendar >= pd.Timestamp("2025-01-01")) & (calendar <= result.date.max())]
    for refit, segment in refit_segments(dates, "monthly"):
        train = recent_window(frame, refit, months)
        pending = result[result.date.isin(segment)]
        if not enough(train, 60) or pending.empty:
            continue
        weight = recency_weights(train.date, refit, half_life)
        head = fit_weighted_subscores(
            embeddings[train.embedding_row], train.net_return.to_numpy(), weight
        )
        scores = predict_subscores(head, embeddings[pending.embedding_row])
        result.loc[pending.index, ["sequence_return", "sequence_profit", "sequence_downside"]] = (
            scores
        )
        result.loc[pending.index, "head_refit_date"] = refit
        torch.save(head.state_dict(), directory / f"{refit:%Y%m%d}.pt")
        audit.append(
            {
                "refit_date": refit,
                "train_start": refit - pd.DateOffset(months=months),
                "train_rows": len(train),
                "effective_rows": float(weight.sum() ** 2 / (weight**2).sum()),
                "max_label_date": train.exit_date.max(),
                "prediction_start": pending.date.min(),
                "prediction_end": pending.date.max(),
            }
        )
    return result, audit


def calibrate_weekly(frame, calendar, directory):
    result = frame.copy()
    result["calibrated"] = np.nan
    directory.mkdir(parents=True)
    dates = calendar[(calendar >= pd.Timestamp("2025-01-01")) & (calendar <= result.date.max())]
    audit = []
    for refit, segment in refit_segments(dates, "weekly"):
        train = recent_window(frame, refit, 3).dropna(subset=FEATURES)
        pending = frame[frame.date.isin(segment)].dropna(subset=FEATURES)
        if not enough(train, 60) or pending.empty:
            continue
        if (train.head_refit_date > train.date).any():
            raise ValueError("calibration contains future-fit head")
        model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000))
        weight = recency_weights(train.date, refit, 90)
        model.fit(train[FEATURES], train.net_return.gt(0), logisticregression__sample_weight=weight)
        result.loc[pending.index, "calibrated"] = model.predict_proba(pending[FEATURES])[:, 1]
        joblib.dump(model, directory / f"{refit:%Y%m%d}.joblib")
        audit.append(
            {
                "refit_date": refit,
                "train_rows": len(train),
                "max_label_date": train.exit_date.max(),
                "max_training_prediction_date": train.date.max(),
                "prediction_start": pending.date.min(),
            }
        )
    return result, audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    started = time.time()
    tracked = [
        Path(__file__),
        ROOT / "src/quant/research/trend_research.py",
        ROOT / "src/quant/research/kronos_multiscale.py",
    ]
    save_json(
        out / "protocol.json",
        {
            "source": str(source),
            "code_hashes": {str(p): sha256(p) for p in tracked},
            "input_hashes": {
                p: sha256(source / p)
                for p in [
                    "embeddings.npy",
                    "left_scored_events.parquet",
                    "right_scored_events.parquet",
                ]
            },
            "grid": {"months": [6, 12], "half_life_days": [None, 90]},
            "head": "monthly,20 epochs,random seed20261021",
            "calibration": "weekly; last3 months matured time-out monthly head predictions; half-life90days",
            "selection": "2025H2 only, 2026 already-observed robustness check",
            "encoder": "frozen",
        },
    )
    histories, calendar, hashes = capital_inputs(source)
    save_json(out / "capital_sources.json", hashes)
    embeddings = np.load(source / "embeddings.npy", mmap_mode="r")
    results = []
    audits = []
    forecasts = {}
    gates = []
    periods = {
        "development": ("2025-04-01", "2025-07-01"),
        "selection": ("2025-07-01", "2026-01-01"),
        "test": ("2026-01-01", "2026-09-01"),
    }
    for side in ["left", "right"]:
        frame = pd.read_parquet(source / f"{side}_scored_events.parquet")
        forecasts[side] = {
            "baseline": (frame, "baseline_score"),
            "frozen_sequence": (frame, "sequence_profit"),
            "frozen_fusion": (frame, "fusion_score"),
        }
        for months in [6, 12]:
            for half_life in [None, 90]:
                name = f"m{months}_" + ("uniform" if half_life is None else "decay90")
                pred, audit = train_predictions(
                    frame, embeddings, calendar, months, half_life, out / "models" / side / name
                )
                forecasts[side][name] = (pred, "sequence_profit")
                audits.extend(
                    [{"side": side, "variant": name, "layer": "head", **row} for row in audit]
                )
                if half_life is not None:
                    pred, cal_audit = calibrate_weekly(
                        pred, calendar, out / "models" / side / f"{name}_weekcal"
                    )
                    forecasts[side][f"{name}_weekcal"] = (pred, "calibrated")
                    audits.extend(
                        [
                            {"side": side, "variant": name, "layer": "calibrator", **row}
                            for row in cal_audit
                        ]
                    )
                pred.to_parquet(out / f"{side}_{name}_predictions.parquet", index=False)
                print(side, name, "complete", flush=True)
        for name, (pred, column) in list(forecasts[side].items()):
            if name in ["baseline", "frozen_sequence", "frozen_fusion"]:
                continue
            selection = pred[pred.date.ge("2025-07-01") & pred.exit_date.lt("2026-01-01")].dropna(
                subset=[column]
            )
            threshold, search = choose_confidence_gate(selection, column, minimum=100)
            gates.append({"side": side, "variant": name, "threshold": threshold, "search": search})
            gated = pred.copy()
            gated[column] = (
                gated[column].where(gated[column].ge(threshold))
                if threshold is not None
                else np.nan
            )
            forecasts[side][name + "_cash"] = (gated, column)
        for period, (start, end) in periods.items():
            test = frame[frame.date.ge(start) & frame.exit_date.lt(end)]
            active = calendar[
                (calendar >= pd.Timestamp(start)) & (calendar <= test.exit_date.max())
            ]
            baseline = None
            for name, (pred, column) in forecasts[side].items():
                if name.endswith("_cash") and period != "test":
                    continue
                part = pred[pred.date.ge(start) & pred.exit_date.lt(end)].dropna(subset=[column])
                picks = top_picks(part, column).copy()
                picks["execution_score"] = picks[column]
                curve, trades, metrics = capital_backtest(
                    picks, histories, active, slots=20, cost=0.002
                )
                if name == "baseline":
                    baseline = curve
                results.append(
                    {
                        "side": side,
                        "period": period,
                        "variant": name,
                        **metrics,
                        "prediction": prediction_metrics(part, column),
                        "gain": paired_block_gain(baseline, curve, repetitions=500)
                        if name != "baseline"
                        else None,
                    }
                )
                curve.to_csv(out / f"{side}_{period}_{name}_equity.csv", index=False)
                trades.to_parquet(out / f"{side}_{period}_{name}_trades.parquet", index=False)
    save_json(out / "results.json", results)
    save_json(out / "cash_selection.json", gates)
    pd.DataFrame(audits).to_csv(out / "refit_audit.csv", index=False)
    choices = []
    for side in ["left", "right"]:
        candidates = [
            r
            for r in results
            if r["side"] == side
            and r["period"] == "selection"
            and r["variant"].startswith("m")
            and not r["variant"].endswith("_cash")
        ]
        best = max(candidates, key=lambda r: (r["total_return"], -abs(r["max_drawdown"])))
        choices.append(
            {
                "side": side,
                "chosen_before_2026": best["variant"],
                "selection_return": best["total_return"],
                "selection_win_rate": best["win_rate"],
                "admitted": bool(
                    best["trades"] >= 100
                    and best["total_return"] > 0
                    and (best["profit_factor"] or 0) > 1
                ),
            }
        )
    save_json(out / "selection_decision.json", choices)
    lines = [
        "# 6/12月窗口与近期加权滚动研究",
        "",
        "冻结原编码器；按月更新子分、加权版本另按周校准。2025下半年选择版本，2026是已观察区间复核。",
        "",
        "|侧|版本|2025H2收益|2026收益|2026胜率|2026回撤|2026敞口|",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in [r for r in results if r["period"] == "test"]:
        prior = next(
            (
                r
                for r in results
                if r["side"] == row["side"]
                and r["period"] == "selection"
                and r["variant"] == row["variant"]
            ),
            None,
        )
        val = f"{prior['total_return']:.2%}" if prior else "—"
        win = f"{row['win_rate']:.1%}" if row["win_rate"] is not None else "—"
        lines.append(
            f"|{row['side']}|{row['variant']}|{val}|{row['total_return']:.2%}|{win}|{row['max_drawdown']:.2%}|{row['mean_exposure']:.1%}|"
        )
    lines += ["", "2025H2选择："] + [
        f"- {c['side']}: {c['chosen_before_2026']}，验证期准入={c['admitted']}。" for c in choices
    ]
    lines += [
        "",
        "现金阈值由2025H2固定；全现金不代表100%胜率。阈值筛选与版本筛选有多重比较，尚未做生产准入。",
        "训练窗口按自然月，权重半衰期按自然日；周校准只用历史各期head的时间外分数，避免拿最终head回填历史。",
        "研究范围仍为原256只抽样股票，不是全市场；未重训或微调编码器。",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    save_json(
        out / "complete.json",
        {"status": "research_complete_not_production", "seconds": time.time() - started},
    )
    print((out / "report.md").read_text(), flush=True)


if __name__ == "__main__":
    main()
