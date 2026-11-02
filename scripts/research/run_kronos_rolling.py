#!/usr/bin/env python
"""Quarter-window monthly/weekly refits and matched portfolio comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import brier_score_loss, roc_auc_score

from quant.research.kronos_rolling import rolling_predictions
from quant.research.sequence_strategy_validation import (
    capital_backtest,
    paired_block_gain,
    top_picks,
)

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def save_json(path, data):
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str, allow_nan=False) + "\n"
    )


def capital_inputs(source):
    config = json.loads((source / "protocol.json").read_text())["config"]
    symbols = json.loads((source / "coverage.json").read_text())["sampled_symbols"]
    raw_path = ROOT / config["raw_bars"]
    raw = pd.read_parquet(raw_path, filters=[("symbol", "in", symbols)])
    raw["date"] = pd.to_datetime(raw.date).dt.normalize()
    calendar = pd.DatetimeIndex(
        pd.read_parquet(raw_path, columns=["date"]).date.unique()
    ).sort_values()
    limits = []
    hashes = {str(raw_path): sha256(raw_path)}
    for path in sorted((ROOT / config["limits_dir"]).glob("*.parquet")):
        part = pd.read_parquet(path)
        part = part[part.ts_code.isin(symbols)].copy()
        if not len(part):
            continue
        part = part.rename(columns={"ts_code": "symbol"})
        part["date"] = pd.to_datetime(part.trade_date.astype(str), format="%Y%m%d")
        limits.append(part[["symbol", "date", "up_limit", "down_limit"]])
        hashes[str(path)] = sha256(path)
    limits = pd.concat(limits).drop_duplicates(["symbol", "date"])
    raw = raw.merge(limits, on=["symbol", "date"], how="left", validate="one_to_one")
    raw = raw[raw[["open", "high", "low", "close", "pre_close"]].gt(0).all(axis=1)]
    histories = {}
    for symbol, group in raw.groupby("symbol"):
        group = group.sort_values("date")
        valid = group.valid_bar.fillna(False).astype(bool) & group.vol.gt(0)
        price = np.exp(np.log(group.close / group.pre_close).cumsum()) * 100
        histories[symbol] = pd.DataFrame(
            {
                "date": group.date,
                "close": price,
                "buy_ok": valid & group.close.lt(group.up_limit - 1e-8),
                "sell_ok": valid & group.close.gt(group.down_limit + 1e-8),
            }
        )
    return histories, calendar, hashes


def prediction_metrics(frame, score):
    part = frame.dropna(subset=[score])
    if part.empty:
        return {"rows": 0, "auc": None, "daily_rank_ic": None, "brier": None, "coverage": 0.0}
    correlations = []
    for _, day in part.groupby("date"):
        if len(day) >= 5 and day[score].nunique() > 1 and day.net_return.nunique() > 1:
            correlations.append(spearmanr(day[score], day.net_return).statistic)
    target = part.net_return.gt(0)
    return {
        "rows": len(part),
        "auc": float(roc_auc_score(target, part[score])) if target.nunique() == 2 else None,
        "daily_rank_ic": float(np.mean(correlations)) if correlations else None,
        "brier": float(brier_score_loss(target, part[score])),
        "coverage": len(part) / len(frame),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    started = time.time()
    tracked = [
        "embeddings.npy",
        "embedding_keys.parquet",
        "encoder.pt",
        "left_scored_events.parquet",
        "right_scored_events.parquet",
    ]
    protocol = {
        "source": str(source),
        "source_hashes": {p: sha256(source / p) for p in tracked},
        "code_hashes": {
            p: sha256(ROOT / p)
            for p in [
                "scripts/research/run_kronos_rolling.py",
                "src/quant/research/kronos_rolling.py",
                "src/quant/research/kronos_multiscale.py",
                "src/quant/research/sequence_strategy_validation.py",
            ]
        },
        "encoder": "frozen original encoder; no pretraining rerun",
        "window": "3 calendar months; signal date in window and exit strictly before refit",
        "refit": "before first trading session of each month/week; freshly initialized heads; 20 epochs",
        "joint_stacking": "auxiliary head fitted first 2 months, last month time-out fusion training; final head refit entire mature quarter",
        "cash_selection": "extra chronological holdout in last month; exact holdout gate model retained; final head has more training than auxiliary",
        "fusion_inputs": [
            "within-day baseline percentile",
            "sequence_return",
            "sequence_profit",
            "sequence_downside",
        ],
        "test": "2026 observed historical period, not new sealed confirmation",
        "seed": 20261021,
        "epochs": 20,
    }
    save_json(output / "protocol.json", protocol)
    embeddings = np.load(source / "embeddings.npy", mmap_mode="r")
    histories, calendar, hashes = capital_inputs(source)
    save_json(output / "capital_sources.json", hashes)
    results = []
    audits = []
    age_metrics = []
    months = []
    for side in ["left", "right"]:
        frame = pd.read_parquet(source / f"{side}_scored_events.parquet")
        test = frame[frame.date.ge("2026-01-01")].copy()
        inference_dates = calendar[
            (calendar >= pd.Timestamp("2026-01-01")) & (calendar <= test.date.max())
        ]
        active_calendar = calendar[
            (calendar >= pd.Timestamp("2026-01-01")) & (calendar <= test.exit_date.max())
        ]
        variants = {
            name: (test, score)
            for name, score in [
                ("baseline", "baseline_score"),
                ("frozen_sequence", "sequence_profit"),
                ("frozen_fusion", "fusion_score"),
            ]
        }
        for frequency in ["monthly", "weekly"]:
            predicted, audit = rolling_predictions(
                frame, embeddings, inference_dates, frequency, output / "models" / side / frequency
            )
            predicted.to_parquet(output / f"{side}_{frequency}_predictions.parquet", index=False)
            audits.extend([{"side": side, "frequency": frequency, **row} for row in audit])
            for variant, column in [
                ("fusion_only", "fusion_only"),
                ("sequence", "rolling_sequence"),
                ("joint", "rolling_joint"),
                ("joint_cash", "rolling_cash"),
            ]:
                variants[f"{frequency}_{variant}"] = (predicted, column)
            predicted["age_sessions"] = [
                int(inference_dates.searchsorted(d) - inference_dates.searchsorted(r))
                if pd.notna(r)
                else -1
                for d, r in zip(predicted.date, predicted.refit_date)
            ]
            for age, part in predicted[predicted.age_sessions.ge(0)].groupby(
                pd.cut(predicted.age_sessions, [-1, 4, 9, 100], labels=["0-4", "5-9", "10+"])
            ):
                if len(part):
                    age_metrics.append(
                        {
                            "side": side,
                            "frequency": frequency,
                            "age_sessions": str(age),
                            **prediction_metrics(part, "rolling_sequence"),
                        }
                    )
        baseline_curve = None
        for name, (predicted, column) in variants.items():
            eligible = predicted.dropna(subset=[column])
            picks = top_picks(eligible, column, 20).copy()
            picks["execution_score"] = picks[column]
            curve, trades, metrics = capital_backtest(
                picks, histories, active_calendar, slots=20, cost=0.002
            )
            if name == "baseline":
                baseline_curve = curve
                old = pd.read_csv(source / f"{side}_baseline_equity.csv", parse_dates=["date"])
                matched = old.merge(
                    curve, on="date", validate="one_to_one", suffixes=("_old", "_new")
                )
                error = float((matched.equity_old - matched.equity_new).abs().max())
                if len(matched) != len(old) or error > 1e-8:
                    raise ValueError(f"capital baseline mismatch: {side} error={error}")
                save_json(
                    output / f"{side}_capital_parity.json",
                    {"max_equity_error": error, "days": len(matched)},
                )
            curve.to_csv(output / f"{side}_{name}_equity.csv", index=False)
            trades.to_parquet(output / f"{side}_{name}_trades.parquet", index=False)
            result = {
                "side": side,
                "variant": name,
                **metrics,
                "prediction": prediction_metrics(predicted, column),
                "increment_vs_baseline": paired_block_gain(baseline_curve, curve)
                if name != "baseline"
                else None,
            }
            results.append(result)
            for month, part in curve.groupby(curve.date.dt.to_period("M")):
                previous = curve.loc[curve.date < part.date.min(), "equity"]
                initial = float(previous.iloc[-1]) if len(previous) else 1.0
                months.append(
                    {
                        "side": side,
                        "variant": name,
                        "month": str(month),
                        "return": float(part.equity.iloc[-1] / initial - 1),
                    }
                )
        # Direct comparisons isolate cadence within otherwise identical rolling designs.
        for design in ["fusion_only", "sequence", "joint", "joint_cash"]:
            monthly = pd.read_csv(
                output / f"{side}_monthly_{design}_equity.csv", parse_dates=["date"]
            )
            weekly = pd.read_csv(
                output / f"{side}_weekly_{design}_equity.csv", parse_dates=["date"]
            )
            save_json(
                output / f"{side}_{design}_weekly_vs_monthly.json",
                paired_block_gain(monthly, weekly),
            )
    save_json(output / "results.json", results)
    pd.DataFrame(audits).to_csv(output / "refit_audit.csv", index=False)
    pd.DataFrame(months).to_csv(output / "monthly_returns.csv", index=False)
    pd.DataFrame(age_metrics).to_csv(output / "age_metrics.csv", index=False)
    lines = [
        "# 近三个月滚动训练验证",
        "",
        "固定原Kronos编码器；滚动更新子分/融合层。2026为已观察的历史复核，使用原256只抽样股票的匹配事件。",
        "每次仅纳入重训前已经成熟的5日标签；相同20仓位上限、交易限制和往返20bps费用。",
        "",
        "|侧|版本|已平仓|胜率|组合收益|最大回撤|平均敞口|AUC|",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        win = f"{row['win_rate']:.1%}" if row["win_rate"] is not None else "—"
        auc = f"{row['prediction']['auc']:.3f}" if row["prediction"]["auc"] is not None else "—"
        lines.append(
            f"|{row['side']}|{row['variant']}|{row['trades']}|{win}|{row['total_return']:.2%}|{row['max_drawdown']:.2%}|{row['mean_exposure']:.1%}|{auc}|"
        )
    lines += [
        "",
        "频率对照：正数表示周度收益高于月度，区间为五日配对区块bootstrap的95%区间，未校正多重比较。",
    ]
    for side in ["left", "right"]:
        for design in ["fusion_only", "sequence", "joint", "joint_cash"]:
            gain = json.loads((output / f"{side}_{design}_weekly_vs_monthly.json").read_text())
            lo, hi = gain["paired_block_95ci"]
            lines.append(f"- {side} {design}: 周减月{gain['gain']:.2%}，区间[{lo:.2%}, {hi:.2%}]。")
    lines += [
        "",
        "这些结果不包含滚动微调编码器；滚动层无法修复冻结表征的全部缺陷。",
        "新融合采用同日原分数分位数，因此对旧冻结融合的差异还包括输入刻度；周/月之间使用完全相同设计。",
        "联合模型的最终head使用完整三个月，辅助head只使用早期样本，因此融合训练和最终head版本存在分布差别；现金阈值使用额外末月时间切分。",
        "原始baseline曲线与前次逐日账户核对通过，误差<1e-8。原模型good_path5分数的Brier不作为净盈利概率校准检验。",
        "不按2026成绩选最佳频率或自动发布生产；空仓版减小敞口也可能改善收益，需要与主动预测能力区分。",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    save_json(
        output / "complete.json",
        {"status": "research_complete_not_production", "seconds": time.time() - started},
    )
    print((output / "report.md").read_text(), flush=True)


if __name__ == "__main__":
    main()
