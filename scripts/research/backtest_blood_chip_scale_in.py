#!/usr/bin/env python3
"""Compare one-shot and staged entries for the frozen blood-chip signal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant.research.blood_chip import (
    BloodChipBacktestConfig,
    load_benchmark,
    summarize_blood_chip_result,
)
from quant.research.blood_chip_scale_in import (
    DEFAULT_SCALE_IN_POLICIES,
    run_blood_chip_scale_in_backtest,
)


PERIODS = (
    ("development_2014_2019", "2014-01-01", "2019-12-31"),
    ("iteration_2020_2022", "2020-01-01", "2022-12-30"),
    ("seen_diagnostic_2023_2026", "2023-01-03", "2026-02-06"),
)
FEATURE_COLUMNS = (
    "ts_code",
    "trade_date",
    "date",
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "pct_chg",
    "vol",
    "amount",
    "adjustment_factor",
    "adjusted_open",
    "adjusted_high",
    "adjusted_low",
    "adjusted_close",
    "residual_return_3d",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features", default="data/research/blood_chip/features.parquet"
    )
    parser.add_argument(
        "--signals",
        default="reports/research/blood_chip_exhaustion/path_signals.parquet",
    )
    parser.add_argument(
        "--benchmark", default="data/raw/index_000300.SH.parquet"
    )
    parser.add_argument(
        "--output-dir", default="reports/research/blood_chip_scale_in"
    )
    return parser.parse_args()


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _stage_metrics(trades: pd.DataFrame) -> dict[str, float]:
    if trades.empty:
        return {
            "average_deployed_fraction": np.nan,
            "average_tranches_filled": np.nan,
            "second_stage_rate": np.nan,
            "third_stage_rate": np.nan,
            "capital_weighted_trade_return": np.nan,
            "capital_profit_factor": np.nan,
        }
    filled = pd.to_numeric(trades["tranches_filled"], errors="coerce")
    deployed = pd.to_numeric(trades["deployed_fraction"], errors="coerce")
    return {
        "average_deployed_fraction": float(deployed.mean()),
        "average_tranches_filled": float(filled.mean()),
        "second_stage_rate": float(filled.ge(2).mean()),
        "third_stage_rate": float(filled.ge(3).mean()),
        **_capital_metrics(trades),
    }


def _capital_metrics(trades: pd.DataFrame) -> dict[str, float]:
    if trades.empty:
        return {
            "capital_weighted_trade_return": np.nan,
            "capital_profit_factor": np.nan,
        }
    pnl = (
        pd.to_numeric(trades["exit_value"], errors="coerce")
        - pd.to_numeric(trades["fees"], errors="coerce")
        - pd.to_numeric(trades["entry_value"], errors="coerce")
    )
    invested = pd.to_numeric(trades["invested_value"], errors="coerce")
    losses = float(-pnl.loc[pnl <= 0].sum())
    return {
        "capital_weighted_trade_return": float(pnl.sum() / invested.sum()),
        "capital_profit_factor": (
            float(pnl.loc[pnl > 0].sum() / losses) if losses > 0 else np.nan
        ),
    }


def _stage_cases(trades: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in trades.groupby(
        ["policy", "period", "tranches_filled"], observed=True, sort=False
    ):
        returns = pd.to_numeric(group["net_return"], errors="coerce")
        rows.append(
            {
                "policy": keys[0],
                "period": keys[1],
                "tranches_filled": int(keys[2]),
                "trades": int(len(group)),
                "win_rate": float(returns.gt(0).mean()),
                "average_net_return": float(returns.mean()),
                "stop_rate": float(group["exit_reason"].eq("stop_loss").mean()),
                **_capital_metrics(group),
            }
        )
    return pd.DataFrame(rows)


def _decision(metrics: pd.DataFrame) -> tuple[str, list[str]]:
    indexed = metrics.set_index(["policy", "period"])
    base = indexed.loc[("one_shot", "iteration_2020_2022")]
    candidate = indexed.loc[("increasing_survival", "iteration_2020_2022")]
    development = indexed.loc[("increasing_survival", "development_2014_2019")]
    checks = {
        "迭代期总收益高于一次性建仓": candidate["total_return"] > base["total_return"],
        "迭代期资金加权盈利因子不低于一次性建仓": (
            candidate["capital_profit_factor"] >= base["capital_profit_factor"]
        ),
        "迭代期最大回撤恶化不超过 3 个百分点": (
            candidate["maximum_drawdown"] >= base["maximum_drawdown"] - 0.03
        ),
        "研发期资金加权单笔收益为正": (
            development["capital_weighted_trade_return"] > 0
        ),
    }
    passed = all(bool(value) for value in checks.values())
    return ("建议进入线上灰度" if passed else "保留研究，暂不替换线上逻辑"), [
        f"{'通过' if value else '未通过'}：{label}" for label, value in checks.items()
    ]


def _percent(value: object) -> str:
    number = pd.to_numeric(value, errors="coerce")
    return "—" if pd.isna(number) else f"{float(number):.2%}"


def _number(value: object) -> str:
    number = pd.to_numeric(value, errors="coerce")
    return "—" if pd.isna(number) else f"{float(number):.2f}"


def _render_report(metrics: pd.DataFrame, decision: str, checks: list[str]) -> str:
    indexed = metrics.set_index(["policy", "period"])
    base = indexed.loc[("one_shot", "iteration_2020_2022")]
    survival = indexed.loc[("increasing_survival", "iteration_2020_2022")]
    risk_capped = indexed.loc[
        ("increasing_survival_risk_capped", "iteration_2020_2022")
    ]
    lines = [
        "# 带血筹分批建仓回测",
        "",
        "## 结论",
        "",
        f"**{decision}。**",
        "",
        *[f"- {item}" for item in checks],
        "",
        "最终候选 `increasing_survival` 的含义是随证据增强而增加资金：20% 试仓；至少存活 5 个交易日、守住信号价 95% 且三日残差收益转正后加 30%；至少存活 10 日、重新站上信号价且残差继续为正后加 50%。`increasing_price_only` 只作为‘越跌越加’的风险对照。",
        "",
        "## 案例迭代发现",
        "",
        f"- 原吸筹区确认第二笔过早，三段完成率只有 {_percent(indexed.loc[('increasing_confirmed', 'iteration_2020_2022'), 'third_stage_rate'])}，平均资金部署仅 {_percent(indexed.loc[('increasing_confirmed', 'iteration_2020_2022'), 'average_deployed_fraction'])}。",
        f"- 生存确认把三段完成率提高到 {_percent(survival['third_stage_rate'])}，总收益由 {_percent(base['total_return'])} 提高至 {_percent(survival['total_return'])}，最大回撤由 {_percent(base['maximum_drawdown'])} 收窄至 {_percent(survival['maximum_drawdown'])}。",
        f"- 每次加仓后机械抬高止损的对照被否决：迭代期总收益只有 {_percent(risk_capped['total_return'])}，最大回撤 {_percent(risk_capped['maximum_drawdown'])}；说明确认后的正常回踩需要空间。",
        "- 普通按笔盈利因子会把 20% 的失败试仓和 100% 的确认赢家等权，无法反映递增加仓效果，因此上线判断使用实际资金盈亏计算的资金加权盈利因子。",
        "",
        "## 汇总指标",
        "",
        "| 区间 | 方案 | 交易数 | 胜率 | 资金加权单笔 | 资金加权PF | 总收益 | 最大回撤 | 平均部署 | 三段完成率 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in metrics.iterrows():
        lines.append(
            "| {period} | {policy} | {trades} | {win} | {avg} | {pf} | {total} | {dd} | {deploy} | {third} |".format(
                period=row["period"],
                policy=row["policy"],
                trades=int(row["trades"]),
                win=_percent(row["win_rate"]),
                avg=_percent(row["capital_weighted_trade_return"]),
                pf=_number(row["capital_profit_factor"]),
                total=_percent(row["total_return"]),
                dd=_percent(row["maximum_drawdown"]),
                deploy=_percent(row["average_deployed_fraction"]),
                third=_percent(row["third_stage_rate"]),
            )
        )
    lines.extend(
        [
            "",
            "## 口径",
            "",
            "- 选股不变：事件低点反弹不超过 15%，同日按 60 日波动率由低到高选择。",
            "- 最多 10 个持仓，首笔起算 120 个交易日，首笔成交价下方 10% 固定止损，允许新事件再次买入。",
            "- 加仓条件仅读取前一交易日收盘数据，下一交易日开盘成交，并计入佣金、印花税、过户费与滑点。",
            "- 2023–2026 为已见诊断区间，不参与上线判断。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("loading selected feature columns", flush=True)
    daily = pd.read_parquet(args.features, columns=list(FEATURE_COLUMNS))
    signals = pd.read_parquet(args.signals)
    signals = signals.loc[
        pd.to_numeric(signals["rebound_from_event_low"], errors="coerce").le(0.15)
    ].copy()
    signals["signal_score"] = -pd.to_numeric(
        signals["volatility_60d"], errors="coerce"
    )
    benchmark = load_benchmark(args.benchmark, "20130104", "20260807")
    config = BloodChipBacktestConfig(
        maximum_positions=10,
        stop_loss=0.10,
        maximum_holding_days=120,
        allow_reentry_after_stop=True,
        require_new_event_for_reentry=True,
    )
    metric_rows: list[dict[str, Any]] = []
    trade_frames: list[pd.DataFrame] = []
    experiments = [
        (name, policy, config.maximum_positions, 1.0 / config.maximum_positions)
        for name, policy in DEFAULT_SCALE_IN_POLICIES.items()
    ]
    experiments.extend(
        [
            (
                "increasing_confirmed_redeploy",
                DEFAULT_SCALE_IN_POLICIES["increasing_confirmed"],
                30,
                0.10,
            ),
            (
                "increasing_price_only_redeploy",
                DEFAULT_SCALE_IN_POLICIES["increasing_price_only"],
                20,
                0.10,
            ),
        ]
    )
    for policy_name, policy, open_plan_limit, target_fraction in experiments:
        for period, start, end in PERIODS:
            print(f"running {policy_name} / {period}", flush=True)
            result = run_blood_chip_scale_in_backtest(
                daily,
                signals,
                config,
                policy,
                start,
                end,
                maximum_open_plans=open_plan_limit,
                target_position_fraction=target_fraction,
            )
            metrics = summarize_blood_chip_result(result, benchmark)
            metrics.update(_stage_metrics(result.trades))
            metrics.update(
                {
                    "policy": policy_name,
                    "period": period,
                    "entry_start": start,
                    "entry_end": end,
                    "maximum_open_plans": open_plan_limit,
                    "target_position_fraction": target_fraction,
                }
            )
            metric_rows.append(metrics)
            if not result.trades.empty:
                trades = result.trades.copy()
                trades["policy"] = policy_name
                trades["period"] = period
                trade_frames.append(trades)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    (output_dir / "metrics.json").write_text(
        json.dumps(_json_value({"metrics": metric_rows}), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    trades = pd.concat(trade_frames, ignore_index=True, sort=False)
    trades.to_parquet(output_dir / "trades.parquet", index=False, compression="zstd")
    stage_cases = _stage_cases(trades)
    stage_cases.to_csv(output_dir / "stage_cases.csv", index=False)
    representative_frames = []
    for _, group in trades.groupby(["policy", "period"], observed=True, sort=False):
        representative_frames.append(
            pd.concat(
                [group.nsmallest(3, "net_return"), group.nlargest(3, "net_return")]
            )
        )
    representatives = pd.concat(representative_frames, ignore_index=True, sort=False)
    representatives.to_csv(output_dir / "representative_cases.csv", index=False)
    decision, checks = _decision(metrics)
    (output_dir / "report.md").write_text(
        _render_report(metrics, decision, checks), encoding="utf-8"
    )
    print(
        metrics[
            [
                "policy",
                "period",
                "trades",
                "win_rate",
                "average_net_return",
                "profit_factor",
                "total_return",
                "maximum_drawdown",
                "average_deployed_fraction",
                "third_stage_rate",
            ]
        ].to_string(index=False),
        flush=True,
    )
    print(decision, flush=True)


if __name__ == "__main__":
    main()
