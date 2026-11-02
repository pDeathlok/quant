#!/usr/bin/env python3
"""Reproducible multi-round BYD MWU research; holdout never selects parameters."""

from __future__ import annotations

import argparse
import hashlib
import html
import itertools
import json
import math
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from quant.research.byd_mwu import (
    ALL_EXPERTS, BASE_EXPERTS, RULE_EXPERTS, ROOT, ExecutionConfig, MixerConfig,
    build_experts, evaluate_experts, load_market, metrics, selection_score, simulate,
)

DEFAULT_OUTPUT = ROOT / "reports/research/byd_mwu_20261012"
GROUPS = (
    BASE_EXPERTS,
    BASE_EXPERTS + ("corrected_score", "cash"),
    RULE_EXPERTS,
    ("original_score", "original_range", "corrected_score", "cash", "trend_120", "pullback", "vol_budget"),
    ("corrected_score", "cash", "trend_60", "trend_120", "pullback", "range_reversal", "vol_budget"),
    RULE_EXPERTS + ("ridge_5d",),
    RULE_EXPERTS + ("tree_5d",),
    ALL_EXPERTS,
)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def key(config: MixerConfig) -> str:
    return json.dumps(config.to_dict(), sort_keys=True)


def candidates(round_number: int, current: MixerConfig) -> list[MixerConfig]:
    if round_number == 1:
        return [replace(current, eta=value) for value in (0.03, 0.10, 0.30)]
    if round_number == 2:
        return [replace(current, eta=eta, share=share, forget=forget)
                for eta, share, forget in itertools.product((0.03, 0.10, 0.30), (0.01, 0.05), (0.97, 0.99, 1.0))]
    if round_number == 3:
        return [replace(current, experts=group, eta=eta)
                for group, eta in itertools.product(GROUPS[1:5], (0.03, 0.10, 0.30))]
    if round_number == 4:
        return [replace(current, experts=group, eta=eta)
                for group, eta in itertools.product(GROUPS[5:], (0.03, 0.10, 0.30))]
    if round_number == 5:
        return [replace(current, deadband=band, max_step=step)
                for band, step in itertools.product((0.0, 0.03, 0.06, 0.10), (0.10, 0.20, 1.0))]
    if round_number == 6:
        return [replace(current, downside=penalty, vol_target=vol)
                for penalty, vol in itertools.product((0.0, 0.5, 1.0), (0.0, 0.20, 0.30))]
    neighbors = [current]
    dimensions = {
        "eta": (0.01, 0.03, 0.10, 0.30, 0.60), "share": (0.0, 0.01, 0.05, 0.10),
        "forget": (0.95, 0.97, 0.99, 1.0), "deadband": (0.0, 0.03, 0.06, 0.10),
        "max_step": (0.10, 0.20, 1.0), "downside": (0.0, 0.5, 1.0),
        "vol_target": (0.0, 0.20, 0.30), "experts": GROUPS,
    }
    for name, values in dimensions.items():
        neighbors.extend(replace(current, **{name: value}) for value in values)
    return list({key(item): item for item in neighbors}.values())


def chart_svg(ledgers: dict[str, pd.DataFrame], path: Path, holdout: bool = False) -> None:
    """Standalone deterministic chart; no plotting-library dependency."""
    colors = ("#0f766e", "#2563eb", "#d97706", "#9333ea", "#64748b")
    w, h, left, right, top, bottom = 1100, 510, 80, 25, 60, 75
    curves = {}
    dates = None
    for name, ledger in ledgers.items():
        frame = ledger[ledger.date.ge("2024-01-01")] if holdout else ledger
        if frame.empty:
            continue
        curves[name] = np.cumprod(1 + frame["return"].to_numpy())
        dates = frame.date
    max_y = max(float(v.max()) for v in curves.values()) * 1.05
    min_y = min(0.8, min(float(v.min()) for v in curves.values()) * 0.95)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
             '<rect width="100%" height="100%" fill="#f8fafc"/>',
             '<style>text{font-family:Arial,sans-serif;fill:#334155;font-size:13px}</style>',
             f'<text x="{left}" y="30" style="font-size:22px;font-weight:bold">BYD | {"Frozen evaluation 2024+" if holdout else "2015+ research"} | net equity</text>']
    for value in np.linspace(min_y, max_y, 6):
        y = top + (max_y - value) / (max_y - min_y) * (h - top - bottom)
        parts += [f'<line x1="{left}" x2="{w-right}" y1="{y:.1f}" y2="{y:.1f}" stroke="#e2e8f0"/>',
                  f'<text x="15" y="{y+4:.1f}">{value:.2f}x</text>']
    for j, (name, values) in enumerate(curves.items()):
        x = np.linspace(left, w - right, len(values))
        y = top + (max_y - values) / (max_y - min_y) * (h - top - bottom)
        points = " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(x, y))
        color = colors[j % len(colors)]
        width = 3 if name == "MWU" else 1.5
        parts += [f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="{width}"/>',
                  f'<text x="{left+j*195}" y="{h-20}" style="fill:{color}">{html.escape(name)}</text>']
    for fraction in (0, 0.25, 0.5, 0.75, 1):
        i = int(fraction * (len(dates) - 1))
        x = left + fraction * (w - right - left)
        parts.append(f'<text x="{x-25:.1f}" y="{h-bottom+25}">{dates.iloc[i]:%Y-%m}</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def table(frame: pd.DataFrame) -> str:
    columns = ("name", "total_return", "cagr", "sharpe", "max_drawdown", "average_position", "orders")
    lines = ["| 名称 | 累计净收益 | 年化收益 | Sharpe | 最大回撤 | 平均仓位 | 订单数 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for _, row in frame.iterrows():
        vals = [str(row["name"])]
        for column in columns[1:]:
            value = row[column]
            vals.append(f"{value:.2f}" if column == "sharpe" else str(int(value)) if column == "orders" else f"{value:.2%}")
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify-frozen", type=Path, help="Assert a correctness replay retains the original frozen configuration")
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    print("Loading raw canonical BYD history", flush=True)
    market, signal = load_market()
    index_calendar = pd.read_parquet(ROOT / "data/raw/index_000300.SH.parquet")
    index_dates = pd.to_datetime(index_calendar.trade_date.astype(str), format="%Y%m%d")
    missing_calendar = index_dates[(index_dates >= market.date.min()) & (index_dates <= market.date.max()) & ~index_dates.isin(market.date)]
    market.to_parquet(out / "market_snapshot.parquet", index=False)
    corporate = market.loc[(market.cash_dividend > 0) | (market["split"] != 1), ["date", "split", "cash_dividend", "bonus_shares"]]
    corporate.to_csv(out / "corporate_actions.csv", index=False)
    digest = hashlib.sha256((out / "market_snapshot.parquet").read_bytes()).hexdigest()
    dev_mask = market.date.le("2023-12-31")
    dev_market, dev_signal = market.loc[dev_mask].reset_index(drop=True), signal.loc[dev_mask].reset_index(drop=True)
    print(f"Development raw bars={len(dev_market)}. Holdout is excluded from search.", flush=True)
    dev_targets, dev_training = build_experts(dev_signal)
    dev_training.to_csv(out / "development_training_audit.csv", index=False)
    execution = ExecutionConfig()
    dev_returns, dev_ledgers = evaluate_experts(dev_market, dev_targets, execution)
    baseline_scores = {name: selection_score(dev_ledgers[name]) for name in BASE_EXPERTS[:2]}
    cache = {}
    trials = []
    rounds = []

    def evaluate(config: MixerConfig, round_number: int) -> float:
        identity = key(config)
        if identity not in cache:
            ledger = simulate(dev_market, dev_targets, execution, config, dev_returns)
            cache[identity] = selection_score(ledger)
            if len(cache) % 10 == 0:
                print(f"  evaluated {len(cache)} unique configurations", flush=True)
        trials.append({"round": round_number, "score": cache[identity], **config.to_dict()})
        return cache[identity]

    current = MixerConfig(eta=0.0)
    best_score = evaluate(current, 0)
    equal_initial_score = best_score
    rounds.append({"round": 0, "score": best_score, "improvement": 0.0, "configuration": current.to_dict(), "candidates": 1})
    stagnant = 0
    converged = False
    for number in range(1, 13):
        options = candidates(number, current)
        old_score = best_score
        for candidate in options:
            score = evaluate(candidate, number)
            if score > best_score + 1e-10:
                current, best_score = candidate, score
        gain = best_score - old_score
        rounds.append({"round": number, "score": best_score, "improvement": gain, "configuration": current.to_dict(), "candidates": len(options)})
        write_json(out / "rounds.json", rounds)
        pd.DataFrame(trials).to_csv(out / "trials.csv", index=False)
        print(f"Round {number}: score={best_score:.4f}, gain={gain:.4f}, experts={len(current.experts)}, candidates={len(options)}", flush=True)
        if number >= 7:
            stagnant = stagnant + 1 if gain <= 0.01 else 0
            if stagnant >= 2:
                converged = True
                break

    if args.verify_frozen:
        expected = json.loads(args.verify_frozen.read_text(encoding="utf-8"))
        if json.loads(key(current)) != expected["config"]:
            raise RuntimeError("Correctness replay changed frozen selection; do not reuse viewed holdout for tuning")
    frozen = {"config": current.to_dict(), "development_score": best_score,
              "converged": converged, "criterion": "two consecutive coordinate rounds improve <=0.01",
              "frozen_at": datetime.now().isoformat(), "holdout_used_for_selection": False,
              "unique_configurations": len(cache), "baseline_scores": baseline_scores,
              "equal_initial_score": equal_initial_score}
    write_json(out / "frozen_config.json", frozen)
    print("Configuration frozen; now computing final 2024+ evaluation", flush=True)
    targets, training = build_experts(signal)
    # Prefix replay includes retraining and rule generation, not just the mixer.
    pd.testing.assert_frame_equal(targets.loc[:len(dev_targets)-1], dev_targets)
    training.to_csv(out / "training_audit.csv", index=False)
    targets.assign(date=market.date).to_parquet(out / "expert_targets.parquet", index=False)
    expert_returns, expert_ledgers = evaluate_experts(market, targets, execution)
    expert_returns.assign(date=market.date).to_parquet(out / "expert_net_returns.parquet", index=False)
    selected = simulate(market, targets, execution, current, expert_returns)
    replay_score = selection_score(selected[selected.date.le("2023-12-31")])
    if abs(replay_score - best_score) > 1e-10:
        raise RuntimeError("Full replay changed development selection score")
    equal_config = replace(current, eta=0.0, forget=1.0, share=0.0)
    equal_matched = simulate(market, targets, execution, equal_config, expert_returns)
    original = expert_ledgers["original_score"]
    original_range = expert_ledgers["original_range"]
    buy_hold = simulate(market, targets, execution, fixed_target=np.ones(len(market)))
    ledgers = {"MWU": selected, "equal_matched": equal_matched, "original_score": original,
               "original_range": original_range, "buy_hold": buy_hold}
    baseline_mwu = simulate(market, targets, execution, MixerConfig(), expert_returns)
    ledgers["basic_MWU"] = baseline_mwu
    equal_core = simulate(market, targets, execution, MixerConfig(eta=0), expert_returns)
    ledgers["equal_core"] = equal_core
    range_matched = simulate(market, targets, execution, replace(current, experts=("original_range",)), expert_returns)
    ledgers["range_matched_controls"] = range_matched
    ledgers["constant_one_third"] = simulate(market, targets, execution, fixed_target=np.full(len(market), 1 / 3))
    summaries = []
    yearly = []
    for name, ledger in ledgers.items():
        ledger.to_csv(out / f"ledger_{name}.csv", index=False)
        for period, start, end in (
            ("development", "2018-01-01", "2023-12-31"),
            ("holdout", "2024-01-01", "2099-12-31"),
            ("full", "2015-01-01", "2099-12-31"),
        ):
            summaries.append({"period": period, "name": name, **metrics(ledger[ledger.date.between(start, end)])})
        for year, frame in ledger.groupby(ledger.date.dt.year):
            yearly.append({"name": name, "year": int(year), **metrics(frame)})
    summary = pd.DataFrame(summaries)
    summary.to_csv(out / "metrics.csv", index=False)
    pd.DataFrame([{"name": name, "score": selection_score(ledger)} for name, ledger in ledgers.items()]).to_csv(out / "development_objective.csv", index=False)
    pd.DataFrame(yearly).to_csv(out / "yearly_metrics.csv", index=False)

    # Post-freeze diagnostics: never promote a variant using holdout performance.
    stress_execution = replace(execution, slippage=execution.slippage * 2)
    stress_returns, _ = evaluate_experts(market, targets, stress_execution)
    diagnostics = []
    stress = simulate(market, targets, stress_execution, current, stress_returns)
    stress.to_csv(out / "ledger_double_slippage.csv", index=False)
    diagnostics.append({"name": "double_slippage", **metrics(stress[stress.date.ge("2024-01-01")])})
    for expert in current.experts:
        reduced = replace(current, experts=tuple(name for name in current.experts if name != expert))
        ledger = simulate(market, targets, execution, reduced, expert_returns)
        diagnostics.append({"name": f"without_{expert}", **metrics(ledger[ledger.date.ge("2024-01-01")])})
    pd.DataFrame(diagnostics).to_csv(out / "post_freeze_diagnostics.csv", index=False)
    chart_ledgers = {name: ledgers[name] for name in ("MWU", "range_matched_controls", "equal_matched", "buy_hold")}
    chart_svg(chart_ledgers, out / "holdout_equity.svg", holdout=True)
    chart_svg(chart_ledgers, out / "full_equity.svg")
    test = summary[summary.period.eq("holdout")].set_index("name")
    mwu, equal = test.loc["MWU"], test.loc["equal_matched"]
    incremental = float(mwu.total_return - equal.total_return)
    accepted = bool(incremental > 0 and mwu.sharpe > equal.sharpe and mwu.total_return > test.loc["original_score", "total_return"])
    weights = selected.loc[selected.date.ge("2024-01-01")].filter(like="weight_")
    weight_summary = pd.DataFrame({"expert": [name.removeprefix("weight_") for name in weights.columns],
                                  "mean_weight": weights.mean().to_numpy(), "final_weight": weights.iloc[-1].to_numpy()})
    weight_summary = weight_summary.sort_values("mean_weight", ascending=False)
    weight_summary.to_csv(out / "weight_summary.csv", index=False)
    dominant = str(weight_summary.expert.iloc[0])
    dominant_share = float(weight_summary.mean_weight.iloc[0])
    # Paired circular moving-block bootstrap. Descriptive uncertainty, never tuning.
    rng = np.random.default_rng(20261012)
    active_rows = []
    test_mwu = selected[selected.date.ge("2024-01-01")]["return"].to_numpy()
    for name in ("equal_matched", "range_matched_controls"):
        other = ledgers[name][ledgers[name].date.ge("2024-01-01")]["return"].to_numpy()
        delta = test_mwu - other
        starts = rng.integers(0, len(delta), size=(2000, math.ceil(len(delta) / 20)))
        indices = ((starts[:, :, None] + np.arange(20)) % len(delta)).reshape(2000, -1)[:, :len(delta)]
        annual_active = delta[indices].mean(axis=1) * 252
        low, high = np.quantile(annual_active, [.025, .975])
        active_rows.append({"comparator": name, "annual_mean_return_difference": float(delta.mean() * 252),
                            "block_bootstrap_95_low": float(low), "block_bootstrap_95_high": float(high)})
    pd.DataFrame(active_rows).to_csv(out / "paired_block_bootstrap.csv", index=False)
    matched_range_incremental = float(mwu.total_return - test.loc["range_matched_controls", "total_return"])
    manifest = {
        "symbol": "002594.SZ", "data_start": str(market.date.min().date()),
        "data_end": str(market.date.max().date()), "rows": len(market), "market_sha256": digest,
        "execution": execution.__dict__, "frozen_config": frozen,
        "official_limit_rows": int(market.limit_source.eq("official_local_partition").sum()),
        "inferred_limit_rows": int(market.limit_source.ne("official_local_partition").sum()),
        "holdout_incremental_return_vs_matched_equal": incremental,
        "holdout_incremental_return_vs_range_matched_controls": matched_range_incremental,
        "dominant_expert": dominant, "dominant_mean_weight": dominant_share,
        "bootstrap": active_rows,
        "passed_experiment_gate": accepted, "production_changed": False,
        "multi_expert_incremental_advantage_established": bool(matched_range_incremental > 0 and dominant_share < .90),
        "recommended_replacement": False,
        "calendar_missing_dates": missing_calendar.dt.strftime("%Y-%m-%d").tolist(),
        "holdout_calendar_missing_dates": int(missing_calendar.ge("2024-01-01").sum()),
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in (
            ROOT / "src/quant/research/byd_mwu.py", ROOT / "scripts/research/run_byd_mwu.py",
            ROOT / "tests/test_byd_mwu.py", ROOT / "scripts/research/byd_002594_trading_strategy.py",
            ROOT / "scripts/research/byd_002594_range_t_strategy.py", ROOT / "src/quant/research/byd_daily_t_plan.py",
        )},
        "runtime": {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__},
        "verification": {"frozen_configuration_checked": bool(args.verify_frozen), "development_score_prefix_replayed": replay_score},
        "assumptions": [
            "100万元研究账户，不沿用历史个人持仓；现金收益为0；无杠杆无做空",
            "日线开盘净交易；没有分钟盘口排队、TWAP或RL成交证据",
            "历史过户费统一万0.2保守假设；印花税按2023-08-28切换",
            "2025-07-29公司行为经公告核验；其他现金分红由前收参考价重构",
            "20%现金与红股税收储备，非真实FIFO结算；其他现金到账日简化为除息日",
            "涨跌停大量使用10%推算；开盘成交和滑点仍是研究模拟",
            "开发验证被反复使用；2024+仅在本实验冻结后读取指标，并非全项目盲测",
            "ML专家在测试期可以用已经成熟的历史标签按固定规则重训，未使用未来标签",
            "停止只证明候选库中的坐标局部平台，不证明全局最优或未来收益收敛",
            "虚拟专家净收益驱动学习，真实组合受整数股数和交易约束影响，不能直接套用线性损失的经典遗憾保证",
        ],
        "sources": [
            "https://theoryofcomputing.org/articles/v008a006/v008a006.pdf",
            "https://mlanthology.org/icml/1995/herbster1995icml-tracking/",
            "https://fgk.chinatax.gov.cn/zcfgk/c102416/c5211343/content.html",
            "https://www1.hkexnews.hk/listedco/listconews/sehk/2025/0721/2025072100912_c.pdf",
        ],
    }
    write_json(out / "manifest.json", manifest)
    conclusion = "通过相对等权与原评分的简单比较门槛；但未证明优于同样交易控制的原区间策略，不建议据此替换原策略。" if accepted and matched_range_incremental <= 0 else "通过相对等权与原评分的简单比较门槛，仅可作为冻结前向模拟候选。" if accepted else "未通过本实验比较门槛，不建议以本结果替换现有策略。"
    confidence_lines = "\n".join(
        f"- 相对 `{row['comparator']}`：年化日均收益差 {row['annual_mean_return_difference']:.2%}，20日分块bootstrap的95%描述区间 [{row['block_bootstrap_95_low']:.2%}, {row['block_bootstrap_95_high']:.2%}]。"
        for row in active_rows
    )
    round_rows = ["| 轮次 | 开发验证分 | 改善 | 专家数 | 候选数 |", "|---|---:|---:|---:|---:|"]
    for row in rounds:
        round_rows.append(f"| {row['round']} | {row['score']:.4f} | {row['improvement']:.4f} | {len(row['configuration']['experts'])} | {row['candidates']} |")
    expert_rows = []
    for name, ledger in expert_ledgers.items():
        expert_rows.append({"name": name, **metrics(ledger[ledger.date.ge("2024-01-01")])})
    pd.DataFrame(expert_rows).to_csv(out / "expert_holdout_metrics.csv", index=False)
    assumptions = "\n".join(f"- {item}" for item in manifest["assumptions"])
    report = f"""# BYD MWU 多轮研究结果

数据截止：{manifest['data_end']}；原始日线 {len(market)} 条；研究账户100万元。

## 结论

{conclusion}
已运行 {len(rounds)-1} 轮优化，评估 {len(cache)} 个不同配置。停止状态：{'达到预注册局部平台条件' if converged else '达到轮数上限，未证明局部平台'}。
冻结测试段相对相同专家及相同仓位控制的等权组合，累计净收益差为 {incremental:.2%}（百分点）。不依据测试段结果再次调参。
相对同样交易控制的原区间单专家，累计净收益差为 {matched_range_incremental:.2%}（百分点）。
评估期平均 {dominant_share:.2%} 的专家权重集中在 `{dominant}`，因此实质接近一个单专家策略，不能将10个专家数量当作有效分散。

## 策略结构

专家输出收盘目标仓位 → MWU 使用本期已实现的专家净收益更新 → 下一期开盘按整数股数净交易。
损失收益尺度0.02，截断到[-1,1]，可加入下行惩罚；遗忘和均匀混合帮助低权重专家恢复。
原区间专家复用现有 `apply_position_delta` 规则，适配为日线目标仓位，并不是分钟做T模型的复刻。
原评分专家保留原有映射；修正版只纠正四项反向区间映射，不修改线上脚本。
新增趋势、回调、箱体反转、现金、波动预算，以及五日预测的Ridge和浅层梯度提升树。
两个模型每21个交易日滚动重训，最多756个成熟样本，标签终点不晚于训练日。未接入NLP或Kronos：缺少可核验的历史文本时点或无泄漏预训练证据。

冻结配置：

```json
{json.dumps(current.to_dict(), ensure_ascii=False, indent=2)}
```

## 逐轮迭代

{chr(10).join(round_rows)}

0等权基线；1基础MWU；2适应机制；3规则专家；4滚动模型；5换手控制；6风险控制；7起坐标邻域复核。
开发验证评分是2018-2019、2020-2021、2022-2023三个块的 `Sharpe+0.5*CAGR-0.5*abs(MDD)` 均值减0.35倍标准差。
这些块被复用，分数不能当作独立样本外证据。连续两轮改善<=0.01才停止，最多12轮。

## 冻结后评估：2024年至数据截止

{table(summary[summary.period.eq('holdout')])}

![冻结评估净值](holdout_equity.svg)

## 开发验证：2018-2023

{table(summary[summary.period.eq('development')])}

## 完整研究：2015年至数据截止

{table(summary[summary.period.eq('full')])}

![完整净值](full_equity.svg)

## 专家与诊断

{table(pd.DataFrame(expert_rows))}

{confidence_lines}

以上区间是固定历史样本的描述性不确定度，不校正项目历史选型、多轮搜索和宏观风格变化，不能当作未来收益保证。
`range_matched_controls` 只保留原区间专家并沿用冻结的仓位控制，用于区分执行纪律和融合本身的贡献；`constant_one_third` 是固定1/3仓位对照。两者均为冻结后的诊断，不参与选型。

冻结后双倍滑点及逐个去除专家的结果见 `post_freeze_diagnostics.csv`，仅作脆弱性诊断，不用于重新选型。
全部年度分段见 `yearly_metrics.csv`。持仓和权重逐日见 `ledger_MWU.csv`。模型重训日期和标签成熟日期见 `training_audit.csv`。
个股有报价日与沪深300交易日历相差 {len(missing_calendar)} 个日期，其中2024+评估段相差 {manifest['holdout_calendar_missing_dates']} 个；缺口不伪造成交，滚动窗口按个股有报价日计数，完整期指标需注意这一口径。
全历史原始涨跌停元数据可用 {manifest['official_limit_rows']} 行，其余 {manifest['inferred_limit_rows']} 行采用推算，结果不具备完整生产成交认证。

## 数据与工程限制

{assumptions}

数值验证：本机NumPy BLAS对有限随机矩阵误报浮点异常，因此Ridge使用等价的标准化正规方程、einsum和正定求解。前缀重放与闭式解测试通过。验证重跑要求与初次冻结参数完全一致；初次指标保留在 `initial_metrics.csv`，没有因已查看测试结果而扩展候选库。

源数据快照SHA256：`{digest}`。
重跑：`PYTHONPATH=src python3 scripts/research/run_byd_mwu.py`。
测试：`PYTHONPATH=src python3 -m unittest discover -s tests -p test_byd_mwu.py -v`。

## 来源

- [MWU理论](https://theoryofcomputing.org/articles/v008a006/v008a006.pdf)
- [跟踪切换专家](https://mlanthology.org/icml/1995/herbster1995icml-tracking/)
- [2023年印花税变更](https://fgk.chinatax.gov.cn/zcfgk/c102416/c5211343/content.html)
- [BYD 2025送转实施公告](https://www1.hkexnews.hk/listedco/listconews/sehk/2025/0721/2025072100912_c.pdf)
"""
    (out / "report.md").write_text(report, encoding="utf-8")
    print(table(summary[summary.period.eq("holdout")]), flush=True)
    print(f"Saved {out / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
