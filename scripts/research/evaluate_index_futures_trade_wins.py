"""Annual, cost-net trade precision research; no OOT-based policy selection."""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from backtest_index_futures import ROOT, build_panel, metrics, simulate
from evaluate_index_futures_directions import CAPITAL, load_bars, margin_reserve
from evaluate_index_futures_lr import ETFS, FUTURES_FACTORS, etf_features
from evaluate_index_futures_reversals import futures_features, short_episodes
from evaluate_index_futures_side_exits import forecasts

HOLD = 20
OUT = ROOT / "trade_wins"
POLICIES = ("baseline", "opposing_exit", "confidence60", "consensus20",
            "win55", "win60", "win65", "win60_equal")


def profit_labels(group, days=HOLD, fee=.0003):
    """One-lot open-to-open PnL, including both legs of every roll."""
    frame = group.sort_values("entry_date").copy()
    for name in ("profit_long", "profit_short", "profit_end"):
        frame[name] = None
    for first in range(len(frame) - days + 1):
        path = frame.iloc[first:first + days]
        if not np.array_equal(path.exit_date.values[:-1], path.entry_date.values[1:]):
            continue
        rolls = path.ts_code.values[1:] != path.ts_code.values[:-1]
        cost = fee * (path.entry_open.iloc[0] + path.exit_open.iloc[-1]
                      + ((path.entry_open.values[1:] + path.exit_open.values[:-1]) * rolls).sum())
        gross = (path.exit_open - path.entry_open).sum()
        idx = frame.index[first]
        frame.loc[idx, ["profit_long", "profit_short", "profit_end"]] = [
            int(gross > cost), int(-gross > cost), path.exit_date.iloc[-1]]
    return frame


def probabilities():
    source, _ = build_panel()
    etf = pd.concat(etf_features(symbol, ticker) for symbol, ticker in ETFS.items())
    future = futures_features(load_bars())
    panel = pd.concat(profit_labels(g) for _, g in source.groupby("symbol"))
    panel = panel.merge(etf, on=["symbol", "trade_date"], how="left", validate="one_to_one")
    panel = panel.merge(future, on=["ts_code", "trade_date"], validate="one_to_one")
    panel = panel.replace([np.inf, -np.inf], np.nan)
    features = FUTURES_FACTORS + etf.filter(regex="^etf_").columns.tolist()
    features += future.filter(regex="^future_").columns.tolist()
    # Recompute existing heads to retain their original chronological protocol.
    old = pd.concat(forecasts(), ignore_index=True)
    audit = []
    for year, test in old.groupby(old.entry_date.str[:4]):
        cutoff = test.trade_date.min()
        beginning = (pd.Timestamp(cutoff) - pd.DateOffset(months=36)).strftime("%Y%m%d")
        train = panel[panel.profit_end.lt(cutoff) & panel.trade_date.ge(beginning)
                      & panel.etf_ret_1.notna()].copy()
        assert len(train) > 100 and train.profit_end.max() < cutoff
        age = (pd.Timestamp(cutoff) - pd.to_datetime(train.profit_end)).dt.days.to_numpy()
        for half_life in (0, 12):
            weights = np.exp2(-age / 365.25) if half_life else np.ones(len(train))
            weights /= weights.mean()
            for side in ("long", "short"):
                model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                      LogisticRegression(C=.1, max_iter=1000))
                model.fit(train[features], train[f"profit_{side}"].astype(int),
                          standardscaler__sample_weight=weights,
                          logisticregression__sample_weight=weights)
                old.loc[test.index, f"win_{side}_{half_life}"] = model.predict_proba(test[features])[:, 1]
                audit.append(dict(year=year, side=side, half_life=half_life, rows=len(train),
                                  cutoff=cutoff, first_signal=train.trade_date.min(),
                                  last_maturity=train.profit_end.max(),
                                  positive_rate=train[f"profit_{side}"].astype(int).mean()))
        print(f"trained {year}: {len(train)} prior mature observations", flush=True)
    pd.DataFrame(audit).to_csv(OUT / "training_audit.csv", index=False)
    old["flat"] = 0.
    for symbol in ETFS:
        old[f"long_{symbol}"] = old.symbol.eq(symbol).astype(float)
        old[f"short_{symbol}"] = -old.symbol.eq(symbol).astype(float)
    return old


def replay(groups, policy, fee, margin):
    by_day = {rows[0].entry_date: rows for rows in groups}
    step, opened = -1, 0

    def select(day, positions):
        nonlocal step, opened
        step += 1
        rows = by_day[day]
        if positions:
            code, (lots, _) = next(iter(positions.items()))
            row = next(r for r in rows if r.symbol == code[:2])
            opposing = row.short_20 if lots > 0 else row.long_20
            if step - opened >= HOLD or (policy == "opposing_exit" and opposing >= .35):
                return ("flat", .5, 1, "both", margin_reserve)
            return None
        candidates = []
        if step % 5 == 0:
            for row in rows:
                side = "long" if row.long_60 > row.short_60 else "short"
                confidence = max(row.long_60, row.short_60)
                if (pd.isna(row.etf_ret_1) or confidence <= .45
                        or abs(row.long_60 - row.short_60) < .10):
                    continue
                if policy == "confidence60" and confidence < .60:
                    continue
                if policy == "consensus20" and (
                    getattr(row, f"{side}_20") < .45
                    or (row.long_20 - row.short_20) * (1 if side == "long" else -1) < .10
                ):
                    continue
                if policy.startswith("win"):
                    half_life = 0 if policy.endswith("equal") else 12
                    if getattr(row, f"win_{side}_{half_life}") < int(policy[3:5]) / 100:
                        continue
                candidates.append((confidence, row.symbol, side))
        if not candidates:
            return ("flat", .5, 1, "both", margin_reserve)
        _, symbol, side = sorted(candidates, key=lambda c: (-c[0], c[1]))[0]
        opened = step
        return (f"{side}_{symbol}", .5, 10_000, "both", margin_reserve)

    ledger = simulate(groups, "flat", .5, 1, "both", 1, fee, margin_rate=margin,
                      capital=CAPITAL, enforce_margin=True, lot_cap=1, policy_selector=select)
    assert ledger.gross_lots.le(1).all()
    assert np.allclose(ledger.equity, CAPITAL + ledger.pnl.cumsum(), atol=1e-6, rtol=0)
    return ledger


def trade_stats(trades):
    values = trades.pnl.to_numpy() if len(trades) else np.array([])
    n, wins = len(values), int((values > 0).sum())
    p = wins / n if n else np.nan
    z = 1.96
    centre = (p + z*z/(2*n)) / (1 + z*z/n) if n else np.nan
    spread = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1+z*z/n) if n else np.nan
    positive, negative = values[values > 0], values[values < 0]
    ratio = positive.mean() / -negative.mean() if len(positive) and len(negative) else np.nan
    factor = positive.sum() / -negative.sum() if len(negative) else np.nan
    return dict(trades=n, wins=wins, losses=len(negative), breakeven=int((values == 0).sum()),
                win_rate=p, win_ci_low=centre-spread, win_ci_high=centre+spread,
                payoff_ratio=ratio, profit_factor=factor,
                avg_net_pnl=values.mean() if n else np.nan,
                closed_net_pnl=values.sum(), worst_trade=values.min() if n else np.nan)


def annual_stats(ledger, trades):
    equity = CAPITAL
    for year, part in ledger.groupby(ledger.exit_date.str[:4]):
        path = np.r_[equity, part.equity]
        completed = trades[trades.exit.str.startswith(year)]
        for side in ("all", "long", "short"):
            subset = completed if side == "all" else completed[completed.side.eq(side)]
            yield dict(year=year, side=side, **trade_stats(subset),
                       total_return=path[-1]/equity-1 if side == "all" else np.nan,
                       max_drawdown=(path/np.maximum.accumulate(path)-1).min() if side == "all" else np.nan,
                       forced=int(part.forced_exit.sum()) if side == "all" else np.nan,
                       fees=part.fees.sum() if side == "all" else np.nan)
        equity = path[-1]


def write_report(annual, summary):
    lines = ["# 股指期货逐笔胜率研究（2026-11-13）", "",
             "本轮没有找到在近期各年份持续提高胜率的方案。2020 年起连续账户中，盈利模型 55% 筛选将总胜率从 55.56% 提高到 60.34%，但 2023–2025 年的胜率均下降；不能用早期改善代替近期验证。", "",
             "50 万本金，最多一手，允许多空；原 60 日方向模型决定方向，每 5 个交易日检查入场，最多持有 20 个交易日。所有方案使用同一个保证金安全储备约束。基准成本为单边名义金额 3 bps、保证金 15%；联合压力情景为 6 bps、25%。保证金会改变实际交易路径，因此联合压力情景不是单纯的成本加倍。", "",
             "新增 LR 分别预测做多/做空持有 20 日、真实换月且扣费后的盈利标签。训练回看最多 36 个月，每年重训，标签必须在该年首个信号日之前成熟；新模型比较 12 个月半衰期与等权。原方向模型保留此前的训练方式。55% 等门槛是原始模型分数，未经校准，不代表真实胜率保证。", "",
             "**统计口径：**换月属于同一交易，开平及换月费用全部计入净盈亏；净利润 > 0 才计为获胜。完整交易归属平仓年份；年度净值收益按每天盯市、继承上年末权益，所以跨年持仓时，两者不能直接相加核对。多头/空头明细、平均每笔净利润、盈亏比、利润因子及 Wilson 区间均在 annual.csv。无交易时胜率留空。Wilson 区间仅作描述，交易间相关性和多次实验使其不能直接用于显著性结论。", "",
             "**两套评估分别看：**continuous 为 2020/2022 年分别以 50 万开始并持续运行；annual_reset 为每年独立以 50 万开始、年底平仓，用于隔离早年权益对开仓资格的影响，不能拼成一条可投资净值。2026 年仅截至 11 月 6 日。", "",
             "## 每年独立 50 万：原策略与新增盈利筛选", "",
             "| 年份 | 原策略胜/笔数 | 原胜率 | 筛选胜/笔数 | 筛选胜率 | 原年收益 | 筛选年收益 | 原年内回撤 | 筛选年内回撤 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    def pct(x):
        return f"{x:.2%}" if pd.notna(x) else "—"
    independent = annual.query("evaluation == 'annual_reset' and side == 'all' and margin == .15")
    for year, rows in independent.groupby("year"):
        base = rows[rows.policy.eq("baseline")].iloc[0]
        candidate = rows[rows.policy.eq("win55")].iloc[0]
        cells = [str(year), f"{base.wins}/{base.trades}", pct(base.win_rate),
                 f"{candidate.wins}/{candidate.trades}", pct(candidate.win_rate),
                 pct(base.total_return), pct(candidate.total_return), pct(base.max_drawdown), pct(candidate.max_drawdown)]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "## 所有方案：2022–2026 逐年独立测试的等权胜率", "",
              "每年权重相同，2026 为不完整年度；这里不以交易数量加权，避免早期交易多的年份主导结果。", "",
              "| 方案 | 基准年度平均胜率 | 基准交易总数 | 压力年度平均胜率 | 压力交易总数 |", "|---|---:|---:|---:|---:|"]
    recent = annual.query("evaluation == 'annual_reset' and side == 'all' and year >= '2022'")
    for policy in POLICIES:
        base = recent[(recent.policy == policy) & (recent.margin == .15)]
        stress = recent[(recent.policy == policy) & (recent.margin == .25)]
        lines.append(f"| {policy} | {pct(base.win_rate.mean())} | {base.trades.sum()} | {pct(stress.win_rate.mean())} | {stress.trades.sum()} |")
    lines += ["", "baseline：固定持有基线；opposing_exit：相反方向 20 日模型 ≥35% 提前平仓；confidence60：原方向模型置信分数 ≥60%；consensus20：60 日与 20 日模型同向确认；win55/60/65：新增盈利模型 12 月半衰期筛选；win60_equal：新增模型等权筛选。", "",
              "## 各方案完整逐年胜率（括号内为交易笔数）", ""]
    for evaluation, start, margin in [("continuous", "20200101", .15), ("continuous", "20220101", .15),
                                       ("annual_reset", None, .15), ("annual_reset", None, .25)]:
        sample = annual[(annual.evaluation == evaluation) & (annual.margin == margin) & (annual.side == "all")]
        if start:
            sample = sample[sample.start == start]
        lines += [f"{evaluation}；起点 {start or '每年重置'}；保证金 {margin:.0%}", "",
                  "| 年份 | " + " | ".join(POLICIES) + " |", "|---|" + "---:|" * len(POLICIES)]
        for year, rows in sample.groupby("year"):
            cells = []
            for policy in POLICIES:
                row = rows[rows.policy == policy].iloc[0]
                cells.append(f"{pct(row.win_rate)} ({row.trades})")
            lines.append(f"| {year} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## 判断与下一轮研究约束", "",
              "暂不替换原策略。提高原方向模型门槛在联合压力情景的年度平均胜率较好，但基准情景并未提高，且部分年份只有 2–7 笔；不应据此确定上线门槛。新盈利 label 在近期也未稳定胜出。", "",
              "后续先研究有明确入场事件的二阶段筛选：用当时模型产生的样本外候选交易训练是否放行，label 与真实退出规则一致；分别评估多空。阈值须在每个评估年份之前的内部时间验证中确定，并设最低交易覆盖率、平均净收益和盈亏比约束。近年权重只作用于当时可见的训练/验证数据，不反向选择已看过年份的最佳参数。", "",
              "限制：本实验的盈利 label 是固定 20 日持有，而真实保证金储备约束可能提前平仓；训练日标签有重叠；日线回测没有模拟盘中追保、涨跌停无法成交及历史费率变化。本轮及此前多次查看过的 2020–2026 历史属于探索性时间外回放，不是全新未触碰 OOT。", "",
              "验证：所有账本最多一手，权益与逐日净 PnL 对账；多空完整交易净利润之和与账本完全对账；训练成熟日期早于预测截止；另有标签换月成本、日期缺口、跨年归属、零收益不计胜的四项单元检查。", "",
              "方法参考：[scikit-learn 时间序列交叉验证](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)。", ""]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-predictions", action="store_true",
                        help="Reuse this experiment's saved chronological model forecasts")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    manifest = dict(capital=CAPITAL, lot_cap=1, hold=HOLD, entry_cadence=5,
                    profit_label="20-session rolled net PnL > 0, each side separately",
                    training_months=36, half_life_months=[0, 12], policies=POLICIES,
                    starts=["20200101", "20220101"], annual_attribution="trade exit year; equity marked daily",
                    evaluations=["continuous", "annual_reset"],
                    selection="none; all variants are exploratory on previously studied history",
                    reserve=True, label_fee=.0003,
                    caveats=["No intraday execution or price-limit constraints",
                             "Profit labels use fixed holding; reserve may exit actual trades earlier",
                             "Overlapping training labels; Wilson intervals are descriptive, not independent-trade inference",
                             "The final open trade is liquidated at the sample endpoint"])
    (OUT / "protocol.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if args.reuse_predictions:
        panel = pd.read_parquet(OUT / "predictions.parquet")
    else:
        panel = probabilities()
        panel.to_parquet(OUT / "predictions.parquet", index=False)
    bars = load_bars()
    summaries, annual, episodes = [], [], []
    windows = [("continuous", start, panel[panel.entry_date.ge(start)]) for start in manifest["starts"]]
    for year in sorted(panel.entry_date.str[:4].unique()):
        # Force liquidation before the next calendar year; each diagnostic
        # account starts with 500k. These accounts cannot be compounded together.
        sample = panel[panel.entry_date.str.startswith(year) & panel.exit_date.str.startswith(year)]
        windows.append(("annual_reset", year + "0101", sample))
    for evaluation, start, sample in windows:
        groups = [list(g.itertuples(index=False)) for _, g in sample.groupby("entry_date")]
        for fee, margin in ((.0003, .15), (.0006, .25)):
            for policy in POLICIES:
                ledger = replay(groups, policy, fee, margin)
                parts = [short_episodes(ledger, bars, fee=fee, direction=direction).assign(side=side)
                         for side, direction in (("long", 1), ("short", -1))]
                trades = pd.concat(parts, ignore_index=True)
                if trades.empty:
                    trades = pd.DataFrame(columns=["entry", "exit", "pnl", "side"])
                assert np.isclose(trades.pnl.sum(), ledger.pnl.sum(), atol=1e-6)
                fields = dict(evaluation=evaluation, start=start, policy=policy, fee=fee, margin=margin)
                ledger.to_csv(OUT / f"ledger_{evaluation}_{start}_{policy}_{margin}.csv", index=False)
                episodes.append(trades.assign(**fields))
                summaries.append(dict(**fields, **metrics(ledger, CAPITAL), **trade_stats(trades)))
                annual.extend(dict(**fields, **row) for row in annual_stats(ledger, trades))
                print(evaluation, start, policy, margin, trade_stats(trades)["win_rate"], flush=True)
    pd.DataFrame(summaries).to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame(annual).to_csv(OUT / "annual.csv", index=False)
    pd.concat(episodes, ignore_index=True).to_csv(OUT / "trades.csv", index=False)
    write_report(pd.DataFrame(annual), pd.DataFrame(summaries))


if __name__ == "__main__":
    main()
