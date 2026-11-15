"""Controlled ablation of causal relative-index features for trade screening."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from backtest_index_futures import ROOT, build_panel, metrics
from evaluate_index_futures_directions import CAPITAL, load_bars
from evaluate_index_futures_lr import ETFS, FUTURES_FACTORS, etf_features
from evaluate_index_futures_reversals import futures_features, short_episodes
from evaluate_index_futures_trade_wins import annual_stats, profit_labels, replay, trade_stats

OUT = ROOT / "relative_indices"
TARGETS = {"IF": "000300.SH", "IH": "000016.SH", "IC": "000905.SH", "IM": "000852.SH"}
EXTERNAL = {"growth": "399006.SZ", "shenzhen": "399001.SZ", "shanghai": "000001.SH"}
VARIANTS = ("base", "size", "external", "combined")


def load_prices():
    series, audit = [], []
    for code in dict.fromkeys([*TARGETS.values(), *EXTERNAL.values()]):
        path = Path("data/raw") / f"index_{code}.parquet"
        frame = pd.read_parquet(path).sort_values("trade_date")
        assert not frame.trade_date.duplicated().any()
        audit.append(dict(code=code, path=str(path), first=frame.trade_date.min(),
                          last=frame.trade_date.max(), rows=len(frame)))
        series.append(frame.set_index("trade_date").close.rename(code))
    pd.DataFrame(audit).to_csv(OUT / "source_audit.csv", index=False)
    return pd.concat(series, axis=1).sort_index()


def relative_pair(target, benchmark):
    """Scale-invariant features; beta uses only observations before each day."""
    x, b = np.log(target), np.log(benchmark)
    spread = x - b
    past = spread.shift()
    z = (spread - past.rolling(60).mean()) / past.rolling(60).std().replace(0, np.nan)
    r, rb = x.diff(), b.diff()
    beta = r.shift().rolling(120).cov(rb.shift()) / rb.shift().rolling(120).var().replace(0, np.nan)
    alpha = r.shift().rolling(120).mean() - beta * rb.shift().rolling(120).mean()
    residual = r - alpha - beta * rb
    return pd.DataFrame(dict(ret_diff_5=spread.diff(5), ret_diff_20=spread.diff(20),
                             ratio_z60=z, ratio_z_change5=z.diff(5),
                             residual20=residual.rolling(20).sum(),
                             corr60=r.rolling(60).corr(rb)))


def relative_features(prices):
    parts = []
    changes = np.log(prices[list(TARGETS.values())]).diff(20)
    for symbol, code in TARGETS.items():
        # Avoid a target compared with itself, without choosing anchors by future correlation.
        large = "000300.SH" if symbol != "IF" else "000016.SH"
        small = "000905.SH" if symbol != "IC" else "000852.SH"
        frame = pd.DataFrame(index=prices.index)
        for name, benchmark in {"large": large, "small": small}.items():
            frame = frame.join(relative_pair(prices[code], prices[benchmark]).add_prefix(f"size_{name}_"))
        frame["size_mid_large20"] = changes["000905.SH"] - changes["000300.SH"]
        frame["size_small_large20"] = changes["000852.SH"] - changes["000300.SH"]
        frame["size_dispersion20"] = changes.std(axis=1, skipna=False)
        for name, benchmark in EXTERNAL.items():
            frame = frame.join(relative_pair(prices[code], prices[benchmark]).add_prefix(f"external_{name}_"))
        frame["symbol"] = symbol
        parts.append(frame.reset_index().rename(columns={"index": "trade_date"}))
    return pd.concat(parts, ignore_index=True).replace([np.inf, -np.inf], np.nan)


def forecasts():
    source, _ = build_panel()
    source = pd.concat(profit_labels(g) for _, g in source.groupby("symbol"))
    etf = pd.concat(etf_features(symbol, ticker) for symbol, ticker in ETFS.items())
    future = futures_features(load_bars())
    source = source.merge(etf, on=["symbol", "trade_date"], how="left", validate="one_to_one")
    source = source.merge(future, on=["ts_code", "trade_date"], validate="one_to_one")
    relative = relative_features(load_prices())
    panel = source.merge(relative, on=["symbol", "trade_date"], how="left", validate="one_to_one")
    panel = panel.replace([np.inf, -np.inf], np.nan)
    old = pd.read_parquet(ROOT / "trade_wins/predictions.parquet")
    old = old.merge(relative, on=["symbol", "trade_date"], how="left", validate="one_to_one")
    base = FUTURES_FACTORS + etf.filter(regex="^etf_").columns.tolist()
    base += future.filter(regex="^future_").columns.tolist()
    size = relative.filter(regex="^size_").columns.tolist()
    external = relative.filter(regex="^external_").columns.tolist()
    sets = dict(base=base, size=base+size, external=base+external, combined=base+size+external)
    all_features = sets["combined"]
    # All ablations use exactly the same eligible training and forecast dates.
    assert old[size + external].notna().all().all()
    audit, coefficients = [], []
    for year, test in old.groupby(old.entry_date.str[:4]):
        cutoff = test.trade_date.min()
        begin = (pd.Timestamp(cutoff) - pd.DateOffset(months=36)).strftime("%Y%m%d")
        train = panel[panel.profit_end.lt(cutoff) & panel.trade_date.ge(begin)
                      & panel.etf_ret_1.notna()].dropna(subset=size+external)
        assert train.profit_end.max() < cutoff
        age = (pd.Timestamp(cutoff) - pd.to_datetime(train.profit_end)).dt.days.to_numpy()
        weight = np.exp2(-age / 365.25)
        weight /= weight.mean()
        for variant, features in sets.items():
            for side in ("long", "short"):
                model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                      LogisticRegression(C=.1, max_iter=1000))
                model.fit(train[features], train[f"profit_{side}"].astype(int),
                          standardscaler__sample_weight=weight,
                          logisticregression__sample_weight=weight)
                old.loc[test.index, f"aux_{variant}_{side}"] = model.predict_proba(test[features])[:, 1]
                audit.append(dict(year=year, variant=variant, side=side, rows=len(train),
                                  feature_count=len(features), cutoff=cutoff,
                                  last_maturity=train.profit_end.max()))
                coefficients.extend(dict(year=year, variant=variant, side=side, feature=f, coef=c)
                                    for f, c in zip(features, model[-1].coef_[0]))
        print("trained", year, len(train), flush=True)
    pd.DataFrame(audit).to_csv(OUT / "training_audit.csv", index=False)
    pd.DataFrame(coefficients).to_csv(OUT / "coefficients.csv", index=False)
    (OUT / "feature_sets.json").write_text(json.dumps(sets, indent=2), encoding="utf-8")
    old[["symbol", "trade_date", *all_features]].to_parquet(OUT / "features.parquet", index=False)
    return old


def matched_screen(trades, panel, fields):
    """Frozen baseline trades: isolate screening from shifts in later entry dates."""
    if trades.empty:
        return []
    lookup = panel.set_index(["entry_date", "symbol"])
    result = []
    for row in trades.itertuples():
        signal = lookup.loc[(row.entry, row.product)]
        for variant in VARIANTS:
            score = signal[f"aux_{variant}_{row.side}"]
            result.append(dict(**fields, variant=variant, entry=row.entry, exit=row.exit,
                               product=row.product, side=row.side, pnl=row.pnl,
                               probability=score, accepted=score >= .55))
    return result


def report(annual, matched):
    def pct(x):
        return "—" if pd.isna(x) else f"{x:.1%}"
    policies = ["no_screen", *VARIANTS]
    lines = ["# 辅助指数相对强弱因子实验", "",
             "固定原60日方向模型、20日盈利label、最长20日持有、55%筛选分数；仅改变盈利LR的因子集合。新模型最多回看36个月，12个月半衰期，每年只使用先前成熟标签。所有特征组使用相同训练样本。50万元，最多一手，多空均可，次日开盘执行。", "",
             "base：原期货与ETF因子；size：加15个规模指数相对因子；external：加18个创业板/深成指/上证综指相对因子；combined：加全部33个；no_screen：原方向策略不经盈利LR筛选。筛选分数并非已校准的真实胜率。", "",
             "辅助信息使用国内指数日线，同日收盘可得；统一样本终点2026-11-06，原始文件更晚的数据不进入当时信号。未引入债券、波动率、海外指数。", "",
             "## 基准情景逐年独立账户胜率（胜/总笔数）", "",
             "单边成本3bps，保证金15%。每年重新以50万元开始、年底平仓；不可拼接成连续复利账户。", "",
             "| 年份 | " + " | ".join(policies) + " |", "|---|" + "---:|" * len(policies)]
    sample = annual.query("evaluation == 'annual_reset' and side == 'all' and fee == .0003 and margin == .15")
    for year, group in sample.groupby("year"):
        cells = []
        for policy in policies:
            row = group[group.policy.eq(policy)].iloc[0]
            cells.append(f"{pct(row.win_rate)} ({row.wins}/{row.trades})")
        lines.append(f"| {year} | " + " | ".join(cells) + " |")
    lines += ["", "## 最近年份逐年独立账户：所有情景", "",
              "2022–2026年等权平均胜率，2026年不完整；无交易年份同时披露。组合年收益均值不是复合年化收益。", "",
              "| 费用bps | 保证金 | 因子组 | 平均胜率 | 有交易年数 | 总笔数 | 平均年度收益 | 最差年内回撤 |", "|---:|---:|---|---:|---:|---:|---:|---:|"]
    recent = annual[(annual.evaluation == "annual_reset") & (annual.side == "all") & (annual.year.astype(int) >= 2022)]
    for (fee, margin, policy), group in recent.groupby(["fee", "margin", "policy"]):
        lines.append(f"| {fee*10000:.0f} | {pct(margin)} | {policy} | {pct(group.win_rate.mean())} | {group.win_rate.notna().sum()} | {group.trades.sum()} | {pct(group.total_return.mean())} | {pct(group.max_drawdown.min())} |")
    lines += ["", "## 相同原策略交易的筛选诊断", "",
              "固定每年独立账户原策略实际交易，分别统计模型保留及拒绝的完整交易。此表不允许拒绝后另开仓，不能视为可执行组合；用途是区分筛选质量与交易时点变化。", "",
              "| 年份 | 因子组 | 保留胜率 | 保留笔数 | 保留净利润元 | 拒绝胜率 | 拒绝笔数 | 拒绝净利润元 |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    frozen = matched[(matched.evaluation == "annual_reset") & (matched.fee == .0003) & (matched.margin == .15)].copy()
    frozen["year"] = frozen.exit.str[:4]
    for (year, variant), group in frozen.groupby(["year", "variant"]):
        accepted, rejected = (trade_stats(group[group.accepted.eq(flag)]) for flag in (True, False))
        lines.append(f"| {year} | {variant} | {pct(accepted['win_rate'])} | {accepted['trades']} | {accepted['closed_net_pnl']:.0f} | {pct(rejected['win_rate'])} | {rejected['trades']} | {rejected['closed_net_pnl']:.0f} |")
    lines += ["", "## 本轮判断与下一步", "",
              "加入全部辅助指数后，基准情景2022–2026年逐年独立账户的平均胜率从盈利筛选基线36.7%升至42.3%，但仍低于不筛选的原策略50.7%。2020年起连续账户中，全部辅助因子将收益从155.6%提高至209.2%，最大回撤从34.7%降至25.2%，胜率却从60.3%略降至59.3%。因此不能把收益/回撤改善写成胜率改善。", "",
              "更严格的相同交易诊断中，2022年后原策略共有40笔交易；全部辅助因子保留33笔，16笔盈利；拒绝7笔，其中6笔盈利。其组合年度胜率改善并非直接证明该筛选能挑出更高胜率的原候选。大小盘因子拒绝的7笔中5笔盈利但合计净亏147884元，提示其可能更适合控制亏损金额，而非提高获胜笔数。上述小样本只作为研究线索。", "",
              "多空拆开看：全部辅助因子在2025年空头只有1/6笔盈利；2026年唯一空头亏损约10.35万元。下阶段应检验方向分别建模的候选交易筛选与继续持有风险label，而不是根据已知2026年亏损事后添加禁止做空规则。阈值和市场状态规则必须由该年之前的内部时间验证确定。", "",
              "建议保留相对指数因子继续研究，但不直接替换现有策略。先测试相对弱势是否正在改善与ETF放量/期货收盘位置的少量交互，再将候选交易的label与实际退出对齐。对于资金约束退出，保存当时账户状态，独立候选诊断与真实组合执行分别核验。", "",
              "全量逐年多空收益、回撤、盈亏比与交易数见annual.csv，逐笔见trades.csv；matched.csv保存固定候选的所有筛选记录，training_audit.csv记录训练截止。", "",
              "限制：本轮为了单独检验辅助因子，沿用固定20日盈利label；资金约束可能使真实交易提前退出，此项尚待下一阶段对齐。没有同时优化退出规则、阈值或模型复杂度。指数差值并不天然预示均值回复；本实验没有声称协整。日线不能检验盘中强平或涨跌停成交。过去历史已经多次研究，此结果不是未触碰OOT。", "",
              "指数选择依据：[中证1000编制方案](https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/indices/detail/files/zh_CN/20231208175402-000852_Index_Methodology_cn.pdf)；预处理原则：[scikit-learn防止信息泄漏](https://scikit-learn.org/stable/common_pitfalls.html)。", ""]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-predictions", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.reuse_predictions:
        panel = pd.read_parquet(OUT / "predictions.parquet")
    else:
        panel = forecasts()
        panel.to_parquet(OUT / "predictions.parquet", index=False)
    protocol = dict(threshold=.55, variants=VARIANTS, fees=[.0003, .0006], margins=[.15, .25],
                    capital=CAPITAL, lot_cap=1, hold=20, half_life=12, train_months=36,
                    first=panel.entry_date.min(), last=panel.exit_date.max(),
                    selection="no selection on evaluation years; fixed feature ablation",
                    labels="previous fixed20 net-positive labels, not yet aligned to cash-driven early exit")
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    bars = load_bars()
    windows = [("continuous", start, panel[panel.entry_date.ge(start)]) for start in ("20200101", "20220101")]
    for year in sorted(panel.entry_date.str[:4].unique()):
        windows.append(("annual_reset", year+"0101", panel[panel.entry_date.str.startswith(year) & panel.exit_date.str.startswith(year)]))
    summaries, annual, episodes, matched = [], [], [], []
    for evaluation, start, window in windows:
        for fee in (.0003, .0006):
            for margin in (.15, .25):
                for variant in ("no_screen", *VARIANTS):
                    sample = window.copy()
                    if variant != "no_screen":
                        for side in ("long", "short"):
                            sample[f"win_{side}_12"] = sample[f"aux_{variant}_{side}"]
                    groups = [list(g.itertuples(index=False)) for _, g in sample.groupby("entry_date")]
                    ledger = replay(groups, "baseline" if variant == "no_screen" else "win55", fee, margin)
                    parts = [short_episodes(ledger, bars, fee, direction).assign(side=side)
                             for side, direction in (("long", 1), ("short", -1))]
                    trades = pd.concat(parts, ignore_index=True)
                    if trades.empty:
                        trades = pd.DataFrame(columns=["entry", "exit", "pnl", "side"])
                    assert np.isclose(trades.pnl.sum(), ledger.pnl.sum(), atol=1e-6)
                    fields = dict(evaluation=evaluation, start=start, fee=fee, margin=margin, policy=variant)
                    summaries.append(dict(**fields, **metrics(ledger, CAPITAL), **trade_stats(trades)))
                    annual.extend(dict(**fields, **row) for row in annual_stats(ledger, trades))
                    episodes.append(trades.assign(**fields))
                    ledger.to_csv(OUT / f"ledger_{evaluation}_{start}_{variant}_{fee}_{margin}.csv", index=False)
                    if variant == "no_screen":
                        matched.extend(matched_screen(trades, panel, fields))
        print("evaluated", evaluation, start, flush=True)
    annual, summary, matched = pd.DataFrame(annual), pd.DataFrame(summaries), pd.DataFrame(matched)
    annual.to_csv(OUT / "annual.csv", index=False)
    summary.to_csv(OUT / "summary.csv", index=False)
    pd.concat(episodes, ignore_index=True).to_csv(OUT / "trades.csv", index=False)
    matched.to_csv(OUT / "matched.csv", index=False)
    report(annual, matched)


if __name__ == "__main__":
    main()
