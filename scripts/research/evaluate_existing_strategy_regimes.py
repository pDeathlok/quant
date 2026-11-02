"""Condition existing frozen strategy results on causal market/industry trends.

Event returns remain event statistics: overlapping labels are never treated as
an investable portfolio. Production files and configurations are read only.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "reports/strategy_lab_20260930/existing_strategies"
RIGHT = ("B2", "B3", "KEY_K", "VIOLENCE_K", "PINGHANG", "DOUBLE_GUN",
         "CHANGAN", "KENGQI", "VEGAS", "TRIPLE_VOLUME_BREAKOUT", "GOLDEN_BOWL",
         "ZAIHOU", "BREATHING", "YUEYUE")
LEFT = ("B1", "SB1", "SUPER_B1", "LOW_PULLBACK")


def dates(values: pd.Series) -> pd.Series:
    text = values.astype(str)
    if text.str.fullmatch(r"\d{8}").all():
        return pd.to_datetime(text, format="%Y%m%d")
    return pd.to_datetime(values).dt.normalize()


def basic() -> pd.DataFrame:
    frame = pd.read_parquet(ROOT / "data/raw/stock_basic_history.parquet")
    frame["list_date"] = pd.to_datetime(frame.list_date.astype(str), format="%Y%m%d", errors="coerce")
    frame["symbol"] = frame.symbol.astype(str).str.zfill(6)
    return frame.drop_duplicates("ts_code", keep="last")


def build_context() -> tuple[pd.DataFrame, pd.DataFrame]:
    industry_path = OUT / "industry_context.parquet"
    index = pd.read_parquet(ROOT / "data/raw/index_000300.SH.parquet")
    index["date"] = dates(index.trade_date)
    index = index.sort_values("date").drop_duplicates("date").set_index("date")
    px = index.close
    ma20, ma60, ma120 = (px.rolling(n).mean() for n in (20, 60, 120))
    index["market_fast"] = np.select(
        [(px > ma20) & (ma20 > ma60), (px < ma20) & (ma20 < ma60)],
        ["up", "down"], default="neutral")
    index["market_slow"] = np.select(
        [(px > ma120) & (ma120 > ma120.shift(20)) & (px.pct_change(60) > 0),
         (px < ma120) & (ma120 < ma120.shift(20))], ["up", "down"], default="neutral")
    index["market_ret20"] = px.pct_change(20, fill_method=None)
    context = index[["market_fast", "market_slow", "market_ret20"]].reset_index()
    context["previous_date"] = context.date.shift(1)
    if industry_path.exists():
        return context, pd.read_parquet(industry_path)
    metadata = basic().set_index("ts_code")
    chunks = []
    paths = sorted((ROOT / "data/raw/daily_partitioned").glob("year_month=*/data.parquet"))
    paths = [path for path in paths if "year_month=2013" <= path.parent.name <= "year_month=202609"]
    for number, path in enumerate(paths):
        frame = pd.read_parquet(path, columns=["ts_code", "date", "pct_chg", "amount", "industry"])
        frame["date"] = dates(frame.date)
        frame = frame.drop_duplicates(["ts_code", "date"])
        listing = frame.ts_code.map(metadata.list_date)
        frame["industry"] = frame.industry.fillna(frame.ts_code.map(metadata.industry))
        valid = ((frame.date - listing).dt.days >= 365) & (frame.amount > 0)
        valid &= frame.pct_chg.notna() & frame.industry.notna()
        grouped = frame.loc[valid].groupby(["date", "industry"]).agg(
            daily_return=("pct_chg", lambda s: s.mean() / 100), members=("pct_chg", "size"))
        chunks.append(grouped[grouped.members >= 5].reset_index())
        if number % 36 == 0:
            print(f"industry context: {number + 1}/{len(paths)} monthly partitions", flush=True)
    panel = pd.concat(chunks, ignore_index=True).sort_values(["industry", "date"])
    panel["proxy_equity"] = panel.groupby("industry").daily_return.transform(lambda s: (1 + s).cumprod())
    grouped_px = panel.groupby("industry").proxy_equity
    panel["sector_ret20"] = grouped_px.transform(lambda s: s.pct_change(20, fill_method=None))
    panel["sector_ma60"] = grouped_px.transform(lambda s: s.rolling(60).mean())
    panel["sector_trend"] = np.select(
        [(panel.proxy_equity > panel.sector_ma60) & (panel.sector_ret20 > 0),
         (panel.proxy_equity < panel.sector_ma60) & (panel.sector_ret20 < 0)],
        ["up", "down"], default="neutral")
    panel["sector_heat"] = pd.cut(panel.sector_ret20,
                                   [-np.inf, 0, .10, .20, np.inf],
                                   labels=["negative", "0_to_10pct", "10_to_20pct", "20pct_plus"],
                                   right=False).astype(str)
    panel = panel.merge(context[["date", "market_ret20"]], on="date", how="left", validate="many_to_one")
    panel["sector_relative20"] = panel.sector_ret20 - panel.market_ret20
    assert not panel.duplicated(["date", "industry"]).any()
    panel.to_parquet(industry_path, index=False)
    context.to_csv(OUT / "market_context.csv", index=False)
    return context, panel


def top20(frame: pd.DataFrame, score: str) -> pd.DataFrame:
    return frame.sort_values(["date", score, "symbol"], ascending=[True, False, True])\
        .groupby("date", sort=False).head(20).copy()


def long_target_labels() -> dict[str, pd.DataFrame]:
    """Diagnostic 60-session price proxies for the two archived target lists.

    These are fresh labels, not a replay of either portfolio's changing weights.
    Exact market-calendar endpoints and a fully quoted path are required.
    """
    paths = {
        "L1_v44_targets_60d": "reports/long_dividend_quality/v44_tea_master_defensive_neutral/l1_dividend_quality_targets.csv",
        "tea_master_targets_60d": "reports/tea_master_long/core14_soft_plus/tea_master_targets.csv",
    }
    cache = OUT / "long_target_labels.parquet"
    if cache.exists():
        combined = pd.read_parquet(cache)
        return {key: group.copy() for key, group in combined.groupby("target_strategy")}
    targets = {key: pd.read_csv(ROOT / path) for key, path in paths.items()}
    symbols = sorted(set().union(*(set(frame.ts_code) for frame in targets.values())))
    chunks = []
    for path in sorted((ROOT / "data/raw/daily_partitioned").glob("year_month=*/data.parquet")):
        if not "year_month=2013" <= path.parent.name <= "year_month=202609":
            continue
        chunk = pd.read_parquet(path, columns=["date", "ts_code", "pct_chg", "amount"],
                                filters=[("ts_code", "in", symbols)])
        chunks.append(chunk)
    raw = pd.concat(chunks, ignore_index=True)
    raw["date"] = dates(raw.date)
    raw = raw.drop_duplicates(["date", "ts_code"])
    calendar = pd.read_parquet(ROOT / "data/raw/index_000300.SH.parquet")
    calendar = pd.DatetimeIndex(dates(calendar.trade_date).sort_values().unique())
    close_returns = (raw.pct_chg / 100).where(raw.amount > 0)
    raw["price_return"] = close_returns
    wide = raw.pivot(index="date", columns="ts_code", values="price_return").reindex(calendar)
    price = (1 + wide).cumprod()
    label_at_entry = price.shift(-60) / price - 1
    full_path = wide.notna().rolling(61).sum().shift(-60).eq(61)
    label_at_entry = label_at_entry.where(full_path)
    labels = label_at_entry.shift(-1).stack().rename("forward_60d_proxy").reset_index()
    labels.columns = ["date", "ts_code", "forward_60d_proxy"]
    result = {}
    for key, target in targets.items():
        target["date"] = dates(target.rebalance_date)
        target = target.merge(labels, on=["date", "ts_code"], how="left", validate="many_to_one")
        target["target_strategy"] = key
        target["target_source"] = paths[key]
        result[key] = target
    pd.concat(result.values(), ignore_index=True).to_parquet(cache, index=False)
    return result


def event_sources() -> tuple[pd.DataFrame, list[dict]]:
    frames, inventory = [], []
    meta = basic()
    symbol_map = meta.set_index("symbol").ts_code.to_dict()
    industry_map = meta.set_index("ts_code").industry.to_dict()

    def add(name, data, source, value, *, fee=0, horizon="5 trading days", kind="forward_label",
            symbol="symbol", timing="close", market="fast"):
        frame = data.copy()
        frame["date"] = dates(frame.date)
        raw_symbol = frame[symbol].astype(str)
        frame["symbol"] = raw_symbol.where(raw_symbol.str.contains(r"\.", regex=True),
            raw_symbol.str.replace(r"\.0$", "", regex=True).str.zfill(6).map(symbol_map))
        frame["industry"] = (frame.industry if "industry" in frame else
                              frame.symbol.map(industry_map))
        frame["industry"] = frame.industry.fillna(frame.symbol.map(industry_map))
        frame["net_return"] = pd.to_numeric(frame[value], errors="coerce") - fee
        frame["strategy"] = name
        frame["timing"] = timing
        frame["market_horizon"] = market
        frame["evidence_kind"] = kind
        frame["year"] = frame.date.dt.year.astype(str)
        if "fold" not in frame:
            frame["fold"] = frame.get("split", frame.year)
        columns = ["strategy", "date", "symbol", "industry", "net_return", "timing",
                   "market_horizon", "evidence_kind", "year", "fold"]
        for optional in ("exit_date", "entry_date", "exit_type", "exit_reason", "return_120d"):
            if optional in frame:
                if optional.endswith("_date"):
                    frame[optional] = pd.to_datetime(frame[optional], errors="coerce")
                columns.append(optional)
        frames.append(frame[columns])
        inventory.append({"strategy": name, "source": source, "evidence_kind": kind,
                          "horizon": horizon, "extra_cost_bps": fee * 10000, "rows": len(frame),
                          "start": str(frame.date.min().date()), "end": str(frame.date.max().date())})

    path = "reports/b1/current/trades.csv"
    b1 = pd.read_csv(ROOT / path, dtype={"symbol": str})
    for key in ("stable", "aggressive"):
        x = b1[b1.combo.str.startswith(key)].copy()
        x["gross_fraction"] = x.return_pct / 100
        add(f"B1_{key}_original_exit", x, path, "gross_fraction", fee=.002,
            horizon="original T+5 / T+9 exit", kind="exit_log_gross_minus_20bps")

    path = "reports/research/right_side_unified_canonical_v5_rule113/test_predictions.parquet"
    right = pd.read_parquet(ROOT / path)
    assert not right.duplicated(["fold", "symbol", "date"]).any()
    add("RIGHT_unified_top20", top20(right, "pred_unified_long_task_deep"), path, "terminal_return", fee=.002)
    for family in RIGHT:
        data = top20(right[right[family].fillna(False).astype(bool)], "pred_unified_long_task_deep")
        add(f"RIGHT_{family}_top20", data, path, "terminal_return", fee=.002)

    path = "reports/research/left_side_regime_gate_v5/fold_predictions.parquet"
    left = pd.read_parquet(ROOT / path)
    flags = pd.read_parquet(ROOT / "reports/research/left_side_unified_v3_group4_input_parity/test_predictions.parquet",
                            columns=["fold", "symbol", "date", *LEFT])
    left = left.merge(flags, on=["fold", "symbol", "date"], how="left", validate="one_to_one")
    assert left[list(LEFT)].notna().all().all(), "missing family membership after merge"
    add("LEFT_unified_top20", top20(left, "production_probability"), path, "terminal_return", fee=.002)
    for family in LEFT:
        data = top20(left[left[family].fillna(False).astype(bool)], "production_probability")
        add(f"LEFT_{family}_top20", data, path, "terminal_return", fee=.002)
    add("LEFT_shadow_blend_regime_top20",
        top20(left[left.trade_gate_open], "blended_percentile"), path, "terminal_return", fee=.002)

    path = "reports/chan_daily/model_strategy/chan_model_strategy_scored.parquet"
    chan = pd.read_parquet(ROOT / path)
    for rule in ("chan_model_primary", "chan_model_expanded", "chan_model_high_conviction"):
        frame = chan[(chan.chan_model_rule_id == rule) & (chan.split != "train")].copy()
        frame["gross_fraction"] = frame.hold_10d_close / 100
        add(rule, frame, path, "gross_fraction", fee=.002, horizon="10 trading days")

    from scripts.research.backtest_triple_volume_breakout import ExitRule, simulate_exit
    for suffix, name in (("201955", "TVB_3x_original_exit"), ("214605", "TVB_2p5x_original_exit")):
        path = f"reports/triple_volume_breakout/triple_volume_breakout_candidates_20260623_{suffix}.parquet"
        candidates = pd.read_parquet(ROOT / path)
        frame = simulate_exit(candidates, ExitRule("fixed_tp5_sl2p5_T9", "fixed", 8, .05, .025))
        frame["gross_fraction"] = frame.return_pct / 100
        add(name, frame, path, "gross_fraction", fee=.002, horizon="original T+9 capped exit",
            kind="replayed_stored_ohlc_exit")

    for folder, name in (("swing_timing_v1", "swing_low_vol_reclaim"),
                         ("swing_timing_long_horizon_v1", "swing_long_horizon_reference")):
        path = f"reports/{folder}/selected_trades.parquet"
        add(name, pd.read_parquet(ROOT / path), path, "net_return", horizon="original capped exit",
            kind="original_net_exit_log", symbol="ts_code", market="slow")

    for file, name in (("locked_policy_confirmation_picks", "tail_overnight_locked"),
                       ("refit_forward_picks", "tail_overnight_refit")):
        path = f"reports/research/tail_overnight_v2/{file}.parquet"
        add(name, pd.read_parquet(ROOT / path), path, "target_net", horizon="next close",
            kind="close_proxy_overnight_net_label", timing="previous_close")

    for folder, prefix in (("right_t012_full_registry_models_v1", "T012"),
                            ("right_t012_full_registry_fusion_v1", "T012_fusion")):
        path = f"reports/research/{folder}/picks.parquet"
        frame = pd.read_parquet(ROOT / path)
        frame = frame[frame.stage == "diagnostic"]
        for (arm, target, policy), group in frame.groupby(["arm", "target", "policy"]):
            add(f"{prefix}_{arm}_{target}_{policy}", group, path, "net", horizon="next-day exit",
                kind="overnight_net_label", timing="previous_close")

    path = "reports/long_quality_iteration_v3/monthly_selections.parquet"
    long = pd.read_parquet(ROOT / path, columns=["date", "ts_code", "industry", "fold", "candidate", "return_52w"])
    for candidate, group in long.groupby("candidate"):
        add(f"LONG_{candidate}_52w", group, path, "return_52w", fee=.003,
            horizon="52 weeks, immature outcomes retained as unknown", kind="forward_52w_not_closed_trade",
            symbol="ts_code", market="slow")

    path = "reports/long_good_stock_price/signals.parquet"
    selected = json.loads((ROOT / "reports/long_good_stock_price/selected_rule.json").read_text())
    frame = pd.read_parquet(ROOT / path)
    frame = frame[(frame.rule == selected["selected_rule"]) &
                  (frame.history_window_months == selected["selected_history_window_months"])]
    add("LONG_good_price_composite_60_guard_12m", frame, path, "return_12m", fee=.003,
        horizon="12 months", kind="forward_12m_not_closed_trade", symbol="ts_code", market="slow")

    for name, target in long_target_labels().items():
        add(name, target, target.target_source.iloc[0], "forward_60d_proxy", fee=.003,
            horizon="next close to 60 sessions later; complete quote path required",
            kind="target_forward_price_proxy_not_closed_trade", symbol="ts_code", market="slow")

    path = "reports/slow_money_follow/slow_money_strategy_trades.csv"
    frame = pd.read_csv(ROOT / path)
    frame["gross_fraction"] = frame.trade_return_pct / 100
    for strategy, group in frame.groupby("strategy"):
        add(f"slow_money_{strategy}", group, path, "gross_fraction", fee=.002,
            horizon="stored strategy horizon; short June 2026 fragment", symbol="ts_code")

    return pd.concat(frames, ignore_index=True), inventory


def attach_context(events: pd.DataFrame, market: pd.DataFrame, industry: pd.DataFrame) -> pd.DataFrame:
    frame = events.copy()
    previous = market.set_index("date").previous_date
    frame["context_date"] = frame.date
    mask = frame.timing.eq("previous_close")
    frame.loc[mask, "context_date"] = frame.loc[mask, "date"].map(previous)
    market = market.rename(columns={"date": "context_date"})
    frame = frame.merge(market[["context_date", "market_fast", "market_slow"]], on="context_date",
                        how="left", validate="many_to_one")
    frame["market_trend"] = frame.market_fast.where(frame.market_horizon.eq("fast"), frame.market_slow)
    frame = frame.merge(industry[["date", "industry", "sector_trend", "sector_heat", "sector_ret20", "sector_relative20"]]
                        .rename(columns={"date": "context_date"}), on=["context_date", "industry"],
                        how="left", validate="many_to_one")
    for field in ("market_trend", "sector_trend", "sector_heat"):
        frame[field] = frame[field].fillna("unknown")
    return frame


def event_metrics(frame: pd.DataFrame) -> dict:
    net = frame.net_return.dropna()
    known_dates = frame.loc[frame.net_return.notna(), "date"].nunique()
    wins = net[net > 0]
    losses = net[net <= 0]
    return {"records": len(frame), "known": len(net), "unknown": len(frame) - len(net),
            "dates": frame.date.nunique(), "known_dates": known_dates, "known_mean_net": net.mean(),
            "median_net": net.median(), "known_win_rate": (net > 0).mean(),
            "win_unknown_as_failure": len(wins) / len(frame) if len(frame) else np.nan,
            "profit_factor": wins.sum() / -losses.sum() if losses.sum() < 0 else np.nan,
            "p10_net": net.quantile(.1), "worst_net": net.min(),
            "mean_win": wins.mean(), "mean_loss": losses.mean(),
            "sample_gate": len(net) >= 30 and known_dates >= 20,
            "sector_unknown": int(frame.sector_trend.eq("unknown").sum())}


def analyze_events(frame: pd.DataFrame) -> None:
    rows, strata = [], []
    for strategy, all_periods in frame.groupby("strategy"):
        samples = [("full", all_periods), *list(all_periods.groupby("year")),
                   *[(f"fold_{fold}", group) for fold, group in all_periods.groupby("fold")]]
        for period, sample in samples:
            masks = {
                "all": pd.Series(True, index=sample.index),
                "sector_known_all": sample.sector_trend.ne("unknown"),
                "market_not_down": sample.market_trend.isin(["up", "neutral"]),
                "market_up": sample.market_trend.eq("up"),
                "sector_up": sample.sector_trend.eq("up"),
                "market_sector_up": sample.market_trend.eq("up") & sample.sector_trend.eq("up"),
                "market_sector_up_not_hot": sample.market_trend.eq("up") & sample.sector_trend.eq("up") &
                                             sample.sector_ret20.lt(.20),
            }
            for policy, mask in masks.items():
                group = sample.loc[mask]
                rows.append({"strategy": strategy, "period": period, "policy": policy,
                             "retained": len(group) / len(sample), **event_metrics(group)})
            for fields in (["market_trend"], ["sector_trend"], ["market_trend", "sector_trend"], ["sector_heat"]):
                for key, group in sample.groupby(fields, dropna=False):
                    key = key if isinstance(key, tuple) else (key,)
                    strata.append({"strategy": strategy, "period": period, "dimension": "+".join(fields),
                                   "state": "+".join(str(v) for v in key), **event_metrics(group)})
    pd.DataFrame(rows).to_csv(OUT / "event_policy_comparison.csv", index=False)
    pd.DataFrame(strata).to_csv(OUT / "event_regime_strata.csv", index=False)
    baseline = pd.DataFrame(rows).query("period=='full' and policy=='all'")
    baseline.to_csv(OUT / "event_baselines.csv", index=False)
    failures = frame[frame.strategy.isin(["B1_stable_original_exit", "LEFT_unified_top20", "RIGHT_unified_top20",
                                          "chan_model_primary", "LONG_xgb_quality_scores_52w", "tail_overnight_locked"])]
    failures = failures.sort_values("net_return").groupby("strategy").head(5)
    failures.to_csv(OUT / "event_failure_cases.csv", index=False)
    successes = frame[frame.strategy.isin(failures.strategy.unique())].sort_values("net_return", ascending=False)
    successes.groupby("strategy").head(5).to_csv(OUT / "event_success_cases.csv", index=False)


def daily_sources() -> dict[str, pd.DataFrame]:
    paths = {
        "L1_v44": "reports/long_dividend_quality/v44_tea_master_defensive_neutral/l1_dividend_quality_equity.csv",
        "tea_master_core14_soft_plus": "reports/tea_master_long/core14_soft_plus/tea_master_equity.csv",
        "long_price_v2_immediate": "reports/long_entry_model_v2/equity_v2_immediate_conservative.csv",
        "long_price_v2_staged": "reports/long_entry_model_v2/equity_v2_staged_50_25_25_conservative.csv",
        "long_quality_v3": "reports/long_quality_iteration_v3/equity_xgb_quality_scores.csv",
        "cb_existing_rotation": "reports/convertible_bond/rotation/20180101_20260616_daily/equity.csv",
        "BYD_trend": "reports/byd_002594/byd_002594_backtest_daily.csv",
        "BYD_range_T": "reports/byd_002594/byd_002594_range_t_backtest_daily.csv",
        "ETF_stability": "reports/strategy_lab_20260930/invvol60_nasdaq_trend_daily.csv",
        "ETF_nasdaq": "reports/strategy_lab_20260930/buy_nasdaq_daily.csv",
    }
    result, checks = {}, []
    for name, path in paths.items():
        source = pd.read_csv(ROOT / path)
        date_column = "date" if "date" in source else "trade_date"
        source["date"] = dates(source[date_column])
        source = source.sort_values("date").drop_duplicates("date").set_index("date")
        if name.startswith("BYD"):
            values = source.strategy_return
        elif "equity" in source:
            values = source.equity.pct_change(fill_method=None)
            initial_capital = 1_000_000 if name == "cb_existing_rotation" else 1.0
            values.iloc[0] = source.equity.iloc[0] / initial_capital - 1
            reconstructed = initial_capital * (1 + values).cumprod()
            assert np.allclose(reconstructed, source.equity, rtol=1e-10, atol=1e-9)
            if "daily_return" in source:
                checks.append({"strategy": name, "source_equity_end": source.equity.iloc[-1] / initial_capital,
                               "csv_daily_return_compounded": (1 + source.daily_return).prod(),
                               "max_daily_return_difference": (values - source.daily_return).abs().max(),
                               "used": "returns derived from after-cost equity"})
        else:
            values = source.daily_return
        frame = pd.DataFrame({"daily_return": values, "source": path})
        if "total_weight" in source:
            frame["exposure"] = source.total_weight
        elif "invested_ratio" in source:
            frame["exposure"] = source.invested_ratio
        elif "exposure" in source:
            frame["exposure"] = source.exposure
        elif "position" in source:
            frame["exposure"] = source.position
        elif "new_position" in source:
            frame["exposure"] = source.new_position
        else:
            frame["exposure"] = np.nan
        result[name] = frame
    pd.DataFrame(checks).to_csv(OUT / "daily_source_reconciliation.csv", index=False)
    return result


def analyze_daily(market: pd.DataFrame) -> None:
    from run_strategy_lab import metrics
    sources = daily_sources()
    common = None
    for frame in sources.values():
        candidate = frame.index[frame.index >= pd.Timestamp("2024-01-03")]
        common = candidate if common is None else common.intersection(candidate)
    common = common.sort_values()
    rows, strata, annual = [], [], []
    market = market.set_index("date")
    prior = market[["market_fast", "market_slow"]].shift(1)
    for name, frame in sources.items():
        for period, sample in (("native", frame), ("common_2024_on", frame.reindex(common)),
                               ("2026", frame.loc["2026-01-01":])):
            if sample.empty:
                continue
            rows.append({"strategy": name, "period": period, "start": str(sample.index.min().date()),
                         "end": str(sample.index.max().date()), "days": len(sample),
                         "source": frame.source.iloc[0], **metrics(sample.daily_return, sample.exposure)})
        work = frame.join(prior)
        for field in ("market_fast", "market_slow"):
            for state, group in work.groupby(field):
                strata.append({"strategy": name, "horizon": field, "state": state, "days": len(group),
                               "mean_daily_return": group.daily_return.mean(),
                               "positive_day_rate": (group.daily_return > 0).mean(),
                               "worst_day": group.daily_return.min(), "mean_exposure": group.exposure.mean()})
        for year, group in frame.groupby(frame.index.year):
            annual.append({"strategy": name, "year": year, "days": len(group),
                           "calendar_year_return": (1 + group.daily_return).prod() - 1})
    pd.DataFrame(rows).to_csv(OUT / "daily_equity_comparison.csv", index=False)
    pd.DataFrame(strata).to_csv(OUT / "daily_market_strata.csv", index=False)
    pd.DataFrame(annual).to_csv(OUT / "daily_annual_returns.csv", index=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    market, industry = build_context()
    events, inventory = event_sources()
    events = attach_context(events, market, industry)
    assert (events.context_date.dropna() <= events.loc[events.context_date.notna(), "date"]).all()
    events.to_parquet(OUT / "events_with_context.parquet", index=False)
    pd.DataFrame(inventory).to_csv(OUT / "strategy_inventory.csv", index=False)
    analyze_events(events)
    analyze_daily(market)
    manifest = {"date": "2026-09-30", "source": "existing local frozen outputs; Tushare market data",
                "event_strategies": len(inventory), "event_rows": len(events),
                "market_missing": int(events.market_trend.eq("unknown").sum()),
                "sector_missing": int(events.sector_trend.eq("unknown").sum()),
                "sector_proxy": "daily equal-weight pct_chg of locally quoted industry members listed >=365 days",
                "mapping_caveat": "cached/current industry mapping is not a vintage industry classification",
                "validation_status": "retrospective diagnostics; no new untouched holdout",
                "production_action": "none"}
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
