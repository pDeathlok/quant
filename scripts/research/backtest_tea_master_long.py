"""Backtest an independent Tea Master long-term strategy.

This script intentionally keeps Tea Master's rules separate from the existing
L1 dividend-quality strategy family. It reuses the project's local qfq daily
data and point-in-time financial loaders, but owns its scoring, position model,
T overlay, and optimization report.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_long_dividend_quality import (
    PROJECT_ROOT,
    REPORT_DIR as L1_REPORT_DIR,
    add_empty_analyst_forecast_columns,
    filter_daily_basic_point_in_time,
    load_benchmark,
    load_daily_basic_monthly,
    load_daily_monthly_features,
    load_financial_asof,
    load_market_regime,
    load_stock_basic,
    max_drawdown,
    parse_date,
    percentile_score,
    summarize,
)


REPORT_DIR = PROJECT_ROOT / "reports/tea_master_long"


@dataclass(frozen=True)
class TeaConfig:
    name: str
    core_slots_risk_on: int
    core_slots_neutral: int
    core_slots_risk_off: int
    satellite_slots: int
    risk_on_weight: float
    neutral_weight: float
    risk_off_weight: float
    satellite_weight: float
    min_tea_score: float
    min_quality_score: float
    min_trend_score: float
    max_symbol_weight: float
    allow_satellite: bool = True
    use_t_overlay: bool = True
    overheat_throttle: bool = True
    max_industry_positions: int | None = None
    risk_off_trend_exit: bool = False
    drawdown_brake: bool = False
    pullback_weight_multiplier: float = 0.70
    pullback_slot_multiplier: float = 0.65
    overheat_weight_multiplier: float = 0.86


CONFIGS = [
    TeaConfig(
        name="core3_original",
        core_slots_risk_on=3,
        core_slots_neutral=3,
        core_slots_risk_off=1,
        satellite_slots=0,
        risk_on_weight=0.75,
        neutral_weight=0.55,
        risk_off_weight=0.15,
        satellite_weight=0.0,
        min_tea_score=74,
        min_quality_score=58,
        min_trend_score=55,
        max_symbol_weight=0.25,
        allow_satellite=False,
    ),
    TeaConfig(
        name="core6_balanced",
        core_slots_risk_on=6,
        core_slots_neutral=5,
        core_slots_risk_off=2,
        satellite_slots=0,
        risk_on_weight=0.78,
        neutral_weight=0.58,
        risk_off_weight=0.20,
        satellite_weight=0.0,
        min_tea_score=72,
        min_quality_score=56,
        min_trend_score=52,
        max_symbol_weight=0.16,
        allow_satellite=False,
    ),
    TeaConfig(
        name="core8_grid",
        core_slots_risk_on=8,
        core_slots_neutral=6,
        core_slots_risk_off=3,
        satellite_slots=0,
        risk_on_weight=0.80,
        neutral_weight=0.60,
        risk_off_weight=0.24,
        satellite_weight=0.0,
        min_tea_score=70,
        min_quality_score=55,
        min_trend_score=50,
        max_symbol_weight=0.12,
        allow_satellite=False,
    ),
    TeaConfig(
        name="core8_plus_satellite",
        core_slots_risk_on=8,
        core_slots_neutral=6,
        core_slots_risk_off=3,
        satellite_slots=2,
        risk_on_weight=0.72,
        neutral_weight=0.58,
        risk_off_weight=0.24,
        satellite_weight=0.08,
        min_tea_score=70,
        min_quality_score=55,
        min_trend_score=50,
        max_symbol_weight=0.12,
        allow_satellite=True,
    ),
    TeaConfig(
        name="core10_defensive",
        core_slots_risk_on=10,
        core_slots_neutral=8,
        core_slots_risk_off=4,
        satellite_slots=0,
        risk_on_weight=0.72,
        neutral_weight=0.56,
        risk_off_weight=0.28,
        satellite_weight=0.0,
        min_tea_score=70,
        min_quality_score=56,
        min_trend_score=48,
        max_symbol_weight=0.10,
        allow_satellite=False,
    ),
    TeaConfig(
        name="core10_industry_cap",
        core_slots_risk_on=10,
        core_slots_neutral=8,
        core_slots_risk_off=4,
        satellite_slots=0,
        risk_on_weight=0.72,
        neutral_weight=0.56,
        risk_off_weight=0.28,
        satellite_weight=0.0,
        min_tea_score=70,
        min_quality_score=56,
        min_trend_score=48,
        max_symbol_weight=0.10,
        allow_satellite=False,
        max_industry_positions=2,
    ),
    TeaConfig(
        name="core10_drawdown_brake",
        core_slots_risk_on=10,
        core_slots_neutral=8,
        core_slots_risk_off=3,
        satellite_slots=0,
        risk_on_weight=0.68,
        neutral_weight=0.50,
        risk_off_weight=0.16,
        satellite_weight=0.0,
        min_tea_score=70,
        min_quality_score=58,
        min_trend_score=50,
        max_symbol_weight=0.10,
        allow_satellite=False,
        max_industry_positions=2,
        risk_off_trend_exit=True,
        drawdown_brake=True,
    ),
    TeaConfig(
        name="core12_quality_spread",
        core_slots_risk_on=12,
        core_slots_neutral=10,
        core_slots_risk_off=4,
        satellite_slots=0,
        risk_on_weight=0.72,
        neutral_weight=0.54,
        risk_off_weight=0.22,
        satellite_weight=0.0,
        min_tea_score=69,
        min_quality_score=60,
        min_trend_score=48,
        max_symbol_weight=0.08,
        allow_satellite=False,
        max_industry_positions=2,
        drawdown_brake=True,
    ),
    TeaConfig(
        name="core10_soft_brake",
        core_slots_risk_on=10,
        core_slots_neutral=8,
        core_slots_risk_off=3,
        satellite_slots=0,
        risk_on_weight=0.72,
        neutral_weight=0.54,
        risk_off_weight=0.22,
        satellite_weight=0.0,
        min_tea_score=70,
        min_quality_score=57,
        min_trend_score=50,
        max_symbol_weight=0.10,
        allow_satellite=False,
        max_industry_positions=2,
        risk_off_trend_exit=True,
        drawdown_brake=True,
        pullback_weight_multiplier=0.84,
        pullback_slot_multiplier=0.85,
        overheat_weight_multiplier=0.92,
    ),
    TeaConfig(
        name="core10_industry_no_t",
        core_slots_risk_on=10,
        core_slots_neutral=8,
        core_slots_risk_off=4,
        satellite_slots=0,
        risk_on_weight=0.72,
        neutral_weight=0.56,
        risk_off_weight=0.28,
        satellite_weight=0.0,
        min_tea_score=70,
        min_quality_score=56,
        min_trend_score=48,
        max_symbol_weight=0.10,
        allow_satellite=False,
        use_t_overlay=False,
        max_industry_positions=2,
    ),
    TeaConfig(
        name="core12_soft_brake",
        core_slots_risk_on=12,
        core_slots_neutral=9,
        core_slots_risk_off=4,
        satellite_slots=0,
        risk_on_weight=0.76,
        neutral_weight=0.56,
        risk_off_weight=0.22,
        satellite_weight=0.0,
        min_tea_score=69,
        min_quality_score=58,
        min_trend_score=48,
        max_symbol_weight=0.085,
        allow_satellite=False,
        max_industry_positions=2,
        drawdown_brake=True,
        pullback_weight_multiplier=0.86,
        pullback_slot_multiplier=0.85,
        overheat_weight_multiplier=0.92,
    ),
    TeaConfig(
        name="core12_soft_brake_plus",
        core_slots_risk_on=12,
        core_slots_neutral=9,
        core_slots_risk_off=4,
        satellite_slots=0,
        risk_on_weight=0.80,
        neutral_weight=0.58,
        risk_off_weight=0.22,
        satellite_weight=0.0,
        min_tea_score=69,
        min_quality_score=58,
        min_trend_score=48,
        max_symbol_weight=0.09,
        allow_satellite=False,
        max_industry_positions=2,
        drawdown_brake=True,
        pullback_weight_multiplier=0.86,
        pullback_slot_multiplier=0.85,
        overheat_weight_multiplier=0.92,
    ),
    TeaConfig(
        name="core14_soft_spread",
        core_slots_risk_on=14,
        core_slots_neutral=10,
        core_slots_risk_off=4,
        satellite_slots=0,
        risk_on_weight=0.78,
        neutral_weight=0.58,
        risk_off_weight=0.22,
        satellite_weight=0.0,
        min_tea_score=68,
        min_quality_score=58,
        min_trend_score=48,
        max_symbol_weight=0.075,
        allow_satellite=False,
        max_industry_positions=2,
        drawdown_brake=True,
        pullback_weight_multiplier=0.86,
        pullback_slot_multiplier=0.85,
        overheat_weight_multiplier=0.92,
    ),
    TeaConfig(
        name="core12_soft_strict_exit",
        core_slots_risk_on=12,
        core_slots_neutral=9,
        core_slots_risk_off=3,
        satellite_slots=0,
        risk_on_weight=0.76,
        neutral_weight=0.56,
        risk_off_weight=0.18,
        satellite_weight=0.0,
        min_tea_score=69,
        min_quality_score=58,
        min_trend_score=48,
        max_symbol_weight=0.085,
        allow_satellite=False,
        max_industry_positions=2,
        risk_off_trend_exit=True,
        drawdown_brake=True,
        pullback_weight_multiplier=0.86,
        pullback_slot_multiplier=0.85,
        overheat_weight_multiplier=0.92,
    ),
    TeaConfig(
        name="core14_soft_plus",
        core_slots_risk_on=14,
        core_slots_neutral=10,
        core_slots_risk_off=4,
        satellite_slots=0,
        risk_on_weight=0.84,
        neutral_weight=0.62,
        risk_off_weight=0.24,
        satellite_weight=0.0,
        min_tea_score=68,
        min_quality_score=58,
        min_trend_score=48,
        max_symbol_weight=0.08,
        allow_satellite=False,
        max_industry_positions=2,
        drawdown_brake=True,
        pullback_weight_multiplier=0.86,
        pullback_slot_multiplier=0.85,
        overheat_weight_multiplier=0.92,
    ),
    TeaConfig(
        name="core16_soft_spread",
        core_slots_risk_on=16,
        core_slots_neutral=12,
        core_slots_risk_off=5,
        satellite_slots=0,
        risk_on_weight=0.80,
        neutral_weight=0.60,
        risk_off_weight=0.24,
        satellite_weight=0.0,
        min_tea_score=67,
        min_quality_score=58,
        min_trend_score=48,
        max_symbol_weight=0.065,
        allow_satellite=False,
        max_industry_positions=2,
        drawdown_brake=True,
        pullback_weight_multiplier=0.86,
        pullback_slot_multiplier=0.85,
        overheat_weight_multiplier=0.92,
    ),
]


def build_tea_scores(features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    out["listing_years"] = (out["date"] - out["list_date"]).dt.days / 365.25
    out["target_price"] = pd.concat(
        [out["ma_60"] * 1.02, out["ma_120"] * 1.05, out["median_close_60"]],
        axis=1,
    ).min(axis=1)
    out["near_year_line"] = out["close"].between(out["ma_120"] * 0.98, out["target_price"] * 1.10)
    out["pullback_or_breakout"] = (
        out["near_year_line"]
        | ((out["close"] > out["ma_120"]) & (out["return_120d"].between(0.0, 0.35)))
    )

    scored: list[pd.DataFrame] = []
    for _, group in out.groupby("date", sort=True):
        group = group.copy()
        group["quality_score"] = (
            percentile_score(group["roe"], True) * 0.35
            + percentile_score(group["netprofit_margin"], True) * 0.20
            + percentile_score(group["or_yoy"], True) * 0.15
            + percentile_score(group["debt_to_assets"], False) * 0.20
            + percentile_score(group["dv_ttm_stability_36m"], True) * 0.10
        ).fillna(50)
        group["value_score"] = (
            percentile_score(1 / group["pe_ttm"].where(group["pe_ttm"] > 0), True) * 0.45
            + percentile_score(1 / group["pb"].where(group["pb"] > 0), True) * 0.35
            + percentile_score(group["dv_ttm"].clip(0, 8), True) * 0.20
        ).fillna(50)
        group["trend_score"] = (
            percentile_score(group["return_120d"].clip(-0.5, 1.0), True) * 0.35
            + percentile_score(group["ma_120_slope_20d"].clip(-0.10, 0.20), True) * 0.35
            + (group["close"] > group["ma_120"]).astype(float) * 20
            + group["near_year_line"].astype(float) * 10
        ).clip(0, 100)
        group["risk_score"] = (
            percentile_score(group["volatility_60d"], False) * 0.45
            + percentile_score(group["downside_volatility_60d"], False) * 0.45
            + percentile_score(group["turnover_rate"].clip(0, 10), False) * 0.10
        ).fillna(50)
        group["volume_score"] = (
            percentile_score(group["turnover_rate"].clip(0, 8), True) * 0.50
            + percentile_score(group["circ_mv"], True) * 0.20
            + percentile_score(group["amount"].fillna(0) if "amount" in group.columns else group["total_mv"], True) * 0.30
        ).fillna(50)
        group["tea_score"] = (
            group["quality_score"] * 0.28
            + group["value_score"] * 0.18
            + group["trend_score"] * 0.28
            + group["risk_score"] * 0.16
            + group["volume_score"] * 0.10
        ).clip(0, 100)
        group["satellite_score"] = (
            percentile_score(group["or_yoy"], True) * 0.24
            + percentile_score(group["basic_eps_yoy"], True) * 0.22
            + group["quality_score"] * 0.20
            + group["trend_score"] * 0.24
            + group["risk_score"] * 0.10
        ).clip(0, 100)
        scored.append(group)
    return pd.concat(scored, ignore_index=True)


def select_targets(
    scored: pd.DataFrame,
    config: TeaConfig,
    initial_current: set[str] | None = None,
) -> pd.DataFrame:
    targets: list[pd.DataFrame] = []
    current: set[str] = set(initial_current or ())
    for date, group in scored.groupby("date", sort=True):
        group = group.copy()
        regime = str(group["market_regime"].dropna().iloc[0]) if not group["market_regime"].dropna().empty else "neutral"
        overheat = bool(group["index_overheat"].fillna(False).iloc[0]) if "index_overheat" in group.columns else False
        pullback_warning = bool(
            (
                (group["index_return_20d"].fillna(0).iloc[0] < -0.04)
                or (group["index_drawdown_60d"].fillna(0).iloc[0] < -0.08)
            )
            if "index_return_20d" in group.columns and "index_drawdown_60d" in group.columns
            else False
        )
        if regime == "risk_on":
            core_slots = config.core_slots_risk_on
            core_weight = config.risk_on_weight
        elif regime == "risk_off":
            core_slots = config.core_slots_risk_off
            core_weight = config.risk_off_weight
        else:
            core_slots = config.core_slots_neutral
            core_weight = config.neutral_weight
        satellite_slots = config.satellite_slots if config.allow_satellite and regime == "risk_on" and not overheat and not pullback_warning else 0
        satellite_weight = config.satellite_weight if satellite_slots else 0.0
        if config.overheat_throttle and regime == "risk_on":
            if pullback_warning:
                core_weight *= 0.72
                satellite_weight = 0.0
            elif overheat:
                core_weight *= 0.82
                satellite_weight *= 0.25
        if config.drawdown_brake:
            if pullback_warning:
                core_weight *= config.pullback_weight_multiplier
                satellite_weight = 0.0
                core_slots = max(1, int(core_slots * config.pullback_slot_multiplier))
            elif overheat:
                core_weight *= config.overheat_weight_multiplier
                satellite_weight *= 0.25

        group_by_symbol = group.set_index("ts_code", drop=False)
        survivors: list[str] = []
        for symbol in current:
            if symbol not in group_by_symbol.index:
                continue
            row = group_by_symbol.loc[symbol]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]
            exit_signal = (
                row["tea_score"] < 55
                or row["close"] < row["ma_120"] * (0.94 if regime == "risk_on" else 0.98)
                or row["ma_120_slope_20d"] < -0.04
                or (regime == "risk_off" and row["risk_score"] < 35)
                or (
                    config.risk_off_trend_exit
                    and regime != "risk_on"
                    and (row["trend_score"] < 45 or row["close"] < row["ma_120"] * 1.01)
                )
            )
            if not exit_signal:
                survivors.append(symbol)

        base = (
            (group["total_mv"] >= 800000)
            & (group["circ_mv"] >= 500000)
            & (group["listing_years"] >= 2)
            & (group["turnover_rate"].between(0.4, 8.0, inclusive="both"))
            & (group["close"] >= group["ma_120"] * (0.96 if regime == "risk_on" else 0.99))
            & (group["ma_120_slope_20d"] >= (-0.015 if regime == "risk_on" else 0.0))
            & (group["pullback_or_breakout"])
            & (group["tea_score"] >= config.min_tea_score)
            & (group["quality_score"] >= config.min_quality_score)
            & (group["trend_score"] >= config.min_trend_score)
            & (group["risk_score"] >= 25)
            & (group["debt_to_assets"].fillna(100) <= 88)
            & (group["pe_ttm"].fillna(999) <= 90)
            & (group["pb"].fillna(999) <= 12)
        )
        if regime == "risk_off":
            base &= (group["quality_score"] >= 65) & (group["risk_score"] >= 55) & (group["value_score"] >= 60)

        core_pool = group[base & ~group["ts_code"].isin(survivors)].sort_values(
            ["tea_score", "trend_score", "quality_score", "risk_score"],
            ascending=False,
        )
        if config.max_industry_positions is not None and not core_pool.empty:
            current_industry_counts = (
                group[group["ts_code"].isin(survivors)]["industry"].value_counts(dropna=False).to_dict()
            )
            capped_rows: list[pd.Series] = []
            for _, candidate in core_pool.iterrows():
                industry = candidate.get("industry")
                used = int(current_industry_counts.get(industry, 0))
                if used >= config.max_industry_positions:
                    continue
                capped_rows.append(candidate)
                current_industry_counts[industry] = used + 1
                if len(capped_rows) >= max(core_slots - len(survivors), 0):
                    break
            core_pool = pd.DataFrame(capped_rows) if capped_rows else core_pool.head(0)
        entrants = core_pool.head(max(core_slots - len(survivors), 0))["ts_code"].tolist()

        satellite: list[str] = []
        if satellite_slots:
            sat_mask = (
                base
                & ~group["ts_code"].isin(survivors)
                & ~group["ts_code"].isin(entrants)
                & (group["satellite_score"] >= 78)
                & (group["trend_score"] >= 68)
                & (group["quality_score"] >= 58)
                & ((group["or_yoy"] >= 12) | (group["basic_eps_yoy"] >= 15))
            )
            satellite = group[sat_mask].sort_values(
                ["satellite_score", "trend_score", "quality_score"],
                ascending=False,
            ).head(satellite_slots)["ts_code"].tolist()

        selected_symbols = survivors + entrants + satellite
        if not selected_symbols:
            current = set()
            continue
        selected = group[group["ts_code"].isin(selected_symbols)].copy()
        selected["sleeve"] = np.where(selected["ts_code"].isin(satellite), "satellite", "core")
        selected["target_weight"] = 0.0
        core_mask = selected["sleeve"] == "core"
        sat_mask = selected["sleeve"] == "satellite"
        if core_mask.any():
            selected.loc[core_mask, "target_weight"] = min(config.max_symbol_weight, core_weight / int(core_mask.sum()))
        if sat_mask.any() and satellite_weight > 0:
            selected.loc[sat_mask, "target_weight"] = min(0.05, satellite_weight / int(sat_mask.sum()))
        selected["rebalance_date"] = date
        selected = selected[selected["target_weight"] > 0].copy()
        current = set(selected["ts_code"])
        targets.append(
            selected[
                [
                    "rebalance_date",
                    "trade_date",
                    "ts_code",
                    "name",
                    "industry",
                    "market_regime",
                    "sleeve",
                    "close",
                    "target_price",
                    "target_weight",
                    "tea_score",
                    "quality_score",
                    "value_score",
                    "trend_score",
                    "risk_score",
                    "satellite_score",
                    "dv_ttm",
                    "pe_ttm",
                    "pb",
                    "roe",
                ]
            ]
        )
    return pd.concat(targets, ignore_index=True) if targets else pd.DataFrame()


def run_tea_backtest(
    daily_returns: pd.DataFrame,
    targets: pd.DataFrame,
    scored: pd.DataFrame,
    config: TeaConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = daily_returns.sort_values(["date", "ts_code"]).copy()
    daily["ret_1d"] = pd.to_numeric(daily["ret_1d"], errors="coerce").fillna(0.0)
    dates = sorted(daily["date"].dropna().unique())
    target_map = {date: frame.copy() for date, frame in targets.groupby("rebalance_date")}
    dv_lookup = scored.set_index(["date", "ts_code"])["dv_ttm"].to_dict()
    daily_by_date = {date: frame for date, frame in daily.groupby("date", sort=True)}

    equity = 1.0
    current_weights: dict[str, float] = {}
    base_weights: dict[str, float] = {}
    rows: list[dict] = []
    trades: list[dict] = []
    current_regime = "neutral"

    for date in dates:
        if date in target_map:
            target = target_map[date]
            next_weights = dict(zip(target["ts_code"], target["target_weight"], strict=False))
            turnover = sum(abs(next_weights.get(s, 0.0) - current_weights.get(s, 0.0)) for s in set(next_weights) | set(current_weights))
            sell_turnover = sum(max(current_weights.get(s, 0.0) - next_weights.get(s, 0.0), 0.0) for s in set(next_weights) | set(current_weights))
            cost = turnover * 0.0003 + sell_turnover * 0.0005
            equity *= 1.0 - cost
            current_weights = next_weights
            base_weights = next_weights.copy()
            current_regime = str(target["market_regime"].dropna().iloc[0]) if not target["market_regime"].dropna().empty else current_regime
            trades.append({"date": date, "kind": "monthly_rebalance", "turnover": turnover, "sell_turnover": sell_turnover, "cost": cost, "positions": len(current_weights), "total_weight": sum(current_weights.values())})

        day = daily_by_date.get(date)
        price_return = 0.0
        dividend_return = 0.0
        if day is not None and current_weights:
            ret_map = dict(zip(day["ts_code"], day["ret_1d"], strict=False))
            for symbol, weight in current_weights.items():
                price_return += weight * float(ret_map.get(symbol, 0.0))
                dv_ttm = dv_lookup.get((date, symbol))
                if pd.notna(dv_ttm):
                    dividend_return += weight * float(dv_ttm) / 100.0 / 252.0
        equity *= 1.0 + price_return + dividend_return

        if config.use_t_overlay and day is not None and current_weights:
            day_by_symbol = day.set_index("ts_code", drop=False)
            adjusted = current_weights.copy()
            turnover = 0.0
            sell_turnover = 0.0
            actions = 0
            for symbol, base_weight in base_weights.items():
                if symbol not in adjusted or symbol not in day_by_symbol.index:
                    continue
                row = day_by_symbol.loc[symbol]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[-1]
                close = float(row.get("close", np.nan))
                ma20 = float(row.get("ma_20", np.nan))
                ma60 = float(row.get("ma_60", np.nan))
                ma120 = float(row.get("ma_120", np.nan))
                if not all(np.isfinite(v) for v in [close, ma20, ma60, ma120]):
                    continue
                core_ratio = 0.82 if current_regime != "risk_on" else 0.78
                sell_signal = close > ma20 * 1.12 or close > ma60 * 1.18
                buy_signal = current_regime != "risk_off" and close >= ma120 * 0.98 and (close <= ma20 * 1.015 or close <= ma60 * 1.035)
                core_weight = base_weight * core_ratio
                new_weight = adjusted[symbol]
                if sell_signal and new_weight > core_weight:
                    new_weight = core_weight
                elif buy_signal and new_weight < base_weight:
                    new_weight = base_weight
                if abs(new_weight - adjusted[symbol]) > 1e-9:
                    delta = abs(new_weight - adjusted[symbol])
                    turnover += delta
                    sell_turnover += max(adjusted[symbol] - new_weight, 0.0)
                    adjusted[symbol] = new_weight
                    actions += 1
            if actions:
                cost = turnover * 0.0003 + sell_turnover * 0.0005
                equity *= 1.0 - cost
                current_weights = adjusted
                trades.append({"date": date, "kind": "t_overlay", "turnover": turnover, "sell_turnover": sell_turnover, "cost": cost, "positions": len(current_weights), "total_weight": sum(current_weights.values()), "actions": actions})

        rows.append({"date": date, "equity": equity, "daily_return": price_return + dividend_return, "price_return": price_return, "dividend_return": dividend_return, "positions": len(current_weights), "total_weight": sum(current_weights.values()) if current_weights else 0.0, "market_regime": current_regime})

    return pd.DataFrame(rows), pd.DataFrame(trades)


def prepare_data(start: str = "20130101", end: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    requested_start = parse_date(start)
    requested_end = parse_date(end)
    if requested_start is None:
        raise ValueError("start must use YYYYMMDD")
    stock_basic = load_stock_basic()
    daily_basic, coverage = load_daily_basic_monthly(requested_start, requested_end)
    coverage["point_in_time_universe"] = True
    daily_features, daily_returns = load_daily_monthly_features(requested_start, requested_end, stock_basic, candidate_symbols=None)
    executable_start_text = max(start, coverage.get("first_trade_date") or start)
    executable_start = parse_date(executable_start_text)
    daily_features = daily_features[daily_features["date"] >= executable_start].copy()
    daily_returns = daily_returns[daily_returns["date"] >= executable_start].copy()
    daily_basic = daily_basic[daily_basic["date"] >= executable_start].copy()
    before_rows = len(daily_basic)
    daily_basic = filter_daily_basic_point_in_time(daily_basic, type("Config", (), {"variant": "tea", "prefilter_min_dv_ttm": 0.0, "prefilter_min_total_mv": 800000.0, "prefilter_min_circ_mv": 500000.0})())
    coverage["point_in_time_universe_rows_before"] = int(before_rows)
    coverage["point_in_time_universe_rows_after"] = int(len(daily_basic))
    merged = daily_features.merge(daily_basic.drop(columns=["trade_date"]), on=["date", "ts_code"], how="inner")
    merged = load_financial_asof(merged)
    merged = add_empty_analyst_forecast_columns(merged)
    market_regime = load_market_regime(merged["date"].min(), merged["date"].max())
    merged = merged.merge(market_regime, on="date", how="left")
    merged["market_regime"] = merged["market_regime"].fillna("neutral")
    for column in ["index_return_20d", "index_return_60d", "index_drawdown_60d"]:
        if column not in merged.columns:
            merged[column] = np.nan
    if "index_overheat" not in merged.columns:
        merged["index_overheat"] = False
    return merged, daily_returns, stock_basic, coverage


def write_outputs(config: TeaConfig, coverage: dict, summary: dict, targets: pd.DataFrame, equity: pd.DataFrame, trades: pd.DataFrame) -> None:
    out_dir = REPORT_DIR / config.name
    out_dir.mkdir(parents=True, exist_ok=True)
    targets.to_csv(out_dir / "tea_master_targets.csv", index=False)
    equity.to_csv(out_dir / "tea_master_equity.csv", index=False)
    trades.to_csv(out_dir / "tea_master_trades.csv", index=False)
    (out_dir / "tea_master_summary.json").write_text(json.dumps({"config": asdict(config), "coverage": coverage, "summary": summary}, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    merged, daily_returns, _, coverage = prepare_data()
    scored = build_tea_scores(merged)
    rows: list[dict] = []
    for config in CONFIGS:
        print(f"running {config.name}", flush=True)
        targets = select_targets(scored, config)
        if targets.empty:
            continue
        equity, trades = run_tea_backtest(daily_returns, targets, scored, config)
        benchmark = load_benchmark(equity["date"].min(), equity["date"].max())
        summary = summarize(equity, trades, benchmark)
        write_outputs(config, coverage, summary, targets, equity, trades)
        rows.append({"variant": config.name, **summary})

    result = pd.DataFrame(rows).sort_values(["sharpe", "annual_return"], ascending=False)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(REPORT_DIR / "optimization_summary.csv", index=False)
    lines = ["# Tea Master Long Optimization", "", result.to_markdown(index=False)]
    (REPORT_DIR / "optimization_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(result[["variant", "annual_return", "max_drawdown", "sharpe", "avg_positions", "avg_total_weight"]].to_string(index=False))


if __name__ == "__main__":
    main()
