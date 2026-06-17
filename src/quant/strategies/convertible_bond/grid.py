from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from quant.strategies.convertible_bond.backtest import (
    ConvertibleBondBacktestConfig,
    ConvertibleBondBacktestResult,
    _active_basic,
    _active_call,
    _portfolio_return,
    _prepare_basic,
    _prepare_call,
    _prepare_daily,
    _should_rebalance,
    _turnover,
    summarize_backtest,
)
from quant.strategies.convertible_bond.rotation import (
    ConvertibleBondFilterConfig,
    ConvertibleBondRotationConfig,
    ConvertibleBondSelector,
    compare_credit_rating,
)
from quant.strategies.convertible_bond.trend_enhanced import add_trend_enhanced_features


@dataclass(frozen=True)
class ConservativeGridConfig:
    name: str = "cb_conservative_grid"
    top_n: int | None = 8
    max_total_weight: float = 0.85
    max_position_weight: float = 0.14
    max_entry_price: float = 118.0
    min_premium_rate: float = -10.0
    max_premium_rate: float = 25.0
    max_double_low: float = 140.0
    max_price_position_252: float = 0.35
    min_drawdown_from_252_high: float = 0.08
    min_amount: float = 2_000.0
    min_remaining_size: float = 1.0
    min_credit_rating: str = "AA-"
    grid_full_price: float = 106.0
    grid_large_price: float = 110.0
    grid_half_price: float = 114.0
    grid_small_price: float = 118.0
    deep_low_position: float = 0.15
    premium_bonus_threshold: float = 12.0
    min_momentum_20d: float | None = None
    score_floor: float = 0.0
    dynamic_grid: bool = False
    low_risk_grid_step: float = 0.030
    medium_risk_grid_step: float = 0.040
    high_risk_grid_step: float = 0.055
    low_risk_position_scale: float = 1.00
    medium_risk_position_scale: float = 0.80
    high_risk_position_scale: float = 0.55

    @property
    def selector_config(self) -> ConvertibleBondRotationConfig:
        selector_top_n = 9999 if self.top_n is None else max(self.top_n * 3, self.top_n)
        return ConvertibleBondRotationConfig(
            top_n=selector_top_n,
            max_position_weight=self.max_position_weight,
            min_score=self.score_floor,
            filter=ConvertibleBondFilterConfig(
                min_price=100.0,
                max_price=self.max_entry_price,
                max_premium_rate=self.max_premium_rate,
                min_amount=self.min_amount,
                min_remaining_size=self.min_remaining_size,
                min_credit_rating=self.min_credit_rating,
                exclude_call_risk=True,
                exclude_not_convertible=True,
            ),
        )


class ConservativeGridStrategy:
    """Only builds sizeable positions when a convertible bond is historically low."""

    def __init__(self, config: ConservativeGridConfig):
        self.config = config
        self.selector = ConvertibleBondSelector(config.selector_config)

    def target_portfolio(
        self,
        daily: pd.DataFrame,
        basic: pd.DataFrame,
        call: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        universe = self.selector.build_universe(daily=daily, basic=basic, call=call)
        if universe.empty:
            return universe
        filtered = self.selector._apply_filters(universe)
        if filtered.empty:
            return filtered
        filtered = self._apply_low_filters(filtered)
        if filtered.empty:
            return filtered
        scored = self._score(filtered)
        selected = scored.sort_values(
            ["grid_score", "double_low", "price_position_252"],
            ascending=[False, True, True],
        )
        if self.config.top_n is not None:
            selected = selected.head(self.config.top_n)
        if selected.empty:
            return selected
        selected = selected.copy()
        selected["grid_multiplier"] = selected.apply(self._grid_multiplier, axis=1)
        selected["risk_level"] = selected.apply(self._risk_level, axis=1)
        selected["grid_step_pct"] = selected.apply(self._grid_step, axis=1)
        selected["position_scale"] = selected.apply(self._risk_position_scale, axis=1)
        selected["target_weight"] = (
            selected["grid_multiplier"]
            * selected["position_scale"]
            * self.config.max_position_weight
        )
        selected = selected[selected["target_weight"] > 0].copy()
        total_weight = float(selected["target_weight"].sum())
        if total_weight > self.config.max_total_weight and total_weight > 0:
            selected["target_weight"] *= self.config.max_total_weight / total_weight
        selected["rank"] = np.arange(1, len(selected) + 1)
        return selected

    def _apply_low_filters(self, frame: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        required = frame.copy()
        if "double_low" not in required.columns:
            required["double_low"] = required["close"] + required["premium_rate"]
        mask = (
            (required["close"] <= cfg.max_entry_price)
            & (required["premium_rate"] >= cfg.min_premium_rate)
            & (required["premium_rate"] <= cfg.max_premium_rate)
            & (required["double_low"] <= cfg.max_double_low)
        )
        if "price_position_252" in required.columns:
            mask &= required["price_position_252"].fillna(1.0) <= cfg.max_price_position_252
        if "drawdown_from_252_high" in required.columns:
            mask &= required["drawdown_from_252_high"].fillna(0.0) <= -cfg.min_drawdown_from_252_high
        if cfg.min_momentum_20d is not None and "momentum_20d" in required.columns:
            mask &= required["momentum_20d"].fillna(-1.0) >= cfg.min_momentum_20d
        return required[mask].copy()

    def _score(self, frame: pd.DataFrame) -> pd.DataFrame:
        scored = self.selector._score(frame)
        low_position = 1.0 - scored.get("price_position_252", pd.Series(1.0, index=scored.index)).fillna(1.0)
        drawdown = scored.get("drawdown_from_252_high", pd.Series(0.0, index=scored.index)).fillna(0.0).abs().clip(0, 0.4) / 0.4
        premium_score = 1.0 - (scored["premium_rate"].clip(0, self.config.max_premium_rate) / self.config.max_premium_rate)
        scored["grid_score"] = (
            0.35 * scored["double_low_score"]
            + 0.25 * low_position
            + 0.20 * drawdown
            + 0.10 * premium_score
            + 0.10 * scored["liquidity_score"]
        )
        return scored

    def _grid_multiplier(self, row: pd.Series) -> float:
        close = float(row["close"])
        if close <= self.config.grid_full_price:
            multiplier = 1.0
        elif close <= self.config.grid_large_price:
            multiplier = 0.75
        elif close <= self.config.grid_half_price:
            multiplier = 0.50
        elif close <= self.config.grid_small_price:
            multiplier = 0.25
        else:
            multiplier = 0.0
        if float(row.get("price_position_252", 1.0)) <= self.config.deep_low_position:
            multiplier = min(1.0, multiplier + 0.15)
        if float(row.get("premium_rate", 100.0)) <= self.config.premium_bonus_threshold:
            multiplier = min(1.0, multiplier + 0.10)
        return float(multiplier)

    def _risk_level(self, row: pd.Series) -> str:
        if not self.config.dynamic_grid:
            return "standard"
        close = float(row.get("close", np.nan))
        premium = float(row.get("premium_rate", np.nan))
        double_low = float(row.get("double_low", np.nan))
        price_position = float(row.get("price_position_252", np.nan))
        amount = float(row.get("amount", np.nan))
        rating = str(row.get("credit_rating", "") or "")
        high_risk = (
            (np.isfinite(close) and close >= 114.0)
            or (np.isfinite(premium) and premium >= 20.0)
            or (np.isfinite(double_low) and double_low >= 136.0)
            or (np.isfinite(price_position) and price_position >= 0.30)
            or (np.isfinite(amount) and amount < 3_000.0)
            or compare_credit_rating(rating, "AA-") < 0
        )
        if high_risk:
            return "high"
        low_risk = (
            (np.isfinite(close) and close <= 110.0)
            and (np.isfinite(premium) and premium <= 12.0)
            and (np.isfinite(double_low) and double_low <= 122.0)
            and (np.isfinite(price_position) and price_position <= 0.15)
            and (not np.isfinite(amount) or amount >= 4_000.0)
            and compare_credit_rating(rating, "AA-") >= 0
        )
        return "low" if low_risk else "medium"

    def _grid_step(self, row: pd.Series) -> float:
        if not self.config.dynamic_grid:
            configured = getattr(self.config, "add_on_drawdown_step", None)
            return float(configured or self.config.medium_risk_grid_step)
        risk_level = self._risk_level(row)
        if risk_level == "low":
            return float(self.config.low_risk_grid_step)
        if risk_level == "high":
            return float(self.config.high_risk_grid_step)
        return float(self.config.medium_risk_grid_step)

    def _risk_position_scale(self, row: pd.Series) -> float:
        if not self.config.dynamic_grid:
            return 1.0
        risk_level = self._risk_level(row)
        if risk_level == "low":
            return float(self.config.low_risk_position_scale)
        if risk_level == "high":
            return float(self.config.high_risk_position_scale)
        return float(self.config.medium_risk_position_scale)


@dataclass(frozen=True)
class HoldingGridConfig(ConservativeGridConfig):
    max_holdings: int | None = None
    market_risk_mode: str = "off"
    market_metric: str = "median_double_low"
    market_ma_window: int = 20
    market_trend_window: int = 20
    market_entry_scale_weak: float = 0.5
    market_entry_scale_strong: float = 1.0
    exit_price: float = 132.0
    exit_premium_rate: float = 45.0
    exit_double_low: float = 165.0
    exit_price_position_252: float = 0.80
    exit_floor_price: float | None = None
    exit_momentum_20d: float | None = None
    stop_loss_from_entry: float | None = None
    rebalance_existing_weight: bool = False
    initial_entry_fraction: float = 1.0
    add_on_drawdown_step: float | None = None
    add_position_fraction: float = 0.25
    max_grid_position_fraction: float = 1.0
    take_profit_1: float | None = None
    take_profit_1_keep_fraction: float = 0.5
    take_profit_2: float | None = None
    min_entry_trend_strength: float | None = None
    min_entry_six_sword: int | None = None
    min_entry_consecutive_six_sword: int | None = None
    min_entry_return_5d: float | None = None
    max_entry_return_5d: float | None = None
    min_entry_return_1d: float | None = None
    max_entry_return_1d: float | None = None
    max_entry_price_position_60d: float | None = None
    max_entry_market_median_double_low: float | None = None
    min_entry_market_trend_20d: float | None = None
    min_entry_market_trend_breadth: float | None = None


class HoldingGridStrategy(ConservativeGridStrategy):
    """Low-entry grid that keeps positions until explicit exit signals appear."""

    def __init__(self, config: HoldingGridConfig):
        super().__init__(config)
        self.config: HoldingGridConfig = config

    def should_exit(self, row: pd.Series, entry_price: float | None = None) -> bool:
        close = float(row.get("close", np.nan))
        premium = float(row.get("premium_rate", np.nan))
        double_low = float(row.get("double_low", np.nan))
        price_position = float(row.get("price_position_252", np.nan))
        if bool(row.get("call_risk", False)):
            return True
        if np.isfinite(close) and close >= self.config.exit_price:
            return True
        if np.isfinite(premium) and premium >= self.config.exit_premium_rate:
            return True
        if np.isfinite(double_low) and double_low >= self.config.exit_double_low:
            return True
        if np.isfinite(price_position) and price_position >= self.config.exit_price_position_252:
            return True
        if self.config.exit_floor_price is not None and np.isfinite(close) and close <= self.config.exit_floor_price:
            return True
        momentum_20d = float(row.get("momentum_20d", np.nan))
        if (
            self.config.exit_momentum_20d is not None
            and np.isfinite(momentum_20d)
            and momentum_20d <= self.config.exit_momentum_20d
        ):
            return True
        if (
            self.config.stop_loss_from_entry is not None
            and entry_price is not None
            and entry_price > 0
            and close / entry_price - 1.0 <= -self.config.stop_loss_from_entry
        ):
            return True
        return False

    def target_with_existing(
        self,
        daily: pd.DataFrame,
        basic: pd.DataFrame,
        current_weights: dict[str, float],
        entry_prices: dict[str, float],
        call: pd.DataFrame | None = None,
    ) -> tuple[dict[str, float], dict[str, float], pd.DataFrame]:
        universe = self.selector.build_universe(daily=daily, basic=basic, call=call)
        if universe.empty:
            return {}, {}, universe
        universe = self._attach_grid_columns(universe)
        universe_by_code = universe.set_index("ts_code", drop=False)
        next_weights: dict[str, float] = {}
        next_entries: dict[str, float] = {}
        kept_codes: set[str] = set()
        for ts_code, weight in current_weights.items():
            if ts_code not in universe_by_code.index:
                continue
            row = universe_by_code.loc[ts_code]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]
            entry_price = entry_prices.get(ts_code)
            if self.should_exit(row, entry_price=entry_price):
                continue
            adjusted_weight, adjusted_entry = self._adjust_existing_weight(
                row=row,
                current_weight=weight,
                entry_price=entry_price,
            )
            if adjusted_weight <= 0:
                continue
            kept_codes.add(ts_code)
            next_weights[ts_code] = adjusted_weight
            next_entries[ts_code] = adjusted_entry

        entry_scale = self._market_entry_scale(universe)
        entries = (
            self.target_portfolio(daily=daily, basic=basic, call=call)
            if entry_scale > 0
            else pd.DataFrame()
        )
        if not entries.empty:
            for row in entries.to_dict(orient="records"):
                ts_code = str(row["ts_code"])
                if ts_code in kept_codes:
                    if self.config.rebalance_existing_weight:
                        next_weights[ts_code] = float(row["target_weight"])
                    continue
                next_weights[ts_code] = (
                    float(row["target_weight"])
                    * entry_scale
                    * self.config.initial_entry_fraction
                )
                next_entries[ts_code] = float(row["close"])

        target_rows = []
        for ts_code, weight in next_weights.items():
            if ts_code not in universe_by_code.index:
                continue
            row = universe_by_code.loc[ts_code]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]
            payload = row.to_dict()
            payload["target_weight"] = weight
            payload["entry_price"] = next_entries.get(ts_code)
            payload["is_existing"] = ts_code in kept_codes
            target_rows.append(payload)
        target = pd.DataFrame(target_rows)
        if not target.empty:
            sort_columns = [
                column
                for column in ["target_weight", "grid_score", "double_low"]
                if column in target.columns
            ]
            ascending = [False if column != "double_low" else True for column in sort_columns]
            target = target.sort_values(sort_columns, ascending=ascending)
            if self.config.max_holdings is not None:
                target = target.head(self.config.max_holdings).copy()
            total_weight = float(target["target_weight"].sum())
            if total_weight > self.config.max_total_weight and total_weight > 0:
                target["target_weight"] *= self.config.max_total_weight / total_weight
            next_weights = dict(zip(target["ts_code"], target["target_weight"]))
            next_entries = {
                str(row["ts_code"]): float(row["entry_price"])
                for row in target[["ts_code", "entry_price"]].to_dict(orient="records")
            }
        return next_weights, next_entries, target

    def _adjust_existing_weight(
        self,
        row: pd.Series,
        current_weight: float,
        entry_price: float | None,
    ) -> tuple[float, float]:
        close = float(row.get("close", np.nan))
        if entry_price is None or entry_price <= 0 or not np.isfinite(close) or close <= 0:
            return current_weight, close if np.isfinite(close) else float(entry_price or 0.0)
        gain = close / entry_price - 1.0
        if self.config.take_profit_2 is not None and gain >= self.config.take_profit_2:
            return 0.0, float(entry_price)

        adjusted_weight = current_weight
        if self.config.take_profit_1 is not None and gain >= self.config.take_profit_1:
            take_profit_weight = self.config.max_position_weight * self.config.take_profit_1_keep_fraction
            adjusted_weight = min(adjusted_weight, take_profit_weight)

        grid_step = self._grid_step(row)
        if grid_step > 0:
            drawdown = max(entry_price / close - 1.0, 0.0)
            add_steps = int(drawdown // grid_step)
            if add_steps > 0 and not self._apply_low_filters(pd.DataFrame([row.to_dict()])).empty:
                fraction = min(
                    self.config.max_grid_position_fraction,
                    self.config.initial_entry_fraction + add_steps * self.config.add_position_fraction,
                )
                add_weight = self.config.max_position_weight * fraction
                adjusted_weight = max(adjusted_weight, add_weight)

        if adjusted_weight > current_weight and adjusted_weight > 0:
            added_weight = adjusted_weight - current_weight
            adjusted_entry = (
                entry_price * current_weight + close * added_weight
            ) / adjusted_weight
            return adjusted_weight, float(adjusted_entry)
        return adjusted_weight, float(entry_price)

    def _attach_grid_columns(self, universe: pd.DataFrame) -> pd.DataFrame:
        frame = universe.copy()
        if "double_low" not in frame.columns:
            frame["double_low"] = frame["close"] + frame["premium_rate"]
        return self._score(frame)

    def _apply_low_filters(self, frame: pd.DataFrame) -> pd.DataFrame:
        filtered = super()._apply_low_filters(frame)
        if filtered.empty:
            return filtered
        return self._apply_trend_entry_filters(filtered)

    def _apply_trend_entry_filters(self, frame: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        mask = pd.Series(True, index=frame.index)
        if cfg.min_entry_trend_strength is not None:
            mask &= frame.get("trend_strength", pd.Series(0.0, index=frame.index)).fillna(0.0) >= cfg.min_entry_trend_strength
        if cfg.min_entry_six_sword is not None:
            mask &= frame.get("six_sword_daily", pd.Series(0.0, index=frame.index)).fillna(0.0) >= cfg.min_entry_six_sword
        if cfg.min_entry_consecutive_six_sword is not None:
            mask &= frame.get("consecutive_six_sword", pd.Series(0.0, index=frame.index)).fillna(0.0) >= cfg.min_entry_consecutive_six_sword
        if cfg.min_entry_return_5d is not None:
            mask &= frame.get("return_5d", pd.Series(0.0, index=frame.index)).fillna(0.0) >= cfg.min_entry_return_5d
        if cfg.max_entry_return_5d is not None:
            mask &= frame.get("return_5d", pd.Series(0.0, index=frame.index)).fillna(0.0) <= cfg.max_entry_return_5d
        if cfg.min_entry_return_1d is not None:
            mask &= frame.get("return_1d", pd.Series(0.0, index=frame.index)).fillna(0.0) >= cfg.min_entry_return_1d
        if cfg.max_entry_return_1d is not None:
            mask &= frame.get("return_1d", pd.Series(0.0, index=frame.index)).fillna(0.0) <= cfg.max_entry_return_1d
        if cfg.max_entry_price_position_60d is not None:
            mask &= frame.get("price_position_60d", pd.Series(0.5, index=frame.index)).fillna(0.5) <= cfg.max_entry_price_position_60d
        if cfg.max_entry_market_median_double_low is not None:
            mask &= frame.get("market_median_double_low", pd.Series(np.inf, index=frame.index)).fillna(np.inf) <= cfg.max_entry_market_median_double_low
        if cfg.min_entry_market_trend_20d is not None:
            mask &= frame.get("market_trend_20d", pd.Series(0.0, index=frame.index)).fillna(0.0) >= cfg.min_entry_market_trend_20d
        if cfg.min_entry_market_trend_breadth is not None:
            mask &= frame.get("market_trend_breadth", pd.Series(0.0, index=frame.index)).fillna(0.0) >= cfg.min_entry_market_trend_breadth
        return frame[mask].copy()

    def _market_entry_scale(self, universe: pd.DataFrame) -> float:
        if self.config.market_risk_mode == "off":
            return 1.0
        trend = universe.get("market_trend_20d")
        price_position = universe.get("market_price_position_252")
        trend_value = float(trend.dropna().iloc[-1]) if trend is not None and trend.dropna().size else 0.0
        position_value = (
            float(price_position.dropna().iloc[-1])
            if price_position is not None and price_position.dropna().size
            else 0.5
        )
        if self.config.market_risk_mode == "block_downtrend":
            return 0.0 if trend_value < 0 else self.config.market_entry_scale_strong
        if self.config.market_risk_mode == "scale_downtrend":
            if trend_value < 0 and position_value > 0.20:
                return self.config.market_entry_scale_weak
            return self.config.market_entry_scale_strong
        raise ValueError(f"Unsupported market_risk_mode: {self.config.market_risk_mode}")


def add_low_position_features(daily: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    frame = daily.copy()
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame["ts_code"] = frame["ts_code"].astype(str).str.upper()
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    grouped = frame.groupby("ts_code", group_keys=False)
    rolling_low = grouped["close"].rolling(window, min_periods=20).min().reset_index(level=0, drop=True)
    rolling_high = grouped["close"].rolling(window, min_periods=20).max().reset_index(level=0, drop=True)
    denominator = (rolling_high - rolling_low).replace(0, np.nan)
    frame["rolling_low_252"] = rolling_low
    frame["rolling_high_252"] = rolling_high
    frame["price_position_252"] = ((frame["close"] - rolling_low) / denominator).clip(0, 1)
    frame["drawdown_from_252_high"] = frame["close"] / rolling_high - 1.0
    frame["momentum_20d"] = grouped["close"].pct_change(20).fillna(0.0)
    if "bond_over_rate" in frame.columns and "premium_rate" not in frame.columns:
        frame["premium_rate"] = pd.to_numeric(frame["bond_over_rate"], errors="coerce")
    if "premium_rate" in frame.columns:
        frame["double_low"] = frame["close"] + pd.to_numeric(frame["premium_rate"], errors="coerce").fillna(0.0)
    return frame


def add_market_state_features(daily: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    frame = daily.copy()
    if "double_low" not in frame.columns:
        premium = pd.to_numeric(
            frame.get("premium_rate", frame.get("bond_over_rate", 0.0)),
            errors="coerce",
        ).fillna(0.0)
        frame["double_low"] = frame["close"] + premium
    market = (
        frame.groupby("trade_date")
        .agg(
            market_median_close=("close", "median"),
            market_median_double_low=("double_low", "median"),
        )
        .sort_index()
    )
    rolling_low = market["market_median_double_low"].rolling(window, min_periods=20).min()
    rolling_high = market["market_median_double_low"].rolling(window, min_periods=20).max()
    denominator = (rolling_high - rolling_low).replace(0, np.nan)
    market["market_price_position_252"] = (
        (market["market_median_double_low"] - rolling_low) / denominator
    ).clip(0, 1)
    market["market_ma_20"] = market["market_median_double_low"].rolling(20, min_periods=5).mean()
    market["market_trend_20d"] = market["market_median_double_low"] / market["market_ma_20"] - 1.0
    return frame.merge(market.reset_index(), on="trade_date", how="left")


def summarize_position_trades(position_trades: pd.DataFrame) -> dict[str, Any]:
    if position_trades.empty:
        return {
            "closed_position_trades": 0,
            "position_win_rate": 0.0,
            "position_profit_factor": 0.0,
            "average_position_return": 0.0,
            "median_position_return": 0.0,
            "average_holding_days": 0.0,
        }
    closed = position_trades[position_trades["status"] == "closed"].copy()
    if closed.empty:
        return {
            "closed_position_trades": 0,
            "position_win_rate": 0.0,
            "position_profit_factor": 0.0,
            "average_position_return": 0.0,
            "median_position_return": 0.0,
            "average_holding_days": 0.0,
        }
    returns = pd.to_numeric(closed["return"], errors="coerce").dropna()
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    gross_profit = float(wins.sum())
    gross_loss = float(losses.abs().sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else None if gross_profit > 0 else 0.0
    holding_days = pd.to_numeric(closed["holding_days"], errors="coerce").dropna()
    return {
        "closed_position_trades": int(len(returns)),
        "position_win_rate": float((returns > 0).mean()) if not returns.empty else 0.0,
        "position_profit_factor": profit_factor,
        "average_position_return": float(returns.mean()) if not returns.empty else 0.0,
        "median_position_return": float(returns.median()) if not returns.empty else 0.0,
        "average_holding_days": float(holding_days.mean()) if not holding_days.empty else 0.0,
    }


def backtest_conservative_grid(
    daily: pd.DataFrame,
    basic: pd.DataFrame,
    call: pd.DataFrame | None,
    backtest_config: ConvertibleBondBacktestConfig,
    grid_config: ConservativeGridConfig,
) -> ConvertibleBondBacktestResult:
    strategy = ConservativeGridStrategy(grid_config)
    prepared_daily = add_trend_enhanced_features(
        add_market_state_features(
            add_low_position_features(_prepare_daily(daily, backtest_config.start_date, backtest_config.end_date))
        )
    )
    prepared_basic = _prepare_basic(basic)
    prepared_call = _prepare_call(call if call is not None else pd.DataFrame())
    trade_dates = sorted(prepared_daily["trade_date"].dropna().astype(str).unique())
    trade_dates = trade_dates[backtest_config.min_history_trade_dates :]
    if len(trade_dates) < 2:
        raise ValueError("Not enough convertible-bond trade dates for backtest")

    weights: dict[str, float] = {}
    previous_close: dict[str, float] = {}
    equity = backtest_config.initial_cash
    equity_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []

    for index, trade_date in enumerate(trade_dates):
        day = prepared_daily[prepared_daily["trade_date"] == trade_date].copy()
        close_map = dict(zip(day["ts_code"], day["close"]))
        gross_return = _portfolio_return(weights, previous_close, close_map)
        equity *= 1.0 + gross_return
        active_basic = _active_basic(prepared_basic, trade_date)
        active_call = _active_call(prepared_call, trade_date)
        if _should_rebalance(index, backtest_config.rebalance):
            target = strategy.target_portfolio(daily=day, basic=active_basic, call=active_call)
            target_weights = (
                dict(zip(target["ts_code"], target["target_weight"])) if not target.empty else {}
            )
            turnover = _turnover(weights, target_weights)
            cost = turnover * (backtest_config.commission_rate + backtest_config.slippage_rate)
            equity *= 1.0 - cost
            for ts_code in sorted(set(weights) | set(target_weights)):
                current_weight = float(weights.get(ts_code, 0.0))
                target_weight = float(target_weights.get(ts_code, 0.0))
                delta = target_weight - current_weight
                if abs(delta) >= grid_config.selector_config.rebalance_threshold:
                    trade_rows.append(
                        {
                            "trade_date": trade_date,
                            "ts_code": ts_code,
                            "current_weight": current_weight,
                            "target_weight": target_weight,
                            "delta_weight": delta,
                            "turnover": abs(delta),
                        }
                    )
            if not target.empty:
                for row in target.to_dict(orient="records"):
                    row["trade_date"] = trade_date
                    target_rows.append(row)
            weights = target_weights
        else:
            turnover = 0.0
            cost = 0.0
        equity_rows.append(
            {
                "trade_date": trade_date,
                "equity": equity,
                "daily_return": gross_return - cost,
                "gross_return": gross_return,
                "cost": cost,
                "turnover": turnover,
                "positions": len(weights),
                "exposure": float(sum(weights.values())),
            }
        )
        previous_close = close_map

    equity_frame = pd.DataFrame(equity_rows)
    targets_frame = pd.DataFrame(target_rows)
    trades_frame = pd.DataFrame(trade_rows)
    summary = summarize_backtest(
        equity=equity_frame,
        trades=trades_frame,
        config=backtest_config,
        first_source_date=str(prepared_daily["trade_date"].min()),
        last_source_date=str(prepared_daily["trade_date"].max()),
    )
    summary["grid_config"] = grid_config.__dict__
    summary["average_exposure"] = float(equity_frame["exposure"].mean()) if "exposure" in equity_frame else 0.0
    summary["invested_days"] = int((equity_frame["exposure"] > 0).sum()) if "exposure" in equity_frame else 0
    return ConvertibleBondBacktestResult(
        summary=summary,
        equity=equity_frame,
        targets=targets_frame,
        trades=trades_frame,
    )


def backtest_holding_grid(
    daily: pd.DataFrame,
    basic: pd.DataFrame,
    call: pd.DataFrame | None,
    backtest_config: ConvertibleBondBacktestConfig,
    grid_config: HoldingGridConfig,
) -> ConvertibleBondBacktestResult:
    strategy = HoldingGridStrategy(grid_config)
    prepared_daily = add_trend_enhanced_features(
        add_market_state_features(
            add_low_position_features(_prepare_daily(daily, backtest_config.start_date, backtest_config.end_date))
        )
    )
    prepared_basic = _prepare_basic(basic)
    prepared_call = _prepare_call(call if call is not None else pd.DataFrame())
    trade_dates = sorted(prepared_daily["trade_date"].dropna().astype(str).unique())
    trade_dates = trade_dates[backtest_config.min_history_trade_dates :]
    if len(trade_dates) < 2:
        raise ValueError("Not enough convertible-bond trade dates for backtest")

    weights: dict[str, float] = {}
    entry_prices: dict[str, float] = {}
    entry_dates: dict[str, str] = {}
    previous_close: dict[str, float] = {}
    equity = backtest_config.initial_cash
    equity_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    position_trade_rows: list[dict[str, Any]] = []

    for index, trade_date in enumerate(trade_dates):
        day = prepared_daily[prepared_daily["trade_date"] == trade_date].copy()
        close_map = dict(zip(day["ts_code"], day["close"]))
        gross_return = _portfolio_return(weights, previous_close, close_map)
        equity *= 1.0 + gross_return
        if _should_rebalance(index, backtest_config.rebalance):
            active_basic = _active_basic(prepared_basic, trade_date)
            active_call = _active_call(prepared_call, trade_date)
            target_weights, next_entries, target = strategy.target_with_existing(
                daily=day,
                basic=active_basic,
                current_weights=weights,
                entry_prices=entry_prices,
                call=active_call,
            )
            turnover = _turnover(weights, target_weights)
            cost = turnover * (backtest_config.commission_rate + backtest_config.slippage_rate)
            equity *= 1.0 - cost
            for ts_code in sorted(set(weights) | set(target_weights)):
                current_weight = float(weights.get(ts_code, 0.0))
                target_weight = float(target_weights.get(ts_code, 0.0))
                delta = target_weight - current_weight
                if current_weight <= 0 < target_weight:
                    entry_dates[ts_code] = trade_date
                if current_weight > 0 and target_weight <= 0:
                    row = day[day["ts_code"] == ts_code]
                    exit_price = float(row["close"].iloc[-1]) if not row.empty else np.nan
                    entry_price = float(entry_prices.get(ts_code, np.nan))
                    entry_date = entry_dates.get(ts_code, trade_date)
                    holding_days = (
                        pd.to_datetime(trade_date) - pd.to_datetime(entry_date)
                    ).days
                    position_trade_rows.append(
                        {
                            "ts_code": ts_code,
                            "entry_date": entry_date,
                            "exit_date": trade_date,
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "return": exit_price / entry_price - 1.0
                            if np.isfinite(entry_price) and entry_price > 0 and np.isfinite(exit_price)
                            else np.nan,
                            "holding_days": holding_days,
                            "status": "closed",
                        }
                    )
                if abs(delta) >= grid_config.selector_config.rebalance_threshold:
                    trade_rows.append(
                        {
                            "trade_date": trade_date,
                            "ts_code": ts_code,
                            "current_weight": current_weight,
                            "target_weight": target_weight,
                            "delta_weight": delta,
                            "turnover": abs(delta),
                        }
                    )
            if not target.empty:
                for row in target.to_dict(orient="records"):
                    row["trade_date"] = trade_date
                    target_rows.append(row)
            weights = target_weights
            entry_prices = next_entries
            entry_dates = {ts_code: entry_dates.get(ts_code, trade_date) for ts_code in weights}
        else:
            turnover = 0.0
            cost = 0.0
        equity_rows.append(
            {
                "trade_date": trade_date,
                "equity": equity,
                "daily_return": gross_return - cost,
                "gross_return": gross_return,
                "cost": cost,
                "turnover": turnover,
                "positions": len(weights),
                "exposure": float(sum(weights.values())),
            }
        )
        previous_close = close_map

    if trade_dates:
        last_trade_date = trade_dates[-1]
        last_day = prepared_daily[prepared_daily["trade_date"] == last_trade_date].copy()
        for ts_code, weight in weights.items():
            if weight <= 0:
                continue
            row = last_day[last_day["ts_code"] == ts_code]
            exit_price = float(row["close"].iloc[-1]) if not row.empty else np.nan
            entry_price = float(entry_prices.get(ts_code, np.nan))
            entry_date = entry_dates.get(ts_code, last_trade_date)
            holding_days = (pd.to_datetime(last_trade_date) - pd.to_datetime(entry_date)).days
            position_trade_rows.append(
                {
                    "ts_code": ts_code,
                    "entry_date": entry_date,
                    "exit_date": last_trade_date,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "return": exit_price / entry_price - 1.0
                    if np.isfinite(entry_price) and entry_price > 0 and np.isfinite(exit_price)
                    else np.nan,
                    "holding_days": holding_days,
                    "status": "open",
                }
            )

    equity_frame = pd.DataFrame(equity_rows)
    targets_frame = pd.DataFrame(target_rows)
    trades_frame = pd.DataFrame(trade_rows)
    summary = summarize_backtest(
        equity=equity_frame,
        trades=trades_frame,
        config=backtest_config,
        first_source_date=str(prepared_daily["trade_date"].min()),
        last_source_date=str(prepared_daily["trade_date"].max()),
    )
    summary["grid_config"] = grid_config.__dict__
    summary["average_exposure"] = float(equity_frame["exposure"].mean()) if "exposure" in equity_frame else 0.0
    summary["invested_days"] = int((equity_frame["exposure"] > 0).sum()) if "exposure" in equity_frame else 0
    position_trades_frame = pd.DataFrame(position_trade_rows)
    summary.update(summarize_position_trades(position_trades_frame))
    return ConvertibleBondBacktestResult(
        summary=summary,
        equity=equity_frame,
        targets=targets_frame,
        trades=trades_frame,
        position_trades=position_trades_frame,
    )
