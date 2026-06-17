from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


_RATING_ORDER = {
    rating: score
    for score, rating in enumerate(
        [
            "",
            "C",
            "CC",
            "CCC",
            "B-",
            "B",
            "B+",
            "BB-",
            "BB",
            "BB+",
            "BBB-",
            "BBB",
            "BBB+",
            "A-",
            "A",
            "A+",
            "AA-",
            "AA",
            "AA+",
            "AAA",
        ]
    )
}


def compare_credit_rating(left: str | None, right: str | None) -> int:
    """Compare two credit ratings, returning positive when left is better."""
    left_score = _RATING_ORDER.get(str(left or "").strip().upper(), 0)
    right_score = _RATING_ORDER.get(str(right or "").strip().upper(), 0)
    return left_score - right_score


@dataclass(frozen=True)
class ConvertibleBondFilterConfig:
    min_price: float = 100.0
    max_price: float = 135.0
    max_premium_rate: float = 35.0
    min_amount: float = 1_000.0
    min_remaining_size: float = 1.0
    min_credit_rating: str = "AA-"
    exclude_call_risk: bool = True
    exclude_not_convertible: bool = True


@dataclass(frozen=True)
class ConvertibleBondRotationConfig:
    top_n: int = 10
    max_position_weight: float = 0.12
    min_score: float = 0.0
    rebalance_threshold: float = 0.01
    filter: ConvertibleBondFilterConfig = ConvertibleBondFilterConfig()


@dataclass(frozen=True)
class RebalanceOrder:
    ts_code: str
    action: Literal["BUY", "SELL"]
    current_weight: float
    target_weight: float
    delta_weight: float
    reason: str


class ConvertibleBondSelector:
    """可转债双低 + 流动性 + 风险过滤轮动选债器。"""

    def __init__(self, config: ConvertibleBondRotationConfig | None = None):
        self.config = config or ConvertibleBondRotationConfig()

    def build_universe(
        self,
        daily: pd.DataFrame,
        basic: pd.DataFrame,
        call: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Merge Tushare可转债行情、基础资料和强赎状态。"""
        if daily.empty:
            return pd.DataFrame()
        frame = self._standardize_daily(daily)
        if not basic.empty:
            frame = frame.merge(self._standardize_basic(basic), on="ts_code", how="left")
        else:
            frame["remain_size"] = np.nan
            frame["credit_rating"] = ""
        frame["call_risk"] = False
        if call is not None and not call.empty:
            frame = self._mark_call_risk(frame, call)
        return frame

    def select(
        self,
        daily: pd.DataFrame,
        basic: pd.DataFrame,
        call: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Return scored target candidates sorted from best to worst."""
        universe = self.build_universe(daily=daily, basic=basic, call=call)
        if universe.empty:
            return universe
        filtered = self._apply_filters(universe)
        if filtered.empty:
            return filtered
        scored = self._score(filtered)
        scored = scored[scored["score"] >= self.config.min_score].copy()
        scored = scored.sort_values(
            ["score", "double_low", "amount"],
            ascending=[False, True, False],
        ).reset_index(drop=True)
        scored["rank"] = np.arange(1, len(scored) + 1)
        return scored.head(self.config.top_n)

    def target_portfolio(
        self,
        daily: pd.DataFrame,
        basic: pd.DataFrame,
        call: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Build equal-weight target portfolio for the selected candidates."""
        selected = self.select(daily=daily, basic=basic, call=call)
        if selected.empty:
            return selected
        weight = min(1.0 / len(selected), self.config.max_position_weight)
        selected = selected.copy()
        selected["target_weight"] = weight
        return selected

    def rebalance_orders(
        self,
        current_weights: dict[str, float],
        daily: pd.DataFrame,
        basic: pd.DataFrame,
        call: pd.DataFrame | None = None,
    ) -> list[RebalanceOrder]:
        """Compare current holdings with target portfolio and emit weight orders."""
        target = self.target_portfolio(daily=daily, basic=basic, call=call)
        target_weights = (
            dict(zip(target["ts_code"], target["target_weight"])) if not target.empty else {}
        )
        all_codes = sorted(set(current_weights) | set(target_weights))
        orders: list[RebalanceOrder] = []
        for ts_code in all_codes:
            current = float(current_weights.get(ts_code, 0.0))
            target_weight = float(target_weights.get(ts_code, 0.0))
            delta = target_weight - current
            if abs(delta) < self.config.rebalance_threshold:
                continue
            if target_weight <= 0:
                reason = "跌出目标组合或触发风险过滤"
                action: Literal["BUY", "SELL"] = "SELL"
            elif current <= 0:
                reason = "进入双低轮动目标组合"
                action = "BUY"
            else:
                reason = "目标权重再平衡"
                action = "BUY" if delta > 0 else "SELL"
            orders.append(
                RebalanceOrder(
                    ts_code=ts_code,
                    action=action,
                    current_weight=current,
                    target_weight=target_weight,
                    delta_weight=delta,
                    reason=reason,
                )
            )
        return orders

    def _apply_filters(self, frame: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config.filter
        mask = (
            frame["close"].between(cfg.min_price, cfg.max_price, inclusive="both")
            & (frame["premium_rate"] <= cfg.max_premium_rate)
            & (frame["amount"] >= cfg.min_amount)
        )
        if "remain_size" in frame.columns:
            mask &= frame["remain_size"].fillna(cfg.min_remaining_size) >= cfg.min_remaining_size
        if cfg.min_credit_rating:
            mask &= frame["credit_rating"].map(
                lambda rating: compare_credit_rating(rating, cfg.min_credit_rating) >= 0
            )
        if cfg.exclude_call_risk:
            mask &= ~frame["call_risk"].fillna(False)
        if cfg.exclude_not_convertible and "conv_start_date" in frame.columns:
            trade_date = frame["trade_date"].astype(str)
            conv_start = frame["conv_start_date"].fillna("").astype(str)
            mask &= (conv_start == "") | (conv_start <= trade_date)
        return frame[mask].copy()

    def _score(self, frame: pd.DataFrame) -> pd.DataFrame:
        scored = frame.copy()
        scored["double_low"] = scored["close"] + scored["premium_rate"]
        scored["double_low_score"] = self._rank_score(scored["double_low"], ascending=True)
        scored["liquidity_score"] = self._rank_score(scored["amount"], ascending=False)
        scored["remain_size_score"] = self._rank_score(
            scored["remain_size"].fillna(scored["remain_size"].median()),
            ascending=False,
        )
        scored["momentum_score"] = self._rank_score(scored["pct_chg"].fillna(0.0), ascending=False)
        scored["score"] = (
            0.50 * scored["double_low_score"]
            + 0.20 * scored["liquidity_score"]
            + 0.15 * scored["remain_size_score"]
            + 0.15 * scored["momentum_score"]
        )
        return scored

    @staticmethod
    def _rank_score(series: pd.Series, ascending: bool) -> pd.Series:
        if series.empty:
            return series
        ranked = series.rank(method="average", pct=True, ascending=ascending)
        return (1.0 - ranked).fillna(0.0)

    @staticmethod
    def _first_existing(frame: pd.DataFrame, columns: list[str], default: float = np.nan) -> pd.Series:
        for column in columns:
            if column in frame.columns:
                return pd.to_numeric(frame[column], errors="coerce")
        return pd.Series(default, index=frame.index)

    def _standardize_daily(self, daily: pd.DataFrame) -> pd.DataFrame:
        frame = daily.copy()
        if "ts_code" not in frame.columns:
            raise ValueError("daily must include ts_code")
        frame["ts_code"] = frame["ts_code"].astype(str).str.upper()
        if "trade_date" in frame.columns:
            frame["trade_date"] = frame["trade_date"].astype(str)
        else:
            frame["trade_date"] = ""
        frame["close"] = self._first_existing(frame, ["close", "price"])
        frame["pct_chg"] = self._first_existing(frame, ["pct_chg", "pct_change"], default=0.0)
        frame["amount"] = self._first_existing(frame, ["amount", "turnover"], default=0.0).fillna(0.0)
        frame["premium_rate"] = self._first_existing(
            frame,
            ["premium_rate", "bond_over_rate", "convert_premium_rate", "conv_premium_rate"],
            default=0.0,
        ).fillna(0.0)
        return frame

    def _standardize_basic(self, basic: pd.DataFrame) -> pd.DataFrame:
        frame = basic.copy()
        frame["ts_code"] = frame["ts_code"].astype(str).str.upper()
        if "remain_size" not in frame.columns:
            frame["remain_size"] = np.nan
        frame["remain_size"] = pd.to_numeric(frame["remain_size"], errors="coerce")
        rating_columns = [column for column in ["newest_rating", "issue_rating", "rate"] if column in frame]
        frame["credit_rating"] = ""
        for column in reversed(rating_columns):
            frame["credit_rating"] = frame["credit_rating"].where(
                frame["credit_rating"].astype(str).str.len() > 0,
                frame[column].fillna("").astype(str).str.upper(),
            )
        keep_columns = [
            column
            for column in [
                "ts_code",
                "bond_short_name",
                "stk_code",
                "stk_short_name",
                "remain_size",
                "credit_rating",
                "list_date",
                "delist_date",
                "conv_start_date",
                "maturity_date",
                "call_clause",
            ]
            if column in frame.columns
        ]
        return frame[keep_columns]

    @staticmethod
    def _mark_call_risk(frame: pd.DataFrame, call: pd.DataFrame) -> pd.DataFrame:
        risk_words = ("公告强赎", "实施强赎", "强赎", "赎回")
        call_frame = call.copy()
        call_frame["ts_code"] = call_frame["ts_code"].astype(str).str.upper()
        risk = pd.Series(False, index=call_frame.index)
        for column in ["is_call", "call_type"]:
            if column in call_frame.columns:
                text = call_frame[column].fillna("").astype(str)
                risk |= text.str.contains("|".join(risk_words), regex=True)
                risk |= text.str.upper().isin({"Y", "YES", "1"})
        risky_codes = set(call_frame.loc[risk, "ts_code"])
        marked = frame.copy()
        marked["call_risk"] = marked["ts_code"].isin(risky_codes)
        return marked
