"""Causal BYD expert allocation research with an integer cash/share ledger.

Research assumptions (not a production execution engine) are recorded in the
experiment manifest. Signals are computed after close; orders execute next open.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BASE_EXPERTS = ("original_score", "original_range", "trend_60")
RULE_EXPERTS = BASE_EXPERTS + (
    "corrected_score", "cash", "trend_120", "pullback", "range_reversal", "vol_budget",
)
ALL_EXPERTS = RULE_EXPERTS + ("ridge_5d", "tree_5d")


@dataclass(frozen=True)
class ExecutionConfig:
    initial_cash: float = 1_000_000.0
    commission: float = 0.0003
    minimum_commission: float = 5.0
    transfer_fee: float = 0.00002
    slippage: float = 0.0005
    participation: float = 0.01
    dividend_tax_reserve: float = 0.20

    def __post_init__(self) -> None:
        if self.initial_cash <= 0 or not 0 < self.participation <= 1:
            raise ValueError("invalid capital or participation")
        if any(getattr(self, name) < 0 for name in (
            "commission", "minimum_commission", "transfer_fee", "slippage",
        )) or not 0 <= self.dividend_tax_reserve <= 1:
            raise ValueError("invalid cost or dividend tax")


@dataclass(frozen=True)
class MixerConfig:
    experts: tuple[str, ...] = BASE_EXPERTS
    eta: float = 0.10
    share: float = 0.0
    forget: float = 1.0
    scale: float = 0.02
    deadband: float = 0.0
    max_step: float = 1.0
    downside: float = 0.0
    vol_target: float = 0.0

    def __post_init__(self) -> None:
        if not self.experts or len(set(self.experts)) != len(self.experts):
            raise ValueError("experts must be nonempty and distinct")
        if self.eta < 0 or not 0 <= self.share <= 1 or not 0 < self.forget <= 1:
            raise ValueError("invalid learning configuration")
        if self.scale <= 0 or not 0 <= self.deadband <= 1 or not 0 < self.max_step <= 1:
            raise ValueError("invalid sizing configuration")
        if self.downside < 0 or self.vol_target < 0:
            raise ValueError("invalid risk configuration")

    def to_dict(self) -> dict:
        return asdict(self)


def update_weights(weights: np.ndarray, returns: np.ndarray, config: MixerConfig) -> np.ndarray:
    """Exponentiated loss update, optional discounted memory and fixed share."""
    if weights.shape != returns.shape or len(weights) != len(config.experts):
        raise ValueError("expert dimensions differ")
    if not np.isfinite(returns).all() or not np.isfinite(weights).all():
        raise ValueError("non-finite expert observation")
    if (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("invalid weights")
    utility = np.clip(returns / config.scale, -1.0, 1.0)
    utility -= config.downside * np.maximum(-utility, 0.0)
    logits = config.forget * np.log(np.maximum(weights, 1e-300)) + config.eta * utility
    shifted = np.exp(logits - logits.max())
    normalized = shifted / shifted.sum()
    return (1 - config.share) * normalized + config.share / len(weights)


def load_market(root: Path = ROOT) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read raw canonical bars; reconstruct only contemporaneous adjustments."""
    frames = []
    for path in sorted((root / "data/raw/daily_partitioned").glob("year_month=*/data.parquet")):
        frame = pd.read_parquet(path, filters=[("ts_code", "==", "002594.SZ")])
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise FileNotFoundError("canonical BYD daily partitions unavailable")
    raw = pd.concat(frames, ignore_index=True).sort_values("trade_date")
    raw = raw.drop_duplicates("trade_date", keep="last").reset_index(drop=True)
    raw["date"] = pd.to_datetime(raw.trade_date.astype(str), format="%Y%m%d")
    for column in ("open", "high", "low", "close", "pre_close", "vol"):
        raw[column] = pd.to_numeric(raw[column], errors="raise")
    if (raw[["open", "close", "pre_close"]] <= 0).any().any():
        raise ValueError("invalid raw price")
    raw["volume_shares"] = raw.vol * 100  # Tushare canonical vol is in board lots.
    raw["split"] = 1
    raw.loc[raw.date.eq(pd.Timestamp("2025-07-29")), "split"] = 3
    inferred_cash = raw.close.shift(1) - raw.pre_close * raw["split"]
    if (inferred_cash.iloc[1:] < -0.02).any():
        raise ValueError("unexplained negative corporate adjustment or data gap")
    raw["cash_dividend"] = inferred_cash.clip(lower=0).fillna(0)
    raw.loc[raw.cash_dividend < 0.005, "cash_dividend"] = 0.0
    raw.loc[raw.date.eq(pd.Timestamp("2025-07-29")), "cash_dividend"] = 3.974
    raw["bonus_shares"] = np.where(raw["split"].eq(3), 0.8, 0.0)
    raw["up_limit"] = (raw.pre_close * 1.10 + 1e-8).round(2)
    raw["down_limit"] = (raw.pre_close * 0.90 + 1e-8).round(2)
    raw["limit_source"] = "inferred_main_board_10pct"
    raw["is_suspended"] = raw.vol.le(0)
    for path in sorted((root / "data/raw/tradability").glob("*.parquet")):
        limits = pd.read_parquet(path, filters=[("ts_code", "==", "002594.SZ")])
        if limits.empty:
            continue
        entry = limits.iloc[-1]
        mask = raw.trade_date.eq(str(entry.trade_date))
        if not mask.any():
            continue
        if pd.notna(entry.up_limit) and pd.notna(entry.down_limit):
            raw.loc[mask, "up_limit"] = float(entry.up_limit)
            raw.loc[mask, "down_limit"] = float(entry.down_limit)
            raw.loc[mask, "limit_source"] = "official_local_partition"
        raw.loc[mask, "is_suspended"] = bool(entry.is_suspended)
    # Forward-only index: extending a data prefix cannot change past features.
    synthetic = (raw.close / raw.pre_close).cumprod() * 100
    multiplier = synthetic / raw.close
    signal = pd.DataFrame({"date": raw.date, "symbol": "002594.SZ"})
    for column in ("open", "high", "low", "close"):
        signal[column] = raw[column] * multiplier
    signal["volume"] = raw.volume_shares / raw["split"].cumprod()
    signal["pct_change"] = (raw.close / raw.pre_close - 1) * 100
    return raw, signal


def _existing_modules() -> tuple:
    modules = []
    for name in ("byd_002594_trading_strategy", "byd_002594_range_t_strategy"):
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(name, ROOT / "scripts/research" / f"{name}.py")
            if spec is None or spec.loader is None:
                raise ImportError(name)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        modules.append(sys.modules[name])
    return tuple(modules)


def build_experts(signal: pd.DataFrame, include_models: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Adapt existing rules, add distinct experts and causally refit two models."""
    scoring, ranging = _existing_modules()
    indicators = scoring.add_indicators(signal)
    original = scoring.add_scores(indicators)
    original_mapper = scoring.bounded_score

    def corrected_mapper(value: pd.Series, low: float, high: float, reverse: bool = False) -> pd.Series:
        low, high = min(low, high), max(low, high)
        score = ((value - low) / (high - low)).clip(0, 1)
        return 1 - score if reverse else score

    # This isolated research adapter never changes the original strategy file.
    scoring.bounded_score = corrected_mapper
    try:
        corrected = scoring.add_scores(indicators)
    finally:
        scoring.bounded_score = original_mapper
    ranged = ranging.add_t_signals(ranging.add_range_indicators(signal))
    range_positions = []
    position = 0.85
    range_config = ranging.RangeTConfig()
    for _, row in ranged.iterrows():
        position = ranging.apply_position_delta(position, float(row.raw_delta), row, range_config)
        range_positions.append(position)
    close = indicators.close
    targets = pd.DataFrame(index=signal.index)
    targets["original_score"] = original.target_position_pct / 100
    targets["original_range"] = range_positions
    targets["trend_60"] = np.where(close > indicators.ma60, 0.95, 0.0)
    targets["corrected_score"] = corrected.target_position_pct / 100
    targets["cash"] = 0.0
    targets["trend_120"] = np.where(close > indicators.ma120, 0.95, 0.0)
    bull = (close > indicators.ma120) & (indicators.ma60_slope20 > 0)
    targets["pullback"] = np.where(bull, np.where(indicators.ret_5 < 0, 0.95, 0.50), 0.0)
    range_pos = ranged.range_pos_60.clip(0, 1)
    targets["range_reversal"] = np.where(ranged.range_regime & ~ranged.breakdown, 0.85 * (1 - range_pos), 0.10)
    vol = close.pct_change().rolling(20).std() * np.sqrt(252)
    targets["vol_budget"] = np.where(close > indicators.ma120, (0.25 / vol).clip(0, 0.95), 0.10)
    training = pd.DataFrame()
    if include_models:
        model_targets, training = rolling_model_experts(signal)
        targets = pd.concat([targets, model_targets], axis=1)
    targets = targets.fillna(0).clip(0, 1)
    targets.loc[:268, :] = 0.0  # Long-average warmup completed before trading begins.
    return targets, training


class _RidgeExpert:
    """Standardized ridge; einsum avoids a faulty local NumPy BLAS matmul path."""

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        from scipy.linalg import solve

        self.mean = x.mean(axis=0)
        self.scale = x.std(axis=0)
        self.scale[self.scale < 1e-12] = 1.0
        normalized = (x - self.mean) / self.scale
        self.intercept = float(y.mean())
        gram = np.einsum("ni,nj->ij", normalized, normalized)
        gram += np.eye(x.shape[1]) * 50.0
        rhs = np.einsum("ni,n->i", normalized, y - self.intercept)
        self.coefficients = solve(gram, rhs, assume_a="pos")
        if not np.isfinite(self.coefficients).all():
            raise ValueError("non-finite ridge fit")

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.einsum("ni,i->n", (x - self.mean) / self.scale, self.coefficients) + self.intercept


def rolling_model_experts(signal: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Five-session open-to-open labels; mature labels only, refit every 21 bars."""
    from sklearn.ensemble import GradientBoostingRegressor

    from quant.research.byd_daily_t_plan import daily_t_features

    features = daily_t_features(signal).reindex(pd.DatetimeIndex(signal.date))
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0)
    x = features.to_numpy(dtype=float)
    opens = signal.open.to_numpy(dtype=float)
    # At close i, sample j is eligible only if j+6 <= i (five open-to-open sessions).
    y = np.full(len(signal), np.nan)
    y[:-6] = np.log(opens[6:] / opens[1:-5])
    predictions = np.zeros((len(signal), 2))
    audit = []
    models = None
    for i in range(520, len(signal)):
        if models is None or (i - 520) % 21 == 0:
            end = i - 6
            begin = max(269, end - 755)
            if end - begin + 1 < 250:
                continue
            models = (
                _RidgeExpert(),
                GradientBoostingRegressor(
                    n_estimators=40, max_depth=2, min_samples_leaf=40,
                    learning_rate=0.05, loss="huber", random_state=20261012,
                ),
            )
            for model in models:
                model.fit(x[begin:end + 1], np.clip(y[begin:end + 1], -0.30, 0.30))
            audit.append({
                "fit_date": str(signal.date.iloc[i].date()),
                "train_start": str(signal.date.iloc[begin].date()),
                "last_feature_date": str(signal.date.iloc[end].date()),
                "last_label_end": str(signal.date.iloc[end + 6].date()),
                "samples": end - begin + 1,
            })
        for j, model in enumerate(models):
            predictions[i, j] = model.predict(x[i:i + 1])[0]
        if not np.isfinite(predictions[i]).all():
            raise ValueError("non-finite rolling prediction")
    # A smooth long/cash policy. The model predicts returns, not portfolio weights.
    targets = np.clip(0.30 + predictions / 0.05, 0.0, 0.95)
    targets[:520] = 0
    return pd.DataFrame(targets, columns=["ridge_5d", "tree_5d"], index=signal.index), pd.DataFrame(audit)


@dataclass
class _Account:
    cash: float
    shares: int = 0
    equity: float = 1_000_000.0


def _trade_day(
    account: _Account, target: float, i: int, arrays: dict[str, np.ndarray],
    execution: ExecutionConfig, deadband: float = 0.0, max_step: float = 1.0,
) -> tuple[float, float, int, float, str]:
    """One open order per day; previous holdings receive corporate entitlements."""
    previous_equity = account.equity
    old_shares = account.shares
    dividend = old_shares * (
        arrays["cash_dividend"][i] * (1 - execution.dividend_tax_reserve)
        - arrays["bonus_shares"][i] * execution.dividend_tax_reserve
    )
    account.cash += dividend
    account.shares *= int(arrays["split"][i])
    open_price = arrays["open"][i]
    equity_at_open = account.cash + account.shares * open_price
    position_at_open = account.shares * open_price / equity_at_open
    requested = float(np.clip(target, 0, 1))
    if abs(requested - position_at_open) <= deadband:
        requested = position_at_open
    requested = float(np.clip(requested, max(0, position_at_open - max_step), min(1, position_at_open + max_step)))
    # Single-asset integer allocation: buy lots; fully selling an odd remainder is allowed.
    desired = int(math.floor(requested * equity_at_open / open_price / 100)) * 100
    change = desired - account.shares
    if requested == 0:
        change = -account.shares
    capacity = int(arrays["prior_volume"][i] * execution.participation / 100) * 100
    if change > 0:
        change = min(change, capacity)
    elif change < 0:
        change = -min(-change, capacity)
    reason = ""
    if bool(arrays["is_suspended"][i]):
        reason = "suspended"
    elif change > 0 and open_price >= arrays["up_limit"][i] - 0.005:
        reason = "buy_at_upper_limit"
    elif change < 0 and open_price <= arrays["down_limit"][i] + 0.005:
        reason = "sell_at_lower_limit"
    if reason:
        change = 0
    cost = 0.0
    turnover = 0.0
    if change:
        buying = change > 0
        price = open_price * (1 + execution.slippage if buying else 1 - execution.slippage)
        # Slippage is capped at the exchange price band.
        price = min(price, arrays["up_limit"][i]) if buying else max(price, arrays["down_limit"][i])
        stamp = 0.0 if buying else (0.001 if arrays["date_int"][i] < 20230828 else 0.0005)
        while change:
            value = abs(change) * price
            fee = max(value * execution.commission, execution.minimum_commission)
            fee += value * (execution.transfer_fee + stamp)
            if not buying or value + fee <= account.cash + 1e-8:
                break
            change = max(0, change - 100)
        if change:
            account.cash -= change * price + fee
            account.shares += change
            cost = fee + abs(change) * abs(price - open_price)
            turnover = abs(change) * open_price / equity_at_open
    account.equity = account.cash + account.shares * arrays["close"][i]
    if account.cash < -1e-6 or account.shares < 0 or account.equity <= 0:
        raise RuntimeError("cash/inventory invariant violated")
    return account.equity / previous_equity - 1, turnover, change, cost, reason


def market_arrays(market: pd.DataFrame) -> dict[str, np.ndarray]:
    result = {key: market[key].to_numpy() for key in (
        "open", "close", "split", "cash_dividend", "bonus_shares", "up_limit",
        "down_limit", "is_suspended",
    )}
    result["prior_volume"] = market.volume_shares.shift(1).fillna(0).to_numpy()
    result["date_int"] = market.date.dt.strftime("%Y%m%d").astype(int).to_numpy()
    return result


def simulate(
    market: pd.DataFrame, targets: pd.DataFrame, execution: ExecutionConfig = ExecutionConfig(),
    mixer: Optional[MixerConfig] = None, expert_returns: Optional[pd.DataFrame] = None,
    fixed_target: Optional[np.ndarray] = None, start: str = "2015-01-01",
) -> pd.DataFrame:
    """Trade t using close t-1 target; update mixer only after observing close t."""
    if len(market) != len(targets):
        raise ValueError("market and targets lengths differ")
    if mixer is None and fixed_target is None:
        raise ValueError("a fixed target or mixer is required")
    if mixer is not None and (expert_returns is None or len(expert_returns) != len(market)):
        raise ValueError("mixer requires aligned net expert returns")
    arrays = market_arrays(market)
    n = len(market)
    first = int(np.searchsorted(market.date.to_numpy(), np.datetime64(start)))
    if first >= n or first < 1:
        raise ValueError("start must have warmup and at least one trade day")
    account = _Account(execution.initial_cash, equity=execution.initial_cash)
    if mixer is not None:
        weights = np.ones(len(mixer.experts)) / len(mixer.experts)
        expert_targets = targets.loc[:, mixer.experts].to_numpy()
        observed_returns = expert_returns.loc[:, mixer.experts].to_numpy()
        daily_market_returns = pd.Series(arrays["close"] / market.pre_close.to_numpy() - 1)
        known_vol = daily_market_returns.rolling(20).std().to_numpy() * np.sqrt(252)
    pending = float(fixed_target[first - 1]) if fixed_target is not None else float(expert_targets[first - 1] @ weights)
    result = []
    for i in range(first, n):
        used_weights = weights.copy() if mixer is not None else None
        ret, turnover, change, cost, blocked = _trade_day(
            account, pending, i, arrays, execution,
            mixer.deadband if mixer else 0, mixer.max_step if mixer else 1,
        )
        record = {
            "date": market.date.iloc[i], "signal_date": market.date.iloc[i - 1],
            "return": ret, "equity": account.equity, "cash": account.cash,
            "shares": account.shares, "position": account.shares * arrays["close"][i] / account.equity,
            "requested_target": pending, "turnover": turnover, "trade_shares": change,
            "cost": cost, "blocked": blocked,
        }
        if mixer is not None:
            record.update({f"weight_{key}": float(value) for key, value in zip(mixer.experts, used_weights)})
            weights = update_weights(weights, observed_returns[i], mixer)
            pending = float(expert_targets[i] @ weights)
            if mixer.vol_target and np.isfinite(known_vol[i]) and known_vol[i] > 0:
                pending *= min(1, mixer.vol_target / known_vol[i])
        else:
            pending = float(fixed_target[i])
        result.append(record)
    return pd.DataFrame(result)


def evaluate_experts(
    market: pd.DataFrame, targets: pd.DataFrame, execution: ExecutionConfig = ExecutionConfig(),
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    returns = pd.DataFrame(0.0, index=market.index, columns=targets.columns)
    ledgers = {}
    for name in targets.columns:
        ledger = simulate(market, targets, execution, fixed_target=targets[name].to_numpy())
        ledgers[name] = ledger
        indices = market.index[market.date.ge(pd.Timestamp("2015-01-01"))]
        returns.loc[indices, name] = ledger["return"].to_numpy()
    return returns, ledgers


def metrics(ledger: pd.DataFrame) -> dict[str, float]:
    if ledger.empty:
        raise ValueError("empty evaluation window")
    ret = ledger["return"].to_numpy(dtype=float)
    equity = np.r_[1.0, np.cumprod(1 + ret)]
    volatility = float(np.std(ret, ddof=1)) if len(ret) > 1 else 0.0
    years = len(ret) / 252
    return {
        "sessions": len(ret), "total_return": float(equity[-1] - 1),
        "cagr": float(equity[-1] ** (1 / years) - 1),
        "sharpe": float(np.mean(ret) / volatility * np.sqrt(252)) if volatility > 1e-12 else 0.0,
        "max_drawdown": float(np.min(equity / np.maximum.accumulate(equity) - 1)),
        "average_position": float(ledger.position.mean()),
        "turnover": float(ledger.turnover.sum()),
        "orders": int(ledger.trade_shares.ne(0).sum()), "cost": float(ledger.cost.sum()),
    }


DEVELOPMENT_BLOCKS = (("2018-01-01", "2019-12-31"), ("2020-01-01", "2021-12-31"), ("2022-01-01", "2023-12-31"))


def selection_score(ledger: pd.DataFrame) -> float:
    scores = []
    for start, end in DEVELOPMENT_BLOCKS:
        block = ledger[ledger.date.between(start, end)]
        m = metrics(block)
        scores.append(m["sharpe"] + 0.5 * m["cagr"] - 0.5 * abs(m["max_drawdown"]))
    return float(np.mean(scores) - 0.35 * np.std(scores))
