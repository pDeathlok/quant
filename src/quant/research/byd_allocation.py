"""Chronological experiments in expert design, online allocation and trading costs.

All policies decide after close and trade next open using the existing integer
ledger. Paper-inspired engineering variants are named explicitly, not advertised
as inheriting regret bounds for a non-linear, taxed trading account.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from quant.research.byd_mwu import (
    ExecutionConfig, _Account, _RidgeExpert, _trade_day, build_experts,
    market_arrays, metrics,
)


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = np.exp(values - np.max(values))
    return shifted / shifted.sum()


def project_simplex(values: np.ndarray) -> np.ndarray:
    """Euclidean projection onto nonnegative weights summing to one."""
    ordered = np.sort(values)[::-1]
    excess = np.cumsum(ordered) - 1
    count = np.arange(1, len(values) + 1)
    active = np.flatnonzero(ordered - excess / count > 0)[-1]
    return np.maximum(values - excess[active] / (active + 1), 0)


@dataclass(frozen=True)
class AllocationConfig:
    method: str = "mwu"
    group: str = "diverse"
    eta: float = .3
    share: float = 0.0
    forget: float = 1.0
    scale: float = .02
    deadband: float = .03
    max_step: float = .10
    cost_gamma: float = 0.0
    risk: float = 0.0
    vol_target: float = 0.0

    def __post_init__(self) -> None:
        if self.method not in (
            "equal", "mwu", "share", "discount", "adah2011", "squint_grid",
            "multiwindow", "context", "omd", "ftrl",
        ):
            raise ValueError("unknown allocation method")
        if self.eta < 0 or self.scale <= 0 or not 0 <= self.share <= 1:
            raise ValueError("invalid rate, scale or share")
        if not 0 < self.forget <= 1 or not 0 <= self.deadband <= 1:
            raise ValueError("invalid memory or deadband")
        if not 0 < self.max_step <= 1 or min(self.cost_gamma, self.risk, self.vol_target) < 0:
            raise ValueError("invalid trading controller")

    def to_dict(self) -> dict:
        return asdict(self)


class OnlineAllocator:
    """Finite full-information learner; observe AFTER the current order."""

    def __init__(self, size: int, config: AllocationConfig):
        if size < 1:
            raise ValueError("empty expert universe")
        self.size, self.config = size, config
        self.p = np.full(size, 1 / size)
        self.losses = np.zeros(size)
        self.scores = np.zeros((6, size))
        self.gap, self.ada_eta = 0.0, 1.0
        self.regret, self.variance = np.zeros(size), np.zeros(size)
        self.etas = np.geomspace(.001, .5, 25)
        self.history: list[np.ndarray] = []
        self.windows = (21, 63, 126, 252)
        self.base_weights = np.full((len(self.windows), size), 1 / size)
        self.meta = np.full(len(self.windows), 1 / len(self.windows))
        self.t = 0

    def weights(self, context: int = 0) -> np.ndarray:
        if not 0 <= context < 6:
            raise ValueError("invalid context")
        if self.config.method == "context":
            return softmax(self.config.eta * self.scores[context])
        return self.p.copy()

    def observe(self, returns: np.ndarray, context: int = 0) -> None:
        if returns.shape != (self.size,) or not np.isfinite(returns).all():
            raise ValueError("invalid expert feedback")
        cfg = self.config
        utility = np.clip(returns / cfg.scale, -1, 1)
        loss = (1 - utility) / 2  # AdaHedge/Squint require losses in [0, 1].
        self.t += 1
        if cfg.method == "equal":
            return
        if cfg.method in ("mwu", "share", "discount"):
            self.p = softmax(cfg.forget * np.log(np.maximum(self.p, 1e-300)) + cfg.eta * utility)
            self.p = (1 - cfg.share) * self.p + cfg.share / self.size
        elif cfg.method == "adah2011":
            # Algorithm 1, van Erven et al. 2011, phi=2; restart when gap budget is spent.
            log_mix = logsumexp(np.log(np.maximum(self.p, 1e-300)) - self.ada_eta * loss)
            self.gap += max(0.0, float(self.p @ loss + log_mix / self.ada_eta))
            self.p = softmax(np.log(np.maximum(self.p, 1e-300)) - self.ada_eta * loss)
            budget = (1 / np.expm1(1) + 1 / self.ada_eta) * np.log(self.size)
            if self.size > 1 and self.gap >= budget:
                self.ada_eta /= 2
                self.gap = 0.0
                self.p[:] = 1 / self.size
        elif cfg.method == "squint_grid":
            excess = float(self.p @ loss) - loss
            self.regret += excess
            self.variance += excess * excess
            # Finite uniform prior on log-spaced rates; not the continuous-prior implementation.
            logits = self.etas[:, None] * self.regret - self.etas[:, None] ** 2 * self.variance
            logits += np.log(self.etas[:, None])
            masses = np.exp(logits - logits.max()).sum(axis=0)
            self.p = masses / masses.sum()
        elif cfg.method == "multiwindow":
            rewards = np.einsum("wi,i->w", self.base_weights, utility)
            self.meta = softmax(np.log(np.maximum(self.meta, 1e-300)) + .1 * rewards)
            self.history.append(utility)
            if len(self.history) > max(self.windows):
                self.history.pop(0)
            history = np.asarray(self.history)
            for j, window in enumerate(self.windows):
                self.base_weights[j] = softmax(cfg.eta * history[-window:].sum(axis=0))
            self.p = np.einsum("w,wi->i", self.meta, self.base_weights)
        elif cfg.method == "context":
            # Forget only when the same state is revisited; no hindsight regime labels.
            self.scores[context] = cfg.forget * self.scores[context] + utility
        elif cfg.method == "omd":
            # Online projected ascent with an L2 pull toward uniform prior.
            gradient = utility - .1 * (self.p - 1 / self.size)
            self.p = project_simplex(self.p + cfg.eta / np.sqrt(self.t) * gradient)
        elif cfg.method == "ftrl":
            self.losses += utility
            # L2-regularized follow-the-leader with a sqrt(t) growing regularizer.
            self.p = project_simplex(1 / self.size + cfg.eta * self.losses / np.sqrt(self.t))


def signal_from_snapshot(market: pd.DataFrame) -> pd.DataFrame:
    synthetic = (market.close / market.pre_close).cumprod() * 100
    multiplier = synthetic / market.close
    signal = pd.DataFrame({"date": market.date, "symbol": "002594.SZ"})
    for column in ("open", "high", "low", "close"):
        signal[column] = market[column] * multiplier
    signal["volume"] = market.volume_shares / market["split"].cumprod()
    signal["pct_change"] = (market.close / market.pre_close - 1) * 100
    return signal


def state_features(signal: pd.DataFrame, index: pd.DataFrame) -> pd.DataFrame:
    """Align benchmark by date and forward-fill past quotations only."""
    dates = pd.DatetimeIndex(signal.date)
    idx = index.copy()
    idx["date"] = pd.to_datetime(idx.trade_date.astype(str), format="%Y%m%d")
    idx = idx.drop_duplicates("date").set_index("date").sort_index()
    # Include benchmark sessions during stock suspensions before aligning BYD dates.
    benchmark = idx.close.reindex(idx.index.union(dates)).sort_index().ffill().reindex(dates).to_numpy()
    c = signal.close
    b = pd.Series(benchmark, index=signal.index)
    frame = pd.DataFrame(index=signal.index)
    for window in (2, 5, 20, 63, 120, 250):
        frame[f"ret_{window}"] = c.pct_change(window)
        frame[f"distance_{window}"] = c / c.rolling(window).mean() - 1
    frame["vol20"] = c.pct_change().rolling(20).std()
    frame["vol63"] = c.pct_change().rolling(63).std()
    frame["range20"] = ((c - signal.low.rolling(20).min()) /
                        (signal.high.rolling(20).max() - signal.low.rolling(20).min()))
    frame["drawdown252"] = c / c.rolling(252).max() - 1
    frame["relative63"] = c.pct_change(63) - b.pct_change(63)
    frame["benchmark120"] = b / b.rolling(120).mean() - 1
    frame["volume_ratio"] = signal.volume / signal.volume.rolling(20).mean()
    frame["gap"] = signal.open / c.shift(1) - 1
    frame["intraday"] = c / signal.open - 1
    return frame.replace([np.inf, -np.inf], np.nan).fillna(0)


def market_context(features: pd.DataFrame) -> np.ndarray:
    # Fixed ex ante thresholds; no quantiles fitted on the whole sample.
    trend = np.where(features.distance_120 > .03, 2, np.where(features.distance_120 < -.03, 0, 1))
    return (2 * trend + (features.vol20.to_numpy() * np.sqrt(252) > .40)).astype(int)


def expanded_experts(
    signal: pd.DataFrame, index: pd.DataFrame, legacy: Optional[pd.DataFrame] = None,
    include_models: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if legacy is None:
        legacy, _ = build_experts(signal, include_models=False)
    targets = legacy.drop(columns="date", errors="ignore").copy()
    features = state_features(signal, index)
    c = signal.close
    targets["reversal_2d"] = (.35 - 3 * features.ret_2).clip(0, .9)
    targets["reversal_5d"] = (.35 - 2 * features.ret_5).clip(0, .9)
    z = (c - c.rolling(20).mean()) / c.rolling(20).std().replace(0, np.nan)
    targets["mean_revert_20d"] = (.35 - .15 * z).clip(0, .9)
    for window in (20, 250):
        targets[f"trend_{window}d"] = np.where(features[f"distance_{window}"] > 0, .9, .05)
    for window in (20, 60):
        # Stateful breakout/exit, both bands exclude today's quote.
        high = signal.high.shift(1).rolling(window).max()
        low = signal.low.shift(1).rolling(max(10, window // 2)).min()
        position, sequence = .05, []
        for close, upper, lower in zip(c, high, low):
            if close > upper:
                position = .9
            elif close < lower:
                position = .05
            sequence.append(position)
        targets[f"breakout_{window}d"] = sequence
    targets["relative_63d"] = np.where((features.relative63 > 0) & (features.ret_63 > 0), .85, .10)
    targets["market_gate"] = targets.original_range * np.where(features.benchmark120 > 0, 1.0, .35)
    targets["drawdown_defense"] = (.65 + 1.5 * features.drawdown252).clip(.05, .65)
    targets["volume_confirm"] = np.where(
        (features.distance_20 > 0) & (features.volume_ratio > 1), .8, .15,
    )
    audit = pd.DataFrame()
    if include_models:
        learned, audit = horizon_models(signal, features)
        targets = pd.concat([targets, learned], axis=1)
    targets = targets.fillna(0).clip(0, 1)
    targets.loc[:268, :] = 0
    return targets, features, audit


def horizon_models(signal: pd.DataFrame, features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.ensemble import GradientBoostingRegressor

    x, opens = features.to_numpy(dtype=float), signal.open.to_numpy(dtype=float)
    columns = {}
    audit = []
    for horizon in (1, 21, 63):
        end_offset = horizon + 1
        labels = np.full(len(signal), np.nan)
        labels[:-end_offset] = np.log(opens[end_offset:] / opens[1:-horizon])
        prediction = np.zeros(len(signal))
        tree_prediction = np.zeros(len(signal))
        models = None
        for i in range(640, len(signal)):
            if models is None or (i - 640) % 21 == 0:
                end = i - end_offset
                begin = max(269, end - 1259)
                if end - begin + 1 < 300:
                    continue
                models = [_RidgeExpert()]
                if horizon == 21:
                    models.append(GradientBoostingRegressor(
                        n_estimators=40, max_depth=2, min_samples_leaf=60,
                        learning_rate=.05, loss="huber", random_state=20261013,
                    ))
                for model in models:
                    model.fit(x[begin:end + 1], np.clip(labels[begin:end + 1], -.5, .5))
                audit.append({
                    "horizon": horizon, "fit_date": str(signal.date.iloc[i].date()),
                    "last_feature_date": str(signal.date.iloc[end].date()),
                    "last_label_end": str(signal.date.iloc[end + end_offset].date()),
                    "samples": end - begin + 1,
                })
            prediction[i] = models[0].predict(x[i:i + 1])[0]
            if horizon == 21:
                tree_prediction[i] = models[1].predict(x[i:i + 1])[0]
        # Convert horizon return forecasts to daily units for comparable exposure.
        columns[f"ridge_{horizon}d"] = np.clip(.35 + prediction / horizon / .004, 0, .95)
        columns[f"ridge_{horizon}d"][:640] = 0
        if horizon == 21:
            columns["tree_21d"] = np.clip(.35 + tree_prediction / horizon / .004, 0, .95)
            columns["tree_21d"][:640] = 0
    result = pd.DataFrame(columns, index=signal.index)
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("non-finite forecast")
    return result, pd.DataFrame(audit)


def expert_groups(columns: list[str]) -> dict[str, tuple[str, ...]]:
    legacy = tuple(name for name in columns if name in (
        "original_score", "original_range", "trend_60", "corrected_score", "cash",
        "trend_120", "pullback", "range_reversal", "vol_budget", "tree_5d",
    ))
    groups = {
        "legacy": legacy,
        "short": ("original_range", "cash", "reversal_2d", "reversal_5d", "mean_revert_20d", "trend_20d", "breakout_20d", "ridge_1d", "ridge_5d"),
        "long": ("cash", "trend_120", "trend_250d", "breakout_60d", "relative_63d", "drawdown_defense", "vol_budget", "ridge_63d"),
        "diverse": ("original_range", "cash", "reversal_5d", "trend_20d", "trend_120", "trend_250d", "relative_63d", "drawdown_defense", "vol_budget", "ridge_21d", "ridge_63d", "tree_21d"),
        "all": tuple(columns),
    }
    if any(any(name not in columns for name in group) for group in groups.values()):
        raise ValueError("missing required expert")
    return groups


def past_pruned_mask(
    market: pd.DataFrame, targets: pd.DataFrame, rewards: pd.DataFrame,
) -> tuple[np.ndarray, list[dict]]:
    """Yearly greedy redundancy pruning on the past 756 observations only."""
    mask = np.ones(targets.shape, dtype=bool)
    audits = []
    for year in sorted(market.date.dt.year.unique()):
        rows = np.flatnonzero(market.date.dt.year.to_numpy() == year)
        start = int(rows[0])
        begin = max(0, start - 756)
        past = rewards.iloc[begin:start]
        if len(past) < 250:
            continue
        vol = past.std().replace(0, np.nan)
        ranking = (past.mean() / vol).fillna(-1e6).sort_values(ascending=False)
        correlations = past.corr().fillna(0)
        positions = targets.iloc[begin:start].corr().fillna(0)
        kept = []
        for name in ranking.index:
            if name == "cash" or all(
                abs(correlations.loc[name, other]) < .95 or abs(positions.loc[name, other]) < .97
                for other in kept if other != "cash"
            ):
                kept.append(name)
        if "cash" not in kept:
            kept.append("cash")
        mask[rows, :] = np.asarray([name in kept for name in targets.columns])[None, :]
        audits.append({"year": int(year), "cutoff": str(market.date.iloc[start - 1].date()),
                       "kept": kept, "count": len(kept)})
    return mask, audits


def allocation_path(
    market: pd.DataFrame, targets: pd.DataFrame, rewards: pd.DataFrame,
    features: pd.DataFrame, config: AllocationConfig, prune: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Weights at close i incorporate feedback through i and affect order i+1."""
    contexts = market_context(features)
    learner = OnlineAllocator(targets.shape[1], config)
    observations = rewards.to_numpy()
    weights = np.full(targets.shape, 1 / targets.shape[1])
    first = int(np.searchsorted(market.date.to_numpy(), np.datetime64("2015-01-01")))
    for i in range(first, len(market)):
        learner.observe(observations[i], int(contexts[i - 1]))
        p = learner.weights(int(contexts[i]))
        if prune is not None:
            p = p * prune[i]
            p = p / p.sum() if p.sum() > 1e-200 else prune[i] / prune[i].sum()
        weights[i] = p
    requested = np.einsum("ni,ni->n", weights, targets.to_numpy())
    return requested, weights


def cost_target(
    proposal: float, position: float, equity: float, variance: float,
    date_int: int, config: AllocationConfig, execution: ExecutionConfig,
) -> float:
    """Finite convex-prox candidate search with minimum commission discontinuity.

    Objective = (x-proposal)^2/(2 gamma) + risk*variance*x^2/2 + cost/equity.
    Include exact current/proposed positions to permit no trade and exact target.
    Estimate fees at today's close; tomorrow's fill still uses actual open/limits.
    """
    if not config.cost_gamma:
        return float(proposal)
    choices = np.unique(np.r_[np.linspace(0, 1, 51), position, proposal])
    delta = choices - position
    notionals = np.abs(delta) * equity
    cost = np.where(
        notionals > 1e-8,
        np.maximum(notionals * execution.commission, execution.minimum_commission)
        + notionals * (execution.slippage + execution.transfer_fee
                      + np.where(delta < 0, .001 if date_int < 20230828 else .0005, 0)),
        0.0,
    ) / equity
    objective = (choices - proposal) ** 2 / (2 * config.cost_gamma)
    objective += .5 * config.risk * variance * choices ** 2 + cost
    return float(choices[np.argmin(objective)])


def simulate_path(
    market: pd.DataFrame, proposals: np.ndarray, configs: list[AllocationConfig],
    execution: ExecutionConfig = ExecutionConfig(), weights: Optional[pd.DataFrame] = None,
    start: str = "2015-01-01",
) -> pd.DataFrame:
    if len(market) != len(proposals) or len(configs) != len(market):
        raise ValueError("unaligned policy")
    arrays = market_arrays(market)
    variance = pd.Series(market.close / market.pre_close - 1).rolling(20).var().fillna(0).to_numpy()
    first = int(np.searchsorted(market.date.to_numpy(), np.datetime64(start)))
    if first < 1 or first >= len(market):
        raise ValueError("start needs a prior signal")
    account = _Account(execution.initial_cash, equity=execution.initial_cash)
    result = []
    for i in range(first, len(market)):
        j, cfg = i - 1, configs[i - 1]
        position = account.shares * arrays["close"][j] / account.equity
        proposal = float(proposals[j])
        if cfg.vol_target and variance[j] > 0:
            proposal *= min(1, cfg.vol_target / np.sqrt(252 * variance[j]))
        target = cost_target(proposal, position, account.equity, variance[j], int(arrays["date_int"][j]), cfg, execution)
        if cfg.cost_gamma and abs(target - position) < 1e-12:
            # A close-time no-trade choice means preserve inventory, not rebalance
            # tomorrow after an overnight gap. Corporate entitlements still accrue.
            previous = account.equity
            account.cash += account.shares * (
                arrays["cash_dividend"][i] * (1 - execution.dividend_tax_reserve)
                - arrays["bonus_shares"][i] * execution.dividend_tax_reserve
            )
            account.shares *= int(arrays["split"][i])
            account.equity = account.cash + account.shares * arrays["close"][i]
            ret, turnover, change, cost, blocked = account.equity / previous - 1, 0., 0, 0., "cost_no_trade"
        else:
            ret, turnover, change, cost, blocked = _trade_day(
                account, target, i, arrays, execution, cfg.deadband, cfg.max_step,
            )
        record = {
            "date": market.date.iloc[i], "signal_date": market.date.iloc[j], "return": ret,
            "equity": account.equity, "cash": account.cash, "shares": account.shares,
            "position": account.shares * arrays["close"][i] / account.equity,
            "proposal": proposal, "requested_target": target, "turnover": turnover,
            "trade_shares": change, "cost": cost, "blocked": blocked,
        }
        if weights is not None:
            record.update({f"weight_{name}": float(value) for name, value in weights.iloc[j].items()})
        result.append(record)
    return pd.DataFrame(result)


def inner_score(ledger: pd.DataFrame, year: int) -> float:
    """Only three latest completed annual validation blocks before outer year."""
    scores = []
    for previous in range(max(2016, year - 3), year):
        frame = ledger[ledger.date.dt.year.eq(previous)]
        if len(frame) < 100:
            continue
        m = metrics(frame)
        scores.append(m["sharpe"] + .5 * m["cagr"] - .5 * abs(m["max_drawdown"]))
    if not scores:
        raise ValueError("no inner validation history")
    return float(np.mean(scores) - .35 * np.std(scores))


def bayesian_search(values: np.ndarray, oracle: np.ndarray, budget: int = 12) -> tuple[int, list[int]]:
    """Finite-space GP expected-improvement search; oracle contains past scores only."""
    from scipy.stats import norm
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

    if len(values) != len(oracle) or not np.isfinite(oracle).all():
        raise ValueError("invalid search observations")
    rng = np.random.default_rng(20261013)
    seen = list(rng.choice(len(values), size=min(5, budget, len(values)), replace=False))
    kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(
        length_scale=1.0, length_scale_bounds="fixed", nu=2.5,
    ) + WhiteKernel(1e-5, noise_level_bounds="fixed")
    while len(seen) < min(budget, len(values)):
        model = GaussianProcessRegressor(kernel=kernel, optimizer=None, normalize_y=True)
        model.fit(values[seen], oracle[seen])
        mean, std = model.predict(values, return_std=True)
        gain = mean - max(oracle[seen]) - .01
        z = gain / np.maximum(std, 1e-12)
        ei = gain * norm.cdf(z) + std * norm.pdf(z)
        ei[seen] = -np.inf
        seen.append(int(np.argmax(ei)))
    best = max(seen, key=lambda i: oracle[i])
    return int(best), [int(i) for i in seen]
