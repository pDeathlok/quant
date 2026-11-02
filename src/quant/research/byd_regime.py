"""Causal trend/range experts and an account-level drawdown circuit breaker."""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant.research.byd_allocation import AllocationConfig, OnlineAllocator, signal_from_snapshot
from quant.research.byd_mwu import _Account, market_arrays
from quant.research.byd_wear import START, WearConfig, wear_cost_target, wear_trade_day


def regime_experts(market: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    signal = signal_from_snapshot(market)
    close = signal.close
    ma60, ma120 = close.rolling(60).mean(), close.rolling(120).mean()
    slope = ma60.pct_change(20)
    distance = close / ma120 - 1
    up = (distance > .03) & (slope > .01)
    down = (distance < -.03) & (slope < -.01)
    regime = np.where(up, 2, np.where(down, 0, 1))
    sigma = (market.close / market.pre_close - 1).rolling(20).std() * np.sqrt(252)
    z = (close - close.rolling(20).mean()) / close.rolling(20).std()
    frame = pd.DataFrame(index=market.index)
    frame['cash'] = 0.
    frame['constant_third'] = 1/3
    for horizon in (60, 120, 250):
        mean = close.rolling(horizon).mean()
        # Hysteresis avoids changing sides around a single moving average.
        held, path = 0., []
        for price, average in zip(close, mean):
            if price > average * 1.02:
                held = .65
            elif price < average * .98:
                held = 0.
            path.append(held)
        frame[f'trend_{horizon}'] = path
    frame['uptrend_only'] = np.where(up, .65, 0.)
    frame['downtrend_defense'] = np.where(down, 0., 1/3)
    frame['range_reversal'] = np.where(regime == 1, (.30 - .12*z).clip(0, .55), 0.)
    frame['regime_switch'] = np.where(up, .65, np.where(down, 0., (.30-.10*z).clip(.1,.45)))
    previous_high = signal.high.shift(1).rolling(60).max()
    previous_low = signal.low.shift(1).rolling(20).min()
    held, path = 0., []
    for price, upper, lower in zip(close, previous_high, previous_low):
        if price > upper:
            held = .65
        elif price < lower:
            held = 0.
        path.append(held)
    frame['breakout_60'] = path
    for target in (.10, .15):
        cap = (target / sigma.replace(0,np.nan)).clip(0, .65).fillna(0)
        frame[f'vol_{int(target*100)}'] = cap
        frame[f'regime_vol_{int(target*100)}'] = np.minimum(frame.regime_switch, cap)
    frame['asset_drawdown_defense'] = frame.regime_switch * ((close/close.rolling(252).max()-1+.30)/.20).clip(0,1)
    frame = frame.fillna(0).clip(0,1)
    frame.iloc[:269] = 0.
    return frame, regime


def mix_path(targets: pd.DataFrame, rewards: pd.DataFrame, regimes: np.ndarray,
             config: AllocationConfig, first: int, downside: float = 0.) -> tuple[np.ndarray, np.ndarray]:
    learner = OnlineAllocator(targets.shape[1], config)
    weights = np.full(targets.shape, 1/targets.shape[1])
    for i in range(first, len(targets)):
        realized = rewards.iloc[i].to_numpy()
        utility = realized - downside * np.maximum(-realized, 0.)
        learner.observe(utility, int(regimes[i-1]))
        weights[i] = learner.weights(int(regimes[i]))
    return np.einsum('ni,ni->n', weights, targets.to_numpy()), weights


def risk_simulate(market: pd.DataFrame, proposals: np.ndarray, configs: list[AllocationConfig],
                  regimes: np.ndarray, wear: WearConfig, stops: np.ndarray,
                  start: str = START) -> pd.DataFrame:
    """Prior-close account drawdown stops; 20-session cooldown, uptrend reentry."""
    arrays = market_arrays(market)
    variance = (market.close/market.pre_close-1).rolling(20).var().fillna(0).to_numpy()
    arrays['known_sigma'] = np.r_[0., np.sqrt(variance[:-1])]
    first = int(np.searchsorted(market.date.to_numpy(), np.datetime64(start)))
    account = _Account(wear.initial_cash, equity=wear.initial_cash)
    peak, halted, since = wear.initial_cash, False, 0
    rows = []
    for i in range(first, len(market)):
        j, cfg = i-1, configs[i-1]
        drawdown = account.equity/peak-1
        if stops[j] > 0 and not halted and drawdown <= -stops[j]:
            halted, since = True, 0
        if halted:
            since += 1
            if since >= 20 and regimes[j] == 2:
                halted, peak = False, account.equity
        proposal = float(proposals[j])
        if cfg.vol_target and variance[j] > 0:
            proposal *= min(1.,cfg.vol_target/np.sqrt(252*variance[j]))
        if halted:
            proposal = 0.
        position = account.shares*arrays['close'][j]/account.equity
        target = wear_cost_target(proposal, position, account.equity, arrays['close'][j],
                                  market.volume_shares.iloc[j], variance[j], int(arrays['date_int'][j]),cfg,wear)
        # Exit bypasses turnover smoothing and the cost no-trade controller.
        if halted:
            target = 0.
            cfg = AllocationConfig(method='equal',deadband=0,max_step=1)
        record = wear_trade_day(account,target,i,arrays,wear,cfg,
                               bool(cfg.cost_gamma and abs(target-position)<1e-12))
        peak = max(peak,account.equity)
        record.update(date=market.date.iloc[i],signal_date=market.date.iloc[j],
                      proposal=proposal,regime=int(regimes[j]),halted=halted,prior_drawdown=drawdown)
        rows.append(record)
    return pd.DataFrame(rows)
