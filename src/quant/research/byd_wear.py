"""Long-history, causal BYD trading-wear ledger and frozen online policies."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from quant.research.byd_allocation import AllocationConfig, OnlineAllocator, market_context
from quant.research.byd_mwu import _Account, market_arrays, metrics

START = "2012-08-10"
COST_COLUMNS = ("commission_cost", "transfer_cost", "stamp_cost", "spread_cost", "impact_cost")


@dataclass(frozen=True)
class WearConfig:
    initial_cash: float = 1_000_000.
    commission: float = .0003
    minimum_commission: float = 5.
    spread: float = .0005
    impact_eta: float = .1
    participation: float = .01
    historical_transfer: bool = True
    fixed_transfer: float = .00002
    charge_stamp: bool = True
    historical_commission: bool = False
    dividend_tax_reserve: float = .20

    def __post_init__(self) -> None:
        if self.initial_cash <= 0 or not 0 < self.participation <= 1:
            raise ValueError("invalid capital or capacity")
        if min(self.commission, self.minimum_commission, self.spread, self.impact_eta, self.fixed_transfer) < 0:
            raise ValueError("negative trading cost")
        if not 0 <= self.dividend_tax_reserve <= 1:
            raise ValueError("invalid dividend reserve")


def transfer_rate(date_int: int, wear: WearConfig) -> float:
    if not wear.historical_transfer:
        return wear.fixed_transfer
    if date_int < 20150801:
        return .0000255
    return .00002 if date_int < 20220429 else .00001


def stamp_rate(date_int: int, wear: WearConfig) -> float:
    if not wear.charge_stamp:
        return 0.
    return .001 if date_int < 20230828 else .0005


def commission_rate(date_int: int, wear: WearConfig) -> float:
    """Broker-history stress assumption, not a claim about the user's contract."""
    if not wear.historical_commission:
        return wear.commission
    return .001 if date_int < 20150101 else .0005 if date_int < 20200101 else .0003


def impact_rate(shares: float, prior_volume: float, known_sigma: float, wear: WearConfig) -> float:
    if prior_volume <= 0 or shares == 0:
        return 0.
    return wear.impact_eta * max(0., known_sigma) * math.sqrt(abs(shares) / prior_volume)


def past_score(ledger: pd.DataFrame, year: int) -> float:
    """Three complete calendar-year validation blocks strictly before outer year."""
    if year < 2016:
        raise ValueError("three full annual blocks require outer year >=2016")
    scores = []
    years = ledger.date.dt.year
    for previous in range(year - 3, year):
        block = ledger[years.eq(previous)]
        if len(block) < 100:
            raise ValueError("incomplete inner validation history")
        m = metrics(block)
        scores.append(m["sharpe"] + .5*m["cagr"] - .5*abs(m["max_drawdown"]))
    return float(np.mean(scores) - .35*np.std(scores))


def long_allocation_path(
    market: pd.DataFrame, targets: pd.DataFrame, rewards: pd.DataFrame,
    features: pd.DataFrame, config: AllocationConfig, start: str = START,
    prune: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    if len(market) != len(targets) or rewards.shape != targets.shape or len(features) != len(market):
        raise ValueError("unaligned expert observations")
    contexts = market_context(features)
    learner = OnlineAllocator(targets.shape[1], config)
    observations = rewards.to_numpy()
    weights = np.full(targets.shape, 1 / targets.shape[1])
    first = int(np.searchsorted(market.date.to_numpy(), np.datetime64(start)))
    for i in range(first, len(market)):
        learner.observe(observations[i], int(contexts[i-1]))
        p = learner.weights(int(contexts[i]))
        if prune is not None:
            p = p * prune[i]
            p = p / p.sum() if p.sum() > 1e-200 else prune[i] / prune[i].sum()
        weights[i] = p
    return np.einsum("ni,ni->n", weights, targets.to_numpy()), weights


def wear_cost_target(
    proposal: float, position: float, equity: float, close: float,
    known_volume: float, variance: float, date_int: int,
    config: AllocationConfig, wear: WearConfig,
) -> float:
    if not config.cost_gamma:
        return proposal
    choices = np.unique(np.r_[np.linspace(0, 1, 51), position, proposal])
    delta = choices - position
    notionals = np.abs(delta) * equity
    quantities = notionals / close
    impact = wear.impact_eta * np.sqrt(max(variance, 0)) * np.sqrt(quantities / max(known_volume, 1))
    cost = np.where(
        notionals > 1e-8,
        np.maximum(notionals * commission_rate(date_int, wear), wear.minimum_commission)
        + notionals * (transfer_rate(date_int, wear) + wear.spread + impact
                      + np.where(delta < 0, stamp_rate(date_int, wear), 0)), 0.,
    ) / equity
    objective = (choices - proposal)**2 / (2 * config.cost_gamma)
    objective += .5 * config.risk * variance * choices**2 + cost
    return float(choices[np.argmin(objective)])


def wear_trade_day(
    account: _Account, target: float, i: int, arrays: dict[str, np.ndarray],
    wear: WearConfig, config: AllocationConfig, hold_inventory: bool = False,
) -> dict:
    """Corporate actions first, constrained next-open order, then close marking."""
    previous = account.equity
    old_shares = account.shares
    account.cash += old_shares * (
        arrays["cash_dividend"][i] * (1-wear.dividend_tax_reserve)
        - arrays["bonus_shares"][i] * wear.dividend_tax_reserve
    )
    account.shares *= int(arrays["split"][i])
    open_price = float(arrays["open"][i])
    equity_open = account.cash + account.shares * open_price
    position_open = account.shares * open_price / equity_open
    requested = min(1., max(0., target))
    within_band = abs(requested-position_open) <= config.deadband
    if within_band:
        requested = position_open
    requested = min(position_open+config.max_step, max(position_open-config.max_step, requested))
    desired = int(math.floor(requested*equity_open/open_price/100 + 1e-9))*100
    if requested <= 0:
        desired = 0
    if hold_inventory or within_band:
        desired = account.shares  # No trade preserves exact inventory, including odd shares.
    requested_change = desired-account.shares
    capacity = int(arrays["prior_volume"][i]*wear.participation/100)*100
    change = min(requested_change, capacity) if requested_change > 0 else -min(-requested_change, capacity)
    capacity_limited = abs(requested_change) > capacity
    reason = "cost_no_trade" if hold_inventory else "deadband_no_trade" if within_band else ""
    if bool(arrays["is_suspended"][i]):
        reason, change = "suspended", 0
    elif change > 0 and open_price >= arrays["up_limit"][i]-.005:
        reason, change = "buy_at_upper_limit", 0
    elif change < 0 and open_price <= arrays["down_limit"][i]+.005:
        reason, change = "sell_at_lower_limit", 0
    costs = {key: 0. for key in COST_COLUMNS}
    price, rate, cash_limited = open_price, 0., False
    if change:
        buying = change > 0
        transfer = transfer_rate(int(arrays["date_int"][i]), wear)
        commission = commission_rate(int(arrays["date_int"][i]), wear)
        stamp = 0. if buying else stamp_rate(int(arrays["date_int"][i]), wear)
        before_cash = change
        # Reprice after reducing unaffordable quantities. The fixed point only
        # decreases size, so it cannot overspend cash or use future liquidity.
        while change:
            rate = impact_rate(change, arrays["prior_volume"][i], arrays["known_sigma"][i], wear)
            price = open_price * (1 + (wear.spread+rate) * (1 if buying else -1))
            price = min(price, arrays["up_limit"][i]) if buying else max(price, arrays["down_limit"][i])
            if not buying:
                break
            variable_bound = account.cash / (price*(1+commission+transfer))
            minimum_bound = max(0., account.cash-wear.minimum_commission) / (price*(1+transfer))
            affordable = max(0, int(math.floor(min(variable_bound, minimum_bound)/100 + 1e-9))*100)
            if change <= affordable:
                break
            change = affordable
        cash_limited = change != before_cash
        if change:
            value = abs(change)*price
            costs["commission_cost"] = max(value*commission, wear.minimum_commission)
            costs["transfer_cost"], costs["stamp_cost"] = value*transfer, value*stamp
            adverse = max(0., (price-open_price)*(1 if buying else -1))
            spread_price = min(adverse, open_price*wear.spread)
            costs["spread_cost"] = abs(change)*spread_price
            costs["impact_cost"] = abs(change)*(adverse-spread_price)
            fees = costs["commission_cost"]+costs["transfer_cost"]+costs["stamp_cost"]
            account.cash -= change*price+fees
            account.shares += change
    if not change:
        price, rate = open_price, 0.
    account.equity = account.cash + account.shares*arrays["close"][i]
    if account.cash < -1e-6 or account.shares < 0 or account.equity <= 0:
        raise RuntimeError("wear account invariant failed")
    return {
        "return": account.equity/previous-1, "equity": account.equity,
        "cash": account.cash, "shares": account.shares,
        "position": account.shares*arrays["close"][i]/account.equity,
        "requested_target": target, "requested_shares": requested_change,
        "trade_shares": change, "unfilled_shares": requested_change-change,
        "turnover": abs(change)*open_price/equity_open,
        "cost": sum(costs.values()), **costs,
        "cost_fraction": sum(costs.values())/equity_open,
        "fill_price": price, "modeled_impact_rate": rate,
        "actual_participation": abs(change)/max(arrays["prior_volume"][i], 1),
        "capacity_limited": capacity_limited, "cash_limited": cash_limited, "blocked": reason,
    }


def wear_simulate(
    market: pd.DataFrame, proposals: np.ndarray, configs: list[AllocationConfig],
    wear: WearConfig = WearConfig(), weights: Optional[pd.DataFrame] = None,
    start: str = START,
) -> pd.DataFrame:
    if len(proposals) != len(market) or len(configs) != len(market):
        raise ValueError("unaligned long policy")
    arrays = market_arrays(market)
    variance = pd.Series(market.close/market.pre_close-1).rolling(20).var().fillna(0).to_numpy()
    arrays["known_sigma"] = np.r_[0., np.sqrt(variance[:-1])]
    dates = market.date.to_numpy()
    volumes = market.volume_shares.to_numpy()
    first = int(np.searchsorted(dates, np.datetime64(start)))
    if first < 1 or first >= len(market):
        raise ValueError("start requires signal warmup")
    account = _Account(wear.initial_cash, equity=wear.initial_cash)
    result = []
    weight_array = weights.to_numpy() if weights is not None else None
    for i in range(first, len(market)):
        j, cfg = i-1, configs[i-1]
        position = account.shares*arrays["close"][j]/account.equity
        proposal = float(proposals[j])
        if cfg.vol_target and variance[j] > 0:
            proposal *= min(1., cfg.vol_target/math.sqrt(252*variance[j]))
        target = wear_cost_target(proposal, position, account.equity, arrays["close"][j],
                                  volumes[j], variance[j], int(arrays["date_int"][j]), cfg, wear)
        hold = bool(cfg.cost_gamma and abs(target-position) < 1e-12)
        record = wear_trade_day(account, target, i, arrays, wear, cfg, hold)
        record.update({"date": dates[i], "signal_date": dates[j], "proposal": proposal})
        if weight_array is not None:
            record.update({f"weight_{name}": float(value) for name, value in zip(weights.columns, weight_array[j])})
        result.append(record)
    return pd.DataFrame(result)


def wear_expert_returns(
    market: pd.DataFrame, targets: pd.DataFrame, wear: WearConfig, start: str = START,
) -> pd.DataFrame:
    rewards = pd.DataFrame(0., index=market.index, columns=targets.columns)
    cfg = AllocationConfig(method="equal", deadband=0, max_step=1)
    first = int(np.searchsorted(market.date.to_numpy(), np.datetime64(start)))
    for name in targets.columns:
        ledger = wear_simulate(market, targets[name].to_numpy(), [cfg]*len(market), wear, start=start)
        rewards.loc[market.index[first:], name] = ledger["return"].to_numpy()
    return rewards


def same_orders_gross(market: pd.DataFrame, net: pd.DataFrame, wear: WearConfig) -> pd.DataFrame:
    """Same net share path, zero trading fees; saved cash is not reinvested."""
    lookup = market.set_index("date")
    account = _Account(wear.initial_cash, equity=wear.initial_cash)
    records = []
    cumulative_cost = 0.
    for row in net.itertuples():
        bar = lookup.loc[row.date]
        previous = account.equity
        account.cash += account.shares * (
            bar.cash_dividend*(1-wear.dividend_tax_reserve)-bar.bonus_shares*wear.dividend_tax_reserve
        )
        account.shares *= int(bar["split"])
        account.cash -= row.trade_shares*bar.open
        account.shares += row.trade_shares
        account.equity = account.cash+account.shares*bar.close
        cumulative_cost += row.cost
        if account.shares != row.shares or abs(account.equity-row.equity-cumulative_cost) > 1e-6*max(1,wear.initial_cash/1e6):
            raise RuntimeError("gross/net decomposition mismatch")
        records.append({"date": row.date, "return": account.equity/previous-1, "equity": account.equity,
                        "cash": account.cash, "shares": account.shares,
                        "position": account.shares*bar.close/account.equity,
                        "trade_shares": row.trade_shares, "turnover": abs(row.trade_shares)*bar.open/previous,
                        "cost": 0., "cumulative_net_cost": cumulative_cost})
    return pd.DataFrame(records)


def wear_metrics(ledger: pd.DataFrame) -> dict:
    result = metrics(ledger)
    result["cagr_252"] = result["cagr"]
    elapsed = (ledger.date.iloc[-1]-ledger.signal_date.iloc[0]).days/365.25
    result["calendar_years"] = float(elapsed)
    result["cagr"] = float((1+result["total_return"])**(1/elapsed)-1) if elapsed > 0 else 0.
    result["realized_volatility"] = float(ledger["return"].std()*np.sqrt(252))
    result["annualized_cost_fraction"] = float(ledger.cost_fraction.mean()*252)
    for key in COST_COLUMNS:
        result[key] = float(ledger[key].sum())
    requested = ledger.requested_shares.ne(0)
    result["requested_orders"] = int(requested.sum())
    result["capacity_limited_days"] = int((requested & ledger.capacity_limited).sum())
    result["cash_limited_days"] = int((requested & ledger.cash_limited).sum())
    result["blocked_orders"] = int((requested & ledger.blocked.fillna("").isin(("suspended", "buy_at_upper_limit", "sell_at_lower_limit"))).sum())
    result["unfilled_share_fraction"] = float(ledger.unfilled_shares.abs().sum()/max(1, ledger.requested_shares.abs().sum()))
    return result
