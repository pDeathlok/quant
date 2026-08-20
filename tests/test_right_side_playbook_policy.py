from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.research.right_side_playbook_policy import (
    DEFAULT_PLAYBOOK_CATALOG,
    EntryConstraint,
    ExitPolicy,
    NO_TRADE_PLAYBOOK,
    PlaybookSpec,
    build_playbook_outcomes,
    default_playbook_catalog,
    playbook_catalog_hash,
    serialize_playbook_catalog,
)


SYMBOL = "000001.SZ"


def _calendar(rows: int = 8) -> pd.DatetimeIndex:
    return pd.bdate_range("2025-01-02", periods=rows)


def _daily(calendar: pd.DatetimeIndex, symbol: str = SYMBOL) -> pd.DataFrame:
    close = np.full(len(calendar), 10.0)
    return pd.DataFrame(
        {
            "ts_code": symbol,
            "trade_date": calendar.strftime("%Y%m%d"),
            "open": close.copy(),
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "pre_close": np.r_[close[0], close[:-1]],
        }
    )


def _sync_pre_close(daily: pd.DataFrame) -> pd.DataFrame:
    out = daily.copy()
    out["pre_close"] = out.groupby("ts_code", sort=False)["close"].shift(1)
    first = out.groupby("ts_code", sort=False).cumcount().eq(0)
    out.loc[first, "pre_close"] = out.loc[first, "close"]
    return out


def _signals(calendar: pd.DatetimeIndex, symbol: str = SYMBOL) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": ["event-1"],
            "symbol": [symbol],
            "date": [calendar[0]],
            "causal_factor": [1.25],
        }
    )


def _spec(
    playbook_id: str,
    *,
    mode: str = "next_open",
    policy: ExitPolicy | None = None,
    cost_bps: float = 15.0,
    entry: EntryConstraint | None = None,
) -> PlaybookSpec:
    actual_entry = entry or EntryConstraint(mode)  # type: ignore[arg-type]
    actual_policy = policy or ExitPolicy("expiry_t3_close", "expiry", 2)
    return PlaybookSpec(playbook_id, actual_entry, actual_policy, cost_bps)


def _outcome(
    daily: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    catalog: tuple[PlaybookSpec, ...],
    *,
    signals: pd.DataFrame | None = None,
    tradability: pd.DataFrame | None = None,
) -> pd.DataFrame:
    return build_playbook_outcomes(
        _signals(calendar) if signals is None else signals,
        _sync_pre_close(daily),
        calendar,
        tradability,
        catalog=catalog,
    )


def test_default_catalog_is_frozen_small_and_unique() -> None:
    assert len(DEFAULT_PLAYBOOK_CATALOG) == 9
    assert len(default_playbook_catalog(include_no_trade=False)) == 8
    assert DEFAULT_PLAYBOOK_CATALOG[-1] == NO_TRADE_PLAYBOOK
    assert len({spec.playbook_id for spec in DEFAULT_PLAYBOOK_CATALOG}) == 9
    assert {
        (spec.entry.mode, spec.exit.policy_id)
        for spec in default_playbook_catalog(include_no_trade=False)
        if spec.entry is not None
    } == {
        (mode, exit_id)
        for mode in ("next_open", "next_close")
        for exit_id in (
            "expiry_t3_close",
            "expiry_t5_close",
            "fixed_tp4_sl2_t5",
            "fixed_tp6_sl3_t5",
        )
    }

    payload = serialize_playbook_catalog()
    assert payload[-1]["playbook_id"] == "no_trade"
    assert len(playbook_catalog_hash()) == 64
    reversed_hash = playbook_catalog_hash(tuple(reversed(DEFAULT_PLAYBOOK_CATALOG)))
    assert reversed_hash != playbook_catalog_hash()


def test_next_day_locked_limit_is_ineligible_but_no_trade_remains_available() -> None:
    calendar = _calendar()
    daily = _daily(calendar)
    daily.loc[1, ["open", "high", "low", "close"]] = 11.0
    tradability = pd.DataFrame(
        {
            "ts_code": [SYMBOL],
            "trade_date": [calendar[1].strftime("%Y%m%d")],
            "up_limit": [11.0],
            "is_suspended": [False],
        }
    )
    catalog = (
        _spec("open", entry=EntryConstraint("next_open", reject_open_at_up_limit=True)),
        _spec("close", entry=EntryConstraint("next_close", reject_close_at_up_limit=True)),
        NO_TRADE_PLAYBOOK,
    )

    result = _outcome(daily, calendar, catalog, tradability=tradability)

    trade = result.loc[result["playbook_id"].ne("no_trade")]
    assert not trade["eligible"].any()
    assert trade["eligibility_reason"].eq("locked_limit_up").all()
    assert trade["locked_limit_up"].all()
    assert trade["maturity_reason"].eq("ineligible_entry").all()
    assert trade["net_return"].isna().all()
    no_trade = result.loc[result["playbook_id"].eq("no_trade")].iloc[0]
    assert no_trade["eligible"]
    assert no_trade["mature"]
    assert no_trade["net_return"] == pytest.approx(0.0)


def test_open_and_close_at_exact_limit_are_mode_specific_gates() -> None:
    calendar = _calendar()
    tradability = pd.DataFrame(
        {
            "ts_code": [SYMBOL],
            "trade_date": [calendar[1].strftime("%Y%m%d")],
            "up_limit": [11.0],
        }
    )
    catalog = (
        _spec("open", entry=EntryConstraint("next_open", reject_open_at_up_limit=True)),
        _spec("close", mode="next_close", entry=EntryConstraint("next_close", reject_close_at_up_limit=True)),
    )

    open_at_limit = _daily(calendar)
    open_at_limit.loc[1, ["open", "high", "low", "close"]] = [11.0, 11.0, 10.5, 10.8]
    open_result = _outcome(open_at_limit, calendar, catalog, tradability=tradability)
    assert open_result.set_index("playbook_id").loc["open", "eligibility_reason"] == "open_at_up_limit"
    assert open_result.set_index("playbook_id").loc["close", "eligible"]

    close_at_limit = _daily(calendar)
    close_at_limit.loc[1, ["open", "high", "low", "close"]] = [10.5, 11.0, 10.4, 11.0]
    close_result = _outcome(close_at_limit, calendar, catalog, tradability=tradability)
    assert close_result.set_index("playbook_id").loc["open", "eligible"]
    assert close_result.set_index("playbook_id").loc["close", "eligibility_reason"] == "close_at_up_limit"


def test_missing_stock_bar_does_not_shift_entry_and_suspension_overrides_stale_bar() -> None:
    calendar = _calendar()
    other = "000002.SZ"
    first = _daily(calendar, SYMBOL)
    second = _daily(calendar, other).drop(index=1)
    daily = pd.concat([first, second], ignore_index=True)
    signals = pd.DataFrame(
        {
            "event_id": ["suspended", "missing"],
            "symbol": [SYMBOL, other],
            "date": [calendar[0], calendar[0]],
        }
    )
    tradability = pd.DataFrame(
        {
            "ts_code": [SYMBOL],
            "trade_date": [calendar[1].strftime("%Y%m%d")],
            "up_limit": [11.0],
            "is_suspended": [True],
        }
    )

    result = _outcome(
        daily,
        calendar,
        (_spec("open"),),
        signals=signals,
        tradability=tradability,
    ).set_index("event_id")

    assert result.loc["suspended", "eligibility_reason"] == "suspended_entry"
    assert result.loc["missing", "eligibility_reason"] == "missing_entry_bar"
    assert result.loc["missing", "entry_date"] == calendar[1]


def test_preregistered_gap_bounds_apply_to_raw_next_open_gap() -> None:
    calendar = _calendar()
    daily = _daily(calendar)
    daily.loc[1, ["open", "high", "low", "close"]] = [10.5, 10.6, 10.4, 10.5]
    restrictive = EntryConstraint("next_open", min_open_gap=-0.02, max_open_gap=0.03)
    permissive = EntryConstraint("next_open", min_open_gap=-0.02, max_open_gap=0.06)

    result = _outcome(
        daily,
        calendar,
        (_spec("restrictive", entry=restrictive), _spec("permissive", entry=permissive)),
    ).set_index("playbook_id")

    assert result.loc["restrictive", "open_gap"] == pytest.approx(0.05)
    assert result.loc["restrictive", "eligibility_reason"] == "open_gap_above_max"
    assert result.loc["permissive", "eligible"]


def test_next_open_and_next_close_use_different_entry_prices() -> None:
    calendar = _calendar()
    daily = _daily(calendar)
    daily.loc[1, ["open", "high", "low", "close"]] = [10.0, 10.3, 9.9, 10.2]
    daily.loc[2:3, ["open", "high", "low", "close"]] = [10.4, 10.5, 10.3, 10.4]
    policy = ExitPolicy("expiry_t3_close", "expiry", 2)

    result = _outcome(
        daily,
        calendar,
        (
            _spec("open", mode="next_open", policy=policy, cost_bps=0),
            _spec("close", mode="next_close", policy=policy, cost_bps=0),
        ),
    ).set_index("playbook_id")

    assert result.loc["open", "entry_price"] == pytest.approx(10.0)
    assert result.loc["close", "entry_price"] == pytest.approx(10.2)
    assert result.loc["open", "exit_date"] == calendar[3]
    assert result.loc["open", "gross_return"] == pytest.approx(0.04)
    assert result.loc["close", "gross_return"] == pytest.approx(10.4 / 10.2 - 1)


def test_entry_session_barriers_cannot_exit_until_t_plus_2() -> None:
    calendar = _calendar()
    daily = _daily(calendar)
    # Both barriers are touched on the T+1 entry session.  They are deliberately
    # ignored for exit purposes, although the open-entry MAE retains the low.
    daily.loc[1, ["open", "high", "low", "close"]] = [10.0, 10.8, 9.7, 10.0]
    daily.loc[2, ["open", "high", "low", "close"]] = [10.0, 10.1, 9.9, 10.0]
    daily.loc[3, ["open", "high", "low", "close"]] = [10.1, 10.5, 10.0, 10.4]
    policy = ExitPolicy("fixed", "fixed", 4, take_profit=0.04, stop_loss=0.02)

    row = _outcome(daily, calendar, (_spec("fixed", policy=policy, cost_bps=0),)).iloc[0]

    assert row["exit_reason"] == "take_profit"
    assert row["exit_date"] == calendar[3]
    assert row["holding_sessions"] == 2
    assert row["mae"] == pytest.approx(-0.03)


def test_fixed_barriers_are_stop_first_and_mark_same_bar_ambiguity() -> None:
    calendar = _calendar()
    daily = _daily(calendar)
    daily.loc[2, ["open", "high", "low", "close"]] = [10.0, 10.5, 9.7, 10.1]
    policy = ExitPolicy("fixed", "fixed", 4, take_profit=0.04, stop_loss=0.02)

    row = _outcome(daily, calendar, (_spec("fixed", policy=policy, cost_bps=0),)).iloc[0]

    assert row["exit_reason"] == "stop_loss"
    assert row["exit_date"] == calendar[2]
    assert row["exit_price"] == pytest.approx(9.8)
    assert row["gross_return"] == pytest.approx(-0.02)
    assert row["ambiguous_bar"]


def test_gap_through_stop_fills_at_open_and_round_trip_cost_is_subtracted_once() -> None:
    calendar = _calendar()
    daily = _daily(calendar)
    daily.loc[2, ["open", "high", "low", "close"]] = [9.5, 9.6, 9.4, 9.5]
    policy = ExitPolicy("fixed", "fixed", 4, take_profit=0.04, stop_loss=0.02)

    row = _outcome(daily, calendar, (_spec("fixed", policy=policy, cost_bps=15),)).iloc[0]

    assert row["exit_reason"] == "stop_loss"
    assert row["exit_price"] == pytest.approx(9.5)
    assert row["gross_return"] == pytest.approx(-0.05)
    assert row["round_trip_cost"] == pytest.approx(0.0015)
    assert row["net_return"] == pytest.approx(-0.0515)


def test_trailing_rule_activates_then_exits_on_a_later_session() -> None:
    calendar = _calendar()
    daily = _daily(calendar)
    daily.loc[2, ["open", "high", "low", "close"]] = [10.1, 10.5, 10.0, 10.4]
    daily.loc[3, ["open", "high", "low", "close"]] = [10.3, 10.4, 10.2, 10.25]
    trailing = ExitPolicy(
        "trailing",
        "trailing",
        4,
        take_profit=0.04,
        stop_loss=0.02,
        trailing_drawdown=0.02,
    )

    row = _outcome(daily, calendar, (_spec("trailing", policy=trailing, cost_bps=0),)).iloc[0]

    assert row["exit_reason"] == "trailing_stop"
    assert row["exit_date"] == calendar[3]
    assert row["exit_price"] == pytest.approx(10.5 * 0.98)
    assert row["holding_sessions"] == 2


def test_next_open_entry_session_high_can_arm_trailing_for_first_sellable_day() -> None:
    calendar = _calendar()
    daily = _daily(calendar)
    daily.loc[1, ["open", "high", "low", "close"]] = [10.0, 10.5, 9.9, 10.4]
    daily.loc[2, ["open", "high", "low", "close"]] = [10.4, 10.4, 10.2, 10.3]
    trailing = ExitPolicy(
        "trailing",
        "trailing",
        4,
        take_profit=0.04,
        stop_loss=0.02,
        trailing_drawdown=0.02,
    )

    row = _outcome(daily, calendar, (_spec("trailing", policy=trailing, cost_bps=0),)).iloc[0]

    assert row["exit_reason"] == "trailing_stop"
    assert row["exit_date"] == calendar[2]
    assert row["holding_sessions"] == 1
    assert row["exit_price"] == pytest.approx(10.5 * 0.98)


def test_early_barrier_is_mature_even_when_full_expiry_tail_is_unavailable() -> None:
    calendar = _calendar(3)
    daily = _daily(calendar)
    daily.loc[2, ["open", "high", "low", "close"]] = [10.1, 10.5, 10.0, 10.4]
    policy = ExitPolicy("fixed", "fixed", 4, take_profit=0.04, stop_loss=0.02)

    row = _outcome(daily, calendar, (_spec("fixed", policy=policy),)).iloc[0]

    assert row["mature"]
    assert row["exit_reason"] == "take_profit"


def test_incomplete_or_missing_future_window_stays_unlabeled() -> None:
    short_calendar = _calendar(3)
    short_daily = _daily(short_calendar)
    expiry = ExitPolicy("expiry_t3_close", "expiry", 2)
    short = _outcome(short_daily, short_calendar, (_spec("expiry", policy=expiry),)).iloc[0]
    assert not short["mature"]
    assert short["maturity_reason"] == "incomplete_market_window"
    assert pd.isna(short["net_return"])

    calendar = _calendar()
    missing_daily = _daily(calendar).drop(index=2)
    missing = _outcome(missing_daily, calendar, (_spec("expiry", policy=expiry),)).iloc[0]
    assert not missing["mature"]
    assert missing["maturity_reason"] == "missing_future_bar"
    assert pd.isna(missing["gross_return"])
    assert pd.isna(missing["mae"])


def test_intermediate_suspension_skips_barriers_but_suspended_expiry_is_unlabeled() -> None:
    calendar = _calendar()
    daily = _daily(calendar)
    # Remove the T+2 OHLC bar and prove the absence is an explicit suspension,
    # not a silent hole in local data.
    daily = daily.drop(index=2)
    tradability = pd.DataFrame(
        {
            "ts_code": [SYMBOL],
            "trade_date": [calendar[2].strftime("%Y%m%d")],
            "is_suspended": [True],
        }
    )
    expiry_t4 = ExitPolicy("expiry_t4_close", "expiry", 3)
    intermediate = _outcome(
        daily,
        calendar,
        (_spec("expiry_t4", policy=expiry_t4, cost_bps=0),),
        tradability=tradability,
    ).iloc[0]

    assert intermediate["mature"]
    assert intermediate["exit_date"] == calendar[4]
    assert intermediate["holding_sessions"] == 3

    expiry_t2 = ExitPolicy("expiry_t2_close", "expiry", 1)
    expiry_suspended = _outcome(
        daily,
        calendar,
        (_spec("expiry_t2", policy=expiry_t2),),
        tradability=tradability,
    ).iloc[0]
    assert not expiry_suspended["mature"]
    assert expiry_suspended["maturity_reason"] == "suspended_expiry_session"
    assert pd.isna(expiry_suspended["net_return"])


def test_appending_future_data_cannot_rewrite_an_already_mature_outcome() -> None:
    calendar = _calendar(5)
    daily = _daily(calendar)
    daily.loc[2, ["open", "high", "low", "close"]] = [10.1, 10.5, 10.0, 10.4]
    policy = ExitPolicy("fixed", "fixed", 4, take_profit=0.04, stop_loss=0.02)
    spec = (_spec("fixed", policy=policy),)
    before = _outcome(daily, calendar, spec).iloc[0]

    extended_calendar = _calendar(7)
    extension = _daily(extended_calendar).iloc[5:].copy()
    extension.loc[:, ["open", "high", "low", "close"]] = [5.0, 6.0, 4.0, 5.0]
    after = _outcome(pd.concat([daily, extension], ignore_index=True), extended_calendar, spec).iloc[0]

    for column in (
        "mature",
        "exit_date",
        "exit_price",
        "exit_reason",
        "gross_return",
        "net_return",
        "mae",
        "holding_sessions",
        "ambiguous_bar",
    ):
        assert after[column] == before[column]


def test_original_causal_signal_columns_are_repeated_for_each_action() -> None:
    calendar = _calendar()
    result = _outcome(
        _daily(calendar),
        calendar,
        (_spec("open"), NO_TRADE_PLAYBOOK),
    )

    assert result["event_id"].eq("event-1").all()
    assert result["causal_factor"].eq(1.25).all()
    assert result[["event_id", "playbook_id"]].duplicated().sum() == 0


def test_auto_event_id_is_stable_and_duplicate_event_keys_are_rejected() -> None:
    calendar = _calendar()
    signals = _signals(calendar).drop(columns="event_id")
    first = _outcome(
        _daily(calendar),
        calendar,
        (_spec("open"),),
        signals=signals,
    )
    second = _outcome(
        _daily(calendar),
        calendar,
        (_spec("open"),),
        signals=signals,
    )
    assert first.loc[0, "event_id"] == f"{SYMBOL}|{calendar[0]:%Y%m%d}"
    assert second.loc[0, "event_id"] == first.loc[0, "event_id"]

    duplicated = pd.concat([signals, signals], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate event_id"):
        _outcome(
            _daily(calendar),
            calendar,
            (_spec("open"),),
            signals=duplicated,
        )
