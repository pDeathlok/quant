import numpy as np
import pandas as pd
import pytest

from quant.research.tail_overnight import (
    attach_lagged_features,
    build_labels,
    build_snapshot_features,
    pool_union,
    purged_split,
)

CAL = pd.to_datetime(["2026-09-03", "2026-09-04", "2026-09-07", "2026-09-08"])


def bars():
    return pd.DataFrame(
        {
            "symbol": ["A"] * 4,
            "date": CAL,
            "open": [10, 10, 10.2, 10.3],
            "high": [10.2, 10.2, 10.5, 10.5],
            "low": [9.8, 9.9, 10.1, 10.1],
            "close": [10, 10, 10.3, 10.4],
            "pre_close": [10, 10, 10, 10.3],
            "volume_shares": [10000] * 4,
        }
    )


def test_lag_uses_previous_market_date_not_previous_available_row():
    events = pd.DataFrame({"symbol": ["A", "A"], "date": CAL[1:3]})
    factors = pd.DataFrame({"symbol": ["A", "A"], "date": [CAL[0], CAL[2]], "close": [10.0, 999.0]})
    out = attach_lagged_features(events, factors, CAL, ["close"])
    assert out.lag1_close.iloc[0] == 10
    assert np.isnan(out.lag1_close.iloc[1])


def test_pool_union_retains_overlap_and_deduplicates():
    a = pd.DataFrame({"symbol": ["A", "A"], "date": [CAL[0]] * 2})
    out = pool_union({"good": a, "watchlist": a})
    assert len(out) == 1 and out.pool_good.iloc[0] and out.pool_watchlist.iloc[0]
    assert not out.pool_short.iloc[0]


def test_labels_use_next_market_session_and_do_not_fill_missing_bar():
    daily = bars().drop(index=2)
    events = daily.iloc[[1]][["symbol", "date"]]
    out = build_labels(events, daily, CAL)
    assert out.exit_date.iloc[0] == CAL[2]
    assert not out.outcome_observed.iloc[0]
    assert np.isnan(out.y_open.iloc[0])
    assert pd.isna(out.y_hit_up_100bps.iloc[0])


def test_labels_high_is_not_executable_exit_and_cost_applied():
    out = build_labels(bars().iloc[[1]][["symbol", "date"]], bars(), CAL, cost_bps=20)
    assert out.y_open.iloc[0] == pytest.approx(0.02)
    assert out.y_high.iloc[0] == pytest.approx(0.05)
    assert out.y_open_net.iloc[0] == pytest.approx(0.018)


def test_exright_is_flagged_and_blocked_exit_is_not_dropped():
    daily = bars()
    daily.loc[2, "pre_close"] = 9
    out = build_labels(daily.iloc[[1]][["symbol", "date"]], daily, CAL)
    assert out.corporate_action_gap.iloc[0]
    assert np.isnan(out.y_open.iloc[0])
    daily = bars()
    daily.loc[2, ["open", "high", "low", "close"]] = 9
    daily["down_limit"] = 9
    out = build_labels(daily.iloc[[1]][["symbol", "date"]], daily, CAL)
    assert len(out) == 1 and not out.exit_open_executable.iloc[0]
    assert out.y_open.iloc[0] == pytest.approx(-0.1)


def snapshot():
    return pd.DataFrame(
        {
            "symbol": ["A"],
            "date": [CAL[1]],
            "quote_time": ["2026-09-04 14:30:00"],
            "price": [10.1],
            "open": [10.0],
            "high": [10.2],
            "low": [9.9],
            "pre_close": [10.0],
            "volume_shares": [2100.0],
            "amount_yuan": [21210.0],
            "turnover_pct": [2.1],
        }
    )


def test_snapshot_volume_units_projection_and_no_future_quote():
    out = build_snapshot_features(snapshot(), mode="snapshot_1430")
    assert out.tail_volume_est_shares.iloc[0] == pytest.approx(2400)
    assert out.tail_turnover_est_pct.iloc[0] == pytest.approx(2.4)
    assert out.tail_price_to_vwap.iloc[0] == pytest.approx(0.0)
    bad = snapshot()
    bad["quote_time"] = "2026-09-04 15:00:00"
    with pytest.raises(ValueError, match="quote"):
        build_snapshot_features(bad, mode="snapshot_1430")


def test_proxy_is_explicit_and_does_not_project_final_volume():
    out = build_snapshot_features(snapshot(), mode="close_proxy")
    assert out.tail_volume_est_shares.iloc[0] == 2100
    assert out.input_mode.iloc[0] == "close_proxy"


def test_purge_excludes_label_end_on_validation_boundary():
    frame = pd.DataFrame({"date": CAL[:3], "exit_date": CAL[1:]})
    train, valid, test = purged_split(frame, CAL[2], CAL[3])
    assert train.date.tolist() == [CAL[0]]
    assert valid.empty and test.empty


def test_live_rejects_nonfinite_or_inconsistent_quotes():
    bad = snapshot()
    bad["amount_yuan"] = np.inf
    with pytest.raises(ValueError, match="nonfinite"):
        build_snapshot_features(bad, mode="snapshot_1430")
    bad = snapshot()
    bad["price"] = 11
    with pytest.raises(ValueError, match="OHLC"):
        build_snapshot_features(bad, mode="snapshot_1430")


def test_timezone_aware_quote_uses_shanghai_cutoff():
    data = snapshot()
    data["quote_time"] = "2026-09-04T06:30:00Z"
    out = build_snapshot_features(data, mode="snapshot_1430")
    assert out.tail_volume_est_shares.iloc[0] == pytest.approx(2400)


def test_exact_limit_overrides_conservative_proxy():
    data = bars()
    data.loc[1, "close"] = 10.5
    data.loc[1, "high"] = 10.5
    data["up_limit"] = 11.0
    out = build_labels(data.iloc[[1]][["symbol", "date"]], data, CAL)
    assert out.entry_executable.iloc[0]
    assert out.entry_limit_source.iloc[0] == "exact"


def test_duplicate_features_or_calendar_raise():
    events = bars().iloc[[1]][["symbol", "date"]]
    features = pd.concat([bars().iloc[[0]], bars().iloc[[0]]])
    with pytest.raises(ValueError, match="duplicate"):
        attach_lagged_features(events, features, CAL, ["close"])
    with pytest.raises(ValueError, match="calendar"):
        build_labels(events, bars(), [CAL[0], CAL[0]])


def test_exit_scenarios_respect_opening_gap_and_ambiguous_order():
    from quant.research.tail_overnight import daily_exit_scenarios

    outcomes = pd.DataFrame(
        {
            "y_open": [-0.04, 0.0, np.nan],
            "y_high": [0.02, 0.02, np.nan],
            "y_low": [-0.05, -0.02, np.nan],
            "y_close": [0.0, 0.0, np.nan],
        }
    )
    out = daily_exit_scenarios(outcomes, take_profit=0.01, stop_loss=0.01)
    assert out.bracket_stop_first_net.iloc[0] == pytest.approx(-0.042)
    assert out.bracket_take_first_net.iloc[0] == pytest.approx(-0.042)
    assert not out.ambiguous_order.iloc[0]
    assert out.ambiguous_order.iloc[1]
    assert out.bracket_stop_first_net.iloc[1] == pytest.approx(-0.012)
    assert out.bracket_take_first_net.iloc[1] == pytest.approx(0.008)
    assert np.isnan(out.next_close_net.iloc[2])


def test_historical_monthly_pool_expires_at_rebalance_not_dataset_tail(tmp_path, monkeypatch):
    from scripts.research import run_tail_overnight as runner

    calendar = pd.bdate_range("2026-01-29", "2026-03-06")
    monthly = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-30", "2026-02-27"]),
            "ts_code": ["A", "B"],
            "rule": ["all_good_stocks"] * 2,
            "history_window_months": [84] * 2,
        }
    )
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner.pd, "read_parquet", lambda *args, **kwargs: monthly.copy())
    short = pd.DataFrame({"symbol": ["S"], "date": [calendar[0]]})
    daily = pd.DataFrame(
        {
            "symbol": ["A"] * len(calendar),
            "date": calendar,
            "close": [10.0] * len(calendar),
            "pre_close": [10.0] * len(calendar),
        }
    )
    result = runner.historical_pools(short, calendar, "2026-01-29", "2026-03-06", daily)
    good = result[result.pool_good]
    assert good[good.symbol.eq("A")].date.max() == pd.Timestamp("2026-02-26")
    assert good[good.symbol.eq("B")].date.min() == pd.Timestamp("2026-02-27")
    assert good[good.symbol.eq("B")].date.max() == pd.Timestamp("2026-03-06")
    assert not result.pool_watchlist.any()
