import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import pytest


def _write_grid_inputs(
    monkeypatch,
    tmp_path: Path,
    daily: pd.DataFrame,
    basic: pd.DataFrame,
):
    import quant.routine.convertible_bond_grid_plan as plan_module

    daily_path = tmp_path / "daily.parquet"
    basic_path = tmp_path / "basic.parquet"
    call_path = tmp_path / "call.parquet"
    daily.to_parquet(daily_path, index=False)
    basic.to_parquet(basic_path, index=False)
    pd.DataFrame().to_parquet(call_path, index=False)
    monkeypatch.setattr(plan_module, "CB_DAILY_PATH", daily_path)
    monkeypatch.setattr(plan_module, "CB_BASIC_PATH", basic_path)
    monkeypatch.setattr(plan_module, "CB_CALL_PATH", call_path)
    return plan_module


def test_convertible_bond_grid_plan_contains_explicit_parts(monkeypatch, tmp_path):
    import quant.routine.convertible_bond_grid_plan as plan_module

    daily_path = tmp_path / "daily.parquet"
    basic_path = tmp_path / "basic.parquet"
    call_path = tmp_path / "call.parquet"
    dates = pd.date_range("2026-01-01", periods=40, freq="D").strftime("%Y%m%d")
    rows = []
    for index, trade_date in enumerate(dates):
        close = 130.0 if index < 20 else 110.0
        rows.append(
            {
                "ts_code": "110001.SH",
                "trade_date": trade_date,
                "close": close,
                "bond_over_rate": 8.0,
                "amount": 8000.0,
                "pct_chg": 0.0,
            }
        )
    daily = pd.DataFrame(rows)
    basic = pd.DataFrame(
        [
            {
                "ts_code": "110001.SH",
                "bond_short_name": "测试转债",
                "stk_short_name": "测试正股",
                "remain_size": 5.0,
                "newest_rating": "AA",
                "list_date": "20200101",
                "delist_date": "",
                "conv_start_date": "20200101",
            }
        ]
    )
    daily.to_parquet(daily_path, index=False)
    basic.to_parquet(basic_path, index=False)
    pd.DataFrame().to_parquet(call_path, index=False)
    monkeypatch.setattr(plan_module, "CB_DAILY_PATH", daily_path)
    monkeypatch.setattr(plan_module, "CB_BASIC_PATH", basic_path)
    monkeypatch.setattr(plan_module, "CB_CALL_PATH", call_path)

    config = plan_module.default_convertible_bond_grid_config()
    config = type(config)(**{**config.__dict__, "min_momentum_20d": None})
    payload = plan_module.build_convertible_bond_grid_plan(limit=1, config=config)

    assert payload["candidates"]
    candidate = payload["candidates"][0]
    operation = candidate["operation_plan"]
    assert operation["max_parts"] >= 1
    assert "buy_levels" in operation
    assert "sell_levels" in operation
    assert "risk_controls" in operation


def test_convertible_bond_grid_strategy_pool_starts_with_formal_overlay_versions():
    from quant.routine.convertible_bond_grid_plan import convertible_bond_grid_strategy_configs

    configs = convertible_bond_grid_strategy_configs()

    assert configs[0][0].name == "core_market_scaled_market_gate"
    assert configs[0][1]["overlay"] == "market_gate"
    assert configs[0][1]["style"] == "稳健主推"
    assert configs[1][0].name == "return_core_trend_rebound"
    assert configs[1][1]["overlay"] == "trend_rebound"
    assert configs[1][1]["style"] == "收益进攻"


def test_convertible_bond_grid_repairs_entire_twenty_session_premium_window(
    monkeypatch,
    tmp_path,
):
    dates = pd.bdate_range("2026-07-16", periods=20).strftime("%Y%m%d")
    daily = pd.DataFrame(
        {
            "ts_code": "110001.SH",
            "trade_date": dates,
            "close": 110.0,
            "bond_over_rate": float("nan"),
            "amount": 8_000.0,
            "pct_chg": 0.0,
        }
    )
    basic = pd.DataFrame(
        {
            "ts_code": ["110001.SH"],
            "bond_short_name": ["测试转债"],
            "stk_code": ["600001.SH"],
            "stk_short_name": ["测试正股"],
            "remain_size": [5.0],
            "newest_rating": ["AA"],
            "list_date": ["20200101"],
            "delist_date": [""],
            "conv_start_date": ["20200101"],
            "conv_price": [10.0],
        }
    )
    plan_module = _write_grid_inputs(monkeypatch, tmp_path, daily, basic)
    monkeypatch.setattr(
        plan_module,
        "_underlying_stock_daily",
        lambda _basic, trade_date: pd.DataFrame(
            {
                "ts_code": ["600001.SH"],
                "trade_date": [trade_date],
                "close": [11.0],
            }
        ),
    )

    payload = plan_module.build_convertible_bond_grid_plan()

    quality = payload["data_quality"]
    assert quality["premium_repaired_rows"] == 20
    assert quality["premium_coverage_window"]["observed_sessions"] == 20
    assert quality["premium_coverage_window"]["sessions_meeting_requirement"] == 20
    assert quality["premium_coverage_window"]["complete"] is True


def test_convertible_bond_grid_fails_when_twenty_session_premium_window_stays_incomplete(
    monkeypatch,
    tmp_path,
):
    dates = pd.bdate_range("2026-07-16", periods=20).strftime("%Y%m%d")
    daily = pd.DataFrame(
        {
            "ts_code": "110001.SH",
            "trade_date": dates,
            "close": 110.0,
            "bond_over_rate": [float("nan")] + [8.0] * 19,
            "amount": 8_000.0,
            "pct_chg": 0.0,
        }
    )
    basic = pd.DataFrame(
        {
            "ts_code": ["110001.SH"],
            "bond_short_name": ["测试转债"],
            "stk_code": ["600001.SH"],
            "stk_short_name": ["测试正股"],
            "remain_size": [5.0],
            "newest_rating": ["AA"],
            "list_date": ["20200101"],
            "delist_date": [""],
            "conv_start_date": ["20200101"],
        }
    )
    plan_module = _write_grid_inputs(monkeypatch, tmp_path, daily, basic)
    monkeypatch.setattr(
        plan_module,
        "_repair_latest_premium_data",
        lambda frame, _basic, _trade_date: (frame, 0),
    )

    with pytest.raises(ValueError, match="20-session premium coverage incomplete"):
        plan_module.build_convertible_bond_grid_plan()


def test_overlay_entry_permission_fails_closed_when_required_market_features_are_missing():
    import quant.routine.convertible_bond_grid_plan as plan_module

    config = plan_module.convertible_bond_grid_strategy_configs()[0][0]

    allowed, reason = plan_module._overlay_entry_permission(
        {
            "market_median_double_low": 130.0,
            "market_trend_20d": float("nan"),
            "market_trend_breadth": float("nan"),
        },
        config,
    )

    assert allowed is False
    assert "市场20日趋势缺失" in reason
    assert "趋势广度缺失" in reason


def test_convertible_bond_reference_poll_accepts_empty_provider_results() -> None:
    import quant.routine.convertible_bond_grid_plan as plan_module

    class Pro:
        def cb_daily(self, **kwargs):
            return pd.DataFrame()

        def cb_basic(self, **kwargs):
            return pd.DataFrame()

        def cb_call(self, **kwargs):
            return pd.DataFrame()

    result = plan_module.refresh_convertible_bond_daily(
        "20260812",
        retries=1,
        sleep_seconds=0,
        fetcher=type("Fetcher", (), {"pro": Pro()})(),
    )

    assert result["status"] == "no_data"
    assert result["reference_poll_status"] == "success"
    assert result["reference_polled_through"] == "20260812"
    assert result["reference_poll"] == {
        "status": "success",
        "polled_through": "20260812",
    }


def test_convertible_bond_reference_fallback_does_not_advance_poll_watermark() -> None:
    import quant.routine.convertible_bond_grid_plan as plan_module

    class Pro:
        def cb_daily(self, **kwargs):
            return pd.DataFrame()

        def cb_basic(self, **kwargs):
            raise RuntimeError("provider unavailable")

        def cb_call(self, **kwargs):
            return pd.DataFrame()

    result = plan_module.refresh_convertible_bond_daily(
        "20260812",
        retries=1,
        sleep_seconds=0,
        fetcher=type("Fetcher", (), {"pro": Pro()})(),
    )

    assert result["reference_poll_status"] == "failed"
    assert result["reference_polled_through"] is None
    assert result["reference_poll"] == {
        "status": "failed",
        "polled_through": None,
    }
