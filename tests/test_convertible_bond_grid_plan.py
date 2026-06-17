import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd


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
