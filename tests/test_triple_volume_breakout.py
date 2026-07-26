import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant.strategies.custom.triple_volume_breakout import (
    add_triple_volume_breakout_signals,
    add_triple_volume_strategy_pool_signals,
    load_triple_volume_variants,
)


def test_triple_volume_breakout_signal_detects_shrink_consolidation_breakout():
    dates = pd.date_range("2024-01-01", periods=80, freq="D")
    close = np.linspace(10.0, 13.0, len(dates))
    open_ = close - 0.03
    high = close + 0.08
    low = close - 0.08
    volume = np.full(len(dates), 1000.0)

    anchor = 70
    volume[anchor - 1] = 1000
    volume[anchor] = 3200
    close[anchor] = 12.70
    open_[anchor] = 12.65
    high[anchor] = 12.78
    low[anchor] = 12.60

    for idx, price in zip(range(anchor + 1, anchor + 5), [12.72, 12.74, 12.73, 12.76]):
        close[idx] = price
        open_[idx] = price - 0.02
        high[idx] = price + 0.04
        low[idx] = price - 0.04
        volume[idx] = 850

    signal_idx = anchor + 5
    close[signal_idx] = 13.05
    open_[signal_idx] = 12.82
    high[signal_idx] = 13.12
    low[signal_idx] = 12.78
    volume[signal_idx] = 820

    df = pd.DataFrame(
        {
            "date": dates,
            "symbol": "TEST.SZ",
            "name": "测试股份",
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "pre_close": pd.Series(close).shift(1).fillna(close[0]),
            "volume": volume,
        }
    )

    out = add_triple_volume_breakout_signals(df)

    formula_anchor = anchor + 1
    assert out.loc[formula_anchor, "triple_volume_anchor"] == 1
    assert out.loc[signal_idx, "days_since_triple_volume"] == signal_idx - formula_anchor
    assert out.loc[signal_idx, "triple_volume_price"] == close[formula_anchor]
    assert out.loc[signal_idx, "signal_triple_volume_breakout"] == 1


def test_triple_volume_breakout_excludes_st_names():
    dates = pd.date_range("2024-01-01", periods=80, freq="D")
    df = pd.DataFrame(
        {
            "date": dates,
            "symbol": "TEST.SZ",
            "name": "*ST测试",
            "open": np.linspace(10, 13, len(dates)),
            "high": np.linspace(10.1, 13.1, len(dates)),
            "low": np.linspace(9.9, 12.9, len(dates)),
            "close": np.linspace(10.02, 13.02, len(dates)),
            "pre_close": np.linspace(10, 13, len(dates)),
            "volume": np.full(len(dates), 800.0),
        }
    )
    df.loc[70, "volume"] = 3000.0

    out = add_triple_volume_breakout_signals(df)

    assert out["signal_triple_volume_breakout"].sum() == 0


def test_triple_volume_strategy_pool_scores_expanded_and_conservative_tiers():
    dates = pd.date_range("2024-01-01", periods=90, freq="D")
    close = np.linspace(10.0, 14.0, len(dates))
    open_ = close - 0.03
    high = close + 0.08
    low = close - 0.08
    volume = np.full(len(dates), 1000.0)

    anchor = 78
    volume[anchor - 1] = 1000
    volume[anchor] = 2600
    close[anchor] = 13.20
    open_[anchor] = 13.12
    high[anchor] = 13.28
    low[anchor] = 13.08
    for idx, price in zip(range(anchor + 1, anchor + 4), [13.24, 13.26, 13.25]):
        close[idx] = price
        open_[idx] = price - 0.02
        high[idx] = price + 0.04
        low[idx] = price - 0.04
        volume[idx] = 850
    signal_idx = anchor + 4
    close[signal_idx] = 13.55
    open_[signal_idx] = 13.32
    high[signal_idx] = 13.62
    low[signal_idx] = 13.28
    volume[signal_idx] = 830

    df = pd.DataFrame(
        {
            "date": dates,
            "symbol": "TEST.SZ",
            "name": "测试股份",
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "pre_close": pd.Series(close).shift(1).fillna(close[0]),
            "volume": volume,
        }
    )
    out = add_triple_volume_strategy_pool_signals(df)

    hit_idx = out.index[out["signal_tvb_merged"] == 1][0]
    assert out.loc[hit_idx, "signal_tvb_expanded"] == 1
    assert out.loc[hit_idx, "signal_tvb_conservative"] == 0
    assert out.loc[hit_idx, "tvb_tier"] == "expanded"
    assert out.loc[hit_idx, "tvb_volume_multiple"] == 2.5

    df.loc[anchor, "volume"] = 3200
    out = add_triple_volume_strategy_pool_signals(df)
    hit_idx = out.index[out["signal_tvb_merged"] == 1][0]

    assert out.loc[hit_idx, "signal_tvb_conservative"] == 1
    assert out.loc[hit_idx, "tvb_tier"] == "conservative"
    assert out.loc[hit_idx, "tvb_volume_multiple"] == 3.0


def test_triple_volume_variants_are_loaded_from_yaml(tmp_path: Path):
    config = tmp_path / "strategy.yaml"
    config.write_text(
        """
strategy:
  enabled: true
  variants:
    - id: conservative
      name: Conservative
      tier: conservative
      volume_multiple: 4.0
      signal_mode: avg_pre_shrink_bull_no60
      base_score: 90
      buy_plan: buy
      sell_plan: sell
      backtest_2024: {trades: 2, avg_return_pct: 1.0}
    - id: expanded
      name: Expanded
      tier: expanded
      volume_multiple: 2.0
      signal_mode: avg_pre_shrink_bull_no60
      base_score: 70
      buy_plan: buy
      sell_plan: sell
      backtest_2024: {trades: 3, avg_return_pct: 0.5}
""",
        encoding="utf-8",
    )

    variants = load_triple_volume_variants(config)

    assert [variant.id for variant in variants] == ["conservative", "expanded"]
    assert [variant.volume_multiple for variant in variants] == [4.0, 2.0]
