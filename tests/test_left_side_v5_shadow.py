from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.features import left_side_v5_shadow as shadow


class _ComponentModel:
    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        positive = frame["component"].to_numpy(float)
        return np.column_stack([1.0 - positive, positive])


class _Gate:
    def predict_score(self, frame: pd.DataFrame) -> np.ndarray:
        return frame["regime"].to_numpy(float)

    def predict_open(self, frame: pd.DataFrame) -> np.ndarray:
        return self.predict_score(frame) >= 0.0


def test_shadow_score_blends_ranks_and_reserves_upper_half_for_open_gate() -> None:
    frame = pd.DataFrame(
        {
            "component": [0.2, 0.8],
            "regime": [-0.1, 0.1],
        }
    )
    bundle = {
        "schema_version": "left-side-absolute-shadow-v5-bundle-v1",
        "component_model": _ComponentModel(),
        "regime_gate": _Gate(),
        "production_weight": 0.5,
    }

    result = shadow.score_left_side_v5_shadow(
        frame,
        production_probability=np.asarray([0.9, 0.1]),
        bundle=bundle,
    )

    assert result["shadow_v5_blended_percentile"].tolist() == pytest.approx(
        [75.0, 75.0]
    )
    assert result["shadow_v5_trade_gate_open"].tolist() == [False, True]
    assert result["shadow_v5_absolute_score"].tolist() == pytest.approx(
        [37.5, 87.5]
    )


def test_shadow_feature_frame_attaches_exact_date_market_values(monkeypatch) -> None:
    target = "2026-08-31"
    candidates = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "trade_date": ["20260831"],
            "date": [pd.Timestamp(target)],
        }
    )
    stock_values = {
        column: [float(index + 1)]
        for index, column in enumerate(shadow.LEFT_SIDE_V5_ACTIVE_FEATURE_COLUMNS[:4])
    }
    market_values = {
        column: [float(index + 10)]
        for index, column in enumerate(shadow.LEFT_SIDE_V5_ACTIVE_FEATURE_COLUMNS[4:])
    }
    breadth_values = {
        column: [float(index + 20)]
        for index, column in enumerate(shadow.LEFT_SIDE_V5_REGIME_FEATURE_COLUMNS[:20])
    }
    monkeypatch.setattr(
        shadow,
        "compute_stock_active_market_value_features",
        lambda _frame: pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "trade_date": ["20260831"],
                **stock_values,
            }
        ),
    )
    monkeypatch.setattr(
        shadow,
        "compute_market_active_market_value_features",
        lambda *_args, **_kwargs: pd.DataFrame(
            {"trade_date": ["20260831"], **market_values}
        ),
    )
    monkeypatch.setattr(
        shadow,
        "_compute_selected_market_breadth_features",
        lambda _frame: pd.DataFrame(
            {"trade_date": ["20260831"], **breadth_values}
        ),
    )

    result = shadow.build_left_side_v5_shadow_feature_frame(
        candidates,
        market_daily=pd.DataFrame(),
        daily_basic=pd.DataFrame(),
        target_date=target,
    )

    assert result.loc[0, "stock_active_share_ratio_13d_proxy"] == 1.0
    assert result.loc[0, "market_breadth_up_ratio_1d"] == 20.0
